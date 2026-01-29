from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.marl import (
    MARLReplayBuffer,
    QMIXLearner,
    MLPAgent,
    DuelingMLPAgent,
    QMixer,
    run_evaluation,
    build_base_qlearning_parser,
    add_qmix_args,
    save_qlearning_checkpoint,
    build_qlearning_hyperparams,
    select_actions_batched,
    stack_obs,
)
from utils.marl.common import epsilon_by_step
from utils.marl_training import (
    setup_device_and_rng,
    load_config_with_overrides,
    create_obs_normalizer,
    print_run_summary,
)
from utils.marl.vector_env import create_async_vector_env, make_env, stack_vector_obs
from utils.reward_normalization import reset_shared_running_normalizers
from utils.run_utils import prepare_run_directory, save_config_with_hyperparameters, BestModelTracker


def parse_args() -> argparse.Namespace:
    parser = build_base_qlearning_parser(
        description="Train QMIX (shared MLP + hypernet mixer) on multi-agent NCS env."
    )
    add_qmix_args(parser)
    return parser.parse_args()


def main() -> None:
    reset_shared_running_normalizers()
    args = parse_args()
    device, rng = setup_device_and_rng(args.device, args.seed)

    _, config_path_str, n_agents, use_agent_id, eval_reward_override, eval_termination_override = (
        load_config_with_overrides(args.config, args.n_agents, not args.no_agent_id)
    )

    run_dir = prepare_run_directory("qmix", args.config, args.output_root)
    rewards_csv_path = run_dir / "training_rewards.csv"
    eval_csv_path = run_dir / "evaluation_rewards.csv"

    if args.n_envs <= 0:
        raise ValueError("n_envs must be positive")
    if args.episodes_per_update is None:
        args.episodes_per_update = args.n_envs
    if args.episodes_per_update <= 0:
        raise ValueError("episodes_per_update must be positive")
    if args.updates_per_batch is None:
        args.updates_per_batch = args.episodes_per_update
    if args.updates_per_batch <= 0:
        raise ValueError("updates_per_batch must be positive")

    env, env_seeds = create_async_vector_env(
        n_envs=args.n_envs,
        n_agents=n_agents,
        episode_length=args.episode_length,
        config_path_str=config_path_str,
        seed=args.seed,
    )
    eval_env = make_env(
        n_agents=n_agents,
        episode_length=args.episode_length,
        config_path_str=config_path_str,
        seed=args.seed,
        reward_override=eval_reward_override,
        termination_override=eval_termination_override,
        freeze_running_normalization=True,
    )

    obs_dim = int(env.single_observation_space.spaces["agent_0"].shape[0])
    n_actions = int(env.single_action_space.spaces["agent_0"].n)
    obs_normalizer = create_obs_normalizer(
        obs_dim, args.normalize_obs, args.obs_norm_clip, args.obs_norm_eps
    )
    input_dim = obs_dim + (n_agents if use_agent_id else 0)
    state_dim = n_agents * obs_dim

    # Select network class based on --dueling flag
    AgentClass = DuelingMLPAgent if args.dueling else MLPAgent
    agent_kwargs = {
        "input_dim": input_dim,
        "n_actions": n_actions,
        "hidden_dims": tuple(args.hidden_dims),
        "activation": args.activation,
        "layer_norm": args.layer_norm,
    }
    if args.dueling:
        agent_kwargs["stream_hidden_dim"] = args.stream_hidden_dim

    if args.independent_agents:
        agent = torch.nn.ModuleList([AgentClass(**agent_kwargs) for _ in range(n_agents)])
    else:
        agent = AgentClass(**agent_kwargs)
    mixer = QMixer(
        n_agents=n_agents,
        state_dim=state_dim,
        mixing_hidden_dim=args.mixer_hidden_dim,
        hypernet_hidden_dim=args.hypernet_hidden_dim,
    )
    learner = QMIXLearner(
        agent=agent,
        mixer=mixer,
        n_agents=n_agents,
        n_actions=n_actions,
        gamma=args.gamma,
        lr=args.learning_rate,
        target_update_interval=args.target_update_interval,
        grad_clip_norm=args.grad_clip_norm,
        use_agent_id=use_agent_id,
        double_q=args.double_q,
        device=device,
        optimizer_type=args.optimizer,
    )
    buffer = MARLReplayBuffer(
        capacity=args.buffer_size,
        n_agents=n_agents,
        obs_dim=obs_dim,
        device=device,
        rng=rng,
    )

    best_model_tracker = BestModelTracker()
    global_step = 0
    episode = 0
    last_eval_step = 0

    # Helper function to save model checkpoint
    def save_checkpoint(path: Path) -> None:
        mixer_params = {
            "state_dim": state_dim,
            "mixer_hidden_dim": args.mixer_hidden_dim,
            "hypernet_hidden_dim": args.hypernet_hidden_dim,
        }
        save_qlearning_checkpoint(
            path=path,
            algorithm="qmix",
            n_agents=n_agents,
            obs_dim=obs_dim,
            n_actions=n_actions,
            use_agent_id=use_agent_id,
            parameter_sharing=(not args.independent_agents),
            agent_hidden_dims=list(args.hidden_dims),
            agent_activation=args.activation,
            agent_layer_norm=args.layer_norm,
            dueling=args.dueling,
            stream_hidden_dim=args.stream_hidden_dim if args.dueling else None,
            agent=learner.agent,
            obs_normalizer=obs_normalizer,
            mixer=learner.mixer,
            mixer_params=mixer_params,
        )

    # Open both CSV files for writing
    with rewards_csv_path.open("w", newline="", encoding="utf-8") as train_f, \
         eval_csv_path.open("w", newline="", encoding="utf-8") as eval_f:
        train_writer = csv.writer(train_f)
        train_writer.writerow(["episode", "reward_sum", "epsilon", "steps"])
        eval_writer = csv.writer(eval_f)
        eval_writer.writerow(["step", "mean_reward", "std_reward"])

        obs_dict, _info = env.reset(seed=env_seeds)
        obs_raw = stack_vector_obs(obs_dict, n_agents)

        episode_reward_sums = np.zeros((args.n_envs,), dtype=np.float32)
        episodes_since_update = 0

        while global_step < args.total_timesteps:
            if obs_normalizer is not None:
                obs = obs_normalizer.normalize(obs_raw, update=True)
            else:
                obs = obs_raw
            epsilon = epsilon_by_step(
                global_step, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps
            )
            actions = select_actions_batched(
                agent=learner.agent,
                obs=obs,
                n_envs=args.n_envs,
                n_agents=n_agents,
                n_actions=n_actions,
                epsilon=epsilon,
                rng=rng,
                device=device,
                use_agent_id=use_agent_id,
            )
            action_dict = {f"agent_{i}": actions[:, i] for i in range(n_agents)}
            next_obs_dict, rewards_arr, terminated, truncated, infos = env.step(action_dict)
            next_obs_raw = stack_vector_obs(next_obs_dict, n_agents)

            rewards = np.asarray(rewards_arr, dtype=np.float32)
            episode_reward_sums += rewards.sum(axis=1)

            terminated_any = np.asarray(terminated, dtype=np.bool_)
            truncated_any = np.asarray(truncated, dtype=np.bool_)
            done_reset = np.logical_or(terminated_any, truncated_any)

            next_obs_for_buffer = next_obs_raw
            if infos.get("final_obs") is not None:
                final_obs = infos["final_obs"]
                done_indices = np.where(done_reset)[0]
                if len(done_indices) > 0:
                    next_obs_for_buffer = next_obs_raw.copy()
                    for env_idx in done_indices:
                        final_env_obs = final_obs[env_idx]
                        if final_env_obs is not None:
                            next_obs_for_buffer[env_idx] = stack_obs(final_env_obs, n_agents)

            buffer.add_batch(
                obs=obs_raw,
                actions=actions.astype(np.int64),
                rewards=rewards,
                next_obs=next_obs_for_buffer,
                dones=terminated_any,
            )

            obs_raw = next_obs_raw
            global_step += args.n_envs

            if np.any(done_reset):
                done_indices = np.where(done_reset)[0]
                for env_idx in done_indices:
                    train_writer.writerow(
                        [episode, float(episode_reward_sums[env_idx]), float(epsilon), global_step]
                    )
                    train_f.flush()

                    best_model_tracker.update(
                        "train",
                        float(episode_reward_sums[env_idx]),
                        run_dir / "best_train_model.pt",
                        save_checkpoint,
                    )

                    if episode % args.log_interval == 0:
                        print(
                            f"[QMIX] episode={episode} steps={global_step} "
                            f"reward_sum={episode_reward_sums[env_idx]:.3f} eps={epsilon:.3f}"
                        )
                    episode += 1
                    episodes_since_update += 1
                    episode_reward_sums[env_idx] = 0.0

                while episodes_since_update >= args.episodes_per_update:
                    if len(buffer) >= args.start_learning:
                        for _ in range(args.updates_per_batch):
                            batch = buffer.sample(args.batch_size, obs_normalizer=obs_normalizer)
                            learner.update(batch)
                    episodes_since_update -= args.episodes_per_update

            # Periodic evaluation
            if global_step - last_eval_step >= args.eval_freq:
                mean_eval_reward, std_eval_reward, _ = run_evaluation(
                    env=eval_env,
                    agent=learner.agent,
                    n_agents=n_agents,
                    n_actions=n_actions,
                    use_agent_id=use_agent_id,
                    device=device,
                    n_episodes=args.n_eval_episodes,
                    seed=args.seed,
                    obs_normalizer=obs_normalizer,
                )
                eval_writer.writerow([global_step, mean_eval_reward, std_eval_reward])
                eval_f.flush()

                # Save best model based on evaluation reward
                best_model_tracker.update(
                    "eval", mean_eval_reward, run_dir / "best_model.pt", save_checkpoint
                )

                print(
                    f"[QMIX] Eval at step {global_step}: "
                    f"mean_reward={mean_eval_reward:.3f} std={std_eval_reward:.3f}"
                )
                last_eval_step = global_step

    latest_path = run_dir / "latest_model.pt"
    save_checkpoint(latest_path)

    mixer_params = {
        "mixer_hidden_dim": args.mixer_hidden_dim,
        "hypernet_hidden_dim": args.hypernet_hidden_dim,
    }
    hyperparams = build_qlearning_hyperparams(
        algorithm="qmix",
        args=args,
        n_agents=n_agents,
        use_agent_id=use_agent_id,
        device=device,
        mixer_params=mixer_params,
    )
    save_config_with_hyperparameters(run_dir, args.config, "qmix", hyperparams)

    print_run_summary(run_dir, latest_path, rewards_csv_path, eval_csv_path)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
