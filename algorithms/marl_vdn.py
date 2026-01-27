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
    VDNLearner,
    MLPAgent,
    DuelingMLPAgent,
    run_evaluation,
    build_base_qlearning_parser,
    save_qlearning_checkpoint,
    build_qlearning_hyperparams,
)
from utils.marl.common import epsilon_by_step, stack_obs, select_actions
from utils.marl_training import (
    setup_device_and_rng,
    load_config_with_overrides,
    create_environments,
    create_obs_normalizer,
    print_run_summary,
)
from utils.reward_normalization import reset_shared_running_normalizers
from utils.run_utils import prepare_run_directory, save_config_with_hyperparameters, BestModelTracker


def parse_args() -> argparse.Namespace:
    parser = build_base_qlearning_parser(
        description="Train VDN (shared MLP + sum mixer) on multi-agent NCS env."
    )
    return parser.parse_args()


def main() -> None:
    reset_shared_running_normalizers()
    args = parse_args()
    device, rng = setup_device_and_rng(args.device, args.seed)

    _, config_path_str, n_agents, use_agent_id, eval_reward_override, eval_termination_override = (
        load_config_with_overrides(args.config, args.n_agents, not args.no_agent_id)
    )

    run_dir = prepare_run_directory("vdn", args.config, args.output_root)
    rewards_csv_path = run_dir / "training_rewards.csv"
    eval_csv_path = run_dir / "evaluation_rewards.csv"

    env, eval_env = create_environments(
        n_agents=n_agents,
        episode_length=args.episode_length,
        config_path_str=config_path_str,
        seed=args.seed,
        eval_reward_override=eval_reward_override,
        eval_termination_override=eval_termination_override,
    )

    obs_dim = int(env.observation_space.spaces["agent_0"].shape[0])
    n_actions = int(env.action_space.spaces["agent_0"].n)
    input_dim = obs_dim + (n_agents if use_agent_id else 0)
    obs_normalizer = create_obs_normalizer(
        obs_dim, args.normalize_obs, args.obs_norm_clip, args.obs_norm_eps
    )

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
    learner = VDNLearner(
        agent=agent,
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
        save_qlearning_checkpoint(
            path=path,
            algorithm="vdn",
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
        )

    # Open both CSV files for writing
    with rewards_csv_path.open("w", newline="", encoding="utf-8") as train_f, \
         eval_csv_path.open("w", newline="", encoding="utf-8") as eval_f:
        train_writer = csv.writer(train_f)
        train_writer.writerow(["episode", "reward_sum", "epsilon", "steps"])
        eval_writer = csv.writer(eval_f)
        eval_writer.writerow(["step", "mean_reward", "std_reward"])

        while global_step < args.total_timesteps:
            episode_seed = None if args.seed is None else args.seed + episode
            obs_dict, _info = env.reset(seed=episode_seed)
            obs_raw = stack_obs(obs_dict, n_agents)
            if obs_normalizer is not None:
                obs = obs_normalizer.normalize(obs_raw, update=True)
            else:
                obs = obs_raw

            episode_reward_sum = 0.0
            done = False
            while not done and global_step < args.total_timesteps:
                epsilon = epsilon_by_step(global_step, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps)
                actions = select_actions(
                    agent=learner.agent,
                    obs=obs,
                    n_agents=n_agents,
                    n_actions=n_actions,
                    epsilon=epsilon,
                    rng=rng,
                    device=device,
                    use_agent_id=use_agent_id,
                )
                action_dict = {f"agent_{i}": int(actions[i]) for i in range(n_agents)}
                next_obs_dict, rewards_dict, terminated, truncated, _infos = env.step(action_dict)
                next_obs_raw = stack_obs(next_obs_dict, n_agents)
                if obs_normalizer is not None:
                    next_obs = obs_normalizer.normalize(next_obs_raw, update=True)
                else:
                    next_obs = next_obs_raw
                rewards = np.asarray([rewards_dict[f"agent_{i}"] for i in range(n_agents)], dtype=np.float32)
                # Distinguish termination (true end) from truncation (time limit)
                # Only terminated should zero out bootstrap; truncated should still bootstrap
                term = any(terminated[f"agent_{i}"] for i in range(n_agents))
                trunc = any(truncated[f"agent_{i}"] for i in range(n_agents))
                done = term or trunc  # For episode reset logic

                buffer.add(
                    obs=obs_raw,
                    actions=actions.astype(np.int64),
                    rewards=rewards,
                    next_obs=next_obs_raw,
                    done=term,  # Store only terminated for correct bootstrapping
                )

                episode_reward_sum += float(rewards.sum())
                obs_raw = next_obs_raw
                obs = next_obs
                global_step += 1

                if len(buffer) >= args.start_learning and global_step % args.train_interval == 0:
                    batch = buffer.sample(args.batch_size, obs_normalizer=obs_normalizer)
                    learner.update(batch)

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

                    print(f"[VDN] Eval at step {global_step}: mean_reward={mean_eval_reward:.3f} std={std_eval_reward:.3f}")
                    last_eval_step = global_step

            train_writer.writerow([episode, episode_reward_sum, epsilon, global_step])
            train_f.flush()

            # Save best model based on training reward
            best_model_tracker.update(
                "train", episode_reward_sum, run_dir / "best_train_model.pt", save_checkpoint
            )

            if episode % args.log_interval == 0:
                print(f"[VDN] episode={episode} steps={global_step} reward_sum={episode_reward_sum:.3f} eps={epsilon:.3f}")
            episode += 1

    latest_path = run_dir / "latest_model.pt"
    save_checkpoint(latest_path)

    hyperparams = build_qlearning_hyperparams(
        algorithm="vdn",
        args=args,
        n_agents=n_agents,
        use_agent_id=use_agent_id,
        device=device,
    )
    save_config_with_hyperparameters(run_dir, args.config, "vdn", hyperparams)

    print_run_summary(run_dir, latest_path, rewards_csv_path, eval_csv_path)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
