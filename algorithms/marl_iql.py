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
    IQLLearner,
    MLPAgent,
    DuelingMLPAgent,
    build_base_qlearning_parser,
    add_team_reward_arg,
    save_qlearning_checkpoint,
    save_qlearning_training_state,
    load_qlearning_training_state,
    build_qlearning_hyperparams,
    qlearning_collect_transition,
    patch_autoreset_final_obs,
)
from utils.marl_training import (
    setup_device_and_rng,
    load_config_with_overrides,
    create_obs_normalizer,
    print_run_summary,
    setup_shared_reward_normalizer,
    evaluate_and_log,
    log_completed_episodes,
)
from utils.marl.vector_env import create_async_vector_env, create_eval_async_vector_env, stack_vector_obs
from utils.reward_normalization import reset_shared_running_normalizers
from utils.run_utils import prepare_run_directory, save_config_with_hyperparameters, BestModelTracker


def parse_args() -> argparse.Namespace:
    parser = build_base_qlearning_parser(
        description="Train IQL (shared MLP) on multi-agent NCS env."
    )
    add_team_reward_arg(parser)
    return parser.parse_args()


def main() -> None:
    reset_shared_running_normalizers()
    args = parse_args()
    device, rng = setup_device_and_rng(args.device, args.seed)

    cfg, config_path_str, n_agents, use_agent_id, eval_reward_override, eval_termination_override, network_override, training_reward_override = (
        load_config_with_overrides(args.config, args.n_agents, not args.no_agent_id, args.set_overrides)
    )

    resuming = args.resume is not None
    if resuming:
        run_dir = Path(args.resume)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {run_dir}")
    else:
        run_dir = prepare_run_directory("iql", args.config, args.output_root)
    rewards_csv_path = run_dir / "training_rewards.csv"
    eval_csv_path = run_dir / "evaluation_rewards.csv"

    if args.n_envs <= 0:
        raise ValueError("n_envs must be positive")
    if args.train_interval <= 0:
        raise ValueError("train_interval must be positive")

    shared_reward_normalizer, _shared_reward_manager = setup_shared_reward_normalizer(
        cfg.get("reward", {}), run_dir
    )

    env, env_seeds = create_async_vector_env(
        n_envs=args.n_envs,
        n_agents=n_agents,
        episode_length=args.episode_length,
        config_path_str=config_path_str,
        seed=args.seed,
        shared_reward_normalizer=shared_reward_normalizer,
        network_override=network_override,
        reward_override=training_reward_override,
        minimal_info=True,
    )
    eval_env = create_eval_async_vector_env(
        n_eval_envs=args.n_eval_envs,
        n_agents=n_agents,
        episode_length=args.episode_length,
        config_path_str=config_path_str,
        seed=args.seed,
        reward_override=eval_reward_override,
        termination_override=eval_termination_override,
    )

    obs_dim = int(env.single_observation_space.spaces["agent_0"].shape[0])
    n_actions = int(env.single_action_space.spaces["agent_0"].n)
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
        "feature_norm": args.feature_norm,
    }
    if args.dueling:
        agent_kwargs["stream_hidden_dim"] = args.stream_hidden_dim

    if args.independent_agents:
        agent = torch.nn.ModuleList([AgentClass(**agent_kwargs) for _ in range(n_agents)])
    else:
        agent = AgentClass(**agent_kwargs)
    learner = IQLLearner(
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
    vector_step = 0

    if resuming:
        training_state_path = run_dir / "training_state.pt"
        if not training_state_path.exists():
            raise FileNotFoundError(f"No training_state.pt found in {run_dir}")
        counters = load_qlearning_training_state(
            training_state_path, learner, buffer, obs_normalizer, best_model_tracker,
        )
        global_step = counters["global_step"]
        episode = counters["episode"]
        last_eval_step = counters["last_eval_step"]
        vector_step = counters["vector_step"]
        print(f"Resumed from {run_dir} at step {global_step}")

    # Helper function to save model checkpoint
    def save_checkpoint(path: Path) -> None:
        save_qlearning_checkpoint(
            path=path,
            algorithm="iql",
            n_agents=n_agents,
            obs_dim=obs_dim,
            n_actions=n_actions,
            use_agent_id=use_agent_id,
            parameter_sharing=(not args.independent_agents),
            agent_hidden_dims=list(args.hidden_dims),
            agent_activation=args.activation,
            dueling=args.dueling,
            stream_hidden_dim=args.stream_hidden_dim if args.dueling else None,
            agent=learner.agent,
            obs_normalizer=obs_normalizer,
            feature_norm=args.feature_norm,
        )

    def save_training_state() -> None:
        save_qlearning_training_state(
            run_dir / "training_state.pt", learner, buffer, obs_normalizer,
            best_model_tracker, global_step, episode, last_eval_step, vector_step,
        )

    csv_mode = "a" if resuming else "w"
    with rewards_csv_path.open(csv_mode, newline="", encoding="utf-8") as train_f, \
         eval_csv_path.open(csv_mode, newline="", encoding="utf-8") as eval_f:
        train_writer = csv.writer(train_f)
        eval_writer = csv.writer(eval_f)
        if not resuming:
            train_writer.writerow(["episode", "reward_sum", "epsilon", "steps"])
            eval_writer.writerow(["step", "mean_reward", "std_reward"])

        obs_dict, _info = env.reset(seed=env_seeds)
        obs_raw = stack_vector_obs(obs_dict, n_agents)

        episode_reward_sums = np.zeros((args.n_envs,), dtype=np.float32)

        while global_step < args.total_timesteps:
            step = qlearning_collect_transition(
                env=env, agent=learner.agent, obs_raw=obs_raw,
                obs_normalizer=obs_normalizer, global_step=global_step,
                epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
                epsilon_decay_steps=args.epsilon_decay_steps,
                n_envs=args.n_envs, n_agents=n_agents, n_actions=n_actions,
                use_agent_id=use_agent_id, rng=rng, device=device,
            )

            raw_rewards = step.rewards_arr
            if args.team_reward:
                team_rewards = raw_rewards.sum(axis=1)
                rewards = np.repeat(team_rewards[:, None], n_agents, axis=1).astype(np.float32)
                episode_reward_sums += team_rewards
            else:
                rewards = raw_rewards
                episode_reward_sums += raw_rewards.sum(axis=1)

            next_obs_for_buffer, _ = patch_autoreset_final_obs(
                step.next_obs_raw, step.infos, step.done_reset, n_agents,
            )

            buffer.add_batch(
                obs=obs_raw,
                actions=step.actions,
                rewards=rewards,
                next_obs=next_obs_for_buffer,
                dones=step.terminated,
            )

            obs_raw = step.next_obs_raw
            global_step += args.n_envs
            vector_step += 1

            episode = log_completed_episodes(
                done_reset=step.done_reset, episode_reward_sums=episode_reward_sums,
                global_step=global_step, episode=episode,
                train_writer=train_writer, train_f=train_f,
                best_model_tracker=best_model_tracker, run_dir=run_dir,
                save_checkpoint=save_checkpoint, log_interval=args.log_interval,
                algo_name="IQL",
                extra_csv_values=step.epsilon,
                extra_log_str=f" eps={step.epsilon:.3f}",
            )

            if len(buffer) >= args.start_learning and vector_step % args.train_interval == 0:
                batch = buffer.sample(args.batch_size, obs_normalizer=obs_normalizer)
                learner.update(batch)

            if global_step - last_eval_step >= args.eval_freq:
                evaluate_and_log(
                    eval_env=eval_env, agent=learner.agent,
                    n_eval_envs=args.n_eval_envs, n_agents=n_agents, n_actions=n_actions,
                    use_agent_id=use_agent_id, device=device,
                    n_episodes=args.n_eval_episodes, seed=args.seed,
                    obs_normalizer=obs_normalizer, eval_writer=eval_writer, eval_f=eval_f,
                    best_model_tracker=best_model_tracker, run_dir=run_dir,
                    save_checkpoint=save_checkpoint, global_step=global_step,
                    algo_name="IQL",
                )
                last_eval_step = global_step
                save_training_state()

    latest_path = run_dir / "latest_model.pt"
    save_checkpoint(latest_path)
    save_training_state()

    hyperparams = build_qlearning_hyperparams(
        algorithm="iql",
        args=args,
        n_agents=n_agents,
        use_agent_id=use_agent_id,
        device=device,
    )
    save_config_with_hyperparameters(run_dir, args.config, "iql", hyperparams)

    print_run_summary(run_dir, latest_path, rewards_csv_path, eval_csv_path)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
