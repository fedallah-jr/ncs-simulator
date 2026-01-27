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
    run_evaluation,
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
    parser = argparse.ArgumentParser(description="Train IQL (shared MLP) on multi-agent NCS env.")
    parser.add_argument("--config", type=Path, default=None, help="Config JSON path.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"), help="Output root directory.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--n-agents", type=int, default=3, help="Number of agents (overridden by config if present).")
    parser.add_argument("--episode-length", type=int, default=500, help="Episode length.")
    parser.add_argument("--total-timesteps", type=int, default=200_000, help="Total environment steps.")

    parser.add_argument("--buffer-size", type=int, default=200_000, help="Replay buffer capacity.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size.")
    parser.add_argument("--start-learning", type=int, default=1_000, help="Start updates after this many steps.")
    parser.add_argument("--train-interval", type=int, default=1, help="Update frequency in env steps.")

    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--target-update-interval", type=int, default=500, help="Hard target update interval (steps).")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clipping L2 norm.")
    parser.add_argument("--double-q", action="store_true", help="Use Double DQN targets.")
    parser.add_argument("--optimizer", type=str, default="rmsprop", choices=["adam", "rmsprop"], help="Optimizer type.")

    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon.")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Final epsilon.")
    parser.add_argument("--epsilon-decay-steps", type=int, default=100_000, help="Linear decay steps.")

    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128], help="MLP hidden dims.")
    parser.add_argument("--activation", type=str, default="tanh", choices=["relu", "tanh", "elu"], help="Activation.")
    parser.add_argument("--layer-norm", action="store_true", help="Enable LayerNorm in MLP.")
    parser.add_argument("--dueling", action="store_true", help="Use Dueling DQN architecture (separate V and A streams).")
    parser.add_argument("--stream-hidden-dim", type=int, default=64, help="Hidden dim for dueling value/advantage streams.")
    parser.add_argument("--no-agent-id", action="store_true", help="Disable appending one-hot agent id.")
    parser.add_argument(
        "--independent-agents",
        action="store_true",
        help="Use independent per-agent networks (disable parameter sharing).",
    )
    parser.add_argument(
        "--team-reward",
        action="store_true",
        help="Use the summed team reward for every agent instead of per-agent rewards.",
    )
    obs_norm_group = parser.add_mutually_exclusive_group()
    obs_norm_group.add_argument(
        "--normalize-obs",
        action="store_true",
        help="Normalize observations with running mean/std (default).",
    )
    obs_norm_group.add_argument(
        "--no-normalize-obs",
        action="store_false",
        dest="normalize_obs",
        help="Disable observation normalization.",
    )
    parser.add_argument(
        "--obs-norm-clip",
        type=float,
        default=5.0,
        help="Clip normalized observations to +/- this value (<=0 disables).",
    )
    parser.add_argument(
        "--obs-norm-eps",
        type=float,
        default=1e-8,
        help="Epsilon for observation normalization.",
    )

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Torch device.")
    parser.add_argument("--log-interval", type=int, default=10, help="Print every N episodes.")

    parser.add_argument("--eval-freq", type=int, default=5000, help="Evaluation frequency in env steps.")
    parser.add_argument("--n-eval-episodes", type=int, default=30, help="Number of evaluation episodes.")
    parser.set_defaults(normalize_obs=True)
    return parser.parse_args()


def main() -> None:
    reset_shared_running_normalizers()
    args = parse_args()
    device, rng = setup_device_and_rng(args.device, args.seed)

    _, config_path_str, n_agents, use_agent_id, eval_reward_override, eval_termination_override = (
        load_config_with_overrides(args.config, args.n_agents, not args.no_agent_id)
    )

    run_dir = prepare_run_directory("iql", args.config, args.output_root)
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

    # Helper function to save model checkpoint
    def save_checkpoint(path: Path) -> None:
        ckpt: Dict[str, Any] = {
            "algorithm": "iql",
            "n_agents": n_agents,
            "obs_dim": obs_dim,
            "n_actions": n_actions,
            "use_agent_id": use_agent_id,
            "parameter_sharing": (not args.independent_agents),
            "agent_hidden_dims": list(args.hidden_dims),
            "agent_activation": args.activation,
            "agent_layer_norm": args.layer_norm,
            "dueling": args.dueling,
            "stream_hidden_dim": args.stream_hidden_dim if args.dueling else None,
        }
        ckpt["obs_normalization"] = (
            obs_normalizer.state_dict() if obs_normalizer is not None else {"enabled": False}
        )
        if args.independent_agents:
            ckpt["agent_state_dicts"] = [net.state_dict() for net in learner.agent]  # type: ignore[union-attr]
        else:
            ckpt["agent_state_dict"] = learner.agent.state_dict()  # type: ignore[union-attr]
        torch.save(ckpt, path)

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
                raw_rewards = np.asarray([rewards_dict[f"agent_{i}"] for i in range(n_agents)], dtype=np.float32)
                if args.team_reward:
                    team_reward = float(raw_rewards.sum())
                    rewards = np.full(n_agents, team_reward, dtype=np.float32)
                    episode_reward_sum += team_reward
                else:
                    rewards = raw_rewards
                    episode_reward_sum += float(raw_rewards.sum())
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

                    print(f"[IQL] Eval at step {global_step}: mean_reward={mean_eval_reward:.3f} std={std_eval_reward:.3f}")
                    last_eval_step = global_step

            train_writer.writerow([episode, episode_reward_sum, epsilon, global_step])
            train_f.flush()

            # Save best model based on training reward
            best_model_tracker.update(
                "train", episode_reward_sum, run_dir / "best_train_model.pt", save_checkpoint
            )

            if episode % args.log_interval == 0:
                print(f"[IQL] episode={episode} steps={global_step} reward_sum={episode_reward_sum:.3f} eps={epsilon:.3f}")
            episode += 1

    latest_path = run_dir / "latest_model.pt"
    save_checkpoint(latest_path)

    hyperparams: Dict[str, Any] = {
        "total_timesteps": args.total_timesteps,
        "episode_length": args.episode_length,
        "n_agents": n_agents,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "start_learning": args.start_learning,
        "train_interval": args.train_interval,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "target_update_interval": args.target_update_interval,
        "grad_clip_norm": args.grad_clip_norm,
        "double_q": args.double_q,
        "optimizer": args.optimizer,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay_steps": args.epsilon_decay_steps,
        "hidden_dims": list(args.hidden_dims),
        "activation": args.activation,
        "layer_norm": args.layer_norm,
        "dueling": args.dueling,
        "stream_hidden_dim": args.stream_hidden_dim,
        "use_agent_id": use_agent_id,
        "independent_agents": args.independent_agents,
        "team_reward": args.team_reward,
        "normalize_obs": args.normalize_obs,
        "obs_norm_clip": args.obs_norm_clip,
        "obs_norm_eps": args.obs_norm_eps,
        "eval_freq": args.eval_freq,
        "n_eval_episodes": args.n_eval_episodes,
        "device": str(device),
        "seed": args.seed,
    }
    save_config_with_hyperparameters(run_dir, args.config, "iql", hyperparams)

    print_run_summary(run_dir, latest_path, rewards_csv_path, eval_csv_path)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
