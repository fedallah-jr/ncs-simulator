from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ncs_env.config import load_config
from ncs_env.env import NCS_Env
from utils.marl import MARLReplayBuffer, VDNLearner, MLPAgent
from utils.marl.common import select_device, epsilon_by_step, stack_obs, select_actions
from utils.run_utils import prepare_run_directory, save_config_with_hyperparameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VDN (shared MLP + sum mixer) on multi-agent NCS env.")
    parser.add_argument("--config", type=Path, default=None, help="Config JSON path.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"), help="Output root directory.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--n-agents", type=int, default=3, help="Number of agents (overridden by config if present).")
    parser.add_argument("--episode-length", type=int, default=500, help="Episode length.")
    parser.add_argument("--total-timesteps", type=int, default=200_000, help="Total environment steps.")

    parser.add_argument("--buffer-size", type=int, default=200_000, help="Replay buffer capacity.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--start-learning", type=int, default=1_000, help="Start updates after this many steps.")
    parser.add_argument("--train-interval", type=int, default=1, help="Update frequency in env steps.")

    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--target-update-interval", type=int, default=500, help="Hard target update interval (steps).")
    parser.add_argument("--grad-clip-norm", type=float, default=10.0, help="Gradient clipping L2 norm.")
    parser.add_argument("--double-q", action="store_true", help="Use Double DQN targets.")
    parser.add_argument("--team-reward", type=str, default="sum", choices=["sum", "mean"], help="Team reward aggregation.")

    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon.")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Final epsilon.")
    parser.add_argument("--epsilon-decay-steps", type=int, default=100_000, help="Linear decay steps.")

    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128], help="MLP hidden dims.")
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "tanh", "elu"], help="Activation.")
    parser.add_argument("--layer-norm", action="store_true", help="Enable LayerNorm in MLP.")
    parser.add_argument("--no-agent-id", action="store_true", help="Disable appending one-hot agent id.")
    parser.add_argument(
        "--independent-agents",
        action="store_true",
        help="Use independent per-agent networks (disable parameter sharing).",
    )

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Torch device.")
    parser.add_argument("--log-interval", type=int, default=10, help="Print every N episodes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed if args.seed is not None else 0)

    config_path_str = str(args.config) if args.config is not None else None
    cfg = load_config(config_path_str)
    system_cfg = cfg.get("system", {})
    n_agents = int(system_cfg.get("n_agents", args.n_agents))
    use_agent_id = not args.no_agent_id

    run_dir, _metadata = prepare_run_directory("vdn", args.config, args.output_root)
    rewards_csv_path = run_dir / "training_rewards.csv"

    env = NCS_Env(
        n_agents=n_agents,
        episode_length=args.episode_length,
        config_path=config_path_str,
        seed=args.seed,
    )

    obs_dim = int(env.observation_space.spaces["agent_0"].shape[0])
    n_actions = int(env.action_space.spaces["agent_0"].n)
    input_dim = obs_dim + (n_agents if use_agent_id else 0)

    if args.independent_agents:
        agent = torch.nn.ModuleList(
            [
                MLPAgent(
                    input_dim=input_dim,
                    n_actions=n_actions,
                    hidden_dims=tuple(args.hidden_dims),
                    activation=args.activation,
                    layer_norm=args.layer_norm,
                )
                for _ in range(n_agents)
            ]
        )
    else:
        agent = MLPAgent(
            input_dim=input_dim,
            n_actions=n_actions,
            hidden_dims=tuple(args.hidden_dims),
            activation=args.activation,
            layer_norm=args.layer_norm,
        )
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
        team_reward=args.team_reward,
    )
    buffer = MARLReplayBuffer(
        capacity=args.buffer_size,
        n_agents=n_agents,
        obs_dim=obs_dim,
        device=device,
        rng=rng,
    )

    best_reward = -float("inf")
    global_step = 0
    episode = 0

    with rewards_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward_sum", "epsilon", "steps"])

        while global_step < args.total_timesteps:
            episode_seed = None if args.seed is None else args.seed + episode
            obs_dict, _info = env.reset(seed=episode_seed)
            obs = stack_obs(obs_dict, n_agents)

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
                next_obs = stack_obs(next_obs_dict, n_agents)
                rewards = np.asarray([rewards_dict[f"agent_{i}"] for i in range(n_agents)], dtype=np.float32)
                done = any(terminated[f"agent_{i}"] or truncated[f"agent_{i}"] for i in range(n_agents))

                buffer.add(
                    obs=obs,
                    actions=actions.astype(np.int64),
                    rewards=rewards,
                    next_obs=next_obs,
                    done=done,
                )

                episode_reward_sum += float(rewards.sum())
                obs = next_obs
                global_step += 1

                if len(buffer) >= args.start_learning and global_step % args.train_interval == 0:
                    batch = buffer.sample(args.batch_size)
                    learner.update(batch)

            writer.writerow([episode, episode_reward_sum, epsilon, global_step])
            if episode_reward_sum > best_reward:
                best_reward = episode_reward_sum
                best_path = run_dir / "best_model.pt"
                ckpt: Dict[str, Any] = {
                    "algorithm": "vdn",
                    "n_agents": n_agents,
                    "obs_dim": obs_dim,
                    "n_actions": n_actions,
                    "use_agent_id": use_agent_id,
                    "parameter_sharing": (not args.independent_agents),
                    "agent_hidden_dims": list(args.hidden_dims),
                    "agent_activation": args.activation,
                    "agent_layer_norm": args.layer_norm,
                    "team_reward": args.team_reward,
                }
                if args.independent_agents:
                    ckpt["agent_state_dicts"] = [net.state_dict() for net in learner.agent]  # type: ignore[union-attr]
                else:
                    ckpt["agent_state_dict"] = learner.agent.state_dict()  # type: ignore[union-attr]
                torch.save(
                    ckpt,
                    best_path,
                )

            if episode % args.log_interval == 0:
                print(f"[VDN] episode={episode} steps={global_step} reward_sum={episode_reward_sum:.3f} eps={epsilon:.3f}")
            episode += 1

    latest_path = run_dir / "latest_model.pt"
    latest_ckpt: Dict[str, Any] = {
        "algorithm": "vdn",
        "n_agents": n_agents,
        "obs_dim": obs_dim,
        "n_actions": n_actions,
        "use_agent_id": use_agent_id,
        "parameter_sharing": (not args.independent_agents),
        "agent_hidden_dims": list(args.hidden_dims),
        "agent_activation": args.activation,
        "agent_layer_norm": args.layer_norm,
        "team_reward": args.team_reward,
    }
    if args.independent_agents:
        latest_ckpt["agent_state_dicts"] = [net.state_dict() for net in learner.agent]  # type: ignore[union-attr]
    else:
        latest_ckpt["agent_state_dict"] = learner.agent.state_dict()  # type: ignore[union-attr]
    torch.save(
        latest_ckpt,
        latest_path,
    )

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
        "team_reward": args.team_reward,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay_steps": args.epsilon_decay_steps,
        "hidden_dims": list(args.hidden_dims),
        "activation": args.activation,
        "layer_norm": args.layer_norm,
        "use_agent_id": use_agent_id,
        "independent_agents": args.independent_agents,
        "device": str(device),
        "seed": args.seed,
    }
    save_config_with_hyperparameters(run_dir, args.config, "vdn", hyperparams)

    print(f"Run artifacts stored in {run_dir}")
    print(f"  - Latest model: {latest_path}")
    print(f"  - Best model: {run_dir / 'best_model.pt'}")
    print(f"  - Training rewards: {rewards_csv_path}")
    print(f"  - Config with hyperparameters: {run_dir / 'config.json'}")

    env.close()


if __name__ == "__main__":
    main()
