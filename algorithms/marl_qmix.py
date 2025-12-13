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
from utils.marl.buffer import MARLReplayBuffer
from utils.marl.learners import QMIXLearner
from utils.marl.networks import MLPAgent, QMixer, append_agent_id
from utils.run_utils import prepare_run_directory, save_config_with_hyperparameters


def _select_device(device_str: str) -> torch.device:
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str == "cuda":
        return torch.device("cuda")
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError("device must be one of: auto, cpu, cuda")


def _epsilon_by_step(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return float(eps_end)
    frac = min(1.0, max(0.0, step / float(decay_steps)))
    return float(eps_start + frac * (eps_end - eps_start))


def _stack_obs(obs_dict: Dict[str, Any], n_agents: int) -> np.ndarray:
    return np.stack([np.asarray(obs_dict[f"agent_{i}"], dtype=np.float32) for i in range(n_agents)], axis=0)


@torch.no_grad()
def _select_actions(
    agent: MLPAgent,
    obs: np.ndarray,
    n_agents: int,
    n_actions: int,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
    use_agent_id: bool,
) -> np.ndarray:
    obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
    if use_agent_id:
        obs_t = append_agent_id(obs_t, n_agents)
    q = agent(obs_t.view(n_agents, -1))
    greedy = q.argmax(dim=-1).cpu().numpy().astype(np.int64)
    actions = greedy.copy()
    explore_mask = rng.random(n_agents) < float(epsilon)
    if np.any(explore_mask):
        actions[explore_mask] = rng.integers(0, n_actions, size=int(explore_mask.sum()), endpoint=False, dtype=np.int64)
    return actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QMIX (shared MLP + hypernet mixer) on multi-agent NCS env.")
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

    parser.add_argument("--mixer-hidden-dim", type=int, default=32, help="QMIX mixing hidden dim.")
    parser.add_argument("--hypernet-hidden-dim", type=int, default=64, help="QMIX hypernet hidden dim.")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Torch device.")
    parser.add_argument("--log-interval", type=int, default=10, help="Print every N episodes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _select_device(args.device)
    rng = np.random.default_rng(int(args.seed) if args.seed is not None else None)
    torch.manual_seed(int(args.seed) if args.seed is not None else 0)

    config_path_str = str(args.config) if args.config is not None else None
    cfg = load_config(config_path_str)
    system_cfg = cfg.get("system", {})
    n_agents = int(system_cfg.get("n_agents", args.n_agents))
    use_agent_id = not bool(args.no_agent_id)

    run_dir, _metadata = prepare_run_directory("qmix", args.config, args.output_root)
    rewards_csv_path = run_dir / "training_rewards.csv"

    env = NCS_Env(
        n_agents=n_agents,
        episode_length=int(args.episode_length),
        config_path=config_path_str,
        seed=int(args.seed) if args.seed is not None else None,
    )

    obs_dim = int(env.observation_space.spaces["agent_0"].shape[0])
    n_actions = int(env.action_space.spaces["agent_0"].n)
    input_dim = obs_dim + (n_agents if use_agent_id else 0)
    state_dim = n_agents * obs_dim

    agent = MLPAgent(
        input_dim=input_dim,
        n_actions=n_actions,
        hidden_dims=tuple(int(x) for x in args.hidden_dims),
        activation=str(args.activation),
        layer_norm=bool(args.layer_norm),
    )
    mixer = QMixer(
        n_agents=n_agents,
        state_dim=state_dim,
        mixing_hidden_dim=int(args.mixer_hidden_dim),
        hypernet_hidden_dim=int(args.hypernet_hidden_dim),
    )
    learner = QMIXLearner(
        agent=agent,
        mixer=mixer,
        n_agents=n_agents,
        n_actions=n_actions,
        gamma=float(args.gamma),
        lr=float(args.learning_rate),
        target_update_interval=int(args.target_update_interval),
        grad_clip_norm=float(args.grad_clip_norm) if args.grad_clip_norm is not None else None,
        use_agent_id=use_agent_id,
        double_q=bool(args.double_q),
        device=device,
        team_reward=str(args.team_reward),
    )
    buffer = MARLReplayBuffer(
        capacity=int(args.buffer_size),
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

        while global_step < int(args.total_timesteps):
            episode_seed = None if args.seed is None else int(args.seed) + episode
            obs_dict, _info = env.reset(seed=episode_seed)
            obs = _stack_obs(obs_dict, n_agents)

            episode_reward_sum = 0.0
            done = False
            while not done and global_step < int(args.total_timesteps):
                epsilon = _epsilon_by_step(global_step, float(args.epsilon_start), float(args.epsilon_end), int(args.epsilon_decay_steps))
                actions = _select_actions(
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
                next_obs = _stack_obs(next_obs_dict, n_agents)
                rewards = np.asarray([float(rewards_dict[f"agent_{i}"]) for i in range(n_agents)], dtype=np.float32)
                done = any(bool(terminated[f"agent_{i}"]) or bool(truncated[f"agent_{i}"]) for i in range(n_agents))

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

                if len(buffer) >= int(args.start_learning) and (global_step % int(args.train_interval) == 0):
                    batch = buffer.sample(int(args.batch_size))
                    learner.update(batch)

            writer.writerow([episode, episode_reward_sum, float(epsilon), global_step])
            if episode_reward_sum > best_reward:
                best_reward = episode_reward_sum
                best_path = run_dir / "best_model.pt"
                torch.save(
                    {
                        "algorithm": "qmix",
                        "n_agents": n_agents,
                        "obs_dim": obs_dim,
                        "n_actions": n_actions,
                        "state_dim": state_dim,
                        "use_agent_id": use_agent_id,
                        "agent_hidden_dims": [int(x) for x in args.hidden_dims],
                        "agent_activation": str(args.activation),
                        "agent_layer_norm": bool(args.layer_norm),
                        "team_reward": str(args.team_reward),
                        "mixer_hidden_dim": int(args.mixer_hidden_dim),
                        "hypernet_hidden_dim": int(args.hypernet_hidden_dim),
                        "agent_state_dict": learner.agent.state_dict(),
                        "mixer_state_dict": learner.mixer.state_dict(),
                    },
                    best_path,
                )

            if episode % int(args.log_interval) == 0:
                print(f"[QMIX] episode={episode} steps={global_step} reward_sum={episode_reward_sum:.3f} eps={epsilon:.3f}")
            episode += 1

    latest_path = run_dir / "latest_model.pt"
    torch.save(
        {
            "algorithm": "qmix",
            "n_agents": n_agents,
            "obs_dim": obs_dim,
            "n_actions": n_actions,
            "state_dim": state_dim,
            "use_agent_id": use_agent_id,
            "agent_hidden_dims": [int(x) for x in args.hidden_dims],
            "agent_activation": str(args.activation),
            "agent_layer_norm": bool(args.layer_norm),
            "team_reward": str(args.team_reward),
            "mixer_hidden_dim": int(args.mixer_hidden_dim),
            "hypernet_hidden_dim": int(args.hypernet_hidden_dim),
            "agent_state_dict": learner.agent.state_dict(),
            "mixer_state_dict": learner.mixer.state_dict(),
        },
        latest_path,
    )

    hyperparams: Dict[str, Any] = {
        "total_timesteps": int(args.total_timesteps),
        "episode_length": int(args.episode_length),
        "n_agents": n_agents,
        "buffer_size": int(args.buffer_size),
        "batch_size": int(args.batch_size),
        "start_learning": int(args.start_learning),
        "train_interval": int(args.train_interval),
        "learning_rate": float(args.learning_rate),
        "gamma": float(args.gamma),
        "target_update_interval": int(args.target_update_interval),
        "grad_clip_norm": float(args.grad_clip_norm),
        "double_q": bool(args.double_q),
        "team_reward": str(args.team_reward),
        "epsilon_start": float(args.epsilon_start),
        "epsilon_end": float(args.epsilon_end),
        "epsilon_decay_steps": int(args.epsilon_decay_steps),
        "hidden_dims": [int(x) for x in args.hidden_dims],
        "activation": str(args.activation),
        "layer_norm": bool(args.layer_norm),
        "use_agent_id": use_agent_id,
        "mixer_hidden_dim": int(args.mixer_hidden_dim),
        "hypernet_hidden_dim": int(args.hypernet_hidden_dim),
        "device": str(device),
        "seed": int(args.seed) if args.seed is not None else None,
    }
    save_config_with_hyperparameters(run_dir, args.config, "qmix", hyperparams)

    print(f"Run artifacts stored in {run_dir}")
    print(f"  - Latest model: {latest_path}")
    print(f"  - Best model: {run_dir / 'best_model.pt'}")
    print(f"  - Training rewards: {rewards_csv_path}")
    print(f"  - Config with hyperparameters: {run_dir / 'config.json'}")

    env.close()


if __name__ == "__main__":
    main()

