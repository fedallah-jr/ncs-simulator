"""
Single-agent PPO training/evaluation wrapper for the NCS environment.

Uses stable-baselines3 PPO with a Gymnasium-compatible wrapper that exposes
the single agent's observation/action spaces as a standard env.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local imports
from ncs_env.env import NCS_Env
from utils import SingleAgentWrapper


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_ncs_single_env(config_path: Optional[str], episode_length: int, seed: Optional[int]) -> SingleAgentWrapper:
    def factory():
        return NCS_Env(
            n_agents=1,
            episode_length=episode_length,
            config_path=config_path,
            seed=seed,
        )

    return SingleAgentWrapper(factory)


def run_evaluation(
    model: PPO,
    env: gym.Env,
    episodes: int,
    max_steps: int,
    trajectory_path: Optional[Path],
) -> None:
    records: list[dict[str, Any]] = []
    for ep in range(1, episodes + 1):
        reset_out = env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, _ = reset_out
        else:
            obs = reset_out
        for step in range(1, max_steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, done, truncated, _info = step_out
            else:
                # VecEnv-style: obs, rewards, dones, infos
                obs, reward, done, info = step_out
                truncated = False
            if trajectory_path:
                action_scalar = int(np.asarray(action).reshape(-1)[0])
                records.append(
                    {
                        "episode": ep,
                        "step": step,
                        "action": action_scalar,
                        "reward": float(np.mean(reward)),
                        "done": bool(np.any(done)),
                        "truncated": bool(truncated),
                    }
                )
            if bool(np.any(done)) or bool(truncated):
                break
    if trajectory_path:
        _write_csv(trajectory_path, records, ["episode", "step", "action", "reward", "done", "truncated"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on single-agent NCS env.")
    parser.add_argument("--config", type=Path, default=None, help="Config JSON path.")
    parser.add_argument("--total-timesteps", type=int, default=200_000, help="Total training timesteps.")
    parser.add_argument("--episode-length", type=int, default=1000, help="Episode length.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="PPO learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO batch size.")
    parser.add_argument("--n-steps", type=int, default=2048, help="PPO rollout length.")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Eval episodes for best model.")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Eval frequency in env steps.")
    parser.add_argument("--best-model-path", type=Path, default=Path("outputs/ppo_best_model"), help="Path to save best model.")
    parser.add_argument("--eval-trajectory-output", type=Path, default=None, help="CSV path to save eval trajectories.")
    parser.add_argument("--load-model", type=Path, default=None, help="Path to load a model for evaluation only.")
    parser.add_argument("--normalize-reward", action="store_true", help="Use VecNormalize for reward normalization.")
    parser.add_argument(
        "--train-log-output",
        type=Path,
        default=None,
        help="CSV path to save per-episode training rewards (from Monitor).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path_str = str(args.config) if args.config is not None else None

    def env_fn():
        env = make_ncs_single_env(config_path_str, args.episode_length, args.seed)
        return Monitor(env)

    train_env: gym.Env = DummyVecEnv([env_fn])
    eval_env: gym.Env = DummyVecEnv([env_fn])
    if args.normalize_reward:
        train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True)
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True)
        eval_env.training = False

    if args.load_model is not None:
        model = PPO.load(args.load_model, env=train_env)
    else:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(args.best_model_path.parent),
            log_path=str(args.best_model_path.parent / "eval_logs"),
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
        )
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            seed=args.seed,
            verbose=1,
        )
        model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)
        model.save(str(args.best_model_path))
        print(f"Saved best model to {args.best_model_path}")
        if args.train_log_output:
            # Extract episode rewards from the monitored environment
            monitor_env = train_env.venv if isinstance(train_env, VecNormalize) else train_env
            if hasattr(monitor_env, "envs") and monitor_env.envs:
                rewards = monitor_env.envs[0].get_episode_rewards()
                args.train_log_output.parent.mkdir(parents=True, exist_ok=True)
                with args.train_log_output.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["episode", "reward"])
                    for idx, rew in enumerate(rewards, start=1):
                        writer.writerow([idx, rew])
                print(f"Saved training rewards to {args.train_log_output}")

    if args.eval_episodes > 0:
        run_evaluation(
            model,
            eval_env,
            episodes=args.eval_episodes,
            max_steps=args.episode_length,
            trajectory_path=args.eval_trajectory_output,
        )
        if args.eval_trajectory_output:
            print(f"Saved evaluation trajectories to {args.eval_trajectory_output}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
