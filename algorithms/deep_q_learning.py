"""
Single-agent DQN training/evaluation for the NCS environment using SB3.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ncs_env.env import NCS_Env
from utils import SingleAgentWrapper
from utils.run_utils import prepare_run_directory, write_details_file


def make_env_fn(config_path: Optional[str], episode_length: int, seed: Optional[int]) -> Callable[[], gym.Env]:
    def factory():
        env = NCS_Env(
            n_agents=1,
            episode_length=episode_length,
            config_path=config_path,
            seed=seed,
        )
        return SingleAgentWrapper(lambda: env)

    return factory


def save_training_rewards(vec_env: gym.Env, output_path: Path) -> None:
    monitor_env = vec_env.venv if isinstance(vec_env, VecNormalize) else vec_env
    rewards = []
    if hasattr(monitor_env, "envs") and monitor_env.envs:
        env = monitor_env.envs[0]
        if hasattr(env, "get_episode_rewards"):
            rewards = env.get_episode_rewards()
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for idx, rew in enumerate(rewards, start=1):
            writer.writerow([idx, rew])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on single-agent NCS env.")
    parser.add_argument("--config", type=Path, default=None, help="Config JSON path.")
    parser.add_argument("--total-timesteps", type=int, default=200_000, help="Total training timesteps.")
    parser.add_argument("--episode-length", type=int, default=1000, help="Episode length.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="DQN learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--batch-size", type=int, default=64, help="DQN batch size.")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="Replay buffer size.")
    parser.add_argument("--train-freq", type=int, default=4, help="Training frequency in steps.")
    parser.add_argument("--target-update-interval", type=int, default=10_000, help="Target network update interval.")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Eval episodes for best model.")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Eval frequency in env steps.")
    parser.add_argument("--normalize-reward", action="store_true", help="Use VecNormalize for reward normalization.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Base directory where run artifacts will be stored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path_str = str(args.config) if args.config is not None else None
    run_dir, metadata = prepare_run_directory("dqn", args.config, args.output_root)

    def env_builder():
        env = NCS_Env(
            n_agents=1,
            episode_length=args.episode_length,
            config_path=config_path_str,
            seed=args.seed,
        )
        return SingleAgentWrapper(lambda: env)

    train_env: gym.Env = DummyVecEnv([lambda: Monitor(env_builder())])
    eval_env: gym.Env = DummyVecEnv([lambda: Monitor(env_builder())])
    if args.normalize_reward:
        train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True)
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True)
        eval_env.training = False

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )
    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        train_freq=args.train_freq,
        target_update_interval=args.target_update_interval,
        seed=args.seed,
        verbose=1,
    )
    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)
    latest_model_path = run_dir / "latest_model"
    model.save(str(latest_model_path))

    training_rewards_path = run_dir / "training_rewards.csv"
    save_training_rewards(train_env, training_rewards_path)

    hyperparams = {
        "total_timesteps": args.total_timesteps,
        "episode_length": args.episode_length,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "train_freq": args.train_freq,
        "target_update_interval": args.target_update_interval,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
        "normalize_reward": args.normalize_reward,
        "seed": args.seed,
    }
    write_details_file(run_dir, metadata, hyperparams)

    print(f"Run artifacts stored in {run_dir}")
    print("Files created:")
    print(f"  - Best model and evaluation logs (EvalCallback outputs) in {run_dir}")
    print(f"  - Latest model: {latest_model_path}.zip")
    print(f"  - Training rewards: {training_rewards_path}")
    print(f"  - Details: {run_dir / 'details.txt'}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
