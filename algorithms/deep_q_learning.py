"""
Single-agent DQN training/evaluation for the NCS environment using SB3.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ncs_env.env import NCS_Env
from ncs_env.config import load_config
from utils import SingleAgentWrapper
from utils.run_utils import prepare_run_directory, save_config_with_hyperparameters


def make_env_fn(
    config_path: Optional[str],
    episode_length: int,
    seed: Optional[int],
    reward_override: Optional[Dict[str, Any]] = None,
) -> Callable[[], gym.Env]:
    def factory():
        def env_factory():
            return NCS_Env(
                n_agents=1,
                episode_length=episode_length,
                config_path=config_path,
                seed=seed,
                reward_override=reward_override,
            )
        return SingleAgentWrapper(env_factory)

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


def _unwrap_base_env(env: Any):
    """Peel common VecEnv/Monitor wrappers to access the underlying env."""
    current = env
    if hasattr(current, "venv"):
        current = current.venv
    if hasattr(current, "envs"):
        current = current.envs[0]
    while hasattr(current, "env"):
        current = current.env
    return current


class RewardMixLoggingEvalCallback(EvalCallback):
    """Eval callback that prints current reward mix weight before each evaluation."""

    def __init__(self, *args, mix_weight_fn: Callable[[], Optional[float]], **kwargs):
        super().__init__(*args, **kwargs)
        self._mix_weight_fn = mix_weight_fn

    def _evaluate_policy(self) -> None:
        mix_weight = None
        if self._mix_weight_fn is not None:
            mix_weight = self._mix_weight_fn()
        if mix_weight is not None:
            print(f"[Eval] reward_mix_weight={mix_weight:.4f}")
        return super()._evaluate_policy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on single-agent NCS env.")
    parser.add_argument("--config", type=Path, default=None, help="Config JSON path.")
    parser.add_argument("--total-timesteps", type=int, default=100_000, help="Total training timesteps.")
    parser.add_argument("--episode-length", type=int, default=1000, help="Episode length.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="DQN learning rate.")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor.")
    parser.add_argument("--batch-size", type=int, default=128, help="DQN batch size.")
    parser.add_argument("--buffer-size", type=int, default=200_000, help="Replay buffer size.")
    parser.add_argument("--train-freq", type=int, default=4, help="Training frequency in steps.")
    parser.add_argument("--target-update-interval", type=int, default=10_000, help="Target network update interval.")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Eval episodes for best model.")
    parser.add_argument("--eval-freq", type=int, default=2_500, help="Eval frequency in env steps.")
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
    eval_reward_override: Optional[Dict[str, Any]] = None
    if config_path_str is not None:
        try:
            cfg = load_config(config_path_str)
            reward_cfg = cfg.get("reward", {})
            eval_reward_cfg = reward_cfg.get("evaluation", None)
            if isinstance(eval_reward_cfg, dict):
                eval_reward_override = eval_reward_cfg
        except Exception:
            eval_reward_override = None

    run_dir, metadata = prepare_run_directory("dqn", args.config, args.output_root)

    def env_builder():
        def factory():
            return NCS_Env(
                n_agents=1,
                episode_length=args.episode_length,
                config_path=config_path_str,
                seed=args.seed,
            )
        return SingleAgentWrapper(factory)

    def eval_builder():
        def factory():
            return NCS_Env(
                n_agents=1,
                episode_length=args.episode_length,
                config_path=config_path_str,
                seed=args.seed,
                reward_override=eval_reward_override,
            )
        return SingleAgentWrapper(factory)

    train_env: gym.Env = DummyVecEnv([lambda: Monitor(env_builder())])
    eval_env: gym.Env = DummyVecEnv([lambda: Monitor(eval_builder())])
    if args.normalize_reward:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
        eval_env.training = False

    def get_mix_weight() -> Optional[float]:
        base_env = _unwrap_base_env(eval_env)
        if hasattr(base_env, "get_reward_mix_weight"):
            try:
                return float(base_env.get_reward_mix_weight())
            except Exception:
                return None
        return None

    eval_callback = RewardMixLoggingEvalCallback(
        eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        mix_weight_fn=get_mix_weight,
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
    save_config_with_hyperparameters(run_dir, args.config, "dqn", hyperparams)

    print(f"Run artifacts stored in {run_dir}")
    print("Files created:")
    print(f"  - Best model and evaluation logs (EvalCallback outputs) in {run_dir}")
    print(f"  - Latest model: {latest_model_path}.zip")
    print(f"  - Training rewards: {training_rewards_path}")
    print(f"  - Config with hyperparameters: {run_dir / 'config.json'}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
