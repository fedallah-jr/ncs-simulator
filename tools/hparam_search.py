"""
Lightweight hyperparameter search helper for PPO and DQN runs on the NCS env.

Usage (examples):
    python -m tools.hparam_search --algorithm ppo --config configs/perfect_comm.json --samples 3
    python -m tools.hparam_search --algorithm dqn --config configs/perfect_comm.json --samples 5

This script performs a simple random search over small, built-in hyperparameter
grids and launches training runs inline (no subprocesses). It reuses the SB3
training logic used in `algorithms/ppo_single.py` and `algorithms/deep_q_learning.py`
so output directories and saved configs stay consistent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ncs_env.env import NCS_Env
from utils import SingleAgentWrapper
from utils.run_utils import prepare_run_directory, save_config_with_hyperparameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random-search hyperparameters for PPO or DQN on NCS.")
    parser.add_argument("--algorithm", choices=["ppo", "dqn"], required=True, help="SB3 algorithm to train.")
    parser.add_argument("--config", type=Path, required=True, help="Path to env config JSON.")
    parser.add_argument("--samples", type=int, default=5, help="Number of random hyperparameter samples.")
    parser.add_argument("--total-timesteps", type=int, default=200_000, help="Training timesteps per trial.")
    parser.add_argument("--episode-length", type=int, default=1000, help="Episode length.")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed for sampling hyperparams.")
    parser.add_argument("--normalize-reward", action="store_true", help="Enable VecNormalize for rewards/obs.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"), help="Root for run artifacts.")
    return parser.parse_args()


def sample_hparams(algo: str, rng: np.random.Generator) -> Dict[str, float]:
    """Sample a hyperparameter dict from small predefined grids."""
    if algo == "ppo":
        grid: Dict[str, List] = {
            "learning_rate": [3e-4, 1e-4, 5e-4],
            "gamma": [0.95, 0.99],
            "batch_size": [64, 128, 256],
            "n_steps": [512, 1024, 2048],
            "ent_coef": [0.0, 0.005, 0.01],
        }
    else:  # dqn
        grid = {
            "learning_rate": [1e-3, 5e-4, 2e-4],
            "gamma": [0.95, 0.99],
            "batch_size": [64, 128, 256],
            "buffer_size": [50_000, 100_000, 200_000],
            "train_freq": [4, 8],
            "target_update_interval": [1_000, 5_000, 10_000],
        }
    return {k: rng.choice(v) for k, v in grid.items()}


def build_envs(config_path: str, episode_length: int, seed: int, normalize_reward: bool) -> Tuple[gym.Env, gym.Env]:
    """Construct SB3 train/eval envs with consistent wrappers."""

    def env_factory():
        def make_env():
            return NCS_Env(
                n_agents=1,
                episode_length=episode_length,
                config_path=config_path,
                seed=seed,
            )

        return SingleAgentWrapper(make_env)

    train_env: gym.Env = DummyVecEnv([lambda: Monitor(env_factory())])
    eval_env: gym.Env = DummyVecEnv([lambda: Monitor(env_factory())])

    if normalize_reward:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
        eval_env.training = False

    return train_env, eval_env


def train_once(algo: str, config_path: Path, hparams: Dict[str, float], args: argparse.Namespace) -> Path:
    """Train a single run with sampled hyperparameters."""
    run_dir, _ = prepare_run_directory(algo, config_path, args.output_root)
    cfg_path_str = str(config_path)

    train_env, eval_env = build_envs(cfg_path_str, args.episode_length, args.seed, args.normalize_reward)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir),
        eval_freq=max(1, args.total_timesteps // 20),
        n_eval_episodes=5,
        deterministic=True,
    )

    if algo == "ppo":
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=hparams["learning_rate"],
            gamma=hparams["gamma"],
            n_steps=int(hparams["n_steps"]),
            batch_size=int(hparams["batch_size"]),
            ent_coef=hparams["ent_coef"],
            seed=args.seed,
            verbose=1,
        )
    else:
        model = DQN(
            "MlpPolicy",
            train_env,
            learning_rate=hparams["learning_rate"],
            gamma=hparams["gamma"],
            batch_size=int(hparams["batch_size"]),
            buffer_size=int(hparams["buffer_size"]),
            train_freq=int(hparams["train_freq"]),
            target_update_interval=int(hparams["target_update_interval"]),
            seed=args.seed,
            verbose=1,
        )

    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)
    model.save(str(run_dir / "latest_model"))

    # Persist config + hyperparams
    def _to_py_scalar(x):
        return x.item() if hasattr(x, "item") else x

    hp_record = {k: _to_py_scalar(v) for k, v in hparams.items()}
    hp_record.update(
        {
            "total_timesteps": args.total_timesteps,
            "episode_length": args.episode_length,
            "normalize_reward": args.normalize_reward,
            "seed": args.seed,
            "algorithm": algo,
            "search_seed": args.seed,
        }
    )
    save_config_with_hyperparameters(run_dir, config_path, algo, hp_record)

    summary = {
        "run_dir": str(run_dir),
        "algorithm": algo,
        "hyperparameters": {k: _to_py_scalar(v) for k, v in hparams.items()},
    }
    with (run_dir / "search_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    train_env.close()
    eval_env.close()
    return run_dir


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    config_path = args.config.resolve()

    print(f"Running random search for {args.algorithm} | samples={args.samples} | config={config_path}")
    global_summaries = []
    for idx in range(args.samples):
        hparams = sample_hparams(args.algorithm, rng)
        print(f"\n[{idx + 1}/{args.samples}] Hyperparameters: {hparams}")
        run_dir = train_once(args.algorithm, config_path, hparams, args)
        # Load eval summary (best last eval mean)
        eval_path = Path(run_dir) / "evaluations.npz"
        try:
            npz = np.load(eval_path)
            mean_rewards = npz["results"].mean(axis=1)
            best_mean = float(np.max(mean_rewards))
            final_mean = float(mean_rewards[-1])
        except Exception:
            best_mean = float("nan")
            final_mean = float("nan")

        summary = {
            "run_dir": str(run_dir),
            "algorithm": args.algorithm,
            "hyperparameters": hparams,
            "best_eval_mean": best_mean,
            "final_eval_mean": final_mean,
        }
        global_summaries.append(summary)
        print(f"Completed run -> {run_dir} | best_eval_mean={best_mean:.2f} | final_eval_mean={final_mean:.2f}")

    # Save global comparison table
    search_table_path = args.output_root / f"search_results_{args.algorithm}.json"
    with search_table_path.open("w", encoding="utf-8") as f:
        json.dump(global_summaries, f, indent=2)
        f.write("\n")
    print(f"\nSaved search comparison to {search_table_path}")


if __name__ == "__main__":
    main()
