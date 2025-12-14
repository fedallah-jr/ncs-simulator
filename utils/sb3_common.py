"""
Common utilities for Stable-Baselines3 algorithm wrappers.

This module contains shared functions used across PPO, DQN, and other SB3-based implementations.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Optional

import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize


def save_training_rewards(vec_env: gym.Env, output_path: Path) -> None:
    """
    Save episode rewards from a VecEnv's Monitor wrapper to a CSV file.

    Args:
        vec_env: The vectorized environment (possibly wrapped with VecNormalize)
        output_path: Path to save the CSV file
    """
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


def unwrap_base_env(env: Any) -> Any:
    """
    Peel common VecEnv/Monitor wrappers to access the underlying environment.

    Args:
        env: A possibly wrapped environment

    Returns:
        The unwrapped base environment
    """
    current = env
    if hasattr(current, "venv"):
        current = current.venv
    if hasattr(current, "envs"):
        current = current.envs[0]
    while hasattr(current, "env"):
        current = current.env
    return current


class RewardMixLoggingEvalCallback(EvalCallback):
    """
    Eval callback that prints current reward mix weight before each evaluation.

    This is useful for curriculum learning scenarios where the reward function
    changes during training.
    """

    def __init__(self, *args, mix_weight_fn: Callable[[], Optional[float]], **kwargs):
        """
        Initialize the callback.

        Args:
            *args: Arguments passed to EvalCallback
            mix_weight_fn: Function that returns the current reward mix weight
            **kwargs: Keyword arguments passed to EvalCallback
        """
        super().__init__(*args, **kwargs)
        self._mix_weight_fn = mix_weight_fn

    def _evaluate_policy(self) -> None:
        mix_weight = None
        if self._mix_weight_fn is not None:
            mix_weight = self._mix_weight_fn()
        if mix_weight is not None:
            print(f"[Eval] reward_mix_weight={mix_weight:.4f}")
        return super()._evaluate_policy()
