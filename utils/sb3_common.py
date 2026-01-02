"""
Common utilities for Stable-Baselines3 algorithm wrappers.

This module contains shared functions used across PPO, DQN, and other SB3-based implementations.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from stable_baselines3.common.vec_env import VecNormalize

from ncs_env.config import load_config


def load_eval_overrides(
    config_path: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Load evaluation reward and termination overrides from a config file.

    Args:
        config_path: Path to the config JSON file, or None.

    Returns:
        Tuple of (eval_reward_override, eval_termination_override), each may be None.
    """
    if config_path is None:
        return None, None
    try:
        cfg = load_config(config_path)
        reward_cfg = cfg.get("reward", {})
        eval_reward_cfg = reward_cfg.get("evaluation", None)
        eval_reward_override = eval_reward_cfg if isinstance(eval_reward_cfg, dict) else None

        termination_cfg = cfg.get("termination", {})
        eval_termination_cfg = termination_cfg.get("evaluation", None)
        eval_termination_override = eval_termination_cfg if isinstance(eval_termination_cfg, dict) else None

        return eval_reward_override, eval_termination_override
    except Exception:
        return None, None


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
