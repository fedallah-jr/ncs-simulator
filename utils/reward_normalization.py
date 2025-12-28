"""
Utilities for estimating reward normalization statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Any, Dict

import numpy as np
from gymnasium import spaces


@dataclass
class ZScoreRewardNormalizer:
    """Z-score normalizer built from empirical reward mean/std."""

    mean: float
    std: float
    scale: float = 1.0
    eps: float = 1e-8

    def __call__(self, reward: float) -> float:
        denom = self.std if self.std > self.eps else self.eps
        return self.scale * (reward - self.mean) / denom


# VecNormalize-like running normalization (tracks return variance, scales reward).
@dataclass
class RunningRewardNormalizer:
    mean: float = 0.0
    m2: float = 0.0
    count: int = 0
    eps: float = 1e-8

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0
        return max(float(np.sqrt(self.m2 / self.count)), self.eps)

    def __call__(self, reward: float) -> float:
        return reward / self.std


_SHARED_RUNNING_NORMALIZERS: Dict[str, RunningRewardNormalizer] = {}


def get_shared_running_normalizer(key: str) -> RunningRewardNormalizer:
    normalizer = _SHARED_RUNNING_NORMALIZERS.get(key)
    if normalizer is None:
        normalizer = RunningRewardNormalizer()
        _SHARED_RUNNING_NORMALIZERS[key] = normalizer
    return normalizer


def reset_shared_running_normalizers() -> None:
    _SHARED_RUNNING_NORMALIZERS.clear()


# Backward-compatible alias
RewardNormalizer = ZScoreRewardNormalizer


def _sample_random_action(action_space: spaces.Space, rng: np.random.Generator):
    """Sample an action from the given space using the provided RNG."""
    if isinstance(action_space, spaces.Dict):
        return {
            key: _sample_random_action(subspace, rng)
            for key, subspace in action_space.spaces.items()
        }
    if isinstance(action_space, spaces.Discrete):
        return int(rng.integers(action_space.n))
    # Fall back to the built-in sampler if the space is something unexpected.
    return action_space.sample(seed=rng.integers(0, 1_000_000))


def compute_reward_normalizer(
    make_env: Callable[[], Any],
    *,
    episodes: int = 30,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> ZScoreRewardNormalizer:
    """
    Roll out a random policy to estimate reward mean/std for z-score normalization.

    Parameters
    ----------
    make_env : Callable
        Zero-arg callable that returns a fresh environment instance.
    episodes : int
        Number of random episodes to sample.
    max_steps : Optional[int]
        Optional step cap per episode; defaults to the environment's episode_length.
    seed : Optional[int]
        Seed for reproducible random actions.
    """
    rng = np.random.default_rng(seed)
    per_step_rewards = []

    for _ in range(episodes):
        env = make_env()
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        horizon = max_steps or getattr(env, "episode_length", None) or 1000
        for _ in range(horizon):
            action = _sample_random_action(env.action_space, rng)
            obs, rewards, terminated, truncated, _ = env.step(action)
            # Rewards are dicts keyed by agent; normalize using the sum.
            step_r = float(sum(rewards.values())) if isinstance(rewards, dict) else float(rewards)
            per_step_rewards.append(step_r)
            if isinstance(terminated, dict) and isinstance(truncated, dict):
                done = all(terminated.values()) or all(truncated.values())
            else:
                done = bool(terminated) or bool(truncated)
            if done:
                break
        env.close()

    rewards_arr = np.asarray(per_step_rewards, dtype=np.float32)
    return ZScoreRewardNormalizer(mean=float(rewards_arr.mean()), std=float(rewards_arr.std()))


# Backward-compatible alias for clarity
compute_zscore_reward_normalizer = compute_reward_normalizer
