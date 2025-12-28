"""
Utilities for estimating reward normalization statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Any, Dict, MutableMapping

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


class SharedRunningRewardNormalizer(RunningRewardNormalizer):
    """
    Running reward normalizer backed by a shared, process-safe store.

    This mirrors RunningRewardNormalizer's interface but reads/writes its state
    from a shared mapping (e.g., multiprocessing.Manager().dict()) so that
    multiple processes can accumulate identical statistics.
    """

    def __init__(
        self,
        key: str,
        store: MutableMapping[str, Dict[str, float]],
        lock: Optional[Any] = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(mean=0.0, m2=0.0, count=0, eps=eps)
        self._key = key
        self._store = store
        self._lock = lock

    def _ensure_state(self) -> Dict[str, float]:
        state = self._store.get(self._key)
        if state is None:
            state = {"mean": 0.0, "m2": 0.0, "count": 0}
            self._store[self._key] = state
        return state

    def _update_state(self, value: float) -> None:
        state = dict(self._ensure_state())
        mean = float(state.get("mean", 0.0))
        m2 = float(state.get("m2", 0.0))
        count = int(state.get("count", 0))

        count += 1
        delta = value - mean
        mean += delta / count
        delta2 = value - mean
        m2 += delta * delta2

        self._store[self._key] = {"mean": mean, "m2": m2, "count": count}

    def update(self, value: float) -> None:
        if self._lock is not None:
            with self._lock:
                self._update_state(value)
        else:
            self._update_state(value)

    def _compute_std(self) -> float:
        state = self._ensure_state()
        count = int(state.get("count", 0))
        if count < 2:
            return 1.0
        m2 = float(state.get("m2", 0.0))
        return max(float(np.sqrt(m2 / count)), self.eps)

    @property
    def std(self) -> float:
        if self._lock is not None:
            with self._lock:
                return self._compute_std()
        return self._compute_std()


_SHARED_RUNNING_NORMALIZERS: Dict[str, RunningRewardNormalizer] = {}
_SHARED_RUNNING_NORMALIZER_STORE: Optional[MutableMapping[str, Dict[str, float]]] = None
_SHARED_RUNNING_NORMALIZER_LOCK: Optional[Any] = None


def get_shared_running_normalizer(key: str) -> RunningRewardNormalizer:
    normalizer = _SHARED_RUNNING_NORMALIZERS.get(key)
    if normalizer is not None:
        return normalizer

    if _SHARED_RUNNING_NORMALIZER_STORE is not None:
        normalizer = SharedRunningRewardNormalizer(
            key=key,
            store=_SHARED_RUNNING_NORMALIZER_STORE,
            lock=_SHARED_RUNNING_NORMALIZER_LOCK,
        )
        normalizer._ensure_state()
    else:
        normalizer = RunningRewardNormalizer()

    _SHARED_RUNNING_NORMALIZERS[key] = normalizer
    return normalizer


def configure_shared_running_normalizers(
    store: Optional[MutableMapping[str, Dict[str, float]]],
    lock: Optional[Any] = None,
) -> None:
    """
    Configure the backend store used for shared running reward normalizers.

    Pass a multiprocessing.Manager().dict() (and optional lock) when using
    multiple processes so statistics stay synchronized across workers. Passing
    None reverts to a process-local dictionary.
    """
    global _SHARED_RUNNING_NORMALIZER_STORE, _SHARED_RUNNING_NORMALIZER_LOCK
    _SHARED_RUNNING_NORMALIZER_STORE = store
    _SHARED_RUNNING_NORMALIZER_LOCK = lock
    _SHARED_RUNNING_NORMALIZERS.clear()
    if _SHARED_RUNNING_NORMALIZER_STORE is not None:
        if _SHARED_RUNNING_NORMALIZER_LOCK is not None:
            with _SHARED_RUNNING_NORMALIZER_LOCK:
                _SHARED_RUNNING_NORMALIZER_STORE.clear()
        else:
            _SHARED_RUNNING_NORMALIZER_STORE.clear()


def reset_shared_running_normalizers() -> None:
    _SHARED_RUNNING_NORMALIZERS.clear()
    if _SHARED_RUNNING_NORMALIZER_STORE is not None:
        if _SHARED_RUNNING_NORMALIZER_LOCK is not None:
            with _SHARED_RUNNING_NORMALIZER_LOCK:
                _SHARED_RUNNING_NORMALIZER_STORE.clear()
        else:
            _SHARED_RUNNING_NORMALIZER_STORE.clear()


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
