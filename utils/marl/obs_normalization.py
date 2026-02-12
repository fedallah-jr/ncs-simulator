from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class RunningObsNormalizer:
    mean: np.ndarray
    m2: np.ndarray
    count: int
    clip: Optional[float] = 5.0
    eps: float = 1e-8

    @classmethod
    def create(
        cls, obs_dim: int, *, clip: Optional[float] = 5.0, eps: float = 1e-8
    ) -> "RunningObsNormalizer":
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        mean = np.zeros((obs_dim,), dtype=np.float64)
        m2 = np.zeros((obs_dim,), dtype=np.float64)
        return cls(mean=mean, m2=m2, count=0, clip=clip, eps=float(eps))

    @property
    def obs_dim(self) -> int:
        return int(self.mean.shape[0])

    def _flatten_obs(self, obs: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
        obs_arr = np.asarray(obs, dtype=np.float64)
        if obs_arr.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Observation last dimension {obs_arr.shape[-1]} does not match obs_dim={self.obs_dim}"
            )
        flat = obs_arr.reshape(-1, self.obs_dim)
        return flat, obs_arr.shape

    @property
    def var(self) -> np.ndarray:
        if self.count < 2:
            return np.ones_like(self.mean, dtype=np.float64)
        return self.m2 / float(self.count)

    def update(self, obs: np.ndarray) -> None:
        flat, _ = self._flatten_obs(obs)
        if flat.size == 0:
            return
        batch_count = int(flat.shape[0])
        batch_mean = flat.mean(axis=0)
        batch_var = flat.var(axis=0)
        if self.count == 0:
            self.mean = batch_mean
            self.m2 = batch_var * batch_count
            self.count = batch_count
            return

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean = self.mean + delta * (batch_count / float(total_count))
        self.m2 = (
            self.m2
            + batch_var * batch_count
            + (delta ** 2) * self.count * batch_count / float(total_count)
        )
        self.count = total_count

    def update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ) -> None:
        if batch_count <= 0:
            return
        mean = np.asarray(batch_mean, dtype=np.float64)
        var = np.asarray(batch_var, dtype=np.float64)
        if mean.shape != self.mean.shape or var.shape != self.mean.shape:
            raise ValueError("Moment shapes must match obs_dim")
        if self.count == 0:
            self.mean = mean
            self.m2 = var * batch_count
            self.count = int(batch_count)
            return
        delta = mean - self.mean
        total_count = self.count + int(batch_count)
        self.mean = self.mean + delta * (batch_count / float(total_count))
        self.m2 = (
            self.m2
            + var * batch_count
            + (delta ** 2) * self.count * batch_count / float(total_count)
        )
        self.count = total_count

    def set_state(self, mean: np.ndarray, m2: np.ndarray, count: int) -> None:
        mean_arr = np.asarray(mean, dtype=np.float64)
        m2_arr = np.asarray(m2, dtype=np.float64)
        if mean_arr.shape != self.mean.shape or m2_arr.shape != self.mean.shape:
            raise ValueError("State shapes must match obs_dim")
        self.mean = mean_arr
        self.m2 = m2_arr
        self.count = int(count)

    def normalize(self, obs: np.ndarray, *, update: bool = True) -> np.ndarray:
        flat, shape = self._flatten_obs(obs)
        if update and flat.size > 0:
            self.update_from_moments(
                batch_mean=flat.mean(axis=0),
                batch_var=flat.var(axis=0),
                batch_count=int(flat.shape[0]),
            )
        std = np.sqrt(self.var) + float(self.eps)
        normalized = (flat - self.mean) / std
        if self.clip is not None and self.clip > 0:
            normalized = np.clip(normalized, -float(self.clip), float(self.clip))
        return normalized.reshape(shape).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "mean": self.mean.astype(np.float64).tolist(),
            "m2": self.m2.astype(np.float64).tolist(),
            "count": int(self.count),
            "clip": None if self.clip is None else float(self.clip),
            "eps": float(self.eps),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "RunningObsNormalizer":
        mean = np.asarray(state.get("mean", []), dtype=np.float64)
        if mean.size == 0:
            raise ValueError("obs_normalization state_dict must include non-empty mean")
        if "m2" in state and state["m2"] is not None:
            m2 = np.asarray(state.get("m2", []), dtype=np.float64)
        else:
            var = np.asarray(state.get("var", np.ones_like(mean)), dtype=np.float64)
            count = int(state.get("count", 1))
            m2 = var * max(count, 1)
        count = int(state.get("count", 0))
        clip = state.get("clip", 5.0)
        clip_value = None if clip is None else float(clip)
        eps = float(state.get("eps", 1e-8))
        return cls(mean=mean, m2=m2, count=count, clip=clip_value, eps=eps)
