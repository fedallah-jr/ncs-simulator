from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import numpy as np


@dataclass
class BCDataset:
    obs: np.ndarray
    actions: np.ndarray
    agent_ids: np.ndarray
    metadata: Dict[str, Any]

    @property
    def size(self) -> int:
        return int(self.actions.shape[0])

    def iter_batches(
        self, batch_size: int, rng: np.random.Generator
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = np.arange(self.size, dtype=np.int64)
        rng.shuffle(indices)
        for start in range(0, self.size, batch_size):
            batch_idx = indices[start : start + batch_size]
            yield self.obs[batch_idx], self.actions[batch_idx], self.agent_ids[batch_idx]


def _read_scalar(data: np.lib.npyio.NpzFile, key: str, default: Any) -> Any:
    if key not in data:
        return default
    value = data[key]
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.item()
    return value


def load_bc_dataset(path: Path) -> BCDataset:
    if not path.exists():
        raise FileNotFoundError(f"BC dataset not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "obs" not in data or "actions" not in data:
            raise ValueError("BC dataset must contain 'obs' and 'actions' arrays.")
        if "agent_ids" not in data:
            raise ValueError("BC dataset must include 'agent_ids' for actor pretraining.")

        obs = np.asarray(data["obs"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.int64)
        agent_ids = np.asarray(data["agent_ids"], dtype=np.int64)

        if obs.ndim != 2:
            raise ValueError("BC dataset 'obs' must have shape (N, obs_dim).")
        if actions.ndim != 1 or agent_ids.ndim != 1:
            raise ValueError("BC dataset 'actions' and 'agent_ids' must be 1D arrays.")
        if obs.shape[0] != actions.shape[0] or obs.shape[0] != agent_ids.shape[0]:
            raise ValueError("BC dataset arrays must align on the first dimension.")

        metadata: Dict[str, Any] = {
            "n_agents": _read_scalar(data, "n_agents", None),
            "obs_dim": _read_scalar(data, "obs_dim", obs.shape[1]),
            "n_actions": _read_scalar(data, "n_actions", None),
            "episodes": _read_scalar(data, "episodes", None),
            "episode_length": _read_scalar(data, "episode_length", None),
            "seed": _read_scalar(data, "seed", None),
            "policy_name": _read_scalar(data, "policy_name", None),
            "config_path": _read_scalar(data, "config_path", None),
            "use_agent_id": _read_scalar(data, "use_agent_id", None),
        }

    return BCDataset(obs=obs, actions=actions, agent_ids=agent_ids, metadata=metadata)
