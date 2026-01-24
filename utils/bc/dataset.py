from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from utils.marl.obs_normalization import RunningObsNormalizer


@dataclass
class BCDataset:
    obs: np.ndarray
    actions: np.ndarray
    rewards: Optional[np.ndarray]
    dones: Optional[np.ndarray]
    episode_ids: Optional[np.ndarray]
    metadata: Dict[str, Any]

    @property
    def num_steps(self) -> int:
        return int(self.obs.shape[0])

    @property
    def n_agents(self) -> int:
        return int(self.obs.shape[1])

    def iter_actor_batches(
        self, batch_size: int, rng: np.random.Generator
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        total_samples = int(self.num_steps * self.n_agents)
        indices = np.arange(total_samples, dtype=np.int64)
        rng.shuffle(indices)
        for start in range(0, total_samples, batch_size):
            batch_idx = indices[start : start + batch_size]
            step_idx = batch_idx // self.n_agents
            agent_idx = batch_idx % self.n_agents
            obs_mb = self.obs[step_idx, agent_idx]
            actions_mb = self.actions[step_idx, agent_idx]
            yield obs_mb, actions_mb, agent_idx

    def compute_obs_normalizer(
        self,
        clip: Optional[float] = 5.0,
        eps: float = 1e-8,
    ) -> "RunningObsNormalizer":
        """Compute observation normalization statistics from the dataset.

        Returns a RunningObsNormalizer initialized with the dataset's observation
        mean and variance, ready for use in BC pretraining and ES training.
        """
        from utils.marl.obs_normalization import RunningObsNormalizer

        obs_dim = int(self.obs.shape[2])
        normalizer = RunningObsNormalizer.create(obs_dim, clip=clip, eps=eps)

        # Flatten observations: (steps, n_agents, obs_dim) -> (steps * n_agents, obs_dim)
        flat_obs = self.obs.reshape(-1, obs_dim).astype(np.float64)
        normalizer.update(flat_obs)

        return normalizer


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

        obs = np.asarray(data["obs"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.int64)
        rewards = np.asarray(data["rewards"], dtype=np.float32) if "rewards" in data else None
        dones = np.asarray(data["dones"], dtype=bool) if "dones" in data else None
        episode_ids = (
            np.asarray(data["episode_ids"], dtype=np.int64)
            if "episode_ids" in data
            else None
        )

        if obs.ndim != 3:
            raise ValueError("BC dataset 'obs' must have shape (steps, n_agents, obs_dim).")
        if actions.ndim != 2:
            raise ValueError("BC dataset 'actions' must have shape (steps, n_agents).")
        if obs.shape[0] != actions.shape[0]:
            raise ValueError("BC dataset arrays must align on the first dimension.")
        if actions.shape[1] != obs.shape[1]:
            raise ValueError("BC dataset n_agents dimension must align across arrays.")

        if rewards is not None:
            if rewards.ndim != 2:
                raise ValueError("BC dataset 'rewards' must have shape (steps, n_agents).")
            if rewards.shape[0] != obs.shape[0] or rewards.shape[1] != obs.shape[1]:
                raise ValueError("BC dataset rewards must align with obs/actions.")

        if dones is not None:
            if dones.ndim != 1:
                raise ValueError("BC dataset 'dones' must be shape (steps,).")
            if dones.shape[0] != obs.shape[0]:
                raise ValueError("BC dataset dones must align on the first dimension.")

        if episode_ids is not None:
            if episode_ids.ndim != 1:
                raise ValueError("BC dataset 'episode_ids' must be shape (steps,).")
            if episode_ids.shape[0] != obs.shape[0]:
                raise ValueError("BC dataset episode_ids must align on the first dimension.")

        metadata: Dict[str, Any] = {
            "n_agents": _read_scalar(data, "n_agents", None),
            "obs_dim": _read_scalar(data, "obs_dim", obs.shape[2]),
            "n_actions": _read_scalar(data, "n_actions", None),
            "episodes": _read_scalar(data, "episodes", None),
            "episode_length": _read_scalar(data, "episode_length", None),
            "seed": _read_scalar(data, "seed", None),
            "policy_name": _read_scalar(data, "policy_name", None),
            "config_path": _read_scalar(data, "config_path", None),
            "use_agent_id": _read_scalar(data, "use_agent_id", None),
            "format_version": _read_scalar(data, "format_version", None),
        }

    return BCDataset(
        obs=obs,
        actions=actions,
        rewards=rewards,
        dones=dones,
        episode_ids=episode_ids,
        metadata=metadata,
    )
