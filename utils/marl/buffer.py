from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from utils.marl.obs_normalization import RunningObsNormalizer

@dataclass(frozen=True)
class MARLBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor
    states: torch.Tensor
    next_states: torch.Tensor


class MARLReplayBuffer:
    """
    Simple transition replay buffer for MLP-based MARL (IQL/VDN/QMIX).

    Stores per-step transitions with shapes:
      obs:        (buffer, n_agents, obs_dim)
      actions:    (buffer, n_agents)
      rewards:    (buffer, n_agents)
      next_obs:   (buffer, n_agents, obs_dim)
      dones:      (buffer,)
      states:     (buffer, state_dim)
      next_states:(buffer, state_dim)
    """

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_dim: int,
        device: torch.device,
        state_dim: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if n_agents <= 0 or obs_dim <= 0:
            raise ValueError("n_agents and obs_dim must be positive")
        self.capacity = int(capacity)
        self.n_agents = int(n_agents)
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim) if state_dim is not None else self.n_agents * self.obs_dim
        self._state_dim_from_obs = self.state_dim == self.n_agents * self.obs_dim
        self.device = device
        self.rng = rng if rng is not None else np.random.default_rng()

        self._ptr = 0
        self._size = 0

        self._obs = np.zeros((self.capacity, self.n_agents, self.obs_dim), dtype=np.float32)
        self._actions = np.zeros((self.capacity, self.n_agents), dtype=np.int64)
        self._rewards = np.zeros((self.capacity, self.n_agents), dtype=np.float32)
        self._next_obs = np.zeros((self.capacity, self.n_agents, self.obs_dim), dtype=np.float32)
        self._dones = np.zeros((self.capacity,), dtype=np.float32)
        self._states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self._next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        done: bool,
        *,
        states: Optional[np.ndarray] = None,
        next_states: Optional[np.ndarray] = None,
    ) -> None:
        if obs.shape != (self.n_agents, self.obs_dim):
            raise ValueError("obs must have shape (n_agents, obs_dim)")
        if next_obs.shape != (self.n_agents, self.obs_dim):
            raise ValueError("next_obs must have shape (n_agents, obs_dim)")
        if actions.shape != (self.n_agents,):
            raise ValueError("actions must have shape (n_agents,)")
        if rewards.shape != (self.n_agents,):
            raise ValueError("rewards must have shape (n_agents,)")

        idx = self._ptr
        self._obs[idx] = obs
        self._actions[idx] = actions
        self._rewards[idx] = rewards
        self._next_obs[idx] = next_obs
        self._dones[idx] = float(done)
        if states is None or next_states is None:
            if not self._state_dim_from_obs:
                raise ValueError("states and next_states must be provided when state_dim differs from n_agents * obs_dim")
            states = obs.reshape(-1)
            next_states = next_obs.reshape(-1)
        if states.shape != (self.state_dim,) or next_states.shape != (self.state_dim,):
            raise ValueError("states and next_states must have shape (state_dim,)")
        self._states[idx] = states
        self._next_states[idx] = next_states

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
        *,
        states: Optional[np.ndarray] = None,
        next_states: Optional[np.ndarray] = None,
    ) -> None:
        if obs.ndim != 3:
            raise ValueError("obs must have shape (batch, n_agents, obs_dim)")
        if next_obs.ndim != 3:
            raise ValueError("next_obs must have shape (batch, n_agents, obs_dim)")
        if actions.ndim != 2:
            raise ValueError("actions must have shape (batch, n_agents)")
        if rewards.ndim != 2:
            raise ValueError("rewards must have shape (batch, n_agents)")
        if dones.ndim != 1:
            raise ValueError("dones must have shape (batch,)")

        batch_size = int(obs.shape[0])
        if batch_size == 0:
            return
        if obs.shape[1:] != (self.n_agents, self.obs_dim):
            raise ValueError("obs must have shape (batch, n_agents, obs_dim)")
        if next_obs.shape[1:] != (self.n_agents, self.obs_dim):
            raise ValueError("next_obs must have shape (batch, n_agents, obs_dim)")
        if actions.shape[1] != self.n_agents:
            raise ValueError("actions must have shape (batch, n_agents)")
        if rewards.shape[1] != self.n_agents:
            raise ValueError("rewards must have shape (batch, n_agents)")
        if dones.shape[0] != batch_size:
            raise ValueError("dones must have shape (batch,)")

        idx = self._ptr
        end = idx + batch_size
        obs_flat = obs.reshape(batch_size, -1)
        next_obs_flat = next_obs.reshape(batch_size, -1)
        if states is None or next_states is None:
            if not self._state_dim_from_obs:
                raise ValueError("states and next_states must be provided when state_dim differs from n_agents * obs_dim")
            states = obs_flat
            next_states = next_obs_flat
        if states.shape != (batch_size, self.state_dim) or next_states.shape != (batch_size, self.state_dim):
            raise ValueError("states and next_states must have shape (batch, state_dim)")

        if end <= self.capacity:
            self._obs[idx:end] = obs
            self._actions[idx:end] = actions
            self._rewards[idx:end] = rewards
            self._next_obs[idx:end] = next_obs
            self._dones[idx:end] = dones.astype(np.float32)
            self._states[idx:end] = states
            self._next_states[idx:end] = next_states
        else:
            first = self.capacity - idx
            second = batch_size - first
            self._obs[idx:] = obs[:first]
            self._actions[idx:] = actions[:first]
            self._rewards[idx:] = rewards[:first]
            self._next_obs[idx:] = next_obs[:first]
            self._dones[idx:] = dones[:first].astype(np.float32)
            self._states[idx:] = states[:first]
            self._next_states[idx:] = next_states[:first]

            self._obs[:second] = obs[first:]
            self._actions[:second] = actions[first:]
            self._rewards[:second] = rewards[first:]
            self._next_obs[:second] = next_obs[first:]
            self._dones[:second] = dones[first:].astype(np.float32)
            self._states[:second] = states[first:]
            self._next_states[:second] = next_states[first:]

        self._ptr = end % self.capacity
        self._size = min(self._size + batch_size, self.capacity)

    def sample(self, batch_size: int, *, obs_normalizer: Optional[RunningObsNormalizer] = None) -> MARLBatch:
        if self._size == 0:
            raise RuntimeError("Cannot sample from an empty buffer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = self.rng.integers(0, self._size, size=int(batch_size), endpoint=False)
        obs = self._obs[indices]
        next_obs = self._next_obs[indices]
        if obs_normalizer is not None:
            # Normalize on sample to keep stats consistent across the batch.
            obs = obs_normalizer.normalize(obs, update=False)
            next_obs = obs_normalizer.normalize(next_obs, update=False)
            if self._state_dim_from_obs:
                states = obs.reshape(obs.shape[0], -1)
                next_states = next_obs.reshape(next_obs.shape[0], -1)
            else:
                states = self._states[indices]
                next_states = self._next_states[indices]
        else:
            states = self._states[indices]
            next_states = self._next_states[indices]

        return MARLBatch(
            obs=torch.as_tensor(obs, device=self.device, dtype=torch.float32),
            actions=torch.as_tensor(self._actions[indices], device=self.device, dtype=torch.long),
            rewards=torch.as_tensor(self._rewards[indices], device=self.device, dtype=torch.float32),
            next_obs=torch.as_tensor(next_obs, device=self.device, dtype=torch.float32),
            dones=torch.as_tensor(self._dones[indices], device=self.device, dtype=torch.float32),
            states=torch.as_tensor(states, device=self.device, dtype=torch.float32),
            next_states=torch.as_tensor(next_states, device=self.device, dtype=torch.float32),
        )
