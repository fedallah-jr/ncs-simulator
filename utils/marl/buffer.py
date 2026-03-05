from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

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
    gamma_n: torch.Tensor  # per-sample effective discount gamma^actual_n


@dataclass
class _NStepTransition:
    """Single transition stored in the n-step deque."""
    obs: np.ndarray       # (n_agents, obs_dim)
    actions: np.ndarray   # (n_agents,)
    rewards: np.ndarray   # (n_agents,)
    next_obs: np.ndarray  # (n_agents, obs_dim)
    done: float
    states: np.ndarray    # (state_dim,)
    next_states: np.ndarray  # (state_dim,)


class NStepHelper:
    """Accumulates n-step returns per parallel env.

    On each ``add_batch``, transitions are pushed into per-env deques.
    When an env resets (``resets[e]`` is True) or a deque reaches ``n_step``
    length, completed n-step transitions are emitted.
    """

    def __init__(self, n_step: int, gamma: float, n_envs: int) -> None:
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.n_envs = int(n_envs)
        self._deques: List[deque] = [deque() for _ in range(n_envs)]

    def _flush_deque(self, dq: deque) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                      np.ndarray, float, np.ndarray,
                                                      np.ndarray, float]]:
        """Flush all pending transitions from a deque, computing n-step returns."""
        results = []
        L = len(dq)
        for k in range(L):
            actual_n = L - k
            # Accumulate discounted rewards
            reward_accum = np.zeros_like(dq[k].rewards)
            for j in range(k, L):
                reward_accum += (self.gamma ** (j - k)) * dq[j].rewards
            gamma_n = self.gamma ** actual_n
            results.append((
                dq[k].obs, dq[k].actions, reward_accum,
                dq[L - 1].next_obs, dq[L - 1].done,
                dq[k].states, dq[L - 1].next_states,
                gamma_n,
            ))
        dq.clear()
        return results

    def add_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
        states: np.ndarray,
        next_states: np.ndarray,
        resets: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                     float, np.ndarray, np.ndarray, float]]:
        """Process a batch of transitions and return completed n-step transitions."""
        n_envs = obs.shape[0]
        emitted = []
        for e in range(n_envs):
            t = _NStepTransition(
                obs=obs[e], actions=actions[e], rewards=rewards[e],
                next_obs=next_obs[e], done=dones[e],
                states=states[e], next_states=next_states[e],
            )
            dq = self._deques[e]
            dq.append(t)

            if resets[e]:
                # Episode boundary: flush everything
                emitted.extend(self._flush_deque(dq))
            elif len(dq) >= self.n_step:
                # Deque full: emit oldest with full n-step return
                oldest = dq[0]
                reward_accum = np.zeros_like(oldest.rewards)
                for j in range(self.n_step):
                    reward_accum += (self.gamma ** j) * dq[j].rewards
                emitted.append((
                    oldest.obs, oldest.actions, reward_accum,
                    dq[self.n_step - 1].next_obs, dq[self.n_step - 1].done,
                    oldest.states, dq[self.n_step - 1].next_states,
                    self.gamma ** self.n_step,
                ))
                dq.popleft()
        return emitted

    def state_dict(self) -> dict:
        """Serialize deque state for checkpointing."""
        deques_data = []
        for dq in self._deques:
            dq_items = []
            for t in dq:
                dq_items.append({
                    "obs": t.obs.copy(), "actions": t.actions.copy(),
                    "rewards": t.rewards.copy(), "next_obs": t.next_obs.copy(),
                    "done": t.done, "states": t.states.copy(),
                    "next_states": t.next_states.copy(),
                })
            deques_data.append(dq_items)
        return {"n_step": self.n_step, "gamma": self.gamma, "deques": deques_data}

    def load_state_dict(self, d: dict) -> None:
        """Restore deque state from checkpoint."""
        for e, dq_items in enumerate(d["deques"]):
            self._deques[e].clear()
            for item in dq_items:
                self._deques[e].append(_NStepTransition(
                    obs=item["obs"], actions=item["actions"],
                    rewards=item["rewards"], next_obs=item["next_obs"],
                    done=item["done"], states=item["states"],
                    next_states=item["next_states"],
                ))


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
        n_step: int = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
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

        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self._nstep_helper: Optional[NStepHelper] = None
        if self.n_step > 1:
            self._nstep_helper = NStepHelper(self.n_step, self.gamma, int(n_envs))

        self._ptr = 0
        self._size = 0

        self._obs = np.zeros((self.capacity, self.n_agents, self.obs_dim), dtype=np.float32)
        self._actions = np.zeros((self.capacity, self.n_agents), dtype=np.int64)
        self._rewards = np.zeros((self.capacity, self.n_agents), dtype=np.float32)
        self._next_obs = np.zeros((self.capacity, self.n_agents, self.obs_dim), dtype=np.float32)
        self._dones = np.zeros((self.capacity,), dtype=np.float32)
        self._states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self._next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self._gamma_n = np.full((self.capacity,), self.gamma, dtype=np.float32)

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

    def _store_single(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        done: float,
        states: np.ndarray,
        next_states: np.ndarray,
        gamma_n: float,
    ) -> None:
        """Store a single transition at the current pointer."""
        idx = self._ptr
        self._obs[idx] = obs
        self._actions[idx] = actions
        self._rewards[idx] = rewards
        self._next_obs[idx] = next_obs
        self._dones[idx] = done
        self._states[idx] = states
        self._next_states[idx] = next_states
        self._gamma_n[idx] = gamma_n
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
        resets: Optional[np.ndarray] = None,
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

        obs_flat = obs.reshape(batch_size, -1)
        next_obs_flat = next_obs.reshape(batch_size, -1)
        if states is None or next_states is None:
            if not self._state_dim_from_obs:
                raise ValueError("states and next_states must be provided when state_dim differs from n_agents * obs_dim")
            states = obs_flat
            next_states = next_obs_flat
        if states.shape != (batch_size, self.state_dim) or next_states.shape != (batch_size, self.state_dim):
            raise ValueError("states and next_states must have shape (batch, state_dim)")

        # N-step path: feed transitions through NStepHelper
        if self._nstep_helper is not None:
            if resets is None:
                raise ValueError("resets must be provided when n_step > 1")
            emitted = self._nstep_helper.add_batch(
                obs=obs, actions=actions, rewards=rewards,
                next_obs=next_obs, dones=dones.astype(np.float32),
                states=states, next_states=next_states,
                resets=resets,
            )
            for (e_obs, e_act, e_rew, e_nobs, e_done,
                 e_st, e_nst, e_gamma_n) in emitted:
                self._store_single(e_obs, e_act, e_rew, e_nobs,
                                   e_done, e_st, e_nst, e_gamma_n)
            return

        # Standard 1-step path (backward compatible)
        idx = self._ptr
        end = idx + batch_size
        gamma_n_arr = np.full((batch_size,), self.gamma, dtype=np.float32)

        if end <= self.capacity:
            self._obs[idx:end] = obs
            self._actions[idx:end] = actions
            self._rewards[idx:end] = rewards
            self._next_obs[idx:end] = next_obs
            self._dones[idx:end] = dones.astype(np.float32)
            self._states[idx:end] = states
            self._next_states[idx:end] = next_states
            self._gamma_n[idx:end] = gamma_n_arr
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
            self._gamma_n[idx:] = gamma_n_arr[:first]

            self._obs[:second] = obs[first:]
            self._actions[:second] = actions[first:]
            self._rewards[:second] = rewards[first:]
            self._next_obs[:second] = next_obs[first:]
            self._dones[:second] = dones[first:].astype(np.float32)
            self._states[:second] = states[first:]
            self._next_states[:second] = next_states[first:]
            self._gamma_n[:second] = gamma_n_arr[first:]

        self._ptr = end % self.capacity
        self._size = min(self._size + batch_size, self.capacity)

    def state_dict(self) -> dict:
        n = self._size
        d = {
            "obs": self._obs[:n].copy(),
            "actions": self._actions[:n].copy(),
            "rewards": self._rewards[:n].copy(),
            "next_obs": self._next_obs[:n].copy(),
            "dones": self._dones[:n].copy(),
            "states": self._states[:n].copy(),
            "next_states": self._next_states[:n].copy(),
            "gamma_n": self._gamma_n[:n].copy(),
            "_ptr": self._ptr,
            "_size": self._size,
        }
        if self._nstep_helper is not None:
            d["nstep_helper"] = self._nstep_helper.state_dict()
        return d

    def load_state_dict(self, d: dict) -> None:
        n = int(d["_size"])
        self._obs[:n] = d["obs"][:n]
        self._actions[:n] = d["actions"][:n]
        self._rewards[:n] = d["rewards"][:n]
        self._next_obs[:n] = d["next_obs"][:n]
        self._dones[:n] = d["dones"][:n]
        self._states[:n] = d["states"][:n]
        self._next_states[:n] = d["next_states"][:n]
        # Backward compat: old checkpoints won't have gamma_n
        if "gamma_n" in d:
            self._gamma_n[:n] = d["gamma_n"][:n]
        else:
            self._gamma_n[:n] = self.gamma
        self._ptr = int(d["_ptr"])
        self._size = n
        # Restore NStepHelper if present
        if self._nstep_helper is not None and "nstep_helper" in d:
            self._nstep_helper.load_state_dict(d["nstep_helper"])

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
            gamma_n=torch.as_tensor(self._gamma_n[indices], device=self.device, dtype=torch.float32),
        )
