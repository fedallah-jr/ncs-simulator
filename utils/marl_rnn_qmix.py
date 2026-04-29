"""Isolated RNN + QMIX implementation.

This module is intentionally self-contained so the entire feature can be
removed by deleting two files: this file and ``algorithms/marl_rnn_qmix.py``.
It reuses stable, algorithm-agnostic infrastructure from ``utils/marl`` (the
QMIX/VDN mixer modules and generic env helpers) but does not add or modify
anything in those shared files.

Architecture mirrors ``NDQRNNAgent`` minus the NDQ communication additions:

    obs_encoder(obs) + agent_lookup(agent_id) + prev_action_lookup(prev_action)
    -> GRU -> Q-head

No message encoder, no differentiable comm, no comm-inference head. The
learner is a proper DRQN + monotonic-mixer pipeline: episode replay buffer,
full-trajectory RNN unroll at training time, target network + (optional)
double-Q / TD(lambda). The QMIX mixer receives the env's ``global_state``
channel (same signal ``algorithms/marl_qmix.py`` feeds to ``QMixer``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Tuple

import numpy as np
import torch
from torch import nn

from utils.marl.common import epsilon_by_step, patch_autoreset_final_obs, stack_obs
from utils.marl.networks import QMixer, VDNMixer
from utils.marl.obs_normalization import RunningObsNormalizer
from utils.marl.torch_policy import MARLTorchCheckpointMetadata
from utils.marl.vector_env import stack_vector_obs


# ---------------------------------------------------------------------------
# Agent network
# ---------------------------------------------------------------------------


class RNNAgent(nn.Module):
    """GRU-based shared agent (NDQ architecture without the comm encoder)."""

    def __init__(
        self,
        obs_dim: int,
        n_agents: int,
        n_actions: int,
        rnn_hidden_dim: int = 64,
        rnn_layers: int = 1,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_agents = int(n_agents)
        self.n_actions = int(n_actions)
        self.rnn_hidden_dim = int(rnn_hidden_dim)
        self.rnn_layers = int(rnn_layers)

        self.obs_encoder = nn.Linear(obs_dim, rnn_hidden_dim)
        self.agent_lookup = nn.Embedding(n_agents, rnn_hidden_dim)
        # +1 for the start-of-episode sentinel (matches NDQ convention).
        self.prev_action_lookup = nn.Embedding(n_actions + 1, rnn_hidden_dim)
        self.rnn = nn.GRU(
            input_size=rnn_hidden_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
        )
        self.output_head = nn.Sequential(
            nn.Linear(rnn_hidden_dim, rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(rnn_hidden_dim, n_actions),
        )

    def forward(
        self,
        obs: torch.Tensor,
        agent_idx: torch.Tensor,
        prev_action: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = (
            self.obs_encoder(obs)
            + self.agent_lookup(agent_idx)
            + self.prev_action_lookup(prev_action)
        )
        rnn_out, h_out = self.rnn(z.unsqueeze(1), hidden)
        q_values = self.output_head(rnn_out.squeeze(1))
        return q_values, h_out

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(self.rnn_layers, batch_size, self.rnn_hidden_dim)


def rnn_forward_batched(
    agent: RNNAgent,
    obs: torch.Tensor,
    prev_action: torch.Tensor,
    hidden: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run one recurrent step over a vectorized batch.

    Args:
        obs:         (B, N, obs_dim)
        prev_action: (B, N)                  int64; ``n_actions`` = start token
        hidden:      (rnn_layers, B * N, rnn_hidden_dim)
    Returns:
        q_values:    (B, N, n_actions)
        h_out:       (rnn_layers, B * N, rnn_hidden_dim)
    """
    if obs.ndim != 3:
        raise ValueError("obs must have shape (B, N, obs_dim)")
    if prev_action.shape != obs.shape[:2]:
        raise ValueError("prev_action must have shape (B, N)")

    batch_size, n_agents, _ = obs.shape
    agent_idx = torch.arange(
        n_agents, device=obs.device, dtype=torch.long,
    ).unsqueeze(0).expand(batch_size, -1).reshape(batch_size * n_agents)
    q_values, h_out = agent(
        obs.reshape(batch_size * n_agents, -1),
        agent_idx,
        prev_action.reshape(batch_size * n_agents),
        hidden,
    )
    return q_values.view(batch_size, n_agents, -1), h_out


# ---------------------------------------------------------------------------
# Episode batch / collector / buffer (carry the env's global_state)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RNNQMIXEpisodeBatch:
    obs: torch.Tensor            # (B, max_T, N, obs_dim)
    actions: torch.Tensor        # (B, max_T, N)           int64
    rewards: torch.Tensor        # (B, max_T, N)           float32
    terminated: torch.Tensor     # (B, max_T)              float32
    mask: torch.Tensor           # (B, max_T)              float32
    next_obs: torch.Tensor       # (B, N, obs_dim)         bootstrap obs
    states: torch.Tensor         # (B, max_T, state_dim)
    next_states: torch.Tensor    # (B, state_dim)          bootstrap state


class RNNQMIXEpisodeCollector:
    """Accumulate complete episodes from vectorized envs (with global state)."""

    def __init__(self, n_envs: int, device: torch.device) -> None:
        self.n_envs = n_envs
        self.device = device
        self._buffers: List[List[dict]] = [[] for _ in range(n_envs)]
        self._completed: List[dict] = []

    def add(self, env_idx: int, transition: dict) -> None:
        self._buffers[env_idx].append(transition)
        if transition.get("reset", transition["done"]):
            self._completed.append(self._finalize(env_idx))
            self._buffers[env_idx] = []

    def _finalize(self, env_idx: int) -> dict:
        buf = self._buffers[env_idx]
        return {
            "obs": np.stack([t["obs"] for t in buf]),
            "actions": np.stack([t["actions"] for t in buf]),
            "rewards": np.stack([t["rewards"] for t in buf]),
            "terminated": np.array([t["done"] for t in buf], dtype=np.float32),
            "next_obs": buf[-1]["next_obs"],
            "states": np.stack([t["state"] for t in buf]),
            "next_state": buf[-1]["next_state"],
        }

    def __len__(self) -> int:
        return len(self._completed)

    def state_dict(self) -> dict:
        serialized_buffers = [
            [
                {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in t.items()}
                for t in buf
            ]
            for buf in self._buffers
        ]
        serialized_completed = [
            {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in ep.items()}
            for ep in self._completed
        ]
        return {"buffers": serialized_buffers, "completed": serialized_completed}

    def load_state_dict(self, d: dict) -> None:
        self._buffers = d.get("buffers", [[] for _ in range(self.n_envs)])
        self._completed = d.get("completed", [])


class RNNQMIXEpisodeReplayBuffer:
    """Circular episode replay buffer with uniform random sampling.

    Stores finalized episode dicts produced by :class:`RNNQMIXEpisodeCollector`
    and samples random batches, returning a :class:`RNNQMIXEpisodeBatch` with
    padding and masking. Carries per-step global state plus the bootstrap
    state for the last transition of each episode.
    """

    def __init__(
        self,
        capacity: int,
        device: torch.device,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.device = device
        self.rng = rng if rng is not None else np.random.default_rng()
        self._storage: List[dict] = []
        self._ptr: int = 0

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, episode: dict) -> None:
        if len(self._storage) < self.capacity:
            self._storage.append(episode)
        else:
            self._storage[self._ptr] = episode
        self._ptr = (self._ptr + 1) % self.capacity

    def add_episodes(self, episodes: List[dict]) -> None:
        for ep in episodes:
            self.add(ep)

    def can_sample(self, batch_size: int) -> bool:
        return len(self._storage) >= batch_size

    def sample(
        self,
        batch_size: int,
        obs_normalizer: Optional[RunningObsNormalizer] = None,
    ) -> RNNQMIXEpisodeBatch:
        if not self.can_sample(batch_size):
            raise RuntimeError(
                f"Cannot sample {batch_size} episodes from buffer with {len(self._storage)}"
            )
        indices = self.rng.choice(len(self._storage), size=batch_size, replace=False)
        episodes = [self._storage[i] for i in indices]

        lengths = [ep["obs"].shape[0] for ep in episodes]
        max_T = max(lengths)
        B = len(episodes)

        sample_ep = episodes[0]
        N, obs_dim = sample_ep["obs"].shape[1], sample_ep["obs"].shape[2]
        state_dim = sample_ep["states"].shape[-1]

        obs = np.zeros((B, max_T, N, obs_dim), dtype=np.float32)
        actions = np.zeros((B, max_T, N), dtype=np.int64)
        rewards = np.zeros((B, max_T, N), dtype=np.float32)
        terminated = np.zeros((B, max_T), dtype=np.float32)
        mask = np.zeros((B, max_T), dtype=np.float32)
        states = np.zeros((B, max_T, state_dim), dtype=np.float32)
        next_obs = np.stack([ep["next_obs"] for ep in episodes])
        next_states = np.stack([ep["next_state"] for ep in episodes])

        for i, ep in enumerate(episodes):
            T = lengths[i]
            obs[i, :T] = ep["obs"]
            actions[i, :T] = ep["actions"]
            rewards[i, :T] = ep["rewards"]
            terminated[i, :T] = ep["terminated"]
            states[i, :T] = ep["states"]
            mask[i, :T] = 1.0

        if obs_normalizer is not None:
            obs = obs_normalizer.normalize(
                obs.reshape(B * max_T * N, obs_dim), update=False,
            ).reshape(B, max_T, N, obs_dim)
            next_obs = obs_normalizer.normalize(
                next_obs.reshape(B * N, obs_dim), update=False,
            ).reshape(B, N, obs_dim)

        return RNNQMIXEpisodeBatch(
            obs=torch.as_tensor(obs, device=self.device, dtype=torch.float32),
            actions=torch.as_tensor(actions, device=self.device, dtype=torch.long),
            rewards=torch.as_tensor(rewards, device=self.device, dtype=torch.float32),
            terminated=torch.as_tensor(terminated, device=self.device, dtype=torch.float32),
            mask=torch.as_tensor(mask, device=self.device, dtype=torch.float32),
            next_obs=torch.as_tensor(next_obs, device=self.device, dtype=torch.float32),
            states=torch.as_tensor(states, device=self.device, dtype=torch.float32),
            next_states=torch.as_tensor(next_states, device=self.device, dtype=torch.float32),
        )

    def state_dict(self) -> dict:
        serialized = [
            {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in ep.items()}
            for ep in self._storage
        ]
        return {"storage": serialized, "ptr": self._ptr}

    def load_state_dict(self, d: dict) -> None:
        self._storage = d.get("storage", [])
        self._ptr = int(d.get("ptr", 0)) % max(self.capacity, 1)


# ---------------------------------------------------------------------------
# Learner
# ---------------------------------------------------------------------------


def _hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def _make_optimizer(
    params,
    lr: float,
    optimizer_type: str,
    rmsprop_alpha: float,
    rmsprop_eps: float,
    momentum: float,
) -> torch.optim.Optimizer:
    if optimizer_type == "rmsprop":
        return torch.optim.RMSprop(
            params, lr=lr, alpha=rmsprop_alpha, eps=rmsprop_eps, momentum=momentum,
        )
    return torch.optim.Adam(params, lr=lr)


def _apply_td_lambda(
    targets: List[torch.Tensor],
    rewards: List[torch.Tensor],
    mask: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float,
    td_lambda: float,
) -> List[torch.Tensor]:
    """Backward recursion G^lambda_t = (1-lambda)*T_t + lambda*(r_t + gamma*(1-d_t)*G^lambda_{t+1})."""
    T = len(targets)
    ret = [t.clone() for t in targets]
    for t in range(T - 2, -1, -1):
        not_done = 1.0 - terminated[:, t]
        valid_next = mask[:, t + 1]
        gate = not_done * valid_next
        shape = [gate.shape[0]] + [1] * (targets[t].ndim - 1)
        gate = gate.view(*shape)
        ret[t] = (1.0 - td_lambda) * targets[t] + td_lambda * (
            rewards[t] + gamma * gate * ret[t + 1]
        )
    return ret


class MARLRNNQMIXLearner:
    """Recurrent Q-learner with QMIX / VDN / per-agent IQL mixing.

    QMIX consumes the env's ``global_state`` directly (passed via
    ``batch.states`` / ``batch.next_states``). The mixer's ``state_dim`` must
    be provided at construction time.
    """

    def __init__(
        self,
        agent: RNNAgent,
        n_agents: int,
        n_actions: int,
        gamma: float,
        lr: float,
        mixer_type: str = "qmix",
        state_dim: int = 0,
        qmix_mixing_hidden_dim: int = 32,
        qmix_hypernet_hidden_dim: int = 64,
        target_update_steps: int = 25,
        grad_clip_norm: Optional[float] = 10.0,
        td_lambda: float = 0.0,
        double_q: bool = True,
        device: Optional[torch.device] = None,
        optimizer_type: str = "rmsprop",
        momentum: float = 0.0,
        rmsprop_alpha: float = 0.99,
        rmsprop_eps: float = 1e-5,
    ) -> None:
        td_lambda = float(td_lambda)
        if not 0.0 <= td_lambda <= 1.0:
            raise ValueError("td_lambda must be in [0, 1]")

        self.agent = agent
        self.target_agent = copy.deepcopy(agent)
        for p in self.target_agent.parameters():
            p.requires_grad = False

        self.n_agents = int(n_agents)
        self.n_actions = int(n_actions)
        self.state_dim = int(state_dim)
        self.gamma = float(gamma)
        self.mixer_type = str(mixer_type)
        self.grad_clip_norm = grad_clip_norm
        self.target_update_steps = int(target_update_steps)
        self.td_lambda = td_lambda
        self.double_q = bool(double_q)

        self.device = device if device is not None else torch.device("cpu")
        self.agent.to(self.device)
        self.target_agent.to(self.device)

        self.mixer: Optional[nn.Module] = None
        self.target_mixer: Optional[nn.Module] = None
        if self.mixer_type == "vdn":
            self.mixer = VDNMixer().to(self.device)
            self.target_mixer = VDNMixer().to(self.device)
        elif self.mixer_type == "qmix":
            if self.state_dim <= 0:
                raise ValueError("QMIX mixer requires state_dim > 0")
            self.mixer = QMixer(
                n_agents=self.n_agents,
                state_dim=self.state_dim,
                mixing_hidden_dim=qmix_mixing_hidden_dim,
                hypernet_hidden_dim=qmix_hypernet_hidden_dim,
            ).to(self.device)
            self.target_mixer = copy.deepcopy(self.mixer).to(self.device)
            for p in self.target_mixer.parameters():
                p.requires_grad = False
        elif self.mixer_type != "none":
            raise ValueError("mixer_type must be one of: none, vdn, qmix")

        self._all_params = list(self.agent.parameters())
        if self.mixer is not None:
            self._all_params += list(self.mixer.parameters())
        self.optimizer = _make_optimizer(
            self._all_params,
            lr=float(lr),
            optimizer_type=optimizer_type,
            rmsprop_alpha=float(rmsprop_alpha),
            rmsprop_eps=float(rmsprop_eps),
            momentum=float(momentum),
        )

        self.episodes_seen = 0
        self._train_steps = 0

    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "agent": self.agent.state_dict(),
            "target_agent": self.target_agent.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "episodes_seen": self.episodes_seen,
            "train_steps": self._train_steps,
        }
        if self.mixer is not None:
            state["mixer"] = self.mixer.state_dict()
        if self.target_mixer is not None:
            state["target_mixer"] = self.target_mixer.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.agent.load_state_dict(state["agent"])
        self.target_agent.load_state_dict(state["target_agent"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.episodes_seen = int(state.get("episodes_seen", 0))
        self._train_steps = int(state.get("train_steps", 0))
        if self.mixer is not None and "mixer" in state:
            self.mixer.load_state_dict(state["mixer"])
        if self.target_mixer is not None and "target_mixer" in state:
            self.target_mixer.load_state_dict(state["target_mixer"])

    # ------------------------------------------------------------------
    def _forward_trace(
        self,
        agent: RNNAgent,
        batch: RNNQMIXEpisodeBatch,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        batch_size, max_t, n_agents = batch.obs.shape[:3]
        hidden = agent.init_hidden(batch_size * n_agents).to(self.device)
        prev_action = torch.full(
            (batch_size, n_agents), self.n_actions, dtype=torch.long, device=self.device,
        )

        all_q: List[torch.Tensor] = []
        for t in range(max_t):
            q_t, new_hidden = rnn_forward_batched(
                agent, batch.obs[:, t], prev_action, hidden,
            )
            all_q.append(q_t)

            valid = batch.mask[:, t].bool()
            valid_bn = valid.unsqueeze(1).expand(batch_size, n_agents).reshape(batch_size * n_agents)
            v_h = valid_bn.view(1, batch_size * n_agents, 1).expand_as(new_hidden)
            hidden = torch.where(v_h, new_hidden, hidden)
            v_a = valid.unsqueeze(1).expand(batch_size, n_agents)
            prev_action = torch.where(v_a, batch.actions[:, t], prev_action)

        q_boot, _ = rnn_forward_batched(agent, batch.next_obs, prev_action, hidden)
        return all_q, q_boot

    # ------------------------------------------------------------------
    def update(self, batch: RNNQMIXEpisodeBatch) -> None:
        online_q, online_q_boot = self._forward_trace(self.agent, batch)
        with torch.no_grad():
            target_q, target_q_boot = self._forward_trace(self.target_agent, batch)

        td_loss = self._compute_td_loss(
            batch, online_q, online_q_boot, target_q, target_q_boot,
        )

        self.optimizer.zero_grad(set_to_none=True)
        td_loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self._all_params, float(self.grad_clip_norm))
        self.optimizer.step()

        self.episodes_seen += batch.obs.shape[0]
        self._train_steps += 1
        if self._train_steps % self.target_update_steps == 0:
            _hard_update(self.target_agent, self.agent)
            if self.target_mixer is not None and self.mixer is not None:
                _hard_update(self.target_mixer, self.mixer)

    # ------------------------------------------------------------------
    def _compute_td_loss(
        self,
        batch: RNNQMIXEpisodeBatch,
        online_q_list: List[torch.Tensor],
        online_q_boot: torch.Tensor,
        target_q_list: List[torch.Tensor],
        target_q_boot: torch.Tensor,
    ) -> torch.Tensor:
        if self.mixer_type == "vdn":
            return self._compute_vdn_td_loss(
                batch, online_q_list, online_q_boot, target_q_list, target_q_boot,
            )
        if self.mixer_type == "qmix":
            return self._compute_qmix_td_loss(
                batch, online_q_list, online_q_boot, target_q_list, target_q_boot,
            )
        return self._compute_iql_td_loss(
            batch, online_q_list, online_q_boot, target_q_list, target_q_boot,
        )

    def _bootstrap_actions(
        self,
        online_q_boot: torch.Tensor,
        target_q_boot: torch.Tensor,
    ) -> torch.Tensor:
        if self.double_q:
            boot_actions = online_q_boot.argmax(dim=-1, keepdim=True)
            return target_q_boot.gather(-1, boot_actions).squeeze(-1)
        return target_q_boot.max(dim=-1).values

    def _next_q(
        self,
        online_q_next: torch.Tensor,
        target_q_next: torch.Tensor,
    ) -> torch.Tensor:
        if self.double_q:
            next_actions = online_q_next.argmax(dim=-1, keepdim=True)
            return target_q_next.gather(-1, next_actions).squeeze(-1)
        return target_q_next.max(dim=-1).values

    def _compute_iql_td_loss(
        self,
        batch: RNNQMIXEpisodeBatch,
        online_q_list: List[torch.Tensor],
        online_q_boot: torch.Tensor,
        target_q_list: List[torch.Tensor],
        target_q_boot: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_t = batch.obs.shape[:2]
        n_agents = self.n_agents

        with torch.no_grad():
            boot_q = self._bootstrap_actions(online_q_boot, target_q_boot)
            targets: List[torch.Tensor] = []
            rewards: List[torch.Tensor] = []
            for t in range(max_t):
                if t < max_t - 1:
                    has_next = batch.mask[:, t + 1].bool()
                    next_q_vals = self._next_q(online_q_list[t + 1], target_q_list[t + 1])
                    bootstrap_q = torch.where(
                        has_next.unsqueeze(1).expand(batch_size, n_agents),
                        next_q_vals,
                        boot_q,
                    )
                else:
                    bootstrap_q = boot_q
                rewards.append(batch.rewards[:, t])
                not_done = (1.0 - batch.terminated[:, t]).unsqueeze(-1)
                targets.append(rewards[-1] + self.gamma * not_done * bootstrap_q)
            if self.td_lambda > 0:
                targets = _apply_td_lambda(
                    targets, rewards, batch.mask, batch.terminated, self.gamma, self.td_lambda,
                )

        total_loss = torch.tensor(0.0, device=self.device)
        for t in range(max_t):
            q_taken = online_q_list[t].gather(-1, batch.actions[:, t].unsqueeze(-1)).squeeze(-1)
            td_error = (q_taken - targets[t]) ** 2
            total_loss = total_loss + (td_error * batch.mask[:, t].unsqueeze(-1)).sum()
        return total_loss / (batch.mask.sum() * float(n_agents)).clamp_min(1.0)

    def _compute_vdn_td_loss(
        self,
        batch: RNNQMIXEpisodeBatch,
        online_q_list: List[torch.Tensor],
        online_q_boot: torch.Tensor,
        target_q_list: List[torch.Tensor],
        target_q_boot: torch.Tensor,
    ) -> torch.Tensor:
        assert self.mixer is not None and self.target_mixer is not None
        batch_size, max_t = batch.obs.shape[:2]

        with torch.no_grad():
            boot_q = self._bootstrap_actions(online_q_boot, target_q_boot)
            targets: List[torch.Tensor] = []
            rewards: List[torch.Tensor] = []
            for t in range(max_t):
                if t < max_t - 1:
                    has_next = batch.mask[:, t + 1].bool()
                    next_q_vals = self._next_q(online_q_list[t + 1], target_q_list[t + 1])
                    bootstrap_q = torch.where(
                        has_next.unsqueeze(1).expand(batch_size, self.n_agents),
                        next_q_vals,
                        boot_q,
                    )
                else:
                    bootstrap_q = boot_q
                next_q_tot = self.target_mixer(bootstrap_q)
                # rnn_collect_transition stores the team-summed reward
                # replicated across the n_agents slots, so taking the first
                # slot recovers the team scalar. Summing across the agent
                # dim would multiply it by n_agents (matches NDQ /
                # MARLDIALLearner._dial_team_reward).
                r_tot = batch.rewards[:, t, :1]
                rewards.append(r_tot)
                not_done = (1.0 - batch.terminated[:, t]).unsqueeze(-1)
                targets.append(r_tot + self.gamma * not_done * next_q_tot)
            if self.td_lambda > 0:
                targets = _apply_td_lambda(
                    targets, rewards, batch.mask, batch.terminated, self.gamma, self.td_lambda,
                )

        total_loss = torch.tensor(0.0, device=self.device)
        for t in range(max_t):
            q_taken = online_q_list[t].gather(-1, batch.actions[:, t].unsqueeze(-1)).squeeze(-1)
            q_tot = self.mixer(q_taken)
            td_error = (q_tot - targets[t]) ** 2
            total_loss = total_loss + (td_error * batch.mask[:, t].unsqueeze(-1)).sum()
        return total_loss / batch.mask.sum().clamp_min(1.0)

    def _compute_qmix_td_loss(
        self,
        batch: RNNQMIXEpisodeBatch,
        online_q_list: List[torch.Tensor],
        online_q_boot: torch.Tensor,
        target_q_list: List[torch.Tensor],
        target_q_boot: torch.Tensor,
    ) -> torch.Tensor:
        assert self.mixer is not None and self.target_mixer is not None
        batch_size, max_t = batch.obs.shape[:2]
        states = batch.states  # (B, max_T, state_dim)
        state_dim = states.shape[-1]
        boot_state = batch.next_states  # (B, state_dim)

        with torch.no_grad():
            boot_q = self._bootstrap_actions(online_q_boot, target_q_boot)
            targets: List[torch.Tensor] = []
            rewards: List[torch.Tensor] = []
            for t in range(max_t):
                if t < max_t - 1:
                    has_next = batch.mask[:, t + 1].bool()
                    next_q_vals = self._next_q(online_q_list[t + 1], target_q_list[t + 1])
                    bootstrap_q = torch.where(
                        has_next.unsqueeze(1).expand(batch_size, self.n_agents),
                        next_q_vals,
                        boot_q,
                    )
                    next_state = torch.where(
                        has_next.view(batch_size, 1).expand(batch_size, state_dim),
                        states[:, t + 1],
                        boot_state,
                    )
                else:
                    bootstrap_q = boot_q
                    next_state = boot_state
                next_q_tot = self.target_mixer(bootstrap_q, next_state)
                # See note in _compute_vdn_td_loss: per-agent slots are
                # already team-replicated by rnn_collect_transition.
                r_tot = batch.rewards[:, t, :1]
                rewards.append(r_tot)
                not_done = (1.0 - batch.terminated[:, t]).unsqueeze(-1)
                targets.append(r_tot + self.gamma * not_done * next_q_tot)
            if self.td_lambda > 0:
                targets = _apply_td_lambda(
                    targets, rewards, batch.mask, batch.terminated, self.gamma, self.td_lambda,
                )

        total_loss = torch.tensor(0.0, device=self.device)
        for t in range(max_t):
            q_taken = online_q_list[t].gather(-1, batch.actions[:, t].unsqueeze(-1)).squeeze(-1)
            q_tot = self.mixer(q_taken, states[:, t])
            td_error = (q_tot - targets[t]) ** 2
            total_loss = total_loss + (td_error * batch.mask[:, t].unsqueeze(-1)).sum()
        return total_loss / batch.mask.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# Environment collection
# ---------------------------------------------------------------------------


class RNNStepResult(NamedTuple):
    obs: np.ndarray
    epsilon: float
    actions: np.ndarray
    next_obs_raw: np.ndarray
    next_global_state_raw: np.ndarray
    rewards_arr: np.ndarray
    terminated: np.ndarray
    done_reset: np.ndarray
    infos: Dict[str, Any]


def rnn_collect_transition(
    *,
    env: Any,
    agent: RNNAgent,
    obs_raw: np.ndarray,
    global_state_raw: np.ndarray,
    obs_normalizer: Optional[RunningObsNormalizer],
    global_step: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_steps: int,
    n_envs: int,
    n_agents: int,
    n_actions: int,
    rng: np.random.Generator,
    device: torch.device,
    prev_actions: np.ndarray,
    hidden_states: torch.Tensor,
    collector: RNNQMIXEpisodeCollector,
) -> RNNStepResult:
    """Collect one vectorized recurrent transition, carrying global state.

    ``global_state_raw`` is the state at step ``t`` (same semantics as
    ``marl_qmix.py``'s state buffer). The next-state is pulled from the
    env's info dict and patched through :func:`patch_autoreset_final_obs`
    to recover the true terminal state at episode boundaries.
    """
    if obs_normalizer is not None:
        obs = obs_normalizer.normalize(obs_raw, update=True)
    else:
        obs = obs_raw

    eps = epsilon_by_step(global_step, epsilon_start, epsilon_end, epsilon_decay_steps)

    with torch.no_grad():
        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
        prev_act_t = torch.as_tensor(prev_actions, device=device, dtype=torch.long)
        q_values, new_hidden = rnn_forward_batched(agent, obs_t, prev_act_t, hidden_states)

    greedy = q_values.argmax(dim=-1).cpu().numpy().astype(np.int64)
    actions = greedy.copy()
    explore_mask = rng.random((n_envs, n_agents)) < float(eps)
    if np.any(explore_mask):
        actions[explore_mask] = rng.integers(
            0, n_actions, size=int(explore_mask.sum()), endpoint=False, dtype=np.int64,
        )

    action_dict = {f"agent_{i}": actions[:, i] for i in range(n_agents)}
    next_obs_dict, rewards_arr, terminated, truncated, infos = env.step(action_dict)
    next_obs_raw = stack_vector_obs(next_obs_dict, n_agents)
    next_global_state_raw = np.asarray(infos.get("global_state"), dtype=np.float32)
    if next_global_state_raw.ndim != 2:
        raise ValueError("global_state must have shape (n_envs, state_dim)")

    terminated_any = np.asarray(terminated, dtype=np.bool_)
    truncated_any = np.asarray(truncated, dtype=np.bool_)
    done_reset = np.logical_or(terminated_any, truncated_any)

    next_obs_for_buffer, next_state_for_buffer = patch_autoreset_final_obs(
        next_obs_raw, infos, done_reset, n_agents,
        next_global_state_raw=next_global_state_raw,
    )
    if next_state_for_buffer is None:
        next_state_for_buffer = next_global_state_raw

    rewards_np = np.asarray(rewards_arr, dtype=np.float32)
    team_rewards = rewards_np.sum(axis=1, keepdims=True)
    rewards_team = np.repeat(team_rewards, n_agents, axis=1).astype(np.float32)

    for env_idx in range(n_envs):
        collector.add(env_idx, {
            "obs": obs_raw[env_idx],
            "actions": actions[env_idx],
            "rewards": rewards_team[env_idx],
            "next_obs": next_obs_for_buffer[env_idx],
            "state": global_state_raw[env_idx],
            "next_state": next_state_for_buffer[env_idx],
            "done": float(terminated_any[env_idx]),
            "reset": bool(done_reset[env_idx]),
        })

    prev_actions[:] = actions
    hidden_states.copy_(new_hidden)

    start_token = n_actions
    for env_idx in range(n_envs):
        if done_reset[env_idx]:
            prev_actions[env_idx] = start_token
            hidden_states[:, env_idx * n_agents : (env_idx + 1) * n_agents, :] = 0.0

    return RNNStepResult(
        obs=obs,
        epsilon=eps,
        actions=actions,
        next_obs_raw=next_obs_raw,
        next_global_state_raw=next_global_state_raw,
        rewards_arr=rewards_np,
        terminated=terminated_any,
        done_reset=done_reset,
        infos=infos,
    )


# ---------------------------------------------------------------------------
# Evaluation action selector
# ---------------------------------------------------------------------------


def make_rnn_eval_action_selector(
    agent: RNNAgent,
    n_eval_envs: int,
    n_agents: int,
    n_actions: int,
    device: torch.device,
    rnn_layers: int,
    rnn_hidden_dim: int,
) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[], None]]:
    """Return (action_selector, reset_fn) for use with ``run_evaluation_vectorized_seeded``."""
    hidden = torch.zeros(rnn_layers, n_eval_envs * n_agents, rnn_hidden_dim, device=device)
    prev_actions = np.full((n_eval_envs, n_agents), n_actions, dtype=np.int64)

    def reset_eval_state() -> None:
        hidden.zero_()
        prev_actions[:] = n_actions

    @torch.no_grad()
    def action_selector(obs: np.ndarray) -> np.ndarray:
        nonlocal hidden
        q_values, new_hidden = rnn_forward_batched(
            agent,
            torch.as_tensor(obs, device=device, dtype=torch.float32),
            torch.as_tensor(prev_actions, device=device, dtype=torch.long),
            hidden,
        )
        hidden = new_hidden
        actions = q_values.argmax(dim=-1).cpu().numpy().astype(np.int64)
        prev_actions[:] = actions
        return actions

    return action_selector, reset_eval_state


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_rnn_qmix_checkpoint(
    path: Path,
    n_agents: int,
    obs_dim: int,
    state_dim: int,
    n_actions: int,
    agent: RNNAgent,
    obs_normalizer: Optional[RunningObsNormalizer],
    rnn_hidden_dim: int,
    rnn_layers: int,
    mixer_type: str,
    qmix_mixing_hidden_dim: int,
    qmix_hypernet_hidden_dim: int,
    mixer: Optional[nn.Module] = None,
) -> None:
    ckpt: Dict[str, Any] = {
        "algorithm": "marl_rnn_qmix",
        "rnn_qmix": True,
        "n_agents": n_agents,
        "obs_dim": obs_dim,
        "state_dim": state_dim,
        "n_actions": n_actions,
        "use_agent_id": True,
        "parameter_sharing": True,
        "rnn_hidden_dim": rnn_hidden_dim,
        "rnn_layers": rnn_layers,
        "mixer_type": mixer_type,
        "qmix_mixing_hidden_dim": qmix_mixing_hidden_dim,
        "qmix_hypernet_hidden_dim": qmix_hypernet_hidden_dim,
        "agent_state_dict": agent.state_dict(),
    }
    if mixer is not None:
        ckpt["mixer_state_dict"] = mixer.state_dict()
    ckpt["obs_normalization"] = (
        obs_normalizer.state_dict() if obs_normalizer is not None else {"enabled": False}
    )
    torch.save(ckpt, path)


def save_rnn_qmix_training_state(
    path: Path,
    learner: MARLRNNQMIXLearner,
    obs_normalizer: Any,
    best_model_tracker: Any,
    global_step: int,
    episode: int,
    last_eval_step: int,
    vector_step: int,
    replay_buffer: Any = None,
) -> None:
    state: Dict[str, Any] = {
        "learner": learner.state_dict(),
        "obs_normalizer": obs_normalizer.state_dict() if obs_normalizer is not None else None,
        "best_model_tracker": dict(best_model_tracker._best),
        "global_step": global_step,
        "episode": episode,
        "last_eval_step": last_eval_step,
        "vector_step": vector_step,
    }
    if replay_buffer is not None:
        state["replay_buffer"] = replay_buffer.state_dict()
    torch.save(state, path)


def load_rnn_qmix_training_state(
    path: Path,
    learner: MARLRNNQMIXLearner,
    obs_normalizer: Any,
    best_model_tracker: Any,
    replay_buffer: Any = None,
) -> Dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    learner.load_state_dict(state["learner"])
    if state["obs_normalizer"] is not None and obs_normalizer is not None:
        restored = RunningObsNormalizer.from_state_dict(state["obs_normalizer"])
        obs_normalizer.mean = restored.mean
        obs_normalizer.m2 = restored.m2
        obs_normalizer.count = restored.count
    best_model_tracker._best = dict(state["best_model_tracker"])
    if replay_buffer is not None and "replay_buffer" in state:
        replay_buffer.load_state_dict(state["replay_buffer"])
    return {
        "global_step": int(state["global_step"]),
        "episode": int(state["episode"]),
        "last_eval_step": int(state["last_eval_step"]),
        "vector_step": int(state["vector_step"]),
    }


# ---------------------------------------------------------------------------
# Inference-time policy loader (used by tools/policy_tester and visualizers)
# ---------------------------------------------------------------------------


def load_rnn_qmix_agent_from_checkpoint(
    model_path: Path,
) -> Tuple[RNNAgent, MARLTorchCheckpointMetadata]:
    """Load an RNN-QMIX agent for inference from a saved checkpoint."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    ckpt = torch.load(str(model_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("MARL RNN-QMIX checkpoint must be a dict")

    n_agents = int(ckpt.get("n_agents", 1))
    obs_dim = int(ckpt.get("obs_dim", 0))
    n_actions = int(ckpt.get("n_actions", 0))
    rnn_hidden_dim = int(ckpt.get("rnn_hidden_dim", 64))
    rnn_layers = int(ckpt.get("rnn_layers", 1))
    if n_agents <= 0 or obs_dim <= 0 or n_actions <= 0:
        raise ValueError(
            "Checkpoint must include positive 'n_agents', 'obs_dim', and 'n_actions'"
        )

    obs_normalizer: Optional[RunningObsNormalizer] = None
    obs_norm_state = ckpt.get("obs_normalization", None)
    if isinstance(obs_norm_state, dict) and bool(obs_norm_state.get("enabled", False)):
        obs_normalizer = RunningObsNormalizer.from_state_dict(obs_norm_state)
        if obs_normalizer.obs_dim != obs_dim:
            raise ValueError(
                f"Checkpoint obs_normalization dim={obs_normalizer.obs_dim} does not match obs_dim={obs_dim}"
            )

    agent = RNNAgent(
        obs_dim=obs_dim,
        n_agents=n_agents,
        n_actions=n_actions,
        rnn_hidden_dim=rnn_hidden_dim,
        rnn_layers=rnn_layers,
    )
    agent.load_state_dict(ckpt["agent_state_dict"])

    metadata = MARLTorchCheckpointMetadata(
        algorithm=str(ckpt.get("algorithm", "marl_rnn_qmix")),
        n_agents=n_agents,
        obs_dim=obs_dim,
        n_actions=n_actions,
        use_agent_id=True,
        parameter_sharing=True,
        agent_hidden_dims=(),
        agent_activation="relu",
        obs_normalizer=obs_normalizer,
    )
    return agent, metadata


class MARLRNNQMIXTorchPolicy:
    """Stateful recurrent RNN-QMIX policy for inference."""

    def __init__(
        self,
        agent: RNNAgent,
        metadata: MARLTorchCheckpointMetadata,
        *,
        device: Optional[torch.device] = None,
    ) -> None:
        self.agent = agent
        self.metadata = metadata
        self.device = device or torch.device("cpu")
        self.agent.to(self.device)

        n_agents = metadata.n_agents
        self.hidden = agent.init_hidden(n_agents).to(self.device)
        self.prev_action = torch.full(
            (1, n_agents), agent.n_actions, dtype=torch.long, device=self.device,
        )

    def reset(self) -> None:
        self.hidden.zero_()
        self.prev_action.fill_(self.agent.n_actions)

    @torch.no_grad()
    def act(self, obs_dict: Mapping[str, Any]) -> Dict[str, int]:
        n_agents = self.metadata.n_agents
        obs = stack_obs(dict(obs_dict), n_agents)
        if self.metadata.obs_normalizer is not None:
            obs = self.metadata.obs_normalizer.normalize(obs, update=False)

        obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
        q_values, new_hidden = rnn_forward_batched(
            self.agent, obs_t, self.prev_action, self.hidden,
        )
        actions = q_values.squeeze(0).argmax(dim=-1).cpu().numpy().astype(np.int64)
        self.hidden = new_hidden
        self.prev_action = torch.as_tensor(
            actions, device=self.device, dtype=torch.long,
        ).unsqueeze(0)
        return {f"agent_{i}": int(actions[i]) for i in range(n_agents)}
