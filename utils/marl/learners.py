from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import copy
import torch
from torch import nn
from torch.nn import functional as F

from utils.marl.buffer import MARLBatch
from utils.marl.networks import MLPAgent, QMixer, VDNMixer, append_agent_id


@dataclass(frozen=True)
class LearnerStats:
    loss: float
    q_mean: float
    target_mean: float


def _hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def _q_values(agent: nn.Module, obs: torch.Tensor, n_agents: int, n_actions: int) -> torch.Tensor:
    """
    Compute per-agent Q-values for either:
      - shared parameters: a single MLPAgent applied to all agents
      - independent parameters: a ModuleList[MLPAgent] with one net per agent

    Args:
        agent: MLPAgent (shared) or nn.ModuleList (independent)
        obs: Tensor of shape (batch, n_agents, input_dim)
        n_agents: Number of agents
        n_actions: Number of discrete actions

    Returns:
        Tensor of shape (batch, n_agents, n_actions)
    """
    if obs.ndim != 3:
        raise ValueError("obs must have shape (batch, n_agents, input_dim)")
    batch_size = obs.shape[0]

    if isinstance(agent, MLPAgent):
        obs_flat = obs.view(batch_size * n_agents, -1)
        return agent(obs_flat).view(batch_size, n_agents, n_actions)

    if isinstance(agent, nn.ModuleList):
        if len(agent) != n_agents:
            raise ValueError("Independent agents ModuleList length must equal n_agents")
        q_per_agent: list[torch.Tensor] = []
        for agent_idx in range(n_agents):
            q_i = agent[agent_idx](obs[:, agent_idx, :]).unsqueeze(1)
            q_per_agent.append(q_i)
        return torch.cat(q_per_agent, dim=1)

    raise TypeError("agent must be an MLPAgent (shared) or nn.ModuleList (independent)")


class IQLLearner:
    def __init__(
        self,
        agent: nn.Module,
        n_agents: int,
        n_actions: int,
        gamma: float,
        lr: float,
        target_update_interval: int = 200,
        grad_clip_norm: Optional[float] = 10.0,
        use_agent_id: bool = True,
        double_q: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        self.agent = agent
        self.target_agent = copy.deepcopy(agent)

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = float(gamma)
        self.target_update_interval = int(target_update_interval)
        self.grad_clip_norm = grad_clip_norm
        self.use_agent_id = bool(use_agent_id)
        self.double_q = bool(double_q)

        self.device = device if device is not None else torch.device("cpu")
        self.agent.to(self.device)
        self.target_agent.to(self.device)

        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=float(lr))
        self.train_steps = 0

    def _maybe_append_id(self, obs: torch.Tensor) -> torch.Tensor:
        return append_agent_id(obs, self.n_agents) if self.use_agent_id else obs

    def update(self, batch: MARLBatch) -> LearnerStats:
        self.train_steps += 1
        obs = self._maybe_append_id(batch.obs)
        next_obs = self._maybe_append_id(batch.next_obs)
        actions = batch.actions
        rewards = batch.rewards
        dones = batch.dones

        batch_size = obs.shape[0]
        q_all = _q_values(self.agent, obs, self.n_agents, self.n_actions)
        q_taken = q_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            if self.double_q:
                online_next_q = _q_values(self.agent, next_obs, self.n_agents, self.n_actions)
                next_actions = online_next_q.argmax(dim=-1, keepdim=True)
                target_next_q = _q_values(self.target_agent, next_obs, self.n_agents, self.n_actions)
                next_q = target_next_q.gather(-1, next_actions).squeeze(-1)
            else:
                target_next_q = _q_values(self.target_agent, next_obs, self.n_agents, self.n_actions)
                next_q = target_next_q.max(dim=-1).values

            not_done = (1.0 - dones).view(batch_size, 1)
            targets = rewards + self.gamma * not_done * next_q

        loss = F.mse_loss(q_taken, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.agent.parameters(), float(self.grad_clip_norm))
        self.optimizer.step()

        if self.train_steps % self.target_update_interval == 0:
            _hard_update(self.target_agent, self.agent)

        return LearnerStats(
            loss=float(loss.item()),
            q_mean=float(q_taken.mean().item()),
            target_mean=float(targets.mean().item()),
        )


class VDNLearner:
    def __init__(
        self,
        agent: nn.Module,
        n_agents: int,
        n_actions: int,
        gamma: float,
        lr: float,
        target_update_interval: int = 200,
        grad_clip_norm: Optional[float] = 10.0,
        use_agent_id: bool = True,
        double_q: bool = True,
        device: Optional[torch.device] = None,
        team_reward: str = "sum",
    ) -> None:
        self.agent = agent
        self.target_agent = copy.deepcopy(agent)

        self.mixer = VDNMixer()
        self.target_mixer = VDNMixer()

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = float(gamma)
        self.target_update_interval = int(target_update_interval)
        self.grad_clip_norm = grad_clip_norm
        self.use_agent_id = bool(use_agent_id)
        self.double_q = bool(double_q)
        if team_reward not in {"sum", "mean"}:
            raise ValueError("team_reward must be 'sum' or 'mean'")
        self.team_reward = team_reward

        self.device = device if device is not None else torch.device("cpu")
        self.agent.to(self.device)
        self.target_agent.to(self.device)
        self.mixer.to(self.device)
        self.target_mixer.to(self.device)

        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=float(lr))
        self.train_steps = 0

    def _maybe_append_id(self, obs: torch.Tensor) -> torch.Tensor:
        return append_agent_id(obs, self.n_agents) if self.use_agent_id else obs

    def update(self, batch: MARLBatch) -> LearnerStats:
        self.train_steps += 1
        obs = self._maybe_append_id(batch.obs)
        next_obs = self._maybe_append_id(batch.next_obs)
        actions = batch.actions
        rewards = batch.rewards
        dones = batch.dones

        batch_size = obs.shape[0]
        q_all = _q_values(self.agent, obs, self.n_agents, self.n_actions)
        q_taken = q_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        q_tot = self.mixer(q_taken).squeeze(-1)

        with torch.no_grad():
            if self.double_q:
                online_next_q = _q_values(self.agent, next_obs, self.n_agents, self.n_actions)
                next_actions = online_next_q.argmax(dim=-1, keepdim=True)
                target_next_q = _q_values(self.target_agent, next_obs, self.n_agents, self.n_actions)
                next_q = target_next_q.gather(-1, next_actions).squeeze(-1)
            else:
                target_next_q = _q_values(self.target_agent, next_obs, self.n_agents, self.n_actions)
                next_q = target_next_q.max(dim=-1).values

            next_q_tot = self.target_mixer(next_q).squeeze(-1)
            r_tot = rewards.sum(dim=1) if self.team_reward == "sum" else rewards.mean(dim=1)
            not_done = (1.0 - dones)
            targets = r_tot + self.gamma * not_done * next_q_tot

        loss = F.mse_loss(q_tot, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.agent.parameters(), float(self.grad_clip_norm))
        self.optimizer.step()

        if self.train_steps % self.target_update_interval == 0:
            _hard_update(self.target_agent, self.agent)

        return LearnerStats(
            loss=float(loss.item()),
            q_mean=float(q_tot.mean().item()),
            target_mean=float(targets.mean().item()),
        )


class QMIXLearner:
    def __init__(
        self,
        agent: nn.Module,
        mixer: QMixer,
        n_agents: int,
        n_actions: int,
        gamma: float,
        lr: float,
        target_update_interval: int = 200,
        grad_clip_norm: Optional[float] = 10.0,
        use_agent_id: bool = True,
        double_q: bool = True,
        device: Optional[torch.device] = None,
        team_reward: str = "sum",
    ) -> None:
        self.agent = agent
        self.target_agent = copy.deepcopy(agent)

        self.mixer = mixer
        self.target_mixer = copy.deepcopy(mixer)

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = float(gamma)
        self.target_update_interval = int(target_update_interval)
        self.grad_clip_norm = grad_clip_norm
        self.use_agent_id = bool(use_agent_id)
        self.double_q = bool(double_q)
        if team_reward not in {"sum", "mean"}:
            raise ValueError("team_reward must be 'sum' or 'mean'")
        self.team_reward = team_reward

        self.device = device if device is not None else torch.device("cpu")
        self.agent.to(self.device)
        self.target_agent.to(self.device)
        self.mixer.to(self.device)
        self.target_mixer.to(self.device)

        self.optimizer = torch.optim.Adam(list(self.agent.parameters()) + list(self.mixer.parameters()), lr=float(lr))
        self.train_steps = 0

    def _maybe_append_id(self, obs: torch.Tensor) -> torch.Tensor:
        return append_agent_id(obs, self.n_agents) if self.use_agent_id else obs

    def update(self, batch: MARLBatch) -> LearnerStats:
        self.train_steps += 1
        obs = self._maybe_append_id(batch.obs)
        next_obs = self._maybe_append_id(batch.next_obs)
        actions = batch.actions
        rewards = batch.rewards
        dones = batch.dones
        states = batch.states
        next_states = batch.next_states

        batch_size = obs.shape[0]
        q_all = _q_values(self.agent, obs, self.n_agents, self.n_actions)
        q_taken = q_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        q_tot = self.mixer(q_taken, states).squeeze(-1)

        with torch.no_grad():
            if self.double_q:
                online_next_q = _q_values(self.agent, next_obs, self.n_agents, self.n_actions)
                next_actions = online_next_q.argmax(dim=-1, keepdim=True)
                target_next_q = _q_values(self.target_agent, next_obs, self.n_agents, self.n_actions)
                next_q = target_next_q.gather(-1, next_actions).squeeze(-1)
            else:
                target_next_q = _q_values(self.target_agent, next_obs, self.n_agents, self.n_actions)
                next_q = target_next_q.max(dim=-1).values

            next_q_tot = self.target_mixer(next_q, next_states).squeeze(-1)
            r_tot = rewards.sum(dim=1) if self.team_reward == "sum" else rewards.mean(dim=1)
            not_done = (1.0 - dones)
            targets = r_tot + self.gamma * not_done * next_q_tot

        loss = F.mse_loss(q_tot, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(list(self.agent.parameters()) + list(self.mixer.parameters()), float(self.grad_clip_norm))
        self.optimizer.step()

        if self.train_steps % self.target_update_interval == 0:
            _hard_update(self.target_agent, self.agent)
            _hard_update(self.target_mixer, self.mixer)

        return LearnerStats(
            loss=float(loss.item()),
            q_mean=float(q_tot.mean().item()),
            target_mean=float(targets.mean().item()),
        )
