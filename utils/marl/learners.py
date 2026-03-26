from __future__ import annotations

from typing import List, Optional

import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

from utils.marl.buffer import MARLBatch, DialRNNEpisodeBatch
from utils.marl.networks import MLPAgent, DuelingMLPAgent, DialRNNAgent, DRU, QMixer, QPLEXMixer, VDNMixer, TwinQNetwork, append_agent_id, route_messages
from utils.marl.value_norm import ValueNorm


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

    if isinstance(agent, (MLPAgent, DuelingMLPAgent)):
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

    raise TypeError("agent must be an MLPAgent/DuelingMLPAgent (shared) or nn.ModuleList (independent)")


def _make_optimizer(
    params, lr: float, optimizer_type: str = "adam",
    rmsprop_alpha: float = 0.99, rmsprop_eps: float = 1e-5,
    momentum: float = 0.0,
) -> torch.optim.Optimizer:
    """Create optimizer based on type. RMSprop uses PyMARL defaults."""
    if optimizer_type == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, alpha=rmsprop_alpha, eps=rmsprop_eps, momentum=momentum)
    return torch.optim.Adam(params, lr=lr)


def _maybe_append_id(obs: torch.Tensor, n_agents: int, use_agent_id: bool) -> torch.Tensor:
    """Conditionally append one-hot agent IDs to observations."""
    return append_agent_id(obs, n_agents) if use_agent_id else obs


class IQLLearner:
    def __init__(
        self,
        agent: nn.Module,
        n_agents: int,
        n_actions: int,
        gamma: float,
        lr: float,
        target_update_interval: int = 200,
        grad_clip_norm: Optional[float] = 1.0,
        use_agent_id: bool = True,
        double_q: bool = True,
        device: Optional[torch.device] = None,
        optimizer_type: str = "rmsprop",
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

        self.optimizer = _make_optimizer(self.agent.parameters(), lr=float(lr), optimizer_type=optimizer_type)
        self.train_steps = 0

    def update(self, batch: MARLBatch) -> None:
        self.train_steps += 1
        obs = _maybe_append_id(batch.obs, self.n_agents, self.use_agent_id)
        next_obs = _maybe_append_id(batch.next_obs, self.n_agents, self.use_agent_id)
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

    def state_dict(self) -> dict:
        return {
            "agent": self.agent.state_dict(),
            "target_agent": self.target_agent.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_steps": self.train_steps,
        }

    def load_state_dict(self, d: dict) -> None:
        self.agent.load_state_dict(d["agent"])
        self.target_agent.load_state_dict(d["target_agent"])
        self.optimizer.load_state_dict(d["optimizer"])
        self.train_steps = int(d["train_steps"])


class VDNLearner:
    def __init__(
        self,
        agent: nn.Module,
        n_agents: int,
        n_actions: int,
        gamma: float,
        lr: float,
        target_update_interval: int = 200,
        grad_clip_norm: Optional[float] = 1.0,
        use_agent_id: bool = True,
        double_q: bool = True,
        device: Optional[torch.device] = None,
        optimizer_type: str = "rmsprop",
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

        self.device = device if device is not None else torch.device("cpu")
        self.agent.to(self.device)
        self.target_agent.to(self.device)
        self.mixer.to(self.device)
        self.target_mixer.to(self.device)

        self.optimizer = _make_optimizer(self.agent.parameters(), lr=float(lr), optimizer_type=optimizer_type)
        self.train_steps = 0

    def update(self, batch: MARLBatch) -> None:
        self.train_steps += 1
        obs = _maybe_append_id(batch.obs, self.n_agents, self.use_agent_id)
        next_obs = _maybe_append_id(batch.next_obs, self.n_agents, self.use_agent_id)
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
            r_tot = rewards.sum(dim=1)  # VDN always sums rewards (Q_tot = sum of Q_i)
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

    def state_dict(self) -> dict:
        return {
            "agent": self.agent.state_dict(),
            "target_agent": self.target_agent.state_dict(),
            "mixer": self.mixer.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_steps": self.train_steps,
        }

    def load_state_dict(self, d: dict) -> None:
        self.agent.load_state_dict(d["agent"])
        self.target_agent.load_state_dict(d["target_agent"])
        self.mixer.load_state_dict(d["mixer"])
        self.target_mixer.load_state_dict(d["target_mixer"])
        self.optimizer.load_state_dict(d["optimizer"])
        self.train_steps = int(d["train_steps"])


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
        grad_clip_norm: Optional[float] = 1.0,
        use_agent_id: bool = True,
        double_q: bool = True,
        device: Optional[torch.device] = None,
        optimizer_type: str = "rmsprop",  # RMSprop default matches PyMARL
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

        self.device = device if device is not None else torch.device("cpu")
        self.agent.to(self.device)
        self.target_agent.to(self.device)
        self.mixer.to(self.device)
        self.target_mixer.to(self.device)

        self._all_params = list(self.agent.parameters()) + list(self.mixer.parameters())
        self.optimizer = _make_optimizer(self._all_params, lr=float(lr), optimizer_type=optimizer_type)
        self.train_steps = 0

    def update(self, batch: MARLBatch) -> None:
        self.train_steps += 1
        obs = _maybe_append_id(batch.obs, self.n_agents, self.use_agent_id)
        next_obs = _maybe_append_id(batch.next_obs, self.n_agents, self.use_agent_id)
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
            r_tot = rewards.sum(dim=1)  # QMIX uses team reward sum
            not_done = (1.0 - dones)
            targets = r_tot + self.gamma * not_done * next_q_tot

        loss = F.mse_loss(q_tot, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self._all_params, float(self.grad_clip_norm))
        self.optimizer.step()

        if self.train_steps % self.target_update_interval == 0:
            _hard_update(self.target_agent, self.agent)
            _hard_update(self.target_mixer, self.mixer)

    def state_dict(self) -> dict:
        return {
            "agent": self.agent.state_dict(),
            "target_agent": self.target_agent.state_dict(),
            "mixer": self.mixer.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_steps": self.train_steps,
        }

    def load_state_dict(self, d: dict) -> None:
        self.agent.load_state_dict(d["agent"])
        self.target_agent.load_state_dict(d["target_agent"])
        self.mixer.load_state_dict(d["mixer"])
        self.target_mixer.load_state_dict(d["target_mixer"])
        self.optimizer.load_state_dict(d["optimizer"])
        self.train_steps = int(d["train_steps"])


class QPLEXLearner:
    def __init__(
        self,
        agent: nn.Module,
        mixer: QPLEXMixer,
        n_agents: int,
        n_actions: int,
        gamma: float,
        lr: float,
        target_update_interval: int = 200,
        grad_clip_norm: Optional[float] = 1.0,
        use_agent_id: bool = True,
        double_q: bool = True,
        device: Optional[torch.device] = None,
        optimizer_type: str = "rmsprop",
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

        self.device = device if device is not None else torch.device("cpu")
        self.agent.to(self.device)
        self.target_agent.to(self.device)
        self.mixer.to(self.device)
        self.target_mixer.to(self.device)

        self._all_params = list(self.agent.parameters()) + list(self.mixer.parameters())
        self.optimizer = _make_optimizer(self._all_params, lr=float(lr), optimizer_type=optimizer_type)
        self.train_steps = 0

    def update(self, batch: MARLBatch) -> None:
        self.train_steps += 1
        obs = _maybe_append_id(batch.obs, self.n_agents, self.use_agent_id)
        next_obs = _maybe_append_id(batch.next_obs, self.n_agents, self.use_agent_id)
        actions = batch.actions
        rewards = batch.rewards
        dones = batch.dones
        states = batch.states
        next_states = batch.next_states

        q_all = _q_values(self.agent, obs, self.n_agents, self.n_actions)
        q_taken = q_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        max_q_i = q_all.max(dim=-1).values
        actions_onehot = F.one_hot(actions, num_classes=self.n_actions).float()

        q_tot_raw, attend_reg, _ = self.mixer(
            q_taken, states, actions_onehot=actions_onehot, max_q_i=max_q_i,
        )
        q_tot = q_tot_raw.squeeze(-1)

        with torch.no_grad():
            if self.double_q:
                online_next_q = _q_values(self.agent, next_obs, self.n_agents, self.n_actions)
                next_actions = online_next_q.argmax(dim=-1)
                target_next_q = _q_values(self.target_agent, next_obs, self.n_agents, self.n_actions)
                target_chosen_qvals = target_next_q.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)
                target_max_qvals = target_next_q.max(dim=-1).values
                next_actions_onehot = F.one_hot(next_actions, num_classes=self.n_actions).float()

                target_q_tot, _, _ = self.target_mixer(
                    target_chosen_qvals,
                    next_states,
                    actions_onehot=next_actions_onehot,
                    max_q_i=target_max_qvals,
                )
                target_q_tot = target_q_tot.squeeze(-1)
            else:
                target_next_q = _q_values(self.target_agent, next_obs, self.n_agents, self.n_actions)
                target_max_qvals = target_next_q.max(dim=-1).values
                target_q_tot, _, _ = self.target_mixer(target_max_qvals, next_states)
                target_q_tot = target_q_tot.squeeze(-1)

            r_tot = rewards.sum(dim=1)
            not_done = (1.0 - dones)
            targets = r_tot + self.gamma * not_done * target_q_tot

        loss = F.mse_loss(q_tot, targets) + attend_reg
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self._all_params, float(self.grad_clip_norm))
        self.optimizer.step()

        if self.train_steps % self.target_update_interval == 0:
            _hard_update(self.target_agent, self.agent)
            _hard_update(self.target_mixer, self.mixer)

    def state_dict(self) -> dict:  # QPLEXLearner state_dict
        return {
            "agent": self.agent.state_dict(),
            "target_agent": self.target_agent.state_dict(),
            "mixer": self.mixer.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_steps": self.train_steps,
        }

    def load_state_dict(self, d: dict) -> None:  # QPLEXLearner load_state_dict
        self.agent.load_state_dict(d["agent"])
        self.target_agent.load_state_dict(d["target_agent"])
        self.mixer.load_state_dict(d["mixer"])
        self.target_mixer.load_state_dict(d["target_mixer"])
        self.optimizer.load_state_dict(d["optimizer"])
        self.train_steps = int(d["train_steps"])


class IQLDIALRNNLearner:
    """IQL learner with recurrent DIAL (GRU-based).

    Recomputes the full online and target recurrent traces over padded episode
    batches, with mask-aware hidden state freezing and per-sample bootstrap
    selection.  Target network is updated every ``target_update_episodes``
    episodes (accumulator pattern for B > 1).
    """

    def __init__(
        self,
        agent: DialRNNAgent,
        n_agents: int,
        n_actions: int,
        comm_dim: int,
        dru: DRU,
        gamma: float,
        lr: float,
        target_update_steps: int = 100,
        grad_clip_norm: Optional[float] = 10.0,
        device: Optional[torch.device] = None,
        momentum: float = 0.05,
        optimizer_type: str = "rmsprop",
    ) -> None:
        self.agent = agent
        self.target_agent = copy.deepcopy(agent)
        for p in self.target_agent.parameters():
            p.requires_grad = False
        self.target_agent.eval()

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.comm_dim = comm_dim
        self.dru = dru
        self.gamma = float(gamma)
        self.target_update_steps = int(target_update_steps)
        self.grad_clip_norm = grad_clip_norm

        self.device = device if device is not None else torch.device("cpu")
        self.agent.to(self.device)
        self.target_agent.to(self.device)

        self.optimizer = _make_optimizer(
            self.agent.parameters(), lr=float(lr),
            optimizer_type=optimizer_type, momentum=float(momentum),
        )
        self.episodes_seen = 0
        self._train_steps = 0

    # ------------------------------------------------------------------
    def _forward_trace(
        self,
        agent: DialRNNAgent,
        batch: DialRNNEpisodeBatch,
        train_mode: bool,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Run mask-aware recurrent trace. Returns (per_step_q, q_boot)."""
        B, max_T, N = batch.obs.shape[:3]
        hidden = agent.init_hidden(B * N).to(self.device)
        prev_action = torch.full(
            (B, N), self.n_actions, dtype=torch.long, device=self.device,
        )
        recv_msg = torch.zeros(B, N, N * self.comm_dim, device=self.device)
        agent_idx = (
            torch.arange(N, device=self.device).unsqueeze(0).expand(B, -1)
        )

        all_q: list[torch.Tensor] = []
        for t in range(max_T):
            obs_t = batch.obs[:, t]
            q_t, m_t, new_hidden = agent(
                obs_t.reshape(B * N, -1),
                agent_idx.reshape(B * N),
                prev_action.reshape(B * N),
                recv_msg.reshape(B * N, -1),
                hidden,
            )
            q_t = q_t.view(B, N, -1)
            m_t = m_t.view(B, N, -1)
            all_q.append(q_t)

            new_recv = route_messages(self.dru(m_t, train_mode=train_mode), N)

            # Freeze state for padded timesteps
            valid = batch.mask[:, t].bool()
            valid_bn = valid.unsqueeze(1).expand(B, N).reshape(B * N)
            v_h = valid_bn.view(1, B * N, 1).expand_as(new_hidden)
            hidden = torch.where(v_h, new_hidden, hidden)
            v_a = valid.unsqueeze(1).expand(B, N)
            prev_action = torch.where(v_a, batch.actions[:, t], prev_action)
            v_m = valid.view(B, 1, 1).expand_as(new_recv)
            recv_msg = torch.where(v_m, new_recv, recv_msg)

        # Bootstrap step (reference runs nsteps+1)
        q_boot, _, _ = agent(
            batch.next_obs.reshape(B * N, -1),
            agent_idx.reshape(B * N),
            prev_action.reshape(B * N),
            recv_msg.reshape(B * N, -1),
            hidden,
        )
        q_boot = q_boot.view(B, N, -1)
        return all_q, q_boot

    # ------------------------------------------------------------------
    def update(self, batch: DialRNNEpisodeBatch) -> None:
        B, max_T = batch.obs.shape[:2]
        N = self.n_agents

        # Online trace WITH gradients
        online_q, _ = self._forward_trace(self.agent, batch, train_mode=True)

        # Target trace WITHOUT gradients
        with torch.no_grad():
            target_q, target_q_boot = self._forward_trace(
                self.target_agent, batch, train_mode=True,
            )

        # TD targets — mask-aware per-sample bootstrap
        with torch.no_grad():
            targets: list[torch.Tensor] = []
            boot_max = target_q_boot.max(-1).values  # (B, N)
            for t in range(max_T):
                if t < max_T - 1:
                    has_next = batch.mask[:, t + 1].bool()
                    next_max = target_q[t + 1].max(-1).values
                    bootstrap_q = torch.where(
                        has_next.unsqueeze(1).expand(B, N), next_max, boot_max,
                    )
                else:
                    bootstrap_q = boot_max
                not_done = (1.0 - batch.terminated[:, t]).unsqueeze(-1)
                targets.append(batch.rewards[:, t] + self.gamma * not_done * bootstrap_q)

        # Masked MSE loss
        total_loss = torch.tensor(0.0, device=self.device)
        for t in range(max_T):
            q_taken = online_q[t].gather(
                -1, batch.actions[:, t].unsqueeze(-1),
            ).squeeze(-1)
            td_error = (q_taken - targets[t]) ** 2
            total_loss = total_loss + (td_error * batch.mask[:, t].unsqueeze(-1)).sum()
        total_weight = (batch.mask.sum() * float(N)).clamp_min(1.0)
        total_loss = total_loss / total_weight

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                self.agent.parameters(), float(self.grad_clip_norm),
            )
        self.optimizer.step()

        # Target update — per-step counter (reference: agent.py:142-144)
        self.episodes_seen += B
        self._train_steps += 1
        if self._train_steps % self.target_update_steps == 0:
            _hard_update(self.target_agent, self.agent)

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "agent": self.agent.state_dict(),
            "target_agent": self.target_agent.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "episodes_seen": self.episodes_seen,
            "train_steps": self._train_steps,
        }

    def load_state_dict(self, d: dict) -> None:
        self.agent.load_state_dict(d["agent"])
        self.target_agent.load_state_dict(d["target_agent"])
        self.optimizer.load_state_dict(d["optimizer"])
        self.episodes_seen = int(d["episodes_seen"])
        self._train_steps = int(d.get("train_steps", 0))


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak averaging: target = (1 - tau) * target + tau * source."""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


class HASACLearner:
    """Heterogeneous-Agent SAC for discrete actions.

    Per-agent actors with sequential updates, SAC-style entropy regularization,
    and a centralized twin Q-critic with soft target updates.
    """

    def __init__(
        self,
        actors: List[MLPAgent],
        critic: TwinQNetwork,
        n_agents: int,
        n_actions: int,
        global_state_dim: int,
        gamma: float = 0.99,
        polyak: float = 0.005,
        actor_lr: float = 5e-4,
        critic_lr: float = 5e-4,
        alpha: float = 0.001,
        auto_alpha: bool = False,
        alpha_lr: float = 3e-4,
        target_entropy: Optional[float] = None,
        value_normalizer: Optional[ValueNorm] = None,
        grad_clip_norm: Optional[float] = 10.0,
        fixed_order: bool = False,
        use_huber_loss: bool = True,
        huber_delta: float = 10.0,
        device: Optional[torch.device] = None,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.global_state_dim = global_state_dim
        self.gamma = float(gamma)
        self.polyak = float(polyak)
        self.grad_clip_norm = grad_clip_norm
        self.fixed_order = bool(fixed_order)
        self.use_huber_loss = bool(use_huber_loss)
        self.huber_delta = float(huber_delta)
        self.auto_alpha = bool(auto_alpha)
        self.value_normalizer = value_normalizer

        self.device = device if device is not None else torch.device("cpu")
        self.rng = rng

        # Actors (one per agent, independent parameters)
        self.actors = actors
        for actor in self.actors:
            actor.to(self.device)

        # Twin Q-critic + target
        self.critic = critic.to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)

        # Optimizers
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=float(actor_lr)) for actor in self.actors
        ]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(critic_lr))

        # Per-agent entropy temperature
        if target_entropy is None:
            self.target_entropy = math.log(n_actions) * 0.98
        else:
            self.target_entropy = float(target_entropy)

        if self.auto_alpha:
            # HARL semantics: auto-alpha starts from log_alpha=0 (alpha=1.0).
            self.log_alphas = [
                torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=True)
                for _ in range(n_agents)
            ]
            self.alpha_optimizers = [
                torch.optim.Adam([la], lr=float(alpha_lr)) for la in self.log_alphas
            ]
            # Separate critic-side alpha (HARL has both actor-side and critic-side alpha updates).
            self.log_alpha_critic = torch.tensor(
                0.0, dtype=torch.float32, device=self.device, requires_grad=True
            )
            self.critic_alpha_optimizer = torch.optim.Adam(
                [self.log_alpha_critic], lr=float(alpha_lr)
            )
        else:
            init_log_alpha = math.log(max(float(alpha), 1e-8))
            self.log_alphas = [
                torch.tensor(
                    init_log_alpha,
                    dtype=torch.float32,
                    device=self.device,
                    requires_grad=False,
                )
                for _ in range(n_agents)
            ]
            self.alpha_optimizers = []
            self.log_alpha_critic = torch.tensor(
                init_log_alpha,
                dtype=torch.float32,
                device=self.device,
                requires_grad=False,
            )
            self.critic_alpha_optimizer = None

        self.train_steps = 0

    def _get_alphas(self) -> List[float]:
        return [la.exp().item() for la in self.log_alphas]

    def _get_critic_alpha(self) -> float:
        return float(self.log_alpha_critic.exp().item())

    def _build_critic_input(self, states: torch.Tensor, actions_onehot: torch.Tensor) -> torch.Tensor:
        """Concatenate global state with flattened one-hot joint actions."""
        return torch.cat([states, actions_onehot], dim=-1)

    def update(self, batch: MARLBatch) -> None:
        self.train_steps += 1
        obs = batch.obs              # (batch, n_agents, obs_dim)
        next_obs = batch.next_obs    # (batch, n_agents, obs_dim)
        actions = batch.actions      # (batch, n_agents) int64
        rewards = batch.rewards      # (batch, n_agents)
        dones = batch.dones          # (batch,)
        states = batch.states        # (batch, state_dim)
        next_states = batch.next_states

        batch_size = obs.shape[0]
        r_tot = rewards.sum(dim=1)
        not_done = (1.0 - dones)

        def _sample_onehot_and_logp(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Sample discrete one-hot action and its log-prob surrogate."""
            log_probs = F.log_softmax(logits, dim=-1)
            action_onehot = F.gumbel_softmax(log_probs, hard=True)
            logp = (action_onehot * log_probs).sum(dim=-1)
            return action_onehot, logp

        # ---- 1. Critic update ----
        with torch.no_grad():
            next_actions_onehot_list = []
            next_logprobs = []
            for i, actor in enumerate(self.actors):
                logits_i = actor(next_obs[:, i, :])
                next_act_onehot_i, next_logp_i = _sample_onehot_and_logp(logits_i)
                next_actions_onehot_list.append(next_act_onehot_i)
                next_logprobs.append(next_logp_i)

            next_actions_onehot = torch.cat(next_actions_onehot_list, dim=-1)
            next_critic_input = self._build_critic_input(next_states, next_actions_onehot)
            next_q1, next_q2 = self.target_critic(next_critic_input)
            next_q_min = torch.min(next_q1, next_q2)
            if self.value_normalizer is not None:
                next_q_min = self.value_normalizer.denormalize(next_q_min.unsqueeze(-1)).squeeze(-1)

            critic_alpha = self._get_critic_alpha()
            entropy_bonus = sum(
                -critic_alpha * next_logprobs[i] for i in range(self.n_agents)
            )
            gamma_n = batch.gamma_n  # per-sample discount (gamma^n for n-step)
            target_q = r_tot + gamma_n * not_done * (next_q_min + entropy_bonus)
            if self.value_normalizer is not None:
                target_q_expanded = target_q.unsqueeze(-1)
                self.value_normalizer.update(target_q_expanded)
                target_q = self.value_normalizer.normalize(target_q_expanded).squeeze(-1)

        replay_onehot = F.one_hot(actions, num_classes=self.n_actions).float()
        replay_onehot_flat = replay_onehot.view(batch_size, -1)
        critic_input = self._build_critic_input(states, replay_onehot_flat)
        q1, q2 = self.critic(critic_input)

        if self.use_huber_loss:
            critic_loss = F.huber_loss(q1, target_q, delta=self.huber_delta) + \
                          F.huber_loss(q2, target_q, delta=self.huber_delta)
        else:
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), float(self.grad_clip_norm))
        self.critic_optimizer.step()

        # ---- 2. Sequential actor updates ----
        # Match HARL training flow: keep critic as a fixed function during actor updates
        # so no critic parameter gradients are accumulated in this phase.
        critic_requires_grad = [param.requires_grad for param in self.critic.parameters()]
        for param in self.critic.parameters():
            param.requires_grad = False
        try:
            if self.fixed_order:
                order = list(range(self.n_agents))
            else:
                order = torch.randperm(self.n_agents, generator=self.rng).tolist()

            # Initialize joint actions/log-probs from current policy (not replay actions).
            with torch.no_grad():
                init_onehot_list = []
                log_probs_list = []
                for idx, actor in enumerate(self.actors):
                    logits_init = actor(obs[:, idx, :])
                    act_init, logp_init = _sample_onehot_and_logp(logits_init)
                    init_onehot_list.append(act_init)
                    log_probs_list.append(logp_init.detach())
                joint_onehot = torch.stack(init_onehot_list, dim=1)  # (batch, n_agents, n_actions)

            for i in order:
                actor = self.actors[i]
                logits_i = actor(obs[:, i, :])
                action_onehot_i, log_prob_surrogate = _sample_onehot_and_logp(logits_i)
                log_probs_list[i] = log_prob_surrogate.detach()

                joint_onehot_for_i = joint_onehot.clone()
                joint_onehot_for_i[:, i, :] = action_onehot_i
                joint_flat = joint_onehot_for_i.view(batch_size, -1)

                critic_in = self._build_critic_input(states.detach(), joint_flat)
                q1_pi, q2_pi = self.critic(critic_in)
                q_pi = torch.min(q1_pi, q2_pi)

                alpha_i = self.log_alphas[i].exp().detach()
                actor_loss = (-q_pi + alpha_i * log_prob_surrogate).mean()

                self.actor_optimizers[i].zero_grad(set_to_none=True)
                actor_loss.backward()
                if self.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(actor.parameters(), float(self.grad_clip_norm))
                self.actor_optimizers[i].step()

                # Per-agent alpha update uses sampled log-prob from the actor loss path.
                if self.auto_alpha:
                    alpha_loss = -(
                        self.log_alphas[i] * (log_prob_surrogate.detach() + self.target_entropy)
                    ).mean()
                    self.alpha_optimizers[i].zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    self.alpha_optimizers[i].step()

                with torch.no_grad():
                    # Refresh this agent's action after the optimizer step so later
                    # agents condition on the updated policy.
                    logits_post = actor(obs[:, i, :])
                    action_post, _ = _sample_onehot_and_logp(logits_post)
                    joint_onehot[:, i, :] = action_post

            # HARL also applies a separate critic-level alpha update using summed log-probs.
            if self.auto_alpha and self.critic_alpha_optimizer is not None:
                target_entropy_total = float(self.target_entropy) * float(self.n_agents)
                summed_logprob = torch.stack(log_probs_list, dim=0).sum(dim=0)
                critic_alpha_loss = -(
                    self.log_alpha_critic * (summed_logprob.detach() + target_entropy_total)
                ).mean()
                self.critic_alpha_optimizer.zero_grad(set_to_none=True)
                critic_alpha_loss.backward()
                self.critic_alpha_optimizer.step()
        finally:
            for param, requires_grad in zip(self.critic.parameters(), critic_requires_grad):
                param.requires_grad = requires_grad

        # ---- 3. Soft target update ----
        _soft_update(self.target_critic, self.critic, self.polyak)

    def state_dict(self) -> dict:  # HASACLearner state_dict
        d = {
            "actors": [actor.state_dict() for actor in self.actors],
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizers": [opt.state_dict() for opt in self.actor_optimizers],
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alphas": [la.detach().cpu().item() for la in self.log_alphas],
            "log_alpha_critic": self.log_alpha_critic.detach().cpu().item(),
            "train_steps": self.train_steps,
        }
        if self.auto_alpha:
            d["alpha_optimizers"] = [opt.state_dict() for opt in self.alpha_optimizers]
            if self.critic_alpha_optimizer is not None:
                d["critic_alpha_optimizer"] = self.critic_alpha_optimizer.state_dict()
        return d

    def load_state_dict(self, d: dict) -> None:  # HASACLearner load_state_dict
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(d["actors"][i])
        self.critic.load_state_dict(d["critic"])
        self.target_critic.load_state_dict(d["target_critic"])
        for i, opt in enumerate(self.actor_optimizers):
            opt.load_state_dict(d["actor_optimizers"][i])
        self.critic_optimizer.load_state_dict(d["critic_optimizer"])
        for i, val in enumerate(d["log_alphas"]):
            self.log_alphas[i].data.fill_(float(val))
        if "log_alpha_critic" in d:
            self.log_alpha_critic.data.fill_(float(d["log_alpha_critic"]))
        elif len(d.get("log_alphas", [])) > 0:
            # Backward compatibility for old checkpoints without critic alpha state.
            self.log_alpha_critic.data.fill_(float(d["log_alphas"][0]))
        self.train_steps = int(d["train_steps"])
        if self.auto_alpha and "alpha_optimizers" in d:
            for i, opt in enumerate(self.alpha_optimizers):
                opt.load_state_dict(d["alpha_optimizers"][i])
        if self.auto_alpha and "critic_alpha_optimizer" in d and self.critic_alpha_optimizer is not None:
            self.critic_alpha_optimizer.load_state_dict(d["critic_alpha_optimizer"])
