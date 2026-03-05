from __future__ import annotations

from typing import List, Optional

import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

from utils.marl.buffer import MARLBatch
from utils.marl.networks import MLPAgent, DuelingMLPAgent, QMixer, QPLEXMixer, VDNMixer, TwinQNetwork, append_agent_id


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
    params, lr: float, optimizer_type: str = "adam", rmsprop_alpha: float = 0.99, rmsprop_eps: float = 1e-5
) -> torch.optim.Optimizer:
    """Create optimizer based on type. RMSprop uses PyMARL defaults."""
    if optimizer_type == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, alpha=rmsprop_alpha, eps=rmsprop_eps)
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

        self.log_alphas = [
            torch.tensor(math.log(max(float(alpha), 1e-8)), dtype=torch.float32,
                         device=self.device, requires_grad=True)
            for _ in range(n_agents)
        ]
        self.alpha_optimizers = [
            torch.optim.Adam([la], lr=float(alpha_lr)) for la in self.log_alphas
        ] if self.auto_alpha else []

        self.train_steps = 0

    def _get_alphas(self) -> List[float]:
        return [la.exp().item() for la in self.log_alphas]

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

        # ---- 1. Critic update ----
        with torch.no_grad():
            next_actions_onehot_list = []
            next_logprobs = []
            for i, actor in enumerate(self.actors):
                logits_i = actor(next_obs[:, i, :])
                probs_i = F.softmax(logits_i, dim=-1)
                log_probs_i = F.log_softmax(logits_i, dim=-1)
                expected_logp_i = (probs_i * log_probs_i).sum(dim=-1)
                next_logprobs.append(expected_logp_i)
                dist_i = torch.distributions.Categorical(probs=probs_i)
                next_act_i = dist_i.sample()
                next_actions_onehot_list.append(
                    F.one_hot(next_act_i, num_classes=self.n_actions).float()
                )

            next_actions_onehot = torch.cat(next_actions_onehot_list, dim=-1)
            next_critic_input = self._build_critic_input(next_states, next_actions_onehot)
            next_q1, next_q2 = self.target_critic(next_critic_input)
            next_q_min = torch.min(next_q1, next_q2)

            alphas = self._get_alphas()
            entropy_bonus = sum(
                -alphas[i] * next_logprobs[i] for i in range(self.n_agents)
            )
            gamma_n = batch.gamma_n  # per-sample discount (gamma^n for n-step)
            target_q = r_tot + gamma_n * not_done * (next_q_min + entropy_bonus)

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
        if self.fixed_order:
            order = list(range(self.n_agents))
        else:
            order = torch.randperm(self.n_agents, generator=self.rng).tolist()

        # Initialize joint actions from current policy (not stale replay actions)
        with torch.no_grad():
            init_onehot_list = []
            for idx, actor in enumerate(self.actors):
                logits_init = actor(obs[:, idx, :])
                probs_init = F.softmax(logits_init, dim=-1)
                dist_init = torch.distributions.Categorical(probs=probs_init)
                act_init = dist_init.sample()
                init_onehot_list.append(
                    F.one_hot(act_init, num_classes=self.n_actions).float()
                )
            joint_onehot = torch.stack(init_onehot_list, dim=1)  # (batch, n_agents, n_actions)

        for i in order:
            actor = self.actors[i]
            logits_i = actor(obs[:, i, :])
            action_onehot_i = F.gumbel_softmax(logits_i, hard=True)

            joint_onehot_for_i = joint_onehot.clone()
            joint_onehot_for_i[:, i, :] = action_onehot_i
            joint_flat = joint_onehot_for_i.view(batch_size, -1)

            critic_in = self._build_critic_input(states.detach(), joint_flat)
            q1_pi, q2_pi = self.critic(critic_in)
            q_pi = torch.min(q1_pi, q2_pi)

            log_prob_surrogate = (action_onehot_i * logits_i).sum(dim=-1)
            alpha_i = self.log_alphas[i].exp().detach()
            actor_loss = (-q_pi + alpha_i * log_prob_surrogate).mean()

            self.actor_optimizers[i].zero_grad(set_to_none=True)
            actor_loss.backward()
            if self.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(actor.parameters(), float(self.grad_clip_norm))
            self.actor_optimizers[i].step()

            with torch.no_grad():
                joint_onehot[:, i, :] = action_onehot_i.detach()

        # ---- 3. Alpha updates ----
        if self.auto_alpha:
            for i in range(self.n_agents):
                with torch.no_grad():
                    logits_i = self.actors[i](obs[:, i, :])
                    probs_i = F.softmax(logits_i, dim=-1)
                    log_probs_i = F.log_softmax(logits_i, dim=-1)
                    expected_logp_i = (probs_i * log_probs_i).sum(dim=-1)

                alpha_loss = -(self.log_alphas[i] * (expected_logp_i + self.target_entropy).detach()).mean()
                self.alpha_optimizers[i].zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_optimizers[i].step()

        # ---- 4. Soft target update ----
        _soft_update(self.target_critic, self.critic, self.polyak)

    def state_dict(self) -> dict:  # HASACLearner state_dict
        d = {
            "actors": [actor.state_dict() for actor in self.actors],
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizers": [opt.state_dict() for opt in self.actor_optimizers],
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alphas": [la.detach().cpu().item() for la in self.log_alphas],
            "train_steps": self.train_steps,
        }
        if self.auto_alpha:
            d["alpha_optimizers"] = [opt.state_dict() for opt in self.alpha_optimizers]
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
        self.train_steps = int(d["train_steps"])
        if self.auto_alpha and "alpha_optimizers" in d:
            for i, opt in enumerate(self.alpha_optimizers):
                opt.load_state_dict(d["alpha_optimizers"][i])
