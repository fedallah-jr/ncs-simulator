from __future__ import annotations

from typing import List, Optional

import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils.marl.buffer import MARLBatch, DialRNNEpisodeBatch
from utils.marl.networks import (
    MLPAgent,
    DuelingMLPAgent,
    DialRNNAgent,
    DRU,
    NDQRNNAgent,
    NDQCommEncoder,
    QMixer,
    VDNMixer,
    TwinQNetwork,
    append_agent_id,
    route_messages,
    dial_rnn_forward_batched,
    ndq_rnn_forward_batched,
)
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
            gamma_n = batch.gamma_n.view(batch_size, 1)  # per-sample discount (gamma^n for n-step)
            targets = rewards + gamma_n * not_done * next_q

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
            gamma_n = batch.gamma_n  # per-sample discount (gamma^n for n-step)
            targets = r_tot + gamma_n * not_done * next_q_tot

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
            gamma_n = batch.gamma_n  # per-sample discount (gamma^n for n-step)
            targets = r_tot + gamma_n * not_done * next_q_tot

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


class MARLDIALLearner:
    """IQL learner with recurrent DIAL (GRU-based).

    Recomputes the full online and target recurrent traces over padded episode
    batches, with mask-aware hidden state freezing and per-sample bootstrap
    selection.

    Target sync cadence: every ``target_update_steps`` calls to ``update``
    (one call = one optimizer step on a batch of B episodes). With B = 32 and
    target_update_steps = 100 this matches the reference's
    ``step_target=100`` (which counts ``learn_from_episode`` calls, each over
    ``bs=32`` parallel rollouts).

    DRU noise replay: the online trace replays ``batch.dru_noise`` so the
    recomputed recv_msg matches what the policy actually saw at collection
    time (variance reduction). The target trace samples fresh noise to remain
    independent of the online trace, mirroring the reference's separate
    target rollout in ``arena.run_episode``.

    ``episodes_seen`` is kept as an informational counter (also persisted in
    ``state_dict``) for downstream logging / resume; it is *not* used to
    decide target syncs — that is keyed off ``_train_steps``.
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
        mixer_type: str = "none",
        state_dim: int = 0,
        qmix_mixing_hidden_dim: int = 32,
        qmix_hypernet_hidden_dim: int = 64,
        td_lambda: float = 0.0,
    ) -> None:
        td_lambda = float(td_lambda)
        if not 0.0 <= td_lambda <= 1.0:
            raise ValueError("td_lambda must be in [0, 1]")
        self.agent = agent
        self.target_agent = copy.deepcopy(agent)
        for p in self.target_agent.parameters():
            p.requires_grad = False
        self.target_agent.train()

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.comm_dim = comm_dim
        self.dru = dru
        self.gamma = float(gamma)
        self.td_lambda = td_lambda
        self.target_update_steps = int(target_update_steps)
        self.grad_clip_norm = grad_clip_norm

        self.mixer_type = mixer_type

        self.device = device if device is not None else torch.device("cpu")
        self.agent.to(self.device)
        self.target_agent.to(self.device)

        self.mixer: Optional[nn.Module] = None
        self.target_mixer: Optional[nn.Module] = None
        if self.mixer_type == "vdn":
            self.mixer = VDNMixer().to(self.device)
            self.target_mixer = VDNMixer().to(self.device)
        elif self.mixer_type == "qmix":
            if int(state_dim) <= 0:
                raise ValueError("QMIX mixer requires state_dim > 0 (env global_state channel)")
            self.mixer = QMixer(
                n_agents=n_agents, state_dim=int(state_dim),
                mixing_hidden_dim=qmix_mixing_hidden_dim,
                hypernet_hidden_dim=qmix_hypernet_hidden_dim,
            ).to(self.device)
            self.target_mixer = copy.deepcopy(self.mixer).to(self.device)
            for p in self.target_mixer.parameters():
                p.requires_grad = False

        all_params = list(self.agent.parameters())
        if self.mixer is not None:
            all_params += list(self.mixer.parameters())
        self.optimizer = _make_optimizer(
            all_params, lr=float(lr),
            optimizer_type=optimizer_type, momentum=float(momentum),
        )
        self.episodes_seen = 0
        self._train_steps = 0

    @staticmethod
    def _dial_team_reward(batch: DialRNNEpisodeBatch, t: int) -> torch.Tensor:
        # dial_rnn_collect_transition stores the already-summed team reward
        # repeated per agent so the IQL path can keep its existing per-agent
        # target shape. Mixed VDN/QMIX losses must consume it once.
        return batch.rewards[:, t, :1]

    # ------------------------------------------------------------------
    def _forward_trace(
        self,
        agent: DialRNNAgent,
        batch: DialRNNEpisodeBatch,
        train_mode: bool,
        dru_noise: Optional[torch.Tensor] = None,
        init_hidden: Optional[torch.Tensor] = None,
        init_prev_action: Optional[torch.Tensor] = None,
        init_recv_msg: Optional[torch.Tensor] = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Run mask-aware recurrent trace. Returns (per_step_q, q_boot).

        If *dru_noise* ``(B, max_T, N, comm_dim)`` is provided, the stored
        noise is replayed so the communication trace matches collection.
        ``dru_noise[:, t]`` is the noise that produced ``recv_msg`` at step
        *t* during collection. DRU at learning step *t* produces messages
        for step *t+1*, so we look up ``dru_noise[:, t+1]``.
        """
        B, max_T, N = batch.obs.shape[:3]
        if init_hidden is None:
            hidden = agent.init_hidden(B * N).to(self.device)
        else:
            hidden = init_hidden.to(self.device)
            if hidden.ndim == 4:
                hidden = hidden.reshape(agent.rnn_layers, B * N, agent.rnn_hidden_dim)
            hidden = hidden.clone()

        if init_prev_action is None:
            prev_action = torch.full(
                (B, N), self.n_actions, dtype=torch.long, device=self.device,
            )
        else:
            prev_action = init_prev_action.to(self.device, dtype=torch.long).clone()

        if init_recv_msg is None:
            recv_msg = torch.zeros(B, N, N * self.comm_dim, device=self.device)
        else:
            recv_msg = init_recv_msg.to(self.device, dtype=torch.float32).clone()
        all_q: list[torch.Tensor] = []
        for t in range(max_T):
            obs_t = batch.obs[:, t]
            valid = batch.mask[:, t].bool()
            q_t, m_t, new_hidden = self._forward_with_valid_mask(
                agent,
                obs_t=obs_t,
                prev_action=prev_action,
                recv_msg=recv_msg,
                hidden=hidden,
                valid=valid,
                B=B,
                N=N,
            )
            all_q.append(q_t)

            # Replay stored noise so recv_msg at step t+1 matches collection.
            step_noise: Optional[torch.Tensor] = None
            if dru_noise is not None and t + 1 < max_T:
                step_noise = dru_noise[:, t + 1]
            new_recv = route_messages(
                self.dru(m_t, train_mode=train_mode, noise=step_noise), N,
            )

            # Freeze state for padded timesteps
            valid_bn = valid.unsqueeze(1).expand(B, N).reshape(B * N)
            v_h = valid_bn.view(1, B * N, 1).expand_as(new_hidden)
            hidden = torch.where(v_h, new_hidden, hidden)
            v_a = valid.unsqueeze(1).expand(B, N)
            prev_action = torch.where(v_a, batch.actions[:, t], prev_action)
            v_m = valid.view(B, 1, 1).expand_as(new_recv)
            recv_msg = torch.where(v_m, new_recv, recv_msg)

        # Bootstrap step (reference runs nsteps+1). Only forward over rows
        # whose episode actually reached this point; otherwise BN running
        # stats would absorb post-end (padded) inputs.
        boot_valid = batch.mask.any(dim=1)  # (B,) — true if episode had ≥1 valid step
        q_boot, _, _ = self._forward_with_valid_mask(
            agent,
            obs_t=batch.next_obs,
            prev_action=prev_action,
            recv_msg=recv_msg,
            hidden=hidden,
            valid=boot_valid,
            B=B,
            N=N,
        )
        return all_q, q_boot

    # ------------------------------------------------------------------
    def _forward_with_valid_mask(
        self,
        agent: DialRNNAgent,
        *,
        obs_t: torch.Tensor,
        prev_action: torch.Tensor,
        recv_msg: torch.Tensor,
        hidden: torch.Tensor,
        valid: torch.Tensor,
        B: int,
        N: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward only the valid (non-padded) rows through the agent.

        Avoids BatchNorm contamination from padded timesteps: with batch-mode
        BN, every row in the batch contributes to running stats and to the
        per-batch normalization, so naively forwarding all B rows leaks
        garbage from envs whose episodes already ended.

        Outputs are scattered back to full batch shape; invalid rows hold
        zero Q/message values (loss is masked out anyway) and the *original*
        hidden state (subsequent ``torch.where`` masking is then a no-op
        for those rows).
        """
        if valid.all():
            return dial_rnn_forward_batched(
                agent, obs_t, prev_action, recv_msg, hidden,
            )

        device = obs_t.device
        q_full = torch.zeros(B, N, agent.n_actions, device=device, dtype=obs_t.dtype)
        m_full = torch.zeros(B, N, agent.comm_dim, device=device, dtype=obs_t.dtype)
        if not valid.any():
            return q_full, m_full, hidden

        valid_idx = valid.nonzero(as_tuple=True)[0]
        V = int(valid_idx.numel())

        obs_v = obs_t.index_select(0, valid_idx).contiguous()
        pa_v = prev_action.index_select(0, valid_idx).contiguous()
        rm_v = recv_msg.index_select(0, valid_idx).contiguous()

        hidden_view = hidden.reshape(agent.rnn_layers, B, N, agent.rnn_hidden_dim)
        h_v = hidden_view.index_select(1, valid_idx).reshape(
            agent.rnn_layers, V * N, agent.rnn_hidden_dim,
        ).contiguous()

        q_v, m_v, h_v_out = dial_rnn_forward_batched(
            agent, obs_v, pa_v, rm_v, h_v,
        )

        q_full = q_full.index_copy(0, valid_idx, q_v)
        m_full = m_full.index_copy(0, valid_idx, m_v)

        new_hidden_view = hidden.reshape(
            agent.rnn_layers, B, N, agent.rnn_hidden_dim,
        ).clone()
        h_v_out_4d = h_v_out.reshape(agent.rnn_layers, V, N, agent.rnn_hidden_dim)
        new_hidden_view = new_hidden_view.index_copy(1, valid_idx, h_v_out_4d)
        new_hidden = new_hidden_view.reshape(
            agent.rnn_layers, B * N, agent.rnn_hidden_dim,
        )

        return q_full, m_full, new_hidden

    # ------------------------------------------------------------------
    @staticmethod
    def _apply_td_lambda(
        targets: list[torch.Tensor],
        rewards: list[torch.Tensor],
        mask: torch.Tensor,
        terminated: torch.Tensor,
        gamma: float,
        td_lambda: float,
    ) -> list[torch.Tensor]:
        """Convert 1-step TD targets to TD(lambda) returns via backward recursion.

        G^lambda_t = (1-lambda)*T_t + lambda*(r_t + gamma*(1-d_t)*G^lambda_{t+1})

        When td_lambda=0 this returns the original 1-step targets unchanged.
        When td_lambda=1 this becomes the full sampled return, with the final
        step bootstrapped only if the episode ended by truncation.
        """
        T = len(targets)
        ret = [t.clone() for t in targets]
        for t in range(T - 2, -1, -1):
            not_done = (1.0 - terminated[:, t])
            valid_next = mask[:, t + 1]
            gate = not_done * valid_next
            # Reshape for broadcasting against target shape (B, ...) e.g. (B, N) or (B, 1)
            shape = [gate.shape[0]] + [1] * (targets[t].ndim - 1)
            gate = gate.view(*shape)
            ret[t] = (1.0 - td_lambda) * targets[t] + td_lambda * (
                rewards[t] + gamma * gate * ret[t + 1]
            )
        return ret

    # ------------------------------------------------------------------
    def update(self, batch: DialRNNEpisodeBatch) -> None:
        B, max_T = batch.obs.shape[:2]
        N = self.n_agents

        # Online: replay stored DRU noise so the recomputed communication
        # trace matches the noise actually applied at collection time. This
        # is a variance-reduction technique that pins the online trace to
        # the rolled-out recv_msg modulo any drift in agent params between
        # collection and learning.
        online_noise = batch.dru_noise if batch.dru_noise.shape[-1] > 0 else None

        # Online trace WITH gradients
        online_q, _ = self._forward_trace(
            self.agent, batch, train_mode=True, dru_noise=online_noise,
        )

        # Target trace WITHOUT gradients. Reference samples target's DRU
        # noise independently of the online net's noise (separate rollout in
        # arena.run_episode). Pass dru_noise=None so each target step
        # samples fresh noise. The bootstrap recv_msg is similarly
        # freshly-sampled because there is no stored noise for the
        # bootstrap timestep.
        with torch.no_grad():
            target_q, target_q_boot = self._forward_trace(
                self.target_agent, batch, train_mode=True, dru_noise=None,
            )

        if self.mixer_type == "vdn":
            self._update_vdn(batch, online_q, target_q, target_q_boot, B, max_T, N)
        elif self.mixer_type == "qmix":
            self._update_qmix(batch, online_q, target_q, target_q_boot, B, max_T, N)
        else:
            self._update_iql(batch, online_q, target_q, target_q_boot, B, max_T, N)

        # Target update — per-step counter (reference: agent.py:142-144)
        self.episodes_seen += B
        self._train_steps += 1
        if self._train_steps % self.target_update_steps == 0:
            _hard_update(self.target_agent, self.agent)
            if self.target_mixer is not None:
                _hard_update(self.target_mixer, self.mixer)

    def _update_iql(
        self, batch: DialRNNEpisodeBatch,
        online_q: list[torch.Tensor], target_q: list[torch.Tensor],
        target_q_boot: torch.Tensor, B: int, max_T: int, N: int,
    ) -> None:
        """Original per-agent TD loss."""
        # TD targets — mask-aware per-sample bootstrap
        with torch.no_grad():
            targets: list[torch.Tensor] = []
            rewards: list[torch.Tensor] = []
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
                rewards.append(batch.rewards[:, t])
                targets.append(rewards[-1] + self.gamma * not_done * bootstrap_q)
            if self.td_lambda > 0:
                targets = self._apply_td_lambda(
                    targets, rewards, batch.mask, batch.terminated,
                    self.gamma, self.td_lambda,
                )

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

    def _update_vdn(
        self, batch: DialRNNEpisodeBatch,
        online_q: list[torch.Tensor], target_q: list[torch.Tensor],
        target_q_boot: torch.Tensor, B: int, max_T: int, N: int,
    ) -> None:
        """VDN-mixed loss: TD error on Q_tot = sum(Q_i)."""
        with torch.no_grad():
            # Target: mix per-agent max-Q into Q_tot
            boot_max = target_q_boot.max(-1).values  # (B, N)
            targets: list[torch.Tensor] = []
            rewards: list[torch.Tensor] = []
            for t in range(max_T):
                if t < max_T - 1:
                    has_next = batch.mask[:, t + 1].bool()
                    next_max = target_q[t + 1].max(-1).values
                    bootstrap_q = torch.where(
                        has_next.unsqueeze(1).expand(B, N), next_max, boot_max,
                    )
                else:
                    bootstrap_q = boot_max
                next_q_tot = self.target_mixer(bootstrap_q)  # (B, 1)
                r_tot = self._dial_team_reward(batch, t)  # (B, 1)
                not_done = (1.0 - batch.terminated[:, t]).unsqueeze(-1)  # (B, 1)
                rewards.append(r_tot)
                targets.append(r_tot + self.gamma * not_done * next_q_tot)
            if self.td_lambda > 0:
                targets = self._apply_td_lambda(
                    targets, rewards, batch.mask, batch.terminated,
                    self.gamma, self.td_lambda,
                )

        # Online: mix per-agent Q(s,a) into Q_tot
        total_loss = torch.tensor(0.0, device=self.device)
        for t in range(max_T):
            q_taken = online_q[t].gather(
                -1, batch.actions[:, t].unsqueeze(-1),
            ).squeeze(-1)  # (B, N)
            q_tot = self.mixer(q_taken)  # (B, 1)
            td_error = (q_tot - targets[t]) ** 2
            total_loss = total_loss + (td_error * batch.mask[:, t].unsqueeze(-1)).sum()
        total_weight = batch.mask.sum().clamp_min(1.0)
        total_loss = total_loss / total_weight

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if self.grad_clip_norm is not None:
            all_params = list(self.agent.parameters()) + list(self.mixer.parameters())
            nn.utils.clip_grad_norm_(all_params, float(self.grad_clip_norm))
        self.optimizer.step()

    def _update_qmix(
        self, batch: DialRNNEpisodeBatch,
        online_q: list[torch.Tensor], target_q: list[torch.Tensor],
        target_q_boot: torch.Tensor, B: int, max_T: int, N: int,
    ) -> None:
        """QMIX-mixed loss: TD error on Q_tot from state-conditioned hypernetwork."""
        states = batch.states                      # (B, T, state_dim)
        boot_state = batch.next_states             # (B, state_dim)
        state_dim = states.shape[-1]

        with torch.no_grad():
            boot_max = target_q_boot.max(-1).values  # (B, N)
            targets: list[torch.Tensor] = []
            rewards: list[torch.Tensor] = []
            for t in range(max_T):
                if t < max_T - 1:
                    has_next = batch.mask[:, t + 1].bool()
                    next_max = target_q[t + 1].max(-1).values
                    bootstrap_q = torch.where(
                        has_next.unsqueeze(1).expand(B, N), next_max, boot_max,
                    )
                    next_state = torch.where(
                        has_next.view(B, 1).expand(B, state_dim),
                        states[:, t + 1], boot_state,
                    )
                else:
                    bootstrap_q = boot_max
                    next_state = boot_state
                next_q_tot = self.target_mixer(bootstrap_q, next_state)  # (B, 1)
                r_tot = self._dial_team_reward(batch, t)  # (B, 1)
                not_done = (1.0 - batch.terminated[:, t]).unsqueeze(-1)  # (B, 1)
                rewards.append(r_tot)
                targets.append(r_tot + self.gamma * not_done * next_q_tot)
            if self.td_lambda > 0:
                targets = self._apply_td_lambda(
                    targets, rewards, batch.mask, batch.terminated,
                    self.gamma, self.td_lambda,
                )

        total_loss = torch.tensor(0.0, device=self.device)
        for t in range(max_T):
            q_taken = online_q[t].gather(
                -1, batch.actions[:, t].unsqueeze(-1),
            ).squeeze(-1)  # (B, N)
            q_tot = self.mixer(q_taken, states[:, t])  # (B, 1)
            td_error = (q_tot - targets[t]) ** 2
            total_loss = total_loss + (td_error * batch.mask[:, t].unsqueeze(-1)).sum()
        total_weight = batch.mask.sum().clamp_min(1.0)
        total_loss = total_loss / total_weight

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if self.grad_clip_norm is not None:
            all_params = list(self.agent.parameters()) + list(self.mixer.parameters())
            nn.utils.clip_grad_norm_(all_params, float(self.grad_clip_norm))
        self.optimizer.step()

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        d = {
            "agent": self.agent.state_dict(),
            "target_agent": self.target_agent.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "episodes_seen": self.episodes_seen,
            "train_steps": self._train_steps,
            "mixer_type": self.mixer_type,
        }
        if self.mixer is not None:
            d["mixer"] = self.mixer.state_dict()
            d["target_mixer"] = self.target_mixer.state_dict()
        return d

    def load_state_dict(self, d: dict) -> None:
        self.agent.load_state_dict(d["agent"])
        self.target_agent.load_state_dict(d["target_agent"])
        self.optimizer.load_state_dict(d["optimizer"])
        self.episodes_seen = int(d["episodes_seen"])
        self._train_steps = int(d.get("train_steps", 0))
        if self.mixer is not None and "mixer" in d:
            self.mixer.load_state_dict(d["mixer"])
            self.target_mixer.load_state_dict(d["target_mixer"])


class MARLNDQLearner:
    """Recurrent NDQ learner with information-bottleneck communication loss."""

    def __init__(
        self,
        agent: NDQRNNAgent,
        comm_encoder: NDQCommEncoder,
        n_agents: int,
        n_actions: int,
        comm_embed_dim: int,
        gamma: float,
        lr: float,
        c_beta: float = 1.0,
        comm_beta: float = 0.001,
        comm_entropy_beta: float = 1e-6,
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

        self.comm_encoder = comm_encoder
        self.target_comm_encoder = copy.deepcopy(comm_encoder)
        for p in self.target_comm_encoder.parameters():
            p.requires_grad = False

        self.n_agents = int(n_agents)
        self.n_actions = int(n_actions)
        self.comm_embed_dim = int(comm_embed_dim)
        self.gamma = float(gamma)
        self.c_beta = float(c_beta)
        self.comm_beta = float(comm_beta)
        self.comm_entropy_beta = float(comm_entropy_beta)
        self.mixer_type = str(mixer_type)
        self.grad_clip_norm = grad_clip_norm
        self.target_update_steps = int(target_update_steps)
        self.td_lambda = td_lambda
        self.double_q = bool(double_q)

        self.device = device if device is not None else torch.device("cpu")
        self.agent.to(self.device)
        self.target_agent.to(self.device)
        self.comm_encoder.to(self.device)
        self.target_comm_encoder.to(self.device)

        self.mixer: Optional[nn.Module] = None
        self.target_mixer: Optional[nn.Module] = None
        if self.mixer_type == "vdn":
            self.mixer = VDNMixer().to(self.device)
            self.target_mixer = VDNMixer().to(self.device)
        elif self.mixer_type == "qmix":
            if int(state_dim) <= 0:
                raise ValueError("QMIX mixer requires state_dim > 0 (env global_state channel)")
            self.mixer = QMixer(
                n_agents=self.n_agents,
                state_dim=int(state_dim),
                mixing_hidden_dim=qmix_mixing_hidden_dim,
                hypernet_hidden_dim=qmix_hypernet_hidden_dim,
            ).to(self.device)
            self.target_mixer = copy.deepcopy(self.mixer).to(self.device)
            for p in self.target_mixer.parameters():
                p.requires_grad = False
        elif self.mixer_type != "none":
            raise ValueError("mixer_type must be one of: none, vdn, qmix")

        self._all_params = list(self.agent.parameters()) + list(self.comm_encoder.parameters())
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

        self.s_mu = torch.zeros(1, device=self.device)
        self.s_sigma = torch.ones(1, device=self.device)
        self.episodes_seen = 0
        self._train_steps = 0

    @staticmethod
    def _ndq_team_reward(batch: DialRNNEpisodeBatch, t: int) -> torch.Tensor:
        # ndq_rnn_collect_transition stores the already-summed team reward
        # repeated per agent so the IQL path can keep its existing per-agent
        # target shape. Mixed VDN/QMIX losses must consume it once.
        return batch.rewards[:, t, :1]

    def _forward_trace(
        self,
        agent: NDQRNNAgent,
        comm_encoder: NDQCommEncoder,
        batch: DialRNNEpisodeBatch,
    ) -> tuple[
        list[torch.Tensor], torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor],
    ]:
        batch_size, max_t, n_agents = batch.obs.shape[:3]
        hidden = agent.init_hidden(batch_size * n_agents).to(self.device)
        prev_action = torch.full(
            (batch_size, n_agents), self.n_actions, dtype=torch.long, device=self.device,
        )

        all_q: list[torch.Tensor] = []
        all_mu: list[torch.Tensor] = []
        all_m_sample: list[torch.Tensor] = []
        all_logits: list[torch.Tensor] = []
        for t in range(max_t):
            q_t, mu_t, m_t, logits_t, new_hidden = ndq_rnn_forward_batched(
                agent,
                comm_encoder,
                batch.obs[:, t],
                prev_action,
                hidden,
                n_actions=self.n_actions,
                comm_embed_dim=self.comm_embed_dim,
            )
            all_q.append(q_t)
            all_mu.append(mu_t)
            all_m_sample.append(m_t)
            all_logits.append(logits_t)

            valid = batch.mask[:, t].bool()
            valid_bn = valid.unsqueeze(1).expand(batch_size, n_agents).reshape(batch_size * n_agents)
            v_h = valid_bn.view(1, batch_size * n_agents, 1).expand_as(new_hidden)
            hidden = torch.where(v_h, new_hidden, hidden)
            v_a = valid.unsqueeze(1).expand(batch_size, n_agents)
            prev_action = torch.where(v_a, batch.actions[:, t], prev_action)

        q_boot, _, _, _, _ = ndq_rnn_forward_batched(
            agent,
            comm_encoder,
            batch.next_obs,
            prev_action,
            hidden,
            n_actions=self.n_actions,
            comm_embed_dim=self.comm_embed_dim,
        )
        return all_q, q_boot, all_mu, all_m_sample, all_logits

    def update(self, batch: DialRNNEpisodeBatch) -> None:
        batch_size = batch.obs.shape[0]

        online_q, online_q_boot, online_mu, online_m_sample, online_logits = self._forward_trace(
            self.agent, self.comm_encoder, batch,
        )
        with torch.no_grad():
            target_q, target_q_boot, _, _, _ = self._forward_trace(
                self.target_agent, self.target_comm_encoder, batch,
            )

        td_loss = self._compute_td_loss(
            batch, online_q, online_q_boot, target_q, target_q_boot,
        )
        comm_loss = self._compute_comm_loss(
            batch, online_mu, online_m_sample, online_logits, target_q,
        )
        loss = td_loss + self.c_beta * comm_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self._all_params, float(self.grad_clip_norm))
        self.optimizer.step()

        self.episodes_seen += batch_size
        self._train_steps += 1
        if self._train_steps % self.target_update_steps == 0:
            _hard_update(self.target_agent, self.agent)
            _hard_update(self.target_comm_encoder, self.comm_encoder)
            if self.target_mixer is not None and self.mixer is not None:
                _hard_update(self.target_mixer, self.mixer)

    def _compute_td_loss(
        self,
        batch: DialRNNEpisodeBatch,
        online_q_list: list[torch.Tensor],
        online_q_boot: torch.Tensor,
        target_q_list: list[torch.Tensor],
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

    def _compute_iql_td_loss(
        self,
        batch: DialRNNEpisodeBatch,
        online_q_list: list[torch.Tensor],
        online_q_boot: torch.Tensor,
        target_q_list: list[torch.Tensor],
        target_q_boot: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_t = batch.obs.shape[:2]
        n_agents = self.n_agents

        with torch.no_grad():
            if self.double_q:
                boot_actions = online_q_boot.argmax(dim=-1, keepdim=True)
                boot_q = target_q_boot.gather(-1, boot_actions).squeeze(-1)
            else:
                boot_q = target_q_boot.max(dim=-1).values

            targets: list[torch.Tensor] = []
            rewards: list[torch.Tensor] = []
            for t in range(max_t):
                if t < max_t - 1:
                    has_next = batch.mask[:, t + 1].bool()
                    if self.double_q:
                        next_actions = online_q_list[t + 1].argmax(dim=-1, keepdim=True)
                        next_q_vals = target_q_list[t + 1].gather(-1, next_actions).squeeze(-1)
                    else:
                        next_q_vals = target_q_list[t + 1].max(dim=-1).values
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
                targets = MARLDIALLearner._apply_td_lambda(
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
        batch: DialRNNEpisodeBatch,
        online_q_list: list[torch.Tensor],
        online_q_boot: torch.Tensor,
        target_q_list: list[torch.Tensor],
        target_q_boot: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_t = batch.obs.shape[:2]

        with torch.no_grad():
            if self.double_q:
                boot_actions = online_q_boot.argmax(dim=-1, keepdim=True)
                boot_q = target_q_boot.gather(-1, boot_actions).squeeze(-1)
            else:
                boot_q = target_q_boot.max(dim=-1).values

            targets: list[torch.Tensor] = []
            rewards: list[torch.Tensor] = []
            for t in range(max_t):
                if t < max_t - 1:
                    has_next = batch.mask[:, t + 1].bool()
                    if self.double_q:
                        next_actions = online_q_list[t + 1].argmax(dim=-1, keepdim=True)
                        next_q_vals = target_q_list[t + 1].gather(-1, next_actions).squeeze(-1)
                    else:
                        next_q_vals = target_q_list[t + 1].max(dim=-1).values
                    bootstrap_q = torch.where(
                        has_next.unsqueeze(1).expand(batch_size, self.n_agents),
                        next_q_vals,
                        boot_q,
                    )
                else:
                    bootstrap_q = boot_q
                next_q_tot = self.target_mixer(bootstrap_q)
                r_tot = self._ndq_team_reward(batch, t)
                rewards.append(r_tot)
                not_done = (1.0 - batch.terminated[:, t]).unsqueeze(-1)
                targets.append(r_tot + self.gamma * not_done * next_q_tot)
            if self.td_lambda > 0:
                targets = MARLDIALLearner._apply_td_lambda(
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
        batch: DialRNNEpisodeBatch,
        online_q_list: list[torch.Tensor],
        online_q_boot: torch.Tensor,
        target_q_list: list[torch.Tensor],
        target_q_boot: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_t = batch.obs.shape[:2]
        states = batch.states                    # (B, max_T, state_dim)
        boot_state = batch.next_states           # (B, state_dim)
        state_dim = states.shape[-1]

        with torch.no_grad():
            if self.double_q:
                boot_actions = online_q_boot.argmax(dim=-1, keepdim=True)
                boot_q = target_q_boot.gather(-1, boot_actions).squeeze(-1)
            else:
                boot_q = target_q_boot.max(dim=-1).values

            targets: list[torch.Tensor] = []
            rewards: list[torch.Tensor] = []
            for t in range(max_t):
                if t < max_t - 1:
                    has_next = batch.mask[:, t + 1].bool()
                    if self.double_q:
                        next_actions = online_q_list[t + 1].argmax(dim=-1, keepdim=True)
                        next_q_vals = target_q_list[t + 1].gather(-1, next_actions).squeeze(-1)
                    else:
                        next_q_vals = target_q_list[t + 1].max(dim=-1).values
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
                r_tot = self._ndq_team_reward(batch, t)
                rewards.append(r_tot)
                not_done = (1.0 - batch.terminated[:, t]).unsqueeze(-1)
                targets.append(r_tot + self.gamma * not_done * next_q_tot)
            if self.td_lambda > 0:
                targets = MARLDIALLearner._apply_td_lambda(
                    targets, rewards, batch.mask, batch.terminated, self.gamma, self.td_lambda,
                )

        total_loss = torch.tensor(0.0, device=self.device)
        for t in range(max_t):
            q_taken = online_q_list[t].gather(-1, batch.actions[:, t].unsqueeze(-1)).squeeze(-1)
            q_tot = self.mixer(q_taken, states[:, t])
            td_error = (q_tot - targets[t]) ** 2
            total_loss = total_loss + (td_error * batch.mask[:, t].unsqueeze(-1)).sum()
        return total_loss / batch.mask.sum().clamp_min(1.0)

    def _compute_comm_loss(
        self,
        batch: DialRNNEpisodeBatch,
        mu_list: list[torch.Tensor],
        m_sample_list: list[torch.Tensor],
        logits_list: list[torch.Tensor],
        target_q_list: list[torch.Tensor],
    ) -> torch.Tensor:
        mask = batch.mask
        mask_sum = mask.sum().clamp_min(1.0)
        mask_bt = mask.unsqueeze(-1).unsqueeze(-1)

        mu_out = torch.stack(mu_list, dim=1)
        m_sample_out = torch.stack(m_sample_list, dim=1)
        logits_out = torch.stack(logits_list, dim=1)
        target_q = torch.stack(target_q_list, dim=1)

        label_target_actions = target_q.max(dim=3, keepdim=True)[1]
        label_prob = logits_out.gather(3, label_target_actions).squeeze(3)
        expressiveness_loss = ((-torch.log(label_prob + 1e-6)) * mask.unsqueeze(-1)).sum() / mask_sum

        kl = torch.distributions.kl_divergence(
            torch.distributions.Normal(mu_out, torch.ones_like(mu_out)),
            torch.distributions.Normal(self.s_mu, self.s_sigma),
        )
        compactness_loss = (kl * mask_bt).sum() / mask_sum

        entropy = -torch.distributions.Normal(self.s_mu, self.s_sigma).log_prob(m_sample_out)
        entropy_loss = (entropy * mask_bt).sum() / mask_sum

        return expressiveness_loss + self.comm_beta * compactness_loss + self.comm_entropy_beta * entropy_loss

    def state_dict(self) -> dict:
        state = {
            "agent": self.agent.state_dict(),
            "target_agent": self.target_agent.state_dict(),
            "comm_encoder": self.comm_encoder.state_dict(),
            "target_comm_encoder": self.target_comm_encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "episodes_seen": self.episodes_seen,
            "train_steps": self._train_steps,
            "mixer_type": self.mixer_type,
        }
        if self.mixer is not None:
            state["mixer"] = self.mixer.state_dict()
            state["target_mixer"] = self.target_mixer.state_dict()
        return state

    def load_state_dict(self, d: dict) -> None:
        self.agent.load_state_dict(d["agent"])
        self.target_agent.load_state_dict(d["target_agent"])
        self.comm_encoder.load_state_dict(d["comm_encoder"])
        self.target_comm_encoder.load_state_dict(d["target_comm_encoder"])
        self.optimizer.load_state_dict(d["optimizer"])
        self.episodes_seen = int(d["episodes_seen"])
        self._train_steps = int(d.get("train_steps", 0))
        if self.mixer is not None and "mixer" in d:
            self.mixer.load_state_dict(d["mixer"])
            self.target_mixer.load_state_dict(d["target_mixer"])


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
        self._user_supplied_target_entropy = target_entropy is not None
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

    def update_target_entropy(self, n_valid_actions: int) -> None:
        """Adjust target entropy for curriculum phase changes.

        No-op when the user explicitly supplied ``--target-entropy``.
        """
        if self._user_supplied_target_entropy:
            return
        self.target_entropy = math.log(n_valid_actions) * 0.98

    def update(self, batch: MARLBatch, *, action_mask: Optional[torch.Tensor] = None) -> None:
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
            if action_mask is not None:
                logits = logits + action_mask
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
        self.log_alpha_critic.data.fill_(float(d["log_alpha_critic"]))
        self.train_steps = int(d["train_steps"])
        if self.auto_alpha and "alpha_optimizers" in d:
            for i, opt in enumerate(self.alpha_optimizers):
                opt.load_state_dict(d["alpha_optimizers"][i])
        if self.auto_alpha and "critic_alpha_optimizer" in d and self.critic_alpha_optimizer is not None:
            self.critic_alpha_optimizer.load_state_dict(d["critic_alpha_optimizer"])
