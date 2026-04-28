from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from utils.marl.popart import PopArtLayer


MIXER_INIT_GAIN = float(nn.init.calculate_gain("relu"))


def _get_activation(name: str) -> nn.Module:
    """Return an activation module instance by name."""
    if name == "relu":
        return nn.ReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "elu":
        return nn.ELU()
    raise ValueError("activation must be one of: relu, tanh, elu")


def _get_activation_gain(name: str) -> float:
    """Return orthogonal initialization gain for hidden layers."""
    if name == "relu":
        return float(nn.init.calculate_gain("relu"))
    elif name == "tanh":
        return float(nn.init.calculate_gain("tanh"))
    elif name == "elu":
        # PyTorch does not expose a dedicated ELU gain.
        # ReLU gain is a practical default for ELU-based MLPs.
        return float(nn.init.calculate_gain("relu"))
    raise ValueError("activation must be one of: relu, tanh, elu")


def _init_linear(linear: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(linear.weight, gain=gain)
    if linear.bias is not None:
        nn.init.constant_(linear.bias, 0.0)


def _apply_orthogonal_init(module: nn.Module, gain: float) -> None:
    """Apply orthogonal init to all linear layers in a module tree."""
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
            _init_linear(submodule, gain=gain)


def build_mlp_hidden(
    input_dim: int,
    hidden_dims: Sequence[int],
    activation: str = "relu",
    layer_norm: bool = False,
) -> tuple[nn.Sequential, int]:
    """Build hidden layers: [Linear -> Activation -> Optional LayerNorm] * N.

    Returns (sequential_module, last_hidden_dim).
    """
    act = _get_activation(activation)
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(act)
        if layer_norm:
            layers.append(nn.LayerNorm(int(hidden_dim)))
        last_dim = int(hidden_dim)
    return nn.Sequential(*layers), last_dim


_AGENT_ID_CACHE: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def append_agent_id(obs: torch.Tensor, n_agents: int) -> torch.Tensor:
    """
    Append a one-hot agent id to observations.

    Args:
        obs: Tensor of shape (batch, n_agents, obs_dim).
        n_agents: Number of agents.

    Returns:
        Tensor of shape (batch, n_agents, obs_dim + n_agents).
    """
    if obs.ndim != 3:
        raise ValueError("obs must have shape (batch, n_agents, obs_dim)")
    batch_size, agent_count, _ = obs.shape
    if agent_count != n_agents:
        raise ValueError("obs second dimension must equal n_agents")
    key = (n_agents, obs.device, obs.dtype)
    eye = _AGENT_ID_CACHE.get(key)
    if eye is None:
        eye = torch.eye(n_agents, device=obs.device, dtype=obs.dtype)
        _AGENT_ID_CACHE[key] = eye
    return torch.cat([obs, eye.unsqueeze(0).expand(batch_size, -1, -1)], dim=-1)


class MLPAgent(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        hidden_dims: Sequence[int] = (128, 128),
        activation: str = "relu",
        feature_norm: bool = False,
        layer_norm: bool = False,
        output_gain: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or n_actions <= 0:
            raise ValueError("input_dim and n_actions must be positive")
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")

        self.feature_norm_layer = nn.LayerNorm(input_dim) if feature_norm else None

        hidden, last_dim = build_mlp_hidden(input_dim, hidden_dims, activation, layer_norm)
        output_linear = nn.Linear(last_dim, n_actions)
        self.net = nn.Sequential(*hidden, output_linear)
        hidden_gain = _get_activation_gain(activation)
        _apply_orthogonal_init(self.net, gain=hidden_gain)
        _init_linear(output_linear, gain=float(output_gain))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.feature_norm_layer is not None:
            obs = self.feature_norm_layer(obs)
        return self.net(obs)


class DuelingMLPAgent(nn.Module):
    """
    Dueling DQN architecture that separates Q-values into value and advantage streams.

    Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))

    References:
        Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning"
        https://arxiv.org/abs/1511.06581
    """

    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        hidden_dims: Sequence[int] = (128, 128),
        stream_hidden_dim: int = 64,
        activation: str = "relu",
        feature_norm: bool = False,
        layer_norm: bool = False,
        output_gain: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or n_actions <= 0:
            raise ValueError("input_dim and n_actions must be positive")
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")

        self.feature_norm_layer = nn.LayerNorm(input_dim) if feature_norm else None

        # Shared feature extractor
        self.features, last_dim = build_mlp_hidden(input_dim, hidden_dims, activation, layer_norm)

        act = _get_activation(activation)
        # Value stream: outputs V(s) scalar
        self.value_stream = nn.Sequential(
            nn.Linear(last_dim, stream_hidden_dim),
            act,
            nn.Linear(stream_hidden_dim, 1),
        )

        # Advantage stream: outputs A(s,a) for each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(last_dim, stream_hidden_dim),
            _get_activation(activation),
            nn.Linear(stream_hidden_dim, n_actions),
        )
        hidden_gain = _get_activation_gain(activation)
        _apply_orthogonal_init(self, gain=hidden_gain)
        _init_linear(self.value_stream[-1], gain=float(output_gain))
        _init_linear(self.advantage_stream[-1], gain=float(output_gain))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.feature_norm_layer is not None:
            obs = self.feature_norm_layer(obs)
        features = self.features(obs)
        value = self.value_stream(features)  # [batch, 1]
        advantage = self.advantage_stream(features)  # [batch, n_actions]
        # Combine with mean subtraction for identifiability
        q = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q


class CentralValueMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_outputs: int,
        hidden_dims: Sequence[int] = (128, 128),
        activation: str = "relu",
        feature_norm: bool = False,
        layer_norm: bool = False,
        use_popart: bool = False,
        popart_beta: float = 0.999,
        output_gain: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or n_outputs <= 0:
            raise ValueError("input_dim and n_outputs must be positive")
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")

        self._use_popart = use_popart
        self.feature_norm_layer = nn.LayerNorm(input_dim) if feature_norm else None

        hidden, last_dim = build_mlp_hidden(input_dim, hidden_dims, activation, layer_norm)

        if use_popart:
            self._output_layer = PopArtLayer(last_dim, n_outputs, beta=popart_beta)
        else:
            self._output_layer = nn.Linear(last_dim, n_outputs)
        self.net = nn.Sequential(*hidden, self._output_layer)
        hidden_gain = _get_activation_gain(activation)
        _apply_orthogonal_init(self.net, gain=hidden_gain)
        if use_popart:
            _init_linear(self._output_layer.linear, gain=float(output_gain))
        else:
            _init_linear(self._output_layer, gain=float(output_gain))

    def popart_layer(self) -> PopArtLayer:
        """Return the PopArt output layer (raises if not using PopArt)."""
        if not self._use_popart:
            raise RuntimeError("CentralValueMLP was not constructed with use_popart=True")
        return self._output_layer  # type: ignore[return-value]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.feature_norm_layer is not None:
            obs = self.feature_norm_layer(obs)
        return self.net(obs)


class DRU(nn.Module):
    """Discretize/Regularize Unit for DIAL communication.

    During training: noisy sigmoid regularization.
    During evaluation: hard binary discretization.
    """

    def __init__(self, sigma: float = 2.0) -> None:
        super().__init__()
        self.sigma = float(sigma)

    def forward(
        self,
        msg_logits: torch.Tensor,
        train_mode: bool = True,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if train_mode:
            if noise is None:
                noise = torch.randn_like(msg_logits) * self.sigma
            return torch.sigmoid(msg_logits + noise)
        else:
            return (msg_logits > 0.0).float()


def route_messages(msg_post_dru: torch.Tensor, n_agents: int) -> torch.Tensor:
    """Route messages all-to-all with self-slot zeroed.

    Args:
        msg_post_dru: (batch, n_agents, comm_dim)
    Returns:
        recv_msgs: (batch, n_agents, n_agents * comm_dim)
    """
    B, N, C = msg_post_dru.shape
    expanded = msg_post_dru.unsqueeze(1).expand(B, N, N, C).clone()
    idx = torch.arange(N, device=msg_post_dru.device)
    expanded[:, idx, idx, :] = 0.0
    return expanded.reshape(B, N, N * C)


class DialRNNAgent(nn.Module):
    """GRU-based DIAL agent following the reference SwitchCNet architecture.

    Inputs are projected to rnn_hidden_dim and summed (not concatenated).
    A single output head produces both action Q-values and communication logits.
    BatchNorm follows the paper/reference on the message preprocessing path and
    the first layer of the output head.
    """

    init_param_range = (-0.08, 0.08)

    def __init__(
        self,
        obs_dim: int,
        n_agents: int,
        n_actions: int,
        comm_dim: int,
        rnn_hidden_dim: int = 128,
        rnn_layers: int = 2,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.comm_dim = comm_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_layers = rnn_layers

        # Input encoders — all project to rnn_hidden_dim, then summed
        self.obs_encoder = nn.Linear(obs_dim, rnn_hidden_dim)
        self.agent_lookup = nn.Embedding(n_agents, rnn_hidden_dim)
        self.prev_action_lookup = nn.Embedding(n_actions + 1, rnn_hidden_dim)  # +1 for start token
        self.msg_encoder = nn.Sequential(
            # The paper/reference batch-normalize the incoming message vector
            # before the message MLP. In this codebase the receiver sees the
            # routed all-to-all message vector, so BN is applied to that full
            # flattened receive tensor of shape (batch * agents, N * comm_dim).
            nn.BatchNorm1d(n_agents * comm_dim),
            nn.Linear(n_agents * comm_dim, rnn_hidden_dim),
            nn.ReLU(),
        )

        # GRU core
        self.rnn = nn.GRU(
            input_size=rnn_hidden_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
        )

        # Single combined output head (reference-faithful)
        self.output_head = nn.Sequential(
            nn.Linear(rnn_hidden_dim, rnn_hidden_dim),
            nn.BatchNorm1d(rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(rnn_hidden_dim, n_actions + comm_dim),
        )
        self.reset_parameters()

    def forward(
        self,
        obs: torch.Tensor,
        agent_idx: torch.Tensor,
        prev_action: torch.Tensor,
        recv_msg: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs:         (B*N, obs_dim)
            agent_idx:   (B*N,)          int64
            prev_action: (B*N,)          int64
            recv_msg:    (B*N, N*comm_dim)
            hidden:      (rnn_layers, B*N, rnn_hidden_dim)
        Returns:
            q_values:    (B*N, n_actions)
            msg_logits:  (B*N, comm_dim)
            h_out:       (rnn_layers, B*N, rnn_hidden_dim)
        """
        z = (
            self.obs_encoder(obs)
            + self.agent_lookup(agent_idx)
            + self.prev_action_lookup(prev_action)
            + self.msg_encoder(recv_msg)
        )
        rnn_out, h_out = self.rnn(z.unsqueeze(1), hidden)
        out = self.output_head(rnn_out.squeeze(1))
        return out[:, : self.n_actions], out[:, self.n_actions :], h_out

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """Return zero initial hidden state of shape (rnn_layers, batch_size, rnn_hidden_dim)."""
        return torch.zeros(self.rnn_layers, batch_size, self.rnn_hidden_dim)

    def reset_parameters(self) -> None:
        """Reference-style parameter reset adapted to the continuous-observation model."""
        self.obs_encoder.reset_parameters()
        self.agent_lookup.reset_parameters()
        self.prev_action_lookup.reset_parameters()
        self.msg_encoder[0].reset_parameters()
        self.msg_encoder[1].reset_parameters()
        self.rnn.reset_parameters()
        self.output_head[0].reset_parameters()
        self.output_head[1].reset_parameters()
        self.output_head[3].reset_parameters()
        for param in self.rnn.parameters():
            param.data.uniform_(*self.init_param_range)


@contextmanager
def _dial_batchnorm_fallback_eval(agent: DialRNNAgent, enabled: bool):
    """Temporarily use BN running stats for degenerate per-agent batch size 1.

    The reference code always trains with batch size > 1 per agent. In this
    codebase, truncated vectorized segments can occasionally produce a single
    sample. For that case only, use running stats instead of mixing agents into
    one BatchNorm batch.
    """
    if not enabled:
        yield
        return

    restored: list[tuple[nn.BatchNorm1d, bool]] = []
    for module in agent.modules():
        if isinstance(module, nn.BatchNorm1d):
            restored.append((module, module.training))
            module.eval()
    try:
        yield
    finally:
        for module, was_training in restored:
            module.train(was_training)


def dial_rnn_forward_batched(
    agent: DialRNNAgent,
    obs: torch.Tensor,
    prev_action: torch.Tensor,
    recv_msg: torch.Tensor,
    hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run shared-parameter DIAL one agent at a time over a vectorized batch.

    This preserves the reference BatchNorm semantics: the shared network is
    reused across agents, but BatchNorm statistics are computed over the batch
    of environments/episodes for a single agent instead of mixing agents
    together inside one flattened batch.

    Args:
        obs:         (B, N, obs_dim)
        prev_action: (B, N)
        recv_msg:    (B, N, N * comm_dim)
        hidden:      (rnn_layers, B * N, rnn_hidden_dim)
    Returns:
        q_values:    (B, N, n_actions)
        msg_logits:  (B, N, comm_dim)
        h_out:       (rnn_layers, B * N, rnn_hidden_dim)
    """
    if obs.ndim != 3:
        raise ValueError("obs must have shape (B, N, obs_dim)")
    if prev_action.shape != obs.shape[:2]:
        raise ValueError("prev_action must have shape (B, N)")
    if recv_msg.ndim != 3 or recv_msg.shape[:2] != obs.shape[:2]:
        raise ValueError("recv_msg must have shape (B, N, N * comm_dim)")
    if hidden.ndim != 3:
        raise ValueError("hidden must have shape (rnn_layers, B * N, rnn_hidden_dim)")

    B, N = obs.shape[:2]
    if N != agent.n_agents:
        raise ValueError("obs agent dimension must equal agent.n_agents")

    hidden_view = hidden.reshape(agent.rnn_layers, B, N, agent.rnn_hidden_dim)
    q_per_agent: list[torch.Tensor] = []
    m_per_agent: list[torch.Tensor] = []
    h_per_agent: list[torch.Tensor] = []

    # BatchNorm cannot estimate variance from a single sample. In that
    # degenerate vectorized case, preserve per-agent isolation and fall back to
    # running stats instead of mixing multiple agents into one BN batch.
    with _dial_batchnorm_fallback_eval(agent, enabled=bool(agent.training and B < 2)):
        for agent_id in range(N):
            agent_idx = torch.full((B,), agent_id, device=obs.device, dtype=torch.long)
            q_i, m_i, h_i = agent(
                obs[:, agent_id, :],
                agent_idx,
                prev_action[:, agent_id],
                recv_msg[:, agent_id, :],
                hidden_view[:, :, agent_id, :].contiguous(),
            )
            q_per_agent.append(q_i)
            m_per_agent.append(m_i)
            h_per_agent.append(h_i)

    q_values = torch.stack(q_per_agent, dim=1)
    msg_logits = torch.stack(m_per_agent, dim=1)
    h_out = torch.stack(h_per_agent, dim=2).reshape(
        agent.rnn_layers, B * N, agent.rnn_hidden_dim,
    )
    return q_values, msg_logits, h_out


class NDQRNNAgent(nn.Module):
    """GRU-based NDQ agent with separate communication encoder."""

    def __init__(
        self,
        obs_dim: int,
        n_agents: int,
        n_actions: int,
        comm_embed_dim: int,
        rnn_hidden_dim: int = 64,
        rnn_layers: int = 1,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_agents = int(n_agents)
        self.n_actions = int(n_actions)
        self.comm_embed_dim = int(comm_embed_dim)
        self.rnn_hidden_dim = int(rnn_hidden_dim)
        self.rnn_layers = int(rnn_layers)

        self.obs_encoder = nn.Linear(obs_dim, rnn_hidden_dim)
        self.agent_lookup = nn.Embedding(n_agents, rnn_hidden_dim)
        self.prev_action_lookup = nn.Embedding(n_actions + 1, rnn_hidden_dim)
        self.msg_encoder = nn.Sequential(
            nn.Linear(n_agents * comm_embed_dim, rnn_hidden_dim),
            nn.ReLU(),
        )
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
        recv_msg: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = (
            self.obs_encoder(obs)
            + self.agent_lookup(agent_idx)
            + self.prev_action_lookup(prev_action)
            + self.msg_encoder(recv_msg)
        )
        rnn_out, h_out = self.rnn(z.unsqueeze(1), hidden)
        q_values = self.output_head(rnn_out.squeeze(1))
        return q_values, h_out

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(self.rnn_layers, batch_size, self.rnn_hidden_dim)


class NDQCommEncoder(nn.Module):
    """Continuous communication encoder used by NDQ."""

    def __init__(
        self,
        input_dim: int,
        n_agents: int,
        comm_embed_dim: int,
        n_actions: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_agents = int(n_agents)
        self.comm_embed_dim = int(comm_embed_dim)
        self.n_actions = int(n_actions)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_agents * comm_embed_dim)
        infer_hidden = 4 * n_agents * comm_embed_dim
        self.inference_model = nn.Sequential(
            nn.Linear(input_dim + n_agents * comm_embed_dim, infer_hidden),
            nn.ReLU(),
            nn.Linear(infer_hidden, infer_hidden),
            nn.ReLU(),
            nn.Linear(infer_hidden, n_actions),
        )

    def forward(self, base_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(base_input))
        x = F.relu(self.fc2(x))
        mu = self.fc3(x)
        sigma = torch.ones_like(mu)
        return mu, sigma

    def infer(self, base_input: torch.Tensor, recv_msg: torch.Tensor) -> torch.Tensor:
        logits = self.inference_model(torch.cat([base_input, recv_msg], dim=-1))
        return F.softmax(logits, dim=-1)


def route_ndq_messages(
    messages: torch.Tensor, n_agents: int, comm_embed_dim: int,
) -> torch.Tensor:
    """Route NDQ messages from sender-major to receiver-major layout."""
    batch_size = messages.shape[0]
    msg = messages.view(batch_size, n_agents, n_agents, comm_embed_dim)
    msg = msg.permute(0, 2, 1, 3).contiguous()
    return msg.view(batch_size, n_agents, n_agents * comm_embed_dim)


def ndq_rnn_forward_batched(
    agent: NDQRNNAgent,
    comm_encoder: NDQCommEncoder,
    obs: torch.Tensor,
    prev_action: torch.Tensor,
    hidden: torch.Tensor,
    *,
    n_actions: int,
    comm_embed_dim: int,
    cut_mu_thres: float = 0.0,
    comm_stats: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one NDQ recurrent step over a vectorized batch.

    When ``cut_mu_thres > 0``, message dimensions with ``|mu| < thres`` are
    zeroed before routing — the inference-time gating from the NDQ paper
    (``_cut_mu`` in ``cate_broadcast_comm_controller_full.py``). Self-slots
    (sender i's own receiver slot) are always considered "not sent" since
    they are overwritten during routing.

    If ``comm_stats`` is provided, accumulates four integer keys:
    ``total_dims``, ``dropped_dims``, ``total_messages``, ``silent_messages``
    — counted over off-diagonal (sender, receiver, embed) triples.
    """
    if obs.ndim != 3:
        raise ValueError("obs must have shape (B, N, obs_dim)")
    if prev_action.shape != obs.shape[:2]:
        raise ValueError("prev_action must have shape (B, N)")

    batch_size, n_agents, _ = obs.shape
    agent_ids_onehot = torch.eye(
        n_agents, device=obs.device, dtype=obs.dtype,
    ).unsqueeze(0).expand(batch_size, -1, -1)

    prev_action_safe = prev_action.clamp(max=n_actions - 1)
    prev_action_onehot = F.one_hot(prev_action_safe, n_actions).to(dtype=obs.dtype)
    start_mask = prev_action >= n_actions
    prev_action_onehot[start_mask] = 0.0

    base_input = torch.cat([obs, prev_action_onehot, agent_ids_onehot], dim=-1)
    base_flat = base_input.reshape(batch_size * n_agents, -1)
    mu, _sigma = comm_encoder(base_flat)

    m_sample = mu + torch.randn_like(mu)
    if cut_mu_thres > 0.0 or comm_stats is not None:
        mu_grid = mu.view(batch_size, n_agents, n_agents, comm_embed_dim)
        diag = torch.eye(n_agents, device=mu.device, dtype=torch.bool)
        off_diag = ~diag
        if cut_mu_thres > 0.0:
            keep_dim = mu_grid.abs() >= float(cut_mu_thres)
            m_grid = m_sample.view(batch_size, n_agents, n_agents, comm_embed_dim)
            m_grid = m_grid * keep_dim.to(m_sample.dtype)
            m_sample = m_grid.view(batch_size * n_agents, n_agents * comm_embed_dim)
        if comm_stats is not None:
            if cut_mu_thres > 0.0:
                kept = keep_dim
            else:
                kept = torch.ones_like(mu_grid, dtype=torch.bool)
            off_mask = off_diag.view(1, n_agents, n_agents, 1)
            total_dims = int(off_mask.expand_as(kept).sum().item())
            dropped_dims = int(((~kept) & off_mask).sum().item())
            pair_any_kept = (kept & off_mask).any(dim=-1)
            total_messages = int(off_diag.sum().item()) * batch_size
            silent_messages = int((~pair_any_kept & off_diag.view(1, n_agents, n_agents)).sum().item())
            comm_stats["total_dims"] = comm_stats.get("total_dims", 0) + total_dims
            comm_stats["dropped_dims"] = comm_stats.get("dropped_dims", 0) + dropped_dims
            comm_stats["total_messages"] = comm_stats.get("total_messages", 0) + total_messages
            comm_stats["silent_messages"] = comm_stats.get("silent_messages", 0) + silent_messages
    messages = m_sample.view(batch_size, n_agents, n_agents * comm_embed_dim)
    recv_msg = route_ndq_messages(messages, n_agents, comm_embed_dim)
    recv_flat = recv_msg.reshape(batch_size * n_agents, -1)

    agent_idx = torch.arange(
        n_agents, device=obs.device, dtype=torch.long,
    ).unsqueeze(0).expand(batch_size, -1).reshape(batch_size * n_agents)
    q_values, h_out = agent(
        obs.reshape(batch_size * n_agents, -1),
        agent_idx,
        prev_action.reshape(batch_size * n_agents),
        recv_flat,
        hidden,
    )

    logits = comm_encoder.infer(base_flat, recv_flat)
    return (
        q_values.view(batch_size, n_agents, -1),
        mu.view(batch_size, n_agents, -1),
        m_sample.view(batch_size, n_agents, -1),
        logits.view(batch_size, n_agents, -1),
        h_out,
    )


class VDNMixer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, agent_qs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_qs: Tensor of shape (batch, n_agents).
        Returns:
            q_tot: Tensor of shape (batch, 1).
        """
        if agent_qs.ndim != 2:
            raise ValueError("agent_qs must have shape (batch, n_agents)")
        return agent_qs.sum(dim=1, keepdim=True)


class QMixer(nn.Module):
    """
    QMIX mixer with hypernetworks (MLP variant).

    References:
        Rashid et al., "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning"
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        mixing_hidden_dim: int = 32,
        hypernet_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if n_agents <= 0 or state_dim <= 0:
            raise ValueError("n_agents and state_dim must be positive")
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.mixing_hidden_dim = mixing_hidden_dim

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, n_agents * mixing_hidden_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, mixing_hidden_dim)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, mixing_hidden_dim * 1),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, 1),
        )
        _apply_orthogonal_init(self, gain=MIXER_INIT_GAIN)

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_qs: Tensor of shape (batch, n_agents).
            states: Tensor of shape (batch, state_dim).
        Returns:
            q_tot: Tensor of shape (batch, 1).
        """
        if agent_qs.ndim != 2:
            raise ValueError("agent_qs must have shape (batch, n_agents)")
        if states.ndim != 2:
            raise ValueError("states must have shape (batch, state_dim)")
        batch_size, n_agents = agent_qs.shape
        if n_agents != self.n_agents:
            raise ValueError("agent_qs second dimension must equal mixer n_agents")
        if states.shape[1] != self.state_dim:
            raise ValueError("states second dimension must equal mixer state_dim")

        w1 = torch.abs(self.hyper_w1(states)).view(batch_size, self.n_agents, self.mixing_hidden_dim)
        b1 = self.hyper_b1(states).view(batch_size, 1, self.mixing_hidden_dim)

        agent_qs_ = agent_qs.view(batch_size, 1, self.n_agents)
        hidden = F.elu(torch.bmm(agent_qs_, w1) + b1)

        w2 = torch.abs(self.hyper_w2(states)).view(batch_size, self.mixing_hidden_dim, 1)
        b2 = self.hyper_b2(states).view(batch_size, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(batch_size, 1)


class TwinQNetwork(nn.Module):
    """Centralized twin Q-critic for HASAC.

    Input: concat(global_state, all_agents_one_hot_actions).
    Output: two scalar Q-values from independent Q-networks.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "relu",
        feature_norm: bool = False,
        layer_norm: bool = False,
        output_gain: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")

        # Keep the two Q-functions fully independent, including input normalization.
        self.q1_feature_norm_layer = nn.LayerNorm(input_dim) if feature_norm else None
        self.q2_feature_norm_layer = nn.LayerNorm(input_dim) if feature_norm else None

        hidden1, last_dim1 = build_mlp_hidden(input_dim, hidden_dims, activation, layer_norm)
        out1 = nn.Linear(last_dim1, 1)
        self.q1 = nn.Sequential(*hidden1, out1)

        hidden2, last_dim2 = build_mlp_hidden(input_dim, hidden_dims, activation, layer_norm)
        out2 = nn.Linear(last_dim2, 1)
        self.q2 = nn.Sequential(*hidden2, out2)

        hidden_gain = _get_activation_gain(activation)
        _apply_orthogonal_init(self.q1, gain=hidden_gain)
        _apply_orthogonal_init(self.q2, gain=hidden_gain)
        _init_linear(out1, gain=float(output_gain))
        _init_linear(out2, gain=float(output_gain))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = self.q1_feature_norm_layer(x) if self.q1_feature_norm_layer is not None else x
        x2 = self.q2_feature_norm_layer(x) if self.q2_feature_norm_layer is not None else x
        return self.q1(x1).squeeze(-1), self.q2(x2).squeeze(-1)

    def q1_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.q1_feature_norm_layer is not None:
            x = self.q1_feature_norm_layer(x)
        return self.q1(x).squeeze(-1)


class QPLEXSIWeight(nn.Module):
    def __init__(
        self,
        n_agents: int,
        n_actions: int,
        state_dim: int,
        num_kernel: int = 10,
        adv_hypernet_layers: int = 3,
        adv_hypernet_embed: int = 64,
    ) -> None:
        super().__init__()
        if n_agents <= 0 or n_actions <= 0 or state_dim <= 0:
            raise ValueError("n_agents, n_actions, and state_dim must be positive")
        if num_kernel <= 0:
            raise ValueError("num_kernel must be positive")
        if adv_hypernet_layers <= 0:
            raise ValueError("adv_hypernet_layers must be positive")
        if adv_hypernet_embed <= 0:
            raise ValueError("adv_hypernet_embed must be positive")

        self.n_agents = int(n_agents)
        self.n_actions = int(n_actions)
        self.state_dim = int(state_dim)
        self.action_dim = self.n_agents * self.n_actions
        self.num_kernel = int(num_kernel)

        self.key_extractors = nn.ModuleList()
        self.agents_extractors = nn.ModuleList()
        self.action_extractors = nn.ModuleList()

        for _ in range(self.num_kernel):
            self.key_extractors.append(
                self._make_mlp(self.state_dim, 1, adv_hypernet_embed, adv_hypernet_layers)
            )
            self.agents_extractors.append(
                self._make_mlp(self.state_dim, self.n_agents, adv_hypernet_embed, adv_hypernet_layers)
            )
            self.action_extractors.append(
                self._make_mlp(
                    self.state_dim + self.action_dim,
                    self.n_agents,
                    adv_hypernet_embed,
                    adv_hypernet_layers,
                )
            )
        _apply_orthogonal_init(self, gain=MIXER_INIT_GAIN)

    @staticmethod
    def _make_mlp(input_dim: int, output_dim: int, hidden_dim: int, n_layers: int) -> nn.Module:
        if n_layers == 1:
            return nn.Linear(input_dim, output_dim)
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, states: torch.Tensor, actions_onehot: torch.Tensor) -> torch.Tensor:
        if states.ndim != 2:
            raise ValueError("states must have shape (batch, state_dim)")
        if actions_onehot.ndim != 3:
            raise ValueError("actions_onehot must have shape (batch, n_agents, n_actions)")
        batch_size = states.shape[0]
        if states.shape[1] != self.state_dim:
            raise ValueError("states second dimension must equal state_dim")
        if actions_onehot.shape[1] != self.n_agents or actions_onehot.shape[2] != self.n_actions:
            raise ValueError("actions_onehot dimensions must match n_agents and n_actions")

        actions_flat = actions_onehot.to(dtype=states.dtype).reshape(batch_size, -1)
        data = torch.cat([states, actions_flat], dim=1)

        head_weights: list[torch.Tensor] = []
        for key_extractor, agents_extractor, action_extractor in zip(
            self.key_extractors, self.agents_extractors, self.action_extractors
        ):
            x_key = torch.abs(key_extractor(states)).expand(-1, self.n_agents) + 1e-10
            x_agents = torch.sigmoid(agents_extractor(states))
            x_action = torch.sigmoid(action_extractor(data))
            head_weights.append(x_key * x_agents * x_action)

        head_attend = torch.stack(head_weights, dim=1).sum(dim=1)
        return head_attend


class QPLEXQattenWeight(nn.Module):
    def __init__(
        self,
        n_agents: int,
        n_actions: int,
        state_dim: int,
        unit_dim: int,
        mixing_embed_dim: int = 32,
        hypernet_embed: int = 64,
        n_head: int = 4,
        attend_reg_coef: float = 0.001,
        weighted_head: bool = False,
        state_bias: bool = True,
        nonlinear: bool = False,
    ) -> None:
        super().__init__()
        if n_agents <= 0 or n_actions <= 0 or state_dim <= 0:
            raise ValueError("n_agents, n_actions, and state_dim must be positive")
        if unit_dim <= 0:
            raise ValueError("unit_dim must be positive")
        if mixing_embed_dim <= 0:
            raise ValueError("mixing_embed_dim must be positive")
        if hypernet_embed <= 0:
            raise ValueError("hypernet_embed must be positive")
        if n_head <= 0:
            raise ValueError("n_head must be positive")
        if attend_reg_coef < 0:
            raise ValueError("attend_reg_coef must be non-negative")
        if state_dim < n_agents * unit_dim:
            raise ValueError("state_dim must be >= n_agents * unit_dim")

        self.n_agents = int(n_agents)
        self.n_actions = int(n_actions)
        self.state_dim = int(state_dim)
        self.unit_dim = int(unit_dim)
        self.embed_dim = int(mixing_embed_dim)
        self.n_head = int(n_head)
        self.attend_reg_coef = float(attend_reg_coef)
        self.weighted_head = bool(weighted_head)
        self.state_bias = bool(state_bias)
        self.nonlinear = bool(nonlinear)

        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()

        key_input_dim = self.unit_dim + (1 if self.nonlinear else 0)
        for _ in range(self.n_head):
            self.selector_extractors.append(
                nn.Sequential(
                    nn.Linear(self.state_dim, hypernet_embed),
                    nn.ReLU(),
                    nn.Linear(hypernet_embed, self.embed_dim, bias=False),
                )
            )
            self.key_extractors.append(nn.Linear(key_input_dim, self.embed_dim, bias=False))

        if self.weighted_head:
            self.hyper_w_head = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.n_head),
            )

        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )
        _apply_orthogonal_init(self, gain=MIXER_INIT_GAIN)

    def forward(
        self,
        agent_qs: torch.Tensor,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        if agent_qs.ndim != 2:
            raise ValueError("agent_qs must have shape (batch, n_agents)")
        if states.ndim != 2:
            raise ValueError("states must have shape (batch, state_dim)")
        if states.shape[1] != self.state_dim:
            raise ValueError("states second dimension must equal state_dim")
        if agent_qs.shape[1] != self.n_agents:
            raise ValueError("agent_qs second dimension must equal n_agents")

        states = states.reshape(-1, self.state_dim)
        unit_states = states[:, : self.unit_dim * self.n_agents]
        unit_states = unit_states.reshape(-1, self.n_agents, self.unit_dim)
        unit_states = unit_states.permute(1, 0, 2)

        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        if self.nonlinear:
            unit_states = torch.cat((unit_states, agent_qs.permute(2, 0, 1)), dim=2)

        all_head_selectors = [sel_ext(states) for sel_ext in self.selector_extractors]
        all_head_keys = [[k_ext(enc) for enc in unit_states] for k_ext in self.key_extractors]

        head_attend_logits: list[torch.Tensor] = []
        head_attend_weights: list[torch.Tensor] = []
        for curr_head_keys, curr_head_selector in zip(all_head_keys, all_head_selectors):
            attend_logits = torch.matmul(
                curr_head_selector.view(-1, 1, self.embed_dim),
                torch.stack(curr_head_keys).permute(1, 2, 0),
            )
            scaled_attend_logits = attend_logits / math.sqrt(self.embed_dim)
            attend_weights = F.softmax(scaled_attend_logits, dim=2)
            head_attend_logits.append(attend_logits)
            head_attend_weights.append(attend_weights)

        head_attend = torch.stack(head_attend_weights, dim=1).view(-1, self.n_head, self.n_agents)
        v = self.V(states).view(-1, 1)

        if self.weighted_head:
            w_head = torch.abs(self.hyper_w_head(states))
            w_head = w_head.view(-1, self.n_head, 1).expand(-1, -1, self.n_agents)
            head_attend = head_attend * w_head

        head_attend = torch.sum(head_attend, dim=1)

        if not self.state_bias:
            v = v * 0.0

        attend_mag_regs = self.attend_reg_coef * sum((logit ** 2).mean() for logit in head_attend_logits)
        head_entropies = [
            -(probs * (probs + 1e-8).log()).sum(dim=2).mean() for probs in head_attend_weights
        ]

        return head_attend, v, attend_mag_regs, head_entropies


class QPLEXMixer(nn.Module):
    def __init__(
        self,
        n_agents: int,
        n_actions: int,
        state_dim: int,
        unit_dim: Optional[int] = None,
        mixing_embed_dim: int = 32,
        hypernet_embed: int = 64,
        num_kernel: int = 10,
        adv_hypernet_layers: int = 3,
        adv_hypernet_embed: int = 64,
        n_head: int = 4,
        attend_reg_coef: float = 0.001,
        state_bias: bool = True,
        nonlinear: bool = False,
        weighted_head: bool = True,
        is_minus_one: bool = True,
    ) -> None:
        super().__init__()
        if n_agents <= 0 or n_actions <= 0 or state_dim <= 0:
            raise ValueError("n_agents, n_actions, and state_dim must be positive")
        if hypernet_embed <= 0:
            raise ValueError("hypernet_embed must be positive")
        if unit_dim is None:
            if state_dim % n_agents != 0:
                raise ValueError("state_dim must be divisible by n_agents when unit_dim is not provided")
            unit_dim = state_dim // n_agents
        if unit_dim <= 0:
            raise ValueError("unit_dim must be positive")
        self.n_agents = int(n_agents)
        self.n_actions = int(n_actions)
        self.state_dim = int(state_dim)
        self.unit_dim = int(unit_dim)
        self.mixing_embed_dim = int(mixing_embed_dim)
        self.weighted_head = bool(weighted_head)
        self.is_minus_one = bool(is_minus_one)
        self.attention_weight = QPLEXQattenWeight(
            n_agents=self.n_agents,
            n_actions=self.n_actions,
            state_dim=self.state_dim,
            unit_dim=self.unit_dim,
            mixing_embed_dim=self.mixing_embed_dim,
            hypernet_embed=hypernet_embed,
            n_head=n_head,
            attend_reg_coef=attend_reg_coef,
            weighted_head=weighted_head,
            state_bias=state_bias,
            nonlinear=nonlinear,
        )
        self.si_weight = QPLEXSIWeight(
            n_agents=self.n_agents,
            n_actions=self.n_actions,
            state_dim=self.state_dim,
            num_kernel=num_kernel,
            adv_hypernet_layers=adv_hypernet_layers,
            adv_hypernet_embed=adv_hypernet_embed,
        )

    def _calc_v(self, agent_qs: torch.Tensor) -> torch.Tensor:
        if agent_qs.ndim != 2:
            raise ValueError("agent_qs must have shape (batch, n_agents)")
        return agent_qs.sum(dim=1, keepdim=True)

    def _calc_adv(
        self,
        agent_qs: torch.Tensor,
        states: torch.Tensor,
        actions_onehot: torch.Tensor,
        max_q_i: torch.Tensor,
    ) -> torch.Tensor:
        adv_q = (agent_qs - max_q_i).detach()
        adv_w = self.si_weight(states, actions_onehot)
        if self.is_minus_one:
            return (adv_q * (adv_w - 1.0)).sum(dim=1, keepdim=True)
        return (adv_q * adv_w).sum(dim=1, keepdim=True)

    def forward(
        self,
        agent_qs: torch.Tensor,
        states: torch.Tensor,
        actions_onehot: Optional[torch.Tensor] = None,
        max_q_i: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        if agent_qs.ndim != 2:
            raise ValueError("agent_qs must have shape (batch, n_agents)")
        if states.ndim != 2:
            raise ValueError("states must have shape (batch, state_dim)")
        batch_size, n_agents = agent_qs.shape
        if n_agents != self.n_agents:
            raise ValueError("agent_qs second dimension must equal n_agents")
        if states.shape[0] != batch_size or states.shape[1] != self.state_dim:
            raise ValueError("states must have shape (batch, state_dim)")

        w_final, v, attend_mag_regs, head_entropies = self.attention_weight(
            agent_qs=agent_qs,
            states=states,
        )
        w_final = w_final.view(-1, self.n_agents) + 1e-10
        v = v.view(-1, 1).expand(-1, self.n_agents) / float(self.n_agents)

        agent_qs = agent_qs.view(-1, self.n_agents)
        agent_qs = w_final * agent_qs + v

        # V component (always computed)
        q_tot = self._calc_v(agent_qs)

        # A component (when advantage inputs are provided)
        if actions_onehot is not None and max_q_i is not None:
            max_q_i = max_q_i.view(-1, self.n_agents)
            max_q_i = w_final * max_q_i + v
            q_tot = q_tot + self._calc_adv(agent_qs, states, actions_onehot, max_q_i)

        return q_tot, attend_mag_regs, head_entropies
