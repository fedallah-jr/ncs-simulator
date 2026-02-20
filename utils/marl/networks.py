from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from utils.marl.popart import PopArtLayer


def _get_activation(name: str) -> nn.Module:
    """Return an activation module instance by name."""
    if name == "relu":
        return nn.ReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "elu":
        return nn.ELU()
    raise ValueError("activation must be one of: relu, tanh, elu")


def _get_activation_class(name: str) -> type[nn.Module]:
    """Return an activation module class by name."""
    if name == "relu":
        return nn.ReLU
    elif name == "tanh":
        return nn.Tanh
    elif name == "elu":
        return nn.ELU
    raise ValueError("activation must be one of: relu, tanh, elu")


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
    ) -> None:
        super().__init__()
        if input_dim <= 0 or n_actions <= 0:
            raise ValueError("input_dim and n_actions must be positive")
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")

        act = _get_activation(activation)
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, int(hidden_dim)))
            layers.append(act)
            last_dim = int(hidden_dim)
        layers.append(nn.Linear(last_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
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
    ) -> None:
        super().__init__()
        if input_dim <= 0 or n_actions <= 0:
            raise ValueError("input_dim and n_actions must be positive")
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")

        act_fn = _get_activation_class(activation)

        # Shared feature extractor
        feature_layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            feature_layers.append(nn.Linear(last_dim, int(hidden_dim)))
            feature_layers.append(act_fn())
            last_dim = int(hidden_dim)
        self.features = nn.Sequential(*feature_layers)

        # Value stream: outputs V(s) scalar
        self.value_stream = nn.Sequential(
            nn.Linear(last_dim, stream_hidden_dim),
            act_fn(),
            nn.Linear(stream_hidden_dim, 1),
        )

        # Advantage stream: outputs A(s,a) for each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(last_dim, stream_hidden_dim),
            act_fn(),
            nn.Linear(stream_hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
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
        use_popart: bool = False,
        popart_beta: float = 0.999,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or n_outputs <= 0:
            raise ValueError("input_dim and n_outputs must be positive")
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")

        self._use_popart = use_popart

        act = _get_activation(activation)
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, int(hidden_dim)))
            layers.append(act)
            last_dim = int(hidden_dim)

        if use_popart:
            self._output_layer = PopArtLayer(last_dim, n_outputs, beta=popart_beta)
        else:
            self._output_layer = nn.Linear(last_dim, n_outputs)
        layers.append(self._output_layer)
        self.net = nn.Sequential(*layers)

    def popart_layer(self) -> PopArtLayer:
        """Return the PopArt output layer (raises if not using PopArt)."""
        if not self._use_popart:
            raise RuntimeError("CentralValueMLP was not constructed with use_popart=True")
        return self._output_layer  # type: ignore[return-value]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


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
