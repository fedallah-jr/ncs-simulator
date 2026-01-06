from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F


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
    eye = torch.eye(n_agents, device=obs.device, dtype=obs.dtype).unsqueeze(0).expand(batch_size, -1, -1)
    return torch.cat([obs, eye], dim=-1)


class MLPAgent(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        hidden_dims: Sequence[int] = (128, 128),
        activation: str = "relu",
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or n_actions <= 0:
            raise ValueError("input_dim and n_actions must be positive")
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")

        act: nn.Module
        if activation == "relu":
            act = nn.ReLU()
        elif activation == "tanh":
            act = nn.Tanh()
        elif activation == "elu":
            act = nn.ELU()
        else:
            raise ValueError("activation must be one of: relu, tanh, elu")

        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, int(hidden_dim)))
            if layer_norm:
                layers.append(nn.LayerNorm(int(hidden_dim)))
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
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or n_actions <= 0:
            raise ValueError("input_dim and n_actions must be positive")
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")

        act_fn: type[nn.Module]
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "elu":
            act_fn = nn.ELU
        else:
            raise ValueError("activation must be one of: relu, tanh, elu")

        # Shared feature extractor
        feature_layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            feature_layers.append(nn.Linear(last_dim, int(hidden_dim)))
            if layer_norm:
                feature_layers.append(nn.LayerNorm(int(hidden_dim)))
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
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or n_outputs <= 0:
            raise ValueError("input_dim and n_outputs must be positive")
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")

        act: nn.Module
        if activation == "relu":
            act = nn.ReLU()
        elif activation == "tanh":
            act = nn.Tanh()
        elif activation == "elu":
            act = nn.ELU()
        else:
            raise ValueError("activation must be one of: relu, tanh, elu")

        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, int(hidden_dim)))
            if layer_norm:
                layers.append(nn.LayerNorm(int(hidden_dim)))
            layers.append(act)
            last_dim = int(hidden_dim)
        layers.append(nn.Linear(last_dim, n_outputs))
        self.net = nn.Sequential(*layers)

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
            x_key = torch.abs(key_extractor(states)).repeat(1, self.n_agents) + 1e-10
            x_agents = torch.sigmoid(agents_extractor(states))
            x_action = torch.sigmoid(action_extractor(data))
            head_weights.append(x_key * x_agents * x_action)

        head_attend = torch.stack(head_weights, dim=1).sum(dim=1)
        return head_attend


class QPLEXMixer(nn.Module):
    def __init__(
        self,
        n_agents: int,
        n_actions: int,
        state_dim: int,
        mixing_embed_dim: int = 32,
        hypernet_embed: int = 64,
        num_kernel: int = 10,
        adv_hypernet_layers: int = 3,
        adv_hypernet_embed: int = 64,
        weighted_head: bool = True,
        is_minus_one: bool = True,
    ) -> None:
        super().__init__()
        if n_agents <= 0 or n_actions <= 0 or state_dim <= 0:
            raise ValueError("n_agents, n_actions, and state_dim must be positive")
        if hypernet_embed <= 0:
            raise ValueError("hypernet_embed must be positive")
        self.n_agents = int(n_agents)
        self.n_actions = int(n_actions)
        self.state_dim = int(state_dim)
        self.mixing_embed_dim = int(mixing_embed_dim)
        self.weighted_head = bool(weighted_head)
        self.is_minus_one = bool(is_minus_one)

        self.hyper_w_final = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, self.n_agents),
        )
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, self.n_agents),
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
        is_v: bool = False,
    ) -> torch.Tensor:
        if agent_qs.ndim != 2:
            raise ValueError("agent_qs must have shape (batch, n_agents)")
        if states.ndim != 2:
            raise ValueError("states must have shape (batch, state_dim)")
        batch_size, n_agents = agent_qs.shape
        if n_agents != self.n_agents:
            raise ValueError("agent_qs second dimension must equal n_agents")
        if states.shape[0] != batch_size or states.shape[1] != self.state_dim:
            raise ValueError("states must have shape (batch, state_dim)")

        if not is_v and (actions_onehot is None or max_q_i is None):
            raise ValueError("actions_onehot and max_q_i are required when is_v is False")

        w_final = torch.abs(self.hyper_w_final(states)) + 1e-10
        v = self.V(states)
        if self.weighted_head:
            agent_qs = w_final * agent_qs + v
            if not is_v:
                max_q_i = w_final * max_q_i + v

        if is_v:
            return self._calc_v(agent_qs)

        return self._calc_adv(agent_qs, states, actions_onehot, max_q_i)
