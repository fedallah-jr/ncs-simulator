from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from utils.marl.common import select_actions, stack_obs
from utils.marl.networks import (
    MLPAgent,
    DuelingMLPAgent,
    DialRNNAgent,
    DRU,
    NDQRNNAgent,
    NDQCommEncoder,
    route_messages,
    dial_rnn_forward_batched,
    ndq_rnn_forward_batched,
)
from utils.marl.obs_normalization import RunningObsNormalizer


@dataclass(frozen=True)
class MARLTorchCheckpointMetadata:
    algorithm: str
    n_agents: int
    obs_dim: int
    n_actions: int
    use_agent_id: bool
    parameter_sharing: bool
    agent_hidden_dims: Tuple[int, ...]
    agent_activation: str
    feature_norm: bool = False
    layer_norm: bool = False
    dueling: bool = False
    stream_hidden_dim: Optional[int] = None
    obs_normalizer: Optional[RunningObsNormalizer] = None


MARLAgent = Union[MLPAgent, DuelingMLPAgent, Sequence[Union[MLPAgent, DuelingMLPAgent]]]


def _load_checkpoint_dict(model_path: Path) -> Dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    ckpt = torch.load(str(model_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("MARL torch checkpoint must be a dict")
    return ckpt


def load_marl_torch_agents_from_checkpoint(model_path: Path) -> Tuple[MARLAgent, MARLTorchCheckpointMetadata]:
    ckpt = _load_checkpoint_dict(model_path)

    algorithm = str(ckpt.get("algorithm", "marl_torch"))
    n_agents = int(ckpt.get("n_agents", 1))
    obs_dim = int(ckpt.get("obs_dim", 0))
    n_actions = int(ckpt.get("n_actions", 0))
    if n_agents <= 0:
        raise ValueError("Checkpoint must include positive 'n_agents'")
    if obs_dim <= 0 or n_actions <= 0:
        raise ValueError("Checkpoint must include positive 'obs_dim' and 'n_actions'")

    use_agent_id = bool(ckpt.get("use_agent_id", n_agents > 1))
    input_dim = obs_dim + (n_agents if use_agent_id else 0)
    hidden_dims = tuple(int(x) for x in ckpt.get("agent_hidden_dims", [128, 128]))
    activation = str(ckpt.get("agent_activation", "relu"))
    feature_norm = bool(ckpt.get("feature_norm", False))
    layer_norm = bool(ckpt.get("layer_norm", False))
    dueling = bool(ckpt.get("dueling", False))
    stream_hidden_dim = ckpt.get("stream_hidden_dim", 64) if dueling else None
    obs_normalizer: Optional[RunningObsNormalizer] = None
    obs_norm_state = ckpt.get("obs_normalization", None)
    if isinstance(obs_norm_state, dict) and bool(obs_norm_state.get("enabled", False)):
        obs_normalizer = RunningObsNormalizer.from_state_dict(obs_norm_state)
        if obs_normalizer.obs_dim != obs_dim:
            raise ValueError(
                f"Checkpoint obs_normalization dim={obs_normalizer.obs_dim} does not match obs_dim={obs_dim}"
            )

    # Select network class based on dueling flag
    AgentClass = DuelingMLPAgent if dueling else MLPAgent
    agent_kwargs: Dict[str, Any] = {
        "input_dim": input_dim,
        "n_actions": n_actions,
        "hidden_dims": hidden_dims,
        "activation": activation,
        "feature_norm": feature_norm,
        "layer_norm": layer_norm,
    }
    if dueling:
        agent_kwargs["stream_hidden_dim"] = stream_hidden_dim

    state_dicts = ckpt.get("agent_state_dicts", None)
    state_dict = ckpt.get("agent_state_dict", None)

    if state_dicts is not None:
        if not isinstance(state_dicts, (list, tuple)):
            raise ValueError("Checkpoint 'agent_state_dicts' must be a list/tuple")
        if len(state_dicts) != n_agents:
            raise ValueError("Checkpoint 'agent_state_dicts' length must equal n_agents")
        agents: List[Union[MLPAgent, DuelingMLPAgent]] = []
        for idx in range(n_agents):
            agent = AgentClass(**agent_kwargs)
            agent.load_state_dict(state_dicts[idx])
            agent.eval()
            agents.append(agent)
        parameter_sharing = False
        agent_or_agents: MARLAgent = agents
    else:
        if state_dict is None:
            raise ValueError("Checkpoint must include 'agent_state_dict' (shared) or 'agent_state_dicts' (independent)")
        agent = AgentClass(**agent_kwargs)
        agent.load_state_dict(state_dict)
        agent.eval()
        parameter_sharing = True
        agent_or_agents = agent

    metadata = MARLTorchCheckpointMetadata(
        algorithm=algorithm,
        n_agents=n_agents,
        obs_dim=obs_dim,
        n_actions=n_actions,
        use_agent_id=use_agent_id,
        parameter_sharing=parameter_sharing,
        agent_hidden_dims=hidden_dims,
        agent_activation=activation,
        feature_norm=feature_norm,
        layer_norm=layer_norm,
        dueling=dueling,
        stream_hidden_dim=stream_hidden_dim,
        obs_normalizer=obs_normalizer,
    )
    return agent_or_agents, metadata


class MARLTorchMultiAgentPolicy:
    def __init__(
        self,
        agent: MARLAgent,
        metadata: MARLTorchCheckpointMetadata,
        *,
        device: Optional[torch.device] = None,
    ) -> None:
        self.agent = agent
        self.metadata = metadata
        self.device = device or torch.device("cpu")
        self._rng = np.random.default_rng(0)

        if isinstance(self.agent, (MLPAgent, DuelingMLPAgent)):
            self.agent.to(self.device)
        else:
            for net in self.agent:
                net.to(self.device)

    def reset(self) -> None:
        return

    def act(self, obs_dict: Mapping[str, Any]) -> Dict[str, int]:
        obs = stack_obs(dict(obs_dict), self.metadata.n_agents)
        if self.metadata.obs_normalizer is not None:
            obs = self.metadata.obs_normalizer.normalize(obs, update=False)
        actions = select_actions(
            agent=self.agent,
            obs=obs,
            n_agents=self.metadata.n_agents,
            n_actions=self.metadata.n_actions,
            epsilon=0.0,
            rng=self._rng,
            device=self.device,
            use_agent_id=self.metadata.use_agent_id,
        )
        return {f"agent_{i}": int(actions[i]) for i in range(self.metadata.n_agents)}


def load_dial_rnn_agent_from_checkpoint(
    model_path: Path,
) -> Tuple[DialRNNAgent, MARLTorchCheckpointMetadata, int, float]:
    """Load a recurrent DIAL agent from checkpoint.

    Returns (agent, metadata, comm_dim, dru_sigma).
    """
    ckpt = _load_checkpoint_dict(model_path)

    n_agents = int(ckpt.get("n_agents", 0))
    obs_dim = int(ckpt.get("obs_dim", 0))
    n_actions = int(ckpt.get("n_actions", 0))
    comm_dim = int(ckpt.get("comm_dim", 0))
    dru_sigma = float(ckpt.get("dru_sigma", 2.0))
    rnn_hidden_dim = int(ckpt.get("rnn_hidden_dim", 0))
    rnn_layers = int(ckpt.get("rnn_layers", 0))

    if (
        n_agents <= 0 or obs_dim <= 0 or n_actions <= 0
        or comm_dim <= 0 or rnn_hidden_dim <= 0 or rnn_layers <= 0
    ):
        raise ValueError(
            "DIAL checkpoint must include positive 'n_agents', 'obs_dim', "
            "'n_actions', 'comm_dim', 'rnn_hidden_dim', and 'rnn_layers'; "
            f"got n_agents={n_agents}, obs_dim={obs_dim}, n_actions={n_actions}, "
            f"comm_dim={comm_dim}, rnn_hidden_dim={rnn_hidden_dim}, rnn_layers={rnn_layers}."
        )

    obs_normalizer: Optional[RunningObsNormalizer] = None
    obs_norm_state = ckpt.get("obs_normalization", None)
    if isinstance(obs_norm_state, dict) and bool(obs_norm_state.get("enabled", False)):
        obs_normalizer = RunningObsNormalizer.from_state_dict(obs_norm_state)

    agent = DialRNNAgent(
        obs_dim=obs_dim,
        n_agents=n_agents,
        n_actions=n_actions,
        comm_dim=comm_dim,
        rnn_hidden_dim=rnn_hidden_dim,
        rnn_layers=rnn_layers,
    )
    agent.load_state_dict(ckpt["agent_state_dict"])

    metadata = MARLTorchCheckpointMetadata(
        algorithm=str(ckpt.get("algorithm", "marl_dial")),
        n_agents=n_agents,
        obs_dim=obs_dim,
        n_actions=n_actions,
        use_agent_id=True,
        parameter_sharing=True,
        agent_hidden_dims=(),
        agent_activation="relu",
        obs_normalizer=obs_normalizer,
    )
    return agent, metadata, comm_dim, dru_sigma


class MARLDialRNNTorchPolicy:
    """Stateful recurrent DIAL policy for inference."""

    def __init__(
        self,
        agent: DialRNNAgent,
        metadata: MARLTorchCheckpointMetadata,
        dru: DRU,
        comm_dim: int,
        *,
        device: Optional[torch.device] = None,
    ) -> None:
        self.agent = agent
        self.metadata = metadata
        self.dru = dru
        self.comm_dim = comm_dim
        self.device = device or torch.device("cpu")
        self.agent.to(self.device)

        n_agents = metadata.n_agents
        self.hidden = agent.init_hidden(n_agents).to(self.device)
        self.prev_action = torch.full(
            (n_agents,), agent.n_actions, dtype=torch.long, device=self.device,
        )
        self.prev_msg_logits = torch.zeros(
            n_agents, comm_dim, device=self.device,
        )
        # Reference initializes recv_msg to literal zeros at episode start.
        # DRU(0, train_mode=False) returns sigmoid(-20) ≈ 2e-9, so without
        # this flag the first step would feed near-zero (but not exactly zero)
        # messages into the network.
        self._episode_start = True

    def reset(self) -> None:
        self.hidden.zero_()
        self.prev_action.fill_(self.agent.n_actions)
        self.prev_msg_logits.zero_()
        self._episode_start = True

    @torch.no_grad()
    def act(self, obs_dict: Mapping[str, Any]) -> Dict[str, int]:
        n_agents = self.metadata.n_agents
        obs = stack_obs(dict(obs_dict), n_agents)
        if self.metadata.obs_normalizer is not None:
            obs = self.metadata.obs_normalizer.normalize(obs, update=False)

        obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        msg_post_dru = self.dru(self.prev_msg_logits.unsqueeze(0), train_mode=False)
        recv_msg = route_messages(msg_post_dru, n_agents).squeeze(0)
        if self._episode_start:
            recv_msg = torch.zeros_like(recv_msg)
            self._episode_start = False
        q_values, msg_logits, new_hidden = dial_rnn_forward_batched(
            self.agent,
            obs_t.unsqueeze(0),
            self.prev_action.unsqueeze(0),
            recv_msg.unsqueeze(0),
            self.hidden,
        )
        actions = q_values.squeeze(0).argmax(dim=-1).cpu().numpy().astype(np.int64)
        self.hidden = new_hidden
        self.prev_action = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        self.prev_msg_logits = msg_logits.squeeze(0)
        return {f"agent_{i}": int(actions[i]) for i in range(n_agents)}


def load_ndq_agent_from_checkpoint(
    model_path: Path,
) -> Tuple[NDQRNNAgent, NDQCommEncoder, MARLTorchCheckpointMetadata, int]:
    """Load a recurrent NDQ agent and comm encoder from checkpoint."""
    ckpt = _load_checkpoint_dict(model_path)

    n_agents = int(ckpt.get("n_agents", 1))
    obs_dim = int(ckpt.get("obs_dim", 0))
    n_actions = int(ckpt.get("n_actions", 0))
    comm_embed_dim = int(ckpt.get("comm_embed_dim", 1))
    rnn_hidden_dim = int(ckpt.get("rnn_hidden_dim", 64))
    rnn_layers = int(ckpt.get("rnn_layers", 1))

    obs_normalizer: Optional[RunningObsNormalizer] = None
    obs_norm_state = ckpt.get("obs_normalization", None)
    if isinstance(obs_norm_state, dict) and bool(obs_norm_state.get("enabled", False)):
        obs_normalizer = RunningObsNormalizer.from_state_dict(obs_norm_state)

    agent = NDQRNNAgent(
        obs_dim=obs_dim,
        n_agents=n_agents,
        n_actions=n_actions,
        comm_embed_dim=comm_embed_dim,
        rnn_hidden_dim=rnn_hidden_dim,
        rnn_layers=rnn_layers,
    )
    agent.load_state_dict(ckpt["agent_state_dict"])

    comm_encoder = NDQCommEncoder(
        input_dim=obs_dim + n_actions + n_agents,
        n_agents=n_agents,
        comm_embed_dim=comm_embed_dim,
        n_actions=n_actions,
        hidden_dim=rnn_hidden_dim,
    )
    comm_encoder.load_state_dict(ckpt["comm_encoder_state_dict"])

    metadata = MARLTorchCheckpointMetadata(
        algorithm=str(ckpt.get("algorithm", "marl_ndq")),
        n_agents=n_agents,
        obs_dim=obs_dim,
        n_actions=n_actions,
        use_agent_id=True,
        parameter_sharing=True,
        agent_hidden_dims=(),
        agent_activation="relu",
        obs_normalizer=obs_normalizer,
    )
    return agent, comm_encoder, metadata, comm_embed_dim


class MARLNDQTorchPolicy:
    """Stateful recurrent NDQ policy for inference."""

    def __init__(
        self,
        agent: NDQRNNAgent,
        comm_encoder: NDQCommEncoder,
        metadata: MARLTorchCheckpointMetadata,
        comm_embed_dim: int,
        *,
        device: Optional[torch.device] = None,
        cut_mu_thres: float = 0.0,
    ) -> None:
        self.agent = agent
        self.comm_encoder = comm_encoder
        self.metadata = metadata
        self.comm_embed_dim = int(comm_embed_dim)
        self.device = device or torch.device("cpu")
        self.agent.to(self.device)
        self.comm_encoder.to(self.device)
        self.cut_mu_thres = float(cut_mu_thres)
        self.comm_stats: Dict[str, int] = {
            "total_dims": 0,
            "dropped_dims": 0,
            "total_messages": 0,
            "silent_messages": 0,
        }

        n_agents = metadata.n_agents
        self.hidden = agent.init_hidden(n_agents).to(self.device)
        self.prev_action = torch.full(
            (1, n_agents), agent.n_actions, dtype=torch.long, device=self.device,
        )

    def reset(self) -> None:
        self.hidden.zero_()
        self.prev_action.fill_(self.agent.n_actions)
        self.comm_stats = {
            "total_dims": 0,
            "dropped_dims": 0,
            "total_messages": 0,
            "silent_messages": 0,
        }

    @torch.no_grad()
    def act(self, obs_dict: Mapping[str, Any]) -> Dict[str, int]:
        n_agents = self.metadata.n_agents
        obs = stack_obs(dict(obs_dict), n_agents)
        if self.metadata.obs_normalizer is not None:
            obs = self.metadata.obs_normalizer.normalize(obs, update=False)

        q_values, _, _, _, new_hidden = ndq_rnn_forward_batched(
            self.agent,
            self.comm_encoder,
            torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0),
            self.prev_action,
            self.hidden,
            n_actions=self.metadata.n_actions,
            comm_embed_dim=self.comm_embed_dim,
            cut_mu_thres=self.cut_mu_thres,
            comm_stats=self.comm_stats,
        )
        actions = q_values.squeeze(0).argmax(dim=-1).cpu().numpy().astype(np.int64)
        self.hidden = new_hidden
        self.prev_action = torch.as_tensor(actions, device=self.device, dtype=torch.long).unsqueeze(0)
        return {f"agent_{i}": int(actions[i]) for i in range(n_agents)}
