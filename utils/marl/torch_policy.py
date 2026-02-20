from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from utils.marl.common import select_actions, stack_obs
from utils.marl.networks import MLPAgent, DuelingMLPAgent
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
