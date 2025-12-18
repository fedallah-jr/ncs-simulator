"""
Common utilities for MARL algorithms.

This module contains shared functions used across IQL, VDN, and QMIX implementations.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Union

import numpy as np
import torch

from utils.marl.networks import MLPAgent, append_agent_id


def select_device(device_str: str) -> torch.device:
    """Select torch device based on string specification."""
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str == "cuda":
        return torch.device("cuda")
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError("device must be one of: auto, cpu, cuda")


def epsilon_by_step(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    """Compute epsilon for epsilon-greedy exploration using linear decay."""
    if decay_steps <= 0:
        return float(eps_end)
    frac = min(1.0, max(0.0, step / float(decay_steps)))
    return float(eps_start + frac * (eps_end - eps_start))


def stack_obs(obs_dict: Dict[str, Any], n_agents: int) -> np.ndarray:
    """Stack agent observations from dict into array of shape (n_agents, obs_dim)."""
    return np.stack([np.asarray(obs_dict[f"agent_{i}"], dtype=np.float32) for i in range(n_agents)], axis=0)


@torch.no_grad()
def select_actions(
    agent: Union[MLPAgent, Sequence[MLPAgent]],
    obs: np.ndarray,
    n_agents: int,
    n_actions: int,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
    use_agent_id: bool,
) -> np.ndarray:
    """
    Select actions using epsilon-greedy policy.

    Args:
        agent: The Q-network agent
        obs: Observations of shape (n_agents, obs_dim)
        n_agents: Number of agents
        n_actions: Number of possible actions
        epsilon: Exploration probability
        rng: Random number generator
        device: Torch device
        use_agent_id: Whether to append agent ID to observations

    Returns:
        Actions array of shape (n_agents,)
    """
    obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
    if use_agent_id:
        obs_t = append_agent_id(obs_t, n_agents)

    if isinstance(agent, MLPAgent):
        q = agent(obs_t.view(n_agents, -1))
    else:
        if len(agent) != n_agents:
            raise ValueError("Independent agents sequence length must equal n_agents")
        q_per_agent: list[torch.Tensor] = []
        for agent_idx, agent_net in enumerate(agent):
            q_per_agent.append(agent_net(obs_t[:, agent_idx, :]))
        q = torch.cat(q_per_agent, dim=0)
    greedy = q.argmax(dim=-1).cpu().numpy().astype(np.int64)
    actions = greedy.copy()
    explore_mask = rng.random(n_agents) < float(epsilon)
    if np.any(explore_mask):
        actions[explore_mask] = rng.integers(0, n_actions, size=int(explore_mask.sum()), endpoint=False, dtype=np.int64)
    return actions
