"""
Common utilities for MARL algorithms.

This module contains shared functions used across IQL, VDN, and QMIX implementations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv

from utils.marl.networks import MLPAgent, DuelingMLPAgent, append_agent_id
from utils.marl.obs_normalization import RunningObsNormalizer
from utils.marl.vector_env import stack_vector_obs


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
    agent: Union[MLPAgent, DuelingMLPAgent, Sequence[Union[MLPAgent, DuelingMLPAgent]]],
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

    if isinstance(agent, (MLPAgent, DuelingMLPAgent)):
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


@torch.no_grad()
def select_actions_batched(
    agent: Union[MLPAgent, DuelingMLPAgent, Sequence[Union[MLPAgent, DuelingMLPAgent]]],
    obs: np.ndarray,
    n_envs: int,
    n_agents: int,
    n_actions: int,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
    use_agent_id: bool,
) -> np.ndarray:
    """
    Select actions using epsilon-greedy policy for batched vector environments.

    Args:
        agent: The Q-network agent
        obs: Observations of shape (n_envs, n_agents, obs_dim)
        n_envs: Number of parallel environments
        n_agents: Number of agents
        n_actions: Number of possible actions
        epsilon: Exploration probability
        rng: Random number generator
        device: Torch device
        use_agent_id: Whether to append agent ID to observations

    Returns:
        Actions array of shape (n_envs, n_agents)
    """
    obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
    if obs_t.ndim != 3:
        raise ValueError("obs must have shape (n_envs, n_agents, obs_dim)")
    if obs_t.shape[0] != n_envs or obs_t.shape[1] != n_agents:
        raise ValueError("obs leading dimensions must match n_envs and n_agents")

    if use_agent_id:
        obs_t = append_agent_id(obs_t, n_agents)

    if isinstance(agent, (MLPAgent, DuelingMLPAgent)):
        q = agent(obs_t.reshape(n_envs * n_agents, -1))
        q = q.view(n_envs, n_agents, n_actions)
    else:
        if len(agent) != n_agents:
            raise ValueError("Independent agents sequence length must equal n_agents")
        q_per_agent: list[torch.Tensor] = []
        for agent_idx, agent_net in enumerate(agent):
            q_per_agent.append(agent_net(obs_t[:, agent_idx, :]))
        q = torch.stack(q_per_agent, dim=1)

    greedy = q.argmax(dim=-1).cpu().numpy().astype(np.int64)
    actions = greedy.copy()
    explore_mask = rng.random((n_envs, n_agents)) < float(epsilon)
    if np.any(explore_mask):
        random_actions = rng.integers(
            0, n_actions, size=int(explore_mask.sum()), endpoint=False, dtype=np.int64
        )
        actions[explore_mask] = random_actions
    return actions


def run_evaluation(
    env: Any,
    agent: Union[MLPAgent, Sequence[MLPAgent]],
    n_agents: int,
    n_actions: int,
    use_agent_id: bool,
    device: torch.device,
    n_episodes: int,
    seed: Optional[int] = None,
    obs_normalizer: Optional[RunningObsNormalizer] = None,
) -> Tuple[float, float, List[float]]:
    """
    Run deterministic evaluation episodes for MARL algorithms.

    Args:
        env: The evaluation environment (NCS_Env or similar multi-agent env)
        agent: The agent network(s) - either shared MLPAgent or list of independent agents
        n_agents: Number of agents
        n_actions: Number of actions per agent
        use_agent_id: Whether to append one-hot agent ID to observations
        device: Torch device
        n_episodes: Number of evaluation episodes to run
        seed: Optional seed for reproducibility
        obs_normalizer: Optional observation normalizer (stats frozen during eval)

    Returns:
        Tuple of (mean_reward, std_reward, episode_rewards)
    """
    episode_rewards: List[float] = []
    dummy_rng = np.random.default_rng(0)  # Not used with epsilon=0

    for ep in range(n_episodes):
        episode_seed = None if seed is None else seed + ep
        obs_dict, _info = env.reset(seed=episode_seed)
        obs = stack_obs(obs_dict, n_agents)
        if obs_normalizer is not None:
            obs = obs_normalizer.normalize(obs, update=False)

        episode_reward_sum = 0.0
        done = False

        while not done:
            # Deterministic action selection (epsilon=0)
            actions = select_actions(
                agent=agent,
                obs=obs,
                n_agents=n_agents,
                n_actions=n_actions,
                epsilon=0.0,  # Deterministic evaluation
                rng=dummy_rng,
                device=device,
                use_agent_id=use_agent_id,
            )
            action_dict = {f"agent_{i}": int(actions[i]) for i in range(n_agents)}
            next_obs_dict, rewards_dict, terminated, truncated, _infos = env.step(action_dict)
            next_obs = stack_obs(next_obs_dict, n_agents)
            if obs_normalizer is not None:
                next_obs = obs_normalizer.normalize(next_obs, update=False)
            rewards = np.asarray([rewards_dict[f"agent_{i}"] for i in range(n_agents)], dtype=np.float32)
            done = any(terminated[f"agent_{i}"] or truncated[f"agent_{i}"] for i in range(n_agents))

            episode_reward_sum += float(rewards.sum())
            obs = next_obs

        episode_rewards.append(episode_reward_sum)

    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    return mean_reward, std_reward, episode_rewards


def run_evaluation_vectorized(
    eval_env: AsyncVectorEnv,
    agent: Union[MLPAgent, Sequence[MLPAgent]],
    n_eval_envs: int,
    n_agents: int,
    n_actions: int,
    use_agent_id: bool,
    device: torch.device,
    n_episodes: int,
    seed: Optional[int] = None,
    obs_normalizer: Optional[RunningObsNormalizer] = None,
) -> Tuple[float, float, List[float]]:
    """
    Run deterministic evaluation episodes using parallel async vector environments.

    Args:
        eval_env: AsyncVectorEnv wrapping n_eval_envs evaluation environments.
        agent: The agent network(s).
        n_eval_envs: Number of parallel evaluation environments.
        n_agents: Number of agents per environment.
        n_actions: Number of actions per agent.
        use_agent_id: Whether to append one-hot agent ID to observations.
        device: Torch device.
        n_episodes: Total number of evaluation episodes to collect.
        seed: Optional base seed for reproducibility.
        obs_normalizer: Optional observation normalizer (stats frozen during eval).

    Returns:
        Tuple of (mean_reward, std_reward, episode_rewards).
    """
    episode_rewards: List[float] = []
    dummy_rng = np.random.default_rng(0)

    env_seeds = None if seed is None else [seed + i for i in range(n_eval_envs)]
    obs_dict, _infos = eval_env.reset(seed=env_seeds)
    obs = stack_vector_obs(obs_dict, n_agents)
    if obs_normalizer is not None:
        obs = obs_normalizer.normalize(obs, update=False)

    env_reward_sums = np.zeros(n_eval_envs, dtype=np.float64)

    while len(episode_rewards) < n_episodes:
        actions = select_actions_batched(
            agent=agent,
            obs=obs,
            n_envs=n_eval_envs,
            n_agents=n_agents,
            n_actions=n_actions,
            epsilon=0.0,
            rng=dummy_rng,
            device=device,
            use_agent_id=use_agent_id,
        )
        action_dict = {f"agent_{i}": actions[:, i] for i in range(n_agents)}
        next_obs_dict, rewards, terminated, truncated, _infos = eval_env.step(action_dict)

        rewards_arr = np.asarray(rewards, dtype=np.float64)
        env_reward_sums += rewards_arr.sum(axis=1)

        done = np.logical_or(terminated, truncated)
        if np.any(done):
            for env_idx in np.where(done)[0]:
                if len(episode_rewards) < n_episodes:
                    episode_rewards.append(float(env_reward_sums[env_idx]))
                env_reward_sums[env_idx] = 0.0

        next_obs = stack_vector_obs(next_obs_dict, n_agents)
        if obs_normalizer is not None:
            next_obs = obs_normalizer.normalize(next_obs, update=False)
        obs = next_obs

    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    return mean_reward, std_reward, episode_rewards
