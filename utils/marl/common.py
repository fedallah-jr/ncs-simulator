"""
Common utilities for MARL algorithms.

This module contains shared functions used across IQL, VDN, and QMIX implementations.
"""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

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
    episode_seeds: List[Optional[int]]
    if seed is None:
        episode_seeds = [None for _ in range(int(n_episodes))]
    else:
        episode_seeds = [int(seed) + ep for ep in range(int(n_episodes))]

    return run_evaluation_vectorized_seeded(
        eval_env=eval_env,
        agent=agent,
        n_eval_envs=n_eval_envs,
        n_agents=n_agents,
        n_actions=n_actions,
        use_agent_id=use_agent_id,
        device=device,
        episode_seeds=episode_seeds,
        obs_normalizer=obs_normalizer,
    )


@torch.no_grad()
def run_evaluation_vectorized_seeded(
    eval_env: AsyncVectorEnv,
    agent: Union[MLPAgent, Sequence[MLPAgent]],
    n_eval_envs: int,
    n_agents: int,
    n_actions: int,
    use_agent_id: bool,
    device: torch.device,
    episode_seeds: Sequence[Optional[int]],
    obs_normalizer: Optional[RunningObsNormalizer] = None,
    heuristic_policy_name: Optional[str] = None,
    heuristic_deterministic: bool = True,
    fixed_action: Optional[int] = None,
) -> Tuple[float, float, List[float]]:
    """Run vectorized evaluation on an explicit seed list (paired-seed friendly).

    Each seed corresponds to one complete episode reward in the returned list.
    """
    if n_eval_envs <= 0:
        raise ValueError("n_eval_envs must be positive")
    if not episode_seeds:
        raise ValueError("episode_seeds must not be empty")
    if fixed_action is not None and heuristic_policy_name is not None:
        raise ValueError("fixed_action and heuristic_policy_name are mutually exclusive")

    dummy_rng = np.random.default_rng(0)
    all_episode_rewards: List[float] = []

    for start in range(0, len(episode_seeds), n_eval_envs):
        batch_seeds = list(episode_seeds[start:start + n_eval_envs])
        active_count = len(batch_seeds)
        if active_count < n_eval_envs:
            pad_seed = batch_seeds[-1]
            batch_seeds.extend([pad_seed] * (n_eval_envs - active_count))

        if all(s is None for s in batch_seeds):
            reset_seeds: Optional[List[int]] = None
        else:
            if any(s is None for s in batch_seeds):
                raise ValueError("episode_seeds must be all None or all integers")
            reset_seeds = [int(s) for s in batch_seeds]  # type: ignore[arg-type]

        obs_dict, _infos = eval_env.reset(seed=reset_seeds)
        obs = stack_vector_obs(obs_dict, n_agents)
        if obs_normalizer is not None:
            obs = obs_normalizer.normalize(obs, update=False)

        active_mask = np.zeros(n_eval_envs, dtype=np.bool_)
        active_mask[:active_count] = True
        done = np.zeros(n_eval_envs, dtype=np.bool_)
        env_reward_sums = np.zeros(n_eval_envs, dtype=np.float64)

        heuristic_policies: Optional[List[List[Any]]] = None
        if heuristic_policy_name is not None:
            from tools.heuristic_policies import get_heuristic_policy

            heuristic_policies = []
            for env_idx in range(n_eval_envs):
                env_seed = None if reset_seeds is None else int(reset_seeds[env_idx])
                env_policies: List[Any] = []
                for agent_idx in range(n_agents):
                    policy_seed = None if env_seed is None else int(env_seed) + agent_idx
                    policy = get_heuristic_policy(
                        heuristic_policy_name,
                        n_agents=n_agents,
                        seed=policy_seed,
                        agent_index=agent_idx,
                    )
                    if hasattr(policy, "reset"):
                        policy.reset()
                    env_policies.append(policy)
                heuristic_policies.append(env_policies)

        while not np.all(done[:active_count]):
            if fixed_action is not None:
                actions = np.full((n_eval_envs, n_agents), int(fixed_action), dtype=np.int64)
            elif heuristic_policies is not None:
                actions = np.zeros((n_eval_envs, n_agents), dtype=np.int64)
                for env_idx in range(active_count):
                    if done[env_idx]:
                        continue
                    for agent_idx in range(n_agents):
                        action, _ = heuristic_policies[env_idx][agent_idx].predict(
                            obs[env_idx, agent_idx], deterministic=heuristic_deterministic
                        )
                        actions[env_idx, agent_idx] = int(action)
            else:
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
            team_rewards = rewards_arr.sum(axis=1)
            running = np.logical_and(active_mask, np.logical_not(done))
            env_reward_sums[running] += team_rewards[running]

            done_now = np.logical_or(
                np.asarray(terminated, dtype=np.bool_),
                np.asarray(truncated, dtype=np.bool_),
            )
            done = np.logical_or(done, np.logical_and(done_now, active_mask))

            next_obs = stack_vector_obs(next_obs_dict, n_agents)
            if obs_normalizer is not None:
                next_obs = obs_normalizer.normalize(next_obs, update=False)
            obs = next_obs

        all_episode_rewards.extend(float(env_reward_sums[idx]) for idx in range(active_count))

    mean_reward = float(np.mean(all_episode_rewards))
    std_reward = float(np.std(all_episode_rewards))
    return mean_reward, std_reward, all_episode_rewards


def patch_autoreset_final_obs(
    next_obs_raw: np.ndarray,
    infos: Dict[str, Any],
    done_reset: np.ndarray,
    n_agents: int,
    next_global_state_raw: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Patch next_obs (and optionally global_state) with final_obs/final_info from auto-reset envs.

    Returns:
        Tuple of (next_obs_for_buffer, next_global_state_for_buffer).
        When *next_global_state_raw* is None the second element is None.
    """
    next_obs_for_buffer = next_obs_raw
    next_global_state_for_buffer = next_global_state_raw

    if infos.get("final_obs") is not None:
        final_obs = infos["final_obs"]
        done_indices = np.where(done_reset)[0]
        if len(done_indices) > 0:
            next_obs_for_buffer = next_obs_raw.copy()
            for env_idx in done_indices:
                final_env_obs = final_obs[env_idx]
                if final_env_obs is not None:
                    next_obs_for_buffer[env_idx] = stack_obs(final_env_obs, n_agents)

            if next_global_state_raw is not None:
                final_info = infos.get("final_info")
                if final_info is not None:
                    final_global = final_info.get("global_state")
                    if final_global is not None:
                        next_global_state_for_buffer = next_global_state_raw.copy()
                        for env_idx in done_indices:
                            final_env_state = final_global[env_idx]
                            if final_env_state is not None:
                                next_global_state_for_buffer[env_idx] = final_env_state

    return next_obs_for_buffer, next_global_state_for_buffer


class QLearnStepResult(NamedTuple):
    obs: np.ndarray            # normalized obs
    epsilon: float
    actions: np.ndarray        # (n_envs, n_agents)
    next_obs_raw: np.ndarray
    rewards_arr: np.ndarray    # raw rewards from env
    terminated: np.ndarray
    done_reset: np.ndarray
    infos: Dict[str, Any]


def qlearning_collect_transition(
    *,
    env: Any,
    agent: Any,
    obs_raw: np.ndarray,
    obs_normalizer: Optional[RunningObsNormalizer],
    global_step: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_steps: int,
    n_envs: int,
    n_agents: int,
    n_actions: int,
    use_agent_id: bool,
    rng: np.random.Generator,
    device: torch.device,
) -> QLearnStepResult:
    """Collect a single vectorized transition for Q-learning algorithms.

    Normalizes observations, computes epsilon, selects actions, and steps the env.
    """
    if obs_normalizer is not None:
        obs = obs_normalizer.normalize(obs_raw, update=True)
    else:
        obs = obs_raw

    eps = epsilon_by_step(global_step, epsilon_start, epsilon_end, epsilon_decay_steps)

    actions = select_actions_batched(
        agent=agent,
        obs=obs,
        n_envs=n_envs,
        n_agents=n_agents,
        n_actions=n_actions,
        epsilon=eps,
        rng=rng,
        device=device,
        use_agent_id=use_agent_id,
    )

    action_dict = {f"agent_{i}": actions[:, i] for i in range(n_agents)}
    next_obs_dict, rewards_arr, terminated, truncated, infos = env.step(action_dict)
    next_obs_raw = stack_vector_obs(next_obs_dict, n_agents)

    terminated_any = np.asarray(terminated, dtype=np.bool_)
    truncated_any = np.asarray(truncated, dtype=np.bool_)
    done_reset = np.logical_or(terminated_any, truncated_any)

    return QLearnStepResult(
        obs=obs,
        epsilon=eps,
        actions=actions,
        next_obs_raw=next_obs_raw,
        rewards_arr=np.asarray(rewards_arr, dtype=np.float32),
        terminated=terminated_any,
        done_reset=done_reset,
        infos=infos,
    )
