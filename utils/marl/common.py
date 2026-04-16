"""
Common utilities for MARL algorithms.

This module contains shared functions used across IQL, VDN, and QMIX implementations.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv

from utils.marl.networks import (
    MLPAgent,
    DuelingMLPAgent,
    DRU,
    NDQRNNAgent,
    NDQCommEncoder,
    append_agent_id,
    route_messages,
    dial_rnn_forward_batched,
    ndq_rnn_forward_batched,
)
from utils.marl.obs_normalization import RunningObsNormalizer
from utils.marl.vector_env import stack_vector_obs


# Large finite negative value used for action masking.  ``exp(-1e9)`` underflows
# to exactly 0 in float32, so softmax probabilities on masked entries are 0 and
# argmax never picks them -- but unlike ``-inf``, ``0 * -1e9 = 0`` (not NaN), so
# expressions like ``(action_onehot * log_probs).sum(-1)`` stay well-defined.
CURRICULUM_MASK_NEG = -1e9


def compute_broadcast_curriculum_mask(
    global_step: int,
    total_timesteps: int,
    n_actions: int,
    phase1_ratio: float,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Return a logit mask for broadcast curriculum, or None when unneeded.

    During Phase 1 (``global_step < phase1_ratio * total_timesteps``), actions
    0 and 1 (the non-broadcast actions) are masked to ``CURRICULUM_MASK_NEG``
    so agents are forced to broadcast.  During Phase 2 (remainder) no masking
    is applied and the function returns ``None``.

    The caller adds the returned tensor to raw logits before creating a
    ``Categorical`` or computing ``log_softmax``.
    """
    if n_actions != 4:
        return None
    if global_step >= phase1_ratio * total_timesteps:
        return None
    mask = torch.zeros(n_actions, device=device)
    mask[0] = CURRICULUM_MASK_NEG
    mask[1] = CURRICULUM_MASK_NEG
    return mask


def curriculum_n_valid_actions(
    global_step: int,
    total_timesteps: int,
    n_actions: int,
    phase1_ratio: float,
) -> int:
    """Return the number of valid (unmasked) actions for the current phase."""
    if n_actions != 4:
        return n_actions
    if global_step < phase1_ratio * total_timesteps:
        return 2
    return n_actions


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
    action_mask: Optional[torch.Tensor] = None,
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
        action_mask: Optional logit-space mask (shape ``(n_actions,)``).
            ``-inf`` entries suppress the corresponding actions.

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

    if action_mask is not None:
        q = q + action_mask

    greedy = q.argmax(dim=-1).cpu().numpy().astype(np.int64)
    actions = greedy.copy()
    explore_mask = rng.random((n_envs, n_agents)) < float(epsilon)
    if np.any(explore_mask):
        if action_mask is not None:
            # Valid actions are those whose mask entry is 0; masked entries
            # hold a large negative value (see CURRICULUM_MASK_NEG).
            valid = (action_mask >= 0.0).cpu().numpy()
            valid_indices = np.where(valid)[0]
            random_actions = rng.choice(
                valid_indices, size=int(explore_mask.sum()),
            ).astype(np.int64)
        else:
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
    action_selector: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    action_mask: Optional[torch.Tensor] = None,
) -> Tuple[float, float, List[float]]:
    """Run vectorized evaluation on an explicit seed list (paired-seed friendly).

    Each seed corresponds to one complete episode reward in the returned list.
    When ``action_selector`` is provided, it is used instead of ``agent`` for
    learned-policy action selection.
    """
    if n_eval_envs <= 0:
        raise ValueError("n_eval_envs must be positive")
    if not episode_seeds:
        raise ValueError("episode_seeds must not be empty")
    if fixed_action is not None and heuristic_policy_name is not None:
        raise ValueError("fixed_action and heuristic_policy_name are mutually exclusive")

    dummy_rng = np.random.default_rng(0)
    all_episode_rewards: List[float] = []

    if heuristic_policy_name is not None:
        from tools.heuristic_policies import get_heuristic_policy

    if fixed_action is not None:
        precomputed_fixed_actions = np.full((n_eval_envs, n_agents), int(fixed_action), dtype=np.int64)
    else:
        precomputed_fixed_actions = None
        
    heuristic_actions_buffer = np.zeros((n_eval_envs, n_agents), dtype=np.int64) if heuristic_policy_name is not None else None

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
                actions = precomputed_fixed_actions
            elif heuristic_policies is not None:
                actions = heuristic_actions_buffer
                actions.fill(0)
                for env_idx in range(active_count):
                    if done[env_idx]:
                        continue
                    for agent_idx in range(n_agents):
                        action, _ = heuristic_policies[env_idx][agent_idx].predict(
                            obs[env_idx, agent_idx], deterministic=heuristic_deterministic
                        )
                        actions[env_idx, agent_idx] = int(action)
            elif action_selector is not None:
                actions = np.asarray(action_selector(obs), dtype=np.int64)
                if actions.shape != (n_eval_envs, n_agents):
                    raise ValueError(
                        "action_selector must return an array with shape "
                        f"({n_eval_envs}, {n_agents})"
                    )
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
                    action_mask=action_mask,
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


# ---------------------------------------------------------------------------
# Recurrent DIAL collection utilities
# ---------------------------------------------------------------------------

class DialRNNStepResult(NamedTuple):
    obs: np.ndarray
    epsilon: float
    actions: np.ndarray
    next_obs_raw: np.ndarray
    rewards_arr: np.ndarray
    terminated: np.ndarray
    done_reset: np.ndarray
    infos: Dict[str, Any]
    msg_logits: np.ndarray


class NDQRNNStepResult(NamedTuple):
    obs: np.ndarray
    epsilon: float
    actions: np.ndarray
    next_obs_raw: np.ndarray
    rewards_arr: np.ndarray
    terminated: np.ndarray
    done_reset: np.ndarray
    infos: Dict[str, Any]


@torch.no_grad()
def select_actions_dial_rnn_batched(
    agent: "DialRNNAgent",
    obs: np.ndarray,
    recv_msg: np.ndarray,
    hidden: torch.Tensor,
    prev_actions: np.ndarray,
    n_envs: int,
    n_agents: int,
    n_actions: int,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Select actions for recurrent DIAL agents.

    Returns (actions, msg_logits_np, new_hidden).
    """
    obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
    recv_msg_t = torch.as_tensor(recv_msg, device=device, dtype=torch.float32)
    prev_act_t = torch.as_tensor(prev_actions, device=device, dtype=torch.long)
    q_values, msg_logits, new_hidden = dial_rnn_forward_batched(
        agent,
        obs_t,
        prev_act_t,
        recv_msg_t,
        hidden,
    )

    greedy = q_values.argmax(dim=-1).cpu().numpy().astype(np.int64)
    actions = greedy.copy()
    explore_mask = rng.random((n_envs, n_agents)) < float(epsilon)
    if np.any(explore_mask):
        actions[explore_mask] = rng.integers(
            0, n_actions, size=int(explore_mask.sum()), endpoint=False, dtype=np.int64,
        )

    return actions, msg_logits.cpu().numpy(), new_hidden


def dial_rnn_collect_transition(
    *,
    env: Any,
    agent: "DialRNNAgent",
    obs_raw: np.ndarray,
    obs_normalizer: Optional[RunningObsNormalizer],
    global_step: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_steps: int,
    n_envs: int,
    n_agents: int,
    n_actions: int,
    rng: np.random.Generator,
    device: torch.device,
    prev_msg_logits: np.ndarray,
    prev_actions: np.ndarray,
    hidden_states: torch.Tensor,
    dru: DRU,
    collector: Any,
    comm_dim: int,
    episode_start_mask: np.ndarray,
) -> DialRNNStepResult:
    """Collect a single vectorized recurrent DIAL transition."""
    if obs_normalizer is not None:
        obs = obs_normalizer.normalize(obs_raw, update=True)
    else:
        obs = obs_raw

    eps = epsilon_by_step(global_step, epsilon_start, epsilon_end, epsilon_decay_steps)

    # DRU on previous message logits → recv messages (no graph needed)
    # Sample noise explicitly so we can store and replay it during learning.
    with torch.no_grad():
        msg_logits_t = torch.as_tensor(prev_msg_logits, device=device, dtype=torch.float32)
        dru_noise = torch.randn_like(msg_logits_t) * dru.sigma
        msg_post_dru = dru(msg_logits_t, train_mode=True, noise=dru_noise)
        recv_msg = route_messages(msg_post_dru, n_agents).cpu().numpy()
        dru_noise_np = dru_noise.cpu().numpy()
    # At episode start no messages have been sent yet — use clean zeros
    # instead of DRU(0) = sigmoid(noise) ≈ 0.5 (reference: arena.py init)
    if np.any(episode_start_mask):
        recv_msg[episode_start_mask] = 0.0

    actions, new_msg_logits, new_hidden = select_actions_dial_rnn_batched(
        agent=agent, obs=obs, recv_msg=recv_msg,
        hidden=hidden_states, prev_actions=prev_actions,
        n_envs=n_envs, n_agents=n_agents, n_actions=n_actions,
        epsilon=eps, rng=rng, device=device,
    )

    action_dict = {f"agent_{i}": actions[:, i] for i in range(n_agents)}
    next_obs_dict, rewards_arr, terminated, truncated, infos = env.step(action_dict)
    next_obs_raw = stack_vector_obs(next_obs_dict, n_agents)

    terminated_any = np.asarray(terminated, dtype=np.bool_)
    truncated_any = np.asarray(truncated, dtype=np.bool_)
    done_reset = np.logical_or(terminated_any, truncated_any)

    next_obs_for_buffer, _ = patch_autoreset_final_obs(
        next_obs_raw, infos, done_reset, n_agents,
    )

    rewards_np = np.asarray(rewards_arr, dtype=np.float32)
    team_rewards = rewards_np.sum(axis=1, keepdims=True)
    rewards_team = np.repeat(team_rewards, n_agents, axis=1).astype(np.float32)

    for e in range(n_envs):
        transition = {
            "obs": obs_raw[e],
            "actions": actions[e],
            "rewards": rewards_team[e],
            "next_obs": next_obs_for_buffer[e],
            "done": float(terminated_any[e]),
            "reset": bool(done_reset[e]),
            "dru_noise": dru_noise_np[e],
        }
        collector.add(e, transition)

    # Update state
    prev_msg_logits[:] = new_msg_logits
    prev_actions[:] = actions
    hidden_states.copy_(new_hidden)
    episode_start_mask[:] = False

    # Reset state for envs that ended
    start_token = n_actions
    for e in range(n_envs):
        if done_reset[e]:
            prev_msg_logits[e] = 0.0
            prev_actions[e] = start_token
            hidden_states[:, e * n_agents : (e + 1) * n_agents, :] = 0.0
            episode_start_mask[e] = True

    return DialRNNStepResult(
        obs=obs,
        epsilon=eps,
        actions=actions,
        next_obs_raw=next_obs_raw,
        rewards_arr=rewards_np,
        terminated=terminated_any,
        done_reset=done_reset,
        infos=infos,
        msg_logits=new_msg_logits,
    )


def ndq_rnn_collect_transition(
    *,
    env: Any,
    agent: NDQRNNAgent,
    comm_encoder: NDQCommEncoder,
    obs_raw: np.ndarray,
    obs_normalizer: Optional[RunningObsNormalizer],
    global_step: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_steps: int,
    n_envs: int,
    n_agents: int,
    n_actions: int,
    comm_embed_dim: int,
    rng: np.random.Generator,
    device: torch.device,
    prev_actions: np.ndarray,
    hidden_states: torch.Tensor,
    collector: Any,
) -> NDQRNNStepResult:
    """Collect a single vectorized recurrent NDQ transition."""
    if obs_normalizer is not None:
        obs = obs_normalizer.normalize(obs_raw, update=True)
    else:
        obs = obs_raw

    eps = epsilon_by_step(global_step, epsilon_start, epsilon_end, epsilon_decay_steps)

    with torch.no_grad():
        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
        prev_act_t = torch.as_tensor(prev_actions, device=device, dtype=torch.long)
        q_values, _, _, _, new_hidden = ndq_rnn_forward_batched(
            agent,
            comm_encoder,
            obs_t,
            prev_act_t,
            hidden_states,
            n_actions=n_actions,
            comm_embed_dim=comm_embed_dim,
        )

    greedy = q_values.argmax(dim=-1).cpu().numpy().astype(np.int64)
    actions = greedy.copy()
    explore_mask = rng.random((n_envs, n_agents)) < float(eps)
    if np.any(explore_mask):
        actions[explore_mask] = rng.integers(
            0, n_actions, size=int(explore_mask.sum()), endpoint=False, dtype=np.int64,
        )

    action_dict = {f"agent_{i}": actions[:, i] for i in range(n_agents)}
    next_obs_dict, rewards_arr, terminated, truncated, infos = env.step(action_dict)
    next_obs_raw = stack_vector_obs(next_obs_dict, n_agents)

    terminated_any = np.asarray(terminated, dtype=np.bool_)
    truncated_any = np.asarray(truncated, dtype=np.bool_)
    done_reset = np.logical_or(terminated_any, truncated_any)

    next_obs_for_buffer, _ = patch_autoreset_final_obs(
        next_obs_raw, infos, done_reset, n_agents,
    )

    rewards_np = np.asarray(rewards_arr, dtype=np.float32)
    team_rewards = rewards_np.sum(axis=1, keepdims=True)
    rewards_team = np.repeat(team_rewards, n_agents, axis=1).astype(np.float32)

    for env_idx in range(n_envs):
        collector.add(env_idx, {
            "obs": obs_raw[env_idx],
            "actions": actions[env_idx],
            "rewards": rewards_team[env_idx],
            "next_obs": next_obs_for_buffer[env_idx],
            "done": float(terminated_any[env_idx]),
            "reset": bool(done_reset[env_idx]),
        })

    prev_actions[:] = actions
    hidden_states.copy_(new_hidden)

    start_token = n_actions
    for env_idx in range(n_envs):
        if done_reset[env_idx]:
            prev_actions[env_idx] = start_token
            hidden_states[:, env_idx * n_agents : (env_idx + 1) * n_agents, :] = 0.0

    return NDQRNNStepResult(
        obs=obs,
        epsilon=eps,
        actions=actions,
        next_obs_raw=next_obs_raw,
        rewards_arr=rewards_np,
        terminated=terminated_any,
        done_reset=done_reset,
        infos=infos,
    )
