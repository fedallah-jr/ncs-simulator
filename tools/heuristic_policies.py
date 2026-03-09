"""
Heuristic policies for networked control systems.

This module provides simple baseline policies that can be used for
comparison with learned policies. All policies follow the same interface
used by the visualization tools for easy integration.
"""

from __future__ import annotations

import re
from typing import Tuple, Optional, Any, List

import numpy as np


class BaseHeuristicPolicy:
    """Base class for heuristic policies."""

    def __init__(self, n_agents: int = 1) -> None:
        """
        Initialize heuristic policy.

        Args:
            n_agents: Number of agents in the system
        """
        self.n_agents = n_agents

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Predict action given observation.

        Args:
            observation: Current observation
            deterministic: Whether to use deterministic policy (ignored for heuristics)

        Returns:
            action: Action to take
            state: Internal state (None for stateless policies)
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset policy state. Override in subclasses if needed."""
        pass


class AlwaysSendPolicy(BaseHeuristicPolicy):
    """Policy that always attempts to send measurements."""

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, None]:
        """
        Always return action=1 (send).

        Args:
            observation: Current observation (unused)
            deterministic: Whether to use deterministic policy (unused)

        Returns:
            action: Always 1 (send)
            state: None
        """
        return 1, None


class NeverSendPolicy(BaseHeuristicPolicy):
    """Policy that never sends measurements."""

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, None]:
        """
        Always return action=0 (don't send).

        Args:
            observation: Current observation (unused)
            deterministic: Whether to use deterministic policy (unused)

        Returns:
            action: Always 0 (don't send)
            state: None
        """
        return 0, None


class SendEveryNPolicy(BaseHeuristicPolicy):
    """Policy that sends every N timesteps."""

    def __init__(self, n: int = 5, n_agents: int = 1) -> None:
        """
        Initialize send-every-N policy.

        Args:
            n: Send measurement every n timesteps
            n_agents: Number of agents in the system
        """
        super().__init__(n_agents)
        self.n = n
        self.timestep = 0

    def reset(self) -> None:
        """Reset timestep counter."""
        self.timestep = 0

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, None]:
        """
        Send every N timesteps.

        Args:
            observation: Current observation (unused)
            deterministic: Whether to use deterministic policy (unused)

        Returns:
            action: 1 if timestep % n == 0, else 0
            state: None
        """
        action = 1 if self.timestep % self.n == 0 else 0
        self.timestep += 1
        return action, None


class RandomSendPolicy(BaseHeuristicPolicy):
    """Policy that sends with a fixed probability."""

    def __init__(self, prob: float = 0.5, n_agents: int = 1, seed: Optional[int] = None) -> None:
        """
        Initialize random send policy.

        Args:
            prob: Probability of sending at each timestep
            n_agents: Number of agents in the system
            seed: Random seed for reproducibility
        """
        super().__init__(n_agents)
        self.prob = prob
        self.rng = np.random.RandomState(seed)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, None]:
        """
        Send with probability prob.

        Args:
            observation: Current observation (unused)
            deterministic: Whether to use deterministic policy (if True, uses prob as threshold)

        Returns:
            action: 1 with probability prob, else 0
            state: None
        """
        if deterministic:
            return int(self.prob >= 0.5), None
        else:
            return int(self.rng.rand() < self.prob), None



class ZeroWaitPolicy(BaseHeuristicPolicy):
    """Policy that waits for ACK before sending again (transport layer)."""

    def __init__(self, n_agents: int = 1, history_window: Optional[int] = None) -> None:
        super().__init__(n_agents)
        self.history_window = history_window
        self._cached_history_window: Optional[int] = history_window

    def reset(self) -> None:
        """Reset cached history metadata between episodes."""
        self._cached_history_window = self.history_window

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, None]:
        """
        Send only when no transmission is pending.

        Args:
            observation: Current observation vector
            deterministic: Whether to use deterministic policy (unused)

        Returns:
            action: 1 if no pending packet, else 0
            state: None
        """
        status_values = self._extract_status_history(observation)
        pending = any(value == 2 for value in status_values)
        return (0 if pending else 1), None

    def _extract_status_history(self, observation: np.ndarray) -> List[int]:
        obs = np.asarray(observation, dtype=float).ravel()
        window = self._resolve_history_window(obs)
        if window <= 0 or obs.size < window * 2:
            return []
        status_slice = obs[-2 * window : -window]
        return [int(round(value)) for value in status_slice]

    def _resolve_history_window(self, observation: np.ndarray) -> int:
        if self._cached_history_window is not None:
            return self._cached_history_window
        obs_len = int(observation.size)
        remainder = obs_len - 3
        if remainder > 0 and remainder % 4 == 0:
            self._cached_history_window = max(1, remainder // 4)
            return self._cached_history_window
        self._cached_history_window = 10
        return self._cached_history_window


class PerfectSyncPolicy(BaseHeuristicPolicy):
    """
    Deterministic time-slot policy with one sender at a time.

    With slot multiplier ``n=1``, agents transmit in strict round-robin order:
    agent_0, agent_1, ..., agent_{N-1}, repeat.

    With ``n>1``, each sender slot is separated by ``n-1`` idle steps:
    agent_0, idle..., agent_1, idle..., etc.
    """

    def __init__(
        self,
        n_agents: int = 1,
        agent_index: int = 0,
        slot_spacing_multiplier: int = 1,
    ) -> None:
        super().__init__(n_agents)
        if self.n_agents <= 0:
            raise ValueError("n_agents must be positive")
        if slot_spacing_multiplier <= 0:
            raise ValueError("slot_spacing_multiplier must be positive")
        if agent_index < 0 or agent_index >= self.n_agents:
            raise ValueError("agent_index must be in [0, n_agents)")
        self.agent_index = int(agent_index)
        self.slot_spacing_multiplier = int(slot_spacing_multiplier)
        self.timestep = 0

    def reset(self) -> None:
        """Reset timestep counter."""
        self.timestep = 0

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, None]:
        """
        Send only in this agent's assigned synchronized slot.

        Args:
            observation: Current observation (unused)
            deterministic: Whether to use deterministic policy (unused)

        Returns:
            action: 1 in this agent's slot, else 0
            state: None
        """
        cycle_len = self.n_agents * self.slot_spacing_multiplier
        agent_slot = (self.agent_index) * self.slot_spacing_multiplier
        action = 1 if (self.timestep % cycle_len) == agent_slot else 0
        self.timestep += 1
        return action, None


def extract_predicted_local_gap(
    observation: np.ndarray,
    *,
    state_dim: int,
) -> np.ndarray:
    """
    Extract the predicted remote-estimation gap from the observation vector.

    The environment appends a local sensor KF estimate block immediately after
    the quantized local measurement, and the gap block directly after that:

        e_predicted = x_hat_sensor_local - x_hat_controller_shadow
    """
    obs = np.asarray(observation, dtype=float).ravel()
    if state_dim <= 0:
        raise ValueError("state_dim must be positive")
    min_obs_dim = 3 * int(state_dim)
    if obs.size < min_obs_dim:
        raise ValueError(
            "Observation is too short to contain the local estimation-gap block."
        )
    return obs[2 * state_dim : min_obs_dim]


def compute_value_of_update_score(
    observation: np.ndarray,
    *,
    weight_matrix: np.ndarray,
    state_dim: int,
) -> float:
    """
    Compute the quadratic value-of-update score ``e_predicted^T M e_predicted``.

    ``weight_matrix`` is typically the control-aware information matrix ``M_t``
    used by the environment's Kalman/LQR reward shaping.
    """
    predicted_gap = extract_predicted_local_gap(observation, state_dim=state_dim)
    weight = np.asarray(weight_matrix, dtype=float)
    if weight.shape != (state_dim, state_dim):
        raise ValueError(
            "weight_matrix must have shape "
            f"({state_dim}, {state_dim}), got {weight.shape}"
        )
    return float(predicted_gap @ weight @ predicted_gap)


def value_of_update_decision(
    observation: np.ndarray,
    *,
    threshold: float,
    weight_matrix: np.ndarray,
    state_dim: int,
) -> Tuple[int, float]:
    """
    Apply a threshold decision rule to the value-of-update score.

    Returns:
        action: ``1`` if the update value exceeds the threshold, else ``0``.
        score: The computed quadratic value-of-update score.
    """
    score = compute_value_of_update_score(
        observation,
        weight_matrix=weight_matrix,
        state_dim=state_dim,
    )
    return int(score > float(threshold)), score


class ValueOfUpdatePolicy(BaseHeuristicPolicy):
    """
    Threshold policy based on the control-aware value of update.

    At each step, this policy computes

        score_t = e_predicted,t^T M_t e_predicted,t

    and transmits iff ``score_t > threshold``.
    """

    def __init__(
        self,
        threshold: float,
        *,
        n_agents: int = 1,
        agent_index: int = 0,
        env: Optional[Any] = None,
    ) -> None:
        super().__init__(n_agents=n_agents)
        if agent_index < 0 or agent_index >= self.n_agents:
            raise ValueError("agent_index must be in [0, n_agents)")
        if env is None:
            raise ValueError(
                "ValueOfUpdatePolicy requires a live environment instance so it can "
                "reuse the environment's time-varying control weight M_t."
            )
        if not hasattr(env, "_get_kf_info_matrix"):
            raise ValueError(
                "Environment does not expose the control-aware information matrix helper."
            )
        self.threshold = float(threshold)
        self.agent_index = int(agent_index)
        self.env = env
        self.state_dim = int(getattr(env, "state_dim"))

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, Any]:
        weight_matrix = np.asarray(
            self.env._get_kf_info_matrix(self.agent_index),
            dtype=float,
        )
        action, score = value_of_update_decision(
            observation,
            threshold=self.threshold,
            weight_matrix=weight_matrix,
            state_dim=self.state_dim,
        )
        return action, {"value_of_update_score": score}


def _parse_perfect_sync_multiplier(policy_name: str) -> Optional[int]:
    """
    Parse perfect-sync policy names.

    Supported names:
    - perfect_sync      -> n = 1
    - perfect_sync_n2   -> n = 2
    - perfect_sync_2    -> n = 2
    """
    if policy_name == "perfect_sync":
        return 1
    match = re.fullmatch(r"perfect_sync(?:_n|_)(\d+)", policy_name)
    if match is None:
        return None
    return int(match.group(1))


def _parse_value_of_update_threshold(policy_name: str) -> Optional[float]:
    """
    Parse value-of-update policy names.

    Supported names:
    - value_of_update_0.1
    - value_of_update_threshold_1e-3
    - vou_0.1
    """
    float_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    match = re.fullmatch(
        rf"(?:value_of_update|vou)(?:_(?:threshold|theta|t))?_({float_pattern})",
        policy_name,
    )
    if match is None:
        return None
    return float(match.group(1))


# Dictionary for easy policy lookup
HEURISTIC_POLICIES = {
    'always_send': AlwaysSendPolicy,
    'never_send': NeverSendPolicy,
    'perfect_sync': lambda n_agents=1, agent_index=0: PerfectSyncPolicy(
        n_agents=n_agents, agent_index=agent_index, slot_spacing_multiplier=1
    ),
    'send_every_2': lambda n_agents=1: SendEveryNPolicy(n=2, n_agents=n_agents),
    'send_every_5': lambda n_agents=1: SendEveryNPolicy(n=5, n_agents=n_agents),
    'send_every_10': lambda n_agents=1: SendEveryNPolicy(n=10, n_agents=n_agents),
    'random_20': lambda n_agents=1, seed=None: RandomSendPolicy(prob=0.20, n_agents=n_agents, seed=seed),
    'random_25': lambda n_agents=1, seed=None: RandomSendPolicy(prob=0.25, n_agents=n_agents, seed=seed),
    'random_33': lambda n_agents=1, seed=None: RandomSendPolicy(prob=0.33, n_agents=n_agents, seed=seed),
    'random_50': lambda n_agents=1, seed=None: RandomSendPolicy(prob=0.5, n_agents=n_agents, seed=seed),
    'random_75': lambda n_agents=1, seed=None: RandomSendPolicy(prob=0.75, n_agents=n_agents, seed=seed),
    'zero_wait': lambda n_agents=1: ZeroWaitPolicy(n_agents=n_agents),
}


def get_heuristic_policy(
    policy_name: str,
    n_agents: int = 1,
    seed: Optional[int] = None,
    agent_index: int = 0,
    env: Optional[Any] = None,
) -> BaseHeuristicPolicy:
    """
    Get a heuristic policy by name.

    Args:
        policy_name: Name of the policy (see HEURISTIC_POLICIES keys)
        n_agents: Number of agents in the system
        seed: Random seed for stochastic policies
        agent_index: Agent index for coordinated multi-agent heuristics
        env: Optional live environment instance for policies that need access
            to time-varying control weights or environment metadata

    Returns:
        Policy instance

    Raises:
        ValueError: If policy_name is not recognized
    """
    sync_multiplier = _parse_perfect_sync_multiplier(policy_name)
    if sync_multiplier is not None:
        if sync_multiplier <= 0:
            raise ValueError("perfect_sync multiplier must be a positive integer")
        return PerfectSyncPolicy(
            n_agents=n_agents,
            agent_index=agent_index,
            slot_spacing_multiplier=sync_multiplier,
        )

    vou_threshold = _parse_value_of_update_threshold(policy_name)
    if vou_threshold is not None:
        return ValueOfUpdatePolicy(
            vou_threshold,
            n_agents=n_agents,
            agent_index=agent_index,
            env=env,
        )

    if policy_name not in HEURISTIC_POLICIES:
        available = ', '.join(sorted(HEURISTIC_POLICIES.keys()))
        raise ValueError(
            f"Unknown policy '{policy_name}'. Available policies: {available}. "
            "Pattern aliases: perfect_sync_n<k>, perfect_sync_<k>, "
            "value_of_update_<threshold>, value_of_update_threshold_<threshold>, "
            "vou_<threshold>."
        )

    policy_factory = HEURISTIC_POLICIES[policy_name]

    # Handle different factory signatures
    if policy_name.startswith('random'):
        return policy_factory(n_agents=n_agents, seed=seed)
    return policy_factory(n_agents=n_agents)
