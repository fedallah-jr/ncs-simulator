"""
Heuristic policies for networked control systems.

This module provides simple baseline policies that can be used for
comparison with learned policies. All policies follow the same interface
used by the visualization tools for easy integration.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional, Any, List

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


class ThresholdPolicy(BaseHeuristicPolicy):
    """Policy that sends when state magnitude exceeds a threshold."""

    def __init__(self, threshold: float = 1.0, n_agents: int = 1) -> None:
        """
        Initialize threshold policy.

        Args:
            threshold: State magnitude threshold for sending
            n_agents: Number of agents in the system
        """
        super().__init__(n_agents)
        self.threshold = threshold

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, None]:
        """
        Send if state magnitude exceeds threshold.

        The state is extracted from the beginning of the observation vector.

        Args:
            observation: Current observation (first elements are current state)
            deterministic: Whether to use deterministic policy (unused)

        Returns:
            action: 1 if ||state|| > threshold, else 0
            state: None
        """
        # Extract current state (first 2 elements by default)
        state_dim = 2
        current_state = observation[:state_dim]
        state_magnitude = np.linalg.norm(current_state)
        return (1 if state_magnitude > self.threshold else 0), None


class AdaptiveThresholdPolicy(BaseHeuristicPolicy):
    """Policy that adapts sending frequency based on state magnitude and recent throughput."""

    def __init__(self, base_threshold: float = 1.0, throughput_weight: float = 0.1, n_agents: int = 1) -> None:
        """
        Initialize adaptive threshold policy.

        Args:
            base_threshold: Base state magnitude threshold
            throughput_weight: Weight for throughput in threshold adaptation
            n_agents: Number of agents in the system
        """
        super().__init__(n_agents)
        self.base_threshold = base_threshold
        self.throughput_weight = throughput_weight

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, None]:
        """
        Send based on adaptive threshold considering state and throughput.

        Args:
            observation: Current observation [state, throughput, history...]
            deterministic: Whether to use deterministic policy (unused)

        Returns:
            action: 1 if state magnitude exceeds adaptive threshold, else 0
            state: None
        """
        # Extract current state and throughput
        state_dim = 2
        current_state = observation[:state_dim]
        current_throughput = observation[state_dim]

        # Compute state magnitude and adaptive threshold
        state_magnitude = np.linalg.norm(current_state)
        adaptive_threshold = self.base_threshold + self.throughput_weight * current_throughput

        return (1 if state_magnitude > adaptive_threshold else 0), None


# Dictionary for easy policy lookup
HEURISTIC_POLICIES = {
    'always_send': AlwaysSendPolicy,
    'never_send': NeverSendPolicy,
    'send_every_2': lambda n_agents=1: SendEveryNPolicy(n=2, n_agents=n_agents),
    'send_every_5': lambda n_agents=1: SendEveryNPolicy(n=5, n_agents=n_agents),
    'send_every_10': lambda n_agents=1: SendEveryNPolicy(n=10, n_agents=n_agents),
    'random_50': lambda n_agents=1, seed=None: RandomSendPolicy(prob=0.5, n_agents=n_agents, seed=seed),
    'random_25': lambda n_agents=1, seed=None: RandomSendPolicy(prob=0.25, n_agents=n_agents, seed=seed),
    'random_75': lambda n_agents=1, seed=None: RandomSendPolicy(prob=0.75, n_agents=n_agents, seed=seed),
    'threshold_1.0': lambda n_agents=1: ThresholdPolicy(threshold=1.0, n_agents=n_agents),
    'threshold_2.0': lambda n_agents=1: ThresholdPolicy(threshold=2.0, n_agents=n_agents),
    'threshold_0.5': lambda n_agents=1: ThresholdPolicy(threshold=0.5, n_agents=n_agents),
    'adaptive': lambda n_agents=1: AdaptiveThresholdPolicy(base_threshold=1.0, throughput_weight=0.1, n_agents=n_agents),
    'zero_wait': lambda n_agents=1: ZeroWaitPolicy(n_agents=n_agents),
}


def get_heuristic_policy(policy_name: str, n_agents: int = 1, seed: Optional[int] = None) -> BaseHeuristicPolicy:
    """
    Get a heuristic policy by name.

    Args:
        policy_name: Name of the policy (see HEURISTIC_POLICIES keys)
        n_agents: Number of agents in the system
        seed: Random seed for stochastic policies

    Returns:
        Policy instance

    Raises:
        ValueError: If policy_name is not recognized
    """
    if policy_name not in HEURISTIC_POLICIES:
        available = ', '.join(HEURISTIC_POLICIES.keys())
        raise ValueError(f"Unknown policy '{policy_name}'. Available policies: {available}")

    policy_factory = HEURISTIC_POLICIES[policy_name]

    # Handle different factory signatures
    if policy_name.startswith('random'):
        return policy_factory(n_agents=n_agents, seed=seed)
    else:
        return policy_factory(n_agents=n_agents)
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
