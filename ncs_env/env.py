from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .controller import Controller, compute_discrete_lqr_gain, compute_finite_horizon_lqr_gains
from .config import DEFAULT_CONFIG_PATH, load_config
from .network import NetworkModel
from .plant import Plant
from utils.schedulers import build_scheduler
from utils.reward_normalization import (
    RunningRewardNormalizer,
    ZScoreRewardNormalizer,
    get_shared_running_normalizer,
)


@dataclass
class RewardDefinition:
    mode: str
    comm_penalty_alpha: float
    simple_comm_penalty_alpha: float
    simple_freshness_decay: float = 0.0
    normalize: Optional[bool] = None  # None = auto-detect, True/False = explicit override
    normalizer: Optional[Union[ZScoreRewardNormalizer, RunningRewardNormalizer]] = None

    def __post_init__(self) -> None:
        if self.mode not in {"difference", "absolute", "simple", "simple_penalty"}:
            raise ValueError("state_error_reward must be 'difference', 'absolute', 'simple', or 'simple_penalty'")
        if float(self.simple_freshness_decay) < 0.0:
            raise ValueError("simple_freshness_decay must be >= 0")


def _should_normalize_reward(definition: RewardDefinition) -> bool:
    """
    Determine if a reward should be normalized based on its type.

    Logic:
    - If explicit override provided: use it
    - Otherwise, auto-detect:
      - simple, simple_penalty: NO (bounded rewards [-1, 0] or [0, 1])
      - absolute, difference: YES (unbounded rewards, scale varies by system)

    Returns:
        True if reward should be normalized, False otherwise
    """
    # Explicit user override takes precedence
    if definition.normalize is not None:
        return definition.normalize

    # Automatic detection based on reward type
    # Simple rewards are bounded and well-scaled, don't normalize
    # Absolute/difference rewards are unbounded, do normalize
    return definition.mode in ["absolute", "difference"]


class NCS_Env(gym.Env):
    """
    Multi-agent Networked Control System Gymnasium Environment.

    Each sensor decides whether to transmit. Observations include historical
    transport outcomes, current throughput estimate, and quantized state.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_agents: int = 3,
        episode_length: int = 1000,
        comm_cost: float = 0.01,
        config_path: Optional[str] = None,
        seed: Optional[int] = None,
        reward_override: Optional[Dict[str, Any]] = None,
        termination_override: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.episode_length = episode_length
        self.comm_cost = comm_cost
        self.reward_override = reward_override
        self.termination_override = termination_override

        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.config = load_config(str(self.config_path))
        self.system_cfg = self.config.get("system", {})
        self.timestep_duration = 0.01
        self.timestep = 0

        self.A = np.array(self.system_cfg.get("A"))
        self.B = np.array(self.system_cfg.get("B"))
        self.state_dim = self.A.shape[0]
        self.control_dim = self.B.shape[1]
        self.initial_state_scale_min, self.initial_state_scale_max = self._resolve_initial_state_scale_range(
            self.system_cfg, self.state_dim
        )

        observation_cfg = self.config.get("observation", {})
        self.history_window = observation_cfg.get("history_window", 10)
        self.state_history_window = observation_cfg.get("state_history_window", self.history_window)
        self.throughput_window = max(1, observation_cfg.get("throughput_window", 50))
        self.quantization_step = observation_cfg.get("quantization_step", 0.05)

        # Create a local RNG instance for this environment
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.action_space = spaces.Dict(
            {f"agent_{i}": spaces.Discrete(2) for i in range(n_agents)}
        )

        obs_dim = (
            self.state_dim  # current state
            + 1  # current throughput
            + self.state_history_window * self.state_dim  # previous states
            + self.history_window  # previous statuses
            + self.history_window  # previous throughputs
        )
        self.observation_space = spaces.Dict(
            {
                f"agent_{i}": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
                )
                for i in range(n_agents)
            }
        )

        self._initialize_systems()
        self._initialize_tracking_structures()

        # Compute reward normalization statistics after systems are initialized
        self._setup_reward_normalization()

    def _initialize_systems(self):
        """Initialize plants, controllers, and the shared network."""
        agent_matrices = self._resolve_agent_system_matrices()
        A, B = agent_matrices[0]
        W = np.array(self.system_cfg.get("process_noise_cov", np.eye(self.state_dim)))
        self.process_noise_cov = W
        self.measurement_noise_cov = np.array(
            self.system_cfg.get("measurement_noise_cov", 0.01 * np.eye(self.state_dim))
        )
        self._has_measurement_noise = not np.allclose(self.measurement_noise_cov, 0.0)
        initial_estimate_cov = np.array(
            self.system_cfg.get("initial_estimate_cov", np.eye(self.state_dim))
        )

        lqr_cfg = self.config.get("lqr", {})
        lqr_Q = np.array(lqr_cfg.get("Q", np.eye(self.state_dim)))
        lqr_R = np.array(lqr_cfg.get("R", np.eye(self.control_dim)))

        # Check if finite-horizon LQR is enabled (default: True)
        self.finite_horizon_enabled = bool(lqr_cfg.get("finite_horizon", True))

        if self.finite_horizon_enabled:
            # Compute time-varying gains for finite horizon (uses self.episode_length)
            self.K_list: List[Union[np.ndarray, List[np.ndarray]]] = [
                compute_finite_horizon_lqr_gains(A_i, B_i, lqr_Q, lqr_R, self.episode_length)
                for (A_i, B_i) in agent_matrices
            ]
        else:
            # Original infinite-horizon DARE solver
            self.K_list: List[np.ndarray] = [
                compute_discrete_lqr_gain(A_i, B_i, lqr_Q, lqr_R)
                for (A_i, B_i) in agent_matrices
            ]

        # For single-agent compatibility, store first agent's gain
        # (For finite-horizon, this will be the list of gains; for infinite-horizon, a single matrix)
        self.K = self.K_list[0]

        base_reward_cfg = self.config.get("reward", {})
        reward_cfg = self._merge_reward_override(base_reward_cfg, self.reward_override)
        self.reward_normalization_type = str(reward_cfg.get("normalization_type", "fixed")).lower()
        self.reward_normalization_episodes = int(reward_cfg.get("normalization_episodes", 30))
        self.reward_normalization_gamma = float(reward_cfg.get("normalization_gamma", 0.99))
        self.state_cost_matrix = np.array(
            reward_cfg.get("state_cost_matrix", np.eye(self.state_dim))
        )
        self.comm_recent_window = int(reward_cfg.get("comm_recent_window", self.history_window))
        self.comm_throughput_window = int(
            reward_cfg.get("comm_throughput_window", max(5 * self.history_window, 50))
        )
        base_comm_penalty_alpha = float(reward_cfg.get("comm_penalty_alpha", self.comm_cost))
        base_simple_comm_penalty_alpha = float(
            reward_cfg.get("simple_comm_penalty_alpha", base_comm_penalty_alpha)
        )
        base_simple_freshness_decay = float(reward_cfg.get("simple_freshness_decay", 0.0))
        reward_mixing_cfg = self._normalize_reward_mixing_cfg(reward_cfg.get("reward_mixing", {}))
        self.reward_definitions = self._build_reward_definitions(
            reward_cfg,
            reward_mixing_cfg,
            base_comm_penalty_alpha,
            base_simple_comm_penalty_alpha,
            base_simple_freshness_decay,
        )
        self.reward_mixing_enabled = bool(reward_mixing_cfg.get("enabled", False)) and len(self.reward_definitions) == 2
        self.comm_throughput_floor = float(reward_cfg.get("comm_throughput_floor", 1e-3))
        self.reward_scheduler: Optional[Callable[[int], float]] = self._build_reward_scheduler(
            reward_mixing_cfg
        )
        self.total_env_steps = 0
        self.last_mix_weight = self._current_mix_weight()
        self._running_reward_returns: Optional[List[List[float]]] = None
        base_termination_cfg = self.config.get("termination", {})
        termination_cfg = self._merge_termination_override(base_termination_cfg, self.termination_override)
        self.termination_enabled = bool(termination_cfg.get("enabled", False))
        self.termination_error_max = None
        if self.termination_enabled:
            self.termination_error_max = termination_cfg.get("state_error_max", None)
            if self.termination_error_max is None:
                raise ValueError("termination.enabled requires termination.state_error_max")
            self.termination_error_max = float(self.termination_error_max)
        self.termination_penalty = float(termination_cfg.get("penalty", 0.0))

        self.plants: List[Plant] = []
        for i in range(self.n_agents):
            x0 = self._sample_initial_state()
            # Pass self.np_random to Plant for isolated RNG
            A_i, B_i = agent_matrices[i]
            self.plants.append(Plant(A_i, B_i, W, x0, rng=self.np_random))

        self.controllers: List[Controller] = []
        for i in range(self.n_agents):
            initial_estimate = np.zeros(self.state_dim)
            self.controllers.append(
                Controller(
                    agent_matrices[i][0],
                    agent_matrices[i][1],
                    self.K_list[i],
                    initial_estimate,
                    process_noise_cov=W,
                    measurement_noise_cov=self.measurement_noise_cov,
                    initial_covariance=initial_estimate_cov,
                )
            )

        network_cfg = self.config.get("network", {})
        self.perfect_communication = bool(network_cfg.get("perfect_communication", False))
        self.network = NetworkModel(
            n_agents=self.n_agents,
            data_rate_kbps=network_cfg.get("data_rate_kbps", 250.0),
            data_packet_size=network_cfg.get("data_packet_size", 50),
            ack_packet_size=network_cfg.get("ack_packet_size", 10),
            max_queue_size=network_cfg.get("max_queue_size", 1),
            slots_per_step=network_cfg.get("slots_per_step", 32),
            mac_min_be=network_cfg.get("mac_min_be", 3),
            mac_max_be=network_cfg.get("mac_max_be", 5),
            max_csma_backoffs=network_cfg.get("max_csma_backoffs", 4),
            max_frame_retries=network_cfg.get("max_frame_retries", 3),
            mac_ack_wait_us=network_cfg.get("mac_ack_wait_us", 864.0),
            mac_ack_turnaround_us=network_cfg.get("mac_ack_turnaround_us", 192.0),
            cca_time_us=network_cfg.get("cca_time_us", 128.0),
            mac_ack_size_bytes=network_cfg.get("mac_ack_size_bytes", 5),
            rng=self.np_random,
        )

    def _initialize_tracking_structures(self):
        """Prepare history and throughput tracking."""
        history_len = max(self.history_window, self.comm_recent_window)
        self.decision_history: List[deque] = []
        for _ in range(self.n_agents):
            history = deque(maxlen=history_len)
            for _ in range(history_len):
                history.append({"timestamp": -1, "status": 0})
            self.decision_history.append(history)

        self.state_history: List[deque] = []
        for _ in range(self.n_agents):
            state_hist = deque(maxlen=self.state_history_window)
            for _ in range(self.state_history_window):
                state_hist.append(np.zeros(self.state_dim, dtype=float))
            self.state_history.append(state_hist)

        self.throughput_history: deque = deque(maxlen=self.history_window)
        for _ in range(self.history_window):
            self.throughput_history.append(0.0)

        self.pending_transmissions: List[Dict[int, Dict[str, int]]] = [
            {} for _ in range(self.n_agents)
        ]
        self.net_tx_attempts = np.zeros(self.n_agents, dtype=np.int64)
        self.net_tx_acks = np.zeros(self.n_agents, dtype=np.int64)
        self.net_tx_drops = np.zeros(self.n_agents, dtype=np.int64)
        self.net_tx_rewrites = np.zeros(self.n_agents, dtype=np.int64)
        self.throughput_records: deque = deque()
        self.comm_success_records: List[deque] = [deque() for _ in range(self.n_agents)]
        self.last_measurements: List[np.ndarray] = [
            np.zeros(self.state_dim, dtype=float) for _ in range(self.n_agents)
        ]
        self.reward_component_stats: Dict[str, Dict[str, float]] = {
            f"agent_{i}": {"prev_error_sum": 0.0, "curr_error_sum": 0.0, "comm_penalty_sum": 0.0, "count": 0.0}
            for i in range(self.n_agents)
        }
        current_mix_weight = self._current_mix_weight()
        self.last_mix_weight = current_mix_weight
        base_components: Dict[str, float] = {
            "prev_error": 0.0,
            "curr_error": 0.0,
            "comm_penalty": 0.0,
        }
        if self.reward_mixing_enabled:
            base_components.update(
                {
                    "primary_reward": 0.0,
                    "secondary_reward": 0.0,
                    "mix_weight": float(current_mix_weight),
                }
            )
        self.last_reward_components: Dict[str, Dict[str, float]] = {
            f"agent_{i}": dict(base_components)
            for i in range(self.n_agents)
        }
        self.last_errors: List[float] = [0.0 for _ in range(self.n_agents)]
        self.last_termination_reasons: List[str] = []
        self.last_termination_agents: List[int] = []
        self.last_bad_termination = False

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment to initial state."""
        # This call updates self.np_random with a new seed if provided
        super().reset(seed=seed)

        self.timestep = 0

        # Update RNG references in subsystems after super().reset()
        # This ensures all subsystems use the same isolated RNG stream
        for plant in self.plants:
            plant.rng = self.np_random
            x0 = self._sample_initial_state()
            plant.reset(x0)

        for controller in self.controllers:
            controller.reset(np.zeros(self.state_dim))

        self.network.rng = self.np_random
        self.network.reset()
        self._initialize_tracking_structures()
        self._reset_running_returns()
        for idx in range(self.n_agents):
            self.last_measurements[idx] = self.plants[idx].get_state().copy()

        observations = self._get_observations()
        for idx in range(self.n_agents):
            self.last_errors[idx] = self._compute_state_error(self.plants[idx].get_state())
        info = self._get_info()
        return observations, info

    def step(self, actions: Dict[str, int]):
        """Execute one timestep of the environment."""
        self.timestep += 1

        # State index is the time index of the current plant state
        # After reset: timestep=0, state is x[0]
        # After first step: timestep=1, state is still x[0] before plant update
        # We use state_index to clearly refer to which x[k] we're measuring
        state_index = self.timestep - 1

        # Store prior for each controller at current state index
        # This enables delayed measurement handling in network mode
        for i in range(self.n_agents):
            self.controllers[i].store_prior(state_index)

        # Kalman filter measurement update (conditionally, based on packet delivery)
        # Note: predict() is called AFTER plant update to maintain correct timing
        delivered_controller_ids = set()
        delivered_message_ages: Dict[int, int] = {}

        if self.perfect_communication:
            for i in range(self.n_agents):
                action = actions[f"agent_{i}"]
                if action == 1:
                    self.net_tx_attempts[i] += 1
                    self.net_tx_acks[i] += 1
                    state = self.plants[i].get_state()
                    measurement = state + self._sample_measurement_noise()
                    self.controllers[i].update(measurement)
                    self.last_measurements[i] = measurement
                    self._record_decision(i, status=1)
                    # Use state_index for consistency (though _log_successful_comm
                    # returns early for perfect_communication anyway)
                    self._log_successful_comm(i, state_index, state_index)
                    delivered_controller_ids.add(i)
                    delivered_message_ages[i] = 0
                else:
                    self._record_decision(i, status=0)
        else:
            delivered_controller_ids = set()
            for i in range(self.n_agents):
                action = actions[f"agent_{i}"]
                if action == 1:
                    self.net_tx_attempts[i] += 1
                    state = self.plants[i].get_state()
                    measurement = state + self._sample_measurement_noise()
                    # Use state_index as the measurement timestamp
                    # This clearly indicates measurement is of x[state_index]
                    measurement_timestamp = state_index
                    overwritten = self.network.queue_data_packet(i, measurement, measurement_timestamp)
                    # Handle queue overwrite: mark the dropped packet as failed (status=3)
                    if overwritten is not None and overwritten.packet_type == "data":
                        self.net_tx_rewrites[i] += 1
                        dropped_timestamp = overwritten.payload.get("timestamp")
                        dropped_entry = self.pending_transmissions[i].pop(dropped_timestamp, None)
                        if dropped_entry is not None:
                            dropped_entry["status"] = 3
                    entry = self._record_decision(i, status=2)
                    entry["send_timestamp"] = measurement_timestamp
                    self.pending_transmissions[i][measurement_timestamp] = entry
                else:
                    self._record_decision(i, status=0)
            # Run the micro-slot network for this environment step
            slots_to_run = self.network.slots_per_step
            for _ in range(slots_to_run):
                network_result = self.network.run_slot()

                for packet in network_result["delivered_data"]:
                    controller_id = packet.dest_id
                    measurement = packet.payload["state"]
                    measurement_timestamp = packet.payload["timestamp"]
                    age_steps = max(0, int(state_index - int(measurement_timestamp)))

                    self.controllers[controller_id].delayed_update(measurement, measurement_timestamp)
                    self.last_measurements[controller_id] = measurement
                    delivered_controller_ids.add(controller_id)
                    existing_age = delivered_message_ages.get(controller_id)
                    if existing_age is None or age_steps < existing_age:
                        delivered_message_ages[controller_id] = age_steps

                for mac_ack in network_result.get("delivered_mac_acks", []):
                    sensor_id = mac_ack.get("sensor_id")
                    measurement_timestamp = mac_ack.get("measurement_timestamp")
                    if sensor_id is None or measurement_timestamp is None:
                        continue
                    if 0 <= int(sensor_id) < self.n_agents:
                        self.net_tx_acks[int(sensor_id)] += 1
                    entry = self.pending_transmissions[sensor_id].pop(measurement_timestamp, None)
                    if entry is not None:
                        entry["status"] = 1
                        self.throughput_records.append(
                            {
                                "timestamp": self.timestep,
                                "bits": self.network.data_packet_size * 8,
                            }
                        )
                        self._log_successful_comm(sensor_id, measurement_timestamp, self.timestep)

                for packet in network_result["dropped_packets"]:
                    if packet.packet_type == "data":
                        sensor_id = packet.source_id
                        if 0 <= int(sensor_id) < self.n_agents:
                            self.net_tx_drops[int(sensor_id)] += 1
                        measurement_timestamp = packet.payload.get("timestamp")
                        entry = self.pending_transmissions[sensor_id].pop(measurement_timestamp, None)
                        if entry is not None:
                            entry["status"] = 3

        # Compute control and update plants
        for i in range(self.n_agents):
            u = self.controllers[i].compute_control()
            self.plants[i].step(u)

        # Kalman filter time update (predict) - done AFTER plant update
        # This propagates estimates forward using dynamics and the control just applied
        # Prepares x_hat[k|k-1] for the next timestep's measurement update
        for i in range(self.n_agents):
            self.controllers[i].predict()

        mix_weight = self._current_mix_weight()
        rewards = {}
        termination_triggered = False
        termination_reasons: List[str] = []
        termination_agents: List[int] = []
        termination_error_max = self.termination_error_max if self.termination_enabled else None
        for i in range(self.n_agents):
            x = self.plants[i].get_state()
            action = actions[f"agent_{i}"]
            prev_error = self.last_errors[i]
            raw_error = self._compute_state_error(x)
            curr_error = raw_error
            if not np.isfinite(raw_error):
                termination_triggered = True
                termination_reasons.append("non_finite")
                termination_agents.append(i)
                curr_error = (
                    float(termination_error_max)
                    if termination_error_max is not None
                    else float(prev_error)
                )
            elif termination_error_max is not None and raw_error >= termination_error_max:
                termination_triggered = True
                termination_reasons.append("state_error_max")
                termination_agents.append(i)
                curr_error = float(termination_error_max)
            info_arrived = i in delivered_controller_ids
            message_age_steps = delivered_message_ages.get(i) if info_arrived else None

            if self.reward_mixing_enabled and len(self.reward_definitions) == 2:
                primary_def, secondary_def = self.reward_definitions
                primary_components = self._compute_reward_for_definition(
                    i, primary_def, 0, prev_error, curr_error, action, info_arrived, message_age_steps
                )
                secondary_components = self._compute_reward_for_definition(
                    i, secondary_def, 1, prev_error, curr_error, action, info_arrived, message_age_steps
                )
                primary_reward = primary_components.pop("reward")
                secondary_reward = secondary_components.pop("reward")
                combined_reward = (1.0 - mix_weight) * primary_reward + mix_weight * secondary_reward
                combined_comm_penalty = (
                    (1.0 - mix_weight) * primary_components["comm_penalty"]
                    + mix_weight * secondary_components["comm_penalty"]
                )
                reward_components: Dict[str, float] = {
                    "prev_error": float(prev_error),
                    "curr_error": float(curr_error),
                    "comm_penalty": float(combined_comm_penalty),
                    "primary_reward": float(primary_reward),
                    "secondary_reward": float(secondary_reward),
                    "mix_weight": float(mix_weight),
                }
                for key in ("info_arrived", "message_age_steps", "freshness"):
                    if key in primary_components or key in secondary_components:
                        reward_components[key] = float(primary_components.get(key, secondary_components.get(key, 0.0)))
            else:
                definition = self.reward_definitions[0]
                components = self._compute_reward_for_definition(
                    i, definition, 0, prev_error, curr_error, action, info_arrived, message_age_steps
                )
                reward_value = components.pop("reward")
                combined_reward = reward_value
                combined_comm_penalty = components.get("comm_penalty", 0.0)
                reward_components = components

            reward = float(combined_reward)
            agent_key = f"agent_{i}"
            rewards[agent_key] = reward
            self.last_reward_components[agent_key] = reward_components
            stats = self.reward_component_stats[agent_key]
            stats["prev_error_sum"] += float(prev_error)
            stats["curr_error_sum"] += float(curr_error)
            stats["comm_penalty_sum"] += float(combined_comm_penalty)
            stats["count"] += 1
            self.last_errors[i] = curr_error

        if termination_triggered and self.termination_penalty != 0.0:
            for i in range(self.n_agents):
                agent_key = f"agent_{i}"
                rewards[agent_key] += self.termination_penalty
                self.last_reward_components[agent_key]["termination_penalty"] = float(
                    self.termination_penalty
                )

        self.last_mix_weight = mix_weight
        self.total_env_steps += 1

        time_limit_reached = self.timestep >= self.episode_length
        finite_horizon_terminal = self.finite_horizon_enabled and time_limit_reached
        terminated = {
            f"agent_{i}": termination_triggered or finite_horizon_terminal
            for i in range(self.n_agents)
        }
        truncated = {
            f"agent_{i}": time_limit_reached and not (termination_triggered or finite_horizon_terminal)
            for i in range(self.n_agents)
        }
        if termination_triggered:
            self.last_termination_reasons = sorted(set(termination_reasons))
            self.last_termination_agents = sorted(set(termination_agents))
            self.last_bad_termination = True
        elif finite_horizon_terminal:
            self.last_termination_reasons = ["finite_horizon"]
            self.last_termination_agents = list(range(self.n_agents))
            self.last_bad_termination = False
        elif time_limit_reached:
            self.last_termination_reasons = ["time_limit"]
            self.last_termination_agents = list(range(self.n_agents))
            self.last_bad_termination = False
        else:
            self.last_termination_reasons = []
            self.last_termination_agents = []
            self.last_bad_termination = False
        if self._running_reward_returns is not None:
            for i in range(self.n_agents):
                agent_key = f"agent_{i}"
                if terminated[agent_key] or truncated[agent_key]:
                    for returns in self._running_reward_returns:
                        returns[i] = 0.0

        # Get observations AFTER plant update (Gym step contract)
        # step() returns the resulting state s[k+1] after action a[k] is applied
        observations = self._get_observations()
        infos = self._get_info()
        return observations, rewards, terminated, truncated, infos

    def _record_decision(self, agent_idx: int, status: int) -> Dict[str, int]:
        """Append a new decision outcome to the history."""
        entry = {"timestamp": self.timestep, "status": status}
        self.decision_history[agent_idx].append(entry)
        return entry

    def _sample_initial_state(self) -> np.ndarray:
        """
        Sample an initial plant state with magnitude constrained to [scale_min, scale_max]
        per dimension. Sign is chosen uniformly.
        """
        scale_min = self.initial_state_scale_min
        scale_max = self.initial_state_scale_max
        magnitudes = self.np_random.uniform(low=scale_min, high=scale_max)
        signs = self.np_random.choice([-1.0, 1.0], size=self.state_dim)
        return signs * magnitudes

    def _resolve_agent_system_matrices(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Resolve per-agent (A, B) matrices.

        Supports `system.heterogeneous_plants` as a list of dictionaries. Each entry may
        override `A` and/or `B` for the corresponding agent index. A separate entry with
        `"default": true` can define a fallback override used for agents whose index is
        not explicitly listed.

        All agents must share the same state/control dimensions because the observation
        and action spaces are shared across agents.
        """
        base_A = np.array(self.system_cfg.get("A"))
        base_B = np.array(self.system_cfg.get("B"))
        het_cfg = self.system_cfg.get("heterogeneous_plants", None)
        if not isinstance(het_cfg, list) or len(het_cfg) == 0:
            return [(base_A, base_B) for _ in range(self.n_agents)]

        default_entry: Optional[Dict[str, Any]] = None
        for entry in het_cfg:
            if isinstance(entry, dict) and bool(entry.get("default", False)):
                default_entry = entry
                break

        matrices: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(self.n_agents):
            entry: Dict[str, Any] = {}
            if i < len(het_cfg) and isinstance(het_cfg[i], dict):
                entry = het_cfg[i]
            elif default_entry is not None:
                entry = default_entry

            A_i = np.array(entry.get("A", base_A))
            B_i = np.array(entry.get("B", base_B))
            if A_i.shape != base_A.shape:
                raise ValueError("heterogeneous_plants entries must not change the A matrix shape")
            if B_i.shape != base_B.shape:
                raise ValueError("heterogeneous_plants entries must not change the B matrix shape")
            matrices.append((A_i, B_i))

        return matrices

    @staticmethod
    def _resolve_initial_state_scale_range(system_cfg: Dict[str, Any], state_dim: int):
        """
        Resolve initial state scale range from config.

        Supports legacy `initial_state_scale` (scalar/list) as a symmetric bound and
        new `initial_state_scale_min`/`initial_state_scale_max` fields. Defaults to
        magnitudes in [0.9, 1.0] per dimension when unspecified.
        """

        def _to_array(cfg_value: Any, default: float) -> np.ndarray:
            arr = np.asarray(cfg_value if cfg_value is not None else default, dtype=float).flatten()
            if arr.size == 0:
                arr = np.asarray(default, dtype=float).flatten()
            if arr.size == 1:
                return np.full(state_dim, float(arr.item()))
            if arr.size != state_dim:
                raise ValueError(
                    "initial_state_scale entries must be scalar or match the state dimension"
                )
            return arr

        legacy_scale = system_cfg.get("initial_state_scale", None)
        min_cfg = system_cfg.get("initial_state_scale_min", None)
        max_cfg = system_cfg.get("initial_state_scale_max", None)

        if legacy_scale is not None:
            # Backward compatibility: symmetric range [-scale, scale]
            legacy = np.abs(_to_array(legacy_scale, 1.0))
            return legacy, legacy

        default_min = 0.9
        default_max = 1.0
        scale_min = np.abs(_to_array(min_cfg, default_min))
        scale_max = np.abs(_to_array(max_cfg, default_max))
        if np.any(scale_max < scale_min):
            raise ValueError("initial_state_scale_max must be >= initial_state_scale_min for all dimensions")
        return scale_min, scale_max

    @staticmethod
    def _merge_reward_override(
        base_reward_cfg: Dict[str, Any], reward_override: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge reward override dict onto the base reward config."""
        if reward_override is None:
            return dict(base_reward_cfg)
        merged = dict(base_reward_cfg)
        merged.update(reward_override)
        return merged

    @staticmethod
    def _merge_termination_override(
        base_termination_cfg: Dict[str, Any],
        termination_override: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Merge termination override dict onto the base termination config."""
        if termination_override is None:
            return dict(base_termination_cfg)
        merged = dict(base_termination_cfg)
        merged.update(termination_override)
        return merged

    @staticmethod
    def _normalize_reward_mixing_cfg(raw_cfg: Any) -> Dict[str, Any]:
        """Normalize the reward_mixing config entry into a dict with an 'enabled' flag."""
        if isinstance(raw_cfg, bool):
            return {"enabled": raw_cfg}
        if raw_cfg is None:
            return {"enabled": False}
        if not isinstance(raw_cfg, dict):
            raise ValueError("reward_mixing must be a boolean or mapping")
        return raw_cfg

    def _build_reward_definitions(
        self,
        reward_cfg: Dict[str, Any],
        mixing_cfg: Dict[str, Any],
        base_comm_penalty_alpha: float,
        base_simple_comm_penalty_alpha: float,
        base_simple_freshness_decay: float,
    ) -> List[RewardDefinition]:
        """Create reward definitions for primary/secondary reward functions."""
        base_mode = reward_cfg.get("state_error_reward", "difference")
        base_normalize = reward_cfg.get("normalize", None)
        if base_normalize is not None:
            base_normalize = bool(base_normalize)
        if not mixing_cfg.get("enabled", False):
            normalize_flag = base_normalize if base_normalize is not None else False
            return [
                RewardDefinition(
                    mode=base_mode,
                    comm_penalty_alpha=base_comm_penalty_alpha,
                    simple_comm_penalty_alpha=base_simple_comm_penalty_alpha,
                    simple_freshness_decay=base_simple_freshness_decay,
                    normalize=normalize_flag,
                    normalizer=None,
                )
            ]

        rewards_list = mixing_cfg.get("rewards", [])
        if not isinstance(rewards_list, list) or len(rewards_list) != 2:
            raise ValueError(
                "reward.reward_mixing.rewards must contain exactly two reward configs "
                "when reward_mixing.enabled is true."
            )

        definitions: List[RewardDefinition] = []
        for reward_entry in rewards_list:
            # Parse optional normalize override
            normalize_override = reward_entry.get("normalize", base_normalize)
            if normalize_override is not None:
                normalize_override = bool(normalize_override)

            definitions.append(
                RewardDefinition(
                    mode=str(reward_entry.get("state_error_reward", base_mode)),
                    comm_penalty_alpha=float(
                        reward_entry.get("comm_penalty_alpha", base_comm_penalty_alpha)
                    ),
                    simple_comm_penalty_alpha=float(
                        reward_entry.get(
                            "simple_comm_penalty_alpha",
                            reward_entry.get("comm_penalty_alpha", base_simple_comm_penalty_alpha),
                        )
                    ),
                    simple_freshness_decay=float(
                        reward_entry.get("simple_freshness_decay", base_simple_freshness_decay)
                    ),
                    normalize=normalize_override,  # Optional user override
                    normalizer=None,  # Will be set during _compute_reward_normalizers()
                )
            )
        return definitions

    def _build_reward_scheduler(self, mixing_cfg: Dict[str, Any]) -> Optional[Callable[[int], float]]:
        """Create the mixing scheduler callable (or None when mixing is disabled)."""
        if not mixing_cfg.get("enabled", False):
            return None
        scheduler_cfg = mixing_cfg.get("scheduler", {})
        if not isinstance(scheduler_cfg, dict):
            scheduler_cfg = {}
        default_steps = int(mixing_cfg.get("total_steps", 100_000))
        default_start = float(mixing_cfg.get("start_value", 0.0))
        default_end = float(mixing_cfg.get("end_value", 1.0))
        return build_scheduler(
            scheduler_cfg,
            default_start=default_start,
            default_end=default_end,
            default_steps=default_steps,
        )

    def _current_mix_weight(self) -> float:
        """Return the current reward mixing weight."""
        if not self.reward_mixing_enabled or self.reward_scheduler is None:
            return 0.0
        return float(self.reward_scheduler(self.total_env_steps))

    def _quantize_state(self, state: np.ndarray) -> np.ndarray:
        """Quantize the measured plant state."""
        if self.quantization_step is None or self.quantization_step <= 0:
            return state.copy()
        step = self.quantization_step
        return np.round(state / step) * step

    def _compute_throughput(self) -> float:
        """Compute throughput (kbps) over the sliding window."""
        if self.perfect_communication:
            return 0.0
        if self.timestep <= 0:
            return 0.0
        window_start = self.timestep - self.throughput_window + 1
        while self.throughput_records and self.throughput_records[0]["timestamp"] < window_start:
            self.throughput_records.popleft()

        total_bits = sum(record["bits"] for record in self.throughput_records)
        effective_steps = min(self.throughput_window, self.timestep)
        window_duration = effective_steps * self.timestep_duration
        if window_duration == 0:
            return 0.0
        throughput_kbps = (total_bits / window_duration) / 1000.0
        return throughput_kbps

    def _sample_measurement_noise(self) -> np.ndarray:
        """Return measurement noise vector (zero when disabled)."""
        if not self._has_measurement_noise:
            return np.zeros(self.state_dim)
        return self.np_random.multivariate_normal(np.zeros(self.state_dim), self.measurement_noise_cov)

    def _log_successful_comm(self, agent_idx: int, send_timestamp: int, ack_timestamp: int) -> None:
        """Record ACKed packet for throughput-based penalties."""
        if self.perfect_communication:
            return
        delay_steps = max(1, ack_timestamp - send_timestamp)
        buffer = self.comm_success_records[agent_idx]
        buffer.append({"timestamp": ack_timestamp, "delay": delay_steps})
        self._prune_comm_records(agent_idx)

    def _prune_comm_records(self, agent_idx: int) -> None:
        """Drop stale ACK records outside the throughput window."""
        window_start = self.timestep - self.comm_throughput_window
        buffer = self.comm_success_records[agent_idx]
        while buffer and buffer[0]["timestamp"] < window_start:
            buffer.popleft()

    def _compute_agent_throughput(self, agent_idx: int) -> float:
        """Return packets-per-step throughput estimate for the agent."""
        if self.perfect_communication:
            return float("inf")
        self._prune_comm_records(agent_idx)
        buffer = self.comm_success_records[agent_idx]
        if not buffer:
            return self.comm_throughput_floor
        total_delay = sum(record["delay"] for record in buffer)
        if total_delay <= 0:
            total_delay = 1.0
        throughput = len(buffer) / total_delay
        return max(throughput, self.comm_throughput_floor)

    def _compute_state_error(self, state: np.ndarray) -> float:
        """Quadratic tracking error (x - x_ref)^T Q (x - x_ref)."""
        dx = state  # reference is zero
        return float(dx.T @ self.state_cost_matrix @ dx)

    def _recent_transmission_count(self, agent_idx: int) -> int:
        """Count transmissions (ACKed or pending) over the short window."""
        recent_entries = list(self.decision_history[agent_idx])[-self.comm_recent_window :]
        return sum(1 for entry in recent_entries if entry["status"] > 0)

    def _setup_reward_normalization(self) -> None:
        if not any(_should_normalize_reward(defn) for defn in self.reward_definitions):
            return
        if self.reward_normalization_type == "running":
            self._init_running_reward_normalizers()
            return
        if self.reward_normalization_type != "fixed":
            raise ValueError("reward.normalization_type must be 'fixed' or 'running'")
        print("\n[Reward Normalization] Computing statistics...")
        self._compute_reward_normalizers(self.reward_normalization_episodes)
        print("[Reward Normalization] Statistics computed.\n")

    def _running_normalizer_key(self, definition_idx: int, definition: RewardDefinition) -> str:
        signature = {
            "config_path": str(self.config_path.resolve()),
            "definition_idx": int(definition_idx),
            "mode": definition.mode,
            "comm_penalty_alpha": float(definition.comm_penalty_alpha),
            "simple_comm_penalty_alpha": float(definition.simple_comm_penalty_alpha),
            "simple_freshness_decay": float(definition.simple_freshness_decay),
            "state_cost_matrix": np.asarray(self.state_cost_matrix).tolist(),
            "comm_recent_window": int(self.comm_recent_window),
            "comm_throughput_window": int(self.comm_throughput_window),
            "comm_throughput_floor": float(self.comm_throughput_floor),
        }
        return json.dumps(signature, sort_keys=True, separators=(",", ":"))

    def _init_running_reward_normalizers(self) -> None:
        self._running_reward_returns = [
            [0.0 for _ in range(self.n_agents)] for _ in range(len(self.reward_definitions))
        ]
        for idx, definition in enumerate(self.reward_definitions):
            if _should_normalize_reward(definition):
                key = self._running_normalizer_key(idx, definition)
                definition.normalizer = get_shared_running_normalizer(key)

    def _reset_running_returns(self) -> None:
        if self._running_reward_returns is None:
            return
        for idx in range(len(self._running_reward_returns)):
            for i in range(self.n_agents):
                self._running_reward_returns[idx][i] = 0.0

    def _compute_reward_normalizers(self, episodes: int) -> None:
        """
        Compute normalization statistics for rewards that need it.

        This runs random rollouts to estimate mean/std of each reward component,
        but ONLY for rewards where normalization is enabled (unbounded rewards).
        Simple/simple_penalty rewards are NOT normalized as they are already bounded.
        """
        saved_total_steps = self.total_env_steps
        saved_last_mix_weight = self.last_mix_weight
        for idx, definition in enumerate(self.reward_definitions):
            if not _should_normalize_reward(definition):
                # Skip normalization for this reward (e.g., simple_penalty)
                print(f"  Reward {idx} ({definition.mode}): No normalization (bounded reward)")
                continue

            print(f"  Reward {idx} ({definition.mode}): Computing normalization statistics...")

            # Compute normalizer for this reward definition
            temp_normalizer = self._compute_normalizer_for_definition(definition, idx, episodes=episodes)
            definition.normalizer = temp_normalizer

            print(f"    → Normalizer: mean={temp_normalizer.mean:.3f}, std={temp_normalizer.std:.3f}")
        self.total_env_steps = saved_total_steps
        self.last_mix_weight = saved_last_mix_weight

    def _compute_normalizer_for_definition(
        self,
        definition: RewardDefinition,
        definition_idx: int,
        episodes: int = 30,
    ) -> ZScoreRewardNormalizer:
        """
        Compute normalizer for a single reward definition.

        Runs random episodes and collects rewards for this specific reward component.

        Args:
            definition: The reward definition to normalize
            definition_idx: Index of the definition (for debugging)
            episodes: Number of random episodes to sample

        Returns:
            ZScoreRewardNormalizer with computed mean and std
        """
        # Use the environment's RNG to generate seeds for normalization episodes
        rng = self.np_random
        collected_rewards = []

        for ep in range(episodes):
            # Reset environment
            self.reset(seed=int(rng.integers(0, 1_000_000)))
            prev_errors = [float(x) for x in self.last_errors]

            for step in range(self.episode_length):
                # Random actions for all agents
                actions = {}
                for i in range(self.n_agents):
                    actions[f"agent_{i}"] = int(rng.integers(0, 2))  # 0 or 1

                # Step environment
                try:
                    _, _, terminated, truncated, info = self.step(actions)
                except Exception:
                    # If step fails during initialization, skip this sample
                    break

                # Collect error rewards for each agent
                for i in range(self.n_agents):
                    action = actions[f"agent_{i}"]
                    x = self.plants[i].x
                    curr_error_raw = self._compute_state_error(x)
                    components = info.get("reward_components", {}).get(f"agent_{i}", {})
                    prev_error = float(components.get("prev_error", prev_errors[i]))
                    curr_error = float(components.get("curr_error", curr_error_raw))
                    info_arrived = bool(components.get("info_arrived", 0.0))
                    message_age_steps = components.get("message_age_steps", None)
                    reward_components = self._compute_reward_for_definition(
                        i,
                        definition,
                        definition_idx,
                        prev_error,
                        curr_error,
                        action,
                        info_arrived,
                        message_age_steps,
                        apply_normalization=False,
                    )

                    collected_rewards.append(float(reward_components["reward"]))
                    prev_errors[i] = curr_error

                if all(terminated.values()) or all(truncated.values()):
                    break

        # Compute statistics
        if len(collected_rewards) == 0:
            # Fallback if no rewards collected
            print(f"    ⚠️  Warning: No rewards collected for normalization, using defaults")
            return ZScoreRewardNormalizer(mean=0.0, std=1.0)

        rewards_array = np.array(collected_rewards, dtype=np.float32)
        mean = float(rewards_array.mean())
        std = float(rewards_array.std())

        # Ensure std is not too small to avoid division issues
        std = max(std, 1e-8)

        return ZScoreRewardNormalizer(mean=mean, std=std)

    def _apply_running_normalization(
        self,
        definition_idx: int,
        agent_idx: int,
        reward_value: float,
    ) -> float:
        if self._running_reward_returns is None:
            return reward_value
        returns = self._running_reward_returns[definition_idx]
        returns[agent_idx] = self.reward_normalization_gamma * returns[agent_idx] + reward_value
        normalizer = self.reward_definitions[definition_idx].normalizer
        if isinstance(normalizer, RunningRewardNormalizer):
            normalizer.update(returns[agent_idx])
            return normalizer(reward_value)
        return reward_value

    def _compute_reward_for_definition(
        self,
        agent_idx: int,
        definition: RewardDefinition,
        definition_idx: int,
        prev_error: float,
        curr_error: float,
        action: int,
        info_arrived: bool,
        message_age_steps: Optional[int] = None,
        apply_normalization: bool = True,
    ) -> Dict[str, float]:
        """Compute reward and components for the given reward definition."""
        comm_penalty = 0.0
        if not self.perfect_communication and action == 1:
            penalty_alpha = (
                definition.comm_penalty_alpha
                if definition.mode not in {"simple", "simple_penalty"}
                else definition.simple_comm_penalty_alpha
            )
            if penalty_alpha > 0:
                recent_tx = self._recent_transmission_count(agent_idx)
                throughput_estimate = self._compute_agent_throughput(agent_idx)
                comm_penalty = penalty_alpha * (recent_tx / throughput_estimate)

        if definition.mode == "difference":
            error_reward = prev_error - curr_error
        elif definition.mode == "absolute":
            error_reward = -curr_error
        elif definition.mode == "simple_penalty":
            # Symmetric version of "simple": 0 if measurement delivered, -1 otherwise
            if info_arrived:
                error_reward = 0.0
            else:
                error_reward = -1.0
        else:  # mode == "simple"
            if not info_arrived:
                error_reward = 0.0
            else:
                age_steps = max(0, int(message_age_steps or 0))
                error_reward = float(np.exp(-float(definition.simple_freshness_decay) * float(age_steps)))

        reward_value = float(error_reward - comm_penalty)

        if apply_normalization:
            # Apply normalization to full reward if configured
            if isinstance(definition.normalizer, RunningRewardNormalizer):
                reward_value = self._apply_running_normalization(definition_idx, agent_idx, reward_value)
            elif definition.normalizer is not None:
                reward_value = definition.normalizer(reward_value)
        components: Dict[str, float] = {
            "prev_error": float(prev_error),
            "curr_error": float(curr_error),
            "comm_penalty": float(comm_penalty),
            "reward": reward_value,
        }
        if definition.mode in {"simple", "simple_penalty"}:
            components["info_arrived"] = 1.0 if info_arrived else 0.0
            components["message_age_steps"] = float(max(0, int(message_age_steps or 0))) if info_arrived else 0.0
            if definition.mode == "simple":
                components["freshness"] = float(error_reward) if info_arrived else 0.0
            else:  # simple_penalty
                components["penalty"] = float(error_reward)
        return components

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Construct observations for all agents."""
        observations = {}
        throughput = self._compute_throughput()

        for i in range(self.n_agents):
            status_history = list(self.decision_history[i])[-self.history_window :]
            if len(status_history) < self.history_window:
                status_history = [{"timestamp": -1, "status": 0}] * (
                    self.history_window - len(status_history)
                ) + status_history
            status_values = [entry["status"] for entry in status_history]

            prev_states = list(self.state_history[i])[-self.state_history_window :]
            if len(prev_states) < self.state_history_window:
                prev_states = [np.zeros(self.state_dim, dtype=float)] * (
                    self.state_history_window - len(prev_states)
                ) + prev_states
            prev_states_flat: List[float] = []
            for state_vec in prev_states:
                prev_states_flat.extend([float(x) for x in state_vec])

            prev_throughputs = list(self.throughput_history)[-self.history_window :]
            if len(prev_throughputs) < self.history_window:
                prev_throughputs = [0.0] * (self.history_window - len(prev_throughputs)) + prev_throughputs

            quantized_state = self._quantize_state(self.plants[i].get_state())
            obs_values: List[float] = []
            obs_values.extend([float(x) for x in quantized_state])
            obs_values.append(float(throughput))
            obs_values.extend(prev_states_flat)
            obs_values.extend([float(x) for x in status_values])
            obs_values.extend([float(x) for x in prev_throughputs])

            observations[f"agent_{i}"] = np.array(obs_values, dtype=np.float32)

        self._update_history_buffers(throughput)
        return observations

    def _update_history_buffers(self, throughput: float) -> None:
        """Push current values so they appear in the 'previous k' slice next step."""
        self.throughput_history.append(float(throughput))
        for i in range(self.n_agents):
            quantized_state = self._quantize_state(self.plants[i].get_state())
            self.state_history[i].append(quantized_state)

    def _get_info(self) -> Dict[str, Any]:
        """Return auxiliary information."""
        collisions = (
            [0 for _ in range(self.n_agents)]
            if self.perfect_communication
            else [int(x) for x in self.network.collisions_per_agent]
        )
        return {
            "timestep": self.timestep,
            "channel_state": "PERFECT" if self.perfect_communication else self.network.channel_state.name,
            "states": [plant.get_state() for plant in self.plants],
            "estimates": [controller.x_hat.copy() for controller in self.controllers],
            "throughput_kbps": 0.0 if self.perfect_communication else self._compute_throughput(),
            "collided_packets": 0 if self.perfect_communication else self.network.total_collided_packets,
            "network_stats": {
                "tx_attempts": [int(x) for x in self.net_tx_attempts],
                "tx_acked": [int(x) for x in self.net_tx_acks],
                "tx_dropped": [int(x) for x in self.net_tx_drops],
                "tx_rewrites": [int(x) for x in self.net_tx_rewrites],
                "tx_collisions": collisions,
            },
            "reward_components": {k: v.copy() for k, v in self.last_reward_components.items()},
            "reward_mix_weight": float(self.last_mix_weight) if self.reward_mixing_enabled else 0.0,
            "termination_reasons": list(self.last_termination_reasons),
            "termination_agents": list(self.last_termination_agents),
            "bad_termination": bool(self.last_bad_termination),
        }

    def render(self):
        """Render environment (not implemented)."""
        pass

    def close(self):
        """Clean up resources."""
        pass

    def get_reward_mix_weight(self) -> float:
        """
        Return the current reward mixing weight (0.0 when mixing is disabled).

        This is useful for logging during evaluation.
        """
        return float(self.last_mix_weight) if hasattr(self, "last_mix_weight") else 0.0
