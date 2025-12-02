from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .controller import Controller, compute_discrete_lqr_gain
from .config import load_config
from .network import NetworkModel
from .plant import Plant


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
    ):
        super().__init__()
        self.n_agents = n_agents
        self.episode_length = episode_length
        self.comm_cost = comm_cost

        self.config = load_config(config_path)
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

    def _initialize_systems(self):
        """Initialize plants, controllers, and the shared network."""
        A = self.A
        B = self.B
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
        self.K = compute_discrete_lqr_gain(A, B, lqr_Q, lqr_R)

        controller_cfg = self.config.get("controller", {})
        self.use_kalman_filter = bool(controller_cfg.get("use_kalman_filter", True))

        reward_cfg = self.config.get("reward", {})
        self.state_cost_matrix = np.array(
            reward_cfg.get("state_cost_matrix", np.eye(self.state_dim))
        )
        self.error_reward_mode = reward_cfg.get("state_error_reward", "difference")
        if self.error_reward_mode not in {"difference", "absolute"}:
            raise ValueError("state_error_reward must be 'difference' or 'absolute'")
        self.comm_recent_window = int(reward_cfg.get("comm_recent_window", self.history_window))
        self.comm_throughput_window = int(
            reward_cfg.get("comm_throughput_window", max(5 * self.history_window, 50))
        )
        self.comm_penalty_alpha = float(reward_cfg.get("comm_penalty_alpha", self.comm_cost))
        self.comm_throughput_floor = float(reward_cfg.get("comm_throughput_floor", 1e-3))

        self.plants: List[Plant] = []
        for _ in range(self.n_agents):
            x0 = self._sample_initial_state()
            # Pass self.np_random to Plant for isolated RNG
            self.plants.append(Plant(A, B, W, x0, rng=self.np_random))

        self.controllers: List[Controller] = []
        for _ in range(self.n_agents):
            initial_estimate = np.zeros(self.state_dim)
            self.controllers.append(
                Controller(
                    A,
                    B,
                    self.K,
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
            backoff_range=tuple(network_cfg.get("backoff_range", (0, 15))),
            max_queue_size=network_cfg.get("max_queue_size", 1),
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
        self.throughput_records: deque = deque()
        self.comm_success_records: List[deque] = [deque() for _ in range(self.n_agents)]
        self.last_measurements: List[np.ndarray] = [
            np.zeros(self.state_dim, dtype=float) for _ in range(self.n_agents)
        ]
        self.reward_component_stats: Dict[str, Dict[str, float]] = {
            f"agent_{i}": {"prev_error_sum": 0.0, "curr_error_sum": 0.0, "comm_penalty_sum": 0.0, "count": 0.0}
            for i in range(self.n_agents)
        }
        self.last_reward_components: Dict[str, Dict[str, float]] = {
            f"agent_{i}": {"prev_error": 0.0, "curr_error": 0.0, "comm_penalty": 0.0}
            for i in range(self.n_agents)
        }
        self.last_errors: List[float] = [0.0 for _ in range(self.n_agents)]

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
        if self.use_kalman_filter:
            for i in range(self.n_agents):
                self.controllers[i].store_prior(state_index)

        # Kalman filter measurement update (conditionally, based on packet delivery)
        # Note: predict() is called AFTER plant update to maintain correct timing
        delivered_controller_ids = set()

        if self.perfect_communication:
            for i in range(self.n_agents):
                action = actions[f"agent_{i}"]
                if action == 1:
                    state = self.plants[i].get_state()
                    measurement = state + self._sample_measurement_noise()
                    if self.use_kalman_filter:
                        self.controllers[i].update(measurement)
                    self.last_measurements[i] = measurement
                    self._record_decision(i, status=1)
                    # Use state_index for consistency (though _log_successful_comm
                    # returns early for perfect_communication anyway)
                    self._log_successful_comm(i, state_index, state_index)
                    delivered_controller_ids.add(i)
                else:
                    self._record_decision(i, status=0)
        else:
            # Advance network clock and process any packets that finished in the previous slot
            network_result = self.network.advance_time()
            ack_ready_indices: List[int] = []

            for packet in network_result["delivered_data"]:
                controller_id = packet.dest_id
                measurement = packet.payload["state"]
                measurement_timestamp = packet.payload["timestamp"]

                if self.use_kalman_filter:
                    # Use delayed_update to properly handle network delays
                    # This retrodicts to measurement time, updates, then predicts forward
                    self.controllers[controller_id].delayed_update(measurement, measurement_timestamp)
                self.last_measurements[controller_id] = measurement
                ack_data = {
                    "ack_timestamp": self.timestep,
                    "measurement_timestamp": measurement_timestamp,
                }
                overwritten_ack = self.network.queue_ack_packet(controller_id, ack_data)
                ack_ready_indices.append(self.n_agents + controller_id)
                # ACK packets being overwritten don't affect sensor status tracking
                # (they're from controller to sensor, status is tracked on sensor side)

            # Allow ACKs to seize the channel immediately after delivery
            ack_drops = self.network.attempt_transmissions(allowed_indices=ack_ready_indices)

            for packet in network_result["delivered_acks"]:
                sensor_id = packet.dest_id
                measurement_timestamp = packet.payload.get("measurement_timestamp")
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

            # Process dropped packets (from collisions in the prior timestep)
            for packet in network_result["dropped_packets"]:
                if packet.packet_type == "data":
                    # Data packet from sensor was dropped
                    sensor_id = packet.source_id
                    measurement_timestamp = packet.payload.get("timestamp")
                    entry = self.pending_transmissions[sensor_id].pop(measurement_timestamp, None)
                    if entry is not None:
                        entry["status"] = 3
                # ACK packets dropped don't need status tracking (they don't affect sensor decision history)

            # Drops that could occur if multiple ACKs collided (rare)
            for packet in ack_drops:
                if packet.packet_type == "data":
                    sensor_id = packet.source_id
                    measurement_timestamp = packet.payload.get("timestamp")
                    entry = self.pending_transmissions[sensor_id].pop(measurement_timestamp, None)
                    if entry is not None:
                        entry["status"] = 3

            for i in range(self.n_agents):
                action = actions[f"agent_{i}"]
                if action == 1:
                    state = self.plants[i].get_state()
                    measurement = state + self._sample_measurement_noise()
                    # Use state_index as the measurement timestamp
                    # This clearly indicates measurement is of x[state_index]
                    measurement_timestamp = state_index
                    overwritten = self.network.queue_data_packet(i, measurement, measurement_timestamp)
                    # Handle queue overwrite: mark the dropped packet as failed (status=3)
                    if overwritten is not None and overwritten.packet_type == "data":
                        dropped_timestamp = overwritten.payload.get("timestamp")
                        dropped_entry = self.pending_transmissions[i].pop(dropped_timestamp, None)
                        if dropped_entry is not None:
                            dropped_entry["status"] = 3
                    entry = self._record_decision(i, status=2)
                    entry["send_timestamp"] = measurement_timestamp
                    self.pending_transmissions[i][measurement_timestamp] = entry
                else:
                    self._record_decision(i, status=0)

            collision_drops = self.network.attempt_transmissions()
            for packet in collision_drops:
                if packet.packet_type == "data":
                    sensor_id = packet.source_id
                    measurement_timestamp = packet.payload.get("timestamp")
                    entry = self.pending_transmissions[sensor_id].pop(measurement_timestamp, None)
                    if entry is not None:
                        entry["status"] = 3
                # ACK packets dropped don't need status tracking (they don't affect sensor status tracking)

            delivered_controller_ids = {p.dest_id for p in network_result["delivered_data"]}

        # Compute control and update plants
        for i in range(self.n_agents):
            if self.use_kalman_filter:
                u = self.controllers[i].compute_control()
            else:
                u = -self.K @ self.last_measurements[i]
            self.plants[i].step(u)

        # Kalman filter time update (predict) - done AFTER plant update
        # This propagates estimates forward using dynamics and the control just applied
        # Prepares x_hat[k|k-1] for the next timestep's measurement update
        if self.use_kalman_filter:
            for i in range(self.n_agents):
                self.controllers[i].predict()

        rewards = {}
        for i in range(self.n_agents):
            x = self.plants[i].get_state()
            action = actions[f"agent_{i}"]
            prev_error = self.last_errors[i]
            curr_error = self._compute_state_error(x)
            comm_penalty = 0.0
            if not self.perfect_communication and action == 1:
                recent_tx = self._recent_transmission_count(i)
                throughput_estimate = self._compute_agent_throughput(i)
                comm_penalty = self.comm_penalty_alpha * (recent_tx / throughput_estimate)
            if self.error_reward_mode == "difference":
                error_reward = prev_error - curr_error
            elif self.error_reward_mode == "absolute":
                error_reward = -curr_error
            else:
                error_reward = 0.0
            reward = float(error_reward - comm_penalty)
            agent_key = f"agent_{i}"
            rewards[agent_key] = reward
            reward_components: Dict[str, float] = {
                "prev_error": float(prev_error),
                "curr_error": float(curr_error),
                "comm_penalty": float(comm_penalty),
            }
            self.last_reward_components[agent_key] = reward_components
            stats = self.reward_component_stats[agent_key]
            stats["prev_error_sum"] += float(prev_error)
            stats["curr_error_sum"] += float(curr_error)
            stats["comm_penalty_sum"] += float(comm_penalty)
            stats["count"] += 1
            self.last_errors[i] = curr_error

        terminated = {f"agent_{i}": False for i in range(self.n_agents)}
        truncated = {
            f"agent_{i}": self.timestep >= self.episode_length for i in range(self.n_agents)
        }

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
        window_start = self.timestep - self.throughput_window + 1
        while self.throughput_records and self.throughput_records[0]["timestamp"] < window_start:
            self.throughput_records.popleft()

        total_bits = sum(record["bits"] for record in self.throughput_records)
        window_duration = self.throughput_window * self.timestep_duration
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
        return {
            "timestep": self.timestep,
            "channel_state": "PERFECT" if self.perfect_communication else self.network.channel_state.name,
            "states": [plant.get_state() for plant in self.plants],
            "estimates": [
                controller.x_hat.copy() if self.use_kalman_filter else self.last_measurements[idx].copy()
                for idx, controller in enumerate(self.controllers)
            ],
            "throughput_kbps": 0.0 if self.perfect_communication else self._compute_throughput(),
            "collided_packets": 0 if self.perfect_communication else self.network.total_collided_packets,
            "reward_components": {k: v.copy() for k, v in self.last_reward_components.items()},
        }

    def render(self):
        """Render environment (not implemented)."""
        pass

    def close(self):
        """Clean up resources."""
        pass
