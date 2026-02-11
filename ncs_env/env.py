from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import islice
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .controller import (
    Controller,
    compute_discrete_lqr_solution,
    compute_finite_horizon_lqr_solution,
)
from .config import DEFAULT_CONFIG_PATH, load_config, resolve_measurement_noise_scale_range
from .network import NetworkModel
from .plant import Plant
from utils.reward_normalization import (
    RunningRewardNormalizer,
    get_shared_running_normalizer,
)


@dataclass
class RewardDefinition:
    mode: str
    comm_penalty_alpha: float
    normalize: bool = False  # Explicit flag; normalization is applied only when True.
    normalizer: Optional[RunningRewardNormalizer] = None
    no_normalization_scale: float = 1.0
    reward_clip_min: Optional[float] = None
    reward_clip_max: Optional[float] = None

    def __post_init__(self) -> None:
        if self.mode not in {
            "absolute",
            "estimation_error",
            "lqr_cost",
            "kf_info",
            "kf_info_s",
            "kf_info_m",
        }:
            raise ValueError(
                "state_error_reward must be 'absolute', 'estimation_error', "
                "'lqr_cost', 'kf_info', 'kf_info_s', or 'kf_info_m'"
            )
        if float(self.no_normalization_scale) <= 0.0:
            raise ValueError("no_normalization_scale must be > 0")
        if self.reward_clip_min is not None and self.reward_clip_max is not None:
            if float(self.reward_clip_min) > float(self.reward_clip_max):
                raise ValueError("reward_clip_min must be <= reward_clip_max")


def _should_normalize_reward(definition: RewardDefinition) -> bool:
    """
    Return whether reward normalization is enabled.

    Normalization is applied only when reward.normalize is explicitly set to True.
    """
    return bool(definition.normalize)


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
        config_path: Optional[str] = None,
        seed: Optional[int] = None,
        reward_override: Optional[Dict[str, Any]] = None,
        termination_override: Optional[Dict[str, Any]] = None,
        freeze_running_normalization: bool = False,
        track_true_goodput: bool = False,
        global_state_enabled: bool = False,
        track_lqr_cost: bool = False,
        track_eval_stats: bool = False,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.episode_length = episode_length
        self.reward_override = reward_override
        self.termination_override = termination_override
        self.freeze_running_normalization = bool(freeze_running_normalization)
        self.track_true_goodput = bool(track_true_goodput)
        self.global_state_enabled = bool(global_state_enabled)
        self.track_lqr_cost = bool(track_lqr_cost)
        self.track_eval_stats = bool(track_eval_stats)
        if self.global_state_enabled:
            self.track_true_goodput = True

        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.config = load_config(str(self.config_path))
        self.system_cfg = self.config.get("system", {})
        self.timestep_duration = 0.01
        self.timestep = 0
        self._init_rng: Optional[np.random.Generator] = None
        self._process_rng: Optional[np.random.Generator] = None
        self._measurement_rng: Optional[np.random.Generator] = None
        self._network_rng: Optional[np.random.Generator] = None

        self.A = np.array(self.system_cfg.get("A"))
        self.B = np.array(self.system_cfg.get("B"))
        self.state_dim = self.A.shape[0]
        self.control_dim = self.B.shape[1]
        self.initial_state_scale_min, self.initial_state_scale_max = self._resolve_initial_state_scale_range(
            self.system_cfg, self.state_dim
        )
        self.initial_state_fixed = bool(self.system_cfg.get("initial_state_fixed", False))
        self.initial_state_fixed_seed = self.system_cfg.get("initial_state_fixed_seed", None)
        if self.initial_state_fixed and self.initial_state_fixed_seed is None:
            self.initial_state_fixed_seed = 0
        self._fixed_initial_states: Optional[List[np.ndarray]] = None

        observation_cfg = self.config.get("observation", {})
        self.history_window = observation_cfg.get("history_window", 10)
        self.state_history_window = observation_cfg.get("state_history_window", self.history_window)
        tp_window = observation_cfg.get("throughput_window", 50)
        self.throughput_windows = [tp_window] if isinstance(tp_window, int) else list(tp_window)
        self.n_throughput_windows = len(self.throughput_windows)
        self.quantization_step = observation_cfg.get("quantization_step", 0.05)

        # Create a local RNG instance for this environment
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._reset_rng_streams()

        self.action_space = spaces.Dict(
            {f"agent_{i}": spaces.Discrete(2) for i in range(n_agents)}
        )

        obs_dim = (
            self.state_dim  # current state
            + self.n_throughput_windows  # current throughputs (one per window)
            + 1  # current measurement noise intensity
            + self.state_history_window * self.state_dim  # previous states
            + self.history_window  # previous statuses
            + self.history_window  # previous throughputs
        )
        self.obs_dim = int(obs_dim)
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
        self.network_trace_enabled = False
        self.network_trace_interval = 50
        self.network_trace_start = 1
        self.last_network_tick_trace: Optional[Dict[str, Any]] = None

        # Compute reward normalization statistics after systems are initialized
        self._setup_reward_normalization()

    def _initialize_systems(self):
        """Initialize plants, controllers, and the shared network."""
        agent_matrices = self._resolve_agent_system_matrices()
        A, B = agent_matrices[0]
        W = np.array(self.system_cfg.get("process_noise_cov", np.eye(self.state_dim)))
        self.process_noise_cov = W
        measurement_noise_scale_range = resolve_measurement_noise_scale_range(self.system_cfg)
        measurement_noise_cov_cfg = self.system_cfg.get("measurement_noise_cov", None)
        use_identity_default = (
            measurement_noise_scale_range is not None
            and "measurement_noise_cov" not in self.system_cfg
        )
        self.measurement_noise_cov = self._resolve_measurement_noise_covariance(
            measurement_noise_cov_cfg,
            self.state_dim,
            use_identity_default=use_identity_default,
        )
        self.measurement_noise_scale_range = measurement_noise_scale_range
        self._has_measurement_noise = (
            not np.allclose(self.measurement_noise_cov, 0.0)
            or (
                self.measurement_noise_scale_range is not None
                and self.measurement_noise_scale_range[1] > 0.0
            )
        )
        self.current_measurement_noise_covs = [
            self.measurement_noise_cov.copy() for _ in range(self.n_agents)
        ]
        base_intensity = self._compute_measurement_noise_intensity(self.measurement_noise_cov)
        self.current_measurement_noise_intensities = [
            base_intensity for _ in range(self.n_agents)
        ]
        initial_estimate_cov = np.array(
            self.system_cfg.get("initial_estimate_cov", np.eye(self.state_dim))
        )

        lqr_cfg = self.config.get("lqr", {})
        self.lqr_Q = np.array(lqr_cfg.get("Q", np.eye(self.state_dim)))
        self.lqr_R = np.array(lqr_cfg.get("R", np.eye(self.control_dim)))
        lqr_Q = self.lqr_Q
        lqr_R = self.lqr_R

        controller_cfg = self.config.get("controller", {})
        self.use_true_state_control = bool(controller_cfg.get("use_true_state_control", False))

        # Check if finite-horizon LQR is enabled (default: True)
        self.finite_horizon_enabled = bool(lqr_cfg.get("finite_horizon", True))
        self.K_list: List[Union[np.ndarray, List[np.ndarray]]] = []
        self.S_list: List[Union[np.ndarray, List[np.ndarray]]] = []
        self.M_list: List[Union[np.ndarray, List[np.ndarray]]] = []

        if self.finite_horizon_enabled:
            # Compute time-varying gains for finite horizon (uses self.episode_length)
            for (A_i, B_i) in agent_matrices:
                gains, costs = compute_finite_horizon_lqr_solution(
                    A_i, B_i, lqr_Q, lqr_R, self.episode_length
                )
                self.K_list.append(gains)
                self.S_list.append(costs)
                m_costs = []
                for t, K_t in enumerate(gains):
                    s_next = costs[t + 1] if t + 1 < len(costs) else lqr_Q
                    gain_weight = lqr_R + B_i.T @ s_next @ B_i
                    m_costs.append(K_t.T @ gain_weight @ K_t)
                self.M_list.append(m_costs)
        else:
            # Original infinite-horizon DARE solver
            for (A_i, B_i) in agent_matrices:
                gain, cost = compute_discrete_lqr_solution(A_i, B_i, lqr_Q, lqr_R)
                self.K_list.append(gain)
                self.S_list.append(cost)
                gain_weight = lqr_R + B_i.T @ cost @ B_i
                self.M_list.append(gain.T @ gain_weight @ gain)

        base_reward_cfg = self.config.get("reward", {})
        reward_cfg = self._merge_config_override(base_reward_cfg, self.reward_override)
        self.reward_normalization_gamma = float(reward_cfg.get("normalization_gamma", 0.99))
        self.state_cost_matrix = self.lqr_Q.copy()
        self.comm_recent_window = int(reward_cfg.get("comm_recent_window", self.history_window))
        self.comm_throughput_window = int(
            reward_cfg.get("comm_throughput_window", max(5 * self.history_window, 50))
        )
        base_comm_penalty_alpha = float(reward_cfg.get("comm_penalty_alpha", 0.0))
        self.reward_definition = self._build_reward_definition(
            reward_cfg,
            base_comm_penalty_alpha,
        )
        self.comm_throughput_floor = float(reward_cfg.get("comm_throughput_floor", 1e-3))
        self._running_reward_returns: Optional[List[float]] = None
        base_termination_cfg = self.config.get("termination", {})
        termination_cfg = self._merge_config_override(base_termination_cfg, self.termination_override)
        self.termination_enabled = bool(termination_cfg.get("enabled", False))
        self.termination_error_max = None
        if self.termination_enabled:
            self.termination_error_max = termination_cfg.get("state_error_max", None)
            if self.termination_error_max is None:
                raise ValueError("termination.enabled requires termination.state_error_max")
            self.termination_error_max = float(self.termination_error_max)
        self.termination_penalty = float(termination_cfg.get("penalty", 0.0))

        if self.initial_state_fixed:
            fixed_rng = np.random.default_rng(int(self.initial_state_fixed_seed))
            self._fixed_initial_states = [
                self._sample_initial_state(rng=fixed_rng) for _ in range(self.n_agents)
            ]

        self.plants: List[Plant] = []
        for i in range(self.n_agents):
            if self._fixed_initial_states is not None:
                x0 = self._fixed_initial_states[i].copy()
            else:
                x0 = self._sample_initial_state()
            A_i, B_i = agent_matrices[i]
            process_rng = self._process_rng if self._process_rng is not None else self.np_random
            self.plants.append(Plant(A_i, B_i, W, x0, rng=process_rng))

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
            slots_per_step=network_cfg.get("slots_per_step", 32),
            mac_min_be=network_cfg.get("mac_min_be", 3),
            mac_max_be=network_cfg.get("mac_max_be", 5),
            max_csma_backoffs=network_cfg.get("max_csma_backoffs", 4),
            max_frame_retries=network_cfg.get("max_frame_retries", 3),
            mac_ack_wait_us=network_cfg.get("mac_ack_wait_us", 864.0),
            mac_ack_turnaround_us=network_cfg.get("mac_ack_turnaround_us", 192.0),
            cca_time_us=network_cfg.get("cca_time_us", 128.0),
            mac_ack_size_bytes=network_cfg.get("mac_ack_size_bytes", 5),
            mac_ifs_sifs_us=network_cfg.get("mac_ifs_sifs_us", 192.0),
            mac_ifs_lifs_us=network_cfg.get("mac_ifs_lifs_us", 640.0),
            mac_ifs_max_sifs_frame_size=network_cfg.get("mac_ifs_max_sifs_frame_size", 18),
            tx_buffer_bytes=network_cfg.get("tx_buffer_bytes", 0),
            app_ack_enabled=network_cfg.get("app_ack_enabled", True),
            app_ack_packet_size=network_cfg.get("app_ack_packet_size", 30),
            app_ack_max_retries=network_cfg.get("app_ack_max_retries", 3),
            rng=self._network_rng if self._network_rng is not None else self.np_random,
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

        self.pending_transmissions: List[Dict[int, Dict[str, int]]] = [
            {} for _ in range(self.n_agents)
        ]
        self._last_cleanup_step = 0  # Track when we last cleaned up pending transmissions
        self.net_tx_attempts = np.zeros(self.n_agents, dtype=np.int64)
        self.net_tx_acks = np.zeros(self.n_agents, dtype=np.int64)
        self.net_tx_drops = np.zeros(self.n_agents, dtype=np.int64)
        self.net_tx_rewrites = np.zeros(self.n_agents, dtype=np.int64)
        self.observed_goodput_records: List[deque] = [deque() for _ in range(self.n_agents)]
        self.observed_goodput_seen: List[set] = [set() for _ in range(self.n_agents)]
        self.throughput_history: List[deque] = [
            deque([0.0 for _ in range(self.history_window)], maxlen=self.history_window)
            for _ in range(self.n_agents)
        ]
        if self.track_true_goodput:
            self.true_goodput_records: Optional[List[deque]] = [deque() for _ in range(self.n_agents)]
            self.true_goodput_seen: Optional[List[set]] = [set() for _ in range(self.n_agents)]
        else:
            self.true_goodput_records = None
            self.true_goodput_seen = None
        self.comm_success_records: List[deque] = [deque() for _ in range(self.n_agents)]
        self.last_measurements: List[np.ndarray] = [
            np.zeros(self.state_dim, dtype=float) for _ in range(self.n_agents)
        ]
        self.last_sensor_measurements: List[np.ndarray] = [
            np.zeros(self.state_dim, dtype=float) for _ in range(self.n_agents)
        ]
        self.last_lqr_costs: List[float] = [0.0 for _ in range(self.n_agents)]
        self.reward_component_stats: Dict[str, Dict[str, float]] = {
            f"agent_{i}": {"prev_error_sum": 0.0, "curr_error_sum": 0.0, "comm_penalty_sum": 0.0, "count": 0.0}
            for i in range(self.n_agents)
        }
        base_components: Dict[str, float] = {
            "prev_error": 0.0,
            "curr_error": 0.0,
            "comm_penalty": 0.0,
            "kf_info_gain": 0.0,
            "kf_info_gain_m": 0.0,
            "kf_info_gain_s": 0.0,
        }
        self.last_reward_components: Dict[str, Dict[str, float]] = {
            f"agent_{i}": dict(base_components)
            for i in range(self.n_agents)
        }
        self.last_errors: List[float] = [0.0 for _ in range(self.n_agents)]
        self.last_kf_info_gains: List[float] = [0.0 for _ in range(self.n_agents)]
        self.last_kf_info_gains_m: List[float] = [0.0 for _ in range(self.n_agents)]
        self.last_kf_info_gains_s: List[float] = [0.0 for _ in range(self.n_agents)]
        self.last_termination_reasons: List[str] = []
        self.last_termination_agents: List[int] = []
        self.last_bad_termination = False
        self.last_dropped_data_packets: List[Dict[str, Any]] = []

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment to initial state."""
        # This call updates self.np_random with a new seed if provided
        super().reset(seed=seed)

        self.timestep = 0
        self._reset_rng_streams()

        # Update RNG references in subsystems after super().reset()
        # This ensures all subsystems use the same isolated RNG stream
        for idx, plant in enumerate(self.plants):
            plant.rng = self._process_rng if self._process_rng is not None else self.np_random
            if self._fixed_initial_states is not None:
                x0 = self._fixed_initial_states[idx].copy()
            else:
                x0 = self._sample_initial_state()
            plant.reset(x0)

        for controller in self.controllers:
            controller.reset(np.zeros(self.state_dim))

        self.network.rng = self._network_rng if self._network_rng is not None else self.np_random
        self.network.reset()
        self._initialize_tracking_structures()
        self._resample_measurement_noise()
        self._update_sensor_measurements()
        self._reset_running_returns()
        self.last_network_tick_trace = None
        for idx in range(self.n_agents):
            self.last_measurements[idx] = self.plants[idx].get_state().copy()

        observations = self._get_observations()
        for idx in range(self.n_agents):
            self.last_errors[idx] = self._compute_reward_error(
                idx, self.plants[idx].get_state()
            )
        info = self._get_info()
        return observations, info

    def step(self, actions: Dict[str, int]):
        """Execute one timestep of the environment."""
        self.timestep += 1
        self.last_network_tick_trace = None
        if self.track_eval_stats:
            self.last_dropped_data_packets = []
        self.network.trace_enabled = bool(self.network_trace_enabled)

        # State index is the time index of the current plant state
        # After reset: timestep=0, state is x[0]
        # After first step: timestep=1, state is still x[0] before plant update
        # We use state_index to clearly refer to which x[k] we're measuring
        state_index = self.timestep - 1
        current_noise_covs = self.current_measurement_noise_covs
        for i in range(self.n_agents):
            self.last_kf_info_gains[i] = 0.0
            self.last_kf_info_gains_m[i] = 0.0
            self.last_kf_info_gains_s[i] = 0.0

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
                    measurement_noise_cov = current_noise_covs[i]
                    measurement = self.last_sensor_measurements[i]
                    self.controllers[i].update(measurement, measurement_noise_cov)
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
            trace_this_tick = self._should_trace_network_tick(self.timestep)
            if trace_this_tick:
                self.network.start_tick_trace(self.timestep)
            for i in range(self.n_agents):
                action = actions[f"agent_{i}"]
                if action == 1:
                    self.net_tx_attempts[i] += 1
                    measurement_noise_cov = current_noise_covs[i]
                    measurement = self.last_sensor_measurements[i]
                    # Use state_index as the measurement timestamp
                    # This clearly indicates measurement is of x[state_index]
                    measurement_timestamp = state_index
                    accepted, _ = self.network.queue_data_packet(
                        i,
                        measurement,
                        measurement_timestamp,
                        measurement_noise_cov=measurement_noise_cov,
                    )
                    if not accepted:
                        # Buffer full or MAC already busy: from transport layer perspective,
                        # packet was sent but will never get ACK (dropped at MAC layer).
                        self.net_tx_drops[i] += 1
                        entry = self._record_decision(i, status=2)
                        entry["send_timestamp"] = measurement_timestamp
                        # Still track as pending since we're waiting for ACK that won't come
                        self.pending_transmissions[i][measurement_timestamp] = entry
                        if self.track_eval_stats:
                            self.last_dropped_data_packets.append(
                                {
                                    "sensor_id": int(i),
                                    "measurement_timestamp": int(measurement_timestamp),
                                    "drop_timestep": int(self.timestep),
                                    "age_steps": 0,
                                    "reason": "queue_reject",
                                }
                            )
                        continue
                    entry = self._record_decision(i, status=2)
                    entry["send_timestamp"] = measurement_timestamp
                    self.pending_transmissions[i][measurement_timestamp] = entry
                else:
                    self._record_decision(i, status=0)
            # Run the micro-slot network for this environment step
            pending_delivered_packets: List[Any] = []
            slots_to_run = self.network.slots_per_step
            for _ in range(slots_to_run):
                network_result = self.network.run_slot()
                if network_result["delivered_data"]:
                    pending_delivered_packets.extend(network_result["delivered_data"])

                # Transport layer uses app-level ACKs to confirm delivery
                for app_ack in network_result.get("delivered_app_acks", []):
                    sensor_id = app_ack.get("sensor_id")
                    measurement_timestamp = app_ack.get("measurement_timestamp")
                    if sensor_id is None or measurement_timestamp is None:
                        continue
                    sensor_id = int(sensor_id)
                    measurement_timestamp = int(measurement_timestamp)
                    if 0 <= sensor_id < self.n_agents:
                        self.net_tx_acks[sensor_id] += 1
                        self._record_observed_goodput(sensor_id, measurement_timestamp)
                    entry = self.pending_transmissions[sensor_id].pop(measurement_timestamp, None)
                    if entry is not None:
                        entry["status"] = 1
                        self._log_successful_comm(sensor_id, measurement_timestamp, self.timestep)

                # Transport layer cannot detect dropped packets - they remain as status=2
                # (waiting for ACK that will never arrive)
                for packet in network_result["dropped_packets"]:
                    if packet.packet_type == "data":
                        sensor_id = packet.source_id
                        if 0 <= int(sensor_id) < self.n_agents:
                            self.net_tx_drops[int(sensor_id)] += 1
                        if self.track_eval_stats:
                            measurement_timestamp = state_index
                            if isinstance(packet.payload, dict):
                                measurement_timestamp = int(packet.payload.get("timestamp", state_index))
                            age_steps = max(0, int(state_index - measurement_timestamp))
                            self.last_dropped_data_packets.append(
                                {
                                    "sensor_id": int(sensor_id),
                                    "measurement_timestamp": int(measurement_timestamp),
                                    "drop_timestep": int(self.timestep),
                                    "age_steps": int(age_steps),
                                    "reason": "network_drop",
                                }
                            )
                        # Note: pending_transmissions entry remains with status=2

            if pending_delivered_packets:
                packets_by_controller: Dict[int, List[Any]] = {}
                for packet in pending_delivered_packets:
                    controller_id = int(packet.dest_id)
                    packets_by_controller.setdefault(controller_id, []).append(packet)

                for controller_id, packets in packets_by_controller.items():
                    packets.sort(
                        key=lambda packet: int(packet.payload.get("timestamp", 0)),
                    )
                    for packet in packets:
                        measurement = packet.payload["state"]
                        measurement_timestamp = packet.payload["timestamp"]
                        measurement_noise_cov = packet.payload.get("measurement_noise_cov")
                        applied = self.controllers[controller_id].delayed_update(
                            measurement, measurement_timestamp, measurement_noise_cov
                        )
                        if not applied:
                            continue
                        age_steps = max(0, int(state_index - int(measurement_timestamp)))

                        if self.track_true_goodput:
                            sensor_id = int(packet.source_id)
                            if 0 <= sensor_id < self.n_agents:
                                self._record_true_goodput(sensor_id, int(measurement_timestamp))

                        self.last_measurements[controller_id] = measurement
                        delivered_controller_ids.add(controller_id)
                        existing_age = delivered_message_ages.get(controller_id)
                        if existing_age is None or age_steps < existing_age:
                            delivered_message_ages[controller_id] = age_steps
            if trace_this_tick:
                self.last_network_tick_trace = self.network.finish_tick_trace()

            # Periodically clean up old pending transmissions to prevent unbounded memory growth
            if self.timestep - self._last_cleanup_step >= 100:
                self._cleanup_old_pending_transmissions(max_age=100)
                self._last_cleanup_step = self.timestep

        for i in range(self.n_agents):
            covariance = self.controllers[i].P
            m_gain = self._compute_kf_step_reward(i, covariance, mode="m")
            s_gain = self._compute_kf_step_reward(i, covariance, mode="s")
            self.last_kf_info_gains_m[i] = m_gain
            self.last_kf_info_gains_s[i] = s_gain
            self.last_kf_info_gains[i] = m_gain + s_gain

        # Compute control and update plants
        lqr_costs = None
        if self.track_lqr_cost or self.reward_definition.mode == "lqr_cost":
            lqr_costs = [0.0 for _ in range(self.n_agents)]
        for i in range(self.n_agents):
            if self.use_true_state_control:
                x = np.asarray(self.plants[i].get_state(), dtype=float).reshape(-1)
                u = self.controllers[i].compute_control_from_state(x)
            else:
                u = self.controllers[i].compute_control()
            if lqr_costs is not None:
                if not self.use_true_state_control:
                    x = np.asarray(self.plants[i].get_state(), dtype=float).reshape(-1)
                u_vec = np.asarray(u, dtype=float).reshape(-1)
                lqr_costs[i] = float(x @ self.lqr_Q @ x + u_vec @ self.lqr_R @ u_vec)
            self.plants[i].step(u)
        if lqr_costs is not None:
            self.last_lqr_costs = lqr_costs

        # Kalman filter time update (predict) - done AFTER plant update
        # This propagates estimates forward using dynamics and the control just applied
        # Prepares x_hat[k|k-1] for the next timestep's measurement update
        for i in range(self.n_agents):
            self.controllers[i].predict()

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
            reward_error = self._compute_reward_error(i, x)
            curr_error = reward_error
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

            info_gain = self._select_kf_info_gain(i, self.reward_definition.mode)
            components = self._compute_reward_for_definition(
                i,
                self.reward_definition,
                prev_error,
                curr_error,
                action,
                info_gain,
                info_arrived,
                message_age_steps,
            )
            components["kf_info_gain_m"] = float(self.last_kf_info_gains_m[i])
            components["kf_info_gain_s"] = float(self.last_kf_info_gains_s[i])
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
                    self._running_reward_returns[i] = 0.0

        # Get observations AFTER plant update (Gym step contract)
        # step() returns the resulting state s[k+1] after action a[k] is applied
        self._resample_measurement_noise()
        self._update_sensor_measurements()
        observations = self._get_observations()
        infos = self._get_info()
        return observations, rewards, terminated, truncated, infos

    def _should_trace_network_tick(self, tick: int) -> bool:
        if not self.network_trace_enabled:
            return False
        if self.network_trace_interval <= 0:
            return False
        if tick < self.network_trace_start:
            return False
        return (tick - self.network_trace_start) % self.network_trace_interval == 0

    def _record_decision(self, agent_idx: int, status: int) -> Dict[str, int]:
        """Append a new decision outcome to the history."""
        entry = {"timestamp": self.timestep, "status": status}
        self.decision_history[agent_idx].append(entry)
        return entry

    def _cleanup_old_pending_transmissions(self, max_age: int = 100) -> None:
        """
        Remove pending transmissions older than max_age steps.

        Since transport layer uses app ACKs and cannot detect MAC-layer drops,
        packets that fail at MAC layer remain in pending_transmissions forever.
        This cleanup prevents unbounded memory growth.

        Args:
            max_age: Maximum age in timesteps before considering a transmission stale
        """
        for agent_idx in range(self.n_agents):
            stale_timestamps = [
                ts for ts, entry in self.pending_transmissions[agent_idx].items()
                if self.timestep - entry.get("timestamp", self.timestep) > max_age
            ]
            for ts in stale_timestamps:
                self.pending_transmissions[agent_idx].pop(ts, None)

    def _sample_initial_state(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Sample an initial plant state with magnitude constrained to [scale_min, scale_max]
        per dimension. Sign is chosen uniformly.
        """
        if rng is None:
            rng = self._init_rng if self._init_rng is not None else self.np_random
        if rng is None:
            rng = np.random.default_rng()
        scale_min = self.initial_state_scale_min
        scale_max = self.initial_state_scale_max
        magnitudes = rng.uniform(low=scale_min, high=scale_max)
        signs = rng.choice([-1.0, 1.0], size=self.state_dim)
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
    def _resolve_measurement_noise_covariance(
        measurement_noise_cov_cfg: Any,
        state_dim: int,
        *,
        use_identity_default: bool,
    ) -> np.ndarray:
        """
        Resolve measurement noise covariance from config.

        If measurement_noise_cov is missing and use_identity_default is True, use I.
        Otherwise default to 0.01 * I.
        """
        if measurement_noise_cov_cfg is None:
            base_value = 1.0 if use_identity_default else 0.01
            return np.eye(state_dim) * base_value
        cov = np.asarray(measurement_noise_cov_cfg, dtype=float)
        if cov.ndim == 0:
            return np.eye(state_dim) * float(cov)
        if cov.shape != (state_dim, state_dim):
            raise ValueError(
                "measurement_noise_cov must be scalar or match the state dimension"
            )
        return cov

    @staticmethod
    def _compute_measurement_noise_intensity(measurement_noise_cov: np.ndarray) -> float:
        """Return a scalar intensity summary for a measurement noise covariance."""
        if measurement_noise_cov.ndim == 0:
            return float(measurement_noise_cov)
        if measurement_noise_cov.shape[0] == 0:
            return 0.0
        return float(np.trace(measurement_noise_cov) / float(measurement_noise_cov.shape[0]))

    @staticmethod
    def _merge_config_override(
        base_cfg: Dict[str, Any], override: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge an override dict onto a base config dict."""
        if override is None:
            return dict(base_cfg)
        merged = dict(base_cfg)
        merged.update(override)
        return merged

    def _build_reward_definition(
        self,
        reward_cfg: Dict[str, Any],
        base_comm_penalty_alpha: float,
    ) -> RewardDefinition:
        base_mode = reward_cfg.get("state_error_reward", "absolute")
        base_normalize = bool(reward_cfg.get("normalize", False))
        no_normalization_scale = reward_cfg.get("no_normalization_scale", 1.0)
        if no_normalization_scale is None:
            no_normalization_scale = 1.0
        no_normalization_scale = float(no_normalization_scale)
        reward_clip_min = reward_cfg.get("reward_clip_min", None)
        reward_clip_max = reward_cfg.get("reward_clip_max", None)
        if reward_clip_min is not None:
            reward_clip_min = float(reward_clip_min)
        if reward_clip_max is not None:
            reward_clip_max = float(reward_clip_max)
        return RewardDefinition(
            mode=str(base_mode),
            comm_penalty_alpha=base_comm_penalty_alpha,
            normalize=base_normalize,
            normalizer=None,
            no_normalization_scale=no_normalization_scale,
            reward_clip_min=reward_clip_min,
            reward_clip_max=reward_clip_max,
        )

    def _quantize_state(self, state: np.ndarray) -> np.ndarray:
        """Quantize the measured plant state."""
        if self.quantization_step is None or self.quantization_step <= 0:
            return state
        step = self.quantization_step
        return np.round(state / step) * step

    def _record_observed_goodput(self, agent_idx: int, measurement_timestamp: int) -> None:
        if not 0 <= agent_idx < self.n_agents:
            return
        seen = self.observed_goodput_seen[agent_idx]
        if measurement_timestamp in seen:
            return
        seen.add(measurement_timestamp)
        self.observed_goodput_records[agent_idx].append(
            {
                "timestamp": self.timestep,
                "bits": self.network.data_packet_size * 8,
            }
        )

    def _record_true_goodput(self, agent_idx: int, measurement_timestamp: int) -> None:
        if not self.track_true_goodput:
            return
        if not 0 <= agent_idx < self.n_agents:
            return
        records = self.true_goodput_records
        seen = self.true_goodput_seen
        if records is None or seen is None:
            return
        agent_seen = seen[agent_idx]
        if measurement_timestamp in agent_seen:
            return
        agent_seen.add(measurement_timestamp)
        records[agent_idx].append(
            {
                "timestamp": self.timestep,
                "bits": self.network.data_packet_size * 8,
            }
        )

    def _compute_agent_goodput_kbps(self, records: deque, window: int, cleanup: bool = True) -> float:
        """Compute goodput (kbps) over the sliding window for one agent."""
        if self.perfect_communication:
            return 0.0
        if self.timestep <= 0:
            return 0.0
        window_start = self.timestep - window + 1

        # Cleanup old records only if requested
        if cleanup:
            while records and records[0]["timestamp"] < window_start:
                records.popleft()

        # Sum bits in window
        total_bits = sum(
            record["bits"] for record in records
            if record["timestamp"] >= window_start
        )
        effective_steps = min(window, self.timestep)
        window_duration = effective_steps * self.timestep_duration
        if window_duration == 0:
            return 0.0
        throughput_kbps = (total_bits / window_duration) / 1000.0
        return throughput_kbps

    def _compute_observed_goodput_kbps(self, agent_idx: int) -> float:
        """Compute ACK-observed goodput (kbps) for a single agent (first window)."""
        return self._compute_agent_goodput_kbps(
            self.observed_goodput_records[agent_idx], self.throughput_windows[0]
        )

    def _compute_observed_goodput_kbps_multi(self, agent_idx: int) -> List[float]:
        """Compute observed goodput for multiple window sizes."""
        records = self.observed_goodput_records[agent_idx]
        # Sort windows largest to smallest for cleanup efficiency
        sorted_windows = sorted(self.throughput_windows, reverse=True)
        throughputs = []
        for i, window in enumerate(sorted_windows):
            # Only cleanup on the largest window (first iteration)
            tp = self._compute_agent_goodput_kbps(records, window, cleanup=(i == 0))
            throughputs.append(tp)

        # Reorder to match original throughput_windows order
        if sorted_windows != self.throughput_windows:
            order_map = {w: tp for w, tp in zip(sorted_windows, throughputs)}
            throughputs = [order_map[w] for w in self.throughput_windows]
        return throughputs

    def _compute_true_goodput_kbps(self, agent_idx: int) -> float:
        """Compute delivered goodput (kbps) for a single agent."""
        if not self.track_true_goodput:
            return 0.0
        records = self.true_goodput_records
        if records is None:
            return 0.0
        return self._compute_agent_goodput_kbps(records[agent_idx], self.throughput_windows[0])

    def _compute_throughput(self) -> float:
        """Compute total ACK-observed goodput (kbps) over the sliding window."""
        return float(sum(self._compute_observed_goodput_kbps(i) for i in range(self.n_agents)))

    def _resample_measurement_noise(self) -> None:
        """Sample the per-agent measurement noise covariance for the next step."""
        rng = self._measurement_rng if self._measurement_rng is not None else self.np_random
        if rng is None:
            rng = np.random.default_rng()
        if self.measurement_noise_scale_range is None:
            base_cov = self.measurement_noise_cov
            intensity = self._compute_measurement_noise_intensity(base_cov)
            self.current_measurement_noise_covs = [
                base_cov.copy() for _ in range(self.n_agents)
            ]
            self.current_measurement_noise_intensities = [
                intensity for _ in range(self.n_agents)
            ]
            return

        min_scale, max_scale = self.measurement_noise_scale_range
        scales = rng.uniform(low=min_scale, high=max_scale, size=self.n_agents)
        covs: List[np.ndarray] = []
        intensities: List[float] = []
        for scale in scales:
            cov = self.measurement_noise_cov * float(scale)
            covs.append(cov)
            intensities.append(self._compute_measurement_noise_intensity(cov))
        self.current_measurement_noise_covs = covs
        self.current_measurement_noise_intensities = intensities

    def _sample_measurement_noise(self, measurement_noise_cov: np.ndarray) -> np.ndarray:
        """Return measurement noise vector (zero when disabled)."""
        if not self._has_measurement_noise:
            return np.zeros(self.state_dim)
        rng = self._measurement_rng if self._measurement_rng is not None else self.np_random
        if rng is None:
            rng = np.random.default_rng()
        return rng.multivariate_normal(
            np.zeros(self.state_dim), measurement_noise_cov
        )

    def _reset_rng_streams(self) -> None:
        """Seed per-episode RNG streams from the global environment RNG."""
        if self.np_random is None:
            self._init_rng = None
            self._process_rng = None
            self._measurement_rng = None
            self._network_rng = None
            return
        seeds = self.np_random.integers(0, 2**32 - 1, size=4, dtype=np.uint32)
        self._init_rng = np.random.default_rng(int(seeds[0]))
        self._process_rng = np.random.default_rng(int(seeds[1]))
        self._measurement_rng = np.random.default_rng(int(seeds[2]))
        self._network_rng = np.random.default_rng(int(seeds[3]))

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

    def _get_kf_info_matrix(self, agent_idx: int) -> np.ndarray:
        info_entry = self.M_list[agent_idx]
        if isinstance(info_entry, list):
            k = min(self.controllers[agent_idx].time_step, len(info_entry) - 1)
            return info_entry[k]
        return info_entry

    def _get_kf_state_cost_matrix(self, agent_idx: int) -> np.ndarray:
        state_entry = self.S_list[agent_idx]
        if isinstance(state_entry, list):
            k = min(self.controllers[agent_idx].time_step, len(state_entry) - 1)
            return state_entry[k]
        return state_entry

    def _compute_kf_step_reward(
        self,
        agent_idx: int,
        covariance: np.ndarray,
        *,
        mode: str = "combined",
    ) -> float:
        info_weight = self._get_kf_info_matrix(agent_idx)
        state_weight = self._get_kf_state_cost_matrix(agent_idx)
        if mode == "m":
            weight = info_weight
        elif mode == "s":
            weight = state_weight
        elif mode == "combined":
            weight = info_weight + state_weight
        else:
            raise ValueError(f"Unsupported kf_info mode: {mode}")
        return -float(np.trace(weight @ covariance))

    def _select_kf_info_gain(self, agent_idx: int, mode: str) -> float:
        if mode == "kf_info_m":
            return self.last_kf_info_gains_m[agent_idx]
        if mode == "kf_info_s":
            return self.last_kf_info_gains_s[agent_idx]
        if mode == "kf_info":
            return self.last_kf_info_gains[agent_idx]
        return 0.0

    def _compute_estimation_error(self, agent_idx: int, state: np.ndarray) -> float:
        """Weighted L1 estimation error ||Q (x - x_hat)||_1."""
        estimate = self.controllers[agent_idx].x_hat
        dx = state - estimate
        weighted = self.state_cost_matrix @ dx
        return float(np.sum(np.abs(weighted)))

    def _compute_reward_error(self, agent_idx: int, state: np.ndarray) -> float:
        """Return the error used by the selected reward mode."""
        if self.reward_definition.mode == "estimation_error":
            return self._compute_estimation_error(agent_idx, state)
        if self.reward_definition.mode == "lqr_cost":
            return float(self.last_lqr_costs[agent_idx])
        if self.reward_definition.mode in {"kf_info", "kf_info_s", "kf_info_m"}:
            return 0.0
        return self._compute_state_error(state)

    def _recent_transmission_count(self, agent_idx: int) -> int:
        """Count transmissions (ACKed or pending) over the short window."""
        dh = self.decision_history[agent_idx]
        recent_entries = islice(dh, max(0, len(dh) - self.comm_recent_window), None)
        return sum(1 for entry in recent_entries if entry["status"] > 0)

    def _setup_reward_normalization(self) -> None:
        if not _should_normalize_reward(self.reward_definition):
            return
        self._init_running_reward_normalizer()

    def _running_normalizer_key(self, definition: RewardDefinition) -> str:
        signature = {
            "config_path": str(self.config_path.resolve()),
            "mode": definition.mode,
            "normalization_gamma": float(self.reward_normalization_gamma),
            "comm_penalty_alpha": float(definition.comm_penalty_alpha),
            "state_cost_matrix": np.asarray(self.state_cost_matrix).tolist(),
            "comm_recent_window": int(self.comm_recent_window),
            "comm_throughput_window": int(self.comm_throughput_window),
            "comm_throughput_floor": float(self.comm_throughput_floor),
        }
        return json.dumps(signature, sort_keys=True, separators=(",", ":"))

    def _init_running_reward_normalizer(self) -> None:
        self._running_reward_returns = [0.0 for _ in range(self.n_agents)]
        if _should_normalize_reward(self.reward_definition):
            key = self._running_normalizer_key(self.reward_definition)
            self.reward_definition.normalizer = get_shared_running_normalizer(key)

    def _reset_running_returns(self) -> None:
        if self._running_reward_returns is None:
            return
        for i in range(self.n_agents):
            self._running_reward_returns[i] = 0.0

    def _apply_running_normalization(
        self,
        agent_idx: int,
        reward_value: float,
    ) -> float:
        if self._running_reward_returns is None:
            return reward_value
        self._running_reward_returns[agent_idx] = (
            self.reward_normalization_gamma * self._running_reward_returns[agent_idx] + reward_value
        )
        normalizer = self.reward_definition.normalizer
        if isinstance(normalizer, RunningRewardNormalizer):
            if not self.freeze_running_normalization:
                normalizer.update(self._running_reward_returns[agent_idx])
            return normalizer(reward_value)
        return reward_value

    def _clip_reward_value(self, reward_value: float) -> float:
        min_value = self.reward_definition.reward_clip_min
        max_value = self.reward_definition.reward_clip_max
        if min_value is not None and reward_value < min_value:
            reward_value = float(min_value)
        if max_value is not None and reward_value > max_value:
            reward_value = float(max_value)
        return reward_value

    def _normalize_reward_value(self, agent_idx: int, reward_value: float) -> float:
        definition = self.reward_definition
        if isinstance(definition.normalizer, RunningRewardNormalizer):
            return self._clip_reward_value(
                self._apply_running_normalization(agent_idx, reward_value)
            )
        if definition.normalizer is not None:
            return self._clip_reward_value(definition.normalizer(reward_value))
        scale = float(definition.no_normalization_scale)
        if scale != 1.0:
            reward_value = reward_value / scale
        return self._clip_reward_value(reward_value)

    def _compute_reward_for_definition(
        self,
        agent_idx: int,
        definition: RewardDefinition,
        prev_error: float,
        curr_error: float,
        action: int,
        info_gain: float,
        info_arrived: bool,
        message_age_steps: Optional[int] = None,
        apply_normalization: bool = True,
    ) -> Dict[str, float]:
        """Compute reward and components for the given reward definition."""
        comm_penalty = 0.0
        if not self.perfect_communication and action == 1:
            penalty_alpha = definition.comm_penalty_alpha
            if penalty_alpha > 0:
                recent_tx = self._recent_transmission_count(agent_idx)
                throughput_estimate = self._compute_agent_throughput(agent_idx)
                comm_penalty = penalty_alpha * (recent_tx / throughput_estimate)

        if definition.mode == "absolute":
            error_reward = -curr_error
        elif definition.mode == "estimation_error":
            error_reward = -curr_error
        elif definition.mode == "lqr_cost":
            error_reward = -curr_error
        elif definition.mode in {"kf_info", "kf_info_s", "kf_info_m"}:
            error_reward = float(info_gain)
        else:
            raise ValueError(f"Unsupported reward mode: {definition.mode}")

        reward_value = float(error_reward - comm_penalty)

        if apply_normalization:
            reward_value = self._normalize_reward_value(agent_idx, reward_value)
        components: Dict[str, float] = {
            "prev_error": float(prev_error),
            "curr_error": float(curr_error),
            "comm_penalty": float(comm_penalty),
            "kf_info_gain": float(info_gain),
            "reward": reward_value,
        }
        if definition.mode == "lqr_cost":
            components["lqr_cost"] = float(curr_error)
        return components

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Construct observations for all agents."""
        observations = {}
        current_throughputs: List[float] = []
        quantized_states: List[np.ndarray] = []

        for i in range(self.n_agents):
            # decision_history maxlen may be larger than history_window, so use islice
            dh = self.decision_history[i]
            status_iter = (
                entry["status"]
                for entry in islice(dh, max(0, len(dh) - self.history_window), None)
            )
            status_values = np.fromiter(
                status_iter, dtype=np.float32, count=self.history_window
            )

            # state_history has maxlen=state_history_window and is pre-filled, always full
            prev_states = np.asarray(self.state_history[i], dtype=np.float32)
            prev_states_flat = prev_states.reshape(-1)

            # throughput_history has maxlen=history_window and is pre-filled, always full
            prev_throughputs = np.asarray(self.throughput_history[i], dtype=np.float32)

            throughputs = self._compute_observed_goodput_kbps_multi(i)
            measurement = self.last_sensor_measurements[i]
            quantized_state = self._quantize_state(measurement)
            obs_values = np.empty(self.obs_dim, dtype=np.float32)
            cursor = 0
            obs_values[cursor : cursor + self.state_dim] = quantized_state
            cursor += self.state_dim
            obs_values[cursor : cursor + self.n_throughput_windows] = throughputs
            cursor += self.n_throughput_windows
            obs_values[cursor] = float(self.current_measurement_noise_intensities[i])
            cursor += 1
            obs_values[cursor : cursor + prev_states_flat.size] = prev_states_flat
            cursor += prev_states_flat.size
            obs_values[cursor : cursor + self.history_window] = status_values
            cursor += self.history_window
            obs_values[cursor : cursor + self.history_window] = prev_throughputs

            observations[f"agent_{i}"] = obs_values
            current_throughputs.append(throughputs[0])  # Use first window for history
            quantized_states.append(quantized_state)

        self._update_history_buffers(current_throughputs, quantized_states)
        return observations

    def _update_history_buffers(
        self,
        throughputs: List[float],
        quantized_states: Optional[List[np.ndarray]] = None,
    ) -> None:
        """Push current values so they appear in the 'previous k' slice next step."""
        for i in range(self.n_agents):
            if i < len(throughputs):
                self.throughput_history[i].append(float(throughputs[i]))
            if quantized_states is not None and i < len(quantized_states):
                self.state_history[i].append(quantized_states[i])
                continue
            quantized_state = self._quantize_state(self.plants[i].get_state())
            self.state_history[i].append(quantized_state)

    def _update_sensor_measurements(self) -> None:
        measurements: List[np.ndarray] = []
        for i in range(self.n_agents):
            state = self.plants[i].get_state()
            measurement_noise_cov = self.current_measurement_noise_covs[i]
            if self._has_measurement_noise:
                measurement = state + self._sample_measurement_noise(measurement_noise_cov)
            else:
                measurement = state
            measurements.append(measurement)
        self.last_sensor_measurements = measurements

    def _get_global_state(self) -> np.ndarray:
        noise_intensity = np.asarray(self.current_measurement_noise_intensities, dtype=np.float32)
        perceived_goodput = np.asarray(
            [self._compute_observed_goodput_kbps(i) for i in range(self.n_agents)],
            dtype=np.float32,
        )
        true_goodput = np.asarray(
            [self._compute_true_goodput_kbps(i) for i in range(self.n_agents)],
            dtype=np.float32,
        )
        channel_state = np.asarray([float(self.network.channel_state.value)], dtype=np.float32)
        backoff_stage = np.zeros(self.n_agents, dtype=np.float32)
        backoff_remaining = np.zeros(self.n_agents, dtype=np.float32)
        for i in range(self.n_agents):
            entity = self.network.entities[i]
            backoff_stage[i] = float(entity.csma_backoffs + (1 if entity.pending_packet is not None else 0))
            backoff_remaining[i] = float(entity.backoff_counter)
        per_agent_features: List[np.ndarray] = []
        for i in range(self.n_agents):
            per_agent_features.append(
                np.concatenate(
                    [
                        self.plants[i].get_state().astype(np.float32),
                        self.controllers[i].x_hat.astype(np.float32),
                        self.last_sensor_measurements[i].astype(np.float32),
                        np.asarray(
                            [
                                noise_intensity[i],
                                perceived_goodput[i],
                                true_goodput[i],
                                backoff_stage[i],
                                backoff_remaining[i],
                            ],
                            dtype=np.float32,
                        ),
                    ]
                )
            )
        return np.concatenate(per_agent_features + [channel_state])

    def _get_info(self) -> Dict[str, Any]:
        """Return auxiliary information."""
        collisions = (
            [0 for _ in range(self.n_agents)]
            if self.perfect_communication
            else [int(x) for x in self.network.collisions_per_agent]
        )
        data_delivered = (
            [int(x) for x in self.net_tx_acks]
            if self.perfect_communication
            else [int(x) for x in self.network.data_delivered_per_agent]
        )
        mac_ack_sent = (
            [0 for _ in range(self.n_agents)]
            if self.perfect_communication
            else [int(x) for x in self.network.mac_ack_sent_per_agent]
        )
        mac_ack_collisions = (
            [0 for _ in range(self.n_agents)]
            if self.perfect_communication
            else [int(x) for x in self.network.mac_ack_collisions_per_agent]
        )
        ack_timeouts = (
            [0 for _ in range(self.n_agents)]
            if self.perfect_communication
            else [int(x) for x in self.network.ack_timeouts_per_agent]
        )
        app_ack_sent = (
            [0 for _ in range(self.n_agents)]
            if self.perfect_communication
            else [int(x) for x in self.network.app_ack_sent_per_agent]
        )
        app_ack_collisions = (
            [0 for _ in range(self.n_agents)]
            if self.perfect_communication
            else [int(x) for x in self.network.app_ack_collisions_per_agent]
        )
        app_ack_drops = (
            [0 for _ in range(self.n_agents)]
            if self.perfect_communication
            else [int(x) for x in self.network.app_ack_drops_per_agent]
        )
        app_ack_delivered = (
            [0 for _ in range(self.n_agents)]
            if self.perfect_communication
            else [int(x) for x in self.network.app_ack_delivered_per_agent]
        )
        info = {
            "timestep": self.timestep,
            "channel_state": "PERFECT" if self.perfect_communication else self.network.channel_state.name,
            "states": [plant.get_state() for plant in self.plants],
            "estimates": [controller.x_hat.copy() for controller in self.controllers],
            "estimate_covariances": [controller.P.copy() for controller in self.controllers],
            "throughput_kbps": 0.0 if self.perfect_communication else self._compute_throughput(),
            "collided_packets": 0 if self.perfect_communication else self.network.total_collided_packets,
            "network_stats": {
                "tx_attempts": [int(x) for x in self.net_tx_attempts],
                "tx_acked": [int(x) for x in self.net_tx_acks],
                "tx_dropped": [int(x) for x in self.net_tx_drops],
                "tx_rewrites": [int(x) for x in self.net_tx_rewrites],
                "tx_collisions": collisions,
                "data_delivered": data_delivered,
                "mac_ack_sent": mac_ack_sent,
                "mac_ack_collisions": mac_ack_collisions,
                "ack_timeouts": ack_timeouts,
                "app_ack_sent": app_ack_sent,
                "app_ack_collisions": app_ack_collisions,
                "app_ack_drops": app_ack_drops,
                "app_ack_delivered": app_ack_delivered,
            },
            "reward_components": {k: v.copy() for k, v in self.last_reward_components.items()},
            "termination_reasons": list(self.last_termination_reasons),
            "termination_agents": list(self.last_termination_agents),
            "bad_termination": bool(self.last_bad_termination),
        }
        if self.track_eval_stats:
            info["dropped_data_packets_step"] = [dict(entry) for entry in self.last_dropped_data_packets]
        if self.track_lqr_cost:
            info["lqr_costs"] = [float(cost) for cost in self.last_lqr_costs]
            info["lqr_cost_total"] = float(sum(self.last_lqr_costs))
        if self.track_true_goodput:
            per_agent_goodput = [self._compute_true_goodput_kbps(i) for i in range(self.n_agents)]
            info["true_goodput_kbps_per_agent"] = per_agent_goodput
            info["true_goodput_kbps_total"] = float(sum(per_agent_goodput))
        if self.global_state_enabled:
            info["global_state"] = self._get_global_state()
        if self.last_network_tick_trace is not None:
            info["network_tick_trace"] = self.last_network_tick_trace
        return info

    def render(self):
        """Render environment (not implemented)."""
        pass

    def close(self):
        """Clean up resources."""
        pass
