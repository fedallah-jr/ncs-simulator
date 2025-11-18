# Networked Control System Simulator

This project models multiple physical plants that share a single IEEE 802.15.4-style channel. Sensors decide each timestep whether to transmit their state, controllers run a Kalman-filtered LQR policy, and a CSMA/CA-inspired medium access layer arbitrates the shared network. The default behavior is configured through `default_config.json` in this directory.

## Simulation Parameters

The configuration file is divided into sections; each key controls a specific aspect of the simulation:

### `system`
- `A`, `B`: Discrete-time state and input matrices used by every plant and controller.
- `process_noise_cov`: Covariance of Gaussian process noise injected into each plant.
- `measurement_noise_cov`: Covariance of sensor noise added before packets are queued; the Kalman filter uses the same value.
- `initial_estimate_cov`: Initial covariance for each controller’s state estimate.
- `initial_state_scale`: Standard-deviation multiplier for sampling each plant’s initial state.

### `lqr`
- `Q`, `R`: Cost matrices used to solve the discrete algebraic Riccati equation. They define how much the optimal controller cares about state deviation versus control effort.

### `reward`
- `state_cost_matrix`: Matrix (Q) used to compute the quadratic tracking error `(x - x_ref)^T Q (x - x_ref)`. Rewards are the improvement in this error between consecutive steps (`r_t = e_{t-1} - e_t`), minus the communication penalty.
- `state_error_reward`: Either `"difference"` (default, reward improvement) or `"absolute"` (reward equals `-e_t`). Use the second option if you want to revert to the original “distance to origin” objective.
- `comm_recent_window`: Short window (steps) used to count how many recent transmission attempts (`p>0`) an agent has initiated.
- `comm_throughput_window`: Long window (steps) used to estimate per-agent throughput from ACKed packets and their delays.
- `comm_penalty_alpha`: Scalar multiplier (`α`) used in the communication penalty `R_{a,\text{comm}} = -α * N_\text{recent}/T`, applied only when `action=1` and the network is not set to `perfect_communication`.
- `comm_throughput_floor`: Small positive value to keep the throughput estimate from collapsing to zero when no ACKs have been observed recently.

With these fields the environment discourages bursts of transmissions in congested conditions: if an agent spams the channel (large `N_recent_tx`) while the measured throughput is low, the penalty grows rapidly; skipping a send (`action=0`) adds no communication cost. When `perfect_communication=true`, the penalty logic is bypassed entirely.

### `observation`
- `history_window`: Number of past steps included for statuses and throughput history.
- `state_history_window`: Number of past states appended to the observation (defaults to `history_window` if omitted).
- `throughput_window`: Sliding window (in steps) used to estimate recent channel throughput (kbps).
- `quantization_step`: Step size for quantizing plant states before they appear in observations. Set to `0` or omit to disable quantization.

Observations are laid out as `[current_state, current_throughput, prev_states..., prev_statuses..., prev_throughputs...]`, where each “prev” block holds `history_window` (or `state_history_window` for states) entries.

### `network`
- `data_rate_kbps`: Physical-layer rate used to convert packet sizes into transmission durations.
- `data_packet_size`, `ack_packet_size`: Sensor/ACK payload sizes in bytes. Larger packets hold the channel for more timesteps.
- `backoff_range`: `(min, max)` slot range for random backoff; the upper bound grows exponentially after collisions.
- `max_queue_size`: Pending-packet capacity per entity. A value of `1` means new data overwrites unsent packets, enforcing “freshest data only.”
- `perfect_communication`: When `true`, disables CSMA behavior entirely—measurements reach controllers instantly, collisions/throughput bookkeeping is skipped, and decision histories record immediate successes for every transmission attempt.

Adjusting these parameters lets you explore different plant dynamics, estimator fidelity, reward shaping, or network congestion levels without modifying the source code.

## Algorithms

Learning-based baselines live under `algorithms/`. To try the tabular independent Q-learning trainer:

```bash
source ~/.venv/bin/activate
python -m algorithms.independent_q_learning \
    --config default_config.json \
    --episodes 100 \
    --episode-length 500
```

Neural baselines:

- PPO (single agent, SB3): `python -m algorithms.ppo_single --config configs/perfect_comm.json --total-timesteps 200000`
- DQN (single agent, SB3): `python -m algorithms.deep_q_learning --config configs/perfect_comm.json --total-timesteps 200000`

CLI flags let you change discretization granularity, exploration schedule, and environment parameters. Pass `--stats-output path/to/file.csv` to persist the learning curve (per-episode totals) and `--trajectory-output path/to/file.csv` to store every per-step `(episode, step, agent_id, action, reward)` entry the trainer observes. The trainer spins up `ncs_env.env.NCS_Env`, discretizes each agent observation (`history`, throughput estimate, quantized state), and updates a per-agent Q-table with epsilon-greedy exploration. Use this as a sanity-check baseline or starting point for more advanced algorithms.

Discretization logic and other reusable helpers live under `utils/`. For example, `utils.discretization.ObservationDiscretizer` is shared between algorithms so new learners can import it without duplicating binning code.

Configuration presets live under `configs/`. `configs/perfect_comm.json` mirrors the default plant/network settings but forces `network.perfect_communication` to `true`, which is useful for debugging algorithms without channel contention.
