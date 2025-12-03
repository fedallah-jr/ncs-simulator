# Networked Control System Simulator

This project models multiple physical plants that share a single IEEE 802.15.4-style channel. Sensors decide each timestep whether to transmit their state, controllers run a Kalman-filtered LQR policy, and a CSMA/CA-inspired medium access layer arbitrates the shared network. The default behavior is configured through `default_config.json` in this directory.

## Simulation Parameters

The configuration file is divided into sections; each key controls a specific aspect of the simulation:

### `system`
- `A`, `B`: Discrete-time state and input matrices used by every plant and controller.
- `process_noise_cov`: Covariance of Gaussian process noise injected into each plant.
- `measurement_noise_cov`: Covariance of sensor noise added before packets are queued; the Kalman filter uses the same value.
- `initial_estimate_cov`: Initial covariance for each controller’s state estimate.
- `initial_state_scale_min` / `initial_state_scale_max`: Magnitude bounds for sampling each plant’s initial state. For each state dimension we draw a magnitude uniformly from `[min, max]` and apply a random sign (`x_i ~ s * U(min_i, max_i)`, `s ∈ {-1,1}`). Provide scalars to share bounds across dimensions or lists/matrices (flattened) matching the state dimension. When omitted, defaults to `[0.9, 1.0]`. Legacy `initial_state_scale` is still accepted as a symmetric bound (equivalent to `min=max=scale`).

### `lqr`
- `Q`, `R`: Cost matrices used to solve the discrete algebraic Riccati equation. They define how much the optimal controller cares about state deviation versus control effort.

### `reward`
- `state_cost_matrix`: Matrix (Q) used to compute the quadratic tracking error `(x - x_ref)^T Q (x - x_ref)`. Rewards are the improvement in this error between consecutive steps (`r_t = e_{t-1} - e_t`), minus the communication penalty.
- `state_error_reward`: Either `"difference"` (default, reward improvement), `"absolute"` (reward equals `-e_t`), or `"simple"` (reward +1 if a new measurement was delivered this step, otherwise 0; no communication penalty).
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

Learning-based baselines live under `algorithms/`:

- PPO (single agent, SB3): `python -m algorithms.ppo_single --config configs/perfect_comm.json --total-timesteps 200000`
- DQN (single agent, SB3): `python -m algorithms.deep_q_learning --config configs/perfect_comm.json --total-timesteps 200000`
- OpenAI-ES (single agent, JAX): `python -m algorithms.openai_es --config configs/perfect_comm.json --generations 1000 --popsize 128`
  - Enable a meta-population search over initializations with `--meta-population-size 4 --truncation-percentage 0.25 --pbt-interval 10`, which periodically copies top-performing strategies into the worst ones.

CLI flags let you change environment parameters. Use `--output-root` (defaults to `outputs/`) to control where training artifacts land. Each run calls `utils.run_utils.prepare_run_directory(...)`, which creates a uniquely named folder that encodes the algorithm and key config values (noise level, initial-state scale, reward mode, etc.) plus an incrementing `run#` suffix. That directory always contains:

- **Model Checkpoints:**
  - For SB3 (PPO/DQN): `best_model.zip`, `latest_model.zip`, and `evaluations.npz` (from `EvalCallback`).
  - For OpenAI-ES: `best_model.npz` (flattened params of best individual) and `latest_model.npz`.
- `training_rewards.csv`: A simple CSV table tracking performance. For SB3 this logs `[episode, reward]`; for OpenAI-ES it logs `[generation, mean_reward, max_reward, time]`.
- **`config.json`**, which combines the full environment configuration with a `training_run` section containing the algorithm name, timestamp, source config path, and all hyperparameters from the run. This structured format makes it easy to reload configurations or use them directly with visualization tools.

Configuration presets live under `configs/`. `configs/perfect_comm.json` mirrors the default plant/network settings but forces `network.perfect_communication` to `true`, which is useful for debugging algorithms without channel contention.

### Saved Configuration Format

The `config.json` file saved in each run directory preserves the complete environment configuration and adds a `training_run` section with metadata:

```json
{
  "system": { ... },
  "lqr": { ... },
  "reward": { ... },
  "observation": { ... },
  "network": { ... },
  "controller": { ... },
  "training_run": {
    "timestamp": "2025-11-20T12:34:56Z",
    "algorithm": "dqn",
    "source_config_path": "/path/to/original/config.json",
    "hyperparameters": {
      "total_timesteps": 200000,
      "episode_length": 1000,
      "learning_rate": 0.001,
      "gamma": 0.99,
      "batch_size": 64,
      ...
    }
  }
}
```

This format allows you to:
- Reproduce the exact run by loading `config.json`
- Pass it directly to visualization or analysis tools
- Track which config file was used as the base
- Compare hyperparameters across different runs programmatically
