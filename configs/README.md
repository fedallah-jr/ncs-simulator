# Configuration Guide

This directory contains the simulator input configs. These files define environment behavior such as plant dynamics, observation layout, reward shaping, network settings, and evaluation defaults.

Training hyperparameters are not read from these input config files. Algorithm flags such as `--learning-rate`, `--double-q`, and `--optimizer` are provided on the CLI and then recorded in the saved run `config.json` under `outputs/`.

## Files In This Directory

- `marl_absolute_plants.json`: 3-agent input config with per-agent plant dynamics, `kf_info_m` training reward, and no random channel drops.
- `marl_absolute_plants_hetero.json`: 3-agent heterogeneous variant with `lqr_cost` reward and per-agent random channel drop rates.

## Input Config Sections

The input config file is divided into sections; each key controls a specific aspect of the simulation.

### `system`

- `A`, `B`: Discrete-time state and input matrices used by every plant and controller.
- `process_noise_cov`: Covariance of Gaussian process noise injected into each plant. Acts as the default for all agents; individual agents can override it via a `process_noise_cov` key inside their `heterogeneous_plants` entry.
- `initial_state_cov`: Covariance used both for initial state sampling and for the Kalman filter's initial estimate covariance. If omitted, it defaults to `4I`.
- `initial_state_fixed`: When `true`, sample one initial state per plant once and reuse it for every reset (training and evaluation).
- `initial_state_fixed_seed`: Optional seed used to sample the fixed initial states. Set this to keep training/eval aligned even if they use different env seeds.

### `lqr`

- `Q`, `R`: Cost matrices used to solve the discrete algebraic Riccati equation. They define how much the optimal controller cares about state deviation versus control effort.

### `reward`

- Tracking error uses `lqr.Q` to compute `(x - x_ref)^T Q (x - x_ref)`.
- `state_error_reward`: Reward mode for tracking error. Options:
  - `"absolute"` (default): Reward equals negative tracking error (`r_t = -e_t`).
  - `"estimation_error"`: Reward equals negative weighted L1 estimation error between true state and controller estimate (`r_t = -||Q (x - x_hat)||_1`).
  - `"lqr_cost"`: Reward equals negative LQR stage cost (`r_t = -(x_t^T Q x_t + u_t^T R u_t)`).
  - `"lqr_cost_immediate_surrogate"`: Reward equals the negative immediate control-mismatch surrogate `r_t = -e_t^T K_t^T R K_t e_t`, where `e_t = x_t - x_hat_t`. This penalizes the extra control effort induced by estimation error at the current step only; unlike `kf_info_m_noise`, it does not include the future state-cost effect via `B^T S_{t+1} B`.
  - `"kf_info"`: Reward equals `r_t = -tr((M_t + S_t) P_t)` where `M_t = K_t^T (R + B^T S_{t+1} B) K_t` (certainty-equivalent LQG weighting).
  - `"kf_info_m"`: Reward equals `r_t = -tr(M_t P_t)` (control-weighted estimation uncertainty).
  - `"kf_info_m_noise"`: Reward equals `r_t = -e_t^T M_t e_t` where `e_t = x_t - x_hat_t` (control-weighted actual estimation error; uses realized error instead of covariance).
  - `"kf_info_s"`: Reward equals `r_t = -tr(S_t P_t)` (state-cost-weighted estimation uncertainty).
  - `"kf_q"`: Reward equals `r_t = -tr(Q P_t)` (Q-weighted estimation uncertainty; expectation of `e_t^T Q e_t`).
  - `"kf_q_noise"`: Reward equals `r_t = -e_t^T Q e_t` where `e_t = x_t - x_hat_t` (Q-weighted actual estimation error; realized counterpart of `kf_q`).
- `comm_recent_window`: Short window (steps) used to count how many recent transmission attempts (`p>0`) an agent has initiated.
- `comm_throughput_window`: Long window (steps) used to estimate per-agent throughput from ACKed packets and their delays.
- `comm_penalty_alpha`: Scalar multiplier (`alpha`) used in the communication penalty `R_{a,comm} = -alpha * N_recent / T`, applied when an action uses the shared packet channel (`1` or `3`) and the network is not set to `perfect_communication`.
- `broadcast_penalty_alpha`: Flat penalty applied when an agent chooses a broadcast action (`2` or `3`). Defaults to `0.0`.
- `normalize`: Explicit flag (default: `false`) for reward normalization in multi-agent runs. When `true`, running normalization scales the per-step reward.
- `no_normalization_scale`: Scalar divisor applied to rewards when normalization is disabled (default `1.0`).
- `reward_clip_min` / `reward_clip_max`: Optional bounds applied to rewards after scaling/normalization.
- `normalization_gamma`: Discount factor for running normalization returns (default `0.999`).
- `comm_throughput_floor`: Small positive value to keep the throughput estimate from collapsing to zero when no ACKs have been observed recently.

When `perfect_communication=true`, the whole communication logic is bypassed directly resulting in instant packet transmission.

### `termination`

- `enabled`: End the episode early when any agent exceeds `state_error_max`.
- `state_error_max`: Threshold on `x^T Q x` using `lqr.Q`. Any agent crossing it terminates the episode for all agents. Use larger values when `Q` scales the error more aggressively.
- `penalty`: Extra reward added when early termination is triggered by `state_error_max` or non-finite errors (default `0.0`). Use a negative value to discourage failures.
- `evaluation`: Optional override dictionary applied to evaluation environments, for example `{"penalty": 0.0}` to disable the termination penalty during eval.
- Non-finite state errors always terminate to avoid NaNs and infinities.
- The `info` dict includes `termination_reasons`, `termination_agents`, and `bad_termination` for logging and analysis.

### `observation`

- `history_window`: Number of past steps included for statuses and throughput history.
- `state_history_window`: Number of past states appended to the observation. If omitted, it defaults to `history_window`.
- `throughput_window`: Sliding window in steps used to estimate recent channel throughput in kbps. It can be a single integer such as `50` or an array such as `[200, 100, 10, 5]` to compute multiple throughput values at different time scales.
- `include_current_throughput`: Whether to append the current throughput feature block to each observation. Defaults to `true`.
- `quantization_step`: Step size for quantizing plant state before it appears in observations. Set to `0` or omit it to disable quantization.
- `error_comm_enabled`: When `true`, append a routed all-to-all handcrafted communication block to each local observation. Defaults to `false`.
- `error_comm_bits`: Number of binary communication features per sender when `error_comm_enabled=true`. Defaults to `1`.
- `error_comm_threshold`: Positive threshold used to quantize the control-aware predicted-gap score when `error_comm_enabled=true`. There is no implicit default when enabled; set it explicitly from config or `--set`.
- `age_comm_enabled`: When `true`, append an all-to-all time-since-last-send communication block to each local observation. Defaults to `false`.
- `age_comm_bits`: Number of quantized age features per sender when `age_comm_enabled=true`. Defaults to `1`.
- `state_comm_enabled`: When `true`, append a sender-slot state-estimate block of size `n_agents * state_dim` to each local observation and expand the action space from 2 to 4 so agents can choose packet send, state broadcast, both, or neither. Defaults to `false`.
- `cevat_state` and `error_comm_enabled` are mutually exclusive.
- `cevat_state` is also mutually exclusive with `age_comm_enabled` and `state_comm_enabled`.
- Local observer features are always included as three per-agent blocks, each of size `state_dim`: first the local sensor Kalman estimate `x_hat_sensor_local`, then the prediction gap `x_hat_sensor_local - x_hat_controller_local`, then the diagonal of the sensor-local Kalman covariance `diag(P_sensor_local)`. The sensor-local tracker updates on every local measurement, while the shadow controller tracker updates only after the matching app ACK is observed.

Observations are laid out as `[current_state, local_kf_prediction, local_estimation_gap, local_sensor_covariance_diag, current_throughput(s)?, prev_states..., prev_statuses..., prev_throughputs..., error_comm?, age_comm?, state_comm?]`. The `current_throughput(s)` block is present when `include_current_throughput=true`. The optional trailing `error_comm` block has size `n_agents * error_comm_bits`, the optional `age_comm` block has size `n_agents`, and the optional `state_comm` block has size `n_agents * state_dim`; all three use a fixed sender-slot layout and zero the receiver's own slot. `error_comm` and `age_comm` are exposed with a one-step delay, while `state_comm` reflects the broadcast controller estimates after one-step-delayed delivery and prediction. Each "prev" block holds `history_window` entries except `prev_states`, which uses `state_history_window`. `current_state` is the quantized plant state.

`history_window` and `state_history_window` are separate on purpose. Past state vectors are much more expensive in observation size than past status or throughput scalars, so you may want longer network/status history without also appending as many previous state vectors.

### `network`

- `data_rate_kbps`: Physical-layer rate used to convert packet sizes into transmission durations.
- `data_packet_size`: Sensor payload size in bytes. Larger packets hold the channel for more timesteps.
- `perfect_communication`: When `true`, disables CSMA behavior entirely. Measurements reach controllers instantly, collisions and throughput bookkeeping are skipped, and decision histories record immediate successes for every transmission attempt.
- `slots_per_step`: Number of micro-slots simulated inside each 10 ms environment step (default 32, about 312 microseconds per slot).
- `mac_min_be` / `mac_max_be`: CSMA/CA backoff exponent bounds (defaults 3 and 5).
- `max_csma_backoffs`: How many CCA failures are allowed before a packet is dropped (default 4).
- `max_frame_retries`: How many collided or NAKed frame retries are attempted before drop (default 3).
- `cca_time_us`, `mac_ack_wait_us`, `mac_ack_turnaround_us`: Timing knobs in microseconds for CCA duration, MAC ACK wait, and ACK turnaround.
- `mac_ack_size_bytes`: Size of the MAC ACK frame (default 5 bytes).
- `mac_ifs_sifs_us`, `mac_ifs_lifs_us`: Inter-frame spacing in microseconds for short and long frames after successful transactions (defaults 192 and 640).
- `mac_ifs_max_sifs_frame_size`: Max frame size in bytes that still uses SIFS (default 18).
- Application-level ACKs are always enabled in network mode. Controllers send app ACKs via CSMA/CA in addition to MAC ACKs.
- `app_ack_packet_size`: Size of app ACK packets in bytes (default 30).
- `app_ack_max_retries`: Maximum retransmission attempts for app ACKs (default 3).
- `tx_buffer_bytes`: Optional per-sensor TX buffer capacity in bytes for queued data packets beyond the in-flight packet. Set to `0` to disable buffering. When set, packets are queued FIFO until the buffer is full.
- `random_drop_rate`: Per-agent probability of physical-layer interference corrupting packets on arrival. This can be a single float applied to all agents or a list of floats with one entry per agent such as `[0.1, 0.0, 0.2]`.

Affected packets still go through the full CSMA/CA process and occupy the channel, but the receiver cannot decode them due to interference. No MAC ACK or app ACK is sent back, so the sender times out and may retry. Interference can also corrupt MAC ACK and app ACK packets in transit, causing unnecessary retries even for successfully received data. This applies only in network mode when `perfect_communication=false`. The default is `0`.

Note: `tx_buffer_bytes` applies only to data packets. MAC ACKs and app ACKs are still sent immediately and are not buffered.

### `controller`

- `use_true_state_control`: When `true`, controllers compute `u = -K x` using the true plant state instead of the Kalman estimate `x_hat`. The Kalman filter still runs, but control no longer depends on it.
- `measurement_noise_cov`: Global covariance matrix for additive sensor measurement noise `z = x + v`, where `v ~ N(0, R)`. The same noisy measurement appears in the agent observation and is transmitted to the controller when `action=1`. The Kalman filter uses this same matrix as `R`, so a zero matrix gives perfect measurements. Individual heterogeneous plants can override this with `system.heterogeneous_plants[i].measurement_noise_cov`.

The sensor-local trackers use the sensor's best local reconstruction of the controller path. If `use_true_state_control=true`, that reconstruction is still approximate because the sensor does not directly observe the controller's true-state control input.

### `training_evaluation`

- `baseline_policy`: Heuristic baseline used during training-time evaluation and model selection.
  - `"perfect_comm"`: Alias for `always_send` with `network.perfect_communication=true`.
  - Any heuristic policy name such as `"random_20"`, `"always_send"`, or `"perfect_sync"`: evaluated with normal network settings.
- The provided config files currently set `training_evaluation.baseline_policy` to `"perfect_sync_n2"`.
- During each evaluation checkpoint, the learned policy and baseline are evaluated seed-by-seed on the same episode seed list. The code reports mean and standard deviation of per-seed drop ratio and selects `best_model.pt` by minimizing mean drop ratio.

Adjusting these parameters lets you explore different plant dynamics, estimator fidelity, reward shaping, and network congestion levels without modifying the source code.

## Saved Run `config.json`

Each training run writes a `config.json` file into its output directory. That saved file preserves the complete effective environment configuration used for training, including CLI `--set` overrides, and adds a `training_run` section with metadata.

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
    "algorithm": "qmix",
    "source_config_path": "/path/to/original/config.json",
    "set_overrides": [
      "reward.state_error_reward=kf_info_m_noise"
    ],
    "hyperparameters": {
      "total_timesteps": 200000,
      "episode_length": 1000,
      "learning_rate": 0.001,
      "gamma": 0.99,
      "batch_size": 64,
      "double_q": false,
      "dueling": false,
      "optimizer": "adam",
      ...
    }
  }
}
```

Input config files in this directory contain environment settings such as `system`, `lqr`, `reward`, `observation`, and `network`. Training hyperparameters such as `double_q`, `learning_rate`, and `optimizer` are supplied via CLI arguments and saved only in the output `config.json` for record-keeping. When `--set` is used, those overrides are applied to the saved top-level config and also listed in `training_run.set_overrides`. The `training_run` section is not read from input configs.

This saved format lets you:

- reproduce the exact run by inspecting the saved `config.json`
- pass the effective config directly to visualization or analysis tools
- track which input config file was used as the base
- compare hyperparameters across runs programmatically
- distinguish between runs such as `iql_0` and `iql_1` by checking their saved hyperparameters
