# Networked Control System Simulator

This project models multiple physical plants that share a single IEEE 802.15.4-style channel. Sensors decide each timestep whether to transmit their state, controllers run a Kalman-filtered LQR policy, and a CSMA/CA-inspired medium access layer arbitrates the shared network. The simulation behaviour is configured with the json files from the configs directory.

## Simulation Parameters

The configuration file is divided into sections; each key controls a specific aspect of the simulation:

### `system`
- `A`, `B`: Discrete-time state and input matrices used by every plant and controller.
- `process_noise_cov`: Covariance of Gaussian process noise injected into each plant.
- `measurement_noise_cov`: Covariance of sensor noise added before packets are queued; the Kalman filter uses the same value.
- `measurement_noise_scale_range`: Optional `[min, max]` range for a per-step scalar applied to `measurement_noise_cov` to produce `R_k`. When set and `measurement_noise_cov` is omitted, the base defaults to the identity matrix.
- Initial states are sampled from `N(0, P_0)` where `P_0 = 4I` (hardcoded). The Kalman filter's initial covariance is set to the same `P_0`, ensuring `P = E[ee^T]` from the start.
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
  - `"kf_info"`: Reward equals `r_t = -tr((M_t + S_t) P_t)` where `M_t = K_t^T (R + B^T S_{t+1} B) K_t` (certainty-equivalent LQG weighting).
  - `"kf_info_m"`: Reward equals `r_t = -tr(M_t P_t)` (control-weighted estimation uncertainty).
  - `"kf_info_m_noise"`: Reward equals `r_t = -e_t^T M_t e_t` where `e_t = x_t - x_hat_t` (control-weighted actual estimation error; uses realized error instead of covariance).
  - `"kf_info_s"`: Reward equals `r_t = -tr(S_t P_t)` (state-cost-weighted estimation uncertainty).
- `comm_recent_window`: Short window (steps) used to count how many recent transmission attempts (`p>0`) an agent has initiated.
- `comm_throughput_window`: Long window (steps) used to estimate per-agent throughput from ACKed packets and their delays.
- `comm_penalty_alpha`: Scalar multiplier (`α`) used in the communication penalty `R_{a,\text{comm}} = -α * N_\text{recent}/T`, applied only when `action=1` and the network is not set to `perfect_communication`.
- `normalize`: Explicit flag (default: `false`) for reward normalization in multi-agent runs. When `true`, running normalization scales the per-step reward.
- `no_normalization_scale`: Scalar divisor applied to rewards when normalization is disabled (default `1.0`).
- `reward_clip_min` / `reward_clip_max`: Optional bounds applied to rewards after scaling/normalization.
- `normalization_gamma`: Discount factor for running normalization returns (default `0.99`).
- `comm_throughput_floor`: Small positive value to keep the throughput estimate from collapsing to zero when no ACKs have been observed recently.

When `perfect_communication=true`, the whole communication logic is bypassed directly resulting in instant package transmission.

### `termination`
- `enabled`: End the episode early when any agent exceeds `state_error_max`.
- `state_error_max`: Threshold on `x^T Q x` using `lqr.Q`. Any agent crossing it terminates the episode for all agents (use larger values when `Q` scales the error more aggressively).
- `penalty`: Extra reward added when early termination is triggered by `state_error_max` or non-finite errors (default `0.0`; use a negative value to discourage failures).
- `evaluation`: Optional override dictionary applied to evaluation environments (e.g., `{"penalty": 0.0}` to disable the termination penalty during eval).
- Non-finite state errors always terminate to avoid NaNs/infs.
- The `info` dict includes `termination_reasons`, `termination_agents`, and `bad_termination` for logging/analysis.

### `observation`
- `history_window`: Number of past steps included for statuses and throughput history.
- `state_history_window`: Number of past states appended to the observation (defaults to `history_window` if omitted).
- `throughput_window`: Sliding window (in steps) used to estimate recent channel throughput (kbps). Can be a single integer (e.g., `50`) or an array of window sizes (e.g., `[200, 100, 10, 5]`) to compute multiple throughput values at different time scales. When an array is provided, all calculated throughput values are included in the observation.
- `quantization_step`: Step size for quantizing sensor measurements (state + measurement noise) before they appear in observations. Set to `0` or omit to disable quantization.

Observations are laid out as `[current_measurement, current_throughput(s), current_measurement_noise, prev_states..., prev_statuses..., prev_throughputs...]`, where each "prev" block holds `history_window` (or `state_history_window` for states) entries. When `throughput_window` is an array, multiple current throughput values are included (one per window size). `current_measurement` is the quantized sensor measurement (`state + measurement noise`). `current_measurement_noise` is a scalar intensity summary of `R_k` (trace divided by state dimension).

### `network`
- `data_rate_kbps`: Physical-layer rate used to convert packet sizes into transmission durations.
- `data_packet_size`: Sensor payload size in bytes. Larger packets hold the channel for more timesteps.
- `perfect_communication`: When `true`, disables CSMA behavior entirely—measurements reach controllers instantly, collisions/throughput bookkeeping is skipped, and decision histories record immediate successes for every transmission attempt.
- `slots_per_step`: Number of micro-slots simulated inside each 10 ms environment step (default 32, ≈312 µs per slot).
- `mac_min_be` / `mac_max_be`: CSMA/CA backoff exponent bounds (defaults 3/5).
- `max_csma_backoffs`: How many CCA failures are allowed before a packet is dropped (default 4).
- `max_frame_retries`: How many collided/NAKed frame retries are attempted before drop (default 3).
- `cca_time_us`, `mac_ack_wait_us`, `mac_ack_turnaround_us`: Timing knobs (µs) for CCA duration, MAC ACK wait, and ACK turnaround.
- `mac_ack_size_bytes`: Size of the MAC ACK frame (default 5 bytes).
- `mac_ifs_sifs_us`, `mac_ifs_lifs_us`: Inter-frame spacing (µs) for short/long frames after successful transactions (defaults 192/640).
- `mac_ifs_max_sifs_frame_size`: Max frame size (bytes) that still uses SIFS (default 18).
- `app_ack_enabled`: When `true`, controllers send application-level ACKs via CSMA/CA in addition to MAC ACKs (default `false`).
- `app_ack_packet_size`: Size of app ACK packets in bytes (default 30).
- `app_ack_max_retries`: Maximum retransmission attempts for app ACKs (default 3).
- `tx_buffer_bytes`: Optional per-sensor TX buffer capacity in bytes for queued data packets (beyond the in-flight packet). Set to `0` to disable buffering (current behavior). When set, packets are queued FIFO until the buffer is full.

Note: `tx_buffer_bytes` applies only to data packets; MAC/app ACKs are still sent immediately and are not buffered.

### `controller`
- `use_true_state_control`: When `true`, controllers compute `u = -K x` using the true plant state instead of the Kalman estimate (`x_hat`). The Kalman filter still runs, but control no longer depends on it.

Adjusting these parameters lets you explore different plant dynamics, estimator fidelity, reward shaping, or network congestion levels without modifying the source code.

## Algorithms

Learning-based baselines live under `algorithms/`:

- OpenAI-ES (shared policy, JAX): `python -m algorithms.openai_es --config configs/perfect_comm.json --generations 1000 --popsize 128`
  - Uses `system.n_agents` from the config and appends a one-hot agent id for parameter sharing when `n_agents > 1`.
  - Observation normalization matches MARL flags: `--no-normalize-obs`, `--obs-norm-clip`, `--obs-norm-eps`.
- IQL (multi-agent, PyTorch): `python -m algorithms.marl_iql --config configs/marl_mixed_plants.json --total-timesteps 200000`
- VDN (multi-agent, PyTorch): `python -m algorithms.marl_vdn --config configs/marl_mixed_plants.json --total-timesteps 200000`
- QMIX (multi-agent, PyTorch): `python -m algorithms.marl_qmix --config configs/marl_mixed_plants.json --total-timesteps 200000`
- QPLEX (multi-agent, PyTorch, Q-attention mixer): `python -m algorithms.marl_qplex --config configs/marl_mixed_plants.json --total-timesteps 200000`
  - QPLEX weights agent Qs via Q-attention (state + per-agent Q-values); tune with `--n-head`, `--attend-reg-coef`, `--nonlinear`, `--no-state-bias`, `--no-weighted-head`.
- MAPPO (multi-agent, PyTorch): `python -m algorithms.marl_mappo --config configs/marl_mixed_plants.json --total-timesteps 200000`

All MARL Q-learning algorithms (IQL, VDN, QMIX, QPLEX) support these architectural enhancements:
- `--double-q`: Enable Double DQN (use online network to select actions, target network to evaluate).
- `--dueling`: Enable Dueling DQN architecture (separate value and advantage streams).
- `--stream-hidden-dim`: Hidden dimension for dueling streams (default: 64).

Example with all enhancements: `python -m algorithms.marl_qmix --config configs/marl_mixed_plants.json --dueling --double-q --total-timesteps 200000`

All MARL algorithms (IQL, VDN, QMIX, QPLEX, MAPPO) support observation normalization (enabled by default):
- `--no-normalize-obs`: Disable running mean/std normalization on per-agent observations.
- `--obs-norm-clip`: Clip normalized observations to +/- this value (<=0 disables).
- `--obs-norm-eps`: Epsilon for observation normalization.

## Behavioral Cloning (Actor-Only)

Generate an actor-only behavioral cloning dataset from a heuristic policy:
```bash
python -m tools.generate_bc_dataset --config configs/marl_mixed_plants.json \
  --episodes 50 --episode-length 500 --output outputs/bc_zero_wait.npz
```

Warm-start OpenAI-ES from the dataset:
```bash
python -m algorithms.openai_es --config configs/marl_mixed_plants.json \
  --bc-dataset outputs/bc_zero_wait.npz --bc-epochs 10 --generations 1000
```

Note: the current dataset stores only observations and actions. We plan to extend
behavioral cloning to include critic and Q-function targets in a future update.

CLI flags let you change environment parameters. Use `--output-root` (defaults to `outputs/`) to control where training artifacts land. Each run calls `utils.run_utils.prepare_run_directory(...)`, which creates a sequentially numbered folder for the algorithm (e.g., `iql_0`, `iql_1`, `vdn_0`, etc.). All training details are preserved in the saved config file. That directory always contains:

- **Model Checkpoints:**
  - For OpenAI-ES: `best_model.npz` (flattened params of best individual) and `latest_model.npz`.
  - For MARL (IQL/VDN/QMIX/MAPPO): `best_model.pt` and `latest_model.pt`.
- `training_rewards.csv`: A simple CSV table tracking performance. For OpenAI-ES it logs `[generation, mean_reward, max_reward, time]`.
- **`config.json`**, which combines the full environment configuration with a `training_run` section containing the algorithm name, timestamp, source config path, and all hyperparameters from the run. 

Configuration presets live under `configs/`. `configs/perfect_comm.json` mirrors the default plant/network settings but forces `network.perfect_communication` to `true`, which is useful for debugging algorithms without channel contention.

## Experiment Scripts

Experiment batches live in the root `run_experiment_*` scripts. Each script writes to a timestamped `outputs/exp_*` folder and renames the per-run directories to match the experiment settings. You can override common knobs via environment variables: `SEED`, `OUTPUT_ROOT`, `TOTAL_TIMESTEPS`, `EPS_DECAY_STEPS`. The current `run_experiment_{1,2,3}` scripts also pin `--batch-size 512` and `--learning-rate 2.5e-5` (half the MARL default).

## Visualization

Post-training visualization lives in `tools/visualize_policy.py`.

- MARL visualization (all agents act): `python -m tools.visualize_policy --config configs/marl_mixed_plants.json --policy outputs/.../best_model.pt --policy-type marl_torch --generate-video --per-agent-videos`
  - Outputs include a coordination action raster, a combined state-space plot, a summary plot, and optional combined/per-agent MP4s (FFmpeg required).
- Visualization uses reward/termination `evaluation` overrides from the config when provided.

## Policy Testing

Policy testing lives in `tools/policy_tester.py` and evaluates a target policy against a fixed heuristic set (default: `zero_wait`, `perfect_sync`, `always_send`, `never_send`, `random_50`) over multiple seeds. The evaluator forces raw absolute reward (no normalization/mixing) while keeping communication penalties and termination settings from the config. Pass `--use-reward-normalization` to enable running reward normalization during evaluation.

- Example (MARL): `python -m tools.policy_tester --config configs/marl_mixed_plants.json --policy outputs/.../best_model.pt --policy-type marl_torch --num-seeds 30`
- Example (batch): `python -m tools.policy_tester --models-root outputs --num-seeds 30`
  - Expects subfolders like `model_1/config.json`, `model_1/best_model.pt`, `model_1/latest_model.pt`.
  - Writes `leaderboard.csv` at the models root plus per-model evaluation folders under `model_*/policy_tests/`.
- Example (heuristics only): `python -m tools.policy_tester --config configs/marl_mixed_plants.json --only-heuristics --num-seeds 50`
  - Evaluates heuristic baselines (`zero_wait`, `perfect_sync`, `always_send`, `never_send`, `random_50`) plus a perfect communication baseline (`always_send` with `network.perfect_communication=true`) and a perfect control baseline (`always_send` with `controller.use_true_state_control=true`).
  - `perfect_sync` supports aliases `perfect_sync_n2`, `perfect_sync_n3`, ... (equivalently `perfect_sync_2`, `perfect_sync_3`, ...) to enforce extra idle spacing.
  - Useful for establishing baseline performance metrics before training.

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
    "algorithm": "qmix",
    "source_config_path": "/path/to/original/config.json",
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

**Important:** Input config files (e.g., `configs/marl_mixed_plants.json`) contain only environment settings (`system`, `lqr`, `reward`, etc.). Training hyperparameters like `double_q`, `learning_rate`, `optimizer` are specified via CLI arguments and saved to the output `config.json` for record-keeping. The `training_run` section is not read from input configs.

This format allows you to:
- Reproduce the exact run by inspecting the saved `config.json`
- Pass it directly to visualization or analysis tools
- Track which config file was used as the base
- Compare hyperparameters across different runs programmatically
- Distinguish between experiments (e.g., `iql_0` vs `iql_1`) by checking their saved hyperparameters
