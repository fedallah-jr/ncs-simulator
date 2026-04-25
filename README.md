# Networked Control System Simulator

This project models multiple physical plants that share a single IEEE 802.15.4-style channel. Sensors decide each timestep whether to transmit their state, controllers run a Kalman-filtered LQR policy, and a CSMA/CA-inspired medium access layer arbitrates the shared network. The simulation behaviour is configured with the json files from the configs directory.

## Algorithms

Learning-based baselines live under `algorithms/`:

- OpenAI-ES (shared policy, JAX): `python -m algorithms.openai_es --config configs/marl_absolute_plants.json --generations 1000 --popsize 128`
  - Uses `system.n_agents` from the config and appends a one-hot agent id for parameter sharing when `n_agents > 1`.
  - Observation normalization matches MARL flags: `--no-normalize-obs`, `--obs-norm-clip`, `--obs-norm-eps`.
- IQL (multi-agent, PyTorch): `python -m algorithms.marl_iql --config configs/marl_absolute_plants.json --total-timesteps 200000`
- DIAL (multi-agent, PyTorch, recurrent, online-only): `python -m algorithms.marl_dial --config configs/marl_absolute_plants.json --total-timesteps 200000`
  - Uses a shared GRU with differentiable communication following the original DIAL paper architecture.
  - Forces a recurrent observation profile during train/eval: `observation.history_window=0`, `observation.state_history_window=0`, and `observation.include_current_throughput=false`.
  - Communication-specific knobs: `--comm-dim`, `--dru-sigma`, `--rnn-hidden-dim`, `--rnn-layers`, `--batch-episodes`, `--momentum`.
  - `--mixer {none,vdn,qmix}`: Value decomposition mixer for joint TD loss (default: `none` = per-agent IQL). `vdn` sums per-agent Q-values; `qmix` uses a state-conditioned hypernetwork with monotonicity constraints. Both smooth noisy per-agent rewards while preserving decentralized execution and differentiable communication.
  - `--qmix-mixing-hidden-dim`, `--qmix-hypernet-hidden-dim`: QMIX architecture knobs (defaults: 32, 64).
- VDN (multi-agent, PyTorch): `python -m algorithms.marl_vdn --config configs/marl_absolute_plants.json --total-timesteps 200000`
- QMIX (multi-agent, PyTorch): `python -m algorithms.marl_qmix --config configs/marl_absolute_plants.json --total-timesteps 200000`
- QPLEX (multi-agent, PyTorch, Q-attention mixer): `python -m algorithms.marl_qplex --config configs/marl_absolute_plants.json --total-timesteps 200000`
  - QPLEX weights agent Qs via Q-attention (state + per-agent Q-values); tune with `--n-head`, `--attend-reg-coef`, `--nonlinear`, `--no-state-bias`, `--no-weighted-head`.
- MAPPO (multi-agent, PyTorch): `python -m algorithms.marl_mappo --config configs/marl_absolute_plants.json --total-timesteps 200000`
  - `--num-mini-batch`: Number of mini-batches per PPO epoch (default `1`, i.e., full-batch updates).
  - Key PPO defaults: `--n-epochs 5`, `--learning-rate 5e-4`, `--vf-coef 1.0`, and LR decay disabled by default (enable with `--lr-decay`).
  - `--popart`: Use PopArt value normalization (output-preserving weight correction) instead of the default EMA-based ValueNorm.
  - PopArt EMA decay can be tuned with `--popart-beta` (default `0.999`).
  - ValueNorm now follows on-policy math by default (`--value-norm-beta 0.99999`, variance floor `1e-2`); enable `--value-norm-per-element-update` to scale decay by rollout batch size.
- HAPPO (multi-agent, PyTorch): `python -m algorithms.marl_happo --config configs/marl_absolute_plants.json --total-timesteps 200000`
  - Independent actor per agent with sequential policy update and importance-weighting factor (monotonic improvement guarantee).
  - `--num-mini-batch`: Number of mini-batches per PPO epoch (default `1`, i.e., full-batch updates).
  - Key PPO defaults: `--n-epochs 5`, `--learning-rate 5e-4`, `--vf-coef 1.0`, and LR decay disabled by default (enable with `--lr-decay`).
  - Uses shared team reward and a scalar centralized critic, matching the paper's fully cooperative formulation.
  - `--fixed-order`: Use fixed agent update order instead of random shuffle each iteration.
  - `--popart`: Use PopArt value normalization instead of the default EMA-based ValueNorm.
  - PopArt EMA decay can be tuned with `--popart-beta` (default `0.999`).

All MARL Q-learning algorithms (IQL, IQL-DIAL, VDN, QMIX, QPLEX) support these architectural enhancements:
- `--double-q`: Enable Double DQN (use online network to select actions, target network to evaluate).
- `--dueling`: Enable Dueling DQN architecture (separate value and advantage streams).
- `--stream-hidden-dim`: Hidden dimension for dueling streams (default: 64).

Example with all enhancements: `python -m algorithms.marl_qmix --config configs/marl_absolute_plants.json --dueling --double-q --total-timesteps 200000`

All MARL algorithms (IQL, IQL-DIAL, VDN, QMIX, QPLEX, MAPPO, HAPPO) support observation normalization (enabled by default):
- `--no-normalize-obs`: Disable running mean/std normalization on per-agent observations.
- `--obs-norm-clip`: Clip normalized observations to +/- this value (<=0 disables).
- `--obs-norm-eps`: Epsilon for observation normalization.

CLI flags let you change environment parameters. Use `--output-root` (defaults to `outputs/`) to control where training artifacts land. Each run calls `utils.run_utils.prepare_run_directory(...)`, which creates a sequentially numbered folder for the algorithm (e.g., `iql_0`, `iql_1`, `vdn_0`, etc.). All training details are preserved in the saved config file. That directory always contains:

- **Model Checkpoints:**
  - For OpenAI-ES: `best_model.npz` (flattened params of best individual) and `latest_model.npz`.
  - For MARL (IQL/VDN/QMIX/MAPPO): `best_model.pt` and `latest_model.pt`.
- `training_rewards.csv`: A simple CSV table tracking performance. For OpenAI-ES it logs `[generation, mean_reward, max_reward, time]`.
- `evaluation_drop_stats.csv`: Training-time baseline comparison stats per eval step: baseline policy, policy/baseline mean/std rewards, and drop-ratio mean/std used for `best_model.pt` selection.
- **`config.json`**, which combines the effective environment configuration used by the run (including CLI `--set` overrides, when provided) with a `training_run` section containing the algorithm name, timestamp, source config path, and all hyperparameters from the run. 

## Experiment Runner

The finalized experiment matrix is orchestrated from `tools/run_experiments.py`, which replaces the legacy `run_experiment_*` bash scripts. It exposes a numbered registry of 20 experiments split across four categories:

- **IDs 1-6** (Cat 1): IQL, QMIX, VDN, MAPPO, HAPPO, HASAC at 15M steps on the heterogeneous config. Q-learners get `--double-q`; n-step returns are 3 (the default for IQL/QMIX/VDN, set explicitly for HASAC). All six use `--no-normalize-obs --feature-norm --layer-norm`; HASAC additionally passes `--value-norm` (MAPPO/HAPPO use ValueNorm via the default code path).
- **IDs 7-12** (Cat 2): NDQ at 30M steps, sweeping `--comm-embed-dim` ∈ {5, 10, 15} × `--mixer` ∈ {vdn, qmix}.
- **IDs 13-16** (Cat 3): VDN, QMIX, HAPPO, HASAC at 15M with 8-bit hand-crafted error communication. Quantile bin edges are computed once via `tools.search_vou_threshold` and cached under `outputs/_shared/vou_search/`.
- **IDs 17-20** (Cat 4): same four algorithms at 15M with 4-bit hand-crafted error comm + 4-bit age comm.

List the registry with `python -m tools.run_experiments --list`. Run a subset (CSV or ranges) with `--ids`:

```bash
python -m tools.run_experiments --list                  # show ID -> name table
python -m tools.run_experiments --ids 1-6               # one batch with all Cat 1 experiments
python -m tools.run_experiments --ids 7-12              # NDQ sweep
python -m tools.run_experiments --ids 1,4-6,9 --dry-run # preview commands without running
```

Each `--ids` invocation forms one **batch**. The orchestrator:

1. Creates `outputs/experiments_<batch_name>/` (auto-named from the ID list, e.g. `experiments_1-6`, `experiments_1_4-6_9`; override with `--batch-name`).
2. If any selected experiment needs hand-crafted comm, runs `tools.search_vou_threshold` once (cached at `outputs/_shared/vou_search/`, reused across batches).
3. Trains each experiment **sequentially** into the batch dir, renaming the auto-numbered subfolder (`{algo}_0`, ...) to its full name (`IQL_doubleq_15mil`, `NDQ_5dim_vdn_30mil`, `HASAC_8bithand_15mil`, ...). A failing training does not abort siblings.
4. Runs `tools.policy_tester --models-root <batch_dir>` once across the whole batch, producing a cross-experiment `leaderboard.csv` plus `leaderboard_network_stats.csv`, a `perfect_comm_baseline/` reference run, and per-model `<NAME>/policy_tests/` aggregates.
5. Zips the entire batch directory to `outputs/experiments_<batch_name>.zip`.

Resulting layout:

```
outputs/experiments_1-6/
  IQL_doubleq_15mil/    best_model.pt, config.json, training_rewards.csv, policy_tests/, ...
  QMIX_doubleq_15mil/   ...
  VDN_doubleq_15mil/    ...
  MAPPO_15mil/          ...
  HAPPO_15mil/          ...
  HASAC_15mil/          ...
  leaderboard.csv
  leaderboard_network_stats.csv
  perfect_comm_baseline/
  logs/                 train_<NAME>.log per experiment + policy_test_batch.log
outputs/experiments_1-6.zip
```

To split the matrix across machines, give each box a disjoint slice of IDs (e.g. machine A `--ids 1-6`, machine B `--ids 7-12`, machine C `--ids 13-20`). Each machine produces its own batch zip independently. Other knobs: `--seed`, `--output-root`, `--num-policy-test-seeds`, `--skip-policy-test`, `--skip-zip`, `--skip-vou`, `--force-vou`, `--dry-run`.

## Visualization

Post-training visualization lives in `tools/visualize_policy.py`.

- MARL visualization (all agents act): `python -m tools.visualize_policy --config configs/marl_absolute_plants.json --policy outputs/.../best_model.pt --policy-type marl_torch --generate-video --per-agent-videos`
  - Outputs include a coordination action raster, a combined state-space plot, a summary plot, and optional combined/per-agent MP4s (FFmpeg required).
- Visualization uses reward/termination `evaluation` overrides from the config when provided.

## Policy Testing

Policy testing lives in `tools/policy_tester.py` and evaluates a target policy against a fixed heuristic set (default: `zero_wait`, `perfect_sync`, `always_send`, `never_send`, `random_50`) over multiple seeds. The evaluator forces raw absolute reward (no normalization/mixing) while keeping communication penalties and termination settings from the config. Pass `--use-reward-normalization` to enable running reward normalization during evaluation.

- Example (MARL): `python -m tools.policy_tester --config configs/marl_absolute_plants.json --policy outputs/.../best_model.pt --policy-type marl_torch --num-seeds 30`
- Example (batch): `python -m tools.policy_tester --models-root outputs --num-seeds 30`
  - Expects subfolders like `model_1/config.json`, `model_1/best_model.pt`, `model_1/latest_model.pt`.
  - Writes `leaderboard.csv` at the models root plus per-model evaluation folders under `model_*/policy_tests/`.
- Example (heuristics only): `python -m tools.policy_tester --config configs/marl_absolute_plants.json --only-heuristics --num-seeds 50`
  - Evaluates heuristic baselines (`zero_wait`, `perfect_sync`, `always_send`, `never_send`, `random_50`) plus a perfect communication baseline (`always_send` with `network.perfect_communication=true`).
  - `perfect_sync` supports aliases `perfect_sync_n2`, `perfect_sync_n3`, ... (equivalently `perfect_sync_2`, `perfect_sync_3`, ...) to enforce extra idle spacing.
- Useful for establishing baseline performance metrics before training.

## Configuration

Input configuration files live under [`configs/`](configs/). Detailed config documentation now lives in [`configs/README.md`](configs/README.md), including:

- input config sections and field semantics
- notes on observation history fields such as `history_window` vs `state_history_window`
- the available config files in this repo
- the saved run `config.json` format written under `outputs/`
