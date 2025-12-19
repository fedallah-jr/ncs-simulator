#!/usr/bin/env bash
set -euo pipefail

# Runs a small MARL experiment grid and zips the resulting outputs.
#
# Experiments:
#   - Algorithms: IQL, VDN, QMIX
#   - Variants:
#       1) shared params + agent-id (default)
#       2) independent params + no agent-id (--independent-agents --no-agent-id)
#
# Training:
#   - timesteps: 1_000_000
#   - epsilon decay steps: 800_000
#   - config: configs/marl_mixed_plants.json, but reward_mixing scheduler:
#       - type: linear
#       - total_steps: 600_000
#
# Usage:
#   ./run_marl_experiments.sh
#   SEED=1 ./run_marl_experiments.sh
#   OUTPUT_ROOT=outputs ./run_marl_experiments.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

SEED="${SEED:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
EPS_DECAY_STEPS="${EPS_DECAY_STEPS:-800000}"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_root="${OUTPUT_ROOT}/marl_grid_${timestamp}_seed${SEED}"

mkdir -p "${run_root}/logs"

PYTHON_BIN=""
if [[ -f "${HOME}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/.venv/bin/activate"
  PYTHON_BIN="python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "Could not find python or python3 (and ~/.venv not found)." >&2
  exit 1
fi

base_config="configs/marl_mixed_plants.json"
if [[ ! -f "${base_config}" ]]; then
  echo "Config not found: ${base_config}" >&2
  exit 1
fi

common_args=(
  --config "${base_config}"
  --output-root "${run_root}"
  --seed "${SEED}"
  --total-timesteps "${TOTAL_TIMESTEPS}"
  --epsilon-decay-steps "${EPS_DECAY_STEPS}"
)

run_one() {
  local algo_module="$1"
  local label="$2"
  shift 2

  local log_path="${run_root}/logs/${label}.log"
  echo "=== ${label} ==="
  echo "Logging to: ${log_path}"
  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m "${algo_module}" "${common_args[@]}" "$@" 2>&1 | tee "${log_path}"
}

# Shared params + agent-id (default)
run_one "algorithms.marl_iql"  "iql_shared_agentid"
run_one "algorithms.marl_vdn"  "vdn_shared_agentid"
run_one "algorithms.marl_qmix" "qmix_shared_agentid"

# Independent params + no agent-id
run_one "algorithms.marl_iql"  "iql_independent_noagentid"  --independent-agents --no-agent-id
run_one "algorithms.marl_vdn"  "vdn_independent_noagentid"  --independent-agents --no-agent-id
run_one "algorithms.marl_qmix" "qmix_independent_noagentid" --independent-agents --no-agent-id

zip_path="${run_root}.zip"
if command -v zip >/dev/null 2>&1; then
  (cd "${OUTPUT_ROOT}" && zip -qr "$(basename "${zip_path}")" "$(basename "${run_root}")")
else
  "${PYTHON_BIN}" - <<PY
import zipfile
from pathlib import Path

src = Path("${run_root}")
dst = Path("${zip_path}")
with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in src.rglob("*"):
        zf.write(p, p.relative_to(src.parent))
print(f"Wrote zip: {dst}")
PY
fi

echo "Done."
echo "Run root: ${run_root}"
echo "Zipped to: ${zip_path}"
