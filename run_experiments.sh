#!/usr/bin/env bash
set -euo pipefail

# Runs MARL experiments (shared params + agent-id) and zips outputs.
#
# Experiments:
#   - Algorithms: IQL, VDN, QMIX
#   - Variant: shared params + agent-id (default)
#
# Training:
#   - timesteps: 1_000_000
#   - epsilon decay steps: 800_000
#   - config: configs/marl_absolute_plants.json (absolute reward only)
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

configs=("configs/marl_absolute_plants.json")

for config_path in "${configs[@]}"; do
  if [[ ! -f "${config_path}" ]]; then
    echo "Config not found: ${config_path}" >&2
    exit 1
  fi
done

run_one() {
  local config_path="$1"
  local algo_module="$2"
  local label="$3"
  shift 3

  local common_args=(
    --config "${config_path}"
    --output-root "${run_root}"
    --seed "${SEED}"
    --total-timesteps "${TOTAL_TIMESTEPS}"
    --epsilon-decay-steps "${EPS_DECAY_STEPS}"
  )

  local log_path="${run_root}/logs/${label}.log"
  echo "=== ${label} ==="
  echo "Logging to: ${log_path}"
  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m "${algo_module}" "${common_args[@]}" "$@" 2>&1 | tee "${log_path}"
}

for config_path in "${configs[@]}"; do
  config_name=$(basename "${config_path}" .json)

  run_one "${config_path}" "algorithms.marl_iql"  "iql_shared_agentid_${config_name}"
  run_one "${config_path}" "algorithms.marl_vdn"  "vdn_shared_agentid_${config_name}"
  run_one "${config_path}" "algorithms.marl_qmix" "qmix_shared_agentid_${config_name}"
done

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
