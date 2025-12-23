#!/usr/bin/env bash
set -euo pipefail

# Wrapper that runs the split experiment scripts.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

"${PROJECT_ROOT}/run_experiments_1.sh"
"${PROJECT_ROOT}/run_experiments_2.sh"
