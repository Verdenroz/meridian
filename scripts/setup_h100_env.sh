#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/setup_h100_env.sh
#
# This script prepares a reproducible Phase 0 training environment
# on an H100 host. It is intentionally explicit and fails fast.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[train,dev]"

echo "Running Meridian environment checks..."
PYTHONPATH=python python -m meridian_train.cli env-check
PYTHONPATH=python python -m meridian_train.cli spec-check > /dev/null
PYTHONPATH=python python -m meridian_train.cli feature-vector-check > /dev/null

echo "H100 training environment is ready."

