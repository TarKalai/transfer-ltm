#!/usr/bin/env bash
# Entry point for a Domino job: the "File to run" is this script, its arguments
# are those of run_experiment.py. Examples (Jobs -> Run):
#
#   bash scripts/domino_job.sh --prepare
#   bash scripts/domino_job.sh experiments/base.yaml
#   bash scripts/domino_job.sh experiments/base.yaml --resume
#
# Where data and results live is set in configs/paths.yaml: point it at a
# project dataset, because a job's own disk is discarded when the job ends.
# PYTHON: the interpreter with the packages (default: the project's .venv).
set -euo pipefail

cd "${DOMINO_WORKING_DIR:-$(dirname "$0")/..}"
PYTHON="${PYTHON:-.venv/bin/python}"
echo "commit : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "python : $PYTHON"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU visible"
exec "$PYTHON" scripts/run_experiment.py "$@"
