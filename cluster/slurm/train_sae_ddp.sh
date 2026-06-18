#!/usr/bin/env bash
set -euo pipefail

: "${CONFIG:?set CONFIG to a YAML config path}"

PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29517}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  -m fsaeter.cli train-sae --config "${CONFIG}" ${EXTRA_ARGS}

