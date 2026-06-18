#!/usr/bin/env bash
set -euo pipefail

: "${CONFIG:?set CONFIG to a YAML config path}"

PYTHON_BIN="${PYTHON_BIN:-python}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

${PYTHON_BIN} -m fsaeter.cli build-h --config "${CONFIG}" ${EXTRA_ARGS}
exec ${PYTHON_BIN} -m fsaeter.cli mine-concepts --config "${CONFIG}" ${EXTRA_ARGS}
