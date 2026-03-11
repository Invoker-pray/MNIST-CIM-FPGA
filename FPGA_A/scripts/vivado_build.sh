#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VIVADO_DIR="${ROOT_DIR}/vivado"
TCL_SCRIPT="${SCRIPT_DIR}/vivado_build.tcl"

mkdir -p "${VIVADO_DIR}"

echo "INFO: ROOT_DIR   = ${ROOT_DIR}"
echo "INFO: VIVADO_DIR = ${VIVADO_DIR}"
echo "INFO: TCL_SCRIPT = ${TCL_SCRIPT}"

cd "${VIVADO_DIR}"

vivado \
	-mode batch \
	-source "${TCL_SCRIPT}" \
	-journal "${VIVADO_DIR}/vivado.jou" \
	-log "${VIVADO_DIR}/vivado.log" \
	-tclargs "${ROOT_DIR}" "${VIVADO_DIR}"
