#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_DIR="${ROOT_DIR}/sim"

TOP_TB="tb_mnist_cim_demo_a_top_led"
SIMV="${SIM_DIR}/${TOP_TB}_simv"
COMPILE_LOG="${SIM_DIR}/compile_${TOP_TB}.log"
RUN_LOG="${SIM_DIR}/sim_${TOP_TB}.log"

mkdir -p "${SIM_DIR}"

echo "INFO: ROOT_DIR   = ${ROOT_DIR}"
echo "INFO: SCRIPT_DIR = ${SCRIPT_DIR}"
echo "INFO: SIM_DIR    = ${SIM_DIR}"

# Clean previous build for this tb only
rm -rf "${SIMV}" "${SIMV}.daidir" "${COMPILE_LOG}" "${RUN_LOG}"

pushd "${SCRIPT_DIR}" >/dev/null

vcs -full64 -sverilog -timescale=1ns/1ps \
	-o "${SIMV}" \
	+v2k \
	../rtl/pkg/package.sv \
	../rtl/ctrl/debounce.sv \
	../rtl/ctrl/onepulse.sv \
	../rtl/mem/fc1_bias_bank.sv \
	../rtl/mem/fc1_multi_block_shared_sample_rom.sv \
	../rtl/mem/fc1_weight_bank.sv \
	../rtl/mem/fc2_bias_bank.sv \
	../rtl/mem/fc2_weight_bank.sv \
	../rtl/mem/mnist_sample_rom.sv \
	../rtl/mem/quantize_param_bank.sv \
	../rtl/core/argmax_int8.sv \
	../rtl/core/cim_tile.sv \
	../rtl/core/fc1_multi_block_shared_input.sv \
	../rtl/core/fc1_ob_engine_shared_input.sv \
	../rtl/core/fc1_relu_requantize_with_file.sv \
	../rtl/core/fc1_to_fc2_top_with_file.sv \
	../rtl/core/fc2_core_with_file.sv \
	../rtl/core/input_buffer.sv \
	../rtl/core/mnist_cim_accel_ip.sv \
	../rtl/core/mnist_inference_core_board.sv \
	../rtl/core/psum_accum.sv \
	../rtl/uart/uart_pred_sender.sv \
	../rtl/uart/uart_tx.sv \
	../rtl/top/mnist_cim_demo_a_top.sv \
	../tb/tb_mnist_cim_demo_a_top.sv \
	-top "${TOP_TB}" \
	-l "${COMPILE_LOG}"

"${SIMV}" -l "${RUN_LOG}"

popd >/dev/null

echo "INFO: compile log = ${COMPILE_LOG}"
echo "INFO: run log     = ${RUN_LOG}"
echo "INFO: simv        = ${SIMV}"
echo "DONE: simulation finished"
