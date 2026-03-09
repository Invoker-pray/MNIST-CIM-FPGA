#!/usr/bin/env bash
set -euo pipefail

# Run this script from onboard_A/scripts/
# Example:
#   cd onboard_A/scripts
#   bash run.sh
# Optional overrides:
#   bash run.sh SAMPLE_ID=3 PRED_FILE=../data/expected/pred_3.txt
#   bash run.sh SIM=vcs

SIM="${SIM:-vcs}"
TOP="tb_mnist_cim_demo_a_top"
BUILD_DIR="../sim"
SIMV="${BUILD_DIR}/${TOP}_simv"
COMPILE_LOG="${BUILD_DIR}/compile_${TOP}.log"
RUN_LOG="${BUILD_DIR}/sim_${TOP}.log"

SAMPLE_ID="${SAMPLE_ID:-0}"
SAMPLE_HEX_FILE="${SAMPLE_HEX_FILE:-../data/samples/mnist_samples_route_b_output_2.hex}"
FC1_WEIGHT_HEX_FILE="${FC1_WEIGHT_HEX_FILE:-../data/weights/fc1_weight_int8.hex}"
FC1_BIAS_HEX_FILE="${FC1_BIAS_HEX_FILE:-../data/weights/fc1_bias_int32.hex}"
QUANT_PARAM_FILE="${QUANT_PARAM_FILE:-../data/quant/quant_params.hex}"
FC2_WEIGHT_HEX_FILE="${FC2_WEIGHT_HEX_FILE:-../data/weights/fc2_weight_int8.hex}"
FC2_BIAS_HEX_FILE="${FC2_BIAS_FILE:-../data/weights/fc2_bias_int32.hex}"
PRED_FILE="${PRED_FILE:-../data/expected/pred_${SAMPLE_ID}.txt}"

mkdir -p "${BUILD_DIR}"
rm -rf "${SIMV}" "${SIMV}.daidir"

SRC_FILES=(
	../rtl/pkg/package.sv
	../rtl/core/cim_tile.sv
	../rtl/core/psum_accum.sv
	../rtl/core/input_buffer.sv
	../rtl/mem/fc1_weight_bank.sv
	../rtl/mem/fc1_bias_bank.sv
	../rtl/core/fc1_ob_engine_shared_input.sv
	../rtl/mem/mnist_sample_rom.sv
	../rtl/mem/fc1_multi_block_shared_sample_rom.sv
	../rtl/core/fc1_multi_block_shared_input.sv
	../rtl/mem/quantize_param_bank.sv
	../rtl/core/fc1_relu_requantize_with_file.sv
	../rtl/mem/fc2_weight_bank.sv
	../rtl/mem/fc2_bias_bank.sv
	../rtl/core/fc2_core_with_file.sv
	../rtl/core/argmax_int8.sv
	../rtl/core/fc1_to_fc2_top_with_file.sv
	../rtl/core/mnist_inference_core_board.sv
	../rtl/uart/uart_tx.sv
	../rtl/uart/uart_pred_sender.sv
	../rtl/top/mnist_cim_demo_a_top.sv
	../tb/tb_mnist_cim_demo_a_top.sv
)

echo "============================================================" | tee "${COMPILE_LOG}"
echo "Compile ${TOP}" | tee -a "${COMPILE_LOG}"
echo "PWD                  = $(pwd)" | tee -a "${COMPILE_LOG}"
echo "SIM                  = ${SIM}" | tee -a "${COMPILE_LOG}"
echo "SAMPLE_ID            = ${SAMPLE_ID}" | tee -a "${COMPILE_LOG}"
echo "SAMPLE_HEX_FILE      = ${SAMPLE_HEX_FILE}" | tee -a "${COMPILE_LOG}"
echo "FC1_WEIGHT_HEX_FILE  = ${FC1_WEIGHT_HEX_FILE}" | tee -a "${COMPILE_LOG}"
echo "FC1_BIAS_HEX_FILE    = ${FC1_BIAS_HEX_FILE}" | tee -a "${COMPILE_LOG}"
echo "QUANT_PARAM_FILE     = ${QUANT_PARAM_FILE}" | tee -a "${COMPILE_LOG}"
echo "FC2_WEIGHT_HEX_FILE  = ${FC2_WEIGHT_HEX_FILE}" | tee -a "${COMPILE_LOG}"
echo "FC2_BIAS_HEX_FILE    = ${FC2_BIAS_HEX_FILE}" | tee -a "${COMPILE_LOG}"
echo "PRED_FILE            = ${PRED_FILE}" | tee -a "${COMPILE_LOG}"
echo "============================================================" | tee -a "${COMPILE_LOG}"

if [[ "${SIM}" == "vcs" ]]; then
	vcs -full64 -sverilog -timescale=1ns/1ps -debug_acc+all -lca \
		-o "${SIMV}" \
		"${SRC_FILES[@]}" \
		>"${COMPILE_LOG}" 2>&1

	"./${SIMV}" \
		+SAMPLE_HEX_FILE="${SAMPLE_HEX_FILE}" \
		+FC1_WEIGHT_HEX_FILE="${FC1_WEIGHT_HEX_FILE}" \
		+FC1_BIAS_FILE="${FC1_BIAS_HEX_FILE}" \
		+WEIGHT_HEX_FILE="${FC1_WEIGHT_HEX_FILE}" \
		+QUANT_PARAM_FILE="${QUANT_PARAM_FILE}" \
		+FC2_WEIGHT_HEX_FILE="${FC2_WEIGHT_HEX_FILE}" \
		+FC2_BIAS_FILE="${FC2_BIAS_HEX_FILE}" \
		+PRED_FILE="${PRED_FILE}" \
		>"${RUN_LOG}" 2>&1
else
	iverilog -g2012 -o "${SIMV}" "${SRC_FILES[@]}" >"${COMPILE_LOG}" 2>&1
	vvp "${SIMV}" \
		+SAMPLE_HEX_FILE="${SAMPLE_HEX_FILE}" \
		+FC1_WEIGHT_HEX_FILE="${FC1_WEIGHT_HEX_FILE}" \
		+FC1_BIAS_FILE="${FC1_BIAS_HEX_FILE}" \
		+WEIGHT_HEX_FILE="${FC1_WEIGHT_HEX_FILE}" \
		+QUANT_PARAM_FILE="${QUANT_PARAM_FILE}" \
		+FC2_WEIGHT_HEX_FILE="${FC2_WEIGHT_HEX_FILE}" \
		+FC2_BIAS_FILE="${FC2_BIAS_HEX_FILE}" \
		+PRED_FILE="${PRED_FILE}" \
		>"${RUN_LOG}" 2>&1
fi

echo "[OK] compile log: ${COMPILE_LOG}"
echo "[OK] run log:     ${RUN_LOG}"
echo "----- tail ${RUN_LOG} -----"
tail -n 40 "${RUN_LOG}" || true
