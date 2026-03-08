#!/usr/bin/env bash
set -euo pipefail

SIM_DIR=../sim
RTL_DIR=../rtl
RTL_SHARED_DIR=../rtl_shared_buffer_ib
TB_DIR=../tb
DATA_DIR=../route_b_output
LOG_DIR=${SIM_DIR}/log_regression

FC1_WEIGHT_HEX="${DATA_DIR}/fc1_weight_int8.hex"
FC1_BIAS_HEX="${DATA_DIR}/fc1_bias_int32.hex"
FC2_WEIGHT_HEX="${DATA_DIR}/fc2_weight_int8.hex"
FC2_BIAS_HEX="${DATA_DIR}/fc2_bias_int32.hex"

mkdir -p "${SIM_DIR}"
mkdir -p "${SIM_DIR}/log"
mkdir -p "${LOG_DIR}"

SIMV=${SIM_DIR}/mnist_inference_top_simv

echo "[INFO] Compiling mnist_inference_top regression binary..."

vcs -full64 -sverilog -timescale=1ns/1ps \
    -debug_access+all \
    -o "${SIMV}" \
    ${RTL_DIR}/package.sv \
    ${RTL_SHARED_DIR}/input_buffer.sv \
    ${RTL_SHARED_DIR}/fc1_weight_bank.sv \
    ${RTL_SHARED_DIR}/fc1_bias_bank.sv \
    ${RTL_SHARED_DIR}/cim_tile.sv \
    ${RTL_SHARED_DIR}/psum_accum.sv \
    ${RTL_SHARED_DIR}/fc1_ob_engine_shared_input.sv \
    ${RTL_SHARED_DIR}/fc1_multi_block_shared_input.sv \
    ${RTL_DIR}/fc1_relu_requantize.sv \
    ${RTL_DIR}/fc2_weight_bank.sv \
    ${RTL_DIR}/fc2_bias_bank.sv \
    ${RTL_DIR}/fc2_core.sv \
    ${RTL_DIR}/fc1_to_fc2_top.sv \
    ${RTL_DIR}/argmax_int8.sv \
    ${RTL_DIR}/mnist_inference_top.sv \
    ${TB_DIR}/tb_mnist_inference_top.sv \
    2>&1 | tee "${SIM_DIR}/log/compile_tb_mnist_inference_top_regression.log"

echo "[INFO] Compile done."

pass_count=0
fail_count=0
total_count=0

shopt -s nullglob
input_files=("${DATA_DIR}"/input_*.hex)

if [ ${#input_files[@]} -eq 0 ]; then
    echo "[ERROR] No input_*.hex found under ${DATA_DIR}"
    exit 1
fi

for input_file in "${input_files[@]}"; do
    base_name=$(basename "${input_file}")
    sample_id=${base_name#input_}
    sample_id=${sample_id%.hex}

    logits_file="${DATA_DIR}/logits_${sample_id}.hex"
    pred_file="${DATA_DIR}/pred_${sample_id}.txt"

    total_count=$((total_count + 1))

    if [ ! -f "${logits_file}" ]; then
        echo "[WARN] Skip sample ${sample_id}: missing ${logits_file}"
        continue
    fi

    if [ ! -f "${pred_file}" ]; then
        echo "[WARN] Skip sample ${sample_id}: missing ${pred_file}"
        continue
    fi

    sim_log="${LOG_DIR}/sim_sample_${sample_id}.log"

    echo "============================================================"
    echo "[INFO] Running sample ${sample_id}"
    echo "[INFO] INPUT : ${input_file}"
    echo "[INFO] LOGITS: ${logits_file}"
    echo "[INFO] PRED  : ${pred_file}"

    set +e
    "${SIMV}" \
        +INPUT_HEX_FILE="${input_file}" \
        +WEIGHT_HEX_FILE="${FC1_WEIGHT_HEX}" \
        +FC1_BIAS_FILE="${FC1_BIAS_HEX}" \
        +FC2_WEIGHT_HEX_FILE="${FC2_WEIGHT_HEX}" \
        +FC2_BIAS_FILE="${FC2_BIAS_HEX}" \
        +LOGITS_FILE="${logits_file}" \
        +PRED_FILE="${pred_file}" \
        > "${sim_log}" 2>&1
    sim_rc=$?
    set -e

    if [ ${sim_rc} -ne 0 ]; then
        echo "[FAIL] sample ${sample_id}: simulator exited with rc=${sim_rc}"
        fail_count=$((fail_count + 1))
        continue
    fi

    if grep -q "PASS: mnist_inference_top matches provided logits/pred golden files." "${sim_log}"; then
        echo "[PASS] sample ${sample_id}"
        pass_count=$((pass_count + 1))
    elif grep -q "PASS: mnist_inference_top matches logits" "${sim_log}"; then
        echo "[PASS] sample ${sample_id}"
        pass_count=$((pass_count + 1))
    else
        echo "[FAIL] sample ${sample_id}"
        fail_count=$((fail_count + 1))
        echo "[INFO] See log: ${sim_log}"
    fi
done

echo "============================================================"
echo "[SUMMARY]"
echo "  total samples : ${total_count}"
echo "  passed        : ${pass_count}"
echo "  failed        : ${fail_count}"

if [ ${fail_count} -eq 0 ]; then
    echo "[RESULT] ALL SAMPLES PASSED"
    exit 0
else
    echo "[RESULT] SOME SAMPLES FAILED"
    exit 1
fi
