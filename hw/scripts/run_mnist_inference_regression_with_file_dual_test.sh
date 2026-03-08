set -euo pipefail

SIM_DIR=../sim
RTL_DIR=../rtl
RTL_SHARED_DIR=../rtl_shared_buffer_ib
TB_DIR=../tb
LOG_DIR=${SIM_DIR}/log_regression_with_file_dual

DATASET_A_DIR=../route_b_output_2
DATASET_B_DIR=../route_b_output_3

mkdir -p "${SIM_DIR}"
mkdir -p "${SIM_DIR}/log"
mkdir -p "${LOG_DIR}"

SIMV=${SIM_DIR}/mnist_inference_top_with_file_simv

echo "[INFO] Compiling mnist_inference_top_with_file regression binary..."

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
	${RTL_DIR}/quantize_param_bank.sv \
	${RTL_DIR}/fc1_relu_requantize_with_file.sv \
	${RTL_DIR}/fc2_weight_bank.sv \
	${RTL_DIR}/fc2_bias_bank.sv \
	${RTL_DIR}/fc2_core_with_file.sv \
	${RTL_DIR}/fc1_to_fc2_top_with_file.sv \
	${RTL_DIR}/argmax_int8.sv \
	${RTL_DIR}/mnist_inference_top_with_file.sv \
	${TB_DIR}/tb_mnist_inference_top_with_file.sv \
	2>&1 | tee "${SIM_DIR}/log/compile_tb_mnist_inference_top_with_file_dual.log"

echo "[INFO] Compile done."

run_one_dataset() {
	local DATA_DIR="$1"
	local TAG="$2"

	local FC1_WEIGHT_HEX="${DATA_DIR}/fc1_weight_int8.hex"
	local FC1_BIAS_HEX="${DATA_DIR}/fc1_bias_int32.hex"
	local FC2_WEIGHT_HEX="${DATA_DIR}/fc2_weight_int8.hex"
	local FC2_BIAS_HEX="${DATA_DIR}/fc2_bias_int32.hex"
	local QUANT_PARAM_HEX="${DATA_DIR}/quant_params.hex"

	local pass_count=0
	local fail_count=0
	local total_count=0

	echo "============================================================"
	echo "[INFO] Running dataset ${TAG}: ${DATA_DIR}"

	shopt -s nullglob
	local input_files=("${DATA_DIR}"/input_*.hex)

	if [ ${#input_files[@]} -eq 0 ]; then
		echo "[ERROR] No input_*.hex found under ${DATA_DIR}"
		return 1
	fi

	for input_file in "${input_files[@]}"; do
		local base_name sample_id logits_file pred_file sim_log
		base_name=$(basename "${input_file}")
		sample_id=${base_name#input_}
		sample_id=${sample_id%.hex}

		logits_file="${DATA_DIR}/logits_${sample_id}.hex"
		pred_file="${DATA_DIR}/pred_${sample_id}.txt"
		sim_log="${LOG_DIR}/${TAG}_sample_${sample_id}.log"

		total_count=$((total_count + 1))

		if [ ! -f "${logits_file}" ]; then
			echo "[WARN] Skip ${TAG} sample ${sample_id}: missing ${logits_file}"
			continue
		fi
		if [ ! -f "${pred_file}" ]; then
			echo "[WARN] Skip ${TAG} sample ${sample_id}: missing ${pred_file}"
			continue
		fi
		if [ ! -f "${QUANT_PARAM_HEX}" ]; then
			echo "[ERROR] Missing ${QUANT_PARAM_HEX}"
			return 1
		fi

		echo "[INFO] ${TAG} sample ${sample_id}"

		set +e
		"${SIMV}" \
			+INPUT_HEX_FILE="${input_file}" \
			+WEIGHT_HEX_FILE="${FC1_WEIGHT_HEX}" \
			+FC1_BIAS_FILE="${FC1_BIAS_HEX}" \
			+FC2_WEIGHT_HEX_FILE="${FC2_WEIGHT_HEX}" \
			+FC2_BIAS_FILE="${FC2_BIAS_HEX}" \
			+QUANT_PARAM_FILE="${QUANT_PARAM_HEX}" \
			+LOGITS_FILE="${logits_file}" \
			+PRED_FILE="${pred_file}" \
			>"${sim_log}" 2>&1
		sim_rc=$?
		set -e

		if [ ${sim_rc} -ne 0 ]; then
			echo "[FAIL] ${TAG} sample ${sample_id}: simulator exited with rc=${sim_rc}"
			fail_count=$((fail_count + 1))
			continue
		fi

		if grep -q "PASS: mnist_inference_top_with_file matches provided logits/pred golden files." "${sim_log}"; then
			echo "[PASS] ${TAG} sample ${sample_id}"
			pass_count=$((pass_count + 1))
		else
			echo "[FAIL] ${TAG} sample ${sample_id}"
			fail_count=$((fail_count + 1))
		fi
	done

	echo "------------------------------------------------------------"
	echo "[SUMMARY] ${TAG}"
	echo "  total samples : ${total_count}"
	echo "  passed        : ${pass_count}"
	echo "  failed        : ${fail_count}"

	if [ ${fail_count} -ne 0 ]; then
		return 1
	fi
	return 0
}

overall_fail=0

run_one_dataset "${DATASET_A_DIR}" "DATASET_A" || overall_fail=1
run_one_dataset "${DATASET_B_DIR}" "DATASET_B" || overall_fail=1

echo "============================================================"
if [ ${overall_fail} -eq 0 ]; then
	echo "[RESULT] ALL DATASETS PASSED"
	exit 0
else
	echo "[RESULT] SOME DATASETS FAILED"
	exit 1
fi
