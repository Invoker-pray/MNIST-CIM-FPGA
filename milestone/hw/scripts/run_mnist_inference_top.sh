SIM_DIR=../sim
RTL_DIR=../rtl
RTL_SHARED_DIR=../rtl_shared_buffer_ib
TB_DIR=../tb

INPUT_HEX="../route_b_output/input_0.hex"
FC1_WEIGHT_HEX="../route_b_output/fc1_weight_int8.hex"
FC1_BIAS_HEX="../route_b_output/fc1_bias_int32.hex"
FC2_WEIGHT_HEX="../route_b_output/fc2_weight_int8.hex"
FC2_BIAS_HEX="../route_b_output/fc2_bias_int32.hex"
LOGITS_HEX="../route_b_output/logits_0.hex"
PRED_TXT="../route_b_output/pred_0.txt"

mkdir -p ${SIM_DIR}
mkdir -p ${SIM_DIR}/log

vcs -full64 -sverilog -timescale=1ns/1ps \
	-debug_access+all \
	-o ${SIM_DIR}/mnist_inference_top_simv \
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
	2>&1 | tee ${SIM_DIR}/log/compile_tb_mnist_inference_top.log

${SIM_DIR}/mnist_inference_top_simv \
	+INPUT_HEX_FILE=${INPUT_HEX} \
	+WEIGHT_HEX_FILE=${FC1_WEIGHT_HEX} \
	+FC1_BIAS_FILE=${FC1_BIAS_HEX} \
	+FC2_WEIGHT_HEX_FILE=${FC2_WEIGHT_HEX} \
	+FC2_BIAS_FILE=${FC2_BIAS_HEX} \
	+LOGITS_FILE=${LOGITS_HEX} \
	+PRED_FILE=${PRED_TXT} \
	2>&1 | tee ${SIM_DIR}/log/sim_tb_mnist_inference_top.log
