SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

FC1_ACC_HEX="../route_b_output/fc1_acc_0.hex"
FC1_RELU_HEX="../route_b_output/fc1_relu_0.hex"
FC1_OUT_HEX="../route_b_output/fc1_out_0.hex"
FC2_WEIGHT_HEX="../route_b_output/fc2_weight_int8.hex"
FC2_BIAS_HEX="../route_b_output/fc2_bias_int32.hex"
FC2_ACC_HEX="../route_b_output/fc2_acc_0.hex"
LOGITS_HEX="../route_b_output/logits_0.hex"

mkdir -p ${SIM_DIR}
mkdir -p ${SIM_DIR}/log

vcs -full64 -sverilog -timescale=1ns/1ps \
	-debug_access+all \
	-o ${SIM_DIR}/fc1_to_fc2_top_simv \
	${RTL_DIR}/package.sv \
	${RTL_DIR}/fc1_relu_requantize.sv \
	${RTL_DIR}/fc2_weight_bank.sv \
	${RTL_DIR}/fc2_bias_bank.sv \
	${RTL_DIR}/fc2_core.sv \
	${RTL_DIR}/fc1_to_fc2_top.sv \
	${TB_DIR}/tb_fc1_to_fc2_top.sv \
	2>&1 | tee ${SIM_DIR}/log/compile_tb_fc1_to_fc2_top.log

${SIM_DIR}/fc1_to_fc2_top_simv \
	+FC1_ACC_FILE=${FC1_ACC_HEX} \
	+FC1_RELU_FILE=${FC1_RELU_HEX} \
	+FC1_OUT_FILE=${FC1_OUT_HEX} \
	+FC2_WEIGHT_HEX_FILE=${FC2_WEIGHT_HEX} \
	+FC2_BIAS_FILE=${FC2_BIAS_HEX} \
	+FC2_ACC_FILE=${FC2_ACC_HEX} \
	+LOGITS_FILE=${LOGITS_HEX} \
	2>&1 | tee ${SIM_DIR}/log/sim_tb_fc1_to_fc2_top.log
