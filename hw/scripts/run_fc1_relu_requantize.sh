SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

FC1_ACC_HEX="../route_b_output/fc1_acc_0.hex"
FC1_RELU_HEX="../route_b_output/fc1_relu_0.hex"
FC1_OUT_HEX="../route_b_output/fc1_out_0.hex"

vcs -full64 -sverilog -timescale=1ns/1ps \
	-debug_access+all \
	-o ${SIM_DIR}/fc1_relu_requantize_simv \
	${RTL_DIR}/package.sv \
	${RTL_DIR}/fc1_relu_requantize.sv \
	${TB_DIR}/tb_fc1_relu_requantize.sv \
	2>&1 | tee ${SIM_DIR}/log/compile_tb_fc1_relu_requantize.log

${SIM_DIR}/fc1_relu_requant_simv \
	+FC1_ACC_FILE=${FC1_ACC_HEX} \
	+FC1_RELU_FILE=${FC1_RELU_HEX} \
	+FC1_OUT_FILE=${FC1_OUT_HEX} \
	2>&1 | tee ${SIM_DIR}/log/sim_tb_fc1_relu_requantize.log
