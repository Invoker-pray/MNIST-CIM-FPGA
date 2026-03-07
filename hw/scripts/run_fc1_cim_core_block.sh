SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

INPUT_HEX="../route_b_output/input_0.hex"
WEIGHT_HEX="../route_b_output/fc1_weight_int8.hex"
FC1_ACC_HEX="../route_b_output/fc1_acc_0.hex"
FC1_BIAS_HEX="../route_b_output/fc1_bias_int32.hex"

vcs -full64 -sverilog -timescale=1ns/1ps \
	-debug_access+all \
	-o ${SIM_DIR}/fc1_cim_core_block_simv \
	${RTL_DIR}/package.sv \
	${RTL_DIR}/input_buffer.sv \
	${RTL_DIR}/fc1_weight_bank.sv \
	${RTL_DIR}/cim_tile.sv \
	${RTL_DIR}/psum_accum.sv \
	${RTL_DIR}/fc1_cim_core_block.sv \
	${RTL_DIR}/fc1_bias_bank.sv \
	${TB_DIR}/tb_fc1_cim_core_block.sv \
	2>&1 | tee ${SIM_DIR}/log/compile_tb_fc1_cim_core_block.log

${SIM_DIR}/fc1_cim_core_block_simv \
	+INPUT_HEX_FILE=${INPUT_HEX} \
	+WEIGHT_HEX_FILE=${WEIGHT_HEX} \
	+FC1_ACC_FILE=${FC1_ACC_HEX} \
	+FC1_BIAS_FILE=${FC1_BIAS_HEX} \
	2>&1 | tee ${SIM_DIR}/log/sim_tb_fc1_cim_core_block.log
