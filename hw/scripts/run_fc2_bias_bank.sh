SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

FC2_BIAS_HEX="../route_b_output/fc2_bias_int32.hex"

vcs -full64 -sverilog -timescale=1ns/1ps \
	-debug_access+all \
	-o ${SIM_DIR}/fc2_bias_bank_simv \
	${RTL_DIR}/package.sv \
	${RTL_DIR}/fc2_bias_bank.sv \
	${TB_DIR}/tb_fc2_bias_bank.sv \
	2>&1 | tee ${SIM_DIR}/log/compile_tb_fc2_bias_bank.log

${SIM_DIR}/fc2_bias_bank_simv \
	+FC2_BIAS_FILE=${FC2_BIAS_HEX} \
	2>&1 | tee ${SIM_DIR}/log/sim_tb_fc2_bias_bank.log
