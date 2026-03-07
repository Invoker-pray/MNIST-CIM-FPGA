SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

vcs -full64 -sverilog -timescale=1ns/1ps \
	-debug_access+all \
	-o ${SIM_DIR}/fc1_weight_bank_simv \
	${RTL_DIR}/package.sv \
	${RTL_DIR}/fc1_weight_bank.sv \
	${TB_DIR}/tb_fc1_weight_bank.sv \
	2>&1 | tee ${SIM_DIR}/log/compile_tb_fc1_weight_bank.log

${SIM_DIR}/fc1_weight_bank_simv +WEIGHT_HEX_FILE=../route_b_output/fc1_weight_int8.hex 2>&1 | tee ${SIM_DIR}/log/sim_tb_fc1_weight_bank.log
