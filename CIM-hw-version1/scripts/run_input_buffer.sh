SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

vcs -full64 -sverilog -timescale=1ns/1ps \
	-debug_access+all \
	-o ${SIM_DIR}/input_buffer_simv \
	${RTL_DIR}/package.sv \
	${RTL_DIR}/input_buffer.sv \
	${TB_DIR}/tb_input_buffer.sv \
	2>&1 | tee ${SIM_DIR}/log/compile_tb_input_buffer.log

${SIM_DIR}/input_buffer_simv +INPUT_HEX_FILE=../route_b_output/input_0.hex 2>&1 | tee ${SIM_DIR}/log/sim_tb_fc1_weight_bank.log
