SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

vcs -full64 -sverilog -timescale=1ns/1ps \
	-debug_access+all \
	-o ${SIM_DIR}/input_buffer_simv \
	${RTL_DIR}/package.sv \
	${RTL_DIR}/input_buffer.sv \
	${TB_DIR}/tb_input_buffer.sv \
	2>&1 | tee ${SIM_DIR}/log/compile_input_buffer.log

${SIM_DIR}/input_buffer_simv 2>&1 | tee ${SIM_DIR}/sim_input_buffer.log
