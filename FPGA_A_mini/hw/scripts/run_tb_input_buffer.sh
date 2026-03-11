
#!/usr/bin/env bash
set -e

SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

mkdir -p ${SIM_DIR}
mkdir -p ${SIM_DIR}/log

vcs -full64 -sverilog -timescale=1ns/1ps \
	-debug_access+all \
	-o ${SIM_DIR}/tb_input_buffer_simv \
	${RTL_DIR}/pkg/package.sv \
	${RTL_DIR}/core/input_buffer.sv \
	${TB_DIR}/tb_input_buffer.sv \
	2>&1 | tee ${SIM_DIR}/log/compile_tb_input_buffer.log

${SIM_DIR}/tb_input_buffer_simv \
	2>&1 | tee ${SIM_DIR}/log/sim_tb_input_buffer.log
