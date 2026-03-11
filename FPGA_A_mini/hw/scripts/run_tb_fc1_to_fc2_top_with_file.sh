#!/usr/bin/env bash
set -e

SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

mkdir -p ${SIM_DIR}
mkdir -p ${SIM_DIR}/log

vcs -full64 -sverilog -timescale=1ns/1ps \
	-debug_access+all \
	-o ${SIM_DIR}/tb_fc1_to_fc2_top_with_file_simv \
	${RTL_DIR}/pkg/package.sv \
	${RTL_DIR}/mem/quantize_param_bank.sv \
	${RTL_DIR}/core/fc1_relu_requantize_with_file.sv \
	${RTL_DIR}/mem/fc2_weight_bank.sv \
	${RTL_DIR}/mem/fc2_bias_bank.sv \
	${RTL_DIR}/core/fc2_core_with_file.sv \
	${RTL_DIR}/core/fc1_to_fc2_top_with_file.sv \
	${TB_DIR}/tb_fc1_to_fc2_top_with_file.sv \
	2>&1 | tee ${SIM_DIR}/log/compile_tb_fc1_to_fc2_top_with_file.log

${SIM_DIR}/tb_fc1_to_fc2_top_with_file_simv \
	2>&1 | tee ${SIM_DIR}/log/sim_tb_fc1_to_fc2_top_with_file.log
