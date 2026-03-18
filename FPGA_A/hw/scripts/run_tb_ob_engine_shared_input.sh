#!/usr/bin/env bash
set -e

SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

mkdir -p ${SIM_DIR}
mkdir -p ${SIM_DIR}/log

vcs -full64 -sverilog -timescale=1ns/1ps \
	-debug_access+all \
	-o ${SIM_DIR}/tb_fc1_ob_engine_shared_input_simv \
	${RTL_DIR}/pkg/package.sv \
	${RTL_DIR}/mem/fc1_weight_bank.sv \
	${RTL_DIR}/mem/fc1_bias_bank.sv \
	${RTL_DIR}/core/cim_tile.sv \
	${RTL_DIR}/core/psum_accum.sv \
	${RTL_DIR}/core/fc1_ob_engine_shared_input.sv \
	${TB_DIR}/tb_fc1_ob_engine_shared_input.sv \
	2>&1 | tee ${SIM_DIR}/log/compile_tb_fc1_ob_engine_shared_input.log

${SIM_DIR}/tb_fc1_ob_engine_shared_input_simv \
	2>&1 | tee ${SIM_DIR}/log/sim_tb_fc1_ob_engine_shared_input.log
