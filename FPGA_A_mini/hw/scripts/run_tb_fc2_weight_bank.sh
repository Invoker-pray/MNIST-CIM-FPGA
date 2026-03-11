#!/usr/bin/env bash
set -e

SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

mkdir -p ${SIM_DIR}
mkdir -p ${SIM_DIR}/log

vcs -full64 -sverilog -timescale=1ns/1ps \
	-debug_access+all \
	-o ${SIM_DIR}/tb_fc2_weight_bank_simv \
	${RTL_DIR}/pkg/package.sv \
	${RTL_DIR}/mem/fc2_weight_bank.sv \
	${TB_DIR}/tb_fc2_weight_bank.sv \
	2>&1 | tee ${SIM_DIR}/log/compile_tb_fc2_weight_bank.log

${SIM_DIR}/tb_fc2_weight_bank_simv \
	2>&1 | tee ${SIM_DIR}/log/sim_tb_fc2_weight_bank.log
