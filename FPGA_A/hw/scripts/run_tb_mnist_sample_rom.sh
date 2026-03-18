#!/usr/bin/env bash
set -e

SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

mkdir -p ${SIM_DIR}
mkdir -p ${SIM_DIR}/log

vcs -full64 -sverilog -timescale=1ns/1ps \
	-debug_access+all \
	-o ${SIM_DIR}/tb_mnist_sample_rom_simv \
	${RTL_DIR}/pkg/package.sv \
	${RTL_DIR}/mem/mnist_sample_rom.sv \
	${TB_DIR}/tb_mnist_sample_rom.sv \
	2>&1 | tee ${SIM_DIR}/log/compile_tb_mnist_sample_rom.log

${SIM_DIR}/tb_mnist_sample_rom_simv \
	2>&1 | tee ${SIM_DIR}/log/sim_tb_mnist_sample_rom.log
