#!/usr/bin/env bash
set -e

SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

mkdir -p ${SIM_DIR}
mkdir -p ${SIM_DIR}/log

vcs -full64 -sverilog -timescale=1ns/1ps \
	-debug_access+all \
	-o ${SIM_DIR}/tb_mnist_inference_core_board_simv \
	${RTL_DIR}/pkg/package.sv \
	${RTL_DIR}/mem/quantize_param_bank.sv \
	${RTL_DIR}/mem/fc1_weight_bank.sv \
	${RTL_DIR}/mem/fc1_bias_bank.sv \
	${RTL_DIR}/mem/mnist_sample_rom.sv \
	${RTL_DIR}/mem/fc2_weight_bank.sv \
	${RTL_DIR}/mem/fc2_bias_bank.sv \
	${RTL_DIR}/core/cim_tile.sv \
	${RTL_DIR}/core/psum_accum.sv \
	${RTL_DIR}/core/fc1_ob_engine_shared_input.sv \
	${RTL_DIR}/mem/fc1_multi_block_shared_sample_rom.sv \
	${RTL_DIR}/core/fc1_relu_requantize_with_file.sv \
	${RTL_DIR}/core/fc2_core_with_file.sv \
	${RTL_DIR}/core/fc1_to_fc2_top_with_file.sv \
	${RTL_DIR}/core/argmax_int8.sv \
	${RTL_DIR}/core/mnist_inference_core_board.sv \
	${TB_DIR}/tb_mnist_inference_core_board.sv \
	2>&1 | tee ${SIM_DIR}/log/compile_tb_mnist_inference_core_board.log

${SIM_DIR}/tb_mnist_inference_core_board_simv \
	2>&1 | tee ${SIM_DIR}/log/sim_tb_mnist_inference_core_board.log
