#!/usr/bin/env bash
set -e

SIM_DIR=../sim
RTL_DIR=../rtl
RTL_IP_DIR=../rtl_ip
RTL_SHARED_DIR=../rtl_shared_buffer_ib
TB_DIR=../tb

mkdir -p ${SIM_DIR}
mkdir -p ${SIM_DIR}/log

vcs -full64 -sverilog -timescale=1ns/1ps \
	-debug_access+all \
	-o ${SIM_DIR}/tb_mnist_cim_demo_a_top_simv \
	${RTL_DIR}/package.sv \
	${RTL_DIR}/quantize_param_bank.sv \
	${RTL_DIR}/fc1_relu_requantize_with_file.sv \
	${RTL_DIR}/fc2_weight_bank.sv \
	${RTL_DIR}/fc2_bias_bank.sv \
	${RTL_DIR}/fc2_core_with_file.sv \
	${RTL_DIR}/fc1_to_fc2_top_with_file.sv \
	${RTL_DIR}/argmax_int8.sv \
	${RTL_IP_DIR}/mnist_sample_rom.sv \
	${RTL_SHARED_DIR}/cim_tile.sv \
	${RTL_SHARED_DIR}/psum_accum.sv \
	${RTL_SHARED_DIR}/fc1_weight_bank.sv \
	${RTL_SHARED_DIR}/fc1_bias_bank.sv \
	${RTL_SHARED_DIR}/fc1_ob_engine_shared_input.sv \
	${RTL_IP_DIR}/fc1_multi_block_shared_sample_rom.sv \
	${RTL_IP_DIR}/mnist_inference_core_board.sv \
	${RTL_IP_DIR}/uart_tx.sv \
	${RTL_IP_DIR}/uart_pred_sender.sv \
	${RTL_IP_DIR}/mnist_cim_demo_a_top.sv \
	${TB_DIR}/tb_mnist_cim_demo_a_top.sv \
	2>&1 | tee ${SIM_DIR}/log/compile_tb_mnist_cim_demo_a_top.log

${SIM_DIR}/tb_mnist_cim_demo_a_top_simv \
	+WEIGHT_HEX_FILE=../route_b_output_2/fc1_weight_int8.hex \
	+FC1_BIAS_FILE=../route_b_output_2/fc1_bias_int32.hex \
	+FC2_WEIGHT_HEX_FILE=../route_b_output_2/fc2_weight_int8.hex \
	+FC2_BIAS_FILE=../route_b_output_2/fc2_bias_int32.hex \
	+QUANT_PARAM_FILE=../route_b_output_2/quant_params.hex \
	2>&1 | tee ${SIM_DIR}/log/sim_tb_mnist_cim_demo_a_top.log
