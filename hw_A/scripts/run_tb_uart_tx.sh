#!/usr/bin/env bash
set -e

SIM_DIR=../sim
RTL_IP_DIR=../rtl_ip
TB_DIR=../tb

mkdir -p ${SIM_DIR}
mkdir -p ${SIM_DIR}/log

vcs -full64 -sverilog -timescale=1ns/1ps \
	-debug_access+all \
	-o ${SIM_DIR}/tb_uart_tx_simv \
	${RTL_IP_DIR}/uart_tx.sv \
	${TB_DIR}/tb_uart_tx.sv \
	2>&1 | tee ${SIM_DIR}/log/compile_tb_uart_tx.log

${SIM_DIR}/tb_uart_tx_simv \
	2>&1 | tee ${SIM_DIR}/log/sim_tb_uart_tx.log
