SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

vcs -full64 -sverilog -timescale=1ns/1ps \
	-debug_access+all \
	-o ${SIM_DIR}/cim_tile_simv \
	${RTL_DIR}/package.sv \
	${RTL_DIR}/cim_tile.sv \
	${TB_DIR}/tb_cim_tile_single.sv \
	2>&1 | tee ${SIM_DIR}/log/compile_tb_cim_tile_single.log

${SIM_DIR}/cim_tile_simv 2>&1 | tee ${SIM_DIR}/log/sim_tb_cim_tile_single.log
