SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

mkdir -p ${SIM_DIR}
mkdir -p ${SIM_DIR}/log

vcs -full64 -sverilog -timescale=1ns/1ps \
	-debug_access+all \
	-o ${SIM_DIR}/psum_accum_simv \
	${RTL_DIR}/package.sv \
	${RTL_DIR}/psum_accum.sv \
	${TB_DIR}/tb_psum_accum.sv \
	2>&1 | tee ${SIM_DIR}/log/compile_tb_psum_accum.log

${SIM_DIR}/psum_accum_simv 2>&1 | tee ${SIM_DIR}/log/sim_tb_psum_accum.log
