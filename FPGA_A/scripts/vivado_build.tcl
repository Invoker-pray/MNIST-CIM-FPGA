
# ============================================================
# Vivado batch build script for FPGA_A
#
# Usage:
#   vivado -mode batch -source scripts/vivado_build.tcl -tclargs <ROOT_DIR> <VIVADO_DIR>
# ============================================================

if { $argc < 2 } {
    puts "ERROR: need 2 args: <ROOT_DIR> <VIVADO_DIR>"
    exit 1
}

set ROOT_DIR   [file normalize [lindex $argv 0]]
set VIVADO_DIR [file normalize [lindex $argv 1]]

set PROJ_NAME  onboard_A
set PROJ_DIR   [file join $VIVADO_DIR $PROJ_NAME]
set PART_NAME  xc7z020clg400-1
set TOP_NAME   mnist_cim_demo_a_top

puts "============================================================"
puts "ROOT_DIR   = $ROOT_DIR"
puts "VIVADO_DIR = $VIVADO_DIR"
puts "PROJ_DIR   = $PROJ_DIR"
puts "PART_NAME  = $PART_NAME"
puts "TOP_NAME   = $TOP_NAME"
puts "============================================================"

file mkdir $VIVADO_DIR

if {[file exists $PROJ_DIR]} {
    puts "INFO: deleting old project dir: $PROJ_DIR"
    file delete -force $PROJ_DIR
}

create_project $PROJ_NAME $PROJ_DIR -part $PART_NAME

# ------------------------------------------------------------
# Source management
# ------------------------------------------------------------
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]
set_property source_mgmt_mode All [current_project]

# ------------------------------------------------------------
# Add design sources
# ------------------------------------------------------------

# pkg
add_files -norecurse [file join $ROOT_DIR rtl pkg package.sv]

# ctrl
add_files -norecurse [file join $ROOT_DIR rtl ctrl debounce.sv]
add_files -norecurse [file join $ROOT_DIR rtl ctrl onepulse.sv]

# mem
add_files -norecurse [file join $ROOT_DIR rtl mem fc1_bias_bank.sv]
add_files -norecurse [file join $ROOT_DIR rtl mem fc1_multi_block_shared_sample_rom.sv]
add_files -norecurse [file join $ROOT_DIR rtl mem fc1_weight_bank.sv]
add_files -norecurse [file join $ROOT_DIR rtl mem fc2_bias_bank.sv]
add_files -norecurse [file join $ROOT_DIR rtl mem fc2_weight_bank.sv]
add_files -norecurse [file join $ROOT_DIR rtl mem mnist_sample_rom.sv]
add_files -norecurse [file join $ROOT_DIR rtl mem quantize_param_bank.sv]

# core
add_files -norecurse [file join $ROOT_DIR rtl core argmax_int8.sv]
add_files -norecurse [file join $ROOT_DIR rtl core cim_tile.sv]
add_files -norecurse [file join $ROOT_DIR rtl core fc1_multi_block_shared_input.sv]
add_files -norecurse [file join $ROOT_DIR rtl core fc1_ob_engine_shared_input.sv]
add_files -norecurse [file join $ROOT_DIR rtl core fc1_relu_requantize_with_file.sv]
add_files -norecurse [file join $ROOT_DIR rtl core fc1_to_fc2_top_with_file.sv]
add_files -norecurse [file join $ROOT_DIR rtl core fc2_core_with_file.sv]
add_files -norecurse [file join $ROOT_DIR rtl core input_buffer.sv]
add_files -norecurse [file join $ROOT_DIR rtl core mnist_cim_accel_ip.sv]
add_files -norecurse [file join $ROOT_DIR rtl core mnist_inference_core_board.sv]
add_files -norecurse [file join $ROOT_DIR rtl core psum_accum.sv]

# uart
add_files -norecurse [file join $ROOT_DIR rtl uart uart_pred_sender.sv]
add_files -norecurse [file join $ROOT_DIR rtl uart uart_tx.sv]

# top
add_files -norecurse [file join $ROOT_DIR rtl top mnist_cim_demo_a_top.sv]

# constraints
add_files -fileset constrs_1 -norecurse [file join $ROOT_DIR constr top.xdc]

# ------------------------------------------------------------
# Add memory initialization/data files
# ------------------------------------------------------------
add_files -norecurse [file join $ROOT_DIR data quant quant_params.hex]
add_files -norecurse [file join $ROOT_DIR data weights fc1_weight_int8.hex]
add_files -norecurse [file join $ROOT_DIR data weights fc1_bias_int32.hex]
add_files -norecurse [file join $ROOT_DIR data weights fc2_weight_int8.hex]
add_files -norecurse [file join $ROOT_DIR data weights fc2_bias_int32.hex]
add_files -norecurse [file join $ROOT_DIR data samples mnist_samples_route_b_output_2.hex]

# Optional sample files
for {set i 0} {$i < 20} {incr i} {
    set f [file join $ROOT_DIR data samples [format "input_%d.hex" $i]]
    if {[file exists $f]} {
        add_files -norecurse $f
    }
}

# ------------------------------------------------------------
# Top and compile order
# ------------------------------------------------------------
set_property top $TOP_NAME [current_fileset]
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

# ------------------------------------------------------------
# Launch synthesis
# ------------------------------------------------------------
launch_runs synth_1 -jobs 8
wait_on_run synth_1

set synth_status [get_property STATUS [get_runs synth_1]]
puts "INFO: synth_1 status = $synth_status"

if {[string match "*ERROR*" $synth_status] || [string match "*failed*" $synth_status] || [string match "*Failed*" $synth_status]} {
    puts "ERROR: synthesis failed"
    exit 1
}

# ------------------------------------------------------------
# Launch implementation to bitstream
# ------------------------------------------------------------
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1

set impl_status [get_property STATUS [get_runs impl_1]]
puts "INFO: impl_1 status = $impl_status"

if {[string match "*ERROR*" $impl_status] || [string match "*failed*" $impl_status] || [string match "*Failed*" $impl_status]} {
    puts "ERROR: implementation failed"
    exit 1
}

# ------------------------------------------------------------
# Reports
# ------------------------------------------------------------
open_run synth_1
report_utilization    -file [file join $VIVADO_DIR synth_utilization.rpt]
report_timing_summary -file [file join $VIVADO_DIR synth_timing_summary.rpt]

open_run impl_1
report_utilization    -file [file join $VIVADO_DIR impl_utilization.rpt]
report_timing_summary -file [file join $VIVADO_DIR impl_timing_summary.rpt]
report_io             -file [file join $VIVADO_DIR impl_io.rpt]
report_drc            -file [file join $VIVADO_DIR impl_drc.rpt]

# ------------------------------------------------------------
# Copy bitstream to a stable top-level path
# ------------------------------------------------------------
set bitfile_src [file join $PROJ_DIR "${PROJ_NAME}.runs" "impl_1" "${TOP_NAME}.bit"]
set bitfile_dst [file join $VIVADO_DIR "${TOP_NAME}.bit"]

if {[file exists $bitfile_src]} {
    file copy -force $bitfile_src $bitfile_dst
    puts "INFO: bitstream copied to: $bitfile_dst"
} else {
    puts "WARN: bitstream not found at: $bitfile_src"
}

puts "============================================================"
puts "DONE: Vivado batch build finished"
puts "Project dir : $PROJ_DIR"
puts "Bitstream   : $bitfile_dst"
puts "Reports dir : $VIVADO_DIR"
puts "============================================================"

exit
