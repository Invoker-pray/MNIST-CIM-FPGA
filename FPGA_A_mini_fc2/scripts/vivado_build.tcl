

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
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]
set_property source_mgmt_mode All [current_project]

# ------------------------------------------------------------
# Collect source files
# ------------------------------------------------------------
set rtl_files [list \
    [file join $ROOT_DIR rtl pkg package.sv] \
    [file join $ROOT_DIR rtl ctrl debounce.sv] \
    [file join $ROOT_DIR rtl ctrl onepulse.sv] \
    [file join $ROOT_DIR rtl mem fc1_bias_bank.sv] \
    [file join $ROOT_DIR rtl mem fc1_multi_block_shared_sample_rom.sv] \
    [file join $ROOT_DIR rtl mem fc1_weight_bank.sv] \
    [file join $ROOT_DIR rtl mem fc2_bias_bank.sv] \
    [file join $ROOT_DIR rtl mem fc2_weight_bank.sv] \
    [file join $ROOT_DIR rtl mem mnist_sample_rom.sv] \
    [file join $ROOT_DIR rtl mem quantize_param_bank.sv] \
    [file join $ROOT_DIR rtl core argmax_int8.sv] \
    [file join $ROOT_DIR rtl core cim_tile.sv] \
    [file join $ROOT_DIR rtl core fc1_multi_block_shared_input.sv] \
    [file join $ROOT_DIR rtl core fc1_ob_engine_shared_input.sv] \
    [file join $ROOT_DIR rtl core fc1_relu_requantize_with_file.sv] \
    [file join $ROOT_DIR rtl core fc1_to_fc2_top_with_file.sv] \
    [file join $ROOT_DIR rtl core fc2_core_with_file.sv] \
    [file join $ROOT_DIR rtl core input_buffer.sv] \
    [file join $ROOT_DIR rtl core mnist_cim_accel_ip.sv] \
    [file join $ROOT_DIR rtl core mnist_inference_core_board.sv] \
    [file join $ROOT_DIR rtl core psum_accum.sv] \
    [file join $ROOT_DIR rtl uart uart_pred_sender.sv] \
    [file join $ROOT_DIR rtl uart uart_tx.sv] \
    [file join $ROOT_DIR rtl top mnist_cim_demo_a_top.sv] \
]

set xdc_files [list \
    [file join $ROOT_DIR constr top.xdc] \
]

set mem_init_files [list \
    [file join $ROOT_DIR data weights fc1_weight_int8.hex] \
    [file join $ROOT_DIR data weights fc1_bias_int32.hex] \
    [file join $ROOT_DIR data samples mnist_samples_route_b_output_2.hex] \
    [file join $ROOT_DIR data quant quant_params.hex] \
    [file join $ROOT_DIR data weights fc2_weight_int8.hex] \
    [file join $ROOT_DIR data weights fc2_bias_int32.hex] \
]

# Optional extra sample files
for {set i 0} {$i < 20} {incr i} {
    set f [file join $ROOT_DIR data samples [format "input_%d.hex" $i]]
    if {[file exists $f]} {
        lappend mem_init_files $f
    }
}

# ------------------------------------------------------------
# Sanity check: required files must exist
# ------------------------------------------------------------
foreach f $rtl_files {
    if {![file exists $f]} {
        puts "ERROR: missing RTL file: $f"
        exit 1
    }
}
foreach f $xdc_files {
    if {![file exists $f]} {
        puts "ERROR: missing XDC file: $f"
        exit 1
    }
}
foreach f $mem_init_files {
    if {![file exists $f]} {
        puts "ERROR: missing memory init file: $f"
        exit 1
    }
}

# ------------------------------------------------------------
# Add files to project
# ------------------------------------------------------------
add_files -norecurse {*}$rtl_files
add_files -fileset constrs_1 -norecurse {*}$xdc_files
add_files -norecurse {*}$mem_init_files

set_property top $TOP_NAME [current_fileset]
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

# ------------------------------------------------------------
# Copy memory init files into run directories so $readmemh()
# can resolve bare filenames during synthesis/implementation
# ------------------------------------------------------------
set SYNTH_RUN_DIR [file join $PROJ_DIR "${PROJ_NAME}.runs" "synth_1"]
set IMPL_RUN_DIR  [file join $PROJ_DIR "${PROJ_NAME}.runs" "impl_1"]

file mkdir $SYNTH_RUN_DIR
file mkdir $IMPL_RUN_DIR

foreach f $mem_init_files {
    set base [file tail $f]
    file copy -force $f [file join $SYNTH_RUN_DIR $base]
    file copy -force $f [file join $IMPL_RUN_DIR  $base]
    puts "INFO: copied $base to synth_1 and impl_1 run dirs"
}

# Also copy into project root for extra robustness
foreach f $mem_init_files {
    set base [file tail $f]
    file copy -force $f [file join $PROJ_DIR $base]
    puts "INFO: copied $base to project dir"
}


# ------------------------------------------------------------
# Save / refresh project state
# ------------------------------------------------------------
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1
# ------------------------------------------------------------
# Launch synthesis
# ------------------------------------------------------------

set_property strategy Flow_RuntimeOptimized [get_runs synth_1]
launch_runs synth_1 -jobs 1
wait_on_run synth_1

set synth_status [get_property STATUS [get_runs synth_1]]
puts "INFO: synth_1 status = $synth_status"

if {[string match "*ERROR*" $synth_status] || [string match "*failed*" $synth_status] || [string match "*Failed*" $synth_status]} {
    puts "ERROR: synthesis failed"
    exit 1
}

open_run synth_1
write_checkpoint -force [file join $VIVADO_DIR post_synth.dcp]
report_utilization    -file [file join $VIVADO_DIR synth_utilization.rpt]
report_timing_summary -file [file join $VIVADO_DIR synth_timing_summary.rpt]

puts "DONE: synthesis finished"
exit
