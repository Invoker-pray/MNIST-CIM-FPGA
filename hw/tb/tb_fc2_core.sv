
`timescale 1ns / 1ps

module tb_fc2_core;
  import mnist_cim_pkg::*;

  logic signed [OUTPUT_WIDTH-1:0] fc1_out_all[0:FC2_IN_DIM-1];
  logic signed [PSUM_WIDTH-1:0] fc2_acc_all[0:FC2_OUT_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] logits_all[0:FC2_OUT_DIM-1];

  logic signed [OUTPUT_WIDTH-1:0] ref_fc1_out_mem[0:FC2_IN_DIM-1];
  logic signed [PSUM_WIDTH-1:0] ref_fc2_acc_mem[0:FC2_OUT_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] ref_logits_mem[0:FC2_OUT_DIM-1];

  string fc1_out_file;
  string fc2_acc_file;
  string logits_file;

  integer i;
  integer error_count;

  fc2_core dut (
      .fc1_out_all(fc1_out_all),
      .fc2_acc_all(fc2_acc_all),
      .logits_all (logits_all)
  );

  initial begin
    fc1_out_file = "../route_b_output/fc1_out_0.hex";
    fc2_acc_file = "../route_b_output/fc2_acc_0.hex";
    logits_file  = "../route_b_output/logits_0.hex";

    if ($value$plusargs("FC1_OUT_FILE=%s", fc1_out_file))
      $display("TB using FC1_OUT_FILE: %s", fc1_out_file);
    else $display("TB using default FC1_OUT_FILE: %s", fc1_out_file);

    if ($value$plusargs("FC2_ACC_FILE=%s", fc2_acc_file))
      $display("TB using FC2_ACC_FILE: %s", fc2_acc_file);
    else $display("TB using default FC2_ACC_FILE: %s", fc2_acc_file);

    if ($value$plusargs("LOGITS_FILE=%s", logits_file))
      $display("TB using LOGITS_FILE: %s", logits_file);
    else $display("TB using default LOGITS_FILE: %s", logits_file);

    $readmemh(fc1_out_file, ref_fc1_out_mem);
    $readmemh(fc2_acc_file, ref_fc2_acc_mem);
    $readmemh(logits_file, ref_logits_mem);
  end

  initial begin
    error_count = 0;

    for (i = 0; i < FC2_IN_DIM; i = i + 1) fc1_out_all[i] = ref_fc1_out_mem[i];

    #1;

    $display("Checking fc2_acc_all ...");
    for (i = 0; i < FC2_OUT_DIM; i = i + 1) begin
      if (fc2_acc_all[i] !== ref_fc2_acc_mem[i]) begin
        $display("ERROR ACC idx=%0d got=%0d expected=%0d", i, fc2_acc_all[i], ref_fc2_acc_mem[i]);
        error_count = error_count + 1;
      end else begin
        $display("MATCH ACC idx=%0d value=%0d", i, fc2_acc_all[i]);
      end
    end

    $display("Checking logits_all ...");
    for (i = 0; i < FC2_OUT_DIM; i = i + 1) begin
      if (logits_all[i] !== ref_logits_mem[i]) begin
        $display("ERROR LOGIT idx=%0d got=%0d expected=%0d", i, logits_all[i], ref_logits_mem[i]);
        error_count = error_count + 1;
      end else begin
        $display("MATCH LOGIT idx=%0d value=%0d", i, logits_all[i]);
      end
    end

    if (error_count == 0) $display("PASS: fc2_core matches fc2_acc_0.hex and logits_0.hex.");
    else $display("FAIL: found %0d mismatches.", error_count);

    $finish;
  end

endmodule
