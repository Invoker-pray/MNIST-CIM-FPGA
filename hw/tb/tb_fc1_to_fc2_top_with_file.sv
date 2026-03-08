
`timescale 1ns / 1ps

module tb_fc1_to_fc2_top_quant_file;
  import mnist_cim_pkg::*;

  logic signed [PSUM_WIDTH-1:0] fc1_acc_all[0:HIDDEN_DIM-1];
  logic signed [PSUM_WIDTH-1:0] fc1_relu_all[0:HIDDEN_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] fc1_out_all[0:HIDDEN_DIM-1];
  logic signed [PSUM_WIDTH-1:0] fc2_acc_all[0:FC2_OUT_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] logits_all[0:FC2_OUT_DIM-1];

  logic signed [PSUM_WIDTH-1:0] ref_fc1_acc_mem[0:HIDDEN_DIM-1];
  logic signed [PSUM_WIDTH-1:0] ref_fc1_relu_mem[0:HIDDEN_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] ref_fc1_out_mem[0:HIDDEN_DIM-1];
  logic signed [PSUM_WIDTH-1:0] ref_fc2_acc_mem[0:FC2_OUT_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] ref_logits_mem[0:FC2_OUT_DIM-1];

  string fc1_acc_file;
  string fc1_relu_file;
  string fc1_out_file;
  string fc2_acc_file;
  string logits_file;
  string quant_param_file;

  integer i;
  integer error_count;

  fc1_to_fc2_top dut (
      .fc1_acc_all (fc1_acc_all),
      .fc1_relu_all(fc1_relu_all),
      .fc1_out_all (fc1_out_all),
      .fc2_acc_all (fc2_acc_all),
      .logits_all  (logits_all)
  );

  initial begin
    fc1_acc_file     = "../route_b_output/fc1_acc_0.hex";
    fc1_relu_file    = "../route_b_output/fc1_relu_0.hex";
    fc1_out_file     = "../route_b_output/fc1_out_0.hex";
    fc2_acc_file     = "../route_b_output/fc2_acc_0.hex";
    logits_file      = "../route_b_output/logits_0.hex";
    quant_param_file = "../route_b_output/quant_params.hex";

    if ($value$plusargs("FC1_ACC_FILE=%s", fc1_acc_file))
      $display("TB using FC1_ACC_FILE: %s", fc1_acc_file);
    if ($value$plusargs("FC1_RELU_FILE=%s", fc1_relu_file))
      $display("TB using FC1_RELU_FILE: %s", fc1_relu_file);
    if ($value$plusargs("FC1_OUT_FILE=%s", fc1_out_file))
      $display("TB using FC1_OUT_FILE: %s", fc1_out_file);
    if ($value$plusargs("FC2_ACC_FILE=%s", fc2_acc_file))
      $display("TB using FC2_ACC_FILE: %s", fc2_acc_file);
    if ($value$plusargs("LOGITS_FILE=%s", logits_file))
      $display("TB using LOGITS_FILE: %s", logits_file);
    if ($value$plusargs("QUANT_PARAM_FILE=%s", quant_param_file))
      $display("TB using QUANT_PARAM_FILE: %s", quant_param_file);

    $readmemh(fc1_acc_file, ref_fc1_acc_mem);
    $readmemh(fc1_relu_file, ref_fc1_relu_mem);
    $readmemh(fc1_out_file, ref_fc1_out_mem);
    $readmemh(fc2_acc_file, ref_fc2_acc_mem);
    $readmemh(logits_file, ref_logits_mem);
  end

  initial begin
    error_count = 0;

    for (i = 0; i < HIDDEN_DIM; i = i + 1) fc1_acc_all[i] = ref_fc1_acc_mem[i];

    #1;

    $display("Checking fc1_relu_all ...");
    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
      if (fc1_relu_all[i] !== ref_fc1_relu_mem[i]) begin
        $display("ERROR RELU idx=%0d got=%0d expected=%0d", i, fc1_relu_all[i],
                 ref_fc1_relu_mem[i]);
        error_count = error_count + 1;
      end
    end

    $display("Checking fc1_out_all ...");
    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
      if (fc1_out_all[i] !== ref_fc1_out_mem[i]) begin
        $display("ERROR OUT idx=%0d got=%0d expected=%0d", i, fc1_out_all[i], ref_fc1_out_mem[i]);
        error_count = error_count + 1;
      end
    end

    $display("Checking fc2_acc_all ...");
    for (i = 0; i < FC2_OUT_DIM; i = i + 1) begin
      if (fc2_acc_all[i] !== ref_fc2_acc_mem[i]) begin
        $display("ERROR ACC idx=%0d got=%0d expected=%0d", i, fc2_acc_all[i], ref_fc2_acc_mem[i]);
        error_count = error_count + 1;
      end
    end

    $display("Checking logits_all ...");
    for (i = 0; i < FC2_OUT_DIM; i = i + 1) begin
      if (logits_all[i] !== ref_logits_mem[i]) begin
        $display("ERROR LOGIT idx=%0d got=%0d expected=%0d", i, logits_all[i], ref_logits_mem[i]);
        error_count = error_count + 1;
      end
    end

    if (error_count == 0)
      $display("PASS: fc1_to_fc2_top matches provided golden files using quant_params.hex.");
    else $display("FAIL: found %0d mismatches.", error_count);

    $finish;
  end

endmodule
