
`timescale 1ns / 1ps

module tb_fc1_to_fc2_top_with_file;
  import mnist_cim_pkg::*;

  parameter string DEFAULT_QUANT_PARAM_FILE = "../data_packed/quant/quant_params.hex";
  parameter string DEFAULT_FC2_WEIGHT_HEX_FILE = "../data_packed/weights/fc2_weight_int8.hex";
  parameter string DEFAULT_FC2_BIAS_HEX_FILE = "../data_packed/weights/fc2_bias_int32.hex";

  logic clk;
  logic rst_n;
  logic start;

  logic busy, done;

  logic signed [PSUM_WIDTH-1:0] fc1_acc_all[0:HIDDEN_DIM-1];
  logic signed [PSUM_WIDTH-1:0] fc1_relu_all[0:HIDDEN_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] fc1_out_all[0:HIDDEN_DIM-1];
  logic signed [PSUM_WIDTH-1:0] fc2_acc_all[0:FC2_OUT_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] logits_all[0:FC2_OUT_DIM-1];

  logic [31:0] quant_mem[0:3];
  logic signed [WEIGHT_WIDTH-1:0] golden_w[0:FC2_WEIGHT_DEPTH-1];
  logic signed [BIAS_WIDTH-1:0] golden_b[0:FC2_OUT_DIM-1];

  logic [31:0] fc1_requant_mult;
  logic [31:0] fc1_requant_shift;
  logic [31:0] fc2_requant_mult;
  logic [31:0] fc2_requant_shift;

  logic signed [PSUM_WIDTH-1:0] golden_fc1_relu[0:HIDDEN_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] golden_fc1_out[0:HIDDEN_DIM-1];
  logic signed [PSUM_WIDTH-1:0] golden_fc2_acc[0:FC2_OUT_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] golden_logits[0:FC2_OUT_DIM-1];

  integer i, o;
  integer err_count;
  longint signed prod, shifted, acc_tmp;

  fc1_to_fc2_top_with_file #(
      .DEFAULT_QUANT_PARAM_FILE   (DEFAULT_QUANT_PARAM_FILE),
      .DEFAULT_FC2_WEIGHT_HEX_FILE(DEFAULT_FC2_WEIGHT_HEX_FILE),
      .DEFAULT_FC2_BIAS_HEX_FILE  (DEFAULT_FC2_BIAS_HEX_FILE)
  ) dut (
      .clk         (clk),
      .rst_n       (rst_n),
      .start       (start),
      .fc1_acc_all (fc1_acc_all),
      .busy        (busy),
      .done        (done),
      .fc1_relu_all(fc1_relu_all),
      .fc1_out_all (fc1_out_all),
      .fc2_acc_all (fc2_acc_all),
      .logits_all  (logits_all)
  );

  initial clk = 1'b0;
  always #5 clk = ~clk;

  function automatic signed [OUTPUT_WIDTH-1:0] requant(
      input logic signed [PSUM_WIDTH-1:0] x, input logic [31:0] mult, input logic [31:0] rshift);
    longint signed f_prod, f_shifted;
    begin
      f_prod = x * $signed(mult);
      if (rshift == 0) f_shifted = f_prod;
      else f_shifted = (f_prod + (64'sd1 <<< (rshift - 1))) >>> rshift;

      if (f_shifted > 127) requant = 8'sd127;
      else if (f_shifted < -128) requant = -8'sd128;
      else requant = f_shifted[OUTPUT_WIDTH-1:0];
    end
  endfunction

  initial begin
    $display("============================================================");
    $display("tb_fc1_to_fc2_top_with_file");
    $display("DEFAULT_QUANT_PARAM_FILE    = %s", DEFAULT_QUANT_PARAM_FILE);
    $display("DEFAULT_FC2_WEIGHT_HEX_FILE = %s", DEFAULT_FC2_WEIGHT_HEX_FILE);
    $display("DEFAULT_FC2_BIAS_HEX_FILE   = %s", DEFAULT_FC2_BIAS_HEX_FILE);
    $display("============================================================");

    $readmemh(DEFAULT_QUANT_PARAM_FILE, quant_mem);
    $readmemh(DEFAULT_FC2_WEIGHT_HEX_FILE, golden_w);
    $readmemh(DEFAULT_FC2_BIAS_HEX_FILE, golden_b);

    fc1_requant_mult = quant_mem[0];
    fc1_requant_shift = quant_mem[1];
    fc2_requant_mult = quant_mem[2];
    fc2_requant_shift = quant_mem[3];

    err_count = 0;
    rst_n = 1'b0;
    start = 1'b0;

    // deterministic FC1 accumulators
    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
      fc1_acc_all[i] = (i % 17) - 8;
    end

    // golden FC1 relu + requant
    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
      if (fc1_acc_all[i] < 0) golden_fc1_relu[i] = 0;
      else golden_fc1_relu[i] = fc1_acc_all[i];

      golden_fc1_out[i] = requant(golden_fc1_relu[i], fc1_requant_mult, fc1_requant_shift);
    end

    // golden FC2
    for (o = 0; o < FC2_OUT_DIM; o = o + 1) begin
      acc_tmp = golden_b[o];
      for (i = 0; i < FC2_IN_DIM; i = i + 1) begin
        acc_tmp = acc_tmp + $signed(golden_fc1_out[i]) * $signed(golden_w[o*FC2_IN_DIM+i]);
      end
      golden_fc2_acc[o] = acc_tmp[PSUM_WIDTH-1:0];
      golden_logits[o]  = requant(golden_fc2_acc[o], fc2_requant_mult, fc2_requant_shift);
    end

    @(posedge clk);
    rst_n <= 1'b1;

    @(posedge clk);
    start <= 1'b1;

    @(posedge clk);
    start <= 1'b0;

    wait (done == 1'b1);
    #1;

    $display("[CHECK] compare fc1_out_all / fc2_acc_all / logits_all");

    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
      if (fc1_out_all[i] !== golden_fc1_out[i]) begin
        $display("ERROR FC1_OUT i=%0d got=%0d exp=%0d", i, fc1_out_all[i], golden_fc1_out[i]);
        err_count = err_count + 1;
      end
    end

    for (o = 0; o < FC2_OUT_DIM; o = o + 1) begin
      if (fc2_acc_all[o] !== golden_fc2_acc[o]) begin
        $display("ERROR FC2_ACC o=%0d got=%0d exp=%0d", o, fc2_acc_all[o], golden_fc2_acc[o]);
        err_count = err_count + 1;
      end

      if (logits_all[o] !== golden_logits[o]) begin
        $display("ERROR LOGIT o=%0d got=%0d exp=%0d", o, logits_all[o], golden_logits[o]);
        err_count = err_count + 1;
      end
    end

    if (err_count == 0) begin
      $display("PASS: fc1_to_fc2_top_with_file integrated FC1 requant + FC2 path is correct.");
    end else begin
      $display("FAIL: fc1_to_fc2_top_with_file found %0d mismatches.", err_count);
    end

    $finish;
  end

endmodule
