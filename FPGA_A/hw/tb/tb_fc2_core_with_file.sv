
`timescale 1ns / 1ps

module tb_fc2_core_with_file;
  import mnist_cim_pkg::*;

  parameter string DEFAULT_WEIGHT_HEX_FILE = "../data_packed/weights/fc2_weight_int8.hex";
  parameter string DEFAULT_BIAS_HEX_FILE = "../data_packed/weights/fc2_bias_int32.hex";

  logic clk;
  logic rst_n;
  logic start;

  logic [31:0] fc2_requant_mult;
  logic [31:0] fc2_requant_shift;

  logic signed [OUTPUT_WIDTH-1:0] fc1_out_all[0:FC2_IN_DIM-1];

  logic busy, done;
  logic signed [PSUM_WIDTH-1:0] fc2_acc_all[0:FC2_OUT_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] logits_all[0:FC2_OUT_DIM-1];

  logic signed [WEIGHT_WIDTH-1:0] golden_weight_mem[0:FC2_WEIGHT_DEPTH-1];
  logic signed [BIAS_WIDTH-1:0] golden_bias_mem[0:FC2_OUT_DIM-1];

  logic signed [PSUM_WIDTH-1:0] golden_fc2_acc[0:FC2_OUT_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] golden_logits[0:FC2_OUT_DIM-1];

  integer o, i;
  integer err_count;
  longint signed tmp_acc;
  longint signed prod;
  longint signed shifted;

  fc2_core_with_file #(
      .DEFAULT_WEIGHT_HEX_FILE(DEFAULT_WEIGHT_HEX_FILE),
      .DEFAULT_BIAS_HEX_FILE  (DEFAULT_BIAS_HEX_FILE)
  ) dut (
      .clk              (clk),
      .rst_n            (rst_n),
      .start            (start),
      .fc2_requant_mult (fc2_requant_mult),
      .fc2_requant_shift(fc2_requant_shift),
      .fc1_out_all      (fc1_out_all),
      .busy             (busy),
      .done             (done),
      .fc2_acc_all      (fc2_acc_all),
      .logits_all       (logits_all)
  );

  initial clk = 1'b0;
  always #5 clk = ~clk;

  function automatic signed [OUTPUT_WIDTH-1:0] golden_requant(
      input logic signed [PSUM_WIDTH-1:0] x, input logic [31:0] mult, input logic [31:0] rshift);
    longint signed f_prod;
    longint signed f_shifted;
    begin
      f_prod = x * $signed(mult);
      if (rshift == 0) f_shifted = f_prod;
      else f_shifted = (f_prod + (64'sd1 <<< (rshift - 1))) >>> rshift;

      if (f_shifted > 127) golden_requant = 8'sd127;
      else if (f_shifted < -128) golden_requant = -8'sd128;
      else golden_requant = f_shifted[OUTPUT_WIDTH-1:0];
    end
  endfunction

  initial begin
    $display("============================================================");
    $display("tb_fc2_core_with_file");
    $display("DEFAULT_WEIGHT_HEX_FILE = %s", DEFAULT_WEIGHT_HEX_FILE);
    $display("DEFAULT_BIAS_HEX_FILE   = %s", DEFAULT_BIAS_HEX_FILE);
    $display("============================================================");

    $readmemh(DEFAULT_WEIGHT_HEX_FILE, golden_weight_mem);
    $readmemh(DEFAULT_BIAS_HEX_FILE, golden_bias_mem);

    err_count = 0;
    rst_n     = 1'b0;
    start     = 1'b0;

    // give deterministic fc1_out_all
    for (i = 0; i < FC2_IN_DIM; i = i + 1) begin
      fc1_out_all[i] = (i % 11) - 5;  // values in a small signed range
    end

    // use a simple requant setting for standalone test
    fc2_requant_mult  = 32'd1;
    fc2_requant_shift = 32'd0;

    // golden
    for (o = 0; o < FC2_OUT_DIM; o = o + 1) begin
      tmp_acc = golden_bias_mem[o];
      for (i = 0; i < FC2_IN_DIM; i = i + 1) begin
        tmp_acc = tmp_acc + $signed(fc1_out_all[i]) * $signed(golden_weight_mem[o*FC2_IN_DIM+i]);
      end
      golden_fc2_acc[o] = tmp_acc[PSUM_WIDTH-1:0];
      golden_logits[o]  = golden_requant(golden_fc2_acc[o], fc2_requant_mult, fc2_requant_shift);
    end

    @(posedge clk);
    rst_n <= 1'b1;

    @(posedge clk);
    start <= 1'b1;

    @(posedge clk);
    start <= 1'b0;

    wait (done == 1'b1);
    #1;

    $display("[CHECK] compare fc2_acc_all / logits_all");
    for (o = 0; o < FC2_OUT_DIM; o = o + 1) begin
      if (fc2_acc_all[o] !== golden_fc2_acc[o]) begin
        $display("ERROR ACC o=%0d got=%0d exp=%0d", o, fc2_acc_all[o], golden_fc2_acc[o]);
        err_count = err_count + 1;
      end

      if (logits_all[o] !== golden_logits[o]) begin
        $display("ERROR LOGIT o=%0d got=%0d exp=%0d", o, logits_all[o], golden_logits[o]);
        err_count = err_count + 1;
      end
    end

    if (err_count == 0) begin
      $display("PASS: fc2_core_with_file sequential BRAM-style MAC is correct.");
    end else begin
      $display("FAIL: fc2_core_with_file found %0d mismatches.", err_count);
    end

    $finish;
  end

endmodule
