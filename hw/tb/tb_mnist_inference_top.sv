
`timescale 1ns / 1ps

module tb_mnist_inference_top;
  import mnist_cim_pkg::*;

  logic clk;
  logic rst_n;
  logic start;
  logic busy;
  logic done;

  logic signed [PSUM_WIDTH-1:0] fc1_acc_all[0:HIDDEN_DIM-1];
  logic signed [PSUM_WIDTH-1:0] fc1_relu_all[0:HIDDEN_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] fc1_out_all[0:HIDDEN_DIM-1];
  logic signed [PSUM_WIDTH-1:0] fc2_acc_all[0:FC2_OUT_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] logits_all[0:FC2_OUT_DIM-1];
  logic [$clog2(FC2_OUT_DIM)-1:0] pred_class;

  logic signed [OUTPUT_WIDTH-1:0] ref_logits_mem[0:FC2_OUT_DIM-1];
  integer ref_pred_class;

  string input_hex_file;
  string fc1_weight_hex_file;
  string fc1_bias_hex_file;
  string fc2_weight_hex_file;
  string fc2_bias_hex_file;
  string logits_file;
  string pred_file;

  integer i;
  integer fd;
  integer r;
  integer error_count;

  mnist_inference_top dut (
      .clk(clk),
      .rst_n(rst_n),
      .start(start),
      .busy(busy),
      .done(done),
      .fc1_acc_all(fc1_acc_all),
      .fc1_relu_all(fc1_relu_all),
      .fc1_out_all(fc1_out_all),
      .fc2_acc_all(fc2_acc_all),
      .logits_all(logits_all),
      .pred_class(pred_class)
  );

  initial clk = 1'b0;
  always #5 clk = ~clk;

  initial begin
    input_hex_file      = "../route_b_output/input_0.hex";
    fc1_weight_hex_file = "../route_b_output/fc1_weight_int8.hex";
    fc1_bias_hex_file   = "../route_b_output/fc1_bias_int32.hex";
    fc2_weight_hex_file = "../route_b_output/fc2_weight_int8.hex";
    fc2_bias_hex_file   = "../route_b_output/fc2_bias_int32.hex";
    logits_file         = "../route_b_output/logits_0.hex";
    pred_file           = "../route_b_output/pred_0.txt";

    if ($value$plusargs("INPUT_HEX_FILE=%s", input_hex_file))
      $display("TB using INPUT_HEX_FILE: %s", input_hex_file);
    else $display("TB using default INPUT_HEX_FILE: %s", input_hex_file);

    if ($value$plusargs("WEIGHT_HEX_FILE=%s", fc1_weight_hex_file))
      $display("TB using WEIGHT_HEX_FILE: %s", fc1_weight_hex_file);
    else $display("TB using default WEIGHT_HEX_FILE: %s", fc1_weight_hex_file);

    if ($value$plusargs("FC1_BIAS_FILE=%s", fc1_bias_hex_file))
      $display("TB using FC1_BIAS_FILE: %s", fc1_bias_hex_file);
    else $display("TB using default FC1_BIAS_FILE: %s", fc1_bias_hex_file);

    if ($value$plusargs("FC2_WEIGHT_HEX_FILE=%s", fc2_weight_hex_file))
      $display("TB using FC2_WEIGHT_HEX_FILE: %s", fc2_weight_hex_file);
    else $display("TB using default FC2_WEIGHT_HEX_FILE: %s", fc2_weight_hex_file);

    if ($value$plusargs("FC2_BIAS_FILE=%s", fc2_bias_hex_file))
      $display("TB using FC2_BIAS_FILE: %s", fc2_bias_hex_file);
    else $display("TB using default FC2_BIAS_FILE: %s", fc2_bias_hex_file);

    if ($value$plusargs("LOGITS_FILE=%s", logits_file))
      $display("TB using LOGITS_FILE: %s", logits_file);
    else $display("TB using default LOGITS_FILE: %s", logits_file);

    if ($value$plusargs("PRED_FILE=%s", pred_file)) $display("TB using PRED_FILE: %s", pred_file);
    else $display("TB using default PRED_FILE: %s", pred_file);

    $readmemh(logits_file, ref_logits_mem);

    fd = $fopen(pred_file, "r");
    if (fd == 0) begin
      $display("ERROR: cannot open pred file: %s", pred_file);
      $finish;
    end
    r = $fscanf(fd, "%d", ref_pred_class);
    $fclose(fd);

    if (r != 1) begin
      $display("ERROR: failed to parse pred file: %s", pred_file);
      $finish;
    end
  end

  initial begin
    error_count = 0;
    rst_n = 1'b0;
    start = 1'b0;

    #12;
    rst_n = 1'b1;

    @(posedge clk);
    start <= 1'b1;

    @(posedge clk);
    start <= 1'b0;

    wait (done == 1'b1);
    @(posedge clk);
    #1;

    $display("Checking logits_all ...");
    for (i = 0; i < FC2_OUT_DIM; i = i + 1) begin
      if (logits_all[i] !== ref_logits_mem[i]) begin
        $display("ERROR LOGIT idx=%0d got=%0d expected=%0d", i, logits_all[i], ref_logits_mem[i]);
        error_count = error_count + 1;
      end else begin
        $display("MATCH LOGIT idx=%0d value=%0d", i, logits_all[i]);
      end
    end

    $display("Checking pred_class ...");
    if (pred_class !== ref_pred_class[$clog2(FC2_OUT_DIM)-1:0]) begin
      $display("ERROR PRED got=%0d expected=%0d", pred_class, ref_pred_class);
      error_count = error_count + 1;
    end else begin
      $display("MATCH PRED value=%0d", pred_class);
    end

    if (error_count == 0)
      $display("PASS: mnist_inference_top matches logits_0.hex and pred_0.txt.");
    else $display("FAIL: found %0d mismatches.", error_count);

    $finish;
  end

endmodule
