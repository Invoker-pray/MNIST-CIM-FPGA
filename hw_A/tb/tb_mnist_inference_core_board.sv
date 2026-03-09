
`timescale 1ns / 1ps

module tb_mnist_inference_core_board;
  import mnist_cim_pkg::*;

  parameter int N_SAMPLES = 20;
  parameter string SAMPLE_HEX_FILE = "../data/mnist_samples_route_b_output_2.hex";
  parameter string FC1_WEIGHT_HEX_FILE = "../route_b_output_2/fc1_weight_int8.hex";
  parameter string FC1_BIAS_HEX_FILE = "../route_b_output_2/fc1_bias_int32.hex";
  parameter string QUANT_PARAM_FILE = "../route_b_output_2/quant_params.hex";
  parameter string FC2_WEIGHT_HEX_FILE = "../route_b_output_2/fc2_weight_int8.hex";
  parameter string FC2_BIAS_HEX_FILE = "../route_b_output_2/fc2_bias_int32.hex";
  parameter string LOGITS_FILE = "../route_b_output_2/logits_0.hex";
  parameter string PRED_FILE = "../route_b_output_2/pred_0.txt";

  logic clk;
  logic rst_n;
  logic start;
  logic [$clog2(N_SAMPLES)-1:0] sample_id;
  logic busy;
  logic done;

  logic signed [OUTPUT_WIDTH-1:0] logits_all[0:FC2_OUT_DIM-1];
  logic [$clog2(FC2_OUT_DIM)-1:0] pred_class;

  logic signed [OUTPUT_WIDTH-1:0] ref_logits[0:FC2_OUT_DIM-1];
  integer ref_pred_class;

  integer i, fd, r;
  integer error_count;

  mnist_inference_core_board #(
      .N_SAMPLES(N_SAMPLES),
      .DEFAULT_SAMPLE_HEX_FILE(SAMPLE_HEX_FILE),
      .DEFAULT_FC1_WEIGHT_HEX_FILE(FC1_WEIGHT_HEX_FILE),
      .DEFAULT_FC1_BIAS_HEX_FILE(FC1_BIAS_HEX_FILE),
      .DEFAULT_QUANT_PARAM_FILE(QUANT_PARAM_FILE),
      .DEFAULT_FC2_WEIGHT_HEX_FILE(FC2_WEIGHT_HEX_FILE),
      .DEFAULT_FC2_BIAS_HEX_FILE(FC2_BIAS_HEX_FILE)
  ) dut (
      .clk(clk),
      .rst_n(rst_n),
      .start(start),
      .sample_id(sample_id),
      .busy(busy),
      .done(done),
      .logits_all(logits_all),
      .pred_class(pred_class)
  );

  initial clk = 1'b0;
  always #5 clk = ~clk;

  initial begin
    $readmemh(LOGITS_FILE, ref_logits);

    fd = $fopen(PRED_FILE, "r");
    if (fd == 0) begin
      $display("ERROR: cannot open pred file.");
      $finish;
    end
    r = $fscanf(fd, "%d", ref_pred_class);
    $fclose(fd);
    if (r != 1) begin
      $display("ERROR: failed to parse pred file.");
      $finish;
    end
  end

  initial begin
    error_count = 0;
    rst_n = 1'b0;
    start = 1'b0;
    sample_id = '0;

    #12;
    rst_n = 1'b1;

    @(posedge clk);
    start <= 1'b1;
    @(posedge clk);
    start <= 1'b0;

    wait (done == 1'b1);
    @(posedge clk);
    #1;

    for (i = 0; i < FC2_OUT_DIM; i = i + 1) begin
      if (logits_all[i] !== ref_logits[i]) begin
        $display("ERROR LOGIT idx=%0d got=%0d expected=%0d", i, logits_all[i], ref_logits[i]);
        error_count = error_count + 1;
      end
    end

    if (pred_class !== ref_pred_class[$clog2(FC2_OUT_DIM)-1:0]) begin
      $display("ERROR PRED got=%0d expected=%0d", pred_class, ref_pred_class);
      error_count = error_count + 1;
    end

    if (error_count == 0) $display("PASS: mnist_inference_core_board matches sample0 logits/pred.");
    else $display("FAIL: found %0d mismatches.", error_count);

    $finish;
  end

endmodule
