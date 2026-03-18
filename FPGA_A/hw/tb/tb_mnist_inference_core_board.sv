
`timescale 1ns / 1ps

module tb_mnist_inference_core_board;
  import mnist_cim_pkg::*;

  parameter int N_SAMPLES = 20;
  parameter string DEFAULT_SAMPLE_HEX_FILE     = "../data_packed/samples/mnist_samples_route_b_output_2.hex";
  parameter string DEFAULT_FC1_WEIGHT_HEX_FILE = "../data_packed/weights/fc1_weight_int8.hex";
  parameter string DEFAULT_FC1_BIAS_HEX_FILE = "../data_packed/weights/fc1_bias_int32.hex";
  parameter string DEFAULT_QUANT_PARAM_FILE = "../data_packed/quant/quant_params.hex";
  parameter string DEFAULT_FC2_WEIGHT_HEX_FILE = "../data_packed/weights/fc2_weight_int8.hex";
  parameter string DEFAULT_FC2_BIAS_HEX_FILE = "../data_packed/weights/fc2_bias_int32.hex";
  parameter string PRED_FILE = "../data_packed/expected/pred_0.txt";

  logic clk;
  logic rst_n;
  logic start;
  logic [$clog2(N_SAMPLES)-1:0] sample_id;

  logic busy, done;
  logic signed [OUTPUT_WIDTH-1:0] logits_all[0:FC2_OUT_DIM-1];
  logic [$clog2(FC2_OUT_DIM)-1:0] pred_class;

  integer fd, r;
  integer ref_pred_class;
  integer err_count;
  integer o;

  mnist_inference_core_board #(
      .N_SAMPLES(N_SAMPLES),
      .DEFAULT_SAMPLE_HEX_FILE(DEFAULT_SAMPLE_HEX_FILE),
      .DEFAULT_FC1_WEIGHT_HEX_FILE(DEFAULT_FC1_WEIGHT_HEX_FILE),
      .DEFAULT_FC1_BIAS_HEX_FILE(DEFAULT_FC1_BIAS_HEX_FILE),
      .DEFAULT_QUANT_PARAM_FILE(DEFAULT_QUANT_PARAM_FILE),
      .DEFAULT_FC2_WEIGHT_HEX_FILE(DEFAULT_FC2_WEIGHT_HEX_FILE),
      .DEFAULT_FC2_BIAS_HEX_FILE(DEFAULT_FC2_BIAS_HEX_FILE)
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
    $display("============================================================");
    $display("tb_mnist_inference_core_board");
    $display("DEFAULT_SAMPLE_HEX_FILE     = %s", DEFAULT_SAMPLE_HEX_FILE);
    $display("DEFAULT_FC1_WEIGHT_HEX_FILE = %s", DEFAULT_FC1_WEIGHT_HEX_FILE);
    $display("DEFAULT_FC1_BIAS_HEX_FILE   = %s", DEFAULT_FC1_BIAS_HEX_FILE);
    $display("DEFAULT_QUANT_PARAM_FILE    = %s", DEFAULT_QUANT_PARAM_FILE);
    $display("DEFAULT_FC2_WEIGHT_HEX_FILE = %s", DEFAULT_FC2_WEIGHT_HEX_FILE);
    $display("DEFAULT_FC2_BIAS_HEX_FILE   = %s", DEFAULT_FC2_BIAS_HEX_FILE);
    $display("PRED_FILE                   = %s", PRED_FILE);
    $display("============================================================");

    fd = $fopen(PRED_FILE, "r");
    if (fd == 0) begin
      $display("ERROR: cannot open pred file: %s", PRED_FILE);
      $finish;
    end
    r = $fscanf(fd, "%d", ref_pred_class);
    $fclose(fd);

    if (r != 1) begin
      $display("ERROR: failed to parse pred file");
      $finish;
    end

    err_count = 0;
    rst_n     = 1'b0;
    start     = 1'b0;
    sample_id = '0;

    @(posedge clk);
    rst_n <= 1'b1;

    @(posedge clk);
    start <= 1'b1;

    @(posedge clk);
    start <= 1'b0;

    wait (done == 1'b1);
    #1;

    $display("[CHECK] pred_class=%0d ref=%0d", pred_class, ref_pred_class);
    for (o = 0; o < FC2_OUT_DIM; o = o + 1) begin
      $display("logits_all[%0d] = %0d", o, logits_all[o]);
    end

    if (pred_class !== ref_pred_class[$clog2(FC2_OUT_DIM)-1:0]) begin
      $display("ERROR: pred_class mismatch got=%0d exp=%0d", pred_class, ref_pred_class);
      err_count = err_count + 1;
    end

    if (err_count == 0)
      $display("PASS: mnist_inference_core_board end-to-end prediction is correct.");
    else $display("FAIL: mnist_inference_core_board found %0d mismatches.", err_count);

    $finish;
  end

  initial begin
    #200_000_000ns;
    $display("ERROR: timeout in tb_mnist_inference_core_board");
    $finish;
  end

endmodule
