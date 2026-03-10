

`timescale 1ns / 1ps

module tb_mnist_cim_demo_a_top_led;

  localparam int CLK_HZ = 100_000_000;
  localparam int BAUD = 115200;
  localparam int N_SAMPLES = 20;

  parameter string DEFAULT_SAMPLE_HEX_FILE = "../data/samples/mnist_samples_route_b_output_2.hex";
  parameter string DEFAULT_FC1_WEIGHT_HEX_FILE = "../data/weights/fc1_weight_int8.hex";
  parameter string DEFAULT_FC1_BIAS_HEX_FILE = "../data/weights/fc1_bias_int32.hex";
  parameter string DEFAULT_QUANT_PARAM_FILE = "../data/quant/quant_params.hex";
  parameter string DEFAULT_FC2_WEIGHT_HEX_FILE = "../data/weights/fc2_weight_int8.hex";
  parameter string DEFAULT_FC2_BIAS_HEX_FILE = "../data/weights/fc2_bias_int32.hex";
  parameter string PRED_FILE = "../data/expected/pred_0.txt";

  logic clk;
  logic rst_n;
  logic btn_start;
  logic [$clog2(N_SAMPLES)-1:0] sample_sel;

  logic uart_tx;
  logic [3:0] led_result;

  integer ref_pred;
  integer fd;
  integer code;

  // DUT
  mnist_cim_demo_a_top #(
      .CLK_HZ(CLK_HZ),
      .BAUD(BAUD),
      .N_SAMPLES(N_SAMPLES),
      .BTN_DEBOUNCE_MS(20),
      .SIM_BYPASS_DEBOUNCE(1'b1),
      .DEFAULT_SAMPLE_HEX_FILE(DEFAULT_SAMPLE_HEX_FILE),
      .DEFAULT_FC1_WEIGHT_HEX_FILE(DEFAULT_FC1_WEIGHT_HEX_FILE),
      .DEFAULT_FC1_BIAS_HEX_FILE(DEFAULT_FC1_BIAS_HEX_FILE),
      .DEFAULT_QUANT_PARAM_FILE(DEFAULT_QUANT_PARAM_FILE),
      .DEFAULT_FC2_WEIGHT_HEX_FILE(DEFAULT_FC2_WEIGHT_HEX_FILE),
      .DEFAULT_FC2_BIAS_HEX_FILE(DEFAULT_FC2_BIAS_HEX_FILE)
  ) dut (
      .clk(clk),
      .rst_n(rst_n),
      .btn_start(btn_start),
      .sample_sel(sample_sel),
      .uart_tx(uart_tx),
      .led_result(led_result)
  );

  // 100 MHz clock
  initial clk = 1'b0;
  always #5 clk = ~clk;

  // basic info
  initial begin
    $display("time=0 : monitor start");
    $display("============================================================");
    $display("  SAMPLE_HEX_FILE     = %s", DEFAULT_SAMPLE_HEX_FILE);
    $display("  FC1_WEIGHT_HEX_FILE = %s", DEFAULT_FC1_WEIGHT_HEX_FILE);
    $display("  FC1_BIAS_HEX_FILE   = %s", DEFAULT_FC1_BIAS_HEX_FILE);
    $display("  QUANT_PARAM_FILE    = %s", DEFAULT_QUANT_PARAM_FILE);
    $display("  FC2_WEIGHT_HEX_FILE = %s", DEFAULT_FC2_WEIGHT_HEX_FILE);
    $display("  FC2_BIAS_HEX_FILE   = %s", DEFAULT_FC2_BIAS_HEX_FILE);
    $display("============================================================");
  end

  // read reference result from file
  initial begin
    ref_pred = -1;
    fd = $fopen(PRED_FILE, "r");
    if (fd == 0) begin
      $fatal(1, "ERROR: cannot open PRED_FILE = %s", PRED_FILE);
    end

    code = $fscanf(fd, "%d", ref_pred);
    $fclose(fd);

    if (code != 1) begin
      $fatal(1, "ERROR: failed to parse integer from %s", PRED_FILE);
    end

    $display("Reference pred_class from file = %0d", ref_pred);
  end

  // stimulus
  initial begin
    rst_n      = 1'b0;
    btn_start  = 1'b0;
    sample_sel = '0;  // test sample 0

    #30;
    rst_n = 1'b1;
    $display("time=%0t : release reset", $time);

    #5;
    btn_start = 1'b1;
    $display("time=%0t : pulse btn_start high", $time);

    #10;
    btn_start = 1'b0;
    $display("time=%0t : pulse btn_start low", $time);
  end

  // monitor LED result
  always @(led_result) begin
    $display("time=%0t : led_result = %0d (bin=%04b)", $time, led_result, led_result);
  end

  // main checker: only verify final latched LED result
  initial begin
    // wait until result becomes non-zero
    wait (led_result != 4'd0);

    // allow settle
    @(posedge clk);
    @(posedge clk);

    if (led_result !== ref_pred[3:0]) begin
      $fatal(1, "FAIL: led_result mismatch, led=%0d ref=%0d", led_result, ref_pred);
    end

    // verify hold behavior
    repeat (20) @(posedge clk);
    if (led_result !== ref_pred[3:0]) begin
      $fatal(1, "FAIL: led_result did not hold, led=%0d ref=%0d", led_result, ref_pred);
    end

    $display("PASS: latched led_result matches pred_class = %0d", ref_pred);
    $finish;
  end

  // timeout
  initial begin
    #5_000_000;
    $fatal(1, "TIMEOUT: simulation did not finish");
  end

endmodule
