
`timescale 1ns / 1ps

module tb_mnist_cim_demo_a_top;
  import mnist_cim_pkg::*;

  parameter int CLK_HZ = 1000;
  parameter int BAUD = 100;
  parameter int N_SAMPLES = 20;
  parameter string SAMPLE_HEX_FILE = "../data/mnist_samples_route_b_output_2.hex";
  parameter string FC1_WEIGHT_HEX_FILE = "../route_b_output_2/fc1_weight_int8.hex";
  parameter string FC1_BIAS_HEX_FILE = "../route_b_output_2/fc1_bias_int32.hex";
  parameter string QUANT_PARAM_FILE = "../route_b_output_2/quant_params.hex";
  parameter string FC2_WEIGHT_HEX_FILE = "../route_b_output_2/fc2_weight_int8.hex";
  parameter string FC2_BIAS_HEX_FILE = "../route_b_output_2/fc2_bias_int32.hex";
  parameter string PRED_FILE = "../route_b_output_2/pred_0.txt";

  localparam int DIV = CLK_HZ / BAUD;
  localparam time CLK_PERIOD = 10ns;
  localparam time BIT_TIME = DIV * CLK_PERIOD;

  logic clk;
  logic rst_n;
  logic btn_start;
  logic [$clog2(N_SAMPLES)-1:0] sample_sel;
  logic uart_tx;
  logic led_busy;
  logic led_done;

  integer ref_pred_class;
  integer fd, r;
  integer i;
  integer error_count;
  reg [7:0] b0, b1, b2;

  mnist_cim_demo_a_top #(
      .CLK_HZ(CLK_HZ),
      .BAUD(BAUD),
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
      .btn_start(btn_start),
      .sample_sel(sample_sel),
      .uart_tx(uart_tx),
      .led_busy(led_busy),
      .led_done(led_done)
  );

  initial clk = 1'b0;
  always #(CLK_PERIOD / 2) clk = ~clk;

  task automatic uart_recv_byte(output reg [7:0] data);
    begin
      wait (uart_tx == 1'b0);
      #(BIT_TIME + BIT_TIME / 2);
      for (i = 0; i < 8; i = i + 1) begin
        data[i] = uart_tx;
        #BIT_TIME;
      end
      #BIT_TIME;
    end
  endtask

  initial begin
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
    btn_start = 1'b0;
    sample_sel = '0;

    #(3 * CLK_PERIOD);
    rst_n = 1'b1;

    @(posedge clk);
    btn_start <= 1'b1;
    @(posedge clk);
    btn_start <= 1'b0;

    uart_recv_byte(b0);
    uart_recv_byte(b1);
    uart_recv_byte(b2);

    if (b0 !== (8'd48 + ref_pred_class[3:0])) error_count = error_count + 1;
    if (b1 !== 8'h0D) error_count = error_count + 1;
    if (b2 !== 8'h0A) error_count = error_count + 1;

    if (error_count == 0) $display("PASS: mnist_cim_demo_a_top sends pred_class over UART.");
    else $display("FAIL: found %0d mismatches.", error_count);

    $finish;
  end

endmodule
