

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
      wait (uart_tx == 1'b0);  // start bit
      #(BIT_TIME + BIT_TIME / 2);  // sample at center of bit0
      for (i = 0; i < 8; i = i + 1) begin
        data[i] = uart_tx;
        #BIT_TIME;
      end
      #BIT_TIME;  // stop bit
    end
  endtask

  initial begin
    $display("============================================================");
    $display("TB_mnist_cim_demo_a_top start");
    $display("  SAMPLE_HEX_FILE     = %s", SAMPLE_HEX_FILE);
    $display("  FC1_WEIGHT_HEX_FILE = %s", FC1_WEIGHT_HEX_FILE);
    $display("  FC1_BIAS_HEX_FILE   = %s", FC1_BIAS_HEX_FILE);
    $display("  QUANT_PARAM_FILE    = %s", QUANT_PARAM_FILE);
    $display("  FC2_WEIGHT_HEX_FILE = %s", FC2_WEIGHT_HEX_FILE);
    $display("  FC2_BIAS_HEX_FILE   = %s", FC2_BIAS_HEX_FILE);
    $display("  PRED_FILE           = %s", PRED_FILE);
    $display("============================================================");

    fd = $fopen(PRED_FILE, "r");
    if (fd == 0) begin
      $display("ERROR: cannot open pred file: %s", PRED_FILE);
      $finish;
    end
    r = $fscanf(fd, "%d", ref_pred_class);
    $fclose(fd);
    if (r != 1) begin
      $display("ERROR: failed to parse pred file: %s", PRED_FILE);
      $finish;
    end

    $display("Reference pred_class from file = %0d", ref_pred_class);
  end

  // 可选：观察关键控制信号变化
  initial begin
    $display("time=%0t : monitor start", $time);
    forever begin
      @(posedge clk);
      $display(
          "time=%0t clk=1 rst_n=%0b btn_start=%0b sample_sel=%0d led_busy=%0b led_done=%0b uart_tx=%0b",
          $time, rst_n, btn_start, sample_sel, led_busy, led_done, uart_tx);
    end
  end

  initial begin
    error_count = 0;
    rst_n = 1'b0;
    btn_start = 1'b0;
    sample_sel = '0;

    b0 = 8'h00;
    b1 = 8'h00;
    b2 = 8'h00;

    #(3 * CLK_PERIOD);
    rst_n = 1'b1;
    $display("time=%0t : release reset", $time);

    @(posedge clk);
    btn_start <= 1'b1;
    $display("time=%0t : pulse btn_start high", $time);

    @(posedge clk);
    btn_start <= 1'b0;
    $display("time=%0t : pulse btn_start low", $time);

    uart_recv_byte(b0);
    $display("UART byte0 received = 0x%02x (%0d, '%s')", b0, b0,
             (b0 >= 8'd32 && b0 <= 8'd126) ? {b0} : "?");

    uart_recv_byte(b1);
    $display("UART byte1 received = 0x%02x (%0d)", b1, b1);

    uart_recv_byte(b2);
    $display("UART byte2 received = 0x%02x (%0d)", b2, b2);

    $display("------------------------------------------------------------");
    $display("Expected byte0 = ASCII('0' + pred) = 0x%02x", (8'd48 + ref_pred_class[3:0]));
    $display("Expected byte1 = 0x0D");
    $display("Expected byte2 = 0x0A");
    $display("------------------------------------------------------------");

    if (b0 !== (8'd48 + ref_pred_class[3:0])) begin
      $display("ERROR: byte0 mismatch, got=0x%02x expected=0x%02x", b0,
               (8'd48 + ref_pred_class[3:0]));
      error_count = error_count + 1;
    end else begin
      $display("MATCH: byte0 correct");
    end

    if (b1 !== 8'h0D) begin
      $display("ERROR: byte1 mismatch, got=0x%02x expected=0x0D", b1);
      error_count = error_count + 1;
    end else begin
      $display("MATCH: byte1 correct (CR)");
    end

    if (b2 !== 8'h0A) begin
      $display("ERROR: byte2 mismatch, got=0x%02x expected=0x0A", b2);
      error_count = error_count + 1;
    end else begin
      $display("MATCH: byte2 correct (LF)");
    end

    $display("Final LED states: led_busy=%0b led_done=%0b", led_busy, led_done);

    if (error_count == 0) begin
      $display("PASS: mnist_cim_demo_a_top sends pred_class over UART correctly.");
    end else begin
      $display("FAIL: found %0d mismatches in UART output.", error_count);
      $display("DETAIL: ref_pred_class=%0d, actual bytes = [%02x %02x %02x]", ref_pred_class, b0,
               b1, b2);
    end

    $finish;
  end

endmodule
