
`timescale 1ns / 1ps

module tb_mnist_cim_demo_a_top;
  import mnist_cim_pkg::*;

  parameter int CLK_HZ = 1000;
  parameter int BAUD = 100;
  parameter int N_SAMPLES = 20;

  parameter string DEFAULT_SAMPLE_HEX_FILE = "../data/samples/mnist_samples_route_b_output_2.hex";
  parameter string DEFAULT_FC1_WEIGHT_HEX_FILE = "../data/weights/fc1_weight_int8.hex";
  parameter string DEFAULT_FC1_BIAS_HEX_FILE = "../data/weights/fc1_bias_int32.hex";
  parameter string DEFAULT_QUANT_PARAM_FILE = "../data/quant/quant_params.hex";
  parameter string DEFAULT_FC2_WEIGHT_HEX_FILE = "../data/weights/fc2_weight_int8.hex";
  parameter string DEFAULT_FC2_BIAS_HEX_FILE = "../data/weights/fc2_bias_int32.hex";
  parameter string PRED_FILE = "../data/expected/pred_0.txt";

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
  integer error_count;

  reg [7:0] rx_bytes[0:15];
  integer rx_count;

  logic monitor_enable;

  mnist_cim_demo_a_top #(
      .CLK_HZ(CLK_HZ),
      .BAUD(BAUD),
      .N_SAMPLES(N_SAMPLES),
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
      .led_busy(led_busy),
      .led_done(led_done)
  );

  initial clk = 1'b0;
  always #(CLK_PERIOD / 2) clk = ~clk;

  // ------------------------------------------------------------
  // Helper: print byte as hex / dec / printable char
  // ------------------------------------------------------------
  task automatic print_uart_byte(input [8*24-1:0] name, input [7:0] data);
    begin
      if (data >= 8'd32 && data <= 8'd126)
        $display("%0s = 0x%02x (%0d, '%s')", name, data, data, {data});
      else $display("%0s = 0x%02x (%0d)", name, data, data);
    end
  endtask

  // ------------------------------------------------------------
  // UART receiver: capture one byte from uart_tx
  // ------------------------------------------------------------
  task automatic uart_capture_one_byte(output reg [7:0] data);
    integer i;
    begin
      data = 8'h00;

      // wait for start bit
      @(negedge uart_tx);

      // move to center of bit0
      #(BIT_TIME + BIT_TIME / 2);

      for (i = 0; i < 8; i = i + 1) begin
        data[i] = uart_tx;
        #BIT_TIME;
      end
    end
  endtask

  // ------------------------------------------------------------
  // Background UART monitor
  // ------------------------------------------------------------
  initial begin : UART_MONITOR
    reg [7:0] tmp;
    rx_count = 0;
    monitor_enable = 1'b0;

    forever begin
      wait (monitor_enable == 1'b1);
      uart_capture_one_byte(tmp);

      if (monitor_enable) begin
        rx_bytes[rx_count] = tmp;
        $display("UART monitor captured byte[%0d] at time=%0t", rx_count, $time);
        print_uart_byte("  captured", tmp);
        rx_count = rx_count + 1;
      end
    end
  end

  // ------------------------------------------------------------
  // Optional signal-change monitor
  // ------------------------------------------------------------
  logic prev_led_busy, prev_led_done, prev_uart_tx;
  initial begin
    prev_led_busy = 1'b0;
    prev_led_done = 1'b0;
    prev_uart_tx  = 1'b1;

    $display("time=%0t : monitor start", $time);

    forever begin
      @(posedge clk);
      if ((led_busy !== prev_led_busy) ||
          (led_done !== prev_led_done) ||
          (uart_tx  !== prev_uart_tx)) begin
        $display(
            "time=%0t rst_n=%0b btn_start=%0b sample_sel=%0d led_busy=%0b led_done=%0b uart_tx=%0b",
            $time, rst_n, btn_start, sample_sel, led_busy, led_done, uart_tx);
        prev_led_busy = led_busy;
        prev_led_done = led_done;
        prev_uart_tx  = uart_tx;
      end
    end
  end

  // ------------------------------------------------------------
  // Read reference prediction
  // ------------------------------------------------------------
  initial begin
    $display("============================================================");
    $display("  SAMPLE_HEX_FILE     = %s", DEFAULT_SAMPLE_HEX_FILE);
    $display("  FC1_WEIGHT_HEX_FILE = %s", DEFAULT_FC1_WEIGHT_HEX_FILE);
    $display("  FC1_BIAS_HEX_FILE   = %s", DEFAULT_FC1_BIAS_HEX_FILE);
    $display("  QUANT_PARAM_FILE    = %s", DEFAULT_QUANT_PARAM_FILE);
    $display("  FC2_WEIGHT_HEX_FILE = %s", DEFAULT_FC2_WEIGHT_HEX_FILE);
    $display("  FC2_BIAS_HEX_FILE   = %s", DEFAULT_FC2_BIAS_HEX_FILE);
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

  // ------------------------------------------------------------
  // Main test flow
  // ------------------------------------------------------------
  initial begin
    error_count = 0;

    rst_n       = 1'b0;
    btn_start   = 1'b0;
    sample_sel  = '0;

    #(3 * CLK_PERIOD);
    rst_n = 1'b1;
    $display("time=%0t : release reset", $time);

    monitor_enable = 1'b1;

    @(posedge clk);
    btn_start <= 1'b1;
    $display("time=%0t : pulse btn_start high", $time);

    @(posedge clk);
    btn_start <= 1'b0;
    $display("time=%0t : pulse btn_start low", $time);

    // FC1 is multi-round; FC2 is now serial MAC, so total delay is much longer
    wait (led_done == 1'b1);
    $display("time=%0t : led_done observed high", $time);

    wait (rx_count >= 3);
    monitor_enable = 1'b0;

    $display("------------------------------------------------------------");
    print_uart_byte("UART byte0 received", rx_bytes[0]);
    print_uart_byte("UART byte1 received", rx_bytes[1]);
    print_uart_byte("UART byte2 received", rx_bytes[2]);
    $display("------------------------------------------------------------");
    $display("Expected byte0 = ASCII('0' + pred) = 0x%02x", (8'd48 + ref_pred_class[3:0]));
    $display("Expected byte1 = 0x0D");
    $display("Expected byte2 = 0x0A");
    $display("------------------------------------------------------------");

    if (rx_bytes[0] !== (8'd48 + ref_pred_class[3:0])) begin
      $display("ERROR: byte0 mismatch, got=0x%02x expected=0x%02x", rx_bytes[0],
               (8'd48 + ref_pred_class[3:0]));
      error_count = error_count + 1;
    end else begin
      $display("MATCH: byte0 correct");
    end

    if (rx_bytes[1] !== 8'h0D) begin
      $display("ERROR: byte1 mismatch, got=0x%02x expected=0x0D", rx_bytes[1]);
      error_count = error_count + 1;
    end else begin
      $display("MATCH: byte1 correct (CR)");
    end

    if (rx_bytes[2] !== 8'h0A) begin
      $display("ERROR: byte2 mismatch, got=0x%02x expected=0x0A", rx_bytes[2]);
      error_count = error_count + 1;
    end else begin
      $display("MATCH: byte2 correct (LF)");
    end

    $display("Final LED states: led_busy=%0b led_done=%0b", led_busy, led_done);

    if (error_count == 0) begin
      $display("PASS: mnist_cim_demo_a_top sends pred_class over UART correctly.");
    end else begin
      $display("FAIL: found %0d mismatches in UART output.", error_count);
      $display("DETAIL: ref_pred_class=%0d, actual bytes=[%02x %02x %02x]", ref_pred_class,
               rx_bytes[0], rx_bytes[1], rx_bytes[2]);
    end

    $finish;
  end

  // ------------------------------------------------------------
  // Timeout: much larger than before, because FC1+FC2 are both slower
  // ------------------------------------------------------------
  initial begin
    #200_000_000ns;
    $display("ERROR: timeout waiting for LED/UART result.");
    $finish;
  end

endmodule
