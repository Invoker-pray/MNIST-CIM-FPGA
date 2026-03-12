
`timescale 1ns / 1ps

module tb_mnist_cim_demo_a_top;
  import mnist_cim_pkg::*;

  parameter int CLK_HZ = 125_000_000;
  parameter int BAUD = 115200;
  parameter int N_SAMPLES = 20;

  parameter string DEFAULT_SAMPLE_HEX_FILE     = "../data_packed/samples/mnist_samples_route_b_output_2.hex";
  parameter string DEFAULT_FC1_WEIGHT_HEX_FILE = "../data_packed/weights/fc1_weight_int8.hex";
  parameter string DEFAULT_FC1_BIAS_HEX_FILE = "../data_packed/weights/fc1_bias_int32.hex";
  parameter string DEFAULT_QUANT_PARAM_FILE = "../data_packed/quant/quant_params.hex";
  parameter string DEFAULT_FC2_WEIGHT_HEX_FILE = "../data_packed/weights/fc2_weight_int8.hex";
  parameter string DEFAULT_FC2_BIAS_HEX_FILE = "../data_packed/weights/fc2_bias_int32.hex";
  parameter string EXPECTED_DIR = "../data_packed/expected";

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

  integer error_count;
  integer pass_count;
  integer fail_count;

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

  task automatic print_uart_byte(input [8*32-1:0] name, input [7:0] data);
    begin
      if (data >= 8'd32 && data <= 8'd126)
        $display("%0s = 0x%02x (%0d, '%s')", name, data, data, {data});
      else $display("%0s = 0x%02x (%0d)", name, data, data);
    end
  endtask

  task automatic uart_capture_one_byte(output reg [7:0] data);
    integer i;
    begin
      data = 8'h00;
      @(negedge uart_tx);
      #(BIT_TIME + BIT_TIME / 2);
      for (i = 0; i < 8; i = i + 1) begin
        data[i] = uart_tx;
        #BIT_TIME;
      end
    end
  endtask

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

  task automatic load_ref_pred(input integer sid, output integer ref_pred_class);
    integer fd, r;
    string pred_file;
    begin
      pred_file = $sformatf("%0s/pred_%0d.txt", EXPECTED_DIR, sid);

      fd = $fopen(pred_file, "r");
      if (fd == 0) begin
        $display("ERROR: cannot open pred file: %0s", pred_file);
        $finish;
      end

      r = $fscanf(fd, "%d", ref_pred_class);
      $fclose(fd);

      if (r != 1) begin
        $display("ERROR: failed to parse pred file: %0s", pred_file);
        $finish;
      end
    end
  endtask

  task automatic run_one_sample(input integer sid);
    integer ref_pred_class;
    begin
      load_ref_pred(sid, ref_pred_class);

      rx_count = 0;
      monitor_enable = 1'b1;
      sample_sel = sid[$clog2(N_SAMPLES)-1:0];

      @(posedge clk);
      btn_start <= 1'b1;
      @(posedge clk);
      btn_start <= 1'b0;

      wait (led_done == 1'b1);
      wait (rx_count >= 3);
      monitor_enable = 1'b0;

      $display("------------------------------------------------------------");
      $display("SAMPLE %0d", sid);
      $display("Reference pred_class from file = %0d", ref_pred_class);
      print_uart_byte("UART byte0 received", rx_bytes[0]);
      print_uart_byte("UART byte1 received", rx_bytes[1]);
      print_uart_byte("UART byte2 received", rx_bytes[2]);
      $display("------------------------------------------------------------");

      if (rx_bytes[0] !== (8'd48 + ref_pred_class[3:0])) begin
        $display("ERROR: sample %0d byte0 mismatch, got=0x%02x expected=0x%02x", sid, rx_bytes[0],
                 (8'd48 + ref_pred_class[3:0]));
        error_count = error_count + 1;
        fail_count  = fail_count + 1;
      end else if (rx_bytes[1] !== 8'h0D) begin
        $display("ERROR: sample %0d byte1 mismatch, got=0x%02x expected=0x0D", sid, rx_bytes[1]);
        error_count = error_count + 1;
        fail_count  = fail_count + 1;
      end else if (rx_bytes[2] !== 8'h0A) begin
        $display("ERROR: sample %0d byte2 mismatch, got=0x%02x expected=0x0A", sid, rx_bytes[2]);
        error_count = error_count + 1;
        fail_count  = fail_count + 1;
      end else begin
        $display("PASS: sample %0d correct.", sid);
        pass_count = pass_count + 1;
      end

      wait (led_done == 1'b0 || led_busy == 1'b0);
      repeat (10) @(posedge clk);
    end
  endtask

  integer sid;

  initial begin
    $display("============================================================");
    $display("  SAMPLE_HEX_FILE     = %s", DEFAULT_SAMPLE_HEX_FILE);
    $display("  FC1_WEIGHT_HEX_FILE = %s", DEFAULT_FC1_WEIGHT_HEX_FILE);
    $display("  FC1_BIAS_HEX_FILE   = %s", DEFAULT_FC1_BIAS_HEX_FILE);
    $display("  QUANT_PARAM_FILE    = %s", DEFAULT_QUANT_PARAM_FILE);
    $display("  FC2_WEIGHT_HEX_FILE = %s", DEFAULT_FC2_WEIGHT_HEX_FILE);
    $display("  FC2_BIAS_HEX_FILE   = %s", DEFAULT_FC2_BIAS_HEX_FILE);
    $display("  EXPECTED_DIR        = %s", EXPECTED_DIR);
    $display("============================================================");

    error_count = 0;
    pass_count  = 0;
    fail_count  = 0;

    rst_n       = 1'b0;
    btn_start   = 1'b0;
    sample_sel  = '0;

    #(3 * CLK_PERIOD);
    rst_n = 1'b1;
    $display("time=%0t : release reset", $time);

    for (sid = 0; sid < N_SAMPLES; sid = sid + 1) begin
      run_one_sample(sid);
    end

    $display("============================================================");
    $display("FINAL SUMMARY: pass=%0d fail=%0d total=%0d", pass_count, fail_count, N_SAMPLES);
    if (error_count == 0) $display("PASS: all %0d samples correct.", N_SAMPLES);
    else $display("FAIL: %0d sample(s) mismatched.", error_count);
    $display("============================================================");

    $finish;
  end

  initial begin
    #5_000_000_000ns;
    $display("ERROR: timeout waiting for LED/UART result.");
    $finish;
  end

endmodule
