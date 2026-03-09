
`timescale 1ns / 1ps

module tb_uart_pred_sender;

  parameter int CLK_HZ = 1000;
  parameter int BAUD = 100;
  localparam int DIV = CLK_HZ / BAUD;
  localparam time CLK_PERIOD = 10ns;
  localparam time BIT_TIME = DIV * CLK_PERIOD;

  logic clk;
  logic rst_n;
  logic trigger;
  logic [3:0] pred_class;
  logic uart_tx;

  reg [7:0] b0, b1, b2;
  integer error_count;
  integer i;

  uart_pred_sender #(
      .CLK_HZ(CLK_HZ),
      .BAUD  (BAUD)
  ) dut (
      .clk(clk),
      .rst_n(rst_n),
      .trigger(trigger),
      .pred_class(pred_class),
      .uart_tx(uart_tx)
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
    error_count = 0;
    rst_n = 1'b0;
    trigger = 1'b0;
    pred_class = 4'd7;

    #(3 * CLK_PERIOD);
    rst_n = 1'b1;

    @(posedge clk);
    trigger <= 1'b1;
    @(posedge clk);
    trigger <= 1'b0;

    uart_recv_byte(b0);
    uart_recv_byte(b1);
    uart_recv_byte(b2);

    if (b0 !== 8'h37) error_count = error_count + 1;
    if (b1 !== 8'h0D) error_count = error_count + 1;
    if (b2 !== 8'h0A) error_count = error_count + 1;

    if (error_count == 0) $display("PASS: uart_pred_sender sends digit/CR/LF correctly.");
    else $display("FAIL: found %0d mismatches.", error_count);

    $finish;
  end

endmodule
