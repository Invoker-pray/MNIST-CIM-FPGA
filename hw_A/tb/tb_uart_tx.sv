
`timescale 1ns / 1ps

module tb_uart_tx;

  parameter int CLK_HZ = 1000;
  parameter int BAUD = 100;
  localparam int DIV = CLK_HZ / BAUD;
  localparam time CLK_PERIOD = 10ns;
  localparam time BIT_TIME = DIV * CLK_PERIOD;

  logic clk;
  logic rst_n;
  logic start;
  logic [7:0] data_in;
  logic tx;
  logic busy;

  reg [7:0] rx_byte;
  integer error_count;
  integer i;

  uart_tx #(
      .CLK_HZ(CLK_HZ),
      .BAUD  (BAUD)
  ) dut (
      .clk(clk),
      .rst_n(rst_n),
      .start(start),
      .data_in(data_in),
      .tx(tx),
      .busy(busy)
  );

  initial clk = 1'b0;
  always #(CLK_PERIOD / 2) clk = ~clk;

  task automatic uart_recv_byte(output reg [7:0] data);
    begin

      @(negedge tx);

      #(BIT_TIME + BIT_TIME / 2);
      for (i = 0; i < 8; i = i + 1) begin
        data[i] = tx;
        #BIT_TIME;
      end
      #BIT_TIME;  // stop bit
    end
  endtask

  initial begin
    error_count = 0;
    rst_n = 1'b0;
    start = 1'b0;
    data_in = 8'h37;  // '7'

    #(3 * CLK_PERIOD);
    rst_n = 1'b1;

    @(posedge clk);
    start <= 1'b1;
    @(posedge clk);
    start <= 1'b0;

    uart_recv_byte(rx_byte);

    if (rx_byte !== 8'h37) begin
      $display("ERROR UART got=0x%02x expected=0x37", rx_byte);
      error_count = error_count + 1;
    end

    if (error_count == 0) $display("PASS: uart_tx sends one byte correctly.");
    else $display("FAIL: found %0d mismatches.", error_count);

    $finish;
  end

endmodule
