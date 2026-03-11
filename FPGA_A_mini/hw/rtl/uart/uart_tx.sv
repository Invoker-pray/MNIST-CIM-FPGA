
module uart_tx #(
    parameter int CLK_HZ = 50_000_000,
    parameter int BAUD   = 115200
) (
    input logic clk,
    input logic rst_n,
    input logic start,
    input logic [7:0] data_in,
    output logic tx,
    output logic busy
);

  localparam int DIV = CLK_HZ / BAUD;

  logic [15:0] baud_cnt;
  logic [ 3:0] bit_idx;
  logic [ 9:0] frame;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      tx       <= 1'b1;
      busy     <= 1'b0;
      baud_cnt <= '0;
      bit_idx  <= '0;
      frame    <= 10'h3FF;
    end else begin
      if (!busy) begin
        tx <= 1'b1;

        if (start) begin
          // frame = {stop, data[7:0], start}
          frame    <= {1'b1, data_in, 1'b0};
          busy     <= 1'b1;
          baud_cnt <= 16'd0;
          bit_idx  <= 4'd0;
          tx       <= 1'b0;  // start bit
        end
      end else begin
        if (baud_cnt == DIV - 1) begin
          baud_cnt <= 16'd0;

          if (bit_idx == 4'd9) begin
            // stop bit 已经保持满 1 bit 时间，现在才真正结束
            busy    <= 1'b0;
            bit_idx <= 4'd0;
            tx      <= 1'b1;
          end else begin
            bit_idx <= bit_idx + 1'b1;
            tx      <= frame[bit_idx+1'b1];
          end
        end else begin
          baud_cnt <= baud_cnt + 1'b1;
        end
      end
    end
  end

endmodule

