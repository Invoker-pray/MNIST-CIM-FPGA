
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
  logic [ 9:0] shifter;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      tx       <= 1'b1;
      busy     <= 1'b0;
      baud_cnt <= '0;
      bit_idx  <= '0;
      shifter  <= 10'h3FF;
    end else begin
      if (!busy) begin
        tx <= 1'b1;
        if (start) begin
          busy     <= 1'b1;
          baud_cnt <= 0;
          bit_idx  <= 0;
          shifter  <= {1'b1, data_in, 1'b0};
          tx       <= 1'b0;
        end
      end else begin
        if (baud_cnt == DIV - 1) begin
          baud_cnt <= 0;
          bit_idx  <= bit_idx + 1'b1;
          shifter  <= {1'b1, shifter[9:1]};
          tx       <= shifter[1];

          if (bit_idx == 9) begin
            busy <= 1'b0;
            tx   <= 1'b1;
          end
        end else begin
          baud_cnt <= baud_cnt + 1'b1;
        end
      end
    end
  end

endmodule
