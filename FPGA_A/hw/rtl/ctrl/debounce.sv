
module debounce #(
    parameter int CLK_HZ = 50_000_000,
    parameter int DEBOUNCE_MS = 20
) (
    input  logic clk,
    input  logic rst_n,
    input  logic din,
    output logic dout
);

  localparam int COUNT_MAX = (CLK_HZ / 1000) * DEBOUNCE_MS;
  localparam int COUNT_W = (COUNT_MAX <= 1) ? 1 : $clog2(COUNT_MAX + 1);

  logic din_meta, din_sync;
  logic din_last;
  logic [COUNT_W-1:0] cnt;

  // 2-flop synchronizer for async button input
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      din_meta <= 1'b0;
      din_sync <= 1'b0;
    end else begin
      din_meta <= din;
      din_sync <= din_meta;
    end
  end

  // Debounce: only update dout after din_sync stays changed for COUNT_MAX cycles
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      din_last <= 1'b0;
      cnt      <= '0;
      dout     <= 1'b0;
    end else begin
      if (din_sync == din_last) begin
        cnt <= '0;
      end else begin
        if (cnt == COUNT_MAX - 1) begin
          din_last <= din_sync;
          dout     <= din_sync;
          cnt      <= '0;
        end else begin
          cnt <= cnt + 1'b1;
        end
      end
    end
  end

endmodule
