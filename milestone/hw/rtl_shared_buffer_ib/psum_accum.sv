
module psum_accum (
    input logic clk,
    input logic rst_n,
    input logic clear,
    input logic en,

    input  logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0]
        tile_psum [0:mnist_cim_pkg::TILE_OUTPUT_SIZE-1],

    output logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0] psum[0:mnist_cim_pkg::TILE_OUTPUT_SIZE-1]
);

  import mnist_cim_pkg::*;

  integer i;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (i = 0; i < TILE_OUTPUT_SIZE; i = i + 1) begin
        psum[i] <= '0;
      end
    end else if (clear) begin
      for (i = 0; i < TILE_OUTPUT_SIZE; i = i + 1) begin
        psum[i] <= '0;
      end
    end else if (en) begin
      for (i = 0; i < TILE_OUTPUT_SIZE; i = i + 1) begin
        psum[i] <= psum[i] + tile_psum[i];
      end
    end
  end

endmodule
