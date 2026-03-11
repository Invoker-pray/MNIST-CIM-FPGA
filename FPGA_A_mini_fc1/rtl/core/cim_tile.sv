


module cim_tile (
    input logic [mnist_cim_pkg::X_EFF_WIDTH-1:0] x_eff_tile[0:mnist_cim_pkg::TILE_INPUT_SIZE-1],

    input  logic signed [mnist_cim_pkg::WEIGHT_WIDTH-1:0]
        w_tile [0:mnist_cim_pkg::TILE_OUTPUT_SIZE-1]
               [0:mnist_cim_pkg::TILE_INPUT_SIZE-1],

    output logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0]
        tile_psum [0:mnist_cim_pkg::TILE_OUTPUT_SIZE-1]
);
  import mnist_cim_pkg::*;

  integer tr, tc;
  logic signed [PSUM_WIDTH-1:0] sum;

  always_comb begin
    for (tr = 0; tr < TILE_OUTPUT_SIZE; tr = tr + 1) begin
      sum = '0;
      for (tc = 0; tc < TILE_INPUT_SIZE; tc = tc + 1) begin
        sum = sum + $signed({1'b0, x_eff_tile[tc]}) * w_tile[tr][tc];
      end
      tile_psum[tr] = sum;
    end
  end

endmodule
