
module fc1_weight_bank #(
    parameter string WEIGHT_HEX_FILE ="../../CIM-sw-version1/sw/train&quantize/route_b_output/fc1_weight_int8.hex"
) (
    input logic [$clog2(mnist_cim_pkg::N_OUTPUT_BLOCKS)-1:0] ob,
    input logic [ $clog2(mnist_cim_pkg::N_INPUT_BLOCKS)-1:0] ib,

    output logic signed [mnist_cim_pkg::WEIGHT_WIDTH-1:0]
        w_tile [0:mnist_cim_pkg::TILE_OUTPUT_SIZE-1]
               [0:mnist_cim_pkg::TILE_INPUT_SIZE-1]
);

  import mnist_cim_pkg::*;

  // --------------------------------------------------
  // Flat weight memory
  // Layout in file:
  //   row-major [out][in]
  // Address:
  //   addr = out * FC1_IN_DIM + in
  // --------------------------------------------------
  logic signed [WEIGHT_WIDTH-1:0] weight_mem[0:FC1_WEIGHT_DEPTH-1];

  initial begin
    $readmemh(WEIGHT_HEX_FILE, weight_mem);
  end

  integer tr, tc;
  integer out_idx, in_idx;
  integer addr;

  always_comb begin
    for (tr = 0; tr < TILE_OUTPUT_SIZE; tr = tr + 1) begin
      for (tc = 0; tc < TILE_INPUT_SIZE; tc = tc + 1) begin
        out_idx = ob * TILE_OUTPUT_SIZE + tr;
        in_idx  = ib  * TILE_INPUT_SIZE  + tc;
        addr    = out_idx * FC1_IN_DIM + in_idx;

        w_tile[tr][tc] = weight_mem[addr];
      end
    end
  end

endmodule
