
module fc1_weight_bank #(
    parameter string DEFAULT_WEIGHT_HEX_FILE = "fc1_weight_int8.hex"
) (
    input logic clk,
    input logic [$clog2(mnist_cim_pkg::N_OUTPUT_BLOCKS)-1:0] ob,
    input logic [$clog2(mnist_cim_pkg::N_INPUT_BLOCKS)-1:0] ib,

    output logic signed [mnist_cim_pkg::WEIGHT_WIDTH-1:0]
        w_tile [0:mnist_cim_pkg::TILE_OUTPUT_SIZE-1]
               [0:mnist_cim_pkg::TILE_INPUT_SIZE-1]
);

  import mnist_cim_pkg::*;

  localparam int TILE_ELEMS = TILE_OUTPUT_SIZE * TILE_INPUT_SIZE;
  localparam int TILE_BITS = TILE_ELEMS * WEIGHT_WIDTH;
  localparam int TILE_DEPTH = N_OUTPUT_BLOCKS * N_INPUT_BLOCKS;

  // One packed line = one 16x16 weight tile
  (* rom_style = "block" *)
  logic [TILE_BITS-1:0] tile_mem[0:TILE_DEPTH-1];

  logic [TILE_BITS-1:0] tile_word_r;
  logic [$clog2(TILE_DEPTH)-1:0] tile_addr;

  integer tr, tc;
  integer flat_idx;

  // Tile index: row-major over [ob][ib]
  assign tile_addr = ob * N_INPUT_BLOCKS + ib;

  initial begin
`ifndef SYNTHESIS
    $display("Using packed FC1 weight tile file: %s", DEFAULT_WEIGHT_HEX_FILE);
`endif
    $readmemh(DEFAULT_WEIGHT_HEX_FILE, tile_mem);
  end

  // Synchronous ROM read: BRAM-friendly
  always_ff @(posedge clk) begin
    tile_word_r <= tile_mem[tile_addr];
  end

  // Unpack one tile into the original 2D interface
  // packed[flat_idx*WEIGHT_WIDTH +: WEIGHT_WIDTH]
  // => element 0 is in the least-significant bits
  always_comb begin
    for (tr = 0; tr < TILE_OUTPUT_SIZE; tr = tr + 1) begin
      for (tc = 0; tc < TILE_INPUT_SIZE; tc = tc + 1) begin
        flat_idx = tr * TILE_INPUT_SIZE + tc;
        w_tile[tr][tc] = tile_word_r[flat_idx*WEIGHT_WIDTH+:WEIGHT_WIDTH];
      end
    end
  end

endmodule
