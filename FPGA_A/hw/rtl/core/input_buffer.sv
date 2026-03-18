
module input_buffer #(
    parameter string DEFAULT_INPUT_HEX_FILE = "input_0.hex"
) (
    input logic clk,
    input logic [$clog2(mnist_cim_pkg::N_INPUT_BLOCKS)-1:0] ib,

    output logic signed [mnist_cim_pkg::INPUT_WIDTH-1:0] x_tile[0:mnist_cim_pkg::TILE_INPUT_SIZE-1],

    output logic [mnist_cim_pkg::X_EFF_WIDTH-1:0] x_eff_tile[0:mnist_cim_pkg::TILE_INPUT_SIZE-1]
);

  import mnist_cim_pkg::*;

  localparam int TILE_BITS = TILE_INPUT_SIZE * INPUT_WIDTH;
  localparam int TILE_DEPTH = N_INPUT_BLOCKS;

  // One packed line = one input tile
  (* rom_style = "block" *)
  logic [TILE_BITS-1:0] input_mem[0:TILE_DEPTH-1];

  logic [TILE_BITS-1:0] tile_word_r;

  integer tc;
  integer x_eff_tmp;

  initial begin
`ifndef SYNTHESIS
    $display("Using packed INPUT_HEX_FILE: %s", DEFAULT_INPUT_HEX_FILE);
`endif
    $readmemh(DEFAULT_INPUT_HEX_FILE, input_mem);
  end

  // synchronous ROM read
  always_ff @(posedge clk) begin
    tile_word_r <= input_mem[ib];
  end

  // unpack tile
  always_comb begin
    for (tc = 0; tc < TILE_INPUT_SIZE; tc = tc + 1) begin
      x_tile[tc] = tile_word_r[tc*INPUT_WIDTH+:INPUT_WIDTH];

      x_eff_tmp  = $signed(x_tile[tc]) - INPUT_ZERO_POINT;
      if (x_eff_tmp < 0) x_eff_tile[tc] = '0;
      else if (x_eff_tmp > ((1 << X_EFF_WIDTH) - 1)) x_eff_tile[tc] = {X_EFF_WIDTH{1'b1}};
      else x_eff_tile[tc] = x_eff_tmp[X_EFF_WIDTH-1:0];
    end
  end

endmodule

