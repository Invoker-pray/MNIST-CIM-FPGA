
module mnist_sample_rom #(
    parameter int N_SAMPLES = 20,
    parameter string DEFAULT_SAMPLE_HEX_FILE = "mnist_samples_route_b_output_2.hex"
) (
    input logic clk,
    input logic [$clog2(N_SAMPLES)-1:0] sample_id,
    input logic [$clog2(mnist_cim_pkg::N_INPUT_BLOCKS)-1:0] ib,

    output logic signed [mnist_cim_pkg::INPUT_WIDTH-1:0] x_tile[0:mnist_cim_pkg::TILE_INPUT_SIZE-1],

    output logic [mnist_cim_pkg::X_EFF_WIDTH-1:0] x_eff_tile[0:mnist_cim_pkg::TILE_INPUT_SIZE-1]
);
  import mnist_cim_pkg::*;

  localparam int TILE_BITS = TILE_INPUT_SIZE * INPUT_WIDTH;
  localparam int TILE_DEPTH = N_SAMPLES * N_INPUT_BLOCKS;

  // One packed line = one input tile
  (* rom_style = "block" *)
  logic [TILE_BITS-1:0] sample_mem[0:TILE_DEPTH-1];

  logic [TILE_BITS-1:0] tile_word_r;
  logic [$clog2(TILE_DEPTH)-1:0] tile_addr;

  integer i;
  integer x_eff_tmp;

  assign tile_addr = sample_id * N_INPUT_BLOCKS + ib;

  initial begin
`ifndef SYNTHESIS
    $display("Using packed sample tile file: %s", DEFAULT_SAMPLE_HEX_FILE);
`endif
    $readmemh(DEFAULT_SAMPLE_HEX_FILE, sample_mem);
  end

  // synchronous ROM read
  always_ff @(posedge clk) begin
    tile_word_r <= sample_mem[tile_addr];
  end

  // unpack tile
  // x_tile[i] = tile_word_r[i*INPUT_WIDTH +: INPUT_WIDTH]
  always_comb begin
    for (i = 0; i < TILE_INPUT_SIZE; i = i + 1) begin
      x_tile[i] = tile_word_r[i*INPUT_WIDTH+:INPUT_WIDTH];

      x_eff_tmp = $signed(x_tile[i]) - INPUT_ZERO_POINT;
      if (x_eff_tmp < 0) x_eff_tile[i] = '0;
      else if (x_eff_tmp > ((1 << X_EFF_WIDTH) - 1)) x_eff_tile[i] = {X_EFF_WIDTH{1'b1}};
      else x_eff_tile[i] = x_eff_tmp[X_EFF_WIDTH-1:0];
    end
  end

endmodule

