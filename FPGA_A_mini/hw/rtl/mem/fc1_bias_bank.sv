
module fc1_bias_bank #(
    parameter string DEFAULT_BIAS_HEX_FILE = "fc1_bias_int32.hex"
) (
    input logic clk,
    input logic [$clog2(mnist_cim_pkg::N_OUTPUT_BLOCKS)-1:0] ob,

    output logic signed [mnist_cim_pkg::BIAS_WIDTH-1:0]
        bias_block [0:mnist_cim_pkg::TILE_OUTPUT_SIZE-1]
);

  import mnist_cim_pkg::*;

  localparam int BLOCK_BITS = TILE_OUTPUT_SIZE * BIAS_WIDTH;

  // One packed line = one 16-element bias block
  (* rom_style = "block" *)
  logic [BLOCK_BITS-1:0] bias_mem[0:N_OUTPUT_BLOCKS-1];

  logic [BLOCK_BITS-1:0] bias_word_r;

  integer i;

  initial begin
`ifndef SYNTHESIS
    $display("Using packed FC1 bias block file: %s", DEFAULT_BIAS_HEX_FILE);
`endif
    $readmemh(DEFAULT_BIAS_HEX_FILE, bias_mem);
  end

  // synchronous ROM read
  always_ff @(posedge clk) begin
    bias_word_r <= bias_mem[ob];
  end

  // unpack block
  // bias_block[i] = bias_word_r[i*BIAS_WIDTH +: BIAS_WIDTH]
  always_comb begin
    for (i = 0; i < TILE_OUTPUT_SIZE; i = i + 1) begin
      bias_block[i] = bias_word_r[i*BIAS_WIDTH+:BIAS_WIDTH];
    end
  end

endmodule


