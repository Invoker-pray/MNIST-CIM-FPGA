
module fc1_bias_bank #(
    parameter string DEFAULT_BIAS_HEX_FILE = "fc1_bias_int32.hex"
) (
    input logic [$clog2(mnist_cim_pkg::N_OUTPUT_BLOCKS)-1:0] ob,

    output logic signed [mnist_cim_pkg::BIAS_WIDTH-1:0]
        bias_block [0:mnist_cim_pkg::TILE_OUTPUT_SIZE-1]
);

  import mnist_cim_pkg::*;

  //string bias_file;

  logic signed [BIAS_WIDTH-1:0] bias_mem[0:FC1_BIAS_DEPTH-1];

  integer i;
  integer idx;


  initial begin
`ifndef SYNTHESIS
    $display("Using default FC1_BIAS_FILE: %s", DEFAULT_BIAS_HEX_FILE);
`endif
    $readmemh(DEFAULT_BIAS_HEX_FILE, bias_mem);
  end


  always_comb begin
    for (i = 0; i < TILE_OUTPUT_SIZE; i = i + 1) begin
      idx = ob * TILE_OUTPUT_SIZE + i;
      bias_block[i] = bias_mem[idx];
    end
  end

endmodule
