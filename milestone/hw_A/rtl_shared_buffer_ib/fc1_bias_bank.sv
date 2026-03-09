
module fc1_bias_bank #(
    parameter string DEFAULT_BIAS_HEX_FILE =
        "../../CIM-sw-version1/sw/train_quantize/route_b_output/fc1_bias_int32.hex"
) (
    input logic [$clog2(mnist_cim_pkg::N_OUTPUT_BLOCKS)-1:0] ob,

    output logic signed [mnist_cim_pkg::BIAS_WIDTH-1:0]
        bias_block [0:mnist_cim_pkg::TILE_OUTPUT_SIZE-1]
);

  import mnist_cim_pkg::*;

  string bias_file;

  logic signed [BIAS_WIDTH-1:0] bias_mem[0:FC1_BIAS_DEPTH-1];

  integer i;
  integer idx;

  initial begin
    bias_file = DEFAULT_BIAS_HEX_FILE;

    if ($value$plusargs("FC1_BIAS_FILE=%s", bias_file)) begin
      $display("Using FC1_BIAS_FILE from plusarg: %s", bias_file);
    end else begin
      $display("Using default FC1_BIAS_FILE: %s", bias_file);
    end

    $readmemh(bias_file, bias_mem);
  end

  always_comb begin
    for (i = 0; i < TILE_OUTPUT_SIZE; i = i + 1) begin
      idx = ob * TILE_OUTPUT_SIZE + i;
      bias_block[i] = bias_mem[idx];
    end
  end

endmodule
