
module fc2_weight_bank #(
    parameter string DEFAULT_WEIGHT_HEX_FILE = "fc2_weight_int8.hex"
) (
    output logic signed [mnist_cim_pkg::WEIGHT_WIDTH-1:0]
        w_all [0:mnist_cim_pkg::FC2_OUT_DIM-1]
              [0:mnist_cim_pkg::FC2_IN_DIM-1]
);
  import mnist_cim_pkg::*;

  // string weight_file;

  logic signed [WEIGHT_WIDTH-1:0] weight_mem[0:FC2_WEIGHT_DEPTH-1];

  integer o, i, addr;


  initial begin
`ifndef SYNTHESIS
    $display("Using default FC2_WEIGHT_HEX_FILE: %s", DEFAULT_WEIGHT_HEX_FILE);
`endif
    $readmemh(DEFAULT_WEIGHT_HEX_FILE, weight_mem);
  end


  always_comb begin
    for (o = 0; o < FC2_OUT_DIM; o = o + 1) begin
      for (i = 0; i < FC2_IN_DIM; i = i + 1) begin
        addr = o * FC2_IN_DIM + i;
        w_all[o][i] = weight_mem[addr];
      end
    end
  end

endmodule
