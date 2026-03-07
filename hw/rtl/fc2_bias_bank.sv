
module fc2_bias_bank #(
    parameter string DEFAULT_BIAS_HEX_FILE = "../route_b_output/fc2_bias_int32.hex"
) (
    output logic signed [mnist_cim_pkg::BIAS_WIDTH-1:0] bias_all[0:mnist_cim_pkg::FC2_OUT_DIM-1]
);
  import mnist_cim_pkg::*;

  string bias_file;

  logic signed [BIAS_WIDTH-1:0] bias_mem[0:FC2_OUT_DIM-1];

  integer i;

  initial begin
    bias_file = DEFAULT_BIAS_HEX_FILE;

    if ($value$plusargs("FC2_BIAS_FILE=%s", bias_file)) begin
      $display("Using FC2_BIAS_FILE from plusarg: %s", bias_file);
    end else begin
      $display("Using default FC2_BIAS_FILE: %s", bias_file);
    end

    $readmemh(bias_file, bias_mem);
  end

  always_comb begin
    for (i = 0; i < FC2_OUT_DIM; i = i + 1) begin
      bias_all[i] = bias_mem[i];
    end
  end

endmodule
