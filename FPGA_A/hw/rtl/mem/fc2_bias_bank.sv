
module fc2_bias_bank #(
    parameter string DEFAULT_BIAS_HEX_FILE = "fc2_bias_int32.hex"
) (
    input logic clk,
    input logic [$clog2(mnist_cim_pkg::FC2_OUT_DIM)-1:0] addr,
    output logic signed [mnist_cim_pkg::BIAS_WIDTH-1:0] bias_data
);
  import mnist_cim_pkg::*;

  (* rom_style = "block" *)
  logic signed [BIAS_WIDTH-1:0] bias_mem[0:FC2_OUT_DIM-1];

  initial begin
`ifndef SYNTHESIS
    $display("Using default FC2_BIAS_FILE: %s", DEFAULT_BIAS_HEX_FILE);
`endif
    $readmemh(DEFAULT_BIAS_HEX_FILE, bias_mem);
  end

  // synchronous ROM read
  always_ff @(posedge clk) begin
    bias_data <= bias_mem[addr];
  end

endmodule

