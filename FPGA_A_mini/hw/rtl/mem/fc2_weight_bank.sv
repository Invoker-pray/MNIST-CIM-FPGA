
module fc2_weight_bank #(
    parameter string DEFAULT_WEIGHT_HEX_FILE = "fc2_weight_int8.hex"
) (
    input logic clk,
    input logic [$clog2(mnist_cim_pkg::FC2_WEIGHT_DEPTH)-1:0] addr,
    output logic signed [mnist_cim_pkg::WEIGHT_WIDTH-1:0] w_data
);
  import mnist_cim_pkg::*;

  (* rom_style = "block" *)
  logic signed [WEIGHT_WIDTH-1:0] weight_mem[0:FC2_WEIGHT_DEPTH-1];

  initial begin
`ifndef SYNTHESIS
    $display("Using default FC2_WEIGHT_HEX_FILE: %s", DEFAULT_WEIGHT_HEX_FILE);
`endif
    $readmemh(DEFAULT_WEIGHT_HEX_FILE, weight_mem);
  end

  // synchronous ROM read
  always_ff @(posedge clk) begin
    w_data <= weight_mem[addr];
  end

endmodule

