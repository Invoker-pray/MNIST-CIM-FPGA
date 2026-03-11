
module quantize_param_bank #(
    parameter string DEFAULT_QUANT_PARAM_FILE = "quant_params.hex"
) (
    output logic [31:0] fc1_requant_mult,
    output logic [31:0] fc1_requant_shift,
    output logic [31:0] fc2_requant_mult,
    output logic [31:0] fc2_requant_shift
);
  logic [31:0] param_mem[0:3];

  initial begin
`ifndef SYNTHESIS
    $display("Using default QUANT_PARAM_FILE: %s", DEFAULT_QUANT_PARAM_FILE);
`endif
    $readmemh(DEFAULT_QUANT_PARAM_FILE, param_mem);
  end

  always_comb begin
    fc1_requant_mult  = param_mem[0];
    fc1_requant_shift = param_mem[1];
    fc2_requant_mult  = param_mem[2];
    fc2_requant_shift = param_mem[3];
  end

endmodule
