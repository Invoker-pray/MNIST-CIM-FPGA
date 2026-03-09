
module quantize_param_bank #(
    parameter string DEFAULT_QUANT_PARAM_FILE = "../route_b_output/quant_params.hex"
) (
    output logic [31:0] fc1_requant_mult,
    output logic [31:0] fc1_requant_shift,
    output logic [31:0] fc2_requant_mult,
    output logic [31:0] fc2_requant_shift
);
  logic [31:0] param_mem[0:3];
  string quant_param_file;

  initial begin
    quant_param_file = DEFAULT_QUANT_PARAM_FILE;

    if ($value$plusargs("QUANT_PARAM_FILE=%s", quant_param_file))
      $display("Using QUANT_PARAM_FILE from plusarg: %s", quant_param_file);
    else $display("Using default QUANT_PARAM_FILE: %s", quant_param_file);

    $readmemh(quant_param_file, param_mem);
  end

  always_comb begin
    fc1_requant_mult  = param_mem[0];
    fc1_requant_shift = param_mem[1];
    fc2_requant_mult  = param_mem[2];
    fc2_requant_shift = param_mem[3];
  end

endmodule
