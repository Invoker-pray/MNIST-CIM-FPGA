

module fc2_core_with_file #(
    parameter string DEFAULT_WEIGHT_HEX_FILE = "fc2_weight_int8.hex",
    parameter string DEFAULT_BIAS_HEX_FILE   = "fc2_bias_int32.hex"
) (
    input logic [31:0] fc2_requant_mult,
    input logic [31:0] fc2_requant_shift,

    input logic signed [mnist_cim_pkg::OUTPUT_WIDTH-1:0] fc1_out_all[0:mnist_cim_pkg::FC2_IN_DIM-1],

    output logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0] fc2_acc_all[0:mnist_cim_pkg::FC2_OUT_DIM-1],

    output logic signed [mnist_cim_pkg::OUTPUT_WIDTH-1:0] logits_all[0:mnist_cim_pkg::FC2_OUT_DIM-1]
);
  import mnist_cim_pkg::*;

  logic signed [WEIGHT_WIDTH-1:0] w_all[0:FC2_OUT_DIM-1][0:FC2_IN_DIM-1];

  logic signed [BIAS_WIDTH-1:0] bias_all[0:FC2_OUT_DIM-1];

  fc2_weight_bank #(
      .DEFAULT_WEIGHT_HEX_FILE(DEFAULT_WEIGHT_HEX_FILE)
  ) u_fc2_weight_bank (
      .w_all(w_all)
  );

  fc2_bias_bank #(
      .DEFAULT_BIAS_HEX_FILE(DEFAULT_BIAS_HEX_FILE)
  ) u_fc2_bias_bank (
      .bias_all(bias_all)
  );

  function automatic signed [OUTPUT_WIDTH-1:0] requantize_int32_to_int8(
      input logic signed [PSUM_WIDTH-1:0] x, input logic [31:0] mult, input logic [31:0] rshift);
    longint signed prod;
    longint signed shifted;
    begin
      prod = x * $signed(mult);
      shifted = (prod + (64'sd1 <<< (rshift - 1))) >>> rshift;

      if (shifted > 127) requantize_int32_to_int8 = 8'sd127;
      else if (shifted < -128) requantize_int32_to_int8 = -8'sd128;
      else requantize_int32_to_int8 = shifted[OUTPUT_WIDTH-1:0];
    end
  endfunction

  integer o, i;
  longint signed acc_tmp;

  always_comb begin
    for (o = 0; o < FC2_OUT_DIM; o = o + 1) begin
      acc_tmp = bias_all[o];

      for (i = 0; i < FC2_IN_DIM; i = i + 1) begin
        acc_tmp = acc_tmp + fc1_out_all[i] * w_all[o][i];
      end

      fc2_acc_all[o] = acc_tmp[PSUM_WIDTH-1:0];

      logits_all[o] = requantize_int32_to_int8(fc2_acc_all[o], fc2_requant_mult, fc2_requant_shift);
    end
  end

endmodule
