
module fc1_relu_requantize #(
    parameter int FC1_REQUANT_MULT  = 2040478460,
    parameter int FC1_REQUANT_SHIFT = 43
) (
    input logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0] fc1_acc_all[0:mnist_cim_pkg::HIDDEN_DIM-1],

    output logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0] fc1_relu_all[0:mnist_cim_pkg::HIDDEN_DIM-1],

    output logic signed [mnist_cim_pkg::OUTPUT_WIDTH-1:0] fc1_out_all[0:mnist_cim_pkg::HIDDEN_DIM-1]
);
  import mnist_cim_pkg::*;

  function automatic signed [OUTPUT_WIDTH-1:0] requantize_int32_to_int8(
      input logic signed [PSUM_WIDTH-1:0] x, input int mult, input int rshift);
    longint signed prod;
    longint signed shifted;
    begin
      prod = x * mult;
      shifted = (prod + (64'sd1 <<< (rshift - 1))) >>> rshift;

      if (shifted > 127) requantize_int32_to_int8 = 8'sd127;
      else if (shifted < -128) requantize_int32_to_int8 = -8'sd128;
      else requantize_int32_to_int8 = shifted[OUTPUT_WIDTH-1:0];
    end
  endfunction

  integer i;
  always_comb begin
    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
      if (fc1_acc_all[i] > 0) fc1_relu_all[i] = fc1_acc_all[i];
      else fc1_relu_all[i] = '0;

      fc1_out_all[i] =
          requantize_int32_to_int8(fc1_relu_all[i], FC1_REQUANT_MULT, FC1_REQUANT_SHIFT);
    end
  end

endmodule
