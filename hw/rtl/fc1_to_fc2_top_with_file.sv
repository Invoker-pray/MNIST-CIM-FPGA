
module fc1_to_fc2_top #(
    parameter string DEFAULT_QUANT_PARAM_FILE = "../route_b_output/quant_params.hex",
    parameter string DEFAULT_FC2_WEIGHT_HEX_FILE = "../route_b_output/fc2_weight_int8.hex",
    parameter string DEFAULT_FC2_BIAS_HEX_FILE = "../route_b_output/fc2_bias_int32.hex"
) (
    input logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0] fc1_acc_all[0:mnist_cim_pkg::HIDDEN_DIM-1],

    output logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0] fc1_relu_all[0:mnist_cim_pkg::HIDDEN_DIM-1],

    output logic signed [mnist_cim_pkg::OUTPUT_WIDTH-1:0]
        fc1_out_all [0:mnist_cim_pkg::HIDDEN_DIM-1],

    output logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0] fc2_acc_all[0:mnist_cim_pkg::FC2_OUT_DIM-1],

    output logic signed [mnist_cim_pkg::OUTPUT_WIDTH-1:0] logits_all[0:mnist_cim_pkg::FC2_OUT_DIM-1]
);
  import mnist_cim_pkg::*;

  logic [31:0] fc1_requant_mult;
  logic [31:0] fc1_requant_shift;
  logic [31:0] fc2_requant_mult;
  logic [31:0] fc2_requant_shift;

  quant_param_bank #(
      .DEFAULT_QUANT_PARAM_FILE(DEFAULT_QUANT_PARAM_FILE)
  ) u_quant_param_bank (
      .fc1_requant_mult (fc1_requant_mult),
      .fc1_requant_shift(fc1_requant_shift),
      .fc2_requant_mult (fc2_requant_mult),
      .fc2_requant_shift(fc2_requant_shift)
  );

  fc1_relu_requantize u_fc1_relu_requantize (
      .fc1_requant_mult (fc1_requant_mult),
      .fc1_requant_shift(fc1_requant_shift),
      .fc1_acc_all      (fc1_acc_all),
      .fc1_relu_all     (fc1_relu_all),
      .fc1_out_all      (fc1_out_all)
  );

  fc2_core #(
      .DEFAULT_WEIGHT_HEX_FILE(DEFAULT_FC2_WEIGHT_HEX_FILE),
      .DEFAULT_BIAS_HEX_FILE  (DEFAULT_FC2_BIAS_HEX_FILE)
  ) u_fc2_core (
      .fc2_requant_mult (fc2_requant_mult),
      .fc2_requant_shift(fc2_requant_shift),
      .fc1_out_all      (fc1_out_all),
      .fc2_acc_all      (fc2_acc_all),
      .logits_all       (logits_all)
  );

endmodule
