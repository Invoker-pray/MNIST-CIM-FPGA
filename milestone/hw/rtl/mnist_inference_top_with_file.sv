
module mnist_inference_top_with_file #(
    parameter string DEFAULT_INPUT_HEX_FILE = "../route_b_output_2/input_0.hex",
    parameter string DEFAULT_FC1_WEIGHT_HEX_FILE = "../route_b_output_2/fc1_weight_int8.hex",
    parameter string DEFAULT_FC1_BIAS_HEX_FILE = "../route_b_output_2/fc1_bias_int32.hex",
    parameter string DEFAULT_QUANT_PARAM_FILE = "../route_b_output_2/quant_params.hex",
    parameter string DEFAULT_FC2_WEIGHT_HEX_FILE = "../route_b_output_2/fc2_weight_int8.hex",
    parameter string DEFAULT_FC2_BIAS_HEX_FILE = "../route_b_output_2/fc2_bias_int32.hex"
) (
    input logic clk,
    input logic rst_n,
    input logic start,

    output logic busy,
    output logic done,

    output logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0] fc1_acc_all[0:mnist_cim_pkg::HIDDEN_DIM-1],

    output logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0] fc1_relu_all[0:mnist_cim_pkg::HIDDEN_DIM-1],

    output logic signed [mnist_cim_pkg::OUTPUT_WIDTH-1:0]
        fc1_out_all [0:mnist_cim_pkg::HIDDEN_DIM-1],

    output logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0] fc2_acc_all[0:mnist_cim_pkg::FC2_OUT_DIM-1],

    output logic signed [mnist_cim_pkg::OUTPUT_WIDTH-1:0]
        logits_all [0:mnist_cim_pkg::FC2_OUT_DIM-1],

    output logic [$clog2(mnist_cim_pkg::FC2_OUT_DIM)-1:0] pred_class
);
  import mnist_cim_pkg::*;

  fc1_multi_block_shared_input #(
      .PAR_OB(8),
      .BASE_OB(0),
      .DEFAULT_INPUT_HEX_FILE(DEFAULT_INPUT_HEX_FILE),
      .DEFAULT_WEIGHT_HEX_FILE(DEFAULT_FC1_WEIGHT_HEX_FILE),
      .DEFAULT_BIAS_HEX_FILE(DEFAULT_FC1_BIAS_HEX_FILE)
  ) u_fc1_multi_block_shared_input (
      .clk(clk),
      .rst_n(rst_n),
      .start(start),
      .busy(busy),
      .done(done),
      .fc1_acc_all(fc1_acc_all)
  );

  fc1_to_fc2_top_with_file #(
      .DEFAULT_QUANT_PARAM_FILE(DEFAULT_QUANT_PARAM_FILE),
      .DEFAULT_FC2_WEIGHT_HEX_FILE(DEFAULT_FC2_WEIGHT_HEX_FILE),
      .DEFAULT_FC2_BIAS_HEX_FILE(DEFAULT_FC2_BIAS_HEX_FILE)
  ) u_fc1_to_fc2_top_with_file (
      .fc1_acc_all (fc1_acc_all),
      .fc1_relu_all(fc1_relu_all),
      .fc1_out_all (fc1_out_all),
      .fc2_acc_all (fc2_acc_all),
      .logits_all  (logits_all)
  );

  argmax_int8 u_argmax_int8 (
      .logits_all(logits_all),
      .pred_class(pred_class)
  );

endmodule
