
module mnist_inference_core_board #(
    parameter int N_SAMPLES = 20,
    parameter string DEFAULT_SAMPLE_HEX_FILE = "../data/mnist_samples_route_b_output_2.hex",
    parameter string DEFAULT_FC1_WEIGHT_HEX_FILE = "../route_b_output_2/fc1_weight_int8.hex",
    parameter string DEFAULT_FC1_BIAS_HEX_FILE = "../route_b_output_2/fc1_bias_int32.hex",
    parameter string DEFAULT_QUANT_PARAM_FILE = "../route_b_output_2/quant_params.hex",
    parameter string DEFAULT_FC2_WEIGHT_HEX_FILE = "../route_b_output_2/fc2_weight_int8.hex",
    parameter string DEFAULT_FC2_BIAS_HEX_FILE = "../route_b_output_2/fc2_bias_int32.hex"
) (
    input logic clk,
    input logic rst_n,
    input logic start,
    input logic [$clog2(N_SAMPLES)-1:0] sample_id,

    output logic busy,
    output logic done,

    output logic signed [mnist_cim_pkg::OUTPUT_WIDTH-1:0]
        logits_all [0:mnist_cim_pkg::FC2_OUT_DIM-1],

    output logic [$clog2(mnist_cim_pkg::FC2_OUT_DIM)-1:0] pred_class
);
  import mnist_cim_pkg::*;

  logic signed [  PSUM_WIDTH-1:0] fc1_acc_all [ 0:HIDDEN_DIM-1];
  logic signed [  PSUM_WIDTH-1:0] fc1_relu_all[ 0:HIDDEN_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] fc1_out_all [ 0:HIDDEN_DIM-1];
  logic signed [  PSUM_WIDTH-1:0] fc2_acc_all [0:FC2_OUT_DIM-1];

  fc1_multi_block_shared_sample_rom #(
      .PAR_OB(8),
      .BASE_OB(0),
      .N_SAMPLES(N_SAMPLES),
      .DEFAULT_SAMPLE_HEX_FILE(DEFAULT_SAMPLE_HEX_FILE),
      .DEFAULT_WEIGHT_HEX_FILE(DEFAULT_FC1_WEIGHT_HEX_FILE),
      .DEFAULT_BIAS_HEX_FILE(DEFAULT_FC1_BIAS_HEX_FILE)
  ) u_fc1 (
      .clk(clk),
      .rst_n(rst_n),
      .start(start),
      .sample_id(sample_id),
      .busy(busy),
      .done(done),
      .fc1_acc_all(fc1_acc_all)
  );

  fc1_to_fc2_top_with_file #(
      .DEFAULT_QUANT_PARAM_FILE(DEFAULT_QUANT_PARAM_FILE),
      .DEFAULT_FC2_WEIGHT_HEX_FILE(DEFAULT_FC2_WEIGHT_HEX_FILE),
      .DEFAULT_FC2_BIAS_HEX_FILE(DEFAULT_FC2_BIAS_HEX_FILE)
  ) u_fc1_to_fc2 (
      .fc1_acc_all (fc1_acc_all),
      .fc1_relu_all(fc1_relu_all),
      .fc1_out_all (fc1_out_all),
      .fc2_acc_all (fc2_acc_all),
      .logits_all  (logits_all)
  );

  argmax_int8 u_argmax (
      .logits_all(logits_all),
      .pred_class(pred_class)
  );

endmodule
