// fc1_to_fc2_top_with_file.sv
// 新增 quantize_param_bank / fc2_weight / fc2_bias 写口透传

module fc1_to_fc2_top_with_file (
    input logic clk,
    input logic rst_n,
    input logic start,

    input logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0]
        fc1_acc_all [0:mnist_cim_pkg::HIDDEN_DIM-1],

    output logic busy,
    output logic done,

    output logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0]
        fc1_relu_all [0:mnist_cim_pkg::HIDDEN_DIM-1],
    output logic signed [mnist_cim_pkg::OUTPUT_WIDTH-1:0]
        fc1_out_all  [0:mnist_cim_pkg::HIDDEN_DIM-1],
    output logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0]
        fc2_acc_all  [0:mnist_cim_pkg::FC2_OUT_DIM-1],
    output logic signed [mnist_cim_pkg::OUTPUT_WIDTH-1:0]
        logits_all   [0:mnist_cim_pkg::FC2_OUT_DIM-1],

    // ── Write port: quantize_param_bank ──────────────────────────────────
    input logic        quant_wr_en,
    input logic [31:0] quant_wr_data,

    // ── Write port: fc2_weight_bank ───────────────────────────────────────
    input logic        fc2w_wr_en,
    input logic [31:0] fc2w_wr_data,

    // ── Write port: fc2_bias_bank ─────────────────────────────────────────
    input logic        fc2b_wr_en,
    input logic [31:0] fc2b_wr_data
);
  import mnist_cim_pkg::*;

  logic [31:0] fc1_requant_mult;
  logic [31:0] fc1_requant_shift;
  logic [31:0] fc2_requant_mult;
  logic [31:0] fc2_requant_shift;

  quantize_param_bank u_quantize_param_bank (
      .clk              (clk),
      .rst_n            (rst_n),
      .fc1_requant_mult (fc1_requant_mult),
      .fc1_requant_shift(fc1_requant_shift),
      .fc2_requant_mult (fc2_requant_mult),
      .fc2_requant_shift(fc2_requant_shift),
      .wr_en            (quant_wr_en),
      .wr_data          (quant_wr_data)
  );

  fc1_relu_requantize_with_file u_fc1_relu_requantize (
      .fc1_requant_mult (fc1_requant_mult),
      .fc1_requant_shift(fc1_requant_shift),
      .fc1_acc_all      (fc1_acc_all),
      .fc1_relu_all     (fc1_relu_all),
      .fc1_out_all      (fc1_out_all)
  );

  fc2_core_with_file u_fc2_core (
      .clk              (clk),
      .rst_n            (rst_n),
      .start            (start),
      .fc2_requant_mult (fc2_requant_mult),
      .fc2_requant_shift(fc2_requant_shift),
      .fc1_out_all      (fc1_out_all),
      .busy             (busy),
      .done             (done),
      .fc2_acc_all      (fc2_acc_all),
      .logits_all       (logits_all),
      .fc2w_wr_en       (fc2w_wr_en),
      .fc2w_wr_data     (fc2w_wr_data),
      .fc2b_wr_en       (fc2b_wr_en),
      .fc2b_wr_data     (fc2b_wr_data)
  );

endmodule
