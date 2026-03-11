

module mnist_inference_core_board #(
    parameter int N_SAMPLES = 20,
    parameter string DEFAULT_SAMPLE_HEX_FILE = "mnist_samples_route_b_output_2.hex",
    parameter string DEFAULT_FC1_WEIGHT_HEX_FILE = "fc1_weight_int8.hex",
    parameter string DEFAULT_FC1_BIAS_HEX_FILE = "fc1_bias_int32.hex",
    parameter string DEFAULT_QUANT_PARAM_FILE = "quant_params.hex",
    parameter string DEFAULT_FC2_WEIGHT_HEX_FILE = "fc2_weight_int8.hex",
    parameter string DEFAULT_FC2_BIAS_HEX_FILE = "fc2_bias_int32.hex"
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

  typedef enum logic [2:0] {
    S_IDLE      = 3'd0,
    S_FC1_WAIT  = 3'd1,
    S_LATCH_FC1 = 3'd2,
    S_FC2_START = 3'd3,
    S_FC2_WAIT  = 3'd4,
    S_DONE      = 3'd5
  } state_t;

  state_t state, state_n;

  logic fc1_busy, fc1_done;
  logic fc2_busy, fc2_done;
  logic fc2_start;

  // FC1 raw output from submodule
  logic signed [PSUM_WIDTH-1:0] fc1_acc_all_wire[0:HIDDEN_DIM-1];
  // Latched FC1 final result for feeding FC2
  logic signed [PSUM_WIDTH-1:0] fc1_acc_all_reg[0:HIDDEN_DIM-1];

  logic signed [PSUM_WIDTH-1:0] fc1_relu_all[0:HIDDEN_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] fc1_out_all[0:HIDDEN_DIM-1];
  logic signed [PSUM_WIDTH-1:0] fc2_acc_all[0:FC2_OUT_DIM-1];

  integer i;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_IDLE;
      for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
        fc1_acc_all_reg[i] <= '0;
      end
    end else begin
      state <= state_n;

      // latch FC1 result exactly once after fc1_done
      if (state == S_LATCH_FC1) begin
        for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
          fc1_acc_all_reg[i] <= fc1_acc_all_wire[i];
        end
      end
    end
  end

  always_comb begin
    state_n   = state;
    fc2_start = 1'b0;
    busy      = 1'b0;
    done      = 1'b0;

    case (state)
      S_IDLE: begin
        if (start) state_n = S_FC1_WAIT;
      end

      S_FC1_WAIT: begin
        busy = 1'b1;
        if (fc1_done) state_n = S_LATCH_FC1;
      end

      S_LATCH_FC1: begin
        busy    = 1'b1;
        state_n = S_FC2_START;
      end

      S_FC2_START: begin
        busy      = 1'b1;
        fc2_start = 1'b1;
        state_n   = S_FC2_WAIT;
      end

      S_FC2_WAIT: begin
        busy = 1'b1;
        if (fc2_done) state_n = S_DONE;
      end

      S_DONE: begin
        done = 1'b1;
        if (!start) state_n = S_IDLE;
      end

      default: begin
        state_n   = S_IDLE;
        fc2_start = 1'b0;
        busy      = 1'b0;
        done      = 1'b0;
      end
    endcase
  end

  fc1_multi_block_shared_sample_rom #(
      .PAR_OB(8),
      .N_SAMPLES(N_SAMPLES),
      .DEFAULT_SAMPLE_HEX_FILE(DEFAULT_SAMPLE_HEX_FILE),
      .DEFAULT_WEIGHT_HEX_FILE(DEFAULT_FC1_WEIGHT_HEX_FILE),
      .DEFAULT_BIAS_HEX_FILE(DEFAULT_FC1_BIAS_HEX_FILE)
  ) u_fc1 (
      .clk        (clk),
      .rst_n      (rst_n),
      .start      (start),
      .sample_id  (sample_id),
      .base_ob    ('0),
      .busy       (fc1_busy),
      .done       (fc1_done),
      .fc1_acc_all(fc1_acc_all_wire)
  );

  fc1_to_fc2_top_with_file #(
      .DEFAULT_QUANT_PARAM_FILE(DEFAULT_QUANT_PARAM_FILE),
      .DEFAULT_FC2_WEIGHT_HEX_FILE(DEFAULT_FC2_WEIGHT_HEX_FILE),
      .DEFAULT_FC2_BIAS_HEX_FILE(DEFAULT_FC2_BIAS_HEX_FILE)
  ) u_fc1_to_fc2 (
      .clk         (clk),
      .rst_n       (rst_n),
      .start       (fc2_start),
      .fc1_acc_all (fc1_acc_all_reg),
      .busy        (fc2_busy),
      .done        (fc2_done),
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
