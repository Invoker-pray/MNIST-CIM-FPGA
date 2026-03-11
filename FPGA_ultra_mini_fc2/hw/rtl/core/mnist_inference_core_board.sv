
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
        logits_all[0:mnist_cim_pkg::FC2_OUT_DIM-1],

    output logic [$clog2(mnist_cim_pkg::FC2_OUT_DIM)-1:0] pred_class
);
  import mnist_cim_pkg::*;

  localparam int FC1_PAR_OB = 1;
  localparam int FC1_PARTIAL_LEN = FC1_PAR_OB * TILE_OUTPUT_SIZE;
  localparam int OB_SEL_W = (N_OUTPUT_BLOCKS > 1) ? $clog2(N_OUTPUT_BLOCKS) : 1;

  typedef enum logic [2:0] {
    S_IDLE      = 3'd0,
    S_FC1_START = 3'd1,
    S_FC1_WAIT  = 3'd2,
    S_FC2_START = 3'd3,
    S_FC2_WAIT  = 3'd4,
    S_DONE      = 3'd5
  } state_t;

  state_t state, state_n;

  logic fc1_start;
  logic fc1_busy, fc1_done;
  logic [OB_SEL_W-1:0] fc1_base_ob;

  logic fc2_start;
  logic fc2_busy, fc2_done;

  logic signed [  PSUM_WIDTH-1:0] fc1_acc_partial [0:FC1_PARTIAL_LEN-1];
  logic signed [  PSUM_WIDTH-1:0] fc1_acc_all_full[     0:HIDDEN_DIM-1];
  logic signed [  PSUM_WIDTH-1:0] fc1_relu_all    [     0:HIDDEN_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] fc1_out_all     [     0:HIDDEN_DIM-1];
  logic signed [  PSUM_WIDTH-1:0] fc2_acc_all     [    0:FC2_OUT_DIM-1];

  integer                         i;

  assign fc1_base_ob = '0;

  // ------------------------------------------------------------
  // State registers + FC1 result capture
  // ------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_IDLE;
      for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
        fc1_acc_all_full[i] <= '0;
      end
    end else begin
      state <= state_n;

      if (state == S_IDLE && start) begin
        for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
          fc1_acc_all_full[i] <= '0;
        end
      end

      // HIDDEN_DIM=16 and FC1_PAR_OB=1 => exactly one block
      if (state == S_FC1_WAIT && fc1_done) begin
        for (i = 0; i < FC1_PARTIAL_LEN; i = i + 1) begin
          fc1_acc_all_full[i] <= fc1_acc_partial[i];
        end
      end
    end
  end

  // ------------------------------------------------------------
  // FSM
  // ------------------------------------------------------------
  always_comb begin
    state_n   = state;
    fc1_start = 1'b0;
    fc2_start = 1'b0;
    busy      = 1'b0;
    done      = 1'b0;

    case (state)
      S_IDLE: begin
        if (start) begin
          state_n = S_FC1_START;
        end
      end

      S_FC1_START: begin
        busy      = 1'b1;
        fc1_start = 1'b1;  // one-cycle pulse
        state_n   = S_FC1_WAIT;
      end

      S_FC1_WAIT: begin
        busy = 1'b1;
        if (fc1_done) begin
          state_n = S_FC2_START;
        end
      end

      S_FC2_START: begin
        busy      = 1'b1;
        fc2_start = 1'b1;  // one-cycle pulse
        state_n   = S_FC2_WAIT;
      end

      S_FC2_WAIT: begin
        busy = 1'b1;
        if (fc2_done) begin
          state_n = S_DONE;
        end
      end

      S_DONE: begin
        done = 1'b1;
        if (!start) begin
          state_n = S_IDLE;
        end
      end

      default: begin
        state_n   = S_IDLE;
        fc1_start = 1'b0;
        fc2_start = 1'b0;
        busy      = 1'b0;
        done      = 1'b0;
      end
    endcase
  end

  // ------------------------------------------------------------
  // FC1 engine
  // ------------------------------------------------------------
  fc1_multi_block_shared_sample_rom #(
      .PAR_OB(FC1_PAR_OB),
      .N_SAMPLES(N_SAMPLES),
      .DEFAULT_SAMPLE_HEX_FILE(DEFAULT_SAMPLE_HEX_FILE),
      .DEFAULT_WEIGHT_HEX_FILE(DEFAULT_FC1_WEIGHT_HEX_FILE),
      .DEFAULT_BIAS_HEX_FILE(DEFAULT_FC1_BIAS_HEX_FILE)
  ) u_fc1 (
      .clk        (clk),
      .rst_n      (rst_n),
      .start      (fc1_start),
      .sample_id  (sample_id),
      .base_ob    (fc1_base_ob),
      .busy       (fc1_busy),
      .done       (fc1_done),
      .fc1_acc_all(fc1_acc_partial)
  );

  // ------------------------------------------------------------
  // FC1 requant + FC2
  // ------------------------------------------------------------
  fc1_to_fc2_top_with_file #(
      .DEFAULT_QUANT_PARAM_FILE(DEFAULT_QUANT_PARAM_FILE),
      .DEFAULT_FC2_WEIGHT_HEX_FILE(DEFAULT_FC2_WEIGHT_HEX_FILE),
      .DEFAULT_FC2_BIAS_HEX_FILE(DEFAULT_FC2_BIAS_HEX_FILE)
  ) u_fc1_to_fc2 (
      .clk         (clk),
      .rst_n       (rst_n),
      .start       (fc2_start),
      .fc1_acc_all (fc1_acc_all_full),
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
