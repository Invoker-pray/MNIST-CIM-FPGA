// fc2_core_with_file.sv
// 新增 fc2_weight_bank 和 fc2_bias_bank 写口透传，移除默认文件参数

module fc2_core_with_file (
    input logic clk,
    input logic rst_n,
    input logic start,

    input logic [31:0] fc2_requant_mult,
    input logic [31:0] fc2_requant_shift,

    input logic signed [mnist_cim_pkg::OUTPUT_WIDTH-1:0]
        fc1_out_all [0:mnist_cim_pkg::FC2_IN_DIM-1],

    output logic busy,
    output logic done,

    output logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0]
        fc2_acc_all [0:mnist_cim_pkg::FC2_OUT_DIM-1],
    output logic signed [mnist_cim_pkg::OUTPUT_WIDTH-1:0]
        logits_all  [0:mnist_cim_pkg::FC2_OUT_DIM-1],

    // ── Write port: fc2_weight_bank ───────────────────────────────────────
    input logic        fc2w_wr_en,
    input logic [31:0] fc2w_wr_data,

    // ── Write port: fc2_bias_bank ─────────────────────────────────────────
    input logic        fc2b_wr_en,
    input logic [31:0] fc2b_wr_data
);
  import mnist_cim_pkg::*;

  typedef enum logic [2:0] {
    S_IDLE      = 3'd0,
    S_INIT      = 3'd1,
    S_BIAS_WAIT = 3'd2,
    S_READ_W    = 3'd3,
    S_W_WAIT    = 3'd4,
    S_MAC       = 3'd5,
    S_STORE     = 3'd6,
    S_DONE      = 3'd7
  } state_t;

  state_t state, state_n;

  logic [$clog2(FC2_OUT_DIM)-1:0] out_idx, out_idx_n;
  logic [$clog2(FC2_IN_DIM)-1:0]  in_idx,  in_idx_n;
  logic signed [PSUM_WIDTH-1:0]   acc_reg, acc_reg_n;

  logic [$clog2(FC2_WEIGHT_DEPTH)-1:0] weight_addr;
  logic signed [WEIGHT_WIDTH-1:0]      weight_data;
  logic [$clog2(FC2_OUT_DIM)-1:0]      bias_addr;
  logic signed [BIAS_WIDTH-1:0]        bias_data;

  logic signed [PSUM_WIDTH-1:0] mul_term;
  logic signed [PSUM_WIDTH-1:0] acc_after_mul;

  integer k;

  assign weight_addr = out_idx * FC2_IN_DIM + in_idx;
  assign bias_addr   = out_idx;

  fc2_weight_bank u_fc2_weight_bank (
      .clk     (clk),
      .rst_n   (rst_n),
      .addr    (weight_addr),
      .w_data  (weight_data),
      .wr_en   (fc2w_wr_en),
      .wr_data (fc2w_wr_data)
  );

  fc2_bias_bank u_fc2_bias_bank (
      .clk      (clk),
      .rst_n    (rst_n),
      .addr     (bias_addr),
      .bias_data(bias_data),
      .wr_en    (fc2b_wr_en),
      .wr_data  (fc2b_wr_data)
  );

  function automatic signed [OUTPUT_WIDTH-1:0] requantize_int32_to_int8(
      input logic signed [PSUM_WIDTH-1:0] x,
      input logic [31:0] mult,
      input logic [31:0] rshift);
    longint signed prod, shifted;
    begin
      prod = x * $signed(mult);
      if (rshift == 0) shifted = prod;
      else shifted = (prod + (64'sd1 <<< (rshift - 1))) >>> rshift;
      if      (shifted >  127) requantize_int32_to_int8 =  8'sd127;
      else if (shifted < -128) requantize_int32_to_int8 = -8'sd128;
      else                     requantize_int32_to_int8 = shifted[OUTPUT_WIDTH-1:0];
    end
  endfunction

  assign mul_term      = $signed(fc1_out_all[in_idx]) * $signed(weight_data);
  assign acc_after_mul = acc_reg + mul_term;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state   <= S_IDLE;
      out_idx <= '0;
      in_idx  <= '0;
      acc_reg <= '0;
      for (k = 0; k < FC2_OUT_DIM; k = k + 1) begin
        fc2_acc_all[k] <= '0;
        logits_all[k]  <= '0;
      end
    end else begin
      state   <= state_n;
      out_idx <= out_idx_n;
      in_idx  <= in_idx_n;
      acc_reg <= acc_reg_n;

      if (state == S_IDLE && start) begin
        for (k = 0; k < FC2_OUT_DIM; k = k + 1) begin
          fc2_acc_all[k] <= '0;
          logits_all[k]  <= '0;
        end
      end

      if (state == S_STORE) begin
        fc2_acc_all[out_idx] <= acc_reg;
        logits_all[out_idx]  <= requantize_int32_to_int8(
            acc_reg, fc2_requant_mult, fc2_requant_shift);
      end
    end
  end

  always_comb begin
    state_n   = state;
    out_idx_n = out_idx;
    in_idx_n  = in_idx;
    acc_reg_n = acc_reg;
    busy = 1'b0;
    done = 1'b0;

    case (state)
      S_IDLE: begin
        if (start) begin
          out_idx_n = '0;
          in_idx_n  = '0;
          acc_reg_n = '0;
          state_n   = S_INIT;
        end
      end
      S_INIT: begin
        busy     = 1'b1;
        in_idx_n = '0;
        state_n  = S_BIAS_WAIT;
      end
      S_BIAS_WAIT: begin
        busy      = 1'b1;
        acc_reg_n = bias_data;
        state_n   = S_READ_W;
      end
      S_READ_W: begin
        busy    = 1'b1;
        state_n = S_W_WAIT;
      end
      S_W_WAIT: begin
        busy    = 1'b1;
        state_n = S_MAC;
      end
      S_MAC: begin
        busy      = 1'b1;
        acc_reg_n = acc_after_mul;
        if (in_idx == FC2_IN_DIM - 1) begin
          state_n = S_STORE;
        end else begin
          in_idx_n = in_idx + 1'b1;
          state_n  = S_READ_W;
        end
      end
      S_STORE: begin
        busy = 1'b1;
        if (out_idx == FC2_OUT_DIM - 1) begin
          state_n = S_DONE;
        end else begin
          out_idx_n = out_idx + 1'b1;
          in_idx_n  = '0;
          acc_reg_n = '0;
          state_n   = S_INIT;
        end
      end
      S_DONE: begin
        done = 1'b1;
        if (!start) state_n = S_IDLE;
      end
      default: begin
        state_n   = S_IDLE;
        out_idx_n = '0;
        in_idx_n  = '0;
        acc_reg_n = '0;
        busy = 1'b0;
        done = 1'b0;
      end
    endcase
  end

endmodule
