
module fc1_multi_block_shared_sample_rom #(
    parameter int PAR_OB = 8,
    parameter int BASE_OB = 0,
    parameter int N_SAMPLES = 20,
    parameter string DEFAULT_SAMPLE_HEX_FILE = "../data/mnist_samples_route_b_output_2.hex",
    parameter string DEFAULT_WEIGHT_HEX_FILE = "../route_b_output_2/fc1_weight_int8.hex",
    parameter string DEFAULT_BIAS_HEX_FILE = "../route_b_output_2/fc1_bias_int32.hex"
) (
    input logic clk,
    input logic rst_n,
    input logic start,
    input logic [$clog2(N_SAMPLES)-1:0] sample_id,

    output logic busy,
    output logic done,

    output logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0]
        fc1_acc_all [0:PAR_OB*mnist_cim_pkg::TILE_OUTPUT_SIZE-1]
);
  import mnist_cim_pkg::*;

  typedef enum logic [1:0] {
    S_IDLE  = 2'd0,
    S_CLEAR = 2'd1,
    S_ACCUM = 2'd2,
    S_DONE  = 2'd3
  } state_t;

  state_t state, state_n;

  logic [$clog2(N_INPUT_BLOCKS)-1:0] ib, ib_n;
  logic clear_psum;
  logic en_psum;

  logic signed [INPUT_WIDTH-1:0] x_tile[0:TILE_INPUT_SIZE-1];

  logic [X_EFF_WIDTH-1:0] x_eff_tile[0:TILE_INPUT_SIZE-1];

  logic signed [PSUM_WIDTH-1:0] fc1_acc_block[0:PAR_OB-1][0:TILE_OUTPUT_SIZE-1];

  // ------------------------------------------------------------
  // Sample ROM: provide one input tile according to sample_id + ib
  // ------------------------------------------------------------
  mnist_sample_rom #(
      .N_SAMPLES(N_SAMPLES),
      .DEFAULT_SAMPLE_HEX_FILE(DEFAULT_SAMPLE_HEX_FILE)
  ) u_mnist_sample_rom (
      .sample_id(sample_id),
      .ib(ib),
      .x_tile(x_tile),
      .x_eff_tile(x_eff_tile)
  );

  // ------------------------------------------------------------
  // State / ib counter registers
  // ------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_IDLE;
      ib    <= '0;
    end else begin
      state <= state_n;
      ib    <= ib_n;
    end
  end

  // ------------------------------------------------------------
  // Control FSM
  // ------------------------------------------------------------
  always_comb begin
    state_n    = state;
    ib_n       = ib;
    clear_psum = 1'b0;
    en_psum    = 1'b0;
    busy       = 1'b0;
    done       = 1'b0;

    case (state)
      S_IDLE: begin
        ib_n = '0;
        if (start) state_n = S_CLEAR;
      end

      S_CLEAR: begin
        busy       = 1'b1;
        clear_psum = 1'b1;
        ib_n       = '0;
        state_n    = S_ACCUM;
      end

      S_ACCUM: begin
        busy    = 1'b1;
        en_psum = 1'b1;

        if (ib == N_INPUT_BLOCKS - 1) begin
          state_n = S_DONE;
          ib_n    = ib;
        end else begin
          ib_n = ib + 1'b1;
        end
      end

      S_DONE: begin
        done = 1'b1;
        if (!start) state_n = S_IDLE;
      end

      default: begin
        state_n    = S_IDLE;
        ib_n       = '0;
        clear_psum = 1'b0;
        en_psum    = 1'b0;
        busy       = 1'b0;
        done       = 1'b0;
      end
    endcase
  end

  // ------------------------------------------------------------
  // Parallel OB engines
  // ------------------------------------------------------------
  genvar g_ob, g_idx;
  generate
    for (g_ob = 0; g_ob < PAR_OB; g_ob = g_ob + 1) begin : GEN_OB_ENGINE
      localparam logic [$clog2(N_OUTPUT_BLOCKS)-1:0] OB_SEL = BASE_OB + g_ob;

      fc1_ob_engine_shared_input #(
          .DEFAULT_WEIGHT_HEX_FILE(DEFAULT_WEIGHT_HEX_FILE),
          .DEFAULT_BIAS_HEX_FILE  (DEFAULT_BIAS_HEX_FILE)
      ) u_fc1_ob_engine_shared_input (
          .clk(clk),
          .rst_n(rst_n),
          .clear_psum(clear_psum),
          .en_psum(en_psum),
          .ib(ib),
          .ob_sel(OB_SEL),
          .x_eff_tile(x_eff_tile),
          .fc1_acc_block(fc1_acc_block[g_ob])
      );

      for (g_idx = 0; g_idx < TILE_OUTPUT_SIZE; g_idx = g_idx + 1) begin : GEN_PACK
        assign fc1_acc_all[g_ob*TILE_OUTPUT_SIZE+g_idx] = fc1_acc_block[g_ob][g_idx];
      end
    end
  endgenerate

endmodule

