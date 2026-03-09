
module fc1_cim_core_block #(
    parameter string DEFAULT_INPUT_HEX_FILE  =
        "../../CIM-sw-version1/sw/train_quantize/route_b_output/input_0.hex",
    parameter string DEFAULT_WEIGHT_HEX_FILE =
        "../../CIM-sw-version1/sw/train_quantize/route_b_output/fc1_weight_int8.hex",
    parameter string DEFAULT_BIAS_HEX_FILE   =
        "../../CIM-sw-version1/sw/train_quantize/route_b_output/fc1_bias_int32.hex"
) (
    input logic clk,
    input logic rst_n,
    input logic start,

    input logic [$clog2(mnist_cim_pkg::N_OUTPUT_BLOCKS)-1:0] ob_sel,

    output logic busy,
    output logic done,

    output logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0]
        fc1_acc_block [0:mnist_cim_pkg::TILE_OUTPUT_SIZE-1]
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

  string input_file;
  string weight_file;
  string bias_file;

  // --------------------------------------------
  // Interconnect wires
  // --------------------------------------------
  logic signed [INPUT_WIDTH-1:0] x_tile[0:TILE_INPUT_SIZE-1];

  logic [X_EFF_WIDTH-1:0] x_eff_tile[0:TILE_INPUT_SIZE-1];

  logic signed [WEIGHT_WIDTH-1:0] w_tile[0:TILE_OUTPUT_SIZE-1][0:TILE_INPUT_SIZE-1];

  logic signed [PSUM_WIDTH-1:0] tile_psum[0:TILE_OUTPUT_SIZE-1];

  logic signed [PSUM_WIDTH-1:0] psum[0:TILE_OUTPUT_SIZE-1];

  logic signed [BIAS_WIDTH-1:0] bias_block[0:TILE_OUTPUT_SIZE-1];

  logic signed [PSUM_WIDTH-1:0] fc1_acc_with_bias[0:TILE_OUTPUT_SIZE-1];

  // --------------------------------------------
  // Read plusargs once
  // --------------------------------------------
  initial begin
    input_file  = DEFAULT_INPUT_HEX_FILE;
    weight_file = DEFAULT_WEIGHT_HEX_FILE;
    bias_file   = DEFAULT_BIAS_HEX_FILE;

    if ($value$plusargs("INPUT_HEX_FILE=%s", input_file)) begin
      $display("Using INPUT_HEX_FILE from plusarg: %s", input_file);
    end else begin
      $display("Using default INPUT_HEX_FILE: %s", input_file);
    end

    if ($value$plusargs("WEIGHT_HEX_FILE=%s", weight_file)) begin
      $display("Using WEIGHT_HEX_FILE from plusarg: %s", weight_file);
    end else begin
      $display("Using default WEIGHT_HEX_FILE: %s", weight_file);
    end

    if ($value$plusargs("FC1_BIAS_FILE=%s", bias_file)) begin
      $display("Using FC1_BIAS_FILE from plusarg: %s", bias_file);
    end else begin
      $display("Using default FC1_BIAS_FILE: %s", bias_file);
    end
  end

  // --------------------------------------------
  // Submodules
  // --------------------------------------------
  input_buffer #(
      .DEFAULT_INPUT_HEX_FILE(DEFAULT_INPUT_HEX_FILE)
  ) u_input_buffer (
      .ib(ib),
      .x_tile(x_tile),
      .x_eff_tile(x_eff_tile)
  );

  fc1_weight_bank #(
      .DEFAULT_WEIGHT_HEX_FILE(DEFAULT_WEIGHT_HEX_FILE)
  ) u_fc1_weight_bank (
      .ob(ob_sel),
      .ib(ib),
      .w_tile(w_tile)
  );

  cim_tile u_cim_tile (
      .x_eff_tile(x_eff_tile),
      .w_tile(w_tile),
      .tile_psum(tile_psum)
  );

  psum_accum u_psum_accum (
      .clk(clk),
      .rst_n(rst_n),
      .clear(clear_psum),
      .en(en_psum),
      .tile_psum(tile_psum),
      .psum(psum)
  );

  fc1_bias_bank #(
      .DEFAULT_BIAS_HEX_FILE(DEFAULT_BIAS_HEX_FILE)
  ) u_fc1_bias_bank (
      .ob(ob_sel),
      .bias_block(bias_block)
  );

  // --------------------------------------------
  // FSM seq
  // --------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_IDLE;
      ib    <= '0;
    end else begin
      state <= state_n;
      ib    <= ib_n;
    end
  end

  // --------------------------------------------
  // FSM comb
  // --------------------------------------------
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
          ib_n    = ib + 1'b1;
          state_n = S_ACCUM;
        end
      end

      S_DONE: begin
        done = 1'b1;
        if (!start) state_n = S_IDLE;
      end

      default: begin
        state_n = S_IDLE;
        ib_n    = '0;
      end
    endcase
  end

  // --------------------------------------------
  // Output connect
  // fc1_acc_0.hex = pure MAC accumulation + bias
  // --------------------------------------------
  genvar g;
  generate
    for (g = 0; g < TILE_OUTPUT_SIZE; g = g + 1) begin : GEN_OUT
      assign fc1_acc_with_bias[g] = psum[g] + bias_block[g];  // Add bias to psum
      assign fc1_acc_block[g]     = fc1_acc_with_bias[g];  // Output the final result
    end
  endgenerate

endmodule

