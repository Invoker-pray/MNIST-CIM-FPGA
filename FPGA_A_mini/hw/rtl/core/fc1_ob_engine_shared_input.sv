
module fc1_ob_engine_shared_input #(
    parameter string DEFAULT_WEIGHT_HEX_FILE = "fc1_weight_int8.hex",
    parameter string DEFAULT_BIAS_HEX_FILE   = "fc1_bias_int32.hex"
) (
    input logic clk,
    input logic rst_n,

    input logic clear_psum,
    input logic en_psum,

    input logic [$clog2(mnist_cim_pkg::N_INPUT_BLOCKS)-1:0] ib,

    input logic [((mnist_cim_pkg::N_OUTPUT_BLOCKS > 1) ? $clog2(
mnist_cim_pkg::N_OUTPUT_BLOCKS
) : 1)-1:0] ob_sel,


    input logic [mnist_cim_pkg::X_EFF_WIDTH-1:0] x_eff_tile[0:mnist_cim_pkg::TILE_INPUT_SIZE-1],

    output logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0]
        fc1_acc_block [0:mnist_cim_pkg::TILE_OUTPUT_SIZE-1]
);
  import mnist_cim_pkg::*;

  logic signed [WEIGHT_WIDTH-1:0] w_tile[0:TILE_OUTPUT_SIZE-1][0:TILE_INPUT_SIZE-1];
  logic signed [PSUM_WIDTH-1:0] tile_psum[0:TILE_OUTPUT_SIZE-1];
  logic signed [PSUM_WIDTH-1:0] psum[0:TILE_OUTPUT_SIZE-1];
  logic signed [BIAS_WIDTH-1:0] bias_block[0:TILE_OUTPUT_SIZE-1];
  logic signed [PSUM_WIDTH-1:0] fc1_acc_with_bias[0:TILE_OUTPUT_SIZE-1];

  fc1_weight_bank #(
      .DEFAULT_WEIGHT_HEX_FILE(DEFAULT_WEIGHT_HEX_FILE)
  ) u_fc1_weight_bank (
      .clk   (clk),
      .ob    (ob_sel),
      .ib    (ib),
      .w_tile(w_tile)
  );

  cim_tile u_cim_tile (
      .x_eff_tile(x_eff_tile),
      .w_tile    (w_tile),
      .tile_psum (tile_psum)
  );

  psum_accum u_psum_accum (
      .clk      (clk),
      .rst_n    (rst_n),
      .clear    (clear_psum),
      .en       (en_psum),
      .tile_psum(tile_psum),
      .psum     (psum)
  );

  fc1_bias_bank #(
      .DEFAULT_BIAS_HEX_FILE(DEFAULT_BIAS_HEX_FILE)
  ) u_fc1_bias_bank (
      .clk       (clk),
      .ob        (ob_sel),
      .bias_block(bias_block)
  );

  genvar g;
  generate
    for (g = 0; g < TILE_OUTPUT_SIZE; g = g + 1) begin : GEN_OUT
      assign fc1_acc_with_bias[g] = psum[g] + bias_block[g];
      assign fc1_acc_block[g]     = fc1_acc_with_bias[g];
    end
  endgenerate

endmodule
