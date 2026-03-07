

module fc1_multi_block_parallel #(
    parameter int PAR_OB = 8
) (
    input logic clk,
    input logic rst_n,
    input logic start,

    output logic busy,
    output logic done,

    output logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0]
        fc1_acc_all [0:PAR_OB*mnist_cim_pkg::TILE_OUTPUT_SIZE-1]
);
  import mnist_cim_pkg::*;

  logic [PAR_OB-1:0] busy_vec;
  logic [PAR_OB-1:0] done_vec;

  logic signed [PSUM_WIDTH-1:0] fc1_acc_block[0:PAR_OB-1][0:TILE_OUTPUT_SIZE-1];

  genvar g_ob, g_idx;
  generate
    for (g_ob = 0; g_ob < PAR_OB; g_ob = g_ob + 1) begin : GEN_OB_CORE
      fc1_cim_core_block u_fc1_cim_core_block (
          .clk(clk),
          .rst_n(rst_n),
          .start(start),
          .ob_sel(g_ob[$clog2(N_OUTPUT_BLOCKS)-1:0]),
          .busy(busy_vec[g_ob]),
          .done(done_vec[g_ob]),
          .fc1_acc_block(fc1_acc_block[g_ob])
      );

      for (g_idx = 0; g_idx < TILE_OUTPUT_SIZE; g_idx = g_idx + 1) begin : GEN_PACK
        assign fc1_acc_all[g_ob*TILE_OUTPUT_SIZE+g_idx] = fc1_acc_block[g_ob][g_idx];
      end
    end
  endgenerate

  assign busy = |busy_vec;
  assign done = &done_vec;

endmodule
