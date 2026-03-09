
module fc1_cim_core_block_dual (
    input logic clk,
    input logic rst_n,
    input logic start,

    output logic busy,
    output logic done,

    output logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0]
        fc1_acc_block0 [0:mnist_cim_pkg::TILE_OUTPUT_SIZE-1],

    output logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0]
        fc1_acc_block1 [0:mnist_cim_pkg::TILE_OUTPUT_SIZE-1]
);
  import mnist_cim_pkg::*;

  logic busy0, done0;
  logic busy1, done1;

  logic [$clog2(N_OUTPUT_BLOCKS)-1:0] ob_sel0, ob_sel1;

  assign ob_sel0 = 0;
  assign ob_sel1 = 1;

  fc1_cim_core_block u_block0 (
      .clk(clk),
      .rst_n(rst_n),
      .start(start),
      .ob_sel(ob_sel0),
      .busy(busy0),
      .done(done0),
      .fc1_acc_block(fc1_acc_block0)
  );

  fc1_cim_core_block u_block1 (
      .clk(clk),
      .rst_n(rst_n),
      .start(start),
      .ob_sel(ob_sel1),
      .busy(busy1),
      .done(done1),
      .fc1_acc_block(fc1_acc_block1)
  );

  assign busy = busy0 | busy1;
  assign done = done0 & done1;

endmodule
