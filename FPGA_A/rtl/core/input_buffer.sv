
module input_buffer #(
    parameter string DEFAULT_INPUT_HEX_FILE = "../data/samples/input_0.hex"
) (
    input logic [$clog2(mnist_cim_pkg::N_INPUT_BLOCKS)-1:0] ib,

    output logic signed [mnist_cim_pkg::INPUT_WIDTH-1:0] x_tile[0:mnist_cim_pkg::TILE_INPUT_SIZE-1],

    output logic [mnist_cim_pkg::X_EFF_WIDTH-1:0] x_eff_tile[0:mnist_cim_pkg::TILE_INPUT_SIZE-1]
);

  import mnist_cim_pkg::*;

  // 运行时文件路径
  //string input_file;

  // 完整输入缓存：784 x int8
  logic signed [INPUT_WIDTH-1:0] input_mem[0:INPUT_DIM-1];

  integer tc;
  integer in_idx;
  integer x_eff_tmp;


  initial begin
`ifndef SYNTHESIS
    $display("Using default INPUT_HEX_FILE: %s", DEFAULT_INPUT_HEX_FILE);
`endif
    $readmemh(DEFAULT_INPUT_HEX_FILE, input_mem);
  end


  always_comb begin
    for (tc = 0; tc < TILE_INPUT_SIZE; tc = tc + 1) begin
      in_idx = ib * TILE_INPUT_SIZE + tc;

      // 原始 int8 输入
      x_tile[tc] = input_mem[in_idx];

      // 零点修正
      x_eff_tmp = input_mem[in_idx] - INPUT_ZERO_POINT;

      // 防御性限幅
      if (x_eff_tmp < 0) x_eff_tile[tc] = '0;
      else if (x_eff_tmp > ((1 << X_EFF_WIDTH) - 1)) x_eff_tile[tc] = {X_EFF_WIDTH{1'b1}};
      else x_eff_tile[tc] = x_eff_tmp[X_EFF_WIDTH-1:0];
    end
  end

endmodule
