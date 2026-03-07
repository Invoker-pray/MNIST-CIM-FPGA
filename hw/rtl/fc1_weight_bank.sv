
module fc1_weight_bank #(
    parameter string DEFAULT_WEIGHT_HEX_FILE = 
        "../../CIM-sw-version1/sw/train_quantize/route_b_output/fc1_weight_int8.hex"
) (
    input logic [$clog2(mnist_cim_pkg::N_OUTPUT_BLOCKS)-1:0] ob,
    input logic [ $clog2(mnist_cim_pkg::N_INPUT_BLOCKS)-1:0] ib,

    output logic signed [mnist_cim_pkg::WEIGHT_WIDTH-1:0]
        w_tile [0:mnist_cim_pkg::TILE_OUTPUT_SIZE-1]
               [0:mnist_cim_pkg::TILE_INPUT_SIZE-1]
);

  import mnist_cim_pkg::*;

  string weight_file;

  logic signed [WEIGHT_WIDTH-1:0] weight_mem[0:FC1_WEIGHT_DEPTH-1];

  initial begin
    // 默认路径
    weight_file = DEFAULT_WEIGHT_HEX_FILE;

    // 从命令行读取
    if ($value$plusargs("WEIGHT_HEX_FILE=%s", weight_file)) begin
      $display("Using WEIGHT_HEX_FILE from plusarg: %s", weight_file);
    end else begin
      $display("Using default WEIGHT_HEX_FILE: %s", weight_file);
    end

    $readmemh(weight_file, weight_mem);
  end

  integer tr, tc;
  integer out_idx, in_idx;
  integer addr;

  always_comb begin
    for (tr = 0; tr < TILE_OUTPUT_SIZE; tr++) begin
      for (tc = 0; tc < TILE_INPUT_SIZE; tc++) begin
        out_idx = ob * TILE_OUTPUT_SIZE + tr;
        in_idx  = ib * TILE_INPUT_SIZE + tc;
        addr    = out_idx * FC1_IN_DIM + in_idx;

        w_tile[tr][tc] = weight_mem[addr];
      end
    end
  end

endmodule

