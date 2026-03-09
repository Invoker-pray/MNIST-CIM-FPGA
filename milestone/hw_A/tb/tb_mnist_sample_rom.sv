
`timescale 1ns / 1ps

module tb_mnist_sample_rom;
  import mnist_cim_pkg::*;

  parameter int N_SAMPLES = 20;
  parameter string SAMPLE_HEX_FILE = "../data/mnist_samples_route_b_output_2.hex";

  logic [$clog2(N_SAMPLES)-1:0] sample_id;
  logic [$clog2(N_INPUT_BLOCKS)-1:0] ib;

  logic signed [INPUT_WIDTH-1:0] x_tile[0:TILE_INPUT_SIZE-1];
  logic [X_EFF_WIDTH-1:0] x_eff_tile[0:TILE_INPUT_SIZE-1];

  logic signed [INPUT_WIDTH-1:0] ref_mem[0:N_SAMPLES*INPUT_DIM-1];

  integer sid, blk, i, addr;
  integer error_count;
  integer x_eff_tmp;

  mnist_sample_rom #(
      .N_SAMPLES(N_SAMPLES),
      .DEFAULT_SAMPLE_HEX_FILE(SAMPLE_HEX_FILE)
  ) dut (
      .sample_id(sample_id),
      .ib(ib),
      .x_tile(x_tile),
      .x_eff_tile(x_eff_tile)
  );

  initial begin
    $display("TB using SAMPLE_HEX_FILE: %s", SAMPLE_HEX_FILE);
    $readmemh(SAMPLE_HEX_FILE, ref_mem);

    error_count = 0;

    for (sid = 0; sid < N_SAMPLES; sid = sid + 1) begin
      sample_id = sid[$clog2(N_SAMPLES)-1:0];
      for (blk = 0; blk < N_INPUT_BLOCKS; blk = blk + 1) begin
        ib = blk[$clog2(N_INPUT_BLOCKS)-1:0];
        #1;
        for (i = 0; i < TILE_INPUT_SIZE; i = i + 1) begin
          addr = sid * INPUT_DIM + blk * TILE_INPUT_SIZE + i;

          if (x_tile[i] !== ref_mem[addr]) begin
            $display("ERROR x_tile sid=%0d ib=%0d i=%0d got=%0d expected=%0d", sid, blk, i,
                     x_tile[i], ref_mem[addr]);
            error_count = error_count + 1;
          end

          x_eff_tmp = ref_mem[addr] - INPUT_ZERO_POINT;
          if (x_eff_tmp < 0) x_eff_tmp = 0;
          if (x_eff_tmp > ((1 << X_EFF_WIDTH) - 1)) x_eff_tmp = (1 << X_EFF_WIDTH) - 1;

          if (x_eff_tile[i] !== x_eff_tmp[X_EFF_WIDTH-1:0]) begin
            $display("ERROR x_eff sid=%0d ib=%0d i=%0d got=%0d expected=%0d", sid, blk, i,
                     x_eff_tile[i], x_eff_tmp);
            error_count = error_count + 1;
          end
        end
      end
    end

    if (error_count == 0)
      $display("PASS: mnist_sample_rom address mapping and x_eff conversion are correct.");
    else $display("FAIL: found %0d mismatches.", error_count);

    $finish;
  end

endmodule
