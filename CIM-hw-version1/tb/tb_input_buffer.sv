
`timescale 1ns / 1ps

module tb_input_buffer;

  import mnist_cim_pkg::*;

  logic        [$clog2(N_INPUT_BLOCKS)-1:0] ib;

  string                                    input_file;

  logic signed [           INPUT_WIDTH-1:0] x_tile     [0:TILE_INPUT_SIZE-1];
  logic        [           X_EFF_WIDTH-1:0] x_eff_tile [0:TILE_INPUT_SIZE-1];

  input_buffer dut (
      .ib(ib),
      .x_tile(x_tile),
      .x_eff_tile(x_eff_tile)
  );

  logic signed [INPUT_WIDTH-1:0] ref_input_mem[0:INPUT_DIM-1];

  integer tc;
  integer expected_idx;
  integer expected_eff;
  integer error_count;

  initial begin

    // 默认路径
    input_file = "../../CIM-sw-version1/sw/train&quantize/route_b_output/input_0.hex";

    // plusarg 覆盖
    if ($value$plusargs("INPUT_HEX_FILE=%s", input_file)) begin
      $display("TB using INPUT_HEX_FILE: %s", input_file);
    end else begin
      $display("TB using default INPUT_HEX_FILE: %s", input_file);
    end

    $readmemh(input_file, ref_input_mem);

    error_count = 0;

    // -------------------------
    // Test 1: block 0
    // -------------------------
    ib = 0;
    #1;

    $display("Checking input block 0...");
    for (tc = 0; tc < TILE_INPUT_SIZE; tc++) begin
      expected_idx = ib * TILE_INPUT_SIZE + tc;
      expected_eff = ref_input_mem[expected_idx] - INPUT_ZERO_POINT;

      if (x_tile[tc] !== ref_input_mem[expected_idx]) begin
        $display("ERROR block0 x_tile tc=%0d got=%0d expected=%0d", tc, x_tile[tc],
                 ref_input_mem[expected_idx]);
        error_count++;
      end

      if (x_eff_tile[tc] !== expected_eff[X_EFF_WIDTH-1:0]) begin
        $display("ERROR block0 x_eff tc=%0d got=%0d expected=%0d", tc, x_eff_tile[tc],
                 expected_eff);
        error_count++;
      end
    end

    // -------------------------
    // Test 2: block 1
    // -------------------------
    ib = 1;
    #1;

    $display("Checking input block 1...");
    for (tc = 0; tc < TILE_INPUT_SIZE; tc++) begin
      expected_idx = ib * TILE_INPUT_SIZE + tc;
      expected_eff = ref_input_mem[expected_idx] - INPUT_ZERO_POINT;

      if (x_tile[tc] !== ref_input_mem[expected_idx]) begin
        $display("ERROR block1 x_tile tc=%0d got=%0d expected=%0d", tc, x_tile[tc],
                 ref_input_mem[expected_idx]);
        error_count++;
      end

      if (x_eff_tile[tc] !== expected_eff[X_EFF_WIDTH-1:0]) begin
        $display("ERROR block1 x_eff tc=%0d got=%0d expected=%0d", tc, x_eff_tile[tc],
                 expected_eff);
        error_count++;
      end
    end

    // -------------------------
    // Test 3: last block
    // -------------------------
    ib = N_INPUT_BLOCKS - 1;
    #1;

    $display("Checking last input block...");
    for (tc = 0; tc < TILE_INPUT_SIZE; tc++) begin
      expected_idx = ib * TILE_INPUT_SIZE + tc;
      expected_eff = ref_input_mem[expected_idx] - INPUT_ZERO_POINT;

      if (x_tile[tc] !== ref_input_mem[expected_idx]) begin
        $display("ERROR last block x_tile tc=%0d got=%0d expected=%0d", tc, x_tile[tc],
                 ref_input_mem[expected_idx]);
        error_count++;
      end

      if (x_eff_tile[tc] !== expected_eff[X_EFF_WIDTH-1:0]) begin
        $display("ERROR last block x_eff tc=%0d got=%0d expected=%0d", tc, x_eff_tile[tc],
                 expected_eff);
        error_count++;
      end
    end

    if (error_count == 0) $display("PASS: input_buffer block extraction is correct.");
    else $display("FAIL: found %0d mismatches.", error_count);

    $finish;
  end

endmodule
