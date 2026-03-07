`timescale 1ns / 1ps

module tb_fc1_weight_bank;
  import mnist_cim_pkg::*;

  logic [$clog2(N_OUTPUT_BLOCKS)-1:0] ob;
  logic [$clog2(N_INPUT_BLOCKS)-1:0] ib;

  parameter string WEIGHT_HEX_FILE="../../CIM-sw-version1/sw/train&quantize/route_b_output/fc1_weight_int8.hex";
  logic signed [WEIGHT_WIDTH-1:0] w_tile[0:TILE_OUTPUT_SIZE-1][0:TILE_INPUT_SIZE-1];

  fc1_weight_bank #(
      .WEIGHT_HEX_FILE("../../CIM-sw-version1/sw/train&quantize/route_b_output/fc1_weight_int8.hex")
  ) dut (
      .ob(ob),
      .ib(ib),
      .w_tile(w_tile)
  );

  logic signed [WEIGHT_WIDTH-1:0] ref_weight_mem[0:FC1_WEIGHT_DEPTH-1];

  integer tr, tc;
  integer expected_addr;
  integer expected_out, expected_in;
  integer error_count;

  initial begin
    $readmemh(WEIGHT_HEX_FILE, ref_weight_mem);

    error_count = 0;

    // -------------------------
    // Test 1: tile (0,0)
    // -------------------------
    ob = 0;
    ib = 0;
    #1;

    $display("Checking tile (0,0)...");
    for (tr = 0; tr < TILE_OUTPUT_SIZE; tr = tr + 1) begin
      for (tc = 0; tc < TILE_INPUT_SIZE; tc = tc + 1) begin
        expected_out  = ob * TILE_OUTPUT_SIZE + tr;
        expected_in   = ib * TILE_INPUT_SIZE + tc;
        expected_addr = expected_out * FC1_IN_DIM + expected_in;

        if (w_tile[tr][tc] !== ref_weight_mem[expected_addr]) begin
          $display("ERROR tile(0,0): tr=%0d tc=%0d got=%0d expected=%0d addr=%0d", tr, tc,
                   w_tile[tr][tc], ref_weight_mem[expected_addr], expected_addr);
          error_count = error_count + 1;
        end
      end
    end

    // -------------------------
    // Test 2: tile (1,2)
    // -------------------------
    ob = 1;
    ib = 2;
    #1;

    $display("Checking tile (1,2)...");
    for (tr = 0; tr < TILE_OUTPUT_SIZE; tr = tr + 1) begin
      for (tc = 0; tc < TILE_INPUT_SIZE; tc = tc + 1) begin
        expected_out  = ob * TILE_OUTPUT_SIZE + tr;
        expected_in   = ib * TILE_INPUT_SIZE + tc;
        expected_addr = expected_out * FC1_IN_DIM + expected_in;

        if (w_tile[tr][tc] !== ref_weight_mem[expected_addr]) begin
          $display("ERROR tile(1,2): tr=%0d tc=%0d got=%0d expected=%0d addr=%0d", tr, tc,
                   w_tile[tr][tc], ref_weight_mem[expected_addr], expected_addr);
          error_count = error_count + 1;
        end
      end
    end

    // -------------------------
    // Test 3: last tile (7,48)
    // -------------------------
    ob = N_OUTPUT_BLOCKS - 1;
    ib = N_INPUT_BLOCKS - 1;
    #1;

    $display("Checking tile (last,last)...");
    for (tr = 0; tr < TILE_OUTPUT_SIZE; tr = tr + 1) begin
      for (tc = 0; tc < TILE_INPUT_SIZE; tc = tc + 1) begin
        expected_out  = ob * TILE_OUTPUT_SIZE + tr;
        expected_in   = ib * TILE_INPUT_SIZE + tc;
        expected_addr = expected_out * FC1_IN_DIM + expected_in;

        if (w_tile[tr][tc] !== ref_weight_mem[expected_addr]) begin
          $display("ERROR tile(last,last): tr=%0d tc=%0d got=%0d expected=%0d addr=%0d", tr, tc,
                   w_tile[tr][tc], ref_weight_mem[expected_addr], expected_addr);
          error_count = error_count + 1;
        end
      end
    end

    if (error_count == 0) begin
      $display("PASS: fc1_weight_bank tile extraction is correct.");
    end else begin
      $display("FAIL: found %0d mismatches.", error_count);
    end

    $finish;
  end
endmodule
