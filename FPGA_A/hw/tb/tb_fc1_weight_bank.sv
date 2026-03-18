
`timescale 1ns / 1ps

module tb_fc1_weight_bank;
  import mnist_cim_pkg::*;

  parameter string DEFAULT_WEIGHT_HEX_FILE = "../data_packed/weights/fc1_weight_int8.hex";

  logic clk;
  logic [$clog2(N_OUTPUT_BLOCKS)-1:0] ob;
  logic [$clog2(N_INPUT_BLOCKS)-1:0] ib;

  logic signed [WEIGHT_WIDTH-1:0] w_tile[0:TILE_OUTPUT_SIZE-1][0:TILE_INPUT_SIZE-1];

  // golden packed memory: one line = one tile
  localparam int TILE_ELEMS = TILE_OUTPUT_SIZE * TILE_INPUT_SIZE;
  localparam int TILE_BITS = TILE_ELEMS * WEIGHT_WIDTH;
  localparam int TILE_DEPTH = N_OUTPUT_BLOCKS * N_INPUT_BLOCKS;

  logic [TILE_BITS-1:0] golden_tile_mem  [0:TILE_DEPTH-1];
  logic [TILE_BITS-1:0] golden_tile_word;

  integer tr, tc;
  integer flat_idx;
  integer err_count;
  integer tile_idx;

  fc1_weight_bank #(
      .DEFAULT_WEIGHT_HEX_FILE(DEFAULT_WEIGHT_HEX_FILE)
  ) dut (
      .clk   (clk),
      .ob    (ob),
      .ib    (ib),
      .w_tile(w_tile)
  );

  initial clk = 1'b0;
  always #5 clk = ~clk;

  initial begin
    $display("============================================================");
    $display("tb_fc1_weight_bank");
    $display("DEFAULT_WEIGHT_HEX_FILE = %s", DEFAULT_WEIGHT_HEX_FILE);
    $display("============================================================");

    $readmemh(DEFAULT_WEIGHT_HEX_FILE, golden_tile_mem);

    err_count = 0;
    ob = '0;
    ib = '0;

    // ----------------------------------------------------------
    // Test 1: first tile ob=0 ib=0
    // ----------------------------------------------------------
    tile_idx = 0 * N_INPUT_BLOCKS + 0;
    golden_tile_word = golden_tile_mem[tile_idx];

    @(posedge clk);
    ob <= '0;
    ib <= '0;

    // synchronous ROM: wait one more clock for output register
    @(posedge clk);
    #1;

    $display("[TEST1] check ob=0 ib=0");

    for (tr = 0; tr < TILE_OUTPUT_SIZE; tr = tr + 1) begin
      for (tc = 0; tc < TILE_INPUT_SIZE; tc = tc + 1) begin
        flat_idx = tr * TILE_INPUT_SIZE + tc;
        if (w_tile[tr][tc] !== golden_tile_word[flat_idx*WEIGHT_WIDTH+:WEIGHT_WIDTH]) begin
          $display("ERROR TEST1 tr=%0d tc=%0d got=0x%02x exp=0x%02x", tr, tc,
                   w_tile[tr][tc] & 8'hFF,
                   golden_tile_word[flat_idx*WEIGHT_WIDTH+:WEIGHT_WIDTH] & 8'hFF);
          err_count = err_count + 1;
        end
      end
    end

    // ----------------------------------------------------------
    // Test 2: another tile, choose last tile to catch addressing
    // ----------------------------------------------------------
    tile_idx = (N_OUTPUT_BLOCKS - 1) * N_INPUT_BLOCKS + (N_INPUT_BLOCKS - 1);
    golden_tile_word = golden_tile_mem[tile_idx];

    @(posedge clk);
    ob <= N_OUTPUT_BLOCKS - 1;
    ib <= N_INPUT_BLOCKS - 1;

    @(posedge clk);
    #1;

    $display("[TEST2] check ob=%0d ib=%0d", N_OUTPUT_BLOCKS - 1, N_INPUT_BLOCKS - 1);

    for (tr = 0; tr < TILE_OUTPUT_SIZE; tr = tr + 1) begin
      for (tc = 0; tc < TILE_INPUT_SIZE; tc = tc + 1) begin
        flat_idx = tr * TILE_INPUT_SIZE + tc;
        if (w_tile[tr][tc] !== golden_tile_word[flat_idx*WEIGHT_WIDTH+:WEIGHT_WIDTH]) begin
          $display("ERROR TEST2 tr=%0d tc=%0d got=0x%02x exp=0x%02x", tr, tc,
                   w_tile[tr][tc] & 8'hFF,
                   golden_tile_word[flat_idx*WEIGHT_WIDTH+:WEIGHT_WIDTH] & 8'hFF);
          err_count = err_count + 1;
        end
      end
    end

    // ----------------------------------------------------------
    // Summary
    // ----------------------------------------------------------
    if (err_count == 0) begin
      $display("PASS: fc1_weight_bank tile addressing/unpacking is correct.");
    end else begin
      $display("FAIL: fc1_weight_bank found %0d mismatches.", err_count);
    end

    $finish;
  end

endmodule
