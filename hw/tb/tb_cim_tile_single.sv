

`timescale 1ns / 1ps

module tb_cim_tile;
  import mnist_cim_pkg::*;

  logic [X_EFF_WIDTH-1:0] x_eff_tile[0:TILE_INPUT_SIZE-1];
  logic signed [WEIGHT_WIDTH-1:0] w_tile[0:TILE_OUTPUT_SIZE-1][0:TILE_INPUT_SIZE-1];
  logic signed [PSUM_WIDTH-1:0] tile_psum[0:TILE_OUTPUT_SIZE-1];

  integer i, j;
  integer error_count;
  integer expected_sum;

  cim_tile dut (
      .x_eff_tile(x_eff_tile),
      .w_tile(w_tile),
      .tile_psum(tile_psum)
  );

  initial begin
    error_count = 0;

    // -------------------------
    // Case 1:
    // x = 1..16
    // row0 weight all 1  => 136
    // row1 weight all 2  => 272
    // -------------------------
    for (i = 0; i < TILE_INPUT_SIZE; i = i + 1) x_eff_tile[i] = i + 1;

    for (j = 0; j < TILE_OUTPUT_SIZE; j = j + 1)
    for (i = 0; i < TILE_INPUT_SIZE; i = i + 1) w_tile[j][i] = 0;

    for (i = 0; i < TILE_INPUT_SIZE; i = i + 1) w_tile[0][i] = 1;

    for (i = 0; i < TILE_INPUT_SIZE; i = i + 1) w_tile[1][i] = 2;

    #1;

    if (tile_psum[0] !== 136) begin
      $display("ERROR Case1 row0: got=%0d expected=136", tile_psum[0]);
      error_count = error_count + 1;
    end

    if (tile_psum[1] !== 272) begin
      $display("ERROR Case1 row1: got=%0d expected=272", tile_psum[1]);
      error_count = error_count + 1;
    end

    // -------------------------
    // Case 2:
    // row2 = alternating 1,0,1,0...
    // expected = 1+3+5+...+15 = 64
    // -------------------------
    for (i = 0; i < TILE_INPUT_SIZE; i = i + 1) w_tile[2][i] = (i % 2 == 0) ? 1 : 0;

    #1;

    if (tile_psum[2] !== 64) begin
      $display("ERROR Case2 row2: got=%0d expected=64", tile_psum[2]);
      error_count = error_count + 1;
    end

    // -------------------------
    // Case 3:
    // row3 all -1 => -(1+...+16) = -136
    // -------------------------
    for (i = 0; i < TILE_INPUT_SIZE; i = i + 1) w_tile[3][i] = -1;

    #1;

    if (tile_psum[3] !== -136) begin
      $display("ERROR Case3 row3: got=%0d expected=-136", tile_psum[3]);
      error_count = error_count + 1;
    end

    if (error_count == 0) $display("PASS: cim_tile basic MAC is correct.");
    else $display("FAIL: found %0d mismatches.", error_count);

    $finish;
  end

endmodule
