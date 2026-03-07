
`timescale 1ns / 1ps

module tb_psum_accum;
  import mnist_cim_pkg::*;

  logic                         clk;
  logic                         rst_n;
  logic                         clear;
  logic                         en;

  logic signed [PSUM_WIDTH-1:0] tile_psum   [0:TILE_OUTPUT_SIZE-1];
  logic signed [PSUM_WIDTH-1:0] psum        [0:TILE_OUTPUT_SIZE-1];

  integer                       i;
  integer                       error_count;

  psum_accum dut (
      .clk(clk),
      .rst_n(rst_n),
      .clear(clear),
      .en(en),
      .tile_psum(tile_psum),
      .psum(psum)
  );

  // 10ns clock
  initial clk = 0;
  always #5 clk = ~clk;

  task check_all_zero;
    begin
      for (i = 0; i < TILE_OUTPUT_SIZE; i = i + 1) begin
        if (psum[i] !== 0) begin
          $display("ERROR zero check: psum[%0d]=%0d expected=0", i, psum[i]);
          error_count = error_count + 1;
        end
      end
    end
  endtask

  initial begin
    error_count = 0;

    // init
    rst_n       = 0;
    clear       = 0;
    en          = 0;

    for (i = 0; i < TILE_OUTPUT_SIZE; i = i + 1) begin
      tile_psum[i] = 0;
    end

    // -------------------------
    // Reset
    // -------------------------
    #12;
    rst_n = 1;
    #2;

    @(posedge clk);
    #1;
    $display("Checking reset result...");
    check_all_zero();

    // -------------------------
    // Case 1: accumulate once
    // tile_psum[i] = i
    // -------------------------
    for (i = 0; i < TILE_OUTPUT_SIZE; i = i + 1) begin
      tile_psum[i] = i;
    end

    en = 1;
    @(posedge clk);
    #1;
    en = 0;

    $display("Checking first accumulation...");
    for (i = 0; i < TILE_OUTPUT_SIZE; i = i + 1) begin
      if (psum[i] !== i) begin
        $display("ERROR Case1: psum[%0d]=%0d expected=%0d", i, psum[i], i);
        error_count = error_count + 1;
      end
    end

    // -------------------------
    // Case 2: accumulate again
    // tile_psum[i] = 2*i
    // expected = i + 2*i = 3*i
    // -------------------------
    for (i = 0; i < TILE_OUTPUT_SIZE; i = i + 1) begin
      tile_psum[i] = 2 * i;
    end

    en = 1;
    @(posedge clk);
    #1;
    en = 0;

    $display("Checking second accumulation...");
    for (i = 0; i < TILE_OUTPUT_SIZE; i = i + 1) begin
      if (psum[i] !== 3 * i) begin
        $display("ERROR Case2: psum[%0d]=%0d expected=%0d", i, psum[i], 3 * i);
        error_count = error_count + 1;
      end
    end

    // -------------------------
    // Case 3: negative accumulation
    // tile_psum[i] = -i
    // expected = 3*i - i = 2*i
    // -------------------------
    for (i = 0; i < TILE_OUTPUT_SIZE; i = i + 1) begin
      tile_psum[i] = -i;
    end

    en = 1;
    @(posedge clk);
    #1;
    en = 0;

    $display("Checking third accumulation (negative values)...");
    for (i = 0; i < TILE_OUTPUT_SIZE; i = i + 1) begin
      if (psum[i] !== 2 * i) begin
        $display("ERROR Case3: psum[%0d]=%0d expected=%0d", i, psum[i], 2 * i);
        error_count = error_count + 1;
      end
    end

    // -------------------------
    // Case 4: clear
    // -------------------------
    clear = 1;
    @(posedge clk);
    #1;
    clear = 0;

    $display("Checking clear...");
    check_all_zero();

    if (error_count == 0) $display("PASS: psum_accum is correct.");
    else $display("FAIL: found %0d mismatches.", error_count);

    $finish;
  end

endmodule
