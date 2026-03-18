
`timescale 1ns / 1ps

module tb_fc1_ob_engine_shared_input;
  import mnist_cim_pkg::*;

  parameter string DEFAULT_WEIGHT_HEX_FILE = "../data_packed/weights/fc1_weight_int8.hex";
  parameter string DEFAULT_BIAS_HEX_FILE = "../data_packed/weights/fc1_bias_int32.hex";

  logic clk;
  logic rst_n;

  logic clear_psum;
  logic en_psum;

  logic [$clog2(N_INPUT_BLOCKS)-1:0] ib;
  logic [$clog2(N_OUTPUT_BLOCKS)-1:0] ob_sel;

  logic [X_EFF_WIDTH-1:0] x_eff_tile[0:TILE_INPUT_SIZE-1];

  logic signed [PSUM_WIDTH-1:0] fc1_acc_block[0:TILE_OUTPUT_SIZE-1];

  // ------------------------------------------------------------
  // golden memories
  // ------------------------------------------------------------
  localparam int TILE_ELEMS = TILE_OUTPUT_SIZE * TILE_INPUT_SIZE;
  localparam int TILE_BITS = TILE_ELEMS * WEIGHT_WIDTH;
  localparam int TILE_DEPTH = N_OUTPUT_BLOCKS * N_INPUT_BLOCKS;
  localparam int BLOCK_BITS = TILE_OUTPUT_SIZE * BIAS_WIDTH;

  logic [TILE_BITS-1:0] golden_weight_mem[0:TILE_DEPTH-1];
  logic [BLOCK_BITS-1:0] golden_bias_mem[0:N_OUTPUT_BLOCKS-1];

  integer r, c;
  integer flat_idx;
  integer err_count;
  integer tile_idx;

  logic signed [WEIGHT_WIDTH-1:0] golden_w;
  logic signed [BIAS_WIDTH-1:0] golden_b;
  logic signed [PSUM_WIDTH-1:0] expected_tile_psum[0:TILE_OUTPUT_SIZE-1];
  logic signed [PSUM_WIDTH-1:0] expected_once[0:TILE_OUTPUT_SIZE-1];
  logic signed [PSUM_WIDTH-1:0] expected_twice[0:TILE_OUTPUT_SIZE-1];
  logic signed [PSUM_WIDTH-1:0] expected_clear[0:TILE_OUTPUT_SIZE-1];

  fc1_ob_engine_shared_input #(
      .DEFAULT_WEIGHT_HEX_FILE(DEFAULT_WEIGHT_HEX_FILE),
      .DEFAULT_BIAS_HEX_FILE  (DEFAULT_BIAS_HEX_FILE)
  ) dut (
      .clk          (clk),
      .rst_n        (rst_n),
      .clear_psum   (clear_psum),
      .en_psum      (en_psum),
      .ib           (ib),
      .ob_sel       (ob_sel),
      .x_eff_tile   (x_eff_tile),
      .fc1_acc_block(fc1_acc_block)
  );

  initial clk = 1'b0;
  always #5 clk = ~clk;

  initial begin
    $display("============================================================");
    $display("tb_fc1_ob_engine_shared_input");
    $display("DEFAULT_WEIGHT_HEX_FILE = %s", DEFAULT_WEIGHT_HEX_FILE);
    $display("DEFAULT_BIAS_HEX_FILE   = %s", DEFAULT_BIAS_HEX_FILE);
    $display("============================================================");

    $readmemh(DEFAULT_WEIGHT_HEX_FILE, golden_weight_mem);
    $readmemh(DEFAULT_BIAS_HEX_FILE, golden_bias_mem);

    err_count  = 0;
    rst_n      = 1'b0;
    clear_psum = 1'b0;
    en_psum    = 1'b0;
    ib         = '0;
    ob_sel     = '0;

    // simple deterministic x_eff_tile
    for (c = 0; c < TILE_INPUT_SIZE; c = c + 1) begin
      x_eff_tile[c] = c + 1;
    end

    // choose ob=0 ib=0 first
    tile_idx = 0 * N_INPUT_BLOCKS + 0;

    // golden compute
    for (r = 0; r < TILE_OUTPUT_SIZE; r = r + 1) begin
      expected_tile_psum[r] = '0;
      for (c = 0; c < TILE_INPUT_SIZE; c = c + 1) begin
        flat_idx = r * TILE_INPUT_SIZE + c;
        golden_w = golden_weight_mem[tile_idx][flat_idx*WEIGHT_WIDTH+:WEIGHT_WIDTH];
        expected_tile_psum[r] = expected_tile_psum[r] +
            $signed(golden_w) * $signed({1'b0, x_eff_tile[c]});
      end

      golden_b = golden_bias_mem[0][r*BIAS_WIDTH+:BIAS_WIDTH];
      expected_once[r] = expected_tile_psum[r] + golden_b;
      expected_twice[r] = (expected_tile_psum[r] <<< 1) + golden_b;
      expected_clear[r] = golden_b;
    end

    // release reset
    @(posedge clk);
    rst_n <= 1'b1;

    // set ob/ib, then wait one full cycle for sync ROMs
    @(posedge clk);
    ob_sel <= '0;
    ib     <= '0;

    @(posedge clk);  // ROM read latency

    // ----------------------------------------------------------
    // TEST1: clear then accumulate once
    // ----------------------------------------------------------
    @(posedge clk);
    clear_psum <= 1'b1;
    en_psum    <= 1'b0;

    @(posedge clk);
    clear_psum <= 1'b0;
    en_psum    <= 1'b1;

    @(posedge clk);
    en_psum <= 1'b0;
    #1;

    $display("[TEST1] check one accumulation");
    for (r = 0; r < TILE_OUTPUT_SIZE; r = r + 1) begin
      if (fc1_acc_block[r] !== expected_once[r]) begin
        $display("ERROR TEST1 r=%0d got=%0d exp=%0d", r, fc1_acc_block[r], expected_once[r]);
        err_count = err_count + 1;
      end
    end

    // ----------------------------------------------------------
    // TEST2: accumulate second time without clear
    // ----------------------------------------------------------
    @(posedge clk);
    en_psum <= 1'b1;

    @(posedge clk);
    en_psum <= 1'b0;
    #1;

    $display("[TEST2] check two accumulations");
    for (r = 0; r < TILE_OUTPUT_SIZE; r = r + 1) begin
      if (fc1_acc_block[r] !== expected_twice[r]) begin
        $display("ERROR TEST2 r=%0d got=%0d exp=%0d", r, fc1_acc_block[r], expected_twice[r]);
        err_count = err_count + 1;
      end
    end

    // ----------------------------------------------------------
    // TEST3: clear
    // ----------------------------------------------------------
    @(posedge clk);
    clear_psum <= 1'b1;
    en_psum    <= 1'b0;

    @(posedge clk);
    clear_psum <= 1'b0;
    #1;

    $display("[TEST3] check clear");
    for (r = 0; r < TILE_OUTPUT_SIZE; r = r + 1) begin
      if (fc1_acc_block[r] !== expected_clear[r]) begin
        $display("ERROR TEST3 r=%0d got=%0d exp=%0d", r, fc1_acc_block[r], expected_clear[r]);
        err_count = err_count + 1;
      end
    end

    if (err_count == 0) begin
      $display("PASS: fc1_ob_engine_shared_input weight/bias/psum behavior is correct.");
    end else begin
      $display("FAIL: fc1_ob_engine_shared_input found %0d mismatches.", err_count);
    end

    $finish;
  end

endmodule
