
`timescale 1ns / 1ps

module tb_fc1_bias_bank;
  import mnist_cim_pkg::*;

  parameter string DEFAULT_BIAS_HEX_FILE = "../data_packed/weights/fc1_bias_int32.hex";

  logic clk;
  logic [$clog2(N_OUTPUT_BLOCKS)-1:0] ob;

  logic signed [BIAS_WIDTH-1:0] bias_block[0:TILE_OUTPUT_SIZE-1];

  localparam int BLOCK_BITS = TILE_OUTPUT_SIZE * BIAS_WIDTH;
  localparam int BLOCK_DEPTH = N_OUTPUT_BLOCKS;

  logic [BLOCK_BITS-1:0] golden_bias_mem[0:BLOCK_DEPTH-1];
  logic [BLOCK_BITS-1:0] golden_bias_word;

  integer i;
  integer err_count;

  fc1_bias_bank #(
      .DEFAULT_BIAS_HEX_FILE(DEFAULT_BIAS_HEX_FILE)
  ) dut (
      .clk       (clk),
      .ob        (ob),
      .bias_block(bias_block)
  );

  initial clk = 1'b0;
  always #5 clk = ~clk;

  initial begin
    $display("============================================================");
    $display("tb_fc1_bias_bank");
    $display("DEFAULT_BIAS_HEX_FILE = %s", DEFAULT_BIAS_HEX_FILE);
    $display("============================================================");

    $readmemh(DEFAULT_BIAS_HEX_FILE, golden_bias_mem);

    err_count = 0;
    ob = '0;

    // ----------------------------------------------------------
    // TEST1: ob = 0
    // ----------------------------------------------------------
    golden_bias_word = golden_bias_mem[0];

    @(posedge clk);
    ob <= '0;

    @(posedge clk);
    #1;

    $display("[TEST1] check ob=0");
    for (i = 0; i < TILE_OUTPUT_SIZE; i = i + 1) begin
      if (bias_block[i] !== golden_bias_word[i*BIAS_WIDTH+:BIAS_WIDTH]) begin
        $display("ERROR TEST1 i=%0d got=0x%08x exp=0x%08x", i, bias_block[i],
                 golden_bias_word[i*BIAS_WIDTH+:BIAS_WIDTH]);
        err_count = err_count + 1;
      end
    end

    // ----------------------------------------------------------
    // TEST2: ob = N_OUTPUT_BLOCKS-1
    // ----------------------------------------------------------
    golden_bias_word = golden_bias_mem[N_OUTPUT_BLOCKS-1];

    @(posedge clk);
    ob <= N_OUTPUT_BLOCKS - 1;

    @(posedge clk);
    #1;

    $display("[TEST2] check ob=%0d", N_OUTPUT_BLOCKS - 1);
    for (i = 0; i < TILE_OUTPUT_SIZE; i = i + 1) begin
      if (bias_block[i] !== golden_bias_word[i*BIAS_WIDTH+:BIAS_WIDTH]) begin
        $display("ERROR TEST2 i=%0d got=0x%08x exp=0x%08x", i, bias_block[i],
                 golden_bias_word[i*BIAS_WIDTH+:BIAS_WIDTH]);
        err_count = err_count + 1;
      end
    end

    if (err_count == 0) begin
      $display("PASS: fc1_bias_bank block addressing/unpacking is correct.");
    end else begin
      $display("FAIL: fc1_bias_bank found %0d mismatches.", err_count);
    end

    $finish;
  end

endmodule
