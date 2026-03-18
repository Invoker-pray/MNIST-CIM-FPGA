
`timescale 1ns / 1ps

module tb_mnist_sample_rom;
  import mnist_cim_pkg::*;

  parameter int N_SAMPLES = 20;
  parameter string DEFAULT_SAMPLE_HEX_FILE = "../data_packed/samples/mnist_samples_route_b_output_2.hex";

  logic                                     clk;
  logic        [     $clog2(N_SAMPLES)-1:0] sample_id;
  logic        [$clog2(N_INPUT_BLOCKS)-1:0] ib;

  logic signed [           INPUT_WIDTH-1:0] x_tile    [0:TILE_INPUT_SIZE-1];
  logic        [           X_EFF_WIDTH-1:0] x_eff_tile[0:TILE_INPUT_SIZE-1];

  localparam int TILE_BITS = TILE_INPUT_SIZE * INPUT_WIDTH;
  localparam int TILE_DEPTH = N_SAMPLES * N_INPUT_BLOCKS;

  logic [TILE_BITS-1:0] golden_sample_mem[0:TILE_DEPTH-1];
  logic [TILE_BITS-1:0] golden_tile_word;

  integer i;
  integer err_count;
  integer tile_idx;
  integer x_eff_exp;

  mnist_sample_rom #(
      .N_SAMPLES(N_SAMPLES),
      .DEFAULT_SAMPLE_HEX_FILE(DEFAULT_SAMPLE_HEX_FILE)
  ) dut (
      .clk       (clk),
      .sample_id (sample_id),
      .ib        (ib),
      .x_tile    (x_tile),
      .x_eff_tile(x_eff_tile)
  );

  initial clk = 1'b0;
  always #5 clk = ~clk;

  initial begin
    $display("============================================================");
    $display("tb_mnist_sample_rom");
    $display("DEFAULT_SAMPLE_HEX_FILE = %s", DEFAULT_SAMPLE_HEX_FILE);
    $display("============================================================");

    $readmemh(DEFAULT_SAMPLE_HEX_FILE, golden_sample_mem);

    err_count        = 0;
    sample_id        = '0;
    ib               = '0;

    // ----------------------------------------------------------
    // TEST1: sample_id=0, ib=0
    // ----------------------------------------------------------
    tile_idx         = 0 * N_INPUT_BLOCKS + 0;
    golden_tile_word = golden_sample_mem[tile_idx];

    @(posedge clk);
    sample_id <= '0;
    ib        <= '0;

    @(posedge clk);
    #1;

    $display("[TEST1] check sample_id=0 ib=0");
    for (i = 0; i < TILE_INPUT_SIZE; i = i + 1) begin
      if (x_tile[i] !== golden_tile_word[i*INPUT_WIDTH+:INPUT_WIDTH]) begin
        $display("ERROR TEST1 x_tile i=%0d got=0x%02x exp=0x%02x", i, x_tile[i] & 8'hFF,
                 golden_tile_word[i*INPUT_WIDTH+:INPUT_WIDTH] & 8'hFF);
        err_count = err_count + 1;
      end

      x_eff_exp = $signed(golden_tile_word[i*INPUT_WIDTH+:INPUT_WIDTH]) - INPUT_ZERO_POINT;
      if (x_eff_exp < 0) x_eff_exp = 0;
      else if (x_eff_exp > ((1 << X_EFF_WIDTH) - 1)) x_eff_exp = (1 << X_EFF_WIDTH) - 1;

      if (x_eff_tile[i] !== x_eff_exp[X_EFF_WIDTH-1:0]) begin
        $display("ERROR TEST1 x_eff i=%0d got=0x%02x exp=0x%02x", i, x_eff_tile[i],
                 x_eff_exp[X_EFF_WIDTH-1:0]);
        err_count = err_count + 1;
      end
    end

    // ----------------------------------------------------------
    // TEST2: sample_id=0, ib=1
    // ----------------------------------------------------------
    tile_idx = 0 * N_INPUT_BLOCKS + 1;
    golden_tile_word = golden_sample_mem[tile_idx];

    @(posedge clk);
    sample_id <= '0;
    ib        <= 1;

    @(posedge clk);
    #1;

    $display("[TEST2] check sample_id=0 ib=1");
    for (i = 0; i < TILE_INPUT_SIZE; i = i + 1) begin
      if (x_tile[i] !== golden_tile_word[i*INPUT_WIDTH+:INPUT_WIDTH]) begin
        $display("ERROR TEST2 x_tile i=%0d got=0x%02x exp=0x%02x", i, x_tile[i] & 8'hFF,
                 golden_tile_word[i*INPUT_WIDTH+:INPUT_WIDTH] & 8'hFF);
        err_count = err_count + 1;
      end

      x_eff_exp = $signed(golden_tile_word[i*INPUT_WIDTH+:INPUT_WIDTH]) - INPUT_ZERO_POINT;
      if (x_eff_exp < 0) x_eff_exp = 0;
      else if (x_eff_exp > ((1 << X_EFF_WIDTH) - 1)) x_eff_exp = (1 << X_EFF_WIDTH) - 1;

      if (x_eff_tile[i] !== x_eff_exp[X_EFF_WIDTH-1:0]) begin
        $display("ERROR TEST2 x_eff i=%0d got=0x%02x exp=0x%02x", i, x_eff_tile[i],
                 x_eff_exp[X_EFF_WIDTH-1:0]);
        err_count = err_count + 1;
      end
    end

    if (err_count == 0) begin
      $display("PASS: mnist_sample_rom tile addressing/unpacking/x_eff is correct.");
    end else begin
      $display("FAIL: mnist_sample_rom found %0d mismatches.", err_count);
    end

    $finish;
  end

endmodule
