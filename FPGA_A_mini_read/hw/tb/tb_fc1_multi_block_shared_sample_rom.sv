
`timescale 1ns / 1ps

module tb_fc1_multi_block_shared_sample_rom;
  import mnist_cim_pkg::*;

  parameter int PAR_OB = 1;
  parameter int N_SAMPLES = 20;
  parameter string DEFAULT_SAMPLE_HEX_FILE = "../data_packed/samples/mnist_samples_route_b_output_2.hex";
  parameter string DEFAULT_WEIGHT_HEX_FILE = "../data_packed/weights/fc1_weight_int8.hex";
  parameter string DEFAULT_BIAS_HEX_FILE = "../data_packed/weights/fc1_bias_int32.hex";

  logic clk;
  logic rst_n;
  logic start;
  logic [$clog2(N_SAMPLES)-1:0] sample_id;
  logic [((N_OUTPUT_BLOCKS > 1) ? $clog2(N_OUTPUT_BLOCKS) : 1)-1:0] base_ob;

  logic busy, done;
  logic signed [PSUM_WIDTH-1:0] fc1_acc_all[0:PAR_OB*TILE_OUTPUT_SIZE-1];

  localparam int SAMPLE_TILE_BITS = TILE_INPUT_SIZE * INPUT_WIDTH;
  localparam int SAMPLE_TILE_DEPTH = N_SAMPLES * N_INPUT_BLOCKS;
  localparam int WEIGHT_TILE_BITS = TILE_OUTPUT_SIZE * TILE_INPUT_SIZE * WEIGHT_WIDTH;
  localparam int WEIGHT_TILE_DEPTH = N_OUTPUT_BLOCKS * N_INPUT_BLOCKS;
  localparam int BIAS_BLOCK_BITS = TILE_OUTPUT_SIZE * BIAS_WIDTH;

  logic [SAMPLE_TILE_BITS-1:0] golden_sample_mem[0:SAMPLE_TILE_DEPTH-1];
  logic [WEIGHT_TILE_BITS-1:0] golden_weight_mem[0:WEIGHT_TILE_DEPTH-1];
  logic [ BIAS_BLOCK_BITS-1:0] golden_bias_mem  [  0:N_OUTPUT_BLOCKS-1];

  integer ib_idx, r, c;
  integer err_count;
  integer sample_tile_idx;
  integer weight_tile_idx;
  integer x_eff_tmp;
  logic signed [INPUT_WIDTH-1:0] sample_val;
  logic signed [WEIGHT_WIDTH-1:0] weight_val;
  logic signed [BIAS_WIDTH-1:0] bias_val;
  logic signed [PSUM_WIDTH-1:0] golden_acc[0:TILE_OUTPUT_SIZE-1];

  fc1_multi_block_shared_sample_rom #(
      .PAR_OB(PAR_OB),
      .N_SAMPLES(N_SAMPLES),
      .DEFAULT_SAMPLE_HEX_FILE(DEFAULT_SAMPLE_HEX_FILE),
      .DEFAULT_WEIGHT_HEX_FILE(DEFAULT_WEIGHT_HEX_FILE),
      .DEFAULT_BIAS_HEX_FILE(DEFAULT_BIAS_HEX_FILE)
  ) dut (
      .clk        (clk),
      .rst_n      (rst_n),
      .start      (start),
      .sample_id  (sample_id),
      .base_ob    (base_ob),
      .busy       (busy),
      .done       (done),
      .fc1_acc_all(fc1_acc_all)
  );

  initial clk = 1'b0;
  always #5 clk = ~clk;

  initial begin
    $display("============================================================");
    $display("tb_fc1_multi_block_shared_sample_rom");
    $display("DEFAULT_SAMPLE_HEX_FILE = %s", DEFAULT_SAMPLE_HEX_FILE);
    $display("DEFAULT_WEIGHT_HEX_FILE = %s", DEFAULT_WEIGHT_HEX_FILE);
    $display("DEFAULT_BIAS_HEX_FILE   = %s", DEFAULT_BIAS_HEX_FILE);
    $display("============================================================");

    $readmemh(DEFAULT_SAMPLE_HEX_FILE, golden_sample_mem);
    $readmemh(DEFAULT_WEIGHT_HEX_FILE, golden_weight_mem);
    $readmemh(DEFAULT_BIAS_HEX_FILE, golden_bias_mem);

    err_count = 0;
    rst_n     = 1'b0;
    start     = 1'b0;
    sample_id = '0;
    base_ob   = '0;

    for (r = 0; r < TILE_OUTPUT_SIZE; r = r + 1) begin
      bias_val = golden_bias_mem[0][r*BIAS_WIDTH+:BIAS_WIDTH];
      golden_acc[r] = bias_val;
    end

    // golden accumulate over all input blocks
    for (ib_idx = 0; ib_idx < N_INPUT_BLOCKS; ib_idx = ib_idx + 1) begin
      sample_tile_idx = 0 * N_INPUT_BLOCKS + ib_idx;
      weight_tile_idx = 0 * N_INPUT_BLOCKS + ib_idx;

      for (r = 0; r < TILE_OUTPUT_SIZE; r = r + 1) begin
        for (c = 0; c < TILE_INPUT_SIZE; c = c + 1) begin
          sample_val = golden_sample_mem[sample_tile_idx][c*INPUT_WIDTH+:INPUT_WIDTH];
          x_eff_tmp  = $signed(sample_val) - INPUT_ZERO_POINT;
          if (x_eff_tmp < 0) x_eff_tmp = 0;
          else if (x_eff_tmp > ((1 << X_EFF_WIDTH) - 1)) x_eff_tmp = (1 << X_EFF_WIDTH) - 1;

          weight_val = golden_weight_mem[weight_tile_idx][(r*TILE_INPUT_SIZE+c)*WEIGHT_WIDTH +: WEIGHT_WIDTH];
          golden_acc[r] = golden_acc[r] + $signed(weight_val) * x_eff_tmp;
        end
      end
    end

    @(posedge clk);
    rst_n <= 1'b1;

    @(posedge clk);
    start <= 1'b1;

    @(posedge clk);
    start <= 1'b0;

    wait (done == 1'b1);
    #1;

    $display("[CHECK] compare fc1_acc_all");
    for (r = 0; r < TILE_OUTPUT_SIZE; r = r + 1) begin
      if (fc1_acc_all[r] !== golden_acc[r]) begin
        $display("ERROR r=%0d got=%0d exp=%0d", r, fc1_acc_all[r], golden_acc[r]);
        err_count = err_count + 1;
      end
    end

    if (err_count == 0) begin
      $display("PASS: fc1_multi_block_shared_sample_rom end-to-end block result is correct.");
    end else begin
      $display("FAIL: fc1_multi_block_shared_sample_rom found %0d mismatches.", err_count);
    end

    $finish;
  end

endmodule
