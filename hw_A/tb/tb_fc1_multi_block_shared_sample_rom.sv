
`timescale 1ns / 1ps

module tb_fc1_multi_block_shared_sample_rom;
  import mnist_cim_pkg::*;

  parameter int N_SAMPLES = 20;
  parameter string SAMPLE_HEX_FILE = "../data/mnist_samples_route_b_output_2.hex";
  parameter string FC1_WEIGHT_HEX_FILE = "../route_b_output_2/fc1_weight_int8.hex";
  parameter string FC1_BIAS_HEX_FILE = "../route_b_output_2/fc1_bias_int32.hex";
  parameter string FC1_ACC_FILE = "../route_b_output_2/fc1_acc_0.hex";

  logic clk;
  logic rst_n;
  logic start;
  logic [$clog2(N_SAMPLES)-1:0] sample_id;
  logic busy;
  logic done;

  logic signed [PSUM_WIDTH-1:0] fc1_acc_all[0:HIDDEN_DIM-1];
  logic signed [PSUM_WIDTH-1:0] ref_fc1_acc[0:HIDDEN_DIM-1];

  integer i;
  integer error_count;

  fc1_multi_block_shared_sample_rom #(
      .PAR_OB(8),
      .BASE_OB(0),
      .N_SAMPLES(N_SAMPLES),
      .DEFAULT_SAMPLE_HEX_FILE(SAMPLE_HEX_FILE),
      .DEFAULT_WEIGHT_HEX_FILE(FC1_WEIGHT_HEX_FILE),
      .DEFAULT_BIAS_HEX_FILE(FC1_BIAS_HEX_FILE)
  ) dut (
      .clk(clk),
      .rst_n(rst_n),
      .start(start),
      .sample_id(sample_id),
      .busy(busy),
      .done(done),
      .fc1_acc_all(fc1_acc_all)
  );

  initial clk = 1'b0;
  always #5 clk = ~clk;

  initial begin
    $display("TB using FC1_ACC_FILE: %s", FC1_ACC_FILE);
    $readmemh(FC1_ACC_FILE, ref_fc1_acc);
  end

  initial begin
    error_count = 0;
    rst_n = 1'b0;
    start = 1'b0;
    sample_id = '0;

    #12;
    rst_n = 1'b1;

    @(posedge clk);
    start <= 1'b1;
    @(posedge clk);
    start <= 1'b0;

    wait (done == 1'b1);
    @(posedge clk);
    #1;

    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
      if (fc1_acc_all[i] !== ref_fc1_acc[i]) begin
        $display("ERROR idx=%0d got=%0d expected=%0d", i, fc1_acc_all[i], ref_fc1_acc[i]);
        error_count = error_count + 1;
      end
    end

    if (error_count == 0)
      $display("PASS: fc1_multi_block_shared_sample_rom matches fc1_acc_0.hex.");
    else $display("FAIL: found %0d mismatches.", error_count);

    $finish;
  end

endmodule
