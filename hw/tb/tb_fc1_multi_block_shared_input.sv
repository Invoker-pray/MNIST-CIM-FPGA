
`timescale 1ns / 1ps

module tb_fc1_multi_block_shared_input;
  import mnist_cim_pkg::*;

  parameter int PAR_OB = 8;
  parameter int BASE_OB = 0;
  localparam int OUT_DIM = PAR_OB * TILE_OUTPUT_SIZE;

  logic clk;
  logic rst_n;
  logic start;
  logic busy;
  logic done;

  logic signed [PSUM_WIDTH-1:0] fc1_acc_all[0:OUT_DIM-1];
  logic signed [PSUM_WIDTH-1:0] ref_fc1_acc_mem[0:HIDDEN_DIM-1];

  string fc1_acc_file;

  integer i;
  integer error_count;
  integer global_idx;

  fc1_multi_block_shared_input #(
      .PAR_OB (PAR_OB),
      .BASE_OB(BASE_OB)
  ) dut (
      .clk(clk),
      .rst_n(rst_n),
      .start(start),
      .busy(busy),
      .done(done),
      .fc1_acc_all(fc1_acc_all)
  );

  initial clk = 1'b0;
  always #5 clk = ~clk;

  initial begin
    fc1_acc_file = "../../CIM-sw-version1/sw/train_quantize/route_b_output/fc1_acc_0.hex";

    if ($value$plusargs("FC1_ACC_FILE=%s", fc1_acc_file)) begin
      $display("TB using FC1_ACC_FILE: %s", fc1_acc_file);
    end else begin
      $display("TB using default FC1_ACC_FILE: %s", fc1_acc_file);
    end

    $readmemh(fc1_acc_file, ref_fc1_acc_mem);
  end

  initial begin
    $display("time=%0t : TB start, PAR_OB=%0d, BASE_OB=%0d, OUT_DIM=%0d", $time, PAR_OB, BASE_OB,
             OUT_DIM);
    $monitor("time=%0t clk=%0b rst_n=%0b start=%0b busy=%0b done=%0b", $time, clk, rst_n, start,
             busy, done);
  end

  initial begin
    error_count = 0;

    rst_n = 1'b0;
    start = 1'b0;

    #12;
    rst_n = 1'b1;

    @(posedge clk);
    start <= 1'b1;

    @(posedge clk);
    start <= 1'b0;

    wait (done == 1'b1);
    @(posedge clk);
    #1;

    $display("Checking fc1_acc_all for shared-input architecture ...");

    for (i = 0; i < OUT_DIM; i = i + 1) begin
      global_idx = BASE_OB * TILE_OUTPUT_SIZE + i;

      if (fc1_acc_all[i] !== ref_fc1_acc_mem[global_idx]) begin
        $display("ERROR idx=%0d global_idx=%0d got=%0d expected=%0d", i, global_idx,
                 fc1_acc_all[i], ref_fc1_acc_mem[global_idx]);
        error_count = error_count + 1;
      end else begin
        $display("MATCH idx=%0d global_idx=%0d value=%0d", i, global_idx, fc1_acc_all[i]);
      end
    end

    if (error_count == 0) $display("PASS: fc1_multi_block_shared_input matches fc1_acc_0.hex.");
    else $display("FAIL: found %0d mismatches.", error_count);

    $finish;
  end

endmodule
