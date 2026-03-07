
`timescale 1ns / 1ps

module tb_fc1_cim_core_block_dual;
  import mnist_cim_pkg::*;

  logic clk;
  logic rst_n;
  logic start;
  logic busy;
  logic done;

  logic signed [PSUM_WIDTH-1:0] fc1_acc_block0[0:TILE_OUTPUT_SIZE-1];
  logic signed [PSUM_WIDTH-1:0] fc1_acc_block1[0:TILE_OUTPUT_SIZE-1];

  logic signed [PSUM_WIDTH-1:0] ref_fc1_acc_mem[0:HIDDEN_DIM-1];

  string fc1_acc_file;

  integer i;
  integer error_count;

  fc1_cim_core_block_dual dut (
      .clk(clk),
      .rst_n(rst_n),
      .start(start),
      .busy(busy),
      .done(done),
      .fc1_acc_block0(fc1_acc_block0),
      .fc1_acc_block1(fc1_acc_block1)
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

    $display("Checking block0 (ob=0) ...");
    for (i = 0; i < TILE_OUTPUT_SIZE; i = i + 1) begin
      if (fc1_acc_block0[i] !== ref_fc1_acc_mem[i]) begin
        $display("ERROR block0 idx=%0d got=%0d expected=%0d", i, fc1_acc_block0[i],
                 ref_fc1_acc_mem[i]);
        error_count = error_count + 1;
      end else begin
        $display("MATCH block0 idx=%0d value=%0d", i, fc1_acc_block0[i]);
      end
    end

    $display("Checking block1 (ob=1) ...");
    for (i = 0; i < TILE_OUTPUT_SIZE; i = i + 1) begin
      if (fc1_acc_block1[i] !== ref_fc1_acc_mem[TILE_OUTPUT_SIZE+i]) begin
        $display("ERROR block1 idx=%0d got=%0d expected=%0d", i, fc1_acc_block1[i],
                 ref_fc1_acc_mem[TILE_OUTPUT_SIZE+i]);
        error_count = error_count + 1;
      end else begin
        $display("MATCH block1 idx=%0d value=%0d", i, fc1_acc_block1[i]);
      end
    end

    if (error_count == 0) $display("PASS: dual fc1_cim_core_block matches ob=0 and ob=1.");
    else $display("FAIL: found %0d mismatches.", error_count);

    $finish;
  end

endmodule
