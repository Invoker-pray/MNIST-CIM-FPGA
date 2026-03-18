
`timescale 1ns / 1ps

module tb_fc2_bias_bank;
  import mnist_cim_pkg::*;

  parameter string DEFAULT_BIAS_HEX_FILE = "../data_packed/weights/fc2_bias_int32.hex";

  logic clk;
  logic [$clog2(FC2_OUT_DIM)-1:0] addr;
  logic signed [BIAS_WIDTH-1:0] bias_data;

  logic signed [BIAS_WIDTH-1:0] golden_bias_mem[0:FC2_OUT_DIM-1];

  integer err_count;

  fc2_bias_bank #(
      .DEFAULT_BIAS_HEX_FILE(DEFAULT_BIAS_HEX_FILE)
  ) dut (
      .clk      (clk),
      .addr     (addr),
      .bias_data(bias_data)
  );

  initial clk = 1'b0;
  always #5 clk = ~clk;

  initial begin
    $display("============================================================");
    $display("tb_fc2_bias_bank");
    $display("DEFAULT_BIAS_HEX_FILE = %s", DEFAULT_BIAS_HEX_FILE);
    $display("============================================================");

    $readmemh(DEFAULT_BIAS_HEX_FILE, golden_bias_mem);

    err_count = 0;
    addr = '0;

    // ----------------------------------------------------------
    // TEST1: addr = 0
    // ----------------------------------------------------------
    @(posedge clk);
    addr <= '0;

    @(posedge clk);
    #1;

    $display("[TEST1] check addr=0");
    if (bias_data !== golden_bias_mem[0]) begin
      $display("ERROR TEST1 got=0x%08x exp=0x%08x", bias_data, golden_bias_mem[0]);
      err_count = err_count + 1;
    end

    // ----------------------------------------------------------
    // TEST2: addr = 1
    // ----------------------------------------------------------
    @(posedge clk);
    addr <= 1;

    @(posedge clk);
    #1;

    $display("[TEST2] check addr=1");
    if (bias_data !== golden_bias_mem[1]) begin
      $display("ERROR TEST2 got=0x%08x exp=0x%08x", bias_data, golden_bias_mem[1]);
      err_count = err_count + 1;
    end

    // ----------------------------------------------------------
    // TEST3: last address
    // ----------------------------------------------------------
    @(posedge clk);
    addr <= FC2_OUT_DIM - 1;

    @(posedge clk);
    #1;

    $display("[TEST3] check addr=%0d", FC2_OUT_DIM - 1);
    if (bias_data !== golden_bias_mem[FC2_OUT_DIM-1]) begin
      $display("ERROR TEST3 got=0x%08x exp=0x%08x", bias_data, golden_bias_mem[FC2_OUT_DIM-1]);
      err_count = err_count + 1;
    end

    if (err_count == 0) begin
      $display("PASS: fc2_bias_bank synchronous single-word read is correct.");
    end else begin
      $display("FAIL: fc2_bias_bank found %0d mismatches.", err_count);
    end

    $finish;
  end

endmodule
