
`timescale 1ns / 1ps

module tb_fc2_weight_bank;
  import mnist_cim_pkg::*;

  parameter string DEFAULT_WEIGHT_HEX_FILE = "../data_packed/weights/fc2_weight_int8.hex";

  logic clk;
  logic [$clog2(FC2_WEIGHT_DEPTH)-1:0] addr;
  logic signed [WEIGHT_WIDTH-1:0] w_data;

  logic signed [WEIGHT_WIDTH-1:0] golden_weight_mem[0:FC2_WEIGHT_DEPTH-1];

  integer err_count;

  fc2_weight_bank #(
      .DEFAULT_WEIGHT_HEX_FILE(DEFAULT_WEIGHT_HEX_FILE)
  ) dut (
      .clk   (clk),
      .addr  (addr),
      .w_data(w_data)
  );

  initial clk = 1'b0;
  always #5 clk = ~clk;

  initial begin
    $display("============================================================");
    $display("tb_fc2_weight_bank");
    $display("DEFAULT_WEIGHT_HEX_FILE = %s", DEFAULT_WEIGHT_HEX_FILE);
    $display("============================================================");

    $readmemh(DEFAULT_WEIGHT_HEX_FILE, golden_weight_mem);

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
    if (w_data !== golden_weight_mem[0]) begin
      $display("ERROR TEST1 got=0x%02x exp=0x%02x", w_data & 8'hFF, golden_weight_mem[0] & 8'hFF);
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
    if (w_data !== golden_weight_mem[1]) begin
      $display("ERROR TEST2 got=0x%02x exp=0x%02x", w_data & 8'hFF, golden_weight_mem[1] & 8'hFF);
      err_count = err_count + 1;
    end

    // ----------------------------------------------------------
    // TEST3: last address
    // ----------------------------------------------------------
    @(posedge clk);
    addr <= FC2_WEIGHT_DEPTH - 1;

    @(posedge clk);
    #1;

    $display("[TEST3] check addr=%0d", FC2_WEIGHT_DEPTH - 1);
    if (w_data !== golden_weight_mem[FC2_WEIGHT_DEPTH-1]) begin
      $display("ERROR TEST3 got=0x%02x exp=0x%02x", w_data & 8'hFF,
               golden_weight_mem[FC2_WEIGHT_DEPTH-1] & 8'hFF);
      err_count = err_count + 1;
    end

    if (err_count == 0) begin
      $display("PASS: fc2_weight_bank synchronous single-word read is correct.");
    end else begin
      $display("FAIL: fc2_weight_bank found %0d mismatches.", err_count);
    end

    $finish;
  end

endmodule
