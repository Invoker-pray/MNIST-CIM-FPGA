
module tb_fc1_cim_core_block;
  import mnist_cim_pkg::*;

  logic clk;
  logic rst_n;
  logic start;
  logic busy;
  logic done;

  logic [$clog2(N_OUTPUT_BLOCKS)-1:0] ob_sel;

  logic signed [PSUM_WIDTH-1:0] fc1_acc_block[0:TILE_OUTPUT_SIZE-1];

  string fc1_acc_file;

  logic signed [PSUM_WIDTH-1:0] ref_fc1_acc_mem[0:HIDDEN_DIM-1];

  integer i;
  integer error_count;
  integer global_idx;

  fc1_cim_core_block dut (
      .clk(clk),
      .rst_n(rst_n),
      .start(start),
      .ob_sel(ob_sel),
      .busy(busy),
      .done(done),
      .fc1_acc_block(fc1_acc_block)
  );

  // --------------------------------------------
  // Clock
  // --------------------------------------------
  initial clk = 1'b0;
  always #5 clk = ~clk;

  // --------------------------------------------
  // TB golden file from plusarg
  // --------------------------------------------
  initial begin
    fc1_acc_file = "../../CIM-sw-version1/sw/train_quantize/route_b_output/fc1_acc_0.hex";

    if ($value$plusargs("FC1_ACC_FILE=%s", fc1_acc_file)) begin
      $display("TB using FC1_ACC_FILE: %s", fc1_acc_file);
    end else begin
      $display("TB using default FC1_ACC_FILE: %s", fc1_acc_file);
    end

    $readmemh(fc1_acc_file, ref_fc1_acc_mem);
  end

  // --------------------------------------------
  // Main test
  // --------------------------------------------
  initial begin
    error_count = 0;

    rst_n = 1'b0;
    start = 1'b0;
    ob_sel = '0;

    #12;
    rst_n  = 1'b1;

    // 固定先测 ob = 0
    ob_sel = 0;

    // start pulse
    @(posedge clk);
    start <= 1'b1;

    @(posedge clk);
    start <= 1'b0;

    wait (done == 1'b1);
    #1;

    $display("Checking fc1_acc_block for ob_sel = %0d ...", ob_sel);

    for (i = 0; i < TILE_OUTPUT_SIZE; i = i + 1) begin
      global_idx = ob_sel * TILE_OUTPUT_SIZE + i;

      if (fc1_acc_block[i] !== ref_fc1_acc_mem[global_idx]) begin
        $display("ERROR ob=%0d idx=%0d got=%0d expected=%0d", ob_sel, i, fc1_acc_block[i],
                 ref_fc1_acc_mem[global_idx]);
        error_count = error_count + 1;
      end else begin
        $display("MATCH ob=%0d idx=%0d value=%0d", ob_sel, i, fc1_acc_block[i]);
      end
    end

    if (error_count == 0) begin
      $display("PASS: fc1_cim_core_block matches fc1_acc_0.hex block.");
    end else begin
      $display("FAIL: found %0d mismatches.", error_count);
    end

    $finish;
  end

endmodule
