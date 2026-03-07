
`timescale 1ns / 1ps

module tb_fc1_relu_requantize;
  import mnist_cim_pkg::*;

  logic signed [PSUM_WIDTH-1:0] fc1_acc_all[0:HIDDEN_DIM-1];
  logic signed [PSUM_WIDTH-1:0] fc1_relu_all[0:HIDDEN_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] fc1_out_all[0:HIDDEN_DIM-1];

  logic signed [PSUM_WIDTH-1:0] ref_fc1_acc_mem[0:HIDDEN_DIM-1];
  logic signed [PSUM_WIDTH-1:0] ref_fc1_relu_mem[0:HIDDEN_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] ref_fc1_out_mem[0:HIDDEN_DIM-1];

  string fc1_acc_file;
  string fc1_relu_file;
  string fc1_out_file;

  integer i;
  integer error_count;

  fc1_relu_requantize dut (
      .fc1_acc_all (fc1_acc_all),
      .fc1_relu_all(fc1_relu_all),
      .fc1_out_all (fc1_out_all)
  );

  initial begin
    fc1_acc_file  = "../route_b_output/fc1_acc_0.hex";
    fc1_relu_file = "../route_b_output/fc1_relu_0.hex";
    fc1_out_file  = "../route_b_output/fc1_out_0.hex";

    if ($value$plusargs("FC1_ACC_FILE=%s", fc1_acc_file))
      $display("TB using FC1_ACC_FILE: %s", fc1_acc_file);
    else $display("TB using default FC1_ACC_FILE: %s", fc1_acc_file);

    if ($value$plusargs("FC1_RELU_FILE=%s", fc1_relu_file))
      $display("TB using FC1_RELU_FILE: %s", fc1_relu_file);
    else $display("TB using default FC1_RELU_FILE: %s", fc1_relu_file);

    if ($value$plusargs("FC1_OUT_FILE=%s", fc1_out_file))
      $display("TB using FC1_OUT_FILE: %s", fc1_out_file);
    else $display("TB using default FC1_OUT_FILE: %s", fc1_out_file);

    $readmemh(fc1_acc_file, ref_fc1_acc_mem);
    $readmemh(fc1_relu_file, ref_fc1_relu_mem);
    $readmemh(fc1_out_file, ref_fc1_out_mem);
  end

  initial begin
    error_count = 0;

    for (i = 0; i < HIDDEN_DIM; i = i + 1) fc1_acc_all[i] = ref_fc1_acc_mem[i];

    #1;

    $display("Checking fc1_relu_all ...");
    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
      if (fc1_relu_all[i] !== ref_fc1_relu_mem[i]) begin
        $display("ERROR RELU idx=%0d got=%0d expected=%0d", i, fc1_relu_all[i],
                 ref_fc1_relu_mem[i]);
        error_count = error_count + 1;
      end else begin
        $display("MATCH RELU idx=%0d value=%0d", i, fc1_relu_all[i]);
      end
    end

    $display("Checking fc1_out_all ...");
    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
      if (fc1_out_all[i] !== ref_fc1_out_mem[i]) begin
        $display("ERROR OUT idx=%0d got=%0d expected=%0d", i, fc1_out_all[i], ref_fc1_out_mem[i]);
        error_count = error_count + 1;
      end else begin
        $display("MATCH OUT idx=%0d value=%0d", i, fc1_out_all[i]);
      end
    end

    if (error_count == 0)
      $display("PASS: fc1_relu_requantize matches fc1_relu_0.hex and fc1_out_0.hex.");
    else $display("FAIL: found %0d mismatches.", error_count);

    $finish;
  end

endmodule
