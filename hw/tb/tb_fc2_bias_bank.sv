
`timescale 1ns / 1ps

module tb_fc2_bias_bank;
  import mnist_cim_pkg::*;

  logic signed [BIAS_WIDTH-1:0] bias_all[0:FC2_OUT_DIM-1];
  logic signed [BIAS_WIDTH-1:0] ref_bias_mem[0:FC2_OUT_DIM-1];

  string fc2_bias_file;

  integer i;
  integer error_count;

  fc2_bias_bank dut (.bias_all(bias_all));

  initial begin
    fc2_bias_file = "../route_b_output/fc2_bias_int32.hex";

    if ($value$plusargs("FC2_BIAS_FILE=%s", fc2_bias_file))
      $display("TB using FC2_BIAS_FILE: %s", fc2_bias_file);
    else $display("TB using default FC2_BIAS_FILE: %s", fc2_bias_file);

    $readmemh(fc2_bias_file, ref_bias_mem);
  end

  initial begin
    error_count = 0;
    #1;

    for (i = 0; i < FC2_OUT_DIM; i = i + 1) begin
      if (bias_all[i] !== ref_bias_mem[i]) begin
        $display("ERROR idx=%0d got=%0d expected=%0d", i, bias_all[i], ref_bias_mem[i]);
        error_count = error_count + 1;
      end else begin
        $display("MATCH idx=%0d value=%0d", i, bias_all[i]);
      end
    end

    if (error_count == 0) $display("PASS: fc2_bias_bank matches fc2_bias_int32.hex.");
    else $display("FAIL: found %0d mismatches.", error_count);

    $finish;
  end

endmodule
