
`timescale 1ns / 1ps

module tb_fc2_weight_bank;
  import mnist_cim_pkg::*;

  logic signed [WEIGHT_WIDTH-1:0] w_all[0:FC2_OUT_DIM-1][0:FC2_IN_DIM-1];
  logic signed [WEIGHT_WIDTH-1:0] ref_weight_mem[0:FC2_WEIGHT_DEPTH-1];

  string fc2_weight_file;

  integer o, i, addr;
  integer error_count;

  fc2_weight_bank dut (.w_all(w_all));

  initial begin
    fc2_weight_file = "../route_b_output/fc2_weight_int8.hex";

    if ($value$plusargs("FC2_WEIGHT_HEX_FILE=%s", fc2_weight_file))
      $display("TB using FC2_WEIGHT_HEX_FILE: %s", fc2_weight_file);
    else $display("TB using default FC2_WEIGHT_HEX_FILE: %s", fc2_weight_file);

    $readmemh(fc2_weight_file, ref_weight_mem);
  end

  initial begin
    error_count = 0;
    #1;

    for (o = 0; o < FC2_OUT_DIM; o = o + 1) begin
      for (i = 0; i < FC2_IN_DIM; i = i + 1) begin
        addr = o * FC2_IN_DIM + i;
        if (w_all[o][i] !== ref_weight_mem[addr]) begin
          $display("ERROR o=%0d i=%0d addr=%0d got=%0d expected=%0d", o, i, addr, w_all[o][i],
                   ref_weight_mem[addr]);
          error_count = error_count + 1;
        end
      end
    end

    if (error_count == 0) $display("PASS: fc2_weight_bank matches fc2_weight_int8.hex layout.");
    else $display("FAIL: found %0d mismatches.", error_count);

    $finish;
  end

endmodule
