
module argmax_int8 (
    input logic signed [mnist_cim_pkg::OUTPUT_WIDTH-1:0] logits_all[0:mnist_cim_pkg::FC2_OUT_DIM-1],

    output logic [$clog2(mnist_cim_pkg::FC2_OUT_DIM)-1:0] pred_class
);
  import mnist_cim_pkg::*;

  integer i;
  logic signed [OUTPUT_WIDTH-1:0] max_val;
  logic [$clog2(FC2_OUT_DIM)-1:0] max_idx;

  always_comb begin
    max_val = logits_all[0];
    max_idx = '0;

    for (i = 1; i < FC2_OUT_DIM; i = i + 1) begin
      if (logits_all[i] > max_val) begin
        max_val = logits_all[i];
        max_idx = i[$clog2(FC2_OUT_DIM)-1:0];
      end
    end

    pred_class = max_idx;
  end

endmodule
