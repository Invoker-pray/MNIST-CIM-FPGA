
`timescale 1ns / 1ps

module tb_mnist_cim_accel_ip;
  import mnist_cim_pkg::*;

  parameter int N_SAMPLES = 20;
  parameter string SAMPLE_HEX_FILE = "../data/mnist_samples_route_b_output_2.hex";
  parameter string FC1_WEIGHT_HEX_FILE = "../route_b_output_2/fc1_weight_int8.hex";
  parameter string FC1_BIAS_HEX_FILE = "../route_b_output_2/fc1_bias_int32.hex";
  parameter string QUANT_PARAM_FILE = "../route_b_output_2/quant_params.hex";
  parameter string FC2_WEIGHT_HEX_FILE = "../route_b_output_2/fc2_weight_int8.hex";
  parameter string FC2_BIAS_HEX_FILE = "../route_b_output_2/fc2_bias_int32.hex";
  parameter string LOGITS_FILE = "../route_b_output_2/logits_0.hex";
  parameter string PRED_FILE = "../route_b_output_2/pred_0.txt";

  logic clk;
  logic rst_n;
  logic csr_we;
  logic csr_re;
  logic [7:0] csr_addr;
  logic [31:0] csr_wdata;
  logic [31:0] csr_rdata;
  logic irq_done;
  logic [$clog2(FC2_OUT_DIM)-1:0] pred_class_dbg;
  logic busy_dbg;
  logic done_dbg;

  logic signed [OUTPUT_WIDTH-1:0] ref_logits[0:FC2_OUT_DIM-1];
  integer ref_pred_class;
  integer i, fd, r;
  integer error_count;

  mnist_cim_accel_ip #(
      .N_SAMPLES(N_SAMPLES),
      .DEFAULT_SAMPLE_HEX_FILE(SAMPLE_HEX_FILE),
      .DEFAULT_FC1_WEIGHT_HEX_FILE(FC1_WEIGHT_HEX_FILE),
      .DEFAULT_FC1_BIAS_HEX_FILE(FC1_BIAS_HEX_FILE),
      .DEFAULT_QUANT_PARAM_FILE(QUANT_PARAM_FILE),
      .DEFAULT_FC2_WEIGHT_HEX_FILE(FC2_WEIGHT_HEX_FILE),
      .DEFAULT_FC2_BIAS_HEX_FILE(FC2_BIAS_HEX_FILE)
  ) dut (
      .clk(clk),
      .rst_n(rst_n),
      .csr_we(csr_we),
      .csr_re(csr_re),
      .csr_addr(csr_addr),
      .csr_wdata(csr_wdata),
      .csr_rdata(csr_rdata),
      .irq_done(irq_done),
      .pred_class_dbg(pred_class_dbg),
      .busy_dbg(busy_dbg),
      .done_dbg(done_dbg)
  );

  initial clk = 1'b0;
  always #5 clk = ~clk;

  task automatic csr_write(input [7:0] addr, input [31:0] data);
    begin
      @(posedge clk);
      csr_we <= 1'b1;
      csr_addr <= addr;
      csr_wdata <= data;
      @(posedge clk);
      csr_we <= 1'b0;
      csr_addr <= 8'h00;
      csr_wdata <= 32'd0;
    end
  endtask

  task automatic csr_read(input [7:0] addr);
    begin
      csr_re   = 1'b1;
      csr_addr = addr;
      #1;
      csr_re = 1'b0;
    end
  endtask

  initial begin
    $readmemh(LOGITS_FILE, ref_logits);

    fd = $fopen(PRED_FILE, "r");
    if (fd == 0) begin
      $display("ERROR: cannot open pred file.");
      $finish;
    end
    r = $fscanf(fd, "%d", ref_pred_class);
    $fclose(fd);
    if (r != 1) begin
      $display("ERROR: failed to parse pred file.");
      $finish;
    end
  end

  initial begin
    error_count = 0;
    rst_n = 1'b0;
    csr_we = 1'b0;
    csr_re = 1'b0;
    csr_addr = 8'h00;
    csr_wdata = 32'd0;

    #12;
    rst_n = 1'b1;

    csr_write(8'h08, 32'd0);  // sample_id
    csr_write(8'h00, 32'h1);  // start

    wait (irq_done == 1'b1);
    #1;

    csr_read(8'h0C);
    if (csr_rdata[$clog2(FC2_OUT_DIM)-1:0] !== ref_pred_class[$clog2(FC2_OUT_DIM)-1:0]) begin
      $display("ERROR PRED got=%0d expected=%0d", csr_rdata[$clog2(FC2_OUT_DIM)-1:0],
               ref_pred_class);
      error_count = error_count + 1;
    end

    for (i = 0; i < FC2_OUT_DIM; i = i + 1) begin
      csr_read(8'h10 + i * 4);
      if ($signed(csr_rdata[OUTPUT_WIDTH-1:0]) !== ref_logits[i]) begin
        $display("ERROR LOGIT idx=%0d got=%0d expected=%0d", i,
                 $signed(csr_rdata[OUTPUT_WIDTH-1:0]), ref_logits[i]);
        error_count = error_count + 1;
      end
    end

    if (error_count == 0) $display("PASS: mnist_cim_accel_ip CSR path works for sample0.");
    else $display("FAIL: found %0d mismatches.", error_count);

    $finish;
  end

endmodule
