
`timescale 1ns / 1ps

module tb_mnist_inference_core_board_debug;
  import mnist_cim_pkg::*;

  parameter int N_SAMPLES = 20;
  parameter string DEFAULT_SAMPLE_HEX_FILE     = "../data_packed/samples/mnist_samples_route_b_output_2.hex";
  parameter string DEFAULT_FC1_WEIGHT_HEX_FILE = "../data_packed/weights/fc1_weight_int8.hex";
  parameter string DEFAULT_FC1_BIAS_HEX_FILE = "../data_packed/weights/fc1_bias_int32.hex";
  parameter string DEFAULT_QUANT_PARAM_FILE = "../data_packed/quant/quant_params.hex";
  parameter string DEFAULT_FC2_WEIGHT_HEX_FILE = "../data_packed/weights/fc2_weight_int8.hex";
  parameter string DEFAULT_FC2_BIAS_HEX_FILE = "../data_packed/weights/fc2_bias_int32.hex";
  parameter string PRED_FILE = "../data_packed/expected/pred_0.txt";

  logic clk;
  logic rst_n;
  logic start;
  logic [$clog2(N_SAMPLES)-1:0] sample_id;

  logic busy, done;
  logic signed [OUTPUT_WIDTH-1:0] logits_all[0:FC2_OUT_DIM-1];
  logic [$clog2(FC2_OUT_DIM)-1:0] pred_class;

  // golden memories
  localparam int SAMPLE_TILE_BITS = TILE_INPUT_SIZE * INPUT_WIDTH;
  localparam int SAMPLE_TILE_DEPTH = N_SAMPLES * N_INPUT_BLOCKS;
  localparam int WEIGHT_TILE_BITS = TILE_OUTPUT_SIZE * TILE_INPUT_SIZE * WEIGHT_WIDTH;
  localparam int WEIGHT_TILE_DEPTH = N_OUTPUT_BLOCKS * N_INPUT_BLOCKS;
  localparam int BIAS_BLOCK_BITS = TILE_OUTPUT_SIZE * BIAS_WIDTH;

  logic [SAMPLE_TILE_BITS-1:0] sample_mem[0:SAMPLE_TILE_DEPTH-1];
  logic [WEIGHT_TILE_BITS-1:0] fc1_w_mem[0:WEIGHT_TILE_DEPTH-1];
  logic [BIAS_BLOCK_BITS-1:0] fc1_b_mem[0:N_OUTPUT_BLOCKS-1];

  logic [31:0] quant_mem[0:3];
  logic signed [WEIGHT_WIDTH-1:0] fc2_w_mem[0:FC2_WEIGHT_DEPTH-1];
  logic signed [BIAS_WIDTH-1:0] fc2_b_mem[0:FC2_OUT_DIM-1];

  logic signed [PSUM_WIDTH-1:0] golden_fc1_acc[0:HIDDEN_DIM-1];
  logic signed [PSUM_WIDTH-1:0] golden_fc1_relu[0:HIDDEN_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] golden_fc1_out[0:HIDDEN_DIM-1];
  logic signed [PSUM_WIDTH-1:0] golden_fc2_acc[0:FC2_OUT_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] golden_logits[0:FC2_OUT_DIM-1];

  logic [31:0] fc1_requant_mult, fc1_requant_shift;
  logic [31:0] fc2_requant_mult, fc2_requant_shift;

  integer fd, rscan, ref_pred_class;
  integer ob_blk, ib_idx, r, c, o, i;
  integer err_count;
  integer sample_tile_idx, weight_tile_idx, flat_idx, x_eff_tmp;
  longint signed acc_tmp;

  logic signed [INPUT_WIDTH-1:0] sample_val;
  logic signed [WEIGHT_WIDTH-1:0] weight_val;
  logic signed [BIAS_WIDTH-1:0] bias_val;

  mnist_inference_core_board #(
      .N_SAMPLES(N_SAMPLES),
      .DEFAULT_SAMPLE_HEX_FILE(DEFAULT_SAMPLE_HEX_FILE),
      .DEFAULT_FC1_WEIGHT_HEX_FILE(DEFAULT_FC1_WEIGHT_HEX_FILE),
      .DEFAULT_FC1_BIAS_HEX_FILE(DEFAULT_FC1_BIAS_HEX_FILE),
      .DEFAULT_QUANT_PARAM_FILE(DEFAULT_QUANT_PARAM_FILE),
      .DEFAULT_FC2_WEIGHT_HEX_FILE(DEFAULT_FC2_WEIGHT_HEX_FILE),
      .DEFAULT_FC2_BIAS_HEX_FILE(DEFAULT_FC2_BIAS_HEX_FILE)
  ) dut (
      .clk(clk),
      .rst_n(rst_n),
      .start(start),
      .sample_id(sample_id),
      .busy(busy),
      .done(done),
      .logits_all(logits_all),
      .pred_class(pred_class)
  );

  initial clk = 1'b0;
  always #5 clk = ~clk;

  function automatic signed [OUTPUT_WIDTH-1:0] requant(
      input logic signed [PSUM_WIDTH-1:0] x, input logic [31:0] mult, input logic [31:0] rshift);
    longint signed prod, shifted;
    begin
      prod = x * $signed(mult);
      if (rshift == 0) shifted = prod;
      else shifted = (prod + (64'sd1 <<< (rshift - 1))) >>> rshift;

      if (shifted > 127) requant = 8'sd127;
      else if (shifted < -128) requant = -8'sd128;
      else requant = shifted[OUTPUT_WIDTH-1:0];
    end
  endfunction

  initial begin
    $display("============================================================");
    $display("tb_mnist_inference_core_board_debug");
    $display("============================================================");

    $readmemh(DEFAULT_SAMPLE_HEX_FILE, sample_mem);
    $readmemh(DEFAULT_FC1_WEIGHT_HEX_FILE, fc1_w_mem);
    $readmemh(DEFAULT_FC1_BIAS_HEX_FILE, fc1_b_mem);
    $readmemh(DEFAULT_QUANT_PARAM_FILE, quant_mem);
    $readmemh(DEFAULT_FC2_WEIGHT_HEX_FILE, fc2_w_mem);
    $readmemh(DEFAULT_FC2_BIAS_HEX_FILE, fc2_b_mem);

    fc1_requant_mult = quant_mem[0];
    fc1_requant_shift = quant_mem[1];
    fc2_requant_mult = quant_mem[2];
    fc2_requant_shift = quant_mem[3];

    fd = $fopen(PRED_FILE, "r");
    if (fd == 0) begin
      $display("ERROR: cannot open pred file");
      $finish;
    end
    rscan = $fscanf(fd, "%d", ref_pred_class);
    $fclose(fd);

    // ----------------------------------------------------------
    // Golden FC1
    // ----------------------------------------------------------
    for (ob_blk = 0; ob_blk < N_OUTPUT_BLOCKS; ob_blk = ob_blk + 1) begin
      for (r = 0; r < TILE_OUTPUT_SIZE; r = r + 1) begin
        flat_idx = ob_blk * TILE_OUTPUT_SIZE + r;
        bias_val = fc1_b_mem[ob_blk][r*BIAS_WIDTH+:BIAS_WIDTH];
        golden_fc1_acc[flat_idx] = bias_val;
      end
    end

    for (ib_idx = 0; ib_idx < N_INPUT_BLOCKS; ib_idx = ib_idx + 1) begin
      sample_tile_idx = 0 * N_INPUT_BLOCKS + ib_idx;

      for (ob_blk = 0; ob_blk < N_OUTPUT_BLOCKS; ob_blk = ob_blk + 1) begin
        weight_tile_idx = ob_blk * N_INPUT_BLOCKS + ib_idx;

        for (r = 0; r < TILE_OUTPUT_SIZE; r = r + 1) begin
          flat_idx = ob_blk * TILE_OUTPUT_SIZE + r;

          for (c = 0; c < TILE_INPUT_SIZE; c = c + 1) begin
            sample_val = sample_mem[sample_tile_idx][c*INPUT_WIDTH+:INPUT_WIDTH];

            x_eff_tmp  = $signed(sample_val) - INPUT_ZERO_POINT;
            if (x_eff_tmp < 0) x_eff_tmp = 0;
            else if (x_eff_tmp > ((1 << X_EFF_WIDTH) - 1)) x_eff_tmp = (1 << X_EFF_WIDTH) - 1;

            weight_val = fc1_w_mem[weight_tile_idx][(r*TILE_INPUT_SIZE+c)*WEIGHT_WIDTH +: WEIGHT_WIDTH];
            golden_fc1_acc[flat_idx] = golden_fc1_acc[flat_idx] + $signed(weight_val) * x_eff_tmp;
          end
        end
      end
    end

    // Golden FC1 relu/requant
    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
      if (golden_fc1_acc[i] < 0) golden_fc1_relu[i] = 0;
      else golden_fc1_relu[i] = golden_fc1_acc[i];

      golden_fc1_out[i] = requant(golden_fc1_relu[i], fc1_requant_mult, fc1_requant_shift);
    end

    // Golden FC2
    for (o = 0; o < FC2_OUT_DIM; o = o + 1) begin
      acc_tmp = fc2_b_mem[o];
      for (i = 0; i < FC2_IN_DIM; i = i + 1) begin
        acc_tmp = acc_tmp + $signed(golden_fc1_out[i]) * $signed(fc2_w_mem[o*FC2_IN_DIM+i]);
      end
      golden_fc2_acc[o] = acc_tmp[PSUM_WIDTH-1:0];
      golden_logits[o]  = requant(golden_fc2_acc[o], fc2_requant_mult, fc2_requant_shift);
    end

    err_count = 0;
    rst_n     = 1'b0;
    start     = 1'b0;
    sample_id = '0;

    @(posedge clk);
    rst_n <= 1'b1;

    @(posedge clk);
    start <= 1'b1;

    @(posedge clk);
    start <= 1'b0;

    wait (done == 1'b1);
    #1;

    // ----------------------------------------------------------
    // Compare stage by stage
    // ----------------------------------------------------------
    $display("========== STAGE CHECK ==========");


    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
      if (dut.fc1_acc_all_reg[i] !== golden_fc1_acc[i]) begin
        $display("FC1_ACC_REG mismatch i=%0d got=%0d exp=%0d", i, dut.fc1_acc_all_reg[i],
                 golden_fc1_acc[i]);
        err_count = err_count + 1;
        disable compare_fc1_acc_reg_done;
      end
    end
    compare_fc1_acc_reg_done : begin
    end

    if (err_count == 0) $display("FC1_ACC OK");

    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
      if (dut.fc1_out_all[i] !== golden_fc1_out[i]) begin
        $display("FC1_OUT mismatch i=%0d got=%0d exp=%0d", i, dut.fc1_out_all[i],
                 golden_fc1_out[i]);
        err_count = err_count + 1;
        disable compare_fc1_out_done;
      end
    end
    compare_fc1_out_done : begin
    end

    for (o = 0; o < FC2_OUT_DIM; o = o + 1) begin
      if (dut.fc2_acc_all[o] !== golden_fc2_acc[o]) begin
        $display("FC2_ACC mismatch o=%0d got=%0d exp=%0d", o, dut.fc2_acc_all[o],
                 golden_fc2_acc[o]);
        err_count = err_count + 1;
        disable compare_fc2_acc_done;
      end
    end
    compare_fc2_acc_done : begin
    end

    for (o = 0; o < FC2_OUT_DIM; o = o + 1) begin
      if (dut.logits_all[o] !== golden_logits[o]) begin
        $display("LOGIT mismatch o=%0d got=%0d exp=%0d", o, dut.logits_all[o], golden_logits[o]);
        err_count = err_count + 1;
        disable compare_logits_done;
      end
    end
    compare_logits_done : begin
    end

    $display("[FINAL] pred_class=%0d ref=%0d", pred_class, ref_pred_class);

    $finish;
  end

  initial begin
    #200_000_000ns;
    $display("ERROR: timeout in tb_mnist_inference_core_board_debug");
    $finish;
  end

endmodule
