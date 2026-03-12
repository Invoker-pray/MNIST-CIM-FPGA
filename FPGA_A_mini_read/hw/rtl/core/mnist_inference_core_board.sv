// mnist_inference_core_board.sv
// 新增统一写总线接口，地址译码后分发给各 bank
// start 信号受 load_done 保护，未加载完成时无法启动推理

module mnist_inference_core_board #(
    parameter int N_SAMPLES = 20
) (
    input logic clk,
    input logic rst_n,

    // Inference control
    input  logic start,
    input  logic [$clog2(N_SAMPLES)-1:0] sample_id,
    output logic busy,
    output logic done,

    output logic signed [mnist_cim_pkg::OUTPUT_WIDTH-1:0]
        logits_all [0:mnist_cim_pkg::FC2_OUT_DIM-1],
    output logic [$clog2(mnist_cim_pkg::FC2_OUT_DIM)-1:0] pred_class,

    // ── 统一写总线（32-bit word 地址）────────────────────────────────────
    // 外部按地址顺序将数据写入对应 bank
    // 地址映射见下方 localparam
    input logic        mem_we,
    input logic [15:0] mem_waddr,   // 32-bit word 地址
    input logic [31:0] mem_wdata
);
  import mnist_cim_pkg::*;

  // ── 地址映射 ──────────────────────────────────────────────────────────────
  // FC1 Weight : [FC1W_BASE .. FC1W_BASE + FC1W_WORDS - 1]
  //   TILE_DEPTH=49, WORDS_PER_TILE=64  => 3136 words
  // FC1 Bias   : [FC1B_BASE .. + FC1B_WORDS - 1]
  //   N_OUTPUT_BLOCKS=1, WORDS_PER_BLOCK=16 => 16 words
  // FC2 Weight : [FC2W_BASE .. + FC2W_WORDS - 1]
  //   FC2_WEIGHT_DEPTH=160 => 160 words
  // FC2 Bias   : [FC2B_BASE .. + FC2B_WORDS - 1]
  //   FC2_OUT_DIM=10 => 10 words
  // Quant Param: [QUANT_BASE .. + 4 - 1] => 4 words
  // Sample ROM : [SAMP_BASE .. + SAMP_WORDS - 1]
  //   TILE_DEPTH=980, WORDS_PER_TILE=4 => 3920 words
  // ─────────────────────────────────────────────────────────────────────────

  localparam int FC1W_TILE_DEPTH     = N_OUTPUT_BLOCKS * N_INPUT_BLOCKS; // 49
  localparam int FC1W_WORDS_PER_TILE = (TILE_OUTPUT_SIZE * TILE_INPUT_SIZE * WEIGHT_WIDTH) / 32; // 64
  localparam int FC1W_WORDS          = FC1W_TILE_DEPTH * FC1W_WORDS_PER_TILE; // 3136

  localparam int FC1B_WORDS_PER_BLK  = (TILE_OUTPUT_SIZE * BIAS_WIDTH) / 32; // 16
  localparam int FC1B_WORDS          = N_OUTPUT_BLOCKS * FC1B_WORDS_PER_BLK; // 16

  localparam int FC2W_WORDS          = FC2_WEIGHT_DEPTH;                 // 160
  localparam int FC2B_WORDS          = FC2_OUT_DIM;                      // 10
  localparam int QUANT_WORDS         = 4;

  localparam int SAMP_TILE_DEPTH     = N_SAMPLES * N_INPUT_BLOCKS;       // 980
  localparam int SAMP_WORDS_PER_TILE = (TILE_INPUT_SIZE * INPUT_WIDTH) / 32; // 4
  localparam int SAMP_WORDS          = SAMP_TILE_DEPTH * SAMP_WORDS_PER_TILE; // 3920

  localparam int FC1W_BASE  = 0;
  localparam int FC1B_BASE  = FC1W_BASE  + FC1W_WORDS;   // 3136
  localparam int FC2W_BASE  = FC1B_BASE  + FC1B_WORDS;   // 3152
  localparam int FC2B_BASE  = FC2W_BASE  + FC2W_WORDS;   // 3312
  localparam int QUANT_BASE = FC2B_BASE  + FC2B_WORDS;   // 3322
  localparam int SAMP_BASE  = QUANT_BASE + QUANT_WORDS;  // 3326

  // ── 地址译码 → 各 bank 写使能 ─────────────────────────────────────────
  logic fc1w_wr_en,  fc1b_wr_en;
  logic fc2w_wr_en,  fc2b_wr_en;
  logic quant_wr_en, samp_wr_en;

  always_comb begin
    fc1w_wr_en  = mem_we && (mem_waddr >= FC1W_BASE)  && (mem_waddr < FC1W_BASE  + FC1W_WORDS);
    fc1b_wr_en  = mem_we && (mem_waddr >= FC1B_BASE)  && (mem_waddr < FC1B_BASE  + FC1B_WORDS);
    fc2w_wr_en  = mem_we && (mem_waddr >= FC2W_BASE)  && (mem_waddr < FC2W_BASE  + FC2W_WORDS);
    fc2b_wr_en  = mem_we && (mem_waddr >= FC2B_BASE)  && (mem_waddr < FC2B_BASE  + FC2B_WORDS);
    quant_wr_en = mem_we && (mem_waddr >= QUANT_BASE) && (mem_waddr < QUANT_BASE + QUANT_WORDS);
    samp_wr_en  = mem_we && (mem_waddr >= SAMP_BASE)  && (mem_waddr < SAMP_BASE  + SAMP_WORDS);
  end

  // ── 内部信号 ──────────────────────────────────────────────────────────────
  typedef enum logic [2:0] {
    S_IDLE      = 3'd0,
    S_FC1_WAIT  = 3'd1,
    S_LATCH_FC1 = 3'd2,
    S_FC2_START = 3'd3,
    S_FC2_WAIT  = 3'd4,
    S_DONE      = 3'd5
  } state_t;

  state_t state, state_n;

  logic fc1_busy, fc1_done;
  logic fc2_busy, fc2_done;
  logic fc2_start;

  logic signed [PSUM_WIDTH-1:0]   fc1_acc_all_wire [0:HIDDEN_DIM-1];
  logic signed [PSUM_WIDTH-1:0]   fc1_acc_all_reg  [0:HIDDEN_DIM-1];
  logic signed [PSUM_WIDTH-1:0]   fc1_relu_all     [0:HIDDEN_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] fc1_out_all      [0:HIDDEN_DIM-1];
  logic signed [PSUM_WIDTH-1:0]   fc2_acc_all      [0:FC2_OUT_DIM-1];

  integer i;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_IDLE;
      for (i = 0; i < HIDDEN_DIM; i = i + 1)
        fc1_acc_all_reg[i] <= '0;
    end else begin
      state <= state_n;
      if (state == S_LATCH_FC1)
        for (i = 0; i < HIDDEN_DIM; i = i + 1)
          fc1_acc_all_reg[i] <= fc1_acc_all_wire[i];
    end
  end

  always_comb begin
    state_n   = state;
    fc2_start = 1'b0;
    busy      = 1'b0;
    done      = 1'b0;

    case (state)
      S_IDLE:      if (start) state_n = S_FC1_WAIT;
      S_FC1_WAIT:  begin busy = 1'b1; if (fc1_done) state_n = S_LATCH_FC1; end
      S_LATCH_FC1: begin busy = 1'b1; state_n = S_FC2_START; end
      S_FC2_START: begin busy = 1'b1; fc2_start = 1'b1; state_n = S_FC2_WAIT; end
      S_FC2_WAIT:  begin busy = 1'b1; if (fc2_done) state_n = S_DONE; end
      S_DONE:      begin done = 1'b1; if (!start) state_n = S_IDLE; end
      default:     begin state_n = S_IDLE; fc2_start = 1'b0; busy = 1'b0; done = 1'b0; end
    endcase
  end

  // ── FC1 子模块 ─────────────────────────────────────────────────────────
  fc1_multi_block_shared_sample_rom #(
      .PAR_OB   (1),
      .N_SAMPLES(N_SAMPLES)
  ) u_fc1 (
      .clk         (clk),
      .rst_n       (rst_n),
      .start       (start),
      .sample_id   (sample_id),
      .base_ob     ('0),
      .busy        (fc1_busy),
      .done        (fc1_done),
      .fc1_acc_all (fc1_acc_all_wire),
      .samp_wr_en  (samp_wr_en),
      .samp_wr_data(mem_wdata),
      .fc1w_wr_en  (fc1w_wr_en),
      .fc1w_wr_data(mem_wdata),
      .fc1b_wr_en  (fc1b_wr_en),
      .fc1b_wr_data(mem_wdata)
  );

  // ── FC2 子模块 ─────────────────────────────────────────────────────────
  fc1_to_fc2_top_with_file u_fc1_to_fc2 (
      .clk          (clk),
      .rst_n        (rst_n),
      .start        (fc2_start),
      .fc1_acc_all  (fc1_acc_all_reg),
      .busy         (fc2_busy),
      .done         (fc2_done),
      .fc1_relu_all (fc1_relu_all),
      .fc1_out_all  (fc1_out_all),
      .fc2_acc_all  (fc2_acc_all),
      .logits_all   (logits_all),
      .quant_wr_en  (quant_wr_en),
      .quant_wr_data(mem_wdata),
      .fc2w_wr_en   (fc2w_wr_en),
      .fc2w_wr_data (mem_wdata),
      .fc2b_wr_en   (fc2b_wr_en),
      .fc2b_wr_data (mem_wdata)
  );

  argmax_int8 u_argmax (
      .logits_all(logits_all),
      .pred_class(pred_class)
  );

endmodule
