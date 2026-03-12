// tb_load_helper.sv
// Testbench 辅助 task：演示如何通过 mem_we/mem_waddr/mem_wdata 总线
// 将外部 hex 文件加载到片上 BRAM
//
// 使用方法：在 tb 中 `include "tb_load_helper.sv" 后调用 load_all_data()

// ─── 地址映射常量（与 mnist_inference_core_board 保持一致）───────────────
localparam int FC1W_TILE_DEPTH      = 49;
localparam int FC1W_WORDS_PER_TILE  = 64;   // 2048-bit tile / 32
localparam int FC1W_WORDS           = FC1W_TILE_DEPTH * FC1W_WORDS_PER_TILE; // 3136

localparam int FC1B_WORDS_PER_BLK   = 16;   // 512-bit block / 32
localparam int FC1B_WORDS           = 1 * FC1B_WORDS_PER_BLK;  // 16

localparam int FC2W_WORDS           = 160;
localparam int FC2B_WORDS           = 10;
localparam int QUANT_WORDS          = 4;

localparam int SAMP_WORDS_PER_TILE  = 4;    // 128-bit tile / 32
localparam int SAMP_WORDS           = 980 * SAMP_WORDS_PER_TILE; // 3920

localparam int FC1W_BASE  = 0;
localparam int FC1B_BASE  = FC1W_BASE  + FC1W_WORDS;   // 3136
localparam int FC2W_BASE  = FC1B_BASE  + FC1B_WORDS;   // 3152
localparam int FC2B_BASE  = FC2W_BASE  + FC2W_WORDS;   // 3312
localparam int QUANT_BASE = FC2B_BASE  + FC2B_WORDS;   // 3322
localparam int SAMP_BASE  = QUANT_BASE + QUANT_WORDS;  // 3326

// ─── 写总线 task ──────────────────────────────────────────────────────────

// 单次写操作（需要 tb 中存在对应信号）
task automatic bus_write(
    input int          addr,
    input logic [31:0] data
);
    @(posedge clk);
    mem_we    <= 1'b1;
    mem_waddr <= 16'(addr);
    mem_wdata <= data;
    @(posedge clk);
    mem_we    <= 1'b0;
    mem_waddr <= '0;
    mem_wdata <= '0;
endtask

// ─── 加载 FC1 Weight（2048-bit tile packed hex，共 49 tiles × 64 words）──

task automatic load_fc1_weight(input string hex_file);
    // 每行一个 64-bit hex 值（tile packed 为 512 hex digits = 256 bytes per tile）
    // 这里将 tile 分解为 64 个 32-bit word 后写入
    logic [2047:0] tile_buf [0:48];
    logic  [31:0]  word;
    int tile_idx, word_idx;

    $display("[LOAD] Loading FC1 weights from: %s", hex_file);
    $readmemh(hex_file, tile_buf);

    for (tile_idx = 0; tile_idx < FC1W_TILE_DEPTH; tile_idx++) begin
        for (word_idx = 0; word_idx < FC1W_WORDS_PER_TILE; word_idx++) begin
            word = tile_buf[tile_idx][word_idx*32 +: 32];
            bus_write(FC1W_BASE + tile_idx * FC1W_WORDS_PER_TILE + word_idx, word);
        end
    end
    $display("[LOAD] FC1 weights done.");
endtask

// ─── 加载 FC1 Bias（512-bit block packed hex，共 1 block × 16 words）──────

task automatic load_fc1_bias(input string hex_file);
    logic [511:0] block_buf [0:0];
    logic  [31:0] word;
    int word_idx;

    $display("[LOAD] Loading FC1 bias from: %s", hex_file);
    $readmemh(hex_file, block_buf);

    for (word_idx = 0; word_idx < FC1B_WORDS; word_idx++) begin
        word = block_buf[0][word_idx*32 +: 32];
        bus_write(FC1B_BASE + word_idx, word);
    end
    $display("[LOAD] FC1 bias done.");
endtask

// ─── 加载 FC2 Weight（8-bit per entry，共 160 entries）────────────────────

task automatic load_fc2_weight(input string hex_file);
    logic [7:0] w_buf [0:159];
    int idx;

    $display("[LOAD] Loading FC2 weights from: %s", hex_file);
    $readmemh(hex_file, w_buf);

    for (idx = 0; idx < FC2W_WORDS; idx++) begin
        bus_write(FC2W_BASE + idx, {24'h0, w_buf[idx]});
    end
    $display("[LOAD] FC2 weights done.");
endtask

// ─── 加载 FC2 Bias（32-bit per entry，共 10 entries）─────────────────────

task automatic load_fc2_bias(input string hex_file);
    logic [31:0] b_buf [0:9];
    int idx;

    $display("[LOAD] Loading FC2 bias from: %s", hex_file);
    $readmemh(hex_file, b_buf);

    for (idx = 0; idx < FC2B_WORDS; idx++) begin
        bus_write(FC2B_BASE + idx, b_buf[idx]);
    end
    $display("[LOAD] FC2 bias done.");
endtask

// ─── 加载量化参数（4 个 32-bit word）─────────────────────────────────────

task automatic load_quant_params(input string hex_file);
    logic [31:0] q_buf [0:3];
    int idx;

    $display("[LOAD] Loading quant params from: %s", hex_file);
    $readmemh(hex_file, q_buf);

    for (idx = 0; idx < QUANT_WORDS; idx++) begin
        bus_write(QUANT_BASE + idx, q_buf[idx]);
    end
    $display("[LOAD] Quant params done.");
endtask

// ─── 加载 Sample ROM（128-bit tile packed hex，共 980 tiles × 4 words）───

task automatic load_samples(input string hex_file);
    logic [127:0] tile_buf [0:979];
    logic  [31:0] word;
    int tile_idx, word_idx;

    $display("[LOAD] Loading samples from: %s", hex_file);
    $readmemh(hex_file, tile_buf);

    for (tile_idx = 0; tile_idx < 980; tile_idx++) begin
        for (word_idx = 0; word_idx < SAMP_WORDS_PER_TILE; word_idx++) begin
            word = tile_buf[tile_idx][word_idx*32 +: 32];
            bus_write(SAMP_BASE + tile_idx * SAMP_WORDS_PER_TILE + word_idx, word);
        end
    end
    $display("[LOAD] Samples done.");
endtask

// ─── 一键加载全部数据 ─────────────────────────────────────────────────────

task automatic load_all_data(
    input string sample_file,
    input string fc1w_file,
    input string fc1b_file,
    input string quant_file,
    input string fc2w_file,
    input string fc2b_file
);
    $display("[LOAD] === Starting data load ===");
    load_fc1_weight(fc1w_file);
    load_fc1_bias(fc1b_file);
    load_fc2_weight(fc2w_file);
    load_fc2_bias(fc2b_file);
    load_quant_params(quant_file);
    load_samples(sample_file);
    $display("[LOAD] === All data loaded ===");

    // 拉高 load_done
    @(posedge clk);
    load_done <= 1'b1;
    $display("[LOAD] load_done asserted, inference enabled.");
endtask
