// fc1_weight_bank.sv
// 移除 $readmemh，改为运行时32-bit串行写入接口
// 写入方式：顺序流式写入，内部自动将32-bit word打包成2048-bit tile后写入BRAM

module fc1_weight_bank (
    input logic clk,
    input logic rst_n,

    // Read port (unchanged)
    input logic [$clog2(mnist_cim_pkg::N_OUTPUT_BLOCKS)-1:0] ob,
    input logic [$clog2(mnist_cim_pkg::N_INPUT_BLOCKS)-1:0]  ib,

    output logic signed [mnist_cim_pkg::WEIGHT_WIDTH-1:0]
        w_tile [0:mnist_cim_pkg::TILE_OUTPUT_SIZE-1]
               [0:mnist_cim_pkg::TILE_INPUT_SIZE-1],

    // Write port: 32-bit serial streaming
    // 调用方按顺序逐字写入，每64个word完成一个tile的写入
    input logic        wr_en,
    input logic [31:0] wr_data
);
  import mnist_cim_pkg::*;

  localparam int TILE_ELEMS     = TILE_OUTPUT_SIZE * TILE_INPUT_SIZE; // 256
  localparam int TILE_BITS      = TILE_ELEMS * WEIGHT_WIDTH;          // 2048
  localparam int WORDS_PER_TILE = TILE_BITS / 32;                     // 64
  localparam int TILE_DEPTH     = N_OUTPUT_BLOCKS * N_INPUT_BLOCKS;   // 49

  // ── BRAM ──────────────────────────────────────────────────────────────────
  (* ram_style = "block" *)
  logic [TILE_BITS-1:0] tile_mem [0:TILE_DEPTH-1];

  // ── Read path ─────────────────────────────────────────────────────────────
  logic [TILE_BITS-1:0]              tile_word_r;
  logic [$clog2(TILE_DEPTH)-1:0]     tile_addr;

  assign tile_addr = ob * N_INPUT_BLOCKS + ib;

  always_ff @(posedge clk) begin
    tile_word_r <= tile_mem[tile_addr];
  end

  integer tr, tc, flat_idx;
  always_comb begin
    for (tr = 0; tr < TILE_OUTPUT_SIZE; tr = tr + 1)
      for (tc = 0; tc < TILE_INPUT_SIZE; tc = tc + 1) begin
        flat_idx = tr * TILE_INPUT_SIZE + tc;
        w_tile[tr][tc] = tile_word_r[flat_idx * WEIGHT_WIDTH +: WEIGHT_WIDTH];
      end
  end

  // ── Write path ────────────────────────────────────────────────────────────
  // 用2048-bit打包缓冲区，每收集满64个32-bit word后一次写入BRAM
  logic [TILE_BITS-1:0]             wr_pack_buf;
  logic [$clog2(WORDS_PER_TILE)-1:0] wr_word_cnt;   // 0 .. 63
  logic [$clog2(TILE_DEPTH)-1:0]    wr_tile_cnt;    // 0 .. 48

  // 延迟一拍提交：保证 wr_pack_buf 已完整更新后再写 tile_mem
  logic                             wr_commit;
  logic [$clog2(TILE_DEPTH)-1:0]   wr_commit_idx;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wr_word_cnt  <= '0;
      wr_tile_cnt  <= '0;
      wr_commit    <= 1'b0;
      wr_pack_buf  <= '0;
    end else begin
      // 默认清除提交标志
      wr_commit <= 1'b0;

      if (wr_en) begin
        // 将当前 word 写入打包缓冲区对应位置
        wr_pack_buf[wr_word_cnt * 32 +: 32] <= wr_data;

        if (wr_word_cnt == WORDS_PER_TILE - 1) begin
          // 最后一个 word：触发下一拍提交
          wr_commit     <= 1'b1;
          wr_commit_idx <= wr_tile_cnt;
          wr_tile_cnt   <= (wr_tile_cnt == TILE_DEPTH - 1) ? '0 : wr_tile_cnt + 1;
          wr_word_cnt   <= '0;
        end else begin
          wr_word_cnt <= wr_word_cnt + 1;
        end
      end

      // 将完整的打包缓冲区写入 BRAM
      // 此时 wr_pack_buf 已包含上一拍写入的最后一个 word
      if (wr_commit) begin
        tile_mem[wr_commit_idx] <= wr_pack_buf;
      end
    end
  end

endmodule
