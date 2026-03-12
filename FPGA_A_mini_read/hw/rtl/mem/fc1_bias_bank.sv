// fc1_bias_bank.sv
// 移除 $readmemh，改为运行时32-bit串行写入接口
// 每16个word完成一个 bias block (512-bit) 的写入

module fc1_bias_bank (
    input logic clk,
    input logic rst_n,

    // Read port (unchanged)
    input logic [$clog2(mnist_cim_pkg::N_OUTPUT_BLOCKS)-1:0] ob,

    output logic signed [mnist_cim_pkg::BIAS_WIDTH-1:0]
        bias_block [0:mnist_cim_pkg::TILE_OUTPUT_SIZE-1],

    // Write port: 32-bit serial streaming
    // 每 TILE_OUTPUT_SIZE(16) 个 word 完成一个 block 的写入
    input logic        wr_en,
    input logic [31:0] wr_data
);
  import mnist_cim_pkg::*;

  localparam int BLOCK_BITS      = TILE_OUTPUT_SIZE * BIAS_WIDTH; // 512
  localparam int WORDS_PER_BLOCK = BLOCK_BITS / 32;               // 16
  localparam int BLOCK_DEPTH     = N_OUTPUT_BLOCKS;               // 1

  // ── BRAM ──────────────────────────────────────────────────────────────────
  (* ram_style = "block" *)
  logic [BLOCK_BITS-1:0] bias_mem [0:BLOCK_DEPTH-1];

  // ── Read path ─────────────────────────────────────────────────────────────
  logic [BLOCK_BITS-1:0] bias_word_r;

  always_ff @(posedge clk) begin
    bias_word_r <= bias_mem[ob];
  end

  integer i;
  always_comb begin
    for (i = 0; i < TILE_OUTPUT_SIZE; i = i + 1)
      bias_block[i] = bias_word_r[i * BIAS_WIDTH +: BIAS_WIDTH];
  end

  // ── Write path ────────────────────────────────────────────────────────────
  logic [BLOCK_BITS-1:0]              wr_pack_buf;
  logic [$clog2(WORDS_PER_BLOCK)-1:0] wr_word_cnt;
  logic [$clog2(BLOCK_DEPTH)-1:0]     wr_block_cnt;

  logic                               wr_commit;
  logic [$clog2(BLOCK_DEPTH)-1:0]    wr_commit_idx;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wr_word_cnt  <= '0;
      wr_block_cnt <= '0;
      wr_commit    <= 1'b0;
      wr_pack_buf  <= '0;
    end else begin
      wr_commit <= 1'b0;

      if (wr_en) begin
        wr_pack_buf[wr_word_cnt * 32 +: 32] <= wr_data;

        if (wr_word_cnt == WORDS_PER_BLOCK - 1) begin
          wr_commit     <= 1'b1;
          wr_commit_idx <= wr_block_cnt;
          wr_block_cnt  <= (wr_block_cnt == BLOCK_DEPTH - 1) ? '0 : wr_block_cnt + 1;
          wr_word_cnt   <= '0;
        end else begin
          wr_word_cnt <= wr_word_cnt + 1;
        end
      end

      if (wr_commit) begin
        bias_mem[wr_commit_idx] <= wr_pack_buf;
      end
    end
  end

endmodule
