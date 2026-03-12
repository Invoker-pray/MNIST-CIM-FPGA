// mnist_sample_rom.sv
// 移除 $readmemh，改为运行时32-bit串行写入接口
// 每4个word完成一个128-bit样本tile的写入

module mnist_sample_rom #(
    parameter int N_SAMPLES = 20
) (
    input logic clk,
    input logic rst_n,

    // Read port (unchanged)
    input logic [$clog2(N_SAMPLES)-1:0]                              sample_id,
    input logic [$clog2(mnist_cim_pkg::N_INPUT_BLOCKS)-1:0]          ib,

    output logic signed [mnist_cim_pkg::INPUT_WIDTH-1:0]
        x_tile [0:mnist_cim_pkg::TILE_INPUT_SIZE-1],
    output logic [mnist_cim_pkg::X_EFF_WIDTH-1:0]
        x_eff_tile [0:mnist_cim_pkg::TILE_INPUT_SIZE-1],

    // Write port: 32-bit serial streaming
    // 每 4 个 word 完成一个 tile (128-bit = 16×8-bit) 的写入
    // 写入顺序：sample 0 tile 0, sample 0 tile 1, ..., sample 19 tile 48
    input logic        wr_en,
    input logic [31:0] wr_data
);
  import mnist_cim_pkg::*;

  localparam int TILE_BITS      = TILE_INPUT_SIZE * INPUT_WIDTH;    // 128
  localparam int WORDS_PER_TILE = TILE_BITS / 32;                   // 4
  localparam int TILE_DEPTH     = N_SAMPLES * N_INPUT_BLOCKS;       // 980

  // ── BRAM ──────────────────────────────────────────────────────────────────
  (* ram_style = "block" *)
  logic [TILE_BITS-1:0] sample_mem [0:TILE_DEPTH-1];

  // ── Read path ─────────────────────────────────────────────────────────────
  logic [TILE_BITS-1:0]              tile_word_r;
  logic [$clog2(TILE_DEPTH)-1:0]     tile_addr;

  assign tile_addr = sample_id * N_INPUT_BLOCKS + ib;

  always_ff @(posedge clk) begin
    tile_word_r <= sample_mem[tile_addr];
  end

  integer i;
  integer x_eff_tmp;
  always_comb begin
    for (i = 0; i < TILE_INPUT_SIZE; i = i + 1) begin
      x_tile[i]    = tile_word_r[i * INPUT_WIDTH +: INPUT_WIDTH];
      x_eff_tmp    = $signed(x_tile[i]) - INPUT_ZERO_POINT;
      if      (x_eff_tmp < 0)                         x_eff_tile[i] = '0;
      else if (x_eff_tmp > ((1 << X_EFF_WIDTH) - 1))  x_eff_tile[i] = {X_EFF_WIDTH{1'b1}};
      else                                             x_eff_tile[i] = x_eff_tmp[X_EFF_WIDTH-1:0];
    end
  end

  // ── Write path ────────────────────────────────────────────────────────────
  logic [TILE_BITS-1:0]              wr_pack_buf;
  logic [$clog2(WORDS_PER_TILE)-1:0] wr_word_cnt;
  logic [$clog2(TILE_DEPTH)-1:0]     wr_tile_cnt;

  logic                              wr_commit;
  logic [$clog2(TILE_DEPTH)-1:0]    wr_commit_idx;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wr_word_cnt  <= '0;
      wr_tile_cnt  <= '0;
      wr_commit    <= 1'b0;
      wr_pack_buf  <= '0;
    end else begin
      wr_commit <= 1'b0;

      if (wr_en) begin
        wr_pack_buf[wr_word_cnt * 32 +: 32] <= wr_data;

        if (wr_word_cnt == WORDS_PER_TILE - 1) begin
          wr_commit     <= 1'b1;
          wr_commit_idx <= wr_tile_cnt;
          wr_tile_cnt   <= (wr_tile_cnt == TILE_DEPTH - 1) ? '0 : wr_tile_cnt + 1;
          wr_word_cnt   <= '0;
        end else begin
          wr_word_cnt <= wr_word_cnt + 1;
        end
      end

      if (wr_commit) begin
        sample_mem[wr_commit_idx] <= wr_pack_buf;
      end
    end
  end

endmodule
