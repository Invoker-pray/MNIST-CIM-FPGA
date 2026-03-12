// fc2_bias_bank.sv
// 移除 $readmemh，改为运行时32-bit直接写入

module fc2_bias_bank (
    input logic clk,
    input logic rst_n,

    // Read port (unchanged)
    input  logic [$clog2(mnist_cim_pkg::FC2_OUT_DIM)-1:0] addr,
    output logic signed [mnist_cim_pkg::BIAS_WIDTH-1:0]   bias_data,

    // Write port: 每次写入一个32-bit bias，顺序写入
    input logic        wr_en,
    input logic [31:0] wr_data
);
  import mnist_cim_pkg::*;

  (* ram_style = "block" *)
  logic signed [BIAS_WIDTH-1:0] bias_mem [0:FC2_OUT_DIM-1];

  // ── Read path ─────────────────────────────────────────────────────────────
  always_ff @(posedge clk) begin
    bias_data <= bias_mem[addr];
  end

  // ── Write path ────────────────────────────────────────────────────────────
  logic [$clog2(FC2_OUT_DIM)-1:0] wr_addr_cnt;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wr_addr_cnt <= '0;
    end else begin
      if (wr_en) begin
        bias_mem[wr_addr_cnt] <= wr_data;
        wr_addr_cnt <= (wr_addr_cnt == FC2_OUT_DIM - 1) ? '0 : wr_addr_cnt + 1;
      end
    end
  end

endmodule
