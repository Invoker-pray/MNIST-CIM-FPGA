// fc2_weight_bank.sv
// 移除 $readmemh，改为运行时写入接口
// FC2权重为8-bit，每次32-bit写入的低8位为有效数据，顺序写入

module fc2_weight_bank (
    input logic clk,
    input logic rst_n,

    // Read port (unchanged)
    input  logic [$clog2(mnist_cim_pkg::FC2_WEIGHT_DEPTH)-1:0] addr,
    output logic signed [mnist_cim_pkg::WEIGHT_WIDTH-1:0]      w_data,

    // Write port: 每次写入一个8-bit权重（放在wr_data低8位），顺序写入
    input logic        wr_en,
    input logic [31:0] wr_data   // [7:0] 有效
);
  import mnist_cim_pkg::*;

  (* ram_style = "block" *)
  logic signed [WEIGHT_WIDTH-1:0] weight_mem [0:FC2_WEIGHT_DEPTH-1];

  // ── Read path ─────────────────────────────────────────────────────────────
  always_ff @(posedge clk) begin
    w_data <= weight_mem[addr];
  end

  // ── Write path ────────────────────────────────────────────────────────────
  logic [$clog2(FC2_WEIGHT_DEPTH)-1:0] wr_addr_cnt;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wr_addr_cnt <= '0;
    end else begin
      if (wr_en) begin
        weight_mem[wr_addr_cnt] <= wr_data[WEIGHT_WIDTH-1:0];
        wr_addr_cnt <= (wr_addr_cnt == FC2_WEIGHT_DEPTH - 1) ? '0 : wr_addr_cnt + 1;
      end
    end
  end

endmodule
