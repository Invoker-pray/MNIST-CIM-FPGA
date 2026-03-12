// quantize_param_bank.sv
// 移除 $readmemh，改为4个32-bit寄存器，运行时顺序写入

module quantize_param_bank (
    input logic clk,
    input logic rst_n,

    // Read port (unchanged, combinational)
    output logic [31:0] fc1_requant_mult,
    output logic [31:0] fc1_requant_shift,
    output logic [31:0] fc2_requant_mult,
    output logic [31:0] fc2_requant_shift,

    // Write port: 顺序写入4个参数
    // 顺序: [0]=fc1_mult, [1]=fc1_shift, [2]=fc2_mult, [3]=fc2_shift
    input logic        wr_en,
    input logic [31:0] wr_data
);

  logic [31:0] param_reg [0:3];
  logic [1:0]  wr_addr_cnt;

  // ── Read (combinational) ──────────────────────────────────────────────────
  assign fc1_requant_mult  = param_reg[0];
  assign fc1_requant_shift = param_reg[1];
  assign fc2_requant_mult  = param_reg[2];
  assign fc2_requant_shift = param_reg[3];

  // ── Write ─────────────────────────────────────────────────────────────────
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wr_addr_cnt <= '0;
      param_reg[0] <= '0;
      param_reg[1] <= '0;
      param_reg[2] <= '0;
      param_reg[3] <= '0;
    end else begin
      if (wr_en) begin
        param_reg[wr_addr_cnt] <= wr_data;
        wr_addr_cnt <= wr_addr_cnt + 1; // 自然溢出回0
      end
    end
  end

endmodule
