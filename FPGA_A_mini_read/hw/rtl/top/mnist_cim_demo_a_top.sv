// mnist_cim_demo_a_top.sv
// 新增统一写总线和 load_done 接口
// load_done=0 时禁止启动推理，保证数据加载完成后才能运行

module mnist_cim_demo_a_top #(
    parameter int CLK_HZ              = 125_000_000,
    parameter int BAUD                = 115200,
    parameter int N_SAMPLES           = 20,
    parameter int BTN_DEBOUNCE_MS     = 20,
    parameter bit SIM_BYPASS_DEBOUNCE = 1'b0
) (
    input logic clk,
    input logic rst_n,

    // ── 推理触发 ──────────────────────────────────────────────────────────
    input logic btn_start,
    input logic [$clog2(N_SAMPLES)-1:0] sample_sel,

    // ── UART 输出 ─────────────────────────────────────────────────────────
    output logic uart_tx,
    output logic led_busy,
    output logic led_done,

    // ── 外部数据写总线 ────────────────────────────────────────────────────
    // 上电后先通过此接口将所有权重/偏置/样本/量化参数写入片上 BRAM
    // 完成后拉高 load_done，之后才可触发推理
    input logic        mem_we,       // 写使能（每拍写一个 32-bit word）
    input logic [15:0] mem_waddr,    // 32-bit word 地址（见 inference_core 中的地址映射）
    input logic [31:0] mem_wdata,    // 写数据

    // load_done=1 表示所有数据已加载完毕，允许推理
    // load_done=0 时 btn_start 无效
    input logic load_done
);
  import mnist_cim_pkg::*;

  logic btn_start_db;
  logic start_pulse;
  logic start_gated;     // 受 load_done 保护的 start 信号

  logic busy, done;
  logic done_d;
  logic done_latched;

  logic signed [OUTPUT_WIDTH-1:0]        logits_all [0:FC2_OUT_DIM-1];
  logic [$clog2(FC2_OUT_DIM)-1:0]        pred_class;

  // ── Button conditioning ───────────────────────────────────────────────────

  generate
    if (SIM_BYPASS_DEBOUNCE) begin : GEN_BYPASS_DEBOUNCE
      assign btn_start_db = btn_start;
`ifndef SYNTHESIS
      initial $display("[TOP] SIM_BYPASS_DEBOUNCE=1, bypass debounce");
`endif
    end else begin : GEN_REAL_DEBOUNCE
`ifndef SYNTHESIS
      initial $display("[TOP] SIM_BYPASS_DEBOUNCE=0, use real debounce");
`endif
      debounce #(
          .CLK_HZ(CLK_HZ),
          .DEBOUNCE_MS(BTN_DEBOUNCE_MS)
      ) u_btn_start_debounce (
          .clk  (clk),
          .rst_n(rst_n),
          .din  (btn_start),
          .dout (btn_start_db)
      );
    end
  endgenerate

  onepulse u_btn_start_onepulse (
      .clk  (clk),
      .rst_n(rst_n),
      .din  (btn_start_db),
      .pulse(start_pulse)
  );

  // 只有 load_done=1 时才允许触发推理
  assign start_gated = start_pulse & load_done;

  // ── done edge detect for UART ─────────────────────────────────────────────

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) done_d <= 1'b0;
    else        done_d <= done;
  end

  // ── done latch for LED ────────────────────────────────────────────────────

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)            done_latched <= 1'b0;
    else if (start_gated)  done_latched <= 1'b0;
    else if (done)         done_latched <= 1'b1;
  end

  // ── Core inference block ──────────────────────────────────────────────────

  mnist_inference_core_board #(
      .N_SAMPLES(N_SAMPLES)
  ) u_core (
      .clk       (clk),
      .rst_n     (rst_n),
      .start     (start_gated),
      .sample_id (sample_sel),
      .busy      (busy),
      .done      (done),
      .logits_all(logits_all),
      .pred_class(pred_class),
      .mem_we    (mem_we),
      .mem_waddr (mem_waddr),
      .mem_wdata (mem_wdata)
  );

  // ── UART sender ───────────────────────────────────────────────────────────

  uart_pred_sender #(
      .CLK_HZ(CLK_HZ),
      .BAUD  (BAUD)
  ) u_uart_pred_sender (
      .clk       (clk),
      .rst_n     (rst_n),
      .trigger   (done & ~done_d),
      .pred_class(pred_class),
      .uart_tx   (uart_tx)
  );

  // ── LEDs ──────────────────────────────────────────────────────────────────

  assign led_busy = busy;
  assign led_done = done_latched;

endmodule
