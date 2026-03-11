
module uart_pred_sender #(
    parameter int CLK_HZ = 50_000_000,
    parameter int BAUD   = 115200
) (
    input logic clk,
    input logic rst_n,
    input logic trigger,
    input logic [3:0] pred_class,

    output logic uart_tx
);
  typedef enum logic [1:0] {
    S_IDLE = 2'd0,
    S_DIGIT = 2'd1,
    S_CR = 2'd2,
    S_LF = 2'd3
  } state_t;

  state_t state, state_n;
  logic tx_start;
  logic [7:0] tx_data;
  logic tx_busy;

  uart_tx #(
      .CLK_HZ(CLK_HZ),
      .BAUD  (BAUD)
  ) u_uart_tx (
      .clk(clk),
      .rst_n(rst_n),
      .start(tx_start),
      .data_in(tx_data),
      .tx(uart_tx),
      .busy(tx_busy)
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) state <= S_IDLE;
    else state <= state_n;
  end

  always_comb begin
    state_n  = state;
    tx_start = 1'b0;
    tx_data  = 8'h00;

    case (state)
      S_IDLE: begin
        if (trigger && !tx_busy) begin
          tx_start = 1'b1;
          tx_data  = 8'd48 + pred_class;
          state_n  = S_DIGIT;
        end
      end

      S_DIGIT: begin
        if (!tx_busy) begin
          tx_start = 1'b1;
          tx_data  = 8'h0D;
          state_n  = S_CR;
        end
      end

      S_CR: begin
        if (!tx_busy) begin
          tx_start = 1'b1;
          tx_data  = 8'h0A;
          state_n  = S_LF;
        end
      end

      S_LF: begin
        if (!tx_busy) state_n = S_IDLE;
      end
    endcase
  end

endmodule
