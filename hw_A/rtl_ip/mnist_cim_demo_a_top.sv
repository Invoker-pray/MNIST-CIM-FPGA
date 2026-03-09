

module mnist_cim_demo_a_top #(
    parameter int CLK_HZ = 50_000_000,
    parameter int BAUD = 115200,
    parameter int N_SAMPLES = 20,
    parameter string DEFAULT_SAMPLE_HEX_FILE = "../data/mnist_samples_route_b_output_2.hex",
    parameter string DEFAULT_FC1_WEIGHT_HEX_FILE = "../route_b_output_2/fc1_weight_int8.hex",
    parameter string DEFAULT_FC1_BIAS_HEX_FILE = "../route_b_output_2/fc1_bias_int32.hex",
    parameter string DEFAULT_QUANT_PARAM_FILE = "../route_b_output_2/quant_params.hex",
    parameter string DEFAULT_FC2_WEIGHT_HEX_FILE = "../route_b_output_2/fc2_weight_int8.hex",
    parameter string DEFAULT_FC2_BIAS_HEX_FILE = "../route_b_output_2/fc2_bias_int32.hex"
) (
    input logic clk,
    input logic rst_n,
    input logic btn_start,
    input logic [$clog2(N_SAMPLES)-1:0] sample_sel,

    output logic uart_tx,
    output logic led_busy,
    output logic led_done
);
  import mnist_cim_pkg::*;

  logic btn_d;
  logic start_pulse;
  logic busy, done;
  logic done_d;

  logic signed [OUTPUT_WIDTH-1:0] logits_all[0:FC2_OUT_DIM-1];
  logic [$clog2(FC2_OUT_DIM)-1:0] pred_class;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      btn_d  <= 1'b0;
      done_d <= 1'b0;
    end else begin
      btn_d  <= btn_start;
      done_d <= done;
    end
  end

  assign start_pulse = btn_start & ~btn_d;

  mnist_inference_core_board #(
      .N_SAMPLES(N_SAMPLES),
      .DEFAULT_SAMPLE_HEX_FILE(DEFAULT_SAMPLE_HEX_FILE),
      .DEFAULT_FC1_WEIGHT_HEX_FILE(DEFAULT_FC1_WEIGHT_HEX_FILE),
      .DEFAULT_FC1_BIAS_HEX_FILE(DEFAULT_FC1_BIAS_HEX_FILE),
      .DEFAULT_QUANT_PARAM_FILE(DEFAULT_QUANT_PARAM_FILE),
      .DEFAULT_FC2_WEIGHT_HEX_FILE(DEFAULT_FC2_WEIGHT_HEX_FILE),
      .DEFAULT_FC2_BIAS_HEX_FILE(DEFAULT_FC2_BIAS_HEX_FILE)
  ) u_core (
      .clk(clk),
      .rst_n(rst_n),
      .start(start_pulse),
      .sample_id(sample_sel),
      .busy(busy),
      .done(done),
      .logits_all(logits_all),
      .pred_class(pred_class)
  );

  uart_pred_sender #(
      .CLK_HZ(CLK_HZ),
      .BAUD  (BAUD)
  ) u_uart_pred_sender (
      .clk(clk),
      .rst_n(rst_n),
      .trigger(done & ~done_d),
      .pred_class(pred_class),
      .uart_tx(uart_tx)
  );

  assign led_busy = busy;
  assign led_done = done;

endmodule
