

module mnist_cim_demo_a_top #(
    parameter int CLK_HZ = 125_000_000,
    parameter int BAUD = 115200,
    parameter int N_SAMPLES = 20,
    parameter int BTN_DEBOUNCE_MS = 20,
    parameter bit SIM_BYPASS_DEBOUNCE = 1'b0,

    parameter string DEFAULT_SAMPLE_HEX_FILE = "../data/samples/mnist_samples_route_b_output_2.hex",
    parameter string DEFAULT_FC1_WEIGHT_HEX_FILE = "../data/weights/fc1_weight_int8.hex",
    parameter string DEFAULT_FC1_BIAS_HEX_FILE = "../data/weights/fc1_bias_int32.hex",
    parameter string DEFAULT_QUANT_PARAM_FILE = "../data/quant/quant_params.hex",
    parameter string DEFAULT_FC2_WEIGHT_HEX_FILE = "../data/weights/fc2_weight_int8.hex",
    parameter string DEFAULT_FC2_BIAS_HEX_FILE = "../data/weights/fc2_bias_int32.hex"
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

  //  logic btn_start_db_raw;
  logic btn_start_db;
  logic start_pulse;

  logic busy, done;
  logic done_d;

  logic busy_seen;
  logic done_latched;

  logic signed [OUTPUT_WIDTH-1:0] logits_all[0:FC2_OUT_DIM-1];
  logic [$clog2(FC2_OUT_DIM)-1:0] pred_class;

  //--------------------------------------------------------------------------
  // Button conditioning: debounce + onepulse
  //--------------------------------------------------------------------------


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


  //--------------------------------------------------------------------------
  // done rising-edge detect for UART trigger
  //--------------------------------------------------------------------------

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      done_d <= 1'b0;
    end else begin
      done_d <= done;
    end
  end

  //--------------------------------------------------------------------------
  // done
  //--------------------------------------------------------------------------

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      done_latched <= 1'b0;
    end else if (start_pulse) begin
      done_latched <= 1'b0;
    end else if (done) begin
      done_latched <= 1'b1;
    end
  end



  //--------------------------------------------------------------------------
  // busy seen 
  //--------------------------------------------------------------------------

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      busy_seen <= 1'b0;
    end else if (start_pulse) begin
      busy_seen <= 1'b0;
    end else if (busy) begin
      busy_seen <= 1'b1;
    end
  end
  //--------------------------------------------------------------------------
  // Core inference block
  //--------------------------------------------------------------------------

  mnist_inference_core_board #(
      .N_SAMPLES(N_SAMPLES),
      .DEFAULT_SAMPLE_HEX_FILE(DEFAULT_SAMPLE_HEX_FILE),
      .DEFAULT_FC1_WEIGHT_HEX_FILE(DEFAULT_FC1_WEIGHT_HEX_FILE),
      .DEFAULT_FC1_BIAS_HEX_FILE(DEFAULT_FC1_BIAS_HEX_FILE),
      .DEFAULT_QUANT_PARAM_FILE(DEFAULT_QUANT_PARAM_FILE),
      .DEFAULT_FC2_WEIGHT_HEX_FILE(DEFAULT_FC2_WEIGHT_HEX_FILE),
      .DEFAULT_FC2_BIAS_HEX_FILE(DEFAULT_FC2_BIAS_HEX_FILE)
  ) u_core (
      .clk       (clk),
      .rst_n     (rst_n),
      .start     (start_pulse),
      .sample_id (sample_sel),
      .busy      (busy),
      .done      (done),
      .logits_all(logits_all),
      .pred_class(pred_class)
  );

  //--------------------------------------------------------------------------
  // UART sender
  //--------------------------------------------------------------------------

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

  //--------------------------------------------------------------------------
  // LEDs
  //--------------------------------------------------------------------------

  assign led_busy = busy_seen;
  //assign led_done = done;
  assign led_done = done_latched;

endmodule
