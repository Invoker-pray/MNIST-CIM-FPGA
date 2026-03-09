
module mnist_cim_accel_ip #(
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

    input  logic        csr_we,
    input  logic        csr_re,
    input  logic [ 7:0] csr_addr,
    input  logic [31:0] csr_wdata,
    output logic [31:0] csr_rdata,

    output logic irq_done,

    output logic [$clog2(mnist_cim_pkg::FC2_OUT_DIM)-1:0] pred_class_dbg,
    output logic                                          busy_dbg,
    output logic                                          done_dbg
);
  import mnist_cim_pkg::*;

  localparam [7:0] ADDR_CTRL = 8'h00;
  localparam [7:0] ADDR_STATUS = 8'h04;
  localparam [7:0] ADDR_SAMPLE_ID = 8'h08;
  localparam [7:0] ADDR_PRED = 8'h0C;
  localparam [7:0] ADDR_LOGIT0 = 8'h10;

  logic [$clog2(N_SAMPLES)-1:0] sample_id_reg;
  logic start_pulse;
  logic done_sticky;

  logic core_busy, core_done;
  logic signed [OUTPUT_WIDTH-1:0] logits_all[0:FC2_OUT_DIM-1];
  logic [$clog2(FC2_OUT_DIM)-1:0] pred_class;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      sample_id_reg <= '0;
      start_pulse   <= 1'b0;
      done_sticky   <= 1'b0;
    end else begin
      start_pulse <= 1'b0;

      if (csr_we && csr_addr == ADDR_SAMPLE_ID) sample_id_reg <= csr_wdata[$clog2(N_SAMPLES)-1:0];

      if (csr_we && csr_addr == ADDR_CTRL && csr_wdata[0] && !core_busy) start_pulse <= 1'b1;

      if (csr_we && csr_addr == ADDR_CTRL && csr_wdata[1]) done_sticky <= 1'b0;
      else if (core_done) done_sticky <= 1'b1;
    end
  end

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
      .sample_id(sample_id_reg),
      .busy(core_busy),
      .done(core_done),
      .logits_all(logits_all),
      .pred_class(pred_class)
  );

  always_comb begin
    csr_rdata = 32'd0;
    case (csr_addr)
      ADDR_STATUS:    csr_rdata = {30'd0, done_sticky, core_busy};
      ADDR_SAMPLE_ID: csr_rdata = sample_id_reg;
      ADDR_PRED:      csr_rdata = pred_class;
      8'h10:          csr_rdata = $signed(logits_all[0]);
      8'h14:          csr_rdata = $signed(logits_all[1]);
      8'h18:          csr_rdata = $signed(logits_all[2]);
      8'h1C:          csr_rdata = $signed(logits_all[3]);
      8'h20:          csr_rdata = $signed(logits_all[4]);
      8'h24:          csr_rdata = $signed(logits_all[5]);
      8'h28:          csr_rdata = $signed(logits_all[6]);
      8'h2C:          csr_rdata = $signed(logits_all[7]);
      8'h30:          csr_rdata = $signed(logits_all[8]);
      8'h34:          csr_rdata = $signed(logits_all[9]);
      default:        csr_rdata = 32'd0;
    endcase
  end

  assign irq_done       = done_sticky;
  assign pred_class_dbg = pred_class;
  assign busy_dbg       = core_busy;
  assign done_dbg       = done_sticky;

endmodule
