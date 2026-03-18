
module onepulse (
    input  logic clk,
    input  logic rst_n,
    input  logic din,
    output logic pulse
);

  logic din_d;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      din_d <= 1'b0;
      pulse <= 1'b0;
    end else begin
      pulse <= din & ~din_d;
      din_d <= din;
    end
  end

endmodule
