

module mnist_sample_rom #(
    parameter int N_SAMPLES = 20,
    parameter string DEFAULT_SAMPLE_HEX_FILE = "../data/samples/mnist_samples_route_b_output_2.hex"
) (
    input logic [$clog2(N_SAMPLES)-1:0] sample_id,
    input logic [$clog2(mnist_cim_pkg::N_INPUT_BLOCKS)-1:0] ib,

    output logic signed [mnist_cim_pkg::INPUT_WIDTH-1:0] x_tile[0:mnist_cim_pkg::TILE_INPUT_SIZE-1],

    output logic [mnist_cim_pkg::X_EFF_WIDTH-1:0] x_eff_tile[0:mnist_cim_pkg::TILE_INPUT_SIZE-1]
);
  import mnist_cim_pkg::*;

  (* rom_style = "block" *)
  logic signed [INPUT_WIDTH-1:0] sample_mem[0:N_SAMPLES*INPUT_DIM-1];

  integer i, addr;
  integer x_eff_tmp;

  initial begin
    $readmemh(DEFAULT_SAMPLE_HEX_FILE, sample_mem);
  end

  always_comb begin
    for (i = 0; i < TILE_INPUT_SIZE; i = i + 1) begin
      addr = sample_id * INPUT_DIM + ib * TILE_INPUT_SIZE + i;

      x_tile[i] = sample_mem[addr];

      x_eff_tmp = sample_mem[addr] - INPUT_ZERO_POINT;
      if (x_eff_tmp < 0) x_eff_tile[i] = '0;
      else if (x_eff_tmp > ((1 << X_EFF_WIDTH) - 1)) x_eff_tile[i] = {X_EFF_WIDTH{1'b1}};
      else x_eff_tile[i] = x_eff_tmp[X_EFF_WIDTH-1:0];
    end
  end

endmodule
