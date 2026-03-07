

package mnist_cim_pkg;

  // -----------------------------
  // Network dimensions
  // -----------------------------
  parameter int INPUT_DIM = 784;
  parameter int HIDDEN_DIM = 128;
  parameter int OUTPUT_DIM = 10;

  // -----------------------------
  // Tile configuration
  // -----------------------------
  parameter int TILE_INPUT_SIZE = 16;
  parameter int TILE_OUTPUT_SIZE = 16;

  // -----------------------------
  // Block numbers
  // -----------------------------
  parameter int N_INPUT_BLOCKS = INPUT_DIM / TILE_INPUT_SIZE;  // 49
  parameter int N_OUTPUT_BLOCKS = HIDDEN_DIM / TILE_OUTPUT_SIZE;  // 8

  // -----------------------------
  // Data widths
  // -----------------------------
  parameter int INPUT_WIDTH = 8;
  parameter int WEIGHT_WIDTH = 8;
  parameter int BIAS_WIDTH = 32;
  parameter int PSUM_WIDTH = 32;
  parameter int OUTPUT_WIDTH = 8;

  // 因为 x_q - zp 后范围可到 0~255，建议预留 9 bit
  parameter int X_EFF_WIDTH = 9;

  // -----------------------------
  // Quantization zero-points
  // -----------------------------
  parameter int signed INPUT_ZERO_POINT = -128;
  parameter int signed FC1_WEIGHT_ZERO_POINT = 0;
  parameter int signed FC1_OUTPUT_ZERO_POINT = 0;

  // -----------------------------
  // fc1&fc2 dimension configuration
  // -----------------------------

  parameter int FC1_IN_DIM = INPUT_DIM;
  parameter int FC1_OUT_DIM = HIDDEN_DIM;

  parameter int FC2_IN_DIM = HIDDEN_DIM;
  parameter int FC2_OUT_DIM = OUTPUT_DIM;


  // -----------------------------
  // fc1&fc2 address configuration
  // -----------------------------

  parameter int FC1_WEIGHT_DEPTH = HIDDEN_DIM * INPUT_DIM;
  parameter int FC1_BIAS_DEPTH = HIDDEN_DIM;
  parameter int FC2_WEIGHT_DEPTH = OUTPUT_DIM * HIDDEN_DIM;
  parameter int FC2_BIAS_DEPTH = OUTPUT_DIM;



  // -----------------------------
  // fc1 to fc2 
  // -----------------------------


  parameter int FC2_IN_DIM = HIDDEN_DIM;
  parameter int FC2_OUT_DIM = 10;
  parameter int FC2_WEIGHT_DEPTH = FC2_OUT_DIM * FC2_IN_DIM;
  parameter int FC2_BIAS_DEPTH = FC2_OUT_DIM;


endpackage



