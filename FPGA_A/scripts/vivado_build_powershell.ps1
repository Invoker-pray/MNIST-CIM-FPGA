$ROOT_DIR="D:\MNIST-CIM-FPGA-master\MNIST-CIM-FPGA-master\FPGA_A"
$VIVADO_DIR="$ROOT_DIR/vivado"

& "D:/vivado/Vivado/2024.2/bin/vivado.bat" `
-mode batch `
-source scripts/vivado_build.tcl `
-tclargs $ROOT_DIR $VIVADO_DIR
