
## ============================================================
## top.xdc for onboard_A / mnist_cim_demo_a_top
## Board: PYNQ-Z2 (Zynq-7020)
##
## Top ports:
##   input  clk
##   input  rst_n
##   input  btn_start
##   input  [4:0] sample_sel
##   output uart_tx
##   output led_busy
##   output led_done
##
## Suggested board mapping:
##   clk         -> sysclk (125 MHz)
##   rst_n       -> SW1  (keep ON = not reset)
##   btn_start   -> BTN0
##   sample_sel0 -> BTN1
##   sample_sel1 -> BTN2
##   sample_sel2 -> BTN3
##   sample_sel3 -> SW0
##   sample_sel4 -> JA0
##   uart_tx     -> JA1   (connect to USB-TTL RX if needed)
##   led_busy    -> LED0
##   led_done    -> LED1
## ============================================================

## -------------------------
## Clock: 125 MHz board clock
## -------------------------
set_property PACKAGE_PIN H16 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]
create_clock -name sys_clk -period 8.000 [get_ports clk]

## -------------------------
## Reset / Start
## -------------------------
## rst_n is active-low in RTL
## Here it is mapped to SW1:
##   SW1 = 1 -> normal run
##   SW1 = 0 -> reset asserted
set_property PACKAGE_PIN M19 [get_ports rst_n]
set_property IOSTANDARD LVCMOS33 [get_ports rst_n]

## btn_start mapped to BTN0
set_property PACKAGE_PIN D19 [get_ports btn_start]
set_property IOSTANDARD LVCMOS33 [get_ports btn_start]

## -------------------------
## Sample selection [4:0]
## -------------------------
## sample_sel[0] -> BTN1
set_property PACKAGE_PIN D20 [get_ports {sample_sel[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {sample_sel[0]}]

## sample_sel[1] -> BTN2
set_property PACKAGE_PIN L20 [get_ports {sample_sel[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {sample_sel[1]}]

## sample_sel[2] -> BTN3
set_property PACKAGE_PIN L19 [get_ports {sample_sel[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {sample_sel[2]}]

## sample_sel[3] -> SW0
set_property PACKAGE_PIN M20 [get_ports {sample_sel[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {sample_sel[3]}]

## sample_sel[4] -> PMODA JA0
set_property PACKAGE_PIN Y18 [get_ports {sample_sel[4]}]
set_property IOSTANDARD LVCMOS33 [get_ports {sample_sel[4]}]

## -------------------------
## UART TX
## -------------------------
## uart_tx -> PMODA JA1
## Connect JA1 to USB-TTL RX for serial observation
set_property PACKAGE_PIN Y19 [get_ports uart_tx]
set_property IOSTANDARD LVCMOS33 [get_ports uart_tx]

## -------------------------
## LEDs
## -------------------------
## led_busy -> LED0
set_property PACKAGE_PIN R14 [get_ports led_busy]
set_property IOSTANDARD LVCMOS33 [get_ports led_busy]

## led_done -> LED1
set_property PACKAGE_PIN P14 [get_ports led_done]
set_property IOSTANDARD LVCMOS33 [get_ports led_done]
