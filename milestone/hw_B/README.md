# 方案B : 软核 CPU + BRAM + 自定义 accelerator + UART

板上放 MicroBlaze / Nios / RISC-V 软核，CPU 通过寄存器控制 accelerator，BRAM 存输入/模型，UART 打印结果。
