# 方案 A : BRAM 预装载 + 按按钮/寄存器启动 + UART 打印

输入样本预先烧到 BRAM/ROM，权重、bias、quant params 也预装载。

板上按钮或简单寄存器触发 start，推理完成后 UART 输出 pred_class。

方案A的设计不需要PC实时的发出命令，也不需要软核CPU，用于实现简单的上板demo.

# file arch

`data/`:保存板上初始化数据。

`rtl/`:保存验证通过的后处理，FC2，argmax，quant_bank等。主要是模型后半段和公用的模块。

`rtl_shared_buffer_ib/`:FC1的共享buffer的架构。

`rtl_ip/`:方案A的核心目录。存放样本ROM，板级FC1输入通路，推理核心等。
