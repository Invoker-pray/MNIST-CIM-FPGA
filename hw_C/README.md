# 方案C : PC 通过串口发 sample_id，板上跑出 pred 再回传

FPGA 上仍然是 accelerator + BRAM/ROM，PC 通过串口发sample_id或简单命令，FPGA 读取对应样本，运行推理，再通过 UART 回传 pred_class.
