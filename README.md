# MNIST CIM FPGA

这是一个面向**MNIST推理任务**的**存算一体 / FPGA 验证系统**。

目前在FPGA_A中实现了`784 -> 128 -> 10`的模型，fc1使用compute in memory结构，将本次计算所需要的weight, bias, quant parameter等加载到BROM，计算时先将weight预读取，然后读取输入数据做计算。

最后推理结果会通过PL侧的uart发送，可以通过一个USB-TTL接入电脑，打开串口读取结果。

# FPGA_A_mini

这是一个缩小版，将模型规模缩小为`7884 -> 16 ->10`，fc1中管理并行度的参数`PAR_OB`设置为1.

适用于资源紧张的FPGA.

# bugs

项目有两个问题，第一个是连续推理会出现结果乱漂，猜测是内存污染导致，复位后的第一次推理结果是正确的，暂未解决；

第二个是，没能写出uart/AXI读取weight等文件，每次都是载入BROM，缺乏灵活性。

另外，没有把`PAR_OB`这个关键参数写入packages.sv.
