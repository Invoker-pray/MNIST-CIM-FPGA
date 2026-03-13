# 关于FPGA_A_mini

这是一个缩小化的模型，相比于FPGA_A，他的中间隐藏层从128削减为16，同时将PAR_OB缩小为1，通过减小并行度加速vivado synth/impl的过程，同时减少板上需要的资源，适用于资源紧张的板子。

约束文件是按照zynq 7020写的，板子型号在hw/scripts/vivado_build.tcl中有写。

# 食用方法

对于mnist数据集，可以通过sw/mini_test.ipynb生成想要的数据集，在mini_test.ipynb中我搭建的是一个简单的小模型，784->16->10，可以选择生成固定或者随机的测试数据。

之后在hw下运行scripts/set_datas.sh完成文件的加载。
hw下，data_packed为经过处理更加方便CIM加载读取的数据，也是实际上被使用的文件。

如果想要运行仿真脚本，请进入scripts后运行run.sh，此时会在hw下生成sim directory，除了top文件的tb直接将compile/sim.log放到sim/，其他仿真记录都会放到sim/log.

如果需要上板测试，请回到hw/，先将constr/下的约束文件修改为符合你实际板子的引脚约束，然后运行scripts/vivado_build.sh（如果在windows下运行则需要修改两个路径，一个是当前project在你pc中的实际路径，一个是pc中vivado.bat的实际路径，这个可以通过查看vivado的属性得到），之后就会自动进行synth/impl，同时在hw下生成vivdao directory，其中会有vivado的工程指针，你可以通过vivado GUI打开工程并查看。

在vivado目录下，会有mnist_cim_damo_a_top.bit，这个文件可以通过vivado
GUI或者开源工具比如OpenFPGALoader实现烧录。

# 一些进阶

- hw/rtl/pkg/package.sv中多数参数是可以修改的，可以适用于不同的模型规模（前提是计算逻辑相匹配）。

- scripts/vivado_build.tcl中有很多vivdao运行配置，包括是否开启综合优化，生成hierarchial report等。

# 一部分问题和优化

当前有一个小bug，系统可能存在一些内存污染的问题，每次复位之后的第一次推理是正确的，但是之后进行的推理会出现错误，结果乱飘，重新复位一下即可（仍在修复中）。

将文件读入BRAM相对来说实现简单，但是灵活性不够，之后可能会修改为通过uart/AXI实现weight/bias/input/sample/quant等文件的外部读取，扩展项目的灵活性。
