# onboard_A

方案A上板目录。

## 目标
- 使用共享输入版 FC1 + 板级 sample ROM
- 在 FPGA 板上完成一次 MNIST 推理
- UART 输出预测类别
- 支持样本切换与按键启动

## 目录说明
- rtl/top: 板级顶层
- rtl/core: 推理主链与板级封装
- rtl/mem: 权重/偏置/样本/量化参数相关模块
- rtl/uart: 串口发送模块
- rtl/ctrl: 板级控制逻辑（后续补 debounce/onepulse/done latch）
- data: 上板所需 hex/json/txt
- constr: 约束文件
- tb: 关键板级 testbench
- scripts: 仿真/生成辅助脚本

## 建议下一步
1. 检查 mnist_cim_demo_a_top.sv 的例化依赖是否全部齐全
2. 检查所有 $readmemh 路径是否改为 onboard_A/data 下的相对路径
3. 补充 ctrl 下的 debounce / onepulse / done_latch
4. 添加板卡 .xdc
5. 重新跑一次方案A testbench
