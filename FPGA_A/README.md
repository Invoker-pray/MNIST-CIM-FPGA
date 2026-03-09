# onboard_A

## 1. 项目简介

`onboard_A` 是 **方案 A 的 FPGA 上板收敛工程**。  
它从原始 `hw` / `hw_A` 工程中抽取出与 **MNIST 推理演示** 直接相关的模块、数据和脚本，目标是形成一个：

- 目录清晰
- 依赖收敛
- 可独立仿真
- 便于后续上板综合实现

的最小可用工程。

当前工程聚焦于 **MNIST 两层 MLP 推理链** 在 FPGA 上的验证，核心特征包括：

- FC1 使用 **共享输入（shared-input）** 的分块并行结构
- 样本输入使用板级可加载的 **sample ROM**
- FC1 输出经过 ReLU / requantize 后进入 FC2
- 最终分类结果通过 **UART 串口输出**
- 板级顶层支持 **样本选择 + 按键启动 + LED 状态显示**

---

## 2. 当前目标

本目录的直接目标不是继续扩展算法，而是把**已经仿真通过的方案 A**整理成一个适合上板的工程版本，用于后续：

- 板级综合与实现
- UART 串口打印预测类别
- 开发板上按键触发推理
- 通过 LED / UART 观察系统运行状态
- 为答辩、报告和演示提供统一工程入口

---

## 3. 预期实现效果

最终希望在 FPGA 板上实现下面的最小闭环：

1. 上电后系统空闲
2. 通过 `sample_sel` 选择一个样本
3. 按下 `btn_start` 触发一次推理
4. 推理过程中 `led_busy` 指示工作状态
5. 推理完成后通过 UART 输出预测类别（例如 `7\r\n`）
6. `led_done` 指示一次推理完成
7. 预测结果与参考标签一致

---

## 4. 目录结构说明

```text
onboard_A/
├── archive/     # 历史说明、原工程树、参考 README
├── constr/      # 板级约束文件（后续补 top.xdc）
├── data/        # 上板/仿真所需数据文件
│   ├── expected/
│   ├── quant/
│   ├── samples/
│   └── weights/
├── docs/        # 上板检查清单、说明文档
├── rtl/         # RTL 主体
│   ├── core/    # 推理主链核心模块
│   ├── ctrl/    # 板级控制逻辑（后续补）
│   ├── mem/     # 权重/偏置/样本/量化参数相关模块
│   ├── pkg/     # 统一参数包
│   ├── top/     # 顶层模块
│   └── uart/    # UART 发送模块
├── scripts/     # 仿真与辅助脚本
├── tb/          # 关键 testbench
└── vivado/      # Vivado 工程目录（后续使用）

```

## 5. 数据文件说明

### `data/samples/`

存放样本输入相关文件：

- `mnist_samples_route_b_output_2.hex`  
  板级 sample ROM 使用的多样本打包文件
- `input_0.hex`, `input_1.hex`, ...  
  单样本输入文件，可用于调试或比对

### `data/weights/`

存放网络参数：

- `fc1_weight_int8.hex`
- `fc1_bias_int32.hex`
- `fc2_weight_int8.hex`
- `fc2_bias_int32.hex`

### `data/quant/`

存放量化相关参数：

- `quant_params.hex`
- `quant_config.json`

### `data/expected/`

存放参考输出：

- `pred_0.txt`, `pred_1.txt`, ...
- `preds.txt`
- `labels.txt`

这些文件主要用于 testbench 对拍和结果核验。

## 6. RTL 模块说明

### 6.1 顶层模块

#### `rtl/top/mnist_cim_demo_a_top.sv`

方案 A 的板级顶层模块，负责连接：

- 时钟 / 复位
- 启动按键
- 样本选择
- 推理核心
- UART 输出
- LED 状态指示

输入：

- `clk`
- `rst_n`
- `btn_start`
- `sample_sel`

输出：

- `uart_tx`
- `led_busy`
- `led_done`

---

### 6.2 推理核心

#### `rtl/core/mnist_inference_core_board.sv`

板级推理主链封装，负责组织：

1. FC1 共享输入推理
2. FC1 ReLU / requantize
3. FC2 全连接
4. argmax 输出预测类别

输出：

- `busy`
- `done`
- `logits_all`
- `pred_class`

---

### 6.3 FC1 共享输入链

#### `rtl/mem/fc1_multi_block_shared_sample_rom.sv`

带 sample ROM 的 FC1 前级控制模块。  
根据 `sample_id + ib` 读取一块输入 tile，并驱动多个输出 block 并行计算。

#### `rtl/core/fc1_ob_engine_shared_input.sv`

FC1 单个输出 block 的计算单元，读取对应权重/偏置，并对输入 tile 做乘加。

#### `rtl/core/cim_tile.sv`

tile 级计算模块，实现单 tile 的并行乘加行为。

#### `rtl/core/psum_accum.sv`

部分和累加逻辑。

#### `rtl/mem/mnist_sample_rom.sv`

板级样本 ROM，从打包样本文件中取出指定 sample 的指定输入 block。

#### `rtl/mem/fc1_weight_bank.sv`

FC1 权重读取模块。

#### `rtl/mem/fc1_bias_bank.sv`

FC1 偏置读取模块。

---

### 6.4 FC1 到 FC2 的中间处理

#### `rtl/core/fc1_relu_requantize_with_file.sv`

FC1 输出的 ReLU 与重定标处理。

#### `rtl/core/fc1_to_fc2_top_with_file.sv`

完成 FC1 输出到 FC2 输入的衔接。

#### `rtl/mem/quantize_param_bank.sv`

读取量化参数文件。

---

### 6.5 FC2 与分类输出

#### `rtl/core/fc2_core_with_file.sv`

FC2 推理模块。

#### `rtl/mem/fc2_weight_bank.sv`

FC2 权重读取模块。

#### `rtl/mem/fc2_bias_bank.sv`

FC2 偏置读取模块。

#### `rtl/core/argmax_int8.sv`

从最终 logits 中选择最大值下标，输出预测类别。

---

### 6.6 UART 模块

#### `rtl/uart/uart_tx.sv`

基础 UART 发送器。

#### `rtl/uart/uart_pred_sender.sv`

将 `pred_class` 格式化为 ASCII 字符，并通过 UART 发送。

---

### 6.7 参数包

#### `rtl/pkg/package.sv`

统一定义网络规模、tile 配置、位宽、block 数等全局参数。  
当前主要配置为：

- 输入维度：784
- 隐层维度：128
- 输出维度：10
- tile 大小：16 × 16
- 输入 block 数：49
- 输出 block 数：8

## 7. 当前工程状态

当前 `onboard_A` 已经完成：

- 关键 RTL 文件收敛
- 数据文件整理到 `data/`
- 独立 testbench 建立
- 仿真脚本收敛到 `scripts/`
- 方案 A 串口输出仿真通过

当前已经验证过的闭环是：

- 读取样本
- 完成推理
- 得到预测类别
- UART 输出分类结果
- 与参考输出对拍通过
