# route_b_output 使用指南

本文档面向 **FPGA / CIM 系统联调与上板验证**，用于说明 `route_b_output` 目录中各文件的格式、含义、推荐使用方式、对拍流程，以及根据当前导出数据所反映出的 **硬件设计注意事项**。

---

## 1. 总体结论

本次检查表明，`route_b_output` 的核心文件格式 **整体正确且自洽**，已经具备作为两层 MLP 整数推理参考数据集的条件。目录中的关键文件满足以下特征：

- 第一层权重：`fc1_weight_int8.hex`
  - 行数：`100352`
  - 对应 shape：`[128, 784]`
  - 每行 2 位十六进制，表示一个 **signed int8** 元素
- 第一层偏置：`fc1_bias_int32.hex`
  - 行数：`128`
  - 每行 8 位十六进制，表示一个 **signed int32** 元素
- 第二层权重：`fc2_weight_int8.hex`
  - 行数：`1280`
  - 对应 shape：`[10, 128]`
  - 每行 2 位十六进制，表示一个 **signed int8** 元素
- 第二层偏置：`fc2_bias_int32.hex`
  - 行数：`10`
  - 每行 8 位十六进制，表示一个 **signed int32** 元素
- 输入样本：`input_i.hex`
  - 每个文件 784 行
  - 每行 2 位十六进制，表示一个 **signed int8** 元素
- 中间 golden：
  - `fc1_acc_i.hex`：128 行，**signed int32**
  - `fc1_relu_i.hex`：128 行，**signed int32**
  - `fc1_out_i.hex`：128 行，**signed int8**
  - `fc2_acc_i.hex`：10 行，**signed int32**
  - `logits_i.hex`：10 行，**signed int8**
- 标签与预测：
  - `labels.txt` 与 `preds.txt` 均为 20 行
  - 当前导出的 20 个样本里，`preds.txt` 与 `labels.txt` **完全一致**

这说明当前 `route_b_output` 已经不仅仅是“文件能导出来”，而是已经形成了一套相对完整的：

**输入 -> 第一层 MAC -> ReLU -> 第一层量化输出 -> 第二层 MAC -> 输出 logits -> 分类结果**

的整数参考链路。

---

## 2. route_b_output 目录文件总览

目录中主要包括以下几类文件：

### 2.1 模型参数类

- `fc1_weight_int8.hex`
- `fc1_bias_int32.hex`
- `fc2_weight_int8.hex`
- `fc2_bias_int32.hex`
- `quant_config.json`
- `mlp_route_b.pth`

### 2.2 输入样本类

- `input_0.hex` ~ `input_19.hex`
- `labels.txt`

### 2.3 第一层 golden 类

- `fc1_acc_0.hex` ~ `fc1_acc_19.hex`
- `fc1_relu_0.hex` ~ `fc1_relu_19.hex`
- `fc1_out_0.hex` ~ `fc1_out_19.hex`

### 2.4 第二层 golden 类

- `fc2_acc_0.hex` ~ `fc2_acc_19.hex`
- `logits_0.hex` ~ `logits_19.hex`
- `pred_0.txt` ~ `pred_19.txt`
- `preds.txt`

---

## 3. 网络结构与数据流

当前 `route_b_output` 对应网络结构为：

- 输入层：`784`
- 隐藏层：`128`
- 输出层：`10`

即：

```text
784 -> 128 -> 10
```

其中整数推理链路可以理解为：

1. 读入 `input_i.hex`，得到长度为 784 的 int8 输入向量
2. 用 `fc1_weight_int8.hex` 和 `fc1_bias_int32.hex` 做第一层乘加
3. 第一层未激活结果与 `fc1_acc_i.hex` 对拍
4. 第一层 ReLU 后结果与 `fc1_relu_i.hex` 对拍
5. 第一层 requant 后的 int8 输出与 `fc1_out_i.hex` 对拍
6. 用 `fc2_weight_int8.hex` 和 `fc2_bias_int32.hex` 做第二层乘加
7. 第二层未缩放结果与 `fc2_acc_i.hex` 对拍
8. 第二层 requant 后的 10 类 logits 与 `logits_i.hex` 对拍
9. 对 `logits_i.hex` 做 `argmax`，结果应等于 `pred_i.txt`
10. `pred_i.txt` 应与 `labels.txt` 中对应样本标签一致

---

## 4. 各文件的详细说明

## 4.1 `fc1_weight_int8.hex`

### 含义
第一层全连接层 `fc1` 的量化权重。

### 形状

```text
[128, 784]
```

也就是：
- 128 个输出神经元
- 每个神经元对应 784 个输入权重

### 文件格式
- 文本文件
- 每行一个元素
- 每行 2 位十六进制
- 表示 **signed int8** 的二补码值

示例：

```text
03
04
fb
06
```

解释为：
- `03` = 3
- `04` = 4
- `fb` = -5
- `06` = 6

### 行数
`100352 = 128 * 784`

### 布局
根据 `quant_config.json`：

```json
"fc1": "row-major [out][in] = [128][784]"
```

所以导出顺序为：

```text
w[0,0], w[0,1], ..., w[0,783], w[1,0], ..., w[127,783]
```

### 硬件读取方式
对于第一层输出神经元 `j`，输入索引 `i`：

```text
addr = j * 784 + i
```

也就是说，硬件地址生成逻辑应采用：
- 外层遍历输出通道 `j = 0..127`
- 内层遍历输入通道 `i = 0..783`

### 注意事项
1. **必须按 signed int8 使用**
   - `8'hfb` 必须被解释为 `-5`
2. **必须与输入顺序严格匹配**
   - 若输入展平顺序与权重索引顺序不一致，结果会整体错误
3. **非常适合做 BRAM/ROM 初始化**
   - 一行一个 int8，适合 `$readmemh`

---

## 4.2 `fc1_bias_int32.hex`

### 含义
第一层偏置。

### 形状

```text
[128]
```

### 文件格式
- 文本文件
- 每行一个元素
- 每行 8 位十六进制
- 表示 **signed int32** 二补码值

示例：

```text
000008e5
0000183f
000001a2
ffffe6b8
```

解释为：
- `000008e5` = 2277
- `0000183f` = 6207
- `000001a2` = 418
- `ffffe6b8` = -6472

### 行数
`128`

### 硬件读取方式
偏置索引与输出神经元一一对应：

```text
bias[j]
```

### 注意事项
1. **这是 int32，不是 int8**
   - 必须进入累加器同量纲的路径
2. **加法位置必须在 MAC 累加之后**
   - 即先完成 784 项乘加，再加偏置
3. **宽度不能截断**
   - 如果你把 bias 误当作 8 bit 或 16 bit，会直接导致结果错误

---

## 4.3 `fc2_weight_int8.hex`

### 含义
第二层全连接层 `fc2` 的量化权重。

### 形状

```text
[10, 128]
```

### 文件格式
- 文本文件
- 每行一个元素
- 每行 2 位十六进制
- 表示 **signed int8**

### 行数
`1280 = 10 * 128`

### 布局
根据 `quant_config.json`：

```json
"fc2": "row-major [out][in] = [10][128]"
```

顺序为：

```text
w2[0,0], ..., w2[0,127], w2[1,0], ..., w2[9,127]
```

### 地址生成
对第二层输出类 `k`、输入索引 `h`：

```text
addr = k * 128 + h
```

### 注意事项
1. 第二层输入不是原始像素，而是 `fc1_out_i.hex`
2. 第二层输入长度是 128，不是 784
3. 第二层同样必须按 signed int8 处理

---

## 4.4 `fc2_bias_int32.hex`

### 含义
第二层偏置。

### 形状

```text
[10]
```

### 文件格式
- 每行 8 位十六进制
- 表示 **signed int32**

### 行数
`10`

### 注意事项
与 `fc1_bias_int32.hex` 一样，必须在 int32 累加路径中加入，不能缩窄。

---

## 4.5 `input_i.hex`

### 含义
第 `i` 个测试样本的输入向量。

### 形状

```text
[784]
```

### 文件格式
- 每行一个元素
- 每行 2 位十六进制
- 表示 **signed int8**

### 行数
`784`

### 数据来源
当前量化策略表现为：背景像素大量出现 `80`，也就是 `-128`。这说明输入采用的是类似：

```text
x_q = round(x * 255) - 128
```

因此：
- 黑色背景（原始像素接近 0） -> `0 - 128 = -128 = 0x80`
- 白色高亮像素（原始像素接近 1） -> `255 - 128 = 127 = 0x7f`

### 展平顺序
输入图像是 `28 x 28`，展平为 784 维向量。一般约定为：

```text
index = row * 28 + col
```

也就是图像按行优先（row-major）展开。

### 硬件使用方式
推荐两种方式：

#### 方式 A：预装载到输入 RAM
1. 通过 `$readmemh` 读入 testbench memory
2. 逐拍写入 DUT 的输入缓存
3. 拉高 `start`

#### 方式 B：流式输入
1. 按索引 `0..783` 逐拍送入 `data_in`
2. 同时置 `valid`
3. 784 个元素送完后等待结果输出

### 注意事项
1. `0x80` 是 `-128`，不是 128
2. 输入零点若为 `-128`，则硬件链路中要明确是否需要做 `(x_q - z_x)`
3. 如果你的设计把输入当作“无符号像素 0~255”，则会与当前数据不兼容

---

## 4.6 `fc1_acc_i.hex`

### 含义
第 `i` 个样本经过第一层乘加并加偏置后的 **int32 累加结果**，尚未经过 ReLU。

### 形状

```text
[128]
```

### 文件格式
- 每行 8 位十六进制
- 表示 **signed int32**

### 行数
`128`

### 使用意义
这是第一层最关键的中间 golden，用来检查：
- 权重读取是否正确
- 输入读取是否正确
- 符号位解释是否正确
- MAC 累加是否正确
- bias 加法是否正确

### 典型对拍位置
如果硬件第一层完成后内部有一个 `acc[127:0]` 或串行输出的中间结果，优先与本文件对拍。

### 注意事项
1. 如果这里就对不上，后面的 ReLU、requant、fc2 都不用继续查
2. 这是第一层定位问题最有效的文件

---

## 4.7 `fc1_relu_i.hex`

### 含义
第一层 `fc1_acc_i.hex` 经过 ReLU 后的结果。

### 形状

```text
[128]
```

### 文件格式
- 每行 8 位十六进制
- 表示 **signed int32**

### 行数
`128`

### 逻辑关系

```text
fc1_relu = max(fc1_acc, 0)
```

### 检查方法
你可以抽查若干行：
- `fc1_acc_i.hex` 为负时，对应 `fc1_relu_i.hex` 必须为 `00000000`
- `fc1_acc_i.hex` 为正时，对应 `fc1_relu_i.hex` 应与原值一致

### 注意事项
1. ReLU 是在 int32 域里做的，不是在 int8 域做的
2. 若硬件先截断再 ReLU，会与当前 golden 不一致

---

## 4.8 `fc1_out_i.hex`

### 含义
第一层 ReLU 输出经 requant 后得到的 **int8 隐层向量**。

### 形状

```text
[128]
```

### 文件格式
- 每行 2 位十六进制
- 表示 **signed int8**

### 行数
`128`

### 使用意义
这是第二层的实际输入参考值。

### 硬件使用方式
如果你的实现是分两拍或分两阶段：
- 第一层算完后写入中间缓存
- 第二层从中间缓存读入

那么该文件应当正好对应中间缓存中的内容。

### 注意事项
1. 这里不是 `fc1_relu` 原值，而是 **requant 后的 int8**
2. 第二层若直接吃 int32，而不是 int8 requant 结果，则会与当前 golden 链路不同

---

## 4.9 `fc2_acc_i.hex`

### 含义
第 `i` 个样本在第二层完成 int8 x int8 乘加并加偏置后的 **int32 累加结果**。

### 形状

```text
[10]
```

### 文件格式
- 每行 8 位十六进制
- 表示 **signed int32**

### 行数
`10`

### 使用意义
这是第二层 MAC 的关键 golden。

### 注意事项
若这里正确，而 `logits_i.hex` 错误，则问题大概率在 requant 阶段。

---

## 4.10 `logits_i.hex`

### 含义
第二层 int32 累加输出经 requant 后得到的 **10 类 int8 logits**。

### 形状

```text
[10]
```

### 文件格式
- 每行 2 位十六进制
- 表示 **signed int8**

### 行数
`10`

### 使用意义
最终分类参考向量。

### 分类方法
对这 10 个 int8 值做 `argmax`，得到预测类别。

### 注意事项
1. 这里是量化后的 logits，不是 softmax 概率
2. 硬件无需做 softmax，只需做 `argmax`
3. `argmax(logits_i.hex)` 应与 `pred_i.txt` 一致

---

## 4.11 `pred_i.txt`、`preds.txt`、`labels.txt`

### `pred_i.txt`
- 第 `i` 个样本的预测类别
- 文件只有一个字符或一行数字，例如 `7`

### `preds.txt`
- 20 个样本的预测结果汇总

### `labels.txt`
- 20 个样本的真实标签汇总

### 当前检查结果
本次检查中：

- `preds.txt` 与 `labels.txt` **完全一致**

说明当前导出的 20 个样本，Python 参考链路全部分类正确。

### 用途
这是系统级正确性的最后检查。

---

## 4.12 `quant_config.json`

### 含义
量化配置文件。

### 当前已包含信息
- 网络结构
- 权重布局
- 输入/隐藏层/输出层维度说明
- 输入量化参数
- `fc1` 的量化参数
- `fc2` 的量化参数

### 作用
硬件设计和 testbench 需要用它确认：
- 当前假定的 weight layout
- 每层输出的 scale / zero-point
- requant 所需的 multiplier / shift

### 注意事项
1. `quant_config.json` 是 **量化语义的来源**，不是可有可无的附属文件
2. 如果 RTL 中手写的量化参数与 json 不一致，对拍一定失败

---

## 4.13 `mlp_route_b.pth`

### 含义
PyTorch 浮点模型参数文件。

### 用途
- 重新生成数据
- 做软件侧 debug
- 验证量化前后的对齐情况

### 注意事项
该文件不是 FPGA 直接使用文件，但建议保留，便于将来再导出不同格式。

---

## 5. 推荐使用流程（硬件联调顺序）

建议按下面顺序使用这些文件，而不要一上来只盯最终分类结果。

### 第一步：验证参数加载
检查：
- `fc1_weight_int8.hex` 是否完整装载到第一层权重存储
- `fc1_bias_int32.hex` 是否完整装载
- `fc2_weight_int8.hex`、`fc2_bias_int32.hex` 是否完整装载

最基本检查项：
- 行数是否一致
- 地址是否从 0 连续递增
- 是否按 signed 解释

### 第二步：只跑第一层 MAC
使用：
- `input_0.hex`
- `fc1_weight_int8.hex`
- `fc1_bias_int32.hex`

对拍：
- `fc1_acc_0.hex`

如果这里不一致，优先检查：
- 输入顺序
- 权重顺序
- signed 解释
- bias 加法位置
- 是否错误使用了 zero-point 修正

### 第三步：验证第一层 ReLU
对拍：
- `fc1_relu_0.hex`

### 第四步：验证第一层 requant
对拍：
- `fc1_out_0.hex`

### 第五步：跑第二层 MAC
对拍：
- `fc2_acc_0.hex`

### 第六步：验证第二层 requant 和 argmax
对拍：
- `logits_0.hex`
- `pred_0.txt`

### 第七步：批量回归
使用 20 个样本全部跑一遍，对比：
- `preds.txt`
- `labels.txt`

---

## 6. Verilog / Testbench 侧推荐读法

## 6.1 典型 `$readmemh` 方式

```verilog
reg [7:0]  fc1_w_mem [0:100351];
reg [31:0] fc1_b_mem [0:127];
reg [7:0]  fc2_w_mem [0:1279];
reg [31:0] fc2_b_mem [0:9];
reg [7:0]  input_mem [0:783];
reg [31:0] fc1_acc_golden [0:127];
reg [31:0] fc1_relu_golden [0:127];
reg [7:0]  fc1_out_golden [0:127];
reg [31:0] fc2_acc_golden [0:9];
reg [7:0]  logits_golden [0:9];

initial begin
    $readmemh("fc1_weight_int8.hex", fc1_w_mem);
    $readmemh("fc1_bias_int32.hex", fc1_b_mem);
    $readmemh("fc2_weight_int8.hex", fc2_w_mem);
    $readmemh("fc2_bias_int32.hex", fc2_b_mem);
    $readmemh("input_0.hex", input_mem);
    $readmemh("fc1_acc_0.hex", fc1_acc_golden);
    $readmemh("fc1_relu_0.hex", fc1_relu_golden);
    $readmemh("fc1_out_0.hex", fc1_out_golden);
    $readmemh("fc2_acc_0.hex", fc2_acc_golden);
    $readmemh("logits_0.hex", logits_golden);
end
```

## 6.2 signed 解释

```verilog
wire signed [7:0]  w1 = $signed(fc1_w_mem[w1_addr]);
wire signed [31:0] b1 = $signed(fc1_b_mem[out_idx]);
wire signed [7:0]  x  = $signed(input_mem[in_idx]);
```

**注意：** `$readmemh` 读入的是位模式，不会自动替你完成 signed 语义，运算时必须显式使用 `$signed(...)`。

---

## 7. 根据当前数据，硬件设计需要特别注意的地方

下面这部分是最重要的。不是泛泛而谈，而是结合你当前 `route_b_output` 的实际数据和文件组织得出的结论。

## 7.1 输入 `0x80` 很多，说明输入零点设计非常关键

在 `input_0.hex` 的开头可以看到大量：

```text
80
80
80
80
```

这意味着大量背景像素被量化成 `-128`。

### 硬件含义
如果你的量化约定是：

```text
x_q = round(x * 255) - 128
```

那么：
- 零像素不是 0，而是 `-128`
- 实际乘加时必须明确当前链路是否需要计算 `(x_q - z_x)`

### 风险
如果硬件误把输入 `0x80` 当“数值 128”或当“零值”，都会造成系统性错误。

### 建议
在 RTL 中先做一个极小规模自检：
- 输入全为 `0x80`
- 权重选定一组已知值
- 手工算出第一层一个神经元的结果
- 与 `fc1_acc` 对拍

---

## 7.2 第一层累加范围不小，accumulator 必须至少 32 位

从 `fc1_acc_0.hex` 中已经能看到：

```text
00014be1
0001416e
ffff87c6
```

这表明第一层 MAC 后的结果已经达到数万量级，属于正常现象。因为第一层是：

```text
784 次 int8 × int8 累加 + int32 bias
```

### 风险
- 若使用 16 位累加器，极易溢出
- 若部分和宽度不足，也会导致结果错但不容易看出

### 建议
- 第一层全路径至少保持 `signed int32`
- 如果做分块累加，局部和与总和都不要窄化

---

## 7.3 bias 必须在 int32 域里加入，不能提前缩位

当前 `fc1_bias_int32.hex` 和 `fc2_bias_int32.hex` 已经明确是 32 位。

### 风险
常见错误包括：
- 把 bias 当 8 位读入
- bias 在加到 acc 前被截断成 16 位
- bias 加在 requant 之后

### 正确做法
- `acc = sum(...) + bias_int32`
- 然后再过 ReLU / requant

---

## 7.4 ReLU 必须先于 requant

当前目录同时给出了：
- `fc1_acc_i.hex`
- `fc1_relu_i.hex`
- `fc1_out_i.hex`

这其实已经把正确顺序告诉你了：

```text
fc1_acc -> ReLU -> requant -> fc1_out
```

### 风险
如果硬件顺序写成：

```text
fc1_acc -> requant -> ReLU
```

那么负数在 requant 截断后的行为会和当前 golden 不一致。

### 建议
确保 ReLU 发生在 int32 域。

---

## 7.5 第二层输入应来自 `fc1_out_i.hex`，而不是 `fc1_relu_i.hex`

这是一个特别容易犯的错误。

### 正确链路
第二层读取的是：

```text
fc1_out_i.hex   （int8）
```

而不是：

```text
fc1_relu_i.hex  （int32）
```

### 风险
如果你的第二层直接消费第一层 ReLU 的 int32 原值，那么：
- fc2_acc 会全部错
- 但 fc1_acc / fc1_relu 仍可能是对的，容易误判

---

## 7.6 权重地址顺序必须完全按 `[out][in]`

当前 `quant_config.json` 已经明确：

- `fc1`: `[128][784]`, row-major `[out][in]`
- `fc2`: `[10][128]`, row-major `[out][in]`

### 风险
若你的硬件阵列天然按列优先、块优先、交织优先读取，而没有在加载阶段做重排，就会整体错位。

### 建议
先在普通 testbench 中用逻辑顺序跑通，再考虑阵列物理映射优化。

---

## 7.7 CIM 物理阵列若做 tile，需要额外做权重重排

当前导出的 `fc1_weight_int8.hex` 和 `fc2_weight_int8.hex` 是 **逻辑布局文件**，适合：
- 软件校验
- testbench
- 通用 BRAM 初始化

但如果你的 CIM 阵列物理组织例如是：
- 每个 tile 只支持 64 个输入
- 或每阵列 32 行 x 64 列

那么真实上板时通常还需要把逻辑权重重排成：
- tile0
- tile1
- ...

### 结论
当前文件 **适合逻辑验证**，但未必等于最终阵列物理烧写格式。

---

## 7.8 `argmax(logits)` 即可，不需要 softmax

当前输出目录只给了 `logits_i.hex` 和 `pred_i.txt`，这说明分类标准就是：

```text
pred = argmax(logits)
```

### 建议
硬件上不要额外实现 softmax。对 MNIST 分类展示完全没有必要，只会增加复杂度和资源占用。

---

## 7.9 `pred_0.txt` 这类单字符文件可能没有换行，解析时不要依赖行数

检查到 `pred_0.txt` 文件大小为 1 字节，这说明它可能只有一个字符，如 `7`，没有额外换行。

### 建议
若用脚本自动读取，不要用“必须有一行换行结尾”的假设。

---

## 8. route_a_output 简要检查结论

`route_a_output` 已做快速检查，核心文件存在且格式方向正确：

- `fc1_weight_int8.hex`
- `fc1_bias_int32.hex`
- `input_i.hex`
- `fc1_acc_golden_i.hex`
- `fc1_relu_golden_i.hex`
- `quant_config.json`

这套文件适合：
- 第一层 `fc1 + ReLU` 的单层验证
- 不涉及第二层分类输出

### route_a_output 额外注意
目录中存在：

```text
.ipynb_checkpoints/
```

以及 checkpoint 文件。上板或提交工程时建议清理掉，避免误用或混淆。

---

## 9. 推荐的联调策略

建议你把 `route_b_output` 的联调拆成下面四级：

### 级别 1：文件与存储正确性
- `$readmemh` 是否读满
- 行数是否正确
- 地址映射是否正确

### 级别 2：第一层正确性
- 对拍 `fc1_acc_i.hex`
- 对拍 `fc1_relu_i.hex`
- 对拍 `fc1_out_i.hex`

### 级别 3：第二层正确性
- 对拍 `fc2_acc_i.hex`
- 对拍 `logits_i.hex`

### 级别 4：系统级正确性
- 检查 `pred_i.txt`
- 批量检查 `preds.txt` 与 `labels.txt`

这样定位问题最快，也最不容易在后期陷入“最终分类错了但不知道错在哪一层”的状态。

---

## 10. 建议的后续改进

虽然当前 `route_b_output` 已经可用，但若要更贴近实际 FPGA/CIM 交付，建议后续继续补充：

1. `manifest.json`
   - 统一列出每个文件的 dtype / shape / layout / 用途
2. 阵列物理映射版权重文件
   - 例如 tile 化后的 `fc1_tile0.hex` 等
3. 打包宽字版本
   - 便于 BRAM 初始化或 DMA 传输
4. 明确的 testbench 脚本
   - 自动依次加载 20 个样本并打印 mismatch

---

## 11. 最终结论

`route_b_output` 当前已经是一套 **格式正确、层次完整、可用于 FPGA/CIM 逻辑验证与系统联调** 的输出目录。核心上没有发现明显格式错误。

特别值得肯定的是：
- 参数文件已区分 int8 权重与 int32 bias
- 中间 golden 分层齐全
- 两层网络输出链路完整
- 20 个样本的预测与标签完全一致

当前最需要硬件侧重点关注的不是“文件是否还能读”，而是：

- signed 解释是否统一
- 输入零点是否处理正确
- ReLU 与 requant 顺序是否正确
- 第二层是否吃的是 `fc1_out` 而不是 `fc1_relu`
- 权重地址生成是否严格匹配 `[out][in]` 布局
- CIM 阵列是否需要从逻辑布局进一步重排为物理 tile 布局

只要这些点处理正确，当前 `route_b_output` 已经足够作为你后续 FPGA/CIM 系统验证的标准数据集。
