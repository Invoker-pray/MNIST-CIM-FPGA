# file analysis(in the test vector generate)

导出的文件格式主要有以下几类：

- weights.hex(for instance: fc1_weight_int8.hex)
- bias.hex(fc1_bias_int32.hex)
- quantized_param.npz
- input.hex
- output.hex

# weights.hex

这是第一层全连接层 fc1.weight 的量化权重文件。对于这类文件，每行一个8bit数值，用2位十六进制表示，负数采用8位二补码的低8位。

因为权重 shape 是 \[128, 784\]，所以总共有：128×784=100352

也就是weights.hex 一共 100352 行.

文件的展开顺序是按输出神经元 row-major 展开格式为 \[out\]\[in\]，原型代码是：

```python
for row in q_weight:
    for val in row:
```

所以输出顺序是w\[0,0\],w\[0,1\],...,w\[0,783\],w\[1,0\],...,w\[127,783\]

# bias.hex

这是第一层的偏置。

每行一个量化后的偏置2位十六进制，当前代码里它被导成了 int8，偏置长度是 128　所以 bias.hex 一共 128 行。
现在是把 bias 当 int8 存，但硬件 MAC 累加结果通常是 int32，所以这个 bias.hex 现在不适合直接用于真实整数推理链路。

原型代码是：

```python
bias = model.fc1.bias.data      # [128]
...
with open(f'{output_dir}/bias.hex', 'w') as f:
    for val in q_bias:
        f.write(f'{val.item() & 0xFF:02x}\n')

```
# quantized_param.hex

这是量化参数文件。

原型代码是：

```python
np.savez(f'{output_dir}/quant_params.npz',
         w_scale=w_scale, w_zero=w_zero,
         b_scale=b_scale, b_zero=b_zero)

```

保存了w_scale, w_zero, b_scale, b_zero.
它用于：软件端反量化,硬件端知道量化参数

# input_i.hex

文本文件,每行一个输入元素,2 位十六进制,量化后为 int8
总行数因为 MNIST 一张图是 28×28=784，所以每个 input_i.hex 一共 784 行

```python
img_flat = img.view(-1, 784)
for val in q_img.flatten():

```
文件顺序是，像素 (0,0)，像素 (0,1)...像素 (0,27)，像素 (1,0)
输入向量索引 i = row * 28 + col

# golden_i.hex

这是预期的正确输出，由软件预先计算好。
文件格式是文本文件，每行一个输出元素，2 位十六进制，量化后为 int8

```python
self.fc1 = nn.Linear(784,128)
self.relu = nn.ReLU()
```

也就是输出128行，每行784个结果

当前含义：

这个 golden_i.hex 表示：先用浮点模型跑出 fc1 + relu 的浮点输出，再把这个浮点输出单独量化成 int8，所以它的意义是：“浮点模型输出的量化结果”，不是严格意义上的“整数硬件链路 golden”

# useage

Verilog $readmemh(filename, target);

$readmemh 读进来的是 reg \[7:0\]，本身只是无符号位模式。
如果你想让它参与有符号运算，需要做符号解释。
例如：

wire signed \[7:0\] w = weight_mem\[addr\];
wire signed \[7:0\] x = input_mem\[idx\];
wire signed \[7:0\] b = bias_mem\[out_idx\];


或者：

wire signed \[7:0\] w = $signed(weight_mem\[addr\]);


这样：

8'hff 才会被解释成 -1

8'h80 才会被解释成 -128

# 第一层硬件如何读取 weights.hex
前权重顺序是：

w\[j\]\[i\]


其中：

j = 0..127，输出神经元

i = 0..783，输入索引

所以硬件最自然的读取方式是：addr=j×784+i

计算流程,对于每个输出神经元 j：

累加器清零
从 i=0 到 783
读取：

x = input_mem\[i\]

w = weight_mem\[j*784+i\]

做乘加：

accj+=x⋅w

循环结束后加偏置：

accj+=bias\[j\]


过 ReLU

输出第 j 个结果

```verilog
for (j = 0; j < 128; j = j + 1) begin
    acc = 0;
    for (i = 0; i < 784; i = i + 1) begin
        acc = acc + $signed(input_mem[i]) * $signed(weight_mem[j*784+i]);
    end
    acc = acc + $signed(bias_mem[j]);
    if (acc < 0) acc = 0;
    out[j] = acc; // 或量化后输出
end

```

# input_i.hex 如何使用

```verilog
$readmemh("input_0.hex", input_mem);
```

然后把 input_mem\[0:783\] 依次送给你的计算模块。

方式 1：一次性 preload 到 BRAM

如果你的设计是：

输入先装入片上 RAM

再启动计算

那就：

先把 input_i.hex 用 $readmemh 读入 testbench memory

再驱动写接口，把 784 个数据写入 DUT 内部 input RAM

拉高 start

方式 2：流式输入

如果你的 DUT 是流式输入：

每拍输入一个 x_i

那就：

从 input_mem\[0\] 到 input_mem\[783\] 依次送

每个时钟给一个有效数据

送满 784 个后等待结果

# golden_i.hex 如何使用

```verilog
$readmemh("golden_0.hex", golden_mem);

```

当 DUT 计算完成后，把 DUT 的 128 个输出与 golden_mem 逐项比对。

```verilog
integer k;
initial begin
    for (k = 0; k < 128; k = k + 1) begin
        if (dut_out[k] !== golden_mem[k]) begin
            $display("Mismatch at %0d: dut=%h, golden=%h", k, dut_out[k], golden_mem[k]);
        end
    end
end

```

但要注意

你现在的 golden_i.hex 是：

浮点输出再量化

所以前提是你的硬件计算链路也得尽量模拟这个逻辑。
如果你 RTL 里已经做成了严格整数推理，那这个 golden 就不再是最优基准了。

