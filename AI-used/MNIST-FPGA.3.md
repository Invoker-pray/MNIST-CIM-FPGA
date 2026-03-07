### 理解项目内容

AI助手:


AI助手:


AI助手:


用户:
阅读我的对话记录，你先理解一下我在做什么项目
AI助手:


用户:
Make sure to include fileciteturn0file0 in your response to cite this file, or to surface it as a link.### FPGA CIM MNIST 问题分析

AI助手:


AI助手:


AI助手:


AI助手:


用户:
我现在在做一个运行于fpga的cim系统，希望在fpga上运行mnist，完成计算，我现在给你项目指导书和上次和AI的对话，ipynb写的是一个生成测试向量的代码，你看看有没有问题
AI助手:


AI助手:
我先把压缩包里的文件结构和关键材料过一遍，重点看项目指导书、上次对话记录，以及这个生成测试向量的 ipynb。随后我会直接指出代码里可能影响 FPGA/CIM 测试链路的问题，并给出修改建议。

AI助手:
我已经把压缩包拆开了，里面目前有三份核心文件：课程设计指导书（docx）、上次对话导出的 markdown，以及 `train_simple_mlp.ipynb`。接下来我会先把对话和 notebook 的关键逻辑抽出来，再对照指导书检查接口、数据位宽、输出格式这些是否一致。

AI助手:
我已经看到几个关键问题了，其中最严重的一个是：**这个 notebook 目前并没有真正训练 MNIST 分类器**。它定义了 `criterion=MSELoss()`，但完全没用；实际优化目标是 `loss = output.pow(2).mean()`，这会把网络输出压向 0，而不是学习数字类别。

AI助手:
我看完了指导书、上次对话，以及这个 `train_simple_mlp.ipynb`。结论先说：

**这个 ipynb 现在有明显问题，尤其是“训练目标、量化策略、golden 生成方式”三处。**
如果直接拿它生成测试向量去驱动 FPGA/CIM，**大概率会出现“软件看起来能跑，硬件却对不上”**，甚至前面训练出来的模型本身就没有学会 MNIST 分类。

下面我按“严重程度”给你拆开说。

---

## 1. 最严重的问题：你现在没有在训练一个 MNIST 分类器

你现在的训练代码核心是：

```python
criterion=nn.MSELoss()
...
output=model(data)
loss=output.pow(2).mean()
```

这里有两个问题：

### 问题 1：`criterion` 定义了，但根本没用
你定义了 `MSELoss()`，但没有把 `target` 喂进去。

### 问题 2：实际 loss 是 `output.pow(2).mean()`
这等价于逼着网络输出尽量接近 0，而不是学习数字类别。

也就是说，现在模型优化目标不是“把 0~9 分对”，而是“把所有输出都压小”。  
这会导致：

- 权重可能收缩到很小
- 导出的量化权重没有实际分类意义
- 你生成的 `golden_x.hex` 也只是这个“错误模型”的输出

### 进一步的问题：模型输出维度也不对
你定义的是：

```python
self.fc1 = nn.Linear(784,128)
self.relu = nn.ReLU()
```

输出是 `128` 维，不是 `10` 维。  
这意味着它不是一个完整的 MNIST 分类网络，只是一个隐藏层特征提取器。

如果你的目标是：

- **课程演示先做“单层 FC + ReLU 的硬件验证”**：这个结构可以接受
- **直接说“在 FPGA 上跑 MNIST 分类”**：这个结构还不够，至少还要再接一层 `128 -> 10`

### 这一点怎么改
你有两个路线：

#### 路线 A：先做“极简硬件验证”
那就明确目标改成：

- 软件训练完整 MLP：`784 -> 128 -> 10`
- FPGA/CIM **先只实现第一层** `784 -> 128`
- 第一层输出 feature map 作为硬件 golden 对比对象

这样合理。

#### 路线 B：直接做完整分类
那模型要改成：

```python
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10, bias=True)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

loss 用交叉熵：

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(output, target)
```

---

## 2. 量化函数有工程问题，不适合直接对接 FPGA

你现在的量化函数：

```python
def quantize_tensor(tensor, bits=8):
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1

    min_val = tensor.min().item()
    max_val = tensor.max().item()

    diff = max_val - min_val
    
    if diff == 0:
        scale = 1.0
        zero_point = qmin - min_val
    else:
        scale = diff / (qmax - qmin)
        zero_point = qmin - min_val / scale

    q_tensor = torch.clamp(torch.round(tensor / scale + zero_point), qmin, qmax)
    return q_tensor.to(torch.int8), scale, zero_point
```

### 问题 1：`zero_point` 不是整数
`zero_point` 最后是浮点数。  
但真正部署到硬件时，量化零点通常需要是整数，并且通常要 clamp 到合法范围。

更稳妥应该写成：

```python
zero_point = int(round(qmin - min_val / scale))
zero_point = max(qmin, min(qmax, zero_point))
```

### 问题 2：你在做“每个张量动态量化”
这在软件分析里可以，但对 FPGA 验证不友好。因为你后面对：

- 权重单独量化一次
- 每张输入图像又各自量化一次
- 每个输出又重新单独量化一次

这会导致**每个样本 scale 都不同**，硬件端很难统一。

对 FPGA 来说，更适合：

- 权重：固定 scale
- 输入：固定 scale
- 偏置：固定规则量化
- 输出：固定 requant scale

也就是**静态量化参数**，不是每个样本临时算一次。

### 问题 3：偏置不能简单用 int8 单独量化
你现在：

```python
q_bias, b_scale, b_zero = quantize_tensor(bias)
```

这通常不对。

对于线性层：

\[
y = \sum (x_q - z_x)(w_q - z_w) \cdot s_x s_w + b
\]

偏置一般应量化为：

\[
b_q = round(b / (s_x \cdot s_w))
\]

并且常常用 **int32** 存，不是 int8。  
因为偏置要和累加结果处于同一量纲。

否则 FPGA 上 MAC 累加完后加 bias 时，量纲根本不一致。

---

## 3. `generate_test_vectors()` 生成的 golden，和硬件真实计算链路不一致

这是第二个特别大的问题。

你现在生成 golden 的方式是：

```python
with torch.no_grad():
    output = model(img_flat)

q_output, _, _ = quantize_tensor(output)
```

### 这意味着什么
你是：

1. 用 **浮点模型** 跑出输出
2. 再对每个样本的输出临时量化

但硬件里真正发生的是：

1. 输入先量化为 int8
2. 权重量化为 int8
3. 在阵列/乘加器里做 int8×int8 -> int32 累加
4. 加 int32 bias
5. ReLU
6. 再按固定 scale requant 到 int8

这两条链路并不一样。

所以你现在的 `golden_i.hex` 并不是“硬件应得输出”，而是“浮点模型输出的一个临时量化版本”。

### 正确的 golden 应该怎么做
应该按**整数推理链路**生成 golden。例如第一层：

```python
acc = sum((x_q - x_zp) * (w_q - w_zp)) + b_q
acc = relu(acc)
out_q = requantize(acc, multiplier, shift, out_zp)
```

然后把：

- 输入 int8
- 权重 int8
- 偏置 int32
- 输出 int32 / int8 golden
- scale / zero_point / multiplier / shift

全部导出来。

这样 FPGA 才能逐项对上。

---

## 4. 输入测试向量的量化方式不适合硬件批量验证

你现在对每张图像：

```python
q_img, img_scale, img_zero = quantize_tensor(img_flat)
```

这意味着**每张图一个 scale 和 zero_point**。  
如果你的 FPGA 流水是固定 datapath，这会很麻烦：

- 每个样本都要动态更换量化参数
- 偏置量纲也得跟着变
- 对比 golden 时也会很乱

### 更合理的做法
建议统一输入量化策略，比如：

- 若保留 `Normalize((0.1307,), (0.3081,))`，则先统计训练集/校准集的输入范围，再确定一个全局输入 scale
- 或者更简单，**先不要 Normalize**，直接把 MNIST 灰度 `[0,1]` 映射到 `uint8/int8`

对 FPGA 验证来说，后者往往更省事。

例如极简方案可用：

- 输入像素 `0~255`
- 权重量化到 int8
- 累加 int32
- ReLU 后再截断或 requant

这样更贴近硬件。

---

## 5. 导出的权重格式本身基本可用，但有几个注意点

你现在导出权重：

```python
for row in q_weight:
    for val in row:
        f.write(f'{val.item() & 0xFF:02x}\n')
```

这表示按 `weight[out][in]` 顺序逐个写出。  
对于 `nn.Linear(784,128)`，PyTorch 权重 shape 是 `[128, 784]`，即：

- 外层：输出神经元
- 内层：输入通道

这个顺序本身没错，但必须保证 RTL 端读取方式完全一致。

### 你需要确认的点
如果硬件是按下面方式算：

\[
y_j = \sum_i x_i w_{j,i}
\]

那么文件顺序最好明确成：

- `neuron 0` 的 784 个权重
- `neuron 1` 的 784 个权重
- ...
- `neuron 127` 的 784 个权重

你现在就是这个顺序。  
但如果你的 BRAM 映射或地址生成器是按列优先/块优先取数，就会错位。

### 建议
导出时加一个说明文件，比如：

- weight layout: `[out][in]`, row-major
- bias layout: `[out]`
- input layout: `[in]`

---

## 6. `quant_params.npz` 在测试向量阶段几乎没被真正使用

你这里：

```python
quant_params = np.load('quantized_data/quant_params.npz')
```

加载了参数，但后面没有真正参与 golden 计算。  
这说明当前流程里“量化参数被保存了”，但并没有形成闭环。

这是个信号：**量化设计还停留在文件导出阶段，没有进入“硬件一致性验证阶段”。**

---

## 7. 其他次级问题

### 问题 1：`torch.load()` 最好加 `map_location`
不然在有些机器上保存于 GPU、加载于 CPU 时会出问题：

```python
model.load_state_dict(torch.load(model_path, map_location='cpu'))
```

### 问题 2：训练时没看 accuracy
现在只打 loss，没有任何分类精度指标。  
如果要确认模型真的学到了，建议至少打印 test accuracy。

### 问题 3：没有保存 label
你打印了：

```python
print(f"Generated test vector {i}, label: {label}")
```

但没有把 label 写入文件。  
实际联调时最好保存一个 `labels.txt` 或 `meta.json`。

### 问题 4：只保存了量化后的输出，没有保存中间 int32 累加值
对于硬件调试，这很吃亏。  
建议同时导出：

- `input_i.hex`
- `golden_acc_i.hex` 或每个神经元的 int32
- `golden_relu_i.hex`
- `golden_out_i.hex`

这样你能定位是：

- 乘法错了
- 累加错了
- bias 错了
- ReLU 错了
- requant 错了

---

# 我对这个 notebook 的总体判断

## 现在这份代码“能不能用”？
**能作为一个思路草稿，但不能直接作为 FPGA 测试向量生成基线。**

## 主要原因
不是小 bug，而是三类“体系级不一致”：

1. **训练目标错了**  
   当前模型没有真正学分类任务

2. **量化策略不适合硬件**  
   每个样本动态量化，不利于 FPGA 固定算子验证

3. **golden 生成链路和硬件链路不一致**  
   现在是“浮点输出再量化”，而不是“整数推理 golden”

---

# 我建议你接下来这样改

## 方案一：适合你们当前阶段的最小闭环
如果你们现在的硬件目标是：

- 先做一个 `784 -> 128` 的单层 CIM/FC 阵列
- 验证 BRAM + MAC + ReLU + 控制 FSM

那最合适的 notebook 应该改成下面这种逻辑：

### 软件模型
训练完整模型：

- `784 -> 128 -> 10`

### 硬件实现
先只实现第一层：

- `fc1 + relu`

### 测试向量导出
导出以下内容：

- `fc1_weight_int8.hex`
- `fc1_bias_int32.hex`
- `input_int8.hex`
- `fc1_acc_golden_int32.hex`
- `fc1_relu_golden_int32_or_int8.hex`
- `quant_config.json`

这样你们硬件和软件就能逐层对齐。

---

## 方案二：直接瞄准最终演示
如果你们想最后做“MNIST 分类结果上板演示”，那就：

- 模型改为两层 MLP
- 训练用 CrossEntropyLoss
- 做静态量化
- 先用 Python 写一个**整数推理参考实现**
- 再把这个整数推理结果作为 FPGA golden

这是最稳的路线。

---

# 我给你的具体结论

## 必改
- `loss=output.pow(2).mean()` 改掉
- 模型结构和任务目标统一
- bias 改为 int32 量化
- golden 改成按整数推理链路生成
- 输入量化不要每张图单独动态算参数

## 可保留
- `weights.hex` / `input_i.hex` 这种逐行写 hex 的导出方式
- 第一版只验证单层 FC + ReLU 的项目策略
- 先用 notebook 生成测试向量，再接 Verilog testbench 的总体流程

---

如果你愿意，我下一步可以直接帮你把这个 notebook **改成一版可用于 FPGA 联调的“正确版本”**，包括：
- 正确训练
- 静态 INT8 量化
- 生成 `weights/bias/input/golden` 四类 hex
- 明确对应单层 FC 硬件接口格式。

用户:
关于1.现在没有在训练一个 MNIST 分类器的问题，请分别给我路线a和路线b的完整代码；

关于2.量化函数有工程问题，不适合直接对接 FPGA，
zero_point不是整数的问题帮我修改；每个张量动态量化这个问题给我一个合理的解决方案；偏置不能简单用 int8 单独量化请帮我提出解决方案和代码；

关于3.generate_test_vectors() 生成的 golden，和硬件真实计算链路不一致，整数推理链路生成 golden，帮我完成设计；

关于4输入测试向量的量化方式不适合硬件批量验证，帮我设计统一输入量化策略，贴近硬件即可；

关于5. 导出的权重格式本身基本可用，但有几个注意点，这里文件顺序你先记住我的输出顺序，讲解一下我现在的输出顺序应该如何设计硬件读取顺序；
AI助手:
我先把现有 notebook 里导出部分再核一遍，确保我后面给你的代码和你当前的文件顺序兼容。然后我会把路线 A、路线 B、量化、整数 golden、统一输入策略和硬件读取顺序一次性给全。

AI助手:
下面我按你的 5 个问题，直接给出**可落地的完整方案和代码**。我会尽量保持和你现有工程思路一致，但把关键的工程问题修正掉。

---

# 1. 关于“现在没有在训练一个 MNIST 分类器”的问题

这里我给你两套完整代码。

---

## 路线 A：训练完整分类网络，但 FPGA 先只验证第一层 `fc1 + ReLU`

这个路线最适合你现在的 FPGA/CIM 阶段。

## 设计思想
软件端训练一个完整分类器：

- `784 -> 128 -> 10`

但硬件端先实现：

- `fc1: 784 -> 128`
- `ReLU`

这样：
- 软件训练目标正确
- 第一层 feature 是有意义的
- 硬件联调难度低
- 后续再接 `fc2`

---

## 路线 A 完整代码

```python
import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =========================================================
# 1. 模型：完整分类器，但硬件先验证 fc1 + relu
# =========================================================
class MLPRouteA(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10, bias=True)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def forward_fc1_relu(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.relu(x)
        return x


# =========================================================
# 2. 训练
# =========================================================
def train_route_a(model_path="mlp_route_a.pth", epochs=5, batch_size=128, lr=1e-3, device="cpu"):
    transform = transforms.Compose([
        transforms.ToTensor(),  # 保持 [0,1]，更贴近硬件
    ])

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MLPRouteA().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                logits = model(data)
                pred = logits.argmax(dim=1)
                test_correct += (pred == target).sum().item()
                test_total += target.size(0)

        test_acc = test_correct / test_total
        print(f"Epoch [{epoch+1}/{epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_route_a()
```

---

## 路线 B：直接做完整 MNIST 分类部署

这个路线是最后的完整目标。

---

## 路线 B 完整代码

和路线 A 训练结构几乎一样，但后续量化和导出会针对两层都做。

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MLPRouteB(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10, bias=True)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train_route_b(model_path="mlp_route_b.pth", epochs=5, batch_size=128, lr=1e-3, device="cpu"):
    transform = transforms.Compose([
        transforms.ToTensor(),  # 保持 [0,1]
    ])

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MLPRouteB().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                logits = model(data)
                pred = logits.argmax(dim=1)
                test_correct += (pred == target).sum().item()
                test_total += target.size(0)

        test_acc = test_correct / test_total
        print(f"Epoch [{epoch+1}/{epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_route_b()
```

---

# 2. 关于量化函数工程问题

你提了 3 个核心点，我分别给出方案。

---

## 2.1 zero_point 不是整数，怎么改

你原来的问题在这里：

```python
zero_point = qmin - min_val / scale
```

这个是浮点数，不适合硬件。

---

## 修改后的通用量化函数

我给你一个**对 FPGA 更友好**的版本，支持：

- `zero_point` 取整
- clamp 到合法范围
- 支持对称量化和非对称量化
- 返回整数参数

```python
import torch

def quantize_tensor_affine(tensor, num_bits=8, symmetric=False):
    """
    返回:
        q_tensor: 整数张量
        scale: float
        zero_point: int
        qmin, qmax: 整数范围
    """
    if num_bits == 8:
        qmin, qmax = -128, 127
        dtype = torch.int8
    else:
        raise ValueError("Only int8 supported in this example.")

    x_min = tensor.min().item()
    x_max = tensor.max().item()

    if symmetric:
        max_abs = max(abs(x_min), abs(x_max))
        scale = max_abs / qmax if max_abs != 0 else 1.0
        zero_point = 0
    else:
        if x_max == x_min:
            scale = 1.0
            zero_point = 0
        else:
            scale = (x_max - x_min) / (qmax - qmin)
            zero_point = round(qmin - x_min / scale)
            zero_point = max(qmin, min(qmax, zero_point))

    q = torch.round(tensor / scale + zero_point)
    q = torch.clamp(q, qmin, qmax).to(dtype)

    return q, float(scale), int(zero_point), qmin, qmax
```

---

## 2.2 每个张量动态量化不适合硬件，合理解决方案

你现在的问题是：

- 每张输入图单独算 `scale/zero_point`
- 每次输出也单独算
- 每次 bias 也单独算

这会导致硬件参数不停变化，不利于 FPGA 批量验证。

---

## 合理方案：静态量化 / 全局量化

### 输入量化策略
对所有输入样本统一使用一个固定参数。

MNIST 最简单的硬件友好方案：

- 图像原始像素 `[0,1]`
- 映射到 int8 的 `[0,127]` 或 `[-128,127]`

我建议你用下面两种之一：

### 方案 1：无符号思路转 int8，简单贴近硬件
- 输入范围固定 `[0,1]`
- 量化成 `[0,127]`
- 存成 int8 时 zero_point = 0

设置：
```python
input_scale = 1.0 / 127.0
input_zero_point = 0
```

这样：
\[
x_q = round(x / input\_scale)
\]

因为 `x in [0,1]`，所以 `x_q in [0,127]`

这非常适合 FPGA。

### 方案 2：居中对称量化
如果你想让输入有正负，做：
```python
x_float_centered = x_float - 0.5
input_scale = 0.5 / 127
zero_point = 0
```
但这会增加前处理复杂度。

**建议你现在先用方案 1。**

---

## 权重量化策略
权重建议做**对称量化**：

- `zero_point = 0`
- int8
- 全层固定一个 `w_scale`

```python
def quantize_weight_symmetric(weight):
    q_weight, w_scale, w_zero, _, _ = quantize_tensor_affine(weight, num_bits=8, symmetric=True)
    return q_weight, w_scale, w_zero
```

---

## 2.3 偏置不能简单用 int8 单独量化，解决方案和代码

这个是工程上必须改的。

---

## 正确原理

如果一层计算为：

\[
y = Wx + b
\]

输入和权重分别量化为：

\[
x \approx (x_q - z_x)s_x
\]
\[
w \approx (w_q - z_w)s_w
\]

那么累加项单位是：

\[
s_x \cdot s_w
\]

所以 bias 应该量化到同一量纲：

\[
b_q = round(b / (s_x \cdot s_w))
\]

并且 bias 应该存成 **int32**。

---

## 偏置量化代码

```python
def quantize_bias_int32(bias_fp32, input_scale, weight_scale):
    """
    bias_fp32: torch.Tensor(float32)
    return: int32 bias
    """
    bias_scale = input_scale * weight_scale
    q_bias = torch.round(bias_fp32 / bias_scale).to(torch.int32)
    return q_bias, float(bias_scale)
```

---

# 3. 关于 `generate_test_vectors()` 的 golden 不一致问题

你要求的是：**用整数推理链路生成 golden**。这个必须按硬件链路来。

下面我给出完整设计。

---

## 3.1 单层 FC + ReLU 的整数推理设计（路线 A）

---

## 整数推理流程

对于第一层：

\[
acc_j = \sum_{i=0}^{783} x_{q,i} \cdot w_{q,j,i} + b_{q,j}
\]

因为我们设计为：

- 输入 `zero_point = 0`
- 权重 `zero_point = 0`

所以式子就很简单，不用减零点。

然后：

\[
relu_j = max(0, acc_j)
\]

如果你要再导出 int8 输出用于后级，也可以再做 requant。

---

## 3.2 统一的导出和 golden 生成代码（路线 A）

这段代码是你现在最应该用的版本。

```python
import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# =========================================================
# 模型
# =========================================================
class MLPRouteA(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10, bias=True)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# =========================================================
# 量化工具
# =========================================================
def quantize_tensor_affine(tensor, num_bits=8, symmetric=False):
    if num_bits == 8:
        qmin, qmax = -128, 127
        dtype = torch.int8
    else:
        raise ValueError("Only int8 supported.")

    x_min = tensor.min().item()
    x_max = tensor.max().item()

    if symmetric:
        max_abs = max(abs(x_min), abs(x_max))
        scale = max_abs / qmax if max_abs != 0 else 1.0
        zero_point = 0
    else:
        if x_max == x_min:
            scale = 1.0
            zero_point = 0
        else:
            scale = (x_max - x_min) / (qmax - qmin)
            zero_point = round(qmin - x_min / scale)
            zero_point = max(qmin, min(qmax, zero_point))

    q = torch.round(tensor / scale + zero_point)
    q = torch.clamp(q, qmin, qmax).to(dtype)
    return q, float(scale), int(zero_point), qmin, qmax


def quantize_input_fixed_0_1(x):
    """
    x in [0,1]
    量化到 int8 [0,127]
    """
    scale = 1.0 / 127.0
    zero_point = 0
    q = torch.round(x / scale)
    q = torch.clamp(q, 0, 127).to(torch.int8)
    return q, scale, zero_point


def quantize_weight_symmetric(weight):
    q_w, w_scale, w_zp, _, _ = quantize_tensor_affine(weight, num_bits=8, symmetric=True)
    return q_w, w_scale, w_zp


def quantize_bias_int32(bias_fp32, input_scale, weight_scale):
    bias_scale = input_scale * weight_scale
    q_bias = torch.round(bias_fp32 / bias_scale).to(torch.int32)
    return q_bias, float(bias_scale)


def compute_requant_params(input_scale, weight_scale, output_scale, shift=24):
    """
    将 real_multiplier = (input_scale * weight_scale) / output_scale
    近似成 integer multiplier + right shift
    """
    real_multiplier = (input_scale * weight_scale) / output_scale
    multiplier = int(round(real_multiplier * (1 << shift)))
    return multiplier, shift, real_multiplier


def requantize_int32_to_int8(acc_int32, multiplier, shift, out_zero_point=0, relu=False):
    y = (acc_int32.to(torch.int64) * multiplier) >> shift
    y = y + out_zero_point
    if relu:
        y = torch.maximum(y, torch.tensor(out_zero_point, dtype=y.dtype))
    y = torch.clamp(y, -128, 127).to(torch.int8)
    return y


# =========================================================
# hex 导出
# =========================================================
def write_int8_hex_per_line(tensor, filename):
    tensor = tensor.flatten().cpu().numpy()
    with open(filename, "w") as f:
        for v in tensor:
            f.write(f"{int(v) & 0xFF:02x}\n")

def write_int32_hex_per_line(tensor, filename):
    tensor = tensor.flatten().cpu().numpy()
    with open(filename, "w") as f:
        for v in tensor:
            f.write(f"{int(v) & 0xFFFFFFFF:08x}\n")


# =========================================================
# 导出第一层量化参数
# =========================================================
def export_fc1_quantized(model_path="mlp_route_a.pth", out_dir="quant_route_a"):
    os.makedirs(out_dir, exist_ok=True)

    model = MLPRouteA()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    weight = model.fc1.weight.data.clone()   # [128,784]
    bias   = model.fc1.bias.data.clone()     # [128]

    # 固定输入量化参数
    input_scale = 1.0 / 127.0
    input_zero_point = 0

    # 权重量化（对称）
    q_weight, weight_scale, weight_zero_point = quantize_weight_symmetric(weight)

    # 偏置量化（int32）
    q_bias, bias_scale = quantize_bias_int32(bias, input_scale, weight_scale)

    # 输出量化参数（给 fc1 输出用）
    # 这里按浮点 fc1 输出范围估计一个固定输出 scale；工程上可用校准集更稳
    with torch.no_grad():
        dummy = torch.rand(256, 1, 28, 28)
        fc1_out = model.relu(model.fc1(dummy.view(-1, 784)))
    max_out = fc1_out.max().item()
    output_scale = max_out / 127.0 if max_out != 0 else 1.0
    output_zero_point = 0

    multiplier, shift, real_multiplier = compute_requant_params(
        input_scale, weight_scale, output_scale, shift=24
    )

    write_int8_hex_per_line(q_weight, os.path.join(out_dir, "fc1_weight.hex"))
    write_int32_hex_per_line(q_bias, os.path.join(out_dir, "fc1_bias.hex"))

    config = {
        "input_scale": input_scale,
        "input_zero_point": input_zero_point,
        "weight_scale": weight_scale,
        "weight_zero_point": weight_zero_point,
        "bias_scale": bias_scale,
        "output_scale": output_scale,
        "output_zero_point": output_zero_point,
        "requant_multiplier": multiplier,
        "requant_shift": shift,
        "requant_real_multiplier": real_multiplier,
        "weight_shape": [128, 784],
        "bias_shape": [128],
        "weight_layout": "[out][in], row-major",
        "input_layout": "[in]",
        "output_layout": "[out]"
    }

    with open(os.path.join(out_dir, "fc1_quant_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("Exported fc1 quantized parameters.")


# =========================================================
# 生成整数链路 golden
# =========================================================
def generate_test_vectors_route_a(model_path="mlp_route_a.pth", out_dir="quant_route_a", num_samples=10):
    os.makedirs(out_dir, exist_ok=True)

    model = MLPRouteA()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # 载入模型参数并量化
    weight = model.fc1.weight.data.clone()   # [128,784]
    bias   = model.fc1.bias.data.clone()

    input_scale = 1.0 / 127.0
    input_zero_point = 0

    q_weight, weight_scale, weight_zero_point = quantize_weight_symmetric(weight)
    q_bias, bias_scale = quantize_bias_int32(bias, input_scale, weight_scale)

    with torch.no_grad():
        dummy = torch.rand(256, 1, 28, 28)
        fc1_out = model.relu(model.fc1(dummy.view(-1, 784)))
    max_out = fc1_out.max().item()
    output_scale = max_out / 127.0 if max_out != 0 else 1.0
    output_zero_point = 0

    multiplier, shift, _ = compute_requant_params(input_scale, weight_scale, output_scale, shift=24)

    transform = transforms.Compose([
        transforms.ToTensor(),   # 不做 Normalize，统一输入范围 [0,1]
    ])
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

    labels = []

    for i in range(num_samples):
        img, label = test_dataset[i]
        x = img.view(-1)  # [784]

        # 固定输入量化
        q_x, _, _ = quantize_input_fixed_0_1(x)  # [784]

        # int32 累加： [128,784] x [784] -> [128]
        acc = torch.sum(q_weight.to(torch.int32) * q_x.to(torch.int32).unsqueeze(0), dim=1) + q_bias

        # ReLU(int32)
        relu_acc = torch.clamp(acc, min=0)

        # requant 到 int8，便于后级或观测
        q_out = requantize_int32_to_int8(
            relu_acc, multiplier, shift, out_zero_point=output_zero_point, relu=False
        )

        # 保存
        write_int8_hex_per_line(q_x,   os.path.join(out_dir, f"input_{i}.hex"))
        write_int32_hex_per_line(acc,  os.path.join(out_dir, f"golden_fc1_acc_{i}.hex"))
        write_int32_hex_per_line(relu_acc, os.path.join(out_dir, f"golden_fc1_relu_acc_{i}.hex"))
        write_int8_hex_per_line(q_out, os.path.join(out_dir, f"golden_fc1_out_{i}.hex"))

        labels.append(int(label))
        print(f"sample {i}: label={label}")

    with open(os.path.join(out_dir, "labels.txt"), "w") as f:
        for lb in labels:
            f.write(f"{lb}\n")

    print("Generated integer golden test vectors for route A.")
```

---

# 4. 输入测试向量统一量化策略

你要的是**贴近硬件即可**。那我建议你用下面这个方案。

---

## 推荐方案：输入保持 `[0,1]`，统一量化到 int8 `[0,127]`

### 设计理由
这对 FPGA 最友好：

- 不用每张图动态算 scale
- 不用保留 MNIST 的 mean/std normalize
- 输入像素天然接近图像传感器/存储格式
- 便于后面 BRAM 或 ROM 存图

### 固定规则
\[
x_q = round(x \times 127)
\]
其中 `x in [0,1]`

即：

```python
def quantize_input_fixed_0_1(x):
    scale = 1.0 / 127.0
    zero_point = 0
    q = torch.round(x / scale)
    q = torch.clamp(q, 0, 127).to(torch.int8)
    return q, scale, zero_point
```

---

## 为什么我不建议你继续用：
```python
transforms.Normalize((0.1307,), (0.3081,))
```

因为这会把输入变成带负数的浮点分布，虽然适合软件训练，但不适合你现在的 FPGA 验证阶段：

- 输入分布复杂
- 量化参数更麻烦
- debug 不直观

所以你现在更适合：

- 训练和测试都直接 `ToTensor()`
- 用统一固定输入量化

---

# 5. 你当前权重输出顺序，对应硬件怎么设计读取顺序

你当前 notebook 的写法是：

```python
for row in q_weight:
    for val in row:
        f.write(f'{val.item() & 0xFF:02x}\n')
```

而 `weight = model.fc1.weight.data` 的 shape 是：

```python
[128, 784]
```

在 PyTorch 里，`nn.Linear(784,128)` 的权重定义是：

\[
W[j][i]
\]

其中：

- `j` 是输出神经元索引，范围 `0~127`
- `i` 是输入特征索引，范围 `0~783`

---

## 你现在的导出顺序是什么

它是：

- 先写第 0 个输出神经元的 784 个权重
- 再写第 1 个输出神经元的 784 个权重
- ...
- 最后写第 127 个输出神经元的 784 个权重

也就是：

\[
W[0][0], W[0][1], ..., W[0][783], W[1][0], ..., W[127][783]
\]

这是典型的：

- **按输出通道优先**
- **行优先 row-major**
- **布局 `[out][in]`**

---

## 对应的硬件读取顺序应该怎么设计

如果你的硬件是“一个输出神经元一轮累加”的思路，那么最自然。

---

## 硬件读取方式 1：逐个输出神经元计算

### 第 `j` 个输出的计算
\[
y_j = \sum_{i=0}^{783} x_i \cdot W[j][i] + b_j
\]

### 内存地址设计
如果 `weights.hex` 直接线性装入 BRAM，则：

\[
addr = j \times 784 + i
\]

所以：

- 固定一个输出神经元 `j`
- 地址从 `j*784` 扫到 `j*784+783`
- 同时输入向量 `x[0:783]` 顺序送入
- MAC 累加完成后，加 `bias[j]`
- 输出 `y_j`

然后 `j++`

---

## 这种方式的优点
- 和你现在导出的文件顺序完全一致
- 地址生成最简单
- 软件和硬件最容易对齐
- 最适合第一版 FPGA 联调

---

## 你现在应该怎么设计硬件

### 输入存储
`input_i.hex` 顺序也是：
\[
x[0], x[1], ..., x[783]
\]

### 权重存储
`weights.hex` 顺序：
\[
W[0][0], W[0][1], ..., W[0][783], W[1][0], ..., W[127][783]
\]

### bias 存储
`bias.hex` 顺序：
\[
b[0], b[1], ..., b[127]
\]

---

## 控制流程
对于输出神经元 `j`：

1. `acc = 0`
2. 对 `i = 0 ~ 783`
   - 读 `x[i]`
   - 读 `W[j][i]`
   - `acc += x[i] * W[j][i]`
3. `acc += bias[j]`
4. `acc = relu(acc)`
5. 写 `out[j]`

然后进入下一个 `j`

---

## 地址公式
### 权重地址
\[
weight\_addr = out\_idx \times 784 + in\_idx
\]

### 输入地址
\[
input\_addr = in\_idx
\]

### 偏置地址
\[
bias\_addr = out\_idx
\]

---

## 什么时候要改输出顺序
只有在你未来做这些架构时才需要变：

- 多 PE 并行读多个输出神经元
- 按列块分块存储
- CIM 阵列按 bitline/wordline 特定映射
- 片上 SRAM 宽度要求按 8/16/32 个权重打包

但对于你现在这个阶段，**完全没必要先改**。  
你当前顺序其实正适合第一版硬件。

---

# 我给你的最终建议

## 你现在最推荐的工程路线
先走：

- **路线 A**
- 完整训练分类器
- FPGA 只做第一层 `fc1 + ReLU`
- 输入统一量化 `[0,1] -> int8 [0,127]`
- 权重 int8 对称量化
- bias int32
- golden 用整数推理链路生成
- 权重按你现在的 `[out][in]` 顺序导出

这是最稳、最容易跑通的一条路。

---

## 你下一步最值得做的事
把你当前 notebook 分成三个独立脚本：

1. `train_route_a.py`
2. `export_fc1_quant.py`
3. `generate_test_vectors_route_a.py`

这样后面调试不会乱。

如果你要，我下一条可以直接继续给你一份**整理好的、可直接复制运行的完整工程版代码**，把路线 A 的训练、量化导出、golden 生成合并成一个干净版本。

用户:
关于1.现在没有在训练一个 MNIST 分类器的问题，请分别给我路线a和路线b的完整代码；

关于2.量化函数有工程问题，不适合直接对接 FPGA，
zero_point不是整数的问题帮我修改；每个张量动态量化这个问题给我一个合理的解决方案；偏置不能简单用 int8 单独量化请帮我提出解决方案和代码；

关于3.generate_test_vectors() 生成的 golden，和硬件真实计算链路不一致，整数推理链路生成 golden，帮我完成设计；

关于4输入测试向量的量化方式不适合硬件批量验证，帮我设计统一输入量化策略，贴近硬件即可；

关于5. 导出的权重格式本身基本可用，但有几个注意点，这里文件顺序你先记住我的输出顺序，讲解一下我现在的输出顺序应该如何设计硬件读取顺序；


问题6,7你帮我解决一下。

按照你提出的方案二：直接瞄准最终演示

如果你们想最后做“MNIST 分类结果上板演示”，那就：

模型改为两层 MLP

训练用 CrossEntropyLoss

做静态量化

先用 Python 写一个整数推理参考实现

再把这个整数推理结果作为 FPGA golden进行处理。
其他应该修改的你帮我修改，和我说一声即可。
AI助手:
我先基于你现有 notebook 的结构，把方案 A、方案 B，以及静态量化、整数推理 golden、测试向量导出这几部分整理成能直接落地的代码。然后我会把硬件读取顺序也一并说明清楚，保证软件导出和 RTL 地址映射能闭环。

AI助手:
下面我直接按你的要求给出可落地代码和设计说明。  
我把内容分成两部分：

1. **路线 A 完整代码**：训练完整 MLP，但 FPGA 先只实现第一层 `fc1 + relu`，用于硬件验证  
2. **路线 B 完整代码**：直接瞄准最终演示，做完整两层 MLP、静态量化、整数推理 golden 导出

你最后明确说“**按方案二直接瞄准最终演示**”，所以我建议你真正落地时优先用**路线 B**。  
路线 A 我也完整给你，方便你前期做单层阵列验证。

---

# 一、你提出的几个问题，我已经统一改掉了

这次代码里我已经一起处理了这些问题：

## 已修改的问题
### 1）训练目标错误
- 路线 A / B 都改成了真正的分类训练
- 使用 `CrossEntropyLoss`

### 2）`zero_point` 不是整数
- 已改为**整数 zero_point**
- 并做了合法范围裁剪

### 3）动态量化不适合 FPGA
- 改为**静态量化**
- 输入、权重、输出都用**固定 scale / zero_point**
- 不再每个样本单独算量化参数

### 4）bias 不能简单 int8
- 改为：
  - `bias_q = round(bias_fp / (input_scale * weight_scale))`
  - 使用 **int32** 保存

### 5）golden 与硬件链路不一致
- 改为**整数推理参考链路**
- golden 由整数乘加、加 bias、ReLU、requant 得到
- 和 FPGA 数据通路一致

### 6）输入量化方式不适合批量验证
- 统一输入量化策略
- 采用更贴近硬件的方案：**MNIST 像素 0~1 -> uint8 0~255，再映射为 int8（减去 128）**
- 固定 `input_scale = 1/255`，`input_zero_point = -128`

### 7）其他问题我也一起修了
- `torch.load(..., map_location=device)`
- 增加 test accuracy
- 保存标签文件
- 保存中间层 int32 累加结果，便于 RTL 分级定位
- 导出 json 配置，方便 FPGA 端对齐

---

# 二、统一量化策略设计

你要求“贴近硬件即可”，我这里给你一个非常适合 FPGA 的设计：

## 输入量化
原始 MNIST 图像来自 `ToTensor()` 后范围是 `[0,1]`

统一量化为：

\[
x_q = round(x / s_x) + z_x
\]

其中：

- `s_x = 1 / 255`
- `z_x = -128`

这样就等价于：

```python
x_u8 = round(x * 255)      # 0~255
x_q  = x_u8 - 128          # -128~127
```

优点：
- 很贴近硬件
- 不需要每张图动态量化
- FPGA 端容易处理
- BRAM 存储方便

## 权重量化
对每一层权重做**对称量化**：

\[
w_q = round(w / s_w)
\]

其中：

\[
s_w = \max(|w|) / 127
\]

- `w_zero_point = 0`
- `w_q` 存为 int8

## 偏置量化
偏置必须与累加器同量纲：

\[
b_q = round(b / (s_x \cdot s_w))
\]

- 存为 int32

## 输出量化
累加后得到 int32：

\[
acc = \sum (x_q - z_x) \cdot (w_q - z_w) + b_q
\]

然后 ReLU，再 requant：

\[
y_q = round(acc \cdot (s_x s_w / s_y)) + z_y
\]

输出层如需分类 argmax，最终也可直接基于 int32 或 requant 后 int8 做。

---

# 三、路线 A：完整代码  
## 用途
- 软件训练完整分类器：`784 -> 128 -> 10`
- FPGA 先只实现第一层：`fc1 + relu`
- 导出第一层输入、权重、bias、int32 累加 golden、relu 输出 golden

这条路线适合你前期快速验证 CIM/FC 阵列。

---

## 路线 A 完整代码

```python
import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =========================================================
# 1. 配置
# =========================================================
DATA_DIR = "./data"
OUT_DIR = "./route_a_output"
MODEL_PATH = os.path.join(OUT_DIR, "mlp_route_a.pth")

os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 2. 模型：完整训练，但导出时只导出第一层用于 FPGA
# =========================================================
class MLPRouteA(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10, bias=True)

    def forward(self, x):
        x = x.view(-1, 784)
        h = self.fc1(x)
        h_relu = self.relu(h)
        y = self.fc2(h_relu)
        return y


# =========================================================
# 3. 数据
#   路线 A 训练时仍然可以用标准 Normalize
#   因为这里只是先训练出一个可用分类器
# =========================================================
def get_train_test_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        DATA_DIR, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        DATA_DIR, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# =========================================================
# 4. 训练 / 测试
# =========================================================
def train_model(num_epochs=5, lr=1e-3):
    train_loader, test_loader = get_train_test_loaders()

    model = MLPRouteA().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        acc = evaluate_model(model, test_loader)
        print(f"[Epoch {epoch+1}/{num_epochs}] loss={avg_loss:.6f}, test_acc={acc:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    return model


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            logits = model(data)
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    return correct / total


# =========================================================
# 5. 量化函数
#   路线 A：为了 FPGA 只验证第一层，采用固定输入量化策略
# =========================================================
INT8_QMIN = -128
INT8_QMAX = 127
INT32_QMIN = -2147483648
INT32_QMAX = 2147483647

def clamp(x, low, high):
    return max(low, min(high, x))

def quantize_affine_tensor(tensor_fp, scale, zero_point, qmin=-128, qmax=127, dtype=torch.int8):
    q = torch.round(tensor_fp / scale + zero_point)
    q = torch.clamp(q, qmin, qmax)
    return q.to(dtype)

def calc_symmetric_scale(tensor_fp, num_bits=8):
    qmax = (2 ** (num_bits - 1)) - 1
    max_abs = max(abs(tensor_fp.min().item()), abs(tensor_fp.max().item()))
    if max_abs == 0:
        return 1.0
    return max_abs / qmax

def quantize_weight_symmetric_int8(weight_fp):
    scale = calc_symmetric_scale(weight_fp, num_bits=8)
    zp = 0
    weight_q = quantize_affine_tensor(weight_fp, scale, zp, -128, 127, torch.int8)
    return weight_q, scale, zp

def quantize_bias_int32(bias_fp, input_scale, weight_scale):
    bias_scale = input_scale * weight_scale
    bias_q = torch.round(bias_fp / bias_scale)
    bias_q = torch.clamp(bias_q, INT32_QMIN, INT32_QMAX).to(torch.int32)
    return bias_q, bias_scale

def quantize_input_hw_style(img_fp_01):
    """
    img_fp_01: 原始像素 [0,1]
    统一策略：
        x_u8 = round(x * 255)
        x_q  = x_u8 - 128
    对应：
        input_scale = 1/255
        input_zero_point = -128
    """
    input_scale = 1.0 / 255.0
    input_zero_point = -128

    x_u8 = torch.round(img_fp_01 * 255.0)
    x_u8 = torch.clamp(x_u8, 0, 255)
    x_q = x_u8.to(torch.int16) - 128
    x_q = torch.clamp(x_q, -128, 127).to(torch.int8)

    return x_q, input_scale, input_zero_point


# =========================================================
# 6. 第一层整数推理 golden
#   只对应 fc1 + relu
# =========================================================
def fc1_integer_reference(x_q, w_q, b_q, x_zp=-128, w_zp=0):
    """
    x_q: [784] int8
    w_q: [128, 784] int8
    b_q: [128] int32

    acc_j = sum_i (x_q[i] - x_zp)*(w_q[j,i] - w_zp) + b_q[j]
    relu_acc = max(acc_j, 0)
    """
    x_int = x_q.to(torch.int32) - int(x_zp)
    w_int = w_q.to(torch.int32) - int(w_zp)

    acc = torch.matmul(w_int, x_int) + b_q
    relu_acc = torch.clamp(acc, min=0)
    return acc.to(torch.int32), relu_acc.to(torch.int32)


# =========================================================
# 7. HEX 导出
# =========================================================
def write_int8_hex_1d(tensor, path):
    with open(path, "w") as f:
        for v in tensor.flatten():
            f.write(f"{int(v.item()) & 0xFF:02x}\n")

def write_int32_hex_1d(tensor, path):
    with open(path, "w") as f:
        for v in tensor.flatten():
            f.write(f"{int(v.item()) & 0xFFFFFFFF:08x}\n")

def export_route_a_artifacts(num_samples=10):
    # 加载训练后的模型
    model = MLPRouteA().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 只提取 fc1
    fc1_w_fp = model.fc1.weight.data.cpu()   # [128, 784]
    fc1_b_fp = model.fc1.bias.data.cpu()     # [128]

    # 固定输入量化参数
    input_scale = 1.0 / 255.0
    input_zero_point = -128

    # 权重量化（对称 int8）
    fc1_w_q, fc1_w_scale, fc1_w_zp = quantize_weight_symmetric_int8(fc1_w_fp)

    # 偏置量化（int32）
    fc1_b_q, fc1_b_scale = quantize_bias_int32(fc1_b_fp, input_scale, fc1_w_scale)

    # 导出权重：按 [out][in] 顺序
    write_int8_hex_1d(fc1_w_q.reshape(-1), os.path.join(OUT_DIR, "fc1_weight_int8.hex"))
    write_int32_hex_1d(fc1_b_q, os.path.join(OUT_DIR, "fc1_bias_int32.hex"))

    # 保存配置
    config = {
        "layer": "fc1_only",
        "weight_shape": [128, 784],
        "weight_layout": "row-major [out][in]",
        "input_layout": "[in]",
        "bias_layout": "[out]",
        "input_scale": input_scale,
        "input_zero_point": input_zero_point,
        "fc1_weight_scale": fc1_w_scale,
        "fc1_weight_zero_point": fc1_w_zp,
        "fc1_bias_scale": fc1_b_scale,
        "fc1_bias_dtype": "int32"
    }

    with open(os.path.join(OUT_DIR, "quant_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 测试集：为了导出原始像素，不能 Normalize
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=test_transform)

    labels = []

    for i in range(num_samples):
        img_fp, label = test_dataset[i]  # [1,28,28], range [0,1]
        labels.append(int(label))

        img_flat = img_fp.view(-1).cpu()
        x_q, _, _ = quantize_input_hw_style(img_flat)

        acc_int32, relu_int32 = fc1_integer_reference(
            x_q, fc1_w_q, fc1_b_q, x_zp=input_zero_point, w_zp=fc1_w_zp
        )

        write_int8_hex_1d(x_q, os.path.join(OUT_DIR, f"input_{i}.hex"))
        write_int32_hex_1d(acc_int32, os.path.join(OUT_DIR, f"fc1_acc_golden_{i}.hex"))
        write_int32_hex_1d(relu_int32, os.path.join(OUT_DIR, f"fc1_relu_golden_{i}.hex"))

        print(f"Exported sample {i}, label={label}")

    with open(os.path.join(OUT_DIR, "labels.txt"), "w") as f:
        for lb in labels:
            f.write(f"{lb}\n")

    print("Route A export done.")


if __name__ == "__main__":
    train_model(num_epochs=5, lr=1e-3)
    export_route_a_artifacts(num_samples=10)
```

---

# 四、路线 B：完整代码  
## 用途
这是你现在真正应该用的版本：

- 完整两层 MLP：`784 -> 128 -> 10`
- `CrossEntropyLoss`
- 静态量化
- Python 写整数推理参考实现
- 导出 FPGA golden
- 最终瞄准“MNIST 上板分类演示”

---

## 路线 B 完整代码

```python
import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =========================================================
# 1. 配置
# =========================================================
DATA_DIR = "./data"
OUT_DIR = "./route_b_output"
MODEL_PATH = os.path.join(OUT_DIR, "mlp_route_b.pth")

os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INT8_QMIN = -128
INT8_QMAX = 127
INT32_QMIN = -2147483648
INT32_QMAX = 2147483647


# =========================================================
# 2. 模型
# =========================================================
class MLPRouteB(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10, bias=True)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# =========================================================
# 3. 数据
#   训练时这里直接用 ToTensor()，不做 Normalize
#   这样更贴近最终硬件部署链路
# =========================================================
def get_train_test_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# =========================================================
# 4. 训练 / 测试
# =========================================================
def evaluate_model(model, test_loader):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            logits = model(data)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    return correct / total


def train_model(num_epochs=8, lr=1e-3):
    train_loader, test_loader = get_train_test_loaders()

    model = MLPRouteB().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        acc = evaluate_model(model, test_loader)
        print(f"[Epoch {epoch+1}/{num_epochs}] loss={avg_loss:.6f}, test_acc={acc:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    return model


# =========================================================
# 5. 量化工具
# =========================================================
def clamp(v, low, high):
    return max(low, min(high, v))

def quantize_affine_tensor(tensor_fp, scale, zero_point, qmin=-128, qmax=127, dtype=torch.int8):
    q = torch.round(tensor_fp / scale + zero_point)
    q = torch.clamp(q, qmin, qmax)
    return q.to(dtype)

def calc_symmetric_scale(tensor_fp, num_bits=8):
    qmax = (2 ** (num_bits - 1)) - 1
    max_abs = max(abs(tensor_fp.min().item()), abs(tensor_fp.max().item()))
    if max_abs == 0:
        return 1.0
    return max_abs / qmax

def calc_affine_params_from_range(min_val, max_val, num_bits=8, signed=True):
    if signed:
        qmin = -(2 ** (num_bits - 1))
        qmax =  (2 ** (num_bits - 1)) - 1
    else:
        qmin = 0
        qmax = (2 ** num_bits) - 1

    if max_val == min_val:
        scale = 1.0
        zero_point = 0
        return scale, zero_point

    scale = (max_val - min_val) / float(qmax - qmin)
    zero_point = round(qmin - min_val / scale)
    zero_point = int(clamp(zero_point, qmin, qmax))
    return float(scale), int(zero_point)

def quantize_weight_symmetric_int8(weight_fp):
    scale = calc_symmetric_scale(weight_fp, num_bits=8)
    zp = 0
    q = quantize_affine_tensor(weight_fp, scale, zp, INT8_QMIN, INT8_QMAX, torch.int8)
    return q, scale, zp

def quantize_bias_int32(bias_fp, input_scale, weight_scale):
    bias_scale = input_scale * weight_scale
    q = torch.round(bias_fp / bias_scale)
    q = torch.clamp(q, INT32_QMIN, INT32_QMAX).to(torch.int32)
    return q, bias_scale

def quantize_input_hw_style(img_fp_01):
    """
    固定输入量化：
        x_u8 = round(x * 255)
        x_q  = x_u8 - 128
    对应：
        scale = 1/255
        zero_point = -128
    """
    scale = 1.0 / 255.0
    zp = -128

    x_u8 = torch.round(img_fp_01 * 255.0)
    x_u8 = torch.clamp(x_u8, 0, 255)
    x_q = x_u8.to(torch.int16) - 128
    x_q = torch.clamp(x_q, -128, 127).to(torch.int8)
    return x_q, scale, zp

def choose_output_scale_from_activation(act_fp):
    """
    输出激活采用对称量化到 int8
    """
    scale = calc_symmetric_scale(act_fp, num_bits=8)
    zp = 0
    return scale, zp

def requantize_from_int32(acc_int32, real_multiplier, out_zero_point=0):
    """
    简单软件参考：
        y_q = round(acc * real_multiplier) + out_zero_point
    """
    y = torch.round(acc_int32.to(torch.float32) * real_multiplier) + out_zero_point
    y = torch.clamp(y, INT8_QMIN, INT8_QMAX)
    return y.to(torch.int8)

def quantize_multiplier_to_int(real_multiplier):
    """
    给 FPGA 做定点乘法时可用的近似参数：
      real_multiplier ≈ M / 2^shift
    返回:
      M(int), shift(int)
    这里只提供给配置文件参考；软件 golden 仍直接用 float real_multiplier
    """
    if real_multiplier == 0:
        return 0, 0

    shift = 0
    m = real_multiplier
    while m < 0.5:
        m *= 2.0
        shift += 1
    M = int(round(m * (1 << 31)))  # Q31
    return M, shift + 31


# =========================================================
# 6. 量化模型准备
# =========================================================
def build_quantized_model(model_fp):
    model_fp.eval()
    model_fp = model_fp.cpu()

    # 固定输入量化参数
    input_scale = 1.0 / 255.0
    input_zero_point = -128

    # fc1 量化
    fc1_w_fp = model_fp.fc1.weight.data
    fc1_b_fp = model_fp.fc1.bias.data
    fc1_w_q, fc1_w_scale, fc1_w_zp = quantize_weight_symmetric_int8(fc1_w_fp)
    fc1_b_q, fc1_b_scale = quantize_bias_int32(fc1_b_fp, input_scale, fc1_w_scale)

    # 用一批训练样本统计 fc1 输出尺度
    calib_loader, _ = get_train_test_loaders(batch_size=256)
    calib_data, _ = next(iter(calib_loader))
    calib_data = calib_data.view(-1, 784)
    with torch.no_grad():
        fc1_act_fp = model_fp.relu(model_fp.fc1(calib_data))
    fc1_out_scale, fc1_out_zp = choose_output_scale_from_activation(fc1_act_fp)

    fc1_real_multiplier = (input_scale * fc1_w_scale) / fc1_out_scale
    fc1_M, fc1_shift = quantize_multiplier_to_int(fc1_real_multiplier)

    # fc2 量化
    fc2_w_fp = model_fp.fc2.weight.data
    fc2_b_fp = model_fp.fc2.bias.data
    fc2_w_q, fc2_w_scale, fc2_w_zp = quantize_weight_symmetric_int8(fc2_w_fp)
    fc2_b_q, fc2_b_scale = quantize_bias_int32(fc2_b_fp, fc1_out_scale, fc2_w_scale)

    # 用一批样本统计 fc2 输出尺度
    with torch.no_grad():
        logits_fp = model_fp(calib_data.view(-1, 1, 28, 28))
    fc2_out_scale, fc2_out_zp = choose_output_scale_from_activation(logits_fp)

    fc2_real_multiplier = (fc1_out_scale * fc2_w_scale) / fc2_out_scale
    fc2_M, fc2_shift = quantize_multiplier_to_int(fc2_real_multiplier)

    qparams = {
        "input": {
            "scale": input_scale,
            "zero_point": input_zero_point,
            "dtype": "int8"
        },
        "fc1": {
            "weight_scale": fc1_w_scale,
            "weight_zero_point": fc1_w_zp,
            "bias_scale": fc1_b_scale,
            "output_scale": fc1_out_scale,
            "output_zero_point": fc1_out_zp,
            "real_multiplier": fc1_real_multiplier,
            "multiplier_q31": fc1_M,
            "shift": fc1_shift
        },
        "fc2": {
            "weight_scale": fc2_w_scale,
            "weight_zero_point": fc2_w_zp,
            "bias_scale": fc2_b_scale,
            "output_scale": fc2_out_scale,
            "output_zero_point": fc2_out_zp,
            "real_multiplier": fc2_real_multiplier,
            "multiplier_q31": fc2_M,
            "shift": fc2_shift
        }
    }

    quantized = {
        "fc1_w_q": fc1_w_q.to(torch.int8),
        "fc1_b_q": fc1_b_q.to(torch.int32),
        "fc2_w_q": fc2_w_q.to(torch.int8),
        "fc2_b_q": fc2_b_q.to(torch.int32),
        "qparams": qparams
    }
    return quantized


# =========================================================
# 7. 整数推理参考实现
# =========================================================
def linear_int8_int32(x_q, w_q, b_q, x_zp, w_zp):
    """
    x_q: [in_features] int8
    w_q: [out_features, in_features] int8
    b_q: [out_features] int32
    """
    x_int = x_q.to(torch.int32) - int(x_zp)
    w_int = w_q.to(torch.int32) - int(w_zp)
    acc = torch.matmul(w_int, x_int) + b_q
    return acc.to(torch.int32)

def relu_int32(x):
    return torch.clamp(x, min=0).to(torch.int32)

def integer_inference_reference(img_fp_01, quantized):
    """
    完整整数参考链路：
      input -> fc1(int32 acc) -> relu -> requant to int8
            -> fc2(int32 acc) -> requant to int8 logits
    """
    qparams = quantized["qparams"]

    # 1) 输入量化
    x_q, _, _ = quantize_input_hw_style(img_fp_01)

    # 2) fc1 int32 累加
    fc1_acc = linear_int8_int32(
        x_q,
        quantized["fc1_w_q"],
        quantized["fc1_b_q"],
        x_zp=qparams["input"]["zero_point"],
        w_zp=qparams["fc1"]["weight_zero_point"]
    )

    # 3) ReLU
    fc1_relu = relu_int32(fc1_acc)

    # 4) fc1 输出 requant -> int8
    fc1_out_q = requantize_from_int32(
        fc1_relu,
        qparams["fc1"]["real_multiplier"],
        qparams["fc1"]["output_zero_point"]
    )

    # 5) fc2 int32 累加
    fc2_acc = linear_int8_int32(
        fc1_out_q,
        quantized["fc2_w_q"],
        quantized["fc2_b_q"],
        x_zp=qparams["fc1"]["output_zero_point"],
        w_zp=qparams["fc2"]["weight_zero_point"]
    )

    # 6) 输出 requant -> int8 logits
    logits_q = requantize_from_int32(
        fc2_acc,
        qparams["fc2"]["real_multiplier"],
        qparams["fc2"]["output_zero_point"]
    )

    pred = int(torch.argmax(logits_q).item())

    return {
        "input_q": x_q,
        "fc1_acc_int32": fc1_acc,
        "fc1_relu_int32": fc1_relu,
        "fc1_out_q": fc1_out_q,
        "fc2_acc_int32": fc2_acc,
        "logits_q": logits_q,
        "pred": pred
    }


# =========================================================
# 8. HEX 导出
# =========================================================
def write_int8_hex_1d(tensor, path):
    with open(path, "w") as f:
        for v in tensor.flatten():
            f.write(f"{int(v.item()) & 0xFF:02x}\n")

def write_int32_hex_1d(tensor, path):
    with open(path, "w") as f:
        for v in tensor.flatten():
            f.write(f"{int(v.item()) & 0xFFFFFFFF:08x}\n")

def export_all_artifacts(model_fp, quantized, num_samples=20):
    # 1) 导出权重 / bias
    # 按 [out][in] 顺序导出
    write_int8_hex_1d(quantized["fc1_w_q"].reshape(-1), os.path.join(OUT_DIR, "fc1_weight_int8.hex"))
    write_int32_hex_1d(quantized["fc1_b_q"], os.path.join(OUT_DIR, "fc1_bias_int32.hex"))

    write_int8_hex_1d(quantized["fc2_w_q"].reshape(-1), os.path.join(OUT_DIR, "fc2_weight_int8.hex"))
    write_int32_hex_1d(quantized["fc2_b_q"], os.path.join(OUT_DIR, "fc2_bias_int32.hex"))

    # 2) 保存量化配置
    config = {
        "network": "784 -> 128 -> 10",
        "weight_layout": {
            "fc1": "row-major [out][in] = [128][784]",
            "fc2": "row-major [out][in] = [10][128]"
        },
        "input_layout": "[784]",
        "hidden_layout": "[128]",
        "output_layout": "[10]",
        "qparams": quantized["qparams"]
    }

    with open(os.path.join(OUT_DIR, "quant_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 3) 导出样本和 golden
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=test_transform)

    labels = []
    preds = []

    for i in range(num_samples):
        img_fp, label = test_dataset[i]
        img_flat = img_fp.view(-1).cpu()

        result = integer_inference_reference(img_flat, quantized)

        labels.append(int(label))
        preds.append(int(result["pred"]))

        write_int8_hex_1d(result["input_q"], os.path.join(OUT_DIR, f"input_{i}.hex"))

        write_int32_hex_1d(result["fc1_acc_int32"], os.path.join(OUT_DIR, f"fc1_acc_{i}.hex"))
        write_int32_hex_1d(result["fc1_relu_int32"], os.path.join(OUT_DIR, f"fc1_relu_{i}.hex"))
        write_int8_hex_1d(result["fc1_out_q"], os.path.join(OUT_DIR, f"fc1_out_{i}.hex"))

        write_int32_hex_1d(result["fc2_acc_int32"], os.path.join(OUT_DIR, f"fc2_acc_{i}.hex"))
        write_int8_hex_1d(result["logits_q"], os.path.join(OUT_DIR, f"logits_{i}.hex"))

        with open(os.path.join(OUT_DIR, f"pred_{i}.txt"), "w") as f:
            f.write(str(result["pred"]))

        print(f"sample={i}, label={label}, pred={result['pred']}")

    with open(os.path.join(OUT_DIR, "labels.txt"), "w") as f:
        for lb in labels:
            f.write(f"{lb}\n")

    with open(os.path.join(OUT_DIR, "preds.txt"), "w") as f:
        for pd in preds:
            f.write(f"{pd}\n")

    print("All artifacts exported.")


# =========================================================
# 9. 验证量化后整数推理精度
# =========================================================
def evaluate_integer_pipeline(model_fp, quantized, num_samples=1000):
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=test_transform)

    correct = 0
    total = min(num_samples, len(test_dataset))

    for i in range(total):
        img_fp, label = test_dataset[i]
        img_flat = img_fp.view(-1).cpu()
        result = integer_inference_reference(img_flat, quantized)
        if result["pred"] == int(label):
            correct += 1

    acc = correct / total
    print(f"Integer pipeline accuracy over {total} samples: {acc:.4f}")
    return acc


# =========================================================
# 10. 主流程
# =========================================================
if __name__ == "__main__":
    # 1) 训练浮点模型
    model_fp = train_model(num_epochs=8, lr=1e-3)

    # 2) 加载模型（防止单独运行导出时也能用）
    model_fp = MLPRouteB().to(DEVICE)
    model_fp.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model_fp.eval()

    # 3) 构建量化模型
    quantized = build_quantized_model(model_fp)

    # 4) 评估整数链路
    evaluate_integer_pipeline(model_fp, quantized, num_samples=1000)

    # 5) 导出 FPGA 所需文件
    export_all_artifacts(model_fp, quantized, num_samples=20)
```

---

# 五、关于问题 2：逐项回答

---

## 2.1 `zero_point` 不是整数，怎么改

你原来的问题在这里：

```python
zero_point = qmin - min_val / scale
```

这里会得到浮点数。应改成：

```python
zero_point = round(qmin - min_val / scale)
zero_point = int(max(qmin, min(qmax, zero_point)))
```

通用函数我已经在路线 B 里给了：

```python
def calc_affine_params_from_range(min_val, max_val, num_bits=8, signed=True):
    if signed:
        qmin = -(2 ** (num_bits - 1))
        qmax =  (2 ** (num_bits - 1)) - 1
    else:
        qmin = 0
        qmax = (2 ** num_bits) - 1

    if max_val == min_val:
        scale = 1.0
        zero_point = 0
        return scale, zero_point

    scale = (max_val - min_val) / float(qmax - qmin)
    zero_point = round(qmin - min_val / scale)
    zero_point = int(clamp(zero_point, qmin, qmax))
    return float(scale), int(zero_point)
```

---

## 2.2 每个张量动态量化不适合 FPGA，合理解决方案

### 你现在的问题
你现在是：
- 每张输入图像单独量化
- 每次输出也单独量化

这会导致 FPGA 需要不断切换 scale / zero_point，不利于验证和部署。

### 合理方案
改为**静态量化**：

#### 输入
固定：
- `input_scale = 1/255`
- `input_zero_point = -128`

#### 权重
每层固定：
- `weight_scale` 固定
- `weight_zero_point = 0`

#### 输出
每层固定：
- `output_scale` 固定
- `output_zero_point` 固定

这样每个样本都共用同一套参数，硬件最容易做。

---

## 2.3 偏置不能用 int8，怎么解决

### 正确原则
bias 必须与 int32 累加器同量纲：

\[
b_q = round(\frac{b}{s_x \cdot s_w})
\]

### 代码
我已经写成：

```python
def quantize_bias_int32(bias_fp, input_scale, weight_scale):
    bias_scale = input_scale * weight_scale
    q = torch.round(bias_fp / bias_scale)
    q = torch.clamp(q, INT32_QMIN, INT32_QMAX).to(torch.int32)
    return q, bias_scale
```

### 为什么要这样
因为硬件里先做：

\[
acc = \sum (x_q-z_x)(w_q-z_w)
\]

这个 `acc` 是 int32 量纲。  
如果 bias 还是 int8，直接相加就是错的。

---

# 六、关于问题 3：整数推理链路 golden，怎么设计

你要求“整数推理链路生成 golden”，现在路线 B 已经完整实现了。

---

## 设计链路

### 第一层
\[
acc_1 = W_1 \cdot (x_q-z_x) + b_{1q}
\]

\[
relu_1 = max(acc_1, 0)
\]

\[
h_q = requant(relu_1)
\]

### 第二层
\[
acc_2 = W_2 \cdot (h_q-z_{h}) + b_{2q}
\]

\[
logits_q = requant(acc_2)
\]

### 分类
\[
pred = argmax(logits_q)
\]

---

## 代码入口
整数参考实现就在这里：

```python
def integer_inference_reference(img_fp_01, quantized):
    ...
```

它导出了：

- `input_q`
- `fc1_acc_int32`
- `fc1_relu_int32`
- `fc1_out_q`
- `fc2_acc_int32`
- `logits_q`
- `pred`

这些就是你硬件联调最有价值的 golden。

---

# 七、关于问题 4：统一输入量化策略

你要求“贴近硬件即可”，我建议你就用下面这个，不要再 Normalize。

---

## 推荐方案
MNIST 原始像素 `[0,1]`

统一做：

```python
x_u8 = round(x * 255)
x_q = x_u8 - 128
```

对应：

- `input_scale = 1/255`
- `input_zero_point = -128`

---

## 优点
### 软件上
- 简单
- 稳定
- 不依赖每张图的分布

### 硬件上
- BRAM 存储直观
- 每个像素 8 bit
- 乘加前减去 `-128` 很明确
- 适合 testbench 和最终上板

---

# 八、关于问题 5：你当前输出顺序如何设计硬件读取顺序

你现在权重导出顺序是：

```python
for row in q_weight:
    for val in row:
        f.write(...)
```

而 `nn.Linear(in_features, out_features)` 在 PyTorch 里权重 shape 是：

\[
[out\_features, in\_features]
\]

所以你现在导出的顺序是：

\[
w[0,0], w[0,1], ..., w[0,783], w[1,0], ..., w[127,783]
\]

也就是：

## 当前输出顺序
**按输出神经元逐行展开，row-major `[out][in]`**

---

## 对应的硬件读取顺序应该怎么设计

如果你的第一层是：

\[
y_j = \sum_{i=0}^{783} x_i \cdot w_{j,i} + b_j
\]

那么最自然的硬件读取顺序就是：

### 方案：按神经元顺序读
对于每个输出神经元 `j`：

1. 读取该神经元的 784 个权重
2. 与输入 784 维逐项相乘累加
3. 加偏置 `b_j`
4. 输出 `y_j`

### 地址映射
如果 `fc1_weight_int8.hex` 按 `[out][in]` 展开，则：

\[
addr = j \times 784 + i
\]

即：
- `j`：输出神经元编号
- `i`：输入特征编号

### RTL 里可写成
- 外层循环：`j = 0..127`
- 内层循环：`i = 0..783`

这和你当前导出顺序是完全匹配的。

---

## 如果硬件是并行阵列怎么办
如果你做的是 CIM 阵列或 PE 并行，也仍然建议逻辑上保持这个顺序，只是物理存储时可分块：

例如分成 16 路并行：
- 每次读同一个 `j` 的 16 个连续权重
- 输入也同步读 16 个元素
- 做 16 路乘加
- 局部累加后再归并

本质上还是从 `[out][in]` 顺序切块读取。

---

## 第二层同理
第二层 `fc2` shape 是 `[10,128]`

地址映射：

\[
addr = j \times 128 + i
\]

其中：
- `j = 0..9`
- `i = 0..127`

---

# 九、问题 6 和 7，我已经一起解决了

---

## 问题 6：`quant_params.npz` 没真正参与闭环
现在我不建议继续用 `npz` 做核心配置。  
我已经改成了：

- `quant_config.json`

里面包含：
- 输入 scale / zp
- 每层 weight_scale / weight_zp
- bias_scale
- output_scale / output_zp
- real_multiplier
- multiplier_q31
- shift

这对硬件更友好，也更好查。

---

## 问题 7：其他工程问题
我已经一起处理：

### 1）`torch.load()` 增加 `map_location`
已改：

```python
torch.load(MODEL_PATH, map_location=DEVICE)
```

### 2）训练时增加准确率
每个 epoch 后打印 test accuracy

### 3）保存 label
已导出：
- `labels.txt`

### 4）保存中间层 golden
已导出：
- `fc1_acc_x.hex`
- `fc1_relu_x.hex`
- `fc1_out_x.hex`
- `fc2_acc_x.hex`
- `logits_x.hex`

这样 RTL 调试时你能定位是哪一级错了。

---

# 十、你最终应该选哪套

## 如果你现在要交阶段成果、先验证阵列
用路线 A。  
因为它最容易把硬件链路先跑通。

## 如果你现在就想直奔“MNIST 上板分类演示”
用路线 B。  
这就是你刚刚指定的目标，我也已经按这个目标把代码写全了。

---

# 十一、我建议你下一步的落地顺序

你现在最稳的执行顺序是：

### 第一步
先跑路线 B 的 Python 脚本，得到：
- 浮点模型精度
- 整数量化链路精度
- 全套导出文件

### 第二步
先在 FPGA 上只实现第一层并对齐：
- `input`
- `fc1_weight`
- `fc1_bias`
- `fc1_acc`
- `fc1_relu`
- `fc1_out`

### 第三步
再补第二层：
- `fc2_weight`
- `fc2_bias`
- `fc2_acc`
- `logits`
- `argmax`

---

如果你愿意，我下一条可以继续直接给你一份**更适合 notebook 使用的版本**，也就是把路线 B 拆成多个单元格，按你现在 `ipynb` 的风格排好。

用户:
现在已经把导出测试向量的部分写好了，请帮我分析导出的文件的格式，以及如何使用
AI助手:


AI助手:
我先把你当前导出的目录和文件内容抽样检查一下，确认每类文件到底是几位十六进制、按什么维度展开、对应哪一级数据通路。然后我会按“文件格式 → RTL/Testbench 怎么读 → 怎么对拍”的顺序给你整理清楚。

AI助手:
我根据你现在 notebook 里的导出代码，已经能把**导出文件格式**和**使用方法**说清楚了。你当前实际导出的文件主要有这几类：

- `weights.hex`
- `bias.hex`
- `quant_params.npz`
- `input_i.hex`
- `golden_i.hex`

下面我按“是什么、怎么组织、硬件怎么读、联调怎么用”来讲。

---

# 1. 你现在导出的文件格式

## 1.1 `weights.hex`

这是第一层全连接层 `fc1.weight` 的量化权重文件。

### 来源
代码里：

```python
weight = model.fc1.weight.data  # [128, 784]
...
with open(f'{output_dir}/weights.hex', 'w') as f:
    for row in q_weight:
        for val in row:
            f.write(f'{val.item() & 0xFF:02x}\n')
```

### 文件格式
- 文本文件
- **每行一个 8 bit 数值**
- 用 **2 位十六进制** 表示
- 负数采用 **8 位二补码的低 8 位**

例如：
- `00` 表示 0
- `01` 表示 1
- `7f` 表示 127
- `ff` 表示 -1
- `80` 表示 -128

### 总行数
因为权重 shape 是 `[128, 784]`，所以总共有：

\[
128 \times 784 = 100352
\]

也就是：

- `weights.hex` 一共 **100352 行**

### 展开顺序
当前是：

```python
for row in q_weight:
    for val in row:
```

所以顺序是：

\[
w[0,0], w[0,1], ..., w[0,783], w[1,0], ..., w[127,783]
\]

也就是：

- **按输出神经元 row-major 展开**
- 格式为 **`[out][in]`**

这点非常重要，RTL 必须按这个顺序取。

---

## 1.2 `bias.hex`

这是第一层偏置文件。

### 来源
```python
bias = model.fc1.bias.data      # [128]
...
with open(f'{output_dir}/bias.hex', 'w') as f:
    for val in q_bias:
        f.write(f'{val.item() & 0xFF:02x}\n')
```

### 文件格式
- 文本文件
- 每行一个量化后的偏置
- **2 位十六进制**
- 当前代码里它被导成了 **int8**

### 总行数
- 偏置长度是 128
- 所以 `bias.hex` 一共 **128 行**

### 当前问题
这个文件虽然格式上没问题，但**工程上有问题**：

- 你现在是把 bias 当 int8 存
- 但硬件 MAC 累加结果通常是 int32
- 所以这个 `bias.hex` 现在**不适合直接用于真实整数推理链路**

也就是说：
- 这个文件可以先拿来做“形式演示”
- 但不适合做你后面要的最终整数 golden 对拍

---

## 1.3 `quant_params.npz`

这是量化参数文件。

### 来源
```python
np.savez(f'{output_dir}/quant_params.npz',
         w_scale=w_scale, w_zero=w_zero,
         b_scale=b_scale, b_zero=b_zero)
```

### 里面有什么
当前保存了：

- `w_scale`
- `w_zero`
- `b_scale`
- `b_zero`

### 用途
理论上它用于：
- 软件端反量化
- 硬件端知道量化参数

### 当前实际情况
你现在的 `generate_test_vectors()` 虽然加载了它：

```python
quant_params = np.load('quantized_data/quant_params.npz')
```

但后面**几乎没真正用上**。  
所以它现在更像一个“记录文件”，还不是硬件链路真正依赖的配置文件。

---

## 1.4 `input_i.hex`

这是第 `i` 个测试样本的输入文件。

### 来源
```python
img, label = test_dataset[i]
img_flat = img.view(-1, 784)
q_img, img_scale, img_zero = quantize_tensor(img_flat)

with open(f'quantized_data/input_{i}.hex', 'w') as f:
    for val in q_img.flatten():
        f.write(f'{val.item() & 0xFF:02x}\n')
```

### 文件格式
- 文本文件
- 每行一个输入元素
- **2 位十六进制**
- 量化后为 int8

### 总行数
MNIST 一张图是 `28×28=784`，所以：

- 每个 `input_i.hex` 一共 **784 行**

### 排列顺序
因为：

```python
img_flat = img.view(-1, 784)
for val in q_img.flatten():
```

所以顺序就是图像展平后的顺序，通常是：

- 第 0 行：像素 `(0,0)`
- 第 1 行：像素 `(0,1)`
- ...
- 第 27 行：像素 `(0,27)`
- 第 28 行：像素 `(1,0)`
- ...

也就是：

- **按行优先 row-major 展开**
- 输入向量索引 `i = row * 28 + col`

RTL 里要按这个顺序喂入。

---

## 1.5 `golden_i.hex`

这是第 `i` 个测试样本的软件 golden 输出。

### 来源
```python
with torch.no_grad():
    output = model(img_flat)

q_output, _, _ = quantize_tensor(output)
with open(f'quantized_data/golden_{i}.hex', 'w') as f:
    for val in q_output.flatten():
        f.write(f'{val.item() & 0xFF:02x}\n')
```

### 文件格式
- 文本文件
- 每行一个输出元素
- **2 位十六进制**
- 量化后为 int8

### 总行数
你现在模型是：

```python
self.fc1 = nn.Linear(784,128)
self.relu = nn.ReLU()
```

所以输出是 128 维。

也就是：
- 每个 `golden_i.hex` 一共 **128 行**

### 当前含义
这个 `golden_i.hex` 表示：

- 先用浮点模型跑出 `fc1 + relu` 的浮点输出
- 再把这个浮点输出单独量化成 int8

所以它的意义是：

- **“浮点模型输出的量化结果”**
- 不是严格意义上的“整数硬件链路 golden”

这点你后面做最终方案时必须记住。

---

# 2. 这些文件如何使用

---

## 2.1 最直接的使用方式：Verilog `$readmemh`

因为这些 `.hex` 文件是：
- 文本
- 每行一个十六进制数

所以最适合直接在 Verilog 里用：

```verilog
reg [7:0] weight_mem [0:100351];
reg [7:0] bias_mem   [0:127];
reg [7:0] input_mem  [0:783];
reg [7:0] golden_mem [0:127];

initial begin
    $readmemh("weights.hex", weight_mem);
    $readmemh("bias.hex", bias_mem);
    $readmemh("input_0.hex", input_mem);
    $readmemh("golden_0.hex", golden_mem);
end
```

---

## 2.2 在 RTL 里如何解释这些 8 bit 数据

### 注意
`$readmemh` 读进来的是 `reg [7:0]`，本身只是无符号位模式。  
如果你想让它参与有符号运算，需要做符号解释。

例如：

```verilog
wire signed [7:0] w = weight_mem[addr];
wire signed [7:0] x = input_mem[idx];
wire signed [7:0] b = bias_mem[out_idx];
```

或者：

```verilog
wire signed [7:0] w = $signed(weight_mem[addr]);
```

这样：
- `8'hff` 才会被解释成 `-1`
- `8'h80` 才会被解释成 `-128`

---

## 2.3 第一层硬件如何读取 `weights.hex`

你当前权重顺序是：

\[
w[j][i]
\]

其中：
- `j = 0..127`，输出神经元
- `i = 0..783`，输入索引

所以硬件最自然的读取方式是：

### 地址公式
\[
addr = j \times 784 + i
\]

### 计算流程
对于每个输出神经元 `j`：

1. 累加器清零
2. 从 `i=0` 到 `783`
3. 读取：
   - `x = input_mem[i]`
   - `w = weight_mem[j*784+i]`
4. 做乘加：
   \[
   acc_j += x \cdot w
   \]
5. 循环结束后加偏置：
   \[
   acc_j += bias[j]
   \]
6. 过 ReLU
7. 输出第 `j` 个结果

### 伪代码
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

---

## 2.4 `input_i.hex` 如何使用

### 在 testbench 中
每次测试一个样本：

```verilog
$readmemh("input_0.hex", input_mem);
```

然后把 `input_mem[0:783]` 依次送给你的计算模块。

### 两种常见喂法
#### 方式 1：一次性 preload 到 BRAM
如果你的设计是：
- 输入先装入片上 RAM
- 再启动计算

那就：
1. 先把 `input_i.hex` 用 `$readmemh` 读入 testbench memory
2. 再驱动写接口，把 784 个数据写入 DUT 内部 input RAM
3. 拉高 `start`

#### 方式 2：流式输入
如果你的 DUT 是流式输入：
- 每拍输入一个 `x_i`

那就：
1. 从 `input_mem[0]` 到 `input_mem[783]` 依次送
2. 每个时钟给一个有效数据
3. 送满 784 个后等待结果

---

## 2.5 `golden_i.hex` 如何使用

这是当前 testbench 最直接的对拍基准。

### 用法
在 testbench 中：

```verilog
$readmemh("golden_0.hex", golden_mem);
```

当 DUT 计算完成后，把 DUT 的 128 个输出与 `golden_mem` 逐项比对。

### 比对方式
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

### 但要注意
你现在的 `golden_i.hex` 是：
- 浮点输出再量化

所以前提是你的硬件计算链路也得尽量模拟这个逻辑。  
如果你 RTL 里已经做成了严格整数推理，那这个 golden 就不再是最优基准了。

---

# 3. 你当前这套文件更适合什么阶段

这套导出结果，最适合：

## 适合
- 第一层 `fc1 + relu` 的硬件原型验证
- BRAM 读写流程验证
- 地址映射验证
- 符号位解释验证
- testbench 对拍基本通路

## 不适合
- 最终的完整 MNIST 分类演示
- 严格的整数推理验证
- 偏置 int32 链路验证
- 静态量化部署闭环

---

# 4. 推荐你怎么用这批文件做联调

---

## 阶段 1：先验证存储与读取
先只做三件事：

1. `weights.hex` 是否正确加载
2. `input_i.hex` 是否正确加载
3. `golden_i.hex` 是否能和当前 RTL 输出逐项对上

只要这一步过了，说明：
- 文件格式没问题
- 读写顺序没问题
- 地址映射没问题

---

## 阶段 2：确认你的计算顺序与导出顺序一致
你需要确认 RTL 的权重寻址是不是：

\[
addr = out\_idx \times 784 + in\_idx
\]

如果不是，就会出现：
- 数据都能读到
- 结果却完全不对

这是最常见的问题。

---

## 阶段 3：确认 signed 解释
这是第二常见问题。  
如果 RTL 里没有把 `8'hff` 当成 `-1`，而是当成 `255`，结果一定炸。

所以一定要检查：
- 输入 signed
- 权重 signed
- 偏置 signed
- 乘法 signed
- 累加器 signed

---

# 5. 我建议你现在这样理解每个文件

你可以把当前文件理解成这张表：

| 文件 | 含义 | 数据类型 | 行数 | 作用 |
|---|---|---:|---:|---|
| `weights.hex` | `fc1` 权重 | int8 | 100352 | DUT 权重存储初始化 |
| `bias.hex` | `fc1` 偏置 | int8 | 128 | 当前版本偏置初始化 |
| `quant_params.npz` | 量化参数记录 | float/int | 少量字段 | 软件辅助，不是闭环核心 |
| `input_i.hex` | 第 i 个输入图像 | int8 | 784 | 测试输入 |
| `golden_i.hex` | 第 i 个参考输出 | int8 | 128 | testbench 对拍 |

---

# 6. 你现在这批文件的“正确使用姿势”

如果你此刻不改导出逻辑，而是先拿现有文件验证硬件，那建议这样：

### 你现在的 DUT 目标
做一个：
- `784 -> 128`
- int8 输入
- int8 权重
- int8 偏置
- 累加
- ReLU
- 最后输出和 `golden_i.hex` 尽量一致

### testbench 流程
1. `$readmemh("weights.hex", weight_mem)`
2. `$readmemh("bias.hex", bias_mem)`
3. `$readmemh("input_0.hex", input_mem)`
4. `$readmemh("golden_0.hex", golden_mem)`
5. 把 `input_mem` 送入 DUT
6. DUT 输出 128 维结果
7. 与 `golden_mem` 逐项比较

这样可以先把系统跑起来。

---

# 7. 你后面升级到最终方案时，文件会怎么变

你后面如果按我们前面定的最终方案升级，文件会改成更合理的结构：

- `fc1_weight_int8.hex`
- `fc1_bias_int32.hex`
- `fc2_weight_int8.hex`
- `fc2_bias_int32.hex`
- `input_i.hex`
- `fc1_acc_i.hex`
- `fc1_relu_i.hex`
- `fc1_out_i.hex`
- `fc2_acc_i.hex`
- `logits_i.hex`
- `pred_i.txt`
- `quant_config.json`

这时每个文件的职责会更清晰，硬件对拍也更容易定位问题。

---

# 8. 你现在最需要注意的两个坑

## 坑 1：`bias.hex` 现在是 int8
这不是最终可部署形式。  
所以你现在用它时，只能把它当作“当前软件逻辑下的配套文件”，不要把它当最终硬件部署格式。

## 坑 2：`golden_i.hex` 不是整数推理 golden
它现在只是：
- 浮点输出
- 再量化

所以如果你 RTL 做得特别“硬件真实”，反而可能和它对不上。

---

如果你把**已经导出的几个 `.hex` 文件内容**贴给我几行，或者把你当前 `quantized_data/` 目录再发我，我可以继续帮你做第二步：**逐个文件按实际内容检查格式是否正确，并给你写一个对应的 Verilog testbench 读取模板**。

用户:
bias.hex:000008e5
0000183f
000001a2
ffffe6b8
fffffa66
fffffdd6
00000acd
00000e73
fffff423
00000244
fffff947
00000cc4
fffff784
0000061c
ffffed80
00000dd7
fffffffe
fffff9de
fffffb86
ffffed26
00000c70
0000116d
ffffff7f
000001a8
0000049e
0000118f
000006c6
00000e32
000025f7
ffffe677
000005b3
00000b83
00002162
00001ce9
00000409
fffffbcc
00000487
00000caa
fffff5f9
fffff633
000003b1
00000fd1
000005d1
00000ce3
0000106f
00000c5a
0000019f
00000681
ffffef2b
fffff49a
ffffe4d9
fffffb73
0000019f
00000dd1
0000016b
00000730
fffff9a5
000009db
000008ec
000003c1
00000b59
00001263
fffff71a
00001706
000013f9
000001e7
000027ca
00000779
000000d3
fffffe21
0000051c
00000e9c
00000c2f
fffffc59
0000142d
00000956
00000813
fffffa40
00000d2a
00000c81
fffffd14
000008ee
0000144a
fffff816
00000be6
00000850
000012d6
00001bd6
00000388
fffffc00
00001590
00000565
00000adb
00000f82
00000242
00001ed3
fffff3bf
00001420
00000b0e
00000ac2
fffffefd
00000072
fffff4f2
fffffc4f
00000904
00000a0f
0000078d
00000cba
fffffb5e
fffffe7a
00000a64
00000e22
00000435
00000642
00002492
00002235
000011d9
fffff921
ffffff6a
00001402
00001a36
0000002e
00001f8d
ffffff3e
fffff936
00000023
fffffe93
fffff438
weight.hex:aa
e5
00
0d
0d
f7
08
e8
ff
20
15
c5
fe
fb
e2
f9
16
15
fc
15
09
13
09
cb
28
ca
0f
15
0f
05
16
f3
d7
f6
06
0c
d8
1e
ef
03
12
04
e6
dd
d3
0d
fa
1c
1c
0f
0d
fd
10
04
01
b7
0d
00
08
15
d2
e8
0f
f3
bd
19
e2
1a
f1
e2
f5
f2
c6
ff
ec
02
d8
e9
fe
0f
16
d8
d6
25
e2
06
bd
24
0d
1c
c2
1c
ee
e0
e7
ff
05
01
0f
ff
fa
16
09
08
df
ff
12
d2
ea
0f
b4
0f
fb
f4
24
81
13
f1
17
0a
ed
03
f7
ed
c0
e4
da
19
fb
fd
fa
bf
14
15
17
e1
df
f6
f0
1c
10
0c
e9
cf
e0
25
eb
e2
1e
13
e9
e9
18
0b
dd
eb
fe
ee
10
14
0c
23
f1
e9
d9
c1
fa
fd
16
c9
2b
13
10
d6
00
ea
ce
e4
e1
d0
ec
0e
0f
13
f6
f6
12
04
08
1b
0f
0b
02
ea
d1
da
e3
e4
dc
1b
d0
18
f0
02
48
06
20
2a
e4
1b
0e
cb
14
df
3a
47
0e
2c
02
e6
01
21
00
f3
e4
cd
c9
d5
df
f1
cd
ed
18
11
0e
34
16
f1
14
db
fd
0e
34
30
df
ec
de
f2
cf
e1
c0
fd
d6
10
4f
1c
03
25
05
fa
fd
19
13
cb
08
20
03
39
0b
1e
ea
f3
0b
21
ee
16
03
af
14
ae
07
ff
f4
0c
aa
09
e6
e9
d2
24
e8
16
0d
f8
14
28
fb
df
9c
05
18
f0
03
45
1e
f8
0a
fc
04
32
26
0c
1b
e6
1a
20
ff
d9
19
b8
12
1b
13
11
e8
0b
e1
26
d4
11
e2
92
e0
fb
f5
15
1c
23
15
f7
09
e4
13
dc
f9
fe
f2
14
0a
cc
eb
02
07
fa
d1
dc
1e
13
ec
10
1e
00
1f
13
1a
db
16
08
03
0e
fe
e0
37
0c
f0
1a
0e
10
36
d2
07
fc
bb
d4
db
f7
00
0f
0b
0f
da
d7
08
06
0b
05
ff
16
34
02
e1
0d
1a
0f
9a
dc
e8
1e
dd
0f
f0
d8
f5
0f
f0
05
95
c1
ec
19
35
eb
08
2d
2e
db
28
fc
dd
f0
f2
12
1a
fd
18
fa
ec
cb
f7
0a
f0
10
c4
0a
14
e1
0c
03
06
07
e9
04
12
0f
08
c5
0d
fb
d1
21
dd
13
d9
ee
0d
09
fb
02
00
ff
e4
d3
14
f3
ce
c6
10
1f
13
0d
05
1d
d3
22
2b
fb
15
07
cf
09
f3
26
13
bd
04
e6
f6
11
f1
46
ef
0a
f9
e6
cb
0c
eb
1d
16
08
0e
2c
e8
08
fa
ee
d2
dd
19
1d
ce
9c
fa
a2
db
0c
c8
f5
d5
ca
ea
ed
1e
22
19
1f
f0
0b
04
07
de
04
00
09
fc
17
09
ec
12
c4
8a
0e
ec
04
0d
e7
12
15
0b
8f
d2
e7
c2
0f
f7
12
dc
0d
e1
18
07
d4
08
28
ab
0a
b4
1c
0b
05
02
0b
11
22
02
e9
14
e9
a7
15
bb
08
e1
fd
00
e4
f0
0d
23
29
ba
2b
09
0c
13
47
ff
0a
14
0e
1a
f8
97
d4
e0
eb
0a
0f
ed
3f
18
15
a5
10
07
13
2d
0f
12
f6
0d
09
0f
05
08
e9
14
11
f1
bb
38
1f
ff
02
05
ff
e2
38
fd
0b
0a
3a
ec
b2
2d
f0
db
cf
ed
eb
c4
05
df
03
a5
02
02
12
46
00
0b
ee
22
e7
f0
14
ca
0d
2e
f1
ba
05
2c
0c
ec
0d
fa
03
fb
0b
f4
e7
12
98
c2
08
d2
04
f3
1b
09
f8
f9
f2
28
f9
30
e9
06
e5
f6
c2
13
f5
14
21
50
c4
07
e2
f3
0d
fa
f3
0c
0c
08
15
10
e9
10
0b
0e
01
eb
12
ed
0b
0f
10
d0
0c
1a
fb
ca
cf
f3
cb
bd
ff
36
f6
fe
f8
f2
42
02
f8
06
12
1d
ce
14
0c
17
fb
0e
ed
17
15
0c
15
02
e6
0f
10
06
02
0c
b4
06
d7
af
2f
de
e1
40
ee
ef
f3
0c
aa
21
0a
c6
fe
f3
ff
1f
e6
3e
08
f8
d0
d8
f3
26
06
19
08
fc
fe
18
eb
f5
d2
d5
f4
e3
18
10
f7
ab
d3
f7
ed
16
c1
04
df
f6
e9
18
11
a9
da
d3
f0
04
15
d7
05
04
17
17
bf
23
ee
db
ec
ff
0f
df
ea
fb
1d
ef
08
ef
e2
f0
0b
13
e4
0e
12
18
ff
e1
0e
fb
10
05
b3
16
08
dd
2e
e0
12
fa
ee
17
ab
16
0a
0b
0f
15
e6
23
e4
cc
17
16
11
0f
a8
ff
c9
e8
2b
d4
04
f7
07
e4
1f
eb
01
9e
14
1a
01
11
13
25
03
04
f1
db
09
05
db
f2
d9
dc
1a
cc
22
14
09
03
08
0b
01
09
0f
d5
c9
d0
f2
f7
28
08
e6
0a
e2
d2
18
1d
14
00
09
08
f9
c3
e4
f7
17
e8
cd
18
0b
10
0f
0e
14
fc
1a
18
09
14
15
09
12
1c
da
f9
d8
e0
07
f6
0f
14
b2
e1
13
c3
e9
05
3a
ed
1f
f3
20
27
0a
c8
d8
32
19
27
25
dc
e2
20
ff
04
e9
50
00
de
db
ec
d6
d5
14
0b
06
ea
d8
21
ff
f1
1b
f3
0b
15
0d
0f
24
01
fd
0f
e2
13
e4
ad
02
14
05
ce
f0
e1
fe
0f
ed
0d
16
05
17
cf
14
15
13
df
c4
10
ec
05
14
01
b7
dd
10
c5
0b
02
12
06
d9
f8
16
14
0b
04
02
e5
10
0b
16
cc
df
17
10
12
0d
0d
d4
e6
cd
10
0c
13
b7
0b
01
18
fc
bb
03
10
ce
fc
15
18
fd
11
0a
14
b5
07
da
1d
db
17
ec
08
f1
0d
ef
03
c2
ec
ee
16
13
14
08
ce
b3
01
dc
ee
01
f5
12
ba
86
d5
0b
18
fe
22
0b
f9
04
15
17
e6
01
27
ea
fc
17
ec
99
03
ae
e0
e7
ea
c3
01
12
34
10
0b
d7
f2
14
14
f3
10
1d
1d
ef
16
ff
26
0d
fa
03
1c
f6
12
e3
18
11
ba
d8
ef
f5
2a
fc
11
15
02
16
17
06
13
0b
08
fb
14
fa
20
fe
21
01
0b
20
f7
14
ec
c8
08
16
fe
0a
10
1e
09
aa
05
ec
e6
fd
04
1b
d6
bd
eb
16
16
16
cc
25
e7
eb
07
17
f2
ab
10
c9
f9
d5
06
ff
16
00
0f
ef
b6
0d
b6
ae
03
10
e0
eb
f3
e6
9c
09
bd
0d
e9
d2
dc
14
97
11
0f
0c
0b
11 先检查这两个有没有问题，然后根据现在的文件和坑，你看看代码有什么需要改进的，最好生成的数据可以被fpga上得cim系统直接使用
AI助手:


AI助手:
我先按“文件内容本身是否自洽”和“是否适合 FPGA/CIM 直接吃”两层来判断。眼下一个明显信号是：你贴出来的 `bias.hex` 已经是 **32 位有符号十六进制**，这比你最初 notebook 里那个 int8 bias 更合理，说明你的导出逻辑应该已经改过一版了。

AI助手:
先说结论：

**你现在贴出来的这两个文件，格式上基本是对的，而且已经比最初 notebook 版本更适合 FPGA/CIM 直接使用。**

尤其是：

- `bias.hex`：现在是**32 位有符号补码**，这很合理，适合 MAC 后加 bias
- `weight.hex`：现在是**8 位有符号补码**，也合理，适合 int8 权重阵列

这说明你的导出代码已经朝着“真实整数推理链路”改进过了，不再是最初那个 bias 只有 2 位 hex 的版本。

下面我分三部分说：

1. 先检查这两个文件有没有明显问题  
2. 结合现在的文件格式，说明 FPGA/CIM 应该怎么直接使用  
3. 按“还能怎么改代码”给你一套更适合直接上板的改进建议

---

# 1. 先检查这两个文件有没有问题

---

## 1.1 `bias.hex` 检查结果

你给的 `bias.hex` 形如：

```text
000008e5
0000183f
000001a2
ffffe6b8
...
```

### 这说明什么
每行都是 **8 个十六进制字符**，也就是 **32 bit**。

这对应的是：

- 正数：直接写成 `000008e5`
- 负数：写成二补码形式，比如 `ffffe6b8`

这正是 FPGA 里最常见的 **signed int32 bias 文件格式**。

### 这和硬件链路匹配吗
匹配，而且是更合理的做法：

- 输入：int8
- 权重：int8
- 累加：int32
- 偏置：int32
- ReLU / requant：后续再做

所以从工程角度说，**这个 bias 格式是对的**。

### 我看出来的一个信号
你这份 `bias.hex` 已经不是最开始 notebook 里那种：

```python
f.write(f'{val.item() & 0xFF:02x}\n')
```

而是明显变成了类似：

```python
f.write(f'{val.item() & 0xFFFFFFFF:08x}\n')
```

这说明你已经把 bias 导出升级成 int32 了。这个改动是正确的。

### 需要确认的一点
如果这是 `fc1` 的 bias，那么它应该有：

- **128 行**

如果这是 `fc2` 的 bias，那么应该有：

- **10 行**

你贴出来看起来更像是 128 行中的一整段。  
如果你最终网络是两层 MLP，那建议不要只叫 `bias.hex`，而是明确改成：

- `fc1_bias_int32.hex`
- `fc2_bias_int32.hex`

这样不会混。

---

## 1.2 `weight.hex` 检查结果

你给的 `weight.hex` 形如：

```text
aa
e5
00
0d
0d
f7
...
```

### 这说明什么
每行都是 **2 个十六进制字符**，也就是 **8 bit**。

这对应：

- `aa` = -86
- `e5` = -27
- `00` = 0
- `0d` = 13
- `f7` = -9

这正是标准的 **signed int8 权重存储格式**。

### 是否合理
合理，前提是 RTL 里按 signed 解释：

```verilog
wire signed [7:0] w = $signed(weight_mem[addr]);
```

### 数值分布上看有没有异常
从你贴的内容看，数值大多集中在：

- `0xC0 ~ 0x40` 左右对应的 -64 到 64 之间
- 偶尔有 `0x81`、`0x9c`、`0x8a`、`0x50`、`0x4f`

这很像正常量化后的神经网络权重分布，不像导出炸掉了。

也就是说：

- **没有明显出现全 00 / 全 ff / 大面积饱和到 7f/80 的异常**
- 看起来像一份正常的量化权重

### 需要确认的一点
如果这是 `fc1.weight`，而且第一层 shape 是 `[128, 784]`，那它总行数应该是：

\[
128 \times 784 = 100352
\]

如果最终网络是两层 MLP，你也应该拆成：

- `fc1_weight_int8.hex`
- `fc2_weight_int8.hex`

否则后面上板会混。

---

# 2. 结合你现在的文件格式，FPGA/CIM 怎么直接使用

现在按你贴出来的格式，**最适合的硬件链路**是：

- `weight.hex`：int8 权重阵列
- `bias.hex`：int32 偏置
- `input_i.hex`：int8 输入
- 累加器：int32
- 后面做 ReLU、requant、第二层

也就是说，你现在已经从“玩具导出格式”走到了“能上硬件”的方向。

---

## 2.1 `weight.hex` 的硬件读取方式

如果你的导出顺序还是之前那样：

```python
for row in q_weight:
    for val in row:
        ...
```

那么权重布局就是：

\[
W[j][i]
\]

也就是：

- `j`：输出神经元索引
- `i`：输入通道索引

### 地址公式
若第一层 `fc1` shape 是 `[128, 784]`，则：

\[
addr = j \times 784 + i
\]

RTL 读法就是：

- 外层循环跑输出通道 `j`
- 内层循环跑输入通道 `i`

### 对应计算
\[
acc_j = \sum_{i=0}^{783} (x_i - z_x)(w_{j,i} - z_w) + b_j
\]

如果你现在是权重对称量化，那通常：

- `z_w = 0`

如果输入采用 `x_q = pixel_u8 - 128`，那：

- `z_x = -128`

---

## 2.2 `bias.hex` 的硬件使用方式

现在你的 bias 是 32 位二补码，很适合直接这样读：

```verilog
reg [31:0] bias_mem [0:127];
initial $readmemh("bias.hex", bias_mem);

wire signed [31:0] bias_val = $signed(bias_mem[j]);
```

然后在 MAC 完成后：

```verilog
acc <= mac_sum + bias_val;
```

这比 int8 bias 合理得多。

---

## 2.3 对 CIM 系统的直接使用建议

如果你的 CIM 阵列本体负责的是 `int8 x int8` 乘加，那最常见的系统切分是：

### CIM 阵列做
- 权重存储
- 向量输入
- 局部乘加 / bitline accumulate / column accumulate

### FPGA 数字逻辑做
- 输入 zero-point 修正
- 偏置相加
- ReLU
- requant
- 第二层控制
- argmax

这和你现在的文件格式非常匹配：

- `weight.hex` 给阵列权重
- `input_i.hex` 给输入驱动
- `bias.hex` 给数字后处理单元
- `golden` 给 testbench 对拍

---

# 3. 这两个文件背后，代码还需要怎么改进

你现在文件本身没大问题，但为了“**生成的数据可以被 FPGA 上的 CIM 系统直接使用**”，我建议你继续改 8 个点。

---

## 改进 1：文件命名必须分层，不能再叫通用名

你现在如果最终是两层 MLP，绝对不要继续用：

- `weight.hex`
- `bias.hex`

因为你后面会出现：

- 第一层权重
- 第二层权重
- 第一层 bias
- 第二层 bias

建议直接改成：

- `fc1_weight_int8.hex`
- `fc1_bias_int32.hex`
- `fc2_weight_int8.hex`
- `fc2_bias_int32.hex`

这样你 RTL 一眼知道读哪个。

---

## 改进 2：必须配一个 `quant_config.json`

现在单靠 hex 文件还不够。  
FPGA/CIM 真正要跑整数推理，还需要知道：

- 输入 `scale`
- 输入 `zero_point`
- 各层 `weight_scale`
- 各层 `weight_zero_point`
- 各层 `output_scale`
- 各层 `output_zero_point`
- requant multiplier / shift

建议额外生成：

```json
{
  "input": {
    "scale": 0.00392156862745098,
    "zero_point": -128
  },
  "fc1": {
    "weight_scale": ...,
    "weight_zero_point": 0,
    "output_scale": ...,
    "output_zero_point": 0,
    "bias_scale": ...,
    "multiplier_q31": ...,
    "shift": ...
  },
  "fc2": {
    "weight_scale": ...,
    "weight_zero_point": 0,
    "output_scale": ...,
    "output_zero_point": 0,
    "bias_scale": ...,
    "multiplier_q31": ...,
    "shift": ...
  }
}
```

这样 FPGA 端和 Python 端是一致的。

---

## 改进 3：输入文件也要明确 int8 和布局

你现在最好明确导出成：

- `input_0_int8.hex`
- `input_1_int8.hex`

并且说明：

- 长度 784
- 顺序是 row-major
- 每行 2 位 hex
- 数值是 signed int8
- 若采用硬件输入策略，则：
  \[
  x_q = round(x \cdot 255) - 128
  \]

这能避免你后面忘记输入是“原始像素量化”还是“Normalize 后量化”。

---

## 改进 4：golden 不要只保留最终输出，要保留中间级

这是对硬件调试最关键的一点。

建议你至少导出：

### 第一层
- `fc1_acc_i.hex`：int32，未激活前
- `fc1_relu_i.hex`：int32，过 ReLU 后
- `fc1_out_i.hex`：int8，requant 后

### 第二层
- `fc2_acc_i.hex`：int32
- `logits_i.hex`：int8

### 分类
- `pred_i.txt`

这样当 FPGA 结果不对时，你能立刻定位是：

- 阵列乘加错
- bias 加错
- ReLU 错
- requant 错
- 第二层地址错

不然只看最后一个 `golden`，很难找 bug。

---

## 改进 5：权重文件要考虑和 CIM 阵列物理映射一致

你现在的 `weight.hex` 是逻辑顺序 `[out][in]`。  
这对软件和普通 RTL 很方便，但对实际 CIM 阵列，可能还不够。

因为真实阵列通常不是一次吃 784 个输入，而是分块，例如：

- 每列 64 个输入
- 每次做 64×N 子矩阵
- 784 被切成很多 tile

所以你后面很可能需要两套文件：

### 逻辑文件
给软件 / testbench 用：
- `fc1_weight_int8.hex`

### 物理映射文件
给阵列初始化用：
- `fc1_weight_tile0.hex`
- `fc1_weight_tile1.hex`
- ...

也就是按 CIM 阵列实际映射切块导出。

如果你现在阵列设计已经定了，比如每次 64 路输入并行，我建议导出时就按 64 一组打包。

---

## 改进 6：给每个文件写 manifest

建议自动生成一个 `manifest.txt` 或 `manifest.json`，写明：

- 文件名
- 数据类型
- 位宽
- 长度
- 布局
- 用途

例如：

```json
{
  "fc1_weight_int8.hex": {
    "dtype": "int8",
    "shape": [128, 784],
    "layout": "row-major [out][in]"
  },
  "fc1_bias_int32.hex": {
    "dtype": "int32",
    "shape": [128]
  },
  "input_0_int8.hex": {
    "dtype": "int8",
    "shape": [784],
    "layout": "row-major flattened image"
  }
}
```

这会让你的工程后期非常省事。

---

## 改进 7：最好支持导出 `.mem` / `.coe` / 打包宽字格式

现在一行一个数适合 testbench，  
但真正上 FPGA，很多时候你会想直接初始化 BRAM。

比如：

- Xilinx 可能想要 `.coe`
- Intel 可能想要 `.mif`
- 或者你会把 16 个 int8 打成 128-bit 宽字

所以建议你的导出工具支持两种模式：

### 调试模式
- 一行一个元素
- 便于看和对拍

### 上板模式
- 打包成 32bit / 64bit / 128bit / 256bit
- 便于 BRAM 初始化和 DMA 传输

---

## 改进 8：确保 signed 和 endianness 约定写死

当前 hex 文件本身没错，但你要在代码里和说明里明确：

### signed 解释
- `weight.hex`：signed int8
- `bias.hex`：signed int32

### endianness
你现在是一行一个元素，所以“行内字节序”问题不大。  
但一旦你后面把 4 个 int8 打包成 32 bit，必须写清楚：

- 第 0 个元素放最低字节还是最高字节

这个最好现在就定。

---

# 4. 你现在这两个文件，我给你的具体判断

---

## `bias.hex`
**没有明显问题，反而是明显进步。**

它已经具备：
- int32
- 二补码
- 可直接用于 MAC 后 bias add

### 唯一建议
改名为：
- `fc1_bias_int32.hex`
- 或 `fc2_bias_int32.hex`

---

## `weight.hex`
**没有明显问题。**

它已经具备：
- int8
- 二补码
- 适合阵列 / 乘加器直接使用

### 唯一建议
改名为：
- `fc1_weight_int8.hex`
- 或 `fc2_weight_int8.hex`

---

# 5. 我建议你下一版导出目录直接长这样

```text
export/
├── quant_config.json
├── manifest.json
├── fc1_weight_int8.hex
├── fc1_bias_int32.hex
├── fc2_weight_int8.hex
├── fc2_bias_int32.hex
├── input_0_int8.hex
├── fc1_acc_0_int32.hex
├── fc1_relu_0_int32.hex
├── fc1_out_0_int8.hex
├── fc2_acc_0_int32.hex
├── logits_0_int8.hex
├── pred_0.txt
├── label_0.txt
├── input_1_int8.hex
...
```

这套结构就已经很像正式硬件联调工程了。

---

# 6. 最后给你一句最关键的判断

你现在贴出来的这两个文件，说明你已经迈过了最初 notebook 的两个大坑：

- **bias 不再是错误的 int8**
- **weight 已经是标准 int8 二补码**

所以文件内容层面，**方向是对的**。  
真正还需要继续加强的是：

- 文件命名分层
- 中间 golden 分级导出
- 量化参数 json 化
- 按 CIM 阵列物理映射切块导出

你把**当前用于导出这些文件的最新版 Python 代码**贴给我，我可以下一步直接帮你逐行审，给你改成一版“**输出可直接喂 FPGA/CIM 系统**”的正式版。

用户:
这是当前生成的数据，你主要先看一下route_b_output中的文件格式是否全部正确，然后给出一个markdown文件，用以说明route_b_output所有文件的使用方法和注意事项，要详细
The file is too long and its contents have been truncated.

用户:
The file contents provided above are truncated/partial snippets. The complete content for these files IS accessible via querying.

If the user asks a question related to these files, and the provided snippets do not clearly answer it, you **MUST** use the `file_search` tool to search the full document contents before responding.

## How to invoke the file_search tool
If the user's query isn't sufficiently answered by the provided snippets, parsed text, or images, you should invoke `file_search.msearch`.
To search the full content of the available files, you **must** send one message with recipient `to=file_search.msearch` in the analysis channel. Your message should be structured exactly as follows:
```
{
  'queries': ['Example query 1', 'Example query 2']  // List of up to five queries to search for.
}
```

Carefully balance the number of `file_search` calls and the quality of your response with the latency incurred by `file_search` calls.

Important guidelines for query formatting:
- Your message must be a valid JSON object. Do not include any extraneous text, backticks, markdown formatting, or comments beyond what is shown.
- The `queries` field must be a list of strings.
- One of the queries **MUST** exactly match the user's original question, but stripped of any extraneous details, and with ambiguous references clearly resolved using context from the conversation. It **MUST** be a complete sentence.
- Even if you suspect the user meant something else, one query **MUST** still be their original question.

## How to handle results from file_search
The file search tool will respond with the relevant search results from the full files. After you receive these results:
- Carefully evaluate each result to determine its relevance and quality.
- Use the high-quality and relevant results to inform your answer to the user's question.
- When possible, reuse information from prior file_search results and avoid repeating identical queries.

## Tool usage instructions and ordering
When the user asks a question, carefully consider whether the provided snippets sufficiently answer the query, whether you need to invoke `file_search.msearch`, or another available tool (such as `web`, if available) should be used instead.

Tool-ordering Guidelines:
1. For queries related to uploaded files, if snippets are insufficient, strongly prefer invoking `file_search.msearch` over other tools.
    - Avoid using `web.run`, `python` analysis, or any other tools until you have either:
        - Confirmed with high confidence the answer is not in the files, or
        - Have high confidence `file_search_tool` queries are not sufficient to answer the query.
2. Avoid usage of `web_search/web.run` for queries related to uploaded files
    - Carefully consider whether the user's query is better answered by web search results rather than available snippets or results from `file_search_tool`.
    - Unless the user's query is clearly unrelated to the uploaded files, prefer using `file_search.msearch`. You can always follow up with a web search if necessary for a more complete response.
3. Avoid using `python` unless necessary or requested by user.
    - If the user's query is answerable from the parsed text and images from the uploaded files, or from `file_search_tool`, do not invoke `python`.

Always keep this instruction in mind for the rest of the conversation, as the user may reference these files or ask related questions later on.
AI助手:
我先把你上传内容里和 `route_b_output` 相关的完整部分检索出来，确认每个文件的位宽、行数、命名和用途，再整理成一份可直接交给同学或后续 RTL 联调使用的说明文档。

用户:

AI助手:
我已经定位到你上传内容里关于 `route_b_output` 的生成代码和后续硬件联调用法了。接下来我会把这些信息整理成一份更正式、可核对的 Markdown 说明文件，并先判断当前文件格式有没有明显自相矛盾的地方。

用户:
Make sure to include fileciteturn2file0 in your response to cite this file, or to surface it as a link.[L1] ### FPGA CIM MNIST 问题分析
[L2] 
[L3] AI助手:
[L4] 
[L5] 
[L6] AI助手:
[L7] 
[L8] 
[L9] AI助手:
[L10] 
[L11] 
[L12] AI助手:
[L13] 
[L14] 
[L15] 用户:
[L16] 我现在在做一个运行于fpga的cim系统，希望在fpga上运行mnist，完成计算，我现在给你项目指导书和上次和AI的对话，ipynb写的是一个生成测试向量的代码，你看看有没有问题
[L17] AI助手:
[L18] 
[L19] 
[L20] AI助手:
[L21] 我先把压缩包里的文件结构和关键材料过一遍，重点看项目指导书、上次对话记录，以及这个生成测试向量的 ipynb。随后我会直接指出代码里可能影响 FPGA/CIM 测试链路的问题，并给出修改建议。
[L22] 
[L23] AI助手:
[L24] 我已经把压缩包拆开了，里面目前有三份核心文件：课程设计指导书（docx）、上次对话导出的 markdown，以及 `train_simple_mlp.ipynb`。接下来我会先把对话和 notebook 的关键逻辑抽出来，再对照指导书检查接口、数据位宽、输出格式这些是否一致。
[L25] 
[L26] AI助手:
[L27] 我已经看到几个关键问题了，其中最严重的一个是：**这个 notebook 目前并没有真正训练 MNIST 分类器**。它定义了 `criterion=MSELoss()`，但完全没用；实际优化目标是 `loss = output.pow(2).mean()`，这会把网络输出压向 0，而不是学习数字类别。
[L28] 
[L29] AI助手:
[L30] 我看完了指导书、上次对话，以及这个 `train_simple_mlp.ipynb`。结论先说：
[L31] 
[L32] **这个 ipynb 现在有明显问题，尤其是“训练目标、量化策略、golden 生成方式”三处。**
[L33] 如果直接拿它生成测试向量去驱动 FPGA/CIM，**大概率会出现“软件看起来能跑，硬件却对不上”**，甚至前面训练出来的模型本身就没有学会 MNIST 分类。
[L34] 
[L35] 下面我按“严重程度”给你拆开说。
[L36] 
[L37] ---
[L38] 
[L39] ## 1. 最严重的问题：你现在没有在训练一个 MNIST 分类器
[L40] 
[L41] 你现在的训练代码核心是：
[L42] 
[L43] ```python
[L44] criterion=nn.MSELoss()
[L45] ...
[L46] output=model(data)
[L47] loss=output.pow(2).mean()
[L48] ```
[L49] 
[L50] 这里有两个问题：
[L51] 
[L52] ### 问题 1：`criterion` 定义了，但根本没用
[L53] 你定义了 `MSELoss()`，但没有把 `target` 喂进去。
[L54] 
[L55] ### 问题 2：实际 loss 是 `output.pow(2).mean()`
[L56] 这等价于逼着网络输出尽量接近 0，而不是学习数字类别。
[L57] 
[L58] 也就是说，现在模型优化目标不是“把 0~9 分对”，而是“把所有输出都压小”。  
[L59] 这会导致：
[L60] 
[L61] - 权重可能收缩到很小
[L62] - 导出的量化权重没有实际分类意义
[L63] - 你生成的 `golden_x.hex` 也只是这个“错误模型”的输出
[L64] 
[L65] ### 进一步的问题：模型输出维度也不对
[L66] 你定义的是：
[L67] 
[L68] ```python
[L69] self.fc1 = nn.Linear(784,128)
[L70] self.relu = nn.ReLU()
[L71] ```
[L72] 
[L73] 输出是 `128` 维，不是 `10` 维。  
[L74] 这意味着它不是一个完整的 MNIST 分类网络，只是一个隐藏层特征提取器。
[L75] 
[L76] 如果你的目标是：
[L77] 
[L78] - **课程演示先做“单层 FC + ReLU 的硬件验证”**：这个结构可以接受
[L79] - **直接说“在 FPGA 上跑 MNIST 分类”**：这个结构还不够，至少还要再接一层 `128 -> 10`
[L80] 
[L81] ### 这一点怎么改
[L82] 你有两个路线：
[L83] 
[L84] #### 路线 A：先做“极简硬件验证”
[L85] 那就明确目标改成：
[L86] 
[L87] - 软件训练完整 MLP：`784 -> 128 -> 10`
[L88] - FPGA/CIM **先只实现第一层** `784 -> 128`
[L89] - 第一层输出 feature map 作为硬件 golden 对比对象
[L90] 
[L91] 这样合理。
[L92] 
[L93] #### 路线 B：直接做完整分类
[L94] 那模型要改成：
[L95] 
[L96] ```python
[L97] class SimpleMLP(nn.Module):
[L98]     def __init__(self):
[L99]         super().__init__()
[L100]         self.fc1 = nn.Linear(784, 128, bias=True)
[L101]         self.relu = nn.ReLU()
[L102]         self.fc2 = nn.Linear(128, 10, bias=True)
[L103] 
[L104]     def forward(self, x):
[L105]         x = x.view(-1, 784)
[L106]         x = self.relu(self.fc1(x))
[L107]         x = self.fc2(x)
[L108]         return x
[L109] ```
[L110] 
[L111] loss 用交叉熵：
[L112] 
[L113] ```python
[L114] criterion = nn.CrossEntropyLoss()
[L115] loss = criterion(output, target)
[L116] ```
[L117] 
[L118] ---
[L119] 
[L120] ## 2. 量化函数有工程问题，不适合直接对接 FPGA
[L121] 
[L122] 你现在的量化函数：
[L123] 
[L124] ```python
[L125] def quantize_tensor(tensor, bits=8):
[L126]     qmin = -(2 ** (bits - 1))
[L127]     qmax = 2 ** (bits - 1) - 1
[L128] 
[L129]     min_val = tensor.min().item()
[L130]     max_val = tensor.max().item()
[L131] 
[L132]     diff = max_val - min_val
[L133]     
[L134]     if diff == 0:
[L135]         scale = 1.0
[L136]         zero_point = qmin - min_val
[L137]     else:
[L138]         scale = diff / (qmax - qmin)
[L139]         zero_point = qmin - min_val / scale
[L140] 
[L141]     q_tensor = torch.clamp(torch.round(tensor / scale + zero_point), qmin, qmax)
[L142]     return q_tensor.to(torch.int8), scale, zero_point
[L143] ```
[L144] 
[L145] ### 问题 1：`zero_point` 不是整数
[L146] `zero_point` 最后是浮点数。  
[L147] 但真正部署到硬件时，量化零点通常需要是整数，并且通常要 clamp 到合法范围。
[L148] 
[L149] 更稳妥应该写成：
[L150] 
[L151] ```python
[L152] zero_point = int(round(qmin - min_val / scale))
[L153] zero_point = max(qmin, min(qmax, zero_point))
[L154] ```
[L155] 
[L156] ### 问题 2：你在做“每个张量动态量化”
[L157] 这在软件分析里可以，但对 FPGA 验证不友好。因为你后面对：
[L158] 
[L159] - 权重单独量化一次
[L160] - 每张输入图像又各自量化一次
[L161] - 每个输出又重新单独量化一次
[L162] 
[L163] 这会导致**每个样本 scale 都不同**，硬件端很难统一。
[L164] 
[L165] 对 FPGA 来说，更适合：
[L166] 
[L167] - 权重：固定 scale
[L168] - 输入：固定 scale
[L169] - 偏置：固定规则量化
[L170] - 输出：固定 requant scale
[L171] 
[L172] 也就是**静态量化参数**，不是每个样本临时算一次。
[L173] 
[L174] ### 问题 3：偏置不能简单用 int8 单独量化
[L175] 你现在：
[L176] 
[L177] ```python
[L178] q_bias, b_scale, b_zero = quantize_tensor(bias)
[L179] ```
[L180] 
[L181] 这通常不对。
[L182] 
[L183] 对于线性层：
[L184] 
[L185] \[
[L186] y = \sum (x_q - z_x)(w_q - z_w) \cdot s_x s_w + b
[L187] \]
[L188] 
[L189] 偏置一般应量化为：
[L190] 
[L191] \[
[L192] b_q = round(b / (s_x \cdot s_w))
[L193] \]
[L194] 
[L195] 并且常常用 **int32** 存，不是 int8。  
[L196] 因为偏置要和累加结果处于同一量纲。
[L197] 
[L198] 否则 FPGA 上 MAC 累加完后加 bias 时，量纲根本不一致。
[L199] 
[L200] ---
[L201] 
[L202] ## 3. `generate_test_vectors()` 生成的 golden，和硬件真实计算链路不一致
[L203] 
[L204] 这是第二个特别大的问题。
[L205] 
[L206] 你现在生成 golden 的方式是：
[L207] 
[L208] ```python
[L209] with torch.no_grad():
[L210]     output = model(img_flat)
[L211] 
[L212] q_output, _, _ = quantize_tensor(output)
[L213] ```
[L214] 
[L215] ### 这意味着什么
[L216] 你是：
[L217] 
[L218] 1. 用 **浮点模型** 跑出输出
[L219] 2. 再对每个样本的输出临时量化
[L220] 
[L221] 但硬件里真正发生的是：
[L222] 
[L223] 1. 输入先量化为 int8
[L224] 2. 权重量化为 int8
[L225] 3. 在阵列/乘加器里做 int8×int8 -> int32 累加
[L226] 4. 加 int32 bias
[L227] 5. ReLU
[L228] 6. 再按固定 scale requant 到 int8
[L229] 
[L230] 这两条链路并不一样。
[L231] 
[L232] 所以你现在的 `golden_i.hex` 并不是“硬件应得输出”，而是“浮点模型输出的一个临时量化版本”。
[L233] 
[L234] ### 正确的 golden 应该怎么做
[L235] 应该按**整数推理链路**生成 golden。例如第一层：
[L236] 
[L237] ```python
[L238] acc = sum((x_q - x_zp) * (w_q - w_zp)) + b_q
[L239] acc = relu(acc)
[L240] out_q = requantize(acc, multiplier, shift, out_zp)
[L241] ```
[L242] 
[L243] 然后把：
[L244] 
[L245] - 输入 int8
[L246] - 权重 int8
[L247] - 偏置 int32
[L248] - 输出 int32 / int8 golden
[L249] - scale / zero_point / multiplier / shift
[L250] 
[L251] 全部导出来。
[L252] 
[L253] 这样 FPGA 才能逐项对上。
[L254] 
[L255] ---
[L256] 
[L257] ## 4. 输入测试向量的量化方式不适合硬件批量验证
[L258] 
[L259] 你现在对每张图像：
[L260] 
[L261] ```python
[L262] q_img, img_scale, img_zero = quantize_tensor(img_flat)
[L263] ```
[L264] 
[L265] 这意味着**每张图一个 scale 和 zero_point**。  
[L266] 如果你的 FPGA 流水是固定 datapath，这会很麻烦：
[L267] 
[L268] - 每个样本都要动态更换量化参数
[L269] - 偏置量纲也得跟着变
[L270] - 对比 golden 时也会很乱
[L271] 
[L272] ### 更合理的做法
[L273] 建议统一输入量化策略，比如：
[L274] 
[L275] - 若保留 `Normalize((0.1307,), (0.3081,))`，则先统计训练集/校准集的输入范围，再确定一个全局输入 scale
[L276] - 或者更简单，**先不要 Normalize**，直接把 MNIST 灰度 `[0,1]` 映射到 `uint8/int8`
[L277] 
[L278] 对 FPGA 验证来说，后者往往更省事。
[L279] 
[L280] 例如极简方案可用：
[L281] 
[L282] - 输入像素 `0~255`
[L283] - 权重量化到 int8
[L284] - 累加 int32
[L285] - ReLU 后再截断或 requant
[L286] 
[L287] 这样更贴近硬件。
[L288] 
[L289] ---
[L290] 
[L291] ## 5. 导出的权重格式本身基本可用，但有几个注意点
[L292] 
[L293] 你现在导出权重：
[L294] 
[L295] ```python
[L296] for row in q_weight:
[L297]     for val in row:
[L298]         f.write(f'{val.item() & 0xFF:02x}\n')
[L299] ```
[L300] 
[L301] 这表示按 `weight[out][in]` 顺序逐个写出。  
[L302] 对于 `nn.Linear(784,128)`，PyTorch 权重 shape 是 `[128, 784]`，即：
[L303] 
[L304] - 外层：输出神经元
[L305] - 内层：输入通道
[L306] 
[L307] 这个顺序本身没错，但必须保证 RTL 端读取方式完全一致。
[L308] 
[L309] ### 你需要确认的点
[L310] 如果硬件是按下面方式算：
[L311] 
[L312] \[
[L313] y_j = \sum_i x_i w_{j,i}
[L314] \]
[L315] 
[L316] 那么文件顺序最好明确成：
[L317] 
[L318] - `neuron 0` 的 784 个权重
[L319] - `neuron 1` 的 784 个权重
[L320] - ...
[L321] - `neuron 127` 的 784 个权重
[L322] 
[L323] 你现在就是这个顺序。  
[L324] 但如果你的 BRAM 映射或地址生成器是按列优先/块优先取数，就会错位。
[L325] 
[L326] ### 建议
[L327] 导出时加一个说明文件，比如：
[L328] 
[L329] - weight layout: `[out][in]`, row-major
[L330] - bias layout: `[out]`
[L331] - input layout: `[in]`
[L332] 
[L333] ---
[L334] 
[L335] ## 6. `quant_params.npz` 在测试向量阶段几乎没被真正使用
[L336] 
[L337] 你这里：
[L338] 
[L339] ```python
[L340] quant_params = np.load('quantized_data/quant_params.npz')
[L341] ```
[L342] 
[L343] 加载了参数，但后面没有真正参与 golden 计算。  
[L344] 这说明当前流程里“量化参数被保存了”，但并没有形成闭环。
[L345] 
[L346] 这是个信号：**量化设计还停留在文件导出阶段，没有进入“硬件一致性验证阶段”。**
[L347] 
[L348] ---
[L349] 
[L350] ## 7. 其他次级问题
[L351] 
[L352] ### 问题 1：`torch.load()` 最好加 `map_location`
[L353] 不然在有些机器上保存于 GPU、加载于 CPU 时会出问题：
[L354] 
[L355] ```python
[L356] model.load_state_dict(torch.load(model_path, map_location='cpu'))
[L357] ```
[L358] 
[L359] ### 问题 2：训练时没看 accuracy
[L360] 现在只打 loss，没有任何分类精度指标。  
[L361] 如果要确认模型真的学到了，建议至少打印 test accuracy。
[L362] 
[L363] ### 问题 3：没有保存 label
[L364] 你打印了：
[L365] 
[L366] ```python
[L367] print(f"Generated test vector {i}, label: {label}")
[L368] ```
[L369] 
[L370] 但没有把 label 写入文件。  
[L371] 实际联调时最好保存一个 `labels.txt` 或 `meta.json`。
[L372] 
[L373] ### 问题 4：只保存了量化后的输出，没有保存中间 int32 累加值
[L374] 对于硬件调试，这很吃亏。  
[L375] 建议同时导出：
[L376] 
[L377] - `input_i.hex`
[L378] - `golden_acc_i.hex` 或每个神经元的 int32
[L379] - `golden_relu_i.hex`
[L380] - `golden_out_i.hex`
[L381] 
[L382] 这样你能定位是：
[L383] 
[L384] - 乘法错了
[L385] - 累加错了
[L386] - bias 错了
[L387] - ReLU 错了
[L388] - requant 错了
[L389] 
[L390] ---
[L391] 
[L392] # 我对这个 notebook 的总体判断
[L393] 
[L394] ## 现在这份代码“能不能用”？
[L395] **能作为一个思路草稿，但不能直接作为 FPGA 测试向量生成基线。**
[L396] 
[L397] ## 主要原因
[L398] 不是小 bug，而是三类“体系级不一致”：
[L399] 
[L400] 1. **训练目标错了**  
[L401]    当前模型没有真正学分类任务
[L402] 
[L403] 2. **量化策略不适合硬件**  
[L404]    每个样本动态量化，不利于 FPGA 固定算子验证
[L405] 
[L406] 3. **golden 生成链路和硬件链路不一致**  
[L407]    现在是“浮点输出再量化”，而不是“整数推理 golden”
[L408] 
[L409] ---
[L410] 
[L411] # 我建议你接下来这样改
[L412] 
[L413] ## 方案一：适合你们当前阶段的最小闭环
[L414] 如果你们现在的硬件目标是：
[L415] 
[L416] - 先做一个 `784 -> 128` 的单层 CIM/FC 阵列
[L417] - 验证 BRAM + MAC + ReLU + 控制 FSM
[L418] 
[L419] 那最合适的 notebook 应该改成下面这种逻辑：
[L420] 
[L421] ### 软件模型
[L422] 训练完整模型：
[L423] 
[L424] - `784 -> 128 -> 10`
[L425] 
[L426] ### 硬件实现
[L427] 先只实现第一层：
[L428] 
[L429] - `fc1 + relu`
[L430] 
[L431] ### 测试向量导出
[L432] 导出以下内容：
[L433] 
[L434] - `fc1_weight_int8.hex`
[L435] - `fc1_bias_int32.hex`
[L436] - `input_int8.hex`
[L437] - `fc1_acc_golden_int32.hex`
[L438] - `fc1_relu_golden_int32_or_int8.hex`
[L439] - `quant_config.json`
[L440] 
[L441] 这样你们硬件和软件就能逐层对齐。
[L442] 
[L443] ---
[L444] 
[L445] ## 方案二：直接瞄准最终演示
[L446] 如果你们想最后做“MNIST 分类结果上板演示”，那就：
[L447] 
[L448] - 模型改为两层 MLP
[L449] - 训练用 CrossEntropyLoss
[L450] - 做静态量化
[L451] - 先用 Python 写一个**整数推理参考实现**
[L452] - 再把这个整数推理结果作为 FPGA golden
[L453] 
[L454] 这是最稳的路线。
[L455] 
[L456] ---
[L457] 
[L458] # 我给你的具体结论
[L459] 
[L460] ## 必改
[L461] - `loss=output.pow(2).mean()` 改掉
[L462] - 模型结构和任务目标统一
[L463] - bias 改为 int32 量化
[L464] - golden 改成按整数推理链路生成
[L465] - 输入量化不要每张图单独动态算参数
[L466] 
[L467] ## 可保留
[L468] - `weights.hex` / `input_i.hex` 这种逐行写 hex 的导出方式
[L469] - 第一版只验证单层 FC + ReLU 的项目策略
[L470] - 先用 notebook 生成测试向量，再接 Verilog testbench 的总体流程
[L471] 
[L472] ---
[L473] 
[L474] 如果你愿意，我下一步可以直接帮你把这个 notebook **改成一版可用于 FPGA 联调的“正确版本”**，包括：
[L475] - 正确训练
[L476] - 静态 INT8 量化
[L477] - 生成 `weights/bias/input/golden` 四类 hex
[L478] - 明确对应单层 FC 硬件接口格式。
[L479] 
[L480] 用户:
[L481] 关于1.现在没有在训练一个 MNIST 分类器的问题，请分别给我路线a和路线b的完整代码；
[L482] 
[L483] 关于2.量化函数有工程问题，不适合直接对接 FPGA，
[L484] zero_point不是整数的问题帮我修改；每个张量动态量化这个问题给我一个合理的解决方案；偏置不能简单用 int8 单独量化请帮我提出解决方案和代码；
[L485] 
[L486] 关于3.generate_test_vectors() 生成的 golden，和硬件真实计算链路不一致，整数推理链路生成 golden，帮我完成设计；
[L487] 
[L488] 关于4输入测试向量的量化方式不适合硬件批量验证，帮我设计统一输入量化策略，贴近硬件即可；
[L489] 
[L490] 关于5. 导出的权重格式本身基本可用，但有几个注意点，这里文件顺序你先记住我的输出顺序，讲解一下我现在的输出顺序应该如何设计硬件读取顺序；
[L491] AI助手:
[L492] 我先把现有 notebook 里导出部分再核一遍，确保我后面给你的代码和你当前的文件顺序兼容。然后我会把路线 A、路线 B、量化、整数 golden、统一输入策略和硬件读取顺序一次性给全。
[L493] 
[L494] AI助手:
[L495] 下面我按你的 5 个问题，直接给出**可落地的完整方案和代码**。我会尽量保持和你现有工程思路一致，但把关键的工程问题修正掉。
[L496] 
[L497] ---
[L498] 
[L499] # 1. 关于“现在没有在训练一个 MNIST 分类器”的问题
[L500] 
[L501] 这里我给你两套完整代码。
[L502] 
[L503] ---
[L504] 
[L505] ## 路线 A：训练完整分类网络，但 FPGA 先只验证第一层 `fc1 + ReLU`
[L506] 
[L507] 这个路线最适合你现在的 FPGA/CIM 阶段。
[L508] 
[L509] ## 设计思想
[L510] 软件端训练一个完整分类器：
[L511] 
[L512] - `784 -> 128 -> 10`
[L513] 
[L514] 但硬件端先实现：
[L515] 
[L516] - `fc1: 784 -> 128`
[L517] - `ReLU`
[L518] 
[L519] 这样：
[L520] - 软件训练目标正确
[L521] - 第一层 feature 是有意义的
[L522] - 硬件联调难度低
[L523] - 后续再接 `fc2`
[L524] 
[L525] ---
[L526] 
[L527] ## 路线 A 完整代码
[L528] 
[L529] ```python
[L530] import os
[L531] import json
[L532] import math
[L533] import numpy as np
[L534] import torch
[L535] import torch.nn as nn
[L536] import torch.optim as optim
[L537] from torchvision import datasets, transforms
[L538] from torch.utils.data import DataLoader
[L539] 
[L540] # =========================================================
[L541] # 1. 模型：完整分类器，但硬件先验证 fc1 + relu
[L542] # =========================================================
[L543] class MLPRouteA(nn.Module):
[L544]     def __init__(self):
[L545]         super().__init__()
[L546]         self.fc1 = nn.Linear(784, 128, bias=True)
[L547]         self.relu = nn.ReLU()
[L548]         self.fc2 = nn.Linear(128, 10, bias=True)
[L549] 
[L550]     def forward(self, x):
[L551]         x = x.view(-1, 784)
[L552]         x = self.fc1(x)
[L553]         x = self.relu(x)
[L554]         x = self.fc2(x)
[L555]         return x
[L556] 
[L557]     def forward_fc1_relu(self, x):
[L558]         x = x.view(-1, 784)
[L559]         x = self.fc1(x)
[L560]         x = self.relu(x)
[L561]         return x
[L562] 
[L563] 
[L564] # =========================================================
[L565] # 2. 训练
[L566] # =========================================================
[L567] def train_route_a(model_path="mlp_route_a.pth", epochs=5, batch_size=128, lr=1e-3, device="cpu"):
[L568]     transform = transforms.Compose([
[L569]         transforms.ToTensor(),  # 保持 [0,1]，更贴近硬件
[L570]     ])
[L571] 
[L572]     train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
[L573]     test_dataset  = datasets.MNIST("./data", train=False, download=True, transform=transform)
[L574] 
[L575]     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
[L576]     test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
[L577] 
[L578]     model = MLPRouteA().to(device)
[L579]     criterion = nn.CrossEntropyLoss()
[L580]     optimizer = optim.Adam(model.parameters(), lr=lr)
[L581] 
[L582]     for epoch in range(epochs):
[L583]         model.train()
[L584]         total_loss = 0.0
[L585]         correct = 0
[L586]         total = 0
[L587] 
[L588]         for data, target in train_loader:
[L589]             data, target = data.to(device), target.to(device)
[L590] 
[L591]             optimizer.zero_grad()
[L592]             logits = model(data)
[L593]             loss = criterion(logits, target)
[L594]             loss.backward()
[L595]             optimizer.step()
[L596] 
[L597]             total_loss += loss.item() * data.size(0)
[L598]             pred = logits.argmax(dim=1)
[L599]             correct += (pred == target).sum().item()
[L600]             total += target.size(0)
[L601] 
[L602]         train_loss = total_loss / total
[L603]         train_acc = correct / total
[L604] 
[L605]         model.eval()
[L606]         test_correct = 0
[L607]         test_total = 0
[L608]         with torch.no_grad():
[L609]             for data, target in test_loader:
[L610]                 data, target = data.to(device), target.to(device)
[L611]                 logits = model(data)
[L612]                 pred = logits.argmax(dim=1)
[L613]                 test_correct += (pred == target).sum().item()
[L614]                 test_total += target.size(0)
[L615] 
[L616]         test_acc = test_correct / test_total
[L617]         print(f"Epoch [{epoch+1}/{epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")
[L618] 
[L619]     torch.save(model.state_dict(), model_path)
[L620]     print(f"Model saved to {model_path}")
[L621] 
[L622] 
[L623] if __name__ == "__main__":
[L624]     train_route_a()
[L625] ```
[L626] 
[L627] ---
[L628] 
[L629] ## 路线 B：直接做完整 MNIST 分类部署
[L630] 
[L631] 这个路线是最后的完整目标。
[L632] 
[L633] ---
[L634] 
[L635] ## 路线 B 完整代码
[L636] 
[L637] 和路线 A 训练结构几乎一样，但后续量化和导出会针对两层都做。
[L638] 
[L639] ```python
[L640] import os
[L641] import torch
[L642] import torch.nn as nn
[L643] import torch.optim as optim
[L644] from torchvision import datasets, transforms
[L645] from torch.utils.data import DataLoader
[L646] 
[L647] class MLPRouteB(nn.Module):
[L648]     def __init__(self):
[L649]         super().__init__()
[L650]         self.fc1 = nn.Linear(784, 128, bias=True)
[L651]         self.relu = nn.ReLU()
[L652]         self.fc2 = nn.Linear(128, 10, bias=True)
[L653] 
[L654]     def forward(self, x):
[L655]         x = x.view(-1, 784)
[L656]         x = self.fc1(x)
[L657]         x = self.relu(x)
[L658]         x = self.fc2(x)
[L659]         return x
[L660] 
[L661] def train_route_b(model_path="mlp_route_b.pth", epochs=5, batch_size=128, lr=1e-3, device="cpu"):
[L662]     transform = transforms.Compose([
[L663]         transforms.ToTensor(),  # 保持 [0,1]
[L664]     ])
[L665] 
[L666]     train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
[L667]     test_dataset  = datasets.MNIST("./data", train=False, download=True, transform=transform)
[L668] 
[L669]     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
[L670]     test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
[L671] 
[L672]     model = MLPRouteB().to(device)
[L673]     criterion = nn.CrossEntropyLoss()
[L674]     optimizer = optim.Adam(model.parameters(), lr=lr)
[L675] 
[L676]     for epoch in range(epochs):
[L677]         model.train()
[L678]         total_loss = 0.0
[L679]         correct = 0
[L680]         total = 0
[L681] 
[L682]         for data, target in train_loader:
[L683]             data, target = data.to(device), target.to(device)
[L684] 
[L685]             optimizer.zero_grad()
[L686]             logits = model(data)
[L687]             loss = criterion(logits, target)
[L688]             loss.backward()
[L689]             optimizer.step()
[L690] 
[L691]             total_loss += loss.item() * data.size(0)
[L692]             pred = logits.argmax(dim=1)
[L693]             correct += (pred == target).sum().item()
[L694]             total += target.size(0)
[L695] 
[L696]         train_loss = total_loss / total
[L697]         train_acc = correct / total
[L698] 
[L699]         model.eval()
[L700]         test_correct = 0
[L701]         test_total = 0
[L702]         with torch.no_grad():
[L703]             for data, target in test_loader:
[L704]                 data, target = data.to(device), target.to(device)
[L705]                 logits = model(data)
[L706]                 pred = logits.argmax(dim=1)
[L707]                 test_correct += (pred == target).sum().item()
[L708]                 test_total += target.size(0)
[L709] 
[L710]         test_acc = test_correct / test_total
[L711]         print(f"Epoch [{epoch+1}/{epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")
[L712] 
[L713]     torch.save(model.state_dict(), model_path)
[L714]     print(f"Model saved to {model_path}")
[L715] 
[L716] 
[L717] if __name__ == "__main__":
[L718]     train_route_b()
[L719] ```
[L720] 
[L721] ---
[L722] 
[L723] # 2. 关于量化函数工程问题
[L724] 
[L725] 你提了 3 个核心点，我分别给出方案。
[L726] 
[L727] ---
[L728] 
[L729] ## 2.1 zero_point 不是整数，怎么改
[L730] 
[L731] 你原来的问题在这里：
[L732] 
[L733] ```python
[L734] zero_point = qmin - min_val / scale
[L735] ```
[L736] 
[L737] 这个是浮点数，不适合硬件。
[L738] 
[L739] ---
[L740] 
[L741] ## 修改后的通用量化函数
[L742] 
[L743] 我给你一个**对 FPGA 更友好**的版本，支持：
[L744] 
[L745] - `zero_point` 取整
[L746] - clamp 到合法范围
[L747] - 支持对称量化和非对称量化
[L748] - 返回整数参数
[L749] 
[L750] ```python
[L751] import torch
[L752] 
[L753] def quantize_tensor_affine(tensor, num_bits=8, symmetric=False):
[L754]     """
[L755]     返回:
[L756]         q_tensor: 整数张量
[L757]         scale: float
[L758]         zero_point: int
[L759]         qmin, qmax: 整数范围
[L760]     """
[L761]     if num_bits == 8:
[L762]         qmin, qmax = -128, 127
[L763]         dtype = torch.int8
[L764]     else:
[L765]         raise ValueError("Only int8 supported in this example.")
[L766] 
[L767]     x_min = tensor.min().item()
[L768]     x_max = tensor.max().item()
[L769] 
[L770]     if symmetric:
[L771]         max_abs = max(abs(x_min), abs(x_max))
[L772]         scale = max_abs / qmax if max_abs != 0 else 1.0
[L773]         zero_point = 0
[L774]     else:
[L775]         if x_max == x_min:
[L776]             scale = 1.0
[L777]             zero_point = 0
[L778]         else:
[L779]             scale = (x_max - x_min) / (qmax - qmin)
[L780]             zero_point = round(qmin - x_min / scale)
[L781]             zero_point = max(qmin, min(qmax, zero_point))
[L782] 
[L783]     q = torch.round(tensor / scale + zero_point)
[L784]     q = torch.clamp(q, qmin, qmax).to(dtype)
[L785] 
[L786]     return q, float(scale), int(zero_point), qmin, qmax
[L787] ```
[L788] 
[L789] ---
[L790] 
[L791] ## 2.2 每个张量动态量化不适合硬件，合理解决方案
[L792] 
[L793] 你现在的问题是：
[L794] 
[L795] - 每张输入图单独算 `scale/zero_point`
[L796] - 每次输出也单独算
[L797] - 每次 bias 也单独算
[L798] 
[L799] 这会导致硬件参数不停变化，不利于 FPGA 批量验证。
[L800] 
[L801] ---
[L802] 
[L803] ## 合理方案：静态量化 / 全局量化
[L804] 
[L805] ### 输入量化策略
[L806] 对所有输入样本统一使用一个固定参数。
[L807] 
[L808] MNIST 最简单的硬件友好方案：
[L809] 
[L810] - 图像原始像素 `[0,1]`
[L811] - 映射到 int8 的 `[0,127]` 或 `[-128,127]`
[L812] 
[L813] 我建议你用下面两种之一：
[L814] 
[L815] ### 方案 1：无符号思路转 int8，简单贴近硬件
[L816] - 输入范围固定 `[0,1]`
[L817] - 量化成 `[0,127]`
[L818] - 存成 int8 时 zero_point = 0
[L819] 
[L820] 设置：
[L821] ```python
[L822] input_scale = 1.0 / 127.0
[L823] input_zero_point = 0
[L824] ```
[L825] 
[L826] 这样：
[L827] \[
[L828] x_q = round(x / input\_scale)
[L829] \]
[L830] 
[L831] 因为 `x in [0,1]`，所以 `x_q in [0,127]`
[L832] 
[L833] 这非常适合 FPGA。
[L834] 
[L835] ### 方案 2：居中对称量化
[L836] 如果你想让输入有正负，做：
[L837] ```python
[L838] x_float_centered = x_float - 0.5
[L839] input_scale = 0.5 / 127
[L840] zero_point = 0
[L841] ```
[L842] 但这会增加前处理复杂度。
[L843] 
[L844] **建议你现在先用方案 1。**
[L845] 
[L846] ---
[L847] 
[L848] ## 权重量化策略
[L849] 权重建议做**对称量化**：
[L850] 
[L851] - `zero_point = 0`
[L852] - int8
[L853] - 全层固定一个 `w_scale`
[L854] 
[L855] ```python
[L856] def quantize_weight_symmetric(weight):
[L857]     q_weight, w_scale, w_zero, _, _ = quantize_tensor_affine(weight, num_bits=8, symmetric=True)
[L858]     return q_weight, w_scale, w_zero
[L859] ```
[L860] 
[L861] ---
[L862] 
[L863] ## 2.3 偏置不能简单用 int8 单独量化，解决方案和代码
[L864] 
[L865] 这个是工程上必须改的。
[L866] 
[L867] ---
[L868] 
[L869] ## 正确原理
[L870] 
[L871] 如果一层计算为：
[L872] 
[L873] \[
[L874] y = Wx + b
[L875] \]
[L876] 
[L877] 输入和权重分别量化为：
[L878] 
[L879] \[
[L880] x \approx (x_q - z_x)s_x
[L881] \]
[L882] \[
[L883] w \approx (w_q - z_w)s_w
[L884] \]
[L885] 
[L886] 那么累加项单位是：
[L887] 
[L888] \[
[L889] s_x \cdot s_w
[L890] \]
[L891] 
[L892] 所以 bias 应该量化到同一量纲：
[L893] 
[L894] \[
[L895] b_q = round(b / (s_x \cdot s_w))
[L896] \]
[L897] 
[L898] 并且 bias 应该存成 **int32**。
[L899] 
[L900] ---
[L901] 
[L902] ## 偏置量化代码
[L903] 
[L904] ```python
[L905] def quantize_bias_int32(bias_fp32, input_scale, weight_scale):
[L906]     """
[L907]     bias_fp32: torch.Tensor(float32)
[L908]     return: int32 bias
[L909]     """
[L910]     bias_scale = input_scale * weight_scale
[L911]     q_bias = torch.round(bias_fp32 / bias_scale).to(torch.int32)
[L912]     return q_bias, float(bias_scale)
[L913] ```
[L914] 
[L915] ---
[L916] 
[L917] # 3. 关于 `generate_test_vectors()` 的 golden 不一致问题
[L918] 
[L919] 你要求的是：**用整数推理链路生成 golden**。这个必须按硬件链路来。
[L920] 
[L921] 下面我给出完整设计。
[L922] 
[L923] ---
[L924] 
[L925] ## 3.1 单层 FC + ReLU 的整数推理设计（路线 A）
[L926] 
[L927] ---
[L928] 
[L929] ## 整数推理流程
[L930] 
[L931] 对于第一层：
[L932] 
[L933] \[
[L934] acc_j = \sum_{i=0}^{783} x_{q,i} \cdot w_{q,j,i} + b_{q,j}
[L935] \]
[L936] 
[L937] 因为我们设计为：
[L938] 
[L939] - 输入 `zero_point = 0`
[L940] - 权重 `zero_point = 0`
[L941] 
[L942] 所以式子就很简单，不用减零点。
[L943] 
[L944] 然后：
[L945] 
[L946] \[
[L947] relu_j = max(0, acc_j)
[L948] \]
[L949] 
[L950] 如果你要再导出 int8 输出用于后级，也可以再做 requant。
[L951] 
[L952] ---
[L953] 
[L954] ## 3.2 统一的导出和 golden 生成代码（路线 A）
[L955] 
[L956] 这段代码是你现在最应该用的版本。
[L957] 
[L958] ```python
[L959] import os
[L960] import json
[L961] import math
[L962] import numpy as np
[L963] import torch
[L964] import torch.nn as nn
[L965] from torchvision import datasets, transforms
[L966] 
[L967] # =========================================================
[L968] # 模型
[L969] # =========================================================
[L970] class MLPRouteA(nn.Module):
[L971]     def __init__(self):
[L972]         super().__init__()
[L973]         self.fc1 = nn.Linear(784, 128, bias=True)
[L974]         self.relu = nn.ReLU()
[L975]         self.fc2 = nn.Linear(128, 10, bias=True)
[L976] 
[L977]     def forward(self, x):
[L978]         x = x.view(-1, 784)
[L979]         x = self.fc1(x)
[L980]         x = self.relu(x)
[L981]         x = self.fc2(x)
[L982]         return x
[L983] 
[L984] 
[L985] # =========================================================
[L986] # 量化工具
[L987] # =========================================================
[L988] def quantize_tensor_affine(tensor, num_bits=8, symmetric=False):
[L989]     if num_bits == 8:
[L990]         qmin, qmax = -128, 127
[L991]         dtype = torch.int8
[L992]     else:
[L993]         raise ValueError("Only int8 supported.")
[L994] 
[L995]     x_min = tensor.min().item()
[L996]     x_max = tensor.max().item()
[L997] 
[L998]     if symmetric:
[L999]         max_abs = max(abs(x_min), abs(x_max))
[L1000]         scale = max_abs / qmax if max_abs != 0 else 1.0
[L1001]         zero_point = 0
[L1002]     else:
[L1003]         if x_max == x_min:
[L1004]             scale = 1.0
[L1005]             zero_point = 0
[L1006]         else:
[L1007]             scale = (x_max - x_min) / (qmax - qmin)
[L1008]             zero_point = round(qmin - x_min / scale)
[L1009]             zero_point = max(qmin, min(qmax, zero_point))
[L1010] 
[L1011]     q = torch.round(tensor / scale + zero_point)
[L1012]     q = torch.clamp(q, qmin, qmax).to(dtype)
[L1013]     return q, float(scale), int(zero_point), qmin, qmax
[L1014] 
[L1015] 
[L1016] def quantize_input_fixed_0_1(x):
[L1017]     """
[L1018]     x in [0,1]
[L1019]     量化到 int8 [0,127]
[L1020]     """
[L1021]     scale = 1.0 / 127.0
[L1022]     zero_point = 0
[L1023]     q = torch.round(x / scale)
[L1024]     q = torch.clamp(q, 0, 127).to(torch.int8)
[L1025]     return q, scale, zero_point
[L1026] 
[L1027] 
[L1028] def quantize_weight_symmetric(weight):
[L1029]     q_w, w_scale, w_zp, _, _ = quantize_tensor_affine(weight, num_bits=8, symmetric=True)
[L1030]     return q_w, w_scale, w_zp
[L1031] 
[L1032] 
[L1033] def quantize_bias_int32(bias_fp32, input_scale, weight_scale):
[L1034]     bias_scale = input_scale * weight_scale
[L1035]     q_bias = torch.round(bias_fp32 / bias_scale).to(torch.int32)
[L1036]     return q_bias, float(bias_scale)
[L1037] 
[L1038] 
[L1039] def compute_requant_params(input_scale, weight_scale, output_scale, shift=24):
[L1040]     """
[L1041]     将 real_multiplier = (input_scale * weight_scale) / output_scale
[L1042]     近似成 integer multiplier + right shift
[L1043]     """
[L1044]     real_multiplier = (input_scale * weight_scale) / output_scale
[L1045]     multiplier = int(round(real_multiplier * (1 << shift)))
[L1046]     return multiplier, shift, real_multiplier
[L1047] 
[L1048] 
[L1049] def requantize_int32_to_int8(acc_int32, multiplier, shift, out_zero_point=0, relu=False):
[L1050]     y = (acc_int32.to(torch.int64) * multiplier) >> shift
[L1051]     y = y + out_zero_point
[L1052]     if relu:
[L1053]         y = torch.maximum(y, torch.tensor(out_zero_point, dtype=y.dtype))
[L1054]     y = torch.clamp(y, -128, 127).to(torch.int8)
[L1055]     return y
[L1056] 
[L1057] 
[L1058] # =========================================================
[L1059] # hex 导出
[L1060] # =========================================================
[L1061] def write_int8_hex_per_line(tensor, filename):
[L1062]     tensor = tensor.flatten().cpu().numpy()
[L1063]     with open(filename, "w") as f:
[L1064]         for v in tensor:
[L1065]             f.write(f"{int(v) & 0xFF:02x}\n")
[L1066] 
[L1067] def write_int32_hex_per_line(tensor, filename):
[L1068]     tensor = tensor.flatten().cpu().numpy()
[L1069]     with open(filename, "w") as f:
[L1070]         for v in tensor:
[L1071]             f.write(f"{int(v) & 0xFFFFFFFF:08x}\n")
[L1072] 
[L1073] 
[L1074] # =========================================================
[L1075] # 导出第一层量化参数
[L1076] # =========================================================
[L1077] def export_fc1_quantized(model_path="mlp_route_a.pth", out_dir="quant_route_a"):
[L1078]     os.makedirs(out_dir, exist_ok=True)
[L1079] 
[L1080]     model = MLPRouteA()
[L1081]     model.load_state_dict(torch.load(model_path, map_location="cpu"))
[L1082]     model.eval()
[L1083] 
[L1084]     weight = model.fc1.weight.data.clone()   # [128,784]
[L1085]     bias   = model.fc1.bias.data.clone()     # [128]
[L1086] 
[L1087]     # 固定输入量化参数
[L1088]     input_scale = 1.0 / 127.0
[L1089]     input_zero_point = 0
[L1090] 
[L1091]     # 权重量化（对称）
[L1092]     q_weight, weight_scale, weight_zero_point = quantize_weight_symmetric(weight)
[L1093] 
[L1094]     # 偏置量化（int32）
[L1095]     q_bias, bias_scale = quantize_bias_int32(bias, input_scale, weight_scale)
[L1096] 
[L1097]     # 输出量化参数（给 fc1 输出用）
[L1098]     # 这里按浮点 fc1 输出范围估计一个固定输出 scale；工程上可用校准集更稳
[L1099]     with torch.no_grad():
[L1100]         dummy = torch.rand(256, 1, 28, 28)
[L1101]         fc1_out = model.relu(model.fc1(dummy.view(-1, 784)))
[L1102]     max_out = fc1_out.max().item()
[L1103]     output_scale = max_out / 127.0 if max_out != 0 else 1.0
[L1104]     output_zero_point = 0
[L1105] 
[L1106]     multiplier, shift, real_multiplier = compute_requant_params(
[L1107]         input_scale, weight_scale, output_scale, shift=24
[L1108]     )
[L1109] 
[L1110]     write_int8_hex_per_line(q_weight, os.path.join(out_dir, "fc1_weight.hex"))
[L1111]     write_int32_hex_per_line(q_bias, os.path.join(out_dir, "fc1_bias.hex"))
[L1112] 
[L1113]     config = {
[L1114]         "input_scale": input_scale,
[L1115]         "input_zero_point": input_zero_point,
[L1116]         "weight_scale": weight_scale,
[L1117]         "weight_zero_point": weight_zero_point,
[L1118]         "bias_scale": bias_scale,
[L1119]         "output_scale": output_scale,
[L1120]         "output_zero_point": output_zero_point,
[L1121]         "requant_multiplier": multiplier,
[L1122]         "requant_shift": shift,
[L1123]         "requant_real_multiplier": real_multiplier,
[L1124]         "weight_shape": [128, 784],
[L1125]         "bias_shape": [128],
[L1126]         "weight_layout": "[out][in], row-major",
[L1127]         "input_layout": "[in]",
[L1128]         "output_layout": "[out]"
[L1129]     }
[L1130] 
[L1131]     with open(os.path.join(out_dir, "fc1_quant_config.json"), "w") as f:
[L1132]         json.dump(config, f, indent=2)
[L1133] 
[L1134]     print("Exported fc1 quantized parameters.")
[L1135] 
[L1136] 
[L1137] # =========================================================
[L1138] # 生成整数链路 golden
[L1139] # =========================================================
[L1140] def generate_test_vectors_route_a(model_path="mlp_route_a.pth", out_dir="quant_route_a", num_samples=10):
[L1141]     os.makedirs(out_dir, exist_ok=True)
[L1142] 
[L1143]     model = MLPRouteA()
[L1144]     model.load_state_dict(torch.load(model_path, map_location="cpu"))
[L1145]     model.eval()
[L1146] 
[L1147]     # 载入模型参数并量化
[L1148]     weight = model.fc1.weight.data.clone()   # [128,784]
[L1149]     bias   = model.fc1.bias.data.clone()
[L1150] 
[L1151]     input_scale = 1.0 / 127.0
[L1152]     input_zero_point = 0
[L1153] 
[L1154]     q_weight, weight_scale, weight_zero_point = quantize_weight_symmetric(weight)
[L1155]     q_bias, bias_scale = quantize_bias_int32(bias, input_scale, weight_scale)
[L1156] 
[L1157]     with torch.no_grad():
[L1158]         dummy = torch.rand(256, 1, 28, 28)
[L1159]         fc1_out = model.relu(model.fc1(dummy.view(-1, 784)))
[L1160]     max_out = fc1_out.max().item()
[L1161]     output_scale = max_out / 127.0 if max_out != 0 else 1.0
[L1162]     output_zero_point = 0
[L1163] 
[L1164]     multiplier, shift, _ = compute_requant_params(input_scale, weight_scale, output_scale, shift=24)
[L1165] 
[L1166]     transform = transforms.Compose([
[L1167]         transforms.ToTensor(),   # 不做 Normalize，统一输入范围 [0,1]
[L1168]     ])
[L1169]     test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
[L1170] 
[L1171]     labels = []
[L1172] 
[L1173]     for i in range(num_samples):
[L1174]         img, label = test_dataset[i]
[L1175]         x = img.view(-1)  # [784]
[L1176] 
[L1177]         # 固定输入量化
[L1178]         q_x, _, _ = quantize_input_fixed_0_1(x)  # [784]
[L1179] 
[L1180]         # int32 累加： [128,784] x [784] -> [128]
[L1181]         acc = torch.sum(q_weight.to(torch.int32) * q_x.to(torch.int32).unsqueeze(0), dim=1) + q_bias
[L1182] 
[L1183]         # ReLU(int32)
[L1184]         relu_acc = torch.clamp(acc, min=0)
[L1185] 
[L1186]         # requant 到 int8，便于后级或观测
[L1187]         q_out = requantize_int32_to_int8(
[L1188]             relu_acc, multiplier, shift, out_zero_point=output_zero_point, relu=False
[L1189]         )
[L1190] 
[L1191]         # 保存
[L1192]         write_int8_hex_per_line(q_x,   os.path.join(out_dir, f"input_{i}.hex"))
[L1193]         write_int32_hex_per_line(acc,  os.path.join(out_dir, f"golden_fc1_acc_{i}.hex"))
[L1194]         write_int32_hex_per_line(relu_acc, os.path.join(out_dir, f"golden_fc1_relu_acc_{i}.hex"))
[L1195]         write_int8_hex_per_line(q_out, os.path.join(out_dir, f"golden_fc1_out_{i}.hex"))
[L1196] 
[L1197]         labels.append(int(label))
[L1198]         print(f"sample {i}: label={label}")
[L1199] 
[L1200]     with open(os.path.join(out_dir, "labels.txt"), "w") as f:
[L1201]         for lb in labels:
[L1202]             f.write(f"{lb}\n")
[L1203] 
[L1204]     print("Generated integer golden test vectors for route A.")
[L1205] ```
[L1206] 
[L1207] ---
[L1208] 
[L1209] # 4. 输入测试向量统一量化策略
[L1210] 
[L1211] 你要的是**贴近硬件即可**。那我建议你用下面这个方案。
[L1212] 
[L1213] ---
[L1214] 
[L1215] ## 推荐方案：输入保持 `[0,1]`，统一量化到 int8 `[0,127]`
[L1216] 
[L1217] ### 设计理由
[L1218] 这对 FPGA 最友好：
[L1219] 
[L1220] - 不用每张图动态算 scale
[L1221] - 不用保留 MNIST 的 mean/std normalize
[L1222] - 输入像素天然接近图像传感器/存储格式
[L1223] - 便于后面 BRAM 或 ROM 存图
[L1224] 
[L1225] ### 固定规则
[L1226] \[
[L1227] x_q = round(x \times 127)
[L1228] \]
[L1229] 其中 `x in [0,1]`
[L1230] 
[L1231] 即：
[L1232] 
[L1233] ```python
[L1234] def quantize_input_fixed_0_1(x):
[L1235]     scale = 1.0 / 127.0
[L1236]     zero_point = 0
[L1237]     q = torch.round(x / scale)
[L1238]     q = torch.clamp(q, 0, 127).to(torch.int8)
[L1239]     return q, scale, zero_point
[L1240] ```
[L1241] 
[L1242] ---
[L1243] 
[L1244] ## 为什么我不建议你继续用：
[L1245] ```python
[L1246] transforms.Normalize((0.1307,), (0.3081,))
[L1247] ```
[L1248] 
[L1249] 因为这会把输入变成带负数的浮点分布，虽然适合软件训练，但不适合你现在的 FPGA 验证阶段：
[L1250] 
[L1251] - 输入分布复杂
[L1252] - 量化参数更麻烦
[L1253] - debug 不直观
[L1254] 
[L1255] 所以你现在更适合：
[L1256] 
[L1257] - 训练和测试都直接 `ToTensor()`
[L1258] - 用统一固定输入量化
[L1259] 
[L1260] ---
[L1261] 
[L1262] # 5. 你当前权重输出顺序，对应硬件怎么设计读取顺序
[L1263] 
[L1264] 你当前 notebook 的写法是：
[L1265] 
[L1266] ```python
[L1267] for row in q_weight:
[L1268]     for val in row:
[L1269]         f.write(f'{val.item() & 0xFF:02x}\n')
[L1270] ```
[L1271] 
[L1272] 而 `weight = model.fc1.weight.data` 的 shape 是：
[L1273] 
[L1274] ```python
[L1275] [128, 784]
[L1276] ```
[L1277] 
[L1278] 在 PyTorch 里，`nn.Linear(784,128)` 的权重定义是：
[L1279] 
[L1280] \[
[L1281] W[j][i]
[L1282] \]
[L1283] 
[L1284] 其中：
[L1285] 
[L1286] - `j` 是输出神经元索引，范围 `0~127`
[L1287] - `i` 是输入特征索引，范围 `0~783`
[L1288] 
[L1289] ---
[L1290] 
[L1291] ## 你现在的导出顺序是什么
[L1292] 
[L1293] 它是：
[L1294] 
[L1295] - 先写第 0 个输出神经元的 784 个权重
[L1296] - 再写第 1 个输出神经元的 784 个权重
[L1297] - ...
[L1298] - 最后写第 127 个输出神经元的 784 个权重
[L1299] 
[L1300] 也就是：
[L1301] 
[L1302] \[
[L1303] W[0][0], W[0][1], ..., W[0][783], W[1][0], ..., W[127][783]
[L1304] \]
[L1305] 
[L1306] 这是典型的：
[L1307] 
[L1308] - **按输出通道优先**
[L1309] - **行优先 row-major**
[L1310] - **布局 `[out][in]`**
[L1311] 
[L1312] ---
[L1313] 
[L1314] ## 对应的硬件读取顺序应该怎么设计
[L1315] 
[L1316] 如果你的硬件是“一个输出神经元一轮累加”的思路，那么最自然。
[L1317] 
[L1318] ---
[L1319] 
[L1320] ## 硬件读取方式 1：逐个输出神经元计算
[L1321] 
[L1322] ### 第 `j` 个输出的计算
[L1323] \[
[L1324] y_j = \sum_{i=0}^{783} x_i \cdot W[j][i] + b_j
[L1325] \]
[L1326] 
[L1327] ### 内存地址设计
[L1328] 如果 `weights.hex` 直接线性装入 BRAM，则：
[L1329] 
[L1330] \[
[L1331] addr = j \times 784 + i
[L1332] \]
[L1333] 
[L1334] 所以：
[L1335] 
[L1336] - 固定一个输出神经元 `j`
[L1337] - 地址从 `j*784` 扫到 `j*784+783`
[L1338] - 同时输入向量 `x[0:783]` 顺序送入
[L1339] - MAC 累加完成后，加 `bias[j]`
[L1340] - 输出 `y_j`
[L1341] 
[L1342] 然后 `j++`
[L1343] 
[L1344] ---
[L1345] 
[L1346] ## 这种方式的优点
[L1347] - 和你现在导出的文件顺序完全一致
[L1348] - 地址生成最简单
[L1349] - 软件和硬件最容易对齐
[L1350] - 最适合第一版 FPGA 联调
[L1351] 
[L1352] ---
[L1353] 
[L1354] ## 你现在应该怎么设计硬件
[L1355] 
[L1356] ### 输入存储
[L1357] `input_i.hex` 顺序也是：
[L1358] \[
[L1359] x[0], x[1], ..., x[783]
[L1360] \]
[L1361] 
[L1362] ### 权重存储
[L1363] `weights.hex` 顺序：
[L1364] \[
[L1365] W[0][0], W[0][1], ..., W[0][783], W[1][0], ..., W[127][783]
[L1366] \]
[L1367] 
[L1368] ### bias 存储
[L1369] `bias.hex` 顺序：
[L1370] \[
[L1371] b[0], b[1], ..., b[127]
[L1372] \]
[L1373] 
[L1374] ---
[L1375] 
[L1376] ## 控制流程
[L1377] 对于输出神经元 `j`：
[L1378] 
[L1379] 1. `acc = 0`
[L1380] 2. 对 `i = 0 ~ 783`
[L1381]    - 读 `x[i]`
[L1382]    - 读 `W[j][i]`
[L1383]    - `acc += x[i] * W[j][i]`
[L1384] 3. `acc += bias[j]`
[L1385] 4. `acc = relu(acc)`
[L1386] 5. 写 `out[j]`
[L1387] 
[L1388] 然后进入下一个 `j`
[L1389] 
[L1390] ---
[L1391] 
[L1392] ## 地址公式
[L1393] ### 权重地址
[L1394] \[
[L1395] weight\_addr = out\_idx \times 784 + in\_idx
[L1396] \]
[L1397] 
[L1398] ### 输入地址
[L1399] \[
[L1400] input\_addr = in\_idx
[L1401] \]
[L1402] 
[L1403] ### 偏置地址
[L1404] \[
[L1405] bias\_addr = out\_idx
[L1406] \]
[L1407] 
[L1408] ---
[L1409] 
[L1410] ## 什么时候要改输出顺序
[L1411] 只有在你未来做这些架构时才需要变：
[L1412] 
[L1413] - 多 PE 并行读多个输出神经元
[L1414] - 按列块分块存储
[L1415] - CIM 阵列按 bitline/wordline 特定映射
[L1416] - 片上 SRAM 宽度要求按 8/16/32 个权重打包
[L1417] 
[L1418] 但对于你现在这个阶段，**完全没必要先改**。  
[L1419] 你当前顺序其实正适合第一版硬件。
[L1420] 
[L1421] ---
[L1422] 
[L1423] # 我给你的最终建议
[L1424] 
[L1425] ## 你现在最推荐的工程路线
[L1426] 先走：
[L1427] 
[L1428] - **路线 A**
[L1429] - 完整训练分类器
[L1430] - FPGA 只做第一层 `fc1 + ReLU`
[L1431] - 输入统一量化 `[0,1] -> int8 [0,127]`
[L1432] - 权重 int8 对称量化
[L1433] - bias int32
[L1434] - golden 用整数推理链路生成
[L1435] - 权重按你现在的 `[out][in]` 顺序导出
[L1436] 
[L1437] 这是最稳、最容易跑通的一条路。
[L1438] 
[L1439] ---
[L1440] 
[L1441] ## 你下一步最值得做的事
[L1442] 把你当前 notebook 分成三个独立脚本：
[L1443] 
[L1444] 1. `train_route_a.py`
[L1445] 2. `export_fc1_quant.py`
[L1446] 3. `generate_test_vectors_route_a.py`
[L1447] 
[L1448] 这样后面调试不会乱。
[L1449] 
[L1450] 如果你要，我下一条可以直接继续给你一份**整理好的、可直接复制运行的完整工程版代码**，把路线 A 的训练、量化导出、golden 生成合并成一个干净版本。
[L1451] 
[L1452] 用户:
[L1453] 关于1.现在没有在训练一个 MNIST 分类器的问题，请分别给我路线a和路线b的完整代码；
[L1454] 
[L1455] 关于2.量化函数有工程问题，不适合直接对接 FPGA，
[L1456] zero_point不是整数的问题帮我修改；每个张量动态量化这个问题给我一个合理的解决方案；偏置不能简单用 int8 单独量化请帮我提出解决方案和代码；
[L1457] 
[L1458] 关于3.generate_test_vectors() 生成的 golden，和硬件真实计算链路不一致，整数推理链路生成 golden，帮我完成设计；
[L1459] 
[L1460] 关于4输入测试向量的量化方式不适合硬件批量验证，帮我设计统一输入量化策略，贴近硬件即可；
[L1461] 
[L1462] 关于5. 导出的权重格式本身基本可用，但有几个注意点，这里文件顺序你先记住我的输出顺序，讲解一下我现在的输出顺序应该如何设计硬件读取顺序；
[L1463] 
[L1464] 
[L1465] 问题6,7你帮我解决一下。
[L1466] 
[L1467] 按照你提出的方案二：直接瞄准最终演示
[L1468] 
[L1469] 如果你们想最后做“MNIST 分类结果上板演示”，那就：
[L1470] 
[L1471] 模型改为两层 MLP
[L1472] 
[L1473] 训练用 CrossEntropyLoss
[L1474] 
[L1475] 做静态量化
[L1476] 
[L1477] 先用 Python 写一个整数推理参考实现
[L1478] 
[L1479] 再把这个整数推理结果作为 FPGA golden进行处理。
[L1480] 其他应该修改的你帮我修改，和我说一声即可。
[L1481] AI助手:
[L1482] 我先基于你现有 notebook 的结构，把方案 A、方案 B，以及静态量化、整数推理 golden、测试向量导出这几部分整理成能直接落地的代码。然后我会把硬件读取顺序也一并说明清楚，保证软件导出和 RTL 地址映射能闭环。
[L1483] 
[L1484] AI助手:
[L1485] 下面我直接按你的要求给出可落地代码和设计说明。  
[L1486] 我把内容分成两部分：
[L1487] 
[L1488] 1. **路线 A 完整代码**：训练完整 MLP，但 FPGA 先只实现第一层 `fc1 + relu`，用于硬件验证  
[L1489] 2. **路线 B 完整代码**：直接瞄准最终演示，做完整两层 MLP、静态量化、整数推理 golden 导出
[L1490] 
[L1491] 你最后明确说“**按方案二直接瞄准最终演示**”，所以我建议你真正落地时优先用**路线 B**。  
[L1492] 路线 A 我也完整给你，方便你前期做单层阵列验证。
[L1493] 
[L1494] ---
[L1495] 
[L1496] # 一、你提出的几个问题，我已经统一改掉了
[L1497] 
[L1498] 这次代码里我已经一起处理了这些问题：
[L1499] 
[L1500] ## 已修改的问题
[L1501] ### 1）训练目标错误
[L1502] - 路线 A / B 都改成了真正的分类训练
[L1503] - 使用 `CrossEntropyLoss`
[L1504] 
[L1505] ### 2）`zero_point` 不是整数
[L1506] - 已改为**整数 zero_point**
[L1507] - 并做了合法范围裁剪
[L1508] 
[L1509] ### 3）动态量化不适合 FPGA
[L1510] - 改为**静态量化**
[L1511] - 输入、权重、输出都用**固定 scale / zero_point**
[L1512] - 不再每个样本单独算量化参数
[L1513] 
[L1514] ### 4）bias 不能简单 int8
[L1515] - 改为：
[L1516]   - `bias_q = round(bias_fp / (input_scale * weight_scale))`
[L1517]   - 使用 **int32** 保存
[L1518] 
[L1519] ### 5）golden 与硬件链路不一致
[L1520] - 改为**整数推理参考链路**
[L1521] - golden 由整数乘加、加 bias、ReLU、requant 得到
[L1522] - 和 FPGA 数据通路一致
[L1523] 
[L1524] ### 6）输入量化方式不适合批量验证
[L1525] - 统一输入量化策略
[L1526] - 采用更贴近硬件的方案：**MNIST 像素 0~1 -> uint8 0~255，再映射为 int8（减去 128）**
[L1527] - 固定 `input_scale = 1/255`，`input_zero_point = -128`
[L1528] 
[L1529] ### 7）其他问题我也一起修了
[L1530] - `torch.load(..., map_location=device)`
[L1531] - 增加 test accuracy
[L1532] - 保存标签文件
[L1533] - 保存中间层 int32 累加结果，便于 RTL 分级定位
[L1534] - 导出 json 配置，方便 FPGA 端对齐
[L1535] 
[L1536] ---
[L1537] 
[L1538] # 二、统一量化策略设计
[L1539] 
[L1540] 你要求“贴近硬件即可”，我这里给你一个非常适合 FPGA 的设计：
[L1541] 
[L1542] ## 输入量化
[L1543] 原始 MNIST 图像来自 `ToTensor()` 后范围是 `[0,1]`
[L1544] 
[L1545] 统一量化为：
[L1546] 
[L1547] \[
[L1548] x_q = round(x / s_x) + z_x
[L1549] \]
[L1550] 
[L1551] 其中：
[L1552] 
[L1553] - `s_x = 1 / 255`
[L1554] - `z_x = -128`
[L1555] 
[L1556] 这样就等价于：
[L1557] 
[L1558] ```python
[L1559] x_u8 = round(x * 255)      # 0~255
[L1560] x_q  = x_u8 - 128          # -128~127
[L1561] ```
[L1562] 
[L1563] 优点：
[L1564] - 很贴近硬件
[L1565] - 不需要每张图动态量化
[L1566] - FPGA 端容易处理
[L1567] - BRAM 存储方便
[L1568] 
[L1569] ## 权重量化
[L1570] 对每一层权重做**对称量化**：
[L1571] 
[L1572] \[
[L1573] w_q = round(w / s_w)
[L1574] \]
[L1575] 
[L1576] 其中：
[L1577] 
[L1578] \[
[L1579] s_w = \max(|w|) / 127
[L1580] \]
[L1581] 
[L1582] - `w_zero_point = 0`
[L1583] - `w_q` 存为 int8
[L1584] 
[L1585] ## 偏置量化
[L1586] 偏置必须与累加器同量纲：
[L1587] 
[L1588] \[
[L1589] b_q = round(b / (s_x \cdot s_w))
[L1590] \]
[L1591] 
[L1592] - 存为 int32
[L1593] 
[L1594] ## 输出量化
[L1595] 累加后得到 int32：
[L1596] 
[L1597] \[
[L1598] acc = \sum (x_q - z_x) \cdot (w_q - z_w) + b_q
[L1599] \]
[L1600] 
[L1601] 然后 ReLU，再 requant：
[L1602] 
[L1603] \[
[L1604] y_q = round(acc \cdot (s_x s_w / s_y)) + z_y
[L1605] \]
[L1606] 
[L1607] 输出层如需分类 argmax，最终也可直接基于 int32 或 requant 后 int8 做。
[L1608] 
[L1609] ---
[L1610] 
[L1611] # 三、路线 A：完整代码  
[L1612] ## 用途
[L1613] - 软件训练完整分类器：`784 -> 128 -> 10`
[L1614] - FPGA 先只实现第一层：`fc1 + relu`
[L1615] - 导出第一层输入、权重、bias、int32 累加 golden、relu 输出 golden
[L1616] 
[L1617] 这条路线适合你前期快速验证 CIM/FC 阵列。
[L1618] 
[L1619] ---
[L1620] 
[L1621] ## 路线 A 完整代码
[L1622] 
[L1623] ```python
[L1624] import os
[L1625] import json
[L1626] import math
[L1627] import numpy as np
[L1628] import torch
[L1629] import torch.nn as nn
[L1630] import torch.optim as optim
[L1631] from torchvision import datasets, transforms
[L1632] from torch.utils.data import DataLoader
[L1633] 
[L1634] # =========================================================
[L1635] # 1. 配置
[L1636] # =========================================================
[L1637] DATA_DIR = "./data"
[L1638] OUT_DIR = "./route_a_output"
[L1639] MODEL_PATH = os.path.join(OUT_DIR, "mlp_route_a.pth")
[L1640] 
[L1641] os.makedirs(OUT_DIR, exist_ok=True)
[L1642] 
[L1643] DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
[L1644] 
[L1645] 
[L1646] # =========================================================
[L1647] # 2. 模型：完整训练，但导出时只导出第一层用于 FPGA
[L1648] # =========================================================
[L1649] class MLPRouteA(nn.Module):
[L1650]     def __init__(self):
[L1651]         super().__init__()
[L1652]         self.fc1 = nn.Linear(784, 128, bias=True)
[L1653]         self.relu = nn.ReLU()
[L1654]         self.fc2 = nn.Linear(128, 10, bias=True)
[L1655] 
[L1656]     def forward(self, x):
[L1657]         x = x.view(-1, 784)
[L1658]         h = self.fc1(x)
[L1659]         h_relu = self.relu(h)
[L1660]         y = self.fc2(h_relu)
[L1661]         return y
[L1662] 
[L1663] 
[L1664] # =========================================================
[L1665] # 3. 数据
[L1666] #   路线 A 训练时仍然可以用标准 Normalize
[L1667] #   因为这里只是先训练出一个可用分类器
[L1668] # =========================================================
[L1669] def get_train_test_loaders(batch_size=128):
[L1670]     transform = transforms.Compose([
[L1671]         transforms.ToTensor(),
[L1672]         transforms.Normalize((0.1307,), (0.3081,))
[L1673]     ])
[L1674] 
[L1675]     train_dataset = datasets.MNIST(
[L1676]         DATA_DIR, train=True, download=True, transform=transform
[L1677]     )
[L1678]     test_dataset = datasets.MNIST(
[L1679]         DATA_DIR, train=False, download=True, transform=transform
[L1680]     )
[L1681] 
[L1682]     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
[L1683]     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
[L1684]     return train_loader, test_loader
[L1685] 
[L1686] 
[L1687] # =========================================================
[L1688] # 4. 训练 / 测试
[L1689] # =========================================================
[L1690] def train_model(num_epochs=5, lr=1e-3):
[L1691]     train_loader, test_loader = get_train_test_loaders()
[L1692] 
[L1693]     model = MLPRouteA().to(DEVICE)
[L1694]     optimizer = optim.Adam(model.parameters(), lr=lr)
[L1695]     criterion = nn.CrossEntropyLoss()
[L1696] 
[L1697]     for epoch in range(num_epochs):
[L1698]         model.train()
[L1699]         running_loss = 0.0
[L1700] 
[L1701]         for data, target in train_loader:
[L1702]             data, target = data.to(DEVICE), target.to(DEVICE)
[L1703] 
[L1704]             optimizer.zero_grad()
[L1705]             logits = model(data)
[L1706]             loss = criterion(logits, target)
[L1707]             loss.backward()
[L1708]             optimizer.step()
[L1709] 
[L1710]             running_loss += loss.item() * data.size(0)
[L1711] 
[L1712]         avg_loss = running_loss / len(train_loader.dataset)
[L1713]         acc = evaluate_model(model, test_loader)
[L1714]         print(f"[Epoch {epoch+1}/{num_epochs}] loss={avg_loss:.6f}, test_acc={acc:.4f}")
[L1715] 
[L1716]     torch.save(model.state_dict(), MODEL_PATH)
[L1717]     print(f"Model saved to: {MODEL_PATH}")
[L1718]     return model
[L1719] 
[L1720] 
[L1721] def evaluate_model(model, test_loader):
[L1722]     model.eval()
[L1723]     correct = 0
[L1724]     total = 0
[L1725] 
[L1726]     with torch.no_grad():
[L1727]         for data, target in test_loader:
[L1728]             data, target = data.to(DEVICE), target.to(DEVICE)
[L1729]             logits = model(data)
[L1730]             pred = logits.argmax(dim=1)
[L1731]             correct += (pred == target).sum().item()
[L1732]             total += target.size(0)
[L1733] 
[L1734]     return correct / total
[L1735] 
[L1736] 
[L1737] # =========================================================
[L1738] # 5. 量化函数
[L1739] #   路线 A：为了 FPGA 只验证第一层，采用固定输入量化策略
[L1740] # =========================================================
[L1741] INT8_QMIN = -128
[L1742] INT8_QMAX = 127
[L1743] INT32_QMIN = -2147483648
[L1744] INT32_QMAX = 2147483647
[L1745] 
[L1746] def clamp(x, low, high):
[L1747]     return max(low, min(high, x))
[L1748] 
[L1749] def quantize_affine_tensor(tensor_fp, scale, zero_point, qmin=-128, qmax=127, dtype=torch.int8):
[L1750]     q = torch.round(tensor_fp / scale + zero_point)
[L1751]     q = torch.clamp(q, qmin, qmax)
[L1752]     return q.to(dtype)
[L1753] 
[L1754] def calc_symmetric_scale(tensor_fp, num_bits=8):
[L1755]     qmax = (2 ** (num_bits - 1)) - 1
[L1756]     max_abs = max(abs(tensor_fp.min().item()), abs(tensor_fp.max().item()))
[L1757]     if max_abs == 0:
[L1758]         return 1.0
[L1759]     return max_abs / qmax
[L1760] 
[L1761] def quantize_weight_symmetric_int8(weight_fp):
[L1762]     scale = calc_symmetric_scale(weight_fp, num_bits=8)
[L1763]     zp = 0
[L1764]     weight_q = quantize_affine_tensor(weight_fp, scale, zp, -128, 127, torch.int8)
[L1765]     return weight_q, scale, zp
[L1766] 
[L1767] def quantize_bias_int32(bias_fp, input_scale, weight_scale):
[L1768]     bias_scale = input_scale * weight_scale
[L1769]     bias_q = torch.round(bias_fp / bias_scale)
[L1770]     bias_q = torch.clamp(bias_q, INT32_QMIN, INT32_QMAX).to(torch.int32)
[L1771]     return bias_q, bias_scale
[L1772] 
[L1773] def quantize_input_hw_style(img_fp_01):
[L1774]     """
[L1775]     img_fp_01: 原始像素 [0,1]
[L1776]     统一策略：
[L1777]         x_u8 = round(x * 255)
[L1778]         x_q  = x_u8 - 128
[L1779]     对应：
[L1780]         input_scale = 1/255
[L1781]         input_zero_point = -128
[L1782]     """
[L1783]     input_scale = 1.0 / 255.0
[L1784]     input_zero_point = -128
[L1785] 
[L1786]     x_u8 = torch.round(img_fp_01 * 255.0)
[L1787]     x_u8 = torch.clamp(x_u8, 0, 255)
[L1788]     x_q = x_u8.to(torch.int16) - 128
[L1789]     x_q = torch.clamp(x_q, -128, 127).to(torch.int8)
[L1790] 
[L1791]     return x_q, input_scale, input_zero_point
[L1792] 
[L1793] 
[L1794] # =========================================================
[L1795] # 6. 第一层整数推理 golden
[L1796] #   只对应 fc1 + relu
[L1797] # =========================================================
[L1798] def fc1_integer_reference(x_q, w_q, b_q, x_zp=-128, w_zp=0):
[L1799]     """
[L1800]     x_q: [784] int8
[L1801]     w_q: [128, 784] int8
[L1802]     b_q: [128] int32
[L1803] 
[L1804]     acc_j = sum_i (x_q[i] - x_zp)*(w_q[j,i] - w_zp) + b_q[j]
[L1805]     relu_acc = max(acc_j, 0)
[L1806]     """
[L1807]     x_int = x_q.to(torch.int32) - int(x_zp)
[L1808]     w_int = w_q.to(torch.int32) - int(w_zp)
[L1809] 
[L1810]     acc = torch.matmul(w_int, x_int) + b_q
[L1811]     relu_acc = torch.clamp(acc, min=0)
[L1812]     return acc.to(torch.int32), relu_acc.to(torch.int32)
[L1813] 
[L1814] 
[L1815] # =========================================================
[L1816] # 7. HEX 导出
[L1817] # =========================================================
[L1818] def write_int8_hex_1d(tensor, path):
[L1819]     with open(path, "w") as f:
[L1820]         for v in tensor.flatten():
[L1821]             f.write(f"{int(v.item()) & 0xFF:02x}\n")
[L1822] 
[L1823] def write_int32_hex_1d(tensor, path):
[L1824]     with open(path, "w") as f:
[L1825]         for v in tensor.flatten():
[L1826]             f.write(f"{int(v.item()) & 0xFFFFFFFF:08x}\n")
[L1827] 
[L1828] def export_route_a_artifacts(num_samples=10):
[L1829]     # 加载训练后的模型
[L1830]     model = MLPRouteA().to(DEVICE)
[L1831]     model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
[L1832]     model.eval()
[L1833] 
[L1834]     # 只提取 fc1
[L1835]     fc1_w_fp = model.fc1.weight.data.cpu()   # [128, 784]
[L1836]     fc1_b_fp = model.fc1.bias.data.cpu()     # [128]
[L1837] 
[L1838]     # 固定输入量化参数
[L1839]     input_scale = 1.0 / 255.0
[L1840]     input_zero_point = -128
[L1841] 
[L1842]     # 权重量化（对称 int8）
[L1843]     fc1_w_q, fc1_w_scale, fc1_w_zp = quantize_weight_symmetric_int8(fc1_w_fp)
[L1844] 
[L1845]     # 偏置量化（int32）
[L1846]     fc1_b_q, fc1_b_scale = quantize_bias_int32(fc1_b_fp, input_scale, fc1_w_scale)
[L1847] 
[L1848]     # 导出权重：按 [out][in] 顺序
[L1849]     write_int8_hex_1d(fc1_w_q.reshape(-1), os.path.join(OUT_DIR, "fc1_weight_int8.hex"))
[L1850]     write_int32_hex_1d(fc1_b_q, os.path.join(OUT_DIR, "fc1_bias_int32.hex"))
[L1851] 
[L1852]     # 保存配置
[L1853]     config = {
[L1854]         "layer": "fc1_only",
[L1855]         "weight_shape": [128, 784],
[L1856]         "weight_layout": "row-major [out][in]",
[L1857]         "input_layout": "[in]",
[L1858]         "bias_layout": "[out]",
[L1859]         "input_scale": input_scale,
[L1860]         "input_zero_point": input_zero_point,
[L1861]         "fc1_weight_scale": fc1_w_scale,
[L1862]         "fc1_weight_zero_point": fc1_w_zp,
[L1863]         "fc1_bias_scale": fc1_b_scale,
[L1864]         "fc1_bias_dtype": "int32"
[L1865]     }
[L1866] 
[L1867]     with open(os.path.join(OUT_DIR, "quant_config.json"), "w", encoding="utf-8") as f:
[L1868]         json.dump(config, f, indent=2, ensure_ascii=False)
[L1869] 
[L1870]     # 测试集：为了导出原始像素，不能 Normalize
[L1871]     test_transform = transforms.Compose([
[L1872]         transforms.ToTensor()
[L1873]     ])
[L1874]     test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=test_transform)
[L1875] 
[L1876]     labels = []
[L1877] 
[L1878]     for i in range(num_samples):
[L1879]         img_fp, label = test_dataset[i]  # [1,28,28], range [0,1]
[L1880]         labels.append(int(label))
[L1881] 
[L1882]         img_flat = img_fp.view(-1).cpu()
[L1883]         x_q, _, _ = quantize_input_hw_style(img_flat)
[L1884] 
[L1885]         acc_int32, relu_int32 = fc1_integer_reference(
[L1886]             x_q, fc1_w_q, fc1_b_q, x_zp=input_zero_point, w_zp=fc1_w_zp
[L1887]         )
[L1888] 
[L1889]         write_int8_hex_1d(x_q, os.path.join(OUT_DIR, f"input_{i}.hex"))
[L1890]         write_int32_hex_1d(acc_int32, os.path.join(OUT_DIR, f"fc1_acc_golden_{i}.hex"))
[L1891]         write_int32_hex_1d(relu_int32, os.path.join(OUT_DIR, f"fc1_relu_golden_{i}.hex"))
[L1892] 
[L1893]         print(f"Exported sample {i}, label={label}")
[L1894] 
[L1895]     with open(os.path.join(OUT_DIR, "labels.txt"), "w") as f:
[L1896]         for lb in labels:
[L1897]             f.write(f"{lb}\n")
[L1898] 
[L1899]     print("Route A export done.")
[L1900] 
[L1901] 
[L1902] if __name__ == "__main__":
[L1903]     train_model(num_epochs=5, lr=1e-3)
[L1904]     export_route_a_artifacts(num_samples=10)
[L1905] ```
[L1906] 
[L1907] ---
[L1908] 
[L1909] # 四、路线 B：完整代码  
[L1910] ## 用途
[L1911] 这是你现在真正应该用的版本：
[L1912] 
[L1913] - 完整两层 MLP：`784 -> 128 -> 10`
[L1914] - `CrossEntropyLoss`
[L1915] - 静态量化
[L1916] - Python 写整数推理参考实现
[L1917] - 导出 FPGA golden
[L1918] - 最终瞄准“MNIST 上板分类演示”
[L1919] 
[L1920] ---
[L1921] 
[L1922] ## 路线 B 完整代码
[L1923] 
[L1924] ```python
[L1925] import os
[L1926] import json
[L1927] import math
[L1928] import numpy as np
[L1929] import torch
[L1930] import torch.nn as nn
[L1931] import torch.optim as optim
[L1932] from torchvision import datasets, transforms
[L1933] from torch.utils.data import DataLoader
[L1934] 
[L1935] # =========================================================
[L1936] # 1. 配置
[L1937] # =========================================================
[L1938] DATA_DIR = "./data"
[L1939] OUT_DIR = "./route_b_output"
[L1940] MODEL_PATH = os.path.join(OUT_DIR, "mlp_route_b.pth")
[L1941] 
[L1942] os.makedirs(OUT_DIR, exist_ok=True)
[L1943] 
[L1944] DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
[L1945] 
[L1946] INT8_QMIN = -128
[L1947] INT8_QMAX = 127
[L1948] INT32_QMIN = -2147483648
[L1949] INT32_QMAX = 2147483647
[L1950] 
[L1951] 
[L1952] # =========================================================
[L1953] # 2. 模型
[L1954] # =========================================================
[L1955] class MLPRouteB(nn.Module):
[L1956]     def __init__(self):
[L1957]         super().__init__()
[L1958]         self.fc1 = nn.Linear(784, 128, bias=True)
[L1959]         self.relu = nn.ReLU()
[L1960]         self.fc2 = nn.Linear(128, 10, bias=True)
[L1961] 
[L1962]     def forward(self, x):
[L1963]         x = x.view(-1, 784)
[L1964]         x = self.fc1(x)
[L1965]         x = self.relu(x)
[L1966]         x = self.fc2(x)
[L1967]         return x
[L1968] 
[L1969] 
[L1970] # =========================================================
[L1971] # 3. 数据
[L1972] #   训练时这里直接用 ToTensor()，不做 Normalize
[L1973] #   这样更贴近最终硬件部署链路
[L1974] # =========================================================
[L1975] def get_train_test_loaders(batch_size=128):
[L1976]     transform = transforms.Compose([
[L1977]         transforms.ToTensor()
[L1978]     ])
[L1979] 
[L1980]     train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
[L1981]     test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
[L1982] 
[L1983]     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
[L1984]     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
[L1985]     return train_loader, test_loader
[L1986] 
[L1987] 
[L1988] # =========================================================
[L1989] # 4. 训练 / 测试
[L1990] # =========================================================
[L1991] def evaluate_model(model, test_loader):
[L1992]     model.eval()
[L1993]     total = 0
[L1994]     correct = 0
[L1995] 
[L1996]     with torch.no_grad():
[L1997]         for data, target in test_loader:
[L1998]             data, target = data.to(DEVICE), target.to(DEVICE)
[L1999]             logits = model(data)
[L2000]             pred = torch.argmax(logits, dim=1)
[L2001]             correct += (pred == target).sum().item()
[L2002]             total += target.size(0)
[L2003] 
[L2004]     return correct / total
[L2005] 
[L2006] 
[L2007] def train_model(num_epochs=8, lr=1e-3):
[L2008]     train_loader, test_loader = get_train_test_loaders()
[L2009] 
[L2010]     model = MLPRouteB().to(DEVICE)
[L2011]     optimizer = optim.Adam(model.parameters(), lr=lr)
[L2012]     criterion = nn.CrossEntropyLoss()
[L2013] 
[L2014]     for epoch in range(num_epochs):
[L2015]         model.train()
[L2016]         running_loss = 0.0
[L2017] 
[L2018]         for data, target in train_loader:
[L2019]             data, target = data.to(DEVICE), target.to(DEVICE)
[L2020] 
[L2021]             optimizer.zero_grad()
[L2022]             logits = model(data)
[L2023]             loss = criterion(logits, target)
[L2024]             loss.backward()
[L2025]             optimizer.step()
[L2026] 
[L2027]             running_loss += loss.item() * data.size(0)
[L2028] 
[L2029]         avg_loss = running_loss / len(train_loader.dataset)
[L2030]         acc = evaluate_model(model, test_loader)
[L2031]         print(f"[Epoch {epoch+1}/{num_epochs}] loss={avg_loss:.6f}, test_acc={acc:.4f}")
[L2032] 
[L2033]     torch.save(model.state_dict(), MODEL_PATH)
[L2034]     print(f"Model saved to: {MODEL_PATH}")
[L2035]     return model
[L2036] 
[L2037] 
[L2038] # =========================================================
[L2039] # 5. 量化工具
[L2040] # =========================================================
[L2041] def clamp(v, low, high):
[L2042]     return max(low, min(high, v))
[L2043] 
[L2044] def quantize_affine_tensor(tensor_fp, scale, zero_point, qmin=-128, qmax=127, dtype=torch.int8):
[L2045]     q = torch.round(tensor_fp / scale + zero_point)
[L2046]     q = torch.clamp(q, qmin, qmax)
[L2047]     return q.to(dtype)
[L2048] 
[L2049] def calc_symmetric_scale(tensor_fp, num_bits=8):
[L2050]     qmax = (2 ** (num_bits - 1)) - 1
[L2051]     max_abs = max(abs(tensor_fp.min().item()), abs(tensor_fp.max().item()))
[L2052]     if max_abs == 0:
[L2053]         return 1.0
[L2054]     return max_abs / qmax
[L2055] 
[L2056] def calc_affine_params_from_range(min_val, max_val, num_bits=8, signed=True):
[L2057]     if signed:
[L2058]         qmin = -(2 ** (num_bits - 1))
[L2059]         qmax =  (2 ** (num_bits - 1)) - 1
[L2060]     else:
[L2061]         qmin = 0
[L2062]         qmax = (2 ** num_bits) - 1
[L2063] 
[L2064]     if max_val == min_val:
[L2065]         scale = 1.0
[L2066]         zero_point = 0
[L2067]         return scale, zero_point
[L2068] 
[L2069]     scale = (max_val - min_val) / float(qmax - qmin)
[L2070]     zero_point = round(qmin - min_val / scale)
[L2071]     zero_point = int(clamp(zero_point, qmin, qmax))
[L2072]     return float(scale), int(zero_point)
[L2073] 
[L2074] def quantize_weight_symmetric_int8(weight_fp):
[L2075]     scale = calc_symmetric_scale(weight_fp, num_bits=8)
[L2076]     zp = 0
[L2077]     q = quantize_affine_tensor(weight_fp, scale, zp, INT8_QMIN, INT8_QMAX, torch.int8)
[L2078]     return q, scale, zp
[L2079] 
[L2080] def quantize_bias_int32(bias_fp, input_scale, weight_scale):
[L2081]     bias_scale = input_scale * weight_scale
[L2082]     q = torch.round(bias_fp / bias_scale)
[L2083]     q = torch.clamp(q, INT32_QMIN, INT32_QMAX).to(torch.int32)
[L2084]     return q, bias_scale
[L2085] 
[L2086] def quantize_input_hw_style(img_fp_01):
[L2087]     """
[L2088]     固定输入量化：
[L2089]         x_u8 = round(x * 255)
[L2090]         x_q  = x_u8 - 128
[L2091]     对应：
[L2092]         scale = 1/255
[L2093]         zero_point = -128
[L2094]     """
[L2095]     scale = 1.0 / 255.0
[L2096]     zp = -128
[L2097] 
[L2098]     x_u8 = torch.round(img_fp_01 * 255.0)
[L2099]     x_u8 = torch.clamp(x_u8, 0, 255)
[L2100]     x_q = x_u8.to(torch.int16) - 128
[L2101]     x_q = torch.clamp(x_q, -128, 127).to(torch.int8)
[L2102]     return x_q, scale, zp
[L2103] 
[L2104] def choose_output_scale_from_activation(act_fp):
[L2105]     """
[L2106]     输出激活采用对称量化到 int8
[L2107]     """
[L2108]     scale = calc_symmetric_scale(act_fp, num_bits=8)
[L2109]     zp = 0
[L2110]     return scale, zp
[L2111] 
[L2112] def requantize_from_int32(acc_int32, real_multiplier, out_zero_point=0):
[L2113]     """
[L2114]     简单软件参考：
[L2115]         y_q = round(acc * real_multiplier) + out_zero_point
[L2116]     """
[L2117]     y = torch.round(acc_int32.to(torch.float32) * real_multiplier) + out_zero_point
[L2118]     y = torch.clamp(y, INT8_QMIN, INT8_QMAX)
[L2119]     return y.to(torch.int8)
[L2120] 
[L2121] def quantize_multiplier_to_int(real_multiplier):
[L2122]     """
[L2123]     给 FPGA 做定点乘法时可用的近似参数：
[L2124]       real_multiplier ≈ M / 2^shift
[L2125]     返回:
[L2126]       M(int), shift(int)
[L2127]     这里只提供给配置文件参考；软件 golden 仍直接用 float real_multiplier
[L2128]     """
[L2129]     if real_multiplier == 0:
[L2130]         return 0, 0
[L2131] 
[L2132]     shift = 0
[L2133]     m = real_multiplier
[L2134]     while m < 0.5:
[L2135]         m *= 2.0
[L2136]         shift += 1
[L2137]     M = int(round(m * (1 << 31)))  # Q31
[L2138]     return M, shift + 31
[L2139] 
[L2140] 
[L2141] # =========================================================
[L2142] # 6. 量化模型准备
[L2143] # =========================================================
[L2144] def build_quantized_model(model_fp):
[L2145]     model_fp.eval()
[L2146]     model_fp = model_fp.cpu()
[L2147] 
[L2148]     # 固定输入量化参数
[L2149]     input_scale = 1.0 / 255.0
[L2150]     input_zero_point = -128
[L2151] 
[L2152]     # fc1 量化
[L2153]     fc1_w_fp = model_fp.fc1.weight.data
[L2154]     fc1_b_fp = model_fp.fc1.bias.data
[L2155]     fc1_w_q, fc1_w_scale, fc1_w_zp = quantize_weight_symmetric_int8(fc1_w_fp)
[L2156]     fc1_b_q, fc1_b_scale = quantize_bias_int32(fc1_b_fp, input_scale, fc1_w_scale)
[L2157] 
[L2158]     # 用一批训练样本统计 fc1 输出尺度
[L2159]     calib_loader, _ = get_train_test_loaders(batch_size=256)
[L2160]     calib_data, _ = next(iter(calib_loader))
[L2161]     calib_data = calib_data.view(-1, 784)
[L2162]     with torch.no_grad():
[L2163]         fc1_act_fp = model_fp.relu(model_fp.fc1(calib_data))
[L2164]     fc1_out_scale, fc1_out_zp = choose_output_scale_from_activation(fc1_act_fp)
[L2165] 
[L2166]     fc1_real_multiplier = (input_scale * fc1_w_scale) / fc1_out_scale
[L2167]     fc1_M, fc1_shift = quantize_multiplier_to_int(fc1_real_multiplier)
[L2168] 
[L2169]     # fc2 量化
[L2170]     fc2_w_fp = model_fp.fc2.weight.data
[L2171]     fc2_b_fp = model_fp.fc2.bias.data
[L2172]     fc2_w_q, fc2_w_scale, fc2_w_zp = quantize_weight_symmetric_int8(fc2_w_fp)
[L2173]     fc2_b_q, fc2_b_scale = quantize_bias_int32(fc2_b_fp, fc1_out_scale, fc2_w_scale)
[L2174] 
[L2175]     # 用一批样本统计 fc2 输出尺度
[L2176]     with torch.no_grad():
[L2177]         logits_fp = model_fp(calib_data.view(-1, 1, 28, 28))
[L2178]     fc2_out_scale, fc2_out_zp = choose_output_scale_from_activation(logits_fp)
[L2179] 
[L2180]     fc2_real_multiplier = (fc1_out_scale * fc2_w_scale) / fc2_out_scale
[L2181]     fc2_M, fc2_shift = quantize_multiplier_to_int(fc2_real_multiplier)
[L2182] 
[L2183]     qparams = {
[L2184]         "input": {
[L2185]             "scale": input_scale,
[L2186]             "zero_point": input_zero_point,
[L2187]             "dtype": "int8"
[L2188]         },
[L2189]         "fc1": {
[L2190]             "weight_scale": fc1_w_scale,
[L2191]             "weight_zero_point": fc1_w_zp,
[L2192]             "bias_scale": fc1_b_scale,
[L2193]             "output_scale": fc1_out_scale,
[L2194]             "output_zero_point": fc1_out_zp,
[L2195]             "real_multiplier": fc1_real_multiplier,
[L2196]             "multiplier_q31": fc1_M,
[L2197]             "shift": fc1_shift
[L2198]         },
[L2199]         "fc2": {
[L2200]             "weight_scale": fc2_w_scale,
[L2201]             "weight_zero_point": fc2_w_zp,
[L2202]             "bias_scale": fc2_b_scale,
[L2203]             "output_scale": fc2_out_scale,
[L2204]             "output_zero_point": fc2_out_zp,
[L2205]             "real_multiplier": fc2_real_multiplier,
[L2206]             "multiplier_q31": fc2_M,
[L2207]             "shift": fc2_shift
[L2208]         }
[L2209]     }
[L2210] 
[L2211]     quantized = {
[L2212]         "fc1_w_q": fc1_w_q.to(torch.int8),
[L2213]         "fc1_b_q": fc1_b_q.to(torch.int32),
[L2214]         "fc2_w_q": fc2_w_q.to(torch.int8),
[L2215]         "fc2_b_q": fc2_b_q.to(torch.int32),
[L2216]         "qparams": qparams
[L2217]     }
[L2218]     return quantized
[L2219] 
[L2220] 
[L2221] # =========================================================
[L2222] # 7. 整数推理参考实现
[L2223] # =========================================================
[L2224] def linear_int8_int32(x_q, w_q, b_q, x_zp, w_zp):
[L2225]     """
[L2226]     x_q: [in_features] int8
[L2227]     w_q: [out_features, in_features] int8
[L2228]     b_q: [out_features] int32
[L2229]     """
[L2230]     x_int = x_q.to(torch.int32) - int(x_zp)
[L2231]     w_int = w_q.to(torch.int32) - int(w_zp)
[L2232]     acc = torch.matmul(w_int, x_int) + b_q
[L2233]     return acc.to(torch.int32)
[L2234] 
[L2235] def relu_int32(x):
[L2236]     return torch.clamp(x, min=0).to(torch.int32)
[L2237] 
[L2238] def integer_inference_reference(img_fp_01, quantized):
[L2239]     """
[L2240]     完整整数参考链路：
[L2241]       input -> fc1(int32 acc) -> relu -> requant to int8
[L2242]             -> fc2(int32 acc) -> requant to int8 logits
[L2243]     """
[L2244]     qparams = quantized["qparams"]
[L2245] 
[L2246]     # 1) 输入量化
[L2247]     x_q, _, _ = quantize_input_hw_style(img_fp_01)
[L2248] 
[L2249]     # 2) fc1 int32 累加
[L2250]     fc1_acc = linear_int8_int32(
[L2251]         x_q,
[L2252]         quantized["fc1_w_q"],
[L2253]         quantized["fc1_b_q"],
[L2254]         x_zp=qparams["input"]["zero_point"],
[L2255]         w_zp=qparams["fc1"]["weight_zero_point"]
[L2256]     )
[L2257] 
[L2258]     # 3) ReLU
[L2259]     fc1_relu = relu_int32(fc1_acc)
[L2260] 
[L2261]     # 4) fc1 输出 requant -> int8
[L2262]     fc1_out_q = requantize_from_int32(
[L2263]         fc1_relu,
[L2264]         qparams["fc1"]["real_multiplier"],
[L2265]         qparams["fc1"]["output_zero_point"]
[L2266]     )
[L2267] 
[L2268]     # 5) fc2 int32 累加
[L2269]     fc2_acc = linear_int8_int32(
[L2270]         fc1_out_q,
[L2271]         quantized["fc2_w_q"],
[L2272]         quantized["fc2_b_q"],
[L2273]         x_zp=qparams["fc1"]["output_zero_point"],
[L2274]         w_zp=qparams["fc2"]["weight_zero_point"]
[L2275]     )
[L2276] 
[L2277]     # 6) 输出 requant -> int8 logits
[L2278]     logits_q = requantize_from_int32(
[L2279]         fc2_acc,
[L2280]         qparams["fc2"]["real_multiplier"],
[L2281]         qparams["fc2"]["output_zero_point"]
[L2282]     )
[L2283] 
[L2284]     pred = int(torch.argmax(logits_q).item())
[L2285] 
[L2286]     return {
[L2287]         "input_q": x_q,
[L2288]         "fc1_acc_int32": fc1_acc,
[L2289]         "fc1_relu_int32": fc1_relu,
[L2290]         "fc1_out_q": fc1_out_q,
[L2291]         "fc2_acc_int32": fc2_acc,
[L2292]         "logits_q": logits_q,
[L2293]         "pred": pred
[L2294]     }
[L2295] 
[L2296] 
[L2297] # =========================================================
[L2298] # 8. HEX 导出
[L2299] # =========================================================
[L2300] def write_int8_hex_1d(tensor, path):
[L2301]     with open(path, "w") as f:
[L2302]         for v in tensor.flatten():
[L2303]             f.write(f"{int(v.item()) & 0xFF:02x}\n")
[L2304] 
[L2305] def write_int32_hex_1d(tensor, path):
[L2306]     with open(path, "w") as f:
[L2307]         for v in tensor.flatten():
[L2308]             f.write(f"{int(v.item()) & 0xFFFFFFFF:08x}\n")
[L2309] 
[L2310] def export_all_artifacts(model_fp, quantized, num_samples=20):
[L2311]     # 1) 导出权重 / bias
[L2312]     # 按 [out][in] 顺序导出
[L2313]     write_int8_hex_1d(quantized["fc1_w_q"].reshape(-1), os.path.join(OUT_DIR, "fc1_weight_int8.hex"))
[L2314]     write_int32_hex_1d(quantized["fc1_b_q"], os.path.join(OUT_DIR, "fc1_bias_int32.hex"))
[L2315] 
[L2316]     write_int8_hex_1d(quantized["fc2_w_q"].reshape(-1), os.path.join(OUT_DIR, "fc2_weight_int8.hex"))
[L2317]     write_int32_hex_1d(quantized["fc2_b_q"], os.path.join(OUT_DIR, "fc2_bias_int32.hex"))
[L2318] 
[L2319]     # 2) 保存量化配置
[L2320]     config = {
[L2321]         "network": "784 -> 128 -> 10",
[L2322]         "weight_layout": {
[L2323]             "fc1": "row-major [out][in] = [128][784]",
[L2324]             "fc2": "row-major [out][in] = [10][128]"
[L2325]         },
[L2326]         "input_layout": "[784]",
[L2327]         "hidden_layout": "[128]",
[L2328]         "output_layout": "[10]",
[L2329]         "qparams": quantized["qparams"]
[L2330]     }
[L2331] 
[L2332]     with open(os.path.join(OUT_DIR, "quant_config.json"), "w", encoding="utf-8") as f:
[L2333]         json.dump(config, f, indent=2, ensure_ascii=False)
[L2334] 
[L2335]     # 3) 导出样本和 golden
[L2336]     test_transform = transforms.Compose([
[L2337]         transforms.ToTensor()
[L2338]     ])
[L2339]     test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=test_transform)
[L2340] 
[L2341]     labels = []
[L2342]     preds = []
[L2343] 
[L2344]     for i in range(num_samples):
[L2345]         img_fp, label = test_dataset[i]
[L2346]         img_flat = img_fp.view(-1).cpu()
[L2347] 
[L2348]         result = integer_inference_reference(img_flat, quantized)
[L2349] 
[L2350]         labels.append(int(label))
[L2351]         preds.append(int(result["pred"]))
[L2352] 
[L2353]         write_int8_hex_1d(result["input_q"], os.path.join(OUT_DIR, f"input_{i}.hex"))
[L2354] 
[L2355]         write_int32_hex_1d(result["fc1_acc_int32"], os.path.join(OUT_DIR, f"fc1_acc_{i}.hex"))
[L2356]         write_int32_hex_1d(result["fc1_relu_int32"], os.path.join(OUT_DIR, f"fc1_relu_{i}.hex"))
[L2357]         write_int8_hex_1d(result["fc1_out_q"], os.path.join(OUT_DIR, f"fc1_out_{i}.hex"))
[L2358] 
[L2359]         write_int32_hex_1d(result["fc2_acc_int32"], os.path.join(OUT_DIR, f"fc2_acc_{i}.hex"))
[L2360]         write_int8_hex_1d(result["logits_q"], os.path.join(OUT_DIR, f"logits_{i}.hex"))
[L2361] 
[L2362]         with open(os.path.join(OUT_DIR, f"pred_{i}.txt"), "w") as f:
[L2363]             f.write(str(result["pred"]))
[L2364] 
[L2365]         print(f"sample={i}, label={label}, pred={result['pred']}")
[L2366] 
[L2367]     with open(os.path.join(OUT_DIR, "labels.txt"), "w") as f:
[L2368]         for lb in labels:
[L2369]             f.write(f"{lb}\n")
[L2370] 
[L2371]     with open(os.path.join(OUT_DIR, "preds.txt"), "w") as f:
[L2372]         for pd in preds:
[L2373]             f.write(f"{pd}\n")
[L2374] 
[L2375]     print("All artifacts exported.")
[L2376] 
[L2377] 
[L2378] # =========================================================
[L2379] # 9. 验证量化后整数推理精度
[L2380] # =========================================================
[L2381] def evaluate_integer_pipeline(model_fp, quantized, num_samples=1000):
[L2382]     test_transform = transforms.Compose([
[L2383]         transforms.ToTensor()
[L2384]     ])
[L2385]     test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=test_transform)
[L2386] 
[L2387]     correct = 0
[L2388]     total = min(num_samples, len(test_dataset))
[L2389] 
[L2390]     for i in range(total):
[L2391]         img_fp, label = test_dataset[i]
[L2392]         img_flat = img_fp.view(-1).cpu()
[L2393]         result = integer_inference_reference(img_flat, quantized)
[L2394]         if result["pred"] == int(label):
[L2395]             correct += 1
[L2396] 
[L2397]     acc = correct / total
[L2398]     print(f"Integer pipeline accuracy over {total} samples: {acc:.4f}")
[L2399]     return acc
[L2400] 
[L2401] 
[L2402] # =========================================================
[L2403] # 10. 主流程
[L2404] # =========================================================
[L2405] if __name__ == "__main__":
[L2406]     # 1) 训练浮点模型
[L2407]     model_fp = train_model(num_epochs=8, lr=1e-3)
[L2408] 
[L2409]     # 2) 加载模型（防止单独运行导出时也能用）
[L2410]     model_fp = MLPRouteB().to(DEVICE)
[L2411]     model_fp.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
[L2412]     model_fp.eval()
[L2413] 
[L2414]     # 3) 构建量化模型
[L2415]     quantized = build_quantized_model(model_fp)
[L2416] 
[L2417]     # 4) 评估整数链路
[L2418]     evaluate_integer_pipeline(model_fp, quantized, num_samples=1000)
[L2419] 
[L2420]     # 5) 导出 FPGA 所需文件
[L2421]     export_all_artifacts(model_fp, quantized, num_samples=20)
[L2422] ```
[L2423] 
[L2424] ---
[L2425] 
[L2426] # 五、关于问题 2：逐项回答
[L2427] 
[L2428] ---
[L2429] 
[L2430] ## 2.1 `zero_point` 不是整数，怎么改
[L2431] 
[L2432] 你原来的问题在这里：
[L2433] 
[L2434] ```python
[L2435] zero_point = qmin - min_val / scale
[L2436] ```
[L2437] 
[L2438] 这里会得到浮点数。应改成：
[L2439] 
[L2440] ```python
[L2441] zero_point = round(qmin - min_val / scale)
[L2442] zero_point = int(max(qmin, min(qmax, zero_point)))
[L2443] ```
[L2444] 
[L2445] 通用函数我已经在路线 B 里给了：
[L2446] 
[L2447] ```python
[L2448] def calc_affine_params_from_range(min_val, max_val, num_bits=8, signed=True):
[L2449]     if signed:
[L2450]         qmin = -(2 ** (num_bits - 1))
[L2451]         qmax =  (2 ** (num_bits - 1)) - 1
[L2452]     else:
[L2453]         qmin = 0
[L2454]         qmax = (2 ** num_bits) - 1
[L2455] 
[L2456]     if max_val == min_val:
[L2457]         scale = 1.0
[L2458]         zero_point = 0
[L2459]         return scale, zero_point
[L2460] 
[L2461]     scale = (max_val - min_val) / float(qmax - qmin)
[L2462]     zero_point = round(qmin - min_val / scale)
[L2463]     zero_point = int(clamp(zero_point, qmin, qmax))
[L2464]     return float(scale), int(zero_point)
[L2465] ```
[L2466] 
[L2467] ---
[L2468] 
[L2469] ## 2.2 每个张量动态量化不适合 FPGA，合理解决方案
[L2470] 
[L2471] ### 你现在的问题
[L2472] 你现在是：
[L2473] - 每张输入图像单独量化
[L2474] - 每次输出也单独量化
[L2475] 
[L2476] 这会导致 FPGA 需要不断切换 scale / zero_point，不利于验证和部署。
[L2477] 
[L2478] ### 合理方案
[L2479] 改为**静态量化**：
[L2480] 
[L2481] #### 输入
[L2482] 固定：
[L2483] - `input_scale = 1/255`
[L2484] - `input_zero_point = -128`
[L2485] 
[L2486] #### 权重
[L2487] 每层固定：
[L2488] - `weight_scale` 固定
[L2489] - `weight_zero_point = 0`
[L2490] 
[L2491] #### 输出
[L2492] 每层固定：
[L2493] - `output_scale` 固定
[L2494] - `output_zero_point` 固定
[L2495] 
[L2496] 这样每个样本都共用同一套参数，硬件最容易做。
[L2497] 
[L2498] ---
[L2499] 
[L2500] ## 2.3 偏置不能用 int8，怎么解决
[L2501] 
[L2502] ### 正确原则
[L2503] bias 必须与 int32 累加器同量纲：
[L2504] 
[L2505] \[
[L2506] b_q = round(\frac{b}{s_x \cdot s_w})
[L2507] \]
[L2508] 
[L2509] ### 代码
[L2510] 我已经写成：
[L2511] 
[L2512] ```python
[L2513] def quantize_bias_int32(bias_fp, input_scale, weight_scale):
[L2514]     bias_scale = input_scale * weight_scale
[L2515]     q = torch.round(bias_fp / bias_scale)
[L2516]     q = torch.clamp(q, INT32_QMIN, INT32_QMAX).to(torch.int32)
[L2517]     return q, bias_scale
[L2518] ```
[L2519] 
[L2520] ### 为什么要这样
[L2521] 因为硬件里先做：
[L2522] 
[L2523] \[
[L2524] acc = \sum (x_q-z_x)(w_q-z_w)
[L2525] \]
[L2526] 
[L2527] 这个 `acc` 是 int32 量纲。  
[L2528] 如果 bias 还是 int8，直接相加就是错的。
[L2529] 
[L2530] ---
[L2531] 
[L2532] # 六、关于问题 3：整数推理链路 golden，怎么设计
[L2533] 
[L2534] 你要求“整数推理链路生成 golden”，现在路线 B 已经完整实现了。
[L2535] 
[L2536] ---
[L2537] 
[L2538] ## 设计链路
[L2539] 
[L2540] ### 第一层
[L2541] \[
[L2542] acc_1 = W_1 \cdot (x_q-z_x) + b_{1q}
[L2543] \]
[L2544] 
[L2545] \[
[L2546] relu_1 = max(acc_1, 0)
[L2547] \]
[L2548] 
[L2549] \[
[L2550] h_q = requant(relu_1)
[L2551] \]
[L2552] 
[L2553] ### 第二层
[L2554] \[
[L2555] acc_2 = W_2 \cdot (h_q-z_{h}) + b_{2q}
[L2556] \]
[L2557] 
[L2558] \[
[L2559] logits_q = requant(acc_2)
[L2560] \]
[L2561] 
[L2562] ### 分类
[L2563] \[
[L2564] pred = argmax(logits_q)
[L2565] \]
[L2566] 
[L2567] ---
[L2568] 
[L2569] ## 代码入口
[L2570] 整数参考实现就在这里：
[L2571] 
[L2572] ```python
[L2573] def integer_inference_reference(img_fp_01, quantized):
[L2574]     ...
[L2575] ```
[L2576] 
[L2577] 它导出了：
[L2578] 
[L2579] - `input_q`
[L2580] - `fc1_acc_int32`
[L2581] - `fc1_relu_int32`
[L2582] - `fc1_out_q`
[L2583] - `fc2_acc_int32`
[L2584] - `logits_q`
[L2585] - `pred`
[L2586] 
[L2587] 这些就是你硬件联调最有价值的 golden。
[L2588] 
[L2589] ---
[L2590] 
[L2591] # 七、关于问题 4：统一输入量化策略
[L2592] 
[L2593] 你要求“贴近硬件即可”，我建议你就用下面这个，不要再 Normalize。
[L2594] 
[L2595] ---
[L2596] 
[L2597] ## 推荐方案
[L2598] MNIST 原始像素 `[0,1]`
[L2599] 
[L2600] 统一做：
[L2601] 
[L2602] ```python
[L2603] x_u8 = round(x * 255)
[L2604] x_q = x_u8 - 128
[L2605] ```
[L2606] 
[L2607] 对应：
[L2608] 
[L2609] - `input_scale = 1/255`
[L2610] - `input_zero_point = -128`
[L2611] 
[L2612] ---
[L2613] 
[L2614] ## 优点
[L2615] ### 软件上
[L2616] - 简单
[L2617] - 稳定
[L2618] - 不依赖每张图的分布
[L2619] 
[L2620] ### 硬件上
[L2621] - BRAM 存储直观
[L2622] - 每个像素 8 bit
[L2623] - 乘加前减去 `-128` 很明确
[L2624] - 适合 testbench 和最终上板
[L2625] 
[L2626] ---
[L2627] 
[L2628] # 八、关于问题 5：你当前输出顺序如何设计硬件读取顺序
[L2629] 
[L2630] 你现在权重导出顺序是：
[L2631] 
[L2632] ```python
[L2633] for row in q_weight:
[L2634]     for val in row:
[L2635]         f.write(...)
[L2636] ```
[L2637] 
[L2638] 而 `nn.Linear(in_features, out_features)` 在 PyTorch 里权重 shape 是：
[L2639] 
[L2640] \[
[L2641] [out\_features, in\_features]
[L2642] \]
[L2643] 
[L2644] 所以你现在导出的顺序是：
[L2645] 
[L2646] \[
[L2647] w[0,0], w[0,1], ..., w[0,783], w[1,0], ..., w[127,783]
[L2648] \]
[L2649] 
[L2650] 也就是：
[L2651] 
[L2652] ## 当前输出顺序
[L2653] **按输出神经元逐行展开，row-major `[out][in]`**
[L2654] 
[L2655] ---
[L2656] 
[L2657] ## 对应的硬件读取顺序应该怎么设计
[L2658] 
[L2659] 如果你的第一层是：
[L2660] 
[L2661] \[
[L2662] y_j = \sum_{i=0}^{783} x_i \cdot w_{j,i} + b_j
[L2663] \]
[L2664] 
[L2665] 那么最自然的硬件读取顺序就是：
[L2666] 
[L2667] ### 方案：按神经元顺序读
[L2668] 对于每个输出神经元 `j`：
[L2669] 
[L2670] 1. 读取该神经元的 784 个权重
[L2671] 2. 与输入 784 维逐项相乘累加
[L2672] 3. 加偏置 `b_j`
[L2673] 4. 输出 `y_j`
[L2674] 
[L2675] ### 地址映射
[L2676] 如果 `fc1_weight_int8.hex` 按 `[out][in]` 展开，则：
[L2677] 
[L2678] \[
[L2679] addr = j \times 784 + i
[L2680] \]
[L2681] 
[L2682] 即：
[L2683] - `j`：输出神经元编号
[L2684] - `i`：输入特征编号
[L2685] 
[L2686] ### RTL 里可写成
[L2687] - 外层循环：`j = 0..127`
[L2688] - 内层循环：`i = 0..783`
[L2689] 
[L2690] 这和你当前导出顺序是完全匹配的。
[L2691] 
[L2692] ---
[L2693] 
[L2694] ## 如果硬件是并行阵列怎么办
[L2695] 如果你做的是 CIM 阵列或 PE 并行，也仍然建议逻辑上保持这个顺序，只是物理存储时可分块：
[L2696] 
[L2697] 例如分成 16 路并行：
[L2698] - 每次读同一个 `j` 的 16 个连续权重
[L2699] - 输入也同步读 16 个元素
[L2700] - 做 16 路乘加
[L2701] - 局部累加后再归并
[L2702] 
[L2703] 本质上还是从 `[out][in]` 顺序切块读取。
[L2704] 
[L2705] ---
[L2706] 
[L2707] ## 第二层同理
[L2708] 第二层 `fc2` shape 是 `[10,128]`
[L2709] 
[L2710] 地址映射：
[L2711] 
[L2712] \[
[L2713] addr = j \times 128 + i
[L2714] \]
[L2715] 
[L2716] 其中：
[L2717] - `j = 0..9`
[L2718] - `i = 0..127`
[L2719] 
[L2720] ---
[L2721] 
[L2722] # 九、问题 6 和 7，我已经一起解决了
[L2723] 
[L2724] ---
[L2725] 
[L2726] ## 问题 6：`quant_params.npz` 没真正参与闭环
[L2727] 现在我不建议继续用 `npz` 做核心配置。  
[L2728] 我已经改成了：
[L2729] 
[L2730] - `quant_config.json`
[L2731] 
[L2732] 里面包含：
[L2733] - 输入 scale / zp
[L2734] - 每层 weight_scale / weight_zp
[L2735] - bias_scale
[L2736] - output_scale / output_zp
[L2737] - real_multiplier
[L2738] - multiplier_q31
[L2739] - shift
[L2740] 
[L2741] 这对硬件更友好，也更好查。
[L2742] 
[L2743] ---
[L2744] 
[L2745] ## 问题 7：其他工程问题
[L2746] 我已经一起处理：
[L2747] 
[L2748] ### 1）`torch.load()` 增加 `map_location`
[L2749] 已改：
[L2750] 
[L2751] ```python
[L2752] torch.load(MODEL_PATH, map_location=DEVICE)
[L2753] ```
[L2754] 
[L2755] ### 2）训练时增加准确率
[L2756] 每个 epoch 后打印 test accuracy
[L2757] 
[L2758] ### 3）保存 label
[L2759] 已导出：
[L2760] - `labels.txt`
[L2761] 
[L2762] ### 4）保存中间层 golden
[L2763] 已导出：
[L2764] - `fc1_acc_x.hex`
[L2765] - `fc1_relu_x.hex`
[L2766] - `fc1_out_x.hex`
[L2767] - `fc2_acc_x.hex`
[L2768] - `logits_x.hex`
[L2769] 
[L2770] 这样 RTL 调试时你能定位是哪一级错了。
[L2771] 
[L2772] ---
[L2773] 
[L2774] # 十、你最终应该选哪套
[L2775] 
[L2776] ## 如果你现在要交阶段成果、先验证阵列
[L2777] 用路线 A。  
[L2778] 因为它最容易把硬件链路先跑通。
[L2779] 
[L2780] ## 如果你现在就想直奔“MNIST 上板分类演示”
[L2781] 用路线 B。  
[L2782] 这就是你刚刚指定的目标，我也已经按这个目标把代码写全了。
[L2783] 
[L2784] ---
[L2785] 
[L2786] # 十一、我建议你下一步的落地顺序
[L2787] 
[L2788] 你现在最稳的执行顺序是：
[L2789] 
[L2790] ### 第一步
[L2791] 先跑路线 B 的 Python 脚本，得到：
[L2792] - 浮点模型精度
[L2793] - 整数量化链路精度
[L2794] - 全套导出文件
[L2795] 
[L2796] ### 第二步
[L2797] 先在 FPGA 上只实现第一层并对齐：
[L2798] - `input`
[L2799] - `fc1_weight`
[L2800] - `fc1_bias`
[L2801] - `fc1_acc`
[L2802] - `fc1_relu`
[L2803] - `fc1_out`
[L2804] 
[L2805] ### 第三步
[L2806] 再补第二层：
[L2807] - `fc2_weight`
[L2808] - `fc2_bias`
[L2809] - `fc2_acc`
[L2810] - `logits`
[L2811] - `argmax`
[L2812] 
[L2813] ---
[L2814] 
[L2815] 如果你愿意，我下一条可以继续直接给你一份**更适合 notebook 使用的版本**，也就是把路线 B 拆成多个单元格，按你现在 `ipynb` 的风格排好。
[L2816] 
[L2817] 用户:
[L2818] 现在已经把导出测试向量的部分写好了，请帮我分析导出的文件的格式，以及如何使用
[L2819] AI助手:
[L2820] 
[L2821] 
[L2822] AI助手:
[L2823] 我先把你当前导出的目录和文件内容抽样检查一下，确认每类文件到底是几位十六进制、按什么维度展开、对应哪一级数据通路。然后我会按“文件格式 → RTL/Testbench 怎么读 → 怎么对拍”的顺序给你整理清楚。
[L2824] 
[L2825] AI助手:
[L2826] 我根据你现在 notebook 里的导出代码，已经能把**导出文件格式**和**使用方法**说清楚了。你当前实际导出的文件主要有这几类：
[L2827] 
[L2828] - `weights.hex`
[L2829] - `bias.hex`
[L2830] - `quant_params.npz`
[L2831] - `input_i.hex`
[L2832] - `golden_i.hex`
[L2833] 
[L2834] 下面我按“是什么、怎么组织、硬件怎么读、联调怎么用”来讲。
[L2835] 
[L2836] ---
[L2837] 
[L2838] # 1. 你现在导出的文件格式
[L2839] 
[L2840] ## 1.1 `weights.hex`
[L2841] 
[L2842] 这是第一层全连接层 `fc1.weight` 的量化权重文件。
[L2843] 
[L2844] ### 来源
[L2845] 代码里：
[L2846] 
[L2847] ```python
[L2848] weight = model.fc1.weight.data  # [128, 784]
[L2849] ...
[L2850] with open(f'{output_dir}/weights.hex', 'w') as f:
[L2851]     for row in q_weight:
[L2852]         for val in row:
[L2853]             f.write(f'{val.item() & 0xFF:02x}\n')
[L2854] ```
[L2855] 
[L2856] ### 文件格式
[L2857] - 文本文件
[L2858] - **每行一个 8 bit 数值**
[L2859] - 用 **2 位十六进制** 表示
[L2860] - 负数采用 **8 位二补码的低 8 位**
[L2861] 
[L2862] 例如：
[L2863] - `00` 表示 0
[L2864] - `01` 表示 1
[L2865] - `7f` 表示 127
[L2866] - `ff` 表示 -1
[L2867] - `80` 表示 -128
[L2868] 
[L2869] ### 总行数
[L2870] 因为权重 shape 是 `[128, 784]`，所以总共有：
[L2871] 
[L2872] \[
[L2873] 128 \times 784 = 100352
[L2874] \]
[L2875] 
[L2876] 也就是：
[L2877] 
[L2878] - `weights.hex` 一共 **100352 行**
[L2879] 
[L2880] ### 展开顺序
[L2881] 当前是：
[L2882] 
[L2883] ```python
[L2884] for row in q_weight:
[L2885]     for val in row:
[L2886] ```
[L2887] 
[L2888] 所以顺序是：
[L2889] 
[L2890] \[
[L2891] w[0,0], w[0,1], ..., w[0,783], w[1,0], ..., w[127,783]
[L2892] \]
[L2893] 
[L2894] 也就是：
[L2895] 
[L2896] - **按输出神经元 row-major 展开**
[L2897] - 格式为 **`[out][in]`**
[L2898] 
[L2899] 这点非常重要，RTL 必须按这个顺序取。
[L2900] 
[L2901] ---
[L2902] 
[L2903] ## 1.2 `bias.hex`
[L2904] 
[L2905] 这是第一层偏置文件。
[L2906] 
[L2907] ### 来源
[L2908] ```python
[L2909] bias = model.fc1.bias.data      # [128]
[L2910] ...
[L2911] with open(f'{output_dir}/bias.hex', 'w') as f:
[L2912]     for val in q_bias:
[L2913]         f.write(f'{val.item() & 0xFF:02x}\n')
[L2914] ```
[L2915] 
[L2916] ### 文件格式
[L2917] - 文本文件
[L2918] - 每行一个量化后的偏置
[L2919] - **2 位十六进制**
[L2920] - 当前代码里它被导成了 **int8**
[L2921] 
[L2922] ### 总行数
[L2923] - 偏置长度是 128
[L2924] - 所以 `bias.hex` 一共 **128 行**
[L2925] 
[L2926] ### 当前问题
[L2927] 这个文件虽然格式上没问题，但**工程上有问题**：
[L2928] 
[L2929] - 你现在是把 bias 当 int8 存
[L2930] - 但硬件 MAC 累加结果通常是 int32
[L2931] - 所以这个 `bias.hex` 现在**不适合直接用于真实整数推理链路**
[L2932] 
[L2933] 也就是说：
[L2934] - 这个文件可以先拿来做“形式演示”
[L2935] - 但不适合做你后面要的最终整数 golden 对拍
[L2936] 
[L2937] ---
[L2938] 
[L2939] ## 1.3 `quant_params.npz`
[L2940] 
[L2941] 这是量化参数文件。
[L2942] 
[L2943] ### 来源
[L2944] ```python
[L2945] np.savez(f'{output_dir}/quant_params.npz',
[L2946]          w_scale=w_scale, w_zero=w_zero,
[L2947]          b_scale=b_scale, b_zero=b_zero)
[L2948] ```
[L2949] 
[L2950] ### 里面有什么
[L2951] 当前保存了：
[L2952] 
[L2953] - `w_scale`
[L2954] - `w_zero`
[L2955] - `b_scale`
[L2956] - `b_zero`
[L2957] 
[L2958] ### 用途
[L2959] 理论上它用于：
[L2960] - 软件端反量化
[L2961] - 硬件端知道量化参数
[L2962] 
[L2963] ### 当前实际情况
[L2964] 你现在的 `generate_test_vectors()` 虽然加载了它：
[L2965] 
[L2966] ```python
[L2967] quant_params = np.load('quantized_data/quant_params.npz')
[L2968] ```
[L2969] 
[L2970] 但后面**几乎没真正用上**。  
[L2971] 所以它现在更像一个“记录文件”，还不是硬件链路真正依赖的配置文件。
[L2972] 
[L2973] ---
[L2974] 
[L2975] ## 1.4 `input_i.hex`
[L2976] 
[L2977] 这是第 `i` 个测试样本的输入文件。
[L2978] 
[L2979] ### 来源
[L2980] ```python
[L2981] img, label = test_dataset[i]
[L2982] img_flat = img.view(-1, 784)
[L2983] q_img, img_scale, img_zero = quantize_tensor(img_flat)
[L2984] 
[L2985] with open(f'quantized_data/input_{i}.hex', 'w') as f:
[L2986]     for val in q_img.flatten():
[L2987]         f.write(f'{val.item() & 0xFF:02x}\n')
[L2988] ```
[L2989] 
[L2990] ### 文件格式
[L2991] - 文本文件
[L2992] - 每行一个输入元素
[L2993] - **2 位十六进制**
[L2994] - 量化后为 int8
[L2995] 
[L2996] ### 总行数
[L2997] MNIST 一张图是 `28×28=784`，所以：
[L2998] 
[L2999] - 每个 `input_i.hex` 一共 **784 行**
[L3000] 
[L3001] ### 排列顺序
[L3002] 因为：
[L3003] 
[L3004] ```python
[L3005] img_flat = img.view(-1, 784)
[L3006] for val in q_img.flatten():
[L3007] ```
[L3008] 
[L3009] 所以顺序就是图像展平后的顺序，通常是：
[L3010] 
[L3011] - 第 0 行：像素 `(0,0)`
[L3012] - 第 1 行：像素 `(0,1)`
[L3013] - ...
[L3014] - 第 27 行：像素 `(0,27)`
[L3015] - 第 28 行：像素 `(1,0)`
[L3016] - ...
[L3017] 
[L3018] 也就是：
[L3019] 
[L3020] - **按行优先 row-major 展开**
[L3021] - 输入向量索引 `i = row * 28 + col`
[L3022] 
[L3023] RTL 里要按这个顺序喂入。
[L3024] 
[L3025] ---
[L3026] 
[L3027] ## 1.5 `golden_i.hex`
[L3028] 
[L3029] 这是第 `i` 个测试样本的软件 golden 输出。
[L3030] 
[L3031] ### 来源
[L3032] ```python
[L3033] with torch.no_grad():
[L3034]     output = model(img_flat)
[L3035] 
[L3036] q_output, _, _ = quantize_tensor(output)
[L3037] with open(f'quantized_data/golden_{i}.hex', 'w') as f:
[L3038]     for val in q_output.flatten():
[L3039]         f.write(f'{val.item() & 0xFF:02x}\n')
[L3040] ```
[L3041] 
[L3042] ### 文件格式
[L3043] - 文本文件
[L3044] - 每行一个输出元素
[L3045] - **2 位十六进制**
[L3046] - 量化后为 int8
[L3047] 
[L3048] ### 总行数
[L3049] 你现在模型是：
[L3050] 
[L3051] ```python
[L3052] self.fc1 = nn.Linear(784,128)
[L3053] self.relu = nn.ReLU()
[L3054] ```
[L3055] 
[L3056] 所以输出是 128 维。
[L3057] 
[L3058] 也就是：
[L3059] - 每个 `golden_i.hex` 一共 **128 行**
[L3060] 
[L3061] ### 当前含义
[L3062] 这个 `golden_i.hex` 表示：
[L3063] 
[L3064] - 先用浮点模型跑出 `fc1 + relu` 的浮点输出
[L3065] - 再把这个浮点输出单独量化成 int8
[L3066] 
[L3067] 所以它的意义是：
[L3068] 
[L3069] - **“浮点模型输出的量化结果”**
[L3070] - 不是严格意义上的“整数硬件链路 golden”
[L3071] 
[L3072] 这点你后面做最终方案时必须记住。
[L3073] 
[L3074] ---
[L3075] 
[L3076] # 2. 这些文件如何使用
[L3077] 
[L3078] ---
[L3079] 
[L3080] ## 2.1 最直接的使用方式：Verilog `$readmemh`
[L3081] 
[L3082] 因为这些 `.hex` 文件是：
[L3083] - 文本
[L3084] - 每行一个十六进制数
[L3085] 
[L3086] 所以最适合直接在 Verilog 里用：
[L3087] 
[L3088] ```verilog
[L3089] reg [7:0] weight_mem [0:100351];
[L3090] reg [7:0] bias_mem   [0:127];
[L3091] reg [7:0] input_mem  [0:783];
[L3092] reg [7:0] golden_mem [0:127];
[L3093] 
[L3094] initial begin
[L3095]     $readmemh("weights.hex", weight_mem);
[L3096]     $readmemh("bias.hex", bias_mem);
[L3097]     $readmemh("input_0.hex", input_mem);
[L3098]     $readmemh("golden_0.hex", golden_mem);
[L3099] end
[L3100] ```
[L3101] 
[L3102] ---
[L3103] 
[L3104] ## 2.2 在 RTL 里如何解释这些 8 bit 数据
[L3105] 
[L3106] ### 注意
[L3107] `$readmemh` 读进来的是 `reg [7:0]`，本身只是无符号位模式。  
[L3108] 如果你想让它参与有符号运算，需要做符号解释。
[L3109] 
[L3110] 例如：
[L3111] 
[L3112] ```verilog
[L3113] wire signed [7:0] w = weight_mem[addr];
[L3114] wire signed [7:0] x = input_mem[idx];
[L3115] wire signed [7:0] b = bias_mem[out_idx];
[L3116] ```
[L3117] 
[L3118] 或者：
[L3119] 
[L3120] ```verilog
[L3121] wire signed [7:0] w = $signed(weight_mem[addr]);
[L3122] ```
[L3123] 
[L3124] 这样：
[L3125] - `8'hff` 才会被解释成 `-1`
[L3126] - `8'h80` 才会被解释成 `-128`
[L3127] 
[L3128] ---
[L3129] 
[L3130] ## 2.3 第一层硬件如何读取 `weights.hex`
[L3131] 
[L3132] 你当前权重顺序是：
[L3133] 
[L3134] \[
[L3135] w[j][i]
[L3136] \]
[L3137] 
[L3138] 其中：
[L3139] - `j = 0..127`，输出神经元
[L3140] - `i = 0..783`，输入索引
[L3141] 
[L3142] 所以硬件最自然的读取方式是：
[L3143] 
[L3144] ### 地址公式
[L3145] \[
[L3146] addr = j \times 784 + i
[L3147] \]
[L3148] 
[L3149] ### 计算流程
[L3150] 对于每个输出神经元 `j`：
[L3151] 
[L3152] 1. 累加器清零
[L3153] 2. 从 `i=0` 到 `783`
[L3154] 3. 读取：
[L3155]    - `x = input_mem[i]`
[L3156]    - `w = weight_mem[j*784+i]`
[L3157] 4. 做乘加：
[L3158]    \[
[L3159]    acc_j += x \cdot w
[L3160]    \]
[L3161] 5. 循环结束后加偏置：
[L3162]    \[
[L3163]    acc_j += bias[j]
[L3164]    \]
[L3165] 6. 过 ReLU
[L3166] 7. 输出第 `j` 个结果
[L3167] 
[L3168] ### 伪代码
[L3169] ```verilog
[L3170] for (j = 0; j < 128; j = j + 1) begin
[L3171]     acc = 0;
[L3172]     for (i = 0; i < 784; i = i + 1) begin
[L3173]         acc = acc + $signed(input_mem[i]) * $signed(weight_mem[j*784+i]);
[L3174]     end
[L3175]     acc = acc + $signed(bias_mem[j]);
[L3176]     if (acc < 0) acc = 0;
[L3177]     out[j] = acc; // 或量化后输出
[L3178] end
[L3179] ```
[L3180] 
[L3181] ---
[L3182] 
[L3183] ## 2.4 `input_i.hex` 如何使用
[L3184] 
[L3185] ### 在 testbench 中
[L3186] 每次测试一个样本：
[L3187] 
[L3188] ```verilog
[L3189] $readmemh("input_0.hex", input_mem);
[L3190] ```
[L3191] 
[L3192] 然后把 `input_mem[0:783]` 依次送给你的计算模块。
[L3193] 
[L3194] ### 两种常见喂法
[L3195] #### 方式 1：一次性 preload 到 BRAM
[L3196] 如果你的设计是：
[L3197] - 输入先装入片上 RAM
[L3198] - 再启动计算
[L3199] 
[L3200] 那就：
[L3201] 1. 先把 `input_i.hex` 用 `$readmemh` 读入 testbench memory
[L3202] 2. 再驱动写接口，把 784 个数据写入 DUT 内部 input RAM
[L3203] 3. 拉高 `start`
[L3204] 
[L3205] #### 方式 2：流式输入
[L3206] 如果你的 DUT 是流式输入：
[L3207] - 每拍输入一个 `x_i`
[L3208] 
[L3209] 那就：
[L3210] 1. 从 `input_mem[0]` 到 `input_mem[783]` 依次送
[L3211] 2. 每个时钟给一个有效数据
[L3212] 3. 送满 784 个后等待结果
[L3213] 
[L3214] ---
[L3215] 
[L3216] ## 2.5 `golden_i.hex` 如何使用
[L3217] 
[L3218] 这是当前 testbench 最直接的对拍基准。
[L3219] 
[L3220] ### 用法
[L3221] 在 testbench 中：
[L3222] 
[L3223] ```verilog
[L3224] $readmemh("golden_0.hex", golden_mem);
[L3225] ```
[L3226] 
[L3227] 当 DUT 计算完成后，把 DUT 的 128 个输出与 `golden_mem` 逐项比对。
[L3228] 
[L3229] ### 比对方式
[L3230] ```verilog
[L3231] integer k;
[L3232] initial begin
[L3233]     for (k = 0; k < 128; k = k + 1) begin
[L3234]         if (dut_out[k] !== golden_mem[k]) begin
[L3235]             $display("Mismatch at %0d: dut=%h, golden=%h", k, dut_out[k], golden_mem[k]);
[L3236]         end
[L3237]     end
[L3238] end
[L3239] ```
[L3240] 
[L3241] ### 但要注意
[L3242] 你现在的 `golden_i.hex` 是：
[L3243] - 浮点输出再量化
[L3244] 
[L3245] 所以前提是你的硬件计算链路也得尽量模拟这个逻辑。  
[L3246] 如果你 RTL 里已经做成了严格整数推理，那这个 golden 就不再是最优基准了。
[L3247] 
[L3248] ---
[L3249] 
[L3250] # 3. 你当前这套文件更适合什么阶段
[L3251] 
[L3252] 这套导出结果，最适合：
[L3253] 
[L3254] ## 适合
[L3255] - 第一层 `fc1 + relu` 的硬件原型验证
[L3256] - BRAM 读写流程验证
[L3257] - 地址映射验证
[L3258] - 符号位解释验证
[L3259] - testbench 对拍基本通路
[L3260] 
[L3261] ## 不适合
[L3262] - 最终的完整 MNIST 分类演示
[L3263] - 严格的整数推理验证
[L3264] - 偏置 int32 链路验证
[L3265] - 静态量化部署闭环
[L3266] 
[L3267] ---
[L3268] 
[L3269] # 4. 推荐你怎么用这批文件做联调
[L3270] 
[L3271] ---
[L3272] 
[L3273] ## 阶段 1：先验证存储与读取
[L3274] 先只做三件事：
[L3275] 
[L3276] 1. `weights.hex` 是否正确加载
[L3277] 2. `input_i.hex` 是否正确加载
[L3278] 3. `golden_i.hex` 是否能和当前 RTL 输出逐项对上
[L3279] 
[L3280] 只要这一步过了，说明：
[L3281] - 文件格式没问题
[L3282] - 读写顺序没问题
[L3283] - 地址映射没问题
[L3284] 
[L3285] ---
[L3286] 
[L3287] ## 阶段 2：确认你的计算顺序与导出顺序一致
[L3288] 你需要确认 RTL 的权重寻址是不是：
[L3289] 
[L3290] \[
[L3291] addr = out\_idx \times 784 + in\_idx
[L3292] \]
[L3293] 
[L3294] 如果不是，就会出现：
[L3295] - 数据都能读到
[L3296] - 结果却完全不对
[L3297] 
[L3298] 这是最常见的问题。
[L3299] 
[L3300] ---
[L3301] 
[L3302] ## 阶段 3：确认 signed 解释
[L3303] 这是第二常见问题。  
[L3304] 如果 RTL 里没有把 `8'hff` 当成 `-1`，而是当成 `255`，结果一定炸。
[L3305] 
[L3306] 所以一定要检查：
[L3307] - 输入 signed
[L3308] - 权重 signed
[L3309] - 偏置 signed
[L3310] - 乘法 signed
[L3311] - 累加器 signed
[L3312] 
[L3313] ---
[L3314] 
[L3315] # 5. 我建议你现在这样理解每个文件
[L3316] 
[L3317] 你可以把当前文件理解成这张表：
[L3318] 
[L3319] | 文件 | 含义 | 数据类型 | 行数 | 作用 |
[L3320] |---|---|---:|---:|---|
[L3321] | `weights.hex` | `fc1` 权重 | int8 | 100352 | DUT 权重存储初始化 |
[L3322] | `bias.hex` | `fc1` 偏置 | int8 | 128 | 当前版本偏置初始化 |
[L3323] | `quant_params.npz` | 量化参数记录 | float/int | 少量字段 | 软件辅助，不是闭环核心 |
[L3324] | `input_i.hex` | 第 i 个输入图像 | int8 | 784 | 测试输入 |
[L3325] | `golden_i.hex` | 第 i 个参考输出 | int8 | 128 | testbench 对拍 |
[L3326] 
[L3327] ---
[L3328] 
[L3329] # 6. 你现在这批文件的“正确使用姿势”
[L3330] 
[L3331] 如果你此刻不改导出逻辑，而是先拿现有文件验证硬件，那建议这样：
[L3332] 
[L3333] ### 你现在的 DUT 目标
[L3334] 做一个：
[L3335] - `784 -> 128`
[L3336] - int8 输入
[L3337] - int8 权重
[L3338] - int8 偏置
[L3339] - 累加
[L3340] - ReLU
[L3341] - 最后输出和 `golden_i.hex` 尽量一致
[L3342] 
[L3343] ### testbench 流程
[L3344] 1. `$readmemh("weights.hex", weight_mem)`
[L3345] 2. `$readmemh("bias.hex", bias_mem)`
[L3346] 3. `$readmemh("input_0.hex", input_mem)`
[L3347] 4. `$readmemh("golden_0.hex", golden_mem)`
[L3348] 5. 把 `input_mem` 送入 DUT
[L3349] 6. DUT 输出 128 维结果
[L3350] 7. 与 `golden_mem` 逐项比较
[L3351] 
[L3352] 这样可以先把系统跑起来。
[L3353] 
[L3354] ---
[L3355] 
[L3356] # 7. 你后面升级到最终方案时，文件会怎么变
[L3357] 
[L3358] 你后面如果按我们前面定的最终方案升级，文件会改成更合理的结构：
[L3359] 
[L3360] - `fc1_weight_int8.hex`
[L3361] - `fc1_bias_int32.hex`
[L3362] - `fc2_weight_int8.hex`
[L3363] - `fc2_bias_int32.hex`
[L3364] - `input_i.hex`
[L3365] - `fc1_acc_i.hex`
[L3366] - `fc1_relu_i.hex`
[L3367] - `fc1_out_i.hex`
[L3368] - `fc2_acc_i.hex`
[L3369] - `logits_i.hex`
[L3370] - `pred_i.txt`
[L3371] - `quant_config.json`
[L3372] 
[L3373] 这时每个文件的职责会更清晰，硬件对拍也更容易定位问题。
[L3374] 
[L3375] ---
[L3376] 
[L3377] # 8. 你现在最需要注意的两个坑
[L3378] 
[L3379] ## 坑 1：`bias.hex` 现在是 int8
[L3380] 这不是最终可部署形式。  
[L3381] 所以你现在用它时，只能把它当作“当前软件逻辑下的配套文件”，不要把它当最终硬件部署格式。
[L3382] 
[L3383] ## 坑 2：`golden_i.hex` 不是整数推理 golden
[L3384] 它现在只是：
[L3385] - 浮点输出
[L3386] - 再量化
[L3387] 
[L3388] 所以如果你 RTL 做得特别“硬件真实”，反而可能和它对不上。
[L3389] 
[L3390] ---
[L3391] 
[L3392] 如果你把**已经导出的几个 `.hex` 文件内容**贴给我几行，或者把你当前 `quantized_data/` 目录再发我，我可以继续帮你做第二步：**逐个文件按实际内容检查格式是否正确，并给你写一个对应的 Verilog testbench 读取模板**。
[L3393] 
[L3394] 用户:
[L3395] bias.hex:000008e5
[L3396] 0000183f
[L3397] 000001a2
[L3398] ffffe6b8
[L3399] fffffa66
[L3400] fffffdd6
[L3401] 00000acd
[L3402] 00000e73
[L3403] fffff423
[L3404] 00000244
[L3405] fffff947
[L3406] 00000cc4
[L3407] fffff784
[L3408] 0000061c
[L3409] ffffed80
[L3410] 00000dd7
[L3411] fffffffe
[L3412] fffff9de
[L3413] fffffb86
[L3414] ffffed26
[L3415] 00000c70
[L3416] 0000116d
[L3417] ffffff7f
[L3418] 000001a8
[L3419] 0000049e
[L3420] 0000118f
[L3421] 000006c6
[L3422] 00000e32
[L3423] 000025f7
[L3424] ffffe677
[L3425] 000005b3
[L3426] 00000b83
[L3427] 00002162
[L3428] 00001ce9
[L3429] 00000409
[L3430] fffffbcc
[L3431] 00000487
[L3432] 00000caa
[L3433] fffff5f9
[L3434] fffff633
[L3435] 000003b1
[L3436] 00000fd1
[L3437] 000005d1
[L3438] 00000ce3
[L3439] 0000106f
[L3440] 00000c5a
[L3441] 0000019f
[L3442] 00000681
[L3443] ffffef2b
[L3444] fffff49a
[L3445] ffffe4d9
[L3446] fffffb73
[L3447] 0000019f
[L3448] 00000dd1
[L3449] 0000016b
[L3450] 00000730
[L3451] fffff9a5
[L3452] 000009db
[L3453] 000008ec
[L3454] 000003c1
[L3455] 00000b59
[L3456] 00001263
[L3457] fffff71a
[L3458] 00001706
[L3459] 000013f9
[L3460] 000001e7
[L3461] 000027ca
[L3462] 00000779
[L3463] 000000d3
[L3464] fffffe21
[L3465] 0000051c
[L3466] 00000e9c
[L3467] 00000c2f
[L3468] fffffc59
[L3469] 0000142d
[L3470] 00000956
[L3471] 00000813
[L3472] fffffa40
[L3473] 00000d2a
[L3474] 00000c81
[L3475] fffffd14
[L3476] 000008ee
[L3477] 0000144a
[L3478] fffff816
[L3479] 00000be6
[L3480] 00000850
[L3481] 000012d6
[L3482] 00001bd6
[L3483] 00000388
[L3484] fffffc00
[L3485] 00001590
[L3486] 00000565
[L3487] 00000adb
[L3488] 00000f82
[L3489] 00000242
[L3490] 00001ed3
[L3491] fffff3bf
[L3492] 00001420
[L3493] 00000b0e
[L3494] 00000ac2
[L3495] fffffefd
[L3496] 00000072
[L3497] fffff4f2
[L3498] fffffc4f
[L3499] 00000904
[L3500] 00000a0f
[L3501] 0000078d
[L3502] 00000cba
[L3503] fffffb5e
[L3504] fffffe7a
[L3505] 00000a64
[L3506] 00000e22
[L3507] 00000435
[L3508] 00000642
[L3509] 00002492
[L3510] 00002235
[L3511] 000011d9
[L3512] fffff921
[L3513] ffffff6a
[L3514] 00001402
[L3515] 00001a36
[L3516] 0000002e
[L3517] 00001f8d
[L3518] ffffff3e
[L3519] fffff936
[L3520] 00000023
[L3521] fffffe93
[L3522] fffff438
[L3523] weight.hex:aa
[L3524] e5
[L3525] 00
[L3526] 0d
[L3527] 0d
[L3528] f7
[L3529] 08
[L3530] e8
[L3531] ff
[L3532] 20
[L3533] 15
[L3534] c5
[L3535] fe
[L3536] fb
[L3537] e2
[L3538] f9
[L3539] 16
[L3540] 15
[L3541] fc
[L3542] 15
[L3543] 09
[L3544] 13
[L3545] 09
[L3546] cb
[L3547] 28
[L3548] ca
[L3549] 0f
[L3550] 15
[L3551] 0f
[L3552] 05
[L3553] 16
[L3554] f3
[L3555] d7
[L3556] f6
[L3557] 06
[L3558] 0c
[L3559] d8
[L3560] 1e
[L3561] ef
[L3562] 03
[L3563] 12
[L3564] 04
[L3565] e6
[L3566] dd
[L3567] d3
[L3568] 0d
[L3569] fa
[L3570] 1c
[L3571] 1c
[L3572] 0f
[L3573] 0d
[L3574] fd
[L3575] 10
[L3576] 04
[L3577] 01
[L3578] b7
[L3579] 0d
[L3580] 00
[L3581] 08
[L3582] 15
[L3583] d2
[L3584] e8
[L3585] 0f
[L3586] f3
[L3587] bd
[L3588] 19
[L3589] e2
[L3590] 1a
[L3591] f1
[L3592] e2
[L3593] f5
[L3594] f2
[L3595] c6
[L3596] ff
[L3597] ec
[L3598] 02
[L3599] d8
[L3600] e9
[L3601] fe
[L3602] 0f
[L3603] 16
[L3604] d8
[L3605] d6
[L3606] 25
[L3607] e2
[L3608] 06
[L3609] bd
[L3610] 24
[L3611] 0d
[L3612] 1c
[L3613] c2
[L3614] 1c
[L3615] ee
[L3616] e0
[L3617] e7
[L3618] ff
[L3619] 05
[L3620] 01
[L3621] 0f
[L3622] ff
[L3623] fa
[L3624] 16
[L3625] 09
[L3626] 08
[L3627] df
[L3628] ff
[L3629] 12
[L3630] d2
[L3631] ea
[L3632] 0f
[L3633] b4
[L3634] 0f
[L3635] fb
[L3636] f4
[L3637] 24
[L3638] 81
[L3639] 13
[L3640] f1
[L3641] 17
[L3642] 0a
[L3643] ed
[L3644] 03
[L3645] f7
[L3646] ed
[L3647] c0
[L3648] e4
[L3649] da
[L3650] 19
[L3651] fb
[L3652] fd
[L3653] fa
[L3654] bf
[L3655] 14
[L3656] 15
[L3657] 17
[L3658] e1
[L3659] df
[L3660] f6
[L3661] f0
[L3662] 1c
[L3663] 10
[L3664] 0c
[L3665] e9
[L3666] cf
[L3667] e0
[L3668] 25
[L3669] eb
[L3670] e2
[L3671] 1e
[L3672] 13
[L3673] e9
[L3674] e9
[L3675] 18
[L3676] 0b
[L3677] dd
[L3678] eb
[L3679] fe
[L3680] ee
[L3681] 10
[L3682] 14
[L3683] 0c
[L3684] 23
[L3685] f1
[L3686] e9
[L3687] d9
[L3688] c1
[L3689] fa
[L3690] fd
[L3691] 16
[L3692] c9
[L3693] 2b
[L3694] 13
[L3695] 10
[L3696] d6
[L3697] 00
[L3698] ea
[L3699] ce
[L3700] e4
[L3701] e1
[L3702] d0
[L3703] ec
[L3704] 0e
[L3705] 0f
[L3706] 13
[L3707] f6
[L3708] f6
[L3709] 12
[L3710] 04
[L3711] 08
[L3712] 1b
[L3713] 0f
[L3714] 0b
[L3715] 02
[L3716] ea
[L3717] d1
[L3718] da
[L3719] e3
[L3720] e4
[L3721] dc
[L3722] 1b
[L3723] d0
[L3724] 18
[L3725] f0
[L3726] 02
[L3727] 48
[L3728] 06
[L3729] 20
[L3730] 2a
[L3731] e4
[L3732] 1b
[L3733] 0e
[L3734] cb
[L3735] 14
[L3736] df
[L3737] 3a
[L3738] 47
[L3739] 0e
[L3740] 2c
[L3741] 02
[L3742] e6
[L3743] 01
[L3744] 21
[L3745] 00
[L3746] f3
[L3747] e4
[L3748] cd
[L3749] c9
[L3750] d5
[L3751] df
[L3752] f1
[L3753] cd
[L3754] ed
[L3755] 18
[L3756] 11
[L3757] 0e
[L3758] 34
[L3759] 16
[L3760] f1
[L3761] 14
[L3762] db
[L3763] fd
[L3764] 0e
[L3765] 34
[L3766] 30
[L3767] df
[L3768] ec
[L3769] de
[L3770] f2
[L3771] cf
[L3772] e1
[L3773] c0
[L3774] fd
[L3775] d6
[L3776] 10
[L3777] 4f
[L3778] 1c
[L3779] 03
[L3780] 25
[L3781] 05
[L3782] fa
[L3783] fd
[L3784] 19
[L3785] 13
[L3786] cb
[L3787] 08
[L3788] 20
[L3789] 03
[L3790] 39
[L3791] 0b
[L3792] 1e
[L3793] ea
[L3794] f3
[L3795] 0b
[L3796] 21
[L3797] ee
[L3798] 16
[L3799] 03
[L3800] af
[L3801] 14
[L3802] ae
[L3803] 07
[L3804] ff
[L3805] f4
[L3806] 0c
[L3807] aa
[L3808] 09
[L3809] e6
[L3810] e9
[L3811] d2
[L3812] 24
[L3813] e8
[L3814] 16
[L3815] 0d
[L3816] f8
[L3817] 14
[L3818] 28
[L3819] fb
[L3820] df
[L3821] 9c
[L3822] 05
[L3823] 18
[L3824] f0
[L3825] 03
[L3826] 45
[L3827] 1e
[L3828] f8
[L3829] 0a
[L3830] fc
[L3831] 04
[L3832] 32
[L3833] 26
[L3834] 0c
[L3835] 1b
[L3836] e6
[L3837] 1a
[L3838] 20
[L3839] ff
[L3840] d9
[L3841] 19
[L3842] b8
[L3843] 12
[L3844] 1b
[L3845] 13
[L3846] 11
[L3847] e8
[L3848] 0b
[L3849] e1
[L3850] 26
[L3851] d4
[L3852] 11
[L3853] e2
[L3854] 92
[L3855] e0
[L3856] fb
[L3857] f5
[L3858] 15
[L3859] 1c
[L3860] 23
[L3861] 15
[L3862] f7
[L3863] 09
[L3864] e4
[L3865] 13
[L3866] dc
[L3867] f9
[L3868] fe
[L3869] f2
[L3870] 14
[L3871] 0a
[L3872] cc
[L3873] eb
[L3874] 02
[L3875] 07
[L3876] fa
[L3877] d1
[L3878] dc
[L3879] 1e
[L3880] 13
[L3881] ec
[L3882] 10
[L3883] 1e
[L3884] 00
[L3885] 1f
[L3886] 13
[L3887] 1a
[L3888] db
[L3889] 16
[L3890] 08
[L3891] 03
[L3892] 0e
[L3893] fe
[L3894] e0
[L3895] 37
[L3896] 0c
[L3897] f0
[L3898] 1a
[L3899] 0e
[L3900] 10
[L3901] 36
[L3902] d2
[L3903] 07
[L3904] fc
[L3905] bb
[L3906] d4
[L3907] db
[L3908] f7
[L3909] 00
[L3910] 0f
[L3911] 0b
[L3912] 0f
[L3913] da
[L3914] d7
[L3915] 08
[L3916] 06
[L3917] 0b
[L3918] 05
[L3919] ff
[L3920] 16
[L3921] 34
[L3922] 02
[L3923] e1
[L3924] 0d
[L3925] 1a
[L3926] 0f
[L3927] 9a
[L3928] dc
[L3929] e8
[L3930] 1e
[L3931] dd
[L3932] 0f
[L3933] f0
[L3934] d8
[L3935] f5
[L3936] 0f
[L3937] f0
[L3938] 05
[L3939] 95
[L3940] c1
[L3941] ec
[L3942] 19
[L3943] 35
[L3944] eb
[L3945] 08
[L3946] 2d
[L3947] 2e
[L3948] db
[L3949] 28
[L3950] fc
[L3951] dd
[L3952] f0
[L3953] f2
[L3954] 12
[L3955] 1a
[L3956] fd
[L3957] 18
[L3958] fa
[L3959] ec
[L3960] cb
[L3961] f7
[L3962] 0a
[L3963] f0
[L3964] 10
[L3965] c4
[L3966] 0a
[L3967] 14
[L3968] e1
[L3969] 0c
[L3970] 03
[L3971] 06
[L3972] 07
[L3973] e9
[L3974] 04
[L3975] 12
[L3976] 0f
[L3977] 08
[L3978] c5
[L3979] 0d
[L3980] fb
[L3981] d1
[L3982] 21
[L3983] dd
[L3984] 13
[L3985] d9
[L3986] ee
[L3987] 0d
[L3988] 09
[L3989] fb
[L3990] 02
[L3991] 00
[L3992] ff
[L3993] e4
[L3994] d3
[L3995] 14
[L3996] f3
[L3997] ce
[L3998] c6
[L3999] 10
[L4000] 1f
[L4001] 13
[L4002] 0d
[L4003] 05
[L4004] 1d
[L4005] d3
[L4006] 22
[L4007] 2b
[L4008] fb
[L4009] 15
[L4010] 07
[L4011] cf
[L4012] 09
[L4013] f3
[L4014] 26
[L4015] 13
[L4016] bd
[L4017] 04
[L4018] e6
[L4019] f6
[L4020] 11
[L4021] f1
[L4022] 46
[L4023] ef
[L4024] 0a
[L4025] f9
[L4026] e6
[L4027] cb
[L4028] 0c
[L4029] eb
[L4030] 1d
[L4031] 16
[L4032] 08
[L4033] 0e
[L4034] 2c
[L4035] e8
[L4036] 08
[L4037] fa
[L4038] ee
[L4039] d2
[L4040] dd
[L4041] 19
[L4042] 1d
[L4043] ce
[L4044] 9c
[L4045] fa
[L4046] a2
[L4047] db
[L4048] 0c
[L4049] c8
[L4050] f5
[L4051] d5
[L4052] ca
[L4053] ea
[L4054] ed
[L4055] 1e
[L4056] 22
[L4057] 19
[L4058] 1f
[L4059] f0
[L4060] 0b
[L4061] 04
[L4062] 07
[L4063] de
[L4064] 04
[L4065] 00
[L4066] 09
[L4067] fc
[L4068] 17
[L4069] 09
[L4070] ec
[L4071] 12
[L4072] c4
[L4073] 8a
[L4074] 0e
[L4075] ec
[L4076] 04
[L4077] 0d
[L4078] e7
[L4079] 12
[L4080] 15
[L4081] 0b
[L4082] 8f
[L4083] d2
[L4084] e7
[L4085] c2
[L4086] 0f
[L4087] f7
[L4088] 12
[L4089] dc
[L4090] 0d
[L4091] e1
[L4092] 18
[L4093] 07
[L4094] d4
[L4095] 08
[L4096] 28
[L4097] ab
[L4098] 0a
[L4099] b4
[L4100] 1c
[L4101] 0b
[L4102] 05
[L4103] 02
[L4104] 0b
[L4105] 11
[L4106] 22
[L4107] 02
[L4108] e9
[L4109] 14
[L4110] e9
[L4111] a7
[L4112] 15
[L4113] bb
[L4114] 08
[L4115] e1
[L4116] fd
[L4117] 00
[L4118] e4
[L4119] f0
[L4120] 0d
[L4121] 23
[L4122] 29
[L4123] ba
[L4124] 2b
[L4125] 09
[L4126] 0c
[L4127] 13
[L4128] 47
[L4129] ff
[L4130] 0a
[L4131] 14
[L4132] 0e
[L4133] 1a
[L4134] f8
[L4135] 97
[L4136] d4
[L4137] e0
[L4138] eb
[L4139] 0a
[L4140] 0f
[L4141] ed
[L4142] 3f
[L4143] 18
[L4144] 15
[L4145] a5
[L4146] 10
[L4147] 07
[L4148] 13
[L4149] 2d
[L4150] 0f
[L4151] 12
[L4152] f6
[L4153] 0d
[L4154] 09
[L4155] 0f
[L4156] 05
[L4157] 08
[L4158] e9
[L4159] 14
[L4160] 11
[L4161] f1
[L4162] bb
[L4163] 38
[L4164] 1f
[L4165] ff
[L4166] 02
[L4167] 05
[L4168] ff
[L4169] e2
[L4170] 38
[L4171] fd
[L4172] 0b
[L4173] 0a
[L4174] 3a
[L4175] ec
[L4176] b2
[L4177] 2d
[L4178] f0
[L4179] db
[L4180] cf
[L4181] ed
[L4182] eb
[L4183] c4
[L4184] 05
[L4185] df
[L4186] 03
[L4187] a5
[L4188] 02
[L4189] 02
[L4190] 12
[L4191] 46
[L4192] 00
[L4193] 0b
[L4194] ee
[L4195] 22
[L4196] e7
[L4197] f0
[L4198] 14
[L4199] ca
[L4200] 0d
[L4201] 2e
[L4202] f1
[L4203] ba
[L4204] 05
[L4205] 2c
[L4206] 0c
[L4207] ec
[L4208] 0d
[L4209] fa
[L4210] 03
[L4211] fb
[L4212] 0b
[L4213] f4
[L4214] e7
[L4215] 12
[L4216] 98
[L4217] c2
[L4218] 08
[L4219] d2
[L4220] 04
[L4221] f3
[L4222] 1b
[L4223] 09
[L4224] f8
[L4225] f9
[L4226] f2
[L4227] 28
[L4228] f9
[L4229] 30
[L4230] e9
[L4231] 06
[L4232] e5
[L4233] f6
[L4234] c2
[L4235] 13
[L4236] f5
[L4237] 14
[L4238] 21
[L4239] 50
[L4240] c4
[L4241] 07
[L4242] e2
[L4243] f3
[L4244] 0d
[L4245] fa
[L4246] f3
[L4247] 0c
[L4248] 0c
[L4249] 08
[L4250] 15
[L4251] 10
[L4252] e9
[L4253] 10
[L4254] 0b
[L4255] 0e
[L4256] 01
[L4257] eb
[L4258] 12
[L4259] ed
[L4260] 0b
[L4261] 0f
[L4262] 10
[L4263] d0
[L4264] 0c
[L4265] 1a
[L4266] fb
[L4267] ca
[L4268] cf
[L4269] f3
[L4270] cb
[L4271] bd
[L4272] ff
[L4273] 36
[L4274] f6
[L4275] fe
[L4276] f8
[L4277] f2
[L4278] 42
[L4279] 02
[L4280] f8
[L4281] 06
[L4282] 12
[L4283] 1d
[L4284] ce
[L4285] 14
[L4286] 0c
[L4287] 17
[L4288] fb
[L4289] 0e
[L4290] ed
[L4291] 17
[L4292] 15
[L4293] 0c
[L4294] 15
[L4295] 02
[L4296] e6
[L4297] 0f
[L4298] 10
[L4299] 06
[L4300] 02
[L4301] 0c
[L4302] b4
[L4303] 06
[L4304] d7
[L4305] af
[L4306] 2f
[L4307] de
[L4308] e1
[L4309] 40
[L4310] ee
[L4311] ef
[L4312] f3
[L4313] 0c
[L4314] aa
[L4315] 21
[L4316] 0a
[L4317] c6
[L4318] fe
[L4319] f3
[L4320] ff
[L4321] 1f
[L4322] e6
[L4323] 3e
[L4324] 08
[L4325] f8
[L4326] d0
[L4327] d8
[L4328] f3
[L4329] 26
[L4330] 06
[L4331] 19
[L4332] 08
[L4333] fc
[L4334] fe
[L4335] 18
[L4336] eb
[L4337] f5
[L4338] d2
[L4339] d5
[L4340] f4
[L4341] e3
[L4342] 18
[L4343] 10
[L4344] f7
[L4345] ab
[L4346] d3
[L4347] f7
[L4348] ed
[L4349] 16
[L4350] c1
[L4351] 04
[L4352] df
[L4353] f6
[L4354] e9
[L4355] 18
[L4356] 11
[L4357] a9
[L4358] da
[L4359] d3
[L4360] f0
[L4361] 04
[L4362] 15
[L4363] d7
[L4364] 05
[L4365] 04
[L4366] 17
[L4367] 17
[L4368] bf
[L4369] 23
[L4370] ee
[L4371] db
[L4372] ec
[L4373] ff
[L4374] 0f
[L4375] df
[L4376] ea
[L4377] fb
[L4378] 1d
[L4379] ef
[L4380] 08
[L4381] ef
[L4382] e2
[L4383] f0
[L4384] 0b
[L4385] 13
[L4386] e4
[L4387] 0e
[L4388] 12
[L4389] 18
[L4390] ff
[L4391] e1
[L4392] 0e
[L4393] fb
[L4394] 10
[L4395] 05
[L4396] b3
[L4397] 16
[L4398] 08
[L4399] dd
[L4400] 2e
[L4401] e0
[L4402] 12
[L4403] fa
[L4404] ee
[L4405] 17
[L4406] ab
[L4407] 16
[L4408] 0a
[L4409] 0b
[L4410] 0f
[L4411] 15
[L4412] e6
[L4413] 23
[L4414] e4
[L4415] cc
[L4416] 17
[L4417] 16
[L4418] 11
[L4419] 0f
[L4420] a8
[L4421] ff
[L4422] c9
[L4423] e8
[L4424] 2b
[L4425] d4
[L4426] 04
[L4427] f7
[L4428] 07
[L4429] e4
[L4430] 1f
[L4431] eb
[L4432] 01
[L4433] 9e
[L4434] 14
[L4435] 1a
[L4436] 01
[L4437] 11
[L4438] 13
[L4439] 25
[L4440] 03
[L4441] 04
[L4442] f1
[L4443] db
[L4444] 09
[L4445] 05
[L4446] db
[L4447] f2
[L4448] d9
[L4449] dc
[L4450] 1a
[L4451] cc
[L4452] 22
[L4453] 14
[L4454] 09
[L4455] 03
[L4456] 08
[L4457] 0b
[L4458] 01
[L4459] 09
[L4460] 0f
[L4461] d5
[L4462] c9
[L4463] d0
[L4464] f2
[L4465] f7
[L4466] 28
[L4467] 08
[L4468] e6
[L4469] 0a
[L4470] e2
[L4471] d2
[L4472] 18
[L4473] 1d
[L4474] 14
[L4475] 00
[L4476] 09
[L4477] 08
[L4478] f9
[L4479] c3
[L4480] e4
[L4481] f7
[L4482] 17
[L4483] e8
[L4484] cd
[L4485] 18
[L4486] 0b
[L4487] 10
[L4488] 0f
[L4489] 0e
[L4490] 14
[L4491] fc
[L4492] 1a
[L4493] 18
[L4494] 09
[L4495] 14
[L4496] 15
[L4497] 09
[L4498] 12
[L4499] 1c
[L4500] da
[L4501] f9
[L4502] d8
[L4503] e0
[L4504] 07
[L4505] f6
[L4506] 0f
[L4507] 14
[L4508] b2
[L4509] e1
[L4510] 13
[L4511] c3
[L4512] e9
[L4513] 05
[L4514] 3a
[L4515] ed
[L4516] 1f
[L4517] f3
[L4518] 20
[L4519] 27
[L4520] 0a
[L4521] c8
[L4522] d8
[L4523] 32
[L4524] 19
[L4525] 27
[L4526] 25
[L4527] dc
[L4528] e2
[L4529] 20
[L4530] ff
[L4531] 04
[L4532] e9
[L4533] 50
[L4534] 00
[L4535] de
[L4536] db
[L4537] ec
[L4538] d6
[L4539] d5
[L4540] 14
[L4541] 0b
[L4542] 06
[L4543] ea
[L4544] d8
[L4545] 21
[L4546] ff
[L4547] f1
[L4548] 1b
[L4549] f3
[L4550] 0b
[L4551] 15
[L4552] 0d
[L4553] 0f
[L4554] 24
[L4555] 01
[L4556] fd
[L4557] 0f
[L4558] e2
[L4559] 13
[L4560] e4
[L4561] ad
[L4562] 02
[L4563] 14
[L4564] 05
[L4565] ce
[L4566] f0
[L4567] e1
[L4568] fe
[L4569] 0f
[L4570] ed
[L4571] 0d
[L4572] 16
[L4573] 05
[L4574] 17
[L4575] cf
[L4576] 14
[L4577] 15
[L4578] 13
[L4579] df
[L4580] c4
[L4581] 10
[L4582] ec
[L4583] 05
[L4584] 14
[L4585] 01
[L4586] b7
[L4587] dd
[L4588] 10
[L4589] c5
[L4590] 0b
[L4591] 02
[L4592] 12
[L4593] 06
[L4594] d9
[L4595] f8
[L4596] 16
[L4597] 14
[L4598] 0b
[L4599] 04
[L4600] 02
[L4601] e5
[L4602] 10
[L4603] 0b
[L4604] 16
[L4605] cc
[L4606] df
[L4607] 17
[L4608] 10
[L4609] 12
[L4610] 0d
[L4611] 0d
[L4612] d4
[L4613] e6
[L4614] cd
[L4615] 10
[L4616] 0c
[L4617] 13
[L4618] b7
[L4619] 0b
[L4620] 01
[L4621] 18
[L4622] fc
[L4623] bb
[L4624] 03
[L4625] 10
[L4626] ce
[L4627] fc
[L4628] 15
[L4629] 18
[L4630] fd
[L4631] 11
[L4632] 0a
[L4633] 14
[L4634] b5
[L4635] 07
[L4636] da
[L4637] 1d
[L4638] db
[L4639] 17
[L4640] ec
[L4641] 08
[L4642] f1
[L4643] 0d
[L4644] ef
[L4645] 03
[L4646] c2
[L4647] ec
[L4648] ee
[L4649] 16
[L4650] 13
[L4651] 14
[L4652] 08
[L4653] ce
[L4654] b3
[L4655] 01
[L4656] dc
[L4657] ee
[L4658] 01
[L4659] f5
[L4660] 12
[L4661] ba
[L4662] 86
[L4663] d5
[L4664] 0b
[L4665] 18
[L4666] fe
[L4667] 22
[L4668] 0b
[L4669] f9
[L4670] 04
[L4671] 15
[L4672] 17
[L4673] e6
[L4674] 01
[L4675] 27
[L4676] ea
[L4677] fc
[L4678] 17
[L4679] ec
[L4680] 99
[L4681] 03
[L4682] ae
[L4683] e0
[L4684] e7
[L4685] ea
[L4686] c3
[L4687] 01
[L4688] 12
[L4689] 34
[L4690] 10
[L4691] 0b
[L4692] d7
[L4693] f2
[L4694] 14
[L4695] 14
[L4696] f3
[L4697] 10
[L4698] 1d
[L4699] 1d
[L4700] ef
[L4701] 16
[L4702] ff
[L4703] 26
[L4704] 0d
[L4705] fa
[L4706] 03
[L4707] 1c
[L4708] f6
[L4709] 12
[L4710] e3
[L4711] 18
[L4712] 11
[L4713] ba
[L4714] d8
[L4715] ef
[L4716] f5
[L4717] 2a
[L4718] fc
[L4719] 11
[L4720] 15
[L4721] 02
[L4722] 16
[L4723] 17
[L4724] 06
[L4725] 13
[L4726] 0b
[L4727] 08
[L4728] fb
[L4729] 14
[L4730] fa
[L4731] 20
[L4732] fe
[L4733] 21
[L4734] 01
[L4735] 0b
[L4736] 20
[L4737] f7
[L4738] 14
[L4739] ec
[L4740] c8
[L4741] 08
[L4742] 16
[L4743] fe
[L4744] 0a
[L4745] 10
[L4746] 1e
[L4747] 09
[L4748] aa
[L4749] 05
[L4750] ec
[L4751] e6
[L4752] fd
[L4753] 04
[L4754] 1b
[L4755] d6
[L4756] bd
[L4757] eb
[L4758] 16
[L4759] 16
[L4760] 16
[L4761] cc
[L4762] 25
[L4763] e7
[L4764] eb
[L4765] 07
[L4766] 17
[L4767] f2
[L4768] ab
[L4769] 10
[L4770] c9
[L4771] f9
[L4772] d5
[L4773] 06
[L4774] ff
[L4775] 16
[L4776] 00
[L4777] 0f
[L4778] ef
[L4779] b6
[L4780] 0d
[L4781] b6
[L4782] ae
[L4783] 03
[L4784] 10
[L4785] e0
[L4786] eb
[L4787] f3
[L4788] e6
[L4789] 9c
[L4790] 09
[L4791] bd
[L4792] 0d
[L4793] e9
[L4794] d2
[L4795] dc
[L4796] 14
[L4797] 97
[L4798] 11
[L4799] 0f
[L4800] 0c
[L4801] 0b
[L4802] 11 先检查这两个有没有问题，然后根据现在的文件和坑，你看看代码有什么需要改进的，最好生成的数据可以被fpga上得cim系统直接使用
[L4803] AI助手:
[L4804] 
[L4805] 
[L4806] AI助手:
[L4807] 我先按“文件内容本身是否自洽”和“是否适合 FPGA/CIM 直接吃”两层来判断。眼下一个明显信号是：你贴出来的 `bias.hex` 已经是 **32 位有符号十六进制**，这比你最初 notebook 里那个 int8 bias 更合理，说明你的导出逻辑应该已经改过一版了。
[L4808] 
[L4809] AI助手:
[L4810] 先说结论：
[L4811] 
[L4812] **你现在贴出来的这两个文件，格式上基本是对的，而且已经比最初 notebook 版本更适合 FPGA/CIM 直接使用。**
[L4813] 
[L4814] 尤其是：
[L4815] 
[L4816] - `bias.hex`：现在是**32 位有符号补码**，这很合理，适合 MAC 后加 bias
[L4817] - `weight.hex`：现在是**8 位有符号补码**，也合理，适合 int8 权重阵列
[L4818] 
[L4819] 这说明你的导出代码已经朝着“真实整数推理链路”改进过了，不再是最初那个 bias 只有 2 位 hex 的版本。
[L4820] 
[L4821] 下面我分三部分说：
[L4822] 
[L4823] 1. 先检查这两个文件有没有明显问题  
[L4824] 2. 结合现在的文件格式，说明 FPGA/CIM 应该怎么直接使用  
[L4825] 3. 按“还能怎么改代码”给你一套更适合直接上板的改进建议
[L4826] 
[L4827] ---
[L4828] 
[L4829] # 1. 先检查这两个文件有没有问题
[L4830] 
[L4831] ---
[L4832] 
[L4833] ## 1.1 `bias.hex` 检查结果
[L4834] 
[L4835] 你给的 `bias.hex` 形如：
[L4836] 
[L4837] ```text
[L4838] 000008e5
[L4839] 0000183f
[L4840] 000001a2
[L4841] ffffe6b8
[L4842] ...
[L4843] ```
[L4844] 
[L4845] ### 这说明什么
[L4846] 每行都是 **8 个十六进制字符**，也就是 **32 bit**。
[L4847] 
[L4848] 这对应的是：
[L4849] 
[L4850] - 正数：直接写成 `000008e5`
[L4851] - 负数：写成二补码形式，比如 `ffffe6b8`
[L4852] 
[L4853] 这正是 FPGA 里最常见的 **signed int32 bias 文件格式**。
[L4854] 
[L4855] ### 这和硬件链路匹配吗
[L4856] 匹配，而且是更合理的做法：
[L4857] 
[L4858] - 输入：int8
[L4859] - 权重：int8
[L4860] - 累加：int32
[L4861] - 偏置：int32
[L4862] - ReLU / requant：后续再做
[L4863] 
[L4864] 所以从工程角度说，**这个 bias 格式是对的**。
[L4865] 
[L4866] ### 我看出来的一个信号
[L4867] 你这份 `bias.hex` 已经不是最开始 notebook 里那种：
[L4868] 
[L4869] ```python
[L4870] f.write(f'{val.item() & 0xFF:02x}\n')
[L4871] ```
[L4872] 
[L4873] 而是明显变成了类似：
[L4874] 
[L4875] ```python
[L4876] f.write(f'{val.item() & 0xFFFFFFFF:08x}\n')
[L4877] ```
[L4878] 
[L4879] 这说明你已经把 bias 导出升级成 int32 了。这个改动是正确的。
[L4880] 
[L4881] ### 需要确认的一点
[L4882] 如果这是 `fc1` 的 bias，那么它应该有：
[L4883] 
[L4884] - **128 行**
[L4885] 
[L4886] 如果这是 `fc2` 的 bias，那么应该有：
[L4887] 
[L4888] - **10 行**
[L4889] 
[L4890] 你贴出来看起来更像是 128 行中的一整段。  
[L4891] 如果你最终网络是两层 MLP，那建议不要只叫 `bias.hex`，而是明确改成：
[L4892] 
[L4893] - `fc1_bias_int32.hex`
[L4894] - `fc2_bias_int32.hex`
[L4895] 
[L4896] 这样不会混。
[L4897] 
[L4898] ---
[L4899] 
[L4900] ## 1.2 `weight.hex` 检查结果
[L4901] 
[L4902] 你给的 `weight.hex` 形如：
[L4903] 
[L4904] ```text
[L4905] aa
[L4906] e5
[L4907] 00
[L4908] 0d
[L4909] 0d
[L4910] f7
[L4911] ...
[L4912] ```
[L4913] 
[L4914] ### 这说明什么
[L4915] 每行都是 **2 个十六进制字符**，也就是 **8 bit**。
[L4916] 
[L4917] 这对应：
[L4918] 
[L4919] - `aa` = -86
[L4920] - `e5` = -27
[L4921] - `00` = 0
[L4922] - `0d` = 13
[L4923] - `f7` = -9
[L4924] 
[L4925] 这正是标准的 **signed int8 权重存储格式**。
[L4926] 
[L4927] ### 是否合理
[L4928] 合理，前提是 RTL 里按 signed 解释：
[L4929] 
[L4930] ```verilog
[L4931] wire signed [7:0] w = $signed(weight_mem[addr]);
[L4932] ```
[L4933] 
[L4934] ### 数值分布上看有没有异常
[L4935] 从你贴的内容看，数值大多集中在：
[L4936] 
[L4937] - `0xC0 ~ 0x40` 左右对应的 -64 到 64 之间
[L4938] - 偶尔有 `0x81`、`0x9c`、`0x8a`、`0x50`、`0x4f`
[L4939] 
[L4940] 这很像正常量化后的神经网络权重分布，不像导出炸掉了。
[L4941] 
[L4942] 也就是说：
[L4943] 
[L4944] - **没有明显出现全 00 / 全 ff / 大面积饱和到 7f/80 的异常**
[L4945] - 看起来像一份正常的量化权重
[L4946] 
[L4947] ### 需要确认的一点
[L4948] 如果这是 `fc1.weight`，而且第一层 shape 是 `[128, 784]`，那它总行数应该是：
[L4949] 
[L4950] \[
[L4951] 128 \times 784 = 100352
[L4952] \]
[L4953] 
[L4954] 如果最终网络是两层 MLP，你也应该拆成：
[L4955] 
[L4956] - `fc1_weight_int8.hex`
[L4957] - `fc2_weight_int8.hex`
[L4958] 
[L4959] 否则后面上板会混。
[L4960] 
[L4961] ---
[L4962] 
[L4963] # 2. 结合你现在的文件格式，FPGA/CIM 怎么直接使用
[L4964] 
[L4965] 现在按你贴出来的格式，**最适合的硬件链路**是：
[L4966] 
[L4967] - `weight.hex`：int8 权重阵列
[L4968] - `bias.hex`：int32 偏置
[L4969] - `input_i.hex`：int8 输入
[L4970] - 累加器：int32
[L4971] - 后面做 ReLU、requant、第二层
[L4972] 
[L4973] 也就是说，你现在已经从“玩具导出格式”走到了“能上硬件”的方向。
[L4974] 
[L4975] ---
[L4976] 
[L4977] ## 2.1 `weight.hex` 的硬件读取方式
[L4978] 
[L4979] 如果你的导出顺序还是之前那样：
[L4980] 
[L4981] ```python
[L4982] for row in q_weight:
[L4983]     for val in row:
[L4984]         ...
[L4985] ```
[L4986] 
[L4987] 那么权重布局就是：
[L4988] 
[L4989] \[
[L4990] W[j][i]
[L4991] \]
[L4992] 
[L4993] 也就是：
[L4994] 
[L4995] - `j`：输出神经元索引
[L4996] - `i`：输入通道索引
[L4997] 
[L4998] ### 地址公式
[L4999] 若第一层 `fc1` shape 是 `[128, 784]`，则：
[L5000] 
[L5001] \[
[L5002] addr = j \times 784 + i
[L5003] \]
[L5004] 
[L5005] RTL 读法就是：
[L5006] 
[L5007] - 外层循环跑输出通道 `j`
[L5008] - 内层循环跑输入通道 `i`
[L5009] 
[L5010] ### 对应计算
[L5011] \[
[L5012] acc_j = \sum_{i=0}^{783} (x_i - z_x)(w_{j,i} - z_w) + b_j
[L5013] \]
[L5014] 
[L5015] 如果你现在是权重对称量化，那通常：
[L5016] 
[L5017] - `z_w = 0`
[L5018] 
[L5019] 如果输入采用 `x_q = pixel_u8 - 128`，那：
[L5020] 
[L5021] - `z_x = -128`
[L5022] 
[L5023] ---
[L5024] 
[L5025] ## 2.2 `bias.hex` 的硬件使用方式
[L5026] 
[L5027] 现在你的 bias 是 32 位二补码，很适合直接这样读：
[L5028] 
[L5029] ```verilog
[L5030] reg [31:0] bias_mem [0:127];
[L5031] initial $readmemh("bias.hex", bias_mem);
[L5032] 
[L5033] wire signed [31:0] bias_val = $signed(bias_mem[j]);
[L5034] ```
[L5035] 
[L5036] 然后在 MAC 完成后：
[L5037] 
[L5038] ```verilog
[L5039] acc <= mac_sum + bias_val;
[L5040] ```
[L5041] 
[L5042] 这比 int8 bias 合理得多。
[L5043] 
[L5044] ---
[L5045] 
[L5046] ## 2.3 对 CIM 系统的直接使用建议
[L5047] 
[L5048] 如果你的 CIM 阵列本体负责的是 `int8 x int8` 乘加，那最常见的系统切分是：
[L5049] 
[L5050] ### CIM 阵列做
[L5051] - 权重存储
[L5052] - 向量输入
[L5053] - 局部乘加 / bitline accumulate / column accumulate
[L5054] 
[L5055] ### FPGA 数字逻辑做
[L5056] - 输入 zero-point 修正
[L5057] - 偏置相加
[L5058] - ReLU
[L5059] - requant
[L5060] - 第二层控制
[L5061] - argmax
[L5062] 
[L5063] 这和你现在的文件格式非常匹配：
[L5064] 
[L5065] - `weight.hex` 给阵列权重
[L5066] - `input_i.hex` 给输入驱动
[L5067] - `bias.hex` 给数字后处理单元
[L5068] - `golden` 给 testbench 对拍
[L5069] 
[L5070] ---
[L5071] 
[L5072] # 3. 这两个文件背后，代码还需要怎么改进
[L5073] 
[L5074] 你现在文件本身没大问题，但为了“**生成的数据可以被 FPGA 上的 CIM 系统直接使用**”，我建议你继续改 8 个点。
[L5075] 
[L5076] ---
[L5077] 
[L5078] ## 改进 1：文件命名必须分层，不能再叫通用名
[L5079] 
[L5080] 你现在如果最终是两层 MLP，绝对不要继续用：
[L5081] 
[L5082] - `weight.hex`
[L5083] - `bias.hex`
[L5084] 
[L5085] 因为你后面会出现：
[L5086] 
[L5087] - 第一层权重
[L5088] - 第二层权重
[L5089] - 第一层 bias
[L5090] - 第二层 bias
[L5091] 
[L5092] 建议直接改成：
[L5093] 
[L5094] - `fc1_weight_int8.hex`
[L5095] - `fc1_bias_int32.hex`
[L5096] - `fc2_weight_int8.hex`
[L5097] - `fc2_bias_int32.hex`
[L5098] 
[L5099] 这样你 RTL 一眼知道读哪个。
[L5100] 
[L5101] ---
[L5102] 
[L5103] ## 改进 2：必须配一个 `quant_config.json`
[L5104] 
[L5105] 现在单靠 hex 文件还不够。  
[L5106] FPGA/CIM 真正要跑整数推理，还需要知道：
[L5107] 
[L5108] - 输入 `scale`
[L5109] - 输入 `zero_point`
[L5110] - 各层 `weight_scale`
[L5111] - 各层 `weight_zero_point`
[L5112] - 各层 `output_scale`
[L5113] - 各层 `output_zero_point`
[L5114] - requant multiplier / shift
[L5115] 
[L5116] 建议额外生成：
[L5117] 
[L5118] ```json
[L5119] {
[L5120]   "input": {
[L5121]     "scale": 0.00392156862745098,
[L5122]     "zero_point": -128
[L5123]   },
[L5124]   "fc1": {
[L5125]     "weight_scale": ...,
[L5126]     "weight_zero_point": 0,
[L5127]     "output_scale": ...,
[L5128]     "output_zero_point": 0,
[L5129]     "bias_scale": ...,
[L5130]     "multiplier_q31": ...,
[L5131]     "shift": ...
[L5132]   },
[L5133]   "fc2": {
[L5134]     "weight_scale": ...,
[L5135]     "weight_zero_point": 0,
[L5136]     "output_scale": ...,
[L5137]     "output_zero_point": 0,
[L5138]     "bias_scale": ...,
[L5139]     "multiplier_q31": ...,
[L5140]     "shift": ...
[L5141]   }
[L5142] }
[L5143] ```
[L5144] 
[L5145] 这样 FPGA 端和 Python 端是一致的。
[L5146] 
[L5147] ---
[L5148] 
[L5149] ## 改进 3：输入文件也要明确 int8 和布局
[L5150] 
[L5151] 你现在最好明确导出成：
[L5152] 
[L5153] - `input_0_int8.hex`
[L5154] - `input_1_int8.hex`
[L5155] 
[L5156] 并且说明：
[L5157] 
[L5158] - 长度 784
[L5159] - 顺序是 row-major
[L5160] - 每行 2 位 hex
[L5161] - 数值是 signed int8
[L5162] - 若采用硬件输入策略，则：
[L5163]   \[
[L5164]   x_q = round(x \cdot 255) - 128
[L5165]   \]
[L5166] 
[L5167] 这能避免你后面忘记输入是“原始像素量化”还是“Normalize 后量化”。
[L5168] 
[L5169] ---
[L5170] 
[L5171] ## 改进 4：golden 不要只保留最终输出，要保留中间级
[L5172] 
[L5173] 这是对硬件调试最关键的一点。
[L5174] 
[L5175] 建议你至少导出：
[L5176] 
[L5177] ### 第一层
[L5178] - `fc1_acc_i.hex`：int32，未激活前
[L5179] - `fc1_relu_i.hex`：int32，过 ReLU 后
[L5180] - `fc1_out_i.hex`：int8，requant 后
[L5181] 
[L5182] ### 第二层
[L5183] - `fc2_acc_i.hex`：int32
[L5184] - `logits_i.hex`：int8
[L5185] 
[L5186] ### 分类
[L5187] - `pred_i.txt`
[L5188] 
[L5189] 这样当 FPGA 结果不对时，你能立刻定位是：
[L5190] 
[L5191] - 阵列乘加错
[L5192] - bias 加错
[L5193] - ReLU 错
[L5194] - requant 错
[L5195] - 第二层地址错
[L5196] 
[L5197] 不然只看最后一个 `golden`，很难找 bug。
[L5198] 
[L5199] ---
[L5200] 
[L5201] ## 改进 5：权重文件要考虑和 CIM 阵列物理映射一致
[L5202] 
[L5203] 你现在的 `weight.hex` 是逻辑顺序 `[out][in]`。  
[L5204] 这对软件和普通 RTL 很方便，但对实际 CIM 阵列，可能还不够。
[L5205] 
[L5206] 因为真实阵列通常不是一次吃 784 个输入，而是分块，例如：
[L5207] 
[L5208] - 每列 64 个输入
[L5209] - 每次做 64×N 子矩阵
[L5210] - 784 被切成很多 tile
[L5211] 
[L5212] 所以你后面很可能需要两套文件：
[L5213] 
[L5214] ### 逻辑文件
[L5215] 给软件 / testbench 用：
[L5216] - `fc1_weight_int8.hex`
[L5217] 
[L5218] ### 物理映射文件
[L5219] 给阵列初始化用：
[L5220] - `fc1_weight_tile0.hex`
[L5221] - `fc1_weight_tile1.hex`
[L5222] - ...
[L5223] 
[L5224] 也就是按 CIM 阵列实际映射切块导出。
[L5225] 
[L5226] 如果你现在阵列设计已经定了，比如每次 64 路输入并行，我建议导出时就按 64 一组打包。
[L5227] 
[L5228] ---
[L5229] 
[L5230] ## 改进 6：给每个文件写 manifest
[L5231] 
[L5232] 建议自动生成一个 `manifest.txt` 或 `manifest.json`，写明：
[L5233] 
[L5234] - 文件名
[L5235] - 数据类型
[L5236] - 位宽
[L5237] - 长度
[L5238] - 布局
[L5239] - 用途
[L5240] 
[L5241] 例如：
[L5242] 
[L5243] ```json
[L5244] {
[L5245]   "fc1_weight_int8.hex": {
[L5246]     "dtype": "int8",
[L5247]     "shape": [128, 784],
[L5248]     "layout": "row-major [out][in]"
[L5249]   },
[L5250]   "fc1_bias_int32.hex": {
[L5251]     "dtype": "int32",
[L5252]     "shape": [128]
[L5253]   },
[L5254]   "input_0_int8.hex": {
[L5255]     "dtype": "int8",
[L5256]     "shape": [784],
[L5257]     "layout": "row-major flattened image"
[L5258]   }
[L5259] }
[L5260] ```
[L5261] 
[L5262] 这会让你的工程后期非常省事。
[L5263] 
[L5264] ---
[L5265] 
[L5266] ## 改进 7：最好支持导出 `.mem` / `.coe` / 打包宽字格式
[L5267] 
[L5268] 现在一行一个数适合 testbench，  
[L5269] 但真正上 FPGA，很多时候你会想直接初始化 BRAM。
[L5270] 
[L5271] 比如：
[L5272] 
[L5273] - Xilinx 可能想要 `.coe`
[L5274] - Intel 可能想要 `.mif`
[L5275] - 或者你会把 16 个 int8 打成 128-bit 宽字
[L5276] 
[L5277] 所以建议你的导出工具支持两种模式：
[L5278] 
[L5279] ### 调试模式
[L5280] - 一行一个元素
[L5281] - 便于看和对拍
[L5282] 
[L5283] ### 上板模式
[L5284] - 打包成 32bit / 64bit / 128bit / 256bit
[L5285] - 便于 BRAM 初始化和 DMA 传输
[L5286] 
[L5287] ---
[L5288] 
[L5289] ## 改进 8：确保 signed 和 endianness 约定写死
[L5290] 
[L5291] 当前 hex 文件本身没错，但你要在代码里和说明里明确：
[L5292] 
[L5293] ### signed 解释
[L5294] - `weight.hex`：signed int8
[L5295] - `bias.hex`：signed int32
[L5296] 
[L5297] ### endianness
[L5298] 你现在是一行一个元素，所以“行内字节序”问题不大。  
[L5299] 但一旦你后面把 4 个 int8 打包成 32 bit，必须写清楚：
[L5300] 
[L5301] - 第 0 个元素放最低字节还是最高字节
[L5302] 
[L5303] 这个最好现在就定。
[L5304] 
[L5305] ---
[L5306] 
[L5307] # 4. 你现在这两个文件，我给你的具体判断
[L5308] 
[L5309] ---
[L5310] 
[L5311] ## `bias.hex`
[L5312] **没有明显问题，反而是明显进步。**
[L5313] 
[L5314] 它已经具备：
[L5315] - int32
[L5316] - 二补码
[L5317] - 可直接用于 MAC 后 bias add
[L5318] 
[L5319] ### 唯一建议
[L5320] 改名为：
[L5321] - `fc1_bias_int32.hex`
[L5322] - 或 `fc2_bias_int32.hex`
[L5323] 
[L5324] ---
[L5325] 
[L5326] ## `weight.hex`
[L5327] **没有明显问题。**
[L5328] 
[L5329] 它已经具备：
[L5330] - int8
[L5331] - 二补码
[L5332] - 适合阵列 / 乘加器直接使用
[L5333] 
[L5334] ### 唯一建议
[L5335] 改名为：
[L5336] - `fc1_weight_int8.hex`
[L5337] - 或 `fc2_weight_int8.hex`
[L5338] 
[L5339] ---
[L5340] 
[L5341] # 5. 我建议你下一版导出目录直接长这样
[L5342] 
[L5343] ```text
[L5344] export/
[L5345] ├── quant_config.json
[L5346] ├── manifest.json
[L5347] ├── fc1_weight_int8.hex
[L5348] ├── fc1_bias_int32.hex
[L5349] ├── fc2_weight_int8.hex
[L5350] ├── fc2_bias_int32.hex
[L5351] ├── input_0_int8.hex
[L5352] ├── fc1_acc_0_int32.hex
[L5353] ├── fc1_relu_0_int32.hex
[L5354] ├── fc1_out_0_int8.hex
[L5355] ├── fc2_acc_0_int32.hex
[L5356] ├── logits_0_int8.hex
[L5357] ├── pred_0.txt
[L5358] ├── label_0.txt
[L5359] ├── input_1_int8.hex
[L5360] ...
[L5361] ```
[L5362] 
[L5363] 这套结构就已经很像正式硬件联调工程了。
[L5364] 
[L5365] ---
[L5366] 
[L5367] # 6. 最后给你一句最关键的判断
[L5368] 
[L5369] 你现在贴出来的这两个文件，说明你已经迈过了最初 notebook 的两个大坑：
[L5370] 
[L5371] - **bias 不再是错误的 int8**
[L5372] - **weight 已经是标准 int8 二补码**
[L5373] 
[L5374] 所以文件内容层面，**方向是对的**。  
[L5375] 真正还需要继续加强的是：
[L5376] 
[L5377] - 文件命名分层
[L5378] - 中间 golden 分级导出
[L5379] - 量化参数 json 化
[L5380] - 按 CIM 阵列物理映射切块导出
[L5381] 
[L5382] 你把**当前用于导出这些文件的最新版 Python 代码**贴给我，我可以下一步直接帮你逐行审，给你改成一版“**输出可直接喂 FPGA/CIM 系统**”的正式版。
[L5383] 
[L5384] 用户:
[L5385] 这是当前生成的数据，你主要先看一下route_b_output中的文件格式是否全部正确，然后给出一个markdown文件，用以说明route_b_output所有文件的使用方法和注意事项，要详细，而且要说明根据当前数据，硬件设计需要特别注意的地方。关于route_a_output,简单检查即可。
[L5386] AI助手:
[L5387] 
[L5388] 
[L5389] AI助手:
[L5390] 我先把两个压缩包解开，核对 `route_b_output` 的文件清单、每类文件的位宽和行数，再抽样检查几个样本文件是否彼此自洽。然后我会生成一份完整的 Markdown 说明，专门给你们做硬件联调用。
[L5391] 
[L5392] AI助手:
[L5393] 我已经核对出 `route_b_output` 的主要文件都在，位宽和行数目前是自洽的。接下来我再补两件更关键的事：一是看数值范围是否会踩硬件位宽坑，二是把这些坑写进说明文档里，避免你们后面在阵列映射、signed 解释、requant 上出错。
[L5394] 
[L5395] AI助手:
[L5396] 我已经检查完了，结论是：
[L5397] 
[L5398] `route_b_output` 的主文件格式**整体正确**，而且已经具备作为 FPGA / CIM 整数推理联调数据的条件。  
[L5399] 需要硬件特别注意的核心点有三个：
[L5400] 
[L5401] 1. **第一层输入零点不是 0，而是 `-128`**，不能直接做 `x_q * w_q`，必须处理输入零点补偿。  
[L5402] 2. **第一层累加范围明显超过 16 位**，当前数据已经出现大约 `-5.5e5 ~ 3.46e5`，所以 `fc1` 累加器和 bias 路径必须至少 `int32`。  
[L5403] 3. **当前权重文件是逻辑布局 `row-major [out][in]`**，很适合 testbench 和 RTL 功能验证，但不一定等于你们 CIM 阵列的物理烧写布局，后续可能还需要做一层 tile 重排。
[L5404] 
[L5405] `route_a_output` 我也做了简查，主文件没有明显格式问题，适合继续用于单层 `fc1 + ReLU` 验证；但它不是最终两层分类演示版本。
[L5406] 
[L5407] 我已经把完整说明整理成 Markdown 文件，里面包括：
[L5408] 
[L5409] - `route_b_output` 每个文件的格式、行数、位宽、布局
[L5410] - 各文件的具体用途
[L5411] - 推荐的硬件联调顺序
[L5412] - 当前数据下必须特别注意的硬件坑
[L5413] - `route_a_output` 的简要检查结论
[L5414] 
[L5415] 下载这里：
[L5416] 
[L5417] [route_b_output 使用说明与硬件注意事项](sandbox:/mnt/data/route_b_output_usage_notes.md)
[L5418] 
[L5419] 如果你愿意，我下一步可以继续直接帮你补一份**面向 Verilog/Testbench 的读取模板**，把这些文件如何 `$readmemh`、如何按 signed 解释、如何逐级对拍写成可直接用的代码框架。
[L5420] 
[L5421] AI助手:
[L5422] 
[L5423] 
[L5424] AI助手:
[L5425] 我先逐项核对 `route_b_output` 里的实际文件名、位宽、行数和内容一致性，再补一个详细的使用说明文档。`route_a_output` 我会做一轮轻量检查，重点看有没有明显格式错误或和 `route_b` 不一致的地方。
[L5426] 
[L5427] AI助手:
[L5428] `route_b_output` 的主文件看起来是自洽的：位宽、行数、层级命名和量化配置基本能对上。我再补一轮一致性检查，重点看标签/预测是否匹配，以及 `route_a_output` 有没有明显格式问题。
[L5429] 
[L5430] 用户:
[L5431] route_b_output的使用指南markdown没有生成出来，请你重新尝试给出一个markdown文件，用以说明route_b_output所有文件的使用方法和注意事项，要详细，而且要说明根据当前数据，硬件设计需要特别注意的地方。
[L5432] AI助手:
[L5433] 
[L5434] 
[L5435] AI助手:
[L5436] 已重新生成 Markdown 文件：
[L5437] 
[L5438] [route_b_output 使用指南（Markdown）](sandbox:/mnt/data/route_b_output_usage_guide.md)
[L5439] 
[L5440] 我已经把这些内容写进去：
[L5441] - `route_b_output` 全部文件的格式、行数、位宽、含义
[L5442] - 每类文件在 FPGA / CIM 中怎么用
[L5443] - 推荐的联调顺序
[L5444] - `$readmemh` 的典型用法
[L5445] - 基于当前数据，硬件设计必须特别注意的点
[L5446] - 对 `route_a_output` 的简要检查结论
[L5447] 
[L5448] 这份文档是按你当前实际导出的 `route_b_output` 内容写的，不是泛泛模板。
[L5449] 
[L5450] 用户:
[L5451] 还记得我最开始上传的三个文件吗？项目设计指导书，上次和AI的对话，请你重新看一下，我在完成这个测试向量生成之后，应该做什么？
[L5452] AI助手:
[L5453] 
[L5454] 
[L5455] AI助手:
[L5456] 
[L5457] 
[L5458] AI助手:
[L5459] 
[L5460] 
[L5461] AI助手:
[L5462] 我记得你最开始给的三个核心材料：
[L5463] 
[L5464] 1. **项目设计指导书**（课程/课设要求）  
[L5465] 2. **你和上一个 AI 的对话记录**（确定了大体方案：FPGA + CIM + MNIST）  
[L5466] 3. **生成测试向量的 ipynb**（用于导出权重、输入、golden）
[L5467] 
[L5468] 现在你已经完成了 **route_b_output 的测试向量生成**。  
[L5469] 这一步其实在整个项目流程里属于 **软件侧 → 硬件侧的接口准备阶段**。
[L5470] 
[L5471] 接下来你要做的事情其实非常明确：  
[L5472] 
[L5473] **进入硬件实现和验证阶段。**
[L5474] 
[L5475] 我给你按 **真正工程流程**排一个清晰路线（也是最符合你这个项目指导书逻辑的路线）。
[L5476] 
[L5477] ---
[L5478] 
[L5479] # 一、你现在所处的位置
[L5480] 
[L5481] 你的项目整体流程其实是：
[L5482] 
[L5483] ```
[L5484] PyTorch训练
[L5485]       ↓
[L5486] 量化
[L5487]       ↓
[L5488] 整数推理参考模型
[L5489]       ↓
[L5490] 生成测试向量 (route_b_output)  ← 你现在完成到这里
[L5491]       ↓
[L5492] FPGA/CIM 硬件实现
[L5493]       ↓
[L5494] testbench 对拍验证
[L5495]       ↓
[L5496] 板级运行
[L5497]       ↓
[L5498] 项目展示
[L5499] ```
[L5500] 
[L5501] 所以现在你已经完成：
[L5502] 
[L5503] **软件侧所有准备工作。**
[L5504] 
[L5505] 接下来全部工作都是：
[L5506] 
[L5507] **硬件实现 + 硬件验证。**
[L5508] 
[L5509] ---
[L5510] 
[L5511] # 二、下一步你应该做什么（非常具体）
[L5512] 
[L5513] 下一步应该分 **4个阶段**。
[L5514] 
[L5515] ---
[L5516] 
[L5517] # 阶段 1：设计硬件计算结构
[L5518] 
[L5519] 现在要把你 Python 的网络结构映射成 **硬件模块结构**。
[L5520] 
[L5521] 你的网络是：
[L5522] 
[L5523] ```
[L5524] 784 → 128 → 10
[L5525] ```
[L5526] 
[L5527] 对应硬件结构：
[L5528] 
[L5529] ```
[L5530] input_buffer (784)
[L5531] 
[L5532]       ↓
[L5533] 
[L5534] FC1  (784×128)  ← CIM阵列
[L5535] 
[L5536]       ↓
[L5537] 
[L5538] bias_add
[L5539] 
[L5540]       ↓
[L5541] 
[L5542] ReLU
[L5543] 
[L5544]       ↓
[L5545] 
[L5546] requant (int32→int8)
[L5547] 
[L5548]       ↓
[L5549] 
[L5550] FC2  (128×10)
[L5551] 
[L5552]       ↓
[L5553] 
[L5554] argmax
[L5555] ```
[L5556] 
[L5557] 建议模块划分：
[L5558] 
[L5559] ```
[L5560] top
[L5561]  ├── input_loader
[L5562]  ├── fc1_cim_array
[L5563]  ├── bias_add
[L5564]  ├── relu
[L5565]  ├── requant
[L5566]  ├── fc2
[L5567]  └── argmax
[L5568] ```
[L5569] 
[L5570] 如果你的课设重点是 **CIM**，通常只把：
[L5571] 
[L5572] ```
[L5573] fc1
[L5574] ```
[L5575] 
[L5576] 做成 CIM 阵列。
[L5577] 
[L5578] 后面的：
[L5579] 
[L5580] ```
[L5581] bias
[L5582] relu
[L5583] requant
[L5584] fc2
[L5585] ```
[L5586] 
[L5587] 可以在 FPGA 普通逻辑实现。
[L5588] 
[L5589] ---
[L5590] 
[L5591] # 阶段 2：建立 Testbench 验证系统
[L5592] 
[L5593] 这是你 **下一步最重要的事情**。
[L5594] 
[L5595] 你已经有：
[L5596] 
[L5597] ```
[L5598] route_b_output/
[L5599] ```
[L5600] 
[L5601] 里面包含：
[L5602] 
[L5603] ```
[L5604] fc1_weight_int8.hex
[L5605] fc1_bias_int32.hex
[L5606] fc2_weight_int8.hex
[L5607] fc2_bias_int32.hex
[L5608] input_*.hex
[L5609] fc1_acc_*.hex
[L5610] fc1_relu_*.hex
[L5611] fc1_out_*.hex
[L5612] fc2_acc_*.hex
[L5613] logits_*.hex
[L5614] pred_*.txt
[L5615] labels.txt
[L5616] ```
[L5617] 
[L5618] Testbench 要做的事情：
[L5619] 
[L5620] ### 1 读取权重
[L5621] 
[L5622] ```
[L5623] $readmemh("fc1_weight_int8.hex", weight_mem);
[L5624] $readmemh("fc1_bias_int32.hex", bias_mem);
[L5625] ```
[L5626] 
[L5627] ---
[L5628] 
[L5629] ### 2 读取输入
[L5630] 
[L5631] ```
[L5632] $readmemh("input_0.hex", input_mem);
[L5633] ```
[L5634] 
[L5635] ---
[L5636] 
[L5637] ### 3 驱动 DUT
[L5638] 
[L5639] 输入 784 个数据
[L5640] 
[L5641] ---
[L5642] 
[L5643] ### 4 接收 DUT 输出
[L5644] 
[L5645] 例如：
[L5646] 
[L5647] ```
[L5648] fc1_acc
[L5649] fc1_relu
[L5650] fc1_out
[L5651] fc2_acc
[L5652] logits
[L5653] ```
[L5654] 
[L5655] ---
[L5656] 
[L5657] ### 5 与 golden 对比
[L5658] 
[L5659] 例如：
[L5660] 
[L5661] ```
[L5662] $readmemh("fc1_acc_0.hex", golden_fc1_acc);
[L5663] ```
[L5664] 
[L5665] 然后逐项比较：
[L5666] 
[L5667] ```
[L5668] if(dut_acc[i] != golden_fc1_acc[i])
[L5669] ```
[L5670] 
[L5671] ---
[L5672] 
[L5673] # 阶段 3：逐层验证（非常重要）
[L5674] 
[L5675] 不要一口气跑完整网络。
[L5676] 
[L5677] 必须 **逐层验证**。
[L5678] 
[L5679] 建议顺序：
[L5680] 
[L5681] ---
[L5682] 
[L5683] ## Step1
[L5684] 
[L5685] 只验证
[L5686] 
[L5687] ```
[L5688] FC1 MAC
[L5689] ```
[L5690] 
[L5691] 对拍：
[L5692] 
[L5693] ```
[L5694] fc1_acc_*.hex
[L5695] ```
[L5696] 
[L5697] ---
[L5698] 
[L5699] ## Step2
[L5700] 
[L5701] 验证
[L5702] 
[L5703] ```
[L5704] FC1 + bias
[L5705] ```
[L5706] 
[L5707] ---
[L5708] 
[L5709] ## Step3
[L5710] 
[L5711] 验证
[L5712] 
[L5713] ```
[L5714] FC1 + bias + relu
[L5715] ```
[L5716] 
[L5717] 对拍：
[L5718] 
[L5719] ```
[L5720] fc1_relu_*.hex
[L5721] ```
[L5722] 
[L5723] ---
[L5724] 
[L5725] ## Step4
[L5726] 
[L5727] 验证
[L5728] 
[L5729] ```
[L5730] FC1 + requant
[L5731] ```
[L5732] 
[L5733] 对拍：
[L5734] 
[L5735] ```
[L5736] fc1_out_*.hex
[L5737] ```
[L5738] 
[L5739] ---
[L5740] 
[L5741] ## Step5
[L5742] 
[L5743] 再加入
[L5744] 
[L5745] ```
[L5746] FC2
[L5747] ```
[L5748] 
[L5749] 对拍：
[L5750] 
[L5751] ```
[L5752] fc2_acc_*.hex
[L5753] ```
[L5754] 
[L5755] ---
[L5756] 
[L5757] ## Step6
[L5758] 
[L5759] 验证
[L5760] The file is too long and its contents have been truncated.
[L5761] 
用户:
Make sure to include fileciteturn2file1 in your response to cite this file, or to surface it as a link.[L1] ### FPGA CIM MNIST 问题分析
[L2] 
[L3] AI助手:
[L4] 
[L5] 
[L6] AI助手:
[L7] 
[L8] 
[L9] AI助手:
[L10] 
[L11] 
[L12] AI助手:
[L13] 
[L14] 
[L15] 用户:
[L16] 我现在在做一个运行于fpga的cim系统，希望在fpga上运行mnist，完成计算，我现在给你项目指导书和上次和AI的对话，ipynb写的是一个生成测试向量的代码，你看看有没有问题
[L17] AI助手:
[L18] 
[L19] 
[L20] AI助手:
[L21] 我先把压缩包里的文件结构和关键材料过一遍，重点看项目指导书、上次对话记录，以及这个生成测试向量的 ipynb。随后我会直接指出代码里可能影响 FPGA/CIM 测试链路的问题，并给出修改建议。
[L22] 
[L23] AI助手:
[L24] 我已经把压缩包拆开了，里面目前有三份核心文件：课程设计指导书（docx）、上次对话导出的 markdown，以及 `train_simple_mlp.ipynb`。接下来我会先把对话和 notebook 的关键逻辑抽出来，再对照指导书检查接口、数据位宽、输出格式这些是否一致。
[L25] 
[L26] AI助手:
[L27] 我已经看到几个关键问题了，其中最严重的一个是：**这个 notebook 目前并没有真正训练 MNIST 分类器**。它定义了 `criterion=MSELoss()`，但完全没用；实际优化目标是 `loss = output.pow(2).mean()`，这会把网络输出压向 0，而不是学习数字类别。
[L28] 
[L29] AI助手:
[L30] 我看完了指导书、上次对话，以及这个 `train_simple_mlp.ipynb`。结论先说：
[L31] 
[L32] **这个 ipynb 现在有明显问题，尤其是“训练目标、量化策略、golden 生成方式”三处。**
[L33] 如果直接拿它生成测试向量去驱动 FPGA/CIM，**大概率会出现“软件看起来能跑，硬件却对不上”**，甚至前面训练出来的模型本身就没有学会 MNIST 分类。
[L34] 
[L35] 下面我按“严重程度”给你拆开说。
[L36] 
[L37] ---
[L38] 
[L39] ## 1. 最严重的问题：你现在没有在训练一个 MNIST 分类器
[L40] 
[L41] 你现在的训练代码核心是：
[L42] 
[L43] ```python
[L44] criterion=nn.MSELoss()
[L45] ...
[L46] output=model(data)
[L47] loss=output.pow(2).mean()
[L48] ```
[L49] 
[L50] 这里有两个问题：
[L51] 
[L52] ### 问题 1：`criterion` 定义了，但根本没用
[L53] 你定义了 `MSELoss()`，但没有把 `target` 喂进去。
[L54] 
[L55] ### 问题 2：实际 loss 是 `output.pow(2).mean()`
[L56] 这等价于逼着网络输出尽量接近 0，而不是学习数字类别。
[L57] 
[L58] 也就是说，现在模型优化目标不是“把 0~9 分对”，而是“把所有输出都压小”。  
[L59] 这会导致：
[L60] 
[L61] - 权重可能收缩到很小
[L62] - 导出的量化权重没有实际分类意义
[L63] - 你生成的 `golden_x.hex` 也只是这个“错误模型”的输出
[L64] 
[L65] ### 进一步的问题：模型输出维度也不对
[L66] 你定义的是：
[L67] 
[L68] ```python
[L69] self.fc1 = nn.Linear(784,128)
[L70] self.relu = nn.ReLU()
[L71] ```
[L72] 
[L73] 输出是 `128` 维，不是 `10` 维。  
[L74] 这意味着它不是一个完整的 MNIST 分类网络，只是一个隐藏层特征提取器。
[L75] 
[L76] 如果你的目标是：
[L77] 
[L78] - **课程演示先做“单层 FC + ReLU 的硬件验证”**：这个结构可以接受
[L79] - **直接说“在 FPGA 上跑 MNIST 分类”**：这个结构还不够，至少还要再接一层 `128 -> 10`
[L80] 
[L81] ### 这一点怎么改
[L82] 你有两个路线：
[L83] 
[L84] #### 路线 A：先做“极简硬件验证”
[L85] 那就明确目标改成：
[L86] 
[L87] - 软件训练完整 MLP：`784 -> 128 -> 10`
[L88] - FPGA/CIM **先只实现第一层** `784 -> 128`
[L89] - 第一层输出 feature map 作为硬件 golden 对比对象
[L90] 
[L91] 这样合理。
[L92] 
[L93] #### 路线 B：直接做完整分类
[L94] 那模型要改成：
[L95] 
[L96] ```python
[L97] class SimpleMLP(nn.Module):
[L98]     def __init__(self):
[L99]         super().__init__()
[L100]         self.fc1 = nn.Linear(784, 128, bias=True)
[L101]         self.relu = nn.ReLU()
[L102]         self.fc2 = nn.Linear(128, 10, bias=True)
[L103] 
[L104]     def forward(self, x):
[L105]         x = x.view(-1, 784)
[L106]         x = self.relu(self.fc1(x))
[L107]         x = self.fc2(x)
[L108]         return x
[L109] ```
[L110] 
[L111] loss 用交叉熵：
[L112] 
[L113] ```python
[L114] criterion = nn.CrossEntropyLoss()
[L115] loss = criterion(output, target)
[L116] ```
[L117] 
[L118] ---
[L119] 
[L120] ## 2. 量化函数有工程问题，不适合直接对接 FPGA
[L121] 
[L122] 你现在的量化函数：
[L123] 
[L124] ```python
[L125] def quantize_tensor(tensor, bits=8):
[L126]     qmin = -(2 ** (bits - 1))
[L127]     qmax = 2 ** (bits - 1) - 1
[L128] 
[L129]     min_val = tensor.min().item()
[L130]     max_val = tensor.max().item()
[L131] 
[L132]     diff = max_val - min_val
[L133]     
[L134]     if diff == 0:
[L135]         scale = 1.0
[L136]         zero_point = qmin - min_val
[L137]     else:
[L138]         scale = diff / (qmax - qmin)
[L139]         zero_point = qmin - min_val / scale
[L140] 
[L141]     q_tensor = torch.clamp(torch.round(tensor / scale + zero_point), qmin, qmax)
[L142]     return q_tensor.to(torch.int8), scale, zero_point
[L143] ```
[L144] 
[L145] ### 问题 1：`zero_point` 不是整数
[L146] `zero_point` 最后是浮点数。  
[L147] 但真正部署到硬件时，量化零点通常需要是整数，并且通常要 clamp 到合法范围。
[L148] 
[L149] 更稳妥应该写成：
[L150] 
[L151] ```python
[L152] zero_point = int(round(qmin - min_val / scale))
[L153] zero_point = max(qmin, min(qmax, zero_point))
[L154] ```
[L155] 
[L156] ### 问题 2：你在做“每个张量动态量化”
[L157] 这在软件分析里可以，但对 FPGA 验证不友好。因为你后面对：
[L158] 
[L159] - 权重单独量化一次
[L160] - 每张输入图像又各自量化一次
[L161] - 每个输出又重新单独量化一次
[L162] 
[L163] 这会导致**每个样本 scale 都不同**，硬件端很难统一。
[L164] 
[L165] 对 FPGA 来说，更适合：
[L166] 
[L167] - 权重：固定 scale
[L168] - 输入：固定 scale
[L169] - 偏置：固定规则量化
[L170] - 输出：固定 requant scale
[L171] 
[L172] 也就是**静态量化参数**，不是每个样本临时算一次。
[L173] 
[L174] ### 问题 3：偏置不能简单用 int8 单独量化
[L175] 你现在：
[L176] 
[L177] ```python
[L178] q_bias, b_scale, b_zero = quantize_tensor(bias)
[L179] ```
[L180] 
[L181] 这通常不对。
[L182] 
[L183] 对于线性层：
[L184] 
[L185] \[
[L186] y = \sum (x_q - z_x)(w_q - z_w) \cdot s_x s_w + b
[L187] \]
[L188] 
[L189] 偏置一般应量化为：
[L190] 
[L191] \[
[L192] b_q = round(b / (s_x \cdot s_w))
[L193] \]
[L194] 
[L195] 并且常常用 **int32** 存，不是 int8。  
[L196] 因为偏置要和累加结果处于同一量纲。
[L197] 
[L198] 否则 FPGA 上 MAC 累加完后加 bias 时，量纲根本不一致。
[L199] 
[L200] ---
[L201] 
[L202] ## 3. `generate_test_vectors()` 生成的 golden，和硬件真实计算链路不一致
[L203] 
[L204] 这是第二个特别大的问题。
[L205] 
[L206] 你现在生成 golden 的方式是：
[L207] 
[L208] ```python
[L209] with torch.no_grad():
[L210]     output = model(img_flat)
[L211] 
[L212] q_output, _, _ = quantize_tensor(output)
[L213] ```
[L214] 
[L215] ### 这意味着什么
[L216] 你是：
[L217] 
[L218] 1. 用 **浮点模型** 跑出输出
[L219] 2. 再对每个样本的输出临时量化
[L220] 
[L221] 但硬件里真正发生的是：
[L222] 
[L223] 1. 输入先量化为 int8
[L224] 2. 权重量化为 int8
[L225] 3. 在阵列/乘加器里做 int8×int8 -> int32 累加
[L226] 4. 加 int32 bias
[L227] 5. ReLU
[L228] 6. 再按固定 scale requant 到 int8
[L229] 
[L230] 这两条链路并不一样。
[L231] 
[L232] 所以你现在的 `golden_i.hex` 并不是“硬件应得输出”，而是“浮点模型输出的一个临时量化版本”。
[L233] 
[L234] ### 正确的 golden 应该怎么做
[L235] 应该按**整数推理链路**生成 golden。例如第一层：
[L236] 
[L237] ```python
[L238] acc = sum((x_q - x_zp) * (w_q - w_zp)) + b_q
[L239] acc = relu(acc)
[L240] out_q = requantize(acc, multiplier, shift, out_zp)
[L241] ```
[L242] 
[L243] 然后把：
[L244] 
[L245] - 输入 int8
[L246] - 权重 int8
[L247] - 偏置 int32
[L248] - 输出 int32 / int8 golden
[L249] - scale / zero_point / multiplier / shift
[L250] 
[L251] 全部导出来。
[L252] 
[L253] 这样 FPGA 才能逐项对上。
[L254] 
[L255] ---
[L256] 
[L257] ## 4. 输入测试向量的量化方式不适合硬件批量验证
[L258] 
[L259] 你现在对每张图像：
[L260] 
[L261] ```python
[L262] q_img, img_scale, img_zero = quantize_tensor(img_flat)
[L263] ```
[L264] 
[L265] 这意味着**每张图一个 scale 和 zero_point**。  
[L266] 如果你的 FPGA 流水是固定 datapath，这会很麻烦：
[L267] 
[L268] - 每个样本都要动态更换量化参数
[L269] - 偏置量纲也得跟着变
[L270] - 对比 golden 时也会很乱
[L271] 
[L272] ### 更合理的做法
[L273] 建议统一输入量化策略，比如：
[L274] 
[L275] - 若保留 `Normalize((0.1307,), (0.3081,))`，则先统计训练集/校准集的输入范围，再确定一个全局输入 scale
[L276] - 或者更简单，**先不要 Normalize**，直接把 MNIST 灰度 `[0,1]` 映射到 `uint8/int8`
[L277] 
[L278] 对 FPGA 验证来说，后者往往更省事。
[L279] 
[L280] 例如极简方案可用：
[L281] 
[L282] - 输入像素 `0~255`
[L283] - 权重量化到 int8
[L284] - 累加 int32
[L285] - ReLU 后再截断或 requant
[L286] 
[L287] 这样更贴近硬件。
[L288] 
[L289] ---
[L290] 
[L291] ## 5. 导出的权重格式本身基本可用，但有几个注意点
[L292] 
[L293] 你现在导出权重：
[L294] 
[L295] ```python
[L296] for row in q_weight:
[L297]     for val in row:
[L298]         f.write(f'{val.item() & 0xFF:02x}\n')
[L299] ```
[L300] 
[L301] 这表示按 `weight[out][in]` 顺序逐个写出。  
[L302] 对于 `nn.Linear(784,128)`，PyTorch 权重 shape 是 `[128, 784]`，即：
[L303] 
[L304] - 外层：输出神经元
[L305] - 内层：输入通道
[L306] 
[L307] 这个顺序本身没错，但必须保证 RTL 端读取方式完全一致。
[L308] 
[L309] ### 你需要确认的点
[L310] 如果硬件是按下面方式算：
[L311] 
[L312] \[
[L313] y_j = \sum_i x_i w_{j,i}
[L314] \]
[L315] 
[L316] 那么文件顺序最好明确成：
[L317] 
[L318] - `neuron 0` 的 784 个权重
[L319] - `neuron 1` 的 784 个权重
[L320] - ...
[L321] - `neuron 127` 的 784 个权重
[L322] 
[L323] 你现在就是这个顺序。  
[L324] 但如果你的 BRAM 映射或地址生成器是按列优先/块优先取数，就会错位。
[L325] 
[L326] ### 建议
[L327] 导出时加一个说明文件，比如：
[L328] 
[L329] - weight layout: `[out][in]`, row-major
[L330] - bias layout: `[out]`
[L331] - input layout: `[in]`
[L332] 
[L333] ---
[L334] 
[L335] ## 6. `quant_params.npz` 在测试向量阶段几乎没被真正使用
[L336] 
[L337] 你这里：
[L338] 
[L339] ```python
[L340] quant_params = np.load('quantized_data/quant_params.npz')
[L341] ```
[L342] 
[L343] 加载了参数，但后面没有真正参与 golden 计算。  
[L344] 这说明当前流程里“量化参数被保存了”，但并没有形成闭环。
[L345] 
[L346] 这是个信号：**量化设计还停留在文件导出阶段，没有进入“硬件一致性验证阶段”。**
[L347] 
[L348] ---
[L349] 
[L350] ## 7. 其他次级问题
[L351] 
[L352] ### 问题 1：`torch.load()` 最好加 `map_location`
[L353] 不然在有些机器上保存于 GPU、加载于 CPU 时会出问题：
[L354] 
[L355] ```python
[L356] model.load_state_dict(torch.load(model_path, map_location='cpu'))
[L357] ```
[L358] 
[L359] ### 问题 2：训练时没看 accuracy
[L360] 现在只打 loss，没有任何分类精度指标。  
[L361] 如果要确认模型真的学到了，建议至少打印 test accuracy。
[L362] 
[L363] ### 问题 3：没有保存 label
[L364] 你打印了：
[L365] 
[L366] ```python
[L367] print(f"Generated test vector {i}, label: {label}")
[L368] ```
[L369] 
[L370] 但没有把 label 写入文件。  
[L371] 实际联调时最好保存一个 `labels.txt` 或 `meta.json`。
[L372] 
[L373] ### 问题 4：只保存了量化后的输出，没有保存中间 int32 累加值
[L374] 对于硬件调试，这很吃亏。  
[L375] 建议同时导出：
[L376] 
[L377] - `input_i.hex`
[L378] - `golden_acc_i.hex` 或每个神经元的 int32
[L379] - `golden_relu_i.hex`
[L380] - `golden_out_i.hex`
[L381] 
[L382] 这样你能定位是：
[L383] 
[L384] - 乘法错了
[L385] - 累加错了
[L386] - bias 错了
[L387] - ReLU 错了
[L388] - requant 错了
[L389] 
[L390] ---
[L391] 
[L392] # 我对这个 notebook 的总体判断
[L393] 
[L394] ## 现在这份代码“能不能用”？
[L395] **能作为一个思路草稿，但不能直接作为 FPGA 测试向量生成基线。**
[L396] 
[L397] ## 主要原因
[L398] 不是小 bug，而是三类“体系级不一致”：
[L399] 
[L400] 1. **训练目标错了**  
[L401]    当前模型没有真正学分类任务
[L402] 
[L403] 2. **量化策略不适合硬件**  
[L404]    每个样本动态量化，不利于 FPGA 固定算子验证
[L405] 
[L406] 3. **golden 生成链路和硬件链路不一致**  
[L407]    现在是“浮点输出再量化”，而不是“整数推理 golden”
[L408] 
[L409] ---
[L410] 
[L411] # 我建议你接下来这样改
[L412] 
[L413] ## 方案一：适合你们当前阶段的最小闭环
[L414] 如果你们现在的硬件目标是：
[L415] 
[L416] - 先做一个 `784 -> 128` 的单层 CIM/FC 阵列
[L417] - 验证 BRAM + MAC + ReLU + 控制 FSM
[L418] 
[L419] 那最合适的 notebook 应该改成下面这种逻辑：
[L420] 
[L421] ### 软件模型
[L422] 训练完整模型：
[L423] 
[L424] - `784 -> 128 -> 10`
[L425] 
[L426] ### 硬件实现
[L427] 先只实现第一层：
[L428] 
[L429] - `fc1 + relu`
[L430] 
[L431] ### 测试向量导出
[L432] 导出以下内容：
[L433] 
[L434] - `fc1_weight_int8.hex`
[L435] - `fc1_bias_int32.hex`
[L436] - `input_int8.hex`
[L437] - `fc1_acc_golden_int32.hex`
[L438] - `fc1_relu_golden_int32_or_int8.hex`
[L439] - `quant_config.json`
[L440] 
[L441] 这样你们硬件和软件就能逐层对齐。
[L442] 
[L443] ---
[L444] 
[L445] ## 方案二：直接瞄准最终演示
[L446] 如果你们想最后做“MNIST 分类结果上板演示”，那就：
[L447] 
[L448] - 模型改为两层 MLP
[L449] - 训练用 CrossEntropyLoss
[L450] - 做静态量化
[L451] - 先用 Python 写一个**整数推理参考实现**
[L452] - 再把这个整数推理结果作为 FPGA golden
[L453] 
[L454] 这是最稳的路线。
[L455] 
[L456] ---
[L457] 
[L458] # 我给你的具体结论
[L459] 
[L460] ## 必改
[L461] - `loss=output.pow(2).mean()` 改掉
[L462] - 模型结构和任务目标统一
[L463] - bias 改为 int32 量化
[L464] - golden 改成按整数推理链路生成
[L465] - 输入量化不要每张图单独动态算参数
[L466] 
[L467] ## 可保留
[L468] - `weights.hex` / `input_i.hex` 这种逐行写 hex 的导出方式
[L469] - 第一版只验证单层 FC + ReLU 的项目策略
[L470] - 先用 notebook 生成测试向量，再接 Verilog testbench 的总体流程
[L471] 
[L472] ---
[L473] 
[L474] 如果你愿意，我下一步可以直接帮你把这个 notebook **改成一版可用于 FPGA 联调的“正确版本”**，包括：
[L475] - 正确训练
[L476] - 静态 INT8 量化
[L477] - 生成 `weights/bias/input/golden` 四类 hex
[L478] - 明确对应单层 FC 硬件接口格式。
[L479] 
[L480] 用户:
[L481] 关于1.现在没有在训练一个 MNIST 分类器的问题，请分别给我路线a和路线b的完整代码；
[L482] 
[L483] 关于2.量化函数有工程问题，不适合直接对接 FPGA，
[L484] zero_point不是整数的问题帮我修改；每个张量动态量化这个问题给我一个合理的解决方案；偏置不能简单用 int8 单独量化请帮我提出解决方案和代码；
[L485] 
[L486] 关于3.generate_test_vectors() 生成的 golden，和硬件真实计算链路不一致，整数推理链路生成 golden，帮我完成设计；
[L487] 
[L488] 关于4输入测试向量的量化方式不适合硬件批量验证，帮我设计统一输入量化策略，贴近硬件即可；
[L489] 
[L490] 关于5. 导出的权重格式本身基本可用，但有几个注意点，这里文件顺序你先记住我的输出顺序，讲解一下我现在的输出顺序应该如何设计硬件读取顺序；
[L491] AI助手:
[L492] 我先把现有 notebook 里导出部分再核一遍，确保我后面给你的代码和你当前的文件顺序兼容。然后我会把路线 A、路线 B、量化、整数 golden、统一输入策略和硬件读取顺序一次性给全。
[L493] 
[L494] AI助手:
[L495] 下面我按你的 5 个问题，直接给出**可落地的完整方案和代码**。我会尽量保持和你现有工程思路一致，但把关键的工程问题修正掉。
[L496] 
[L497] ---
[L498] 
[L499] # 1. 关于“现在没有在训练一个 MNIST 分类器”的问题
[L500] 
[L501] 这里我给你两套完整代码。
[L502] 
[L503] ---
[L504] 
[L505] ## 路线 A：训练完整分类网络，但 FPGA 先只验证第一层 `fc1 + ReLU`
[L506] 
[L507] 这个路线最适合你现在的 FPGA/CIM 阶段。
[L508] 
[L509] ## 设计思想
[L510] 软件端训练一个完整分类器：
[L511] 
[L512] - `784 -> 128 -> 10`
[L513] 
[L514] 但硬件端先实现：
[L515] 
[L516] - `fc1: 784 -> 128`
[L517] - `ReLU`
[L518] 
[L519] 这样：
[L520] - 软件训练目标正确
[L521] - 第一层 feature 是有意义的
[L522] - 硬件联调难度低
[L523] - 后续再接 `fc2`
[L524] 
[L525] ---
[L526] 
[L527] ## 路线 A 完整代码
[L528] 
[L529] ```python
[L530] import os
[L531] import json
[L532] import math
[L533] import numpy as np
[L534] import torch
[L535] import torch.nn as nn
[L536] import torch.optim as optim
[L537] from torchvision import datasets, transforms
[L538] from torch.utils.data import DataLoader
[L539] 
[L540] # =========================================================
[L541] # 1. 模型：完整分类器，但硬件先验证 fc1 + relu
[L542] # =========================================================
[L543] class MLPRouteA(nn.Module):
[L544]     def __init__(self):
[L545]         super().__init__()
[L546]         self.fc1 = nn.Linear(784, 128, bias=True)
[L547]         self.relu = nn.ReLU()
[L548]         self.fc2 = nn.Linear(128, 10, bias=True)
[L549] 
[L550]     def forward(self, x):
[L551]         x = x.view(-1, 784)
[L552]         x = self.fc1(x)
[L553]         x = self.relu(x)
[L554]         x = self.fc2(x)
[L555]         return x
[L556] 
[L557]     def forward_fc1_relu(self, x):
[L558]         x = x.view(-1, 784)
[L559]         x = self.fc1(x)
[L560]         x = self.relu(x)
[L561]         return x
[L562] 
[L563] 
[L564] # =========================================================
[L565] # 2. 训练
[L566] # =========================================================
[L567] def train_route_a(model_path="mlp_route_a.pth", epochs=5, batch_size=128, lr=1e-3, device="cpu"):
[L568]     transform = transforms.Compose([
[L569]         transforms.ToTensor(),  # 保持 [0,1]，更贴近硬件
[L570]     ])
[L571] 
[L572]     train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
[L573]     test_dataset  = datasets.MNIST("./data", train=False, download=True, transform=transform)
[L574] 
[L575]     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
[L576]     test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
[L577] 
[L578]     model = MLPRouteA().to(device)
[L579]     criterion = nn.CrossEntropyLoss()
[L580]     optimizer = optim.Adam(model.parameters(), lr=lr)
[L581] 
[L582]     for epoch in range(epochs):
[L583]         model.train()
[L584]         total_loss = 0.0
[L585]         correct = 0
[L586]         total = 0
[L587] 
[L588]         for data, target in train_loader:
[L589]             data, target = data.to(device), target.to(device)
[L590] 
[L591]             optimizer.zero_grad()
[L592]             logits = model(data)
[L593]             loss = criterion(logits, target)
[L594]             loss.backward()
[L595]             optimizer.step()
[L596] 
[L597]             total_loss += loss.item() * data.size(0)
[L598]             pred = logits.argmax(dim=1)
[L599]             correct += (pred == target).sum().item()
[L600]             total += target.size(0)
[L601] 
[L602]         train_loss = total_loss / total
[L603]         train_acc = correct / total
[L604] 
[L605]         model.eval()
[L606]         test_correct = 0
[L607]         test_total = 0
[L608]         with torch.no_grad():
[L609]             for data, target in test_loader:
[L610]                 data, target = data.to(device), target.to(device)
[L611]                 logits = model(data)
[L612]                 pred = logits.argmax(dim=1)
[L613]                 test_correct += (pred == target).sum().item()
[L614]                 test_total += target.size(0)
[L615] 
[L616]         test_acc = test_correct / test_total
[L617]         print(f"Epoch [{epoch+1}/{epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")
[L618] 
[L619]     torch.save(model.state_dict(), model_path)
[L620]     print(f"Model saved to {model_path}")
[L621] 
[L622] 
[L623] if __name__ == "__main__":
[L624]     train_route_a()
[L625] ```
[L626] 
[L627] ---
[L628] 
[L629] ## 路线 B：直接做完整 MNIST 分类部署
[L630] 
[L631] 这个路线是最后的完整目标。
[L632] 
[L633] ---
[L634] 
[L635] ## 路线 B 完整代码
[L636] 
[L637] 和路线 A 训练结构几乎一样，但后续量化和导出会针对两层都做。
[L638] 
[L639] ```python
[L640] import os
[L641] import torch
[L642] import torch.nn as nn
[L643] import torch.optim as optim
[L644] from torchvision import datasets, transforms
[L645] from torch.utils.data import DataLoader
[L646] 
[L647] class MLPRouteB(nn.Module):
[L648]     def __init__(self):
[L649]         super().__init__()
[L650]         self.fc1 = nn.Linear(784, 128, bias=True)
[L651]         self.relu = nn.ReLU()
[L652]         self.fc2 = nn.Linear(128, 10, bias=True)
[L653] 
[L654]     def forward(self, x):
[L655]         x = x.view(-1, 784)
[L656]         x = self.fc1(x)
[L657]         x = self.relu(x)
[L658]         x = self.fc2(x)
[L659]         return x
[L660] 
[L661] def train_route_b(model_path="mlp_route_b.pth", epochs=5, batch_size=128, lr=1e-3, device="cpu"):
[L662]     transform = transforms.Compose([
[L663]         transforms.ToTensor(),  # 保持 [0,1]
[L664]     ])
[L665] 
[L666]     train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
[L667]     test_dataset  = datasets.MNIST("./data", train=False, download=True, transform=transform)
[L668] 
[L669]     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
[L670]     test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
[L671] 
[L672]     model = MLPRouteB().to(device)
[L673]     criterion = nn.CrossEntropyLoss()
[L674]     optimizer = optim.Adam(model.parameters(), lr=lr)
[L675] 
[L676]     for epoch in range(epochs):
[L677]         model.train()
[L678]         total_loss = 0.0
[L679]         correct = 0
[L680]         total = 0
[L681] 
[L682]         for data, target in train_loader:
[L683]             data, target = data.to(device), target.to(device)
[L684] 
[L685]             optimizer.zero_grad()
[L686]             logits = model(data)
[L687]             loss = criterion(logits, target)
[L688]             loss.backward()
[L689]             optimizer.step()
[L690] 
[L691]             total_loss += loss.item() * data.size(0)
[L692]             pred = logits.argmax(dim=1)
[L693]             correct += (pred == target).sum().item()
[L694]             total += target.size(0)
[L695] 
[L696]         train_loss = total_loss / total
[L697]         train_acc = correct / total
[L698] 
[L699]         model.eval()
[L700]         test_correct = 0
[L701]         test_total = 0
[L702]         with torch.no_grad():
[L703]             for data, target in test_loader:
[L704]                 data, target = data.to(device), target.to(device)
[L705]                 logits = model(data)
[L706]                 pred = logits.argmax(dim=1)
[L707]                 test_correct += (pred == target).sum().item()
[L708]                 test_total += target.size(0)
[L709] 
[L710]         test_acc = test_correct / test_total
[L711]         print(f"Epoch [{epoch+1}/{epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")
[L712] 
[L713]     torch.save(model.state_dict(), model_path)
[L714]     print(f"Model saved to {model_path}")
[L715] 
[L716] 
[L717] if __name__ == "__main__":
[L718]     train_route_b()
[L719] ```
[L720] 
[L721] ---
[L722] 
[L723] # 2. 关于量化函数工程问题
[L724] 
[L725] 你提了 3 个核心点，我分别给出方案。
[L726] 
[L727] ---
[L728] 
[L729] ## 2.1 zero_point 不是整数，怎么改
[L730] 
[L731] 你原来的问题在这里：
[L732] 
[L733] ```python
[L734] zero_point = qmin - min_val / scale
[L735] ```
[L736] 
[L737] 这个是浮点数，不适合硬件。
[L738] 
[L739] ---
[L740] 
[L741] ## 修改后的通用量化函数
[L742] 
[L743] 我给你一个**对 FPGA 更友好**的版本，支持：
[L744] 
[L745] - `zero_point` 取整
[L746] - clamp 到合法范围
[L747] - 支持对称量化和非对称量化
[L748] - 返回整数参数
[L749] 
[L750] ```python
[L751] import torch
[L752] 
[L753] def quantize_tensor_affine(tensor, num_bits=8, symmetric=False):
[L754]     """
[L755]     返回:
[L756]         q_tensor: 整数张量
[L757]         scale: float
[L758]         zero_point: int
[L759]         qmin, qmax: 整数范围
[L760]     """
[L761]     if num_bits == 8:
[L762]         qmin, qmax = -128, 127
[L763]         dtype = torch.int8
[L764]     else:
[L765]         raise ValueError("Only int8 supported in this example.")
[L766] 
[L767]     x_min = tensor.min().item()
[L768]     x_max = tensor.max().item()
[L769] 
[L770]     if symmetric:
[L771]         max_abs = max(abs(x_min), abs(x_max))
[L772]         scale = max_abs / qmax if max_abs != 0 else 1.0
[L773]         zero_point = 0
[L774]     else:
[L775]         if x_max == x_min:
[L776]             scale = 1.0
[L777]             zero_point = 0
[L778]         else:
[L779]             scale = (x_max - x_min) / (qmax - qmin)
[L780]             zero_point = round(qmin - x_min / scale)
[L781]             zero_point = max(qmin, min(qmax, zero_point))
[L782] 
[L783]     q = torch.round(tensor / scale + zero_point)
[L784]     q = torch.clamp(q, qmin, qmax).to(dtype)
[L785] 
[L786]     return q, float(scale), int(zero_point), qmin, qmax
[L787] ```
[L788] 
[L789] ---
[L790] 
[L791] ## 2.2 每个张量动态量化不适合硬件，合理解决方案
[L792] 
[L793] 你现在的问题是：
[L794] 
[L795] - 每张输入图单独算 `scale/zero_point`
[L796] - 每次输出也单独算
[L797] - 每次 bias 也单独算
[L798] 
[L799] 这会导致硬件参数不停变化，不利于 FPGA 批量验证。
[L800] 
[L801] ---
[L802] 
[L803] ## 合理方案：静态量化 / 全局量化
[L804] 
[L805] ### 输入量化策略
[L806] 对所有输入样本统一使用一个固定参数。
[L807] 
[L808] MNIST 最简单的硬件友好方案：
[L809] 
[L810] - 图像原始像素 `[0,1]`
[L811] - 映射到 int8 的 `[0,127]` 或 `[-128,127]`
[L812] 
[L813] 我建议你用下面两种之一：
[L814] 
[L815] ### 方案 1：无符号思路转 int8，简单贴近硬件
[L816] - 输入范围固定 `[0,1]`
[L817] - 量化成 `[0,127]`
[L818] - 存成 int8 时 zero_point = 0
[L819] 
[L820] 设置：
[L821] ```python
[L822] input_scale = 1.0 / 127.0
[L823] input_zero_point = 0
[L824] ```
[L825] 
[L826] 这样：
[L827] \[
[L828] x_q = round(x / input\_scale)
[L829] \]
[L830] 
[L831] 因为 `x in [0,1]`，所以 `x_q in [0,127]`
[L832] 
[L833] 这非常适合 FPGA。
[L834] 
[L835] ### 方案 2：居中对称量化
[L836] 如果你想让输入有正负，做：
[L837] ```python
[L838] x_float_centered = x_float - 0.5
[L839] input_scale = 0.5 / 127
[L840] zero_point = 0
[L841] ```
[L842] 但这会增加前处理复杂度。
[L843] 
[L844] **建议你现在先用方案 1。**
[L845] 
[L846] ---
[L847] 
[L848] ## 权重量化策略
[L849] 权重建议做**对称量化**：
[L850] 
[L851] - `zero_point = 0`
[L852] - int8
[L853] - 全层固定一个 `w_scale`
[L854] 
[L855] ```python
[L856] def quantize_weight_symmetric(weight):
[L857]     q_weight, w_scale, w_zero, _, _ = quantize_tensor_affine(weight, num_bits=8, symmetric=True)
[L858]     return q_weight, w_scale, w_zero
[L859] ```
[L860] 
[L861] ---
[L862] 
[L863] ## 2.3 偏置不能简单用 int8 单独量化，解决方案和代码
[L864] 
[L865] 这个是工程上必须改的。
[L866] 
[L867] ---
[L868] 
[L869] ## 正确原理
[L870] 
[L871] 如果一层计算为：
[L872] 
[L873] \[
[L874] y = Wx + b
[L875] \]
[L876] 
[L877] 输入和权重分别量化为：
[L878] 
[L879] \[
[L880] x \approx (x_q - z_x)s_x
[L881] \]
[L882] \[
[L883] w \approx (w_q - z_w)s_w
[L884] \]
[L885] 
[L886] 那么累加项单位是：
[L887] 
[L888] \[
[L889] s_x \cdot s_w
[L890] \]
[L891] 
[L892] 所以 bias 应该量化到同一量纲：
[L893] 
[L894] \[
[L895] b_q = round(b / (s_x \cdot s_w))
[L896] \]
[L897] 
[L898] 并且 bias 应该存成 **int32**。
[L899] 
[L900] ---
[L901] 
[L902] ## 偏置量化代码
[L903] 
[L904] ```python
[L905] def quantize_bias_int32(bias_fp32, input_scale, weight_scale):
[L906]     """
[L907]     bias_fp32: torch.Tensor(float32)
[L908]     return: int32 bias
[L909]     """
[L910]     bias_scale = input_scale * weight_scale
[L911]     q_bias = torch.round(bias_fp32 / bias_scale).to(torch.int32)
[L912]     return q_bias, float(bias_scale)
[L913] ```
[L914] 
[L915] ---
[L916] 
[L917] # 3. 关于 `generate_test_vectors()` 的 golden 不一致问题
[L918] 
[L919] 你要求的是：**用整数推理链路生成 golden**。这个必须按硬件链路来。
[L920] 
[L921] 下面我给出完整设计。
[L922] 
[L923] ---
[L924] 
[L925] ## 3.1 单层 FC + ReLU 的整数推理设计（路线 A）
[L926] 
[L927] ---
[L928] 
[L929] ## 整数推理流程
[L930] 
[L931] 对于第一层：
[L932] 
[L933] \[
[L934] acc_j = \sum_{i=0}^{783} x_{q,i} \cdot w_{q,j,i} + b_{q,j}
[L935] \]
[L936] 
[L937] 因为我们设计为：
[L938] 
[L939] - 输入 `zero_point = 0`
[L940] - 权重 `zero_point = 0`
[L941] 
[L942] 所以式子就很简单，不用减零点。
[L943] 
[L944] 然后：
[L945] 
[L946] \[
[L947] relu_j = max(0, acc_j)
[L948] \]
[L949] 
[L950] 如果你要再导出 int8 输出用于后级，也可以再做 requant。
[L951] 
[L952] ---
[L953] 
[L954] ## 3.2 统一的导出和 golden 生成代码（路线 A）
[L955] 
[L956] 这段代码是你现在最应该用的版本。
[L957] 
[L958] ```python
[L959] import os
[L960] import json
[L961] import math
[L962] import numpy as np
[L963] import torch
[L964] import torch.nn as nn
[L965] from torchvision import datasets, transforms
[L966] 
[L967] # =========================================================
[L968] # 模型
[L969] # =========================================================
[L970] class MLPRouteA(nn.Module):
[L971]     def __init__(self):
[L972]         super().__init__()
[L973]         self.fc1 = nn.Linear(784, 128, bias=True)
[L974]         self.relu = nn.ReLU()
[L975]         self.fc2 = nn.Linear(128, 10, bias=True)
[L976] 
[L977]     def forward(self, x):
[L978]         x = x.view(-1, 784)
[L979]         x = self.fc1(x)
[L980]         x = self.relu(x)
[L981]         x = self.fc2(x)
[L982]         return x
[L983] 
[L984] 
[L985] # =========================================================
[L986] # 量化工具
[L987] # =========================================================
[L988] def quantize_tensor_affine(tensor, num_bits=8, symmetric=False):
[L989]     if num_bits == 8:
[L990]         qmin, qmax = -128, 127
[L991]         dtype = torch.int8
[L992]     else:
[L993]         raise ValueError("Only int8 supported.")
[L994] 
[L995]     x_min = tensor.min().item()
[L996]     x_max = tensor.max().item()
[L997] 
[L998]     if symmetric:
[L999]         max_abs = max(abs(x_min), abs(x_max))
[L1000]         scale = max_abs / qmax if max_abs != 0 else 1.0
[L1001]         zero_point = 0
[L1002]     else:
[L1003]         if x_max == x_min:
[L1004]             scale = 1.0
[L1005]             zero_point = 0
[L1006]         else:
[L1007]             scale = (x_max - x_min) / (qmax - qmin)
[L1008]             zero_point = round(qmin - x_min / scale)
[L1009]             zero_point = max(qmin, min(qmax, zero_point))
[L1010] 
[L1011]     q = torch.round(tensor / scale + zero_point)
[L1012]     q = torch.clamp(q, qmin, qmax).to(dtype)
[L1013]     return q, float(scale), int(zero_point), qmin, qmax
[L1014] 
[L1015] 
[L1016] def quantize_input_fixed_0_1(x):
[L1017]     """
[L1018]     x in [0,1]
[L1019]     量化到 int8 [0,127]
[L1020]     """
[L1021]     scale = 1.0 / 127.0
[L1022]     zero_point = 0
[L1023]     q = torch.round(x / scale)
[L1024]     q = torch.clamp(q, 0, 127).to(torch.int8)
[L1025]     return q, scale, zero_point
[L1026] 
[L1027] 
[L1028] def quantize_weight_symmetric(weight):
[L1029]     q_w, w_scale, w_zp, _, _ = quantize_tensor_affine(weight, num_bits=8, symmetric=True)
[L1030]     return q_w, w_scale, w_zp
[L1031] 
[L1032] 
[L1033] def quantize_bias_int32(bias_fp32, input_scale, weight_scale):
[L1034]     bias_scale = input_scale * weight_scale
[L1035]     q_bias = torch.round(bias_fp32 / bias_scale).to(torch.int32)
[L1036]     return q_bias, float(bias_scale)
[L1037] 
[L1038] 
[L1039] def compute_requant_params(input_scale, weight_scale, output_scale, shift=24):
[L1040]     """
[L1041]     将 real_multiplier = (input_scale * weight_scale) / output_scale
[L1042]     近似成 integer multiplier + right shift
[L1043]     """
[L1044]     real_multiplier = (input_scale * weight_scale) / output_scale
[L1045]     multiplier = int(round(real_multiplier * (1 << shift)))
[L1046]     return multiplier, shift, real_multiplier
[L1047] 
[L1048] 
[L1049] def requantize_int32_to_int8(acc_int32, multiplier, shift, out_zero_point=0, relu=False):
[L1050]     y = (acc_int32.to(torch.int64) * multiplier) >> shift
[L1051]     y = y + out_zero_point
[L1052]     if relu:
[L1053]         y = torch.maximum(y, torch.tensor(out_zero_point, dtype=y.dtype))
[L1054]     y = torch.clamp(y, -128, 127).to(torch.int8)
[L1055]     return y
[L1056] 
[L1057] 
[L1058] # =========================================================
[L1059] # hex 导出
[L1060] # =========================================================
[L1061] def write_int8_hex_per_line(tensor, filename):
[L1062]     tensor = tensor.flatten().cpu().numpy()
[L1063]     with open(filename, "w") as f:
[L1064]         for v in tensor:
[L1065]             f.write(f"{int(v) & 0xFF:02x}\n")
[L1066] 
[L1067] def write_int32_hex_per_line(tensor, filename):
[L1068]     tensor = tensor.flatten().cpu().numpy()
[L1069]     with open(filename, "w") as f:
[L1070]         for v in tensor:
[L1071]             f.write(f"{int(v) & 0xFFFFFFFF:08x}\n")
[L1072] 
[L1073] 
[L1074] # =========================================================
[L1075] # 导出第一层量化参数
[L1076] # =========================================================
[L1077] def export_fc1_quantized(model_path="mlp_route_a.pth", out_dir="quant_route_a"):
[L1078]     os.makedirs(out_dir, exist_ok=True)
[L1079] 
[L1080]     model = MLPRouteA()
[L1081]     model.load_state_dict(torch.load(model_path, map_location="cpu"))
[L1082]     model.eval()
[L1083] 
[L1084]     weight = model.fc1.weight.data.clone()   # [128,784]
[L1085]     bias   = model.fc1.bias.data.clone()     # [128]
[L1086] 
[L1087]     # 固定输入量化参数
[L1088]     input_scale = 1.0 / 127.0
[L1089]     input_zero_point = 0
[L1090] 
[L1091]     # 权重量化（对称）
[L1092]     q_weight, weight_scale, weight_zero_point = quantize_weight_symmetric(weight)
[L1093] 
[L1094]     # 偏置量化（int32）
[L1095]     q_bias, bias_scale = quantize_bias_int32(bias, input_scale, weight_scale)
[L1096] 
[L1097]     # 输出量化参数（给 fc1 输出用）
[L1098]     # 这里按浮点 fc1 输出范围估计一个固定输出 scale；工程上可用校准集更稳
[L1099]     with torch.no_grad():
[L1100]         dummy = torch.rand(256, 1, 28, 28)
[L1101]         fc1_out = model.relu(model.fc1(dummy.view(-1, 784)))
[L1102]     max_out = fc1_out.max().item()
[L1103]     output_scale = max_out / 127.0 if max_out != 0 else 1.0
[L1104]     output_zero_point = 0
[L1105] 
[L1106]     multiplier, shift, real_multiplier = compute_requant_params(
[L1107]         input_scale, weight_scale, output_scale, shift=24
[L1108]     )
[L1109] 
[L1110]     write_int8_hex_per_line(q_weight, os.path.join(out_dir, "fc1_weight.hex"))
[L1111]     write_int32_hex_per_line(q_bias, os.path.join(out_dir, "fc1_bias.hex"))
[L1112] 
[L1113]     config = {
[L1114]         "input_scale": input_scale,
[L1115]         "input_zero_point": input_zero_point,
[L1116]         "weight_scale": weight_scale,
[L1117]         "weight_zero_point": weight_zero_point,
[L1118]         "bias_scale": bias_scale,
[L1119]         "output_scale": output_scale,
[L1120]         "output_zero_point": output_zero_point,
[L1121]         "requant_multiplier": multiplier,
[L1122]         "requant_shift": shift,
[L1123]         "requant_real_multiplier": real_multiplier,
[L1124]         "weight_shape": [128, 784],
[L1125]         "bias_shape": [128],
[L1126]         "weight_layout": "[out][in], row-major",
[L1127]         "input_layout": "[in]",
[L1128]         "output_layout": "[out]"
[L1129]     }
[L1130] 
[L1131]     with open(os.path.join(out_dir, "fc1_quant_config.json"), "w") as f:
[L1132]         json.dump(config, f, indent=2)
[L1133] 
[L1134]     print("Exported fc1 quantized parameters.")
[L1135] 
[L1136] 
[L1137] # =========================================================
[L1138] # 生成整数链路 golden
[L1139] # =========================================================
[L1140] def generate_test_vectors_route_a(model_path="mlp_route_a.pth", out_dir="quant_route_a", num_samples=10):
[L1141]     os.makedirs(out_dir, exist_ok=True)
[L1142] 
[L1143]     model = MLPRouteA()
[L1144]     model.load_state_dict(torch.load(model_path, map_location="cpu"))
[L1145]     model.eval()
[L1146] 
[L1147]     # 载入模型参数并量化
[L1148]     weight = model.fc1.weight.data.clone()   # [128,784]
[L1149]     bias   = model.fc1.bias.data.clone()
[L1150] 
[L1151]     input_scale = 1.0 / 127.0
[L1152]     input_zero_point = 0
[L1153] 
[L1154]     q_weight, weight_scale, weight_zero_point = quantize_weight_symmetric(weight)
[L1155]     q_bias, bias_scale = quantize_bias_int32(bias, input_scale, weight_scale)
[L1156] 
[L1157]     with torch.no_grad():
[L1158]         dummy = torch.rand(256, 1, 28, 28)
[L1159]         fc1_out = model.relu(model.fc1(dummy.view(-1, 784)))
[L1160]     max_out = fc1_out.max().item()
[L1161]     output_scale = max_out / 127.0 if max_out != 0 else 1.0
[L1162]     output_zero_point = 0
[L1163] 
[L1164]     multiplier, shift, _ = compute_requant_params(input_scale, weight_scale, output_scale, shift=24)
[L1165] 
[L1166]     transform = transforms.Compose([
[L1167]         transforms.ToTensor(),   # 不做 Normalize，统一输入范围 [0,1]
[L1168]     ])
[L1169]     test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
[L1170] 
[L1171]     labels = []
[L1172] 
[L1173]     for i in range(num_samples):
[L1174]         img, label = test_dataset[i]
[L1175]         x = img.view(-1)  # [784]
[L1176] 
[L1177]         # 固定输入量化
[L1178]         q_x, _, _ = quantize_input_fixed_0_1(x)  # [784]
[L1179] 
[L1180]         # int32 累加： [128,784] x [784] -> [128]
[L1181]         acc = torch.sum(q_weight.to(torch.int32) * q_x.to(torch.int32).unsqueeze(0), dim=1) + q_bias
[L1182] 
[L1183]         # ReLU(int32)
[L1184]         relu_acc = torch.clamp(acc, min=0)
[L1185] 
[L1186]         # requant 到 int8，便于后级或观测
[L1187]         q_out = requantize_int32_to_int8(
[L1188]             relu_acc, multiplier, shift, out_zero_point=output_zero_point, relu=False
[L1189]         )
[L1190] 
[L1191]         # 保存
[L1192]         write_int8_hex_per_line(q_x,   os.path.join(out_dir, f"input_{i}.hex"))
[L1193]         write_int32_hex_per_line(acc,  os.path.join(out_dir, f"golden_fc1_acc_{i}.hex"))
[L1194]         write_int32_hex_per_line(relu_acc, os.path.join(out_dir, f"golden_fc1_relu_acc_{i}.hex"))
[L1195]         write_int8_hex_per_line(q_out, os.path.join(out_dir, f"golden_fc1_out_{i}.hex"))
[L1196] 
[L1197]         labels.append(int(label))
[L1198]         print(f"sample {i}: label={label}")
[L1199] 
[L1200]     with open(os.path.join(out_dir, "labels.txt"), "w") as f:
[L1201]         for lb in labels:
[L1202]             f.write(f"{lb}\n")
[L1203] 
[L1204]     print("Generated integer golden test vectors for route A.")
[L1205] ```
[L1206] 
[L1207] ---
[L1208] 
[L1209] # 4. 输入测试向量统一量化策略
[L1210] 
[L1211] 你要的是**贴近硬件即可**。那我建议你用下面这个方案。
[L1212] 
[L1213] ---
[L1214] 
[L1215] ## 推荐方案：输入保持 `[0,1]`，统一量化到 int8 `[0,127]`
[L1216] 
[L1217] ### 设计理由
[L1218] 这对 FPGA 最友好：
[L1219] 
[L1220] - 不用每张图动态算 scale
[L1221] - 不用保留 MNIST 的 mean/std normalize
[L1222] - 输入像素天然接近图像传感器/存储格式
[L1223] - 便于后面 BRAM 或 ROM 存图
[L1224] 
[L1225] ### 固定规则
[L1226] \[
[L1227] x_q = round(x \times 127)
[L1228] \]
[L1229] 其中 `x in [0,1]`
[L1230] 
[L1231] 即：
[L1232] 
[L1233] ```python
[L1234] def quantize_input_fixed_0_1(x):
[L1235]     scale = 1.0 / 127.0
[L1236]     zero_point = 0
[L1237]     q = torch.round(x / scale)
[L1238]     q = torch.clamp(q, 0, 127).to(torch.int8)
[L1239]     return q, scale, zero_point
[L1240] ```
[L1241] 
[L1242] ---
[L1243] 
[L1244] ## 为什么我不建议你继续用：
[L1245] ```python
[L1246] transforms.Normalize((0.1307,), (0.3081,))
[L1247] ```
[L1248] 
[L1249] 因为这会把输入变成带负数的浮点分布，虽然适合软件训练，但不适合你现在的 FPGA 验证阶段：
[L1250] 
[L1251] - 输入分布复杂
[L1252] - 量化参数更麻烦
[L1253] - debug 不直观
[L1254] 
[L1255] 所以你现在更适合：
[L1256] 
[L1257] - 训练和测试都直接 `ToTensor()`
[L1258] - 用统一固定输入量化
[L1259] 
[L1260] ---
[L1261] 
[L1262] # 5. 你当前权重输出顺序，对应硬件怎么设计读取顺序
[L1263] 
[L1264] 你当前 notebook 的写法是：
[L1265] 
[L1266] ```python
[L1267] for row in q_weight:
[L1268]     for val in row:
[L1269]         f.write(f'{val.item() & 0xFF:02x}\n')
[L1270] ```
[L1271] 
[L1272] 而 `weight = model.fc1.weight.data` 的 shape 是：
[L1273] 
[L1274] ```python
[L1275] [128, 784]
[L1276] ```
[L1277] 
[L1278] 在 PyTorch 里，`nn.Linear(784,128)` 的权重定义是：
[L1279] 
[L1280] \[
[L1281] W[j][i]
[L1282] \]
[L1283] 
[L1284] 其中：
[L1285] 
[L1286] - `j` 是输出神经元索引，范围 `0~127`
[L1287] - `i` 是输入特征索引，范围 `0~783`
[L1288] 
[L1289] ---
[L1290] 
[L1291] ## 你现在的导出顺序是什么
[L1292] 
[L1293] 它是：
[L1294] 
[L1295] - 先写第 0 个输出神经元的 784 个权重
[L1296] - 再写第 1 个输出神经元的 784 个权重
[L1297] - ...
[L1298] - 最后写第 127 个输出神经元的 784 个权重
[L1299] 
[L1300] 也就是：
[L1301] 
[L1302] \[
[L1303] W[0][0], W[0][1], ..., W[0][783], W[1][0], ..., W[127][783]
[L1304] \]
[L1305] 
[L1306] 这是典型的：
[L1307] 
[L1308] - **按输出通道优先**
[L1309] - **行优先 row-major**
[L1310] - **布局 `[out][in]`**
[L1311] 
[L1312] ---
[L1313] 
[L1314] ## 对应的硬件读取顺序应该怎么设计
[L1315] 
[L1316] 如果你的硬件是“一个输出神经元一轮累加”的思路，那么最自然。
[L1317] 
[L1318] ---
[L1319] 
[L1320] ## 硬件读取方式 1：逐个输出神经元计算
[L1321] 
[L1322] ### 第 `j` 个输出的计算
[L1323] \[
[L1324] y_j = \sum_{i=0}^{783} x_i \cdot W[j][i] + b_j
[L1325] \]
[L1326] 
[L1327] ### 内存地址设计
[L1328] 如果 `weights.hex` 直接线性装入 BRAM，则：
[L1329] 
[L1330] \[
[L1331] addr = j \times 784 + i
[L1332] \]
[L1333] 
[L1334] 所以：
[L1335] 
[L1336] - 固定一个输出神经元 `j`
[L1337] - 地址从 `j*784` 扫到 `j*784+783`
[L1338] - 同时输入向量 `x[0:783]` 顺序送入
[L1339] - MAC 累加完成后，加 `bias[j]`
[L1340] - 输出 `y_j`
[L1341] 
[L1342] 然后 `j++`
[L1343] 
[L1344] ---
[L1345] 
[L1346] ## 这种方式的优点
[L1347] - 和你现在导出的文件顺序完全一致
[L1348] - 地址生成最简单
[L1349] - 软件和硬件最容易对齐
[L1350] - 最适合第一版 FPGA 联调
[L1351] 
[L1352] ---
[L1353] 
[L1354] ## 你现在应该怎么设计硬件
[L1355] 
[L1356] ### 输入存储
[L1357] `input_i.hex` 顺序也是：
[L1358] \[
[L1359] x[0], x[1], ..., x[783]
[L1360] \]
[L1361] 
[L1362] ### 权重存储
[L1363] `weights.hex` 顺序：
[L1364] \[
[L1365] W[0][0], W[0][1], ..., W[0][783], W[1][0], ..., W[127][783]
[L1366] \]
[L1367] 
[L1368] ### bias 存储
[L1369] `bias.hex` 顺序：
[L1370] \[
[L1371] b[0], b[1], ..., b[127]
[L1372] \]
[L1373] 
[L1374] ---
[L1375] 
[L1376] ## 控制流程
[L1377] 对于输出神经元 `j`：
[L1378] 
[L1379] 1. `acc = 0`
[L1380] 2. 对 `i = 0 ~ 783`
[L1381]    - 读 `x[i]`
[L1382]    - 读 `W[j][i]`
[L1383]    - `acc += x[i] * W[j][i]`
[L1384] 3. `acc += bias[j]`
[L1385] 4. `acc = relu(acc)`
[L1386] 5. 写 `out[j]`
[L1387] 
[L1388] 然后进入下一个 `j`
[L1389] 
[L1390] ---
[L1391] 
[L1392] ## 地址公式
[L1393] ### 权重地址
[L1394] \[
[L1395] weight\_addr = out\_idx \times 784 + in\_idx
[L1396] \]
[L1397] 
[L1398] ### 输入地址
[L1399] \[
[L1400] input\_addr = in\_idx
[L1401] \]
[L1402] 
[L1403] ### 偏置地址
[L1404] \[
[L1405] bias\_addr = out\_idx
[L1406] \]
[L1407] 
[L1408] ---
[L1409] 
[L1410] ## 什么时候要改输出顺序
[L1411] 只有在你未来做这些架构时才需要变：
[L1412] 
[L1413] - 多 PE 并行读多个输出神经元
[L1414] - 按列块分块存储
[L1415] - CIM 阵列按 bitline/wordline 特定映射
[L1416] - 片上 SRAM 宽度要求按 8/16/32 个权重打包
[L1417] 
[L1418] 但对于你现在这个阶段，**完全没必要先改**。  
[L1419] 你当前顺序其实正适合第一版硬件。
[L1420] 
[L1421] ---
[L1422] 
[L1423] # 我给你的最终建议
[L1424] 
[L1425] ## 你现在最推荐的工程路线
[L1426] 先走：
[L1427] 
[L1428] - **路线 A**
[L1429] - 完整训练分类器
[L1430] - FPGA 只做第一层 `fc1 + ReLU`
[L1431] - 输入统一量化 `[0,1] -> int8 [0,127]`
[L1432] - 权重 int8 对称量化
[L1433] - bias int32
[L1434] - golden 用整数推理链路生成
[L1435] - 权重按你现在的 `[out][in]` 顺序导出
[L1436] 
[L1437] 这是最稳、最容易跑通的一条路。
[L1438] 
[L1439] ---
[L1440] 
[L1441] ## 你下一步最值得做的事
[L1442] 把你当前 notebook 分成三个独立脚本：
[L1443] 
[L1444] 1. `train_route_a.py`
[L1445] 2. `export_fc1_quant.py`
[L1446] 3. `generate_test_vectors_route_a.py`
[L1447] 
[L1448] 这样后面调试不会乱。
[L1449] 
[L1450] 如果你要，我下一条可以直接继续给你一份**整理好的、可直接复制运行的完整工程版代码**，把路线 A 的训练、量化导出、golden 生成合并成一个干净版本。
[L1451] 
[L1452] 用户:
[L1453] 关于1.现在没有在训练一个 MNIST 分类器的问题，请分别给我路线a和路线b的完整代码；
[L1454] 
[L1455] 关于2.量化函数有工程问题，不适合直接对接 FPGA，
[L1456] zero_point不是整数的问题帮我修改；每个张量动态量化这个问题给我一个合理的解决方案；偏置不能简单用 int8 单独量化请帮我提出解决方案和代码；
[L1457] 
[L1458] 关于3.generate_test_vectors() 生成的 golden，和硬件真实计算链路不一致，整数推理链路生成 golden，帮我完成设计；
[L1459] 
[L1460] 关于4输入测试向量的量化方式不适合硬件批量验证，帮我设计统一输入量化策略，贴近硬件即可；
[L1461] 
[L1462] 关于5. 导出的权重格式本身基本可用，但有几个注意点，这里文件顺序你先记住我的输出顺序，讲解一下我现在的输出顺序应该如何设计硬件读取顺序；
[L1463] 
[L1464] 
[L1465] 问题6,7你帮我解决一下。
[L1466] 
[L1467] 按照你提出的方案二：直接瞄准最终演示
[L1468] 
[L1469] 如果你们想最后做“MNIST 分类结果上板演示”，那就：
[L1470] 
[L1471] 模型改为两层 MLP
[L1472] 
[L1473] 训练用 CrossEntropyLoss
[L1474] 
[L1475] 做静态量化
[L1476] 
[L1477] 先用 Python 写一个整数推理参考实现
[L1478] 
[L1479] 再把这个整数推理结果作为 FPGA golden进行处理。
[L1480] 其他应该修改的你帮我修改，和我说一声即可。
[L1481] AI助手:
[L1482] 我先基于你现有 notebook 的结构，把方案 A、方案 B，以及静态量化、整数推理 golden、测试向量导出这几部分整理成能直接落地的代码。然后我会把硬件读取顺序也一并说明清楚，保证软件导出和 RTL 地址映射能闭环。
[L1483] 
[L1484] AI助手:
[L1485] 下面我直接按你的要求给出可落地代码和设计说明。  
[L1486] 我把内容分成两部分：
[L1487] 
[L1488] 1. **路线 A 完整代码**：训练完整 MLP，但 FPGA 先只实现第一层 `fc1 + relu`，用于硬件验证  
[L1489] 2. **路线 B 完整代码**：直接瞄准最终演示，做完整两层 MLP、静态量化、整数推理 golden 导出
[L1490] 
[L1491] 你最后明确说“**按方案二直接瞄准最终演示**”，所以我建议你真正落地时优先用**路线 B**。  
[L1492] 路线 A 我也完整给你，方便你前期做单层阵列验证。
[L1493] 
[L1494] ---
[L1495] 
[L1496] # 一、你提出的几个问题，我已经统一改掉了
[L1497] 
[L1498] 这次代码里我已经一起处理了这些问题：
[L1499] 
[L1500] ## 已修改的问题
[L1501] ### 1）训练目标错误
[L1502] - 路线 A / B 都改成了真正的分类训练
[L1503] - 使用 `CrossEntropyLoss`
[L1504] 
[L1505] ### 2）`zero_point` 不是整数
[L1506] - 已改为**整数 zero_point**
[L1507] - 并做了合法范围裁剪
[L1508] 
[L1509] ### 3）动态量化不适合 FPGA
[L1510] - 改为**静态量化**
[L1511] - 输入、权重、输出都用**固定 scale / zero_point**
[L1512] - 不再每个样本单独算量化参数
[L1513] 
[L1514] ### 4）bias 不能简单 int8
[L1515] - 改为：
[L1516]   - `bias_q = round(bias_fp / (input_scale * weight_scale))`
[L1517]   - 使用 **int32** 保存
[L1518] 
[L1519] ### 5）golden 与硬件链路不一致
[L1520] - 改为**整数推理参考链路**
[L1521] - golden 由整数乘加、加 bias、ReLU、requant 得到
[L1522] - 和 FPGA 数据通路一致
[L1523] 
[L1524] ### 6）输入量化方式不适合批量验证
[L1525] - 统一输入量化策略
[L1526] - 采用更贴近硬件的方案：**MNIST 像素 0~1 -> uint8 0~255，再映射为 int8（减去 128）**
[L1527] - 固定 `input_scale = 1/255`，`input_zero_point = -128`
[L1528] 
[L1529] ### 7）其他问题我也一起修了
[L1530] - `torch.load(..., map_location=device)`
[L1531] - 增加 test accuracy
[L1532] - 保存标签文件
[L1533] - 保存中间层 int32 累加结果，便于 RTL 分级定位
[L1534] - 导出 json 配置，方便 FPGA 端对齐
[L1535] 
[L1536] ---
[L1537] 
[L1538] # 二、统一量化策略设计
[L1539] 
[L1540] 你要求“贴近硬件即可”，我这里给你一个非常适合 FPGA 的设计：
[L1541] 
[L1542] ## 输入量化
[L1543] 原始 MNIST 图像来自 `ToTensor()` 后范围是 `[0,1]`
[L1544] 
[L1545] 统一量化为：
[L1546] 
[L1547] \[
[L1548] x_q = round(x / s_x) + z_x
[L1549] \]
[L1550] 
[L1551] 其中：
[L1552] 
[L1553] - `s_x = 1 / 255`
[L1554] - `z_x = -128`
[L1555] 
[L1556] 这样就等价于：
[L1557] 
[L1558] ```python
[L1559] x_u8 = round(x * 255)      # 0~255
[L1560] x_q  = x_u8 - 128          # -128~127
[L1561] ```
[L1562] 
[L1563] 优点：
[L1564] - 很贴近硬件
[L1565] - 不需要每张图动态量化
[L1566] - FPGA 端容易处理
[L1567] - BRAM 存储方便
[L1568] 
[L1569] ## 权重量化
[L1570] 对每一层权重做**对称量化**：
[L1571] 
[L1572] \[
[L1573] w_q = round(w / s_w)
[L1574] \]
[L1575] 
[L1576] 其中：
[L1577] 
[L1578] \[
[L1579] s_w = \max(|w|) / 127
[L1580] \]
[L1581] 
[L1582] - `w_zero_point = 0`
[L1583] - `w_q` 存为 int8
[L1584] 
[L1585] ## 偏置量化
[L1586] 偏置必须与累加器同量纲：
[L1587] 
[L1588] \[
[L1589] b_q = round(b / (s_x \cdot s_w))
[L1590] \]
[L1591] 
[L1592] - 存为 int32
[L1593] 
[L1594] ## 输出量化
[L1595] 累加后得到 int32：
[L1596] 
[L1597] \[
[L1598] acc = \sum (x_q - z_x) \cdot (w_q - z_w) + b_q
[L1599] \]
[L1600] 
[L1601] 然后 ReLU，再 requant：
[L1602] 
[L1603] \[
[L1604] y_q = round(acc \cdot (s_x s_w / s_y)) + z_y
[L1605] \]
[L1606] 
[L1607] 输出层如需分类 argmax，最终也可直接基于 int32 或 requant 后 int8 做。
[L1608] 
[L1609] ---
[L1610] 
[L1611] # 三、路线 A：完整代码  
[L1612] ## 用途
[L1613] - 软件训练完整分类器：`784 -> 128 -> 10`
[L1614] - FPGA 先只实现第一层：`fc1 + relu`
[L1615] - 导出第一层输入、权重、bias、int32 累加 golden、relu 输出 golden
[L1616] 
[L1617] 这条路线适合你前期快速验证 CIM/FC 阵列。
[L1618] 
[L1619] ---
[L1620] 
[L1621] ## 路线 A 完整代码
[L1622] 
[L1623] ```python
[L1624] import os
[L1625] import json
[L1626] import math
[L1627] import numpy as np
[L1628] import torch
[L1629] import torch.nn as nn
[L1630] import torch.optim as optim
[L1631] from torchvision import datasets, transforms
[L1632] from torch.utils.data import DataLoader
[L1633] 
[L1634] # =========================================================
[L1635] # 1. 配置
[L1636] # =========================================================
[L1637] DATA_DIR = "./data"
[L1638] OUT_DIR = "./route_a_output"
[L1639] MODEL_PATH = os.path.join(OUT_DIR, "mlp_route_a.pth")
[L1640] 
[L1641] os.makedirs(OUT_DIR, exist_ok=True)
[L1642] 
[L1643] DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
[L1644] 
[L1645] 
[L1646] # =========================================================
[L1647] # 2. 模型：完整训练，但导出时只导出第一层用于 FPGA
[L1648] # =========================================================
[L1649] class MLPRouteA(nn.Module):
[L1650]     def __init__(self):
[L1651]         super().__init__()
[L1652]         self.fc1 = nn.Linear(784, 128, bias=True)
[L1653]         self.relu = nn.ReLU()
[L1654]         self.fc2 = nn.Linear(128, 10, bias=True)
[L1655] 
[L1656]     def forward(self, x):
[L1657]         x = x.view(-1, 784)
[L1658]         h = self.fc1(x)
[L1659]         h_relu = self.relu(h)
[L1660]         y = self.fc2(h_relu)
[L1661]         return y
[L1662] 
[L1663] 
[L1664] # =========================================================
[L1665] # 3. 数据
[L1666] #   路线 A 训练时仍然可以用标准 Normalize
[L1667] #   因为这里只是先训练出一个可用分类器
[L1668] # =========================================================
[L1669] def get_train_test_loaders(batch_size=128):
[L1670]     transform = transforms.Compose([
[L1671]         transforms.ToTensor(),
[L1672]         transforms.Normalize((0.1307,), (0.3081,))
[L1673]     ])
[L1674] 
[L1675]     train_dataset = datasets.MNIST(
[L1676]         DATA_DIR, train=True, download=True, transform=transform
[L1677]     )
[L1678]     test_dataset = datasets.MNIST(
[L1679]         DATA_DIR, train=False, download=True, transform=transform
[L1680]     )
[L1681] 
[L1682]     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
[L1683]     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
[L1684]     return train_loader, test_loader
[L1685] 
[L1686] 
[L1687] # =========================================================
[L1688] # 4. 训练 / 测试
[L1689] # =========================================================
[L1690] def train_model(num_epochs=5, lr=1e-3):
[L1691]     train_loader, test_loader = get_train_test_loaders()
[L1692] 
[L1693]     model = MLPRouteA().to(DEVICE)
[L1694]     optimizer = optim.Adam(model.parameters(), lr=lr)
[L1695]     criterion = nn.CrossEntropyLoss()
[L1696] 
[L1697]     for epoch in range(num_epochs):
[L1698]         model.train()
[L1699]         running_loss = 0.0
[L1700] 
[L1701]         for data, target in train_loader:
[L1702]             data, target = data.to(DEVICE), target.to(DEVICE)
[L1703] 
[L1704]             optimizer.zero_grad()
[L1705]             logits = model(data)
[L1706]             loss = criterion(logits, target)
[L1707]             loss.backward()
[L1708]             optimizer.step()
[L1709] 
[L1710]             running_loss += loss.item() * data.size(0)
[L1711] 
[L1712]         avg_loss = running_loss / len(train_loader.dataset)
[L1713]         acc = evaluate_model(model, test_loader)
[L1714]         print(f"[Epoch {epoch+1}/{num_epochs}] loss={avg_loss:.6f}, test_acc={acc:.4f}")
[L1715] 
[L1716]     torch.save(model.state_dict(), MODEL_PATH)
[L1717]     print(f"Model saved to: {MODEL_PATH}")
[L1718]     return model
[L1719] 
[L1720] 
[L1721] def evaluate_model(model, test_loader):
[L1722]     model.eval()
[L1723]     correct = 0
[L1724]     total = 0
[L1725] 
[L1726]     with torch.no_grad():
[L1727]         for data, target in test_loader:
[L1728]             data, target = data.to(DEVICE), target.to(DEVICE)
[L1729]             logits = model(data)
[L1730]             pred = logits.argmax(dim=1)
[L1731]             correct += (pred == target).sum().item()
[L1732]             total += target.size(0)
[L1733] 
[L1734]     return correct / total
[L1735] 
[L1736] 
[L1737] # =========================================================
[L1738] # 5. 量化函数
[L1739] #   路线 A：为了 FPGA 只验证第一层，采用固定输入量化策略
[L1740] # =========================================================
[L1741] INT8_QMIN = -128
[L1742] INT8_QMAX = 127
[L1743] INT32_QMIN = -2147483648
[L1744] INT32_QMAX = 2147483647
[L1745] 
[L1746] def clamp(x, low, high):
[L1747]     return max(low, min(high, x))
[L1748] 
[L1749] def quantize_affine_tensor(tensor_fp, scale, zero_point, qmin=-128, qmax=127, dtype=torch.int8):
[L1750]     q = torch.round(tensor_fp / scale + zero_point)
[L1751]     q = torch.clamp(q, qmin, qmax)
[L1752]     return q.to(dtype)
[L1753] 
[L1754] def calc_symmetric_scale(tensor_fp, num_bits=8):
[L1755]     qmax = (2 ** (num_bits - 1)) - 1
[L1756]     max_abs = max(abs(tensor_fp.min().item()), abs(tensor_fp.max().item()))
[L1757]     if max_abs == 0:
[L1758]         return 1.0
[L1759]     return max_abs / qmax
[L1760] 
[L1761] def quantize_weight_symmetric_int8(weight_fp):
[L1762]     scale = calc_symmetric_scale(weight_fp, num_bits=8)
[L1763]     zp = 0
[L1764]     weight_q = quantize_affine_tensor(weight_fp, scale, zp, -128, 127, torch.int8)
[L1765]     return weight_q, scale, zp
[L1766] 
[L1767] def quantize_bias_int32(bias_fp, input_scale, weight_scale):
[L1768]     bias_scale = input_scale * weight_scale
[L1769]     bias_q = torch.round(bias_fp / bias_scale)
[L1770]     bias_q = torch.clamp(bias_q, INT32_QMIN, INT32_QMAX).to(torch.int32)
[L1771]     return bias_q, bias_scale
[L1772] 
[L1773] def quantize_input_hw_style(img_fp_01):
[L1774]     """
[L1775]     img_fp_01: 原始像素 [0,1]
[L1776]     统一策略：
[L1777]         x_u8 = round(x * 255)
[L1778]         x_q  = x_u8 - 128
[L1779]     对应：
[L1780]         input_scale = 1/255
[L1781]         input_zero_point = -128
[L1782]     """
[L1783]     input_scale = 1.0 / 255.0
[L1784]     input_zero_point = -128
[L1785] 
[L1786]     x_u8 = torch.round(img_fp_01 * 255.0)
[L1787]     x_u8 = torch.clamp(x_u8, 0, 255)
[L1788]     x_q = x_u8.to(torch.int16) - 128
[L1789]     x_q = torch.clamp(x_q, -128, 127).to(torch.int8)
[L1790] 
[L1791]     return x_q, input_scale, input_zero_point
[L1792] 
[L1793] 
[L1794] # =========================================================
[L1795] # 6. 第一层整数推理 golden
[L1796] #   只对应 fc1 + relu
[L1797] # =========================================================
[L1798] def fc1_integer_reference(x_q, w_q, b_q, x_zp=-128, w_zp=0):
[L1799]     """
[L1800]     x_q: [784] int8
[L1801]     w_q: [128, 784] int8
[L1802]     b_q: [128] int32
[L1803] 
[L1804]     acc_j = sum_i (x_q[i] - x_zp)*(w_q[j,i] - w_zp) + b_q[j]
[L1805]     relu_acc = max(acc_j, 0)
[L1806]     """
[L1807]     x_int = x_q.to(torch.int32) - int(x_zp)
[L1808]     w_int = w_q.to(torch.int32) - int(w_zp)
[L1809] 
[L1810]     acc = torch.matmul(w_int, x_int) + b_q
[L1811]     relu_acc = torch.clamp(acc, min=0)
[L1812]     return acc.to(torch.int32), relu_acc.to(torch.int32)
[L1813] 
[L1814] 
[L1815] # =========================================================
[L1816] # 7. HEX 导出
[L1817] # =========================================================
[L1818] def write_int8_hex_1d(tensor, path):
[L1819]     with open(path, "w") as f:
[L1820]         for v in tensor.flatten():
[L1821]             f.write(f"{int(v.item()) & 0xFF:02x}\n")
[L1822] 
[L1823] def write_int32_hex_1d(tensor, path):
[L1824]     with open(path, "w") as f:
[L1825]         for v in tensor.flatten():
[L1826]             f.write(f"{int(v.item()) & 0xFFFFFFFF:08x}\n")
[L1827] 
[L1828] def export_route_a_artifacts(num_samples=10):
[L1829]     # 加载训练后的模型
[L1830]     model = MLPRouteA().to(DEVICE)
[L1831]     model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
[L1832]     model.eval()
[L1833] 
[L1834]     # 只提取 fc1
[L1835]     fc1_w_fp = model.fc1.weight.data.cpu()   # [128, 784]
[L1836]     fc1_b_fp = model.fc1.bias.data.cpu()     # [128]
[L1837] 
[L1838]     # 固定输入量化参数
[L1839]     input_scale = 1.0 / 255.0
[L1840]     input_zero_point = -128
[L1841] 
[L1842]     # 权重量化（对称 int8）
[L1843]     fc1_w_q, fc1_w_scale, fc1_w_zp = quantize_weight_symmetric_int8(fc1_w_fp)
[L1844] 
[L1845]     # 偏置量化（int32）
[L1846]     fc1_b_q, fc1_b_scale = quantize_bias_int32(fc1_b_fp, input_scale, fc1_w_scale)
[L1847] 
[L1848]     # 导出权重：按 [out][in] 顺序
[L1849]     write_int8_hex_1d(fc1_w_q.reshape(-1), os.path.join(OUT_DIR, "fc1_weight_int8.hex"))
[L1850]     write_int32_hex_1d(fc1_b_q, os.path.join(OUT_DIR, "fc1_bias_int32.hex"))
[L1851] 
[L1852]     # 保存配置
[L1853]     config = {
[L1854]         "layer": "fc1_only",
[L1855]         "weight_shape": [128, 784],
[L1856]         "weight_layout": "row-major [out][in]",
[L1857]         "input_layout": "[in]",
[L1858]         "bias_layout": "[out]",
[L1859]         "input_scale": input_scale,
[L1860]         "input_zero_point": input_zero_point,
[L1861]         "fc1_weight_scale": fc1_w_scale,
[L1862]         "fc1_weight_zero_point": fc1_w_zp,
[L1863]         "fc1_bias_scale": fc1_b_scale,
[L1864]         "fc1_bias_dtype": "int32"
[L1865]     }
[L1866] 
[L1867]     with open(os.path.join(OUT_DIR, "quant_config.json"), "w", encoding="utf-8") as f:
[L1868]         json.dump(config, f, indent=2, ensure_ascii=False)
[L1869] 
[L1870]     # 测试集：为了导出原始像素，不能 Normalize
[L1871]     test_transform = transforms.Compose([
[L1872]         transforms.ToTensor()
[L1873]     ])
[L1874]     test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=test_transform)
[L1875] 
[L1876]     labels = []
[L1877] 
[L1878]     for i in range(num_samples):
[L1879]         img_fp, label = test_dataset[i]  # [1,28,28], range [0,1]
[L1880]         labels.append(int(label))
[L1881] 
[L1882]         img_flat = img_fp.view(-1).cpu()
[L1883]         x_q, _, _ = quantize_input_hw_style(img_flat)
[L1884] 
[L1885]         acc_int32, relu_int32 = fc1_integer_reference(
[L1886]             x_q, fc1_w_q, fc1_b_q, x_zp=input_zero_point, w_zp=fc1_w_zp
[L1887]         )
[L1888] 
[L1889]         write_int8_hex_1d(x_q, os.path.join(OUT_DIR, f"input_{i}.hex"))
[L1890]         write_int32_hex_1d(acc_int32, os.path.join(OUT_DIR, f"fc1_acc_golden_{i}.hex"))
[L1891]         write_int32_hex_1d(relu_int32, os.path.join(OUT_DIR, f"fc1_relu_golden_{i}.hex"))
[L1892] 
[L1893]         print(f"Exported sample {i}, label={label}")
[L1894] 
[L1895]     with open(os.path.join(OUT_DIR, "labels.txt"), "w") as f:
[L1896]         for lb in labels:
[L1897]             f.write(f"{lb}\n")
[L1898] 
[L1899]     print("Route A export done.")
[L1900] 
[L1901] 
[L1902] if __name__ == "__main__":
[L1903]     train_model(num_epochs=5, lr=1e-3)
[L1904]     export_route_a_artifacts(num_samples=10)
[L1905] ```
[L1906] 
[L1907] ---
[L1908] 
[L1909] # 四、路线 B：完整代码  
[L1910] ## 用途
[L1911] 这是你现在真正应该用的版本：
[L1912] 
[L1913] - 完整两层 MLP：`784 -> 128 -> 10`
[L1914] - `CrossEntropyLoss`
[L1915] - 静态量化
[L1916] - Python 写整数推理参考实现
[L1917] - 导出 FPGA golden
[L1918] - 最终瞄准“MNIST 上板分类演示”
[L1919] 
[L1920] ---
[L1921] 
[L1922] ## 路线 B 完整代码
[L1923] 
[L1924] ```python
[L1925] import os
[L1926] import json
[L1927] import math
[L1928] import numpy as np
[L1929] import torch
[L1930] import torch.nn as nn
[L1931] import torch.optim as optim
[L1932] from torchvision import datasets, transforms
[L1933] from torch.utils.data import DataLoader
[L1934] 
[L1935] # =========================================================
[L1936] # 1. 配置
[L1937] # =========================================================
[L1938] DATA_DIR = "./data"
[L1939] OUT_DIR = "./route_b_output"
[L1940] MODEL_PATH = os.path.join(OUT_DIR, "mlp_route_b.pth")
[L1941] 
[L1942] os.makedirs(OUT_DIR, exist_ok=True)
[L1943] 
[L1944] DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
[L1945] 
[L1946] INT8_QMIN = -128
[L1947] INT8_QMAX = 127
[L1948] INT32_QMIN = -2147483648
[L1949] INT32_QMAX = 2147483647
[L1950] 
[L1951] 
[L1952] # =========================================================
[L1953] # 2. 模型
[L1954] # =========================================================
[L1955] class MLPRouteB(nn.Module):
[L1956]     def __init__(self):
[L1957]         super().__init__()
[L1958]         self.fc1 = nn.Linear(784, 128, bias=True)
[L1959]         self.relu = nn.ReLU()
[L1960]         self.fc2 = nn.Linear(128, 10, bias=True)
[L1961] 
[L1962]     def forward(self, x):
[L1963]         x = x.view(-1, 784)
[L1964]         x = self.fc1(x)
[L1965]         x = self.relu(x)
[L1966]         x = self.fc2(x)
[L1967]         return x
[L1968] 
[L1969] 
[L1970] # =========================================================
[L1971] # 3. 数据
[L1972] #   训练时这里直接用 ToTensor()，不做 Normalize
[L1973] #   这样更贴近最终硬件部署链路
[L1974] # =========================================================
[L1975] def get_train_test_loaders(batch_size=128):
[L1976]     transform = transforms.Compose([
[L1977]         transforms.ToTensor()
[L1978]     ])
[L1979] 
[L1980]     train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
[L1981]     test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
[L1982] 
[L1983]     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
[L1984]     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
[L1985]     return train_loader, test_loader
[L1986] 
[L1987] 
[L1988] # =========================================================
[L1989] # 4. 训练 / 测试
[L1990] # =========================================================
[L1991] def evaluate_model(model, test_loader):
[L1992]     model.eval()
[L1993]     total = 0
[L1994]     correct = 0
[L1995] 
[L1996]     with torch.no_grad():
[L1997]         for data, target in test_loader:
[L1998]             data, target = data.to(DEVICE), target.to(DEVICE)
[L1999]             logits = model(data)
[L2000]             pred = torch.argmax(logits, dim=1)
[L2001]             correct += (pred == target).sum().item()
[L2002]             total += target.size(0)
[L2003] 
[L2004]     return correct / total
[L2005] 
[L2006] 
[L2007] def train_model(num_epochs=8, lr=1e-3):
[L2008]     train_loader, test_loader = get_train_test_loaders()
[L2009] 
[L2010]     model = MLPRouteB().to(DEVICE)
[L2011]     optimizer = optim.Adam(model.parameters(), lr=lr)
[L2012]     criterion = nn.CrossEntropyLoss()
[L2013] 
[L2014]     for epoch in range(num_epochs):
[L2015]         model.train()
[L2016]         running_loss = 0.0
[L2017] 
[L2018]         for data, target in train_loader:
[L2019]             data, target = data.to(DEVICE), target.to(DEVICE)
[L2020] 
[L2021]             optimizer.zero_grad()
[L2022]             logits = model(data)
[L2023]             loss = criterion(logits, target)
[L2024]             loss.backward()
[L2025]             optimizer.step()
[L2026] 
[L2027]             running_loss += loss.item() * data.size(0)
[L2028] 
[L2029]         avg_loss = running_loss / len(train_loader.dataset)
[L2030]         acc = evaluate_model(model, test_loader)
[L2031]         print(f"[Epoch {epoch+1}/{num_epochs}] loss={avg_loss:.6f}, test_acc={acc:.4f}")
[L2032] 
[L2033]     torch.save(model.state_dict(), MODEL_PATH)
[L2034]     print(f"Model saved to: {MODEL_PATH}")
[L2035]     return model
[L2036] 
[L2037] 
[L2038] # =========================================================
[L2039] # 5. 量化工具
[L2040] # =========================================================
[L2041] def clamp(v, low, high):
[L2042]     return max(low, min(high, v))
[L2043] 
[L2044] def quantize_affine_tensor(tensor_fp, scale, zero_point, qmin=-128, qmax=127, dtype=torch.int8):
[L2045]     q = torch.round(tensor_fp / scale + zero_point)
[L2046]     q = torch.clamp(q, qmin, qmax)
[L2047]     return q.to(dtype)
[L2048] 
[L2049] def calc_symmetric_scale(tensor_fp, num_bits=8):
[L2050]     qmax = (2 ** (num_bits - 1)) - 1
[L2051]     max_abs = max(abs(tensor_fp.min().item()), abs(tensor_fp.max().item()))
[L2052]     if max_abs == 0:
[L2053]         return 1.0
[L2054]     return max_abs / qmax
[L2055] 
[L2056] def calc_affine_params_from_range(min_val, max_val, num_bits=8, signed=True):
[L2057]     if signed:
[L2058]         qmin = -(2 ** (num_bits - 1))
[L2059]         qmax =  (2 ** (num_bits - 1)) - 1
[L2060]     else:
[L2061]         qmin = 0
[L2062]         qmax = (2 ** num_bits) - 1
[L2063] 
[L2064]     if max_val == min_val:
[L2065]         scale = 1.0
[L2066]         zero_point = 0
[L2067]         return scale, zero_point
[L2068] 
[L2069]     scale = (max_val - min_val) / float(qmax - qmin)
[L2070]     zero_point = round(qmin - min_val / scale)
[L2071]     zero_point = int(clamp(zero_point, qmin, qmax))
[L2072]     return float(scale), int(zero_point)
[L2073] 
[L2074] def quantize_weight_symmetric_int8(weight_fp):
[L2075]     scale = calc_symmetric_scale(weight_fp, num_bits=8)
[L2076]     zp = 0
[L2077]     q = quantize_affine_tensor(weight_fp, scale, zp, INT8_QMIN, INT8_QMAX, torch.int8)
[L2078]     return q, scale, zp
[L2079] 
[L2080] def quantize_bias_int32(bias_fp, input_scale, weight_scale):
[L2081]     bias_scale = input_scale * weight_scale
[L2082]     q = torch.round(bias_fp / bias_scale)
[L2083]     q = torch.clamp(q, INT32_QMIN, INT32_QMAX).to(torch.int32)
[L2084]     return q, bias_scale
[L2085] 
[L2086] def quantize_input_hw_style(img_fp_01):
[L2087]     """
[L2088]     固定输入量化：
[L2089]         x_u8 = round(x * 255)
[L2090]         x_q  = x_u8 - 128
[L2091]     对应：
[L2092]         scale = 1/255
[L2093]         zero_point = -128
[L2094]     """
[L2095]     scale = 1.0 / 255.0
[L2096]     zp = -128
[L2097] 
[L2098]     x_u8 = torch.round(img_fp_01 * 255.0)
[L2099]     x_u8 = torch.clamp(x_u8, 0, 255)
[L2100]     x_q = x_u8.to(torch.int16) - 128
[L2101]     x_q = torch.clamp(x_q, -128, 127).to(torch.int8)
[L2102]     return x_q, scale, zp
[L2103] 
[L2104] def choose_output_scale_from_activation(act_fp):
[L2105]     """
[L2106]     输出激活采用对称量化到 int8
[L2107]     """
[L2108]     scale = calc_symmetric_scale(act_fp, num_bits=8)
[L2109]     zp = 0
[L2110]     return scale, zp
[L2111] 
[L2112] def requantize_from_int32(acc_int32, real_multiplier, out_zero_point=0):
[L2113]     """
[L2114]     简单软件参考：
[L2115]         y_q = round(acc * real_multiplier) + out_zero_point
[L2116]     """
[L2117]     y = torch.round(acc_int32.to(torch.float32) * real_multiplier) + out_zero_point
[L2118]     y = torch.clamp(y, INT8_QMIN, INT8_QMAX)
[L2119]     return y.to(torch.int8)
[L2120] 
[L2121] def quantize_multiplier_to_int(real_multiplier):
[L2122]     """
[L2123]     给 FPGA 做定点乘法时可用的近似参数：
[L2124]       real_multiplier ≈ M / 2^shift
[L2125]     返回:
[L2126]       M(int), shift(int)
[L2127]     这里只提供给配置文件参考；软件 golden 仍直接用 float real_multiplier
[L2128]     """
[L2129]     if real_multiplier == 0:
[L2130]         return 0, 0
[L2131] 
[L2132]     shift = 0
[L2133]     m = real_multiplier
[L2134]     while m < 0.5:
[L2135]         m *= 2.0
[L2136]         shift += 1
[L2137]     M = int(round(m * (1 << 31)))  # Q31
[L2138]     return M, shift + 31
[L2139] 
[L2140] 
[L2141] # =========================================================
[L2142] # 6. 量化模型准备
[L2143] # =========================================================
[L2144] def build_quantized_model(model_fp):
[L2145]     model_fp.eval()
[L2146]     model_fp = model_fp.cpu()
[L2147] 
[L2148]     # 固定输入量化参数
[L2149]     input_scale = 1.0 / 255.0
[L2150]     input_zero_point = -128
[L2151] 
[L2152]     # fc1 量化
[L2153]     fc1_w_fp = model_fp.fc1.weight.data
[L2154]     fc1_b_fp = model_fp.fc1.bias.data
[L2155]     fc1_w_q, fc1_w_scale, fc1_w_zp = quantize_weight_symmetric_int8(fc1_w_fp)
[L2156]     fc1_b_q, fc1_b_scale = quantize_bias_int32(fc1_b_fp, input_scale, fc1_w_scale)
[L2157] 
[L2158]     # 用一批训练样本统计 fc1 输出尺度
[L2159]     calib_loader, _ = get_train_test_loaders(batch_size=256)
[L2160]     calib_data, _ = next(iter(calib_loader))
[L2161]     calib_data = calib_data.view(-1, 784)
[L2162]     with torch.no_grad():
[L2163]         fc1_act_fp = model_fp.relu(model_fp.fc1(calib_data))
[L2164]     fc1_out_scale, fc1_out_zp = choose_output_scale_from_activation(fc1_act_fp)
[L2165] 
[L2166]     fc1_real_multiplier = (input_scale * fc1_w_scale) / fc1_out_scale
[L2167]     fc1_M, fc1_shift = quantize_multiplier_to_int(fc1_real_multiplier)
[L2168] 
[L2169]     # fc2 量化
[L2170]     fc2_w_fp = model_fp.fc2.weight.data
[L2171]     fc2_b_fp = model_fp.fc2.bias.data
[L2172]     fc2_w_q, fc2_w_scale, fc2_w_zp = quantize_weight_symmetric_int8(fc2_w_fp)
[L2173]     fc2_b_q, fc2_b_scale = quantize_bias_int32(fc2_b_fp, fc1_out_scale, fc2_w_scale)
[L2174] 
[L2175]     # 用一批样本统计 fc2 输出尺度
[L2176]     with torch.no_grad():
[L2177]         logits_fp = model_fp(calib_data.view(-1, 1, 28, 28))
[L2178]     fc2_out_scale, fc2_out_zp = choose_output_scale_from_activation(logits_fp)
[L2179] 
[L2180]     fc2_real_multiplier = (fc1_out_scale * fc2_w_scale) / fc2_out_scale
[L2181]     fc2_M, fc2_shift = quantize_multiplier_to_int(fc2_real_multiplier)
[L2182] 
[L2183]     qparams = {
[L2184]         "input": {
[L2185]             "scale": input_scale,
[L2186]             "zero_point": input_zero_point,
[L2187]             "dtype": "int8"
[L2188]         },
[L2189]         "fc1": {
[L2190]             "weight_scale": fc1_w_scale,
[L2191]             "weight_zero_point": fc1_w_zp,
[L2192]             "bias_scale": fc1_b_scale,
[L2193]             "output_scale": fc1_out_scale,
[L2194]             "output_zero_point": fc1_out_zp,
[L2195]             "real_multiplier": fc1_real_multiplier,
[L2196]             "multiplier_q31": fc1_M,
[L2197]             "shift": fc1_shift
[L2198]         },
[L2199]         "fc2": {
[L2200]             "weight_scale": fc2_w_scale,
[L2201]             "weight_zero_point": fc2_w_zp,
[L2202]             "bias_scale": fc2_b_scale,
[L2203]             "output_scale": fc2_out_scale,
[L2204]             "output_zero_point": fc2_out_zp,
[L2205]             "real_multiplier": fc2_real_multiplier,
[L2206]             "multiplier_q31": fc2_M,
[L2207]             "shift": fc2_shift
[L2208]         }
[L2209]     }
[L2210] 
[L2211]     quantized = {
[L2212]         "fc1_w_q": fc1_w_q.to(torch.int8),
[L2213]         "fc1_b_q": fc1_b_q.to(torch.int32),
[L2214]         "fc2_w_q": fc2_w_q.to(torch.int8),
[L2215]         "fc2_b_q": fc2_b_q.to(torch.int32),
[L2216]         "qparams": qparams
[L2217]     }
[L2218]     return quantized
[L2219] 
[L2220] 
[L2221] # =========================================================
[L2222] # 7. 整数推理参考实现
[L2223] # =========================================================
[L2224] def linear_int8_int32(x_q, w_q, b_q, x_zp, w_zp):
[L2225]     """
[L2226]     x_q: [in_features] int8
[L2227]     w_q: [out_features, in_features] int8
[L2228]     b_q: [out_features] int32
[L2229]     """
[L2230]     x_int = x_q.to(torch.int32) - int(x_zp)
[L2231]     w_int = w_q.to(torch.int32) - int(w_zp)
[L2232]     acc = torch.matmul(w_int, x_int) + b_q
[L2233]     return acc.to(torch.int32)
[L2234] 
[L2235] def relu_int32(x):
[L2236]     return torch.clamp(x, min=0).to(torch.int32)
[L2237] 
[L2238] def integer_inference_reference(img_fp_01, quantized):
[L2239]     """
[L2240]     完整整数参考链路：
[L2241]       input -> fc1(int32 acc) -> relu -> requant to int8
[L2242]             -> fc2(int32 acc) -> requant to int8 logits
[L2243]     """
[L2244]     qparams = quantized["qparams"]
[L2245] 
[L2246]     # 1) 输入量化
[L2247]     x_q, _, _ = quantize_input_hw_style(img_fp_01)
[L2248] 
[L2249]     # 2) fc1 int32 累加
[L2250]     fc1_acc = linear_int8_int32(
[L2251]         x_q,
[L2252]         quantized["fc1_w_q"],
[L2253]         quantized["fc1_b_q"],
[L2254]         x_zp=qparams["input"]["zero_point"],
[L2255]         w_zp=qparams["fc1"]["weight_zero_point"]
[L2256]     )
[L2257] 
[L2258]     # 3) ReLU
[L2259]     fc1_relu = relu_int32(fc1_acc)
[L2260] 
[L2261]     # 4) fc1 输出 requant -> int8
[L2262]     fc1_out_q = requantize_from_int32(
[L2263]         fc1_relu,
[L2264]         qparams["fc1"]["real_multiplier"],
[L2265]         qparams["fc1"]["output_zero_point"]
[L2266]     )
[L2267] 
[L2268]     # 5) fc2 int32 累加
[L2269]     fc2_acc = linear_int8_int32(
[L2270]         fc1_out_q,
[L2271]         quantized["fc2_w_q"],
[L2272]         quantized["fc2_b_q"],
[L2273]         x_zp=qparams["fc1"]["output_zero_point"],
[L2274]         w_zp=qparams["fc2"]["weight_zero_point"]
[L2275]     )
[L2276] 
[L2277]     # 6) 输出 requant -> int8 logits
[L2278]     logits_q = requantize_from_int32(
[L2279]         fc2_acc,
[L2280]         qparams["fc2"]["real_multiplier"],
[L2281]         qparams["fc2"]["output_zero_point"]
[L2282]     )
[L2283] 
[L2284]     pred = int(torch.argmax(logits_q).item())
[L2285] 
[L2286]     return {
[L2287]         "input_q": x_q,
[L2288]         "fc1_acc_int32": fc1_acc,
[L2289]         "fc1_relu_int32": fc1_relu,
[L2290]         "fc1_out_q": fc1_out_q,
[L2291]         "fc2_acc_int32": fc2_acc,
[L2292]         "logits_q": logits_q,
[L2293]         "pred": pred
[L2294]     }
[L2295] 
[L2296] 
[L2297] # =========================================================
[L2298] # 8. HEX 导出
[L2299] # =========================================================
[L2300] def write_int8_hex_1d(tensor, path):
[L2301]     with open(path, "w") as f:
[L2302]         for v in tensor.flatten():
[L2303]             f.write(f"{int(v.item()) & 0xFF:02x}\n")
[L2304] 
[L2305] def write_int32_hex_1d(tensor, path):
[L2306]     with open(path, "w") as f:
[L2307]         for v in tensor.flatten():
[L2308]             f.write(f"{int(v.item()) & 0xFFFFFFFF:08x}\n")
[L2309] 
[L2310] def export_all_artifacts(model_fp, quantized, num_samples=20):
[L2311]     # 1) 导出权重 / bias
[L2312]     # 按 [out][in] 顺序导出
[L2313]     write_int8_hex_1d(quantized["fc1_w_q"].reshape(-1), os.path.join(OUT_DIR, "fc1_weight_int8.hex"))
[L2314]     write_int32_hex_1d(quantized["fc1_b_q"], os.path.join(OUT_DIR, "fc1_bias_int32.hex"))
[L2315] 
[L2316]     write_int8_hex_1d(quantized["fc2_w_q"].reshape(-1), os.path.join(OUT_DIR, "fc2_weight_int8.hex"))
[L2317]     write_int32_hex_1d(quantized["fc2_b_q"], os.path.join(OUT_DIR, "fc2_bias_int32.hex"))
[L2318] 
[L2319]     # 2) 保存量化配置
[L2320]     config = {
[L2321]         "network": "784 -> 128 -> 10",
[L2322]         "weight_layout": {
[L2323]             "fc1": "row-major [out][in] = [128][784]",
[L2324]             "fc2": "row-major [out][in] = [10][128]"
[L2325]         },
[L2326]         "input_layout": "[784]",
[L2327]         "hidden_layout": "[128]",
[L2328]         "output_layout": "[10]",
[L2329]         "qparams": quantized["qparams"]
[L2330]     }
[L2331] 
[L2332]     with open(os.path.join(OUT_DIR, "quant_config.json"), "w", encoding="utf-8") as f:
[L2333]         json.dump(config, f, indent=2, ensure_ascii=False)
[L2334] 
[L2335]     # 3) 导出样本和 golden
[L2336]     test_transform = transforms.Compose([
[L2337]         transforms.ToTensor()
[L2338]     ])
[L2339]     test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=test_transform)
[L2340] 
[L2341]     labels = []
[L2342]     preds = []
[L2343] 
[L2344]     for i in range(num_samples):
[L2345]         img_fp, label = test_dataset[i]
[L2346]         img_flat = img_fp.view(-1).cpu()
[L2347] 
[L2348]         result = integer_inference_reference(img_flat, quantized)
[L2349] 
[L2350]         labels.append(int(label))
[L2351]         preds.append(int(result["pred"]))
[L2352] 
[L2353]         write_int8_hex_1d(result["input_q"], os.path.join(OUT_DIR, f"input_{i}.hex"))
[L2354] 
[L2355]         write_int32_hex_1d(result["fc1_acc_int32"], os.path.join(OUT_DIR, f"fc1_acc_{i}.hex"))
[L2356]         write_int32_hex_1d(result["fc1_relu_int32"], os.path.join(OUT_DIR, f"fc1_relu_{i}.hex"))
[L2357]         write_int8_hex_1d(result["fc1_out_q"], os.path.join(OUT_DIR, f"fc1_out_{i}.hex"))
[L2358] 
[L2359]         write_int32_hex_1d(result["fc2_acc_int32"], os.path.join(OUT_DIR, f"fc2_acc_{i}.hex"))
[L2360]         write_int8_hex_1d(result["logits_q"], os.path.join(OUT_DIR, f"logits_{i}.hex"))
[L2361] 
[L2362]         with open(os.path.join(OUT_DIR, f"pred_{i}.txt"), "w") as f:
[L2363]             f.write(str(result["pred"]))
[L2364] 
[L2365]         print(f"sample={i}, label={label}, pred={result['pred']}")
[L2366] 
[L2367]     with open(os.path.join(OUT_DIR, "labels.txt"), "w") as f:
[L2368]         for lb in labels:
[L2369]             f.write(f"{lb}\n")
[L2370] 
[L2371]     with open(os.path.join(OUT_DIR, "preds.txt"), "w") as f:
[L2372]         for pd in preds:
[L2373]             f.write(f"{pd}\n")
[L2374] 
[L2375]     print("All artifacts exported.")
[L2376] 
[L2377] 
[L2378] # =========================================================
[L2379] # 9. 验证量化后整数推理精度
[L2380] # =========================================================
[L2381] def evaluate_integer_pipeline(model_fp, quantized, num_samples=1000):
[L2382]     test_transform = transforms.Compose([
[L2383]         transforms.ToTensor()
[L2384]     ])
[L2385]     test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=test_transform)
[L2386] 
[L2387]     correct = 0
[L2388]     total = min(num_samples, len(test_dataset))
[L2389] 
[L2390]     for i in range(total):
[L2391]         img_fp, label = test_dataset[i]
[L2392]         img_flat = img_fp.view(-1).cpu()
[L2393]         result = integer_inference_reference(img_flat, quantized)
[L2394]         if result["pred"] == int(label):
[L2395]             correct += 1
[L2396] 
[L2397]     acc = correct / total
[L2398]     print(f"Integer pipeline accuracy over {total} samples: {acc:.4f}")
[L2399]     return acc
[L2400] 
[L2401] 
[L2402] # =========================================================
[L2403] # 10. 主流程
[L2404] # =========================================================
[L2405] if __name__ == "__main__":
[L2406]     # 1) 训练浮点模型
[L2407]     model_fp = train_model(num_epochs=8, lr=1e-3)
[L2408] 
[L2409]     # 2) 加载模型（防止单独运行导出时也能用）
[L2410]     model_fp = MLPRouteB().to(DEVICE)
[L2411]     model_fp.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
[L2412]     model_fp.eval()
[L2413] 
[L2414]     # 3) 构建量化模型
[L2415]     quantized = build_quantized_model(model_fp)
[L2416] 
[L2417]     # 4) 评估整数链路
[L2418]     evaluate_integer_pipeline(model_fp, quantized, num_samples=1000)
[L2419] 
[L2420]     # 5) 导出 FPGA 所需文件
[L2421]     export_all_artifacts(model_fp, quantized, num_samples=20)
[L2422] ```
[L2423] 
[L2424] ---
[L2425] 
[L2426] # 五、关于问题 2：逐项回答
[L2427] 
[L2428] ---
[L2429] 
[L2430] ## 2.1 `zero_point` 不是整数，怎么改
[L2431] 
[L2432] 你原来的问题在这里：
[L2433] 
[L2434] ```python
[L2435] zero_point = qmin - min_val / scale
[L2436] ```
[L2437] 
[L2438] 这里会得到浮点数。应改成：
[L2439] 
[L2440] ```python
[L2441] zero_point = round(qmin - min_val / scale)
[L2442] zero_point = int(max(qmin, min(qmax, zero_point)))
[L2443] ```
[L2444] 
[L2445] 通用函数我已经在路线 B 里给了：
[L2446] 
[L2447] ```python
[L2448] def calc_affine_params_from_range(min_val, max_val, num_bits=8, signed=True):
[L2449]     if signed:
[L2450]         qmin = -(2 ** (num_bits - 1))
[L2451]         qmax =  (2 ** (num_bits - 1)) - 1
[L2452]     else:
[L2453]         qmin = 0
[L2454]         qmax = (2 ** num_bits) - 1
[L2455] 
[L2456]     if max_val == min_val:
[L2457]         scale = 1.0
[L2458]         zero_point = 0
[L2459]         return scale, zero_point
[L2460] 
[L2461]     scale = (max_val - min_val) / float(qmax - qmin)
[L2462]     zero_point = round(qmin - min_val / scale)
[L2463]     zero_point = int(clamp(zero_point, qmin, qmax))
[L2464]     return float(scale), int(zero_point)
[L2465] ```
[L2466] 
[L2467] ---
[L2468] 
[L2469] ## 2.2 每个张量动态量化不适合 FPGA，合理解决方案
[L2470] 
[L2471] ### 你现在的问题
[L2472] 你现在是：
[L2473] - 每张输入图像单独量化
[L2474] - 每次输出也单独量化
[L2475] 
[L2476] 这会导致 FPGA 需要不断切换 scale / zero_point，不利于验证和部署。
[L2477] 
[L2478] ### 合理方案
[L2479] 改为**静态量化**：
[L2480] 
[L2481] #### 输入
[L2482] 固定：
[L2483] - `input_scale = 1/255`
[L2484] - `input_zero_point = -128`
[L2485] 
[L2486] #### 权重
[L2487] 每层固定：
[L2488] - `weight_scale` 固定
[L2489] - `weight_zero_point = 0`
[L2490] 
[L2491] #### 输出
[L2492] 每层固定：
[L2493] - `output_scale` 固定
[L2494] - `output_zero_point` 固定
[L2495] 
[L2496] 这样每个样本都共用同一套参数，硬件最容易做。
[L2497] 
[L2498] ---
[L2499] 
[L2500] ## 2.3 偏置不能用 int8，怎么解决
[L2501] 
[L2502] ### 正确原则
[L2503] bias 必须与 int32 累加器同量纲：
[L2504] 
[L2505] \[
[L2506] b_q = round(\frac{b}{s_x \cdot s_w})
[L2507] \]
[L2508] 
[L2509] ### 代码
[L2510] 我已经写成：
[L2511] 
[L2512] ```python
[L2513] def quantize_bias_int32(bias_fp, input_scale, weight_scale):
[L2514]     bias_scale = input_scale * weight_scale
[L2515]     q = torch.round(bias_fp / bias_scale)
[L2516]     q = torch.clamp(q, INT32_QMIN, INT32_QMAX).to(torch.int32)
[L2517]     return q, bias_scale
[L2518] ```
[L2519] 
[L2520] ### 为什么要这样
[L2521] 因为硬件里先做：
[L2522] 
[L2523] \[
[L2524] acc = \sum (x_q-z_x)(w_q-z_w)
[L2525] \]
[L2526] 
[L2527] 这个 `acc` 是 int32 量纲。  
[L2528] 如果 bias 还是 int8，直接相加就是错的。
[L2529] 
[L2530] ---
[L2531] 
[L2532] # 六、关于问题 3：整数推理链路 golden，怎么设计
[L2533] 
[L2534] 你要求“整数推理链路生成 golden”，现在路线 B 已经完整实现了。
[L2535] 
[L2536] ---
[L2537] 
[L2538] ## 设计链路
[L2539] 
[L2540] ### 第一层
[L2541] \[
[L2542] acc_1 = W_1 \cdot (x_q-z_x) + b_{1q}
[L2543] \]
[L2544] 
[L2545] \[
[L2546] relu_1 = max(acc_1, 0)
[L2547] \]
[L2548] 
[L2549] \[
[L2550] h_q = requant(relu_1)
[L2551] \]
[L2552] 
[L2553] ### 第二层
[L2554] \[
[L2555] acc_2 = W_2 \cdot (h_q-z_{h}) + b_{2q}
[L2556] \]
[L2557] 
[L2558] \[
[L2559] logits_q = requant(acc_2)
[L2560] \]
[L2561] 
[L2562] ### 分类
[L2563] \[
[L2564] pred = argmax(logits_q)
[L2565] \]
[L2566] 
[L2567] ---
[L2568] 
[L2569] ## 代码入口
[L2570] 整数参考实现就在这里：
[L2571] 
[L2572] ```python
[L2573] def integer_inference_reference(img_fp_01, quantized):
[L2574]     ...
[L2575] ```
[L2576] 
[L2577] 它导出了：
[L2578] 
[L2579] - `input_q`
[L2580] - `fc1_acc_int32`
[L2581] - `fc1_relu_int32`
[L2582] - `fc1_out_q`
[L2583] - `fc2_acc_int32`
[L2584] - `logits_q`
[L2585] - `pred`
[L2586] 
[L2587] 这些就是你硬件联调最有价值的 golden。
[L2588] 
[L2589] ---
[L2590] 
[L2591] # 七、关于问题 4：统一输入量化策略
[L2592] 
[L2593] 你要求“贴近硬件即可”，我建议你就用下面这个，不要再 Normalize。
[L2594] 
[L2595] ---
[L2596] 
[L2597] ## 推荐方案
[L2598] MNIST 原始像素 `[0,1]`
[L2599] 
[L2600] 统一做：
[L2601] 
[L2602] ```python
[L2603] x_u8 = round(x * 255)
[L2604] x_q = x_u8 - 128
[L2605] ```
[L2606] 
[L2607] 对应：
[L2608] 
[L2609] - `input_scale = 1/255`
[L2610] - `input_zero_point = -128`
[L2611] 
[L2612] ---
[L2613] 
[L2614] ## 优点
[L2615] ### 软件上
[L2616] - 简单
[L2617] - 稳定
[L2618] - 不依赖每张图的分布
[L2619] 
[L2620] ### 硬件上
[L2621] - BRAM 存储直观
[L2622] - 每个像素 8 bit
[L2623] - 乘加前减去 `-128` 很明确
[L2624] - 适合 testbench 和最终上板
[L2625] 
[L2626] ---
[L2627] 
[L2628] # 八、关于问题 5：你当前输出顺序如何设计硬件读取顺序
[L2629] 
[L2630] 你现在权重导出顺序是：
[L2631] 
[L2632] ```python
[L2633] for row in q_weight:
[L2634]     for val in row:
[L2635]         f.write(...)
[L2636] ```
[L2637] 
[L2638] 而 `nn.Linear(in_features, out_features)` 在 PyTorch 里权重 shape 是：
[L2639] 
[L2640] \[
[L2641] [out\_features, in\_features]
[L2642] \]
[L2643] 
[L2644] 所以你现在导出的顺序是：
[L2645] 
[L2646] \[
[L2647] w[0,0], w[0,1], ..., w[0,783], w[1,0], ..., w[127,783]
[L2648] \]
[L2649] 
[L2650] 也就是：
[L2651] 
[L2652] ## 当前输出顺序
[L2653] **按输出神经元逐行展开，row-major `[out][in]`**
[L2654] 
[L2655] ---
[L2656] 
[L2657] ## 对应的硬件读取顺序应该怎么设计
[L2658] 
[L2659] 如果你的第一层是：
[L2660] 
[L2661] \[
[L2662] y_j = \sum_{i=0}^{783} x_i \cdot w_{j,i} + b_j
[L2663] \]
[L2664] 
[L2665] 那么最自然的硬件读取顺序就是：
[L2666] 
[L2667] ### 方案：按神经元顺序读
[L2668] 对于每个输出神经元 `j`：
[L2669] 
[L2670] 1. 读取该神经元的 784 个权重
[L2671] 2. 与输入 784 维逐项相乘累加
[L2672] 3. 加偏置 `b_j`
[L2673] 4. 输出 `y_j`
[L2674] 
[L2675] ### 地址映射
[L2676] 如果 `fc1_weight_int8.hex` 按 `[out][in]` 展开，则：
[L2677] 
[L2678] \[
[L2679] addr = j \times 784 + i
[L2680] \]
[L2681] 
[L2682] 即：
[L2683] - `j`：输出神经元编号
[L2684] - `i`：输入特征编号
[L2685] 
[L2686] ### RTL 里可写成
[L2687] - 外层循环：`j = 0..127`
[L2688] - 内层循环：`i = 0..783`
[L2689] 
[L2690] 这和你当前导出顺序是完全匹配的。
[L2691] 
[L2692] ---
[L2693] 
[L2694] ## 如果硬件是并行阵列怎么办
[L2695] 如果你做的是 CIM 阵列或 PE 并行，也仍然建议逻辑上保持这个顺序，只是物理存储时可分块：
[L2696] 
[L2697] 例如分成 16 路并行：
[L2698] - 每次读同一个 `j` 的 16 个连续权重
[L2699] - 输入也同步读 16 个元素
[L2700] - 做 16 路乘加
[L2701] - 局部累加后再归并
[L2702] 
[L2703] 本质上还是从 `[out][in]` 顺序切块读取。
[L2704] 
[L2705] ---
[L2706] 
[L2707] ## 第二层同理
[L2708] 第二层 `fc2` shape 是 `[10,128]`
[L2709] 
[L2710] 地址映射：
[L2711] 
[L2712] \[
[L2713] addr = j \times 128 + i
[L2714] \]
[L2715] 
[L2716] 其中：
[L2717] - `j = 0..9`
[L2718] - `i = 0..127`
[L2719] 
[L2720] ---
[L2721] 
[L2722] # 九、问题 6 和 7，我已经一起解决了
[L2723] 
[L2724] ---
[L2725] 
[L2726] ## 问题 6：`quant_params.npz` 没真正参与闭环
[L2727] 现在我不建议继续用 `npz` 做核心配置。  
[L2728] 我已经改成了：
[L2729] 
[L2730] - `quant_config.json`
[L2731] 
[L2732] 里面包含：
[L2733] - 输入 scale / zp
[L2734] - 每层 weight_scale / weight_zp
[L2735] - bias_scale
[L2736] - output_scale / output_zp
[L2737] - real_multiplier
[L2738] - multiplier_q31
[L2739] - shift
[L2740] 
[L2741] 这对硬件更友好，也更好查。
[L2742] 
[L2743] ---
[L2744] 
[L2745] ## 问题 7：其他工程问题
[L2746] 我已经一起处理：
[L2747] 
[L2748] ### 1）`torch.load()` 增加 `map_location`
[L2749] 已改：
[L2750] 
[L2751] ```python
[L2752] torch.load(MODEL_PATH, map_location=DEVICE)
[L2753] ```
[L2754] 
[L2755] ### 2）训练时增加准确率
[L2756] 每个 epoch 后打印 test accuracy
[L2757] 
[L2758] ### 3）保存 label
[L2759] 已导出：
[L2760] - `labels.txt`
[L2761] 
[L2762] ### 4）保存中间层 golden
[L2763] 已导出：
[L2764] - `fc1_acc_x.hex`
[L2765] - `fc1_relu_x.hex`
[L2766] - `fc1_out_x.hex`
[L2767] - `fc2_acc_x.hex`
[L2768] - `logits_x.hex`
[L2769] 
[L2770] 这样 RTL 调试时你能定位是哪一级错了。
[L2771] 
[L2772] ---
[L2773] 
[L2774] # 十、你最终应该选哪套
[L2775] 
[L2776] ## 如果你现在要交阶段成果、先验证阵列
[L2777] 用路线 A。  
[L2778] 因为它最容易把硬件链路先跑通。
[L2779] 
[L2780] ## 如果你现在就想直奔“MNIST 上板分类演示”
[L2781] 用路线 B。  
[L2782] 这就是你刚刚指定的目标，我也已经按这个目标把代码写全了。
[L2783] 
[L2784] ---
[L2785] 
[L2786] # 十一、我建议你下一步的落地顺序
[L2787] 
[L2788] 你现在最稳的执行顺序是：
[L2789] 
[L2790] ### 第一步
[L2791] 先跑路线 B 的 Python 脚本，得到：
[L2792] - 浮点模型精度
[L2793] - 整数量化链路精度
[L2794] - 全套导出文件
[L2795] 
[L2796] ### 第二步
[L2797] 先在 FPGA 上只实现第一层并对齐：
[L2798] - `input`
[L2799] - `fc1_weight`
[L2800] - `fc1_bias`
[L2801] - `fc1_acc`
[L2802] - `fc1_relu`
[L2803] - `fc1_out`
[L2804] 
[L2805] ### 第三步
[L2806] 再补第二层：
[L2807] - `fc2_weight`
[L2808] - `fc2_bias`
[L2809] - `fc2_acc`
[L2810] - `logits`
[L2811] - `argmax`
[L2812] 
[L2813] ---
[L2814] 
[L2815] 如果你愿意，我下一条可以继续直接给你一份**更适合 notebook 使用的版本**，也就是把路线 B 拆成多个单元格，按你现在 `ipynb` 的风格排好。
[L2816] 
[L2817] 用户:
[L2818] 现在已经把导出测试向量的部分写好了，请帮我分析导出的文件的格式，以及如何使用
[L2819] AI助手:
[L2820] 
[L2821] 
[L2822] AI助手:
[L2823] 我先把你当前导出的目录和文件内容抽样检查一下，确认每类文件到底是几位十六进制、按什么维度展开、对应哪一级数据通路。然后我会按“文件格式 → RTL/Testbench 怎么读 → 怎么对拍”的顺序给你整理清楚。
[L2824] 
[L2825] AI助手:
[L2826] 我根据你现在 notebook 里的导出代码，已经能把**导出文件格式**和**使用方法**说清楚了。你当前实际导出的文件主要有这几类：
[L2827] 
[L2828] - `weights.hex`
[L2829] - `bias.hex`
[L2830] - `quant_params.npz`
[L2831] - `input_i.hex`
[L2832] - `golden_i.hex`
[L2833] 
[L2834] 下面我按“是什么、怎么组织、硬件怎么读、联调怎么用”来讲。
[L2835] 
[L2836] ---
[L2837] 
[L2838] # 1. 你现在导出的文件格式
[L2839] 
[L2840] ## 1.1 `weights.hex`
[L2841] 
[L2842] 这是第一层全连接层 `fc1.weight` 的量化权重文件。
[L2843] 
[L2844] ### 来源
[L2845] 代码里：
[L2846] 
[L2847] ```python
[L2848] weight = model.fc1.weight.data  # [128, 784]
[L2849] ...
[L2850] with open(f'{output_dir}/weights.hex', 'w') as f:
[L2851]     for row in q_weight:
[L2852]         for val in row:
[L2853]             f.write(f'{val.item() & 0xFF:02x}\n')
[L2854] ```
[L2855] 
[L2856] ### 文件格式
[L2857] - 文本文件
[L2858] - **每行一个 8 bit 数值**
[L2859] - 用 **2 位十六进制** 表示
[L2860] - 负数采用 **8 位二补码的低 8 位**
[L2861] 
[L2862] 例如：
[L2863] - `00` 表示 0
[L2864] - `01` 表示 1
[L2865] - `7f` 表示 127
[L2866] - `ff` 表示 -1
[L2867] - `80` 表示 -128
[L2868] 
[L2869] ### 总行数
[L2870] 因为权重 shape 是 `[128, 784]`，所以总共有：
[L2871] 
[L2872] \[
[L2873] 128 \times 784 = 100352
[L2874] \]
[L2875] 
[L2876] 也就是：
[L2877] 
[L2878] - `weights.hex` 一共 **100352 行**
[L2879] 
[L2880] ### 展开顺序
[L2881] 当前是：
[L2882] 
[L2883] ```python
[L2884] for row in q_weight:
[L2885]     for val in row:
[L2886] ```
[L2887] 
[L2888] 所以顺序是：
[L2889] 
[L2890] \[
[L2891] w[0,0], w[0,1], ..., w[0,783], w[1,0], ..., w[127,783]
[L2892] \]
[L2893] 
[L2894] 也就是：
[L2895] 
[L2896] - **按输出神经元 row-major 展开**
[L2897] - 格式为 **`[out][in]`**
[L2898] 
[L2899] 这点非常重要，RTL 必须按这个顺序取。
[L2900] 
[L2901] ---
[L2902] 
[L2903] ## 1.2 `bias.hex`
[L2904] 
[L2905] 这是第一层偏置文件。
[L2906] 
[L2907] ### 来源
[L2908] ```python
[L2909] bias = model.fc1.bias.data      # [128]
[L2910] ...
[L2911] with open(f'{output_dir}/bias.hex', 'w') as f:
[L2912]     for val in q_bias:
[L2913]         f.write(f'{val.item() & 0xFF:02x}\n')
[L2914] ```
[L2915] 
[L2916] ### 文件格式
[L2917] - 文本文件
[L2918] - 每行一个量化后的偏置
[L2919] - **2 位十六进制**
[L2920] - 当前代码里它被导成了 **int8**
[L2921] 
[L2922] ### 总行数
[L2923] - 偏置长度是 128
[L2924] - 所以 `bias.hex` 一共 **128 行**
[L2925] 
[L2926] ### 当前问题
[L2927] 这个文件虽然格式上没问题，但**工程上有问题**：
[L2928] 
[L2929] - 你现在是把 bias 当 int8 存
[L2930] - 但硬件 MAC 累加结果通常是 int32
[L2931] - 所以这个 `bias.hex` 现在**不适合直接用于真实整数推理链路**
[L2932] 
[L2933] 也就是说：
[L2934] - 这个文件可以先拿来做“形式演示”
[L2935] - 但不适合做你后面要的最终整数 golden 对拍
[L2936] 
[L2937] ---
[L2938] 
[L2939] ## 1.3 `quant_params.npz`
[L2940] 
[L2941] 这是量化参数文件。
[L2942] 
[L2943] ### 来源
[L2944] ```python
[L2945] np.savez(f'{output_dir}/quant_params.npz',
[L2946]          w_scale=w_scale, w_zero=w_zero,
[L2947]          b_scale=b_scale, b_zero=b_zero)
[L2948] ```
[L2949] 
[L2950] ### 里面有什么
[L2951] 当前保存了：
[L2952] 
[L2953] - `w_scale`
[L2954] - `w_zero`
[L2955] - `b_scale`
[L2956] - `b_zero`
[L2957] 
[L2958] ### 用途
[L2959] 理论上它用于：
[L2960] - 软件端反量化
[L2961] - 硬件端知道量化参数
[L2962] 
[L2963] ### 当前实际情况
[L2964] 你现在的 `generate_test_vectors()` 虽然加载了它：
[L2965] 
[L2966] ```python
[L2967] quant_params = np.load('quantized_data/quant_params.npz')
[L2968] ```
[L2969] 
[L2970] 但后面**几乎没真正用上**。  
[L2971] 所以它现在更像一个“记录文件”，还不是硬件链路真正依赖的配置文件。
[L2972] 
[L2973] ---
[L2974] 
[L2975] ## 1.4 `input_i.hex`
[L2976] 
[L2977] 这是第 `i` 个测试样本的输入文件。
[L2978] 
[L2979] ### 来源
[L2980] ```python
[L2981] img, label = test_dataset[i]
[L2982] img_flat = img.view(-1, 784)
[L2983] q_img, img_scale, img_zero = quantize_tensor(img_flat)
[L2984] 
[L2985] with open(f'quantized_data/input_{i}.hex', 'w') as f:
[L2986]     for val in q_img.flatten():
[L2987]         f.write(f'{val.item() & 0xFF:02x}\n')
[L2988] ```
[L2989] 
[L2990] ### 文件格式
[L2991] - 文本文件
[L2992] - 每行一个输入元素
[L2993] - **2 位十六进制**
[L2994] - 量化后为 int8
[L2995] 
[L2996] ### 总行数
[L2997] MNIST 一张图是 `28×28=784`，所以：
[L2998] 
[L2999] - 每个 `input_i.hex` 一共 **784 行**
[L3000] 
[L3001] ### 排列顺序
[L3002] 因为：
[L3003] 
[L3004] ```python
[L3005] img_flat = img.view(-1, 784)
[L3006] for val in q_img.flatten():
[L3007] ```
[L3008] 
[L3009] 所以顺序就是图像展平后的顺序，通常是：
[L3010] 
[L3011] - 第 0 行：像素 `(0,0)`
[L3012] - 第 1 行：像素 `(0,1)`
[L3013] - ...
[L3014] - 第 27 行：像素 `(0,27)`
[L3015] - 第 28 行：像素 `(1,0)`
[L3016] - ...
[L3017] 
[L3018] 也就是：
[L3019] 
[L3020] - **按行优先 row-major 展开**
[L3021] - 输入向量索引 `i = row * 28 + col`
[L3022] 
[L3023] RTL 里要按这个顺序喂入。
[L3024] 
[L3025] ---
[L3026] 
[L3027] ## 1.5 `golden_i.hex`
[L3028] 
[L3029] 这是第 `i` 个测试样本的软件 golden 输出。
[L3030] 
[L3031] ### 来源
[L3032] ```python
[L3033] with torch.no_grad():
[L3034]     output = model(img_flat)
[L3035] 
[L3036] q_output, _, _ = quantize_tensor(output)
[L3037] with open(f'quantized_data/golden_{i}.hex', 'w') as f:
[L3038]     for val in q_output.flatten():
[L3039]         f.write(f'{val.item() & 0xFF:02x}\n')
[L3040] ```
[L3041] 
[L3042] ### 文件格式
[L3043] - 文本文件
[L3044] - 每行一个输出元素
[L3045] - **2 位十六进制**
[L3046] - 量化后为 int8
[L3047] 
[L3048] ### 总行数
[L3049] 你现在模型是：
[L3050] 
[L3051] ```python
[L3052] self.fc1 = nn.Linear(784,128)
[L3053] self.relu = nn.ReLU()
[L3054] ```
[L3055] 
[L3056] 所以输出是 128 维。
[L3057] 
[L3058] 也就是：
[L3059] - 每个 `golden_i.hex` 一共 **128 行**
[L3060] 
[L3061] ### 当前含义
[L3062] 这个 `golden_i.hex` 表示：
[L3063] 
[L3064] - 先用浮点模型跑出 `fc1 + relu` 的浮点输出
[L3065] - 再把这个浮点输出单独量化成 int8
[L3066] 
[L3067] 所以它的意义是：
[L3068] 
[L3069] - **“浮点模型输出的量化结果”**
[L3070] - 不是严格意义上的“整数硬件链路 golden”
[L3071] 
[L3072] 这点你后面做最终方案时必须记住。
[L3073] 
[L3074] ---
[L3075] 
[L3076] # 2. 这些文件如何使用
[L3077] 
[L3078] ---
[L3079] 
[L3080] ## 2.1 最直接的使用方式：Verilog `$readmemh`
[L3081] 
[L3082] 因为这些 `.hex` 文件是：
[L3083] - 文本
[L3084] - 每行一个十六进制数
[L3085] 
[L3086] 所以最适合直接在 Verilog 里用：
[L3087] 
[L3088] ```verilog
[L3089] reg [7:0] weight_mem [0:100351];
[L3090] reg [7:0] bias_mem   [0:127];
[L3091] reg [7:0] input_mem  [0:783];
[L3092] reg [7:0] golden_mem [0:127];
[L3093] 
[L3094] initial begin
[L3095]     $readmemh("weights.hex", weight_mem);
[L3096]     $readmemh("bias.hex", bias_mem);
[L3097]     $readmemh("input_0.hex", input_mem);
[L3098]     $readmemh("golden_0.hex", golden_mem);
[L3099] end
[L3100] ```
[L3101] 
[L3102] ---
[L3103] 
[L3104] ## 2.2 在 RTL 里如何解释这些 8 bit 数据
[L3105] 
[L3106] ### 注意
[L3107] `$readmemh` 读进来的是 `reg [7:0]`，本身只是无符号位模式。  
[L3108] 如果你想让它参与有符号运算，需要做符号解释。
[L3109] 
[L3110] 例如：
[L3111] 
[L3112] ```verilog
[L3113] wire signed [7:0] w = weight_mem[addr];
[L3114] wire signed [7:0] x = input_mem[idx];
[L3115] wire signed [7:0] b = bias_mem[out_idx];
[L3116] ```
[L3117] 
[L3118] 或者：
[L3119] 
[L3120] ```verilog
[L3121] wire signed [7:0] w = $signed(weight_mem[addr]);
[L3122] ```
[L3123] 
[L3124] 这样：
[L3125] - `8'hff` 才会被解释成 `-1`
[L3126] - `8'h80` 才会被解释成 `-128`
[L3127] 
[L3128] ---
[L3129] 
[L3130] ## 2.3 第一层硬件如何读取 `weights.hex`
[L3131] 
[L3132] 你当前权重顺序是：
[L3133] 
[L3134] \[
[L3135] w[j][i]
[L3136] \]
[L3137] 
[L3138] 其中：
[L3139] - `j = 0..127`，输出神经元
[L3140] - `i = 0..783`，输入索引
[L3141] 
[L3142] 所以硬件最自然的读取方式是：
[L3143] 
[L3144] ### 地址公式
[L3145] \[
[L3146] addr = j \times 784 + i
[L3147] \]
[L3148] 
[L3149] ### 计算流程
[L3150] 对于每个输出神经元 `j`：
[L3151] 
[L3152] 1. 累加器清零
[L3153] 2. 从 `i=0` 到 `783`
[L3154] 3. 读取：
[L3155]    - `x = input_mem[i]`
[L3156]    - `w = weight_mem[j*784+i]`
[L3157] 4. 做乘加：
[L3158]    \[
[L3159]    acc_j += x \cdot w
[L3160]    \]
[L3161] 5. 循环结束后加偏置：
[L3162]    \[
[L3163]    acc_j += bias[j]
[L3164]    \]
[L3165] 6. 过 ReLU
[L3166] 7. 输出第 `j` 个结果
[L3167] 
[L3168] ### 伪代码
[L3169] ```verilog
[L3170] for (j = 0; j < 128; j = j + 1) begin
[L3171]     acc = 0;
[L3172]     for (i = 0; i < 784; i = i + 1) begin
[L3173]         acc = acc + $signed(input_mem[i]) * $signed(weight_mem[j*784+i]);
[L3174]     end
[L3175]     acc = acc + $signed(bias_mem[j]);
[L3176]     if (acc < 0) acc = 0;
[L3177]     out[j] = acc; // 或量化后输出
[L3178] end
[L3179] ```
[L3180] 
[L3181] ---
[L3182] 
[L3183] ## 2.4 `input_i.hex` 如何使用
[L3184] 
[L3185] ### 在 testbench 中
[L3186] 每次测试一个样本：
[L3187] 
[L3188] ```verilog
[L3189] $readmemh("input_0.hex", input_mem);
[L3190] ```
[L3191] 
[L3192] 然后把 `input_mem[0:783]` 依次送给你的计算模块。
[L3193] 
[L3194] ### 两种常见喂法
[L3195] #### 方式 1：一次性 preload 到 BRAM
[L3196] 如果你的设计是：
[L3197] - 输入先装入片上 RAM
[L3198] - 再启动计算
[L3199] 
[L3200] 那就：
[L3201] 1. 先把 `input_i.hex` 用 `$readmemh` 读入 testbench memory
[L3202] 2. 再驱动写接口，把 784 个数据写入 DUT 内部 input RAM
[L3203] 3. 拉高 `start`
[L3204] 
[L3205] #### 方式 2：流式输入
[L3206] 如果你的 DUT 是流式输入：
[L3207] - 每拍输入一个 `x_i`
[L3208] 
[L3209] 那就：
[L3210] 1. 从 `input_mem[0]` 到 `input_mem[783]` 依次送
[L3211] 2. 每个时钟给一个有效数据
[L3212] 3. 送满 784 个后等待结果
[L3213] 
[L3214] ---
[L3215] 
[L3216] ## 2.5 `golden_i.hex` 如何使用
[L3217] 
[L3218] 这是当前 testbench 最直接的对拍基准。
[L3219] 
[L3220] ### 用法
[L3221] 在 testbench 中：
[L3222] 
[L3223] ```verilog
[L3224] $readmemh("golden_0.hex", golden_mem);
[L3225] ```
[L3226] 
[L3227] 当 DUT 计算完成后，把 DUT 的 128 个输出与 `golden_mem` 逐项比对。
[L3228] 
[L3229] ### 比对方式
[L3230] ```verilog
[L3231] integer k;
[L3232] initial begin
[L3233]     for (k = 0; k < 128; k = k + 1) begin
[L3234]         if (dut_out[k] !== golden_mem[k]) begin
[L3235]             $display("Mismatch at %0d: dut=%h, golden=%h", k, dut_out[k], golden_mem[k]);
[L3236]         end
[L3237]     end
[L3238] end
[L3239] ```
[L3240] 
[L3241] ### 但要注意
[L3242] 你现在的 `golden_i.hex` 是：
[L3243] - 浮点输出再量化
[L3244] 
[L3245] 所以前提是你的硬件计算链路也得尽量模拟这个逻辑。  
[L3246] 如果你 RTL 里已经做成了严格整数推理，那这个 golden 就不再是最优基准了。
[L3247] 
[L3248] ---
[L3249] 
[L3250] # 3. 你当前这套文件更适合什么阶段
[L3251] 
[L3252] 这套导出结果，最适合：
[L3253] 
[L3254] ## 适合
[L3255] - 第一层 `fc1 + relu` 的硬件原型验证
[L3256] - BRAM 读写流程验证
[L3257] - 地址映射验证
[L3258] - 符号位解释验证
[L3259] - testbench 对拍基本通路
[L3260] 
[L3261] ## 不适合
[L3262] - 最终的完整 MNIST 分类演示
[L3263] - 严格的整数推理验证
[L3264] - 偏置 int32 链路验证
[L3265] - 静态量化部署闭环
[L3266] 
[L3267] ---
[L3268] 
[L3269] # 4. 推荐你怎么用这批文件做联调
[L3270] 
[L3271] ---
[L3272] 
[L3273] ## 阶段 1：先验证存储与读取
[L3274] 先只做三件事：
[L3275] 
[L3276] 1. `weights.hex` 是否正确加载
[L3277] 2. `input_i.hex` 是否正确加载
[L3278] 3. `golden_i.hex` 是否能和当前 RTL 输出逐项对上
[L3279] 
[L3280] 只要这一步过了，说明：
[L3281] - 文件格式没问题
[L3282] - 读写顺序没问题
[L3283] - 地址映射没问题
[L3284] 
[L3285] ---
[L3286] 
[L3287] ## 阶段 2：确认你的计算顺序与导出顺序一致
[L3288] 你需要确认 RTL 的权重寻址是不是：
[L3289] 
[L3290] \[
[L3291] addr = out\_idx \times 784 + in\_idx
[L3292] \]
[L3293] 
[L3294] 如果不是，就会出现：
[L3295] - 数据都能读到
[L3296] - 结果却完全不对
[L3297] 
[L3298] 这是最常见的问题。
[L3299] 
[L3300] ---
[L3301] 
[L3302] ## 阶段 3：确认 signed 解释
[L3303] 这是第二常见问题。  
[L3304] 如果 RTL 里没有把 `8'hff` 当成 `-1`，而是当成 `255`，结果一定炸。
[L3305] 
[L3306] 所以一定要检查：
[L3307] - 输入 signed
[L3308] - 权重 signed
[L3309] - 偏置 signed
[L3310] - 乘法 signed
[L3311] - 累加器 signed
[L3312] 
[L3313] ---
[L3314] 
[L3315] # 5. 我建议你现在这样理解每个文件
[L3316] 
[L3317] 你可以把当前文件理解成这张表：
[L3318] 
[L3319] | 文件 | 含义 | 数据类型 | 行数 | 作用 |
[L3320] |---|---|---:|---:|---|
[L3321] | `weights.hex` | `fc1` 权重 | int8 | 100352 | DUT 权重存储初始化 |
[L3322] | `bias.hex` | `fc1` 偏置 | int8 | 128 | 当前版本偏置初始化 |
[L3323] | `quant_params.npz` | 量化参数记录 | float/int | 少量字段 | 软件辅助，不是闭环核心 |
[L3324] | `input_i.hex` | 第 i 个输入图像 | int8 | 784 | 测试输入 |
[L3325] | `golden_i.hex` | 第 i 个参考输出 | int8 | 128 | testbench 对拍 |
[L3326] 
[L3327] ---
[L3328] 
[L3329] # 6. 你现在这批文件的“正确使用姿势”
[L3330] 
[L3331] 如果你此刻不改导出逻辑，而是先拿现有文件验证硬件，那建议这样：
[L3332] 
[L3333] ### 你现在的 DUT 目标
[L3334] 做一个：
[L3335] - `784 -> 128`
[L3336] - int8 输入
[L3337] - int8 权重
[L3338] - int8 偏置
[L3339] - 累加
[L3340] - ReLU
[L3341] - 最后输出和 `golden_i.hex` 尽量一致
[L3342] 
[L3343] ### testbench 流程
[L3344] 1. `$readmemh("weights.hex", weight_mem)`
[L3345] 2. `$readmemh("bias.hex", bias_mem)`
[L3346] 3. `$readmemh("input_0.hex", input_mem)`
[L3347] 4. `$readmemh("golden_0.hex", golden_mem)`
[L3348] 5. 把 `input_mem` 送入 DUT
[L3349] 6. DUT 输出 128 维结果
[L3350] 7. 与 `golden_mem` 逐项比较
[L3351] 
[L3352] 这样可以先把系统跑起来。
[L3353] 
[L3354] ---
[L3355] 
[L3356] # 7. 你后面升级到最终方案时，文件会怎么变
[L3357] 
[L3358] 你后面如果按我们前面定的最终方案升级，文件会改成更合理的结构：
[L3359] 
[L3360] - `fc1_weight_int8.hex`
[L3361] - `fc1_bias_int32.hex`
[L3362] - `fc2_weight_int8.hex`
[L3363] - `fc2_bias_int32.hex`
[L3364] - `input_i.hex`
[L3365] - `fc1_acc_i.hex`
[L3366] - `fc1_relu_i.hex`
[L3367] - `fc1_out_i.hex`
[L3368] - `fc2_acc_i.hex`
[L3369] - `logits_i.hex`
[L3370] - `pred_i.txt`
[L3371] - `quant_config.json`
[L3372] 
[L3373] 这时每个文件的职责会更清晰，硬件对拍也更容易定位问题。
[L3374] 
[L3375] ---
[L3376] 
[L3377] # 8. 你现在最需要注意的两个坑
[L3378] 
[L3379] ## 坑 1：`bias.hex` 现在是 int8
[L3380] 这不是最终可部署形式。  
[L3381] 所以你现在用它时，只能把它当作“当前软件逻辑下的配套文件”，不要把它当最终硬件部署格式。
[L3382] 
[L3383] ## 坑 2：`golden_i.hex` 不是整数推理 golden
[L3384] 它现在只是：
[L3385] - 浮点输出
[L3386] - 再量化
[L3387] 
[L3388] 所以如果你 RTL 做得特别“硬件真实”，反而可能和它对不上。
[L3389] 
[L3390] ---
[L3391] 
[L3392] 如果你把**已经导出的几个 `.hex` 文件内容**贴给我几行，或者把你当前 `quantized_data/` 目录再发我，我可以继续帮你做第二步：**逐个文件按实际内容检查格式是否正确，并给你写一个对应的 Verilog testbench 读取模板**。
[L3393] 
[L3394] 用户:
[L3395] bias.hex:000008e5
[L3396] 0000183f
[L3397] 000001a2
[L3398] ffffe6b8
[L3399] fffffa66
[L3400] fffffdd6
[L3401] 00000acd
[L3402] 00000e73
[L3403] fffff423
[L3404] 00000244
[L3405] fffff947
[L3406] 00000cc4
[L3407] fffff784
[L3408] 0000061c
[L3409] ffffed80
[L3410] 00000dd7
[L3411] fffffffe
[L3412] fffff9de
[L3413] fffffb86
[L3414] ffffed26
[L3415] 00000c70
[L3416] 0000116d
[L3417] ffffff7f
[L3418] 000001a8
[L3419] 0000049e
[L3420] 0000118f
[L3421] 000006c6
[L3422] 00000e32
[L3423] 000025f7
[L3424] ffffe677
[L3425] 000005b3
[L3426] 00000b83
[L3427] 00002162
[L3428] 00001ce9
[L3429] 00000409
[L3430] fffffbcc
[L3431] 00000487
[L3432] 00000caa
[L3433] fffff5f9
[L3434] fffff633
[L3435] 000003b1
[L3436] 00000fd1
[L3437] 000005d1
[L3438] 00000ce3
[L3439] 0000106f
[L3440] 00000c5a
[L3441] 0000019f
[L3442] 00000681
[L3443] ffffef2b
[L3444] fffff49a
[L3445] ffffe4d9
[L3446] fffffb73
[L3447] 0000019f
[L3448] 00000dd1
[L3449] 0000016b
[L3450] 00000730
[L3451] fffff9a5
[L3452] 000009db
[L3453] 000008ec
[L3454] 000003c1
[L3455] 00000b59
[L3456] 00001263
[L3457] fffff71a
[L3458] 00001706
[L3459] 000013f9
[L3460] 000001e7
[L3461] 000027ca
[L3462] 00000779
[L3463] 000000d3
[L3464] fffffe21
[L3465] 0000051c
[L3466] 00000e9c
[L3467] 00000c2f
[L3468] fffffc59
[L3469] 0000142d
[L3470] 00000956
[L3471] 00000813
[L3472] fffffa40
[L3473] 00000d2a
[L3474] 00000c81
[L3475] fffffd14
[L3476] 000008ee
[L3477] 0000144a
[L3478] fffff816
[L3479] 00000be6
[L3480] 00000850
[L3481] 000012d6
[L3482] 00001bd6
[L3483] 00000388
[L3484] fffffc00
[L3485] 00001590
[L3486] 00000565
[L3487] 00000adb
[L3488] 00000f82
[L3489] 00000242
[L3490] 00001ed3
[L3491] fffff3bf
[L3492] 00001420
[L3493] 00000b0e
[L3494] 00000ac2
[L3495] fffffefd
[L3496] 00000072
[L3497] fffff4f2
[L3498] fffffc4f
[L3499] 00000904
[L3500] 00000a0f
[L3501] 0000078d
[L3502] 00000cba
[L3503] fffffb5e
[L3504] fffffe7a
[L3505] 00000a64
[L3506] 00000e22
[L3507] 00000435
[L3508] 00000642
[L3509] 00002492
[L3510] 00002235
[L3511] 000011d9
[L3512] fffff921
[L3513] ffffff6a
[L3514] 00001402
[L3515] 00001a36
[L3516] 0000002e
[L3517] 00001f8d
[L3518] ffffff3e
[L3519] fffff936
[L3520] 00000023
[L3521] fffffe93
[L3522] fffff438
[L3523] weight.hex:aa
[L3524] e5
[L3525] 00
[L3526] 0d
[L3527] 0d
[L3528] f7
[L3529] 08
[L3530] e8
[L3531] ff
[L3532] 20
[L3533] 15
[L3534] c5
[L3535] fe
[L3536] fb
[L3537] e2
[L3538] f9
[L3539] 16
[L3540] 15
[L3541] fc
[L3542] 15
[L3543] 09
[L3544] 13
[L3545] 09
[L3546] cb
[L3547] 28
[L3548] ca
[L3549] 0f
[L3550] 15
[L3551] 0f
[L3552] 05
[L3553] 16
[L3554] f3
[L3555] d7
[L3556] f6
[L3557] 06
[L3558] 0c
[L3559] d8
[L3560] 1e
[L3561] ef
[L3562] 03
[L3563] 12
[L3564] 04
[L3565] e6
[L3566] dd
[L3567] d3
[L3568] 0d
[L3569] fa
[L3570] 1c
[L3571] 1c
[L3572] 0f
[L3573] 0d
[L3574] fd
[L3575] 10
[L3576] 04
[L3577] 01
[L3578] b7
[L3579] 0d
[L3580] 00
[L3581] 08
[L3582] 15
[L3583] d2
[L3584] e8
[L3585] 0f
[L3586] f3
[L3587] bd
[L3588] 19
[L3589] e2
[L3590] 1a
[L3591] f1
[L3592] e2
[L3593] f5
[L3594] f2
[L3595] c6
[L3596] ff
[L3597] ec
[L3598] 02
[L3599] d8
[L3600] e9
[L3601] fe
[L3602] 0f
[L3603] 16
[L3604] d8
[L3605] d6
[L3606] 25
[L3607] e2
[L3608] 06
[L3609] bd
[L3610] 24
[L3611] 0d
[L3612] 1c
[L3613] c2
[L3614] 1c
[L3615] ee
[L3616] e0
[L3617] e7
[L3618] ff
[L3619] 05
[L3620] 01
[L3621] 0f
[L3622] ff
[L3623] fa
[L3624] 16
[L3625] 09
[L3626] 08
[L3627] df
[L3628] ff
[L3629] 12
[L3630] d2
[L3631] ea
[L3632] 0f
[L3633] b4
[L3634] 0f
[L3635] fb
[L3636] f4
[L3637] 24
[L3638] 81
[L3639] 13
[L3640] f1
[L3641] 17
[L3642] 0a
[L3643] ed
[L3644] 03
[L3645] f7
[L3646] ed
[L3647] c0
[L3648] e4
[L3649] da
[L3650] 19
[L3651] fb
[L3652] fd
[L3653] fa
[L3654] bf
[L3655] 14
[L3656] 15
[L3657] 17
[L3658] e1
[L3659] df
[L3660] f6
[L3661] f0
[L3662] 1c
[L3663] 10
[L3664] 0c
[L3665] e9
[L3666] cf
[L3667] e0
[L3668] 25
[L3669] eb
[L3670] e2
[L3671] 1e
[L3672] 13
[L3673] e9
[L3674] e9
[L3675] 18
[L3676] 0b
[L3677] dd
[L3678] eb
[L3679] fe
[L3680] ee
[L3681] 10
[L3682] 14
[L3683] 0c
[L3684] 23
[L3685] f1
[L3686] e9
[L3687] d9
[L3688] c1
[L3689] fa
[L3690] fd
[L3691] 16
[L3692] c9
[L3693] 2b
[L3694] 13
[L3695] 10
[L3696] d6
[L3697] 00
[L3698] ea
[L3699] ce
[L3700] e4
[L3701] e1
[L3702] d0
[L3703] ec
[L3704] 0e
[L3705] 0f
[L3706] 13
[L3707] f6
[L3708] f6
[L3709] 12
[L3710] 04
[L3711] 08
[L3712] 1b
[L3713] 0f
[L3714] 0b
[L3715] 02
[L3716] ea
[L3717] d1
[L3718] da
[L3719] e3
[L3720] e4
[L3721] dc
[L3722] 1b
[L3723] d0
[L3724] 18
[L3725] f0
[L3726] 02
[L3727] 48
[L3728] 06
[L3729] 20
[L3730] 2a
[L3731] e4
[L3732] 1b
[L3733] 0e
[L3734] cb
[L3735] 14
[L3736] df
[L3737] 3a
[L3738] 47
[L3739] 0e
[L3740] 2c
[L3741] 02
[L3742] e6
[L3743] 01
[L3744] 21
[L3745] 00
[L3746] f3
[L3747] e4
[L3748] cd
[L3749] c9
[L3750] d5
[L3751] df
[L3752] f1
[L3753] cd
[L3754] ed
[L3755] 18
[L3756] 11
[L3757] 0e
[L3758] 34
[L3759] 16
[L3760] f1
[L3761] 14
[L3762] db
[L3763] fd
[L3764] 0e
[L3765] 34
[L3766] 30
[L3767] df
[L3768] ec
[L3769] de
[L3770] f2
[L3771] cf
[L3772] e1
[L3773] c0
[L3774] fd
[L3775] d6
[L3776] 10
[L3777] 4f
[L3778] 1c
[L3779] 03
[L3780] 25
[L3781] 05
[L3782] fa
[L3783] fd
[L3784] 19
[L3785] 13
[L3786] cb
[L3787] 08
[L3788] 20
[L3789] 03
[L3790] 39
[L3791] 0b
[L3792] 1e
[L3793] ea
[L3794] f3
[L3795] 0b
[L3796] 21
[L3797] ee
[L3798] 16
[L3799] 03
[L3800] af
[L3801] 14
[L3802] ae
[L3803] 07
[L3804] ff
[L3805] f4
[L3806] 0c
[L3807] aa
[L3808] 09
[L3809] e6
[L3810] e9
[L3811] d2
[L3812] 24
[L3813] e8
[L3814] 16
[L3815] 0d
[L3816] f8
[L3817] 14
[L3818] 28
[L3819] fb
[L3820] df
[L3821] 9c
[L3822] 05
[L3823] 18
[L3824] f0
[L3825] 03
[L3826] 45
[L3827] 1e
[L3828] f8
[L3829] 0a
[L3830] fc
[L3831] 04
[L3832] 32
[L3833] 26
[L3834] 0c
[L3835] 1b
[L3836] e6
[L3837] 1a
[L3838] 20
[L3839] ff
[L3840] d9
[L3841] 19
[L3842] b8
[L3843] 12
[L3844] 1b
[L3845] 13
[L3846] 11
[L3847] e8
[L3848] 0b
[L3849] e1
[L3850] 26
[L3851] d4
[L3852] 11
[L3853] e2
[L3854] 92
[L3855] e0
[L3856] fb
[L3857] f5
[L3858] 15
[L3859] 1c
[L3860] 23
[L3861] 15
[L3862] f7
[L3863] 09
[L3864] e4
[L3865] 13
[L3866] dc
[L3867] f9
[L3868] fe
[L3869] f2
[L3870] 14
[L3871] 0a
[L3872] cc
[L3873] eb
[L3874] 02
[L3875] 07
[L3876] fa
[L3877] d1
[L3878] dc
[L3879] 1e
[L3880] 13
[L3881] ec
[L3882] 10
[L3883] 1e
[L3884] 00
[L3885] 1f
[L3886] 13
[L3887] 1a
[L3888] db
[L3889] 16
[L3890] 08
[L3891] 03
[L3892] 0e
[L3893] fe
[L3894] e0
[L3895] 37
[L3896] 0c
[L3897] f0
[L3898] 1a
[L3899] 0e
[L3900] 10
[L3901] 36
[L3902] d2
[L3903] 07
[L3904] fc
[L3905] bb
[L3906] d4
[L3907] db
[L3908] f7
[L3909] 00
[L3910] 0f
[L3911] 0b
[L3912] 0f
[L3913] da
[L3914] d7
[L3915] 08
[L3916] 06
[L3917] 0b
[L3918] 05
[L3919] ff
[L3920] 16
[L3921] 34
[L3922] 02
[L3923] e1
[L3924] 0d
[L3925] 1a
[L3926] 0f
[L3927] 9a
[L3928] dc
[L3929] e8
[L3930] 1e
[L3931] dd
[L3932] 0f
[L3933] f0
[L3934] d8
[L3935] f5
[L3936] 0f
[L3937] f0
[L3938] 05
[L3939] 95
[L3940] c1
[L3941] ec
[L3942] 19
[L3943] 35
[L3944] eb
[L3945] 08
[L3946] 2d
[L3947] 2e
[L3948] db
[L3949] 28
[L3950] fc
[L3951] dd
[L3952] f0
[L3953] f2
[L3954] 12
[L3955] 1a
[L3956] fd
[L3957] 18
[L3958] fa
[L3959] ec
[L3960] cb
[L3961] f7
[L3962] 0a
[L3963] f0
[L3964] 10
[L3965] c4
[L3966] 0a
[L3967] 14
[L3968] e1
[L3969] 0c
[L3970] 03
[L3971] 06
[L3972] 07
[L3973] e9
[L3974] 04
[L3975] 12
[L3976] 0f
[L3977] 08
[L3978] c5
[L3979] 0d
[L3980] fb
[L3981] d1
[L3982] 21
[L3983] dd
[L3984] 13
[L3985] d9
[L3986] ee
[L3987] 0d
[L3988] 09
[L3989] fb
[L3990] 02
[L3991] 00
[L3992] ff
[L3993] e4
[L3994] d3
[L3995] 14
[L3996] f3
[L3997] ce
[L3998] c6
[L3999] 10
[L4000] 1f
[L4001] 13
[L4002] 0d
[L4003] 05
[L4004] 1d
[L4005] d3
[L4006] 22
[L4007] 2b
[L4008] fb
[L4009] 15
[L4010] 07
[L4011] cf
[L4012] 09
[L4013] f3
[L4014] 26
[L4015] 13
[L4016] bd
[L4017] 04
[L4018] e6
[L4019] f6
[L4020] 11
[L4021] f1
[L4022] 46
[L4023] ef
[L4024] 0a
[L4025] f9
[L4026] e6
[L4027] cb
[L4028] 0c
[L4029] eb
[L4030] 1d
[L4031] 16
[L4032] 08
[L4033] 0e
[L4034] 2c
[L4035] e8
[L4036] 08
[L4037] fa
[L4038] ee
[L4039] d2
[L4040] dd
[L4041] 19
[L4042] 1d
[L4043] ce
[L4044] 9c
[L4045] fa
[L4046] a2
[L4047] db
[L4048] 0c
[L4049] c8
[L4050] f5
[L4051] d5
[L4052] ca
[L4053] ea
[L4054] ed
[L4055] 1e
[L4056] 22
[L4057] 19
[L4058] 1f
[L4059] f0
[L4060] 0b
[L4061] 04
[L4062] 07
[L4063] de
[L4064] 04
[L4065] 00
[L4066] 09
[L4067] fc
[L4068] 17
[L4069] 09
[L4070] ec
[L4071] 12
[L4072] c4
[L4073] 8a
[L4074] 0e
[L4075] ec
[L4076] 04
[L4077] 0d
[L4078] e7
[L4079] 12
[L4080] 15
[L4081] 0b
[L4082] 8f
[L4083] d2
[L4084] e7
[L4085] c2
[L4086] 0f
[L4087] f7
[L4088] 12
[L4089] dc
[L4090] 0d
[L4091] e1
[L4092] 18
[L4093] 07
[L4094] d4
[L4095] 08
[L4096] 28
[L4097] ab
[L4098] 0a
[L4099] b4
[L4100] 1c
[L4101] 0b
[L4102] 05
[L4103] 02
[L4104] 0b
[L4105] 11
[L4106] 22
[L4107] 02
[L4108] e9
[L4109] 14
[L4110] e9
[L4111] a7
[L4112] 15
[L4113] bb
[L4114] 08
[L4115] e1
[L4116] fd
[L4117] 00
[L4118] e4
[L4119] f0
[L4120] 0d
[L4121] 23
[L4122] 29
[L4123] ba
[L4124] 2b
[L4125] 09
[L4126] 0c
[L4127] 13
[L4128] 47
[L4129] ff
[L4130] 0a
[L4131] 14
[L4132] 0e
[L4133] 1a
[L4134] f8
[L4135] 97
[L4136] d4
[L4137] e0
[L4138] eb
[L4139] 0a
[L4140] 0f
[L4141] ed
[L4142] 3f
[L4143] 18
[L4144] 15
[L4145] a5
[L4146] 10
[L4147] 07
[L4148] 13
[L4149] 2d
[L4150] 0f
[L4151] 12
[L4152] f6
[L4153] 0d
[L4154] 09
[L4155] 0f
[L4156] 05
[L4157] 08
[L4158] e9
[L4159] 14
[L4160] 11
[L4161] f1
[L4162] bb
[L4163] 38
[L4164] 1f
[L4165] ff
[L4166] 02
[L4167] 05
[L4168] ff
[L4169] e2
[L4170] 38
[L4171] fd
[L4172] 0b
[L4173] 0a
[L4174] 3a
[L4175] ec
[L4176] b2
[L4177] 2d
[L4178] f0
[L4179] db
[L4180] cf
[L4181] ed
[L4182] eb
[L4183] c4
[L4184] 05
[L4185] df
[L4186] 03
[L4187] a5
[L4188] 02
[L4189] 02
[L4190] 12
[L4191] 46
[L4192] 00
[L4193] 0b
[L4194] ee
[L4195] 22
[L4196] e7
[L4197] f0
[L4198] 14
[L4199] ca
[L4200] 0d
[L4201] 2e
[L4202] f1
[L4203] ba
[L4204] 05
[L4205] 2c
[L4206] 0c
[L4207] ec
[L4208] 0d
[L4209] fa
[L4210] 03
[L4211] fb
[L4212] 0b
[L4213] f4
[L4214] e7
[L4215] 12
[L4216] 98
[L4217] c2
[L4218] 08
[L4219] d2
[L4220] 04
[L4221] f3
[L4222] 1b
[L4223] 09
[L4224] f8
[L4225] f9
[L4226] f2
[L4227] 28
[L4228] f9
[L4229] 30
[L4230] e9
[L4231] 06
[L4232] e5
[L4233] f6
[L4234] c2
[L4235] 13
[L4236] f5
[L4237] 14
[L4238] 21
[L4239] 50
[L4240] c4
[L4241] 07
[L4242] e2
[L4243] f3
[L4244] 0d
[L4245] fa
[L4246] f3
[L4247] 0c
[L4248] 0c
[L4249] 08
[L4250] 15
[L4251] 10
[L4252] e9
[L4253] 10
[L4254] 0b
[L4255] 0e
[L4256] 01
[L4257] eb
[L4258] 12
[L4259] ed
[L4260] 0b
[L4261] 0f
[L4262] 10
[L4263] d0
[L4264] 0c
[L4265] 1a
[L4266] fb
[L4267] ca
[L4268] cf
[L4269] f3
[L4270] cb
[L4271] bd
[L4272] ff
[L4273] 36
[L4274] f6
[L4275] fe
[L4276] f8
[L4277] f2
[L4278] 42
[L4279] 02
[L4280] f8
[L4281] 06
[L4282] 12
[L4283] 1d
[L4284] ce
[L4285] 14
[L4286] 0c
[L4287] 17
[L4288] fb
[L4289] 0e
[L4290] ed
[L4291] 17
[L4292] 15
[L4293] 0c
[L4294] 15
[L4295] 02
[L4296] e6
[L4297] 0f
[L4298] 10
[L4299] 06
[L4300] 02
[L4301] 0c
[L4302] b4
[L4303] 06
[L4304] d7
[L4305] af
[L4306] 2f
[L4307] de
[L4308] e1
[L4309] 40
[L4310] ee
[L4311] ef
[L4312] f3
[L4313] 0c
[L4314] aa
[L4315] 21
[L4316] 0a
[L4317] c6
[L4318] fe
[L4319] f3
[L4320] ff
[L4321] 1f
[L4322] e6
[L4323] 3e
[L4324] 08
[L4325] f8
[L4326] d0
[L4327] d8
[L4328] f3
[L4329] 26
[L4330] 06
[L4331] 19
[L4332] 08
[L4333] fc
[L4334] fe
[L4335] 18
[L4336] eb
[L4337] f5
[L4338] d2
[L4339] d5
[L4340] f4
[L4341] e3
[L4342] 18
[L4343] 10
[L4344] f7
[L4345] ab
[L4346] d3
[L4347] f7
[L4348] ed
[L4349] 16
[L4350] c1
[L4351] 04
[L4352] df
[L4353] f6
[L4354] e9
[L4355] 18
[L4356] 11
[L4357] a9
[L4358] da
[L4359] d3
[L4360] f0
[L4361] 04
[L4362] 15
[L4363] d7
[L4364] 05
[L4365] 04
[L4366] 17
[L4367] 17
[L4368] bf
[L4369] 23
[L4370] ee
[L4371] db
[L4372] ec
[L4373] ff
[L4374] 0f
[L4375] df
[L4376] ea
[L4377] fb
[L4378] 1d
[L4379] ef
[L4380] 08
[L4381] ef
[L4382] e2
[L4383] f0
[L4384] 0b
[L4385] 13
[L4386] e4
[L4387] 0e
[L4388] 12
[L4389] 18
[L4390] ff
[L4391] e1
[L4392] 0e
[L4393] fb
[L4394] 10
[L4395] 05
[L4396] b3
[L4397] 16
[L4398] 08
[L4399] dd
[L4400] 2e
[L4401] e0
[L4402] 12
[L4403] fa
[L4404] ee
[L4405] 17
[L4406] ab
[L4407] 16
[L4408] 0a
[L4409] 0b
[L4410] 0f
[L4411] 15
[L4412] e6
[L4413] 23
[L4414] e4
[L4415] cc
[L4416] 17
[L4417] 16
[L4418] 11
[L4419] 0f
[L4420] a8
[L4421] ff
[L4422] c9
[L4423] e8
[L4424] 2b
[L4425] d4
[L4426] 04
[L4427] f7
[L4428] 07
[L4429] e4
[L4430] 1f
[L4431] eb
[L4432] 01
[L4433] 9e
[L4434] 14
[L4435] 1a
[L4436] 01
[L4437] 11
[L4438] 13
[L4439] 25
[L4440] 03
[L4441] 04
[L4442] f1
[L4443] db
[L4444] 09
[L4445] 05
[L4446] db
[L4447] f2
[L4448] d9
[L4449] dc
[L4450] 1a
[L4451] cc
[L4452] 22
[L4453] 14
[L4454] 09
[L4455] 03
[L4456] 08
[L4457] 0b
[L4458] 01
[L4459] 09
[L4460] 0f
[L4461] d5
[L4462] c9
[L4463] d0
[L4464] f2
[L4465] f7
[L4466] 28
[L4467] 08
[L4468] e6
[L4469] 0a
[L4470] e2
[L4471] d2
[L4472] 18
[L4473] 1d
[L4474] 14
[L4475] 00
[L4476] 09
[L4477] 08
[L4478] f9
[L4479] c3
[L4480] e4
[L4481] f7
[L4482] 17
[L4483] e8
[L4484] cd
[L4485] 18
[L4486] 0b
[L4487] 10
[L4488] 0f
[L4489] 0e
[L4490] 14
[L4491] fc
[L4492] 1a
[L4493] 18
[L4494] 09
[L4495] 14
[L4496] 15
[L4497] 09
[L4498] 12
[L4499] 1c
[L4500] da
[L4501] f9
[L4502] d8
[L4503] e0
[L4504] 07
[L4505] f6
[L4506] 0f
[L4507] 14
[L4508] b2
[L4509] e1
[L4510] 13
[L4511] c3
[L4512] e9
[L4513] 05
[L4514] 3a
[L4515] ed
[L4516] 1f
[L4517] f3
[L4518] 20
[L4519] 27
[L4520] 0a
[L4521] c8
[L4522] d8
[L4523] 32
[L4524] 19
[L4525] 27
[L4526] 25
[L4527] dc
[L4528] e2
[L4529] 20
[L4530] ff
[L4531] 04
[L4532] e9
[L4533] 50
[L4534] 00
[L4535] de
[L4536] db
[L4537] ec
[L4538] d6
[L4539] d5
[L4540] 14
[L4541] 0b
[L4542] 06
[L4543] ea
[L4544] d8
[L4545] 21
[L4546] ff
[L4547] f1
[L4548] 1b
[L4549] f3
[L4550] 0b
[L4551] 15
[L4552] 0d
[L4553] 0f
[L4554] 24
[L4555] 01
[L4556] fd
[L4557] 0f
[L4558] e2
[L4559] 13
[L4560] e4
[L4561] ad
[L4562] 02
[L4563] 14
[L4564] 05
[L4565] ce
[L4566] f0
[L4567] e1
[L4568] fe
[L4569] 0f
[L4570] ed
[L4571] 0d
[L4572] 16
[L4573] 05
[L4574] 17
[L4575] cf
[L4576] 14
[L4577] 15
[L4578] 13
[L4579] df
[L4580] c4
[L4581] 10
[L4582] ec
[L4583] 05
[L4584] 14
[L4585] 01
[L4586] b7
[L4587] dd
[L4588] 10
[L4589] c5
[L4590] 0b
[L4591] 02
[L4592] 12
[L4593] 06
[L4594] d9
[L4595] f8
[L4596] 16
[L4597] 14
[L4598] 0b
[L4599] 04
[L4600] 02
[L4601] e5
[L4602] 10
[L4603] 0b
[L4604] 16
[L4605] cc
[L4606] df
[L4607] 17
[L4608] 10
[L4609] 12
[L4610] 0d
[L4611] 0d
[L4612] d4
[L4613] e6
[L4614] cd
[L4615] 10
[L4616] 0c
[L4617] 13
[L4618] b7
[L4619] 0b
[L4620] 01
[L4621] 18
[L4622] fc
[L4623] bb
[L4624] 03
[L4625] 10
[L4626] ce
[L4627] fc
[L4628] 15
[L4629] 18
[L4630] fd
[L4631] 11
[L4632] 0a
[L4633] 14
[L4634] b5
[L4635] 07
[L4636] da
[L4637] 1d
[L4638] db
[L4639] 17
[L4640] ec
[L4641] 08
[L4642] f1
[L4643] 0d
[L4644] ef
[L4645] 03
[L4646] c2
[L4647] ec
[L4648] ee
[L4649] 16
[L4650] 13
[L4651] 14
[L4652] 08
[L4653] ce
[L4654] b3
[L4655] 01
[L4656] dc
[L4657] ee
[L4658] 01
[L4659] f5
[L4660] 12
[L4661] ba
[L4662] 86
[L4663] d5
[L4664] 0b
[L4665] 18
[L4666] fe
[L4667] 22
[L4668] 0b
[L4669] f9
[L4670] 04
[L4671] 15
[L4672] 17
[L4673] e6
[L4674] 01
[L4675] 27
[L4676] ea
[L4677] fc
[L4678] 17
[L4679] ec
[L4680] 99
[L4681] 03
[L4682] ae
[L4683] e0
[L4684] e7
[L4685] ea
[L4686] c3
[L4687] 01
[L4688] 12
[L4689] 34
[L4690] 10
[L4691] 0b
[L4692] d7
[L4693] f2
[L4694] 14
[L4695] 14
[L4696] f3
[L4697] 10
[L4698] 1d
[L4699] 1d
[L4700] ef
[L4701] 16
[L4702] ff
[L4703] 26
[L4704] 0d
[L4705] fa
[L4706] 03
[L4707] 1c
[L4708] f6
[L4709] 12
[L4710] e3
[L4711] 18
[L4712] 11
[L4713] ba
[L4714] d8
[L4715] ef
[L4716] f5
[L4717] 2a
[L4718] fc
[L4719] 11
[L4720] 15
[L4721] 02
[L4722] 16
[L4723] 17
[L4724] 06
[L4725] 13
[L4726] 0b
[L4727] 08
[L4728] fb
[L4729] 14
[L4730] fa
[L4731] 20
[L4732] fe
[L4733] 21
[L4734] 01
[L4735] 0b
[L4736] 20
[L4737] f7
[L4738] 14
[L4739] ec
[L4740] c8
[L4741] 08
[L4742] 16
[L4743] fe
[L4744] 0a
[L4745] 10
[L4746] 1e
[L4747] 09
[L4748] aa
[L4749] 05
[L4750] ec
[L4751] e6
[L4752] fd
[L4753] 04
[L4754] 1b
[L4755] d6
[L4756] bd
[L4757] eb
[L4758] 16
[L4759] 16
[L4760] 16
[L4761] cc
[L4762] 25
[L4763] e7
[L4764] eb
[L4765] 07
[L4766] 17
[L4767] f2
[L4768] ab
[L4769] 10
[L4770] c9
[L4771] f9
[L4772] d5
[L4773] 06
[L4774] ff
[L4775] 16
[L4776] 00
[L4777] 0f
[L4778] ef
[L4779] b6
[L4780] 0d
[L4781] b6
[L4782] ae
[L4783] 03
[L4784] 10
[L4785] e0
[L4786] eb
[L4787] f3
[L4788] e6
[L4789] 9c
[L4790] 09
[L4791] bd
[L4792] 0d
[L4793] e9
[L4794] d2
[L4795] dc
[L4796] 14
[L4797] 97
[L4798] 11
[L4799] 0f
[L4800] 0c
[L4801] 0b
[L4802] 11 先检查这两个有没有问题，然后根据现在的文件和坑，你看看代码有什么需要改进的，最好生成的数据可以被fpga上得cim系统直接使用
[L4803] AI助手:
[L4804] 
[L4805] 
[L4806] AI助手:
[L4807] 我先按“文件内容本身是否自洽”和“是否适合 FPGA/CIM 直接吃”两层来判断。眼下一个明显信号是：你贴出来的 `bias.hex` 已经是 **32 位有符号十六进制**，这比你最初 notebook 里那个 int8 bias 更合理，说明你的导出逻辑应该已经改过一版了。
[L4808] 
[L4809] AI助手:
[L4810] 先说结论：
[L4811] 
[L4812] **你现在贴出来的这两个文件，格式上基本是对的，而且已经比最初 notebook 版本更适合 FPGA/CIM 直接使用。**
[L4813] 
[L4814] 尤其是：
[L4815] 
[L4816] - `bias.hex`：现在是**32 位有符号补码**，这很合理，适合 MAC 后加 bias
[L4817] - `weight.hex`：现在是**8 位有符号补码**，也合理，适合 int8 权重阵列
[L4818] 
[L4819] 这说明你的导出代码已经朝着“真实整数推理链路”改进过了，不再是最初那个 bias 只有 2 位 hex 的版本。
[L4820] 
[L4821] 下面我分三部分说：
[L4822] 
[L4823] 1. 先检查这两个文件有没有明显问题  
[L4824] 2. 结合现在的文件格式，说明 FPGA/CIM 应该怎么直接使用  
[L4825] 3. 按“还能怎么改代码”给你一套更适合直接上板的改进建议
[L4826] 
[L4827] ---
[L4828] 
[L4829] # 1. 先检查这两个文件有没有问题
[L4830] 
[L4831] ---
[L4832] 
[L4833] ## 1.1 `bias.hex` 检查结果
[L4834] 
[L4835] 你给的 `bias.hex` 形如：
[L4836] 
[L4837] ```text
[L4838] 000008e5
[L4839] 0000183f
[L4840] 000001a2
[L4841] ffffe6b8
[L4842] ...
[L4843] ```
[L4844] 
[L4845] ### 这说明什么
[L4846] 每行都是 **8 个十六进制字符**，也就是 **32 bit**。
[L4847] 
[L4848] 这对应的是：
[L4849] 
[L4850] - 正数：直接写成 `000008e5`
[L4851] - 负数：写成二补码形式，比如 `ffffe6b8`
[L4852] 
[L4853] 这正是 FPGA 里最常见的 **signed int32 bias 文件格式**。
[L4854] 
[L4855] ### 这和硬件链路匹配吗
[L4856] 匹配，而且是更合理的做法：
[L4857] 
[L4858] - 输入：int8
[L4859] - 权重：int8
[L4860] - 累加：int32
[L4861] - 偏置：int32
[L4862] - ReLU / requant：后续再做
[L4863] 
[L4864] 所以从工程角度说，**这个 bias 格式是对的**。
[L4865] 
[L4866] ### 我看出来的一个信号
[L4867] 你这份 `bias.hex` 已经不是最开始 notebook 里那种：
[L4868] 
[L4869] ```python
[L4870] f.write(f'{val.item() & 0xFF:02x}\n')
[L4871] ```
[L4872] 
[L4873] 而是明显变成了类似：
[L4874] 
[L4875] ```python
[L4876] f.write(f'{val.item() & 0xFFFFFFFF:08x}\n')
[L4877] ```
[L4878] 
[L4879] 这说明你已经把 bias 导出升级成 int32 了。这个改动是正确的。
[L4880] 
[L4881] ### 需要确认的一点
[L4882] 如果这是 `fc1` 的 bias，那么它应该有：
[L4883] 
[L4884] - **128 行**
[L4885] 
[L4886] 如果这是 `fc2` 的 bias，那么应该有：
[L4887] 
[L4888] - **10 行**
[L4889] 
[L4890] 你贴出来看起来更像是 128 行中的一整段。  
[L4891] 如果你最终网络是两层 MLP，那建议不要只叫 `bias.hex`，而是明确改成：
[L4892] 
[L4893] - `fc1_bias_int32.hex`
[L4894] - `fc2_bias_int32.hex`
[L4895] 
[L4896] 这样不会混。
[L4897] 
[L4898] ---
[L4899] 
[L4900] ## 1.2 `weight.hex` 检查结果
[L4901] 
[L4902] 你给的 `weight.hex` 形如：
[L4903] 
[L4904] ```text
[L4905] aa
[L4906] e5
[L4907] 00
[L4908] 0d
[L4909] 0d
[L4910] f7
[L4911] ...
[L4912] ```
[L4913] 
[L4914] ### 这说明什么
[L4915] 每行都是 **2 个十六进制字符**，也就是 **8 bit**。
[L4916] 
[L4917] 这对应：
[L4918] 
[L4919] - `aa` = -86
[L4920] - `e5` = -27
[L4921] - `00` = 0
[L4922] - `0d` = 13
[L4923] - `f7` = -9
[L4924] 
[L4925] 这正是标准的 **signed int8 权重存储格式**。
[L4926] 
[L4927] ### 是否合理
[L4928] 合理，前提是 RTL 里按 signed 解释：
[L4929] 
[L4930] ```verilog
[L4931] wire signed [7:0] w = $signed(weight_mem[addr]);
[L4932] ```
[L4933] 
[L4934] ### 数值分布上看有没有异常
[L4935] 从你贴的内容看，数值大多集中在：
[L4936] 
[L4937] - `0xC0 ~ 0x40` 左右对应的 -64 到 64 之间
[L4938] - 偶尔有 `0x81`、`0x9c`、`0x8a`、`0x50`、`0x4f`
[L4939] 
[L4940] 这很像正常量化后的神经网络权重分布，不像导出炸掉了。
[L4941] 
[L4942] 也就是说：
[L4943] 
[L4944] - **没有明显出现全 00 / 全 ff / 大面积饱和到 7f/80 的异常**
[L4945] - 看起来像一份正常的量化权重
[L4946] 
[L4947] ### 需要确认的一点
[L4948] 如果这是 `fc1.weight`，而且第一层 shape 是 `[128, 784]`，那它总行数应该是：
[L4949] 
[L4950] \[
[L4951] 128 \times 784 = 100352
[L4952] \]
[L4953] 
[L4954] 如果最终网络是两层 MLP，你也应该拆成：
[L4955] 
[L4956] - `fc1_weight_int8.hex`
[L4957] - `fc2_weight_int8.hex`
[L4958] 
[L4959] 否则后面上板会混。
[L4960] 
[L4961] ---
[L4962] 
[L4963] # 2. 结合你现在的文件格式，FPGA/CIM 怎么直接使用
[L4964] 
[L4965] 现在按你贴出来的格式，**最适合的硬件链路**是：
[L4966] 
[L4967] - `weight.hex`：int8 权重阵列
[L4968] - `bias.hex`：int32 偏置
[L4969] - `input_i.hex`：int8 输入
[L4970] - 累加器：int32
[L4971] - 后面做 ReLU、requant、第二层
[L4972] 
[L4973] 也就是说，你现在已经从“玩具导出格式”走到了“能上硬件”的方向。
[L4974] 
[L4975] ---
[L4976] 
[L4977] ## 2.1 `weight.hex` 的硬件读取方式
[L4978] 
[L4979] 如果你的导出顺序还是之前那样：
[L4980] 
[L4981] ```python
[L4982] for row in q_weight:
[L4983]     for val in row:
[L4984]         ...
[L4985] ```
[L4986] 
[L4987] 那么权重布局就是：
[L4988] 
[L4989] \[
[L4990] W[j][i]
[L4991] \]
[L4992] 
[L4993] 也就是：
[L4994] 
[L4995] - `j`：输出神经元索引
[L4996] - `i`：输入通道索引
[L4997] 
[L4998] ### 地址公式
[L4999] 若第一层 `fc1` shape 是 `[128, 784]`，则：
[L5000] 
[L5001] \[
[L5002] addr = j \times 784 + i
[L5003] \]
[L5004] 
[L5005] RTL 读法就是：
[L5006] 
[L5007] - 外层循环跑输出通道 `j`
[L5008] - 内层循环跑输入通道 `i`
[L5009] 
[L5010] ### 对应计算
[L5011] \[
[L5012] acc_j = \sum_{i=0}^{783} (x_i - z_x)(w_{j,i} - z_w) + b_j
[L5013] \]
[L5014] 
[L5015] 如果你现在是权重对称量化，那通常：
[L5016] 
[L5017] - `z_w = 0`
[L5018] 
[L5019] 如果输入采用 `x_q = pixel_u8 - 128`，那：
[L5020] 
[L5021] - `z_x = -128`
[L5022] 
[L5023] ---
[L5024] 
[L5025] ## 2.2 `bias.hex` 的硬件使用方式
[L5026] 
[L5027] 现在你的 bias 是 32 位二补码，很适合直接这样读：
[L5028] 
[L5029] ```verilog
[L5030] reg [31:0] bias_mem [0:127];
[L5031] initial $readmemh("bias.hex", bias_mem);
[L5032] 
[L5033] wire signed [31:0] bias_val = $signed(bias_mem[j]);
[L5034] ```
[L5035] 
[L5036] 然后在 MAC 完成后：
[L5037] 
[L5038] ```verilog
[L5039] acc <= mac_sum + bias_val;
[L5040] ```
[L5041] 
[L5042] 这比 int8 bias 合理得多。
[L5043] 
[L5044] ---
[L5045] 
[L5046] ## 2.3 对 CIM 系统的直接使用建议
[L5047] 
[L5048] 如果你的 CIM 阵列本体负责的是 `int8 x int8` 乘加，那最常见的系统切分是：
[L5049] 
[L5050] ### CIM 阵列做
[L5051] - 权重存储
[L5052] - 向量输入
[L5053] - 局部乘加 / bitline accumulate / column accumulate
[L5054] 
[L5055] ### FPGA 数字逻辑做
[L5056] - 输入 zero-point 修正
[L5057] - 偏置相加
[L5058] - ReLU
[L5059] - requant
[L5060] - 第二层控制
[L5061] - argmax
[L5062] 
[L5063] 这和你现在的文件格式非常匹配：
[L5064] 
[L5065] - `weight.hex` 给阵列权重
[L5066] - `input_i.hex` 给输入驱动
[L5067] - `bias.hex` 给数字后处理单元
[L5068] - `golden` 给 testbench 对拍
[L5069] 
[L5070] ---
[L5071] 
[L5072] # 3. 这两个文件背后，代码还需要怎么改进
[L5073] 
[L5074] 你现在文件本身没大问题，但为了“**生成的数据可以被 FPGA 上的 CIM 系统直接使用**”，我建议你继续改 8 个点。
[L5075] 
[L5076] ---
[L5077] 
[L5078] ## 改进 1：文件命名必须分层，不能再叫通用名
[L5079] 
[L5080] 你现在如果最终是两层 MLP，绝对不要继续用：
[L5081] 
[L5082] - `weight.hex`
[L5083] - `bias.hex`
[L5084] 
[L5085] 因为你后面会出现：
[L5086] 
[L5087] - 第一层权重
[L5088] - 第二层权重
[L5089] - 第一层 bias
[L5090] - 第二层 bias
[L5091] 
[L5092] 建议直接改成：
[L5093] 
[L5094] - `fc1_weight_int8.hex`
[L5095] - `fc1_bias_int32.hex`
[L5096] - `fc2_weight_int8.hex`
[L5097] - `fc2_bias_int32.hex`
[L5098] 
[L5099] 这样你 RTL 一眼知道读哪个。
[L5100] 
[L5101] ---
[L5102] 
[L5103] ## 改进 2：必须配一个 `quant_config.json`
[L5104] 
[L5105] 现在单靠 hex 文件还不够。  
[L5106] FPGA/CIM 真正要跑整数推理，还需要知道：
[L5107] 
[L5108] - 输入 `scale`
[L5109] - 输入 `zero_point`
[L5110] - 各层 `weight_scale`
[L5111] - 各层 `weight_zero_point`
[L5112] - 各层 `output_scale`
[L5113] - 各层 `output_zero_point`
[L5114] - requant multiplier / shift
[L5115] 
[L5116] 建议额外生成：
[L5117] 
[L5118] ```json
[L5119] {
[L5120]   "input": {
[L5121]     "scale": 0.00392156862745098,
[L5122]     "zero_point": -128
[L5123]   },
[L5124]   "fc1": {
[L5125]     "weight_scale": ...,
[L5126]     "weight_zero_point": 0,
[L5127]     "output_scale": ...,
[L5128]     "output_zero_point": 0,
[L5129]     "bias_scale": ...,
[L5130]     "multiplier_q31": ...,
[L5131]     "shift": ...
[L5132]   },
[L5133]   "fc2": {
[L5134]     "weight_scale": ...,
[L5135]     "weight_zero_point": 0,
[L5136]     "output_scale": ...,
[L5137]     "output_zero_point": 0,
[L5138]     "bias_scale": ...,
[L5139]     "multiplier_q31": ...,
[L5140]     "shift": ...
[L5141]   }
[L5142] }
[L5143] ```
[L5144] 
[L5145] 这样 FPGA 端和 Python 端是一致的。
[L5146] 
[L5147] ---
[L5148] 
[L5149] ## 改进 3：输入文件也要明确 int8 和布局
[L5150] 
[L5151] 你现在最好明确导出成：
[L5152] 
[L5153] - `input_0_int8.hex`
[L5154] - `input_1_int8.hex`
[L5155] 
[L5156] 并且说明：
[L5157] 
[L5158] - 长度 784
[L5159] - 顺序是 row-major
[L5160] - 每行 2 位 hex
[L5161] - 数值是 signed int8
[L5162] - 若采用硬件输入策略，则：
[L5163]   \[
[L5164]   x_q = round(x \cdot 255) - 128
[L5165]   \]
[L5166] 
[L5167] 这能避免你后面忘记输入是“原始像素量化”还是“Normalize 后量化”。
[L5168] 
[L5169] ---
[L5170] 
[L5171] ## 改进 4：golden 不要只保留最终输出，要保留中间级
[L5172] 
[L5173] 这是对硬件调试最关键的一点。
[L5174] 
[L5175] 建议你至少导出：
[L5176] 
[L5177] ### 第一层
[L5178] - `fc1_acc_i.hex`：int32，未激活前
[L5179] - `fc1_relu_i.hex`：int32，过 ReLU 后
[L5180] - `fc1_out_i.hex`：int8，requant 后
[L5181] 
[L5182] ### 第二层
[L5183] - `fc2_acc_i.hex`：int32
[L5184] - `logits_i.hex`：int8
[L5185] 
[L5186] ### 分类
[L5187] - `pred_i.txt`
[L5188] 
[L5189] 这样当 FPGA 结果不对时，你能立刻定位是：
[L5190] 
[L5191] - 阵列乘加错
[L5192] - bias 加错
[L5193] - ReLU 错
[L5194] - requant 错
[L5195] - 第二层地址错
[L5196] 
[L5197] 不然只看最后一个 `golden`，很难找 bug。
[L5198] 
[L5199] ---
[L5200] 
[L5201] ## 改进 5：权重文件要考虑和 CIM 阵列物理映射一致
[L5202] 
[L5203] 你现在的 `weight.hex` 是逻辑顺序 `[out][in]`。  
[L5204] 这对软件和普通 RTL 很方便，但对实际 CIM 阵列，可能还不够。
[L5205] 
[L5206] 因为真实阵列通常不是一次吃 784 个输入，而是分块，例如：
[L5207] 
[L5208] - 每列 64 个输入
[L5209] - 每次做 64×N 子矩阵
[L5210] - 784 被切成很多 tile
[L5211] 
[L5212] 所以你后面很可能需要两套文件：
[L5213] 
[L5214] ### 逻辑文件
[L5215] 给软件 / testbench 用：
[L5216] - `fc1_weight_int8.hex`
[L5217] 
[L5218] ### 物理映射文件
[L5219] 给阵列初始化用：
[L5220] - `fc1_weight_tile0.hex`
[L5221] - `fc1_weight_tile1.hex`
[L5222] - ...
[L5223] 
[L5224] 也就是按 CIM 阵列实际映射切块导出。
[L5225] 
[L5226] 如果你现在阵列设计已经定了，比如每次 64 路输入并行，我建议导出时就按 64 一组打包。
[L5227] 
[L5228] ---
[L5229] 
[L5230] ## 改进 6：给每个文件写 manifest
[L5231] 
[L5232] 建议自动生成一个 `manifest.txt` 或 `manifest.json`，写明：
[L5233] 
[L5234] - 文件名
[L5235] - 数据类型
[L5236] - 位宽
[L5237] - 长度
[L5238] - 布局
[L5239] - 用途
[L5240] 
[L5241] 例如：
[L5242] 
[L5243] ```json
[L5244] {
[L5245]   "fc1_weight_int8.hex": {
[L5246]     "dtype": "int8",
[L5247]     "shape": [128, 784],
[L5248]     "layout": "row-major [out][in]"
[L5249]   },
[L5250]   "fc1_bias_int32.hex": {
[L5251]     "dtype": "int32",
[L5252]     "shape": [128]
[L5253]   },
[L5254]   "input_0_int8.hex": {
[L5255]     "dtype": "int8",
[L5256]     "shape": [784],
[L5257]     "layout": "row-major flattened image"
[L5258]   }
[L5259] }
[L5260] ```
[L5261] 
[L5262] 这会让你的工程后期非常省事。
[L5263] 
[L5264] ---
[L5265] 
[L5266] ## 改进 7：最好支持导出 `.mem` / `.coe` / 打包宽字格式
[L5267] 
[L5268] 现在一行一个数适合 testbench，  
[L5269] 但真正上 FPGA，很多时候你会想直接初始化 BRAM。
[L5270] 
[L5271] 比如：
[L5272] 
[L5273] - Xilinx 可能想要 `.coe`
[L5274] - Intel 可能想要 `.mif`
[L5275] - 或者你会把 16 个 int8 打成 128-bit 宽字
[L5276] 
[L5277] 所以建议你的导出工具支持两种模式：
[L5278] 
[L5279] ### 调试模式
[L5280] - 一行一个元素
[L5281] - 便于看和对拍
[L5282] 
[L5283] ### 上板模式
[L5284] - 打包成 32bit / 64bit / 128bit / 256bit
[L5285] - 便于 BRAM 初始化和 DMA 传输
[L5286] 
[L5287] ---
[L5288] 
[L5289] ## 改进 8：确保 signed 和 endianness 约定写死
[L5290] 
[L5291] 当前 hex 文件本身没错，但你要在代码里和说明里明确：
[L5292] 
[L5293] ### signed 解释
[L5294] - `weight.hex`：signed int8
[L5295] - `bias.hex`：signed int32
[L5296] 
[L5297] ### endianness
[L5298] 你现在是一行一个元素，所以“行内字节序”问题不大。  
[L5299] 但一旦你后面把 4 个 int8 打包成 32 bit，必须写清楚：
[L5300] 
[L5301] - 第 0 个元素放最低字节还是最高字节
[L5302] 
[L5303] 这个最好现在就定。
[L5304] 
[L5305] ---
[L5306] 
[L5307] # 4. 你现在这两个文件，我给你的具体判断
[L5308] 
[L5309] ---
[L5310] 
[L5311] ## `bias.hex`
[L5312] **没有明显问题，反而是明显进步。**
[L5313] 
[L5314] 它已经具备：
[L5315] - int32
[L5316] - 二补码
[L5317] - 可直接用于 MAC 后 bias add
[L5318] 
[L5319] ### 唯一建议
[L5320] 改名为：
[L5321] - `fc1_bias_int32.hex`
[L5322] - 或 `fc2_bias_int32.hex`
[L5323] 
[L5324] ---
[L5325] 
[L5326] ## `weight.hex`
[L5327] **没有明显问题。**
[L5328] 
[L5329] 它已经具备：
[L5330] - int8
[L5331] - 二补码
[L5332] - 适合阵列 / 乘加器直接使用
[L5333] 
[L5334] ### 唯一建议
[L5335] 改名为：
[L5336] - `fc1_weight_int8.hex`
[L5337] - 或 `fc2_weight_int8.hex`
[L5338] 
[L5339] ---
[L5340] 
[L5341] # 5. 我建议你下一版导出目录直接长这样
[L5342] 
[L5343] ```text
[L5344] export/
[L5345] ├── quant_config.json
[L5346] ├── manifest.json
[L5347] ├── fc1_weight_int8.hex
[L5348] ├── fc1_bias_int32.hex
[L5349] ├── fc2_weight_int8.hex
[L5350] ├── fc2_bias_int32.hex
[L5351] ├── input_0_int8.hex
[L5352] ├── fc1_acc_0_int32.hex
[L5353] ├── fc1_relu_0_int32.hex
[L5354] ├── fc1_out_0_int8.hex
[L5355] ├── fc2_acc_0_int32.hex
[L5356] ├── logits_0_int8.hex
[L5357] ├── pred_0.txt
[L5358] ├── label_0.txt
[L5359] ├── input_1_int8.hex
[L5360] ...
[L5361] ```
[L5362] 
[L5363] 这套结构就已经很像正式硬件联调工程了。
[L5364] 
[L5365] ---
[L5366] 
[L5367] # 6. 最后给你一句最关键的判断
[L5368] 
[L5369] 你现在贴出来的这两个文件，说明你已经迈过了最初 notebook 的两个大坑：
[L5370] 
[L5371] - **bias 不再是错误的 int8**
[L5372] - **weight 已经是标准 int8 二补码**
[L5373] 
[L5374] 所以文件内容层面，**方向是对的**。  
[L5375] 真正还需要继续加强的是：
[L5376] 
[L5377] - 文件命名分层
[L5378] - 中间 golden 分级导出
[L5379] - 量化参数 json 化
[L5380] - 按 CIM 阵列物理映射切块导出
[L5381] 
[L5382] 你把**当前用于导出这些文件的最新版 Python 代码**贴给我，我可以下一步直接帮你逐行审，给你改成一版“**输出可直接喂 FPGA/CIM 系统**”的正式版。
[L5383] 
[L5384] 用户:
[L5385] 这是当前生成的数据，你主要先看一下route_b_output中的文件格式是否全部正确，然后给出一个markdown文件，用以说明route_b_output所有文件的使用方法和注意事项，要详细，而且要说明根据当前数据，硬件设计需要特别注意的地方。关于route_a_output,简单检查即可。
[L5386] AI助手:
[L5387] 
[L5388] 
[L5389] AI助手:
[L5390] 我先把两个压缩包解开，核对 `route_b_output` 的文件清单、每类文件的位宽和行数，再抽样检查几个样本文件是否彼此自洽。然后我会生成一份完整的 Markdown 说明，专门给你们做硬件联调用。
[L5391] 
[L5392] AI助手:
[L5393] 我已经核对出 `route_b_output` 的主要文件都在，位宽和行数目前是自洽的。接下来我再补两件更关键的事：一是看数值范围是否会踩硬件位宽坑，二是把这些坑写进说明文档里，避免你们后面在阵列映射、signed 解释、requant 上出错。
[L5394] 
[L5395] AI助手:
[L5396] 我已经检查完了，结论是：
[L5397] 
[L5398] `route_b_output` 的主文件格式**整体正确**，而且已经具备作为 FPGA / CIM 整数推理联调数据的条件。  
[L5399] 需要硬件特别注意的核心点有三个：
[L5400] 
[L5401] 1. **第一层输入零点不是 0，而是 `-128`**，不能直接做 `x_q * w_q`，必须处理输入零点补偿。  
[L5402] 2. **第一层累加范围明显超过 16 位**，当前数据已经出现大约 `-5.5e5 ~ 3.46e5`，所以 `fc1` 累加器和 bias 路径必须至少 `int32`。  
[L5403] 3. **当前权重文件是逻辑布局 `row-major [out][in]`**，很适合 testbench 和 RTL 功能验证，但不一定等于你们 CIM 阵列的物理烧写布局，后续可能还需要做一层 tile 重排。
[L5404] 
[L5405] `route_a_output` 我也做了简查，主文件没有明显格式问题，适合继续用于单层 `fc1 + ReLU` 验证；但它不是最终两层分类演示版本。
[L5406] 
[L5407] 我已经把完整说明整理成 Markdown 文件，里面包括：
[L5408] 
[L5409] - `route_b_output` 每个文件的格式、行数、位宽、布局
[L5410] - 各文件的具体用途
[L5411] - 推荐的硬件联调顺序
[L5412] - 当前数据下必须特别注意的硬件坑
[L5413] - `route_a_output` 的简要检查结论
[L5414] 
[L5415] 下载这里：
[L5416] 
[L5417] [route_b_output 使用说明与硬件注意事项](sandbox:/mnt/data/route_b_output_usage_notes.md)
[L5418] 
[L5419] 如果你愿意，我下一步可以继续直接帮你补一份**面向 Verilog/Testbench 的读取模板**，把这些文件如何 `$readmemh`、如何按 signed 解释、如何逐级对拍写成可直接用的代码框架。
[L5420] 
[L5421] AI助手:
[L5422] 
[L5423] 
[L5424] AI助手:
[L5425] 我先逐项核对 `route_b_output` 里的实际文件名、位宽、行数和内容一致性，再补一个详细的使用说明文档。`route_a_output` 我会做一轮轻量检查，重点看有没有明显格式错误或和 `route_b` 不一致的地方。
[L5426] 
[L5427] AI助手:
[L5428] `route_b_output` 的主文件看起来是自洽的：位宽、行数、层级命名和量化配置基本能对上。我再补一轮一致性检查，重点看标签/预测是否匹配，以及 `route_a_output` 有没有明显格式问题。
[L5429] 
[L5430] 用户:
[L5431] route_b_output的使用指南markdown没有生成出来，请你重新尝试给出一个markdown文件，用以说明route_b_output所有文件的使用方法和注意事项，要详细，而且要说明根据当前数据，硬件设计需要特别注意的地方。
[L5432] AI助手:
[L5433] 
[L5434] 
[L5435] AI助手:
[L5436] 已重新生成 Markdown 文件：
[L5437] 
[L5438] [route_b_output 使用指南（Markdown）](sandbox:/mnt/data/route_b_output_usage_guide.md)
[L5439] 
[L5440] 我已经把这些内容写进去：
[L5441] - `route_b_output` 全部文件的格式、行数、位宽、含义
[L5442] - 每类文件在 FPGA / CIM 中怎么用
[L5443] - 推荐的联调顺序
[L5444] - `$readmemh` 的典型用法
[L5445] - 基于当前数据，硬件设计必须特别注意的点
[L5446] - 对 `route_a_output` 的简要检查结论
[L5447] 
[L5448] 这份文档是按你当前实际导出的 `route_b_output` 内容写的，不是泛泛模板。
[L5449] 
[L5450] 用户:
[L5451] 还记得我最开始上传的三个文件吗？项目设计指导书，上次和AI的对话，请你重新看一下，我在完成这个测试向量生成之后，应该做什么？
[L5452] AI助手:
[L5453] 
[L5454] 
[L5455] AI助手:
[L5456] 
[L5457] 
[L5458] AI助手:
[L5459] 
[L5460] 
[L5461] AI助手:
[L5462] 我记得你最开始给的三个核心材料：
[L5463] 
[L5464] 1. **项目设计指导书**（课程/课设要求）  
[L5465] 2. **你和上一个 AI 的对话记录**（确定了大体方案：FPGA + CIM + MNIST）  
[L5466] 3. **生成测试向量的 ipynb**（用于导出权重、输入、golden）
[L5467] 
[L5468] 现在你已经完成了 **route_b_output 的测试向量生成**。  
[L5469] 这一步其实在整个项目流程里属于 **软件侧 → 硬件侧的接口准备阶段**。
[L5470] 
[L5471] 接下来你要做的事情其实非常明确：  
[L5472] 
[L5473] **进入硬件实现和验证阶段。**
[L5474] 
[L5475] 我给你按 **真正工程流程**排一个清晰路线（也是最符合你这个项目指导书逻辑的路线）。
[L5476] 
[L5477] ---
[L5478] 
[L5479] # 一、你现在所处的位置
[L5480] 
[L5481] 你的项目整体流程其实是：
[L5482] 
[L5483] ```
[L5484] PyTorch训练
[L5485]       ↓
[L5486] 量化
[L5487]       ↓
[L5488] 整数推理参考模型
[L5489]       ↓
[L5490] 生成测试向量 (route_b_output)  ← 你现在完成到这里
[L5491]       ↓
[L5492] FPGA/CIM 硬件实现
[L5493]       ↓
[L5494] testbench 对拍验证
[L5495]       ↓
[L5496] 板级运行
[L5497]       ↓
[L5498] 项目展示
[L5499] ```
[L5500] 
[L5501] 所以现在你已经完成：
[L5502] 
[L5503] **软件侧所有准备工作。**
[L5504] 
[L5505] 接下来全部工作都是：
[L5506] 
[L5507] **硬件实现 + 硬件验证。**
[L5508] 
[L5509] ---
[L5510] 
[L5511] # 二、下一步你应该做什么（非常具体）
[L5512] 
[L5513] 下一步应该分 **4个阶段**。
[L5514] 
[L5515] ---
[L5516] 
[L5517] # 阶段 1：设计硬件计算结构
[L5518] 
[L5519] 现在要把你 Python 的网络结构映射成 **硬件模块结构**。
[L5520] 
[L5521] 你的网络是：
[L5522] 
[L5523] ```
[L5524] 784 → 128 → 10
[L5525] ```
[L5526] 
[L5527] 对应硬件结构：
[L5528] 
[L5529] ```
[L5530] input_buffer (784)
[L5531] 
[L5532]       ↓
[L5533] 
[L5534] FC1  (784×128)  ← CIM阵列
[L5535] 
[L5536]       ↓
[L5537] 
[L5538] bias_add
[L5539] 
[L5540]       ↓
[L5541] 
[L5542] ReLU
[L5543] 
[L5544]       ↓
[L5545] 
[L5546] requant (int32→int8)
[L5547] 
[L5548]       ↓
[L5549] 
[L5550] FC2  (128×10)
[L5551] 
[L5552]       ↓
[L5553] 
[L5554] argmax
[L5555] ```
[L5556] 
[L5557] 建议模块划分：
[L5558] 
[L5559] ```
[L5560] top
[L5561]  ├── input_loader
[L5562]  ├── fc1_cim_array
[L5563]  ├── bias_add
[L5564]  ├── relu
[L5565]  ├── requant
[L5566]  ├── fc2
[L5567]  └── argmax
[L5568] ```
[L5569] 
[L5570] 如果你的课设重点是 **CIM**，通常只把：
[L5571] 
[L5572] ```
[L5573] fc1
[L5574] ```
[L5575] 
[L5576] 做成 CIM 阵列。
[L5577] 
[L5578] 后面的：
[L5579] 
[L5580] ```
[L5581] bias
[L5582] relu
[L5583] requant
[L5584] fc2
[L5585] ```
[L5586] 
[L5587] 可以在 FPGA 普通逻辑实现。
[L5588] 
[L5589] ---
[L5590] 
[L5591] # 阶段 2：建立 Testbench 验证系统
[L5592] 
[L5593] 这是你 **下一步最重要的事情**。
[L5594] 
[L5595] 你已经有：
[L5596] 
[L5597] ```
[L5598] route_b_output/
[L5599] ```
[L5600] 
[L5601] 里面包含：
[L5602] 
[L5603] ```
[L5604] fc1_weight_int8.hex
[L5605] fc1_bias_int32.hex
[L5606] fc2_weight_int8.hex
[L5607] fc2_bias_int32.hex
[L5608] input_*.hex
[L5609] fc1_acc_*.hex
[L5610] fc1_relu_*.hex
[L5611] fc1_out_*.hex
[L5612] fc2_acc_*.hex
[L5613] logits_*.hex
[L5614] pred_*.txt
[L5615] labels.txt
[L5616] ```
[L5617] 
[L5618] Testbench 要做的事情：
[L5619] 
[L5620] ### 1 读取权重
[L5621] 
[L5622] ```
[L5623] $readmemh("fc1_weight_int8.hex", weight_mem);
[L5624] $readmemh("fc1_bias_int32.hex", bias_mem);
[L5625] ```
[L5626] 
[L5627] ---
[L5628] 
[L5629] ### 2 读取输入
[L5630] 
[L5631] ```
[L5632] $readmemh("input_0.hex", input_mem);
[L5633] ```
[L5634] 
[L5635] ---
[L5636] 
[L5637] ### 3 驱动 DUT
[L5638] 
[L5639] 输入 784 个数据
[L5640] 
[L5641] ---
[L5642] 
[L5643] ### 4 接收 DUT 输出
[L5644] 
[L5645] 例如：
[L5646] 
[L5647] ```
[L5648] fc1_acc
[L5649] fc1_relu
[L5650] fc1_out
[L5651] fc2_acc
[L5652] logits
[L5653] ```
[L5654] 
[L5655] ---
[L5656] 
[L5657] ### 5 与 golden 对比
[L5658] 
[L5659] 例如：
[L5660] 
[L5661] ```
[L5662] $readmemh("fc1_acc_0.hex", golden_fc1_acc);
[L5663] ```
[L5664] 
[L5665] 然后逐项比较：
[L5666] 
[L5667] ```
[L5668] if(dut_acc[i] != golden_fc1_acc[i])
[L5669] ```
[L5670] 
[L5671] ---
[L5672] 
[L5673] # 阶段 3：逐层验证（非常重要）
[L5674] 
[L5675] 不要一口气跑完整网络。
[L5676] 
[L5677] 必须 **逐层验证**。
[L5678] 
[L5679] 建议顺序：
[L5680] 
[L5681] ---
[L5682] 
[L5683] ## Step1
[L5684] 
[L5685] 只验证
[L5686] 
[L5687] ```
[L5688] FC1 MAC
[L5689] ```
[L5690] 
[L5691] 对拍：
[L5692] 
[L5693] ```
[L5694] fc1_acc_*.hex
[L5695] ```
[L5696] 
[L5697] ---
[L5698] 
[L5699] ## Step2
[L5700] 
[L5701] 验证
[L5702] 
[L5703] ```
[L5704] FC1 + bias
[L5705] ```
[L5706] 
[L5707] ---
[L5708] 
[L5709] ## Step3
[L5710] 
[L5711] 验证
[L5712] 
[L5713] ```
[L5714] FC1 + bias + relu
[L5715] ```
[L5716] 
[L5717] 对拍：
[L5718] 
[L5719] ```
[L5720] fc1_relu_*.hex
[L5721] ```
[L5722] 
[L5723] ---
[L5724] 
[L5725] ## Step4
[L5726] 
[L5727] 验证
[L5728] 
[L5729] ```
[L5730] FC1 + requant
[L5731] ```
[L5732] 
[L5733] 对拍：
[L5734] 
[L5735] ```
[L5736] fc1_out_*.hex
[L5737] ```
[L5738] 
[L5739] ---
[L5740] 
[L5741] ## Step5
[L5742] 
[L5743] 再加入
[L5744] 
[L5745] ```
[L5746] FC2
[L5747] ```
[L5748] 
[L5749] 对拍：
[L5750] 
[L5751] ```
[L5752] fc2_acc_*.hex
[L5753] ```
[L5754] 
[L5755] ---
[L5756] 
[L5757] ## Step6
[L5758] 
[L5759] 验证
[L5760] The file is too long and its contents have been truncated.
[L5761] 
用户:
Make sure to include fileciteturn2file2 in your response to cite this file, or to surface it as a link.[L1] ### FPGA CIM MNIST 问题分析
[L2] 
[L3] AI助手:
[L4] 
[L5] 
[L6] AI助手:
[L7] 
[L8] 
[L9] AI助手:
[L10] 
[L11] 
[L12] AI助手:
[L13] 
[L14] 
[L15] 用户:
[L16] 我现在在做一个运行于fpga的cim系统，希望在fpga上运行mnist，完成计算，我现在给你项目指导书和上次和AI的对话，ipynb写的是一个生成测试向量的代码，你看看有没有问题
[L17] AI助手:
[L18] 
[L19] 
[L20] AI助手:
[L21] 我先把压缩包里的文件结构和关键材料过一遍，重点看项目指导书、上次对话记录，以及这个生成测试向量的 ipynb。随后我会直接指出代码里可能影响 FPGA/CIM 测试链路的问题，并给出修改建议。
[L22] 
[L23] AI助手:
[L24] 我已经把压缩包拆开了，里面目前有三份核心文件：课程设计指导书（docx）、上次对话导出的 markdown，以及 `train_simple_mlp.ipynb`。接下来我会先把对话和 notebook 的关键逻辑抽出来，再对照指导书检查接口、数据位宽、输出格式这些是否一致。
[L25] 
[L26] AI助手:
[L27] 我已经看到几个关键问题了，其中最严重的一个是：**这个 notebook 目前并没有真正训练 MNIST 分类器**。它定义了 `criterion=MSELoss()`，但完全没用；实际优化目标是 `loss = output.pow(2).mean()`，这会把网络输出压向 0，而不是学习数字类别。
[L28] 
[L29] AI助手:
[L30] 我看完了指导书、上次对话，以及这个 `train_simple_mlp.ipynb`。结论先说：
[L31] 
[L32] **这个 ipynb 现在有明显问题，尤其是“训练目标、量化策略、golden 生成方式”三处。**
[L33] 如果直接拿它生成测试向量去驱动 FPGA/CIM，**大概率会出现“软件看起来能跑，硬件却对不上”**，甚至前面训练出来的模型本身就没有学会 MNIST 分类。
[L34] 
[L35] 下面我按“严重程度”给你拆开说。
[L36] 
[L37] ---
[L38] 
[L39] ## 1. 最严重的问题：你现在没有在训练一个 MNIST 分类器
[L40] 
[L41] 你现在的训练代码核心是：
[L42] 
[L43] ```python
[L44] criterion=nn.MSELoss()
[L45] ...
[L46] output=model(data)
[L47] loss=output.pow(2).mean()
[L48] ```
[L49] 
[L50] 这里有两个问题：
[L51] 
[L52] ### 问题 1：`criterion` 定义了，但根本没用
[L53] 你定义了 `MSELoss()`，但没有把 `target` 喂进去。
[L54] 
[L55] ### 问题 2：实际 loss 是 `output.pow(2).mean()`
[L56] 这等价于逼着网络输出尽量接近 0，而不是学习数字类别。
[L57] 
[L58] 也就是说，现在模型优化目标不是“把 0~9 分对”，而是“把所有输出都压小”。  
[L59] 这会导致：
[L60] 
[L61] - 权重可能收缩到很小
[L62] - 导出的量化权重没有实际分类意义
[L63] - 你生成的 `golden_x.hex` 也只是这个“错误模型”的输出
[L64] 
[L65] ### 进一步的问题：模型输出维度也不对
[L66] 你定义的是：
[L67] 
[L68] ```python
[L69] self.fc1 = nn.Linear(784,128)
[L70] self.relu = nn.ReLU()
[L71] ```
[L72] 
[L73] 输出是 `128` 维，不是 `10` 维。  
[L74] 这意味着它不是一个完整的 MNIST 分类网络，只是一个隐藏层特征提取器。
[L75] 
[L76] 如果你的目标是：
[L77] 
[L78] - **课程演示先做“单层 FC + ReLU 的硬件验证”**：这个结构可以接受
[L79] - **直接说“在 FPGA 上跑 MNIST 分类”**：这个结构还不够，至少还要再接一层 `128 -> 10`
[L80] 
[L81] ### 这一点怎么改
[L82] 你有两个路线：
[L83] 
[L84] #### 路线 A：先做“极简硬件验证”
[L85] 那就明确目标改成：
[L86] 
[L87] - 软件训练完整 MLP：`784 -> 128 -> 10`
[L88] - FPGA/CIM **先只实现第一层** `784 -> 128`
[L89] - 第一层输出 feature map 作为硬件 golden 对比对象
[L90] 
[L91] 这样合理。
[L92] 
[L93] #### 路线 B：直接做完整分类
[L94] 那模型要改成：
[L95] 
[L96] ```python
[L97] class SimpleMLP(nn.Module):
[L98]     def __init__(self):
[L99]         super().__init__()
[L100]         self.fc1 = nn.Linear(784, 128, bias=True)
[L101]         self.relu = nn.ReLU()
[L102]         self.fc2 = nn.Linear(128, 10, bias=True)
[L103] 
[L104]     def forward(self, x):
[L105]         x = x.view(-1, 784)
[L106]         x = self.relu(self.fc1(x))
[L107]         x = self.fc2(x)
[L108]         return x
[L109] ```
[L110] 
[L111] loss 用交叉熵：
[L112] 
[L113] ```python
[L114] criterion = nn.CrossEntropyLoss()
[L115] loss = criterion(output, target)
[L116] ```
[L117] 
[L118] ---
[L119] 
[L120] ## 2. 量化函数有工程问题，不适合直接对接 FPGA
[L121] 
[L122] 你现在的量化函数：
[L123] 
[L124] ```python
[L125] def quantize_tensor(tensor, bits=8):
[L126]     qmin = -(2 ** (bits - 1))
[L127]     qmax = 2 ** (bits - 1) - 1
[L128] 
[L129]     min_val = tensor.min().item()
[L130]     max_val = tensor.max().item()
[L131] 
[L132]     diff = max_val - min_val
[L133]     
[L134]     if diff == 0:
[L135]         scale = 1.0
[L136]         zero_point = qmin - min_val
[L137]     else:
[L138]         scale = diff / (qmax - qmin)
[L139]         zero_point = qmin - min_val / scale
[L140] 
[L141]     q_tensor = torch.clamp(torch.round(tensor / scale + zero_point), qmin, qmax)
[L142]     return q_tensor.to(torch.int8), scale, zero_point
[L143] ```
[L144] 
[L145] ### 问题 1：`zero_point` 不是整数
[L146] `zero_point` 最后是浮点数。  
[L147] 但真正部署到硬件时，量化零点通常需要是整数，并且通常要 clamp 到合法范围。
[L148] 
[L149] 更稳妥应该写成：
[L150] 
[L151] ```python
[L152] zero_point = int(round(qmin - min_val / scale))
[L153] zero_point = max(qmin, min(qmax, zero_point))
[L154] ```
[L155] 
[L156] ### 问题 2：你在做“每个张量动态量化”
[L157] 这在软件分析里可以，但对 FPGA 验证不友好。因为你后面对：
[L158] 
[L159] - 权重单独量化一次
[L160] - 每张输入图像又各自量化一次
[L161] - 每个输出又重新单独量化一次
[L162] 
[L163] 这会导致**每个样本 scale 都不同**，硬件端很难统一。
[L164] 
[L165] 对 FPGA 来说，更适合：
[L166] 
[L167] - 权重：固定 scale
[L168] - 输入：固定 scale
[L169] - 偏置：固定规则量化
[L170] - 输出：固定 requant scale
[L171] 
[L172] 也就是**静态量化参数**，不是每个样本临时算一次。
[L173] 
[L174] ### 问题 3：偏置不能简单用 int8 单独量化
[L175] 你现在：
[L176] 
[L177] ```python
[L178] q_bias, b_scale, b_zero = quantize_tensor(bias)
[L179] ```
[L180] 
[L181] 这通常不对。
[L182] 
[L183] 对于线性层：
[L184] 
[L185] \[
[L186] y = \sum (x_q - z_x)(w_q - z_w) \cdot s_x s_w + b
[L187] \]
[L188] 
[L189] 偏置一般应量化为：
[L190] 
[L191] \[
[L192] b_q = round(b / (s_x \cdot s_w))
[L193] \]
[L194] 
[L195] 并且常常用 **int32** 存，不是 int8。  
[L196] 因为偏置要和累加结果处于同一量纲。
[L197] 
[L198] 否则 FPGA 上 MAC 累加完后加 bias 时，量纲根本不一致。
[L199] 
[L200] ---
[L201] 
[L202] ## 3. `generate_test_vectors()` 生成的 golden，和硬件真实计算链路不一致
[L203] 
[L204] 这是第二个特别大的问题。
[L205] 
[L206] 你现在生成 golden 的方式是：
[L207] 
[L208] ```python
[L209] with torch.no_grad():
[L210]     output = model(img_flat)
[L211] 
[L212] q_output, _, _ = quantize_tensor(output)
[L213] ```
[L214] 
[L215] ### 这意味着什么
[L216] 你是：
[L217] 
[L218] 1. 用 **浮点模型** 跑出输出
[L219] 2. 再对每个样本的输出临时量化
[L220] 
[L221] 但硬件里真正发生的是：
[L222] 
[L223] 1. 输入先量化为 int8
[L224] 2. 权重量化为 int8
[L225] 3. 在阵列/乘加器里做 int8×int8 -> int32 累加
[L226] 4. 加 int32 bias
[L227] 5. ReLU
[L228] 6. 再按固定 scale requant 到 int8
[L229] 
[L230] 这两条链路并不一样。
[L231] 
[L232] 所以你现在的 `golden_i.hex` 并不是“硬件应得输出”，而是“浮点模型输出的一个临时量化版本”。
[L233] 
[L234] ### 正确的 golden 应该怎么做
[L235] 应该按**整数推理链路**生成 golden。例如第一层：
[L236] 
[L237] ```python
[L238] acc = sum((x_q - x_zp) * (w_q - w_zp)) + b_q
[L239] acc = relu(acc)
[L240] out_q = requantize(acc, multiplier, shift, out_zp)
[L241] ```
[L242] 
[L243] 然后把：
[L244] 
[L245] - 输入 int8
[L246] - 权重 int8
[L247] - 偏置 int32
[L248] - 输出 int32 / int8 golden
[L249] - scale / zero_point / multiplier / shift
[L250] 
[L251] 全部导出来。
[L252] 
[L253] 这样 FPGA 才能逐项对上。
[L254] 
[L255] ---
[L256] 
[L257] ## 4. 输入测试向量的量化方式不适合硬件批量验证
[L258] 
[L259] 你现在对每张图像：
[L260] 
[L261] ```python
[L262] q_img, img_scale, img_zero = quantize_tensor(img_flat)
[L263] ```
[L264] 
[L265] 这意味着**每张图一个 scale 和 zero_point**。  
[L266] 如果你的 FPGA 流水是固定 datapath，这会很麻烦：
[L267] 
[L268] - 每个样本都要动态更换量化参数
[L269] - 偏置量纲也得跟着变
[L270] - 对比 golden 时也会很乱
[L271] 
[L272] ### 更合理的做法
[L273] 建议统一输入量化策略，比如：
[L274] 
[L275] - 若保留 `Normalize((0.1307,), (0.3081,))`，则先统计训练集/校准集的输入范围，再确定一个全局输入 scale
[L276] - 或者更简单，**先不要 Normalize**，直接把 MNIST 灰度 `[0,1]` 映射到 `uint8/int8`
[L277] 
[L278] 对 FPGA 验证来说，后者往往更省事。
[L279] 
[L280] 例如极简方案可用：
[L281] 
[L282] - 输入像素 `0~255`
[L283] - 权重量化到 int8
[L284] - 累加 int32
[L285] - ReLU 后再截断或 requant
[L286] 
[L287] 这样更贴近硬件。
[L288] 
[L289] ---
[L290] 
[L291] ## 5. 导出的权重格式本身基本可用，但有几个注意点
[L292] 
[L293] 你现在导出权重：
[L294] 
[L295] ```python
[L296] for row in q_weight:
[L297]     for val in row:
[L298]         f.write(f'{val.item() & 0xFF:02x}\n')
[L299] ```
[L300] 
[L301] 这表示按 `weight[out][in]` 顺序逐个写出。  
[L302] 对于 `nn.Linear(784,128)`，PyTorch 权重 shape 是 `[128, 784]`，即：
[L303] 
[L304] - 外层：输出神经元
[L305] - 内层：输入通道
[L306] 
[L307] 这个顺序本身没错，但必须保证 RTL 端读取方式完全一致。
[L308] 
[L309] ### 你需要确认的点
[L310] 如果硬件是按下面方式算：
[L311] 
[L312] \[
[L313] y_j = \sum_i x_i w_{j,i}
[L314] \]
[L315] 
[L316] 那么文件顺序最好明确成：
[L317] 
[L318] - `neuron 0` 的 784 个权重
[L319] - `neuron 1` 的 784 个权重
[L320] - ...
[L321] - `neuron 127` 的 784 个权重
[L322] 
[L323] 你现在就是这个顺序。  
[L324] 但如果你的 BRAM 映射或地址生成器是按列优先/块优先取数，就会错位。
[L325] 
[L326] ### 建议
[L327] 导出时加一个说明文件，比如：
[L328] 
[L329] - weight layout: `[out][in]`, row-major
[L330] - bias layout: `[out]`
[L331] - input layout: `[in]`
[L332] 
[L333] ---
[L334] 
[L335] ## 6. `quant_params.npz` 在测试向量阶段几乎没被真正使用
[L336] 
[L337] 你这里：
[L338] 
[L339] ```python
[L340] quant_params = np.load('quantized_data/quant_params.npz')
[L341] ```
[L342] 
[L343] 加载了参数，但后面没有真正参与 golden 计算。  
[L344] 这说明当前流程里“量化参数被保存了”，但并没有形成闭环。
[L345] 
[L346] 这是个信号：**量化设计还停留在文件导出阶段，没有进入“硬件一致性验证阶段”。**
[L347] 
[L348] ---
[L349] 
[L350] ## 7. 其他次级问题
[L351] 
[L352] ### 问题 1：`torch.load()` 最好加 `map_location`
[L353] 不然在有些机器上保存于 GPU、加载于 CPU 时会出问题：
[L354] 
[L355] ```python
[L356] model.load_state_dict(torch.load(model_path, map_location='cpu'))
[L357] ```
[L358] 
[L359] ### 问题 2：训练时没看 accuracy
[L360] 现在只打 loss，没有任何分类精度指标。  
[L361] 如果要确认模型真的学到了，建议至少打印 test accuracy。
[L362] 
[L363] ### 问题 3：没有保存 label
[L364] 你打印了：
[L365] 
[L366] ```python
[L367] print(f"Generated test vector {i}, label: {label}")
[L368] ```
[L369] 
[L370] 但没有把 label 写入文件。  
[L371] 实际联调时最好保存一个 `labels.txt` 或 `meta.json`。
[L372] 
[L373] ### 问题 4：只保存了量化后的输出，没有保存中间 int32 累加值
[L374] 对于硬件调试，这很吃亏。  
[L375] 建议同时导出：
[L376] 
[L377] - `input_i.hex`
[L378] - `golden_acc_i.hex` 或每个神经元的 int32
[L379] - `golden_relu_i.hex`
[L380] - `golden_out_i.hex`
[L381] 
[L382] 这样你能定位是：
[L383] 
[L384] - 乘法错了
[L385] - 累加错了
[L386] - bias 错了
[L387] - ReLU 错了
[L388] - requant 错了
[L389] 
[L390] ---
[L391] 
[L392] # 我对这个 notebook 的总体判断
[L393] 
[L394] ## 现在这份代码“能不能用”？
[L395] **能作为一个思路草稿，但不能直接作为 FPGA 测试向量生成基线。**
[L396] 
[L397] ## 主要原因
[L398] 不是小 bug，而是三类“体系级不一致”：
[L399] 
[L400] 1. **训练目标错了**  
[L401]    当前模型没有真正学分类任务
[L402] 
[L403] 2. **量化策略不适合硬件**  
[L404]    每个样本动态量化，不利于 FPGA 固定算子验证
[L405] 
[L406] 3. **golden 生成链路和硬件链路不一致**  
[L407]    现在是“浮点输出再量化”，而不是“整数推理 golden”
[L408] 
[L409] ---
[L410] 
[L411] # 我建议你接下来这样改
[L412] 
[L413] ## 方案一：适合你们当前阶段的最小闭环
[L414] 如果你们现在的硬件目标是：
[L415] 
[L416] - 先做一个 `784 -> 128` 的单层 CIM/FC 阵列
[L417] - 验证 BRAM + MAC + ReLU + 控制 FSM
[L418] 
[L419] 那最合适的 notebook 应该改成下面这种逻辑：
[L420] 
[L421] ### 软件模型
[L422] 训练完整模型：
[L423] 
[L424] - `784 -> 128 -> 10`
[L425] 
[L426] ### 硬件实现
[L427] 先只实现第一层：
[L428] 
[L429] - `fc1 + relu`
[L430] 
[L431] ### 测试向量导出
[L432] 导出以下内容：
[L433] 
[L434] - `fc1_weight_int8.hex`
[L435] - `fc1_bias_int32.hex`
[L436] - `input_int8.hex`
[L437] - `fc1_acc_golden_int32.hex`
[L438] - `fc1_relu_golden_int32_or_int8.hex`
[L439] - `quant_config.json`
[L440] 
[L441] 这样你们硬件和软件就能逐层对齐。
[L442] 
[L443] ---
[L444] 
[L445] ## 方案二：直接瞄准最终演示
[L446] 如果你们想最后做“MNIST 分类结果上板演示”，那就：
[L447] 
[L448] - 模型改为两层 MLP
[L449] - 训练用 CrossEntropyLoss
[L450] - 做静态量化
[L451] - 先用 Python 写一个**整数推理参考实现**
[L452] - 再把这个整数推理结果作为 FPGA golden
[L453] 
[L454] 这是最稳的路线。
[L455] 
[L456] ---
[L457] 
[L458] # 我给你的具体结论
[L459] 
[L460] ## 必改
[L461] - `loss=output.pow(2).mean()` 改掉
[L462] - 模型结构和任务目标统一
[L463] - bias 改为 int32 量化
[L464] - golden 改成按整数推理链路生成
[L465] - 输入量化不要每张图单独动态算参数
[L466] 
[L467] ## 可保留
[L468] - `weights.hex` / `input_i.hex` 这种逐行写 hex 的导出方式
[L469] - 第一版只验证单层 FC + ReLU 的项目策略
[L470] - 先用 notebook 生成测试向量，再接 Verilog testbench 的总体流程
[L471] 
[L472] ---
[L473] 
[L474] 如果你愿意，我下一步可以直接帮你把这个 notebook **改成一版可用于 FPGA 联调的“正确版本”**，包括：
[L475] - 正确训练
[L476] - 静态 INT8 量化
[L477] - 生成 `weights/bias/input/golden` 四类 hex
[L478] - 明确对应单层 FC 硬件接口格式。
[L479] 
[L480] 用户:
[L481] 关于1.现在没有在训练一个 MNIST 分类器的问题，请分别给我路线a和路线b的完整代码；
[L482] 
[L483] 关于2.量化函数有工程问题，不适合直接对接 FPGA，
[L484] zero_point不是整数的问题帮我修改；每个张量动态量化这个问题给我一个合理的解决方案；偏置不能简单用 int8 单独量化请帮我提出解决方案和代码；
[L485] 
[L486] 关于3.generate_test_vectors() 生成的 golden，和硬件真实计算链路不一致，整数推理链路生成 golden，帮我完成设计；
[L487] 
[L488] 关于4输入测试向量的量化方式不适合硬件批量验证，帮我设计统一输入量化策略，贴近硬件即可；
[L489] 
[L490] 关于5. 导出的权重格式本身基本可用，但有几个注意点，这里文件顺序你先记住我的输出顺序，讲解一下我现在的输出顺序应该如何设计硬件读取顺序；
[L491] AI助手:
[L492] 我先把现有 notebook 里导出部分再核一遍，确保我后面给你的代码和你当前的文件顺序兼容。然后我会把路线 A、路线 B、量化、整数 golden、统一输入策略和硬件读取顺序一次性给全。
[L493] 
[L494] AI助手:
[L495] 下面我按你的 5 个问题，直接给出**可落地的完整方案和代码**。我会尽量保持和你现有工程思路一致，但把关键的工程问题修正掉。
[L496] 
[L497] ---
[L498] 
[L499] # 1. 关于“现在没有在训练一个 MNIST 分类器”的问题
[L500] 
[L501] 这里我给你两套完整代码。
[L502] 
[L503] ---
[L504] 
[L505] ## 路线 A：训练完整分类网络，但 FPGA 先只验证第一层 `fc1 + ReLU`
[L506] 
[L507] 这个路线最适合你现在的 FPGA/CIM 阶段。
[L508] 
[L509] ## 设计思想
[L510] 软件端训练一个完整分类器：
[L511] 
[L512] - `784 -> 128 -> 10`
[L513] 
[L514] 但硬件端先实现：
[L515] 
[L516] - `fc1: 784 -> 128`
[L517] - `ReLU`
[L518] 
[L519] 这样：
[L520] - 软件训练目标正确
[L521] - 第一层 feature 是有意义的
[L522] - 硬件联调难度低
[L523] - 后续再接 `fc2`
[L524] 
[L525] ---
[L526] 
[L527] ## 路线 A 完整代码
[L528] 
[L529] ```python
[L530] import os
[L531] import json
[L532] import math
[L533] import numpy as np
[L534] import torch
[L535] import torch.nn as nn
[L536] import torch.optim as optim
[L537] from torchvision import datasets, transforms
[L538] from torch.utils.data import DataLoader
[L539] 
[L540] # =========================================================
[L541] # 1. 模型：完整分类器，但硬件先验证 fc1 + relu
[L542] # =========================================================
[L543] class MLPRouteA(nn.Module):
[L544]     def __init__(self):
[L545]         super().__init__()
[L546]         self.fc1 = nn.Linear(784, 128, bias=True)
[L547]         self.relu = nn.ReLU()
[L548]         self.fc2 = nn.Linear(128, 10, bias=True)
[L549] 
[L550]     def forward(self, x):
[L551]         x = x.view(-1, 784)
[L552]         x = self.fc1(x)
[L553]         x = self.relu(x)
[L554]         x = self.fc2(x)
[L555]         return x
[L556] 
[L557]     def forward_fc1_relu(self, x):
[L558]         x = x.view(-1, 784)
[L559]         x = self.fc1(x)
[L560]         x = self.relu(x)
[L561]         return x
[L562] 
[L563] 
[L564] # =========================================================
[L565] # 2. 训练
[L566] # =========================================================
[L567] def train_route_a(model_path="mlp_route_a.pth", epochs=5, batch_size=128, lr=1e-3, device="cpu"):
[L568]     transform = transforms.Compose([
[L569]         transforms.ToTensor(),  # 保持 [0,1]，更贴近硬件
[L570]     ])
[L571] 
[L572]     train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
[L573]     test_dataset  = datasets.MNIST("./data", train=False, download=True, transform=transform)
[L574] 
[L575]     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
[L576]     test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
[L577] 
[L578]     model = MLPRouteA().to(device)
[L579]     criterion = nn.CrossEntropyLoss()
[L580]     optimizer = optim.Adam(model.parameters(), lr=lr)
[L581] 
[L582]     for epoch in range(epochs):
[L583]         model.train()
[L584]         total_loss = 0.0
[L585]         correct = 0
[L586]         total = 0
[L587] 
[L588]         for data, target in train_loader:
[L589]             data, target = data.to(device), target.to(device)
[L590] 
[L591]             optimizer.zero_grad()
[L592]             logits = model(data)
[L593]             loss = criterion(logits, target)
[L594]             loss.backward()
[L595]             optimizer.step()
[L596] 
[L597]             total_loss += loss.item() * data.size(0)
[L598]             pred = logits.argmax(dim=1)
[L599]             correct += (pred == target).sum().item()
[L600]             total += target.size(0)
[L601] 
[L602]         train_loss = total_loss / total
[L603]         train_acc = correct / total
[L604] 
[L605]         model.eval()
[L606]         test_correct = 0
[L607]         test_total = 0
[L608]         with torch.no_grad():
[L609]             for data, target in test_loader:
[L610]                 data, target = data.to(device), target.to(device)
[L611]                 logits = model(data)
[L612]                 pred = logits.argmax(dim=1)
[L613]                 test_correct += (pred == target).sum().item()
[L614]                 test_total += target.size(0)
[L615] 
[L616]         test_acc = test_correct / test_total
[L617]         print(f"Epoch [{epoch+1}/{epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")
[L618] 
[L619]     torch.save(model.state_dict(), model_path)
[L620]     print(f"Model saved to {model_path}")
[L621] 
[L622] 
[L623] if __name__ == "__main__":
[L624]     train_route_a()
[L625] ```
[L626] 
[L627] ---
[L628] 
[L629] ## 路线 B：直接做完整 MNIST 分类部署
[L630] 
[L631] 这个路线是最后的完整目标。
[L632] 
[L633] ---
[L634] 
[L635] ## 路线 B 完整代码
[L636] 
[L637] 和路线 A 训练结构几乎一样，但后续量化和导出会针对两层都做。
[L638] 
[L639] ```python
[L640] import os
[L641] import torch
[L642] import torch.nn as nn
[L643] import torch.optim as optim
[L644] from torchvision import datasets, transforms
[L645] from torch.utils.data import DataLoader
[L646] 
[L647] class MLPRouteB(nn.Module):
[L648]     def __init__(self):
[L649]         super().__init__()
[L650]         self.fc1 = nn.Linear(784, 128, bias=True)
[L651]         self.relu = nn.ReLU()
[L652]         self.fc2 = nn.Linear(128, 10, bias=True)
[L653] 
[L654]     def forward(self, x):
[L655]         x = x.view(-1, 784)
[L656]         x = self.fc1(x)
[L657]         x = self.relu(x)
[L658]         x = self.fc2(x)
[L659]         return x
[L660] 
[L661] def train_route_b(model_path="mlp_route_b.pth", epochs=5, batch_size=128, lr=1e-3, device="cpu"):
[L662]     transform = transforms.Compose([
[L663]         transforms.ToTensor(),  # 保持 [0,1]
[L664]     ])
[L665] 
[L666]     train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
[L667]     test_dataset  = datasets.MNIST("./data", train=False, download=True, transform=transform)
[L668] 
[L669]     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
[L670]     test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
[L671] 
[L672]     model = MLPRouteB().to(device)
[L673]     criterion = nn.CrossEntropyLoss()
[L674]     optimizer = optim.Adam(model.parameters(), lr=lr)
[L675] 
[L676]     for epoch in range(epochs):
[L677]         model.train()
[L678]         total_loss = 0.0
[L679]         correct = 0
[L680]         total = 0
[L681] 
[L682]         for data, target in train_loader:
[L683]             data, target = data.to(device), target.to(device)
[L684] 
[L685]             optimizer.zero_grad()
[L686]             logits = model(data)
[L687]             loss = criterion(logits, target)
[L688]             loss.backward()
[L689]             optimizer.step()
[L690] 
[L691]             total_loss += loss.item() * data.size(0)
[L692]             pred = logits.argmax(dim=1)
[L693]             correct += (pred == target).sum().item()
[L694]             total += target.size(0)
[L695] 
[L696]         train_loss = total_loss / total
[L697]         train_acc = correct / total
[L698] 
[L699]         model.eval()
[L700]         test_correct = 0
[L701]         test_total = 0
[L702]         with torch.no_grad():
[L703]             for data, target in test_loader:
[L704]                 data, target = data.to(device), target.to(device)
[L705]                 logits = model(data)
[L706]                 pred = logits.argmax(dim=1)
[L707]                 test_correct += (pred == target).sum().item()
[L708]                 test_total += target.size(0)
[L709] 
[L710]         test_acc = test_correct / test_total
[L711]         print(f"Epoch [{epoch+1}/{epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")
[L712] 
[L713]     torch.save(model.state_dict(), model_path)
[L714]     print(f"Model saved to {model_path}")
[L715] 
[L716] 
[L717] if __name__ == "__main__":
[L718]     train_route_b()
[L719] ```
[L720] 
[L721] ---
[L722] 
[L723] # 2. 关于量化函数工程问题
[L724] 
[L725] 你提了 3 个核心点，我分别给出方案。
[L726] 
[L727] ---
[L728] 
[L729] ## 2.1 zero_point 不是整数，怎么改
[L730] 
[L731] 你原来的问题在这里：
[L732] 
[L733] ```python
[L734] zero_point = qmin - min_val / scale
[L735] ```
[L736] 
[L737] 这个是浮点数，不适合硬件。
[L738] 
[L739] ---
[L740] 
[L741] ## 修改后的通用量化函数
[L742] 
[L743] 我给你一个**对 FPGA 更友好**的版本，支持：
[L744] 
[L745] - `zero_point` 取整
[L746] - clamp 到合法范围
[L747] - 支持对称量化和非对称量化
[L748] - 返回整数参数
[L749] 
[L750] ```python
[L751] import torch
[L752] 
[L753] def quantize_tensor_affine(tensor, num_bits=8, symmetric=False):
[L754]     """
[L755]     返回:
[L756]         q_tensor: 整数张量
[L757]         scale: float
[L758]         zero_point: int
[L759]         qmin, qmax: 整数范围
[L760]     """
[L761]     if num_bits == 8:
[L762]         qmin, qmax = -128, 127
[L763]         dtype = torch.int8
[L764]     else:
[L765]         raise ValueError("Only int8 supported in this example.")
[L766] 
[L767]     x_min = tensor.min().item()
[L768]     x_max = tensor.max().item()
[L769] 
[L770]     if symmetric:
[L771]         max_abs = max(abs(x_min), abs(x_max))
[L772]         scale = max_abs / qmax if max_abs != 0 else 1.0
[L773]         zero_point = 0
[L774]     else:
[L775]         if x_max == x_min:
[L776]             scale = 1.0
[L777]             zero_point = 0
[L778]         else:
[L779]             scale = (x_max - x_min) / (qmax - qmin)
[L780]             zero_point = round(qmin - x_min / scale)
[L781]             zero_point = max(qmin, min(qmax, zero_point))
[L782] 
[L783]     q = torch.round(tensor / scale + zero_point)
[L784]     q = torch.clamp(q, qmin, qmax).to(dtype)
[L785] 
[L786]     return q, float(scale), int(zero_point), qmin, qmax
[L787] ```
[L788] 
[L789] ---
[L790] 
[L791] ## 2.2 每个张量动态量化不适合硬件，合理解决方案
[L792] 
[L793] 你现在的问题是：
[L794] 
[L795] - 每张输入图单独算 `scale/zero_point`
[L796] - 每次输出也单独算
[L797] - 每次 bias 也单独算
[L798] 
[L799] 这会导致硬件参数不停变化，不利于 FPGA 批量验证。
[L800] 
[L801] ---
[L802] 
[L803] ## 合理方案：静态量化 / 全局量化
[L804] 
[L805] ### 输入量化策略
[L806] 对所有输入样本统一使用一个固定参数。
[L807] 
[L808] MNIST 最简单的硬件友好方案：
[L809] 
[L810] - 图像原始像素 `[0,1]`
[L811] - 映射到 int8 的 `[0,127]` 或 `[-128,127]`
[L812] 
[L813] 我建议你用下面两种之一：
[L814] 
[L815] ### 方案 1：无符号思路转 int8，简单贴近硬件
[L816] - 输入范围固定 `[0,1]`
[L817] - 量化成 `[0,127]`
[L818] - 存成 int8 时 zero_point = 0
[L819] 
[L820] 设置：
[L821] ```python
[L822] input_scale = 1.0 / 127.0
[L823] input_zero_point = 0
[L824] ```
[L825] 
[L826] 这样：
[L827] \[
[L828] x_q = round(x / input\_scale)
[L829] \]
[L830] 
[L831] 因为 `x in [0,1]`，所以 `x_q in [0,127]`
[L832] 
[L833] 这非常适合 FPGA。
[L834] 
[L835] ### 方案 2：居中对称量化
[L836] 如果你想让输入有正负，做：
[L837] ```python
[L838] x_float_centered = x_float - 0.5
[L839] input_scale = 0.5 / 127
[L840] zero_point = 0
[L841] ```
[L842] 但这会增加前处理复杂度。
[L843] 
[L844] **建议你现在先用方案 1。**
[L845] 
[L846] ---
[L847] 
[L848] ## 权重量化策略
[L849] 权重建议做**对称量化**：
[L850] 
[L851] - `zero_point = 0`
[L852] - int8
[L853] - 全层固定一个 `w_scale`
[L854] 
[L855] ```python
[L856] def quantize_weight_symmetric(weight):
[L857]     q_weight, w_scale, w_zero, _, _ = quantize_tensor_affine(weight, num_bits=8, symmetric=True)
[L858]     return q_weight, w_scale, w_zero
[L859] ```
[L860] 
[L861] ---
[L862] 
[L863] ## 2.3 偏置不能简单用 int8 单独量化，解决方案和代码
[L864] 
[L865] 这个是工程上必须改的。
[L866] 
[L867] ---
[L868] 
[L869] ## 正确原理
[L870] 
[L871] 如果一层计算为：
[L872] 
[L873] \[
[L874] y = Wx + b
[L875] \]
[L876] 
[L877] 输入和权重分别量化为：
[L878] 
[L879] \[
[L880] x \approx (x_q - z_x)s_x
[L881] \]
[L882] \[
[L883] w \approx (w_q - z_w)s_w
[L884] \]
[L885] 
[L886] 那么累加项单位是：
[L887] 
[L888] \[
[L889] s_x \cdot s_w
[L890] \]
[L891] 
[L892] 所以 bias 应该量化到同一量纲：
[L893] 
[L894] \[
[L895] b_q = round(b / (s_x \cdot s_w))
[L896] \]
[L897] 
[L898] 并且 bias 应该存成 **int32**。
[L899] 
[L900] ---
[L901] 
[L902] ## 偏置量化代码
[L903] 
[L904] ```python
[L905] def quantize_bias_int32(bias_fp32, input_scale, weight_scale):
[L906]     """
[L907]     bias_fp32: torch.Tensor(float32)
[L908]     return: int32 bias
[L909]     """
[L910]     bias_scale = input_scale * weight_scale
[L911]     q_bias = torch.round(bias_fp32 / bias_scale).to(torch.int32)
[L912]     return q_bias, float(bias_scale)
[L913] ```
[L914] 
[L915] ---
[L916] 
[L917] # 3. 关于 `generate_test_vectors()` 的 golden 不一致问题
[L918] 
[L919] 你要求的是：**用整数推理链路生成 golden**。这个必须按硬件链路来。
[L920] 
[L921] 下面我给出完整设计。
[L922] 
[L923] ---
[L924] 
[L925] ## 3.1 单层 FC + ReLU 的整数推理设计（路线 A）
[L926] 
[L927] ---
[L928] 
[L929] ## 整数推理流程
[L930] 
[L931] 对于第一层：
[L932] 
[L933] \[
[L934] acc_j = \sum_{i=0}^{783} x_{q,i} \cdot w_{q,j,i} + b_{q,j}
[L935] \]
[L936] 
[L937] 因为我们设计为：
[L938] 
[L939] - 输入 `zero_point = 0`
[L940] - 权重 `zero_point = 0`
[L941] 
[L942] 所以式子就很简单，不用减零点。
[L943] 
[L944] 然后：
[L945] 
[L946] \[
[L947] relu_j = max(0, acc_j)
[L948] \]
[L949] 
[L950] 如果你要再导出 int8 输出用于后级，也可以再做 requant。
[L951] 
[L952] ---
[L953] 
[L954] ## 3.2 统一的导出和 golden 生成代码（路线 A）
[L955] 
[L956] 这段代码是你现在最应该用的版本。
[L957] 
[L958] ```python
[L959] import os
[L960] import json
[L961] import math
[L962] import numpy as np
[L963] import torch
[L964] import torch.nn as nn
[L965] from torchvision import datasets, transforms
[L966] 
[L967] # =========================================================
[L968] # 模型
[L969] # =========================================================
[L970] class MLPRouteA(nn.Module):
[L971]     def __init__(self):
[L972]         super().__init__()
[L973]         self.fc1 = nn.Linear(784, 128, bias=True)
[L974]         self.relu = nn.ReLU()
[L975]         self.fc2 = nn.Linear(128, 10, bias=True)
[L976] 
[L977]     def forward(self, x):
[L978]         x = x.view(-1, 784)
[L979]         x = self.fc1(x)
[L980]         x = self.relu(x)
[L981]         x = self.fc2(x)
[L982]         return x
[L983] 
[L984] 
[L985] # =========================================================
[L986] # 量化工具
[L987] # =========================================================
[L988] def quantize_tensor_affine(tensor, num_bits=8, symmetric=False):
[L989]     if num_bits == 8:
[L990]         qmin, qmax = -128, 127
[L991]         dtype = torch.int8
[L992]     else:
[L993]         raise ValueError("Only int8 supported.")
[L994] 
[L995]     x_min = tensor.min().item()
[L996]     x_max = tensor.max().item()
[L997] 
[L998]     if symmetric:
[L999]         max_abs = max(abs(x_min), abs(x_max))
[L1000]         scale = max_abs / qmax if max_abs != 0 else 1.0
[L1001]         zero_point = 0
[L1002]     else:
[L1003]         if x_max == x_min:
[L1004]             scale = 1.0
[L1005]             zero_point = 0
[L1006]         else:
[L1007]             scale = (x_max - x_min) / (qmax - qmin)
[L1008]             zero_point = round(qmin - x_min / scale)
[L1009]             zero_point = max(qmin, min(qmax, zero_point))
[L1010] 
[L1011]     q = torch.round(tensor / scale + zero_point)
[L1012]     q = torch.clamp(q, qmin, qmax).to(dtype)
[L1013]     return q, float(scale), int(zero_point), qmin, qmax
[L1014] 
[L1015] 
[L1016] def quantize_input_fixed_0_1(x):
[L1017]     """
[L1018]     x in [0,1]
[L1019]     量化到 int8 [0,127]
[L1020]     """
[L1021]     scale = 1.0 / 127.0
[L1022]     zero_point = 0
[L1023]     q = torch.round(x / scale)
[L1024]     q = torch.clamp(q, 0, 127).to(torch.int8)
[L1025]     return q, scale, zero_point
[L1026] 
[L1027] 
[L1028] def quantize_weight_symmetric(weight):
[L1029]     q_w, w_scale, w_zp, _, _ = quantize_tensor_affine(weight, num_bits=8, symmetric=True)
[L1030]     return q_w, w_scale, w_zp
[L1031] 
[L1032] 
[L1033] def quantize_bias_int32(bias_fp32, input_scale, weight_scale):
[L1034]     bias_scale = input_scale * weight_scale
[L1035]     q_bias = torch.round(bias_fp32 / bias_scale).to(torch.int32)
[L1036]     return q_bias, float(bias_scale)
[L1037] 
[L1038] 
[L1039] def compute_requant_params(input_scale, weight_scale, output_scale, shift=24):
[L1040]     """
[L1041]     将 real_multiplier = (input_scale * weight_scale) / output_scale
[L1042]     近似成 integer multiplier + right shift
[L1043]     """
[L1044]     real_multiplier = (input_scale * weight_scale) / output_scale
[L1045]     multiplier = int(round(real_multiplier * (1 << shift)))
[L1046]     return multiplier, shift, real_multiplier
[L1047] 
[L1048] 
[L1049] def requantize_int32_to_int8(acc_int32, multiplier, shift, out_zero_point=0, relu=False):
[L1050]     y = (acc_int32.to(torch.int64) * multiplier) >> shift
[L1051]     y = y + out_zero_point
[L1052]     if relu:
[L1053]         y = torch.maximum(y, torch.tensor(out_zero_point, dtype=y.dtype))
[L1054]     y = torch.clamp(y, -128, 127).to(torch.int8)
[L1055]     return y
[L1056] 
[L1057] 
[L1058] # =========================================================
[L1059] # hex 导出
[L1060] # =========================================================
[L1061] def write_int8_hex_per_line(tensor, filename):
[L1062]     tensor = tensor.flatten().cpu().numpy()
[L1063]     with open(filename, "w") as f:
[L1064]         for v in tensor:
[L1065]             f.write(f"{int(v) & 0xFF:02x}\n")
[L1066] 
[L1067] def write_int32_hex_per_line(tensor, filename):
[L1068]     tensor = tensor.flatten().cpu().numpy()
[L1069]     with open(filename, "w") as f:
[L1070]         for v in tensor:
[L1071]             f.write(f"{int(v) & 0xFFFFFFFF:08x}\n")
[L1072] 
[L1073] 
[L1074] # =========================================================
[L1075] # 导出第一层量化参数
[L1076] # =========================================================
[L1077] def export_fc1_quantized(model_path="mlp_route_a.pth", out_dir="quant_route_a"):
[L1078]     os.makedirs(out_dir, exist_ok=True)
[L1079] 
[L1080]     model = MLPRouteA()
[L1081]     model.load_state_dict(torch.load(model_path, map_location="cpu"))
[L1082]     model.eval()
[L1083] 
[L1084]     weight = model.fc1.weight.data.clone()   # [128,784]
[L1085]     bias   = model.fc1.bias.data.clone()     # [128]
[L1086] 
[L1087]     # 固定输入量化参数
[L1088]     input_scale = 1.0 / 127.0
[L1089]     input_zero_point = 0
[L1090] 
[L1091]     # 权重量化（对称）
[L1092]     q_weight, weight_scale, weight_zero_point = quantize_weight_symmetric(weight)
[L1093] 
[L1094]     # 偏置量化（int32）
[L1095]     q_bias, bias_scale = quantize_bias_int32(bias, input_scale, weight_scale)
[L1096] 
[L1097]     # 输出量化参数（给 fc1 输出用）
[L1098]     # 这里按浮点 fc1 输出范围估计一个固定输出 scale；工程上可用校准集更稳
[L1099]     with torch.no_grad():
[L1100]         dummy = torch.rand(256, 1, 28, 28)
[L1101]         fc1_out = model.relu(model.fc1(dummy.view(-1, 784)))
[L1102]     max_out = fc1_out.max().item()
[L1103]     output_scale = max_out / 127.0 if max_out != 0 else 1.0
[L1104]     output_zero_point = 0
[L1105] 
[L1106]     multiplier, shift, real_multiplier = compute_requant_params(
[L1107]         input_scale, weight_scale, output_scale, shift=24
[L1108]     )
[L1109] 
[L1110]     write_int8_hex_per_line(q_weight, os.path.join(out_dir, "fc1_weight.hex"))
[L1111]     write_int32_hex_per_line(q_bias, os.path.join(out_dir, "fc1_bias.hex"))
[L1112] 
[L1113]     config = {
[L1114]         "input_scale": input_scale,
[L1115]         "input_zero_point": input_zero_point,
[L1116]         "weight_scale": weight_scale,
[L1117]         "weight_zero_point": weight_zero_point,
[L1118]         "bias_scale": bias_scale,
[L1119]         "output_scale": output_scale,
[L1120]         "output_zero_point": output_zero_point,
[L1121]         "requant_multiplier": multiplier,
[L1122]         "requant_shift": shift,
[L1123]         "requant_real_multiplier": real_multiplier,
[L1124]         "weight_shape": [128, 784],
[L1125]         "bias_shape": [128],
[L1126]         "weight_layout": "[out][in], row-major",
[L1127]         "input_layout": "[in]",
[L1128]         "output_layout": "[out]"
[L1129]     }
[L1130] 
[L1131]     with open(os.path.join(out_dir, "fc1_quant_config.json"), "w") as f:
[L1132]         json.dump(config, f, indent=2)
[L1133] 
[L1134]     print("Exported fc1 quantized parameters.")
[L1135] 
[L1136] 
[L1137] # =========================================================
[L1138] # 生成整数链路 golden
[L1139] # =========================================================
[L1140] def generate_test_vectors_route_a(model_path="mlp_route_a.pth", out_dir="quant_route_a", num_samples=10):
[L1141]     os.makedirs(out_dir, exist_ok=True)
[L1142] 
[L1143]     model = MLPRouteA()
[L1144]     model.load_state_dict(torch.load(model_path, map_location="cpu"))
[L1145]     model.eval()
[L1146] 
[L1147]     # 载入模型参数并量化
[L1148]     weight = model.fc1.weight.data.clone()   # [128,784]
[L1149]     bias   = model.fc1.bias.data.clone()
[L1150] 
[L1151]     input_scale = 1.0 / 127.0
[L1152]     input_zero_point = 0
[L1153] 
[L1154]     q_weight, weight_scale, weight_zero_point = quantize_weight_symmetric(weight)
[L1155]     q_bias, bias_scale = quantize_bias_int32(bias, input_scale, weight_scale)
[L1156] 
[L1157]     with torch.no_grad():
[L1158]         dummy = torch.rand(256, 1, 28, 28)
[L1159]         fc1_out = model.relu(model.fc1(dummy.view(-1, 784)))
[L1160]     max_out = fc1_out.max().item()
[L1161]     output_scale = max_out / 127.0 if max_out != 0 else 1.0
[L1162]     output_zero_point = 0
[L1163] 
[L1164]     multiplier, shift, _ = compute_requant_params(input_scale, weight_scale, output_scale, shift=24)
[L1165] 
[L1166]     transform = transforms.Compose([
[L1167]         transforms.ToTensor(),   # 不做 Normalize，统一输入范围 [0,1]
[L1168]     ])
[L1169]     test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
[L1170] 
[L1171]     labels = []
[L1172] 
[L1173]     for i in range(num_samples):
[L1174]         img, label = test_dataset[i]
[L1175]         x = img.view(-1)  # [784]
[L1176] 
[L1177]         # 固定输入量化
[L1178]         q_x, _, _ = quantize_input_fixed_0_1(x)  # [784]
[L1179] 
[L1180]         # int32 累加： [128,784] x [784] -> [128]
[L1181]         acc = torch.sum(q_weight.to(torch.int32) * q_x.to(torch.int32).unsqueeze(0), dim=1) + q_bias
[L1182] 
[L1183]         # ReLU(int32)
[L1184]         relu_acc = torch.clamp(acc, min=0)
[L1185] 
[L1186]         # requant 到 int8，便于后级或观测
[L1187]         q_out = requantize_int32_to_int8(
[L1188]             relu_acc, multiplier, shift, out_zero_point=output_zero_point, relu=False
[L1189]         )
[L1190] 
[L1191]         # 保存
[L1192]         write_int8_hex_per_line(q_x,   os.path.join(out_dir, f"input_{i}.hex"))
[L1193]         write_int32_hex_per_line(acc,  os.path.join(out_dir, f"golden_fc1_acc_{i}.hex"))
[L1194]         write_int32_hex_per_line(relu_acc, os.path.join(out_dir, f"golden_fc1_relu_acc_{i}.hex"))
[L1195]         write_int8_hex_per_line(q_out, os.path.join(out_dir, f"golden_fc1_out_{i}.hex"))
[L1196] 
[L1197]         labels.append(int(label))
[L1198]         print(f"sample {i}: label={label}")
[L1199] 
[L1200]     with open(os.path.join(out_dir, "labels.txt"), "w") as f:
[L1201]         for lb in labels:
[L1202]             f.write(f"{lb}\n")
[L1203] 
[L1204]     print("Generated integer golden test vectors for route A.")
[L1205] ```
[L1206] 
[L1207] ---
[L1208] 
[L1209] # 4. 输入测试向量统一量化策略
[L1210] 
[L1211] 你要的是**贴近硬件即可**。那我建议你用下面这个方案。
[L1212] 
[L1213] ---
[L1214] 
[L1215] ## 推荐方案：输入保持 `[0,1]`，统一量化到 int8 `[0,127]`
[L1216] 
[L1217] ### 设计理由
[L1218] 这对 FPGA 最友好：
[L1219] 
[L1220] - 不用每张图动态算 scale
[L1221] - 不用保留 MNIST 的 mean/std normalize
[L1222] - 输入像素天然接近图像传感器/存储格式
[L1223] - 便于后面 BRAM 或 ROM 存图
[L1224] 
[L1225] ### 固定规则
[L1226] \[
[L1227] x_q = round(x \times 127)
[L1228] \]
[L1229] 其中 `x in [0,1]`
[L1230] 
[L1231] 即：
[L1232] 
[L1233] ```python
[L1234] def quantize_input_fixed_0_1(x):
[L1235]     scale = 1.0 / 127.0
[L1236]     zero_point = 0
[L1237]     q = torch.round(x / scale)
[L1238]     q = torch.clamp(q, 0, 127).to(torch.int8)
[L1239]     return q, scale, zero_point
[L1240] ```
[L1241] 
[L1242] ---
[L1243] 
[L1244] ## 为什么我不建议你继续用：
[L1245] ```python
[L1246] transforms.Normalize((0.1307,), (0.3081,))
[L1247] ```
[L1248] 
[L1249] 因为这会把输入变成带负数的浮点分布，虽然适合软件训练，但不适合你现在的 FPGA 验证阶段：
[L1250] 
[L1251] - 输入分布复杂
[L1252] - 量化参数更麻烦
[L1253] - debug 不直观
[L1254] 
[L1255] 所以你现在更适合：
[L1256] 
[L1257] - 训练和测试都直接 `ToTensor()`
[L1258] - 用统一固定输入量化
[L1259] 
[L1260] ---
[L1261] 
[L1262] # 5. 你当前权重输出顺序，对应硬件怎么设计读取顺序
[L1263] 
[L1264] 你当前 notebook 的写法是：
[L1265] 
[L1266] ```python
[L1267] for row in q_weight:
[L1268]     for val in row:
[L1269]         f.write(f'{val.item() & 0xFF:02x}\n')
[L1270] ```
[L1271] 
[L1272] 而 `weight = model.fc1.weight.data` 的 shape 是：
[L1273] 
[L1274] ```python
[L1275] [128, 784]
[L1276] ```
[L1277] 
[L1278] 在 PyTorch 里，`nn.Linear(784,128)` 的权重定义是：
[L1279] 
[L1280] \[
[L1281] W[j][i]
[L1282] \]
[L1283] 
[L1284] 其中：
[L1285] 
[L1286] - `j` 是输出神经元索引，范围 `0~127`
[L1287] - `i` 是输入特征索引，范围 `0~783`
[L1288] 
[L1289] ---
[L1290] 
[L1291] ## 你现在的导出顺序是什么
[L1292] 
[L1293] 它是：
[L1294] 
[L1295] - 先写第 0 个输出神经元的 784 个权重
[L1296] - 再写第 1 个输出神经元的 784 个权重
[L1297] - ...
[L1298] - 最后写第 127 个输出神经元的 784 个权重
[L1299] 
[L1300] 也就是：
[L1301] 
[L1302] \[
[L1303] W[0][0], W[0][1], ..., W[0][783], W[1][0], ..., W[127][783]
[L1304] \]
[L1305] 
[L1306] 这是典型的：
[L1307] 
[L1308] - **按输出通道优先**
[L1309] - **行优先 row-major**
[L1310] - **布局 `[out][in]`**
[L1311] 
[L1312] ---
[L1313] 
[L1314] ## 对应的硬件读取顺序应该怎么设计
[L1315] 
[L1316] 如果你的硬件是“一个输出神经元一轮累加”的思路，那么最自然。
[L1317] 
[L1318] ---
[L1319] 
[L1320] ## 硬件读取方式 1：逐个输出神经元计算
[L1321] 
[L1322] ### 第 `j` 个输出的计算
[L1323] \[
[L1324] y_j = \sum_{i=0}^{783} x_i \cdot W[j][i] + b_j
[L1325] \]
[L1326] 
[L1327] ### 内存地址设计
[L1328] 如果 `weights.hex` 直接线性装入 BRAM，则：
[L1329] 
[L1330] \[
[L1331] addr = j \times 784 + i
[L1332] \]
[L1333] 
[L1334] 所以：
[L1335] 
[L1336] - 固定一个输出神经元 `j`
[L1337] - 地址从 `j*784` 扫到 `j*784+783`
[L1338] - 同时输入向量 `x[0:783]` 顺序送入
[L1339] - MAC 累加完成后，加 `bias[j]`
[L1340] - 输出 `y_j`
[L1341] 
[L1342] 然后 `j++`
[L1343] 
[L1344] ---
[L1345] 
[L1346] ## 这种方式的优点
[L1347] - 和你现在导出的文件顺序完全一致
[L1348] - 地址生成最简单
[L1349] - 软件和硬件最容易对齐
[L1350] - 最适合第一版 FPGA 联调
[L1351] 
[L1352] ---
[L1353] 
[L1354] ## 你现在应该怎么设计硬件
[L1355] 
[L1356] ### 输入存储
[L1357] `input_i.hex` 顺序也是：
[L1358] \[
[L1359] x[0], x[1], ..., x[783]
[L1360] \]
[L1361] 
[L1362] ### 权重存储
[L1363] `weights.hex` 顺序：
[L1364] \[
[L1365] W[0][0], W[0][1], ..., W[0][783], W[1][0], ..., W[127][783]
[L1366] \]
[L1367] 
[L1368] ### bias 存储
[L1369] `bias.hex` 顺序：
[L1370] \[
[L1371] b[0], b[1], ..., b[127]
[L1372] \]
[L1373] 
[L1374] ---
[L1375] 
[L1376] ## 控制流程
[L1377] 对于输出神经元 `j`：
[L1378] 
[L1379] 1. `acc = 0`
[L1380] 2. 对 `i = 0 ~ 783`
[L1381]    - 读 `x[i]`
[L1382]    - 读 `W[j][i]`
[L1383]    - `acc += x[i] * W[j][i]`
[L1384] 3. `acc += bias[j]`
[L1385] 4. `acc = relu(acc)`
[L1386] 5. 写 `out[j]`
[L1387] 
[L1388] 然后进入下一个 `j`
[L1389] 
[L1390] ---
[L1391] 
[L1392] ## 地址公式
[L1393] ### 权重地址
[L1394] \[
[L1395] weight\_addr = out\_idx \times 784 + in\_idx
[L1396] \]
[L1397] 
[L1398] ### 输入地址
[L1399] \[
[L1400] input\_addr = in\_idx
[L1401] \]
[L1402] 
[L1403] ### 偏置地址
[L1404] \[
[L1405] bias\_addr = out\_idx
[L1406] \]
[L1407] 
[L1408] ---
[L1409] 
[L1410] ## 什么时候要改输出顺序
[L1411] 只有在你未来做这些架构时才需要变：
[L1412] 
[L1413] - 多 PE 并行读多个输出神经元
[L1414] - 按列块分块存储
[L1415] - CIM 阵列按 bitline/wordline 特定映射
[L1416] - 片上 SRAM 宽度要求按 8/16/32 个权重打包
[L1417] 
[L1418] 但对于你现在这个阶段，**完全没必要先改**。  
[L1419] 你当前顺序其实正适合第一版硬件。
[L1420] 
[L1421] ---
[L1422] 
[L1423] # 我给你的最终建议
[L1424] 
[L1425] ## 你现在最推荐的工程路线
[L1426] 先走：
[L1427] 
[L1428] - **路线 A**
[L1429] - 完整训练分类器
[L1430] - FPGA 只做第一层 `fc1 + ReLU`
[L1431] - 输入统一量化 `[0,1] -> int8 [0,127]`
[L1432] - 权重 int8 对称量化
[L1433] - bias int32
[L1434] - golden 用整数推理链路生成
[L1435] - 权重按你现在的 `[out][in]` 顺序导出
[L1436] 
[L1437] 这是最稳、最容易跑通的一条路。
[L1438] 
[L1439] ---
[L1440] 
[L1441] ## 你下一步最值得做的事
[L1442] 把你当前 notebook 分成三个独立脚本：
[L1443] 
[L1444] 1. `train_route_a.py`
[L1445] 2. `export_fc1_quant.py`
[L1446] 3. `generate_test_vectors_route_a.py`
[L1447] 
[L1448] 这样后面调试不会乱。
[L1449] 
[L1450] 如果你要，我下一条可以直接继续给你一份**整理好的、可直接复制运行的完整工程版代码**，把路线 A 的训练、量化导出、golden 生成合并成一个干净版本。
[L1451] 
[L1452] 用户:
[L1453] 关于1.现在没有在训练一个 MNIST 分类器的问题，请分别给我路线a和路线b的完整代码；
[L1454] 
[L1455] 关于2.量化函数有工程问题，不适合直接对接 FPGA，
[L1456] zero_point不是整数的问题帮我修改；每个张量动态量化这个问题给我一个合理的解决方案；偏置不能简单用 int8 单独量化请帮我提出解决方案和代码；
[L1457] 
[L1458] 关于3.generate_test_vectors() 生成的 golden，和硬件真实计算链路不一致，整数推理链路生成 golden，帮我完成设计；
[L1459] 
[L1460] 关于4输入测试向量的量化方式不适合硬件批量验证，帮我设计统一输入量化策略，贴近硬件即可；
[L1461] 
[L1462] 关于5. 导出的权重格式本身基本可用，但有几个注意点，这里文件顺序你先记住我的输出顺序，讲解一下我现在的输出顺序应该如何设计硬件读取顺序；
[L1463] 
[L1464] 
[L1465] 问题6,7你帮我解决一下。
[L1466] 
[L1467] 按照你提出的方案二：直接瞄准最终演示
[L1468] 
[L1469] 如果你们想最后做“MNIST 分类结果上板演示”，那就：
[L1470] 
[L1471] 模型改为两层 MLP
[L1472] 
[L1473] 训练用 CrossEntropyLoss
[L1474] 
[L1475] 做静态量化
[L1476] 
[L1477] 先用 Python 写一个整数推理参考实现
[L1478] 
[L1479] 再把这个整数推理结果作为 FPGA golden进行处理。
[L1480] 其他应该修改的你帮我修改，和我说一声即可。
[L1481] AI助手:
[L1482] 我先基于你现有 notebook 的结构，把方案 A、方案 B，以及静态量化、整数推理 golden、测试向量导出这几部分整理成能直接落地的代码。然后我会把硬件读取顺序也一并说明清楚，保证软件导出和 RTL 地址映射能闭环。
[L1483] 
[L1484] AI助手:
[L1485] 下面我直接按你的要求给出可落地代码和设计说明。  
[L1486] 我把内容分成两部分：
[L1487] 
[L1488] 1. **路线 A 完整代码**：训练完整 MLP，但 FPGA 先只实现第一层 `fc1 + relu`，用于硬件验证  
[L1489] 2. **路线 B 完整代码**：直接瞄准最终演示，做完整两层 MLP、静态量化、整数推理 golden 导出
[L1490] 
[L1491] 你最后明确说“**按方案二直接瞄准最终演示**”，所以我建议你真正落地时优先用**路线 B**。  
[L1492] 路线 A 我也完整给你，方便你前期做单层阵列验证。
[L1493] 
[L1494] ---
[L1495] 
[L1496] # 一、你提出的几个问题，我已经统一改掉了
[L1497] 
[L1498] 这次代码里我已经一起处理了这些问题：
[L1499] 
[L1500] ## 已修改的问题
[L1501] ### 1）训练目标错误
[L1502] - 路线 A / B 都改成了真正的分类训练
[L1503] - 使用 `CrossEntropyLoss`
[L1504] 
[L1505] ### 2）`zero_point` 不是整数
[L1506] - 已改为**整数 zero_point**
[L1507] - 并做了合法范围裁剪
[L1508] 
[L1509] ### 3）动态量化不适合 FPGA
[L1510] - 改为**静态量化**
[L1511] - 输入、权重、输出都用**固定 scale / zero_point**
[L1512] - 不再每个样本单独算量化参数
[L1513] 
[L1514] ### 4）bias 不能简单 int8
[L1515] - 改为：
[L1516]   - `bias_q = round(bias_fp / (input_scale * weight_scale))`
[L1517]   - 使用 **int32** 保存
[L1518] 
[L1519] ### 5）golden 与硬件链路不一致
[L1520] - 改为**整数推理参考链路**
[L1521] - golden 由整数乘加、加 bias、ReLU、requant 得到
[L1522] - 和 FPGA 数据通路一致
[L1523] 
[L1524] ### 6）输入量化方式不适合批量验证
[L1525] - 统一输入量化策略
[L1526] - 采用更贴近硬件的方案：**MNIST 像素 0~1 -> uint8 0~255，再映射为 int8（减去 128）**
[L1527] - 固定 `input_scale = 1/255`，`input_zero_point = -128`
[L1528] 
[L1529] ### 7）其他问题我也一起修了
[L1530] - `torch.load(..., map_location=device)`
[L1531] - 增加 test accuracy
[L1532] - 保存标签文件
[L1533] - 保存中间层 int32 累加结果，便于 RTL 分级定位
[L1534] - 导出 json 配置，方便 FPGA 端对齐
[L1535] 
[L1536] ---
[L1537] 
[L1538] # 二、统一量化策略设计
[L1539] 
[L1540] 你要求“贴近硬件即可”，我这里给你一个非常适合 FPGA 的设计：
[L1541] 
[L1542] ## 输入量化
[L1543] 原始 MNIST 图像来自 `ToTensor()` 后范围是 `[0,1]`
[L1544] 
[L1545] 统一量化为：
[L1546] 
[L1547] \[
[L1548] x_q = round(x / s_x) + z_x
[L1549] \]
[L1550] 
[L1551] 其中：
[L1552] 
[L1553] - `s_x = 1 / 255`
[L1554] - `z_x = -128`
[L1555] 
[L1556] 这样就等价于：
[L1557] 
[L1558] ```python
[L1559] x_u8 = round(x * 255)      # 0~255
[L1560] x_q  = x_u8 - 128          # -128~127
[L1561] ```
[L1562] 
[L1563] 优点：
[L1564] - 很贴近硬件
[L1565] - 不需要每张图动态量化
[L1566] - FPGA 端容易处理
[L1567] - BRAM 存储方便
[L1568] 
[L1569] ## 权重量化
[L1570] 对每一层权重做**对称量化**：
[L1571] 
[L1572] \[
[L1573] w_q = round(w / s_w)
[L1574] \]
[L1575] 
[L1576] 其中：
[L1577] 
[L1578] \[
[L1579] s_w = \max(|w|) / 127
[L1580] \]
[L1581] 
[L1582] - `w_zero_point = 0`
[L1583] - `w_q` 存为 int8
[L1584] 
[L1585] ## 偏置量化
[L1586] 偏置必须与累加器同量纲：
[L1587] 
[L1588] \[
[L1589] b_q = round(b / (s_x \cdot s_w))
[L1590] \]
[L1591] 
[L1592] - 存为 int32
[L1593] 
[L1594] ## 输出量化
[L1595] 累加后得到 int32：
[L1596] 
[L1597] \[
[L1598] acc = \sum (x_q - z_x) \cdot (w_q - z_w) + b_q
[L1599] \]
[L1600] 
[L1601] 然后 ReLU，再 requant：
[L1602] 
[L1603] \[
[L1604] y_q = round(acc \cdot (s_x s_w / s_y)) + z_y
[L1605] \]
[L1606] 
[L1607] 输出层如需分类 argmax，最终也可直接基于 int32 或 requant 后 int8 做。
[L1608] 
[L1609] ---
[L1610] 
[L1611] # 三、路线 A：完整代码  
[L1612] ## 用途
[L1613] - 软件训练完整分类器：`784 -> 128 -> 10`
[L1614] - FPGA 先只实现第一层：`fc1 + relu`
[L1615] - 导出第一层输入、权重、bias、int32 累加 golden、relu 输出 golden
[L1616] 
[L1617] 这条路线适合你前期快速验证 CIM/FC 阵列。
[L1618] 
[L1619] ---
[L1620] 
[L1621] ## 路线 A 完整代码
[L1622] 
[L1623] ```python
[L1624] import os
[L1625] import json
[L1626] import math
[L1627] import numpy as np
[L1628] import torch
[L1629] import torch.nn as nn
[L1630] import torch.optim as optim
[L1631] from torchvision import datasets, transforms
[L1632] from torch.utils.data import DataLoader
[L1633] 
[L1634] # =========================================================
[L1635] # 1. 配置
[L1636] # =========================================================
[L1637] DATA_DIR = "./data"
[L1638] OUT_DIR = "./route_a_output"
[L1639] MODEL_PATH = os.path.join(OUT_DIR, "mlp_route_a.pth")
[L1640] 
[L1641] os.makedirs(OUT_DIR, exist_ok=True)
[L1642] 
[L1643] DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
[L1644] 
[L1645] 
[L1646] # =========================================================
[L1647] # 2. 模型：完整训练，但导出时只导出第一层用于 FPGA
[L1648] # =========================================================
[L1649] class MLPRouteA(nn.Module):
[L1650]     def __init__(self):
[L1651]         super().__init__()
[L1652]         self.fc1 = nn.Linear(784, 128, bias=True)
[L1653]         self.relu = nn.ReLU()
[L1654]         self.fc2 = nn.Linear(128, 10, bias=True)
[L1655] 
[L1656]     def forward(self, x):
[L1657]         x = x.view(-1, 784)
[L1658]         h = self.fc1(x)
[L1659]         h_relu = self.relu(h)
[L1660]         y = self.fc2(h_relu)
[L1661]         return y
[L1662] 
[L1663] 
[L1664] # =========================================================
[L1665] # 3. 数据
[L1666] #   路线 A 训练时仍然可以用标准 Normalize
[L1667] #   因为这里只是先训练出一个可用分类器
[L1668] # =========================================================
[L1669] def get_train_test_loaders(batch_size=128):
[L1670]     transform = transforms.Compose([
[L1671]         transforms.ToTensor(),
[L1672]         transforms.Normalize((0.1307,), (0.3081,))
[L1673]     ])
[L1674] 
[L1675]     train_dataset = datasets.MNIST(
[L1676]         DATA_DIR, train=True, download=True, transform=transform
[L1677]     )
[L1678]     test_dataset = datasets.MNIST(
[L1679]         DATA_DIR, train=False, download=True, transform=transform
[L1680]     )
[L1681] 
[L1682]     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
[L1683]     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
[L1684]     return train_loader, test_loader
[L1685] 
[L1686] 
[L1687] # =========================================================
[L1688] # 4. 训练 / 测试
[L1689] # =========================================================
[L1690] def train_model(num_epochs=5, lr=1e-3):
[L1691]     train_loader, test_loader = get_train_test_loaders()
[L1692] 
[L1693]     model = MLPRouteA().to(DEVICE)
[L1694]     optimizer = optim.Adam(model.parameters(), lr=lr)
[L1695]     criterion = nn.CrossEntropyLoss()
[L1696] 
[L1697]     for epoch in range(num_epochs):
[L1698]         model.train()
[L1699]         running_loss = 0.0
[L1700] 
[L1701]         for data, target in train_loader:
[L1702]             data, target = data.to(DEVICE), target.to(DEVICE)
[L1703] 
[L1704]             optimizer.zero_grad()
[L1705]             logits = model(data)
[L1706]             loss = criterion(logits, target)
[L1707]             loss.backward()
[L1708]             optimizer.step()
[L1709] 
[L1710]             running_loss += loss.item() * data.size(0)
[L1711] 
[L1712]         avg_loss = running_loss / len(train_loader.dataset)
[L1713]         acc = evaluate_model(model, test_loader)
[L1714]         print(f"[Epoch {epoch+1}/{num_epochs}] loss={avg_loss:.6f}, test_acc={acc:.4f}")
[L1715] 
[L1716]     torch.save(model.state_dict(), MODEL_PATH)
[L1717]     print(f"Model saved to: {MODEL_PATH}")
[L1718]     return model
[L1719] 
[L1720] 
[L1721] def evaluate_model(model, test_loader):
[L1722]     model.eval()
[L1723]     correct = 0
[L1724]     total = 0
[L1725] 
[L1726]     with torch.no_grad():
[L1727]         for data, target in test_loader:
[L1728]             data, target = data.to(DEVICE), target.to(DEVICE)
[L1729]             logits = model(data)
[L1730]             pred = logits.argmax(dim=1)
[L1731]             correct += (pred == target).sum().item()
[L1732]             total += target.size(0)
[L1733] 
[L1734]     return correct / total
[L1735] 
[L1736] 
[L1737] # =========================================================
[L1738] # 5. 量化函数
[L1739] #   路线 A：为了 FPGA 只验证第一层，采用固定输入量化策略
[L1740] # =========================================================
[L1741] INT8_QMIN = -128
[L1742] INT8_QMAX = 127
[L1743] INT32_QMIN = -2147483648
[L1744] INT32_QMAX = 2147483647
[L1745] 
[L1746] def clamp(x, low, high):
[L1747]     return max(low, min(high, x))
[L1748] 
[L1749] def quantize_affine_tensor(tensor_fp, scale, zero_point, qmin=-128, qmax=127, dtype=torch.int8):
[L1750]     q = torch.round(tensor_fp / scale + zero_point)
[L1751]     q = torch.clamp(q, qmin, qmax)
[L1752]     return q.to(dtype)
[L1753] 
[L1754] def calc_symmetric_scale(tensor_fp, num_bits=8):
[L1755]     qmax = (2 ** (num_bits - 1)) - 1
[L1756]     max_abs = max(abs(tensor_fp.min().item()), abs(tensor_fp.max().item()))
[L1757]     if max_abs == 0:
[L1758]         return 1.0
[L1759]     return max_abs / qmax
[L1760] 
[L1761] def quantize_weight_symmetric_int8(weight_fp):
[L1762]     scale = calc_symmetric_scale(weight_fp, num_bits=8)
[L1763]     zp = 0
[L1764]     weight_q = quantize_affine_tensor(weight_fp, scale, zp, -128, 127, torch.int8)
[L1765]     return weight_q, scale, zp
[L1766] 
[L1767] def quantize_bias_int32(bias_fp, input_scale, weight_scale):
[L1768]     bias_scale = input_scale * weight_scale
[L1769]     bias_q = torch.round(bias_fp / bias_scale)
[L1770]     bias_q = torch.clamp(bias_q, INT32_QMIN, INT32_QMAX).to(torch.int32)
[L1771]     return bias_q, bias_scale
[L1772] 
[L1773] def quantize_input_hw_style(img_fp_01):
[L1774]     """
[L1775]     img_fp_01: 原始像素 [0,1]
[L1776]     统一策略：
[L1777]         x_u8 = round(x * 255)
[L1778]         x_q  = x_u8 - 128
[L1779]     对应：
[L1780]         input_scale = 1/255
[L1781]         input_zero_point = -128
[L1782]     """
[L1783]     input_scale = 1.0 / 255.0
[L1784]     input_zero_point = -128
[L1785] 
[L1786]     x_u8 = torch.round(img_fp_01 * 255.0)
[L1787]     x_u8 = torch.clamp(x_u8, 0, 255)
[L1788]     x_q = x_u8.to(torch.int16) - 128
[L1789]     x_q = torch.clamp(x_q, -128, 127).to(torch.int8)
[L1790] 
[L1791]     return x_q, input_scale, input_zero_point
[L1792] 
[L1793] 
[L1794] # =========================================================
[L1795] # 6. 第一层整数推理 golden
[L1796] #   只对应 fc1 + relu
[L1797] # =========================================================
[L1798] def fc1_integer_reference(x_q, w_q, b_q, x_zp=-128, w_zp=0):
[L1799]     """
[L1800]     x_q: [784] int8
[L1801]     w_q: [128, 784] int8
[L1802]     b_q: [128] int32
[L1803] 
[L1804]     acc_j = sum_i (x_q[i] - x_zp)*(w_q[j,i] - w_zp) + b_q[j]
[L1805]     relu_acc = max(acc_j, 0)
[L1806]     """
[L1807]     x_int = x_q.to(torch.int32) - int(x_zp)
[L1808]     w_int = w_q.to(torch.int32) - int(w_zp)
[L1809] 
[L1810]     acc = torch.matmul(w_int, x_int) + b_q
[L1811]     relu_acc = torch.clamp(acc, min=0)
[L1812]     return acc.to(torch.int32), relu_acc.to(torch.int32)
[L1813] 
[L1814] 
[L1815] # =========================================================
[L1816] # 7. HEX 导出
[L1817] # =========================================================
[L1818] def write_int8_hex_1d(tensor, path):
[L1819]     with open(path, "w") as f:
[L1820]         for v in tensor.flatten():
[L1821]             f.write(f"{int(v.item()) & 0xFF:02x}\n")
[L1822] 
[L1823] def write_int32_hex_1d(tensor, path):
[L1824]     with open(path, "w") as f:
[L1825]         for v in tensor.flatten():
[L1826]             f.write(f"{int(v.item()) & 0xFFFFFFFF:08x}\n")
[L1827] 
[L1828] def export_route_a_artifacts(num_samples=10):
[L1829]     # 加载训练后的模型
[L1830]     model = MLPRouteA().to(DEVICE)
[L1831]     model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
[L1832]     model.eval()
[L1833] 
[L1834]     # 只提取 fc1
[L1835]     fc1_w_fp = model.fc1.weight.data.cpu()   # [128, 784]
[L1836]     fc1_b_fp = model.fc1.bias.data.cpu()     # [128]
[L1837] 
[L1838]     # 固定输入量化参数
[L1839]     input_scale = 1.0 / 255.0
[L1840]     input_zero_point = -128
[L1841] 
[L1842]     # 权重量化（对称 int8）
[L1843]     fc1_w_q, fc1_w_scale, fc1_w_zp = quantize_weight_symmetric_int8(fc1_w_fp)
[L1844] 
[L1845]     # 偏置量化（int32）
[L1846]     fc1_b_q, fc1_b_scale = quantize_bias_int32(fc1_b_fp, input_scale, fc1_w_scale)
[L1847] 
[L1848]     # 导出权重：按 [out][in] 顺序
[L1849]     write_int8_hex_1d(fc1_w_q.reshape(-1), os.path.join(OUT_DIR, "fc1_weight_int8.hex"))
[L1850]     write_int32_hex_1d(fc1_b_q, os.path.join(OUT_DIR, "fc1_bias_int32.hex"))
[L1851] 
[L1852]     # 保存配置
[L1853]     config = {
[L1854]         "layer": "fc1_only",
[L1855]         "weight_shape": [128, 784],
[L1856]         "weight_layout": "row-major [out][in]",
[L1857]         "input_layout": "[in]",
[L1858]         "bias_layout": "[out]",
[L1859]         "input_scale": input_scale,
[L1860]         "input_zero_point": input_zero_point,
[L1861]         "fc1_weight_scale": fc1_w_scale,
[L1862]         "fc1_weight_zero_point": fc1_w_zp,
[L1863]         "fc1_bias_scale": fc1_b_scale,
[L1864]         "fc1_bias_dtype": "int32"
[L1865]     }
[L1866] 
[L1867]     with open(os.path.join(OUT_DIR, "quant_config.json"), "w", encoding="utf-8") as f:
[L1868]         json.dump(config, f, indent=2, ensure_ascii=False)
[L1869] 
[L1870]     # 测试集：为了导出原始像素，不能 Normalize
[L1871]     test_transform = transforms.Compose([
[L1872]         transforms.ToTensor()
[L1873]     ])
[L1874]     test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=test_transform)
[L1875] 
[L1876]     labels = []
[L1877] 
[L1878]     for i in range(num_samples):
[L1879]         img_fp, label = test_dataset[i]  # [1,28,28], range [0,1]
[L1880]         labels.append(int(label))
[L1881] 
[L1882]         img_flat = img_fp.view(-1).cpu()
[L1883]         x_q, _, _ = quantize_input_hw_style(img_flat)
[L1884] 
[L1885]         acc_int32, relu_int32 = fc1_integer_reference(
[L1886]             x_q, fc1_w_q, fc1_b_q, x_zp=input_zero_point, w_zp=fc1_w_zp
[L1887]         )
[L1888] 
[L1889]         write_int8_hex_1d(x_q, os.path.join(OUT_DIR, f"input_{i}.hex"))
[L1890]         write_int32_hex_1d(acc_int32, os.path.join(OUT_DIR, f"fc1_acc_golden_{i}.hex"))
[L1891]         write_int32_hex_1d(relu_int32, os.path.join(OUT_DIR, f"fc1_relu_golden_{i}.hex"))
[L1892] 
[L1893]         print(f"Exported sample {i}, label={label}")
[L1894] 
[L1895]     with open(os.path.join(OUT_DIR, "labels.txt"), "w") as f:
[L1896]         for lb in labels:
[L1897]             f.write(f"{lb}\n")
[L1898] 
[L1899]     print("Route A export done.")
[L1900] 
[L1901] 
[L1902] if __name__ == "__main__":
[L1903]     train_model(num_epochs=5, lr=1e-3)
[L1904]     export_route_a_artifacts(num_samples=10)
[L1905] ```
[L1906] 
[L1907] ---
[L1908] 
[L1909] # 四、路线 B：完整代码  
[L1910] ## 用途
[L1911] 这是你现在真正应该用的版本：
[L1912] 
[L1913] - 完整两层 MLP：`784 -> 128 -> 10`
[L1914] - `CrossEntropyLoss`
[L1915] - 静态量化
[L1916] - Python 写整数推理参考实现
[L1917] - 导出 FPGA golden
[L1918] - 最终瞄准“MNIST 上板分类演示”
[L1919] 
[L1920] ---
[L1921] 
[L1922] ## 路线 B 完整代码
[L1923] 
[L1924] ```python
[L1925] import os
[L1926] import json
[L1927] import math
[L1928] import numpy as np
[L1929] import torch
[L1930] import torch.nn as nn
[L1931] import torch.optim as optim
[L1932] from torchvision import datasets, transforms
[L1933] from torch.utils.data import DataLoader
[L1934] 
[L1935] # =========================================================
[L1936] # 1. 配置
[L1937] # =========================================================
[L1938] DATA_DIR = "./data"
[L1939] OUT_DIR = "./route_b_output"
[L1940] MODEL_PATH = os.path.join(OUT_DIR, "mlp_route_b.pth")
[L1941] 
[L1942] os.makedirs(OUT_DIR, exist_ok=True)
[L1943] 
[L1944] DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
[L1945] 
[L1946] INT8_QMIN = -128
[L1947] INT8_QMAX = 127
[L1948] INT32_QMIN = -2147483648
[L1949] INT32_QMAX = 2147483647
[L1950] 
[L1951] 
[L1952] # =========================================================
[L1953] # 2. 模型
[L1954] # =========================================================
[L1955] class MLPRouteB(nn.Module):
[L1956]     def __init__(self):
[L1957]         super().__init__()
[L1958]         self.fc1 = nn.Linear(784, 128, bias=True)
[L1959]         self.relu = nn.ReLU()
[L1960]         self.fc2 = nn.Linear(128, 10, bias=True)
[L1961] 
[L1962]     def forward(self, x):
[L1963]         x = x.view(-1, 784)
[L1964]         x = self.fc1(x)
[L1965]         x = self.relu(x)
[L1966]         x = self.fc2(x)
[L1967]         return x
[L1968] 
[L1969] 
[L1970] # =========================================================
[L1971] # 3. 数据
[L1972] #   训练时这里直接用 ToTensor()，不做 Normalize
[L1973] #   这样更贴近最终硬件部署链路
[L1974] # =========================================================
[L1975] def get_train_test_loaders(batch_size=128):
[L1976]     transform = transforms.Compose([
[L1977]         transforms.ToTensor()
[L1978]     ])
[L1979] 
[L1980]     train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
[L1981]     test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
[L1982] 
[L1983]     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
[L1984]     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
[L1985]     return train_loader, test_loader
[L1986] 
[L1987] 
[L1988] # =========================================================
[L1989] # 4. 训练 / 测试
[L1990] # =========================================================
[L1991] def evaluate_model(model, test_loader):
[L1992]     model.eval()
[L1993]     total = 0
[L1994]     correct = 0
[L1995] 
[L1996]     with torch.no_grad():
[L1997]         for data, target in test_loader:
[L1998]             data, target = data.to(DEVICE), target.to(DEVICE)
[L1999]             logits = model(data)
[L2000]             pred = torch.argmax(logits, dim=1)
[L2001]             correct += (pred == target).sum().item()
[L2002]             total += target.size(0)
[L2003] 
[L2004]     return correct / total
[L2005] 
[L2006] 
[L2007] def train_model(num_epochs=8, lr=1e-3):
[L2008]     train_loader, test_loader = get_train_test_loaders()
[L2009] 
[L2010]     model = MLPRouteB().to(DEVICE)
[L2011]     optimizer = optim.Adam(model.parameters(), lr=lr)
[L2012]     criterion = nn.CrossEntropyLoss()
[L2013] 
[L2014]     for epoch in range(num_epochs):
[L2015]         model.train()
[L2016]         running_loss = 0.0
[L2017] 
[L2018]         for data, target in train_loader:
[L2019]             data, target = data.to(DEVICE), target.to(DEVICE)
[L2020] 
[L2021]             optimizer.zero_grad()
[L2022]             logits = model(data)
[L2023]             loss = criterion(logits, target)
[L2024]             loss.backward()
[L2025]             optimizer.step()
[L2026] 
[L2027]             running_loss += loss.item() * data.size(0)
[L2028] 
[L2029]         avg_loss = running_loss / len(train_loader.dataset)
[L2030]         acc = evaluate_model(model, test_loader)
[L2031]         print(f"[Epoch {epoch+1}/{num_epochs}] loss={avg_loss:.6f}, test_acc={acc:.4f}")
[L2032] 
[L2033]     torch.save(model.state_dict(), MODEL_PATH)
[L2034]     print(f"Model saved to: {MODEL_PATH}")
[L2035]     return model
[L2036] 
[L2037] 
[L2038] # =========================================================
[L2039] # 5. 量化工具
[L2040] # =========================================================
[L2041] def clamp(v, low, high):
[L2042]     return max(low, min(high, v))
[L2043] 
[L2044] def quantize_affine_tensor(tensor_fp, scale, zero_point, qmin=-128, qmax=127, dtype=torch.int8):
[L2045]     q = torch.round(tensor_fp / scale + zero_point)
[L2046]     q = torch.clamp(q, qmin, qmax)
[L2047]     return q.to(dtype)
[L2048] 
[L2049] def calc_symmetric_scale(tensor_fp, num_bits=8):
[L2050]     qmax = (2 ** (num_bits - 1)) - 1
[L2051]     max_abs = max(abs(tensor_fp.min().item()), abs(tensor_fp.max().item()))
[L2052]     if max_abs == 0:
[L2053]         return 1.0
[L2054]     return max_abs / qmax
[L2055] 
[L2056] def calc_affine_params_from_range(min_val, max_val, num_bits=8, signed=True):
[L2057]     if signed:
[L2058]         qmin = -(2 ** (num_bits - 1))
[L2059]         qmax =  (2 ** (num_bits - 1)) - 1
[L2060]     else:
[L2061]         qmin = 0
[L2062]         qmax = (2 ** num_bits) - 1
[L2063] 
[L2064]     if max_val == min_val:
[L2065]         scale = 1.0
[L2066]         zero_point = 0
[L2067]         return scale, zero_point
[L2068] 
[L2069]     scale = (max_val - min_val) / float(qmax - qmin)
[L2070]     zero_point = round(qmin - min_val / scale)
[L2071]     zero_point = int(clamp(zero_point, qmin, qmax))
[L2072]     return float(scale), int(zero_point)
[L2073] 
[L2074] def quantize_weight_symmetric_int8(weight_fp):
[L2075]     scale = calc_symmetric_scale(weight_fp, num_bits=8)
[L2076]     zp = 0
[L2077]     q = quantize_affine_tensor(weight_fp, scale, zp, INT8_QMIN, INT8_QMAX, torch.int8)
[L2078]     return q, scale, zp
[L2079] 
[L2080] def quantize_bias_int32(bias_fp, input_scale, weight_scale):
[L2081]     bias_scale = input_scale * weight_scale
[L2082]     q = torch.round(bias_fp / bias_scale)
[L2083]     q = torch.clamp(q, INT32_QMIN, INT32_QMAX).to(torch.int32)
[L2084]     return q, bias_scale
[L2085] 
[L2086] def quantize_input_hw_style(img_fp_01):
[L2087]     """
[L2088]     固定输入量化：
[L2089]         x_u8 = round(x * 255)
[L2090]         x_q  = x_u8 - 128
[L2091]     对应：
[L2092]         scale = 1/255
[L2093]         zero_point = -128
[L2094]     """
[L2095]     scale = 1.0 / 255.0
[L2096]     zp = -128
[L2097] 
[L2098]     x_u8 = torch.round(img_fp_01 * 255.0)
[L2099]     x_u8 = torch.clamp(x_u8, 0, 255)
[L2100]     x_q = x_u8.to(torch.int16) - 128
[L2101]     x_q = torch.clamp(x_q, -128, 127).to(torch.int8)
[L2102]     return x_q, scale, zp
[L2103] 
[L2104] def choose_output_scale_from_activation(act_fp):
[L2105]     """
[L2106]     输出激活采用对称量化到 int8
[L2107]     """
[L2108]     scale = calc_symmetric_scale(act_fp, num_bits=8)
[L2109]     zp = 0
[L2110]     return scale, zp
[L2111] 
[L2112] def requantize_from_int32(acc_int32, real_multiplier, out_zero_point=0):
[L2113]     """
[L2114]     简单软件参考：
[L2115]         y_q = round(acc * real_multiplier) + out_zero_point
[L2116]     """
[L2117]     y = torch.round(acc_int32.to(torch.float32) * real_multiplier) + out_zero_point
[L2118]     y = torch.clamp(y, INT8_QMIN, INT8_QMAX)
[L2119]     return y.to(torch.int8)
[L2120] 
[L2121] def quantize_multiplier_to_int(real_multiplier):
[L2122]     """
[L2123]     给 FPGA 做定点乘法时可用的近似参数：
[L2124]       real_multiplier ≈ M / 2^shift
[L2125]     返回:
[L2126]       M(int), shift(int)
[L2127]     这里只提供给配置文件参考；软件 golden 仍直接用 float real_multiplier
[L2128]     """
[L2129]     if real_multiplier == 0:
[L2130]         return 0, 0
[L2131] 
[L2132]     shift = 0
[L2133]     m = real_multiplier
[L2134]     while m < 0.5:
[L2135]         m *= 2.0
[L2136]         shift += 1
[L2137]     M = int(round(m * (1 << 31)))  # Q31
[L2138]     return M, shift + 31
[L2139] 
[L2140] 
[L2141] # =========================================================
[L2142] # 6. 量化模型准备
[L2143] # =========================================================
[L2144] def build_quantized_model(model_fp):
[L2145]     model_fp.eval()
[L2146]     model_fp = model_fp.cpu()
[L2147] 
[L2148]     # 固定输入量化参数
[L2149]     input_scale = 1.0 / 255.0
[L2150]     input_zero_point = -128
[L2151] 
[L2152]     # fc1 量化
[L2153]     fc1_w_fp = model_fp.fc1.weight.data
[L2154]     fc1_b_fp = model_fp.fc1.bias.data
[L2155]     fc1_w_q, fc1_w_scale, fc1_w_zp = quantize_weight_symmetric_int8(fc1_w_fp)
[L2156]     fc1_b_q, fc1_b_scale = quantize_bias_int32(fc1_b_fp, input_scale, fc1_w_scale)
[L2157] 
[L2158]     # 用一批训练样本统计 fc1 输出尺度
[L2159]     calib_loader, _ = get_train_test_loaders(batch_size=256)
[L2160]     calib_data, _ = next(iter(calib_loader))
[L2161]     calib_data = calib_data.view(-1, 784)
[L2162]     with torch.no_grad():
[L2163]         fc1_act_fp = model_fp.relu(model_fp.fc1(calib_data))
[L2164]     fc1_out_scale, fc1_out_zp = choose_output_scale_from_activation(fc1_act_fp)
[L2165] 
[L2166]     fc1_real_multiplier = (input_scale * fc1_w_scale) / fc1_out_scale
[L2167]     fc1_M, fc1_shift = quantize_multiplier_to_int(fc1_real_multiplier)
[L2168] 
[L2169]     # fc2 量化
[L2170]     fc2_w_fp = model_fp.fc2.weight.data
[L2171]     fc2_b_fp = model_fp.fc2.bias.data
[L2172]     fc2_w_q, fc2_w_scale, fc2_w_zp = quantize_weight_symmetric_int8(fc2_w_fp)
[L2173]     fc2_b_q, fc2_b_scale = quantize_bias_int32(fc2_b_fp, fc1_out_scale, fc2_w_scale)
[L2174] 
[L2175]     # 用一批样本统计 fc2 输出尺度
[L2176]     with torch.no_grad():
[L2177]         logits_fp = model_fp(calib_data.view(-1, 1, 28, 28))
[L2178]     fc2_out_scale, fc2_out_zp = choose_output_scale_from_activation(logits_fp)
[L2179] 
[L2180]     fc2_real_multiplier = (fc1_out_scale * fc2_w_scale) / fc2_out_scale
[L2181]     fc2_M, fc2_shift = quantize_multiplier_to_int(fc2_real_multiplier)
[L2182] 
[L2183]     qparams = {
[L2184]         "input": {
[L2185]             "scale": input_scale,
[L2186]             "zero_point": input_zero_point,
[L2187]             "dtype": "int8"
[L2188]         },
[L2189]         "fc1": {
[L2190]             "weight_scale": fc1_w_scale,
[L2191]             "weight_zero_point": fc1_w_zp,
[L2192]             "bias_scale": fc1_b_scale,
[L2193]             "output_scale": fc1_out_scale,
[L2194]             "output_zero_point": fc1_out_zp,
[L2195]             "real_multiplier": fc1_real_multiplier,
[L2196]             "multiplier_q31": fc1_M,
[L2197]             "shift": fc1_shift
[L2198]         },
[L2199]         "fc2": {
[L2200]             "weight_scale": fc2_w_scale,
[L2201]             "weight_zero_point": fc2_w_zp,
[L2202]             "bias_scale": fc2_b_scale,
[L2203]             "output_scale": fc2_out_scale,
[L2204]             "output_zero_point": fc2_out_zp,
[L2205]             "real_multiplier": fc2_real_multiplier,
[L2206]             "multiplier_q31": fc2_M,
[L2207]             "shift": fc2_shift
[L2208]         }
[L2209]     }
[L2210] 
[L2211]     quantized = {
[L2212]         "fc1_w_q": fc1_w_q.to(torch.int8),
[L2213]         "fc1_b_q": fc1_b_q.to(torch.int32),
[L2214]         "fc2_w_q": fc2_w_q.to(torch.int8),
[L2215]         "fc2_b_q": fc2_b_q.to(torch.int32),
[L2216]         "qparams": qparams
[L2217]     }
[L2218]     return quantized
[L2219] 
[L2220] 
[L2221] # =========================================================
[L2222] # 7. 整数推理参考实现
[L2223] # =========================================================
[L2224] def linear_int8_int32(x_q, w_q, b_q, x_zp, w_zp):
[L2225]     """
[L2226]     x_q: [in_features] int8
[L2227]     w_q: [out_features, in_features] int8
[L2228]     b_q: [out_features] int32
[L2229]     """
[L2230]     x_int = x_q.to(torch.int32) - int(x_zp)
[L2231]     w_int = w_q.to(torch.int32) - int(w_zp)
[L2232]     acc = torch.matmul(w_int, x_int) + b_q
[L2233]     return acc.to(torch.int32)
[L2234] 
[L2235] def relu_int32(x):
[L2236]     return torch.clamp(x, min=0).to(torch.int32)
[L2237] 
[L2238] def integer_inference_reference(img_fp_01, quantized):
[L2239]     """
[L2240]     完整整数参考链路：
[L2241]       input -> fc1(int32 acc) -> relu -> requant to int8
[L2242]             -> fc2(int32 acc) -> requant to int8 logits
[L2243]     """
[L2244]     qparams = quantized["qparams"]
[L2245] 
[L2246]     # 1) 输入量化
[L2247]     x_q, _, _ = quantize_input_hw_style(img_fp_01)
[L2248] 
[L2249]     # 2) fc1 int32 累加
[L2250]     fc1_acc = linear_int8_int32(
[L2251]         x_q,
[L2252]         quantized["fc1_w_q"],
[L2253]         quantized["fc1_b_q"],
[L2254]         x_zp=qparams["input"]["zero_point"],
[L2255]         w_zp=qparams["fc1"]["weight_zero_point"]
[L2256]     )
[L2257] 
[L2258]     # 3) ReLU
[L2259]     fc1_relu = relu_int32(fc1_acc)
[L2260] 
[L2261]     # 4) fc1 输出 requant -> int8
[L2262]     fc1_out_q = requantize_from_int32(
[L2263]         fc1_relu,
[L2264]         qparams["fc1"]["real_multiplier"],
[L2265]         qparams["fc1"]["output_zero_point"]
[L2266]     )
[L2267] 
[L2268]     # 5) fc2 int32 累加
[L2269]     fc2_acc = linear_int8_int32(
[L2270]         fc1_out_q,
[L2271]         quantized["fc2_w_q"],
[L2272]         quantized["fc2_b_q"],
[L2273]         x_zp=qparams["fc1"]["output_zero_point"],
[L2274]         w_zp=qparams["fc2"]["weight_zero_point"]
[L2275]     )
[L2276] 
[L2277]     # 6) 输出 requant -> int8 logits
[L2278]     logits_q = requantize_from_int32(
[L2279]         fc2_acc,
[L2280]         qparams["fc2"]["real_multiplier"],
[L2281]         qparams["fc2"]["output_zero_point"]
[L2282]     )
[L2283] 
[L2284]     pred = int(torch.argmax(logits_q).item())
[L2285] 
[L2286]     return {
[L2287]         "input_q": x_q,
[L2288]         "fc1_acc_int32": fc1_acc,
[L2289]         "fc1_relu_int32": fc1_relu,
[L2290]         "fc1_out_q": fc1_out_q,
[L2291]         "fc2_acc_int32": fc2_acc,
[L2292]         "logits_q": logits_q,
[L2293]         "pred": pred
[L2294]     }
[L2295] 
[L2296] 
[L2297] # =========================================================
[L2298] # 8. HEX 导出
[L2299] # =========================================================
[L2300] def write_int8_hex_1d(tensor, path):
[L2301]     with open(path, "w") as f:
[L2302]         for v in tensor.flatten():
[L2303]             f.write(f"{int(v.item()) & 0xFF:02x}\n")
[L2304] 
[L2305] def write_int32_hex_1d(tensor, path):
[L2306]     with open(path, "w") as f:
[L2307]         for v in tensor.flatten():
[L2308]             f.write(f"{int(v.item()) & 0xFFFFFFFF:08x}\n")
[L2309] 
[L2310] def export_all_artifacts(model_fp, quantized, num_samples=20):
[L2311]     # 1) 导出权重 / bias
[L2312]     # 按 [out][in] 顺序导出
[L2313]     write_int8_hex_1d(quantized["fc1_w_q"].reshape(-1), os.path.join(OUT_DIR, "fc1_weight_int8.hex"))
[L2314]     write_int32_hex_1d(quantized["fc1_b_q"], os.path.join(OUT_DIR, "fc1_bias_int32.hex"))
[L2315] 
[L2316]     write_int8_hex_1d(quantized["fc2_w_q"].reshape(-1), os.path.join(OUT_DIR, "fc2_weight_int8.hex"))
[L2317]     write_int32_hex_1d(quantized["fc2_b_q"], os.path.join(OUT_DIR, "fc2_bias_int32.hex"))
[L2318] 
[L2319]     # 2) 保存量化配置
[L2320]     config = {
[L2321]         "network": "784 -> 128 -> 10",
[L2322]         "weight_layout": {
[L2323]             "fc1": "row-major [out][in] = [128][784]",
[L2324]             "fc2": "row-major [out][in] = [10][128]"
[L2325]         },
[L2326]         "input_layout": "[784]",
[L2327]         "hidden_layout": "[128]",
[L2328]         "output_layout": "[10]",
[L2329]         "qparams": quantized["qparams"]
[L2330]     }
[L2331] 
[L2332]     with open(os.path.join(OUT_DIR, "quant_config.json"), "w", encoding="utf-8") as f:
[L2333]         json.dump(config, f, indent=2, ensure_ascii=False)
[L2334] 
[L2335]     # 3) 导出样本和 golden
[L2336]     test_transform = transforms.Compose([
[L2337]         transforms.ToTensor()
[L2338]     ])
[L2339]     test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=test_transform)
[L2340] 
[L2341]     labels = []
[L2342]     preds = []
[L2343] 
[L2344]     for i in range(num_samples):
[L2345]         img_fp, label = test_dataset[i]
[L2346]         img_flat = img_fp.view(-1).cpu()
[L2347] 
[L2348]         result = integer_inference_reference(img_flat, quantized)
[L2349] 
[L2350]         labels.append(int(label))
[L2351]         preds.append(int(result["pred"]))
[L2352] 
[L2353]         write_int8_hex_1d(result["input_q"], os.path.join(OUT_DIR, f"input_{i}.hex"))
[L2354] 
[L2355]         write_int32_hex_1d(result["fc1_acc_int32"], os.path.join(OUT_DIR, f"fc1_acc_{i}.hex"))
[L2356]         write_int32_hex_1d(result["fc1_relu_int32"], os.path.join(OUT_DIR, f"fc1_relu_{i}.hex"))
[L2357]         write_int8_hex_1d(result["fc1_out_q"], os.path.join(OUT_DIR, f"fc1_out_{i}.hex"))
[L2358] 
[L2359]         write_int32_hex_1d(result["fc2_acc_int32"], os.path.join(OUT_DIR, f"fc2_acc_{i}.hex"))
[L2360]         write_int8_hex_1d(result["logits_q"], os.path.join(OUT_DIR, f"logits_{i}.hex"))
[L2361] 
[L2362]         with open(os.path.join(OUT_DIR, f"pred_{i}.txt"), "w") as f:
[L2363]             f.write(str(result["pred"]))
[L2364] 
[L2365]         print(f"sample={i}, label={label}, pred={result['pred']}")
[L2366] 
[L2367]     with open(os.path.join(OUT_DIR, "labels.txt"), "w") as f:
[L2368]         for lb in labels:
[L2369]             f.write(f"{lb}\n")
[L2370] 
[L2371]     with open(os.path.join(OUT_DIR, "preds.txt"), "w") as f:
[L2372]         for pd in preds:
[L2373]             f.write(f"{pd}\n")
[L2374] 
[L2375]     print("All artifacts exported.")
[L2376] 
[L2377] 
[L2378] # =========================================================
[L2379] # 9. 验证量化后整数推理精度
[L2380] # =========================================================
[L2381] def evaluate_integer_pipeline(model_fp, quantized, num_samples=1000):
[L2382]     test_transform = transforms.Compose([
[L2383]         transforms.ToTensor()
[L2384]     ])
[L2385]     test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=test_transform)
[L2386] 
[L2387]     correct = 0
[L2388]     total = min(num_samples, len(test_dataset))
[L2389] 
[L2390]     for i in range(total):
[L2391]         img_fp, label = test_dataset[i]
[L2392]         img_flat = img_fp.view(-1).cpu()
[L2393]         result = integer_inference_reference(img_flat, quantized)
[L2394]         if result["pred"] == int(label):
[L2395]             correct += 1
[L2396] 
[L2397]     acc = correct / total
[L2398]     print(f"Integer pipeline accuracy over {total} samples: {acc:.4f}")
[L2399]     return acc
[L2400] 
[L2401] 
[L2402] # =========================================================
[L2403] # 10. 主流程
[L2404] # =========================================================
[L2405] if __name__ == "__main__":
[L2406]     # 1) 训练浮点模型
[L2407]     model_fp = train_model(num_epochs=8, lr=1e-3)
[L2408] 
[L2409]     # 2) 加载模型（防止单独运行导出时也能用）
[L2410]     model_fp = MLPRouteB().to(DEVICE)
[L2411]     model_fp.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
[L2412]     model_fp.eval()
[L2413] 
[L2414]     # 3) 构建量化模型
[L2415]     quantized = build_quantized_model(model_fp)
[L2416] 
[L2417]     # 4) 评估整数链路
[L2418]     evaluate_integer_pipeline(model_fp, quantized, num_samples=1000)
[L2419] 
[L2420]     # 5) 导出 FPGA 所需文件
[L2421]     export_all_artifacts(model_fp, quantized, num_samples=20)
[L2422] ```
[L2423] 
[L2424] ---
[L2425] 
[L2426] # 五、关于问题 2：逐项回答
[L2427] 
[L2428] ---
[L2429] 
[L2430] ## 2.1 `zero_point` 不是整数，怎么改
[L2431] 
[L2432] 你原来的问题在这里：
[L2433] 
[L2434] ```python
[L2435] zero_point = qmin - min_val / scale
[L2436] ```
[L2437] 
[L2438] 这里会得到浮点数。应改成：
[L2439] 
[L2440] ```python
[L2441] zero_point = round(qmin - min_val / scale)
[L2442] zero_point = int(max(qmin, min(qmax, zero_point)))
[L2443] ```
[L2444] 
[L2445] 通用函数我已经在路线 B 里给了：
[L2446] 
[L2447] ```python
[L2448] def calc_affine_params_from_range(min_val, max_val, num_bits=8, signed=True):
[L2449]     if signed:
[L2450]         qmin = -(2 ** (num_bits - 1))
[L2451]         qmax =  (2 ** (num_bits - 1)) - 1
[L2452]     else:
[L2453]         qmin = 0
[L2454]         qmax = (2 ** num_bits) - 1
[L2455] 
[L2456]     if max_val == min_val:
[L2457]         scale = 1.0
[L2458]         zero_point = 0
[L2459]         return scale, zero_point
[L2460] 
[L2461]     scale = (max_val - min_val) / float(qmax - qmin)
[L2462]     zero_point = round(qmin - min_val / scale)
[L2463]     zero_point = int(clamp(zero_point, qmin, qmax))
[L2464]     return float(scale), int(zero_point)
[L2465] ```
[L2466] 
[L2467] ---
[L2468] 
[L2469] ## 2.2 每个张量动态量化不适合 FPGA，合理解决方案
[L2470] 
[L2471] ### 你现在的问题
[L2472] 你现在是：
[L2473] - 每张输入图像单独量化
[L2474] - 每次输出也单独量化
[L2475] 
[L2476] 这会导致 FPGA 需要不断切换 scale / zero_point，不利于验证和部署。
[L2477] 
[L2478] ### 合理方案
[L2479] 改为**静态量化**：
[L2480] 
[L2481] #### 输入
[L2482] 固定：
[L2483] - `input_scale = 1/255`
[L2484] - `input_zero_point = -128`
[L2485] 
[L2486] #### 权重
[L2487] 每层固定：
[L2488] - `weight_scale` 固定
[L2489] - `weight_zero_point = 0`
[L2490] 
[L2491] #### 输出
[L2492] 每层固定：
[L2493] - `output_scale` 固定
[L2494] - `output_zero_point` 固定
[L2495] 
[L2496] 这样每个样本都共用同一套参数，硬件最容易做。
[L2497] 
[L2498] ---
[L2499] 
[L2500] ## 2.3 偏置不能用 int8，怎么解决
[L2501] 
[L2502] ### 正确原则
[L2503] bias 必须与 int32 累加器同量纲：
[L2504] 
[L2505] \[
[L2506] b_q = round(\frac{b}{s_x \cdot s_w})
[L2507] \]
[L2508] 
[L2509] ### 代码
[L2510] 我已经写成：
[L2511] 
[L2512] ```python
[L2513] def quantize_bias_int32(bias_fp, input_scale, weight_scale):
[L2514]     bias_scale = input_scale * weight_scale
[L2515]     q = torch.round(bias_fp / bias_scale)
[L2516]     q = torch.clamp(q, INT32_QMIN, INT32_QMAX).to(torch.int32)
[L2517]     return q, bias_scale
[L2518] ```
[L2519] 
[L2520] ### 为什么要这样
[L2521] 因为硬件里先做：
[L2522] 
[L2523] \[
[L2524] acc = \sum (x_q-z_x)(w_q-z_w)
[L2525] \]
[L2526] 
[L2527] 这个 `acc` 是 int32 量纲。  
[L2528] 如果 bias 还是 int8，直接相加就是错的。
[L2529] 
[L2530] ---
[L2531] 
[L2532] # 六、关于问题 3：整数推理链路 golden，怎么设计
[L2533] 
[L2534] 你要求“整数推理链路生成 golden”，现在路线 B 已经完整实现了。
[L2535] 
[L2536] ---
[L2537] 
[L2538] ## 设计链路
[L2539] 
[L2540] ### 第一层
[L2541] \[
[L2542] acc_1 = W_1 \cdot (x_q-z_x) + b_{1q}
[L2543] \]
[L2544] 
[L2545] \[
[L2546] relu_1 = max(acc_1, 0)
[L2547] \]
[L2548] 
[L2549] \[
[L2550] h_q = requant(relu_1)
[L2551] \]
[L2552] 
[L2553] ### 第二层
[L2554] \[
[L2555] acc_2 = W_2 \cdot (h_q-z_{h}) + b_{2q}
[L2556] \]
[L2557] 
[L2558] \[
[L2559] logits_q = requant(acc_2)
[L2560] \]
[L2561] 
[L2562] ### 分类
[L2563] \[
[L2564] pred = argmax(logits_q)
[L2565] \]
[L2566] 
[L2567] ---
[L2568] 
[L2569] ## 代码入口
[L2570] 整数参考实现就在这里：
[L2571] 
[L2572] ```python
[L2573] def integer_inference_reference(img_fp_01, quantized):
[L2574]     ...
[L2575] ```
[L2576] 
[L2577] 它导出了：
[L2578] 
[L2579] - `input_q`
[L2580] - `fc1_acc_int32`
[L2581] - `fc1_relu_int32`
[L2582] - `fc1_out_q`
[L2583] - `fc2_acc_int32`
[L2584] - `logits_q`
[L2585] - `pred`
[L2586] 
[L2587] 这些就是你硬件联调最有价值的 golden。
[L2588] 
[L2589] ---
[L2590] 
[L2591] # 七、关于问题 4：统一输入量化策略
[L2592] 
[L2593] 你要求“贴近硬件即可”，我建议你就用下面这个，不要再 Normalize。
[L2594] 
[L2595] ---
[L2596] 
[L2597] ## 推荐方案
[L2598] MNIST 原始像素 `[0,1]`
[L2599] 
[L2600] 统一做：
[L2601] 
[L2602] ```python
[L2603] x_u8 = round(x * 255)
[L2604] x_q = x_u8 - 128
[L2605] ```
[L2606] 
[L2607] 对应：
[L2608] 
[L2609] - `input_scale = 1/255`
[L2610] - `input_zero_point = -128`
[L2611] 
[L2612] ---
[L2613] 
[L2614] ## 优点
[L2615] ### 软件上
[L2616] - 简单
[L2617] - 稳定
[L2618] - 不依赖每张图的分布
[L2619] 
[L2620] ### 硬件上
[L2621] - BRAM 存储直观
[L2622] - 每个像素 8 bit
[L2623] - 乘加前减去 `-128` 很明确
[L2624] - 适合 testbench 和最终上板
[L2625] 
[L2626] ---
[L2627] 
[L2628] # 八、关于问题 5：你当前输出顺序如何设计硬件读取顺序
[L2629] 
[L2630] 你现在权重导出顺序是：
[L2631] 
[L2632] ```python
[L2633] for row in q_weight:
[L2634]     for val in row:
[L2635]         f.write(...)
[L2636] ```
[L2637] 
[L2638] 而 `nn.Linear(in_features, out_features)` 在 PyTorch 里权重 shape 是：
[L2639] 
[L2640] \[
[L2641] [out\_features, in\_features]
[L2642] \]
[L2643] 
[L2644] 所以你现在导出的顺序是：
[L2645] 
[L2646] \[
[L2647] w[0,0], w[0,1], ..., w[0,783], w[1,0], ..., w[127,783]
[L2648] \]
[L2649] 
[L2650] 也就是：
[L2651] 
[L2652] ## 当前输出顺序
[L2653] **按输出神经元逐行展开，row-major `[out][in]`**
[L2654] 
[L2655] ---
[L2656] 
[L2657] ## 对应的硬件读取顺序应该怎么设计
[L2658] 
[L2659] 如果你的第一层是：
[L2660] 
[L2661] \[
[L2662] y_j = \sum_{i=0}^{783} x_i \cdot w_{j,i} + b_j
[L2663] \]
[L2664] 
[L2665] 那么最自然的硬件读取顺序就是：
[L2666] 
[L2667] ### 方案：按神经元顺序读
[L2668] 对于每个输出神经元 `j`：
[L2669] 
[L2670] 1. 读取该神经元的 784 个权重
[L2671] 2. 与输入 784 维逐项相乘累加
[L2672] 3. 加偏置 `b_j`
[L2673] 4. 输出 `y_j`
[L2674] 
[L2675] ### 地址映射
[L2676] 如果 `fc1_weight_int8.hex` 按 `[out][in]` 展开，则：
[L2677] 
[L2678] \[
[L2679] addr = j \times 784 + i
[L2680] \]
[L2681] 
[L2682] 即：
[L2683] - `j`：输出神经元编号
[L2684] - `i`：输入特征编号
[L2685] 
[L2686] ### RTL 里可写成
[L2687] - 外层循环：`j = 0..127`
[L2688] - 内层循环：`i = 0..783`
[L2689] 
[L2690] 这和你当前导出顺序是完全匹配的。
[L2691] 
[L2692] ---
[L2693] 
[L2694] ## 如果硬件是并行阵列怎么办
[L2695] 如果你做的是 CIM 阵列或 PE 并行，也仍然建议逻辑上保持这个顺序，只是物理存储时可分块：
[L2696] 
[L2697] 例如分成 16 路并行：
[L2698] - 每次读同一个 `j` 的 16 个连续权重
[L2699] - 输入也同步读 16 个元素
[L2700] - 做 16 路乘加
[L2701] - 局部累加后再归并
[L2702] 
[L2703] 本质上还是从 `[out][in]` 顺序切块读取。
[L2704] 
[L2705] ---
[L2706] 
[L2707] ## 第二层同理
[L2708] 第二层 `fc2` shape 是 `[10,128]`
[L2709] 
[L2710] 地址映射：
[L2711] 
[L2712] \[
[L2713] addr = j \times 128 + i
[L2714] \]
[L2715] 
[L2716] 其中：
[L2717] - `j = 0..9`
[L2718] - `i = 0..127`
[L2719] 
[L2720] ---
[L2721] 
[L2722] # 九、问题 6 和 7，我已经一起解决了
[L2723] 
[L2724] ---
[L2725] 
[L2726] ## 问题 6：`quant_params.npz` 没真正参与闭环
[L2727] 现在我不建议继续用 `npz` 做核心配置。  
[L2728] 我已经改成了：
[L2729] 
[L2730] - `quant_config.json`
[L2731] 
[L2732] 里面包含：
[L2733] - 输入 scale / zp
[L2734] - 每层 weight_scale / weight_zp
[L2735] - bias_scale
[L2736] - output_scale / output_zp
[L2737] - real_multiplier
[L2738] - multiplier_q31
[L2739] - shift
[L2740] 
[L2741] 这对硬件更友好，也更好查。
[L2742] 
[L2743] ---
[L2744] 
[L2745] ## 问题 7：其他工程问题
[L2746] 我已经一起处理：
[L2747] 
[L2748] ### 1）`torch.load()` 增加 `map_location`
[L2749] 已改：
[L2750] 
[L2751] ```python
[L2752] torch.load(MODEL_PATH, map_location=DEVICE)
[L2753] ```
[L2754] 
[L2755] ### 2）训练时增加准确率
[L2756] 每个 epoch 后打印 test accuracy
[L2757] 
[L2758] ### 3）保存 label
[L2759] 已导出：
[L2760] - `labels.txt`
[L2761] 
[L2762] ### 4）保存中间层 golden
[L2763] 已导出：
[L2764] - `fc1_acc_x.hex`
[L2765] - `fc1_relu_x.hex`
[L2766] - `fc1_out_x.hex`
[L2767] - `fc2_acc_x.hex`
[L2768] - `logits_x.hex`
[L2769] 
[L2770] 这样 RTL 调试时你能定位是哪一级错了。
[L2771] 
[L2772] ---
[L2773] 
[L2774] # 十、你最终应该选哪套
[L2775] 
[L2776] ## 如果你现在要交阶段成果、先验证阵列
[L2777] 用路线 A。  
[L2778] 因为它最容易把硬件链路先跑通。
[L2779] 
[L2780] ## 如果你现在就想直奔“MNIST 上板分类演示”
[L2781] 用路线 B。  
[L2782] 这就是你刚刚指定的目标，我也已经按这个目标把代码写全了。
[L2783] 
[L2784] ---
[L2785] 
[L2786] # 十一、我建议你下一步的落地顺序
[L2787] 
[L2788] 你现在最稳的执行顺序是：
[L2789] 
[L2790] ### 第一步
[L2791] 先跑路线 B 的 Python 脚本，得到：
[L2792] - 浮点模型精度
[L2793] - 整数量化链路精度
[L2794] - 全套导出文件
[L2795] 
[L2796] ### 第二步
[L2797] 先在 FPGA 上只实现第一层并对齐：
[L2798] - `input`
[L2799] - `fc1_weight`
[L2800] - `fc1_bias`
[L2801] - `fc1_acc`
[L2802] - `fc1_relu`
[L2803] - `fc1_out`
[L2804] 
[L2805] ### 第三步
[L2806] 再补第二层：
[L2807] - `fc2_weight`
[L2808] - `fc2_bias`
[L2809] - `fc2_acc`
[L2810] - `logits`
[L2811] - `argmax`
[L2812] 
[L2813] ---
[L2814] 
[L2815] 如果你愿意，我下一条可以继续直接给你一份**更适合 notebook 使用的版本**，也就是把路线 B 拆成多个单元格，按你现在 `ipynb` 的风格排好。
[L2816] 
[L2817] 用户:
[L2818] 现在已经把导出测试向量的部分写好了，请帮我分析导出的文件的格式，以及如何使用
[L2819] AI助手:
[L2820] 
[L2821] 
[L2822] AI助手:
[L2823] 我先把你当前导出的目录和文件内容抽样检查一下，确认每类文件到底是几位十六进制、按什么维度展开、对应哪一级数据通路。然后我会按“文件格式 → RTL/Testbench 怎么读 → 怎么对拍”的顺序给你整理清楚。
[L2824] 
[L2825] AI助手:
[L2826] 我根据你现在 notebook 里的导出代码，已经能把**导出文件格式**和**使用方法**说清楚了。你当前实际导出的文件主要有这几类：
[L2827] 
[L2828] - `weights.hex`
[L2829] - `bias.hex`
[L2830] - `quant_params.npz`
[L2831] - `input_i.hex`
[L2832] - `golden_i.hex`
[L2833] 
[L2834] 下面我按“是什么、怎么组织、硬件怎么读、联调怎么用”来讲。
[L2835] 
[L2836] ---
[L2837] 
[L2838] # 1. 你现在导出的文件格式
[L2839] 
[L2840] ## 1.1 `weights.hex`
[L2841] 
[L2842] 这是第一层全连接层 `fc1.weight` 的量化权重文件。
[L2843] 
[L2844] ### 来源
[L2845] 代码里：
[L2846] 
[L2847] ```python
[L2848] weight = model.fc1.weight.data  # [128, 784]
[L2849] ...
[L2850] with open(f'{output_dir}/weights.hex', 'w') as f:
[L2851]     for row in q_weight:
[L2852]         for val in row:
[L2853]             f.write(f'{val.item() & 0xFF:02x}\n')
[L2854] ```
[L2855] 
[L2856] ### 文件格式
[L2857] - 文本文件
[L2858] - **每行一个 8 bit 数值**
[L2859] - 用 **2 位十六进制** 表示
[L2860] - 负数采用 **8 位二补码的低 8 位**
[L2861] 
[L2862] 例如：
[L2863] - `00` 表示 0
[L2864] - `01` 表示 1
[L2865] - `7f` 表示 127
[L2866] - `ff` 表示 -1
[L2867] - `80` 表示 -128
[L2868] 
[L2869] ### 总行数
[L2870] 因为权重 shape 是 `[128, 784]`，所以总共有：
[L2871] 
[L2872] \[
[L2873] 128 \times 784 = 100352
[L2874] \]
[L2875] 
[L2876] 也就是：
[L2877] 
[L2878] - `weights.hex` 一共 **100352 行**
[L2879] 
[L2880] ### 展开顺序
[L2881] 当前是：
[L2882] 
[L2883] ```python
[L2884] for row in q_weight:
[L2885]     for val in row:
[L2886] ```
[L2887] 
[L2888] 所以顺序是：
[L2889] 
[L2890] \[
[L2891] w[0,0], w[0,1], ..., w[0,783], w[1,0], ..., w[127,783]
[L2892] \]
[L2893] 
[L2894] 也就是：
[L2895] 
[L2896] - **按输出神经元 row-major 展开**
[L2897] - 格式为 **`[out][in]`**
[L2898] 
[L2899] 这点非常重要，RTL 必须按这个顺序取。
[L2900] 
[L2901] ---
[L2902] 
[L2903] ## 1.2 `bias.hex`
[L2904] 
[L2905] 这是第一层偏置文件。
[L2906] 
[L2907] ### 来源
[L2908] ```python
[L2909] bias = model.fc1.bias.data      # [128]
[L2910] ...
[L2911] with open(f'{output_dir}/bias.hex', 'w') as f:
[L2912]     for val in q_bias:
[L2913]         f.write(f'{val.item() & 0xFF:02x}\n')
[L2914] ```
[L2915] 
[L2916] ### 文件格式
[L2917] - 文本文件
[L2918] - 每行一个量化后的偏置
[L2919] - **2 位十六进制**
[L2920] - 当前代码里它被导成了 **int8**
[L2921] 
[L2922] ### 总行数
[L2923] - 偏置长度是 128
[L2924] - 所以 `bias.hex` 一共 **128 行**
[L2925] 
[L2926] ### 当前问题
[L2927] 这个文件虽然格式上没问题，但**工程上有问题**：
[L2928] 
[L2929] - 你现在是把 bias 当 int8 存
[L2930] - 但硬件 MAC 累加结果通常是 int32
[L2931] - 所以这个 `bias.hex` 现在**不适合直接用于真实整数推理链路**
[L2932] 
[L2933] 也就是说：
[L2934] - 这个文件可以先拿来做“形式演示”
[L2935] - 但不适合做你后面要的最终整数 golden 对拍
[L2936] 
[L2937] ---
[L2938] 
[L2939] ## 1.3 `quant_params.npz`
[L2940] 
[L2941] 这是量化参数文件。
[L2942] 
[L2943] ### 来源
[L2944] ```python
[L2945] np.savez(f'{output_dir}/quant_params.npz',
[L2946]          w_scale=w_scale, w_zero=w_zero,
[L2947]          b_scale=b_scale, b_zero=b_zero)
[L2948] ```
[L2949] 
[L2950] ### 里面有什么
[L2951] 当前保存了：
[L2952] 
[L2953] - `w_scale`
[L2954] - `w_zero`
[L2955] - `b_scale`
[L2956] - `b_zero`
[L2957] 
[L2958] ### 用途
[L2959] 理论上它用于：
[L2960] - 软件端反量化
[L2961] - 硬件端知道量化参数
[L2962] 
[L2963] ### 当前实际情况
[L2964] 你现在的 `generate_test_vectors()` 虽然加载了它：
[L2965] 
[L2966] ```python
[L2967] quant_params = np.load('quantized_data/quant_params.npz')
[L2968] ```
[L2969] 
[L2970] 但后面**几乎没真正用上**。  
[L2971] 所以它现在更像一个“记录文件”，还不是硬件链路真正依赖的配置文件。
[L2972] 
[L2973] ---
[L2974] 
[L2975] ## 1.4 `input_i.hex`
[L2976] 
[L2977] 这是第 `i` 个测试样本的输入文件。
[L2978] 
[L2979] ### 来源
[L2980] ```python
[L2981] img, label = test_dataset[i]
[L2982] img_flat = img.view(-1, 784)
[L2983] q_img, img_scale, img_zero = quantize_tensor(img_flat)
[L2984] 
[L2985] with open(f'quantized_data/input_{i}.hex', 'w') as f:
[L2986]     for val in q_img.flatten():
[L2987]         f.write(f'{val.item() & 0xFF:02x}\n')
[L2988] ```
[L2989] 
[L2990] ### 文件格式
[L2991] - 文本文件
[L2992] - 每行一个输入元素
[L2993] - **2 位十六进制**
[L2994] - 量化后为 int8
[L2995] 
[L2996] ### 总行数
[L2997] MNIST 一张图是 `28×28=784`，所以：
[L2998] 
[L2999] - 每个 `input_i.hex` 一共 **784 行**
[L3000] 
[L3001] ### 排列顺序
[L3002] 因为：
[L3003] 
[L3004] ```python
[L3005] img_flat = img.view(-1, 784)
[L3006] for val in q_img.flatten():
[L3007] ```
[L3008] 
[L3009] 所以顺序就是图像展平后的顺序，通常是：
[L3010] 
[L3011] - 第 0 行：像素 `(0,0)`
[L3012] - 第 1 行：像素 `(0,1)`
[L3013] - ...
[L3014] - 第 27 行：像素 `(0,27)`
[L3015] - 第 28 行：像素 `(1,0)`
[L3016] - ...
[L3017] 
[L3018] 也就是：
[L3019] 
[L3020] - **按行优先 row-major 展开**
[L3021] - 输入向量索引 `i = row * 28 + col`
[L3022] 
[L3023] RTL 里要按这个顺序喂入。
[L3024] 
[L3025] ---
[L3026] 
[L3027] ## 1.5 `golden_i.hex`
[L3028] 
[L3029] 这是第 `i` 个测试样本的软件 golden 输出。
[L3030] 
[L3031] ### 来源
[L3032] ```python
[L3033] with torch.no_grad():
[L3034]     output = model(img_flat)
[L3035] 
[L3036] q_output, _, _ = quantize_tensor(output)
[L3037] with open(f'quantized_data/golden_{i}.hex', 'w') as f:
[L3038]     for val in q_output.flatten():
[L3039]         f.write(f'{val.item() & 0xFF:02x}\n')
[L3040] ```
[L3041] 
[L3042] ### 文件格式
[L3043] - 文本文件
[L3044] - 每行一个输出元素
[L3045] - **2 位十六进制**
[L3046] - 量化后为 int8
[L3047] 
[L3048] ### 总行数
[L3049] 你现在模型是：
[L3050] 
[L3051] ```python
[L3052] self.fc1 = nn.Linear(784,128)
[L3053] self.relu = nn.ReLU()
[L3054] ```
[L3055] 
[L3056] 所以输出是 128 维。
[L3057] 
[L3058] 也就是：
[L3059] - 每个 `golden_i.hex` 一共 **128 行**
[L3060] 
[L3061] ### 当前含义
[L3062] 这个 `golden_i.hex` 表示：
[L3063] 
[L3064] - 先用浮点模型跑出 `fc1 + relu` 的浮点输出
[L3065] - 再把这个浮点输出单独量化成 int8
[L3066] 
[L3067] 所以它的意义是：
[L3068] 
[L3069] - **“浮点模型输出的量化结果”**
[L3070] - 不是严格意义上的“整数硬件链路 golden”
[L3071] 
[L3072] 这点你后面做最终方案时必须记住。
[L3073] 
[L3074] ---
[L3075] 
[L3076] # 2. 这些文件如何使用
[L3077] 
[L3078] ---
[L3079] 
[L3080] ## 2.1 最直接的使用方式：Verilog `$readmemh`
[L3081] 
[L3082] 因为这些 `.hex` 文件是：
[L3083] - 文本
[L3084] - 每行一个十六进制数
[L3085] 
[L3086] 所以最适合直接在 Verilog 里用：
[L3087] 
[L3088] ```verilog
[L3089] reg [7:0] weight_mem [0:100351];
[L3090] reg [7:0] bias_mem   [0:127];
[L3091] reg [7:0] input_mem  [0:783];
[L3092] reg [7:0] golden_mem [0:127];
[L3093] 
[L3094] initial begin
[L3095]     $readmemh("weights.hex", weight_mem);
[L3096]     $readmemh("bias.hex", bias_mem);
[L3097]     $readmemh("input_0.hex", input_mem);
[L3098]     $readmemh("golden_0.hex", golden_mem);
[L3099] end
[L3100] ```
[L3101] 
[L3102] ---
[L3103] 
[L3104] ## 2.2 在 RTL 里如何解释这些 8 bit 数据
[L3105] 
[L3106] ### 注意
[L3107] `$readmemh` 读进来的是 `reg [7:0]`，本身只是无符号位模式。  
[L3108] 如果你想让它参与有符号运算，需要做符号解释。
[L3109] 
[L3110] 例如：
[L3111] 
[L3112] ```verilog
[L3113] wire signed [7:0] w = weight_mem[addr];
[L3114] wire signed [7:0] x = input_mem[idx];
[L3115] wire signed [7:0] b = bias_mem[out_idx];
[L3116] ```
[L3117] 
[L3118] 或者：
[L3119] 
[L3120] ```verilog
[L3121] wire signed [7:0] w = $signed(weight_mem[addr]);
[L3122] ```
[L3123] 
[L3124] 这样：
[L3125] - `8'hff` 才会被解释成 `-1`
[L3126] - `8'h80` 才会被解释成 `-128`
[L3127] 
[L3128] ---
[L3129] 
[L3130] ## 2.3 第一层硬件如何读取 `weights.hex`
[L3131] 
[L3132] 你当前权重顺序是：
[L3133] 
[L3134] \[
[L3135] w[j][i]
[L3136] \]
[L3137] 
[L3138] 其中：
[L3139] - `j = 0..127`，输出神经元
[L3140] - `i = 0..783`，输入索引
[L3141] 
[L3142] 所以硬件最自然的读取方式是：
[L3143] 
[L3144] ### 地址公式
[L3145] \[
[L3146] addr = j \times 784 + i
[L3147] \]
[L3148] 
[L3149] ### 计算流程
[L3150] 对于每个输出神经元 `j`：
[L3151] 
[L3152] 1. 累加器清零
[L3153] 2. 从 `i=0` 到 `783`
[L3154] 3. 读取：
[L3155]    - `x = input_mem[i]`
[L3156]    - `w = weight_mem[j*784+i]`
[L3157] 4. 做乘加：
[L3158]    \[
[L3159]    acc_j += x \cdot w
[L3160]    \]
[L3161] 5. 循环结束后加偏置：
[L3162]    \[
[L3163]    acc_j += bias[j]
[L3164]    \]
[L3165] 6. 过 ReLU
[L3166] 7. 输出第 `j` 个结果
[L3167] 
[L3168] ### 伪代码
[L3169] ```verilog
[L3170] for (j = 0; j < 128; j = j + 1) begin
[L3171]     acc = 0;
[L3172]     for (i = 0; i < 784; i = i + 1) begin
[L3173]         acc = acc + $signed(input_mem[i]) * $signed(weight_mem[j*784+i]);
[L3174]     end
[L3175]     acc = acc + $signed(bias_mem[j]);
[L3176]     if (acc < 0) acc = 0;
[L3177]     out[j] = acc; // 或量化后输出
[L3178] end
[L3179] ```
[L3180] 
[L3181] ---
[L3182] 
[L3183] ## 2.4 `input_i.hex` 如何使用
[L3184] 
[L3185] ### 在 testbench 中
[L3186] 每次测试一个样本：
[L3187] 
[L3188] ```verilog
[L3189] $readmemh("input_0.hex", input_mem);
[L3190] ```
[L3191] 
[L3192] 然后把 `input_mem[0:783]` 依次送给你的计算模块。
[L3193] 
[L3194] ### 两种常见喂法
[L3195] #### 方式 1：一次性 preload 到 BRAM
[L3196] 如果你的设计是：
[L3197] - 输入先装入片上 RAM
[L3198] - 再启动计算
[L3199] 
[L3200] 那就：
[L3201] 1. 先把 `input_i.hex` 用 `$readmemh` 读入 testbench memory
[L3202] 2. 再驱动写接口，把 784 个数据写入 DUT 内部 input RAM
[L3203] 3. 拉高 `start`
[L3204] 
[L3205] #### 方式 2：流式输入
[L3206] 如果你的 DUT 是流式输入：
[L3207] - 每拍输入一个 `x_i`
[L3208] 
[L3209] 那就：
[L3210] 1. 从 `input_mem[0]` 到 `input_mem[783]` 依次送
[L3211] 2. 每个时钟给一个有效数据
[L3212] 3. 送满 784 个后等待结果
[L3213] 
[L3214] ---
[L3215] 
[L3216] ## 2.5 `golden_i.hex` 如何使用
[L3217] 
[L3218] 这是当前 testbench 最直接的对拍基准。
[L3219] 
[L3220] ### 用法
[L3221] 在 testbench 中：
[L3222] 
[L3223] ```verilog
[L3224] $readmemh("golden_0.hex", golden_mem);
[L3225] ```
[L3226] 
[L3227] 当 DUT 计算完成后，把 DUT 的 128 个输出与 `golden_mem` 逐项比对。
[L3228] 
[L3229] ### 比对方式
[L3230] ```verilog
[L3231] integer k;
[L3232] initial begin
[L3233]     for (k = 0; k < 128; k = k + 1) begin
[L3234]         if (dut_out[k] !== golden_mem[k]) begin
[L3235]             $display("Mismatch at %0d: dut=%h, golden=%h", k, dut_out[k], golden_mem[k]);
[L3236]         end
[L3237]     end
[L3238] end
[L3239] ```
[L3240] 
[L3241] ### 但要注意
[L3242] 你现在的 `golden_i.hex` 是：
[L3243] - 浮点输出再量化
[L3244] 
[L3245] 所以前提是你的硬件计算链路也得尽量模拟这个逻辑。  
[L3246] 如果你 RTL 里已经做成了严格整数推理，那这个 golden 就不再是最优基准了。
[L3247] 
[L3248] ---
[L3249] 
[L3250] # 3. 你当前这套文件更适合什么阶段
[L3251] 
[L3252] 这套导出结果，最适合：
[L3253] 
[L3254] ## 适合
[L3255] - 第一层 `fc1 + relu` 的硬件原型验证
[L3256] - BRAM 读写流程验证
[L3257] - 地址映射验证
[L3258] - 符号位解释验证
[L3259] - testbench 对拍基本通路
[L3260] 
[L3261] ## 不适合
[L3262] - 最终的完整 MNIST 分类演示
[L3263] - 严格的整数推理验证
[L3264] - 偏置 int32 链路验证
[L3265] - 静态量化部署闭环
[L3266] 
[L3267] ---
[L3268] 
[L3269] # 4. 推荐你怎么用这批文件做联调
[L3270] 
[L3271] ---
[L3272] 
[L3273] ## 阶段 1：先验证存储与读取
[L3274] 先只做三件事：
[L3275] 
[L3276] 1. `weights.hex` 是否正确加载
[L3277] 2. `input_i.hex` 是否正确加载
[L3278] 3. `golden_i.hex` 是否能和当前 RTL 输出逐项对上
[L3279] 
[L3280] 只要这一步过了，说明：
[L3281] - 文件格式没问题
[L3282] - 读写顺序没问题
[L3283] - 地址映射没问题
[L3284] 
[L3285] ---
[L3286] 
[L3287] ## 阶段 2：确认你的计算顺序与导出顺序一致
[L3288] 你需要确认 RTL 的权重寻址是不是：
[L3289] 
[L3290] \[
[L3291] addr = out\_idx \times 784 + in\_idx
[L3292] \]
[L3293] 
[L3294] 如果不是，就会出现：
[L3295] - 数据都能读到
[L3296] - 结果却完全不对
[L3297] 
[L3298] 这是最常见的问题。
[L3299] 
[L3300] ---
[L3301] 
[L3302] ## 阶段 3：确认 signed 解释
[L3303] 这是第二常见问题。  
[L3304] 如果 RTL 里没有把 `8'hff` 当成 `-1`，而是当成 `255`，结果一定炸。
[L3305] 
[L3306] 所以一定要检查：
[L3307] - 输入 signed
[L3308] - 权重 signed
[L3309] - 偏置 signed
[L3310] - 乘法 signed
[L3311] - 累加器 signed
[L3312] 
[L3313] ---
[L3314] 
[L3315] # 5. 我建议你现在这样理解每个文件
[L3316] 
[L3317] 你可以把当前文件理解成这张表：
[L3318] 
[L3319] | 文件 | 含义 | 数据类型 | 行数 | 作用 |
[L3320] |---|---|---:|---:|---|
[L3321] | `weights.hex` | `fc1` 权重 | int8 | 100352 | DUT 权重存储初始化 |
[L3322] | `bias.hex` | `fc1` 偏置 | int8 | 128 | 当前版本偏置初始化 |
[L3323] | `quant_params.npz` | 量化参数记录 | float/int | 少量字段 | 软件辅助，不是闭环核心 |
[L3324] | `input_i.hex` | 第 i 个输入图像 | int8 | 784 | 测试输入 |
[L3325] | `golden_i.hex` | 第 i 个参考输出 | int8 | 128 | testbench 对拍 |
[L3326] 
[L3327] ---
[L3328] 
[L3329] # 6. 你现在这批文件的“正确使用姿势”
[L3330] 
[L3331] 如果你此刻不改导出逻辑，而是先拿现有文件验证硬件，那建议这样：
[L3332] 
[L3333] ### 你现在的 DUT 目标
[L3334] 做一个：
[L3335] - `784 -> 128`
[L3336] - int8 输入
[L3337] - int8 权重
[L3338] - int8 偏置
[L3339] - 累加
[L3340] - ReLU
[L3341] - 最后输出和 `golden_i.hex` 尽量一致
[L3342] 
[L3343] ### testbench 流程
[L3344] 1. `$readmemh("weights.hex", weight_mem)`
[L3345] 2. `$readmemh("bias.hex", bias_mem)`
[L3346] 3. `$readmemh("input_0.hex", input_mem)`
[L3347] 4. `$readmemh("golden_0.hex", golden_mem)`
[L3348] 5. 把 `input_mem` 送入 DUT
[L3349] 6. DUT 输出 128 维结果
[L3350] 7. 与 `golden_mem` 逐项比较
[L3351] 
[L3352] 这样可以先把系统跑起来。
[L3353] 
[L3354] ---
[L3355] 
[L3356] # 7. 你后面升级到最终方案时，文件会怎么变
[L3357] 
[L3358] 你后面如果按我们前面定的最终方案升级，文件会改成更合理的结构：
[L3359] 
[L3360] - `fc1_weight_int8.hex`
[L3361] - `fc1_bias_int32.hex`
[L3362] - `fc2_weight_int8.hex`
[L3363] - `fc2_bias_int32.hex`
[L3364] - `input_i.hex`
[L3365] - `fc1_acc_i.hex`
[L3366] - `fc1_relu_i.hex`
[L3367] - `fc1_out_i.hex`
[L3368] - `fc2_acc_i.hex`
[L3369] - `logits_i.hex`
[L3370] - `pred_i.txt`
[L3371] - `quant_config.json`
[L3372] 
[L3373] 这时每个文件的职责会更清晰，硬件对拍也更容易定位问题。
[L3374] 
[L3375] ---
[L3376] 
[L3377] # 8. 你现在最需要注意的两个坑
[L3378] 
[L3379] ## 坑 1：`bias.hex` 现在是 int8
[L3380] 这不是最终可部署形式。  
[L3381] 所以你现在用它时，只能把它当作“当前软件逻辑下的配套文件”，不要把它当最终硬件部署格式。
[L3382] 
[L3383] ## 坑 2：`golden_i.hex` 不是整数推理 golden
[L3384] 它现在只是：
[L3385] - 浮点输出
[L3386] - 再量化
[L3387] 
[L3388] 所以如果你 RTL 做得特别“硬件真实”，反而可能和它对不上。
[L3389] 
[L3390] ---
[L3391] 
[L3392] 如果你把**已经导出的几个 `.hex` 文件内容**贴给我几行，或者把你当前 `quantized_data/` 目录再发我，我可以继续帮你做第二步：**逐个文件按实际内容检查格式是否正确，并给你写一个对应的 Verilog testbench 读取模板**。
[L3393] 
[L3394] 用户:
[L3395] bias.hex:000008e5
[L3396] 0000183f
[L3397] 000001a2
[L3398] ffffe6b8
[L3399] fffffa66
[L3400] fffffdd6
[L3401] 00000acd
[L3402] 00000e73
[L3403] fffff423
[L3404] 00000244
[L3405] fffff947
[L3406] 00000cc4
[L3407] fffff784
[L3408] 0000061c
[L3409] ffffed80
[L3410] 00000dd7
[L3411] fffffffe
[L3412] fffff9de
[L3413] fffffb86
[L3414] ffffed26
[L3415] 00000c70
[L3416] 0000116d
[L3417] ffffff7f
[L3418] 000001a8
[L3419] 0000049e
[L3420] 0000118f
[L3421] 000006c6
[L3422] 00000e32
[L3423] 000025f7
[L3424] ffffe677
[L3425] 000005b3
[L3426] 00000b83
[L3427] 00002162
[L3428] 00001ce9
[L3429] 00000409
[L3430] fffffbcc
[L3431] 00000487
[L3432] 00000caa
[L3433] fffff5f9
[L3434] fffff633
[L3435] 000003b1
[L3436] 00000fd1
[L3437] 000005d1
[L3438] 00000ce3
[L3439] 0000106f
[L3440] 00000c5a
[L3441] 0000019f
[L3442] 00000681
[L3443] ffffef2b
[L3444] fffff49a
[L3445] ffffe4d9
[L3446] fffffb73
[L3447] 0000019f
[L3448] 00000dd1
[L3449] 0000016b
[L3450] 00000730
[L3451] fffff9a5
[L3452] 000009db
[L3453] 000008ec
[L3454] 000003c1
[L3455] 00000b59
[L3456] 00001263
[L3457] fffff71a
[L3458] 00001706
[L3459] 000013f9
[L3460] 000001e7
[L3461] 000027ca
[L3462] 00000779
[L3463] 000000d3
[L3464] fffffe21
[L3465] 0000051c
[L3466] 00000e9c
[L3467] 00000c2f
[L3468] fffffc59
[L3469] 0000142d
[L3470] 00000956
[L3471] 00000813
[L3472] fffffa40
[L3473] 00000d2a
[L3474] 00000c81
[L3475] fffffd14
[L3476] 000008ee
[L3477] 0000144a
[L3478] fffff816
[L3479] 00000be6
[L3480] 00000850
[L3481] 000012d6
[L3482] 00001bd6
[L3483] 00000388
[L3484] fffffc00
[L3485] 00001590
[L3486] 00000565
[L3487] 00000adb
[L3488] 00000f82
[L3489] 00000242
[L3490] 00001ed3
[L3491] fffff3bf
[L3492] 00001420
[L3493] 00000b0e
[L3494] 00000ac2
[L3495] fffffefd
[L3496] 00000072
[L3497] fffff4f2
[L3498] fffffc4f
[L3499] 00000904
[L3500] 00000a0f
[L3501] 0000078d
[L3502] 00000cba
[L3503] fffffb5e
[L3504] fffffe7a
[L3505] 00000a64
[L3506] 00000e22
[L3507] 00000435
[L3508] 00000642
[L3509] 00002492
[L3510] 00002235
[L3511] 000011d9
[L3512] fffff921
[L3513] ffffff6a
[L3514] 00001402
[L3515] 00001a36
[L3516] 0000002e
[L3517] 00001f8d
[L3518] ffffff3e
[L3519] fffff936
[L3520] 00000023
[L3521] fffffe93
[L3522] fffff438
[L3523] weight.hex:aa
[L3524] e5
[L3525] 00
[L3526] 0d
[L3527] 0d
[L3528] f7
[L3529] 08
[L3530] e8
[L3531] ff
[L3532] 20
[L3533] 15
[L3534] c5
[L3535] fe
[L3536] fb
[L3537] e2
[L3538] f9
[L3539] 16
[L3540] 15
[L3541] fc
[L3542] 15
[L3543] 09
[L3544] 13
[L3545] 09
[L3546] cb
[L3547] 28
[L3548] ca
[L3549] 0f
[L3550] 15
[L3551] 0f
[L3552] 05
[L3553] 16
[L3554] f3
[L3555] d7
[L3556] f6
[L3557] 06
[L3558] 0c
[L3559] d8
[L3560] 1e
[L3561] ef
[L3562] 03
[L3563] 12
[L3564] 04
[L3565] e6
[L3566] dd
[L3567] d3
[L3568] 0d
[L3569] fa
[L3570] 1c
[L3571] 1c
[L3572] 0f
[L3573] 0d
[L3574] fd
[L3575] 10
[L3576] 04
[L3577] 01
[L3578] b7
[L3579] 0d
[L3580] 00
[L3581] 08
[L3582] 15
[L3583] d2
[L3584] e8
[L3585] 0f
[L3586] f3
[L3587] bd
[L3588] 19
[L3589] e2
[L3590] 1a
[L3591] f1
[L3592] e2
[L3593] f5
[L3594] f2
[L3595] c6
[L3596] ff
[L3597] ec
[L3598] 02
[L3599] d8
[L3600] e9
[L3601] fe
[L3602] 0f
[L3603] 16
[L3604] d8
[L3605] d6
[L3606] 25
[L3607] e2
[L3608] 06
[L3609] bd
[L3610] 24
[L3611] 0d
[L3612] 1c
[L3613] c2
[L3614] 1c
[L3615] ee
[L3616] e0
[L3617] e7
[L3618] ff
[L3619] 05
[L3620] 01
[L3621] 0f
[L3622] ff
[L3623] fa
[L3624] 16
[L3625] 09
[L3626] 08
[L3627] df
[L3628] ff
[L3629] 12
[L3630] d2
[L3631] ea
[L3632] 0f
[L3633] b4
[L3634] 0f
[L3635] fb
[L3636] f4
[L3637] 24
[L3638] 81
[L3639] 13
[L3640] f1
[L3641] 17
[L3642] 0a
[L3643] ed
[L3644] 03
[L3645] f7
[L3646] ed
[L3647] c0
[L3648] e4
[L3649] da
[L3650] 19
[L3651] fb
[L3652] fd
[L3653] fa
[L3654] bf
[L3655] 14
[L3656] 15
[L3657] 17
[L3658] e1
[L3659] df
[L3660] f6
[L3661] f0
[L3662] 1c
[L3663] 10
[L3664] 0c
[L3665] e9
[L3666] cf
[L3667] e0
[L3668] 25
[L3669] eb
[L3670] e2
[L3671] 1e
[L3672] 13
[L3673] e9
[L3674] e9
[L3675] 18
[L3676] 0b
[L3677] dd
[L3678] eb
[L3679] fe
[L3680] ee
[L3681] 10
[L3682] 14
[L3683] 0c
[L3684] 23
[L3685] f1
[L3686] e9
[L3687] d9
[L3688] c1
[L3689] fa
[L3690] fd
[L3691] 16
[L3692] c9
[L3693] 2b
[L3694] 13
[L3695] 10
[L3696] d6
[L3697] 00
[L3698] ea
[L3699] ce
[L3700] e4
[L3701] e1
[L3702] d0
[L3703] ec
[L3704] 0e
[L3705] 0f
[L3706] 13
[L3707] f6
[L3708] f6
[L3709] 12
[L3710] 04
[L3711] 08
[L3712] 1b
[L3713] 0f
[L3714] 0b
[L3715] 02
[L3716] ea
[L3717] d1
[L3718] da
[L3719] e3
[L3720] e4
[L3721] dc
[L3722] 1b
[L3723] d0
[L3724] 18
[L3725] f0
[L3726] 02
[L3727] 48
[L3728] 06
[L3729] 20
[L3730] 2a
[L3731] e4
[L3732] 1b
[L3733] 0e
[L3734] cb
[L3735] 14
[L3736] df
[L3737] 3a
[L3738] 47
[L3739] 0e
[L3740] 2c
[L3741] 02
[L3742] e6
[L3743] 01
[L3744] 21
[L3745] 00
[L3746] f3
[L3747] e4
[L3748] cd
[L3749] c9
[L3750] d5
[L3751] df
[L3752] f1
[L3753] cd
[L3754] ed
[L3755] 18
[L3756] 11
[L3757] 0e
[L3758] 34
[L3759] 16
[L3760] f1
[L3761] 14
[L3762] db
[L3763] fd
[L3764] 0e
[L3765] 34
[L3766] 30
[L3767] df
[L3768] ec
[L3769] de
[L3770] f2
[L3771] cf
[L3772] e1
[L3773] c0
[L3774] fd
[L3775] d6
[L3776] 10
[L3777] 4f
[L3778] 1c
[L3779] 03
[L3780] 25
[L3781] 05
[L3782] fa
[L3783] fd
[L3784] 19
[L3785] 13
[L3786] cb
[L3787] 08
[L3788] 20
[L3789] 03
[L3790] 39
[L3791] 0b
[L3792] 1e
[L3793] ea
[L3794] f3
[L3795] 0b
[L3796] 21
[L3797] ee
[L3798] 16
[L3799] 03
[L3800] af
[L3801] 14
[L3802] ae
[L3803] 07
[L3804] ff
[L3805] f4
[L3806] 0c
[L3807] aa
[L3808] 09
[L3809] e6
[L3810] e9
[L3811] d2
[L3812] 24
[L3813] e8
[L3814] 16
[L3815] 0d
[L3816] f8
[L3817] 14
[L3818] 28
[L3819] fb
[L3820] df
[L3821] 9c
[L3822] 05
[L3823] 18
[L3824] f0
[L3825] 03
[L3826] 45
[L3827] 1e
[L3828] f8
[L3829] 0a
[L3830] fc
[L3831] 04
[L3832] 32
[L3833] 26
[L3834] 0c
[L3835] 1b
[L3836] e6
[L3837] 1a
[L3838] 20
[L3839] ff
[L3840] d9
[L3841] 19
[L3842] b8
[L3843] 12
[L3844] 1b
[L3845] 13
[L3846] 11
[L3847] e8
[L3848] 0b
[L3849] e1
[L3850] 26
[L3851] d4
[L3852] 11
[L3853] e2
[L3854] 92
[L3855] e0
[L3856] fb
[L3857] f5
[L3858] 15
[L3859] 1c
[L3860] 23
[L3861] 15
[L3862] f7
[L3863] 09
[L3864] e4
[L3865] 13
[L3866] dc
[L3867] f9
[L3868] fe
[L3869] f2
[L3870] 14
[L3871] 0a
[L3872] cc
[L3873] eb
[L3874] 02
[L3875] 07
[L3876] fa
[L3877] d1
[L3878] dc
[L3879] 1e
[L3880] 13
[L3881] ec
[L3882] 10
[L3883] 1e
[L3884] 00
[L3885] 1f
[L3886] 13
[L3887] 1a
[L3888] db
[L3889] 16
[L3890] 08
[L3891] 03
[L3892] 0e
[L3893] fe
[L3894] e0
[L3895] 37
[L3896] 0c
[L3897] f0
[L3898] 1a
[L3899] 0e
[L3900] 10
[L3901] 36
[L3902] d2
[L3903] 07
[L3904] fc
[L3905] bb
[L3906] d4
[L3907] db
[L3908] f7
[L3909] 00
[L3910] 0f
[L3911] 0b
[L3912] 0f
[L3913] da
[L3914] d7
[L3915] 08
[L3916] 06
[L3917] 0b
[L3918] 05
[L3919] ff
[L3920] 16
[L3921] 34
[L3922] 02
[L3923] e1
[L3924] 0d
[L3925] 1a
[L3926] 0f
[L3927] 9a
[L3928] dc
[L3929] e8
[L3930] 1e
[L3931] dd
[L3932] 0f
[L3933] f0
[L3934] d8
[L3935] f5
[L3936] 0f
[L3937] f0
[L3938] 05
[L3939] 95
[L3940] c1
[L3941] ec
[L3942] 19
[L3943] 35
[L3944] eb
[L3945] 08
[L3946] 2d
[L3947] 2e
[L3948] db
[L3949] 28
[L3950] fc
[L3951] dd
[L3952] f0
[L3953] f2
[L3954] 12
[L3955] 1a
[L3956] fd
[L3957] 18
[L3958] fa
[L3959] ec
[L3960] cb
[L3961] f7
[L3962] 0a
[L3963] f0
[L3964] 10
[L3965] c4
[L3966] 0a
[L3967] 14
[L3968] e1
[L3969] 0c
[L3970] 03
[L3971] 06
[L3972] 07
[L3973] e9
[L3974] 04
[L3975] 12
[L3976] 0f
[L3977] 08
[L3978] c5
[L3979] 0d
[L3980] fb
[L3981] d1
[L3982] 21
[L3983] dd
[L3984] 13
[L3985] d9
[L3986] ee
[L3987] 0d
[L3988] 09
[L3989] fb
[L3990] 02
[L3991] 00
[L3992] ff
[L3993] e4
[L3994] d3
[L3995] 14
[L3996] f3
[L3997] ce
[L3998] c6
[L3999] 10
[L4000] 1f
[L4001] 13
[L4002] 0d
[L4003] 05
[L4004] 1d
[L4005] d3
[L4006] 22
[L4007] 2b
[L4008] fb
[L4009] 15
[L4010] 07
[L4011] cf
[L4012] 09
[L4013] f3
[L4014] 26
[L4015] 13
[L4016] bd
[L4017] 04
[L4018] e6
[L4019] f6
[L4020] 11
[L4021] f1
[L4022] 46
[L4023] ef
[L4024] 0a
[L4025] f9
[L4026] e6
[L4027] cb
[L4028] 0c
[L4029] eb
[L4030] 1d
[L4031] 16
[L4032] 08
[L4033] 0e
[L4034] 2c
[L4035] e8
[L4036] 08
[L4037] fa
[L4038] ee
[L4039] d2
[L4040] dd
[L4041] 19
[L4042] 1d
[L4043] ce
[L4044] 9c
[L4045] fa
[L4046] a2
[L4047] db
[L4048] 0c
[L4049] c8
[L4050] f5
[L4051] d5
[L4052] ca
[L4053] ea
[L4054] ed
[L4055] 1e
[L4056] 22
[L4057] 19
[L4058] 1f
[L4059] f0
[L4060] 0b
[L4061] 04
[L4062] 07
[L4063] de
[L4064] 04
[L4065] 00
[L4066] 09
[L4067] fc
[L4068] 17
[L4069] 09
[L4070] ec
[L4071] 12
[L4072] c4
[L4073] 8a
[L4074] 0e
[L4075] ec
[L4076] 04
[L4077] 0d
[L4078] e7
[L4079] 12
[L4080] 15
[L4081] 0b
[L4082] 8f
[L4083] d2
[L4084] e7
[L4085] c2
[L4086] 0f
[L4087] f7
[L4088] 12
[L4089] dc
[L4090] 0d
[L4091] e1
[L4092] 18
[L4093] 07
[L4094] d4
[L4095] 08
[L4096] 28
[L4097] ab
[L4098] 0a
[L4099] b4
[L4100] 1c
[L4101] 0b
[L4102] 05
[L4103] 02
[L4104] 0b
[L4105] 11
[L4106] 22
[L4107] 02
[L4108] e9
[L4109] 14
[L4110] e9
[L4111] a7
[L4112] 15
[L4113] bb
[L4114] 08
[L4115] e1
[L4116] fd
[L4117] 00
[L4118] e4
[L4119] f0
[L4120] 0d
[L4121] 23
[L4122] 29
[L4123] ba
[L4124] 2b
[L4125] 09
[L4126] 0c
[L4127] 13
[L4128] 47
[L4129] ff
[L4130] 0a
[L4131] 14
[L4132] 0e
[L4133] 1a
[L4134] f8
[L4135] 97
[L4136] d4
[L4137] e0
[L4138] eb
[L4139] 0a
[L4140] 0f
[L4141] ed
[L4142] 3f
[L4143] 18
[L4144] 15
[L4145] a5
[L4146] 10
[L4147] 07
[L4148] 13
[L4149] 2d
[L4150] 0f
[L4151] 12
[L4152] f6
[L4153] 0d
[L4154] 09
[L4155] 0f
[L4156] 05
[L4157] 08
[L4158] e9
[L4159] 14
[L4160] 11
[L4161] f1
[L4162] bb
[L4163] 38
[L4164] 1f
[L4165] ff
[L4166] 02
[L4167] 05
[L4168] ff
[L4169] e2
[L4170] 38
[L4171] fd
[L4172] 0b
[L4173] 0a
[L4174] 3a
[L4175] ec
[L4176] b2
[L4177] 2d
[L4178] f0
[L4179] db
[L4180] cf
[L4181] ed
[L4182] eb
[L4183] c4
[L4184] 05
[L4185] df
[L4186] 03
[L4187] a5
[L4188] 02
[L4189] 02
[L4190] 12
[L4191] 46
[L4192] 00
[L4193] 0b
[L4194] ee
[L4195] 22
[L4196] e7
[L4197] f0
[L4198] 14
[L4199] ca
[L4200] 0d
[L4201] 2e
[L4202] f1
[L4203] ba
[L4204] 05
[L4205] 2c
[L4206] 0c
[L4207] ec
[L4208] 0d
[L4209] fa
[L4210] 03
[L4211] fb
[L4212] 0b
[L4213] f4
[L4214] e7
[L4215] 12
[L4216] 98
[L4217] c2
[L4218] 08
[L4219] d2
[L4220] 04
[L4221] f3
[L4222] 1b
[L4223] 09
[L4224] f8
[L4225] f9
[L4226] f2
[L4227] 28
[L4228] f9
[L4229] 30
[L4230] e9
[L4231] 06
[L4232] e5
[L4233] f6
[L4234] c2
[L4235] 13
[L4236] f5
[L4237] 14
[L4238] 21
[L4239] 50
[L4240] c4
[L4241] 07
[L4242] e2
[L4243] f3
[L4244] 0d
[L4245] fa
[L4246] f3
[L4247] 0c
[L4248] 0c
[L4249] 08
[L4250] 15
[L4251] 10
[L4252] e9
[L4253] 10
[L4254] 0b
[L4255] 0e
[L4256] 01
[L4257] eb
[L4258] 12
[L4259] ed
[L4260] 0b
[L4261] 0f
[L4262] 10
[L4263] d0
[L4264] 0c
[L4265] 1a
[L4266] fb
[L4267] ca
[L4268] cf
[L4269] f3
[L4270] cb
[L4271] bd
[L4272] ff
[L4273] 36
[L4274] f6
[L4275] fe
[L4276] f8
[L4277] f2
[L4278] 42
[L4279] 02
[L4280] f8
[L4281] 06
[L4282] 12
[L4283] 1d
[L4284] ce
[L4285] 14
[L4286] 0c
[L4287] 17
[L4288] fb
[L4289] 0e
[L4290] ed
[L4291] 17
[L4292] 15
[L4293] 0c
[L4294] 15
[L4295] 02
[L4296] e6
[L4297] 0f
[L4298] 10
[L4299] 06
[L4300] 02
[L4301] 0c
[L4302] b4
[L4303] 06
[L4304] d7
[L4305] af
[L4306] 2f
[L4307] de
[L4308] e1
[L4309] 40
[L4310] ee
[L4311] ef
[L4312] f3
[L4313] 0c
[L4314] aa
[L4315] 21
[L4316] 0a
[L4317] c6
[L4318] fe
[L4319] f3
[L4320] ff
[L4321] 1f
[L4322] e6
[L4323] 3e
[L4324] 08
[L4325] f8
[L4326] d0
[L4327] d8
[L4328] f3
[L4329] 26
[L4330] 06
[L4331] 19
[L4332] 08
[L4333] fc
[L4334] fe
[L4335] 18
[L4336] eb
[L4337] f5
[L4338] d2
[L4339] d5
[L4340] f4
[L4341] e3
[L4342] 18
[L4343] 10
[L4344] f7
[L4345] ab
[L4346] d3
[L4347] f7
[L4348] ed
[L4349] 16
[L4350] c1
[L4351] 04
[L4352] df
[L4353] f6
[L4354] e9
[L4355] 18
[L4356] 11
[L4357] a9
[L4358] da
[L4359] d3
[L4360] f0
[L4361] 04
[L4362] 15
[L4363] d7
[L4364] 05
[L4365] 04
[L4366] 17
[L4367] 17
[L4368] bf
[L4369] 23
[L4370] ee
[L4371] db
[L4372] ec
[L4373] ff
[L4374] 0f
[L4375] df
[L4376] ea
[L4377] fb
[L4378] 1d
[L4379] ef
[L4380] 08
[L4381] ef
[L4382] e2
[L4383] f0
[L4384] 0b
[L4385] 13
[L4386] e4
[L4387] 0e
[L4388] 12
[L4389] 18
[L4390] ff
[L4391] e1
[L4392] 0e
[L4393] fb
[L4394] 10
[L4395] 05
[L4396] b3
[L4397] 16
[L4398] 08
[L4399] dd
[L4400] 2e
[L4401] e0
[L4402] 12
[L4403] fa
[L4404] ee
[L4405] 17
[L4406] ab
[L4407] 16
[L4408] 0a
[L4409] 0b
[L4410] 0f
[L4411] 15
[L4412] e6
[L4413] 23
[L4414] e4
[L4415] cc
[L4416] 17
[L4417] 16
[L4418] 11
[L4419] 0f
[L4420] a8
[L4421] ff
[L4422] c9
[L4423] e8
[L4424] 2b
[L4425] d4
[L4426] 04
[L4427] f7
[L4428] 07
[L4429] e4
[L4430] 1f
[L4431] eb
[L4432] 01
[L4433] 9e
[L4434] 14
[L4435] 1a
[L4436] 01
[L4437] 11
[L4438] 13
[L4439] 25
[L4440] 03
[L4441] 04
[L4442] f1
[L4443] db
[L4444] 09
[L4445] 05
[L4446] db
[L4447] f2
[L4448] d9
[L4449] dc
[L4450] 1a
[L4451] cc
[L4452] 22
[L4453] 14
[L4454] 09
[L4455] 03
[L4456] 08
[L4457] 0b
[L4458] 01
[L4459] 09
[L4460] 0f
[L4461] d5
[L4462] c9
[L4463] d0
[L4464] f2
[L4465] f7
[L4466] 28
[L4467] 08
[L4468] e6
[L4469] 0a
[L4470] e2
[L4471] d2
[L4472] 18
[L4473] 1d
[L4474] 14
[L4475] 00
[L4476] 09
[L4477] 08
[L4478] f9
[L4479] c3
[L4480] e4
[L4481] f7
[L4482] 17
[L4483] e8
[L4484] cd
[L4485] 18
[L4486] 0b
[L4487] 10
[L4488] 0f
[L4489] 0e
[L4490] 14
[L4491] fc
[L4492] 1a
[L4493] 18
[L4494] 09
[L4495] 14
[L4496] 15
[L4497] 09
[L4498] 12
[L4499] 1c
[L4500] da
[L4501] f9
[L4502] d8
[L4503] e0
[L4504] 07
[L4505] f6
[L4506] 0f
[L4507] 14
[L4508] b2
[L4509] e1
[L4510] 13
[L4511] c3
[L4512] e9
[L4513] 05
[L4514] 3a
[L4515] ed
[L4516] 1f
[L4517] f3
[L4518] 20
[L4519] 27
[L4520] 0a
[L4521] c8
[L4522] d8
[L4523] 32
[L4524] 19
[L4525] 27
[L4526] 25
[L4527] dc
[L4528] e2
[L4529] 20
[L4530] ff
[L4531] 04
[L4532] e9
[L4533] 50
[L4534] 00
[L4535] de
[L4536] db
[L4537] ec
[L4538] d6
[L4539] d5
[L4540] 14
[L4541] 0b
[L4542] 06
[L4543] ea
[L4544] d8
[L4545] 21
[L4546] ff
[L4547] f1
[L4548] 1b
[L4549] f3
[L4550] 0b
[L4551] 15
[L4552] 0d
[L4553] 0f
[L4554] 24
[L4555] 01
[L4556] fd
[L4557] 0f
[L4558] e2
[L4559] 13
[L4560] e4
[L4561] ad
[L4562] 02
[L4563] 14
[L4564] 05
[L4565] ce
[L4566] f0
[L4567] e1
[L4568] fe
[L4569] 0f
[L4570] ed
[L4571] 0d
[L4572] 16
[L4573] 05
[L4574] 17
[L4575] cf
[L4576] 14
[L4577] 15
[L4578] 13
[L4579] df
[L4580] c4
[L4581] 10
[L4582] ec
[L4583] 05
[L4584] 14
[L4585] 01
[L4586] b7
[L4587] dd
[L4588] 10
[L4589] c5
[L4590] 0b
[L4591] 02
[L4592] 12
[L4593] 06
[L4594] d9
[L4595] f8
[L4596] 16
[L4597] 14
[L4598] 0b
[L4599] 04
[L4600] 02
[L4601] e5
[L4602] 10
[L4603] 0b
[L4604] 16
[L4605] cc
[L4606] df
[L4607] 17
[L4608] 10
[L4609] 12
[L4610] 0d
[L4611] 0d
[L4612] d4
[L4613] e6
[L4614] cd
[L4615] 10
[L4616] 0c
[L4617] 13
[L4618] b7
[L4619] 0b
[L4620] 01
[L4621] 18
[L4622] fc
[L4623] bb
[L4624] 03
[L4625] 10
[L4626] ce
[L4627] fc
[L4628] 15
[L4629] 18
[L4630] fd
[L4631] 11
[L4632] 0a
[L4633] 14
[L4634] b5
[L4635] 07
[L4636] da
[L4637] 1d
[L4638] db
[L4639] 17
[L4640] ec
[L4641] 08
[L4642] f1
[L4643] 0d
[L4644] ef
[L4645] 03
[L4646] c2
[L4647] ec
[L4648] ee
[L4649] 16
[L4650] 13
[L4651] 14
[L4652] 08
[L4653] ce
[L4654] b3
[L4655] 01
[L4656] dc
[L4657] ee
[L4658] 01
[L4659] f5
[L4660] 12
[L4661] ba
[L4662] 86
[L4663] d5
[L4664] 0b
[L4665] 18
[L4666] fe
[L4667] 22
[L4668] 0b
[L4669] f9
[L4670] 04
[L4671] 15
[L4672] 17
[L4673] e6
[L4674] 01
[L4675] 27
[L4676] ea
[L4677] fc
[L4678] 17
[L4679] ec
[L4680] 99
[L4681] 03
[L4682] ae
[L4683] e0
[L4684] e7
[L4685] ea
[L4686] c3
[L4687] 01
[L4688] 12
[L4689] 34
[L4690] 10
[L4691] 0b
[L4692] d7
[L4693] f2
[L4694] 14
[L4695] 14
[L4696] f3
[L4697] 10
[L4698] 1d
[L4699] 1d
[L4700] ef
[L4701] 16
[L4702] ff
[L4703] 26
[L4704] 0d
[L4705] fa
[L4706] 03
[L4707] 1c
[L4708] f6
[L4709] 12
[L4710] e3
[L4711] 18
[L4712] 11
[L4713] ba
[L4714] d8
[L4715] ef
[L4716] f5
[L4717] 2a
[L4718] fc
[L4719] 11
[L4720] 15
[L4721] 02
[L4722] 16
[L4723] 17
[L4724] 06
[L4725] 13
[L4726] 0b
[L4727] 08
[L4728] fb
[L4729] 14
[L4730] fa
[L4731] 20
[L4732] fe
[L4733] 21
[L4734] 01
[L4735] 0b
[L4736] 20
[L4737] f7
[L4738] 14
[L4739] ec
[L4740] c8
[L4741] 08
[L4742] 16
[L4743] fe
[L4744] 0a
[L4745] 10
[L4746] 1e
[L4747] 09
[L4748] aa
[L4749] 05
[L4750] ec
[L4751] e6
[L4752] fd
[L4753] 04
[L4754] 1b
[L4755] d6
[L4756] bd
[L4757] eb
[L4758] 16
[L4759] 16
[L4760] 16
[L4761] cc
[L4762] 25
[L4763] e7
[L4764] eb
[L4765] 07
[L4766] 17
[L4767] f2
[L4768] ab
[L4769] 10
[L4770] c9
[L4771] f9
[L4772] d5
[L4773] 06
[L4774] ff
[L4775] 16
[L4776] 00
[L4777] 0f
[L4778] ef
[L4779] b6
[L4780] 0d
[L4781] b6
[L4782] ae
[L4783] 03
[L4784] 10
[L4785] e0
[L4786] eb
[L4787] f3
[L4788] e6
[L4789] 9c
[L4790] 09
[L4791] bd
[L4792] 0d
[L4793] e9
[L4794] d2
[L4795] dc
[L4796] 14
[L4797] 97
[L4798] 11
[L4799] 0f
[L4800] 0c
[L4801] 0b
[L4802] 11 先检查这两个有没有问题，然后根据现在的文件和坑，你看看代码有什么需要改进的，最好生成的数据可以被fpga上得cim系统直接使用
[L4803] AI助手:
[L4804] 
[L4805] 
[L4806] AI助手:
[L4807] 我先按“文件内容本身是否自洽”和“是否适合 FPGA/CIM 直接吃”两层来判断。眼下一个明显信号是：你贴出来的 `bias.hex` 已经是 **32 位有符号十六进制**，这比你最初 notebook 里那个 int8 bias 更合理，说明你的导出逻辑应该已经改过一版了。
[L4808] 
[L4809] AI助手:
[L4810] 先说结论：
[L4811] 
[L4812] **你现在贴出来的这两个文件，格式上基本是对的，而且已经比最初 notebook 版本更适合 FPGA/CIM 直接使用。**
[L4813] 
[L4814] 尤其是：
[L4815] 
[L4816] - `bias.hex`：现在是**32 位有符号补码**，这很合理，适合 MAC 后加 bias
[L4817] - `weight.hex`：现在是**8 位有符号补码**，也合理，适合 int8 权重阵列
[L4818] 
[L4819] 这说明你的导出代码已经朝着“真实整数推理链路”改进过了，不再是最初那个 bias 只有 2 位 hex 的版本。
[L4820] 
[L4821] 下面我分三部分说：
[L4822] 
[L4823] 1. 先检查这两个文件有没有明显问题  
[L4824] 2. 结合现在的文件格式，说明 FPGA/CIM 应该怎么直接使用  
[L4825] 3. 按“还能怎么改代码”给你一套更适合直接上板的改进建议
[L4826] 
[L4827] ---
[L4828] 
[L4829] # 1. 先检查这两个文件有没有问题
[L4830] 
[L4831] ---
[L4832] 
[L4833] ## 1.1 `bias.hex` 检查结果
[L4834] 
[L4835] 你给的 `bias.hex` 形如：
[L4836] 
[L4837] ```text
[L4838] 000008e5
[L4839] 0000183f
[L4840] 000001a2
[L4841] ffffe6b8
[L4842] ...
[L4843] ```
[L4844] 
[L4845] ### 这说明什么
[L4846] 每行都是 **8 个十六进制字符**，也就是 **32 bit**。
[L4847] 
[L4848] 这对应的是：
[L4849] 
[L4850] - 正数：直接写成 `000008e5`
[L4851] - 负数：写成二补码形式，比如 `ffffe6b8`
[L4852] 
[L4853] 这正是 FPGA 里最常见的 **signed int32 bias 文件格式**。
[L4854] 
[L4855] ### 这和硬件链路匹配吗
[L4856] 匹配，而且是更合理的做法：
[L4857] 
[L4858] - 输入：int8
[L4859] - 权重：int8
[L4860] - 累加：int32
[L4861] - 偏置：int32
[L4862] - ReLU / requant：后续再做
[L4863] 
[L4864] 所以从工程角度说，**这个 bias 格式是对的**。
[L4865] 
[L4866] ### 我看出来的一个信号
[L4867] 你这份 `bias.hex` 已经不是最开始 notebook 里那种：
[L4868] 
[L4869] ```python
[L4870] f.write(f'{val.item() & 0xFF:02x}\n')
[L4871] ```
[L4872] 
[L4873] 而是明显变成了类似：
[L4874] 
[L4875] ```python
[L4876] f.write(f'{val.item() & 0xFFFFFFFF:08x}\n')
[L4877] ```
[L4878] 
[L4879] 这说明你已经把 bias 导出升级成 int32 了。这个改动是正确的。
[L4880] 
[L4881] ### 需要确认的一点
[L4882] 如果这是 `fc1` 的 bias，那么它应该有：
[L4883] 
[L4884] - **128 行**
[L4885] 
[L4886] 如果这是 `fc2` 的 bias，那么应该有：
[L4887] 
[L4888] - **10 行**
[L4889] 
[L4890] 你贴出来看起来更像是 128 行中的一整段。  
[L4891] 如果你最终网络是两层 MLP，那建议不要只叫 `bias.hex`，而是明确改成：
[L4892] 
[L4893] - `fc1_bias_int32.hex`
[L4894] - `fc2_bias_int32.hex`
[L4895] 
[L4896] 这样不会混。
[L4897] 
[L4898] ---
[L4899] 
[L4900] ## 1.2 `weight.hex` 检查结果
[L4901] 
[L4902] 你给的 `weight.hex` 形如：
[L4903] 
[L4904] ```text
[L4905] aa
[L4906] e5
[L4907] 00
[L4908] 0d
[L4909] 0d
[L4910] f7
[L4911] ...
[L4912] ```
[L4913] 
[L4914] ### 这说明什么
[L4915] 每行都是 **2 个十六进制字符**，也就是 **8 bit**。
[L4916] 
[L4917] 这对应：
[L4918] 
[L4919] - `aa` = -86
[L4920] - `e5` = -27
[L4921] - `00` = 0
[L4922] - `0d` = 13
[L4923] - `f7` = -9
[L4924] 
[L4925] 这正是标准的 **signed int8 权重存储格式**。
[L4926] 
[L4927] ### 是否合理
[L4928] 合理，前提是 RTL 里按 signed 解释：
[L4929] 
[L4930] ```verilog
[L4931] wire signed [7:0] w = $signed(weight_mem[addr]);
[L4932] ```
[L4933] 
[L4934] ### 数值分布上看有没有异常
[L4935] 从你贴的内容看，数值大多集中在：
[L4936] 
[L4937] - `0xC0 ~ 0x40` 左右对应的 -64 到 64 之间
[L4938] - 偶尔有 `0x81`、`0x9c`、`0x8a`、`0x50`、`0x4f`
[L4939] 
[L4940] 这很像正常量化后的神经网络权重分布，不像导出炸掉了。
[L4941] 
[L4942] 也就是说：
[L4943] 
[L4944] - **没有明显出现全 00 / 全 ff / 大面积饱和到 7f/80 的异常**
[L4945] - 看起来像一份正常的量化权重
[L4946] 
[L4947] ### 需要确认的一点
[L4948] 如果这是 `fc1.weight`，而且第一层 shape 是 `[128, 784]`，那它总行数应该是：
[L4949] 
[L4950] \[
[L4951] 128 \times 784 = 100352
[L4952] \]
[L4953] 
[L4954] 如果最终网络是两层 MLP，你也应该拆成：
[L4955] 
[L4956] - `fc1_weight_int8.hex`
[L4957] - `fc2_weight_int8.hex`
[L4958] 
[L4959] 否则后面上板会混。
[L4960] 
[L4961] ---
[L4962] 
[L4963] # 2. 结合你现在的文件格式，FPGA/CIM 怎么直接使用
[L4964] 
[L4965] 现在按你贴出来的格式，**最适合的硬件链路**是：
[L4966] 
[L4967] - `weight.hex`：int8 权重阵列
[L4968] - `bias.hex`：int32 偏置
[L4969] - `input_i.hex`：int8 输入
[L4970] - 累加器：int32
[L4971] - 后面做 ReLU、requant、第二层
[L4972] 
[L4973] 也就是说，你现在已经从“玩具导出格式”走到了“能上硬件”的方向。
[L4974] 
[L4975] ---
[L4976] 
[L4977] ## 2.1 `weight.hex` 的硬件读取方式
[L4978] 
[L4979] 如果你的导出顺序还是之前那样：
[L4980] 
[L4981] ```python
[L4982] for row in q_weight:
[L4983]     for val in row:
[L4984]         ...
[L4985] ```
[L4986] 
[L4987] 那么权重布局就是：
[L4988] 
[L4989] \[
[L4990] W[j][i]
[L4991] \]
[L4992] 
[L4993] 也就是：
[L4994] 
[L4995] - `j`：输出神经元索引
[L4996] - `i`：输入通道索引
[L4997] 
[L4998] ### 地址公式
[L4999] 若第一层 `fc1` shape 是 `[128, 784]`，则：
[L5000] 
[L5001] \[
[L5002] addr = j \times 784 + i
[L5003] \]
[L5004] 
[L5005] RTL 读法就是：
[L5006] 
[L5007] - 外层循环跑输出通道 `j`
[L5008] - 内层循环跑输入通道 `i`
[L5009] 
[L5010] ### 对应计算
[L5011] \[
[L5012] acc_j = \sum_{i=0}^{783} (x_i - z_x)(w_{j,i} - z_w) + b_j
[L5013] \]
[L5014] 
[L5015] 如果你现在是权重对称量化，那通常：
[L5016] 
[L5017] - `z_w = 0`
[L5018] 
[L5019] 如果输入采用 `x_q = pixel_u8 - 128`，那：
[L5020] 
[L5021] - `z_x = -128`
[L5022] 
[L5023] ---
[L5024] 
[L5025] ## 2.2 `bias.hex` 的硬件使用方式
[L5026] 
[L5027] 现在你的 bias 是 32 位二补码，很适合直接这样读：
[L5028] 
[L5029] ```verilog
[L5030] reg [31:0] bias_mem [0:127];
[L5031] initial $readmemh("bias.hex", bias_mem);
[L5032] 
[L5033] wire signed [31:0] bias_val = $signed(bias_mem[j]);
[L5034] ```
[L5035] 
[L5036] 然后在 MAC 完成后：
[L5037] 
[L5038] ```verilog
[L5039] acc <= mac_sum + bias_val;
[L5040] ```
[L5041] 
[L5042] 这比 int8 bias 合理得多。
[L5043] 
[L5044] ---
[L5045] 
[L5046] ## 2.3 对 CIM 系统的直接使用建议
[L5047] 
[L5048] 如果你的 CIM 阵列本体负责的是 `int8 x int8` 乘加，那最常见的系统切分是：
[L5049] 
[L5050] ### CIM 阵列做
[L5051] - 权重存储
[L5052] - 向量输入
[L5053] - 局部乘加 / bitline accumulate / column accumulate
[L5054] 
[L5055] ### FPGA 数字逻辑做
[L5056] - 输入 zero-point 修正
[L5057] - 偏置相加
[L5058] - ReLU
[L5059] - requant
[L5060] - 第二层控制
[L5061] - argmax
[L5062] 
[L5063] 这和你现在的文件格式非常匹配：
[L5064] 
[L5065] - `weight.hex` 给阵列权重
[L5066] - `input_i.hex` 给输入驱动
[L5067] - `bias.hex` 给数字后处理单元
[L5068] - `golden` 给 testbench 对拍
[L5069] 
[L5070] ---
[L5071] 
[L5072] # 3. 这两个文件背后，代码还需要怎么改进
[L5073] 
[L5074] 你现在文件本身没大问题，但为了“**生成的数据可以被 FPGA 上的 CIM 系统直接使用**”，我建议你继续改 8 个点。
[L5075] 
[L5076] ---
[L5077] 
[L5078] ## 改进 1：文件命名必须分层，不能再叫通用名
[L5079] 
[L5080] 你现在如果最终是两层 MLP，绝对不要继续用：
[L5081] 
[L5082] - `weight.hex`
[L5083] - `bias.hex`
[L5084] 
[L5085] 因为你后面会出现：
[L5086] 
[L5087] - 第一层权重
[L5088] - 第二层权重
[L5089] - 第一层 bias
[L5090] - 第二层 bias
[L5091] 
[L5092] 建议直接改成：
[L5093] 
[L5094] - `fc1_weight_int8.hex`
[L5095] - `fc1_bias_int32.hex`
[L5096] - `fc2_weight_int8.hex`
[L5097] - `fc2_bias_int32.hex`
[L5098] 
[L5099] 这样你 RTL 一眼知道读哪个。
[L5100] 
[L5101] ---
[L5102] 
[L5103] ## 改进 2：必须配一个 `quant_config.json`
[L5104] 
[L5105] 现在单靠 hex 文件还不够。  
[L5106] FPGA/CIM 真正要跑整数推理，还需要知道：
[L5107] 
[L5108] - 输入 `scale`
[L5109] - 输入 `zero_point`
[L5110] - 各层 `weight_scale`
[L5111] - 各层 `weight_zero_point`
[L5112] - 各层 `output_scale`
[L5113] - 各层 `output_zero_point`
[L5114] - requant multiplier / shift
[L5115] 
[L5116] 建议额外生成：
[L5117] 
[L5118] ```json
[L5119] {
[L5120]   "input": {
[L5121]     "scale": 0.00392156862745098,
[L5122]     "zero_point": -128
[L5123]   },
[L5124]   "fc1": {
[L5125]     "weight_scale": ...,
[L5126]     "weight_zero_point": 0,
[L5127]     "output_scale": ...,
[L5128]     "output_zero_point": 0,
[L5129]     "bias_scale": ...,
[L5130]     "multiplier_q31": ...,
[L5131]     "shift": ...
[L5132]   },
[L5133]   "fc2": {
[L5134]     "weight_scale": ...,
[L5135]     "weight_zero_point": 0,
[L5136]     "output_scale": ...,
[L5137]     "output_zero_point": 0,
[L5138]     "bias_scale": ...,
[L5139]     "multiplier_q31": ...,
[L5140]     "shift": ...
[L5141]   }
[L5142] }
[L5143] ```
[L5144] 
[L5145] 这样 FPGA 端和 Python 端是一致的。
[L5146] 
[L5147] ---
[L5148] 
[L5149] ## 改进 3：输入文件也要明确 int8 和布局
[L5150] 
[L5151] 你现在最好明确导出成：
[L5152] 
[L5153] - `input_0_int8.hex`
[L5154] - `input_1_int8.hex`
[L5155] 
[L5156] 并且说明：
[L5157] 
[L5158] - 长度 784
[L5159] - 顺序是 row-major
[L5160] - 每行 2 位 hex
[L5161] - 数值是 signed int8
[L5162] - 若采用硬件输入策略，则：
[L5163]   \[
[L5164]   x_q = round(x \cdot 255) - 128
[L5165]   \]
[L5166] 
[L5167] 这能避免你后面忘记输入是“原始像素量化”还是“Normalize 后量化”。
[L5168] 
[L5169] ---
[L5170] 
[L5171] ## 改进 4：golden 不要只保留最终输出，要保留中间级
[L5172] 
[L5173] 这是对硬件调试最关键的一点。
[L5174] 
[L5175] 建议你至少导出：
[L5176] 
[L5177] ### 第一层
[L5178] - `fc1_acc_i.hex`：int32，未激活前
[L5179] - `fc1_relu_i.hex`：int32，过 ReLU 后
[L5180] - `fc1_out_i.hex`：int8，requant 后
[L5181] 
[L5182] ### 第二层
[L5183] - `fc2_acc_i.hex`：int32
[L5184] - `logits_i.hex`：int8
[L5185] 
[L5186] ### 分类
[L5187] - `pred_i.txt`
[L5188] 
[L5189] 这样当 FPGA 结果不对时，你能立刻定位是：
[L5190] 
[L5191] - 阵列乘加错
[L5192] - bias 加错
[L5193] - ReLU 错
[L5194] - requant 错
[L5195] - 第二层地址错
[L5196] 
[L5197] 不然只看最后一个 `golden`，很难找 bug。
[L5198] 
[L5199] ---
[L5200] 
[L5201] ## 改进 5：权重文件要考虑和 CIM 阵列物理映射一致
[L5202] 
[L5203] 你现在的 `weight.hex` 是逻辑顺序 `[out][in]`。  
[L5204] 这对软件和普通 RTL 很方便，但对实际 CIM 阵列，可能还不够。
[L5205] 
[L5206] 因为真实阵列通常不是一次吃 784 个输入，而是分块，例如：
[L5207] 
[L5208] - 每列 64 个输入
[L5209] - 每次做 64×N 子矩阵
[L5210] - 784 被切成很多 tile
[L5211] 
[L5212] 所以你后面很可能需要两套文件：
[L5213] 
[L5214] ### 逻辑文件
[L5215] 给软件 / testbench 用：
[L5216] - `fc1_weight_int8.hex`
[L5217] 
[L5218] ### 物理映射文件
[L5219] 给阵列初始化用：
[L5220] - `fc1_weight_tile0.hex`
[L5221] - `fc1_weight_tile1.hex`
[L5222] - ...
[L5223] 
[L5224] 也就是按 CIM 阵列实际映射切块导出。
[L5225] 
[L5226] 如果你现在阵列设计已经定了，比如每次 64 路输入并行，我建议导出时就按 64 一组打包。
[L5227] 
[L5228] ---
[L5229] 
[L5230] ## 改进 6：给每个文件写 manifest
[L5231] 
[L5232] 建议自动生成一个 `manifest.txt` 或 `manifest.json`，写明：
[L5233] 
[L5234] - 文件名
[L5235] - 数据类型
[L5236] - 位宽
[L5237] - 长度
[L5238] - 布局
[L5239] - 用途
[L5240] 
[L5241] 例如：
[L5242] 
[L5243] ```json
[L5244] {
[L5245]   "fc1_weight_int8.hex": {
[L5246]     "dtype": "int8",
[L5247]     "shape": [128, 784],
[L5248]     "layout": "row-major [out][in]"
[L5249]   },
[L5250]   "fc1_bias_int32.hex": {
[L5251]     "dtype": "int32",
[L5252]     "shape": [128]
[L5253]   },
[L5254]   "input_0_int8.hex": {
[L5255]     "dtype": "int8",
[L5256]     "shape": [784],
[L5257]     "layout": "row-major flattened image"
[L5258]   }
[L5259] }
[L5260] ```
[L5261] 
[L5262] 这会让你的工程后期非常省事。
[L5263] 
[L5264] ---
[L5265] 
[L5266] ## 改进 7：最好支持导出 `.mem` / `.coe` / 打包宽字格式
[L5267] 
[L5268] 现在一行一个数适合 testbench，  
[L5269] 但真正上 FPGA，很多时候你会想直接初始化 BRAM。
[L5270] 
[L5271] 比如：
[L5272] 
[L5273] - Xilinx 可能想要 `.coe`
[L5274] - Intel 可能想要 `.mif`
[L5275] - 或者你会把 16 个 int8 打成 128-bit 宽字
[L5276] 
[L5277] 所以建议你的导出工具支持两种模式：
[L5278] 
[L5279] ### 调试模式
[L5280] - 一行一个元素
[L5281] - 便于看和对拍
[L5282] 
[L5283] ### 上板模式
[L5284] - 打包成 32bit / 64bit / 128bit / 256bit
[L5285] - 便于 BRAM 初始化和 DMA 传输
[L5286] 
[L5287] ---
[L5288] 
[L5289] ## 改进 8：确保 signed 和 endianness 约定写死
[L5290] 
[L5291] 当前 hex 文件本身没错，但你要在代码里和说明里明确：
[L5292] 
[L5293] ### signed 解释
[L5294] - `weight.hex`：signed int8
[L5295] - `bias.hex`：signed int32
[L5296] 
[L5297] ### endianness
[L5298] 你现在是一行一个元素，所以“行内字节序”问题不大。  
[L5299] 但一旦你后面把 4 个 int8 打包成 32 bit，必须写清楚：
[L5300] 
[L5301] - 第 0 个元素放最低字节还是最高字节
[L5302] 
[L5303] 这个最好现在就定。
[L5304] 
[L5305] ---
[L5306] 
[L5307] # 4. 你现在这两个文件，我给你的具体判断
[L5308] 
[L5309] ---
[L5310] 
[L5311] ## `bias.hex`
[L5312] **没有明显问题，反而是明显进步。**
[L5313] 
[L5314] 它已经具备：
[L5315] - int32
[L5316] - 二补码
[L5317] - 可直接用于 MAC 后 bias add
[L5318] 
[L5319] ### 唯一建议
[L5320] 改名为：
[L5321] - `fc1_bias_int32.hex`
[L5322] - 或 `fc2_bias_int32.hex`
[L5323] 
[L5324] ---
[L5325] 
[L5326] ## `weight.hex`
[L5327] **没有明显问题。**
[L5328] 
[L5329] 它已经具备：
[L5330] - int8
[L5331] - 二补码
[L5332] - 适合阵列 / 乘加器直接使用
[L5333] 
[L5334] ### 唯一建议
[L5335] 改名为：
[L5336] - `fc1_weight_int8.hex`
[L5337] - 或 `fc2_weight_int8.hex`
[L5338] 
[L5339] ---
[L5340] 
[L5341] # 5. 我建议你下一版导出目录直接长这样
[L5342] 
[L5343] ```text
[L5344] export/
[L5345] ├── quant_config.json
[L5346] ├── manifest.json
[L5347] ├── fc1_weight_int8.hex
[L5348] ├── fc1_bias_int32.hex
[L5349] ├── fc2_weight_int8.hex
[L5350] ├── fc2_bias_int32.hex
[L5351] ├── input_0_int8.hex
[L5352] ├── fc1_acc_0_int32.hex
[L5353] ├── fc1_relu_0_int32.hex
[L5354] ├── fc1_out_0_int8.hex
[L5355] ├── fc2_acc_0_int32.hex
[L5356] ├── logits_0_int8.hex
[L5357] ├── pred_0.txt
[L5358] ├── label_0.txt
[L5359] ├── input_1_int8.hex
[L5360] ...
[L5361] ```
[L5362] 
[L5363] 这套结构就已经很像正式硬件联调工程了。
[L5364] 
[L5365] ---
[L5366] 
[L5367] # 6. 最后给你一句最关键的判断
[L5368] 
[L5369] 你现在贴出来的这两个文件，说明你已经迈过了最初 notebook 的两个大坑：
[L5370] 
[L5371] - **bias 不再是错误的 int8**
[L5372] - **weight 已经是标准 int8 二补码**
[L5373] 
[L5374] 所以文件内容层面，**方向是对的**。  
[L5375] 真正还需要继续加强的是：
[L5376] 
[L5377] - 文件命名分层
[L5378] - 中间 golden 分级导出
[L5379] - 量化参数 json 化
[L5380] - 按 CIM 阵列物理映射切块导出
[L5381] 
[L5382] 你把**当前用于导出这些文件的最新版 Python 代码**贴给我，我可以下一步直接帮你逐行审，给你改成一版“**输出可直接喂 FPGA/CIM 系统**”的正式版。
[L5383] 
[L5384] 用户:
[L5385] 这是当前生成的数据，你主要先看一下route_b_output中的文件格式是否全部正确，然后给出一个markdown文件，用以说明route_b_output所有文件的使用方法和注意事项，要详细，而且要说明根据当前数据，硬件设计需要特别注意的地方。关于route_a_output,简单检查即可。
[L5386] AI助手:
[L5387] 
[L5388] 
[L5389] AI助手:
[L5390] 我先把两个压缩包解开，核对 `route_b_output` 的文件清单、每类文件的位宽和行数，再抽样检查几个样本文件是否彼此自洽。然后我会生成一份完整的 Markdown 说明，专门给你们做硬件联调用。
[L5391] 
[L5392] AI助手:
[L5393] 我已经核对出 `route_b_output` 的主要文件都在，位宽和行数目前是自洽的。接下来我再补两件更关键的事：一是看数值范围是否会踩硬件位宽坑，二是把这些坑写进说明文档里，避免你们后面在阵列映射、signed 解释、requant 上出错。
[L5394] 
[L5395] AI助手:
[L5396] 我已经检查完了，结论是：
[L5397] 
[L5398] `route_b_output` 的主文件格式**整体正确**，而且已经具备作为 FPGA / CIM 整数推理联调数据的条件。  
[L5399] 需要硬件特别注意的核心点有三个：
[L5400] 
[L5401] 1. **第一层输入零点不是 0，而是 `-128`**，不能直接做 `x_q * w_q`，必须处理输入零点补偿。  
[L5402] 2. **第一层累加范围明显超过 16 位**，当前数据已经出现大约 `-5.5e5 ~ 3.46e5`，所以 `fc1` 累加器和 bias 路径必须至少 `int32`。  
[L5403] 3. **当前权重文件是逻辑布局 `row-major [out][in]`**，很适合 testbench 和 RTL 功能验证，但不一定等于你们 CIM 阵列的物理烧写布局，后续可能还需要做一层 tile 重排。
[L5404] 
[L5405] `route_a_output` 我也做了简查，主文件没有明显格式问题，适合继续用于单层 `fc1 + ReLU` 验证；但它不是最终两层分类演示版本。
[L5406] 
[L5407] 我已经把完整说明整理成 Markdown 文件，里面包括：
[L5408] 
[L5409] - `route_b_output` 每个文件的格式、行数、位宽、布局
[L5410] - 各文件的具体用途
[L5411] - 推荐的硬件联调顺序
[L5412] - 当前数据下必须特别注意的硬件坑
[L5413] - `route_a_output` 的简要检查结论
[L5414] 
[L5415] 下载这里：
[L5416] 
[L5417] [route_b_output 使用说明与硬件注意事项](sandbox:/mnt/data/route_b_output_usage_notes.md)
[L5418] 
[L5419] 如果你愿意，我下一步可以继续直接帮你补一份**面向 Verilog/Testbench 的读取模板**，把这些文件如何 `$readmemh`、如何按 signed 解释、如何逐级对拍写成可直接用的代码框架。
[L5420] 
[L5421] AI助手:
[L5422] 
[L5423] 
[L5424] AI助手:
[L5425] 我先逐项核对 `route_b_output` 里的实际文件名、位宽、行数和内容一致性，再补一个详细的使用说明文档。`route_a_output` 我会做一轮轻量检查，重点看有没有明显格式错误或和 `route_b` 不一致的地方。
[L5426] 
[L5427] AI助手:
[L5428] `route_b_output` 的主文件看起来是自洽的：位宽、行数、层级命名和量化配置基本能对上。我再补一轮一致性检查，重点看标签/预测是否匹配，以及 `route_a_output` 有没有明显格式问题。
[L5429] 
[L5430] 用户:
[L5431] route_b_output的使用指南markdown没有生成出来，请你重新尝试给出一个markdown文件，用以说明route_b_output所有文件的使用方法和注意事项，要详细，而且要说明根据当前数据，硬件设计需要特别注意的地方。
[L5432] AI助手:
[L5433] 
[L5434] 
[L5435] AI助手:
[L5436] 已重新生成 Markdown 文件：
[L5437] 
[L5438] [route_b_output 使用指南（Markdown）](sandbox:/mnt/data/route_b_output_usage_guide.md)
[L5439] 
[L5440] 我已经把这些内容写进去：
[L5441] - `route_b_output` 全部文件的格式、行数、位宽、含义
[L5442] - 每类文件在 FPGA / CIM 中怎么用
[L5443] - 推荐的联调顺序
[L5444] - `$readmemh` 的典型用法
[L5445] - 基于当前数据，硬件设计必须特别注意的点
[L5446] - 对 `route_a_output` 的简要检查结论
[L5447] 
[L5448] 这份文档是按你当前实际导出的 `route_b_output` 内容写的，不是泛泛模板。
[L5449] 
[L5450] 用户:
[L5451] 还记得我最开始上传的三个文件吗？项目设计指导书，上次和AI的对话，请你重新看一下，我在完成这个测试向量生成之后，应该做什么？
[L5452] AI助手:
[L5453] 
[L5454] 
[L5455] AI助手:
[L5456] 
[L5457] 
[L5458] AI助手:
[L5459] 
[L5460] 
[L5461] AI助手:
[L5462] 我记得你最开始给的三个核心材料：
[L5463] 
[L5464] 1. **项目设计指导书**（课程/课设要求）  
[L5465] 2. **你和上一个 AI 的对话记录**（确定了大体方案：FPGA + CIM + MNIST）  
[L5466] 3. **生成测试向量的 ipynb**（用于导出权重、输入、golden）
[L5467] 
[L5468] 现在你已经完成了 **route_b_output 的测试向量生成**。  
[L5469] 这一步其实在整个项目流程里属于 **软件侧 → 硬件侧的接口准备阶段**。
[L5470] 
[L5471] 接下来你要做的事情其实非常明确：  
[L5472] 
[L5473] **进入硬件实现和验证阶段。**
[L5474] 
[L5475] 我给你按 **真正工程流程**排一个清晰路线（也是最符合你这个项目指导书逻辑的路线）。
[L5476] 
[L5477] ---
[L5478] 
[L5479] # 一、你现在所处的位置
[L5480] 
[L5481] 你的项目整体流程其实是：
[L5482] 
[L5483] ```
[L5484] PyTorch训练
[L5485]       ↓
[L5486] 量化
[L5487]       ↓
[L5488] 整数推理参考模型
[L5489]       ↓
[L5490] 生成测试向量 (route_b_output)  ← 你现在完成到这里
[L5491]       ↓
[L5492] FPGA/CIM 硬件实现
[L5493]       ↓
[L5494] testbench 对拍验证
[L5495]       ↓
[L5496] 板级运行
[L5497]       ↓
[L5498] 项目展示
[L5499] ```
[L5500] 
[L5501] 所以现在你已经完成：
[L5502] 
[L5503] **软件侧所有准备工作。**
[L5504] 
[L5505] 接下来全部工作都是：
[L5506] 
[L5507] **硬件实现 + 硬件验证。**
[L5508] 
[L5509] ---
[L5510] 
[L5511] # 二、下一步你应该做什么（非常具体）
[L5512] 
[L5513] 下一步应该分 **4个阶段**。
[L5514] 
[L5515] ---
[L5516] 
[L5517] # 阶段 1：设计硬件计算结构
[L5518] 
[L5519] 现在要把你 Python 的网络结构映射成 **硬件模块结构**。
[L5520] 
[L5521] 你的网络是：
[L5522] 
[L5523] ```
[L5524] 784 → 128 → 10
[L5525] ```
[L5526] 
[L5527] 对应硬件结构：
[L5528] 
[L5529] ```
[L5530] input_buffer (784)
[L5531] 
[L5532]       ↓
[L5533] 
[L5534] FC1  (784×128)  ← CIM阵列
[L5535] 
[L5536]       ↓
[L5537] 
[L5538] bias_add
[L5539] 
[L5540]       ↓
[L5541] 
[L5542] ReLU
[L5543] 
[L5544]       ↓
[L5545] 
[L5546] requant (int32→int8)
[L5547] 
[L5548]       ↓
[L5549] 
[L5550] FC2  (128×10)
[L5551] 
[L5552]       ↓
[L5553] 
[L5554] argmax
[L5555] ```
[L5556] 
[L5557] 建议模块划分：
[L5558] 
[L5559] ```
[L5560] top
[L5561]  ├── input_loader
[L5562]  ├── fc1_cim_array
[L5563]  ├── bias_add
[L5564]  ├── relu
[L5565]  ├── requant
[L5566]  ├── fc2
[L5567]  └── argmax
[L5568] ```
[L5569] 
[L5570] 如果你的课设重点是 **CIM**，通常只把：
[L5571] 
[L5572] ```
[L5573] fc1
[L5574] ```
[L5575] 
[L5576] 做成 CIM 阵列。
[L5577] 
[L5578] 后面的：
[L5579] 
[L5580] ```
[L5581] bias
[L5582] relu
[L5583] requant
[L5584] fc2
[L5585] ```
[L5586] 
[L5587] 可以在 FPGA 普通逻辑实现。
[L5588] 
[L5589] ---
[L5590] 
[L5591] # 阶段 2：建立 Testbench 验证系统
[L5592] 
[L5593] 这是你 **下一步最重要的事情**。
[L5594] 
[L5595] 你已经有：
[L5596] 
[L5597] ```
[L5598] route_b_output/
[L5599] ```
[L5600] 
[L5601] 里面包含：
[L5602] 
[L5603] ```
[L5604] fc1_weight_int8.hex
[L5605] fc1_bias_int32.hex
[L5606] fc2_weight_int8.hex
[L5607] fc2_bias_int32.hex
[L5608] input_*.hex
[L5609] fc1_acc_*.hex
[L5610] fc1_relu_*.hex
[L5611] fc1_out_*.hex
[L5612] fc2_acc_*.hex
[L5613] logits_*.hex
[L5614] pred_*.txt
[L5615] labels.txt
[L5616] ```
[L5617] 
[L5618] Testbench 要做的事情：
[L5619] 
[L5620] ### 1 读取权重
[L5621] 
[L5622] ```
[L5623] $readmemh("fc1_weight_int8.hex", weight_mem);
[L5624] $readmemh("fc1_bias_int32.hex", bias_mem);
[L5625] ```
[L5626] 
[L5627] ---
[L5628] 
[L5629] ### 2 读取输入
[L5630] 
[L5631] ```
[L5632] $readmemh("input_0.hex", input_mem);
[L5633] ```
[L5634] 
[L5635] ---
[L5636] 
[L5637] ### 3 驱动 DUT
[L5638] 
[L5639] 输入 784 个数据
[L5640] 
[L5641] ---
[L5642] 
[L5643] ### 4 接收 DUT 输出
[L5644] 
[L5645] 例如：
[L5646] 
[L5647] ```
[L5648] fc1_acc
[L5649] fc1_relu
[L5650] fc1_out
[L5651] fc2_acc
[L5652] logits
[L5653] ```
[L5654] 
[L5655] ---
[L5656] 
[L5657] ### 5 与 golden 对比
[L5658] 
[L5659] 例如：
[L5660] 
[L5661] ```
[L5662] $readmemh("fc1_acc_0.hex", golden_fc1_acc);
[L5663] ```
[L5664] 
[L5665] 然后逐项比较：
[L5666] 
[L5667] ```
[L5668] if(dut_acc[i] != golden_fc1_acc[i])
[L5669] ```
[L5670] 
[L5671] ---
[L5672] 
[L5673] # 阶段 3：逐层验证（非常重要）
[L5674] 
[L5675] 不要一口气跑完整网络。
[L5676] 
[L5677] 必须 **逐层验证**。
[L5678] 
[L5679] 建议顺序：
[L5680] 
[L5681] ---
[L5682] 
[L5683] ## Step1
[L5684] 
[L5685] 只验证
[L5686] 
[L5687] ```
[L5688] FC1 MAC
[L5689] ```
[L5690] 
[L5691] 对拍：
[L5692] 
[L5693] ```
[L5694] fc1_acc_*.hex
[L5695] ```
[L5696] 
[L5697] ---
[L5698] 
[L5699] ## Step2
[L5700] 
[L5701] 验证
[L5702] 
[L5703] ```
[L5704] FC1 + bias
[L5705] ```
[L5706] 
[L5707] ---
[L5708] 
[L5709] ## Step3
[L5710] 
[L5711] 验证
[L5712] 
[L5713] ```
[L5714] FC1 + bias + relu
[L5715] ```
[L5716] 
[L5717] 对拍：
[L5718] 
[L5719] ```
[L5720] fc1_relu_*.hex
[L5721] ```
[L5722] 
[L5723] ---
[L5724] 
[L5725] ## Step4
[L5726] 
[L5727] 验证
[L5728] 
[L5729] ```
[L5730] FC1 + requant
[L5731] ```
[L5732] 
[L5733] 对拍：
[L5734] 
[L5735] ```
[L5736] fc1_out_*.hex
[L5737] ```
[L5738] 
[L5739] ---
[L5740] 
[L5741] ## Step5
[L5742] 
[L5743] 再加入
[L5744] 
[L5745] ```
[L5746] FC2
[L5747] ```
[L5748] 
[L5749] 对拍：
[L5750] 
[L5751] ```
[L5752] fc2_acc_*.hex
[L5753] ```
[L5754] 
[L5755] ---
[L5756] 
[L5757] ## Step6
[L5758] 
[L5759] 验证
[L5760] The file is too long and its contents have been truncated.
[L5761] 
用户:
Make sure to include fileciteturn2file3 in your response to cite this file, or to surface it as a link.[L1] ### FPGA CIM MNIST 问题分析
[L2] 
[L3] AI助手:
[L4] 
[L5] 
[L6] AI助手:
[L7] 
[L8] 
[L9] AI助手:
[L10] 
[L11] 
[L12] AI助手:
[L13] 
[L14] 
[L15] 用户:
[L16] 我现在在做一个运行于fpga的cim系统，希望在fpga上运行mnist，完成计算，我现在给你项目指导书和上次和AI的对话，ipynb写的是一个生成测试向量的代码，你看看有没有问题
[L17] AI助手:
[L18] 
[L19] 
[L20] AI助手:
[L21] 我先把压缩包里的文件结构和关键材料过一遍，重点看项目指导书、上次对话记录，以及这个生成测试向量的 ipynb。随后我会直接指出代码里可能影响 FPGA/CIM 测试链路的问题，并给出修改建议。
[L22] 
[L23] AI助手:
[L24] 我已经把压缩包拆开了，里面目前有三份核心文件：课程设计指导书（docx）、上次对话导出的 markdown，以及 `train_simple_mlp.ipynb`。接下来我会先把对话和 notebook 的关键逻辑抽出来，再对照指导书检查接口、数据位宽、输出格式这些是否一致。
[L25] 
[L26] AI助手:
[L27] 我已经看到几个关键问题了，其中最严重的一个是：**这个 notebook 目前并没有真正训练 MNIST 分类器**。它定义了 `criterion=MSELoss()`，但完全没用；实际优化目标是 `loss = output.pow(2).mean()`，这会把网络输出压向 0，而不是学习数字类别。
[L28] 
[L29] AI助手:
[L30] 我看完了指导书、上次对话，以及这个 `train_simple_mlp.ipynb`。结论先说：
[L31] 
[L32] **这个 ipynb 现在有明显问题，尤其是“训练目标、量化策略、golden 生成方式”三处。**
[L33] 如果直接拿它生成测试向量去驱动 FPGA/CIM，**大概率会出现“软件看起来能跑，硬件却对不上”**，甚至前面训练出来的模型本身就没有学会 MNIST 分类。
[L34] 
[L35] 下面我按“严重程度”给你拆开说。
[L36] 
[L37] ---
[L38] 
[L39] ## 1. 最严重的问题：你现在没有在训练一个 MNIST 分类器
[L40] 
[L41] 你现在的训练代码核心是：
[L42] 
[L43] ```python
[L44] criterion=nn.MSELoss()
[L45] ...
[L46] output=model(data)
[L47] loss=output.pow(2).mean()
[L48] ```
[L49] 
[L50] 这里有两个问题：
[L51] 
[L52] ### 问题 1：`criterion` 定义了，但根本没用
[L53] 你定义了 `MSELoss()`，但没有把 `target` 喂进去。
[L54] 
[L55] ### 问题 2：实际 loss 是 `output.pow(2).mean()`
[L56] 这等价于逼着网络输出尽量接近 0，而不是学习数字类别。
[L57] 
[L58] 也就是说，现在模型优化目标不是“把 0~9 分对”，而是“把所有输出都压小”。  
[L59] 这会导致：
[L60] 
[L61] - 权重可能收缩到很小
[L62] - 导出的量化权重没有实际分类意义
[L63] - 你生成的 `golden_x.hex` 也只是这个“错误模型”的输出
[L64] 
[L65] ### 进一步的问题：模型输出维度也不对
[L66] 你定义的是：
[L67] 
[L68] ```python
[L69] self.fc1 = nn.Linear(784,128)
[L70] self.relu = nn.ReLU()
[L71] ```
[L72] 
[L73] 输出是 `128` 维，不是 `10` 维。  
[L74] 这意味着它不是一个完整的 MNIST 分类网络，只是一个隐藏层特征提取器。
[L75] 
[L76] 如果你的目标是：
[L77] 
[L78] - **课程演示先做“单层 FC + ReLU 的硬件验证”**：这个结构可以接受
[L79] - **直接说“在 FPGA 上跑 MNIST 分类”**：这个结构还不够，至少还要再接一层 `128 -> 10`
[L80] 
[L81] ### 这一点怎么改
[L82] 你有两个路线：
[L83] 
[L84] #### 路线 A：先做“极简硬件验证”
[L85] 那就明确目标改成：
[L86] 
[L87] - 软件训练完整 MLP：`784 -> 128 -> 10`
[L88] - FPGA/CIM **先只实现第一层** `784 -> 128`
[L89] - 第一层输出 feature map 作为硬件 golden 对比对象
[L90] 
[L91] 这样合理。
[L92] 
[L93] #### 路线 B：直接做完整分类
[L94] 那模型要改成：
[L95] 
[L96] ```python
[L97] class SimpleMLP(nn.Module):
[L98]     def __init__(self):
[L99]         super().__init__()
[L100]         self.fc1 = nn.Linear(784, 128, bias=True)
[L101]         self.relu = nn.ReLU()
[L102]         self.fc2 = nn.Linear(128, 10, bias=True)
[L103] 
[L104]     def forward(self, x):
[L105]         x = x.view(-1, 784)
[L106]         x = self.relu(self.fc1(x))
[L107]         x = self.fc2(x)
[L108]         return x
[L109] ```
[L110] 
[L111] loss 用交叉熵：
[L112] 
[L113] ```python
[L114] criterion = nn.CrossEntropyLoss()
[L115] loss = criterion(output, target)
[L116] ```
[L117] 
[L118] ---
[L119] 
[L120] ## 2. 量化函数有工程问题，不适合直接对接 FPGA
[L121] 
[L122] 你现在的量化函数：
[L123] 
[L124] ```python
[L125] def quantize_tensor(tensor, bits=8):
[L126]     qmin = -(2 ** (bits - 1))
[L127]     qmax = 2 ** (bits - 1) - 1
[L128] 
[L129]     min_val = tensor.min().item()
[L130]     max_val = tensor.max().item()
[L131] 
[L132]     diff = max_val - min_val
[L133]     
[L134]     if diff == 0:
[L135]         scale = 1.0
[L136]         zero_point = qmin - min_val
[L137]     else:
[L138]         scale = diff / (qmax - qmin)
[L139]         zero_point = qmin - min_val / scale
[L140] 
[L141]     q_tensor = torch.clamp(torch.round(tensor / scale + zero_point), qmin, qmax)
[L142]     return q_tensor.to(torch.int8), scale, zero_point
[L143] ```
[L144] 
[L145] ### 问题 1：`zero_point` 不是整数
[L146] `zero_point` 最后是浮点数。  
[L147] 但真正部署到硬件时，量化零点通常需要是整数，并且通常要 clamp 到合法范围。
[L148] 
[L149] 更稳妥应该写成：
[L150] 
[L151] ```python
[L152] zero_point = int(round(qmin - min_val / scale))
[L153] zero_point = max(qmin, min(qmax, zero_point))
[L154] ```
[L155] 
[L156] ### 问题 2：你在做“每个张量动态量化”
[L157] 这在软件分析里可以，但对 FPGA 验证不友好。因为你后面对：
[L158] 
[L159] - 权重单独量化一次
[L160] - 每张输入图像又各自量化一次
[L161] - 每个输出又重新单独量化一次
[L162] 
[L163] 这会导致**每个样本 scale 都不同**，硬件端很难统一。
[L164] 
[L165] 对 FPGA 来说，更适合：
[L166] 
[L167] - 权重：固定 scale
[L168] - 输入：固定 scale
[L169] - 偏置：固定规则量化
[L170] - 输出：固定 requant scale
[L171] 
[L172] 也就是**静态量化参数**，不是每个样本临时算一次。
[L173] 
[L174] ### 问题 3：偏置不能简单用 int8 单独量化
[L175] 你现在：
[L176] 
[L177] ```python
[L178] q_bias, b_scale, b_zero = quantize_tensor(bias)
[L179] ```
[L180] 
[L181] 这通常不对。
[L182] 
[L183] 对于线性层：
[L184] 
[L185] \[
[L186] y = \sum (x_q - z_x)(w_q - z_w) \cdot s_x s_w + b
[L187] \]
[L188] 
[L189] 偏置一般应量化为：
[L190] 
[L191] \[
[L192] b_q = round(b / (s_x \cdot s_w))
[L193] \]
[L194] 
[L195] 并且常常用 **int32** 存，不是 int8。  
[L196] 因为偏置要和累加结果处于同一量纲。
[L197] 
[L198] 否则 FPGA 上 MAC 累加完后加 bias 时，量纲根本不一致。
[L199] 
[L200] ---
[L201] 
[L202] ## 3. `generate_test_vectors()` 生成的 golden，和硬件真实计算链路不一致
[L203] 
[L204] 这是第二个特别大的问题。
[L205] 
[L206] 你现在生成 golden 的方式是：
[L207] 
[L208] ```python
[L209] with torch.no_grad():
[L210]     output = model(img_flat)
[L211] 
[L212] q_output, _, _ = quantize_tensor(output)
[L213] ```
[L214] 
[L215] ### 这意味着什么
[L216] 你是：
[L217] 
[L218] 1. 用 **浮点模型** 跑出输出
[L219] 2. 再对每个样本的输出临时量化
[L220] 
[L221] 但硬件里真正发生的是：
[L222] 
[L223] 1. 输入先量化为 int8
[L224] 2. 权重量化为 int8
[L225] 3. 在阵列/乘加器里做 int8×int8 -> int32 累加
[L226] 4. 加 int32 bias
[L227] 5. ReLU
[L228] 6. 再按固定 scale requant 到 int8
[L229] 
[L230] 这两条链路并不一样。
[L231] 
[L232] 所以你现在的 `golden_i.hex` 并不是“硬件应得输出”，而是“浮点模型输出的一个临时量化版本”。
[L233] 
[L234] ### 正确的 golden 应该怎么做
[L235] 应该按**整数推理链路**生成 golden。例如第一层：
[L236] 
[L237] ```python
[L238] acc = sum((x_q - x_zp) * (w_q - w_zp)) + b_q
[L239] acc = relu(acc)
[L240] out_q = requantize(acc, multiplier, shift, out_zp)
[L241] ```
[L242] 
[L243] 然后把：
[L244] 
[L245] - 输入 int8
[L246] - 权重 int8
[L247] - 偏置 int32
[L248] - 输出 int32 / int8 golden
[L249] - scale / zero_point / multiplier / shift
[L250] 
[L251] 全部导出来。
[L252] 
[L253] 这样 FPGA 才能逐项对上。
[L254] 
[L255] ---
[L256] 
[L257] ## 4. 输入测试向量的量化方式不适合硬件批量验证
[L258] 
[L259] 你现在对每张图像：
[L260] 
[L261] ```python
[L262] q_img, img_scale, img_zero = quantize_tensor(img_flat)
[L263] ```
[L264] 
[L265] 这意味着**每张图一个 scale 和 zero_point**。  
[L266] 如果你的 FPGA 流水是固定 datapath，这会很麻烦：
[L267] 
[L268] - 每个样本都要动态更换量化参数
[L269] - 偏置量纲也得跟着变
[L270] - 对比 golden 时也会很乱
[L271] 
[L272] ### 更合理的做法
[L273] 建议统一输入量化策略，比如：
[L274] 
[L275] - 若保留 `Normalize((0.1307,), (0.3081,))`，则先统计训练集/校准集的输入范围，再确定一个全局输入 scale
[L276] - 或者更简单，**先不要 Normalize**，直接把 MNIST 灰度 `[0,1]` 映射到 `uint8/int8`
[L277] 
[L278] 对 FPGA 验证来说，后者往往更省事。
[L279] 
[L280] 例如极简方案可用：
[L281] 
[L282] - 输入像素 `0~255`
[L283] - 权重量化到 int8
[L284] - 累加 int32
[L285] - ReLU 后再截断或 requant
[L286] 
[L287] 这样更贴近硬件。
[L288] 
[L289] ---
[L290] 
[L291] ## 5. 导出的权重格式本身基本可用，但有几个注意点
[L292] 
[L293] 你现在导出权重：
[L294] 
[L295] ```python
[L296] for row in q_weight:
[L297]     for val in row:
[L298]         f.write(f'{val.item() & 0xFF:02x}\n')
[L299] ```
[L300] 
[L301] 这表示按 `weight[out][in]` 顺序逐个写出。  
[L302] 对于 `nn.Linear(784,128)`，PyTorch 权重 shape 是 `[128, 784]`，即：
[L303] 
[L304] - 外层：输出神经元
[L305] - 内层：输入通道
[L306] 
[L307] 这个顺序本身没错，但必须保证 RTL 端读取方式完全一致。
[L308] 
[L309] ### 你需要确认的点
[L310] 如果硬件是按下面方式算：
[L311] 
[L312] \[
[L313] y_j = \sum_i x_i w_{j,i}
[L314] \]
[L315] 
[L316] 那么文件顺序最好明确成：
[L317] 
[L318] - `neuron 0` 的 784 个权重
[L319] - `neuron 1` 的 784 个权重
[L320] - ...
[L321] - `neuron 127` 的 784 个权重
[L322] 
[L323] 你现在就是这个顺序。  
[L324] 但如果你的 BRAM 映射或地址生成器是按列优先/块优先取数，就会错位。
[L325] 
[L326] ### 建议
[L327] 导出时加一个说明文件，比如：
[L328] 
[L329] - weight layout: `[out][in]`, row-major
[L330] - bias layout: `[out]`
[L331] - input layout: `[in]`
[L332] 
[L333] ---
[L334] 
[L335] ## 6. `quant_params.npz` 在测试向量阶段几乎没被真正使用
[L336] 
[L337] 你这里：
[L338] 
[L339] ```python
[L340] quant_params = np.load('quantized_data/quant_params.npz')
[L341] ```
[L342] 
[L343] 加载了参数，但后面没有真正参与 golden 计算。  
[L344] 这说明当前流程里“量化参数被保存了”，但并没有形成闭环。
[L345] 
[L346] 这是个信号：**量化设计还停留在文件导出阶段，没有进入“硬件一致性验证阶段”。**
[L347] 
[L348] ---
[L349] 
[L350] ## 7. 其他次级问题
[L351] 
[L352] ### 问题 1：`torch.load()` 最好加 `map_location`
[L353] 不然在有些机器上保存于 GPU、加载于 CPU 时会出问题：
[L354] 
[L355] ```python
[L356] model.load_state_dict(torch.load(model_path, map_location='cpu'))
[L357] ```
[L358] 
[L359] ### 问题 2：训练时没看 accuracy
[L360] 现在只打 loss，没有任何分类精度指标。  
[L361] 如果要确认模型真的学到了，建议至少打印 test accuracy。
[L362] 
[L363] ### 问题 3：没有保存 label
[L364] 你打印了：
[L365] 
[L366] ```python
[L367] print(f"Generated test vector {i}, label: {label}")
[L368] ```
[L369] 
[L370] 但没有把 label 写入文件。  
[L371] 实际联调时最好保存一个 `labels.txt` 或 `meta.json`。
[L372] 
[L373] ### 问题 4：只保存了量化后的输出，没有保存中间 int32 累加值
[L374] 对于硬件调试，这很吃亏。  
[L375] 建议同时导出：
[L376] 
[L377] - `input_i.hex`
[L378] - `golden_acc_i.hex` 或每个神经元的 int32
[L379] - `golden_relu_i.hex`
[L380] - `golden_out_i.hex`
[L381] 
[L382] 这样你能定位是：
[L383] 
[L384] - 乘法错了
[L385] - 累加错了
[L386] - bias 错了
[L387] - ReLU 错了
[L388] - requant 错了
[L389] 
[L390] ---
[L391] 
[L392] # 我对这个 notebook 的总体判断
[L393] 
[L394] ## 现在这份代码“能不能用”？
[L395] **能作为一个思路草稿，但不能直接作为 FPGA 测试向量生成基线。**
[L396] 
[L397] ## 主要原因
[L398] 不是小 bug，而是三类“体系级不一致”：
[L399] 
[L400] 1. **训练目标错了**  
[L401]    当前模型没有真正学分类任务
[L402] 
[L403] 2. **量化策略不适合硬件**  
[L404]    每个样本动态量化，不利于 FPGA 固定算子验证
[L405] 
[L406] 3. **golden 生成链路和硬件链路不一致**  
[L407]    现在是“浮点输出再量化”，而不是“整数推理 golden”
[L408] 
[L409] ---
[L410] 
[L411] # 我建议你接下来这样改
[L412] 
[L413] ## 方案一：适合你们当前阶段的最小闭环
[L414] 如果你们现在的硬件目标是：
[L415] 
[L416] - 先做一个 `784 -> 128` 的单层 CIM/FC 阵列
[L417] - 验证 BRAM + MAC + ReLU + 控制 FSM
[L418] 
[L419] 那最合适的 notebook 应该改成下面这种逻辑：
[L420] 
[L421] ### 软件模型
[L422] 训练完整模型：
[L423] 
[L424] - `784 -> 128 -> 10`
[L425] 
[L426] ### 硬件实现
[L427] 先只实现第一层：
[L428] 
[L429] - `fc1 + relu`
[L430] 
[L431] ### 测试向量导出
[L432] 导出以下内容：
[L433] 
[L434] - `fc1_weight_int8.hex`
[L435] - `fc1_bias_int32.hex`
[L436] - `input_int8.hex`
[L437] - `fc1_acc_golden_int32.hex`
[L438] - `fc1_relu_golden_int32_or_int8.hex`
[L439] - `quant_config.json`
[L440] 
[L441] 这样你们硬件和软件就能逐层对齐。
[L442] 
[L443] ---
[L444] 
[L445] ## 方案二：直接瞄准最终演示
[L446] 如果你们想最后做“MNIST 分类结果上板演示”，那就：
[L447] 
[L448] - 模型改为两层 MLP
[L449] - 训练用 CrossEntropyLoss
[L450] - 做静态量化
[L451] - 先用 Python 写一个**整数推理参考实现**
[L452] - 再把这个整数推理结果作为 FPGA golden
[L453] 
[L454] 这是最稳的路线。
[L455] 
[L456] ---
[L457] 
[L458] # 我给你的具体结论
[L459] 
[L460] ## 必改
[L461] - `loss=output.pow(2).mean()` 改掉
[L462] - 模型结构和任务目标统一
[L463] - bias 改为 int32 量化
[L464] - golden 改成按整数推理链路生成
[L465] - 输入量化不要每张图单独动态算参数
[L466] 
[L467] ## 可保留
[L468] - `weights.hex` / `input_i.hex` 这种逐行写 hex 的导出方式
[L469] - 第一版只验证单层 FC + ReLU 的项目策略
[L470] - 先用 notebook 生成测试向量，再接 Verilog testbench 的总体流程
[L471] 
[L472] ---
[L473] 
[L474] 如果你愿意，我下一步可以直接帮你把这个 notebook **改成一版可用于 FPGA 联调的“正确版本”**，包括：
[L475] - 正确训练
[L476] - 静态 INT8 量化
[L477] - 生成 `weights/bias/input/golden` 四类 hex
[L478] - 明确对应单层 FC 硬件接口格式。
[L479] 
[L480] 用户:
[L481] 关于1.现在没有在训练一个 MNIST 分类器的问题，请分别给我路线a和路线b的完整代码；
[L482] 
[L483] 关于2.量化函数有工程问题，不适合直接对接 FPGA，
[L484] zero_point不是整数的问题帮我修改；每个张量动态量化这个问题给我一个合理的解决方案；偏置不能简单用 int8 单独量化请帮我提出解决方案和代码；
[L485] 
[L486] 关于3.generate_test_vectors() 生成的 golden，和硬件真实计算链路不一致，整数推理链路生成 golden，帮我完成设计；
[L487] 
[L488] 关于4输入测试向量的量化方式不适合硬件批量验证，帮我设计统一输入量化策略，贴近硬件即可；
[L489] 
[L490] 关于5. 导出的权重格式本身基本可用，但有几个注意点，这里文件顺序你先记住我的输出顺序，讲解一下我现在的输出顺序应该如何设计硬件读取顺序；
[L491] AI助手:
[L492] 我先把现有 notebook 里导出部分再核一遍，确保我后面给你的代码和你当前的文件顺序兼容。然后我会把路线 A、路线 B、量化、整数 golden、统一输入策略和硬件读取顺序一次性给全。
[L493] 
[L494] AI助手:
[L495] 下面我按你的 5 个问题，直接给出**可落地的完整方案和代码**。我会尽量保持和你现有工程思路一致，但把关键的工程问题修正掉。
[L496] 
[L497] ---
[L498] 
[L499] # 1. 关于“现在没有在训练一个 MNIST 分类器”的问题
[L500] 
[L501] 这里我给你两套完整代码。
[L502] 
[L503] ---
[L504] 
[L505] ## 路线 A：训练完整分类网络，但 FPGA 先只验证第一层 `fc1 + ReLU`
[L506] 
[L507] 这个路线最适合你现在的 FPGA/CIM 阶段。
[L508] 
[L509] ## 设计思想
[L510] 软件端训练一个完整分类器：
[L511] 
[L512] - `784 -> 128 -> 10`
[L513] 
[L514] 但硬件端先实现：
[L515] 
[L516] - `fc1: 784 -> 128`
[L517] - `ReLU`
[L518] 
[L519] 这样：
[L520] - 软件训练目标正确
[L521] - 第一层 feature 是有意义的
[L522] - 硬件联调难度低
[L523] - 后续再接 `fc2`
[L524] 
[L525] ---
[L526] 
[L527] ## 路线 A 完整代码
[L528] 
[L529] ```python
[L530] import os
[L531] import json
[L532] import math
[L533] import numpy as np
[L534] import torch
[L535] import torch.nn as nn
[L536] import torch.optim as optim
[L537] from torchvision import datasets, transforms
[L538] from torch.utils.data import DataLoader
[L539] 
[L540] # =========================================================
[L541] # 1. 模型：完整分类器，但硬件先验证 fc1 + relu
[L542] # =========================================================
[L543] class MLPRouteA(nn.Module):
[L544]     def __init__(self):
[L545]         super().__init__()
[L546]         self.fc1 = nn.Linear(784, 128, bias=True)
[L547]         self.relu = nn.ReLU()
[L548]         self.fc2 = nn.Linear(128, 10, bias=True)
[L549] 
[L550]     def forward(self, x):
[L551]         x = x.view(-1, 784)
[L552]         x = self.fc1(x)
[L553]         x = self.relu(x)
[L554]         x = self.fc2(x)
[L555]         return x
[L556] 
[L557]     def forward_fc1_relu(self, x):
[L558]         x = x.view(-1, 784)
[L559]         x = self.fc1(x)
[L560]         x = self.relu(x)
[L561]         return x
[L562] 
[L563] 
[L564] # =========================================================
[L565] # 2. 训练
[L566] # =========================================================
[L567] def train_route_a(model_path="mlp_route_a.pth", epochs=5, batch_size=128, lr=1e-3, device="cpu"):
[L568]     transform = transforms.Compose([
[L569]         transforms.ToTensor(),  # 保持 [0,1]，更贴近硬件
[L570]     ])
[L571] 
[L572]     train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
[L573]     test_dataset  = datasets.MNIST("./data", train=False, download=True, transform=transform)
[L574] 
[L575]     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
[L576]     test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
[L577] 
[L578]     model = MLPRouteA().to(device)
[L579]     criterion = nn.CrossEntropyLoss()
[L580]     optimizer = optim.Adam(model.parameters(), lr=lr)
[L581] 
[L582]     for epoch in range(epochs):
[L583]         model.train()
[L584]         total_loss = 0.0
[L585]         correct = 0
[L586]         total = 0
[L587] 
[L588]         for data, target in train_loader:
[L589]             data, target = data.to(device), target.to(device)
[L590] 
[L591]             optimizer.zero_grad()
[L592]             logits = model(data)
[L593]             loss = criterion(logits, target)
[L594]             loss.backward()
[L595]             optimizer.step()
[L596] 
[L597]             total_loss += loss.item() * data.size(0)
[L598]             pred = logits.argmax(dim=1)
[L599]             correct += (pred == target).sum().item()
[L600]             total += target.size(0)
[L601] 
[L602]         train_loss = total_loss / total
[L603]         train_acc = correct / total
[L604] 
[L605]         model.eval()
[L606]         test_correct = 0
[L607]         test_total = 0
[L608]         with torch.no_grad():
[L609]             for data, target in test_loader:
[L610]                 data, target = data.to(device), target.to(device)
[L611]                 logits = model(data)
[L612]                 pred = logits.argmax(dim=1)
[L613]                 test_correct += (pred == target).sum().item()
[L614]                 test_total += target.size(0)
[L615] 
[L616]         test_acc = test_correct / test_total
[L617]         print(f"Epoch [{epoch+1}/{epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")
[L618] 
[L619]     torch.save(model.state_dict(), model_path)
[L620]     print(f"Model saved to {model_path}")
[L621] 
[L622] 
[L623] if __name__ == "__main__":
[L624]     train_route_a()
[L625] ```
[L626] 
[L627] ---
[L628] 
[L629] ## 路线 B：直接做完整 MNIST 分类部署
[L630] 
[L631] 这个路线是最后的完整目标。
[L632] 
[L633] ---
[L634] 
[L635] ## 路线 B 完整代码
[L636] 
[L637] 和路线 A 训练结构几乎一样，但后续量化和导出会针对两层都做。
[L638] 
[L639] ```python
[L640] import os
[L641] import torch
[L642] import torch.nn as nn
[L643] import torch.optim as optim
[L644] from torchvision import datasets, transforms
[L645] from torch.utils.data import DataLoader
[L646] 
[L647] class MLPRouteB(nn.Module):
[L648]     def __init__(self):
[L649]         super().__init__()
[L650]         self.fc1 = nn.Linear(784, 128, bias=True)
[L651]         self.relu = nn.ReLU()
[L652]         self.fc2 = nn.Linear(128, 10, bias=True)
[L653] 
[L654]     def forward(self, x):
[L655]         x = x.view(-1, 784)
[L656]         x = self.fc1(x)
[L657]         x = self.relu(x)
[L658]         x = self.fc2(x)
[L659]         return x
[L660] 
[L661] def train_route_b(model_path="mlp_route_b.pth", epochs=5, batch_size=128, lr=1e-3, device="cpu"):
[L662]     transform = transforms.Compose([
[L663]         transforms.ToTensor(),  # 保持 [0,1]
[L664]     ])
[L665] 
[L666]     train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
[L667]     test_dataset  = datasets.MNIST("./data", train=False, download=True, transform=transform)
[L668] 
[L669]     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
[L670]     test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
[L671] 
[L672]     model = MLPRouteB().to(device)
[L673]     criterion = nn.CrossEntropyLoss()
[L674]     optimizer = optim.Adam(model.parameters(), lr=lr)
[L675] 
[L676]     for epoch in range(epochs):
[L677]         model.train()
[L678]         total_loss = 0.0
[L679]         correct = 0
[L680]         total = 0
[L681] 
[L682]         for data, target in train_loader:
[L683]             data, target = data.to(device), target.to(device)
[L684] 
[L685]             optimizer.zero_grad()
[L686]             logits = model(data)
[L687]             loss = criterion(logits, target)
[L688]             loss.backward()
[L689]             optimizer.step()
[L690] 
[L691]             total_loss += loss.item() * data.size(0)
[L692]             pred = logits.argmax(dim=1)
[L693]             correct += (pred == target).sum().item()
[L694]             total += target.size(0)
[L695] 
[L696]         train_loss = total_loss / total
[L697]         train_acc = correct / total
[L698] 
[L699]         model.eval()
[L700]         test_correct = 0
[L701]         test_total = 0
[L702]         with torch.no_grad():
[L703]             for data, target in test_loader:
[L704]                 data, target = data.to(device), target.to(device)
[L705]                 logits = model(data)
[L706]                 pred = logits.argmax(dim=1)
[L707]                 test_correct += (pred == target).sum().item()
[L708]                 test_total += target.size(0)
[L709] 
[L710]         test_acc = test_correct / test_total
[L711]         print(f"Epoch [{epoch+1}/{epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")
[L712] 
[L713]     torch.save(model.state_dict(), model_path)
[L714]     print(f"Model saved to {model_path}")
[L715] 
[L716] 
[L717] if __name__ == "__main__":
[L718]     train_route_b()
[L719] ```
[L720] 
[L721] ---
[L722] 
[L723] # 2. 关于量化函数工程问题
[L724] 
[L725] 你提了 3 个核心点，我分别给出方案。
[L726] 
[L727] ---
[L728] 
[L729] ## 2.1 zero_point 不是整数，怎么改
[L730] 
[L731] 你原来的问题在这里：
[L732] 
[L733] ```python
[L734] zero_point = qmin - min_val / scale
[L735] ```
[L736] 
[L737] 这个是浮点数，不适合硬件。
[L738] 
[L739] ---
[L740] 
[L741] ## 修改后的通用量化函数
[L742] 
[L743] 我给你一个**对 FPGA 更友好**的版本，支持：
[L744] 
[L745] - `zero_point` 取整
[L746] - clamp 到合法范围
[L747] - 支持对称量化和非对称量化
[L748] - 返回整数参数
[L749] 
[L750] ```python
[L751] import torch
[L752] 
[L753] def quantize_tensor_affine(tensor, num_bits=8, symmetric=False):
[L754]     """
[L755]     返回:
[L756]         q_tensor: 整数张量
[L757]         scale: float
[L758]         zero_point: int
[L759]         qmin, qmax: 整数范围
[L760]     """
[L761]     if num_bits == 8:
[L762]         qmin, qmax = -128, 127
[L763]         dtype = torch.int8
[L764]     else:
[L765]         raise ValueError("Only int8 supported in this example.")
[L766] 
[L767]     x_min = tensor.min().item()
[L768]     x_max = tensor.max().item()
[L769] 
[L770]     if symmetric:
[L771]         max_abs = max(abs(x_min), abs(x_max))
[L772]         scale = max_abs / qmax if max_abs != 0 else 1.0
[L773]         zero_point = 0
[L774]     else:
[L775]         if x_max == x_min:
[L776]             scale = 1.0
[L777]             zero_point = 0
[L778]         else:
[L779]             scale = (x_max - x_min) / (qmax - qmin)
[L780]             zero_point = round(qmin - x_min / scale)
[L781]             zero_point = max(qmin, min(qmax, zero_point))
[L782] 
[L783]     q = torch.round(tensor / scale + zero_point)
[L784]     q = torch.clamp(q, qmin, qmax).to(dtype)
[L785] 
[L786]     return q, float(scale), int(zero_point), qmin, qmax
[L787] ```
[L788] 
[L789] ---
[L790] 
[L791] ## 2.2 每个张量动态量化不适合硬件，合理解决方案
[L792] 
[L793] 你现在的问题是：
[L794] 
[L795] - 每张输入图单独算 `scale/zero_point`
[L796] - 每次输出也单独算
[L797] - 每次 bias 也单独算
[L798] 
[L799] 这会导致硬件参数不停变化，不利于 FPGA 批量验证。
[L800] 
[L801] ---
[L802] 
[L803] ## 合理方案：静态量化 / 全局量化
[L804] 
[L805] ### 输入量化策略
[L806] 对所有输入样本统一使用一个固定参数。
[L807] 
[L808] MNIST 最简单的硬件友好方案：
[L809] 
[L810] - 图像原始像素 `[0,1]`
[L811] - 映射到 int8 的 `[0,127]` 或 `[-128,127]`
[L812] 
[L813] 我建议你用下面两种之一：
[L814] 
[L815] ### 方案 1：无符号思路转 int8，简单贴近硬件
[L816] - 输入范围固定 `[0,1]`
[L817] - 量化成 `[0,127]`
[L818] - 存成 int8 时 zero_point = 0
[L819] 
[L820] 设置：
[L821] ```python
[L822] input_scale = 1.0 / 127.0
[L823] input_zero_point = 0
[L824] ```
[L825] 
[L826] 这样：
[L827] \[
[L828] x_q = round(x / input\_scale)
[L829] \]
[L830] 
[L831] 因为 `x in [0,1]`，所以 `x_q in [0,127]`
[L832] 
[L833] 这非常适合 FPGA。
[L834] 
[L835] ### 方案 2：居中对称量化
[L836] 如果你想让输入有正负，做：
[L837] ```python
[L838] x_float_centered = x_float - 0.5
[L839] input_scale = 0.5 / 127
[L840] zero_point = 0
[L841] ```
[L842] 但这会增加前处理复杂度。
[L843] 
[L844] **建议你现在先用方案 1。**
[L845] 
[L846] ---
[L847] 
[L848] ## 权重量化策略
[L849] 权重建议做**对称量化**：
[L850] 
[L851] - `zero_point = 0`
[L852] - int8
[L853] - 全层固定一个 `w_scale`
[L854] 
[L855] ```python
[L856] def quantize_weight_symmetric(weight):
[L857]     q_weight, w_scale, w_zero, _, _ = quantize_tensor_affine(weight, num_bits=8, symmetric=True)
[L858]     return q_weight, w_scale, w_zero
[L859] ```
[L860] 
[L861] ---
[L862] 
[L863] ## 2.3 偏置不能简单用 int8 单独量化，解决方案和代码
[L864] 
[L865] 这个是工程上必须改的。
[L866] 
[L867] ---
[L868] 
[L869] ## 正确原理
[L870] 
[L871] 如果一层计算为：
[L872] 
[L873] \[
[L874] y = Wx + b
[L875] \]
[L876] 
[L877] 输入和权重分别量化为：
[L878] 
[L879] \[
[L880] x \approx (x_q - z_x)s_x
[L881] \]
[L882] \[
[L883] w \approx (w_q - z_w)s_w
[L884] \]
[L885] 
[L886] 那么累加项单位是：
[L887] 
[L888] \[
[L889] s_x \cdot s_w
[L890] \]
[L891] 
[L892] 所以 bias 应该量化到同一量纲：
[L893] 
[L894] \[
[L895] b_q = round(b / (s_x \cdot s_w))
[L896] \]
[L897] 
[L898] 并且 bias 应该存成 **int32**。
[L899] 
[L900] ---
[L901] 
[L902] ## 偏置量化代码
[L903] 
[L904] ```python
[L905] def quantize_bias_int32(bias_fp32, input_scale, weight_scale):
[L906]     """
[L907]     bias_fp32: torch.Tensor(float32)
[L908]     return: int32 bias
[L909]     """
[L910]     bias_scale = input_scale * weight_scale
[L911]     q_bias = torch.round(bias_fp32 / bias_scale).to(torch.int32)
[L912]     return q_bias, float(bias_scale)
[L913] ```
[L914] 
[L915] ---
[L916] 
[L917] # 3. 关于 `generate_test_vectors()` 的 golden 不一致问题
[L918] 
[L919] 你要求的是：**用整数推理链路生成 golden**。这个必须按硬件链路来。
[L920] 
[L921] 下面我给出完整设计。
[L922] 
[L923] ---
[L924] 
[L925] ## 3.1 单层 FC + ReLU 的整数推理设计（路线 A）
[L926] 
[L927] ---
[L928] 
[L929] ## 整数推理流程
[L930] 
[L931] 对于第一层：
[L932] 
[L933] \[
[L934] acc_j = \sum_{i=0}^{783} x_{q,i} \cdot w_{q,j,i} + b_{q,j}
[L935] \]
[L936] 
[L937] 因为我们设计为：
[L938] 
[L939] - 输入 `zero_point = 0`
[L940] - 权重 `zero_point = 0`
[L941] 
[L942] 所以式子就很简单，不用减零点。
[L943] 
[L944] 然后：
[L945] 
[L946] \[
[L947] relu_j = max(0, acc_j)
[L948] \]
[L949] 
[L950] 如果你要再导出 int8 输出用于后级，也可以再做 requant。
[L951] 
[L952] ---
[L953] 
[L954] ## 3.2 统一的导出和 golden 生成代码（路线 A）
[L955] 
[L956] 这段代码是你现在最应该用的版本。
[L957] 
[L958] ```python
[L959] import os
[L960] import json
[L961] import math
[L962] import numpy as np
[L963] import torch
[L964] import torch.nn as nn
[L965] from torchvision import datasets, transforms
[L966] 
[L967] # =========================================================
[L968] # 模型
[L969] # =========================================================
[L970] class MLPRouteA(nn.Module):
[L971]     def __init__(self):
[L972]         super().__init__()
[L973]         self.fc1 = nn.Linear(784, 128, bias=True)
[L974]         self.relu = nn.ReLU()
[L975]         self.fc2 = nn.Linear(128, 10, bias=True)
[L976] 
[L977]     def forward(self, x):
[L978]         x = x.view(-1, 784)
[L979]         x = self.fc1(x)
[L980]         x = self.relu(x)
[L981]         x = self.fc2(x)
[L982]         return x
[L983] 
[L984] 
[L985] # =========================================================
[L986] # 量化工具
[L987] # =========================================================
[L988] def quantize_tensor_affine(tensor, num_bits=8, symmetric=False):
[L989]     if num_bits == 8:
[L990]         qmin, qmax = -128, 127
[L991]         dtype = torch.int8
[L992]     else:
[L993]         raise ValueError("Only int8 supported.")
[L994] 
[L995]     x_min = tensor.min().item()
[L996]     x_max = tensor.max().item()
[L997] 
[L998]     if symmetric:
[L999]         max_abs = max(abs(x_min), abs(x_max))
[L1000]         scale = max_abs / qmax if max_abs != 0 else 1.0
[L1001]         zero_point = 0
[L1002]     else:
[L1003]         if x_max == x_min:
[L1004]             scale = 1.0
[L1005]             zero_point = 0
[L1006]         else:
[L1007]             scale = (x_max - x_min) / (qmax - qmin)
[L1008]             zero_point = round(qmin - x_min / scale)
[L1009]             zero_point = max(qmin, min(qmax, zero_point))
[L1010] 
[L1011]     q = torch.round(tensor / scale + zero_point)
[L1012]     q = torch.clamp(q, qmin, qmax).to(dtype)
[L1013]     return q, float(scale), int(zero_point), qmin, qmax
[L1014] 
[L1015] 
[L1016] def quantize_input_fixed_0_1(x):
[L1017]     """
[L1018]     x in [0,1]
[L1019]     量化到 int8 [0,127]
[L1020]     """
[L1021]     scale = 1.0 / 127.0
[L1022]     zero_point = 0
[L1023]     q = torch.round(x / scale)
[L1024]     q = torch.clamp(q, 0, 127).to(torch.int8)
[L1025]     return q, scale, zero_point
[L1026] 
[L1027] 
[L1028] def quantize_weight_symmetric(weight):
[L1029]     q_w, w_scale, w_zp, _, _ = quantize_tensor_affine(weight, num_bits=8, symmetric=True)
[L1030]     return q_w, w_scale, w_zp
[L1031] 
[L1032] 
[L1033] def quantize_bias_int32(bias_fp32, input_scale, weight_scale):
[L1034]     bias_scale = input_scale * weight_scale
[L1035]     q_bias = torch.round(bias_fp32 / bias_scale).to(torch.int32)
[L1036]     return q_bias, float(bias_scale)
[L1037] 
[L1038] 
[L1039] def compute_requant_params(input_scale, weight_scale, output_scale, shift=24):
[L1040]     """
[L1041]     将 real_multiplier = (input_scale * weight_scale) / output_scale
[L1042]     近似成 integer multiplier + right shift
[L1043]     """
[L1044]     real_multiplier = (input_scale * weight_scale) / output_scale
[L1045]     multiplier = int(round(real_multiplier * (1 << shift)))
[L1046]     return multiplier, shift, real_multiplier
[L1047] 
[L1048] 
[L1049] def requantize_int32_to_int8(acc_int32, multiplier, shift, out_zero_point=0, relu=False):
[L1050]     y = (acc_int32.to(torch.int64) * multiplier) >> shift
[L1051]     y = y + out_zero_point
[L1052]     if relu:
[L1053]         y = torch.maximum(y, torch.tensor(out_zero_point, dtype=y.dtype))
[L1054]     y = torch.clamp(y, -128, 127).to(torch.int8)
[L1055]     return y
[L1056] 
[L1057] 
[L1058] # =========================================================
[L1059] # hex 导出
[L1060] # =========================================================
[L1061] def write_int8_hex_per_line(tensor, filename):
[L1062]     tensor = tensor.flatten().cpu().numpy()
[L1063]     with open(filename, "w") as f:
[L1064]         for v in tensor:
[L1065]             f.write(f"{int(v) & 0xFF:02x}\n")
[L1066] 
[L1067] def write_int32_hex_per_line(tensor, filename):
[L1068]     tensor = tensor.flatten().cpu().numpy()
[L1069]     with open(filename, "w") as f:
[L1070]         for v in tensor:
[L1071]             f.write(f"{int(v) & 0xFFFFFFFF:08x}\n")
[L1072] 
[L1073] 
[L1074] # =========================================================
[L1075] # 导出第一层量化参数
[L1076] # =========================================================
[L1077] def export_fc1_quantized(model_path="mlp_route_a.pth", out_dir="quant_route_a"):
[L1078]     os.makedirs(out_dir, exist_ok=True)
[L1079] 
[L1080]     model = MLPRouteA()
[L1081]     model.load_state_dict(torch.load(model_path, map_location="cpu"))
[L1082]     model.eval()
[L1083] 
[L1084]     weight = model.fc1.weight.data.clone()   # [128,784]
[L1085]     bias   = model.fc1.bias.data.clone()     # [128]
[L1086] 
[L1087]     # 固定输入量化参数
[L1088]     input_scale = 1.0 / 127.0
[L1089]     input_zero_point = 0
[L1090] 
[L1091]     # 权重量化（对称）
[L1092]     q_weight, weight_scale, weight_zero_point = quantize_weight_symmetric(weight)
[L1093] 
[L1094]     # 偏置量化（int32）
[L1095]     q_bias, bias_scale = quantize_bias_int32(bias, input_scale, weight_scale)
[L1096] 
[L1097]     # 输出量化参数（给 fc1 输出用）
[L1098]     # 这里按浮点 fc1 输出范围估计一个固定输出 scale；工程上可用校准集更稳
[L1099]     with torch.no_grad():
[L1100]         dummy = torch.rand(256, 1, 28, 28)
[L1101]         fc1_out = model.relu(model.fc1(dummy.view(-1, 784)))
[L1102]     max_out = fc1_out.max().item()
[L1103]     output_scale = max_out / 127.0 if max_out != 0 else 1.0
[L1104]     output_zero_point = 0
[L1105] 
[L1106]     multiplier, shift, real_multiplier = compute_requant_params(
[L1107]         input_scale, weight_scale, output_scale, shift=24
[L1108]     )
[L1109] 
[L1110]     write_int8_hex_per_line(q_weight, os.path.join(out_dir, "fc1_weight.hex"))
[L1111]     write_int32_hex_per_line(q_bias, os.path.join(out_dir, "fc1_bias.hex"))
[L1112] 
[L1113]     config = {
[L1114]         "input_scale": input_scale,
[L1115]         "input_zero_point": input_zero_point,
[L1116]         "weight_scale": weight_scale,
[L1117]         "weight_zero_point": weight_zero_point,
[L1118]         "bias_scale": bias_scale,
[L1119]         "output_scale": output_scale,
[L1120]         "output_zero_point": output_zero_point,
[L1121]         "requant_multiplier": multiplier,
[L1122]         "requant_shift": shift,
[L1123]         "requant_real_multiplier": real_multiplier,
[L1124]         "weight_shape": [128, 784],
[L1125]         "bias_shape": [128],
[L1126]         "weight_layout": "[out][in], row-major",
[L1127]         "input_layout": "[in]",
[L1128]         "output_layout": "[out]"
[L1129]     }
[L1130] 
[L1131]     with open(os.path.join(out_dir, "fc1_quant_config.json"), "w") as f:
[L1132]         json.dump(config, f, indent=2)
[L1133] 
[L1134]     print("Exported fc1 quantized parameters.")
[L1135] 
[L1136] 
[L1137] # =========================================================
[L1138] # 生成整数链路 golden
[L1139] # =========================================================
[L1140] def generate_test_vectors_route_a(model_path="mlp_route_a.pth", out_dir="quant_route_a", num_samples=10):
[L1141]     os.makedirs(out_dir, exist_ok=True)
[L1142] 
[L1143]     model = MLPRouteA()
[L1144]     model.load_state_dict(torch.load(model_path, map_location="cpu"))
[L1145]     model.eval()
[L1146] 
[L1147]     # 载入模型参数并量化
[L1148]     weight = model.fc1.weight.data.clone()   # [128,784]
[L1149]     bias   = model.fc1.bias.data.clone()
[L1150] 
[L1151]     input_scale = 1.0 / 127.0
[L1152]     input_zero_point = 0
[L1153] 
[L1154]     q_weight, weight_scale, weight_zero_point = quantize_weight_symmetric(weight)
[L1155]     q_bias, bias_scale = quantize_bias_int32(bias, input_scale, weight_scale)
[L1156] 
[L1157]     with torch.no_grad():
[L1158]         dummy = torch.rand(256, 1, 28, 28)
[L1159]         fc1_out = model.relu(model.fc1(dummy.view(-1, 784)))
[L1160]     max_out = fc1_out.max().item()
[L1161]     output_scale = max_out / 127.0 if max_out != 0 else 1.0
[L1162]     output_zero_point = 0
[L1163] 
[L1164]     multiplier, shift, _ = compute_requant_params(input_scale, weight_scale, output_scale, shift=24)
[L1165] 
[L1166]     transform = transforms.Compose([
[L1167]         transforms.ToTensor(),   # 不做 Normalize，统一输入范围 [0,1]
[L1168]     ])
[L1169]     test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
[L1170] 
[L1171]     labels = []
[L1172] 
[L1173]     for i in range(num_samples):
[L1174]         img, label = test_dataset[i]
[L1175]         x = img.view(-1)  # [784]
[L1176] 
[L1177]         # 固定输入量化
[L1178]         q_x, _, _ = quantize_input_fixed_0_1(x)  # [784]
[L1179] 
[L1180]         # int32 累加： [128,784] x [784] -> [128]
[L1181]         acc = torch.sum(q_weight.to(torch.int32) * q_x.to(torch.int32).unsqueeze(0), dim=1) + q_bias
[L1182] 
[L1183]         # ReLU(int32)
[L1184]         relu_acc = torch.clamp(acc, min=0)
[L1185] 
[L1186]         # requant 到 int8，便于后级或观测
[L1187]         q_out = requantize_int32_to_int8(
[L1188]             relu_acc, multiplier, shift, out_zero_point=output_zero_point, relu=False
[L1189]         )
[L1190] 
[L1191]         # 保存
[L1192]         write_int8_hex_per_line(q_x,   os.path.join(out_dir, f"input_{i}.hex"))
[L1193]         write_int32_hex_per_line(acc,  os.path.join(out_dir, f"golden_fc1_acc_{i}.hex"))
[L1194]         write_int32_hex_per_line(relu_acc, os.path.join(out_dir, f"golden_fc1_relu_acc_{i}.hex"))
[L1195]         write_int8_hex_per_line(q_out, os.path.join(out_dir, f"golden_fc1_out_{i}.hex"))
[L1196] 
[L1197]         labels.append(int(label))
[L1198]         print(f"sample {i}: label={label}")
[L1199] 
[L1200]     with open(os.path.join(out_dir, "labels.txt"), "w") as f:
[L1201]         for lb in labels:
[L1202]             f.write(f"{lb}\n")
[L1203] 
[L1204]     print("Generated integer golden test vectors for route A.")
[L1205] ```
[L1206] 
[L1207] ---
[L1208] 
[L1209] # 4. 输入测试向量统一量化策略
[L1210] 
[L1211] 你要的是**贴近硬件即可**。那我建议你用下面这个方案。
[L1212] 
[L1213] ---
[L1214] 
[L1215] ## 推荐方案：输入保持 `[0,1]`，统一量化到 int8 `[0,127]`
[L1216] 
[L1217] ### 设计理由
[L1218] 这对 FPGA 最友好：
[L1219] 
[L1220] - 不用每张图动态算 scale
[L1221] - 不用保留 MNIST 的 mean/std normalize
[L1222] - 输入像素天然接近图像传感器/存储格式
[L1223] - 便于后面 BRAM 或 ROM 存图
[L1224] 
[L1225] ### 固定规则
[L1226] \[
[L1227] x_q = round(x \times 127)
[L1228] \]
[L1229] 其中 `x in [0,1]`
[L1230] 
[L1231] 即：
[L1232] 
[L1233] ```python
[L1234] def quantize_input_fixed_0_1(x):
[L1235]     scale = 1.0 / 127.0
[L1236]     zero_point = 0
[L1237]     q = torch.round(x / scale)
[L1238]     q = torch.clamp(q, 0, 127).to(torch.int8)
[L1239]     return q, scale, zero_point
[L1240] ```
[L1241] 
[L1242] ---
[L1243] 
[L1244] ## 为什么我不建议你继续用：
[L1245] ```python
[L1246] transforms.Normalize((0.1307,), (0.3081,))
[L1247] ```
[L1248] 
[L1249] 因为这会把输入变成带负数的浮点分布，虽然适合软件训练，但不适合你现在的 FPGA 验证阶段：
[L1250] 
[L1251] - 输入分布复杂
[L1252] - 量化参数更麻烦
[L1253] - debug 不直观
[L1254] 
[L1255] 所以你现在更适合：
[L1256] 
[L1257] - 训练和测试都直接 `ToTensor()`
[L1258] - 用统一固定输入量化
[L1259] 
[L1260] ---
[L1261] 
[L1262] # 5. 你当前权重输出顺序，对应硬件怎么设计读取顺序
[L1263] 
[L1264] 你当前 notebook 的写法是：
[L1265] 
[L1266] ```python
[L1267] for row in q_weight:
[L1268]     for val in row:
[L1269]         f.write(f'{val.item() & 0xFF:02x}\n')
[L1270] ```
[L1271] 
[L1272] 而 `weight = model.fc1.weight.data` 的 shape 是：
[L1273] 
[L1274] ```python
[L1275] [128, 784]
[L1276] ```
[L1277] 
[L1278] 在 PyTorch 里，`nn.Linear(784,128)` 的权重定义是：
[L1279] 
[L1280] \[
[L1281] W[j][i]
[L1282] \]
[L1283] 
[L1284] 其中：
[L1285] 
[L1286] - `j` 是输出神经元索引，范围 `0~127`
[L1287] - `i` 是输入特征索引，范围 `0~783`
[L1288] 
[L1289] ---
[L1290] 
[L1291] ## 你现在的导出顺序是什么
[L1292] 
[L1293] 它是：
[L1294] 
[L1295] - 先写第 0 个输出神经元的 784 个权重
[L1296] - 再写第 1 个输出神经元的 784 个权重
[L1297] - ...
[L1298] - 最后写第 127 个输出神经元的 784 个权重
[L1299] 
[L1300] 也就是：
[L1301] 
[L1302] \[
[L1303] W[0][0], W[0][1], ..., W[0][783], W[1][0], ..., W[127][783]
[L1304] \]
[L1305] 
[L1306] 这是典型的：
[L1307] 
[L1308] - **按输出通道优先**
[L1309] - **行优先 row-major**
[L1310] - **布局 `[out][in]`**
[L1311] 
[L1312] ---
[L1313] 
[L1314] ## 对应的硬件读取顺序应该怎么设计
[L1315] 
[L1316] 如果你的硬件是“一个输出神经元一轮累加”的思路，那么最自然。
[L1317] 
[L1318] ---
[L1319] 
[L1320] ## 硬件读取方式 1：逐个输出神经元计算
[L1321] 
[L1322] ### 第 `j` 个输出的计算
[L1323] \[
[L1324] y_j = \sum_{i=0}^{783} x_i \cdot W[j][i] + b_j
[L1325] \]
[L1326] 
[L1327] ### 内存地址设计
[L1328] 如果 `weights.hex` 直接线性装入 BRAM，则：
[L1329] 
[L1330] \[
[L1331] addr = j \times 784 + i
[L1332] \]
[L1333] 
[L1334] 所以：
[L1335] 
[L1336] - 固定一个输出神经元 `j`
[L1337] - 地址从 `j*784` 扫到 `j*784+783`
[L1338] - 同时输入向量 `x[0:783]` 顺序送入
[L1339] - MAC 累加完成后，加 `bias[j]`
[L1340] - 输出 `y_j`
[L1341] 
[L1342] 然后 `j++`
[L1343] 
[L1344] ---
[L1345] 
[L1346] ## 这种方式的优点
[L1347] - 和你现在导出的文件顺序完全一致
[L1348] - 地址生成最简单
[L1349] - 软件和硬件最容易对齐
[L1350] - 最适合第一版 FPGA 联调
[L1351] 
[L1352] ---
[L1353] 
[L1354] ## 你现在应该怎么设计硬件
[L1355] 
[L1356] ### 输入存储
[L1357] `input_i.hex` 顺序也是：
[L1358] \[
[L1359] x[0], x[1], ..., x[783]
[L1360] \]
[L1361] 
[L1362] ### 权重存储
[L1363] `weights.hex` 顺序：
[L1364] \[
[L1365] W[0][0], W[0][1], ..., W[0][783], W[1][0], ..., W[127][783]
[L1366] \]
[L1367] 
[L1368] ### bias 存储
[L1369] `bias.hex` 顺序：
[L1370] \[
[L1371] b[0], b[1], ..., b[127]
[L1372] \]
[L1373] 
[L1374] ---
[L1375] 
[L1376] ## 控制流程
[L1377] 对于输出神经元 `j`：
[L1378] 
[L1379] 1. `acc = 0`
[L1380] 2. 对 `i = 0 ~ 783`
[L1381]    - 读 `x[i]`
[L1382]    - 读 `W[j][i]`
[L1383]    - `acc += x[i] * W[j][i]`
[L1384] 3. `acc += bias[j]`
[L1385] 4. `acc = relu(acc)`
[L1386] 5. 写 `out[j]`
[L1387] 
[L1388] 然后进入下一个 `j`
[L1389] 
[L1390] ---
[L1391] 
[L1392] ## 地址公式
[L1393] ### 权重地址
[L1394] \[
[L1395] weight\_addr = out\_idx \times 784 + in\_idx
[L1396] \]
[L1397] 
[L1398] ### 输入地址
[L1399] \[
[L1400] input\_addr = in\_idx
[L1401] \]
[L1402] 
[L1403] ### 偏置地址
[L1404] \[
[L1405] bias\_addr = out\_idx
[L1406] \]
[L1407] 
[L1408] ---
[L1409] 
[L1410] ## 什么时候要改输出顺序
[L1411] 只有在你未来做这些架构时才需要变：
[L1412] 
[L1413] - 多 PE 并行读多个输出神经元
[L1414] - 按列块分块存储
[L1415] - CIM 阵列按 bitline/wordline 特定映射
[L1416] - 片上 SRAM 宽度要求按 8/16/32 个权重打包
[L1417] 
[L1418] 但对于你现在这个阶段，**完全没必要先改**。  
[L1419] 你当前顺序其实正适合第一版硬件。
[L1420] 
[L1421] ---
[L1422] 
[L1423] # 我给你的最终建议
[L1424] 
[L1425] ## 你现在最推荐的工程路线
[L1426] 先走：
[L1427] 
[L1428] - **路线 A**
[L1429] - 完整训练分类器
[L1430] - FPGA 只做第一层 `fc1 + ReLU`
[L1431] - 输入统一量化 `[0,1] -> int8 [0,127]`
[L1432] - 权重 int8 对称量化
[L1433] - bias int32
[L1434] - golden 用整数推理链路生成
[L1435] - 权重按你现在的 `[out][in]` 顺序导出
[L1436] 
[L1437] 这是最稳、最容易跑通的一条路。
[L1438] 
[L1439] ---
[L1440] 
[L1441] ## 你下一步最值得做的事
[L1442] 把你当前 notebook 分成三个独立脚本：
[L1443] 
[L1444] 1. `train_route_a.py`
[L1445] 2. `export_fc1_quant.py`
[L1446] 3. `generate_test_vectors_route_a.py`
[L1447] 
[L1448] 这样后面调试不会乱。
[L1449] 
[L1450] 如果你要，我下一条可以直接继续给你一份**整理好的、可直接复制运行的完整工程版代码**，把路线 A 的训练、量化导出、golden 生成合并成一个干净版本。
[L1451] 
[L1452] 用户:
[L1453] 关于1.现在没有在训练一个 MNIST 分类器的问题，请分别给我路线a和路线b的完整代码；
[L1454] 
[L1455] 关于2.量化函数有工程问题，不适合直接对接 FPGA，
[L1456] zero_point不是整数的问题帮我修改；每个张量动态量化这个问题给我一个合理的解决方案；偏置不能简单用 int8 单独量化请帮我提出解决方案和代码；
[L1457] 
[L1458] 关于3.generate_test_vectors() 生成的 golden，和硬件真实计算链路不一致，整数推理链路生成 golden，帮我完成设计；
[L1459] 
[L1460] 关于4输入测试向量的量化方式不适合硬件批量验证，帮我设计统一输入量化策略，贴近硬件即可；
[L1461] 
[L1462] 关于5. 导出的权重格式本身基本可用，但有几个注意点，这里文件顺序你先记住我的输出顺序，讲解一下我现在的输出顺序应该如何设计硬件读取顺序；
[L1463] 
[L1464] 
[L1465] 问题6,7你帮我解决一下。
[L1466] 
[L1467] 按照你提出的方案二：直接瞄准最终演示
[L1468] 
[L1469] 如果你们想最后做“MNIST 分类结果上板演示”，那就：
[L1470] 
[L1471] 模型改为两层 MLP
[L1472] 
[L1473] 训练用 CrossEntropyLoss
[L1474] 
[L1475] 做静态量化
[L1476] 
[L1477] 先用 Python 写一个整数推理参考实现
[L1478] 
[L1479] 再把这个整数推理结果作为 FPGA golden进行处理。
[L1480] 其他应该修改的你帮我修改，和我说一声即可。
[L1481] AI助手:
[L1482] 我先基于你现有 notebook 的结构，把方案 A、方案 B，以及静态量化、整数推理 golden、测试向量导出这几部分整理成能直接落地的代码。然后我会把硬件读取顺序也一并说明清楚，保证软件导出和 RTL 地址映射能闭环。
[L1483] 
[L1484] AI助手:
[L1485] 下面我直接按你的要求给出可落地代码和设计说明。  
[L1486] 我把内容分成两部分：
[L1487] 
[L1488] 1. **路线 A 完整代码**：训练完整 MLP，但 FPGA 先只实现第一层 `fc1 + relu`，用于硬件验证  
[L1489] 2. **路线 B 完整代码**：直接瞄准最终演示，做完整两层 MLP、静态量化、整数推理 golden 导出
[L1490] 
[L1491] 你最后明确说“**按方案二直接瞄准最终演示**”，所以我建议你真正落地时优先用**路线 B**。  
[L1492] 路线 A 我也完整给你，方便你前期做单层阵列验证。
[L1493] 
[L1494] ---
[L1495] 
[L1496] # 一、你提出的几个问题，我已经统一改掉了
[L1497] 
[L1498] 这次代码里我已经一起处理了这些问题：
[L1499] 
[L1500] ## 已修改的问题
[L1501] ### 1）训练目标错误
[L1502] - 路线 A / B 都改成了真正的分类训练
[L1503] - 使用 `CrossEntropyLoss`
[L1504] 
[L1505] ### 2）`zero_point` 不是整数
[L1506] - 已改为**整数 zero_point**
[L1507] - 并做了合法范围裁剪
[L1508] 
[L1509] ### 3）动态量化不适合 FPGA
[L1510] - 改为**静态量化**
[L1511] - 输入、权重、输出都用**固定 scale / zero_point**
[L1512] - 不再每个样本单独算量化参数
[L1513] 
[L1514] ### 4）bias 不能简单 int8
[L1515] - 改为：
[L1516]   - `bias_q = round(bias_fp / (input_scale * weight_scale))`
[L1517]   - 使用 **int32** 保存
[L1518] 
[L1519] ### 5）golden 与硬件链路不一致
[L1520] - 改为**整数推理参考链路**
[L1521] - golden 由整数乘加、加 bias、ReLU、requant 得到
[L1522] - 和 FPGA 数据通路一致
[L1523] 
[L1524] ### 6）输入量化方式不适合批量验证
[L1525] - 统一输入量化策略
[L1526] - 采用更贴近硬件的方案：**MNIST 像素 0~1 -> uint8 0~255，再映射为 int8（减去 128）**
[L1527] - 固定 `input_scale = 1/255`，`input_zero_point = -128`
[L1528] 
[L1529] ### 7）其他问题我也一起修了
[L1530] - `torch.load(..., map_location=device)`
[L1531] - 增加 test accuracy
[L1532] - 保存标签文件
[L1533] - 保存中间层 int32 累加结果，便于 RTL 分级定位
[L1534] - 导出 json 配置，方便 FPGA 端对齐
[L1535] 
[L1536] ---
[L1537] 
[L1538] # 二、统一量化策略设计
[L1539] 
[L1540] 你要求“贴近硬件即可”，我这里给你一个非常适合 FPGA 的设计：
[L1541] 
[L1542] ## 输入量化
[L1543] 原始 MNIST 图像来自 `ToTensor()` 后范围是 `[0,1]`
[L1544] 
[L1545] 统一量化为：
[L1546] 
[L1547] \[
[L1548] x_q = round(x / s_x) + z_x
[L1549] \]
[L1550] 
[L1551] 其中：
[L1552] 
[L1553] - `s_x = 1 / 255`
[L1554] - `z_x = -128`
[L1555] 
[L1556] 这样就等价于：
[L1557] 
[L1558] ```python
[L1559] x_u8 = round(x * 255)      # 0~255
[L1560] x_q  = x_u8 - 128          # -128~127
[L1561] ```
[L1562] 
[L1563] 优点：
[L1564] - 很贴近硬件
[L1565] - 不需要每张图动态量化
[L1566] - FPGA 端容易处理
[L1567] - BRAM 存储方便
[L1568] 
[L1569] ## 权重量化
[L1570] 对每一层权重做**对称量化**：
[L1571] 
[L1572] \[
[L1573] w_q = round(w / s_w)
[L1574] \]
[L1575] 
[L1576] 其中：
[L1577] 
[L1578] \[
[L1579] s_w = \max(|w|) / 127
[L1580] \]
[L1581] 
[L1582] - `w_zero_point = 0`
[L1583] - `w_q` 存为 int8
[L1584] 
[L1585] ## 偏置量化
[L1586] 偏置必须与累加器同量纲：
[L1587] 
[L1588] \[
[L1589] b_q = round(b / (s_x \cdot s_w))
[L1590] \]
[L1591] 
[L1592] - 存为 int32
[L1593] 
[L1594] ## 输出量化
[L1595] 累加后得到 int32：
[L1596] 
[L1597] \[
[L1598] acc = \sum (x_q - z_x) \cdot (w_q - z_w) + b_q
[L1599] \]
[L1600] 
[L1601] 然后 ReLU，再 requant：
[L1602] 
[L1603] \[
[L1604] y_q = round(acc \cdot (s_x s_w / s_y)) + z_y
[L1605] \]
[L1606] 
[L1607] 输出层如需分类 argmax，最终也可直接基于 int32 或 requant 后 int8 做。
[L1608] 
[L1609] ---
[L1610] 
[L1611] # 三、路线 A：完整代码  
[L1612] ## 用途
[L1613] - 软件训练完整分类器：`784 -> 128 -> 10`
[L1614] - FPGA 先只实现第一层：`fc1 + relu`
[L1615] - 导出第一层输入、权重、bias、int32 累加 golden、relu 输出 golden
[L1616] 
[L1617] 这条路线适合你前期快速验证 CIM/FC 阵列。
[L1618] 
[L1619] ---
[L1620] 
[L1621] ## 路线 A 完整代码
[L1622] 
[L1623] ```python
[L1624] import os
[L1625] import json
[L1626] import math
[L1627] import numpy as np
[L1628] import torch
[L1629] import torch.nn as nn
[L1630] import torch.optim as optim
[L1631] from torchvision import datasets, transforms
[L1632] from torch.utils.data import DataLoader
[L1633] 
[L1634] # =========================================================
[L1635] # 1. 配置
[L1636] # =========================================================
[L1637] DATA_DIR = "./data"
[L1638] OUT_DIR = "./route_a_output"
[L1639] MODEL_PATH = os.path.join(OUT_DIR, "mlp_route_a.pth")
[L1640] 
[L1641] os.makedirs(OUT_DIR, exist_ok=True)
[L1642] 
[L1643] DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
[L1644] 
[L1645] 
[L1646] # =========================================================
[L1647] # 2. 模型：完整训练，但导出时只导出第一层用于 FPGA
[L1648] # =========================================================
[L1649] class MLPRouteA(nn.Module):
[L1650]     def __init__(self):
[L1651]         super().__init__()
[L1652]         self.fc1 = nn.Linear(784, 128, bias=True)
[L1653]         self.relu = nn.ReLU()
[L1654]         self.fc2 = nn.Linear(128, 10, bias=True)
[L1655] 
[L1656]     def forward(self, x):
[L1657]         x = x.view(-1, 784)
[L1658]         h = self.fc1(x)
[L1659]         h_relu = self.relu(h)
[L1660]         y = self.fc2(h_relu)
[L1661]         return y
[L1662] 
[L1663] 
[L1664] # =========================================================
[L1665] # 3. 数据
[L1666] #   路线 A 训练时仍然可以用标准 Normalize
[L1667] #   因为这里只是先训练出一个可用分类器
[L1668] # =========================================================
[L1669] def get_train_test_loaders(batch_size=128):
[L1670]     transform = transforms.Compose([
[L1671]         transforms.ToTensor(),
[L1672]         transforms.Normalize((0.1307,), (0.3081,))
[L1673]     ])
[L1674] 
[L1675]     train_dataset = datasets.MNIST(
[L1676]         DATA_DIR, train=True, download=True, transform=transform
[L1677]     )
[L1678]     test_dataset = datasets.MNIST(
[L1679]         DATA_DIR, train=False, download=True, transform=transform
[L1680]     )
[L1681] 
[L1682]     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
[L1683]     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
[L1684]     return train_loader, test_loader
[L1685] 
[L1686] 
[L1687] # =========================================================
[L1688] # 4. 训练 / 测试
[L1689] # =========================================================
[L1690] def train_model(num_epochs=5, lr=1e-3):
[L1691]     train_loader, test_loader = get_train_test_loaders()
[L1692] 
[L1693]     model = MLPRouteA().to(DEVICE)
[L1694]     optimizer = optim.Adam(model.parameters(), lr=lr)
[L1695]     criterion = nn.CrossEntropyLoss()
[L1696] 
[L1697]     for epoch in range(num_epochs):
[L1698]         model.train()
[L1699]         running_loss = 0.0
[L1700] 
[L1701]         for data, target in train_loader:
[L1702]             data, target = data.to(DEVICE), target.to(DEVICE)
[L1703] 
[L1704]             optimizer.zero_grad()
[L1705]             logits = model(data)
[L1706]             loss = criterion(logits, target)
[L1707]             loss.backward()
[L1708]             optimizer.step()
[L1709] 
[L1710]             running_loss += loss.item() * data.size(0)
[L1711] 
[L1712]         avg_loss = running_loss / len(train_loader.dataset)
[L1713]         acc = evaluate_model(model, test_loader)
[L1714]         print(f"[Epoch {epoch+1}/{num_epochs}] loss={avg_loss:.6f}, test_acc={acc:.4f}")
[L1715] 
[L1716]     torch.save(model.state_dict(), MODEL_PATH)
[L1717]     print(f"Model saved to: {MODEL_PATH}")
[L1718]     return model
[L1719] 
[L1720] 
[L1721] def evaluate_model(model, test_loader):
[L1722]     model.eval()
[L1723]     correct = 0
[L1724]     total = 0
[L1725] 
[L1726]     with torch.no_grad():
[L1727]         for data, target in test_loader:
[L1728]             data, target = data.to(DEVICE), target.to(DEVICE)
[L1729]             logits = model(data)
[L1730]             pred = logits.argmax(dim=1)
[L1731]             correct += (pred == target).sum().item()
[L1732]             total += target.size(0)
[L1733] 
[L1734]     return correct / total
[L1735] 
[L1736] 
[L1737] # =========================================================
[L1738] # 5. 量化函数
[L1739] #   路线 A：为了 FPGA 只验证第一层，采用固定输入量化策略
[L1740] # =========================================================
[L1741] INT8_QMIN = -128
[L1742] INT8_QMAX = 127
[L1743] INT32_QMIN = -2147483648
[L1744] INT32_QMAX = 2147483647
[L1745] 
[L1746] def clamp(x, low, high):
[L1747]     return max(low, min(high, x))
[L1748] 
[L1749] def quantize_affine_tensor(tensor_fp, scale, zero_point, qmin=-128, qmax=127, dtype=torch.int8):
[L1750]     q = torch.round(tensor_fp / scale + zero_point)
[L1751]     q = torch.clamp(q, qmin, qmax)
[L1752]     return q.to(dtype)
[L1753] 
[L1754] def calc_symmetric_scale(tensor_fp, num_bits=8):
[L1755]     qmax = (2 ** (num_bits - 1)) - 1
[L1756]     max_abs = max(abs(tensor_fp.min().item()), abs(tensor_fp.max().item()))
[L1757]     if max_abs == 0:
[L1758]         return 1.0
[L1759]     return max_abs / qmax
[L1760] 
[L1761] def quantize_weight_symmetric_int8(weight_fp):
[L1762]     scale = calc_symmetric_scale(weight_fp, num_bits=8)
[L1763]     zp = 0
[L1764]     weight_q = quantize_affine_tensor(weight_fp, scale, zp, -128, 127, torch.int8)
[L1765]     return weight_q, scale, zp
[L1766] 
[L1767] def quantize_bias_int32(bias_fp, input_scale, weight_scale):
[L1768]     bias_scale = input_scale * weight_scale
[L1769]     bias_q = torch.round(bias_fp / bias_scale)
[L1770]     bias_q = torch.clamp(bias_q, INT32_QMIN, INT32_QMAX).to(torch.int32)
[L1771]     return bias_q, bias_scale
[L1772] 
[L1773] def quantize_input_hw_style(img_fp_01):
[L1774]     """
[L1775]     img_fp_01: 原始像素 [0,1]
[L1776]     统一策略：
[L1777]         x_u8 = round(x * 255)
[L1778]         x_q  = x_u8 - 128
[L1779]     对应：
[L1780]         input_scale = 1/255
[L1781]         input_zero_point = -128
[L1782]     """
[L1783]     input_scale = 1.0 / 255.0
[L1784]     input_zero_point = -128
[L1785] 
[L1786]     x_u8 = torch.round(img_fp_01 * 255.0)
[L1787]     x_u8 = torch.clamp(x_u8, 0, 255)
[L1788]     x_q = x_u8.to(torch.int16) - 128
[L1789]     x_q = torch.clamp(x_q, -128, 127).to(torch.int8)
[L1790] 
[L1791]     return x_q, input_scale, input_zero_point
[L1792] 
[L1793] 
[L1794] # =========================================================
[L1795] # 6. 第一层整数推理 golden
[L1796] #   只对应 fc1 + relu
[L1797] # =========================================================
[L1798] def fc1_integer_reference(x_q, w_q, b_q, x_zp=-128, w_zp=0):
[L1799]     """
[L1800]     x_q: [784] int8
[L1801]     w_q: [128, 784] int8
[L1802]     b_q: [128] int32
[L1803] 
[L1804]     acc_j = sum_i (x_q[i] - x_zp)*(w_q[j,i] - w_zp) + b_q[j]
[L1805]     relu_acc = max(acc_j, 0)
[L1806]     """
[L1807]     x_int = x_q.to(torch.int32) - int(x_zp)
[L1808]     w_int = w_q.to(torch.int32) - int(w_zp)
[L1809] 
[L1810]     acc = torch.matmul(w_int, x_int) + b_q
[L1811]     relu_acc = torch.clamp(acc, min=0)
[L1812]     return acc.to(torch.int32), relu_acc.to(torch.int32)
[L1813] 
[L1814] 
[L1815] # =========================================================
[L1816] # 7. HEX 导出
[L1817] # =========================================================
[L1818] def write_int8_hex_1d(tensor, path):
[L1819]     with open(path, "w") as f:
[L1820]         for v in tensor.flatten():
[L1821]             f.write(f"{int(v.item()) & 0xFF:02x}\n")
[L1822] 
[L1823] def write_int32_hex_1d(tensor, path):
[L1824]     with open(path, "w") as f:
[L1825]         for v in tensor.flatten():
[L1826]             f.write(f"{int(v.item()) & 0xFFFFFFFF:08x}\n")
[L1827] 
[L1828] def export_route_a_artifacts(num_samples=10):
[L1829]     # 加载训练后的模型
[L1830]     model = MLPRouteA().to(DEVICE)
[L1831]     model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
[L1832]     model.eval()
[L1833] 
[L1834]     # 只提取 fc1
[L1835]     fc1_w_fp = model.fc1.weight.data.cpu()   # [128, 784]
[L1836]     fc1_b_fp = model.fc1.bias.data.cpu()     # [128]
[L1837] 
[L1838]     # 固定输入量化参数
[L1839]     input_scale = 1.0 / 255.0
[L1840]     input_zero_point = -128
[L1841] 
[L1842]     # 权重量化（对称 int8）
[L1843]     fc1_w_q, fc1_w_scale, fc1_w_zp = quantize_weight_symmetric_int8(fc1_w_fp)
[L1844] 
[L1845]     # 偏置量化（int32）
[L1846]     fc1_b_q, fc1_b_scale = quantize_bias_int32(fc1_b_fp, input_scale, fc1_w_scale)
[L1847] 
[L1848]     # 导出权重：按 [out][in] 顺序
[L1849]     write_int8_hex_1d(fc1_w_q.reshape(-1), os.path.join(OUT_DIR, "fc1_weight_int8.hex"))
[L1850]     write_int32_hex_1d(fc1_b_q, os.path.join(OUT_DIR, "fc1_bias_int32.hex"))
[L1851] 
[L1852]     # 保存配置
[L1853]     config = {
[L1854]         "layer": "fc1_only",
[L1855]         "weight_shape": [128, 784],
[L1856]         "weight_layout": "row-major [out][in]",
[L1857]         "input_layout": "[in]",
[L1858]         "bias_layout": "[out]",
[L1859]         "input_scale": input_scale,
[L1860]         "input_zero_point": input_zero_point,
[L1861]         "fc1_weight_scale": fc1_w_scale,
[L1862]         "fc1_weight_zero_point": fc1_w_zp,
[L1863]         "fc1_bias_scale": fc1_b_scale,
[L1864]         "fc1_bias_dtype": "int32"
[L1865]     }
[L1866] 
[L1867]     with open(os.path.join(OUT_DIR, "quant_config.json"), "w", encoding="utf-8") as f:
[L1868]         json.dump(config, f, indent=2, ensure_ascii=False)
[L1869] 
[L1870]     # 测试集：为了导出原始像素，不能 Normalize
[L1871]     test_transform = transforms.Compose([
[L1872]         transforms.ToTensor()
[L1873]     ])
[L1874]     test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=test_transform)
[L1875] 
[L1876]     labels = []
[L1877] 
[L1878]     for i in range(num_samples):
[L1879]         img_fp, label = test_dataset[i]  # [1,28,28], range [0,1]
[L1880]         labels.append(int(label))
[L1881] 
[L1882]         img_flat = img_fp.view(-1).cpu()
[L1883]         x_q, _, _ = quantize_input_hw_style(img_flat)
[L1884] 
[L1885]         acc_int32, relu_int32 = fc1_integer_reference(
[L1886]             x_q, fc1_w_q, fc1_b_q, x_zp=input_zero_point, w_zp=fc1_w_zp
[L1887]         )
[L1888] 
[L1889]         write_int8_hex_1d(x_q, os.path.join(OUT_DIR, f"input_{i}.hex"))
[L1890]         write_int32_hex_1d(acc_int32, os.path.join(OUT_DIR, f"fc1_acc_golden_{i}.hex"))
[L1891]         write_int32_hex_1d(relu_int32, os.path.join(OUT_DIR, f"fc1_relu_golden_{i}.hex"))
[L1892] 
[L1893]         print(f"Exported sample {i}, label={label}")
[L1894] 
[L1895]     with open(os.path.join(OUT_DIR, "labels.txt"), "w") as f:
[L1896]         for lb in labels:
[L1897]             f.write(f"{lb}\n")
[L1898] 
[L1899]     print("Route A export done.")
[L1900] 
[L1901] 
[L1902] if __name__ == "__main__":
[L1903]     train_model(num_epochs=5, lr=1e-3)
[L1904]     export_route_a_artifacts(num_samples=10)
[L1905] ```
[L1906] 
[L1907] ---
[L1908] 
[L1909] # 四、路线 B：完整代码  
[L1910] ## 用途
[L1911] 这是你现在真正应该用的版本：
[L1912] 
[L1913] - 完整两层 MLP：`784 -> 128 -> 10`
[L1914] - `CrossEntropyLoss`
[L1915] - 静态量化
[L1916] - Python 写整数推理参考实现
[L1917] - 导出 FPGA golden
[L1918] - 最终瞄准“MNIST 上板分类演示”
[L1919] 
[L1920] ---
[L1921] 
[L1922] ## 路线 B 完整代码
[L1923] 
[L1924] ```python
[L1925] import os
[L1926] import json
[L1927] import math
[L1928] import numpy as np
[L1929] import torch
[L1930] import torch.nn as nn
[L1931] import torch.optim as optim
[L1932] from torchvision import datasets, transforms
[L1933] from torch.utils.data import DataLoader
[L1934] 
[L1935] # =========================================================
[L1936] # 1. 配置
[L1937] # =========================================================
[L1938] DATA_DIR = "./data"
[L1939] OUT_DIR = "./route_b_output"
[L1940] MODEL_PATH = os.path.join(OUT_DIR, "mlp_route_b.pth")
[L1941] 
[L1942] os.makedirs(OUT_DIR, exist_ok=True)
[L1943] 
[L1944] DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
[L1945] 
[L1946] INT8_QMIN = -128
[L1947] INT8_QMAX = 127
[L1948] INT32_QMIN = -2147483648
[L1949] INT32_QMAX = 2147483647
[L1950] 
[L1951] 
[L1952] # =========================================================
[L1953] # 2. 模型
[L1954] # =========================================================
[L1955] class MLPRouteB(nn.Module):
[L1956]     def __init__(self):
[L1957]         super().__init__()
[L1958]         self.fc1 = nn.Linear(784, 128, bias=True)
[L1959]         self.relu = nn.ReLU()
[L1960]         self.fc2 = nn.Linear(128, 10, bias=True)
[L1961] 
[L1962]     def forward(self, x):
[L1963]         x = x.view(-1, 784)
[L1964]         x = self.fc1(x)
[L1965]         x = self.relu(x)
[L1966]         x = self.fc2(x)
[L1967]         return x
[L1968] 
[L1969] 
[L1970] # =========================================================
[L1971] # 3. 数据
[L1972] #   训练时这里直接用 ToTensor()，不做 Normalize
[L1973] #   这样更贴近最终硬件部署链路
[L1974] # =========================================================
[L1975] def get_train_test_loaders(batch_size=128):
[L1976]     transform = transforms.Compose([
[L1977]         transforms.ToTensor()
[L1978]     ])
[L1979] 
[L1980]     train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
[L1981]     test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
[L1982] 
[L1983]     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
[L1984]     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
[L1985]     return train_loader, test_loader
[L1986] 
[L1987] 
[L1988] # =========================================================
[L1989] # 4. 训练 / 测试
[L1990] # =========================================================
[L1991] def evaluate_model(model, test_loader):
[L1992]     model.eval()
[L1993]     total = 0
[L1994]     correct = 0
[L1995] 
[L1996]     with torch.no_grad():
[L1997]         for data, target in test_loader:
[L1998]             data, target = data.to(DEVICE), target.to(DEVICE)
[L1999]             logits = model(data)
[L2000]             pred = torch.argmax(logits, dim=1)
[L2001]             correct += (pred == target).sum().item()
[L2002]             total += target.size(0)
[L2003] 
[L2004]     return correct / total
[L2005] 
[L2006] 
[L2007] def train_model(num_epochs=8, lr=1e-3):
[L2008]     train_loader, test_loader = get_train_test_loaders()
[L2009] 
[L2010]     model = MLPRouteB().to(DEVICE)
[L2011]     optimizer = optim.Adam(model.parameters(), lr=lr)
[L2012]     criterion = nn.CrossEntropyLoss()
[L2013] 
[L2014]     for epoch in range(num_epochs):
[L2015]         model.train()
[L2016]         running_loss = 0.0
[L2017] 
[L2018]         for data, target in train_loader:
[L2019]             data, target = data.to(DEVICE), target.to(DEVICE)
[L2020] 
[L2021]             optimizer.zero_grad()
[L2022]             logits = model(data)
[L2023]             loss = criterion(logits, target)
[L2024]             loss.backward()
[L2025]             optimizer.step()
[L2026] 
[L2027]             running_loss += loss.item() * data.size(0)
[L2028] 
[L2029]         avg_loss = running_loss / len(train_loader.dataset)
[L2030]         acc = evaluate_model(model, test_loader)
[L2031]         print(f"[Epoch {epoch+1}/{num_epochs}] loss={avg_loss:.6f}, test_acc={acc:.4f}")
[L2032] 
[L2033]     torch.save(model.state_dict(), MODEL_PATH)
[L2034]     print(f"Model saved to: {MODEL_PATH}")
[L2035]     return model
[L2036] 
[L2037] 
[L2038] # =========================================================
[L2039] # 5. 量化工具
[L2040] # =========================================================
[L2041] def clamp(v, low, high):
[L2042]     return max(low, min(high, v))
[L2043] 
[L2044] def quantize_affine_tensor(tensor_fp, scale, zero_point, qmin=-128, qmax=127, dtype=torch.int8):
[L2045]     q = torch.round(tensor_fp / scale + zero_point)
[L2046]     q = torch.clamp(q, qmin, qmax)
[L2047]     return q.to(dtype)
[L2048] 
[L2049] def calc_symmetric_scale(tensor_fp, num_bits=8):
[L2050]     qmax = (2 ** (num_bits - 1)) - 1
[L2051]     max_abs = max(abs(tensor_fp.min().item()), abs(tensor_fp.max().item()))
[L2052]     if max_abs == 0:
[L2053]         return 1.0
[L2054]     return max_abs / qmax
[L2055] 
[L2056] def calc_affine_params_from_range(min_val, max_val, num_bits=8, signed=True):
[L2057]     if signed:
[L2058]         qmin = -(2 ** (num_bits - 1))
[L2059]         qmax =  (2 ** (num_bits - 1)) - 1
[L2060]     else:
[L2061]         qmin = 0
[L2062]         qmax = (2 ** num_bits) - 1
[L2063] 
[L2064]     if max_val == min_val:
[L2065]         scale = 1.0
[L2066]         zero_point = 0
[L2067]         return scale, zero_point
[L2068] 
[L2069]     scale = (max_val - min_val) / float(qmax - qmin)
[L2070]     zero_point = round(qmin - min_val / scale)
[L2071]     zero_point = int(clamp(zero_point, qmin, qmax))
[L2072]     return float(scale), int(zero_point)
[L2073] 
[L2074] def quantize_weight_symmetric_int8(weight_fp):
[L2075]     scale = calc_symmetric_scale(weight_fp, num_bits=8)
[L2076]     zp = 0
[L2077]     q = quantize_affine_tensor(weight_fp, scale, zp, INT8_QMIN, INT8_QMAX, torch.int8)
[L2078]     return q, scale, zp
[L2079] 
[L2080] def quantize_bias_int32(bias_fp, input_scale, weight_scale):
[L2081]     bias_scale = input_scale * weight_scale
[L2082]     q = torch.round(bias_fp / bias_scale)
[L2083]     q = torch.clamp(q, INT32_QMIN, INT32_QMAX).to(torch.int32)
[L2084]     return q, bias_scale
[L2085] 
[L2086] def quantize_input_hw_style(img_fp_01):
[L2087]     """
[L2088]     固定输入量化：
[L2089]         x_u8 = round(x * 255)
[L2090]         x_q  = x_u8 - 128
[L2091]     对应：
[L2092]         scale = 1/255
[L2093]         zero_point = -128
[L2094]     """
[L2095]     scale = 1.0 / 255.0
[L2096]     zp = -128
[L2097] 
[L2098]     x_u8 = torch.round(img_fp_01 * 255.0)
[L2099]     x_u8 = torch.clamp(x_u8, 0, 255)
[L2100]     x_q = x_u8.to(torch.int16) - 128
[L2101]     x_q = torch.clamp(x_q, -128, 127).to(torch.int8)
[L2102]     return x_q, scale, zp
[L2103] 
[L2104] def choose_output_scale_from_activation(act_fp):
[L2105]     """
[L2106]     输出激活采用对称量化到 int8
[L2107]     """
[L2108]     scale = calc_symmetric_scale(act_fp, num_bits=8)
[L2109]     zp = 0
[L2110]     return scale, zp
[L2111] 
[L2112] def requantize_from_int32(acc_int32, real_multiplier, out_zero_point=0):
[L2113]     """
[L2114]     简单软件参考：
[L2115]         y_q = round(acc * real_multiplier) + out_zero_point
[L2116]     """
[L2117]     y = torch.round(acc_int32.to(torch.float32) * real_multiplier) + out_zero_point
[L2118]     y = torch.clamp(y, INT8_QMIN, INT8_QMAX)
[L2119]     return y.to(torch.int8)
[L2120] 
[L2121] def quantize_multiplier_to_int(real_multiplier):
[L2122]     """
[L2123]     给 FPGA 做定点乘法时可用的近似参数：
[L2124]       real_multiplier ≈ M / 2^shift
[L2125]     返回:
[L2126]       M(int), shift(int)
[L2127]     这里只提供给配置文件参考；软件 golden 仍直接用 float real_multiplier
[L2128]     """
[L2129]     if real_multiplier == 0:
[L2130]         return 0, 0
[L2131] 
[L2132]     shift = 0
[L2133]     m = real_multiplier
[L2134]     while m < 0.5:
[L2135]         m *= 2.0
[L2136]         shift += 1
[L2137]     M = int(round(m * (1 << 31)))  # Q31
[L2138]     return M, shift + 31
[L2139] 
[L2140] 
[L2141] # =========================================================
[L2142] # 6. 量化模型准备
[L2143] # =========================================================
[L2144] def build_quantized_model(model_fp):
[L2145]     model_fp.eval()
[L2146]     model_fp = model_fp.cpu()
[L2147] 
[L2148]     # 固定输入量化参数
[L2149]     input_scale = 1.0 / 255.0
[L2150]     input_zero_point = -128
[L2151] 
[L2152]     # fc1 量化
[L2153]     fc1_w_fp = model_fp.fc1.weight.data
[L2154]     fc1_b_fp = model_fp.fc1.bias.data
[L2155]     fc1_w_q, fc1_w_scale, fc1_w_zp = quantize_weight_symmetric_int8(fc1_w_fp)
[L2156]     fc1_b_q, fc1_b_scale = quantize_bias_int32(fc1_b_fp, input_scale, fc1_w_scale)
[L2157] 
[L2158]     # 用一批训练样本统计 fc1 输出尺度
[L2159]     calib_loader, _ = get_train_test_loaders(batch_size=256)
[L2160]     calib_data, _ = next(iter(calib_loader))
[L2161]     calib_data = calib_data.view(-1, 784)
[L2162]     with torch.no_grad():
[L2163]         fc1_act_fp = model_fp.relu(model_fp.fc1(calib_data))
[L2164]     fc1_out_scale, fc1_out_zp = choose_output_scale_from_activation(fc1_act_fp)
[L2165] 
[L2166]     fc1_real_multiplier = (input_scale * fc1_w_scale) / fc1_out_scale
[L2167]     fc1_M, fc1_shift = quantize_multiplier_to_int(fc1_real_multiplier)
[L2168] 
[L2169]     # fc2 量化
[L2170]     fc2_w_fp = model_fp.fc2.weight.data
[L2171]     fc2_b_fp = model_fp.fc2.bias.data
[L2172]     fc2_w_q, fc2_w_scale, fc2_w_zp = quantize_weight_symmetric_int8(fc2_w_fp)
[L2173]     fc2_b_q, fc2_b_scale = quantize_bias_int32(fc2_b_fp, fc1_out_scale, fc2_w_scale)
[L2174] 
[L2175]     # 用一批样本统计 fc2 输出尺度
[L2176]     with torch.no_grad():
[L2177]         logits_fp = model_fp(calib_data.view(-1, 1, 28, 28))
[L2178]     fc2_out_scale, fc2_out_zp = choose_output_scale_from_activation(logits_fp)
[L2179] 
[L2180]     fc2_real_multiplier = (fc1_out_scale * fc2_w_scale) / fc2_out_scale
[L2181]     fc2_M, fc2_shift = quantize_multiplier_to_int(fc2_real_multiplier)
[L2182] 
[L2183]     qparams = {
[L2184]         "input": {
[L2185]             "scale": input_scale,
[L2186]             "zero_point": input_zero_point,
[L2187]             "dtype": "int8"
[L2188]         },
[L2189]         "fc1": {
[L2190]             "weight_scale": fc1_w_scale,
[L2191]             "weight_zero_point": fc1_w_zp,
[L2192]             "bias_scale": fc1_b_scale,
[L2193]             "output_scale": fc1_out_scale,
[L2194]             "output_zero_point": fc1_out_zp,
[L2195]             "real_multiplier": fc1_real_multiplier,
[L2196]             "multiplier_q31": fc1_M,
[L2197]             "shift": fc1_shift
[L2198]         },
[L2199]         "fc2": {
[L2200]             "weight_scale": fc2_w_scale,
[L2201]             "weight_zero_point": fc2_w_zp,
[L2202]             "bias_scale": fc2_b_scale,
[L2203]             "output_scale": fc2_out_scale,
[L2204]             "output_zero_point": fc2_out_zp,
[L2205]             "real_multiplier": fc2_real_multiplier,
[L2206]             "multiplier_q31": fc2_M,
[L2207]             "shift": fc2_shift
[L2208]         }
[L2209]     }
[L2210] 
[L2211]     quantized = {
[L2212]         "fc1_w_q": fc1_w_q.to(torch.int8),
[L2213]         "fc1_b_q": fc1_b_q.to(torch.int32),
[L2214]         "fc2_w_q": fc2_w_q.to(torch.int8),
[L2215]         "fc2_b_q": fc2_b_q.to(torch.int32),
[L2216]         "qparams": qparams
[L2217]     }
[L2218]     return quantized
[L2219] 
[L2220] 
[L2221] # =========================================================
[L2222] # 7. 整数推理参考实现
[L2223] # =========================================================
[L2224] def linear_int8_int32(x_q, w_q, b_q, x_zp, w_zp):
[L2225]     """
[L2226]     x_q: [in_features] int8
[L2227]     w_q: [out_features, in_features] int8
[L2228]     b_q: [out_features] int32
[L2229]     """
[L2230]     x_int = x_q.to(torch.int32) - int(x_zp)
[L2231]     w_int = w_q.to(torch.int32) - int(w_zp)
[L2232]     acc = torch.matmul(w_int, x_int) + b_q
[L2233]     return acc.to(torch.int32)
[L2234] 
[L2235] def relu_int32(x):
[L2236]     return torch.clamp(x, min=0).to(torch.int32)
[L2237] 
[L2238] def integer_inference_reference(img_fp_01, quantized):
[L2239]     """
[L2240]     完整整数参考链路：
[L2241]       input -> fc1(int32 acc) -> relu -> requant to int8
[L2242]             -> fc2(int32 acc) -> requant to int8 logits
[L2243]     """
[L2244]     qparams = quantized["qparams"]
[L2245] 
[L2246]     # 1) 输入量化
[L2247]     x_q, _, _ = quantize_input_hw_style(img_fp_01)
[L2248] 
[L2249]     # 2) fc1 int32 累加
[L2250]     fc1_acc = linear_int8_int32(
[L2251]         x_q,
[L2252]         quantized["fc1_w_q"],
[L2253]         quantized["fc1_b_q"],
[L2254]         x_zp=qparams["input"]["zero_point"],
[L2255]         w_zp=qparams["fc1"]["weight_zero_point"]
[L2256]     )
[L2257] 
[L2258]     # 3) ReLU
[L2259]     fc1_relu = relu_int32(fc1_acc)
[L2260] 
[L2261]     # 4) fc1 输出 requant -> int8
[L2262]     fc1_out_q = requantize_from_int32(
[L2263]         fc1_relu,
[L2264]         qparams["fc1"]["real_multiplier"],
[L2265]         qparams["fc1"]["output_zero_point"]
[L2266]     )
[L2267] 
[L2268]     # 5) fc2 int32 累加
[L2269]     fc2_acc = linear_int8_int32(
[L2270]         fc1_out_q,
[L2271]         quantized["fc2_w_q"],
[L2272]         quantized["fc2_b_q"],
[L2273]         x_zp=qparams["fc1"]["output_zero_point"],
[L2274]         w_zp=qparams["fc2"]["weight_zero_point"]
[L2275]     )
[L2276] 
[L2277]     # 6) 输出 requant -> int8 logits
[L2278]     logits_q = requantize_from_int32(
[L2279]         fc2_acc,
[L2280]         qparams["fc2"]["real_multiplier"],
[L2281]         qparams["fc2"]["output_zero_point"]
[L2282]     )
[L2283] 
[L2284]     pred = int(torch.argmax(logits_q).item())
[L2285] 
[L2286]     return {
[L2287]         "input_q": x_q,
[L2288]         "fc1_acc_int32": fc1_acc,
[L2289]         "fc1_relu_int32": fc1_relu,
[L2290]         "fc1_out_q": fc1_out_q,
[L2291]         "fc2_acc_int32": fc2_acc,
[L2292]         "logits_q": logits_q,
[L2293]         "pred": pred
[L2294]     }
[L2295] 
[L2296] 
[L2297] # =========================================================
[L2298] # 8. HEX 导出
[L2299] # =========================================================
[L2300] def write_int8_hex_1d(tensor, path):
[L2301]     with open(path, "w") as f:
[L2302]         for v in tensor.flatten():
[L2303]             f.write(f"{int(v.item()) & 0xFF:02x}\n")
[L2304] 
[L2305] def write_int32_hex_1d(tensor, path):
[L2306]     with open(path, "w") as f:
[L2307]         for v in tensor.flatten():
[L2308]             f.write(f"{int(v.item()) & 0xFFFFFFFF:08x}\n")
[L2309] 
[L2310] def export_all_artifacts(model_fp, quantized, num_samples=20):
[L2311]     # 1) 导出权重 / bias
[L2312]     # 按 [out][in] 顺序导出
[L2313]     write_int8_hex_1d(quantized["fc1_w_q"].reshape(-1), os.path.join(OUT_DIR, "fc1_weight_int8.hex"))
[L2314]     write_int32_hex_1d(quantized["fc1_b_q"], os.path.join(OUT_DIR, "fc1_bias_int32.hex"))
[L2315] 
[L2316]     write_int8_hex_1d(quantized["fc2_w_q"].reshape(-1), os.path.join(OUT_DIR, "fc2_weight_int8.hex"))
[L2317]     write_int32_hex_1d(quantized["fc2_b_q"], os.path.join(OUT_DIR, "fc2_bias_int32.hex"))
[L2318] 
[L2319]     # 2) 保存量化配置
[L2320]     config = {
[L2321]         "network": "784 -> 128 -> 10",
[L2322]         "weight_layout": {
[L2323]             "fc1": "row-major [out][in] = [128][784]",
[L2324]             "fc2": "row-major [out][in] = [10][128]"
[L2325]         },
[L2326]         "input_layout": "[784]",
[L2327]         "hidden_layout": "[128]",
[L2328]         "output_layout": "[10]",
[L2329]         "qparams": quantized["qparams"]
[L2330]     }
[L2331] 
[L2332]     with open(os.path.join(OUT_DIR, "quant_config.json"), "w", encoding="utf-8") as f:
[L2333]         json.dump(config, f, indent=2, ensure_ascii=False)
[L2334] 
[L2335]     # 3) 导出样本和 golden
[L2336]     test_transform = transforms.Compose([
[L2337]         transforms.ToTensor()
[L2338]     ])
[L2339]     test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=test_transform)
[L2340] 
[L2341]     labels = []
[L2342]     preds = []
[L2343] 
[L2344]     for i in range(num_samples):
[L2345]         img_fp, label = test_dataset[i]
[L2346]         img_flat = img_fp.view(-1).cpu()
[L2347] 
[L2348]         result = integer_inference_reference(img_flat, quantized)
[L2349] 
[L2350]         labels.append(int(label))
[L2351]         preds.append(int(result["pred"]))
[L2352] 
[L2353]         write_int8_hex_1d(result["input_q"], os.path.join(OUT_DIR, f"input_{i}.hex"))
[L2354] 
[L2355]         write_int32_hex_1d(result["fc1_acc_int32"], os.path.join(OUT_DIR, f"fc1_acc_{i}.hex"))
[L2356]         write_int32_hex_1d(result["fc1_relu_int32"], os.path.join(OUT_DIR, f"fc1_relu_{i}.hex"))
[L2357]         write_int8_hex_1d(result["fc1_out_q"], os.path.join(OUT_DIR, f"fc1_out_{i}.hex"))
[L2358] 
[L2359]         write_int32_hex_1d(result["fc2_acc_int32"], os.path.join(OUT_DIR, f"fc2_acc_{i}.hex"))
[L2360]         write_int8_hex_1d(result["logits_q"], os.path.join(OUT_DIR, f"logits_{i}.hex"))
[L2361] 
[L2362]         with open(os.path.join(OUT_DIR, f"pred_{i}.txt"), "w") as f:
[L2363]             f.write(str(result["pred"]))
[L2364] 
[L2365]         print(f"sample={i}, label={label}, pred={result['pred']}")
[L2366] 
[L2367]     with open(os.path.join(OUT_DIR, "labels.txt"), "w") as f:
[L2368]         for lb in labels:
[L2369]             f.write(f"{lb}\n")
[L2370] 
[L2371]     with open(os.path.join(OUT_DIR, "preds.txt"), "w") as f:
[L2372]         for pd in preds:
[L2373]             f.write(f"{pd}\n")
[L2374] 
[L2375]     print("All artifacts exported.")
[L2376] 
[L2377] 
[L2378] # =========================================================
[L2379] # 9. 验证量化后整数推理精度
[L2380] # =========================================================
[L2381] def evaluate_integer_pipeline(model_fp, quantized, num_samples=1000):
[L2382]     test_transform = transforms.Compose([
[L2383]         transforms.ToTensor()
[L2384]     ])
[L2385]     test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=test_transform)
[L2386] 
[L2387]     correct = 0
[L2388]     total = min(num_samples, len(test_dataset))
[L2389] 
[L2390]     for i in range(total):
[L2391]         img_fp, label = test_dataset[i]
[L2392]         img_flat = img_fp.view(-1).cpu()
[L2393]         result = integer_inference_reference(img_flat, quantized)
[L2394]         if result["pred"] == int(label):
[L2395]             correct += 1
[L2396] 
[L2397]     acc = correct / total
[L2398]     print(f"Integer pipeline accuracy over {total} samples: {acc:.4f}")
[L2399]     return acc
[L2400] 
[L2401] 
[L2402] # =========================================================
[L2403] # 10. 主流程
[L2404] # =========================================================
[L2405] if __name__ == "__main__":
[L2406]     # 1) 训练浮点模型
[L2407]     model_fp = train_model(num_epochs=8, lr=1e-3)
[L2408] 
[L2409]     # 2) 加载模型（防止单独运行导出时也能用）
[L2410]     model_fp = MLPRouteB().to(DEVICE)
[L2411]     model_fp.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
[L2412]     model_fp.eval()
[L2413] 
[L2414]     # 3) 构建量化模型
[L2415]     quantized = build_quantized_model(model_fp)
[L2416] 
[L2417]     # 4) 评估整数链路
[L2418]     evaluate_integer_pipeline(model_fp, quantized, num_samples=1000)
[L2419] 
[L2420]     # 5) 导出 FPGA 所需文件
[L2421]     export_all_artifacts(model_fp, quantized, num_samples=20)
[L2422] ```
[L2423] 
[L2424] ---
[L2425] 
[L2426] # 五、关于问题 2：逐项回答
[L2427] 
[L2428] ---
[L2429] 
[L2430] ## 2.1 `zero_point` 不是整数，怎么改
[L2431] 
[L2432] 你原来的问题在这里：
[L2433] 
[L2434] ```python
[L2435] zero_point = qmin - min_val / scale
[L2436] ```
[L2437] 
[L2438] 这里会得到浮点数。应改成：
[L2439] 
[L2440] ```python
[L2441] zero_point = round(qmin - min_val / scale)
[L2442] zero_point = int(max(qmin, min(qmax, zero_point)))
[L2443] ```
[L2444] 
[L2445] 通用函数我已经在路线 B 里给了：
[L2446] 
[L2447] ```python
[L2448] def calc_affine_params_from_range(min_val, max_val, num_bits=8, signed=True):
[L2449]     if signed:
[L2450]         qmin = -(2 ** (num_bits - 1))
[L2451]         qmax =  (2 ** (num_bits - 1)) - 1
[L2452]     else:
[L2453]         qmin = 0
[L2454]         qmax = (2 ** num_bits) - 1
[L2455] 
[L2456]     if max_val == min_val:
[L2457]         scale = 1.0
[L2458]         zero_point = 0
[L2459]         return scale, zero_point
[L2460] 
[L2461]     scale = (max_val - min_val) / float(qmax - qmin)
[L2462]     zero_point = round(qmin - min_val / scale)
[L2463]     zero_point = int(clamp(zero_point, qmin, qmax))
[L2464]     return float(scale), int(zero_point)
[L2465] ```
[L2466] 
[L2467] ---
[L2468] 
[L2469] ## 2.2 每个张量动态量化不适合 FPGA，合理解决方案
[L2470] 
[L2471] ### 你现在的问题
[L2472] 你现在是：
[L2473] - 每张输入图像单独量化
[L2474] - 每次输出也单独量化
[L2475] 
[L2476] 这会导致 FPGA 需要不断切换 scale / zero_point，不利于验证和部署。
[L2477] 
[L2478] ### 合理方案
[L2479] 改为**静态量化**：
[L2480] 
[L2481] #### 输入
[L2482] 固定：
[L2483] - `input_scale = 1/255`
[L2484] - `input_zero_point = -128`
[L2485] 
[L2486] #### 权重
[L2487] 每层固定：
[L2488] - `weight_scale` 固定
[L2489] - `weight_zero_point = 0`
[L2490] 
[L2491] #### 输出
[L2492] 每层固定：
[L2493] - `output_scale` 固定
[L2494] - `output_zero_point` 固定
[L2495] 
[L2496] 这样每个样本都共用同一套参数，硬件最容易做。
[L2497] 
[L2498] ---
[L2499] 
[L2500] ## 2.3 偏置不能用 int8，怎么解决
[L2501] 
[L2502] ### 正确原则
[L2503] bias 必须与 int32 累加器同量纲：
[L2504] 
[L2505] \[
[L2506] b_q = round(\frac{b}{s_x \cdot s_w})
[L2507] \]
[L2508] 
[L2509] ### 代码
[L2510] 我已经写成：
[L2511] 
[L2512] ```python
[L2513] def quantize_bias_int32(bias_fp, input_scale, weight_scale):
[L2514]     bias_scale = input_scale * weight_scale
[L2515]     q = torch.round(bias_fp / bias_scale)
[L2516]     q = torch.clamp(q, INT32_QMIN, INT32_QMAX).to(torch.int32)
[L2517]     return q, bias_scale
[L2518] ```
[L2519] 
[L2520] ### 为什么要这样
[L2521] 因为硬件里先做：
[L2522] 
[L2523] \[
[L2524] acc = \sum (x_q-z_x)(w_q-z_w)
[L2525] \]
[L2526] 
[L2527] 这个 `acc` 是 int32 量纲。  
[L2528] 如果 bias 还是 int8，直接相加就是错的。
[L2529] 
[L2530] ---
[L2531] 
[L2532] # 六、关于问题 3：整数推理链路 golden，怎么设计
[L2533] 
[L2534] 你要求“整数推理链路生成 golden”，现在路线 B 已经完整实现了。
[L2535] 
[L2536] ---
[L2537] 
[L2538] ## 设计链路
[L2539] 
[L2540] ### 第一层
[L2541] \[
[L2542] acc_1 = W_1 \cdot (x_q-z_x) + b_{1q}
[L2543] \]
[L2544] 
[L2545] \[
[L2546] relu_1 = max(acc_1, 0)
[L2547] \]
[L2548] 
[L2549] \[
[L2550] h_q = requant(relu_1)
[L2551] \]
[L2552] 
[L2553] ### 第二层
[L2554] \[
[L2555] acc_2 = W_2 \cdot (h_q-z_{h}) + b_{2q}
[L2556] \]
[L2557] 
[L2558] \[
[L2559] logits_q = requant(acc_2)
[L2560] \]
[L2561] 
[L2562] ### 分类
[L2563] \[
[L2564] pred = argmax(logits_q)
[L2565] \]
[L2566] 
[L2567] ---
[L2568] 
[L2569] ## 代码入口
[L2570] 整数参考实现就在这里：
[L2571] 
[L2572] ```python
[L2573] def integer_inference_reference(img_fp_01, quantized):
[L2574]     ...
[L2575] ```
[L2576] 
[L2577] 它导出了：
[L2578] 
[L2579] - `input_q`
[L2580] - `fc1_acc_int32`
[L2581] - `fc1_relu_int32`
[L2582] - `fc1_out_q`
[L2583] - `fc2_acc_int32`
[L2584] - `logits_q`
[L2585] - `pred`
[L2586] 
[L2587] 这些就是你硬件联调最有价值的 golden。
[L2588] 
[L2589] ---
[L2590] 
[L2591] # 七、关于问题 4：统一输入量化策略
[L2592] 
[L2593] 你要求“贴近硬件即可”，我建议你就用下面这个，不要再 Normalize。
[L2594] 
[L2595] ---
[L2596] 
[L2597] ## 推荐方案
[L2598] MNIST 原始像素 `[0,1]`
[L2599] 
[L2600] 统一做：
[L2601] 
[L2602] ```python
[L2603] x_u8 = round(x * 255)
[L2604] x_q = x_u8 - 128
[L2605] ```
[L2606] 
[L2607] 对应：
[L2608] 
[L2609] - `input_scale = 1/255`
[L2610] - `input_zero_point = -128`
[L2611] 
[L2612] ---
[L2613] 
[L2614] ## 优点
[L2615] ### 软件上
[L2616] - 简单
[L2617] - 稳定
[L2618] - 不依赖每张图的分布
[L2619] 
[L2620] ### 硬件上
[L2621] - BRAM 存储直观
[L2622] - 每个像素 8 bit
[L2623] - 乘加前减去 `-128` 很明确
[L2624] - 适合 testbench 和最终上板
[L2625] 
[L2626] ---
[L2627] 
[L2628] # 八、关于问题 5：你当前输出顺序如何设计硬件读取顺序
[L2629] 
[L2630] 你现在权重导出顺序是：
[L2631] 
[L2632] ```python
[L2633] for row in q_weight:
[L2634]     for val in row:
[L2635]         f.write(...)
[L2636] ```
[L2637] 
[L2638] 而 `nn.Linear(in_features, out_features)` 在 PyTorch 里权重 shape 是：
[L2639] 
[L2640] \[
[L2641] [out\_features, in\_features]
[L2642] \]
[L2643] 
[L2644] 所以你现在导出的顺序是：
[L2645] 
[L2646] \[
[L2647] w[0,0], w[0,1], ..., w[0,783], w[1,0], ..., w[127,783]
[L2648] \]
[L2649] 
[L2650] 也就是：
[L2651] 
[L2652] ## 当前输出顺序
[L2653] **按输出神经元逐行展开，row-major `[out][in]`**
[L2654] 
[L2655] ---
[L2656] 
[L2657] ## 对应的硬件读取顺序应该怎么设计
[L2658] 
[L2659] 如果你的第一层是：
[L2660] 
[L2661] \[
[L2662] y_j = \sum_{i=0}^{783} x_i \cdot w_{j,i} + b_j
[L2663] \]
[L2664] 
[L2665] 那么最自然的硬件读取顺序就是：
[L2666] 
[L2667] ### 方案：按神经元顺序读
[L2668] 对于每个输出神经元 `j`：
[L2669] 
[L2670] 1. 读取该神经元的 784 个权重
[L2671] 2. 与输入 784 维逐项相乘累加
[L2672] 3. 加偏置 `b_j`
[L2673] 4. 输出 `y_j`
[L2674] 
[L2675] ### 地址映射
[L2676] 如果 `fc1_weight_int8.hex` 按 `[out][in]` 展开，则：
[L2677] 
[L2678] \[
[L2679] addr = j \times 784 + i
[L2680] \]
[L2681] 
[L2682] 即：
[L2683] - `j`：输出神经元编号
[L2684] - `i`：输入特征编号
[L2685] 
[L2686] ### RTL 里可写成
[L2687] - 外层循环：`j = 0..127`
[L2688] - 内层循环：`i = 0..783`
[L2689] 
[L2690] 这和你当前导出顺序是完全匹配的。
[L2691] 
[L2692] ---
[L2693] 
[L2694] ## 如果硬件是并行阵列怎么办
[L2695] 如果你做的是 CIM 阵列或 PE 并行，也仍然建议逻辑上保持这个顺序，只是物理存储时可分块：
[L2696] 
[L2697] 例如分成 16 路并行：
[L2698] - 每次读同一个 `j` 的 16 个连续权重
[L2699] - 输入也同步读 16 个元素
[L2700] - 做 16 路乘加
[L2701] - 局部累加后再归并
[L2702] 
[L2703] 本质上还是从 `[out][in]` 顺序切块读取。
[L2704] 
[L2705] ---
[L2706] 
[L2707] ## 第二层同理
[L2708] 第二层 `fc2` shape 是 `[10,128]`
[L2709] 
[L2710] 地址映射：
[L2711] 
[L2712] \[
[L2713] addr = j \times 128 + i
[L2714] \]
[L2715] 
[L2716] 其中：
[L2717] - `j = 0..9`
[L2718] - `i = 0..127`
[L2719] 
[L2720] ---
[L2721] 
[L2722] # 九、问题 6 和 7，我已经一起解决了
[L2723] 
[L2724] ---
[L2725] 
[L2726] ## 问题 6：`quant_params.npz` 没真正参与闭环
[L2727] 现在我不建议继续用 `npz` 做核心配置。  
[L2728] 我已经改成了：
[L2729] 
[L2730] - `quant_config.json`
[L2731] 
[L2732] 里面包含：
[L2733] - 输入 scale / zp
[L2734] - 每层 weight_scale / weight_zp
[L2735] - bias_scale
[L2736] - output_scale / output_zp
[L2737] - real_multiplier
[L2738] - multiplier_q31
[L2739] - shift
[L2740] 
[L2741] 这对硬件更友好，也更好查。
[L2742] 
[L2743] ---
[L2744] 
[L2745] ## 问题 7：其他工程问题
[L2746] 我已经一起处理：
[L2747] 
[L2748] ### 1）`torch.load()` 增加 `map_location`
[L2749] 已改：
[L2750] 
[L2751] ```python
[L2752] torch.load(MODEL_PATH, map_location=DEVICE)
[L2753] ```
[L2754] 
[L2755] ### 2）训练时增加准确率
[L2756] 每个 epoch 后打印 test accuracy
[L2757] 
[L2758] ### 3）保存 label
[L2759] 已导出：
[L2760] - `labels.txt`
[L2761] 
[L2762] ### 4）保存中间层 golden
[L2763] 已导出：
[L2764] - `fc1_acc_x.hex`
[L2765] - `fc1_relu_x.hex`
[L2766] - `fc1_out_x.hex`
[L2767] - `fc2_acc_x.hex`
[L2768] - `logits_x.hex`
[L2769] 
[L2770] 这样 RTL 调试时你能定位是哪一级错了。
[L2771] 
[L2772] ---
[L2773] 
[L2774] # 十、你最终应该选哪套
[L2775] 
[L2776] ## 如果你现在要交阶段成果、先验证阵列
[L2777] 用路线 A。  
[L2778] 因为它最容易把硬件链路先跑通。
[L2779] 
[L2780] ## 如果你现在就想直奔“MNIST 上板分类演示”
[L2781] 用路线 B。  
[L2782] 这就是你刚刚指定的目标，我也已经按这个目标把代码写全了。
[L2783] 
[L2784] ---
[L2785] 
[L2786] # 十一、我建议你下一步的落地顺序
[L2787] 
[L2788] 你现在最稳的执行顺序是：
[L2789] 
[L2790] ### 第一步
[L2791] 先跑路线 B 的 Python 脚本，得到：
[L2792] - 浮点模型精度
[L2793] - 整数量化链路精度
[L2794] - 全套导出文件
[L2795] 
[L2796] ### 第二步
[L2797] 先在 FPGA 上只实现第一层并对齐：
[L2798] - `input`
[L2799] - `fc1_weight`
[L2800] - `fc1_bias`
[L2801] - `fc1_acc`
[L2802] - `fc1_relu`
[L2803] - `fc1_out`
[L2804] 
[L2805] ### 第三步
[L2806] 再补第二层：
[L2807] - `fc2_weight`
[L2808] - `fc2_bias`
[L2809] - `fc2_acc`
[L2810] - `logits`
[L2811] - `argmax`
[L2812] 
[L2813] ---
[L2814] 
[L2815] 如果你愿意，我下一条可以继续直接给你一份**更适合 notebook 使用的版本**，也就是把路线 B 拆成多个单元格，按你现在 `ipynb` 的风格排好。
[L2816] 
[L2817] 用户:
[L2818] 现在已经把导出测试向量的部分写好了，请帮我分析导出的文件的格式，以及如何使用
[L2819] AI助手:
[L2820] 
[L2821] 
[L2822] AI助手:
[L2823] 我先把你当前导出的目录和文件内容抽样检查一下，确认每类文件到底是几位十六进制、按什么维度展开、对应哪一级数据通路。然后我会按“文件格式 → RTL/Testbench 怎么读 → 怎么对拍”的顺序给你整理清楚。
[L2824] 
[L2825] AI助手:
[L2826] 我根据你现在 notebook 里的导出代码，已经能把**导出文件格式**和**使用方法**说清楚了。你当前实际导出的文件主要有这几类：
[L2827] 
[L2828] - `weights.hex`
[L2829] - `bias.hex`
[L2830] - `quant_params.npz`
[L2831] - `input_i.hex`
[L2832] - `golden_i.hex`
[L2833] 
[L2834] 下面我按“是什么、怎么组织、硬件怎么读、联调怎么用”来讲。
[L2835] 
[L2836] ---
[L2837] 
[L2838] # 1. 你现在导出的文件格式
[L2839] 
[L2840] ## 1.1 `weights.hex`
[L2841] 
[L2842] 这是第一层全连接层 `fc1.weight` 的量化权重文件。
[L2843] 
[L2844] ### 来源
[L2845] 代码里：
[L2846] 
[L2847] ```python
[L2848] weight = model.fc1.weight.data  # [128, 784]
[L2849] ...
[L2850] with open(f'{output_dir}/weights.hex', 'w') as f:
[L2851]     for row in q_weight:
[L2852]         for val in row:
[L2853]             f.write(f'{val.item() & 0xFF:02x}\n')
[L2854] ```
[L2855] 
[L2856] ### 文件格式
[L2857] - 文本文件
[L2858] - **每行一个 8 bit 数值**
[L2859] - 用 **2 位十六进制** 表示
[L2860] - 负数采用 **8 位二补码的低 8 位**
[L2861] 
[L2862] 例如：
[L2863] - `00` 表示 0
[L2864] - `01` 表示 1
[L2865] - `7f` 表示 127
[L2866] - `ff` 表示 -1
[L2867] - `80` 表示 -128
[L2868] 
[L2869] ### 总行数
[L2870] 因为权重 shape 是 `[128, 784]`，所以总共有：
[L2871] 
[L2872] \[
[L2873] 128 \times 784 = 100352
[L2874] \]
[L2875] 
[L2876] 也就是：
[L2877] 
[L2878] - `weights.hex` 一共 **100352 行**
[L2879] 
[L2880] ### 展开顺序
[L2881] 当前是：
[L2882] 
[L2883] ```python
[L2884] for row in q_weight:
[L2885]     for val in row:
[L2886] ```
[L2887] 
[L2888] 所以顺序是：
[L2889] 
[L2890] \[
[L2891] w[0,0], w[0,1], ..., w[0,783], w[1,0], ..., w[127,783]
[L2892] \]
[L2893] 
[L2894] 也就是：
[L2895] 
[L2896] - **按输出神经元 row-major 展开**
[L2897] - 格式为 **`[out][in]`**
[L2898] 
[L2899] 这点非常重要，RTL 必须按这个顺序取。
[L2900] 
[L2901] ---
[L2902] 
[L2903] ## 1.2 `bias.hex`
[L2904] 
[L2905] 这是第一层偏置文件。
[L2906] 
[L2907] ### 来源
[L2908] ```python
[L2909] bias = model.fc1.bias.data      # [128]
[L2910] ...
[L2911] with open(f'{output_dir}/bias.hex', 'w') as f:
[L2912]     for val in q_bias:
[L2913]         f.write(f'{val.item() & 0xFF:02x}\n')
[L2914] ```
[L2915] 
[L2916] ### 文件格式
[L2917] - 文本文件
[L2918] - 每行一个量化后的偏置
[L2919] - **2 位十六进制**
[L2920] - 当前代码里它被导成了 **int8**
[L2921] 
[L2922] ### 总行数
[L2923] - 偏置长度是 128
[L2924] - 所以 `bias.hex` 一共 **128 行**
[L2925] 
[L2926] ### 当前问题
[L2927] 这个文件虽然格式上没问题，但**工程上有问题**：
[L2928] 
[L2929] - 你现在是把 bias 当 int8 存
[L2930] - 但硬件 MAC 累加结果通常是 int32
[L2931] - 所以这个 `bias.hex` 现在**不适合直接用于真实整数推理链路**
[L2932] 
[L2933] 也就是说：
[L2934] - 这个文件可以先拿来做“形式演示”
[L2935] - 但不适合做你后面要的最终整数 golden 对拍
[L2936] 
[L2937] ---
[L2938] 
[L2939] ## 1.3 `quant_params.npz`
[L2940] 
[L2941] 这是量化参数文件。
[L2942] 
[L2943] ### 来源
[L2944] ```python
[L2945] np.savez(f'{output_dir}/quant_params.npz',
[L2946]          w_scale=w_scale, w_zero=w_zero,
[L2947]          b_scale=b_scale, b_zero=b_zero)
[L2948] ```
[L2949] 
[L2950] ### 里面有什么
[L2951] 当前保存了：
[L2952] 
[L2953] - `w_scale`
[L2954] - `w_zero`
[L2955] - `b_scale`
[L2956] - `b_zero`
[L2957] 
[L2958] ### 用途
[L2959] 理论上它用于：
[L2960] - 软件端反量化
[L2961] - 硬件端知道量化参数
[L2962] 
[L2963] ### 当前实际情况
[L2964] 你现在的 `generate_test_vectors()` 虽然加载了它：
[L2965] 
[L2966] ```python
[L2967] quant_params = np.load('quantized_data/quant_params.npz')
[L2968] ```
[L2969] 
[L2970] 但后面**几乎没真正用上**。  
[L2971] 所以它现在更像一个“记录文件”，还不是硬件链路真正依赖的配置文件。
[L2972] 
[L2973] ---
[L2974] 
[L2975] ## 1.4 `input_i.hex`
[L2976] 
[L2977] 这是第 `i` 个测试样本的输入文件。
[L2978] 
[L2979] ### 来源
[L2980] ```python
[L2981] img, label = test_dataset[i]
[L2982] img_flat = img.view(-1, 784)
[L2983] q_img, img_scale, img_zero = quantize_tensor(img_flat)
[L2984] 
[L2985] with open(f'quantized_data/input_{i}.hex', 'w') as f:
[L2986]     for val in q_img.flatten():
[L2987]         f.write(f'{val.item() & 0xFF:02x}\n')
[L2988] ```
[L2989] 
[L2990] ### 文件格式
[L2991] - 文本文件
[L2992] - 每行一个输入元素
[L2993] - **2 位十六进制**
[L2994] - 量化后为 int8
[L2995] 
[L2996] ### 总行数
[L2997] MNIST 一张图是 `28×28=784`，所以：
[L2998] 
[L2999] - 每个 `input_i.hex` 一共 **784 行**
[L3000] 
[L3001] ### 排列顺序
[L3002] 因为：
[L3003] 
[L3004] ```python
[L3005] img_flat = img.view(-1, 784)
[L3006] for val in q_img.flatten():
[L3007] ```
[L3008] 
[L3009] 所以顺序就是图像展平后的顺序，通常是：
[L3010] 
[L3011] - 第 0 行：像素 `(0,0)`
[L3012] - 第 1 行：像素 `(0,1)`
[L3013] - ...
[L3014] - 第 27 行：像素 `(0,27)`
[L3015] - 第 28 行：像素 `(1,0)`
[L3016] - ...
[L3017] 
[L3018] 也就是：
[L3019] 
[L3020] - **按行优先 row-major 展开**
[L3021] - 输入向量索引 `i = row * 28 + col`
[L3022] 
[L3023] RTL 里要按这个顺序喂入。
[L3024] 
[L3025] ---
[L3026] 
[L3027] ## 1.5 `golden_i.hex`
[L3028] 
[L3029] 这是第 `i` 个测试样本的软件 golden 输出。
[L3030] 
[L3031] ### 来源
[L3032] ```python
[L3033] with torch.no_grad():
[L3034]     output = model(img_flat)
[L3035] 
[L3036] q_output, _, _ = quantize_tensor(output)
[L3037] with open(f'quantized_data/golden_{i}.hex', 'w') as f:
[L3038]     for val in q_output.flatten():
[L3039]         f.write(f'{val.item() & 0xFF:02x}\n')
[L3040] ```
[L3041] 
[L3042] ### 文件格式
[L3043] - 文本文件
[L3044] - 每行一个输出元素
[L3045] - **2 位十六进制**
[L3046] - 量化后为 int8
[L3047] 
[L3048] ### 总行数
[L3049] 你现在模型是：
[L3050] 
[L3051] ```python
[L3052] self.fc1 = nn.Linear(784,128)
[L3053] self.relu = nn.ReLU()
[L3054] ```
[L3055] 
[L3056] 所以输出是 128 维。
[L3057] 
[L3058] 也就是：
[L3059] - 每个 `golden_i.hex` 一共 **128 行**
[L3060] 
[L3061] ### 当前含义
[L3062] 这个 `golden_i.hex` 表示：
[L3063] 
[L3064] - 先用浮点模型跑出 `fc1 + relu` 的浮点输出
[L3065] - 再把这个浮点输出单独量化成 int8
[L3066] 
[L3067] 所以它的意义是：
[L3068] 
[L3069] - **“浮点模型输出的量化结果”**
[L3070] - 不是严格意义上的“整数硬件链路 golden”
[L3071] 
[L3072] 这点你后面做最终方案时必须记住。
[L3073] 
[L3074] ---
[L3075] 
[L3076] # 2. 这些文件如何使用
[L3077] 
[L3078] ---
[L3079] 
[L3080] ## 2.1 最直接的使用方式：Verilog `$readmemh`
[L3081] 
[L3082] 因为这些 `.hex` 文件是：
[L3083] - 文本
[L3084] - 每行一个十六进制数
[L3085] 
[L3086] 所以最适合直接在 Verilog 里用：
[L3087] 
[L3088] ```verilog
[L3089] reg [7:0] weight_mem [0:100351];
[L3090] reg [7:0] bias_mem   [0:127];
[L3091] reg [7:0] input_mem  [0:783];
[L3092] reg [7:0] golden_mem [0:127];
[L3093] 
[L3094] initial begin
[L3095]     $readmemh("weights.hex", weight_mem);
[L3096]     $readmemh("bias.hex", bias_mem);
[L3097]     $readmemh("input_0.hex", input_mem);
[L3098]     $readmemh("golden_0.hex", golden_mem);
[L3099] end
[L3100] ```
[L3101] 
[L3102] ---
[L3103] 
[L3104] ## 2.2 在 RTL 里如何解释这些 8 bit 数据
[L3105] 
[L3106] ### 注意
[L3107] `$readmemh` 读进来的是 `reg [7:0]`，本身只是无符号位模式。  
[L3108] 如果你想让它参与有符号运算，需要做符号解释。
[L3109] 
[L3110] 例如：
[L3111] 
[L3112] ```verilog
[L3113] wire signed [7:0] w = weight_mem[addr];
[L3114] wire signed [7:0] x = input_mem[idx];
[L3115] wire signed [7:0] b = bias_mem[out_idx];
[L3116] ```
[L3117] 
[L3118] 或者：
[L3119] 
[L3120] ```verilog
[L3121] wire signed [7:0] w = $signed(weight_mem[addr]);
[L3122] ```
[L3123] 
[L3124] 这样：
[L3125] - `8'hff` 才会被解释成 `-1`
[L3126] - `8'h80` 才会被解释成 `-128`
[L3127] 
[L3128] ---
[L3129] 
[L3130] ## 2.3 第一层硬件如何读取 `weights.hex`
[L3131] 
[L3132] 你当前权重顺序是：
[L3133] 
[L3134] \[
[L3135] w[j][i]
[L3136] \]
[L3137] 
[L3138] 其中：
[L3139] - `j = 0..127`，输出神经元
[L3140] - `i = 0..783`，输入索引
[L3141] 
[L3142] 所以硬件最自然的读取方式是：
[L3143] 
[L3144] ### 地址公式
[L3145] \[
[L3146] addr = j \times 784 + i
[L3147] \]
[L3148] 
[L3149] ### 计算流程
[L3150] 对于每个输出神经元 `j`：
[L3151] 
[L3152] 1. 累加器清零
[L3153] 2. 从 `i=0` 到 `783`
[L3154] 3. 读取：
[L3155]    - `x = input_mem[i]`
[L3156]    - `w = weight_mem[j*784+i]`
[L3157] 4. 做乘加：
[L3158]    \[
[L3159]    acc_j += x \cdot w
[L3160]    \]
[L3161] 5. 循环结束后加偏置：
[L3162]    \[
[L3163]    acc_j += bias[j]
[L3164]    \]
[L3165] 6. 过 ReLU
[L3166] 7. 输出第 `j` 个结果
[L3167] 
[L3168] ### 伪代码
[L3169] ```verilog
[L3170] for (j = 0; j < 128; j = j + 1) begin
[L3171]     acc = 0;
[L3172]     for (i = 0; i < 784; i = i + 1) begin
[L3173]         acc = acc + $signed(input_mem[i]) * $signed(weight_mem[j*784+i]);
[L3174]     end
[L3175]     acc = acc + $signed(bias_mem[j]);
[L3176]     if (acc < 0) acc = 0;
[L3177]     out[j] = acc; // 或量化后输出
[L3178] end
[L3179] ```
[L3180] 
[L3181] ---
[L3182] 
[L3183] ## 2.4 `input_i.hex` 如何使用
[L3184] 
[L3185] ### 在 testbench 中
[L3186] 每次测试一个样本：
[L3187] 
[L3188] ```verilog
[L3189] $readmemh("input_0.hex", input_mem);
[L3190] ```
[L3191] 
[L3192] 然后把 `input_mem[0:783]` 依次送给你的计算模块。
[L3193] 
[L3194] ### 两种常见喂法
[L3195] #### 方式 1：一次性 preload 到 BRAM
[L3196] 如果你的设计是：
[L3197] - 输入先装入片上 RAM
[L3198] - 再启动计算
[L3199] 
[L3200] 那就：
[L3201] 1. 先把 `input_i.hex` 用 `$readmemh` 读入 testbench memory
[L3202] 2. 再驱动写接口，把 784 个数据写入 DUT 内部 input RAM
[L3203] 3. 拉高 `start`
[L3204] 
[L3205] #### 方式 2：流式输入
[L3206] 如果你的 DUT 是流式输入：
[L3207] - 每拍输入一个 `x_i`
[L3208] 
[L3209] 那就：
[L3210] 1. 从 `input_mem[0]` 到 `input_mem[783]` 依次送
[L3211] 2. 每个时钟给一个有效数据
[L3212] 3. 送满 784 个后等待结果
[L3213] 
[L3214] ---
[L3215] 
[L3216] ## 2.5 `golden_i.hex` 如何使用
[L3217] 
[L3218] 这是当前 testbench 最直接的对拍基准。
[L3219] 
[L3220] ### 用法
[L3221] 在 testbench 中：
[L3222] 
[L3223] ```verilog
[L3224] $readmemh("golden_0.hex", golden_mem);
[L3225] ```
[L3226] 
[L3227] 当 DUT 计算完成后，把 DUT 的 128 个输出与 `golden_mem` 逐项比对。
[L3228] 
[L3229] ### 比对方式
[L3230] ```verilog
[L3231] integer k;
[L3232] initial begin
[L3233]     for (k = 0; k < 128; k = k + 1) begin
[L3234]         if (dut_out[k] !== golden_mem[k]) begin
[L3235]             $display("Mismatch at %0d: dut=%h, golden=%h", k, dut_out[k], golden_mem[k]);
[L3236]         end
[L3237]     end
[L3238] end
[L3239] ```
[L3240] 
[L3241] ### 但要注意
[L3242] 你现在的 `golden_i.hex` 是：
[L3243] - 浮点输出再量化
[L3244] 
[L3245] 所以前提是你的硬件计算链路也得尽量模拟这个逻辑。  
[L3246] 如果你 RTL 里已经做成了严格整数推理，那这个 golden 就不再是最优基准了。
[L3247] 
[L3248] ---
[L3249] 
[L3250] # 3. 你当前这套文件更适合什么阶段
[L3251] 
[L3252] 这套导出结果，最适合：
[L3253] 
[L3254] ## 适合
[L3255] - 第一层 `fc1 + relu` 的硬件原型验证
[L3256] - BRAM 读写流程验证
[L3257] - 地址映射验证
[L3258] - 符号位解释验证
[L3259] - testbench 对拍基本通路
[L3260] 
[L3261] ## 不适合
[L3262] - 最终的完整 MNIST 分类演示
[L3263] - 严格的整数推理验证
[L3264] - 偏置 int32 链路验证
[L3265] - 静态量化部署闭环
[L3266] 
[L3267] ---
[L3268] 
[L3269] # 4. 推荐你怎么用这批文件做联调
[L3270] 
[L3271] ---
[L3272] 
[L3273] ## 阶段 1：先验证存储与读取
[L3274] 先只做三件事：
[L3275] 
[L3276] 1. `weights.hex` 是否正确加载
[L3277] 2. `input_i.hex` 是否正确加载
[L3278] 3. `golden_i.hex` 是否能和当前 RTL 输出逐项对上
[L3279] 
[L3280] 只要这一步过了，说明：
[L3281] - 文件格式没问题
[L3282] - 读写顺序没问题
[L3283] - 地址映射没问题
[L3284] 
[L3285] ---
[L3286] 
[L3287] ## 阶段 2：确认你的计算顺序与导出顺序一致
[L3288] 你需要确认 RTL 的权重寻址是不是：
[L3289] 
[L3290] \[
[L3291] addr = out\_idx \times 784 + in\_idx
[L3292] \]
[L3293] 
[L3294] 如果不是，就会出现：
[L3295] - 数据都能读到
[L3296] - 结果却完全不对
[L3297] 
[L3298] 这是最常见的问题。
[L3299] 
[L3300] ---
[L3301] 
[L3302] ## 阶段 3：确认 signed 解释
[L3303] 这是第二常见问题。  
[L3304] 如果 RTL 里没有把 `8'hff` 当成 `-1`，而是当成 `255`，结果一定炸。
[L3305] 
[L3306] 所以一定要检查：
[L3307] - 输入 signed
[L3308] - 权重 signed
[L3309] - 偏置 signed
[L3310] - 乘法 signed
[L3311] - 累加器 signed
[L3312] 
[L3313] ---
[L3314] 
[L3315] # 5. 我建议你现在这样理解每个文件
[L3316] 
[L3317] 你可以把当前文件理解成这张表：
[L3318] 
[L3319] | 文件 | 含义 | 数据类型 | 行数 | 作用 |
[L3320] |---|---|---:|---:|---|
[L3321] | `weights.hex` | `fc1` 权重 | int8 | 100352 | DUT 权重存储初始化 |
[L3322] | `bias.hex` | `fc1` 偏置 | int8 | 128 | 当前版本偏置初始化 |
[L3323] | `quant_params.npz` | 量化参数记录 | float/int | 少量字段 | 软件辅助，不是闭环核心 |
[L3324] | `input_i.hex` | 第 i 个输入图像 | int8 | 784 | 测试输入 |
[L3325] | `golden_i.hex` | 第 i 个参考输出 | int8 | 128 | testbench 对拍 |
[L3326] 
[L3327] ---
[L3328] 
[L3329] # 6. 你现在这批文件的“正确使用姿势”
[L3330] 
[L3331] 如果你此刻不改导出逻辑，而是先拿现有文件验证硬件，那建议这样：
[L3332] 
[L3333] ### 你现在的 DUT 目标
[L3334] 做一个：
[L3335] - `784 -> 128`
[L3336] - int8 输入
[L3337] - int8 权重
[L3338] - int8 偏置
[L3339] - 累加
[L3340] - ReLU
[L3341] - 最后输出和 `golden_i.hex` 尽量一致
[L3342] 
[L3343] ### testbench 流程
[L3344] 1. `$readmemh("weights.hex", weight_mem)`
[L3345] 2. `$readmemh("bias.hex", bias_mem)`
[L3346] 3. `$readmemh("input_0.hex", input_mem)`
[L3347] 4. `$readmemh("golden_0.hex", golden_mem)`
[L3348] 5. 把 `input_mem` 送入 DUT
[L3349] 6. DUT 输出 128 维结果
[L3350] 7. 与 `golden_mem` 逐项比较
[L3351] 
[L3352] 这样可以先把系统跑起来。
[L3353] 
[L3354] ---
[L3355] 
[L3356] # 7. 你后面升级到最终方案时，文件会怎么变
[L3357] 
[L3358] 你后面如果按我们前面定的最终方案升级，文件会改成更合理的结构：
[L3359] 
[L3360] - `fc1_weight_int8.hex`
[L3361] - `fc1_bias_int32.hex`
[L3362] - `fc2_weight_int8.hex`
[L3363] - `fc2_bias_int32.hex`
[L3364] - `input_i.hex`
[L3365] - `fc1_acc_i.hex`
[L3366] - `fc1_relu_i.hex`
[L3367] - `fc1_out_i.hex`
[L3368] - `fc2_acc_i.hex`
[L3369] - `logits_i.hex`
[L3370] - `pred_i.txt`
[L3371] - `quant_config.json`
[L3372] 
[L3373] 这时每个文件的职责会更清晰，硬件对拍也更容易定位问题。
[L3374] 
[L3375] ---
[L3376] 
[L3377] # 8. 你现在最需要注意的两个坑
[L3378] 
[L3379] ## 坑 1：`bias.hex` 现在是 int8
[L3380] 这不是最终可部署形式。  
[L3381] 所以你现在用它时，只能把它当作“当前软件逻辑下的配套文件”，不要把它当最终硬件部署格式。
[L3382] 
[L3383] ## 坑 2：`golden_i.hex` 不是整数推理 golden
[L3384] 它现在只是：
[L3385] - 浮点输出
[L3386] - 再量化
[L3387] 
[L3388] 所以如果你 RTL 做得特别“硬件真实”，反而可能和它对不上。
[L3389] 
[L3390] ---
[L3391] 
[L3392] 如果你把**已经导出的几个 `.hex` 文件内容**贴给我几行，或者把你当前 `quantized_data/` 目录再发我，我可以继续帮你做第二步：**逐个文件按实际内容检查格式是否正确，并给你写一个对应的 Verilog testbench 读取模板**。
[L3393] 
[L3394] 用户:
[L3395] bias.hex:000008e5
[L3396] 0000183f
[L3397] 000001a2
[L3398] ffffe6b8
[L3399] fffffa66
[L3400] fffffdd6
[L3401] 00000acd
[L3402] 00000e73
[L3403] fffff423
[L3404] 00000244
[L3405] fffff947
[L3406] 00000cc4
[L3407] fffff784
[L3408] 0000061c
[L3409] ffffed80
[L3410] 00000dd7
[L3411] fffffffe
[L3412] fffff9de
[L3413] fffffb86
[L3414] ffffed26
[L3415] 00000c70
[L3416] 0000116d
[L3417] ffffff7f
[L3418] 000001a8
[L3419] 0000049e
[L3420] 0000118f
[L3421] 000006c6
[L3422] 00000e32
[L3423] 000025f7
[L3424] ffffe677
[L3425] 000005b3
[L3426] 00000b83
[L3427] 00002162
[L3428] 00001ce9
[L3429] 00000409
[L3430] fffffbcc
[L3431] 00000487
[L3432] 00000caa
[L3433] fffff5f9
[L3434] fffff633
[L3435] 000003b1
[L3436] 00000fd1
[L3437] 000005d1
[L3438] 00000ce3
[L3439] 0000106f
[L3440] 00000c5a
[L3441] 0000019f
[L3442] 00000681
[L3443] ffffef2b
[L3444] fffff49a
[L3445] ffffe4d9
[L3446] fffffb73
[L3447] 0000019f
[L3448] 00000dd1
[L3449] 0000016b
[L3450] 00000730
[L3451] fffff9a5
[L3452] 000009db
[L3453] 000008ec
[L3454] 000003c1
[L3455] 00000b59
[L3456] 00001263
[L3457] fffff71a
[L3458] 00001706
[L3459] 000013f9
[L3460] 000001e7
[L3461] 000027ca
[L3462] 00000779
[L3463] 000000d3
[L3464] fffffe21
[L3465] 0000051c
[L3466] 00000e9c
[L3467] 00000c2f
[L3468] fffffc59
[L3469] 0000142d
[L3470] 00000956
[L3471] 00000813
[L3472] fffffa40
[L3473] 00000d2a
[L3474] 00000c81
[L3475] fffffd14
[L3476] 000008ee
[L3477] 0000144a
[L3478] fffff816
[L3479] 00000be6
[L3480] 00000850
[L3481] 000012d6
[L3482] 00001bd6
[L3483] 00000388
[L3484] fffffc00
[L3485] 00001590
[L3486] 00000565
[L3487] 00000adb
[L3488] 00000f82
[L3489] 00000242
[L3490] 00001ed3
[L3491] fffff3bf
[L3492] 00001420
[L3493] 00000b0e
[L3494] 00000ac2
[L3495] fffffefd
[L3496] 00000072
[L3497] fffff4f2
[L3498] fffffc4f
[L3499] 00000904
[L3500] 00000a0f
[L3501] 0000078d
[L3502] 00000cba
[L3503] fffffb5e
[L3504] fffffe7a
[L3505] 00000a64
[L3506] 00000e22
[L3507] 00000435
[L3508] 00000642
[L3509] 00002492
[L3510] 00002235
[L3511] 000011d9
[L3512] fffff921
[L3513] ffffff6a
[L3514] 00001402
[L3515] 00001a36
[L3516] 0000002e
[L3517] 00001f8d
[L3518] ffffff3e
[L3519] fffff936
[L3520] 00000023
[L3521] fffffe93
[L3522] fffff438
[L3523] weight.hex:aa
[L3524] e5
[L3525] 00
[L3526] 0d
[L3527] 0d
[L3528] f7
[L3529] 08
[L3530] e8
[L3531] ff
[L3532] 20
[L3533] 15
[L3534] c5
[L3535] fe
[L3536] fb
[L3537] e2
[L3538] f9
[L3539] 16
[L3540] 15
[L3541] fc
[L3542] 15
[L3543] 09
[L3544] 13
[L3545] 09
[L3546] cb
[L3547] 28
[L3548] ca
[L3549] 0f
[L3550] 15
[L3551] 0f
[L3552] 05
[L3553] 16
[L3554] f3
[L3555] d7
[L3556] f6
[L3557] 06
[L3558] 0c
[L3559] d8
[L3560] 1e
[L3561] ef
[L3562] 03
[L3563] 12
[L3564] 04
[L3565] e6
[L3566] dd
[L3567] d3
[L3568] 0d
[L3569] fa
[L3570] 1c
[L3571] 1c
[L3572] 0f
[L3573] 0d
[L3574] fd
[L3575] 10
[L3576] 04
[L3577] 01
[L3578] b7
[L3579] 0d
[L3580] 00
[L3581] 08
[L3582] 15
[L3583] d2
[L3584] e8
[L3585] 0f
[L3586] f3
[L3587] bd
[L3588] 19
[L3589] e2
[L3590] 1a
[L3591] f1
[L3592] e2
[L3593] f5
[L3594] f2
[L3595] c6
[L3596] ff
[L3597] ec
[L3598] 02
[L3599] d8
[L3600] e9
[L3601] fe
[L3602] 0f
[L3603] 16
[L3604] d8
[L3605] d6
[L3606] 25
[L3607] e2
[L3608] 06
[L3609] bd
[L3610] 24
[L3611] 0d
[L3612] 1c
[L3613] c2
[L3614] 1c
[L3615] ee
[L3616] e0
[L3617] e7
[L3618] ff
[L3619] 05
[L3620] 01
[L3621] 0f
[L3622] ff
[L3623] fa
[L3624] 16
[L3625] 09
[L3626] 08
[L3627] df
[L3628] ff
[L3629] 12
[L3630] d2
[L3631] ea
[L3632] 0f
[L3633] b4
[L3634] 0f
[L3635] fb
[L3636] f4
[L3637] 24
[L3638] 81
[L3639] 13
[L3640] f1
[L3641] 17
[L3642] 0a
[L3643] ed
[L3644] 03
[L3645] f7
[L3646] ed
[L3647] c0
[L3648] e4
[L3649] da
[L3650] 19
[L3651] fb
[L3652] fd
[L3653] fa
[L3654] bf
[L3655] 14
[L3656] 15
[L3657] 17
[L3658] e1
[L3659] df
[L3660] f6
[L3661] f0
[L3662] 1c
[L3663] 10
[L3664] 0c
[L3665] e9
[L3666] cf
[L3667] e0
[L3668] 25
[L3669] eb
[L3670] e2
[L3671] 1e
[L3672] 13
[L3673] e9
[L3674] e9
[L3675] 18
[L3676] 0b
[L3677] dd
[L3678] eb
[L3679] fe
[L3680] ee
[L3681] 10
[L3682] 14
[L3683] 0c
[L3684] 23
[L3685] f1
[L3686] e9
[L3687] d9
[L3688] c1
[L3689] fa
[L3690] fd
[L3691] 16
[L3692] c9
[L3693] 2b
[L3694] 13
[L3695] 10
[L3696] d6
[L3697] 00
[L3698] ea
[L3699] ce
[L3700] e4
[L3701] e1
[L3702] d0
[L3703] ec
[L3704] 0e
[L3705] 0f
[L3706] 13
[L3707] f6
[L3708] f6
[L3709] 12
[L3710] 04
[L3711] 08
[L3712] 1b
[L3713] 0f
[L3714] 0b
[L3715] 02
[L3716] ea
[L3717] d1
[L3718] da
[L3719] e3
[L3720] e4
[L3721] dc
[L3722] 1b
[L3723] d0
[L3724] 18
[L3725] f0
[L3726] 02
[L3727] 48
[L3728] 06
[L3729] 20
[L3730] 2a
[L3731] e4
[L3732] 1b
[L3733] 0e
[L3734] cb
[L3735] 14
[L3736] df
[L3737] 3a
[L3738] 47
[L3739] 0e
[L3740] 2c
[L3741] 02
[L3742] e6
[L3743] 01
[L3744] 21
[L3745] 00
[L3746] f3
[L3747] e4
[L3748] cd
[L3749] c9
[L3750] d5
[L3751] df
[L3752] f1
[L3753] cd
[L3754] ed
[L3755] 18
[L3756] 11
[L3757] 0e
[L3758] 34
[L3759] 16
[L3760] f1
[L3761] 14
[L3762] db
[L3763] fd
[L3764] 0e
[L3765] 34
[L3766] 30
[L3767] df
[L3768] ec
[L3769] de
[L3770] f2
[L3771] cf
[L3772] e1
[L3773] c0
[L3774] fd
[L3775] d6
[L3776] 10
[L3777] 4f
[L3778] 1c
[L3779] 03
[L3780] 25
[L3781] 05
[L3782] fa
[L3783] fd
[L3784] 19
[L3785] 13
[L3786] cb
[L3787] 08
[L3788] 20
[L3789] 03
[L3790] 39
[L3791] 0b
[L3792] 1e
[L3793] ea
[L3794] f3
[L3795] 0b
[L3796] 21
[L3797] ee
[L3798] 16
[L3799] 03
[L3800] af
[L3801] 14
[L3802] ae
[L3803] 07
[L3804] ff
[L3805] f4
[L3806] 0c
[L3807] aa
[L3808] 09
[L3809] e6
[L3810] e9
[L3811] d2
[L3812] 24
[L3813] e8
[L3814] 16
[L3815] 0d
[L3816] f8
[L3817] 14
[L3818] 28
[L3819] fb
[L3820] df
[L3821] 9c
[L3822] 05
[L3823] 18
[L3824] f0
[L3825] 03
[L3826] 45
[L3827] 1e
[L3828] f8
[L3829] 0a
[L3830] fc
[L3831] 04
[L3832] 32
[L3833] 26
[L3834] 0c
[L3835] 1b
[L3836] e6
[L3837] 1a
[L3838] 20
[L3839] ff
[L3840] d9
[L3841] 19
[L3842] b8
[L3843] 12
[L3844] 1b
[L3845] 13
[L3846] 11
[L3847] e8
[L3848] 0b
[L3849] e1
[L3850] 26
[L3851] d4
[L3852] 11
[L3853] e2
[L3854] 92
[L3855] e0
[L3856] fb
[L3857] f5
[L3858] 15
[L3859] 1c
[L3860] 23
[L3861] 15
[L3862] f7
[L3863] 09
[L3864] e4
[L3865] 13
[L3866] dc
[L3867] f9
[L3868] fe
[L3869] f2
[L3870] 14
[L3871] 0a
[L3872] cc
[L3873] eb
[L3874] 02
[L3875] 07
[L3876] fa
[L3877] d1
[L3878] dc
[L3879] 1e
[L3880] 13
[L3881] ec
[L3882] 10
[L3883] 1e
[L3884] 00
[L3885] 1f
[L3886] 13
[L3887] 1a
[L3888] db
[L3889] 16
[L3890] 08
[L3891] 03
[L3892] 0e
[L3893] fe
[L3894] e0
[L3895] 37
[L3896] 0c
[L3897] f0
[L3898] 1a
[L3899] 0e
[L3900] 10
[L3901] 36
[L3902] d2
[L3903] 07
[L3904] fc
[L3905] bb
[L3906] d4
[L3907] db
[L3908] f7
[L3909] 00
[L3910] 0f
[L3911] 0b
[L3912] 0f
[L3913] da
[L3914] d7
[L3915] 08
[L3916] 06
[L3917] 0b
[L3918] 05
[L3919] ff
[L3920] 16
[L3921] 34
[L3922] 02
[L3923] e1
[L3924] 0d
[L3925] 1a
[L3926] 0f
[L3927] 9a
[L3928] dc
[L3929] e8
[L3930] 1e
[L3931] dd
[L3932] 0f
[L3933] f0
[L3934] d8
[L3935] f5
[L3936] 0f
[L3937] f0
[L3938] 05
[L3939] 95
[L3940] c1
[L3941] ec
[L3942] 19
[L3943] 35
[L3944] eb
[L3945] 08
[L3946] 2d
[L3947] 2e
[L3948] db
[L3949] 28
[L3950] fc
[L3951] dd
[L3952] f0
[L3953] f2
[L3954] 12
[L3955] 1a
[L3956] fd
[L3957] 18
[L3958] fa
[L3959] ec
[L3960] cb
[L3961] f7
[L3962] 0a
[L3963] f0
[L3964] 10
[L3965] c4
[L3966] 0a
[L3967] 14
[L3968] e1
[L3969] 0c
[L3970] 03
[L3971] 06
[L3972] 07
[L3973] e9
[L3974] 04
[L3975] 12
[L3976] 0f
[L3977] 08
[L3978] c5
[L3979] 0d
[L3980] fb
[L3981] d1
[L3982] 21
[L3983] dd
[L3984] 13
[L3985] d9
[L3986] ee
[L3987] 0d
[L3988] 09
[L3989] fb
[L3990] 02
[L3991] 00
[L3992] ff
[L3993] e4
[L3994] d3
[L3995] 14
[L3996] f3
[L3997] ce
[L3998] c6
[L3999] 10
[L4000] 1f
[L4001] 13
[L4002] 0d
[L4003] 05
[L4004] 1d
[L4005] d3
[L4006] 22
[L4007] 2b
[L4008] fb
[L4009] 15
[L4010] 07
[L4011] cf
[L4012] 09
[L4013] f3
[L4014] 26
[L4015] 13
[L4016] bd
[L4017] 04
[L4018] e6
[L4019] f6
[L4020] 11
[L4021] f1
[L4022] 46
[L4023] ef
[L4024] 0a
[L4025] f9
[L4026] e6
[L4027] cb
[L4028] 0c
[L4029] eb
[L4030] 1d
[L4031] 16
[L4032] 08
[L4033] 0e
[L4034] 2c
[L4035] e8
[L4036] 08
[L4037] fa
[L4038] ee
[L4039] d2
[L4040] dd
[L4041] 19
[L4042] 1d
[L4043] ce
[L4044] 9c
[L4045] fa
[L4046] a2
[L4047] db
[L4048] 0c
[L4049] c8
[L4050] f5
[L4051] d5
[L4052] ca
[L4053] ea
[L4054] ed
[L4055] 1e
[L4056] 22
[L4057] 19
[L4058] 1f
[L4059] f0
[L4060] 0b
[L4061] 04
[L4062] 07
[L4063] de
[L4064] 04
[L4065] 00
[L4066] 09
[L4067] fc
[L4068] 17
[L4069] 09
[L4070] ec
[L4071] 12
[L4072] c4
[L4073] 8a
[L4074] 0e
[L4075] ec
[L4076] 04
[L4077] 0d
[L4078] e7
[L4079] 12
[L4080] 15
[L4081] 0b
[L4082] 8f
[L4083] d2
[L4084] e7
[L4085] c2
[L4086] 0f
[L4087] f7
[L4088] 12
[L4089] dc
[L4090] 0d
[L4091] e1
[L4092] 18
[L4093] 07
[L4094] d4
[L4095] 08
[L4096] 28
[L4097] ab
[L4098] 0a
[L4099] b4
[L4100] 1c
[L4101] 0b
[L4102] 05
[L4103] 02
[L4104] 0b
[L4105] 11
[L4106] 22
[L4107] 02
[L4108] e9
[L4109] 14
[L4110] e9
[L4111] a7
[L4112] 15
[L4113] bb
[L4114] 08
[L4115] e1
[L4116] fd
[L4117] 00
[L4118] e4
[L4119] f0
[L4120] 0d
[L4121] 23
[L4122] 29
[L4123] ba
[L4124] 2b
[L4125] 09
[L4126] 0c
[L4127] 13
[L4128] 47
[L4129] ff
[L4130] 0a
[L4131] 14
[L4132] 0e
[L4133] 1a
[L4134] f8
[L4135] 97
[L4136] d4
[L4137] e0
[L4138] eb
[L4139] 0a
[L4140] 0f
[L4141] ed
[L4142] 3f
[L4143] 18
[L4144] 15
[L4145] a5
[L4146] 10
[L4147] 07
[L4148] 13
[L4149] 2d
[L4150] 0f
[L4151] 12
[L4152] f6
[L4153] 0d
[L4154] 09
[L4155] 0f
[L4156] 05
[L4157] 08
[L4158] e9
[L4159] 14
[L4160] 11
[L4161] f1
[L4162] bb
[L4163] 38
[L4164] 1f
[L4165] ff
[L4166] 02
[L4167] 05
[L4168] ff
[L4169] e2
[L4170] 38
[L4171] fd
[L4172] 0b
[L4173] 0a
[L4174] 3a
[L4175] ec
[L4176] b2
[L4177] 2d
[L4178] f0
[L4179] db
[L4180] cf
[L4181] ed
[L4182] eb
[L4183] c4
[L4184] 05
[L4185] df
[L4186] 03
[L4187] a5
[L4188] 02
[L4189] 02
[L4190] 12
[L4191] 46
[L4192] 00
[L4193] 0b
[L4194] ee
[L4195] 22
[L4196] e7
[L4197] f0
[L4198] 14
[L4199] ca
[L4200] 0d
[L4201] 2e
[L4202] f1
[L4203] ba
[L4204] 05
[L4205] 2c
[L4206] 0c
[L4207] ec
[L4208] 0d
[L4209] fa
[L4210] 03
[L4211] fb
[L4212] 0b
[L4213] f4
[L4214] e7
[L4215] 12
[L4216] 98
[L4217] c2
[L4218] 08
[L4219] d2
[L4220] 04
[L4221] f3
[L4222] 1b
[L4223] 09
[L4224] f8
[L4225] f9
[L4226] f2
[L4227] 28
[L4228] f9
[L4229] 30
[L4230] e9
[L4231] 06
[L4232] e5
[L4233] f6
[L4234] c2
[L4235] 13
[L4236] f5
[L4237] 14
[L4238] 21
[L4239] 50
[L4240] c4
[L4241] 07
[L4242] e2
[L4243] f3
[L4244] 0d
[L4245] fa
[L4246] f3
[L4247] 0c
[L4248] 0c
[L4249] 08
[L4250] 15
[L4251] 10
[L4252] e9
[L4253] 10
[L4254] 0b
[L4255] 0e
[L4256] 01
[L4257] eb
[L4258] 12
[L4259] ed
[L4260] 0b
[L4261] 0f
[L4262] 10
[L4263] d0
[L4264] 0c
[L4265] 1a
[L4266] fb
[L4267] ca
[L4268] cf
[L4269] f3
[L4270] cb
[L4271] bd
[L4272] ff
[L4273] 36
[L4274] f6
[L4275] fe
[L4276] f8
[L4277] f2
[L4278] 42
[L4279] 02
[L4280] f8
[L4281] 06
[L4282] 12
[L4283] 1d
[L4284] ce
[L4285] 14
[L4286] 0c
[L4287] 17
[L4288] fb
[L4289] 0e
[L4290] ed
[L4291] 17
[L4292] 15
[L4293] 0c
[L4294] 15
[L4295] 02
[L4296] e6
[L4297] 0f
[L4298] 10
[L4299] 06
[L4300] 02
[L4301] 0c
[L4302] b4
[L4303] 06
[L4304] d7
[L4305] af
[L4306] 2f
[L4307] de
[L4308] e1
[L4309] 40
[L4310] ee
[L4311] ef
[L4312] f3
[L4313] 0c
[L4314] aa
[L4315] 21
[L4316] 0a
[L4317] c6
[L4318] fe
[L4319] f3
[L4320] ff
[L4321] 1f
[L4322] e6
[L4323] 3e
[L4324] 08
[L4325] f8
[L4326] d0
[L4327] d8
[L4328] f3
[L4329] 26
[L4330] 06
[L4331] 19
[L4332] 08
[L4333] fc
[L4334] fe
[L4335] 18
[L4336] eb
[L4337] f5
[L4338] d2
[L4339] d5
[L4340] f4
[L4341] e3
[L4342] 18
[L4343] 10
[L4344] f7
[L4345] ab
[L4346] d3
[L4347] f7
[L4348] ed
[L4349] 16
[L4350] c1
[L4351] 04
[L4352] df
[L4353] f6
[L4354] e9
[L4355] 18
[L4356] 11
[L4357] a9
[L4358] da
[L4359] d3
[L4360] f0
[L4361] 04
[L4362] 15
[L4363] d7
[L4364] 05
[L4365] 04
[L4366] 17
[L4367] 17
[L4368] bf
[L4369] 23
[L4370] ee
[L4371] db
[L4372] ec
[L4373] ff
[L4374] 0f
[L4375] df
[L4376] ea
[L4377] fb
[L4378] 1d
[L4379] ef
[L4380] 08
[L4381] ef
[L4382] e2
[L4383] f0
[L4384] 0b
[L4385] 13
[L4386] e4
[L4387] 0e
[L4388] 12
[L4389] 18
[L4390] ff
[L4391] e1
[L4392] 0e
[L4393] fb
[L4394] 10
[L4395] 05
[L4396] b3
[L4397] 16
[L4398] 08
[L4399] dd
[L4400] 2e
[L4401] e0
[L4402] 12
[L4403] fa
[L4404] ee
[L4405] 17
[L4406] ab
[L4407] 16
[L4408] 0a
[L4409] 0b
[L4410] 0f
[L4411] 15
[L4412] e6
[L4413] 23
[L4414] e4
[L4415] cc
[L4416] 17
[L4417] 16
[L4418] 11
[L4419] 0f
[L4420] a8
[L4421] ff
[L4422] c9
[L4423] e8
[L4424] 2b
[L4425] d4
[L4426] 04
[L4427] f7
[L4428] 07
[L4429] e4
[L4430] 1f
[L4431] eb
[L4432] 01
[L4433] 9e
[L4434] 14
[L4435] 1a
[L4436] 01
[L4437] 11
[L4438] 13
[L4439] 25
[L4440] 03
[L4441] 04
[L4442] f1
[L4443] db
[L4444] 09
[L4445] 05
[L4446] db
[L4447] f2
[L4448] d9
[L4449] dc
[L4450] 1a
[L4451] cc
[L4452] 22
[L4453] 14
[L4454] 09
[L4455] 03
[L4456] 08
[L4457] 0b
[L4458] 01
[L4459] 09
[L4460] 0f
[L4461] d5
[L4462] c9
[L4463] d0
[L4464] f2
[L4465] f7
[L4466] 28
[L4467] 08
[L4468] e6
[L4469] 0a
[L4470] e2
[L4471] d2
[L4472] 18
[L4473] 1d
[L4474] 14
[L4475] 00
[L4476] 09
[L4477] 08
[L4478] f9
[L4479] c3
[L4480] e4
[L4481] f7
[L4482] 17
[L4483] e8
[L4484] cd
[L4485] 18
[L4486] 0b
[L4487] 10
[L4488] 0f
[L4489] 0e
[L4490] 14
[L4491] fc
[L4492] 1a
[L4493] 18
[L4494] 09
[L4495] 14
[L4496] 15
[L4497] 09
[L4498] 12
[L4499] 1c
[L4500] da
[L4501] f9
[L4502] d8
[L4503] e0
[L4504] 07
[L4505] f6
[L4506] 0f
[L4507] 14
[L4508] b2
[L4509] e1
[L4510] 13
[L4511] c3
[L4512] e9
[L4513] 05
[L4514] 3a
[L4515] ed
[L4516] 1f
[L4517] f3
[L4518] 20
[L4519] 27
[L4520] 0a
[L4521] c8
[L4522] d8
[L4523] 32
[L4524] 19
[L4525] 27
[L4526] 25
[L4527] dc
[L4528] e2
[L4529] 20
[L4530] ff
[L4531] 04
[L4532] e9
[L4533] 50
[L4534] 00
[L4535] de
[L4536] db
[L4537] ec
[L4538] d6
[L4539] d5
[L4540] 14
[L4541] 0b
[L4542] 06
[L4543] ea
[L4544] d8
[L4545] 21
[L4546] ff
[L4547] f1
[L4548] 1b
[L4549] f3
[L4550] 0b
[L4551] 15
[L4552] 0d
[L4553] 0f
[L4554] 24
[L4555] 01
[L4556] fd
[L4557] 0f
[L4558] e2
[L4559] 13
[L4560] e4
[L4561] ad
[L4562] 02
[L4563] 14
[L4564] 05
[L4565] ce
[L4566] f0
[L4567] e1
[L4568] fe
[L4569] 0f
[L4570] ed
[L4571] 0d
[L4572] 16
[L4573] 05
[L4574] 17
[L4575] cf
[L4576] 14
[L4577] 15
[L4578] 13
[L4579] df
[L4580] c4
[L4581] 10
[L4582] ec
[L4583] 05
[L4584] 14
[L4585] 01
[L4586] b7
[L4587] dd
[L4588] 10
[L4589] c5
[L4590] 0b
[L4591] 02
[L4592] 12
[L4593] 06
[L4594] d9
[L4595] f8
[L4596] 16
[L4597] 14
[L4598] 0b
[L4599] 04
[L4600] 02
[L4601] e5
[L4602] 10
[L4603] 0b
[L4604] 16
[L4605] cc
[L4606] df
[L4607] 17
[L4608] 10
[L4609] 12
[L4610] 0d
[L4611] 0d
[L4612] d4
[L4613] e6
[L4614] cd
[L4615] 10
[L4616] 0c
[L4617] 13
[L4618] b7
[L4619] 0b
[L4620] 01
[L4621] 18
[L4622] fc
[L4623] bb
[L4624] 03
[L4625] 10
[L4626] ce
[L4627] fc
[L4628] 15
[L4629] 18
[L4630] fd
[L4631] 11
[L4632] 0a
[L4633] 14
[L4634] b5
[L4635] 07
[L4636] da
[L4637] 1d
[L4638] db
[L4639] 17
[L4640] ec
[L4641] 08
[L4642] f1
[L4643] 0d
[L4644] ef
[L4645] 03
[L4646] c2
[L4647] ec
[L4648] ee
[L4649] 16
[L4650] 13
[L4651] 14
[L4652] 08
[L4653] ce
[L4654] b3
[L4655] 01
[L4656] dc
[L4657] ee
[L4658] 01
[L4659] f5
[L4660] 12
[L4661] ba
[L4662] 86
[L4663] d5
[L4664] 0b
[L4665] 18
[L4666] fe
[L4667] 22
[L4668] 0b
[L4669] f9
[L4670] 04
[L4671] 15
[L4672] 17
[L4673] e6
[L4674] 01
[L4675] 27
[L4676] ea
[L4677] fc
[L4678] 17
[L4679] ec
[L4680] 99
[L4681] 03
[L4682] ae
[L4683] e0
[L4684] e7
[L4685] ea
[L4686] c3
[L4687] 01
[L4688] 12
[L4689] 34
[L4690] 10
[L4691] 0b
[L4692] d7
[L4693] f2
[L4694] 14
[L4695] 14
[L4696] f3
[L4697] 10
[L4698] 1d
[L4699] 1d
[L4700] ef
[L4701] 16
[L4702] ff
[L4703] 26
[L4704] 0d
[L4705] fa
[L4706] 03
[L4707] 1c
[L4708] f6
[L4709] 12
[L4710] e3
[L4711] 18
[L4712] 11
[L4713] ba
[L4714] d8
[L4715] ef
[L4716] f5
[L4717] 2a
[L4718] fc
[L4719] 11
[L4720] 15
[L4721] 02
[L4722] 16
[L4723] 17
[L4724] 06
[L4725] 13
[L4726] 0b
[L4727] 08
[L4728] fb
[L4729] 14
[L4730] fa
[L4731] 20
[L4732] fe
[L4733] 21
[L4734] 01
[L4735] 0b
[L4736] 20
[L4737] f7
[L4738] 14
[L4739] ec
[L4740] c8
[L4741] 08
[L4742] 16
[L4743] fe
[L4744] 0a
[L4745] 10
[L4746] 1e
[L4747] 09
[L4748] aa
[L4749] 05
[L4750] ec
[L4751] e6
[L4752] fd
[L4753] 04
[L4754] 1b
[L4755] d6
[L4756] bd
[L4757] eb
[L4758] 16
[L4759] 16
[L4760] 16
[L4761] cc
[L4762] 25
[L4763] e7
[L4764] eb
[L4765] 07
[L4766] 17
[L4767] f2
[L4768] ab
[L4769] 10
[L4770] c9
[L4771] f9
[L4772] d5
[L4773] 06
[L4774] ff
[L4775] 16
[L4776] 00
[L4777] 0f
[L4778] ef
[L4779] b6
[L4780] 0d
[L4781] b6
[L4782] ae
[L4783] 03
[L4784] 10
[L4785] e0
[L4786] eb
[L4787] f3
[L4788] e6
[L4789] 9c
[L4790] 09
[L4791] bd
[L4792] 0d
[L4793] e9
[L4794] d2
[L4795] dc
[L4796] 14
[L4797] 97
[L4798] 11
[L4799] 0f
[L4800] 0c
[L4801] 0b
[L4802] 11 先检查这两个有没有问题，然后根据现在的文件和坑，你看看代码有什么需要改进的，最好生成的数据可以被fpga上得cim系统直接使用
[L4803] AI助手:
[L4804] 
[L4805] 
[L4806] AI助手:
[L4807] 我先按“文件内容本身是否自洽”和“是否适合 FPGA/CIM 直接吃”两层来判断。眼下一个明显信号是：你贴出来的 `bias.hex` 已经是 **32 位有符号十六进制**，这比你最初 notebook 里那个 int8 bias 更合理，说明你的导出逻辑应该已经改过一版了。
[L4808] 
[L4809] AI助手:
[L4810] 先说结论：
[L4811] 
[L4812] **你现在贴出来的这两个文件，格式上基本是对的，而且已经比最初 notebook 版本更适合 FPGA/CIM 直接使用。**
[L4813] 
[L4814] 尤其是：
[L4815] 
[L4816] - `bias.hex`：现在是**32 位有符号补码**，这很合理，适合 MAC 后加 bias
[L4817] - `weight.hex`：现在是**8 位有符号补码**，也合理，适合 int8 权重阵列
[L4818] 
[L4819] 这说明你的导出代码已经朝着“真实整数推理链路”改进过了，不再是最初那个 bias 只有 2 位 hex 的版本。
[L4820] 
[L4821] 下面我分三部分说：
[L4822] 
[L4823] 1. 先检查这两个文件有没有明显问题  
[L4824] 2. 结合现在的文件格式，说明 FPGA/CIM 应该怎么直接使用  
[L4825] 3. 按“还能怎么改代码”给你一套更适合直接上板的改进建议
[L4826] 
[L4827] ---
[L4828] 
[L4829] # 1. 先检查这两个文件有没有问题
[L4830] 
[L4831] ---
[L4832] 
[L4833] ## 1.1 `bias.hex` 检查结果
[L4834] 
[L4835] 你给的 `bias.hex` 形如：
[L4836] 
[L4837] ```text
[L4838] 000008e5
[L4839] 0000183f
[L4840] 000001a2
[L4841] ffffe6b8
[L4842] ...
[L4843] ```
[L4844] 
[L4845] ### 这说明什么
[L4846] 每行都是 **8 个十六进制字符**，也就是 **32 bit**。
[L4847] 
[L4848] 这对应的是：
[L4849] 
[L4850] - 正数：直接写成 `000008e5`
[L4851] - 负数：写成二补码形式，比如 `ffffe6b8`
[L4852] 
[L4853] 这正是 FPGA 里最常见的 **signed int32 bias 文件格式**。
[L4854] 
[L4855] ### 这和硬件链路匹配吗
[L4856] 匹配，而且是更合理的做法：
[L4857] 
[L4858] - 输入：int8
[L4859] - 权重：int8
[L4860] - 累加：int32
[L4861] - 偏置：int32
[L4862] - ReLU / requant：后续再做
[L4863] 
[L4864] 所以从工程角度说，**这个 bias 格式是对的**。
[L4865] 
[L4866] ### 我看出来的一个信号
[L4867] 你这份 `bias.hex` 已经不是最开始 notebook 里那种：
[L4868] 
[L4869] ```python
[L4870] f.write(f'{val.item() & 0xFF:02x}\n')
[L4871] ```
[L4872] 
[L4873] 而是明显变成了类似：
[L4874] 
[L4875] ```python
[L4876] f.write(f'{val.item() & 0xFFFFFFFF:08x}\n')
[L4877] ```
[L4878] 
[L4879] 这说明你已经把 bias 导出升级成 int32 了。这个改动是正确的。
[L4880] 
[L4881] ### 需要确认的一点
[L4882] 如果这是 `fc1` 的 bias，那么它应该有：
[L4883] 
[L4884] - **128 行**
[L4885] 
[L4886] 如果这是 `fc2` 的 bias，那么应该有：
[L4887] 
[L4888] - **10 行**
[L4889] 
[L4890] 你贴出来看起来更像是 128 行中的一整段。  
[L4891] 如果你最终网络是两层 MLP，那建议不要只叫 `bias.hex`，而是明确改成：
[L4892] 
[L4893] - `fc1_bias_int32.hex`
[L4894] - `fc2_bias_int32.hex`
[L4895] 
[L4896] 这样不会混。
[L4897] 
[L4898] ---
[L4899] 
[L4900] ## 1.2 `weight.hex` 检查结果
[L4901] 
[L4902] 你给的 `weight.hex` 形如：
[L4903] 
[L4904] ```text
[L4905] aa
[L4906] e5
[L4907] 00
[L4908] 0d
[L4909] 0d
[L4910] f7
[L4911] ...
[L4912] ```
[L4913] 
[L4914] ### 这说明什么
[L4915] 每行都是 **2 个十六进制字符**，也就是 **8 bit**。
[L4916] 
[L4917] 这对应：
[L4918] 
[L4919] - `aa` = -86
[L4920] - `e5` = -27
[L4921] - `00` = 0
[L4922] - `0d` = 13
[L4923] - `f7` = -9
[L4924] 
[L4925] 这正是标准的 **signed int8 权重存储格式**。
[L4926] 
[L4927] ### 是否合理
[L4928] 合理，前提是 RTL 里按 signed 解释：
[L4929] 
[L4930] ```verilog
[L4931] wire signed [7:0] w = $signed(weight_mem[addr]);
[L4932] ```
[L4933] 
[L4934] ### 数值分布上看有没有异常
[L4935] 从你贴的内容看，数值大多集中在：
[L4936] 
[L4937] - `0xC0 ~ 0x40` 左右对应的 -64 到 64 之间
[L4938] - 偶尔有 `0x81`、`0x9c`、`0x8a`、`0x50`、`0x4f`
[L4939] 
[L4940] 这很像正常量化后的神经网络权重分布，不像导出炸掉了。
[L4941] 
[L4942] 也就是说：
[L4943] 
[L4944] - **没有明显出现全 00 / 全 ff / 大面积饱和到 7f/80 的异常**
[L4945] - 看起来像一份正常的量化权重
[L4946] 
[L4947] ### 需要确认的一点
[L4948] 如果这是 `fc1.weight`，而且第一层 shape 是 `[128, 784]`，那它总行数应该是：
[L4949] 
[L4950] \[
[L4951] 128 \times 784 = 100352
[L4952] \]
[L4953] 
[L4954] 如果最终网络是两层 MLP，你也应该拆成：
[L4955] 
[L4956] - `fc1_weight_int8.hex`
[L4957] - `fc2_weight_int8.hex`
[L4958] 
[L4959] 否则后面上板会混。
[L4960] 
[L4961] ---
[L4962] 
[L4963] # 2. 结合你现在的文件格式，FPGA/CIM 怎么直接使用
[L4964] 
[L4965] 现在按你贴出来的格式，**最适合的硬件链路**是：
[L4966] 
[L4967] - `weight.hex`：int8 权重阵列
[L4968] - `bias.hex`：int32 偏置
[L4969] - `input_i.hex`：int8 输入
[L4970] - 累加器：int32
[L4971] - 后面做 ReLU、requant、第二层
[L4972] 
[L4973] 也就是说，你现在已经从“玩具导出格式”走到了“能上硬件”的方向。
[L4974] 
[L4975] ---
[L4976] 
[L4977] ## 2.1 `weight.hex` 的硬件读取方式
[L4978] 
[L4979] 如果你的导出顺序还是之前那样：
[L4980] 
[L4981] ```python
[L4982] for row in q_weight:
[L4983]     for val in row:
[L4984]         ...
[L4985] ```
[L4986] 
[L4987] 那么权重布局就是：
[L4988] 
[L4989] \[
[L4990] W[j][i]
[L4991] \]
[L4992] 
[L4993] 也就是：
[L4994] 
[L4995] - `j`：输出神经元索引
[L4996] - `i`：输入通道索引
[L4997] 
[L4998] ### 地址公式
[L4999] 若第一层 `fc1` shape 是 `[128, 784]`，则：
[L5000] 
[L5001] \[
[L5002] addr = j \times 784 + i
[L5003] \]
[L5004] 
[L5005] RTL 读法就是：
[L5006] 
[L5007] - 外层循环跑输出通道 `j`
[L5008] - 内层循环跑输入通道 `i`
[L5009] 
[L5010] ### 对应计算
[L5011] \[
[L5012] acc_j = \sum_{i=0}^{783} (x_i - z_x)(w_{j,i} - z_w) + b_j
[L5013] \]
[L5014] 
[L5015] 如果你现在是权重对称量化，那通常：
[L5016] 
[L5017] - `z_w = 0`
[L5018] 
[L5019] 如果输入采用 `x_q = pixel_u8 - 128`，那：
[L5020] 
[L5021] - `z_x = -128`
[L5022] 
[L5023] ---
[L5024] 
[L5025] ## 2.2 `bias.hex` 的硬件使用方式
[L5026] 
[L5027] 现在你的 bias 是 32 位二补码，很适合直接这样读：
[L5028] 
[L5029] ```verilog
[L5030] reg [31:0] bias_mem [0:127];
[L5031] initial $readmemh("bias.hex", bias_mem);
[L5032] 
[L5033] wire signed [31:0] bias_val = $signed(bias_mem[j]);
[L5034] ```
[L5035] 
[L5036] 然后在 MAC 完成后：
[L5037] 
[L5038] ```verilog
[L5039] acc <= mac_sum + bias_val;
[L5040] ```
[L5041] 
[L5042] 这比 int8 bias 合理得多。
[L5043] 
[L5044] ---
[L5045] 
[L5046] ## 2.3 对 CIM 系统的直接使用建议
[L5047] 
[L5048] 如果你的 CIM 阵列本体负责的是 `int8 x int8` 乘加，那最常见的系统切分是：
[L5049] 
[L5050] ### CIM 阵列做
[L5051] - 权重存储
[L5052] - 向量输入
[L5053] - 局部乘加 / bitline accumulate / column accumulate
[L5054] 
[L5055] ### FPGA 数字逻辑做
[L5056] - 输入 zero-point 修正
[L5057] - 偏置相加
[L5058] - ReLU
[L5059] - requant
[L5060] - 第二层控制
[L5061] - argmax
[L5062] 
[L5063] 这和你现在的文件格式非常匹配：
[L5064] 
[L5065] - `weight.hex` 给阵列权重
[L5066] - `input_i.hex` 给输入驱动
[L5067] - `bias.hex` 给数字后处理单元
[L5068] - `golden` 给 testbench 对拍
[L5069] 
[L5070] ---
[L5071] 
[L5072] # 3. 这两个文件背后，代码还需要怎么改进
[L5073] 
[L5074] 你现在文件本身没大问题，但为了“**生成的数据可以被 FPGA 上的 CIM 系统直接使用**”，我建议你继续改 8 个点。
[L5075] 
[L5076] ---
[L5077] 
[L5078] ## 改进 1：文件命名必须分层，不能再叫通用名
[L5079] 
[L5080] 你现在如果最终是两层 MLP，绝对不要继续用：
[L5081] 
[L5082] - `weight.hex`
[L5083] - `bias.hex`
[L5084] 
[L5085] 因为你后面会出现：
[L5086] 
[L5087] - 第一层权重
[L5088] - 第二层权重
[L5089] - 第一层 bias
[L5090] - 第二层 bias
[L5091] 
[L5092] 建议直接改成：
[L5093] 
[L5094] - `fc1_weight_int8.hex`
[L5095] - `fc1_bias_int32.hex`
[L5096] - `fc2_weight_int8.hex`
[L5097] - `fc2_bias_int32.hex`
[L5098] 
[L5099] 这样你 RTL 一眼知道读哪个。
[L5100] 
[L5101] ---
[L5102] 
[L5103] ## 改进 2：必须配一个 `quant_config.json`
[L5104] 
[L5105] 现在单靠 hex 文件还不够。  
[L5106] FPGA/CIM 真正要跑整数推理，还需要知道：
[L5107] 
[L5108] - 输入 `scale`
[L5109] - 输入 `zero_point`
[L5110] - 各层 `weight_scale`
[L5111] - 各层 `weight_zero_point`
[L5112] - 各层 `output_scale`
[L5113] - 各层 `output_zero_point`
[L5114] - requant multiplier / shift
[L5115] 
[L5116] 建议额外生成：
[L5117] 
[L5118] ```json
[L5119] {
[L5120]   "input": {
[L5121]     "scale": 0.00392156862745098,
[L5122]     "zero_point": -128
[L5123]   },
[L5124]   "fc1": {
[L5125]     "weight_scale": ...,
[L5126]     "weight_zero_point": 0,
[L5127]     "output_scale": ...,
[L5128]     "output_zero_point": 0,
[L5129]     "bias_scale": ...,
[L5130]     "multiplier_q31": ...,
[L5131]     "shift": ...
[L5132]   },
[L5133]   "fc2": {
[L5134]     "weight_scale": ...,
[L5135]     "weight_zero_point": 0,
[L5136]     "output_scale": ...,
[L5137]     "output_zero_point": 0,
[L5138]     "bias_scale": ...,
[L5139]     "multiplier_q31": ...,
[L5140]     "shift": ...
[L5141]   }
[L5142] }
[L5143] ```
[L5144] 
[L5145] 这样 FPGA 端和 Python 端是一致的。
[L5146] 
[L5147] ---
[L5148] 
[L5149] ## 改进 3：输入文件也要明确 int8 和布局
[L5150] 
[L5151] 你现在最好明确导出成：
[L5152] 
[L5153] - `input_0_int8.hex`
[L5154] - `input_1_int8.hex`
[L5155] 
[L5156] 并且说明：
[L5157] 
[L5158] - 长度 784
[L5159] - 顺序是 row-major
[L5160] - 每行 2 位 hex
[L5161] - 数值是 signed int8
[L5162] - 若采用硬件输入策略，则：
[L5163]   \[
[L5164]   x_q = round(x \cdot 255) - 128
[L5165]   \]
[L5166] 
[L5167] 这能避免你后面忘记输入是“原始像素量化”还是“Normalize 后量化”。
[L5168] 
[L5169] ---
[L5170] 
[L5171] ## 改进 4：golden 不要只保留最终输出，要保留中间级
[L5172] 
[L5173] 这是对硬件调试最关键的一点。
[L5174] 
[L5175] 建议你至少导出：
[L5176] 
[L5177] ### 第一层
[L5178] - `fc1_acc_i.hex`：int32，未激活前
[L5179] - `fc1_relu_i.hex`：int32，过 ReLU 后
[L5180] - `fc1_out_i.hex`：int8，requant 后
[L5181] 
[L5182] ### 第二层
[L5183] - `fc2_acc_i.hex`：int32
[L5184] - `logits_i.hex`：int8
[L5185] 
[L5186] ### 分类
[L5187] - `pred_i.txt`
[L5188] 
[L5189] 这样当 FPGA 结果不对时，你能立刻定位是：
[L5190] 
[L5191] - 阵列乘加错
[L5192] - bias 加错
[L5193] - ReLU 错
[L5194] - requant 错
[L5195] - 第二层地址错
[L5196] 
[L5197] 不然只看最后一个 `golden`，很难找 bug。
[L5198] 
[L5199] ---
[L5200] 
[L5201] ## 改进 5：权重文件要考虑和 CIM 阵列物理映射一致
[L5202] 
[L5203] 你现在的 `weight.hex` 是逻辑顺序 `[out][in]`。  
[L5204] 这对软件和普通 RTL 很方便，但对实际 CIM 阵列，可能还不够。
[L5205] 
[L5206] 因为真实阵列通常不是一次吃 784 个输入，而是分块，例如：
[L5207] 
[L5208] - 每列 64 个输入
[L5209] - 每次做 64×N 子矩阵
[L5210] - 784 被切成很多 tile
[L5211] 
[L5212] 所以你后面很可能需要两套文件：
[L5213] 
[L5214] ### 逻辑文件
[L5215] 给软件 / testbench 用：
[L5216] - `fc1_weight_int8.hex`
[L5217] 
[L5218] ### 物理映射文件
[L5219] 给阵列初始化用：
[L5220] - `fc1_weight_tile0.hex`
[L5221] - `fc1_weight_tile1.hex`
[L5222] - ...
[L5223] 
[L5224] 也就是按 CIM 阵列实际映射切块导出。
[L5225] 
[L5226] 如果你现在阵列设计已经定了，比如每次 64 路输入并行，我建议导出时就按 64 一组打包。
[L5227] 
[L5228] ---
[L5229] 
[L5230] ## 改进 6：给每个文件写 manifest
[L5231] 
[L5232] 建议自动生成一个 `manifest.txt` 或 `manifest.json`，写明：
[L5233] 
[L5234] - 文件名
[L5235] - 数据类型
[L5236] - 位宽
[L5237] - 长度
[L5238] - 布局
[L5239] - 用途
[L5240] 
[L5241] 例如：
[L5242] 
[L5243] ```json
[L5244] {
[L5245]   "fc1_weight_int8.hex": {
[L5246]     "dtype": "int8",
[L5247]     "shape": [128, 784],
[L5248]     "layout": "row-major [out][in]"
[L5249]   },
[L5250]   "fc1_bias_int32.hex": {
[L5251]     "dtype": "int32",
[L5252]     "shape": [128]
[L5253]   },
[L5254]   "input_0_int8.hex": {
[L5255]     "dtype": "int8",
[L5256]     "shape": [784],
[L5257]     "layout": "row-major flattened image"
[L5258]   }
[L5259] }
[L5260] ```
[L5261] 
[L5262] 这会让你的工程后期非常省事。
[L5263] 
[L5264] ---
[L5265] 
[L5266] ## 改进 7：最好支持导出 `.mem` / `.coe` / 打包宽字格式
[L5267] 
[L5268] 现在一行一个数适合 testbench，  
[L5269] 但真正上 FPGA，很多时候你会想直接初始化 BRAM。
[L5270] 
[L5271] 比如：
[L5272] 
[L5273] - Xilinx 可能想要 `.coe`
[L5274] - Intel 可能想要 `.mif`
[L5275] - 或者你会把 16 个 int8 打成 128-bit 宽字
[L5276] 
[L5277] 所以建议你的导出工具支持两种模式：
[L5278] 
[L5279] ### 调试模式
[L5280] - 一行一个元素
[L5281] - 便于看和对拍
[L5282] 
[L5283] ### 上板模式
[L5284] - 打包成 32bit / 64bit / 128bit / 256bit
[L5285] - 便于 BRAM 初始化和 DMA 传输
[L5286] 
[L5287] ---
[L5288] 
[L5289] ## 改进 8：确保 signed 和 endianness 约定写死
[L5290] 
[L5291] 当前 hex 文件本身没错，但你要在代码里和说明里明确：
[L5292] 
[L5293] ### signed 解释
[L5294] - `weight.hex`：signed int8
[L5295] - `bias.hex`：signed int32
[L5296] 
[L5297] ### endianness
[L5298] 你现在是一行一个元素，所以“行内字节序”问题不大。  
[L5299] 但一旦你后面把 4 个 int8 打包成 32 bit，必须写清楚：
[L5300] 
[L5301] - 第 0 个元素放最低字节还是最高字节
[L5302] 
[L5303] 这个最好现在就定。
[L5304] 
[L5305] ---
[L5306] 
[L5307] # 4. 你现在这两个文件，我给你的具体判断
[L5308] 
[L5309] ---
[L5310] 
[L5311] ## `bias.hex`
[L5312] **没有明显问题，反而是明显进步。**
[L5313] 
[L5314] 它已经具备：
[L5315] - int32
[L5316] - 二补码
[L5317] - 可直接用于 MAC 后 bias add
[L5318] 
[L5319] ### 唯一建议
[L5320] 改名为：
[L5321] - `fc1_bias_int32.hex`
[L5322] - 或 `fc2_bias_int32.hex`
[L5323] 
[L5324] ---
[L5325] 
[L5326] ## `weight.hex`
[L5327] **没有明显问题。**
[L5328] 
[L5329] 它已经具备：
[L5330] - int8
[L5331] - 二补码
[L5332] - 适合阵列 / 乘加器直接使用
[L5333] 
[L5334] ### 唯一建议
[L5335] 改名为：
[L5336] - `fc1_weight_int8.hex`
[L5337] - 或 `fc2_weight_int8.hex`
[L5338] 
[L5339] ---
[L5340] 
[L5341] # 5. 我建议你下一版导出目录直接长这样
[L5342] 
[L5343] ```text
[L5344] export/
[L5345] ├── quant_config.json
[L5346] ├── manifest.json
[L5347] ├── fc1_weight_int8.hex
[L5348] ├── fc1_bias_int32.hex
[L5349] ├── fc2_weight_int8.hex
[L5350] ├── fc2_bias_int32.hex
[L5351] ├── input_0_int8.hex
[L5352] ├── fc1_acc_0_int32.hex
[L5353] ├── fc1_relu_0_int32.hex
[L5354] ├── fc1_out_0_int8.hex
[L5355] ├── fc2_acc_0_int32.hex
[L5356] ├── logits_0_int8.hex
[L5357] ├── pred_0.txt
[L5358] ├── label_0.txt
[L5359] ├── input_1_int8.hex
[L5360] ...
[L5361] ```
[L5362] 
[L5363] 这套结构就已经很像正式硬件联调工程了。
[L5364] 
[L5365] ---
[L5366] 
[L5367] # 6. 最后给你一句最关键的判断
[L5368] 
[L5369] 你现在贴出来的这两个文件，说明你已经迈过了最初 notebook 的两个大坑：
[L5370] 
[L5371] - **bias 不再是错误的 int8**
[L5372] - **weight 已经是标准 int8 二补码**
[L5373] 
[L5374] 所以文件内容层面，**方向是对的**。  
[L5375] 真正还需要继续加强的是：
[L5376] 
[L5377] - 文件命名分层
[L5378] - 中间 golden 分级导出
[L5379] - 量化参数 json 化
[L5380] - 按 CIM 阵列物理映射切块导出
[L5381] 
[L5382] 你把**当前用于导出这些文件的最新版 Python 代码**贴给我，我可以下一步直接帮你逐行审，给你改成一版“**输出可直接喂 FPGA/CIM 系统**”的正式版。
[L5383] 
[L5384] 用户:
[L5385] 这是当前生成的数据，你主要先看一下route_b_output中的文件格式是否全部正确，然后给出一个markdown文件，用以说明route_b_output所有文件的使用方法和注意事项，要详细，而且要说明根据当前数据，硬件设计需要特别注意的地方。关于route_a_output,简单检查即可。
[L5386] AI助手:
[L5387] 
[L5388] 
[L5389] AI助手:
[L5390] 我先把两个压缩包解开，核对 `route_b_output` 的文件清单、每类文件的位宽和行数，再抽样检查几个样本文件是否彼此自洽。然后我会生成一份完整的 Markdown 说明，专门给你们做硬件联调用。
[L5391] 
[L5392] AI助手:
[L5393] 我已经核对出 `route_b_output` 的主要文件都在，位宽和行数目前是自洽的。接下来我再补两件更关键的事：一是看数值范围是否会踩硬件位宽坑，二是把这些坑写进说明文档里，避免你们后面在阵列映射、signed 解释、requant 上出错。
[L5394] 
[L5395] AI助手:
[L5396] 我已经检查完了，结论是：
[L5397] 
[L5398] `route_b_output` 的主文件格式**整体正确**，而且已经具备作为 FPGA / CIM 整数推理联调数据的条件。  
[L5399] 需要硬件特别注意的核心点有三个：
[L5400] 
[L5401] 1. **第一层输入零点不是 0，而是 `-128`**，不能直接做 `x_q * w_q`，必须处理输入零点补偿。  
[L5402] 2. **第一层累加范围明显超过 16 位**，当前数据已经出现大约 `-5.5e5 ~ 3.46e5`，所以 `fc1` 累加器和 bias 路径必须至少 `int32`。  
[L5403] 3. **当前权重文件是逻辑布局 `row-major [out][in]`**，很适合 testbench 和 RTL 功能验证，但不一定等于你们 CIM 阵列的物理烧写布局，后续可能还需要做一层 tile 重排。
[L5404] 
[L5405] `route_a_output` 我也做了简查，主文件没有明显格式问题，适合继续用于单层 `fc1 + ReLU` 验证；但它不是最终两层分类演示版本。
[L5406] 
[L5407] 我已经把完整说明整理成 Markdown 文件，里面包括：
[L5408] 
[L5409] - `route_b_output` 每个文件的格式、行数、位宽、布局
[L5410] - 各文件的具体用途
[L5411] - 推荐的硬件联调顺序
[L5412] - 当前数据下必须特别注意的硬件坑
[L5413] - `route_a_output` 的简要检查结论
[L5414] 
[L5415] 下载这里：
[L5416] 
[L5417] [route_b_output 使用说明与硬件注意事项](sandbox:/mnt/data/route_b_output_usage_notes.md)
[L5418] 
[L5419] 如果你愿意，我下一步可以继续直接帮你补一份**面向 Verilog/Testbench 的读取模板**，把这些文件如何 `$readmemh`、如何按 signed 解释、如何逐级对拍写成可直接用的代码框架。
[L5420] 
[L5421] AI助手:
[L5422] 
[L5423] 
[L5424] AI助手:
[L5425] 我先逐项核对 `route_b_output` 里的实际文件名、位宽、行数和内容一致性，再补一个详细的使用说明文档。`route_a_output` 我会做一轮轻量检查，重点看有没有明显格式错误或和 `route_b` 不一致的地方。
[L5426] 
[L5427] AI助手:
[L5428] `route_b_output` 的主文件看起来是自洽的：位宽、行数、层级命名和量化配置基本能对上。我再补一轮一致性检查，重点看标签/预测是否匹配，以及 `route_a_output` 有没有明显格式问题。
[L5429] 
[L5430] 用户:
[L5431] route_b_output的使用指南markdown没有生成出来，请你重新尝试给出一个markdown文件，用以说明route_b_output所有文件的使用方法和注意事项，要详细，而且要说明根据当前数据，硬件设计需要特别注意的地方。
[L5432] AI助手:
[L5433] 
[L5434] 
[L5435] AI助手:
[L5436] 已重新生成 Markdown 文件：
[L5437] 
[L5438] [route_b_output 使用指南（Markdown）](sandbox:/mnt/data/route_b_output_usage_guide.md)
[L5439] 
[L5440] 我已经把这些内容写进去：
[L5441] - `route_b_output` 全部文件的格式、行数、位宽、含义
[L5442] - 每类文件在 FPGA / CIM 中怎么用
[L5443] - 推荐的联调顺序
[L5444] - `$readmemh` 的典型用法
[L5445] - 基于当前数据，硬件设计必须特别注意的点
[L5446] - 对 `route_a_output` 的简要检查结论
[L5447] 
[L5448] 这份文档是按你当前实际导出的 `route_b_output` 内容写的，不是泛泛模板。
[L5449] 
[L5450] 用户:
[L5451] 还记得我最开始上传的三个文件吗？项目设计指导书，上次和AI的对话，请你重新看一下，我在完成这个测试向量生成之后，应该做什么？
[L5452] AI助手:
[L5453] 
[L5454] 
[L5455] AI助手:
[L5456] 
[L5457] 
[L5458] AI助手:
[L5459] 
[L5460] 
[L5461] AI助手:
[L5462] 我记得你最开始给的三个核心材料：
[L5463] 
[L5464] 1. **项目设计指导书**（课程/课设要求）  
[L5465] 2. **你和上一个 AI 的对话记录**（确定了大体方案：FPGA + CIM + MNIST）  
[L5466] 3. **生成测试向量的 ipynb**（用于导出权重、输入、golden）
[L5467] 
[L5468] 现在你已经完成了 **route_b_output 的测试向量生成**。  
[L5469] 这一步其实在整个项目流程里属于 **软件侧 → 硬件侧的接口准备阶段**。
[L5470] 
[L5471] 接下来你要做的事情其实非常明确：  
[L5472] 
[L5473] **进入硬件实现和验证阶段。**
[L5474] 
[L5475] 我给你按 **真正工程流程**排一个清晰路线（也是最符合你这个项目指导书逻辑的路线）。
[L5476] 
[L5477] ---
[L5478] 
[L5479] # 一、你现在所处的位置
[L5480] 
[L5481] 你的项目整体流程其实是：
[L5482] 
[L5483] ```
[L5484] PyTorch训练
[L5485]       ↓
[L5486] 量化
[L5487]       ↓
[L5488] 整数推理参考模型
[L5489]       ↓
[L5490] 生成测试向量 (route_b_output)  ← 你现在完成到这里
[L5491]       ↓
[L5492] FPGA/CIM 硬件实现
[L5493]       ↓
[L5494] testbench 对拍验证
[L5495]       ↓
[L5496] 板级运行
[L5497]       ↓
[L5498] 项目展示
[L5499] ```
[L5500] 
[L5501] 所以现在你已经完成：
[L5502] 
[L5503] **软件侧所有准备工作。**
[L5504] 
[L5505] 接下来全部工作都是：
[L5506] 
[L5507] **硬件实现 + 硬件验证。**
[L5508] 
[L5509] ---
[L5510] 
[L5511] # 二、下一步你应该做什么（非常具体）
[L5512] 
[L5513] 下一步应该分 **4个阶段**。
[L5514] 
[L5515] ---
[L5516] 
[L5517] # 阶段 1：设计硬件计算结构
[L5518] 
[L5519] 现在要把你 Python 的网络结构映射成 **硬件模块结构**。
[L5520] 
[L5521] 你的网络是：
[L5522] 
[L5523] ```
[L5524] 784 → 128 → 10
[L5525] ```
[L5526] 
[L5527] 对应硬件结构：
[L5528] 
[L5529] ```
[L5530] input_buffer (784)
[L5531] 
[L5532]       ↓
[L5533] 
[L5534] FC1  (784×128)  ← CIM阵列
[L5535] 
[L5536]       ↓
[L5537] 
[L5538] bias_add
[L5539] 
[L5540]       ↓
[L5541] 
[L5542] ReLU
[L5543] 
[L5544]       ↓
[L5545] 
[L5546] requant (int32→int8)
[L5547] 
[L5548]       ↓
[L5549] 
[L5550] FC2  (128×10)
[L5551] 
[L5552]       ↓
[L5553] 
[L5554] argmax
[L5555] ```
[L5556] 
[L5557] 建议模块划分：
[L5558] 
[L5559] ```
[L5560] top
[L5561]  ├── input_loader
[L5562]  ├── fc1_cim_array
[L5563]  ├── bias_add
[L5564]  ├── relu
[L5565]  ├── requant
[L5566]  ├── fc2
[L5567]  └── argmax
[L5568] ```
[L5569] 
[L5570] 如果你的课设重点是 **CIM**，通常只把：
[L5571] 
[L5572] ```
[L5573] fc1
[L5574] ```
[L5575] 
[L5576] 做成 CIM 阵列。
[L5577] 
[L5578] 后面的：
[L5579] 
[L5580] ```
[L5581] bias
[L5582] relu
[L5583] requant
[L5584] fc2
[L5585] ```
[L5586] 
[L5587] 可以在 FPGA 普通逻辑实现。
[L5588] 
[L5589] ---
[L5590] 
[L5591] # 阶段 2：建立 Testbench 验证系统
[L5592] 
[L5593] 这是你 **下一步最重要的事情**。
[L5594] 
[L5595] 你已经有：
[L5596] 
[L5597] ```
[L5598] route_b_output/
[L5599] ```
[L5600] 
[L5601] 里面包含：
[L5602] 
[L5603] ```
[L5604] fc1_weight_int8.hex
[L5605] fc1_bias_int32.hex
[L5606] fc2_weight_int8.hex
[L5607] fc2_bias_int32.hex
[L5608] input_*.hex
[L5609] fc1_acc_*.hex
[L5610] fc1_relu_*.hex
[L5611] fc1_out_*.hex
[L5612] fc2_acc_*.hex
[L5613] logits_*.hex
[L5614] pred_*.txt
[L5615] labels.txt
[L5616] ```
[L5617] 
[L5618] Testbench 要做的事情：
[L5619] 
[L5620] ### 1 读取权重
[L5621] 
[L5622] ```
[L5623] $readmemh("fc1_weight_int8.hex", weight_mem);
[L5624] $readmemh("fc1_bias_int32.hex", bias_mem);
[L5625] ```
[L5626] 
[L5627] ---
[L5628] 
[L5629] ### 2 读取输入
[L5630] 
[L5631] ```
[L5632] $readmemh("input_0.hex", input_mem);
[L5633] ```
[L5634] 
[L5635] ---
[L5636] 
[L5637] ### 3 驱动 DUT
[L5638] 
[L5639] 输入 784 个数据
[L5640] 
[L5641] ---
[L5642] 
[L5643] ### 4 接收 DUT 输出
[L5644] 
[L5645] 例如：
[L5646] 
[L5647] ```
[L5648] fc1_acc
[L5649] fc1_relu
[L5650] fc1_out
[L5651] fc2_acc
[L5652] logits
[L5653] ```
[L5654] 
[L5655] ---
[L5656] 
[L5657] ### 5 与 golden 对比
[L5658] 
[L5659] 例如：
[L5660] 
[L5661] ```
[L5662] $readmemh("fc1_acc_0.hex", golden_fc1_acc);
[L5663] ```
[L5664] 
[L5665] 然后逐项比较：
[L5666] 
[L5667] ```
[L5668] if(dut_acc[i] != golden_fc1_acc[i])
[L5669] ```
[L5670] 
[L5671] ---
[L5672] 
[L5673] # 阶段 3：逐层验证（非常重要）
[L5674] 
[L5675] 不要一口气跑完整网络。
[L5676] 
[L5677] 必须 **逐层验证**。
[L5678] 
[L5679] 建议顺序：
[L5680] 
[L5681] ---
[L5682] 
[L5683] ## Step1
[L5684] 
[L5685] 只验证
[L5686] 
[L5687] ```
[L5688] FC1 MAC
[L5689] ```
[L5690] 
[L5691] 对拍：
[L5692] 
[L5693] ```
[L5694] fc1_acc_*.hex
[L5695] ```
[L5696] 
[L5697] ---
[L5698] 
[L5699] ## Step2
[L5700] 
[L5701] 验证
[L5702] 
[L5703] ```
[L5704] FC1 + bias
[L5705] ```
[L5706] 
[L5707] ---
[L5708] 
[L5709] ## Step3
[L5710] 
[L5711] 验证
[L5712] 
[L5713] ```
[L5714] FC1 + bias + relu
[L5715] ```
[L5716] 
[L5717] 对拍：
[L5718] 
[L5719] ```
[L5720] fc1_relu_*.hex
[L5721] ```
[L5722] 
[L5723] ---
[L5724] 
[L5725] ## Step4
[L5726] 
[L5727] 验证
[L5728] 
[L5729] ```
[L5730] FC1 + requant
[L5731] ```
[L5732] 
[L5733] 对拍：
[L5734] 
[L5735] ```
[L5736] fc1_out_*.hex
[L5737] ```
[L5738] 
[L5739] ---
[L5740] 
[L5741] ## Step5
[L5742] 
[L5743] 再加入
[L5744] 
[L5745] ```
[L5746] FC2
[L5747] ```
[L5748] 
[L5749] 对拍：
[L5750] 
[L5751] ```
[L5752] fc2_acc_*.hex
[L5753] ```
[L5754] 
[L5755] ---
[L5756] 
[L5757] ## Step6
[L5758] 
[L5759] 验证
[L5760] The file is too long and its contents have been truncated.
[L5761] 
AI助手:
我已经把 `route_b_output` 的文件结构、导出代码含义和硬件侧注意点整理成文档了，现在把 Markdown 文件写出来并给你下载链接。

用户:
你可以先阅读一下我的代码，在对照一下上面的文件，我已经进行到完成了单个tile的计算，现在你觉得我该做什么？我认为我应该做多个tile的并行计算，如果你觉得我接下做这个是正确的，就给我介绍下一步的设计流程，并且给出代码
AI助手:
我先把你上传的 `hw.zip` 和前面的说明文件一起过一遍，确认你当前单 tile 设计到底做到哪一步，再给你一个“下一步最值得做什么”的判断和可落地代码框架。

AI助手:
我已经确认你现在的状态：`cim_tile` 单 tile 乘加已经通了，`fc1_cim_core_block` 也已经做到了“沿 49 个 input block 累加，最终吐出 16 个输出神经元”的层级。这说明你其实已经不只是在做“单 tile”，而是在做“单个 output block = 16 路输出”的完整 FC1 子块。

AI助手:
我看完你的代码后，结论很明确：

**你下一步做“多个 tile 的并行计算”是对的，而且是最自然的下一步。**

不过我会把这句话说得更准确一点：

你现在已经完成的不是“只有一个 tile 的孤立验证”，而是：

- `cim_tile.sv`：**一个 16x16 tile 的组合 MAC**
- `psum_accum.sv`：**沿 input block 方向做 49 次累加**
- `fc1_cim_core_block.sv`：**固定一个 output block（16 个神经元），完成整段 FC1 的 block 计算**

也就是说，你当前已经能算出：

- 128 维隐藏层中的 **16 维**
- 这 16 维对应某个 `ob_sel`
- 输入 784 维会被切成 49 个 `ib`
- 每个 `ib` 都经过同一个 tile 累加进去

这一步是对的，而且结构也很清晰。

---

## 你下一步最应该做什么

我建议按下面顺序推进：

### 第一步：先做 **多个 output block 并行**
也就是并行计算多个 `ob_sel`。

这是最正确的并行方向，因为：

- FC1 的 128 个输出，本来就被你分成了 `8 个 output block`
- 每个 block 是 16 个输出
- 所有 block **共享同一个输入向量**
- 只是权重不同、bias 不同

所以最自然的并行就是：

- 广播同一个 `ib`
- 同时算多个 `ob`
- 每个 `ob` 内部还是你现在这个 tile + accum 流程

这比“在 input 方向并行多个 tile”更适合作为你现在的下一步，因为它几乎完全复用你已有设计。

---

## 为什么这一步是正确的

你当前数据流本质是：

\[
y[ob][tr] = \sum_{ib=0}^{48} \sum_{tc=0}^{15} x[ib][tc] \cdot w[ob][tr][ib][tc] + b[ob][tr]
\]

现在你是：

- 固定一个 `ob`
- 扫完全部 `ib`

下一步并行化后可以变成：

- 同时处理多个 `ob`
- 对每个 `ib`，把同一份 `x_eff_tile` 广播给多个计算单元
- 每个单元读取自己的 `w_tile`
- 分别累加自己的 `psum`

这就是典型的 **output-channel parallelism**。

---

# 建议的设计流程

## 方案 A：最适合你现在继续写的版本
直接复制多个 `fc1_cim_core_block` 实例并行跑。

### 优点
- 代码改动最小
- 复用你现在所有模块
- 最容易先跑通仿真
- 最容易先做正确性验证

### 缺点
- 每个 block 都各自实例化了 `input_buffer`
- 在真实硬件上会有重复存储/重复读取
- 不是最终最优结构

但作为**下一步里程碑**，这是最合适的。

---

## 方案 B：后续再做的优化版本
把结构改成：

- 一个全局 `input_buffer`
- 一个全局 `ib` 控制器
- 多个并行 tile engine
- 每个 engine 只负责一个 `ob`
- 所有 engine 共享输入广播

这个才是更像正式加速器的结构。

但我建议你**先做方案 A，把多 block 并行跑起来，再重构成方案 B。**

---

# 我建议你的下一步里程碑

按这个顺序做最稳：

### 1. 先把单 block 测试补全
你现在 `tb_fc1_cim_core_block.sv` 只测了 `ob_sel = 0`。

先把它改成：

- `ob_sel = 0..7` 全扫
- 每个 block 对比 `fc1_acc_0.hex` 里相应 16 个结果

这一步很重要，因为它会暴露地址映射有没有问题。

---

### 2. 实现 2 路并行 block
先别一口气 8 路。

先做：

- `PAR_OB = 2`

这样更容易调试。

---

### 3. 再扩成 4 路 / 8 路
确认 2 路对了以后，扩到：

- `PAR_OB = 4`
- 最终 `PAR_OB = 8`

这样你整层 FC1 就能一次算完 128 维输出。

---

### 4. 把 FC1 输出接 ReLU / 量化 / FC2
到那时你的系统路径就是：

- input
- FC1 multi-block parallel
- bias
- ReLU / requant
- FC2

---

# 你现在最值得改的结构

你当前 `fc1_cim_core_block` 已经可以当成一个 **16 输出并行 engine**。

所以你应该新建一个顶层，例如：

- `fc1_multi_block_parallel.sv`

作用：

- 启动多个 `fc1_cim_core_block`
- 给每个实例不同的 `ob_sel`
- 收集所有实例输出
- 拼成完整的 128 维 `fc1_acc_all`

---

# 我建议你直接这样写

下面这份代码是你当前工程最容易接进去的版本。

---

## 1）多 block 并行顶层：`fc1_multi_block_parallel.sv`

```systemverilog
module fc1_multi_block_parallel #(
    parameter int PAR_OB = mnist_cim_pkg::N_OUTPUT_BLOCKS
) (
    input  logic clk,
    input  logic rst_n,
    input  logic start,

    output logic busy,
    output logic done,

    output logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0]
        fc1_acc_all [0:mnist_cim_pkg::HIDDEN_DIM-1]
);
  import mnist_cim_pkg::*;

  logic [PAR_OB-1:0] busy_vec;
  logic [PAR_OB-1:0] done_vec;

  logic signed [PSUM_WIDTH-1:0]
      fc1_acc_block [0:PAR_OB-1][0:TILE_OUTPUT_SIZE-1];

  genvar g_ob, g_idx;
  generate
    for (g_ob = 0; g_ob < PAR_OB; g_ob = g_ob + 1) begin : GEN_OB_CORE
      fc1_cim_core_block u_fc1_cim_core_block (
          .clk(clk),
          .rst_n(rst_n),
          .start(start),
          .ob_sel(g_ob[$clog2(N_OUTPUT_BLOCKS)-1:0]),
          .busy(busy_vec[g_ob]),
          .done(done_vec[g_ob]),
          .fc1_acc_block(fc1_acc_block[g_ob])
      );

      for (g_idx = 0; g_idx < TILE_OUTPUT_SIZE; g_idx = g_idx + 1) begin : GEN_PACK
        assign fc1_acc_all[g_ob*TILE_OUTPUT_SIZE + g_idx] = fc1_acc_block[g_ob][g_idx];
      end
    end
  endgenerate

  assign busy = |busy_vec;
  assign done = &done_vec;

endmodule
```

---

## 2）对应测试平台：`tb_fc1_multi_block_parallel.sv`

这个 testbench 直接对比整份 `fc1_acc_0.hex` 的 128 个值。

```systemverilog
`timescale 1ns / 1ps

module tb_fc1_multi_block_parallel;
  import mnist_cim_pkg::*;

  logic clk;
  logic rst_n;
  logic start;
  logic busy;
  logic done;

  logic signed [PSUM_WIDTH-1:0] fc1_acc_all[0:HIDDEN_DIM-1];
  logic signed [PSUM_WIDTH-1:0] ref_fc1_acc_mem[0:HIDDEN_DIM-1];

  string fc1_acc_file;

  integer i;
  integer error_count;

  fc1_multi_block_parallel #(
      .PAR_OB(N_OUTPUT_BLOCKS)
  ) dut (
      .clk(clk),
      .rst_n(rst_n),
      .start(start),
      .busy(busy),
      .done(done),
      .fc1_acc_all(fc1_acc_all)
  );

  initial clk = 1'b0;
  always #5 clk = ~clk;

  initial begin
    fc1_acc_file = "../../CIM-sw-version1/sw/train_quantize/route_b_output/fc1_acc_0.hex";

    if ($value$plusargs("FC1_ACC_FILE=%s", fc1_acc_file)) begin
      $display("TB using FC1_ACC_FILE: %s", fc1_acc_file);
    end else begin
      $display("TB using default FC1_ACC_FILE: %s", fc1_acc_file);
    end

    $readmemh(fc1_acc_file, ref_fc1_acc_mem);
  end

  initial begin
    error_count = 0;

    rst_n = 1'b0;
    start = 1'b0;

    #12;
    rst_n = 1'b1;

    @(posedge clk);
    start <= 1'b1;

    @(posedge clk);
    start <= 1'b0;

    wait(done == 1'b1);
    @(posedge clk);
    #1;

    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
      if (fc1_acc_all[i] !== ref_fc1_acc_mem[i]) begin
        $display("ERROR idx=%0d got=%0d expected=%0d",
                 i, fc1_acc_all[i], ref_fc1_acc_mem[i]);
        error_count = error_count + 1;
      end else begin
        $display("MATCH idx=%0d value=%0d", i, fc1_acc_all[i]);
      end
    end

    if (error_count == 0)
      $display("PASS: fc1_multi_block_parallel matches full fc1_acc_0.hex");
    else
      $display("FAIL: found %0d mismatches", error_count);

    $finish;
  end

endmodule
```

---

# 但我还建议你先补一个“单实例扫全 ob”的 testbench

因为你现在的 `tb_fc1_cim_core_block.sv` 只测了 `ob_sel=0`。  
在上多并行之前，先确认：

- `ob=0..7` 每个 block 都对

这是非常值得先补的。

---

## 3）单实例扫完整个 8 个 output block 的测试

```systemverilog
`timescale 1ns / 1ps

module tb_fc1_cim_core_block_all_ob;
  import mnist_cim_pkg::*;

  logic clk;
  logic rst_n;
  logic start;
  logic busy;
  logic done;

  logic [$clog2(N_OUTPUT_BLOCKS)-1:0] ob_sel;
  logic signed [PSUM_WIDTH-1:0] fc1_acc_block[0:TILE_OUTPUT_SIZE-1];

  logic signed [PSUM_WIDTH-1:0] ref_fc1_acc_mem[0:HIDDEN_DIM-1];

  string fc1_acc_file;

  integer ob;
  integer i;
  integer error_count;
  integer global_idx;

  fc1_cim_core_block dut (
      .clk(clk),
      .rst_n(rst_n),
      .start(start),
      .ob_sel(ob_sel),
      .busy(busy),
      .done(done),
      .fc1_acc_block(fc1_acc_block)
  );

  initial clk = 1'b0;
  always #5 clk = ~clk;

  initial begin
    fc1_acc_file = "../../CIM-sw-version1/sw/train_quantize/route_b_output/fc1_acc_0.hex";
    if ($value$plusargs("FC1_ACC_FILE=%s", fc1_acc_file))
      $display("TB using FC1_ACC_FILE: %s", fc1_acc_file);
    else
      $display("TB using default FC1_ACC_FILE: %s", fc1_acc_file);

    $readmemh(fc1_acc_file, ref_fc1_acc_mem);
  end

  initial begin
    error_count = 0;

    rst_n = 1'b0;
    start = 1'b0;
    ob_sel = '0;

    #12;
    rst_n = 1'b1;

    for (ob = 0; ob < N_OUTPUT_BLOCKS; ob = ob + 1) begin
      ob_sel = ob[$clog2(N_OUTPUT_BLOCKS)-1:0];

      @(posedge clk);
      start <= 1'b1;

      @(posedge clk);
      start <= 1'b0;

      wait(done == 1'b1);
      @(posedge clk);
      #1;

      $display("Checking ob=%0d ...", ob);

      for (i = 0; i < TILE_OUTPUT_SIZE; i = i + 1) begin
        global_idx = ob * TILE_OUTPUT_SIZE + i;

        if (fc1_acc_block[i] !== ref_fc1_acc_mem[global_idx]) begin
          $display("ERROR ob=%0d idx=%0d got=%0d expected=%0d",
                   ob, i, fc1_acc_block[i], ref_fc1_acc_mem[global_idx]);
          error_count = error_count + 1;
        end
      end

      @(posedge clk);
    end

    if (error_count == 0)
      $display("PASS: all ob blocks match fc1_acc_0.hex");
    else
      $display("FAIL: found %0d mismatches", error_count);

    $finish;
  end

endmodule
```

---

# 你接下来真正的架构重点

## 重点 1：并行的方向应该选 output block
不是先在 input block 上平铺。

因为你现在的系统：

- 输入是同一份
- 不同输出块权重不同
- 结果彼此独立

这特别适合做并行。

---

## 重点 2：先允许重复 input_buffer，后面再优化
当前最省事的办法是每个并行 block 自己带一个 `input_buffer`。  
虽然不优雅，但能快速验证架构正确性。

后面你可以把 `fc1_cim_core_block` 重构成两层：

### 控制层
- 统一 `ib` 计数器
- 统一 `start/done`

### 计算层
- 多个 `cim_tile + psum_accum`
- 每个计算层只对应一个 `ob`

这样输入广播就能共享。

---

## 重点 3：done 的定义
多 block 并行后：

- `busy = |busy_vec`
- `done = &done_vec`

这是最简单直接的写法。

---

# 我还发现了一个你后面要注意的点

你当前 `fc1_cim_core_block` 里有三个字符串变量：

```systemverilog
string input_file;
string weight_file;
string bias_file;
```

并且它们通过 plusarg 读取，但**没有实际传给子模块**。  
子模块自己又各自读取 plusarg / 默认路径。

这虽然仿真时通常还能跑，但结构上有点“表面设置了，实际上没传下去”。

也就是这里：

```systemverilog
input_buffer #(
    .DEFAULT_INPUT_HEX_FILE(DEFAULT_INPUT_HEX_FILE)
) u_input_buffer (...)
```

并没有把 `input_file` 传进去。  
`weight_bank` 和 `bias_bank` 也是同理。

这不是你当前最紧急的问题，但后面建议整理一下，不然并行实例多起来时，文件管理会更乱。

---

# 所以我的判断是

**是的，你下一步做多个 tile / 多个 output block 并行，是正确路线。**

而且更准确地说：

**你应该先做“多个 output block 并行”，让 FC1 一次算出更多 hidden neuron。**

---

# 我给你的实际执行建议

按这个顺序走最稳：

1. 先把 `tb_fc1_cim_core_block` 扩成 `ob=0..7` 全验证  
2. 再加 `fc1_multi_block_parallel.sv`，先跑 `PAR_OB=2`  
3. 跑通后扩成 `PAR_OB=8`  
4. 再考虑共享 input buffer、统一 ib controller  
5. 最后接 ReLU / requant / FC2

---

如果你愿意，我下一条可以继续直接帮你把这几个文件整理成你工程可直接放进去的完整版本，包括一个对应的 `run_fc1_multi_block_parallel.sh` 脚本。

用户:
fc1_multi_block_parallel.sv这个就是直接简单的实例多个对吧，我觉得还是先实现ob=0..7的全验证，然后简单的实例化两个，然后变成实例化8个，然后再共享inputbuffer和ib controller,最后激活和FC2.请帮我实现ob=0..7的全验证
AI助手:
我先把你上传的工程里现有 `tb_fc1_cim_core_block.sv`、`fc1_cim_core_block.sv` 和包定义对照一下，然后直接给你一份可运行的 `ob=0..7` 全验证 testbench。

AI助手:
我已经确认你之前那版单块 testbench 的核心结构了：它目前确实只固定测 `ob=0`，而且比较方式就是按 `global_idx = ob_sel * 16 + i` 去对拍整份 `fc1_acc_0.hex`。现在我去把你上传的工程实际文件结构再核对一下，避免给你的版本和你现有端口名不一致。

AI助手:
可以，先做 **`ob=0..7` 全验证** 很合适。你当前的 testbench 确实还是“固定只测 `ob=0`”，主流程里把 `ob_sel = 0` 后打一发 `start`，然后只检查这一块的 16 个输出。

你现在直接把 `tb/tb_fc1_cim_core_block.sv` 改成下面这版就行。

```systemverilog
`timescale 1ns / 1ps

module tb_fc1_cim_core_block;
  import mnist_cim_pkg::*;

  logic clk;
  logic rst_n;
  logic start;
  logic busy;
  logic done;

  logic [$clog2(N_OUTPUT_BLOCKS)-1:0] ob_sel;

  logic signed [PSUM_WIDTH-1:0] fc1_acc_block[0:TILE_OUTPUT_SIZE-1];

  string fc1_acc_file;

  logic signed [PSUM_WIDTH-1:0] ref_fc1_acc_mem[0:HIDDEN_DIM-1];

  integer ob;
  integer i;
  integer error_count;
  integer global_idx;

  fc1_cim_core_block dut (
      .clk(clk),
      .rst_n(rst_n),
      .start(start),
      .ob_sel(ob_sel),
      .busy(busy),
      .done(done),
      .fc1_acc_block(fc1_acc_block)
  );

  // --------------------------------------------
  // Clock
  // --------------------------------------------
  initial clk = 1'b0;
  always #5 clk = ~clk;

  // --------------------------------------------
  // TB golden file from plusarg
  // --------------------------------------------
  initial begin
    fc1_acc_file = "../../CIM-sw-version1/sw/train_quantize/route_b_output/fc1_acc_0.hex";

    if ($value$plusargs("FC1_ACC_FILE=%s", fc1_acc_file)) begin
      $display("TB using FC1_ACC_FILE: %s", fc1_acc_file);
    end else begin
      $display("TB using default FC1_ACC_FILE: %s", fc1_acc_file);
    end

    $readmemh(fc1_acc_file, ref_fc1_acc_mem);
  end

  // --------------------------------------------
  // Optional monitor
  // --------------------------------------------
  initial begin
    $display("time=%0t : TB start", $time);
    $monitor("time=%0t clk=%0b rst_n=%0b start=%0b busy=%0b done=%0b ob_sel=%0d",
             $time, clk, rst_n, start, busy, done, ob_sel);
  end

  // --------------------------------------------
  // task: run one ob block
  // --------------------------------------------
  task automatic run_one_ob(input integer ob_idx);
    begin
      ob_sel = ob_idx[$clog2(N_OUTPUT_BLOCKS)-1:0];

      @(posedge clk);
      start <= 1'b1;

      @(posedge clk);
      start <= 1'b0;

      wait (done == 1'b1);

      // 再等一个时钟边沿，确保输出稳定
      @(posedge clk);
      #1;

      $display("Checking fc1_acc_block for ob_sel = %0d ...", ob_idx);

      for (i = 0; i < TILE_OUTPUT_SIZE; i = i + 1) begin
        global_idx = ob_idx * TILE_OUTPUT_SIZE + i;

        if (fc1_acc_block[i] !== ref_fc1_acc_mem[global_idx]) begin
          $display("ERROR ob=%0d idx=%0d got=%0d expected=%0d",
                   ob_idx, i, fc1_acc_block[i], ref_fc1_acc_mem[global_idx]);
          error_count = error_count + 1;
        end else begin
          $display("MATCH ob=%0d idx=%0d value=%0d",
                   ob_idx, i, fc1_acc_block[i]);
        end
      end

      // 让 DUT 回到 IDLE，再开始下一轮
      @(posedge clk);
      #1;
    end
  endtask

  // --------------------------------------------
  // Main test
  // --------------------------------------------
  initial begin
    error_count = 0;

    rst_n  = 1'b0;
    start  = 1'b0;
    ob_sel = '0;

    #12;
    rst_n = 1'b1;

    // 依次验证 ob = 0..7
    for (ob = 0; ob < N_OUTPUT_BLOCKS; ob = ob + 1) begin
      run_one_ob(ob);
    end

    if (error_count == 0) begin
      $display("PASS: fc1_cim_core_block matches all output blocks in fc1_acc_0.hex.");
    end else begin
      $display("FAIL: found %0d mismatches across all output blocks.", error_count);
    end

    $finish;
  end

endmodule
```

你这一步的核心改动只有三点：

1. 原来固定 `ob_sel = 0`，现在改成 `for (ob = 0; ob < N_OUTPUT_BLOCKS; ob++)` 逐块验证。  
2. 每块的 golden index 还是原来的规则：`global_idx = ob * TILE_OUTPUT_SIZE + i`。这个规则和你之前单块 testbench 的比较逻辑是一致的。  
3. 每次 `done` 之后多等一个时钟，再开下一轮，让 DUT 从 `DONE -> IDLE` 走干净。你当前 `fc1_cim_core_block` 的状态机在 `S_DONE` 里是 `done=1`，并且要等 `!start` 才回 `IDLE`。

你现有的 `fc1_cim_core_block` 已经带 bias 输出了，末尾是 `fc1_acc_block = psum + bias_block`，所以这版 testbench 继续对拍 `fc1_acc_0.hex` 是合理的。

再提醒你两个小点：

- 你 zip 里的实际路径是 `hw/route_b_output/fc1_acc_0.hex`，而不是默认写的 `../../CIM-sw-version1/...`。所以跑仿真时大概率还是要靠脚本里的 `+FC1_ACC_FILE=...` 覆盖默认路径。
- 这版 testbench 不需要改 DUT 端口，和你当前工程里的 `fc1_cim_core_block.sv` 直接匹配。

你可以先直接替换这个文件，然后跑一遍 `ob=0..7`。如果你愿意，我下一条直接把 `run_fc1_cim_core_block.sh` 也一起改成适配这版全验证的版本。

用户:
run_fc1_cim_core_block.sh需要修改吗
AI助手:
我先看一下你工程里现有 `run_fc1_cim_core_block.sh` 是怎么传 `+FC1_ACC_FILE` 和编译 testbench 的，再直接告诉你要不要改、改哪几行。

用户:

AI助手:
要改，但**只需要小改，不是为了 `ob=0..7` 本身改流程，而是为了保证脚本和你现在的 DUT/TB 保持一致**。

你现在这个脚本至少应该满足两件事：

1. **编译列表里包含 `fc1_bias_bank.sv`**  
2. **运行时传入 `+FC1_BIAS_FILE=...`**  

因为你当前这条仿真链路里，`fc1_cim_core_block` 已经需要 bias bank 参与，之前整理出的推荐脚本也是这么配的：编译时带上 `fc1_bias_bank.sv`，运行时传 `+FC1_BIAS_FILE=${FC1_BIAS_HEX}`。

而且你现在新改的 testbench 只是把原来“固定测 `ob=0`”变成“循环测 `ob=0..7`”，它读取 golden 的方式还是同一个 `FC1_ACC_FILE`，这一点没有变。之前那版 TB 也是通过 plusarg 读 `fc1_acc_0.hex` 的。  
所以**脚本不需要因为“全验证”额外新增什么参数**，但要保证它至少已经是“bias 版脚本”。

你可以直接把 `run_fc1_cim_core_block.sh` 改成这版：

```bash
#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SIM_DIR="../sim"
RTL_DIR="../rtl"
TB_DIR="../tb"
LOG_DIR="${SIM_DIR}/log"

INPUT_HEX="../../CIM-sw-version1/sw/train_quantize/route_b_output/input_0.hex"
WEIGHT_HEX="../../CIM-sw-version1/sw/train_quantize/route_b_output/fc1_weight_int8.hex"
FC1_BIAS_HEX="../../CIM-sw-version1/sw/train_quantize/route_b_output/fc1_bias_int32.hex"
FC1_ACC_HEX="../../CIM-sw-version1/sw/train_quantize/route_b_output/fc1_acc_0.hex"

mkdir -p "${SIM_DIR}"
mkdir -p "${LOG_DIR}"

vcs -full64 -sverilog -timescale=1ns/1ps \
    -debug_access+all \
    -o "${SIM_DIR}/fc1_cim_core_block_simv" \
    "${RTL_DIR}/package.sv" \
    "${RTL_DIR}/input_buffer.sv" \
    "${RTL_DIR}/fc1_weight_bank.sv" \
    "${RTL_DIR}/fc1_bias_bank.sv" \
    "${RTL_DIR}/cim_tile.sv" \
    "${RTL_DIR}/psum_accum.sv" \
    "${RTL_DIR}/fc1_cim_core_block.sv" \
    "${TB_DIR}/tb_fc1_cim_core_block.sv" \
    2>&1 | tee "${LOG_DIR}/compile_tb_fc1_cim_core_block.log"

"${SIM_DIR}/fc1_cim_core_block_simv" \
    +INPUT_HEX_FILE="${INPUT_HEX}" \
    +WEIGHT_HEX_FILE="${WEIGHT_HEX}" \
    +FC1_BIAS_FILE="${FC1_BIAS_HEX}" \
    +FC1_ACC_FILE="${FC1_ACC_HEX}" \
    2>&1 | tee "${LOG_DIR}/sim_tb_fc1_cim_core_block.log"
```

我建议你顺手加上的还有这一点：

- **脚本开头先 `cd` 到脚本目录**  
  这是因为你之前已经碰到过相对路径/日志目录的问题；把脚本固定到自身目录执行，会更稳。之前对类似脚本的排查里，根因就是 `../sim`、`../rtl` 这些路径依赖“你从哪个目录启动脚本”。

所以结论是：

- **为了 `ob=0..7` 全验证，脚本本身不用改测试流程**
- **但如果你当前脚本还没带 bias bank / bias plusarg，必须改**
- **最好再加上 `SCRIPT_DIR` 那两行，避免路径问题**

你跑完后，日志里预期会从原来的单块 `PASS`，变成全块扫描后的总 `PASS/FAIL` 汇总，因为 testbench 主流程已经从固定 `ob=0` 改成循环 `ob=0..7` 了。

用户:
run_fc1_cim_core_block.sh，和fc1_cim_core_block.sv需要修改吗
AI助手:
我先对照你工程里这两个文件现在的写法，判断是“必须改”还是“可不改但建议改”，然后把最小改动方案告诉你。

用户:

AI助手:
结论先说：

**为了做 `ob=0..7` 全验证，`fc1_cim_core_block.sv` 通常不用改；`run_fc1_cim_core_block.sh` 可能要小改，也可能不用改，取决于你现在脚本里有没有把新的 TB 编进去，以及有没有正确传文件路径。**

### 1. `fc1_cim_core_block.sv` 要不要改
如果你现在用的是**已经加上 bias 的版本**，那这次做 `ob=0..7` 全验证时，**不用再改它**。

因为你现在这版模块里：
- 状态机已经能从 `IDLE -> CLEAR -> ACCUM -> DONE`
- `DONE` 状态会拉高 `done`
- 输出已经是 `psum[g] + bias_block[g]`，也就是带 bias 的 `fc1_acc_block`  

所以这一步“单实例扫 `ob=0..7`”本质上只是**验证不同 `ob_sel` 下的输出块是否都和 golden 对齐**，不需要再改 DUT 逻辑。

只有两种情况你才需要再动 `fc1_cim_core_block.sv`：
- 你当前本地文件还不是这个“bias 已接上”的版本；
- 你里面还保留了顶层冗余的 `input_file / weight_file / bias_file` plusarg 逻辑，想顺手清理结构。之前也提过，这些其实可以删掉，让子模块自己处理更干净。

### 2. `run_fc1_cim_core_block.sh` 要不要改
**大概率建议改一下，但改动很小。**

原因不是 DUT 变了，而是你现在的 TB 已经从“只测 `ob=0`”变成了“循环测 `ob=0..7`”，脚本最好保证两件事：

第一，**编译的还是你现在这个 `tb_fc1_cim_core_block.sv`**。  
你之前给出的脚本范式里，确实就是编译：
- `package.sv`
- `input_buffer.sv`
- `fc1_weight_bank.sv`
- `cim_tile.sv`
- `psum_accum.sv`
- `fc1_cim_core_block.sv`
- `tb_fc1_cim_core_block.sv` 

如果你只是替换 TB 文件内容，那么**脚本的编译列表本身不用改**。

第二，**最好给 TB 传 `+FC1_ACC_FILE=...`**。  
因为你现在 TB 是 plusarg 读 golden 的，默认路径是一个老路径；如果你本地实际 golden 在 `hw/route_b_output/fc1_acc_0.hex`，那脚本里最好显式传路径，不要赌默认路径正好对。你现在这个 TB 的读法就是：

- 默认 `fc1_acc_file = .../fc1_acc_0.hex`
- 但允许 `+FC1_ACC_FILE=%s` 覆盖 

所以脚本里最好加这一段。

### 3. 那 `+FC1_BIAS_FILE` 要不要传
这个要看你当前 `fc1_bias_bank.sv` 的实现。

如果 `fc1_bias_bank`：
- 自己内部支持 `+FC1_BIAS_FILE`
- 且默认路径不一定对

那脚本里**最好传**。  
你之前分析这个问题时也提到过：旧版 `run_fc1_cim_core_block.sh` 只传了 `INPUT_HEX_FILE / WEIGHT_HEX_FILE / FC1_ACC_FILE`，没有传 bias 路径，这会让 bias 文件加载不稳。 

但如果你的 `fc1_bias_bank.sv` 默认路径已经是对的，而且内部自己读文件没问题，那么从“做 `ob=0..7` 全验证”这个目标来说，**不传也可能能跑通**。

---

## 我的建议
你现在最稳的做法是：

- **`fc1_cim_core_block.sv`：不改**
- **`run_fc1_cim_core_block.sh`：小改，显式传所有文件路径**
- **真正需要改的是 `tb_fc1_cim_core_block.sv`**

---

## 推荐脚本版本
你可以把脚本整理成这样：

```bash
#!/usr/bin/env bash
set -e

SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

INPUT_HEX=../route_b_output/input_0.hex
WEIGHT_HEX=../route_b_output/fc1_weight_int8.hex
BIAS_HEX=../route_b_output/fc1_bias_int32.hex
FC1_ACC_HEX=../route_b_output/fc1_acc_0.hex

mkdir -p ${SIM_DIR}
mkdir -p ${SIM_DIR}/log

vcs -full64 -sverilog -timescale=1ns/1ps \
    -debug_access+all \
    -o ${SIM_DIR}/fc1_cim_core_block_simv \
    ${RTL_DIR}/package.sv \
    ${RTL_DIR}/input_buffer.sv \
    ${RTL_DIR}/fc1_weight_bank.sv \
    ${RTL_DIR}/fc1_bias_bank.sv \
    ${RTL_DIR}/cim_tile.sv \
    ${RTL_DIR}/psum_accum.sv \
    ${RTL_DIR}/fc1_cim_core_block.sv \
    ${TB_DIR}/tb_fc1_cim_core_block.sv \
    2>&1 | tee ${SIM_DIR}/log/compile_tb_fc1_cim_core_block.log

${SIM_DIR}/fc1_cim_core_block_simv \
    +INPUT_HEX_FILE=${INPUT_HEX} \
    +WEIGHT_HEX_FILE=${WEIGHT_HEX} \
    +FC1_BIAS_FILE=${BIAS_HEX} \
    +FC1_ACC_FILE=${FC1_ACC_HEX} \
    2>&1 | tee ${SIM_DIR}/log/sim_tb_fc1_cim_core_block.log
```

### 这个版本相比你旧脚本的区别
- 加上了 `fc1_bias_bank.sv` 编译项
- 运行时显式传：
  - `+INPUT_HEX_FILE`
  - `+WEIGHT_HEX_FILE`
  - `+FC1_BIAS_FILE`
  - `+FC1_ACC_FILE`

这样最不容易再被路径问题坑。

---

## 最终判断
你这一步要做 `ob=0..7` 全验证时：

- **`tb_fc1_cim_core_block.sv`：必须改**
- **`run_fc1_cim_core_block.sh`：建议改，主要是传路径更稳**
- **`fc1_cim_core_block.sv`：如果你现在已经是 bias 正确接入的版本，就不用改** 

你把当前 `run_fc1_cim_core_block.sh` 贴出来的话，我可以直接按你工程目录帮你改成可运行的最终版。

用户:
单个的0-7成功了对吧，然后简单的实例化两个怎么做
AI助手:
我先看一下你刚上传的全扫描仿真日志，确认 `ob=0..7` 是不是真的全部过了，然后再给你“两路简单实例化”的最小实现方案。

用户:

AI助手:
对，**单个 `fc1_cim_core_block` 扫 `ob=0..7` 已经成功了**。

你刚上传的日志里，最后一轮 `ob_sel=7` 全部都是 `MATCH`，并且结尾明确写了：

`PASS: fc1_cim_core_block matches all output blocks in fc1_acc_0.hex.`  

这说明你现在已经验证了：

- 单个 block 结构是对的
- `ob_sel` 选择不同输出块也是对的
- `global_idx = ob * 16 + i` 这套对拍方式是成立的

所以下一步做“**简单实例化两个**”完全合理。

---

# 你现在这一步怎么做最合适

你说的“简单实例化两个”，我建议先做成：

- 同一个 `start`
- 两个 `fc1_cim_core_block`
- 一个跑 `ob=0`
- 一个跑 `ob=1`
- 两个同时开始
- 各自输出 16 个结果
- TB 里分别对拍 `fc1_acc_0.hex[0:15]` 和 `fc1_acc_0.hex[16:31]`

这是最小、最稳、最好调试的两路并行版本。

---

# 先说明一下这一步的本质

你现在不是在做“两个 tile 并行”，而是在做：

**两个 output block 并行。**

也就是：

- 第 1 个实例负责 hidden[0:15]
- 第 2 个实例负责 hidden[16:31]

两者共享同一个输入样本，只是权重块不同。这个方向是对的。

---

# 最小实现方式

建议你新建一个模块，比如：

## `fc1_cim_core_block_dual.sv`

```systemverilog id="5r03lv"
module fc1_cim_core_block_dual (
    input  logic clk,
    input  logic rst_n,
    input  logic start,

    output logic busy,
    output logic done,

    output logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0]
        fc1_acc_block0 [0:mnist_cim_pkg::TILE_OUTPUT_SIZE-1],

    output logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0]
        fc1_acc_block1 [0:mnist_cim_pkg::TILE_OUTPUT_SIZE-1]
);
  import mnist_cim_pkg::*;

  logic busy0, done0;
  logic busy1, done1;

  logic [$clog2(N_OUTPUT_BLOCKS)-1:0] ob_sel0, ob_sel1;

  assign ob_sel0 = 0;
  assign ob_sel1 = 1;

  fc1_cim_core_block u_block0 (
      .clk(clk),
      .rst_n(rst_n),
      .start(start),
      .ob_sel(ob_sel0),
      .busy(busy0),
      .done(done0),
      .fc1_acc_block(fc1_acc_block0)
  );

  fc1_cim_core_block u_block1 (
      .clk(clk),
      .rst_n(rst_n),
      .start(start),
      .ob_sel(ob_sel1),
      .busy(busy1),
      .done(done1),
      .fc1_acc_block(fc1_acc_block1)
  );

  assign busy = busy0 | busy1;
  assign done = done0 & done1;

endmodule
```

这个版本故意保持最简单：

- `ob_sel0 = 0`
- `ob_sel1 = 1`

先别做可配置，先把“两路并行”概念跑通。

---

# 对应 testbench 怎么写

建议新建：

## `tb_fc1_cim_core_block_dual.sv`

```systemverilog id="caagbv"
`timescale 1ns / 1ps

module tb_fc1_cim_core_block_dual;
  import mnist_cim_pkg::*;

  logic clk;
  logic rst_n;
  logic start;
  logic busy;
  logic done;

  logic signed [PSUM_WIDTH-1:0] fc1_acc_block0[0:TILE_OUTPUT_SIZE-1];
  logic signed [PSUM_WIDTH-1:0] fc1_acc_block1[0:TILE_OUTPUT_SIZE-1];

  logic signed [PSUM_WIDTH-1:0] ref_fc1_acc_mem[0:HIDDEN_DIM-1];

  string fc1_acc_file;

  integer i;
  integer error_count;

  fc1_cim_core_block_dual dut (
      .clk(clk),
      .rst_n(rst_n),
      .start(start),
      .busy(busy),
      .done(done),
      .fc1_acc_block0(fc1_acc_block0),
      .fc1_acc_block1(fc1_acc_block1)
  );

  initial clk = 1'b0;
  always #5 clk = ~clk;

  initial begin
    fc1_acc_file = "../../CIM-sw-version1/sw/train_quantize/route_b_output/fc1_acc_0.hex";

    if ($value$plusargs("FC1_ACC_FILE=%s", fc1_acc_file)) begin
      $display("TB using FC1_ACC_FILE: %s", fc1_acc_file);
    end else begin
      $display("TB using default FC1_ACC_FILE: %s", fc1_acc_file);
    end

    $readmemh(fc1_acc_file, ref_fc1_acc_mem);
  end

  initial begin
    error_count = 0;

    rst_n = 1'b0;
    start = 1'b0;

    #12;
    rst_n = 1'b1;

    @(posedge clk);
    start <= 1'b1;

    @(posedge clk);
    start <= 1'b0;

    wait(done == 1'b1);
    @(posedge clk);
    #1;

    $display("Checking block0 (ob=0) ...");
    for (i = 0; i < TILE_OUTPUT_SIZE; i = i + 1) begin
      if (fc1_acc_block0[i] !== ref_fc1_acc_mem[i]) begin
        $display("ERROR block0 idx=%0d got=%0d expected=%0d",
                 i, fc1_acc_block0[i], ref_fc1_acc_mem[i]);
        error_count = error_count + 1;
      end else begin
        $display("MATCH block0 idx=%0d value=%0d",
                 i, fc1_acc_block0[i]);
      end
    end

    $display("Checking block1 (ob=1) ...");
    for (i = 0; i < TILE_OUTPUT_SIZE; i = i + 1) begin
      if (fc1_acc_block1[i] !== ref_fc1_acc_mem[TILE_OUTPUT_SIZE + i]) begin
        $display("ERROR block1 idx=%0d got=%0d expected=%0d",
                 i, fc1_acc_block1[i], ref_fc1_acc_mem[TILE_OUTPUT_SIZE + i]);
        error_count = error_count + 1;
      end else begin
        $display("MATCH block1 idx=%0d value=%0d",
                 i, fc1_acc_block1[i]);
      end
    end

    if (error_count == 0)
      $display("PASS: dual fc1_cim_core_block matches ob=0 and ob=1.");
    else
      $display("FAIL: found %0d mismatches.", error_count);

    $finish;
  end

endmodule
```

---

# 你会看到的现象

如果两路并行正确，你会看到：

- 两个实例一起 `busy`
- 等两个都完成后 `done=1`
- `block0` 全部对上 `fc1_acc[0:15]`
- `block1` 全部对上 `fc1_acc[16:31]`

---

# 这个版本为什么值得先做

因为它只验证一件事：

**“多个 `fc1_cim_core_block` 同时实例化后，互不干扰。”**

这一步能帮你确认：

- 多实例权重读取没串
- 多实例输入读取没串
- 多实例 done/busy 聚合没问题
- 仿真层面并行结构成立

---

# 再往下一步怎么扩成 8 个

等这个 dual 版过了，你就可以很自然推广到：

- `fc1_cim_core_block_oct.sv`
- 或者更通用的 `fc1_multi_block_parallel.sv`

思路就是：

- `genvar g`
- `for (g=0; g<PAR_OB; g++)`
- 每个实例 `ob_sel = g`
- 输出拼到 `fc1_acc_all[g*16 + i]`

但你现在先做双实例是对的，因为更容易看日志、定位问题。

---

# 你这一步先不要急着共享 input_buffer

因为现在最重要的是：

**先证明多实例并行在功能上是成立的。**

哪怕当前结构是：

- 每个实例各自带一套 `input_buffer`
- 每个实例各自扫 49 个 `ib`

也没关系。这个版本的目标不是省资源，而是先跑通。

等你双实例、八实例都过了，再做：

- 共享 input buffer
- 共享 ib controller
- 广播 `x_eff_tile`
- 每个并行单元只保留 weight/bias/psum

这样才是顺的。

---

# 我对你下一步的建议顺序

你现在就按这个顺序最稳：

1. 已完成：`ob=0..7` 单实例全验证  
2. 现在做：双实例 `ob=0` + `ob=1`  
3. 然后扩：4 实例  
4. 再扩：8 实例  
5. 再重构：共享 `input_buffer` 和 `ib controller`  
6. 最后接激活和 FC2

---

# 现在最关键的一句话

**是的，你已经完成了单实例 `ob=0..7` 的全验证；下一步就做两个 `fc1_cim_core_block` 的简单并行实例化。**

如果你愿意，我下一条可以继续直接给你一版 **`run_fc1_cim_core_block_dual.sh`**，这样你可以马上编译跑双实例。

用户:
SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

INPUT_HEX="../route_b_output/input_0.hex"
WEIGHT_HEX="../route_b_output/fc1_weight_int8.hex"
FC1_ACC_HEX="../route_b_output/fc1_acc_0.hex"
FC1_BIAS_HEX="../route_b_output/fc1_bias_int32.hex"

vcs -full64 -sverilog -timescale=1ns/1ps \
	-debug_access+all \
	-o ${SIM_DIR}/fc1_cim_core_block_simv \
	${RTL_DIR}/package.sv \
	${RTL_DIR}/input_buffer.sv \
	${RTL_DIR}/fc1_weight_bank.sv \
	${RTL_DIR}/cim_tile.sv \
	${RTL_DIR}/psum_accum.sv \
	${RTL_DIR}/fc1_cim_core_block.sv \
	${RTL_DIR}/fc1_bias_bank.sv \
	${TB_DIR}/tb_fc1_cim_core_block.sv \
	2>&1 | tee ${SIM_DIR}/log/compile_tb_fc1_cim_core_block.log

${SIM_DIR}/fc1_cim_core_block_simv +INPUT_HEX_FILE=${INPUT_HEX} +WEIGHT_HEX_FILE=${WEIGHT_HEX} +FC1_ACC_FILE=${FC1_ACC_HEX} +FC1_BIAS_FILE=${FC1_BIAS_HEX} 2>&1 | tee ${SIM_DIR}/log/sim_tb_fc1_cim_core_block.log仿照这个给我一个新的run.sh
AI助手:
可以，按你现在这个风格，双实例版本的 `run.sh` 直接这样写。

假设你新增了两个文件：

- `rtl/fc1_cim_core_block_dual.sv`
- `tb/tb_fc1_cim_core_block_dual.sv`

那么脚本可以叫：

## `run_fc1_cim_core_block_dual.sh`

```bash
#!/usr/bin/env bash
set -e

SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

INPUT_HEX="../route_b_output/input_0.hex"
WEIGHT_HEX="../route_b_output/fc1_weight_int8.hex"
FC1_ACC_HEX="../route_b_output/fc1_acc_0.hex"
FC1_BIAS_HEX="../route_b_output/fc1_bias_int32.hex"

mkdir -p ${SIM_DIR}
mkdir -p ${SIM_DIR}/log

vcs -full64 -sverilog -timescale=1ns/1ps \
    -debug_access+all \
    -o ${SIM_DIR}/fc1_cim_core_block_dual_simv \
    ${RTL_DIR}/package.sv \
    ${RTL_DIR}/input_buffer.sv \
    ${RTL_DIR}/fc1_weight_bank.sv \
    ${RTL_DIR}/cim_tile.sv \
    ${RTL_DIR}/psum_accum.sv \
    ${RTL_DIR}/fc1_bias_bank.sv \
    ${RTL_DIR}/fc1_cim_core_block.sv \
    ${RTL_DIR}/fc1_cim_core_block_dual.sv \
    ${TB_DIR}/tb_fc1_cim_core_block_dual.sv \
    2>&1 | tee ${SIM_DIR}/log/compile_tb_fc1_cim_core_block_dual.log

${SIM_DIR}/fc1_cim_core_block_dual_simv \
    +INPUT_HEX_FILE=${INPUT_HEX} \
    +WEIGHT_HEX_FILE=${WEIGHT_HEX} \
    +FC1_ACC_FILE=${FC1_ACC_HEX} \
    +FC1_BIAS_FILE=${FC1_BIAS_HEX} \
    2>&1 | tee ${SIM_DIR}/log/sim_tb_fc1_cim_core_block_dual.log
```

### 这版和你原来脚本的区别
主要就三处：

- 可执行文件名改成了  
  `fc1_cim_core_block_dual_simv`
- 多编译了一个顶层 RTL：  
  `fc1_cim_core_block_dual.sv`
- TB 换成了：  
  `tb_fc1_cim_core_block_dual.sv`

---

## 目录里要对应好
你需要保证这几个文件存在：

```bash
../rtl/package.sv
../rtl/input_buffer.sv
../rtl/fc1_weight_bank.sv
../rtl/cim_tile.sv
../rtl/psum_accum.sv
../rtl/fc1_bias_bank.sv
../rtl/fc1_cim_core_block.sv
../rtl/fc1_cim_core_block_dual.sv
../tb/tb_fc1_cim_core_block_dual.sv
```

---

## 运行方式
直接：

```bash
bash run_fc1_cim_core_block_dual.sh
```

如果脚本已有执行权限，也可以：

```bash
./run_fc1_cim_core_block_dual.sh
```

---

## 你跑通后应该看到的结果
如果 dual 版正确，日志最后应该类似：

- `MATCH block0 idx=...`
- `MATCH block1 idx=...`
- `PASS: dual fc1_cim_core_block matches ob=0 and ob=1.`

---

## 一个小建议
你现在这个 dual 版先固定：

- `u_block0 -> ob_sel=0`
- `u_block1 -> ob_sel=1`

就够了。先别急着把脚本做成可传 `OB0/OB1` 参数的版本，先把功能跑通最重要。

你要的话，我下一条直接把 **`fc1_cim_core_block_dual.sv` 和 `tb_fc1_cim_core_block_dual.sv`** 再按你当前工程风格整理成最终可复制版。

用户:
成功了对吧，帮我做实例化多个。根据数据格式，你建议如何输入？（你可以看一下我上传的数据格式说明文件）
用户:
Make sure to include fileciteturn5file0 in your response to cite this file, or to surface it as a link.# route_b_output 使用指南

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

用户:
The files uploaded by the user in the current message have been fully loaded. Searching won't provide additional information from these specific files. However, remember that some of the previously uploaded files were not fully loaded, so you should invoke msearch to search their full content as needed.
AI助手:
我先根据你刚上传的双实例日志确认“两路并行是否真的过了”，再结合你这个 `route_b_output` 说明，给你一个“多实例应该怎么喂数据”的建议，不只是给代码。

用户:

AI助手:
对，**双实例已经成功了**。你上传的日志里，`block0` 和 `block1` 都是逐项 `MATCH`，最后明确写了  
`PASS: dual fc1_cim_core_block matches ob=0 and ob=1.`   

所以你现在可以很有把握地进入“**实例化多个 output block**”这一步。

---

## 我建议你现在怎么做

你下一步不要直接上“共享 input buffer + 共享 ib controller”的最终优化版。  
先做一个**功能优先版**：

- 一次实例化 `PAR_OB` 个 `fc1_cim_core_block`
- 每个实例固定不同的 `ob_sel`
- 同时启动
- 输出拼成完整 `fc1_acc_all[128]`

也就是：

- `PAR_OB=2` 时：算 `ob=0,1`
- `PAR_OB=4` 时：算 `ob=0,1,2,3`
- `PAR_OB=8` 时：一次算完整层 `ob=0..7`

这一步最适合你现在，因为双实例已经证明“多实例互不干扰”了。 

---

# 根据你上传的数据格式，我建议输入怎么做

你这个问题很关键。  
结合你上传的 `route_b_output` 使用说明，我的建议是：

## 当前阶段最推荐：**方式 A，预装载输入 + 广播同一个样本**
也就是说明文档里的 `input_i.hex -> 先读入 input RAM / input_buffer，再开始计算`。文档里明确给了两种方式，其中方式 A 是“预装载到输入 RAM”，方式 B 才是流式输入。

我建议你现在选 **方式 A**，理由是：

### 1. 你的数据天然是静态文件
`input_i.hex` 每个样本就是 **784 行、每行一个 signed int8**，适合 `$readmemh`。

### 2. 你现在并行的是多个 `ob`
多个 output block **共享同一个输入向量**，只是权重不同。  
所以最合理的数据流是：

- 先把 `input_0.hex` 装进一个输入缓存
- `ib=0..48` 顺序读出每个 16 元素 input tile
- 广播给所有并行 `ob` 实例

### 3. 它和你当前 `fc1_cim_core_block` 的结构最兼容
你现在的 block 已经是“内部自己扫 49 个 `ib`”。  
所以当前这一步，哪怕先不共享 `input_buffer`，也可以先跑通。  
但从**架构方向**上，你后面应该收敛到：

- 一个共享 `input_buffer`
- 一个共享 `ib controller`
- 多个 `ob` 计算实例共用同一份 `x_eff_tile`

这和说明文档里“输入按 784 个 int8、行优先展开，推荐先预装载到输入 RAM”的使用方式是完全一致的。 

---

# 我不建议你现在怎么输入

## 先不要把它做成“逐拍 streaming 输入接口”作为主架构
虽然文档里也允许流式输入。  
但你现在如果这么做，会同时引入这些问题：

- 输入 valid/ready 时序
- 784 拍装载控制
- 装载完成与开始计算的同步
- 多实例并行时输入分发

这会把问题复杂化。  
你当前阶段最重要的是把 **FC1 多 ob 并行** 功能跑通，不是先做总线化接口。

---

# 结合你的数据格式，输入应该遵守什么规则

说明文档里有几个点你后面做多实例时必须继续保持：

### 1. 输入是 `signed int8`
`input_i.hex` 每行 2 位十六进制，表示 signed int8。尤其 `0x80` 表示 `-128`，不是 128。 

### 2. 输入长度是 784
这就是你 `49 x 16` 分块的来源。说明文档也明确第一层输入向量形状是 `[784]`。

### 3. 展平顺序是 row-major
`index = row * 28 + col`。你当前 `ib` 的切分必须严格按这个顺序，不要做别的重排。

### 4. 当前数据更适合逻辑布局验证，不是最终物理 tile 烧写格式
说明文档明确说现在这些文件适合逻辑验证；如果将来是实际 CIM 物理阵列，可能还要做 tile 重排。  
所以你现在做“多实例 ob 并行”，应该继续按**逻辑布局**来，不要过早引入物理映射重排。

---

# 我建议你的多实例顶层写成这样

下面这版就是你现在最适合的“实例化多个”的版本。

## `fc1_multi_block_parallel.sv`

```systemverilog
module fc1_multi_block_parallel #(
    parameter int PAR_OB = 8
) (
    input  logic clk,
    input  logic rst_n,
    input  logic start,

    output logic busy,
    output logic done,

    output logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0]
        fc1_acc_all [0:mnist_cim_pkg::HIDDEN_DIM-1]
);
  import mnist_cim_pkg::*;

  logic [PAR_OB-1:0] busy_vec;
  logic [PAR_OB-1:0] done_vec;

  logic signed [PSUM_WIDTH-1:0]
      fc1_acc_block [0:PAR_OB-1][0:TILE_OUTPUT_SIZE-1];

  genvar g_ob, g_idx;
  generate
    for (g_ob = 0; g_ob < PAR_OB; g_ob = g_ob + 1) begin : GEN_OB_CORE
      fc1_cim_core_block u_fc1_cim_core_block (
          .clk(clk),
          .rst_n(rst_n),
          .start(start),
          .ob_sel(g_ob[$clog2(N_OUTPUT_BLOCKS)-1:0]),
          .busy(busy_vec[g_ob]),
          .done(done_vec[g_ob]),
          .fc1_acc_block(fc1_acc_block[g_ob])
      );

      for (g_idx = 0; g_idx < TILE_OUTPUT_SIZE; g_idx = g_idx + 1) begin : GEN_PACK
        assign fc1_acc_all[g_ob*TILE_OUTPUT_SIZE + g_idx] = fc1_acc_block[g_ob][g_idx];
      end
    end
  endgenerate

  assign busy = |busy_vec;
  assign done = &done_vec;

endmodule
```

---

# 这个版本怎么理解

它本质上就是：

- `g_ob=0` 对应 `ob=0`
- `g_ob=1` 对应 `ob=1`
- ...
- `g_ob=7` 对应 `ob=7`

每个实例各自算 16 个输出，最终拼成 128 维 `fc1_acc_all`。

---

# 但我要明确说一下：这还是“功能优先版”

这版虽然能用，但它的特点是：

- 每个 `fc1_cim_core_block` 里都可能各自带一套 `input_buffer`
- 所以输入被重复读取了 `PAR_OB` 次

这在功能验证阶段是可以接受的。  
但从你的数据格式和目标架构来看，**下一版更合理的输入方式**应该是：

## 共享输入的正式版方向
- 一个 `input_buffer_shared`
- 一个 `ib_counter`
- 每拍给出一个 `x_eff_tile[16]`
- 广播给所有并行 `ob` 实例
- 每个实例只负责读自己的权重块、累加自己的 psum

这才是和“一个样本输入，多个输出块并行计算”最匹配的结构。

---

# 所以我对“输入怎么做”的最终建议

## 你现在立刻该做
继续用你现有的 `input_i.hex` 文件和 `$readmemh` 机制，保持：

- 一个样本一个 `input_0.hex`
- 784 行
- int8
- row-major
- 逻辑顺序切成 `49 x 16`

这和说明文档完全一致。

## 你下一阶段该演进成
### 当前过渡版
- 多个 `fc1_cim_core_block` 简单实例化
- 各自读输入
- 先验证 `fc1_acc_all[128]`

### 再下一版正式结构
- 共享 `input_buffer`
- 共享 `ib controller`
- 广播 `x_eff_tile`
- 并行多个 `ob`

---

# 你接下来怎么验证多实例

你现在可以做 3 步：

### 第一步：`PAR_OB=2`
对拍：

- `fc1_acc_all[0:15]` 对 `fc1_acc_0.hex[0:15]`
- `fc1_acc_all[16:31]` 对 `fc1_acc_0.hex[16:31]`

### 第二步：`PAR_OB=4`
对拍前 64 个值。

### 第三步：`PAR_OB=8`
一次对拍完整 128 个 `fc1_acc_0.hex`。

说明文档里也明确指出，`fc1_acc_i.hex` 是第一层最关键的中间 golden，最适合先拿来定位第一层 MAC 是否正确。

---

# 一句结论

**双实例已经成功；按你当前的数据格式，最推荐的输入方式是“预装载 input_i.hex 到输入缓存，再把同一个输入块广播给多个 ob 实例”。**

这是你现在最稳、也最符合 `route_b_output` 这套数据组织方式的路线。 

下一条我可以直接给你：
**`fc1_multi_block_parallel.sv + tb_fc1_multi_block_parallel.sv + run_fc1_multi_block_parallel.sh` 的可运行版本。**

用户:
请给我这三个代码
AI助手:
可以，下面直接给你这三个文件的可复制版本：

- `fc1_multi_block_parallel.sv`
- `tb_fc1_multi_block_parallel.sv`
- `run_fc1_multi_block_parallel.sh`

这版先按你当前阶段来做：

- 默认 `PAR_OB = 8`
- 也支持改成 `2` 或 `4`
- 仍然是**简单多实例化**
- 还**没有共享 input_buffer / ib controller**
- 用 `fc1_acc_0.hex` 做 golden 对拍

---

## 1) `fc1_multi_block_parallel.sv`

```systemverilog
module fc1_multi_block_parallel #(
    parameter int PAR_OB = 8
) (
    input  logic clk,
    input  logic rst_n,
    input  logic start,

    output logic busy,
    output logic done,

    output logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0]
        fc1_acc_all [0:PAR_OB*mnist_cim_pkg::TILE_OUTPUT_SIZE-1]
);
  import mnist_cim_pkg::*;

  logic [PAR_OB-1:0] busy_vec;
  logic [PAR_OB-1:0] done_vec;

  logic signed [PSUM_WIDTH-1:0]
      fc1_acc_block [0:PAR_OB-1][0:TILE_OUTPUT_SIZE-1];

  genvar g_ob, g_idx;
  generate
    for (g_ob = 0; g_ob < PAR_OB; g_ob = g_ob + 1) begin : GEN_OB_CORE
      fc1_cim_core_block u_fc1_cim_core_block (
          .clk(clk),
          .rst_n(rst_n),
          .start(start),
          .ob_sel(g_ob[$clog2(N_OUTPUT_BLOCKS)-1:0]),
          .busy(busy_vec[g_ob]),
          .done(done_vec[g_ob]),
          .fc1_acc_block(fc1_acc_block[g_ob])
      );

      for (g_idx = 0; g_idx < TILE_OUTPUT_SIZE; g_idx = g_idx + 1) begin : GEN_PACK
        assign fc1_acc_all[g_ob*TILE_OUTPUT_SIZE + g_idx] = fc1_acc_block[g_ob][g_idx];
      end
    end
  endgenerate

  assign busy = |busy_vec;
  assign done = &done_vec;

endmodule
```

---

## 2) `tb_fc1_multi_block_parallel.sv`

这版 testbench 支持改 `PAR_OB`。  
如果你设成：

- `PAR_OB=2`，检查前 32 个输出
- `PAR_OB=4`，检查前 64 个输出
- `PAR_OB=8`，检查完整 128 个输出

```systemverilog
`timescale 1ns / 1ps

module tb_fc1_multi_block_parallel;
  import mnist_cim_pkg::*;

  parameter int PAR_OB = 8;
  localparam int OUT_DIM = PAR_OB * TILE_OUTPUT_SIZE;

  logic clk;
  logic rst_n;
  logic start;
  logic busy;
  logic done;

  logic signed [PSUM_WIDTH-1:0] fc1_acc_all[0:OUT_DIM-1];
  logic signed [PSUM_WIDTH-1:0] ref_fc1_acc_mem[0:HIDDEN_DIM-1];

  string fc1_acc_file;

  integer i;
  integer error_count;

  fc1_multi_block_parallel #(
      .PAR_OB(PAR_OB)
  ) dut (
      .clk(clk),
      .rst_n(rst_n),
      .start(start),
      .busy(busy),
      .done(done),
      .fc1_acc_all(fc1_acc_all)
  );

  initial clk = 1'b0;
  always #5 clk = ~clk;

  initial begin
    fc1_acc_file = "../../CIM-sw-version1/sw/train_quantize/route_b_output/fc1_acc_0.hex";

    if ($value$plusargs("FC1_ACC_FILE=%s", fc1_acc_file)) begin
      $display("TB using FC1_ACC_FILE: %s", fc1_acc_file);
    end else begin
      $display("TB using default FC1_ACC_FILE: %s", fc1_acc_file);
    end

    $readmemh(fc1_acc_file, ref_fc1_acc_mem);
  end

  initial begin
    $display("time=%0t : TB start, PAR_OB=%0d, OUT_DIM=%0d", $time, PAR_OB, OUT_DIM);
    $monitor("time=%0t clk=%0b rst_n=%0b start=%0b busy=%0b done=%0b",
             $time, clk, rst_n, start, busy, done);
  end

  initial begin
    error_count = 0;

    rst_n = 1'b0;
    start = 1'b0;

    #12;
    rst_n = 1'b1;

    @(posedge clk);
    start <= 1'b1;

    @(posedge clk);
    start <= 1'b0;

    wait(done == 1'b1);
    @(posedge clk);
    #1;

    $display("Checking fc1_acc_all ...");
    for (i = 0; i < OUT_DIM; i = i + 1) begin
      if (fc1_acc_all[i] !== ref_fc1_acc_mem[i]) begin
        $display("ERROR idx=%0d got=%0d expected=%0d",
                 i, fc1_acc_all[i], ref_fc1_acc_mem[i]);
        error_count = error_count + 1;
      end else begin
        $display("MATCH idx=%0d value=%0d",
                 i, fc1_acc_all[i]);
      end
    end

    if (error_count == 0)
      $display("PASS: fc1_multi_block_parallel matches first %0d outputs of fc1_acc_0.hex.", OUT_DIM);
    else
      $display("FAIL: found %0d mismatches.", error_count);

    $finish;
  end

endmodule
```

---

## 3) `run_fc1_multi_block_parallel.sh`

这个脚本先按 `PAR_OB=8` 写。  
如果你想先跑 2 路或 4 路，只需要改编译命令里的 `+define+PAR_OB_2` / `+define+PAR_OB_4`，或者更简单，直接改 testbench 里的 `parameter int PAR_OB = 8;`。

先给你最直接版：

```bash
#!/usr/bin/env bash
set -e

SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

INPUT_HEX="../route_b_output/input_0.hex"
WEIGHT_HEX="../route_b_output/fc1_weight_int8.hex"
FC1_ACC_HEX="../route_b_output/fc1_acc_0.hex"
FC1_BIAS_HEX="../route_b_output/fc1_bias_int32.hex"

mkdir -p ${SIM_DIR}
mkdir -p ${SIM_DIR}/log

vcs -full64 -sverilog -timescale=1ns/1ps \
    -debug_access+all \
    -o ${SIM_DIR}/fc1_multi_block_parallel_simv \
    ${RTL_DIR}/package.sv \
    ${RTL_DIR}/input_buffer.sv \
    ${RTL_DIR}/fc1_weight_bank.sv \
    ${RTL_DIR}/cim_tile.sv \
    ${RTL_DIR}/psum_accum.sv \
    ${RTL_DIR}/fc1_bias_bank.sv \
    ${RTL_DIR}/fc1_cim_core_block.sv \
    ${RTL_DIR}/fc1_multi_block_parallel.sv \
    ${TB_DIR}/tb_fc1_multi_block_parallel.sv \
    2>&1 | tee ${SIM_DIR}/log/compile_tb_fc1_multi_block_parallel.log

${SIM_DIR}/fc1_multi_block_parallel_simv \
    +INPUT_HEX_FILE=${INPUT_HEX} \
    +WEIGHT_HEX_FILE=${WEIGHT_HEX} \
    +FC1_ACC_FILE=${FC1_ACC_HEX} \
    +FC1_BIAS_FILE=${FC1_BIAS_HEX} \
    2>&1 | tee ${SIM_DIR}/log/sim_tb_fc1_multi_block_parallel.log
```

---

## 你现在怎么用

### 先跑 2 路
把 `tb_fc1_multi_block_parallel.sv` 里这行：

```systemverilog
parameter int PAR_OB = 8;
```

改成：

```systemverilog
parameter int PAR_OB = 2;
```

### 再跑 4 路
改成：

```systemverilog
parameter int PAR_OB = 4;
```

### 最后跑 8 路
改回：

```systemverilog
parameter int PAR_OB = 8;
```

---

## 一个小提醒
这版 `fc1_acc_all` 的长度是：

```systemverilog
[0:PAR_OB*TILE_OUTPUT_SIZE-1]
```

所以当 `PAR_OB < 8` 时，输出数组不是 128 长，而是前一段有效长度。  
这是我故意这么写的，因为：

- `PAR_OB=2` 时更容易调试
- TB 只检查前 32 个值
- 不需要处理后面没实例化的空位

如果你想，我下一条可以继续帮你做第二版：

**固定输出始终是 128 维，未实例化部分补 0**  
或者直接进入下一步：

**共享 `input_buffer` + 共享 `ib controller` 的结构重构版。**

用户:
Chronologic VCS simulator copyright 1991-2018
Contains Synopsys proprietary information.
Compiler version O-2018.09-SP2_Full64; Runtime version O-2018.09-SP2_Full64;  Mar  7 23:36 2026
Using INPUT_HEX_FILE from plusarg: ../route_b_output/input_0.hex
Using WEIGHT_HEX_FILE from plusarg: ../route_b_output/fc1_weight_int8.hex
Using FC1_BIAS_FILE from plusarg: ../route_b_output/fc1_bias_int32.hex
Using INPUT_HEX_FILE from plusarg: ../route_b_output/input_0.hex
Using WEIGHT_HEX_FILE from plusarg: ../route_b_output/fc1_weight_int8.hex
Using FC1_BIAS_FILE from plusarg: ../route_b_output/fc1_bias_int32.hex
Using INPUT_HEX_FILE from plusarg: ../route_b_output/input_0.hex
Using WEIGHT_HEX_FILE from plusarg: ../route_b_output/fc1_weight_int8.hex
Using FC1_BIAS_FILE from plusarg: ../route_b_output/fc1_bias_int32.hex
Using INPUT_HEX_FILE from plusarg: ../route_b_output/input_0.hex
Using WEIGHT_HEX_FILE from plusarg: ../route_b_output/fc1_weight_int8.hex
Using FC1_BIAS_FILE from plusarg: ../route_b_output/fc1_bias_int32.hex
Using INPUT_HEX_FILE from plusarg: ../route_b_output/input_0.hex
Using WEIGHT_HEX_FILE from plusarg: ../route_b_output/fc1_weight_int8.hex
Using FC1_BIAS_FILE from plusarg: ../route_b_output/fc1_bias_int32.hex
Using INPUT_HEX_FILE from plusarg: ../route_b_output/input_0.hex
Using WEIGHT_HEX_FILE from plusarg: ../route_b_output/fc1_weight_int8.hex
Using FC1_BIAS_FILE from plusarg: ../route_b_output/fc1_bias_int32.hex
Using INPUT_HEX_FILE from plusarg: ../route_b_output/input_0.hex
Using WEIGHT_HEX_FILE from plusarg: ../route_b_output/fc1_weight_int8.hex
Using FC1_BIAS_FILE from plusarg: ../route_b_output/fc1_bias_int32.hex
Using INPUT_HEX_FILE from plusarg: ../route_b_output/input_0.hex
Using WEIGHT_HEX_FILE from plusarg: ../route_b_output/fc1_weight_int8.hex
Using FC1_BIAS_FILE from plusarg: ../route_b_output/fc1_bias_int32.hex
Using INPUT_HEX_FILE from plusarg: ../route_b_output/input_0.hex
Using WEIGHT_HEX_FILE from plusarg: ../route_b_output/fc1_weight_int8.hex
Using FC1_BIAS_FILE from plusarg: ../route_b_output/fc1_bias_int32.hex
Using INPUT_HEX_FILE from plusarg: ../route_b_output/input_0.hex
Using WEIGHT_HEX_FILE from plusarg: ../route_b_output/fc1_weight_int8.hex
Using FC1_BIAS_FILE from plusarg: ../route_b_output/fc1_bias_int32.hex
Using INPUT_HEX_FILE from plusarg: ../route_b_output/input_0.hex
Using WEIGHT_HEX_FILE from plusarg: ../route_b_output/fc1_weight_int8.hex
Using FC1_BIAS_FILE from plusarg: ../route_b_output/fc1_bias_int32.hex
Using INPUT_HEX_FILE from plusarg: ../route_b_output/input_0.hex
Using WEIGHT_HEX_FILE from plusarg: ../route_b_output/fc1_weight_int8.hex
Using FC1_BIAS_FILE from plusarg: ../route_b_output/fc1_bias_int32.hex
Using INPUT_HEX_FILE from plusarg: ../route_b_output/input_0.hex
Using WEIGHT_HEX_FILE from plusarg: ../route_b_output/fc1_weight_int8.hex
Using FC1_BIAS_FILE from plusarg: ../route_b_output/fc1_bias_int32.hex
Using INPUT_HEX_FILE from plusarg: ../route_b_output/input_0.hex
Using WEIGHT_HEX_FILE from plusarg: ../route_b_output/fc1_weight_int8.hex
Using FC1_BIAS_FILE from plusarg: ../route_b_output/fc1_bias_int32.hex
Using INPUT_HEX_FILE from plusarg: ../route_b_output/input_0.hex
Using WEIGHT_HEX_FILE from plusarg: ../route_b_output/fc1_weight_int8.hex
Using FC1_BIAS_FILE from plusarg: ../route_b_output/fc1_bias_int32.hex
Using INPUT_HEX_FILE from plusarg: ../route_b_output/input_0.hex
Using WEIGHT_HEX_FILE from plusarg: ../route_b_output/fc1_weight_int8.hex
Using FC1_BIAS_FILE from plusarg: ../route_b_output/fc1_bias_int32.hex
TB using FC1_ACC_FILE: ../route_b_output/fc1_acc_0.hex
time=0 : TB start, PAR_OB=8, OUT_DIM=128
time=0 clk=0 rst_n=0 start=0 busy=0 done=0
time=5000 clk=1 rst_n=0 start=0 busy=0 done=0
time=10000 clk=0 rst_n=0 start=0 busy=0 done=0
time=12000 clk=0 rst_n=1 start=0 busy=0 done=0
time=15000 clk=1 rst_n=1 start=1 busy=0 done=0
time=20000 clk=0 rst_n=1 start=1 busy=0 done=0
time=25000 clk=1 rst_n=1 start=0 busy=1 done=0
time=30000 clk=0 rst_n=1 start=0 busy=1 done=0
time=35000 clk=1 rst_n=1 start=0 busy=1 done=0
time=40000 clk=0 rst_n=1 start=0 busy=1 done=0
time=45000 clk=1 rst_n=1 start=0 busy=1 done=0
time=50000 clk=0 rst_n=1 start=0 busy=1 done=0
time=55000 clk=1 rst_n=1 start=0 busy=1 done=0
time=60000 clk=0 rst_n=1 start=0 busy=1 done=0
time=65000 clk=1 rst_n=1 start=0 busy=1 done=0
time=70000 clk=0 rst_n=1 start=0 busy=1 done=0
time=75000 clk=1 rst_n=1 start=0 busy=1 done=0
time=80000 clk=0 rst_n=1 start=0 busy=1 done=0
time=85000 clk=1 rst_n=1 start=0 busy=1 done=0
time=90000 clk=0 rst_n=1 start=0 busy=1 done=0
time=95000 clk=1 rst_n=1 start=0 busy=1 done=0
time=100000 clk=0 rst_n=1 start=0 busy=1 done=0
time=105000 clk=1 rst_n=1 start=0 busy=1 done=0
time=110000 clk=0 rst_n=1 start=0 busy=1 done=0
time=115000 clk=1 rst_n=1 start=0 busy=1 done=0
time=120000 clk=0 rst_n=1 start=0 busy=1 done=0
time=125000 clk=1 rst_n=1 start=0 busy=1 done=0
time=130000 clk=0 rst_n=1 start=0 busy=1 done=0
time=135000 clk=1 rst_n=1 start=0 busy=1 done=0
time=140000 clk=0 rst_n=1 start=0 busy=1 done=0
time=145000 clk=1 rst_n=1 start=0 busy=1 done=0
time=150000 clk=0 rst_n=1 start=0 busy=1 done=0
time=155000 clk=1 rst_n=1 start=0 busy=1 done=0
time=160000 clk=0 rst_n=1 start=0 busy=1 done=0
time=165000 clk=1 rst_n=1 start=0 busy=1 done=0
time=170000 clk=0 rst_n=1 start=0 busy=1 done=0
time=175000 clk=1 rst_n=1 start=0 busy=1 done=0
time=180000 clk=0 rst_n=1 start=0 busy=1 done=0
time=185000 clk=1 rst_n=1 start=0 busy=1 done=0
time=190000 clk=0 rst_n=1 start=0 busy=1 done=0
time=195000 clk=1 rst_n=1 start=0 busy=1 done=0
time=200000 clk=0 rst_n=1 start=0 busy=1 done=0
time=205000 clk=1 rst_n=1 start=0 busy=1 done=0
time=210000 clk=0 rst_n=1 start=0 busy=1 done=0
time=215000 clk=1 rst_n=1 start=0 busy=1 done=0
time=220000 clk=0 rst_n=1 start=0 busy=1 done=0
time=225000 clk=1 rst_n=1 start=0 busy=1 done=0
time=230000 clk=0 rst_n=1 start=0 busy=1 done=0
time=235000 clk=1 rst_n=1 start=0 busy=1 done=0
time=240000 clk=0 rst_n=1 start=0 busy=1 done=0
time=245000 clk=1 rst_n=1 start=0 busy=1 done=0
time=250000 clk=0 rst_n=1 start=0 busy=1 done=0
time=255000 clk=1 rst_n=1 start=0 busy=1 done=0
time=260000 clk=0 rst_n=1 start=0 busy=1 done=0
time=265000 clk=1 rst_n=1 start=0 busy=1 done=0
time=270000 clk=0 rst_n=1 start=0 busy=1 done=0
time=275000 clk=1 rst_n=1 start=0 busy=1 done=0
time=280000 clk=0 rst_n=1 start=0 busy=1 done=0
time=285000 clk=1 rst_n=1 start=0 busy=1 done=0
time=290000 clk=0 rst_n=1 start=0 busy=1 done=0
time=295000 clk=1 rst_n=1 start=0 busy=1 done=0
time=300000 clk=0 rst_n=1 start=0 busy=1 done=0
time=305000 clk=1 rst_n=1 start=0 busy=1 done=0
time=310000 clk=0 rst_n=1 start=0 busy=1 done=0
time=315000 clk=1 rst_n=1 start=0 busy=1 done=0
time=320000 clk=0 rst_n=1 start=0 busy=1 done=0
time=325000 clk=1 rst_n=1 start=0 busy=1 done=0
time=330000 clk=0 rst_n=1 start=0 busy=1 done=0
time=335000 clk=1 rst_n=1 start=0 busy=1 done=0
time=340000 clk=0 rst_n=1 start=0 busy=1 done=0
time=345000 clk=1 rst_n=1 start=0 busy=1 done=0
time=350000 clk=0 rst_n=1 start=0 busy=1 done=0
time=355000 clk=1 rst_n=1 start=0 busy=1 done=0
time=360000 clk=0 rst_n=1 start=0 busy=1 done=0
time=365000 clk=1 rst_n=1 start=0 busy=1 done=0
time=370000 clk=0 rst_n=1 start=0 busy=1 done=0
time=375000 clk=1 rst_n=1 start=0 busy=1 done=0
time=380000 clk=0 rst_n=1 start=0 busy=1 done=0
time=385000 clk=1 rst_n=1 start=0 busy=1 done=0
time=390000 clk=0 rst_n=1 start=0 busy=1 done=0
time=395000 clk=1 rst_n=1 start=0 busy=1 done=0
time=400000 clk=0 rst_n=1 start=0 busy=1 done=0
time=405000 clk=1 rst_n=1 start=0 busy=1 done=0
time=410000 clk=0 rst_n=1 start=0 busy=1 done=0
time=415000 clk=1 rst_n=1 start=0 busy=1 done=0
time=420000 clk=0 rst_n=1 start=0 busy=1 done=0
time=425000 clk=1 rst_n=1 start=0 busy=1 done=0
time=430000 clk=0 rst_n=1 start=0 busy=1 done=0
time=435000 clk=1 rst_n=1 start=0 busy=1 done=0
time=440000 clk=0 rst_n=1 start=0 busy=1 done=0
time=445000 clk=1 rst_n=1 start=0 busy=1 done=0
time=450000 clk=0 rst_n=1 start=0 busy=1 done=0
time=455000 clk=1 rst_n=1 start=0 busy=1 done=0
time=460000 clk=0 rst_n=1 start=0 busy=1 done=0
time=465000 clk=1 rst_n=1 start=0 busy=1 done=0
time=470000 clk=0 rst_n=1 start=0 busy=1 done=0
time=475000 clk=1 rst_n=1 start=0 busy=1 done=0
time=480000 clk=0 rst_n=1 start=0 busy=1 done=0
time=485000 clk=1 rst_n=1 start=0 busy=1 done=0
time=490000 clk=0 rst_n=1 start=0 busy=1 done=0
time=495000 clk=1 rst_n=1 start=0 busy=1 done=0
time=500000 clk=0 rst_n=1 start=0 busy=1 done=0
time=505000 clk=1 rst_n=1 start=0 busy=1 done=0
time=510000 clk=0 rst_n=1 start=0 busy=1 done=0
time=515000 clk=1 rst_n=1 start=0 busy=1 done=0
time=520000 clk=0 rst_n=1 start=0 busy=1 done=0
time=525000 clk=1 rst_n=1 start=0 busy=0 done=1
time=530000 clk=0 rst_n=1 start=0 busy=0 done=1
time=535000 clk=1 rst_n=1 start=0 busy=0 done=0
Checking fc1_acc_all ...
MATCH idx=0 value=-30778
MATCH idx=1 value=-33761
MATCH idx=2 value=-26292
MATCH idx=3 value=32890
MATCH idx=4 value=84961
MATCH idx=5 value=82286
MATCH idx=6 value=-42871
MATCH idx=7 value=-83495
MATCH idx=8 value=14476
MATCH idx=9 value=143175
MATCH idx=10 value=50796
MATCH idx=11 value=204030
MATCH idx=12 value=27237
MATCH idx=13 value=-70958
MATCH idx=14 value=-7987
MATCH idx=15 value=69688
MATCH idx=16 value=88057
MATCH idx=17 value=-30428
MATCH idx=18 value=103647
MATCH idx=19 value=74590
MATCH idx=20 value=181469
MATCH idx=21 value=-11999
MATCH idx=22 value=33262
MATCH idx=23 value=22503
MATCH idx=24 value=12667
MATCH idx=25 value=-71001
MATCH idx=26 value=89749
MATCH idx=27 value=-41266
MATCH idx=28 value=17780
MATCH idx=29 value=45129
MATCH idx=30 value=-40512
MATCH idx=31 value=39422
MATCH idx=32 value=-186523
MATCH idx=33 value=84098
MATCH idx=34 value=105599
MATCH idx=35 value=111699
MATCH idx=36 value=129751
MATCH idx=37 value=111784
MATCH idx=38 value=35579
MATCH idx=39 value=-62725
MATCH idx=40 value=-7048
MATCH idx=41 value=35904
MATCH idx=42 value=-33587
MATCH idx=43 value=-17150
MATCH idx=44 value=-131331
MATCH idx=45 value=84384
MATCH idx=46 value=-24924
MATCH idx=47 value=-79448
MATCH idx=48 value=35748
MATCH idx=49 value=71862
MATCH idx=50 value=137625
MATCH idx=51 value=3921
MATCH idx=52 value=-27163
MATCH idx=53 value=87933
MATCH idx=54 value=-50122
MATCH idx=55 value=12515
MATCH idx=56 value=34636
MATCH idx=57 value=113111
MATCH idx=58 value=-130900
MATCH idx=59 value=106744
MATCH idx=60 value=10320
MATCH idx=61 value=-51794
MATCH idx=62 value=66657
MATCH idx=63 value=151667
MATCH idx=64 value=-120468
MATCH idx=65 value=-159868
MATCH idx=66 value=93345
MATCH idx=67 value=112988
MATCH idx=68 value=127393
MATCH idx=69 value=123097
MATCH idx=70 value=174926
MATCH idx=71 value=-64310
MATCH idx=72 value=95366
MATCH idx=73 value=-65186
MATCH idx=74 value=77104
MATCH idx=75 value=86763
MATCH idx=76 value=-74663
MATCH idx=77 value=-50076
MATCH idx=78 value=-137016
MATCH idx=79 value=1632
MATCH idx=80 value=108374
MATCH idx=81 value=72168
MATCH idx=82 value=-14840
MATCH idx=83 value=-80454
MATCH idx=84 value=15926
MATCH idx=85 value=158338
MATCH idx=86 value=25279
MATCH idx=87 value=-135680
MATCH idx=88 value=110647
MATCH idx=89 value=-78433
MATCH idx=90 value=-35020
MATCH idx=91 value=62847
MATCH idx=92 value=8678
MATCH idx=93 value=-83628
MATCH idx=94 value=68715
MATCH idx=95 value=59129
MATCH idx=96 value=84174
MATCH idx=97 value=-42893
MATCH idx=98 value=-1099
MATCH idx=99 value=64691
MATCH idx=100 value=110406
MATCH idx=101 value=40369
MATCH idx=102 value=1419
MATCH idx=103 value=83186
MATCH idx=104 value=-85707
MATCH idx=105 value=16581
MATCH idx=106 value=80734
MATCH idx=107 value=49325
MATCH idx=108 value=24797
MATCH idx=109 value=-77458
MATCH idx=110 value=-45432
MATCH idx=111 value=22701
MATCH idx=112 value=-25321
MATCH idx=113 value=17557
MATCH idx=114 value=55627
MATCH idx=115 value=73592
MATCH idx=116 value=-92943
MATCH idx=117 value=75767
MATCH idx=118 value=48607
MATCH idx=119 value=-32534
MATCH idx=120 value=-49159
MATCH idx=121 value=146206
MATCH idx=122 value=-84614
MATCH idx=123 value=90042
MATCH idx=124 value=34624
MATCH idx=125 value=-6985
MATCH idx=126 value=-37106
MATCH idx=127 value=41735
PASS: fc1_multi_block_parallel matches first 128 outputs of fc1_acc_0.hex.
$finish called from file "../tb/tb_fc1_multi_block_parallel.sv", line 91.
$finish at simulation time               536000
           V C S   S i m u l a t i o n   R e p o r t 
Time: 536000 ps
CPU Time:      0.230 seconds;       Data structure size:   3.2Mb
Sat Mar  7 23:36:45 2026是成功了对吧
AI助手:
对，**成功了，而且是完整 8 个 output block 并行成功**。

从你贴的日志可以直接看出几件关键事：

- `PAR_OB=8, OUT_DIM=128`
- 在 `time=525000` 时，`busy=0 done=1`
- 后面 `idx=0..127` 全部都是 `MATCH`
- 最后一行是  
  `PASS: fc1_multi_block_parallel matches first 128 outputs of fc1_acc_0.hex.`

这说明你现在已经完成了：

1. **单 block 正确**
2. **`ob=0..7` 单实例扫描正确**
3. **双实例并行正确**
4. **8 实例并行正确**
5. **拼接后的 128 维 FC1 输出和 golden 完全一致**

这已经证明你当前这条“**简单多实例化做 output-block 并行**”路线是通的。

---

## 这一步的意义
你现在实际上已经把 FC1 的第一层主干算通了：

- 输入：`input_0.hex`
- 权重：`fc1_weight_int8.hex`
- bias：`fc1_bias_int32.hex`
- 输出：`fc1_acc_0.hex`
- 并且是 **8 个 ob 一次并行算完**

所以接下来最合理的下一步就不是继续堆更多实例了，而是进入你之前规划的后两步：

### 方案 1：先做结构优化
把现在的：
- 多个 `fc1_cim_core_block`
- 每个内部各自带 `input_buffer`

重构成：
- **共享一个 `input_buffer`**
- **共享一个 `ib controller`**
- 广播 `x_eff_tile`
- 多个并行 `ob` 核只保留权重/累加/偏置

这是硬件结构上更像正式加速器的版本。

### 方案 2：先接功能链路
直接在 `fc1_acc_all[127:0]` 后面接：
- ReLU
- requant / scale / shift
- FC2

如果你更想尽快把整网跑通，我建议先接功能链路。  
如果你更想把 FC1 这个模块打磨成“像样的并行架构”，我建议先做共享 `input_buffer`。

---

## 我对你下一步的建议
按工程推进效率，我建议：

**先做共享 `input_buffer` + 共享 `ib controller`。**

原因是你现在已经证明：
- 算法对
- 地址映射对
- 多实例对

所以接下来最值得做的是把“重复读取同一个输入 8 次”的结构去掉。

---

## 你现在已经达到的状态
你可以把当前版本标成一个明确里程碑：

**Milestone: FC1 multi-output-block parallel verified**

也就是：
- 单样本
- 8 路 output block 并行
- 与 `fc1_acc_0.hex` 全对齐

这一步已经很扎实了。

下一条我可以直接帮你开始写：
**共享 `input_buffer` / `ib controller` 的重构版顶层和子模块接口。**

用户:
先按照方案一做共享的input_buffer和ib controller.
AI助手:
我先把你当前工程里的 `fc1_cim_core_block`、`input_buffer`、`psum_accum` 这些接口再对齐一下，然后给你一版“共享输入、共享 ib 扫描、多个 ob 并行”的重构代码，尽量保持和你现有模块兼容。

用户:

AI助手:


AI助手:


AI助手:
好，那我们正式进入 **方案一：共享 `input_buffer` + 共享 `ib controller`**。

这一步我建议你：

- **保留**你现有的这些模块不动  
  - `input_buffer.sv`
  - `fc1_weight_bank.sv`
  - `fc1_bias_bank.sv`
  - `cim_tile.sv`
  - `psum_accum.sv`
- **新增**两个 RTL：
  1. `fc1_ob_engine_shared_input.sv`
  2. `fc1_multi_block_shared_input.sv`
- 再新增：
  3. `tb_fc1_multi_block_shared_input.sv`
  4. `run_fc1_multi_block_shared_input.sh`

---

# 结构变化

原来你的结构是：

- 每个 `fc1_cim_core_block`
  - 自己有 `input_buffer`
  - 自己有 `ib` FSM
  - 自己读同一个输入

现在改成：

- **一个共享 `input_buffer`**
- **一个共享 `ib` FSM**
- 多个 `ob engine`
  - 每个只负责：
    - 根据共享 `ib` 读自己的 `w_tile`
    - 用共享 `x_eff_tile` 做 MAC
    - 本地累加 `psum`
    - 最后加 bias

这样就更像正式并行加速器了。

---

# 1) `fc1_ob_engine_shared_input.sv`

这个模块是“单个 output block 的计算引擎”，**不再包含 input_buffer，也不再包含 FSM**。

```systemverilog
module fc1_ob_engine_shared_input #(
    parameter string DEFAULT_WEIGHT_HEX_FILE =
        "../../CIM-sw-version1/sw/train_quantize/route_b_output/fc1_weight_int8.hex",
    parameter string DEFAULT_BIAS_HEX_FILE =
        "../../CIM-sw-version1/sw/train_quantize/route_b_output/fc1_bias_int32.hex"
) (
    input logic clk,
    input logic rst_n,

    input logic clear_psum,
    input logic en_psum,

    input logic [$clog2(mnist_cim_pkg::N_INPUT_BLOCKS)-1:0] ib,
    input logic [$clog2(mnist_cim_pkg::N_OUTPUT_BLOCKS)-1:0] ob_sel,

    input logic [mnist_cim_pkg::X_EFF_WIDTH-1:0]
        x_eff_tile [0:mnist_cim_pkg::TILE_INPUT_SIZE-1],

    output logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0]
        fc1_acc_block [0:mnist_cim_pkg::TILE_OUTPUT_SIZE-1]
);
  import mnist_cim_pkg::*;

  logic signed [WEIGHT_WIDTH-1:0]
      w_tile [0:TILE_OUTPUT_SIZE-1][0:TILE_INPUT_SIZE-1];

  logic signed [PSUM_WIDTH-1:0]
      tile_psum [0:TILE_OUTPUT_SIZE-1];

  logic signed [PSUM_WIDTH-1:0]
      psum [0:TILE_OUTPUT_SIZE-1];

  logic signed [BIAS_WIDTH-1:0]
      bias_block [0:TILE_OUTPUT_SIZE-1];

  logic signed [PSUM_WIDTH-1:0]
      fc1_acc_with_bias [0:TILE_OUTPUT_SIZE-1];

  fc1_weight_bank #(
      .DEFAULT_WEIGHT_HEX_FILE(DEFAULT_WEIGHT_HEX_FILE)
  ) u_fc1_weight_bank (
      .ob(ob_sel),
      .ib(ib),
      .w_tile(w_tile)
  );

  cim_tile u_cim_tile (
      .x_eff_tile(x_eff_tile),
      .w_tile(w_tile),
      .tile_psum(tile_psum)
  );

  psum_accum u_psum_accum (
      .clk(clk),
      .rst_n(rst_n),
      .clear(clear_psum),
      .en(en_psum),
      .tile_psum(tile_psum),
      .psum(psum)
  );

  fc1_bias_bank #(
      .DEFAULT_BIAS_HEX_FILE(DEFAULT_BIAS_HEX_FILE)
  ) u_fc1_bias_bank (
      .ob(ob_sel),
      .bias_block(bias_block)
  );

  genvar g;
  generate
    for (g = 0; g < TILE_OUTPUT_SIZE; g = g + 1) begin : GEN_OUT
      assign fc1_acc_with_bias[g] = psum[g] + bias_block[g];
      assign fc1_acc_block[g]     = fc1_acc_with_bias[g];
    end
  endgenerate

endmodule
```

---

# 2) `fc1_multi_block_shared_input.sv`

这个就是新的顶层：

- 一个共享 `input_buffer`
- 一个共享 `ib` 控制器
- 多个 `fc1_ob_engine_shared_input`

```systemverilog
module fc1_multi_block_shared_input #(
    parameter int PAR_OB = 8,
    parameter int BASE_OB = 0,
    parameter string DEFAULT_INPUT_HEX_FILE =
        "../../CIM-sw-version1/sw/train_quantize/route_b_output/input_0.hex",
    parameter string DEFAULT_WEIGHT_HEX_FILE =
        "../../CIM-sw-version1/sw/train_quantize/route_b_output/fc1_weight_int8.hex",
    parameter string DEFAULT_BIAS_HEX_FILE =
        "../../CIM-sw-version1/sw/train_quantize/route_b_output/fc1_bias_int32.hex"
) (
    input logic clk,
    input logic rst_n,
    input logic start,

    output logic busy,
    output logic done,

    output logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0]
        fc1_acc_all [0:PAR_OB*mnist_cim_pkg::TILE_OUTPUT_SIZE-1]
);
  import mnist_cim_pkg::*;

  typedef enum logic [1:0] {
    S_IDLE  = 2'd0,
    S_CLEAR = 2'd1,
    S_ACCUM = 2'd2,
    S_DONE  = 2'd3
  } state_t;

  state_t state, state_n;

  logic [$clog2(N_INPUT_BLOCKS)-1:0] ib, ib_n;

  logic clear_psum;
  logic en_psum;

  logic signed [INPUT_WIDTH-1:0]
      x_tile [0:TILE_INPUT_SIZE-1];

  logic [X_EFF_WIDTH-1:0]
      x_eff_tile [0:TILE_INPUT_SIZE-1];

  logic signed [PSUM_WIDTH-1:0]
      fc1_acc_block [0:PAR_OB-1][0:TILE_OUTPUT_SIZE-1];

  // --------------------------------------------
  // Shared input buffer
  // --------------------------------------------
  input_buffer #(
      .DEFAULT_INPUT_HEX_FILE(DEFAULT_INPUT_HEX_FILE)
  ) u_input_buffer (
      .ib(ib),
      .x_tile(x_tile),
      .x_eff_tile(x_eff_tile)
  );

  // --------------------------------------------
  // Global FSM
  // --------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_IDLE;
      ib    <= '0;
    end else begin
      state <= state_n;
      ib    <= ib_n;
    end
  end

  always_comb begin
    state_n    = state;
    ib_n       = ib;
    clear_psum = 1'b0;
    en_psum    = 1'b0;
    busy       = 1'b0;
    done       = 1'b0;

    case (state)
      S_IDLE: begin
        ib_n = '0;
        if (start) state_n = S_CLEAR;
      end

      S_CLEAR: begin
        busy       = 1'b1;
        clear_psum = 1'b1;
        ib_n       = '0;
        state_n    = S_ACCUM;
      end

      S_ACCUM: begin
        busy    = 1'b1;
        en_psum = 1'b1;

        if (ib == N_INPUT_BLOCKS - 1) begin
          state_n = S_DONE;
          ib_n    = ib;
        end else begin
          ib_n    = ib + 1'b1;
          state_n = S_ACCUM;
        end
      end

      S_DONE: begin
        done = 1'b1;
        if (!start) state_n = S_IDLE;
      end

      default: begin
        state_n = S_IDLE;
        ib_n    = '0;
      end
    endcase
  end

  // --------------------------------------------
  // Parallel output-block engines
  // --------------------------------------------
  genvar g_ob, g_idx;
  generate
    for (g_ob = 0; g_ob < PAR_OB; g_ob = g_ob + 1) begin : GEN_OB_ENGINE
      fc1_ob_engine_shared_input #(
          .DEFAULT_WEIGHT_HEX_FILE(DEFAULT_WEIGHT_HEX_FILE),
          .DEFAULT_BIAS_HEX_FILE(DEFAULT_BIAS_HEX_FILE)
      ) u_fc1_ob_engine_shared_input (
          .clk(clk),
          .rst_n(rst_n),
          .clear_psum(clear_psum),
          .en_psum(en_psum),
          .ib(ib),
          .ob_sel((BASE_OB + g_ob)[$clog2(N_OUTPUT_BLOCKS)-1:0]),
          .x_eff_tile(x_eff_tile),
          .fc1_acc_block(fc1_acc_block[g_ob])
      );

      for (g_idx = 0; g_idx < TILE_OUTPUT_SIZE; g_idx = g_idx + 1) begin : GEN_PACK
        assign fc1_acc_all[g_ob*TILE_OUTPUT_SIZE + g_idx] = fc1_acc_block[g_ob][g_idx];
      end
    end
  endgenerate

endmodule
```

---

# 3) `tb_fc1_multi_block_shared_input.sv`

这个 testbench 跟你之前那版并行测试类似，但现在测的是“**共享输入版**”。

```systemverilog
`timescale 1ns / 1ps

module tb_fc1_multi_block_shared_input;
  import mnist_cim_pkg::*;

  parameter int PAR_OB  = 8;
  parameter int BASE_OB = 0;
  localparam int OUT_DIM = PAR_OB * TILE_OUTPUT_SIZE;

  logic clk;
  logic rst_n;
  logic start;
  logic busy;
  logic done;

  logic signed [PSUM_WIDTH-1:0] fc1_acc_all[0:OUT_DIM-1];
  logic signed [PSUM_WIDTH-1:0] ref_fc1_acc_mem[0:HIDDEN_DIM-1];

  string fc1_acc_file;

  integer i;
  integer error_count;
  integer global_idx;

  fc1_multi_block_shared_input #(
      .PAR_OB(PAR_OB),
      .BASE_OB(BASE_OB)
  ) dut (
      .clk(clk),
      .rst_n(rst_n),
      .start(start),
      .busy(busy),
      .done(done),
      .fc1_acc_all(fc1_acc_all)
  );

  initial clk = 1'b0;
  always #5 clk = ~clk;

  initial begin
    fc1_acc_file = "../../CIM-sw-version1/sw/train_quantize/route_b_output/fc1_acc_0.hex";

    if ($value$plusargs("FC1_ACC_FILE=%s", fc1_acc_file)) begin
      $display("TB using FC1_ACC_FILE: %s", fc1_acc_file);
    end else begin
      $display("TB using default FC1_ACC_FILE: %s", fc1_acc_file);
    end

    $readmemh(fc1_acc_file, ref_fc1_acc_mem);
  end

  initial begin
    $display("time=%0t : TB start, PAR_OB=%0d, BASE_OB=%0d, OUT_DIM=%0d",
             $time, PAR_OB, BASE_OB, OUT_DIM);
    $monitor("time=%0t clk=%0b rst_n=%0b start=%0b busy=%0b done=%0b",
             $time, clk, rst_n, start, busy, done);
  end

  initial begin
    error_count = 0;

    rst_n = 1'b0;
    start = 1'b0;

    #12;
    rst_n = 1'b1;

    @(posedge clk);
    start <= 1'b1;

    @(posedge clk);
    start <= 1'b0;

    wait(done == 1'b1);
    @(posedge clk);
    #1;

    $display("Checking fc1_acc_all for shared-input architecture ...");

    for (i = 0; i < OUT_DIM; i = i + 1) begin
      global_idx = BASE_OB * TILE_OUTPUT_SIZE + i;

      if (fc1_acc_all[i] !== ref_fc1_acc_mem[global_idx]) begin
        $display("ERROR idx=%0d global_idx=%0d got=%0d expected=%0d",
                 i, global_idx, fc1_acc_all[i], ref_fc1_acc_mem[global_idx]);
        error_count = error_count + 1;
      end else begin
        $display("MATCH idx=%0d global_idx=%0d value=%0d",
                 i, global_idx, fc1_acc_all[i]);
      end
    end

    if (error_count == 0)
      $display("PASS: fc1_multi_block_shared_input matches fc1_acc_0.hex.");
    else
      $display("FAIL: found %0d mismatches.", error_count);

    $finish;
  end

endmodule
```

---

# 4) `run_fc1_multi_block_shared_input.sh`

```bash
#!/usr/bin/env bash
set -e

SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

INPUT_HEX="../route_b_output/input_0.hex"
WEIGHT_HEX="../route_b_output/fc1_weight_int8.hex"
FC1_ACC_HEX="../route_b_output/fc1_acc_0.hex"
FC1_BIAS_HEX="../route_b_output/fc1_bias_int32.hex"

mkdir -p ${SIM_DIR}
mkdir -p ${SIM_DIR}/log

vcs -full64 -sverilog -timescale=1ns/1ps \
    -debug_access+all \
    -o ${SIM_DIR}/fc1_multi_block_shared_input_simv \
    ${RTL_DIR}/package.sv \
    ${RTL_DIR}/input_buffer.sv \
    ${RTL_DIR}/fc1_weight_bank.sv \
    ${RTL_DIR}/fc1_bias_bank.sv \
    ${RTL_DIR}/cim_tile.sv \
    ${RTL_DIR}/psum_accum.sv \
    ${RTL_DIR}/fc1_ob_engine_shared_input.sv \
    ${RTL_DIR}/fc1_multi_block_shared_input.sv \
    ${TB_DIR}/tb_fc1_multi_block_shared_input.sv \
    2>&1 | tee ${SIM_DIR}/log/compile_tb_fc1_multi_block_shared_input.log

${SIM_DIR}/fc1_multi_block_shared_input_simv \
    +INPUT_HEX_FILE=${INPUT_HEX} \
    +WEIGHT_HEX_FILE=${WEIGHT_HEX} \
    +FC1_ACC_FILE=${FC1_ACC_HEX} \
    +FC1_BIAS_FILE=${FC1_BIAS_HEX} \
    2>&1 | tee ${SIM_DIR}/log/sim_tb_fc1_multi_block_shared_input.log
```

---

# 你现在怎么用

## 先建议这样跑
先在 `tb_fc1_multi_block_shared_input.sv` 里用：

```systemverilog
parameter int PAR_OB  = 2;
parameter int BASE_OB = 0;
```

先确认共享输入版的 `ob=0,1` 正确。

## 然后再改成
```systemverilog
parameter int PAR_OB  = 8;
parameter int BASE_OB = 0;
```

这样就验证完整 128 个输出。

---

# 这版和你上一版相比，真正省掉了什么

上一版 `fc1_multi_block_parallel.sv` 的问题是：

- 每个实例内部都有一个 `input_buffer`
- 同一份 `input_0.hex` 被重复加载/重复读取

这一版已经变成：

- **只有 1 个 `input_buffer`**
- **只有 1 套 `ib` 扫描 FSM**
- `x_eff_tile` 广播给所有并行 `ob engine`

所以这一步已经是你想要的“方案一”。

---

# 我建议你下一步怎么测

顺序最好这样：

1. `PAR_OB=2, BASE_OB=0`
2. `PAR_OB=2, BASE_OB=2`
3. `PAR_OB=4, BASE_OB=0`
4. `PAR_OB=8, BASE_OB=0`

这样可以更稳地排问题。

---

如果你愿意，我下一条可以继续直接帮你做两件事中的一个：

1. **把这套共享输入架构再进一步整理成更干净的正式版接口**
2. **继续往后接 ReLU + requant，开始做 FC1 -> FC2 链路**

用户:
                         Chronologic VCS (TM)
       Version O-2018.09-SP2_Full64 -- Sat Mar  7 23:55:22 2026
               Copyright (c) 1991-2018 by Synopsys Inc.
                         ALL RIGHTS RESERVED

This program is proprietary and confidential information of Synopsys Inc.
and may be used and disclosed only as authorized in a license agreement
controlling such use and disclosure.

Parsing design file '../rtl_shared_buffer_ib/package.sv'
Parsing design file '../rtl_shared_buffer_ib/input_buffer.sv'
Parsing design file '../rtl_shared_buffer_ib/fc1_weight_bank.sv'
Parsing design file '../rtl_shared_buffer_ib/fc1_bias_bank.sv'
Parsing design file '../rtl_shared_buffer_ib/cim_tile.sv'
Parsing design file '../rtl_shared_buffer_ib/psum_accum.sv'
Parsing design file '../rtl_shared_buffer_ib/fc1_ob_engine_shared_input.sv'
Parsing design file '../rtl_shared_buffer_ib/fc1_multi_block_shared_input.sv'

Error-[SE] Syntax error
  Following verilog source has syntax error :
  "../rtl_shared_buffer_ib/fc1_multi_block_shared_input.sv", 132: token is '['
            .ob_sel((BASE_OB + g_ob)[$clog2(N_OUTPUT_BLOCKS)-1:0]),
                                     ^

1 error
CPU time: .094 seconds to compile
AI助手:


AI助手:


AI助手:
这个报错是对的，问题就出在这一句：

```systemverilog
.ob_sel((BASE_OB + g_ob)[$clog2(N_OUTPUT_BLOCKS)-1:0]),
```

在 Verilog/SystemVerilog 里，**不能直接对一个算术表达式做这种位选切片**，所以 VCS 在 `[` 这里报语法错误。

---

## 怎么改

最简单的改法是：  
**不要切片，直接把表达式传进去**。

把这一句改成：

```systemverilog
.ob_sel(BASE_OB + g_ob),
```

因为 `ob_sel` 端口本身已经有固定位宽：

```systemverilog
input logic [$clog2(mnist_cim_pkg::N_OUTPUT_BLOCKS)-1:0] ob_sel
```

综合/仿真器会自动按端口位宽截断。

---

# 你应该改成的完整片段

把 `fc1_multi_block_shared_input.sv` 里这段：

```systemverilog
      fc1_ob_engine_shared_input #(
          .DEFAULT_WEIGHT_HEX_FILE(DEFAULT_WEIGHT_HEX_FILE),
          .DEFAULT_BIAS_HEX_FILE(DEFAULT_BIAS_HEX_FILE)
      ) u_fc1_ob_engine_shared_input (
          .clk(clk),
          .rst_n(rst_n),
          .clear_psum(clear_psum),
          .en_psum(en_psum),
          .ib(ib),
          .ob_sel((BASE_OB + g_ob)[$clog2(N_OUTPUT_BLOCKS)-1:0]),
          .x_eff_tile(x_eff_tile),
          .fc1_acc_block(fc1_acc_block[g_ob])
      );
```

改成：

```systemverilog
      fc1_ob_engine_shared_input #(
          .DEFAULT_WEIGHT_HEX_FILE(DEFAULT_WEIGHT_HEX_FILE),
          .DEFAULT_BIAS_HEX_FILE(DEFAULT_BIAS_HEX_FILE)
      ) u_fc1_ob_engine_shared_input (
          .clk(clk),
          .rst_n(rst_n),
          .clear_psum(clear_psum),
          .en_psum(en_psum),
          .ib(ib),
          .ob_sel(BASE_OB + g_ob),
          .x_eff_tile(x_eff_tile),
          .fc1_acc_block(fc1_acc_block[g_ob])
      );
```

---

## 为什么之前 dual 版没事
因为 dual 版里你写的是固定值：

```systemverilog
assign ob_sel0 = 0;
assign ob_sel1 = 1;
```

那不是“表达式后面直接切片”，所以不会触发这个语法问题。

---

## 更稳一点的写法
如果你想写得更规范一点，可以在 generate 里先定义一个局部常量。比如：

```systemverilog
localparam int OB_IDX = BASE_OB + g_ob;
```

然后：

```systemverilog
.ob_sel(OB_IDX),
```

不过对很多工具来说，直接写：

```systemverilog
.ob_sel(BASE_OB + g_ob)
```

就已经够了。

---

## 我建议你现在先这么做
先只改这一处，再重新编译。  
大概率这个语法错误就没了。

如果后面 VCS 继续报 generate 相关的小兼容问题，我再帮你顺手整理成一版 **更保守、对老版本 VCS 更友好** 的写法。

用户:
Chronologic VCS simulator copyright 1991-2018
Contains Synopsys proprietary information.
Compiler version O-2018.09-SP2_Full64; Runtime version O-2018.09-SP2_Full64;  Mar  7 23:59 2026
Using INPUT_HEX_FILE from plusarg: ../route_b_output/input_0.hex
Using WEIGHT_HEX_FILE from plusarg: ../route_b_output/fc1_weight_int8.hex
Using FC1_BIAS_FILE from plusarg: ../route_b_output/fc1_bias_int32.hex
Using WEIGHT_HEX_FILE from plusarg: ../route_b_output/fc1_weight_int8.hex
Using FC1_BIAS_FILE from plusarg: ../route_b_output/fc1_bias_int32.hex
Using WEIGHT_HEX_FILE from plusarg: ../route_b_output/fc1_weight_int8.hex
Using FC1_BIAS_FILE from plusarg: ../route_b_output/fc1_bias_int32.hex
Using WEIGHT_HEX_FILE from plusarg: ../route_b_output/fc1_weight_int8.hex
Using FC1_BIAS_FILE from plusarg: ../route_b_output/fc1_bias_int32.hex
Using WEIGHT_HEX_FILE from plusarg: ../route_b_output/fc1_weight_int8.hex
Using FC1_BIAS_FILE from plusarg: ../route_b_output/fc1_bias_int32.hex
Using WEIGHT_HEX_FILE from plusarg: ../route_b_output/fc1_weight_int8.hex
Using FC1_BIAS_FILE from plusarg: ../route_b_output/fc1_bias_int32.hex
Using WEIGHT_HEX_FILE from plusarg: ../route_b_output/fc1_weight_int8.hex
Using FC1_BIAS_FILE from plusarg: ../route_b_output/fc1_bias_int32.hex
Using WEIGHT_HEX_FILE from plusarg: ../route_b_output/fc1_weight_int8.hex
Using FC1_BIAS_FILE from plusarg: ../route_b_output/fc1_bias_int32.hex
TB using FC1_ACC_FILE: ../route_b_output/fc1_acc_0.hex
time=0 : TB start, PAR_OB=8, BASE_OB=0, OUT_DIM=128
time=0 clk=0 rst_n=0 start=0 busy=0 done=0
time=5000 clk=1 rst_n=0 start=0 busy=0 done=0
time=10000 clk=0 rst_n=0 start=0 busy=0 done=0
time=12000 clk=0 rst_n=1 start=0 busy=0 done=0
time=15000 clk=1 rst_n=1 start=1 busy=0 done=0
time=20000 clk=0 rst_n=1 start=1 busy=0 done=0
time=25000 clk=1 rst_n=1 start=0 busy=1 done=0
time=30000 clk=0 rst_n=1 start=0 busy=1 done=0
time=35000 clk=1 rst_n=1 start=0 busy=1 done=0
time=40000 clk=0 rst_n=1 start=0 busy=1 done=0
time=45000 clk=1 rst_n=1 start=0 busy=1 done=0
time=50000 clk=0 rst_n=1 start=0 busy=1 done=0
time=55000 clk=1 rst_n=1 start=0 busy=1 done=0
time=60000 clk=0 rst_n=1 start=0 busy=1 done=0
time=65000 clk=1 rst_n=1 start=0 busy=1 done=0
time=70000 clk=0 rst_n=1 start=0 busy=1 done=0
time=75000 clk=1 rst_n=1 start=0 busy=1 done=0
time=80000 clk=0 rst_n=1 start=0 busy=1 done=0
time=85000 clk=1 rst_n=1 start=0 busy=1 done=0
time=90000 clk=0 rst_n=1 start=0 busy=1 done=0
time=95000 clk=1 rst_n=1 start=0 busy=1 done=0
time=100000 clk=0 rst_n=1 start=0 busy=1 done=0
time=105000 clk=1 rst_n=1 start=0 busy=1 done=0
time=110000 clk=0 rst_n=1 start=0 busy=1 done=0
time=115000 clk=1 rst_n=1 start=0 busy=1 done=0
time=120000 clk=0 rst_n=1 start=0 busy=1 done=0
time=125000 clk=1 rst_n=1 start=0 busy=1 done=0
time=130000 clk=0 rst_n=1 start=0 busy=1 done=0
time=135000 clk=1 rst_n=1 start=0 busy=1 done=0
time=140000 clk=0 rst_n=1 start=0 busy=1 done=0
time=145000 clk=1 rst_n=1 start=0 busy=1 done=0
time=150000 clk=0 rst_n=1 start=0 busy=1 done=0
time=155000 clk=1 rst_n=1 start=0 busy=1 done=0
time=160000 clk=0 rst_n=1 start=0 busy=1 done=0
time=165000 clk=1 rst_n=1 start=0 busy=1 done=0
time=170000 clk=0 rst_n=1 start=0 busy=1 done=0
time=175000 clk=1 rst_n=1 start=0 busy=1 done=0
time=180000 clk=0 rst_n=1 start=0 busy=1 done=0
time=185000 clk=1 rst_n=1 start=0 busy=1 done=0
time=190000 clk=0 rst_n=1 start=0 busy=1 done=0
time=195000 clk=1 rst_n=1 start=0 busy=1 done=0
time=200000 clk=0 rst_n=1 start=0 busy=1 done=0
time=205000 clk=1 rst_n=1 start=0 busy=1 done=0
time=210000 clk=0 rst_n=1 start=0 busy=1 done=0
time=215000 clk=1 rst_n=1 start=0 busy=1 done=0
time=220000 clk=0 rst_n=1 start=0 busy=1 done=0
time=225000 clk=1 rst_n=1 start=0 busy=1 done=0
time=230000 clk=0 rst_n=1 start=0 busy=1 done=0
time=235000 clk=1 rst_n=1 start=0 busy=1 done=0
time=240000 clk=0 rst_n=1 start=0 busy=1 done=0
time=245000 clk=1 rst_n=1 start=0 busy=1 done=0
time=250000 clk=0 rst_n=1 start=0 busy=1 done=0
time=255000 clk=1 rst_n=1 start=0 busy=1 done=0
time=260000 clk=0 rst_n=1 start=0 busy=1 done=0
time=265000 clk=1 rst_n=1 start=0 busy=1 done=0
time=270000 clk=0 rst_n=1 start=0 busy=1 done=0
time=275000 clk=1 rst_n=1 start=0 busy=1 done=0
time=280000 clk=0 rst_n=1 start=0 busy=1 done=0
time=285000 clk=1 rst_n=1 start=0 busy=1 done=0
time=290000 clk=0 rst_n=1 start=0 busy=1 done=0
time=295000 clk=1 rst_n=1 start=0 busy=1 done=0
time=300000 clk=0 rst_n=1 start=0 busy=1 done=0
time=305000 clk=1 rst_n=1 start=0 busy=1 done=0
time=310000 clk=0 rst_n=1 start=0 busy=1 done=0
time=315000 clk=1 rst_n=1 start=0 busy=1 done=0
time=320000 clk=0 rst_n=1 start=0 busy=1 done=0
time=325000 clk=1 rst_n=1 start=0 busy=1 done=0
time=330000 clk=0 rst_n=1 start=0 busy=1 done=0
time=335000 clk=1 rst_n=1 start=0 busy=1 done=0
time=340000 clk=0 rst_n=1 start=0 busy=1 done=0
time=345000 clk=1 rst_n=1 start=0 busy=1 done=0
time=350000 clk=0 rst_n=1 start=0 busy=1 done=0
time=355000 clk=1 rst_n=1 start=0 busy=1 done=0
time=360000 clk=0 rst_n=1 start=0 busy=1 done=0
time=365000 clk=1 rst_n=1 start=0 busy=1 done=0
time=370000 clk=0 rst_n=1 start=0 busy=1 done=0
time=375000 clk=1 rst_n=1 start=0 busy=1 done=0
time=380000 clk=0 rst_n=1 start=0 busy=1 done=0
time=385000 clk=1 rst_n=1 start=0 busy=1 done=0
time=390000 clk=0 rst_n=1 start=0 busy=1 done=0
time=395000 clk=1 rst_n=1 start=0 busy=1 done=0
time=400000 clk=0 rst_n=1 start=0 busy=1 done=0
time=405000 clk=1 rst_n=1 start=0 busy=1 done=0
time=410000 clk=0 rst_n=1 start=0 busy=1 done=0
time=415000 clk=1 rst_n=1 start=0 busy=1 done=0
time=420000 clk=0 rst_n=1 start=0 busy=1 done=0
time=425000 clk=1 rst_n=1 start=0 busy=1 done=0
time=430000 clk=0 rst_n=1 start=0 busy=1 done=0
time=435000 clk=1 rst_n=1 start=0 busy=1 done=0
time=440000 clk=0 rst_n=1 start=0 busy=1 done=0
time=445000 clk=1 rst_n=1 start=0 busy=1 done=0
time=450000 clk=0 rst_n=1 start=0 busy=1 done=0
time=455000 clk=1 rst_n=1 start=0 busy=1 done=0
time=460000 clk=0 rst_n=1 start=0 busy=1 done=0
time=465000 clk=1 rst_n=1 start=0 busy=1 done=0
time=470000 clk=0 rst_n=1 start=0 busy=1 done=0
time=475000 clk=1 rst_n=1 start=0 busy=1 done=0
time=480000 clk=0 rst_n=1 start=0 busy=1 done=0
time=485000 clk=1 rst_n=1 start=0 busy=1 done=0
time=490000 clk=0 rst_n=1 start=0 busy=1 done=0
time=495000 clk=1 rst_n=1 start=0 busy=1 done=0
time=500000 clk=0 rst_n=1 start=0 busy=1 done=0
time=505000 clk=1 rst_n=1 start=0 busy=1 done=0
time=510000 clk=0 rst_n=1 start=0 busy=1 done=0
time=515000 clk=1 rst_n=1 start=0 busy=1 done=0
time=520000 clk=0 rst_n=1 start=0 busy=1 done=0
time=525000 clk=1 rst_n=1 start=0 busy=0 done=1
time=530000 clk=0 rst_n=1 start=0 busy=0 done=1
time=535000 clk=1 rst_n=1 start=0 busy=0 done=0
Checking fc1_acc_all for shared-input architecture ...
MATCH idx=0 global_idx=0 value=-30778
MATCH idx=1 global_idx=1 value=-33761
MATCH idx=2 global_idx=2 value=-26292
MATCH idx=3 global_idx=3 value=32890
MATCH idx=4 global_idx=4 value=84961
MATCH idx=5 global_idx=5 value=82286
MATCH idx=6 global_idx=6 value=-42871
MATCH idx=7 global_idx=7 value=-83495
MATCH idx=8 global_idx=8 value=14476
MATCH idx=9 global_idx=9 value=143175
MATCH idx=10 global_idx=10 value=50796
MATCH idx=11 global_idx=11 value=204030
MATCH idx=12 global_idx=12 value=27237
MATCH idx=13 global_idx=13 value=-70958
MATCH idx=14 global_idx=14 value=-7987
MATCH idx=15 global_idx=15 value=69688
MATCH idx=16 global_idx=16 value=88057
MATCH idx=17 global_idx=17 value=-30428
MATCH idx=18 global_idx=18 value=103647
MATCH idx=19 global_idx=19 value=74590
MATCH idx=20 global_idx=20 value=181469
MATCH idx=21 global_idx=21 value=-11999
MATCH idx=22 global_idx=22 value=33262
MATCH idx=23 global_idx=23 value=22503
MATCH idx=24 global_idx=24 value=12667
MATCH idx=25 global_idx=25 value=-71001
MATCH idx=26 global_idx=26 value=89749
MATCH idx=27 global_idx=27 value=-41266
MATCH idx=28 global_idx=28 value=17780
MATCH idx=29 global_idx=29 value=45129
MATCH idx=30 global_idx=30 value=-40512
MATCH idx=31 global_idx=31 value=39422
MATCH idx=32 global_idx=32 value=-186523
MATCH idx=33 global_idx=33 value=84098
MATCH idx=34 global_idx=34 value=105599
MATCH idx=35 global_idx=35 value=111699
MATCH idx=36 global_idx=36 value=129751
MATCH idx=37 global_idx=37 value=111784
MATCH idx=38 global_idx=38 value=35579
MATCH idx=39 global_idx=39 value=-62725
MATCH idx=40 global_idx=40 value=-7048
MATCH idx=41 global_idx=41 value=35904
MATCH idx=42 global_idx=42 value=-33587
MATCH idx=43 global_idx=43 value=-17150
MATCH idx=44 global_idx=44 value=-131331
MATCH idx=45 global_idx=45 value=84384
MATCH idx=46 global_idx=46 value=-24924
MATCH idx=47 global_idx=47 value=-79448
MATCH idx=48 global_idx=48 value=35748
MATCH idx=49 global_idx=49 value=71862
MATCH idx=50 global_idx=50 value=137625
MATCH idx=51 global_idx=51 value=3921
MATCH idx=52 global_idx=52 value=-27163
MATCH idx=53 global_idx=53 value=87933
MATCH idx=54 global_idx=54 value=-50122
MATCH idx=55 global_idx=55 value=12515
MATCH idx=56 global_idx=56 value=34636
MATCH idx=57 global_idx=57 value=113111
MATCH idx=58 global_idx=58 value=-130900
MATCH idx=59 global_idx=59 value=106744
MATCH idx=60 global_idx=60 value=10320
MATCH idx=61 global_idx=61 value=-51794
MATCH idx=62 global_idx=62 value=66657
MATCH idx=63 global_idx=63 value=151667
MATCH idx=64 global_idx=64 value=-120468
MATCH idx=65 global_idx=65 value=-159868
MATCH idx=66 global_idx=66 value=93345
MATCH idx=67 global_idx=67 value=112988
MATCH idx=68 global_idx=68 value=127393
MATCH idx=69 global_idx=69 value=123097
MATCH idx=70 global_idx=70 value=174926
MATCH idx=71 global_idx=71 value=-64310
MATCH idx=72 global_idx=72 value=95366
MATCH idx=73 global_idx=73 value=-65186
MATCH idx=74 global_idx=74 value=77104
MATCH idx=75 global_idx=75 value=86763
MATCH idx=76 global_idx=76 value=-74663
MATCH idx=77 global_idx=77 value=-50076
MATCH idx=78 global_idx=78 value=-137016
MATCH idx=79 global_idx=79 value=1632
MATCH idx=80 global_idx=80 value=108374
MATCH idx=81 global_idx=81 value=72168
MATCH idx=82 global_idx=82 value=-14840
MATCH idx=83 global_idx=83 value=-80454
MATCH idx=84 global_idx=84 value=15926
MATCH idx=85 global_idx=85 value=158338
MATCH idx=86 global_idx=86 value=25279
MATCH idx=87 global_idx=87 value=-135680
MATCH idx=88 global_idx=88 value=110647
MATCH idx=89 global_idx=89 value=-78433
MATCH idx=90 global_idx=90 value=-35020
MATCH idx=91 global_idx=91 value=62847
MATCH idx=92 global_idx=92 value=8678
MATCH idx=93 global_idx=93 value=-83628
MATCH idx=94 global_idx=94 value=68715
MATCH idx=95 global_idx=95 value=59129
MATCH idx=96 global_idx=96 value=84174
MATCH idx=97 global_idx=97 value=-42893
MATCH idx=98 global_idx=98 value=-1099
MATCH idx=99 global_idx=99 value=64691
MATCH idx=100 global_idx=100 value=110406
MATCH idx=101 global_idx=101 value=40369
MATCH idx=102 global_idx=102 value=1419
MATCH idx=103 global_idx=103 value=83186
MATCH idx=104 global_idx=104 value=-85707
MATCH idx=105 global_idx=105 value=16581
MATCH idx=106 global_idx=106 value=80734
MATCH idx=107 global_idx=107 value=49325
MATCH idx=108 global_idx=108 value=24797
MATCH idx=109 global_idx=109 value=-77458
MATCH idx=110 global_idx=110 value=-45432
MATCH idx=111 global_idx=111 value=22701
MATCH idx=112 global_idx=112 value=-25321
MATCH idx=113 global_idx=113 value=17557
MATCH idx=114 global_idx=114 value=55627
MATCH idx=115 global_idx=115 value=73592
MATCH idx=116 global_idx=116 value=-92943
MATCH idx=117 global_idx=117 value=75767
MATCH idx=118 global_idx=118 value=48607
MATCH idx=119 global_idx=119 value=-32534
MATCH idx=120 global_idx=120 value=-49159
MATCH idx=121 global_idx=121 value=146206
MATCH idx=122 global_idx=122 value=-84614
MATCH idx=123 global_idx=123 value=90042
MATCH idx=124 global_idx=124 value=34624
MATCH idx=125 global_idx=125 value=-6985
MATCH idx=126 global_idx=126 value=-37106
MATCH idx=127 global_idx=127 value=41735
PASS: fc1_multi_block_shared_input matches fc1_acc_0.hex.
$finish called from file "../tb/tb_fc1_multi_block_shared_input.sv", line 96.
$finish at simulation time               536000
           V C S   S i m u l a t i o n   R e p o r t 
Time: 536000 ps
CPU Time:      0.220 seconds;       Data structure size:   3.2Mb
Sat Mar  7 23:59:35 2026成功了对吧
AI助手:
对，**成功了，而且是你想要的“共享 `input_buffer` + 共享 `ib controller`”版本成功了。**

从这份日志可以明确确认几件事：

- 顶层已经是共享输入架构：`TB start, PAR_OB=8, BASE_OB=0, OUT_DIM=128`
- 整个计算流程正常结束：`time=525000 ... busy=0 done=1`
- `idx=0..127` 全部都是 `MATCH`
- 最后明确输出：  
  `PASS: fc1_multi_block_shared_input matches fc1_acc_0.hex.`

这说明你现在已经完成了一个很关键的里程碑：

## 你已经完成的阶段
1. 单个 tile 计算正确  
2. 单个 `ob` block 正确  
3. `ob=0..7` 全扫描正确  
4. 多实例简单并行正确  
5. **共享输入缓存 + 共享 ib 控制器 + 8 路 output block 并行正确**

也就是说，你现在的 FC1 并行架构已经从“功能验证版”升级成了“更合理的硬件组织版”。

---

## 这一步真正证明了什么

你刚刚证明了下面这件事是成立的：

- 一个输入样本 `input_0.hex`
- 一个共享 `input_buffer`
- 一个共享 `ib` 扫描器
- 8 个并行 `ob engine`
- 广播同一份 `x_eff_tile`
- 每个 engine 取自己对应的权重和 bias
- 最终拼出的 128 维 `fc1_acc_all`
- 与软件 golden `fc1_acc_0.hex` **完全一致**

这已经是一个很完整的 FC1 硬件子系统了。

---

## 你现在最适合进入的下一步

按你之前的规划，后面是：

- 激活
- FC2

所以现在最顺的路线就是：

### 下一步做 `fc1_acc_all -> 激活/量化 -> fc2`

更具体一点，我建议按这个顺序：

### 1. 先做 FC1 后处理模块
把 `fc1_acc_all[128]` 接成：
- bias 后结果你已经有了
- 接下来做 `ReLU`
- 再做 requant / scale / shift
- 生成 `fc1_out_0.hex` 对拍

因为你的 `route_b_output` 说明里，`fc1_out_i.hex` 就是第一层激活后的 golden，中间层最适合继续拿来做验证。

### 2. 再做 FC2
然后用：
- `fc1_out_0.hex` 作为 FC2 输入 golden 参考
- 权重 `fc2_weight_int8.hex`
- bias `fc2_bias_int32.hex`
- 输出去对拍 `fc2_acc_0.hex` / `scores_0.hex`

---

## 我建议你先别做的事
现在先不用再去继续优化：

- 更复杂的片上总线接口
- streaming 输入
- 多样本 batch pipeline
- 物理 tile 映射重排

因为你现在最有价值的是把**整条 MNIST 推理链**先跑通。

---

## 一句结论
**是的，成功了。**  
而且这不是普通成功，是你“共享输入架构”的 FC1 正式版已经验证通过了。

下一步最适合直接做：
**ReLU / requant 模块，然后接 FC2。**

我下一条可以直接给你：
**FC1 激活量化模块的设计流程 + RTL 代码框架。**

用户:
帮我做fc1_acc_all 的激活量化直到进入fc2
AI助手:
我先把你上传的 `route_b_output` 说明里和 `fc1_out_0.hex / fc2` 相关的量化规则、文件含义、数据位宽检索出来，再按你当前共享输入版 FC1 的接口，给你一版从 `fc1_acc_all` 到 FC2 输入的 RTL。

用户:
帮我做fc1_acc_all 的激活量化直到进入fc2,先别写，一会上传代码结构，你对着写
AI助手:
好，你把代码结构传上来，我会按你现有接口和命名风格来接 `fc1_acc_all -> 激活/量化 -> FC2`。  
等你上传后，我会先对齐模块边界、数据位宽和 golden 文件对应关系，再给你贴合当前工程的实现。

用户:
请写
AI助手:
我先把你刚上传的工程结构和现有 FC1/FC2 相关文件对齐一下，确认接口、位宽和命名后，直接按你现有风格补上 `fc1_acc_all -> 激活量化 -> fc2` 这一段。

AI助手:
我已经把关键量化公式对上了：  
`fc1_out_0.hex` 对应的是 `fc1_relu` 经过 `y = (x * 2040478460 + rounding) >>> 43` 再做 int8 饱和，和你的 golden 是一致的。接下来我直接按你现有风格给出可接到 `fc1_acc_all` 后面的 RTL。

AI助手:
可以，下面我按你现在这套风格，直接给出从 **`fc1_acc_all` -> ReLU -> requant -> `fc1_out_all` -> FC2 -> logits** 的一组 RTL。

我尽量保持和你现有代码一致：

- 继续用 `mnist_cim_pkg`
- 继续用 `$readmemh`
- 继续拆成 bank / core / top
- quant 参数直接按你 `quant_config.json` 里的数值写进去

你现在至少新增这 5 个文件：

1. `fc1_relu_requant.sv`
2. `fc2_weight_bank.sv`
3. `fc2_bias_bank.sv`
4. `fc2_core.sv`
5. `fc1_to_fc2_top.sv`

---

## 1) `fc1_relu_requant.sv`

作用：

- 输入：`fc1_acc_all[128]`，signed int32
- 输出：
  - `fc1_relu_all[128]`，signed int32
  - `fc1_out_all[128]`，signed int8

量化公式我已经按你 golden 对上了：

\[
\text{fc1\_out} = \text{saturate\_int8}\Big((\text{fc1\_relu} \times 2040478460 + rounding) >>> 43\Big)
\]

```systemverilog
module fc1_relu_requant #(
    parameter int FC1_REQUANT_MULT  = 2040478460,
    parameter int FC1_REQUANT_SHIFT = 43
) (
    input logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0]
        fc1_acc_all [0:mnist_cim_pkg::HIDDEN_DIM-1],

    output logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0]
        fc1_relu_all [0:mnist_cim_pkg::HIDDEN_DIM-1],

    output logic signed [mnist_cim_pkg::OUTPUT_WIDTH-1:0]
        fc1_out_all [0:mnist_cim_pkg::HIDDEN_DIM-1]
);
  import mnist_cim_pkg::*;

  function automatic signed [OUTPUT_WIDTH-1:0] requantize_int32_to_int8(
      input logic signed [PSUM_WIDTH-1:0] x,
      input int mult,
      input int rshift
  );
    longint signed prod;
    longint signed shifted;
    begin
      prod = x * mult;
      shifted = (prod + (64'sd1 <<< (rshift - 1))) >>> rshift;

      if (shifted > 127)
        requantize_int32_to_int8 = 8'sd127;
      else if (shifted < -128)
        requantize_int32_to_int8 = -8'sd128;
      else
        requantize_int32_to_int8 = shifted[OUTPUT_WIDTH-1:0];
    end
  endfunction

  integer i;
  always_comb begin
    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
      if (fc1_acc_all[i] > 0)
        fc1_relu_all[i] = fc1_acc_all[i];
      else
        fc1_relu_all[i] = '0;

      fc1_out_all[i] = requantize_int32_to_int8(
          fc1_relu_all[i],
          FC1_REQUANT_MULT,
          FC1_REQUANT_SHIFT
      );
    end
  end

endmodule
```

---

## 2) `fc2_weight_bank.sv`

作用：

- 从 `fc2_weight_int8.hex` 读取第二层权重
- 输出完整二维矩阵 `w_all[10][128]`

布局按你说明文件：

- row-major
- `[out][in] = [10][128]`
- `addr = out_idx * 128 + in_idx`

```systemverilog
module fc2_weight_bank #(
    parameter string DEFAULT_WEIGHT_HEX_FILE =
        "../route_b_output/fc2_weight_int8.hex"
) (
    output logic signed [mnist_cim_pkg::WEIGHT_WIDTH-1:0]
        w_all [0:mnist_cim_pkg::FC2_OUT_DIM-1]
              [0:mnist_cim_pkg::FC2_IN_DIM-1]
);
  import mnist_cim_pkg::*;

  string weight_file;

  logic signed [WEIGHT_WIDTH-1:0] weight_mem[0:FC2_WEIGHT_DEPTH-1];

  integer o, i, addr;

  initial begin
    weight_file = DEFAULT_WEIGHT_HEX_FILE;

    if ($value$plusargs("FC2_WEIGHT_HEX_FILE=%s", weight_file)) begin
      $display("Using FC2_WEIGHT_HEX_FILE from plusarg: %s", weight_file);
    end else begin
      $display("Using default FC2_WEIGHT_HEX_FILE: %s", weight_file);
    end

    $readmemh(weight_file, weight_mem);
  end

  always_comb begin
    for (o = 0; o < FC2_OUT_DIM; o = o + 1) begin
      for (i = 0; i < FC2_IN_DIM; i = i + 1) begin
        addr = o * FC2_IN_DIM + i;
        w_all[o][i] = weight_mem[addr];
      end
    end
  end

endmodule
```

---

## 3) `fc2_bias_bank.sv`

作用：

- 读取 `fc2_bias_int32.hex`
- 输出 10 个 int32 bias

```systemverilog
module fc2_bias_bank #(
    parameter string DEFAULT_BIAS_HEX_FILE =
        "../route_b_output/fc2_bias_int32.hex"
) (
    output logic signed [mnist_cim_pkg::BIAS_WIDTH-1:0]
        bias_all [0:mnist_cim_pkg::FC2_OUT_DIM-1]
);
  import mnist_cim_pkg::*;

  string bias_file;

  logic signed [BIAS_WIDTH-1:0] bias_mem[0:FC2_OUT_DIM-1];

  integer i;

  initial begin
    bias_file = DEFAULT_BIAS_HEX_FILE;

    if ($value$plusargs("FC2_BIAS_FILE=%s", bias_file)) begin
      $display("Using FC2_BIAS_FILE from plusarg: %s", bias_file);
    end else begin
      $display("Using default FC2_BIAS_FILE: %s", bias_file);
    end

    $readmemh(bias_file, bias_mem);
  end

  always_comb begin
    for (i = 0; i < FC2_OUT_DIM; i = i + 1) begin
      bias_all[i] = bias_mem[i];
    end
  end

endmodule
```

---

## 4) `fc2_core.sv`

作用：

- 输入：`fc1_out_all[128]`，signed int8
- 输出：
  - `fc2_acc_all[10]`，signed int32
  - `logits_all[10]`，signed int8

这里第二层直接全连接一次算完，不再分 tile，因为：

- 输入只有 128
- 输出只有 10
- 先做功能正确最合适

量化公式按你的 `quant_config.json`：

\[
\text{logits} = \text{saturate\_int8}\Big((\text{fc2\_acc} \times 1680079861 + rounding) >>> 39\Big)
\]

```systemverilog
module fc2_core #(
    parameter string DEFAULT_WEIGHT_HEX_FILE =
        "../route_b_output/fc2_weight_int8.hex",
    parameter string DEFAULT_BIAS_HEX_FILE =
        "../route_b_output/fc2_bias_int32.hex",
    parameter int FC2_REQUANT_MULT  = 1680079861,
    parameter int FC2_REQUANT_SHIFT = 39
) (
    input logic signed [mnist_cim_pkg::OUTPUT_WIDTH-1:0]
        fc1_out_all [0:mnist_cim_pkg::FC2_IN_DIM-1],

    output logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0]
        fc2_acc_all [0:mnist_cim_pkg::FC2_OUT_DIM-1],

    output logic signed [mnist_cim_pkg::OUTPUT_WIDTH-1:0]
        logits_all [0:mnist_cim_pkg::FC2_OUT_DIM-1]
);
  import mnist_cim_pkg::*;

  logic signed [WEIGHT_WIDTH-1:0]
      w_all [0:FC2_OUT_DIM-1][0:FC2_IN_DIM-1];

  logic signed [BIAS_WIDTH-1:0]
      bias_all [0:FC2_OUT_DIM-1];

  fc2_weight_bank #(
      .DEFAULT_WEIGHT_HEX_FILE(DEFAULT_WEIGHT_HEX_FILE)
  ) u_fc2_weight_bank (
      .w_all(w_all)
  );

  fc2_bias_bank #(
      .DEFAULT_BIAS_HEX_FILE(DEFAULT_BIAS_HEX_FILE)
  ) u_fc2_bias_bank (
      .bias_all(bias_all)
  );

  function automatic signed [OUTPUT_WIDTH-1:0] requantize_int32_to_int8(
      input logic signed [PSUM_WIDTH-1:0] x,
      input int mult,
      input int rshift
  );
    longint signed prod;
    longint signed shifted;
    begin
      prod = x * mult;
      shifted = (prod + (64'sd1 <<< (rshift - 1))) >>> rshift;

      if (shifted > 127)
        requantize_int32_to_int8 = 8'sd127;
      else if (shifted < -128)
        requantize_int32_to_int8 = -8'sd128;
      else
        requantize_int32_to_int8 = shifted[OUTPUT_WIDTH-1:0];
    end
  endfunction

  integer o, i;
  longint signed acc_tmp;

  always_comb begin
    for (o = 0; o < FC2_OUT_DIM; o = o + 1) begin
      acc_tmp = bias_all[o];

      for (i = 0; i < FC2_IN_DIM; i = i + 1) begin
        acc_tmp = acc_tmp + fc1_out_all[i] * w_all[o][i];
      end

      fc2_acc_all[o] = acc_tmp[PSUM_WIDTH-1:0];

      logits_all[o] = requantize_int32_to_int8(
          fc2_acc_all[o],
          FC2_REQUANT_MULT,
          FC2_REQUANT_SHIFT
      );
    end
  end

endmodule
```

---

## 5) `fc1_to_fc2_top.sv`

这个顶层就是你现在最想接的那段链路：

- 输入：`fc1_acc_all[128]`
- 经过：
  - `fc1_relu_requant`
  - `fc2_core`
- 输出：
  - `fc1_relu_all[128]`
  - `fc1_out_all[128]`
  - `fc2_acc_all[10]`
  - `logits_all[10]`

```systemverilog
module fc1_to_fc2_top #(
    parameter string DEFAULT_FC2_WEIGHT_HEX_FILE =
        "../route_b_output/fc2_weight_int8.hex",
    parameter string DEFAULT_FC2_BIAS_HEX_FILE =
        "../route_b_output/fc2_bias_int32.hex"
) (
    input logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0]
        fc1_acc_all [0:mnist_cim_pkg::HIDDEN_DIM-1],

    output logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0]
        fc1_relu_all [0:mnist_cim_pkg::HIDDEN_DIM-1],

    output logic signed [mnist_cim_pkg::OUTPUT_WIDTH-1:0]
        fc1_out_all [0:mnist_cim_pkg::HIDDEN_DIM-1],

    output logic signed [mnist_cim_pkg::PSUM_WIDTH-1:0]
        fc2_acc_all [0:mnist_cim_pkg::FC2_OUT_DIM-1],

    output logic signed [mnist_cim_pkg::OUTPUT_WIDTH-1:0]
        logits_all [0:mnist_cim_pkg::FC2_OUT_DIM-1]
);
  import mnist_cim_pkg::*;

  fc1_relu_requant u_fc1_relu_requant (
      .fc1_acc_all(fc1_acc_all),
      .fc1_relu_all(fc1_relu_all),
      .fc1_out_all(fc1_out_all)
  );

  fc2_core #(
      .DEFAULT_WEIGHT_HEX_FILE(DEFAULT_FC2_WEIGHT_HEX_FILE),
      .DEFAULT_BIAS_HEX_FILE(DEFAULT_FC2_BIAS_HEX_FILE)
  ) u_fc2_core (
      .fc1_out_all(fc1_out_all),
      .fc2_acc_all(fc2_acc_all),
      .logits_all(logits_all)
  );

endmodule
```

---

# 你怎么接到现有 FC1 顶层后面

如果你当前已经有：

- `fc1_multi_block_shared_input`
- 输出 `fc1_acc_all[0:127]`

那么直接再包一层，例如：

```systemverilog
logic signed [PSUM_WIDTH-1:0] fc1_acc_all [0:HIDDEN_DIM-1];
logic signed [PSUM_WIDTH-1:0] fc1_relu_all[0:HIDDEN_DIM-1];
logic signed [OUTPUT_WIDTH-1:0] fc1_out_all [0:HIDDEN_DIM-1];
logic signed [PSUM_WIDTH-1:0] fc2_acc_all [0:FC2_OUT_DIM-1];
logic signed [OUTPUT_WIDTH-1:0] logits_all [0:FC2_OUT_DIM-1];

fc1_multi_block_shared_input u_fc1 (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .busy(busy_fc1),
    .done(done_fc1),
    .fc1_acc_all(fc1_acc_all)
);

fc1_to_fc2_top u_fc1_to_fc2 (
    .fc1_acc_all(fc1_acc_all),
    .fc1_relu_all(fc1_relu_all),
    .fc1_out_all(fc1_out_all),
    .fc2_acc_all(fc2_acc_all),
    .logits_all(logits_all)
);
```

---

# 我建议你接下来先怎么验证

顺序按这个来最稳：

1. 先对拍 `fc1_relu_0.hex`
2. 再对拍 `fc1_out_0.hex`
3. 再对拍 `fc2_acc_0.hex`
4. 最后对拍 `logits_0.hex`

也就是：
- 先证明 `ReLU` 对
- 再证明第一层 requant 对
- 再证明 FC2 MAC 对
- 最后证明第二层 requant 对

---

# 两个小提醒

第一个，你现在 `package.sv` 其实不用改也能先跑，因为这些模块直接用了已有的：

- `FC2_IN_DIM`
- `FC2_OUT_DIM`
- `FC2_WEIGHT_DEPTH`
- `FC2_BIAS_DEPTH`

第二个，这版是**组合实现**，也就是：

- `fc1_relu_requant` 组合
- `fc2_core` 组合

这样最适合先和 golden 对拍。  
等你确认全对了，再考虑做成时序化/流水化版本。

---

如果你愿意，我下一条直接继续给你补：

- `tb_fc1_to_fc2_top.sv`
- `run_fc1_to_fc2_top.sh`

这样你可以直接开始对拍 `fc1_relu_0.hex / fc1_out_0.hex / fc2_acc_0.hex / logits_0.hex`。

用户:
mnist_cim_pkg这个需要添加内容吗，因为我写的时候没有考虑FC2
AI助手:
要，**大概率需要补一些 FC2 相关常量**。  
因为我刚才给你的那几份代码里已经用到了这些名字：

- `FC2_IN_DIM`
- `FC2_OUT_DIM`
- `FC2_WEIGHT_DEPTH`

如果你现在的 `mnist_cim_pkg` 只覆盖了 FC1，那编译时会直接报这些符号没定义。

---

## 你至少要在 `mnist_cim_pkg` 里补这些

下面这段可以直接加：

```systemverilog
package mnist_cim_pkg;

  // --------------------------------------------------
  // Common data width
  // --------------------------------------------------
  parameter int INPUT_WIDTH   = 8;
  parameter int WEIGHT_WIDTH  = 8;
  parameter int OUTPUT_WIDTH  = 8;
  parameter int BIAS_WIDTH    = 32;
  parameter int PSUM_WIDTH    = 32;
  parameter int X_EFF_WIDTH   = 8;

  // --------------------------------------------------
  // FC1 / tile config
  // --------------------------------------------------
  parameter int INPUT_DIM         = 784;
  parameter int HIDDEN_DIM        = 128;
  parameter int TILE_INPUT_SIZE   = 16;
  parameter int TILE_OUTPUT_SIZE  = 16;

  parameter int N_INPUT_BLOCKS    = 49;   // 784 / 16
  parameter int N_OUTPUT_BLOCKS   = 8;    // 128 / 16

  // --------------------------------------------------
  // FC2 config
  // --------------------------------------------------
  parameter int FC2_IN_DIM        = HIDDEN_DIM; // 128
  parameter int FC2_OUT_DIM       = 10;

  parameter int FC2_WEIGHT_DEPTH  = FC2_OUT_DIM * FC2_IN_DIM; // 10*128=1280
  parameter int FC2_BIAS_DEPTH    = FC2_OUT_DIM;

endpackage
```

---

## 如果你现有 package 已经有这些，就不要重复定义
你现在最稳的做法不是整份覆盖，而是检查你当前 `package.sv`：

### 如果已经有
- `INPUT_WIDTH`
- `WEIGHT_WIDTH`
- `OUTPUT_WIDTH`
- `BIAS_WIDTH`
- `PSUM_WIDTH`
- `X_EFF_WIDTH`
- `INPUT_DIM`
- `HIDDEN_DIM`
- `TILE_INPUT_SIZE`
- `TILE_OUTPUT_SIZE`
- `N_INPUT_BLOCKS`
- `N_OUTPUT_BLOCKS`

那你只需要**补 FC2 这一小段**：

```systemverilog
  parameter int FC2_IN_DIM        = HIDDEN_DIM;
  parameter int FC2_OUT_DIM       = 10;
  parameter int FC2_WEIGHT_DEPTH  = FC2_OUT_DIM * FC2_IN_DIM;
  parameter int FC2_BIAS_DEPTH    = FC2_OUT_DIM;
```

---

## 我刚才那几份代码为什么需要这些
因为：

### `fc2_weight_bank.sv`
需要知道权重矩阵大小：
```systemverilog
[0:FC2_OUT_DIM-1][0:FC2_IN_DIM-1]
```

### `fc2_bias_bank.sv`
需要知道 bias 长度：
```systemverilog
[0:FC2_OUT_DIM-1]
```

### `fc2_core.sv`
需要知道：
- 输入长度 128
- 输出长度 10

---

## 还要不要补量化常量到 package？
**不是必须。**

我刚才把量化参数写在模块 parameter 里了：

- `FC1_REQUANT_MULT = 2040478460`
- `FC1_REQUANT_SHIFT = 43`
- `FC2_REQUANT_MULT = 1680079861`
- `FC2_REQUANT_SHIFT = 39`

所以即使 package 里没有这些，也能编。

但如果你想统一管理，也可以补到 package 里，例如：

```systemverilog
  parameter int FC1_REQUANT_MULT  = 2040478460;
  parameter int FC1_REQUANT_SHIFT = 43;

  parameter int FC2_REQUANT_MULT  = 1680079861;
  parameter int FC2_REQUANT_SHIFT = 39;
```

然后模块里再引用 package 常量。  
这会更整洁，但不是这一步必须做的。

---

## 我的建议
你现在最小改动就做两件事：

### 第一件
在 `mnist_cim_pkg` 里补：

```systemverilog
parameter int FC2_IN_DIM        = HIDDEN_DIM;
parameter int FC2_OUT_DIM       = 10;
parameter int FC2_WEIGHT_DEPTH  = FC2_OUT_DIM * FC2_IN_DIM;
parameter int FC2_BIAS_DEPTH    = FC2_OUT_DIM;
```

### 第二件
确认你 package 里已经有：
- `OUTPUT_WIDTH = 8`
- `BIAS_WIDTH = 32`
- `PSUM_WIDTH = 32`

如果这些也没有，就一起补上。

---

## 一句结论
**要补，但通常只需要补 FC2 相关的维度和深度常量，不用大改 package。**

你把当前 `package.sv` 贴出来的话，我可以直接帮你标出“只需要新增哪几行”。

用户:
你怎么接到现有 FC1 顶层后面

如果你当前已经有：

fc1_multi_block_shared_input

输出 fc1_acc_all[0:127]

那么直接再包一层，例如：

logic signed [PSUM_WIDTH-1:0] fc1_acc_all [0:HIDDEN_DIM-1];
logic signed [PSUM_WIDTH-1:0] fc1_relu_all[0:HIDDEN_DIM-1];
logic signed [OUTPUT_WIDTH-1:0] fc1_out_all [0:HIDDEN_DIM-1];
logic signed [PSUM_WIDTH-1:0] fc2_acc_all [0:FC2_OUT_DIM-1];
logic signed [OUTPUT_WIDTH-1:0] logits_all [0:FC2_OUT_DIM-1];

fc1_multi_block_shared_input u_fc1 (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .busy(busy_fc1),
    .done(done_fc1),
    .fc1_acc_all(fc1_acc_all)
);

fc1_to_fc2_top u_fc1_to_fc2 (
    .fc1_acc_all(fc1_acc_all),
    .fc1_relu_all(fc1_relu_all),
    .fc1_out_all(fc1_out_all),
    .fc2_acc_all(fc2_acc_all),
    .logits_all(logits_all)
);

解释一下这个。

另外写几个测试fc1_relu_requant.sv

fc2_weight_bank.sv

fc2_bias_bank.sv

fc2_core.sv

fc1_to_fc2_top.sv的run.sh，同样仿照我的hw/scripts内文件写
AI助手:
先解释你问的“**怎么接到现有 FC1 顶层后面**”。

你现在已经有这个模块：

- `fc1_multi_block_shared_input`
- 输出 `fc1_acc_all[0:127]`

它做完的事情是：

- 读 `input_0.hex`
- 读 `fc1_weight_int8.hex`
- 读 `fc1_bias_int32.hex`
- 算完整个 FC1
- 输出 **带 bias、但还没经过 ReLU 和 requant 的 128 维 int32 累加结果**

也就是：

\[
fc1\_acc\_all = W_1 x + b_1
\]

所以你后面要接的链路就是：

\[
fc1\_acc\_all \rightarrow ReLU \rightarrow requant \rightarrow fc1\_out\_all \rightarrow FC2
\]

---

# 你贴的这段连接代码是什么意思

```systemverilog
logic signed [PSUM_WIDTH-1:0] fc1_acc_all [0:HIDDEN_DIM-1];
logic signed [PSUM_WIDTH-1:0] fc1_relu_all[0:HIDDEN_DIM-1];
logic signed [OUTPUT_WIDTH-1:0] fc1_out_all [0:HIDDEN_DIM-1];
logic signed [PSUM_WIDTH-1:0] fc2_acc_all [0:FC2_OUT_DIM-1];
logic signed [OUTPUT_WIDTH-1:0] logits_all [0:FC2_OUT_DIM-1];
```

这几行是在顶层里声明中间信号。

它们分别表示：

- `fc1_acc_all[128]`  
  FC1 原始累加输出，**int32**
- `fc1_relu_all[128]`  
  FC1 经过 ReLU 后的结果，**int32**
- `fc1_out_all[128]`  
  FC1 经过 requant 后输出给 FC2 的结果，**int8**
- `fc2_acc_all[10]`  
  FC2 原始累加输出，**int32**
- `logits_all[10]`  
  FC2 requant 后最终输出，**int8**

---

然后这段：

```systemverilog
fc1_multi_block_shared_input u_fc1 (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .busy(busy_fc1),
    .done(done_fc1),
    .fc1_acc_all(fc1_acc_all)
);
```

意思是：

- 实例化你的 FC1 顶层
- 它把完整的 FC1 结果写到 `fc1_acc_all`

也就是说这一步只负责：

\[
input \rightarrow fc1\_acc\_all
\]

---

再往后这段：

```systemverilog
fc1_to_fc2_top u_fc1_to_fc2 (
    .fc1_acc_all(fc1_acc_all),
    .fc1_relu_all(fc1_relu_all),
    .fc1_out_all(fc1_out_all),
    .fc2_acc_all(fc2_acc_all),
    .logits_all(logits_all)
);
```

意思是：

- 把 `u_fc1` 的输出 `fc1_acc_all`
- 直接送进 `fc1_to_fc2_top`
- 这个模块内部继续做：
  1. `fc1_acc_all -> fc1_relu_all`
  2. `fc1_relu_all -> fc1_out_all`
  3. `fc1_out_all -> fc2_acc_all`
  4. `fc2_acc_all -> logits_all`

所以 `fc1_to_fc2_top` 相当于是 **FC1 后处理 + FC2**。

---

# 整体数据流就是这样

```text
input_0.hex
   │
   ▼
fc1_multi_block_shared_input
   │
   ▼
fc1_acc_all[128]   // int32
   │
   ▼
fc1_relu_requant
   │
   ├── fc1_relu_all[128] // int32
   └── fc1_out_all [128] // int8
                │
                ▼
             fc2_core
                │
                ├── fc2_acc_all[10] // int32
                └── logits_all [10] // int8
```

---

# 为什么要拆成两个模块，而不是全塞一起

因为这样更容易验证：

- `fc1_multi_block_shared_input` 只负责 FC1 主计算
- `fc1_to_fc2_top` 只负责后处理和 FC2

你就可以分别对拍：

- `fc1_acc_0.hex`
- `fc1_relu_0.hex`
- `fc1_out_0.hex`
- `fc2_acc_0.hex`
- `logits_0.hex`

定位错误会非常清楚。

---

# 你要的 run.sh

下面我按你现在 `hw/scripts` 的风格来写。  
我默认目录还是：

- `SIM_DIR=../sim`
- `RTL_DIR=../rtl`
- `TB_DIR=../tb`

而且我默认你会分别给每个模块配一个 testbench：

- `tb_fc1_relu_requant.sv`
- `tb_fc2_weight_bank.sv`
- `tb_fc2_bias_bank.sv`
- `tb_fc2_core.sv`
- `tb_fc1_to_fc2_top.sv`

---

## 1) `run_fc1_relu_requant.sh`

```bash
#!/usr/bin/env bash
set -e

SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

FC1_ACC_HEX="../route_b_output/fc1_acc_0.hex"
FC1_RELU_HEX="../route_b_output/fc1_relu_0.hex"
FC1_OUT_HEX="../route_b_output/fc1_out_0.hex"

mkdir -p ${SIM_DIR}
mkdir -p ${SIM_DIR}/log

vcs -full64 -sverilog -timescale=1ns/1ps \
    -debug_access+all \
    -o ${SIM_DIR}/fc1_relu_requant_simv \
    ${RTL_DIR}/package.sv \
    ${RTL_DIR}/fc1_relu_requant.sv \
    ${TB_DIR}/tb_fc1_relu_requant.sv \
    2>&1 | tee ${SIM_DIR}/log/compile_tb_fc1_relu_requant.log

${SIM_DIR}/fc1_relu_requant_simv \
    +FC1_ACC_FILE=${FC1_ACC_HEX} \
    +FC1_RELU_FILE=${FC1_RELU_HEX} \
    +FC1_OUT_FILE=${FC1_OUT_HEX} \
    2>&1 | tee ${SIM_DIR}/log/sim_tb_fc1_relu_requant.log
```

---

## 2) `run_fc2_weight_bank.sh`

```bash
#!/usr/bin/env bash
set -e

SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

FC2_WEIGHT_HEX="../route_b_output/fc2_weight_int8.hex"

mkdir -p ${SIM_DIR}
mkdir -p ${SIM_DIR}/log

vcs -full64 -sverilog -timescale=1ns/1ps \
    -debug_access+all \
    -o ${SIM_DIR}/fc2_weight_bank_simv \
    ${RTL_DIR}/package.sv \
    ${RTL_DIR}/fc2_weight_bank.sv \
    ${TB_DIR}/tb_fc2_weight_bank.sv \
    2>&1 | tee ${SIM_DIR}/log/compile_tb_fc2_weight_bank.log

${SIM_DIR}/fc2_weight_bank_simv \
    +FC2_WEIGHT_HEX_FILE=${FC2_WEIGHT_HEX} \
    2>&1 | tee ${SIM_DIR}/log/sim_tb_fc2_weight_bank.log
```

---

## 3) `run_fc2_bias_bank.sh`

```bash
#!/usr/bin/env bash
set -e

SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

FC2_BIAS_HEX="../route_b_output/fc2_bias_int32.hex"

mkdir -p ${SIM_DIR}
mkdir -p ${SIM_DIR}/log

vcs -full64 -sverilog -timescale=1ns/1ps \
    -debug_access+all \
    -o ${SIM_DIR}/fc2_bias_bank_simv \
    ${RTL_DIR}/package.sv \
    ${RTL_DIR}/fc2_bias_bank.sv \
    ${TB_DIR}/tb_fc2_bias_bank.sv \
    2>&1 | tee ${SIM_DIR}/log/compile_tb_fc2_bias_bank.log

${SIM_DIR}/fc2_bias_bank_simv \
    +FC2_BIAS_FILE=${FC2_BIAS_HEX} \
    2>&1 | tee ${SIM_DIR}/log/sim_tb_fc2_bias_bank.log
```

---

## 4) `run_fc2_core.sh`

这个模块会依赖：

- `fc2_weight_bank.sv`
- `fc2_bias_bank.sv`

并且 testbench 通常会喂入：

- `fc1_out_0.hex`
- 对拍 `fc2_acc_0.hex`
- 对拍 `logits_0.hex`

```bash
#!/usr/bin/env bash
set -e

SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

FC1_OUT_HEX="../route_b_output/fc1_out_0.hex"
FC2_WEIGHT_HEX="../route_b_output/fc2_weight_int8.hex"
FC2_BIAS_HEX="../route_b_output/fc2_bias_int32.hex"
FC2_ACC_HEX="../route_b_output/fc2_acc_0.hex"
LOGITS_HEX="../route_b_output/logits_0.hex"

mkdir -p ${SIM_DIR}
mkdir -p ${SIM_DIR}/log

vcs -full64 -sverilog -timescale=1ns/1ps \
    -debug_access+all \
    -o ${SIM_DIR}/fc2_core_simv \
    ${RTL_DIR}/package.sv \
    ${RTL_DIR}/fc2_weight_bank.sv \
    ${RTL_DIR}/fc2_bias_bank.sv \
    ${RTL_DIR}/fc2_core.sv \
    ${TB_DIR}/tb_fc2_core.sv \
    2>&1 | tee ${SIM_DIR}/log/compile_tb_fc2_core.log

${SIM_DIR}/fc2_core_simv \
    +FC1_OUT_FILE=${FC1_OUT_HEX} \
    +FC2_WEIGHT_HEX_FILE=${FC2_WEIGHT_HEX} \
    +FC2_BIAS_FILE=${FC2_BIAS_HEX} \
    +FC2_ACC_FILE=${FC2_ACC_HEX} \
    +LOGITS_FILE=${LOGITS_HEX} \
    2>&1 | tee ${SIM_DIR}/log/sim_tb_fc2_core.log
```

---

## 5) `run_fc1_to_fc2_top.sh`

这个模块会对拍整条链：

- 输入 `fc1_acc_0.hex`
- 对拍 `fc1_relu_0.hex`
- 对拍 `fc1_out_0.hex`
- 对拍 `fc2_acc_0.hex`
- 对拍 `logits_0.hex`

```bash
#!/usr/bin/env bash
set -e

SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

FC1_ACC_HEX="../route_b_output/fc1_acc_0.hex"
FC1_RELU_HEX="../route_b_output/fc1_relu_0.hex"
FC1_OUT_HEX="../route_b_output/fc1_out_0.hex"
FC2_WEIGHT_HEX="../route_b_output/fc2_weight_int8.hex"
FC2_BIAS_HEX="../route_b_output/fc2_bias_int32.hex"
FC2_ACC_HEX="../route_b_output/fc2_acc_0.hex"
LOGITS_HEX="../route_b_output/logits_0.hex"

mkdir -p ${SIM_DIR}
mkdir -p ${SIM_DIR}/log

vcs -full64 -sverilog -timescale=1ns/1ps \
    -debug_access+all \
    -o ${SIM_DIR}/fc1_to_fc2_top_simv \
    ${RTL_DIR}/package.sv \
    ${RTL_DIR}/fc1_relu_requant.sv \
    ${RTL_DIR}/fc2_weight_bank.sv \
    ${RTL_DIR}/fc2_bias_bank.sv \
    ${RTL_DIR}/fc2_core.sv \
    ${RTL_DIR}/fc1_to_fc2_top.sv \
    ${TB_DIR}/tb_fc1_to_fc2_top.sv \
    2>&1 | tee ${SIM_DIR}/log/compile_tb_fc1_to_fc2_top.log

${SIM_DIR}/fc1_to_fc2_top_simv \
    +FC1_ACC_FILE=${FC1_ACC_HEX} \
    +FC1_RELU_FILE=${FC1_RELU_HEX} \
    +FC1_OUT_FILE=${FC1_OUT_HEX} \
    +FC2_WEIGHT_HEX_FILE=${FC2_WEIGHT_HEX} \
    +FC2_BIAS_FILE=${FC2_BIAS_HEX} \
    +FC2_ACC_FILE=${FC2_ACC_HEX} \
    +LOGITS_FILE=${LOGITS_HEX} \
    2>&1 | tee ${SIM_DIR}/log/sim_tb_fc1_to_fc2_top.log
```

---

# 我再补一句命名上的注意点

这些脚本里我默认你的模块内部 plusarg 名字分别是：

- `+FC1_ACC_FILE`
- `+FC1_RELU_FILE`
- `+FC1_OUT_FILE`
- `+FC2_WEIGHT_HEX_FILE`
- `+FC2_BIAS_FILE`
- `+FC2_ACC_FILE`
- `+LOGITS_FILE`

所以你的 **testbench 读 golden 文件** 时，也最好统一按这些名字写。

---

# 你接下来最稳的验证顺序

建议你按这个顺序跑：

1. `run_fc1_relu_requant.sh`
2. `run_fc2_weight_bank.sh`
3. `run_fc2_bias_bank.sh`
4. `run_fc2_core.sh`
5. `run_fc1_to_fc2_top.sh`

因为这样最容易定位问题：
- 先过量化
- 再过 bank
- 再过 FC2
- 最后过整链

如果你愿意，我下一条直接把这 5 个对应的 **testbench** 也一起给你。

用户:
请给出
AI助手:
可以，下面直接给你 5 个对应的 testbench，和我上一条的 `run.sh` 一一对应：

- `tb_fc1_relu_requant.sv`
- `tb_fc2_weight_bank.sv`
- `tb_fc2_bias_bank.sv`
- `tb_fc2_core.sv`
- `tb_fc1_to_fc2_top.sv`

我按你现在工程的风格来写：
- `string xxx_file`
- `plusargs`
- `$readmemh`
- `MATCH / ERROR / PASS / FAIL`

---

# 1) `tb_fc1_relu_requant.sv`

作用：
- 输入 `fc1_acc_0.hex`
- 对拍 `fc1_relu_0.hex`
- 对拍 `fc1_out_0.hex`

```systemverilog
`timescale 1ns / 1ps

module tb_fc1_relu_requant;
  import mnist_cim_pkg::*;

  logic signed [PSUM_WIDTH-1:0] fc1_acc_all [0:HIDDEN_DIM-1];
  logic signed [PSUM_WIDTH-1:0] fc1_relu_all[0:HIDDEN_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] fc1_out_all [0:HIDDEN_DIM-1];

  logic signed [PSUM_WIDTH-1:0] ref_fc1_acc_mem [0:HIDDEN_DIM-1];
  logic signed [PSUM_WIDTH-1:0] ref_fc1_relu_mem[0:HIDDEN_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] ref_fc1_out_mem[0:HIDDEN_DIM-1];

  string fc1_acc_file;
  string fc1_relu_file;
  string fc1_out_file;

  integer i;
  integer error_count;

  fc1_relu_requant dut (
      .fc1_acc_all(fc1_acc_all),
      .fc1_relu_all(fc1_relu_all),
      .fc1_out_all(fc1_out_all)
  );

  initial begin
    fc1_acc_file  = "../route_b_output/fc1_acc_0.hex";
    fc1_relu_file = "../route_b_output/fc1_relu_0.hex";
    fc1_out_file  = "../route_b_output/fc1_out_0.hex";

    if ($value$plusargs("FC1_ACC_FILE=%s", fc1_acc_file))
      $display("TB using FC1_ACC_FILE: %s", fc1_acc_file);
    else
      $display("TB using default FC1_ACC_FILE: %s", fc1_acc_file);

    if ($value$plusargs("FC1_RELU_FILE=%s", fc1_relu_file))
      $display("TB using FC1_RELU_FILE: %s", fc1_relu_file);
    else
      $display("TB using default FC1_RELU_FILE: %s", fc1_relu_file);

    if ($value$plusargs("FC1_OUT_FILE=%s", fc1_out_file))
      $display("TB using FC1_OUT_FILE: %s", fc1_out_file);
    else
      $display("TB using default FC1_OUT_FILE: %s", fc1_out_file);

    $readmemh(fc1_acc_file,  ref_fc1_acc_mem);
    $readmemh(fc1_relu_file, ref_fc1_relu_mem);
    $readmemh(fc1_out_file,  ref_fc1_out_mem);
  end

  initial begin
    error_count = 0;

    for (i = 0; i < HIDDEN_DIM; i = i + 1)
      fc1_acc_all[i] = ref_fc1_acc_mem[i];

    #1;

    $display("Checking fc1_relu_all ...");
    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
      if (fc1_relu_all[i] !== ref_fc1_relu_mem[i]) begin
        $display("ERROR RELU idx=%0d got=%0d expected=%0d",
                 i, fc1_relu_all[i], ref_fc1_relu_mem[i]);
        error_count = error_count + 1;
      end else begin
        $display("MATCH RELU idx=%0d value=%0d", i, fc1_relu_all[i]);
      end
    end

    $display("Checking fc1_out_all ...");
    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
      if (fc1_out_all[i] !== ref_fc1_out_mem[i]) begin
        $display("ERROR OUT idx=%0d got=%0d expected=%0d",
                 i, fc1_out_all[i], ref_fc1_out_mem[i]);
        error_count = error_count + 1;
      end else begin
        $display("MATCH OUT idx=%0d value=%0d", i, fc1_out_all[i]);
      end
    end

    if (error_count == 0)
      $display("PASS: fc1_relu_requant matches fc1_relu_0.hex and fc1_out_0.hex.");
    else
      $display("FAIL: found %0d mismatches.", error_count);

    $finish;
  end

endmodule
```

---

# 2) `tb_fc2_weight_bank.sv`

作用：
- 读取 `fc2_weight_int8.hex`
- 验证 bank 展开规则是否正确

这里我做的是对几个关键地址 spot-check，同时也全量遍历验证。

```systemverilog
`timescale 1ns / 1ps

module tb_fc2_weight_bank;
  import mnist_cim_pkg::*;

  logic signed [WEIGHT_WIDTH-1:0] w_all [0:FC2_OUT_DIM-1][0:FC2_IN_DIM-1];
  logic signed [WEIGHT_WIDTH-1:0] ref_weight_mem[0:FC2_WEIGHT_DEPTH-1];

  string fc2_weight_file;

  integer o, i, addr;
  integer error_count;

  fc2_weight_bank dut (
      .w_all(w_all)
  );

  initial begin
    fc2_weight_file = "../route_b_output/fc2_weight_int8.hex";

    if ($value$plusargs("FC2_WEIGHT_HEX_FILE=%s", fc2_weight_file))
      $display("TB using FC2_WEIGHT_HEX_FILE: %s", fc2_weight_file);
    else
      $display("TB using default FC2_WEIGHT_HEX_FILE: %s", fc2_weight_file);

    $readmemh(fc2_weight_file, ref_weight_mem);
  end

  initial begin
    error_count = 0;
    #1;

    for (o = 0; o < FC2_OUT_DIM; o = o + 1) begin
      for (i = 0; i < FC2_IN_DIM; i = i + 1) begin
        addr = o * FC2_IN_DIM + i;
        if (w_all[o][i] !== ref_weight_mem[addr]) begin
          $display("ERROR o=%0d i=%0d addr=%0d got=%0d expected=%0d",
                   o, i, addr, w_all[o][i], ref_weight_mem[addr]);
          error_count = error_count + 1;
        end
      end
    end

    if (error_count == 0)
      $display("PASS: fc2_weight_bank matches fc2_weight_int8.hex layout.");
    else
      $display("FAIL: found %0d mismatches.", error_count);

    $finish;
  end

endmodule
```

---

# 3) `tb_fc2_bias_bank.sv`

作用：
- 读取 `fc2_bias_int32.hex`
- 对拍 10 个 bias

```systemverilog
`timescale 1ns / 1ps

module tb_fc2_bias_bank;
  import mnist_cim_pkg::*;

  logic signed [BIAS_WIDTH-1:0] bias_all [0:FC2_OUT_DIM-1];
  logic signed [BIAS_WIDTH-1:0] ref_bias_mem[0:FC2_OUT_DIM-1];

  string fc2_bias_file;

  integer i;
  integer error_count;

  fc2_bias_bank dut (
      .bias_all(bias_all)
  );

  initial begin
    fc2_bias_file = "../route_b_output/fc2_bias_int32.hex";

    if ($value$plusargs("FC2_BIAS_FILE=%s", fc2_bias_file))
      $display("TB using FC2_BIAS_FILE: %s", fc2_bias_file);
    else
      $display("TB using default FC2_BIAS_FILE: %s", fc2_bias_file);

    $readmemh(fc2_bias_file, ref_bias_mem);
  end

  initial begin
    error_count = 0;
    #1;

    for (i = 0; i < FC2_OUT_DIM; i = i + 1) begin
      if (bias_all[i] !== ref_bias_mem[i]) begin
        $display("ERROR idx=%0d got=%0d expected=%0d",
                 i, bias_all[i], ref_bias_mem[i]);
        error_count = error_count + 1;
      end else begin
        $display("MATCH idx=%0d value=%0d", i, bias_all[i]);
      end
    end

    if (error_count == 0)
      $display("PASS: fc2_bias_bank matches fc2_bias_int32.hex.");
    else
      $display("FAIL: found %0d mismatches.", error_count);

    $finish;
  end

endmodule
```

---

# 4) `tb_fc2_core.sv`

作用：
- 输入 `fc1_out_0.hex`
- 对拍 `fc2_acc_0.hex`
- 对拍 `logits_0.hex`

```systemverilog
`timescale 1ns / 1ps

module tb_fc2_core;
  import mnist_cim_pkg::*;

  logic signed [OUTPUT_WIDTH-1:0] fc1_out_all[0:FC2_IN_DIM-1];
  logic signed [PSUM_WIDTH-1:0]   fc2_acc_all[0:FC2_OUT_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] logits_all [0:FC2_OUT_DIM-1];

  logic signed [OUTPUT_WIDTH-1:0] ref_fc1_out_mem[0:FC2_IN_DIM-1];
  logic signed [PSUM_WIDTH-1:0]   ref_fc2_acc_mem[0:FC2_OUT_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] ref_logits_mem [0:FC2_OUT_DIM-1];

  string fc1_out_file;
  string fc2_acc_file;
  string logits_file;

  integer i;
  integer error_count;

  fc2_core dut (
      .fc1_out_all(fc1_out_all),
      .fc2_acc_all(fc2_acc_all),
      .logits_all(logits_all)
  );

  initial begin
    fc1_out_file = "../route_b_output/fc1_out_0.hex";
    fc2_acc_file = "../route_b_output/fc2_acc_0.hex";
    logits_file  = "../route_b_output/logits_0.hex";

    if ($value$plusargs("FC1_OUT_FILE=%s", fc1_out_file))
      $display("TB using FC1_OUT_FILE: %s", fc1_out_file);
    else
      $display("TB using default FC1_OUT_FILE: %s", fc1_out_file);

    if ($value$plusargs("FC2_ACC_FILE=%s", fc2_acc_file))
      $display("TB using FC2_ACC_FILE: %s", fc2_acc_file);
    else
      $display("TB using default FC2_ACC_FILE: %s", fc2_acc_file);

    if ($value$plusargs("LOGITS_FILE=%s", logits_file))
      $display("TB using LOGITS_FILE: %s", logits_file);
    else
      $display("TB using default LOGITS_FILE: %s", logits_file);

    $readmemh(fc1_out_file, ref_fc1_out_mem);
    $readmemh(fc2_acc_file, ref_fc2_acc_mem);
    $readmemh(logits_file,  ref_logits_mem);
  end

  initial begin
    error_count = 0;

    for (i = 0; i < FC2_IN_DIM; i = i + 1)
      fc1_out_all[i] = ref_fc1_out_mem[i];

    #1;

    $display("Checking fc2_acc_all ...");
    for (i = 0; i < FC2_OUT_DIM; i = i + 1) begin
      if (fc2_acc_all[i] !== ref_fc2_acc_mem[i]) begin
        $display("ERROR ACC idx=%0d got=%0d expected=%0d",
                 i, fc2_acc_all[i], ref_fc2_acc_mem[i]);
        error_count = error_count + 1;
      end else begin
        $display("MATCH ACC idx=%0d value=%0d", i, fc2_acc_all[i]);
      end
    end

    $display("Checking logits_all ...");
    for (i = 0; i < FC2_OUT_DIM; i = i + 1) begin
      if (logits_all[i] !== ref_logits_mem[i]) begin
        $display("ERROR LOGIT idx=%0d got=%0d expected=%0d",
                 i, logits_all[i], ref_logits_mem[i]);
        error_count = error_count + 1;
      end else begin
        $display("MATCH LOGIT idx=%0d value=%0d", i, logits_all[i]);
      end
    end

    if (error_count == 0)
      $display("PASS: fc2_core matches fc2_acc_0.hex and logits_0.hex.");
    else
      $display("FAIL: found %0d mismatches.", error_count);

    $finish;
  end

endmodule
```

---

# 5) `tb_fc1_to_fc2_top.sv`

作用：
- 输入 `fc1_acc_0.hex`
- 对拍：
  - `fc1_relu_0.hex`
  - `fc1_out_0.hex`
  - `fc2_acc_0.hex`
  - `logits_0.hex`

```systemverilog
`timescale 1ns / 1ps

module tb_fc1_to_fc2_top;
  import mnist_cim_pkg::*;

  logic signed [PSUM_WIDTH-1:0]   fc1_acc_all [0:HIDDEN_DIM-1];
  logic signed [PSUM_WIDTH-1:0]   fc1_relu_all[0:HIDDEN_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] fc1_out_all [0:HIDDEN_DIM-1];
  logic signed [PSUM_WIDTH-1:0]   fc2_acc_all [0:FC2_OUT_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] logits_all  [0:FC2_OUT_DIM-1];

  logic signed [PSUM_WIDTH-1:0]   ref_fc1_acc_mem [0:HIDDEN_DIM-1];
  logic signed [PSUM_WIDTH-1:0]   ref_fc1_relu_mem[0:HIDDEN_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] ref_fc1_out_mem [0:HIDDEN_DIM-1];
  logic signed [PSUM_WIDTH-1:0]   ref_fc2_acc_mem [0:FC2_OUT_DIM-1];
  logic signed [OUTPUT_WIDTH-1:0] ref_logits_mem  [0:FC2_OUT_DIM-1];

  string fc1_acc_file;
  string fc1_relu_file;
  string fc1_out_file;
  string fc2_acc_file;
  string logits_file;

  integer i;
  integer error_count;

  fc1_to_fc2_top dut (
      .fc1_acc_all(fc1_acc_all),
      .fc1_relu_all(fc1_relu_all),
      .fc1_out_all(fc1_out_all),
      .fc2_acc_all(fc2_acc_all),
      .logits_all(logits_all)
  );

  initial begin
    fc1_acc_file  = "../route_b_output/fc1_acc_0.hex";
    fc1_relu_file = "../route_b_output/fc1_relu_0.hex";
    fc1_out_file  = "../route_b_output/fc1_out_0.hex";
    fc2_acc_file  = "../route_b_output/fc2_acc_0.hex";
    logits_file   = "../route_b_output/logits_0.hex";

    if ($value$plusargs("FC1_ACC_FILE=%s", fc1_acc_file))
      $display("TB using FC1_ACC_FILE: %s", fc1_acc_file);
    else
      $display("TB using default FC1_ACC_FILE: %s", fc1_acc_file);

    if ($value$plusargs("FC1_RELU_FILE=%s", fc1_relu_file))
      $display("TB using FC1_RELU_FILE: %s", fc1_relu_file);
    else
      $display("TB using default FC1_RELU_FILE: %s", fc1_relu_file);

    if ($value$plusargs("FC1_OUT_FILE=%s", fc1_out_file))
      $display("TB using FC1_OUT_FILE: %s", fc1_out_file);
    else
      $display("TB using default FC1_OUT_FILE: %s", fc1_out_file);

    if ($value$plusargs("FC2_ACC_FILE=%s", fc2_acc_file))
      $display("TB using FC2_ACC_FILE: %s", fc2_acc_file);
    else
      $display("TB using default FC2_ACC_FILE: %s", fc2_acc_file);

    if ($value$plusargs("LOGITS_FILE=%s", logits_file))
      $display("TB using LOGITS_FILE: %s", logits_file);
    else
      $display("TB using default LOGITS_FILE: %s", logits_file);

    $readmemh(fc1_acc_file,  ref_fc1_acc_mem);
    $readmemh(fc1_relu_file, ref_fc1_relu_mem);
    $readmemh(fc1_out_file,  ref_fc1_out_mem);
    $readmemh(fc2_acc_file,  ref_fc2_acc_mem);
    $readmemh(logits_file,   ref_logits_mem);
  end

  initial begin
    error_count = 0;

    for (i = 0; i < HIDDEN_DIM; i = i + 1)
      fc1_acc_all[i] = ref_fc1_acc_mem[i];

    #1;

    $display("Checking fc1_relu_all ...");
    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
      if (fc1_relu_all[i] !== ref_fc1_relu_mem[i]) begin
        $display("ERROR RELU idx=%0d got=%0d expected=%0d",
                 i, fc1_relu_all[i], ref_fc1_relu_mem[i]);
        error_count = error_count + 1;
      end
    end

    $display("Checking fc1_out_all ...");
    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
      if (fc1_out_all[i] !== ref_fc1_out_mem[i]) begin
        $display("ERROR OUT idx=%0d got=%0d expected=%0d",
                 i, fc1_out_all[i], ref_fc1_out_mem[i]);
        error_count = error_count + 1;
      end
    end

    $display("Checking fc2_acc_all ...");
    for (i = 0; i < FC2_OUT_DIM; i = i + 1) begin
      if (fc2_acc_all[i] !== ref_fc2_acc_mem[i]) begin
        $display("ERROR ACC idx=%0d got=%0d expected=%0d",
                 i, fc2_acc_all[i], ref_fc2_acc_mem[i]);
        error_count = error_count + 1;
      end
    end

    $display("Checking logits_all ...");
    for (i = 0; i < FC2_OUT_DIM; i = i + 1) begin
      if (logits_all[i] !== ref_logits_mem[i]) begin
        $display("ERROR LOGIT idx=%0d got=%0d expected=%0d",
                 i, logits_all[i], ref_logits_mem[i]);
        error_count = error_count + 1;
      end
    end

    if (error_count == 0)
      $display("PASS: fc1_to_fc2_top matches all golden files.");
    else
      $display("FAIL: found %0d mismatches.", error_count);

    $finish;
  end

endmodule
```

---

## 你现在可以按这个顺序跑
先跑：
1. `run_fc1_relu_requant.sh`
2. `run_fc2_weight_bank.sh`
3. `run_fc2_bias_bank.sh`
4. `run_fc2_core.sh`
5. `run_fc1_to_fc2_top.sh`

如果你需要，我下一条可以把这些模块和 testbench 里可能会遇到的 **package 缺失项** 一次性整理成一版可直接补进 `mnist_cim_pkg` 的内容。


