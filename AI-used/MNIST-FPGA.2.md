### FPGA CIM MNIST 问题分析

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
这是当前生成的数据，你主要先看一下route_b_output中的文件格式是否全部正确，然后给出一个markdown文件，用以说明route_b_output所有文件的使用方法和注意事项，要详细，而且要说明根据当前数据，硬件设计需要特别注意的地方。关于route_a_output,简单检查即可。
AI助手:


AI助手:
我先把两个压缩包解开，核对 `route_b_output` 的文件清单、每类文件的位宽和行数，再抽样检查几个样本文件是否彼此自洽。然后我会生成一份完整的 Markdown 说明，专门给你们做硬件联调用。

AI助手:
我已经核对出 `route_b_output` 的主要文件都在，位宽和行数目前是自洽的。接下来我再补两件更关键的事：一是看数值范围是否会踩硬件位宽坑，二是把这些坑写进说明文档里，避免你们后面在阵列映射、signed 解释、requant 上出错。

AI助手:
我已经检查完了，结论是：

`route_b_output` 的主文件格式**整体正确**，而且已经具备作为 FPGA / CIM 整数推理联调数据的条件。  
需要硬件特别注意的核心点有三个：

1. **第一层输入零点不是 0，而是 `-128`**，不能直接做 `x_q * w_q`，必须处理输入零点补偿。  
2. **第一层累加范围明显超过 16 位**，当前数据已经出现大约 `-5.5e5 ~ 3.46e5`，所以 `fc1` 累加器和 bias 路径必须至少 `int32`。  
3. **当前权重文件是逻辑布局 `row-major [out][in]`**，很适合 testbench 和 RTL 功能验证，但不一定等于你们 CIM 阵列的物理烧写布局，后续可能还需要做一层 tile 重排。

`route_a_output` 我也做了简查，主文件没有明显格式问题，适合继续用于单层 `fc1 + ReLU` 验证；但它不是最终两层分类演示版本。

我已经把完整说明整理成 Markdown 文件，里面包括：

- `route_b_output` 每个文件的格式、行数、位宽、布局
- 各文件的具体用途
- 推荐的硬件联调顺序
- 当前数据下必须特别注意的硬件坑
- `route_a_output` 的简要检查结论

下载这里：

[route_b_output 使用说明与硬件注意事项](sandbox:/mnt/data/route_b_output_usage_notes.md)

如果你愿意，我下一步可以继续直接帮你补一份**面向 Verilog/Testbench 的读取模板**，把这些文件如何 `$readmemh`、如何按 signed 解释、如何逐级对拍写成可直接用的代码框架。

AI助手:


AI助手:
我先逐项核对 `route_b_output` 里的实际文件名、位宽、行数和内容一致性，再补一个详细的使用说明文档。`route_a_output` 我会做一轮轻量检查，重点看有没有明显格式错误或和 `route_b` 不一致的地方。

AI助手:
`route_b_output` 的主文件看起来是自洽的：位宽、行数、层级命名和量化配置基本能对上。我再补一轮一致性检查，重点看标签/预测是否匹配，以及 `route_a_output` 有没有明显格式问题。

用户:
route_b_output的使用指南markdown没有生成出来，请你重新尝试给出一个markdown文件，用以说明route_b_output所有文件的使用方法和注意事项，要详细，而且要说明根据当前数据，硬件设计需要特别注意的地方。
AI助手:


AI助手:
已重新生成 Markdown 文件：

[route_b_output 使用指南（Markdown）](sandbox:/mnt/data/route_b_output_usage_guide.md)

我已经把这些内容写进去：
- `route_b_output` 全部文件的格式、行数、位宽、含义
- 每类文件在 FPGA / CIM 中怎么用
- 推荐的联调顺序
- `$readmemh` 的典型用法
- 基于当前数据，硬件设计必须特别注意的点
- 对 `route_a_output` 的简要检查结论

这份文档是按你当前实际导出的 `route_b_output` 内容写的，不是泛泛模板。

用户:
还记得我最开始上传的三个文件吗？项目设计指导书，上次和AI的对话，请你重新看一下，我在完成这个测试向量生成之后，应该做什么？
AI助手:


AI助手:


AI助手:


AI助手:
我记得你最开始给的三个核心材料：

1. **项目设计指导书**（课程/课设要求）  
2. **你和上一个 AI 的对话记录**（确定了大体方案：FPGA + CIM + MNIST）  
3. **生成测试向量的 ipynb**（用于导出权重、输入、golden）

现在你已经完成了 **route_b_output 的测试向量生成**。  
这一步其实在整个项目流程里属于 **软件侧 → 硬件侧的接口准备阶段**。

接下来你要做的事情其实非常明确：  

**进入硬件实现和验证阶段。**

我给你按 **真正工程流程**排一个清晰路线（也是最符合你这个项目指导书逻辑的路线）。

---

# 一、你现在所处的位置

你的项目整体流程其实是：

```
PyTorch训练
      ↓
量化
      ↓
整数推理参考模型
      ↓
生成测试向量 (route_b_output)  ← 你现在完成到这里
      ↓
FPGA/CIM 硬件实现
      ↓
testbench 对拍验证
      ↓
板级运行
      ↓
项目展示
```

所以现在你已经完成：

**软件侧所有准备工作。**

接下来全部工作都是：

**硬件实现 + 硬件验证。**

---

# 二、下一步你应该做什么（非常具体）

下一步应该分 **4个阶段**。

---

# 阶段 1：设计硬件计算结构

现在要把你 Python 的网络结构映射成 **硬件模块结构**。

你的网络是：

```
784 → 128 → 10
```

对应硬件结构：

```
input_buffer (784)

      ↓

FC1  (784×128)  ← CIM阵列

      ↓

bias_add

      ↓

ReLU

      ↓

requant (int32→int8)

      ↓

FC2  (128×10)

      ↓

argmax
```

建议模块划分：

```
top
 ├── input_loader
 ├── fc1_cim_array
 ├── bias_add
 ├── relu
 ├── requant
 ├── fc2
 └── argmax
```

如果你的课设重点是 **CIM**，通常只把：

```
fc1
```

做成 CIM 阵列。

后面的：

```
bias
relu
requant
fc2
```

可以在 FPGA 普通逻辑实现。

---

# 阶段 2：建立 Testbench 验证系统

这是你 **下一步最重要的事情**。

你已经有：

```
route_b_output/
```

里面包含：

```
fc1_weight_int8.hex
fc1_bias_int32.hex
fc2_weight_int8.hex
fc2_bias_int32.hex
input_*.hex
fc1_acc_*.hex
fc1_relu_*.hex
fc1_out_*.hex
fc2_acc_*.hex
logits_*.hex
pred_*.txt
labels.txt
```

Testbench 要做的事情：

### 1 读取权重

```
$readmemh("fc1_weight_int8.hex", weight_mem);
$readmemh("fc1_bias_int32.hex", bias_mem);
```

---

### 2 读取输入

```
$readmemh("input_0.hex", input_mem);
```

---

### 3 驱动 DUT

输入 784 个数据

---

### 4 接收 DUT 输出

例如：

```
fc1_acc
fc1_relu
fc1_out
fc2_acc
logits
```

---

### 5 与 golden 对比

例如：

```
$readmemh("fc1_acc_0.hex", golden_fc1_acc);
```

然后逐项比较：

```
if(dut_acc[i] != golden_fc1_acc[i])
```

---

# 阶段 3：逐层验证（非常重要）

不要一口气跑完整网络。

必须 **逐层验证**。

建议顺序：

---

## Step1

只验证

```
FC1 MAC
```

对拍：

```
fc1_acc_*.hex
```

---

## Step2

验证

```
FC1 + bias
```

---

## Step3

验证

```
FC1 + bias + relu
```

对拍：

```
fc1_relu_*.hex
```

---

## Step4

验证

```
FC1 + requant
```

对拍：

```
fc1_out_*.hex
```

---

## Step5

再加入

```
FC2
```

对拍：

```
fc2_acc_*.hex
```

---

## Step6

验证

```
logits
```

对拍：

```
logits_*.hex
```

---

## Step7

验证

```
argmax
```

对拍：

```
pred_i.txt
```

---

# 阶段 4：板级运行

如果仿真完全正确，就可以上 FPGA。

板级流程：

```
UART / DMA / BRAM
      ↓
输入 784
      ↓
网络计算
      ↓
输出 10 logits
      ↓
argmax
      ↓
输出数字
```

展示方式可以是：

- 串口打印
- 数码管显示
- HDMI 输出
- PC terminal

---

# 三、你这个项目的真正重点

根据你给的指导书和对话，我基本可以判断：

老师真正想看的不是：

```
PyTorch训练
```

而是：

**FPGA + CIM实现。**

重点通常是：

### 1 CIM阵列设计

例如：

```
bitline accumulate
wordline control
ADC / 数字累加
```

### 2 数据流设计

```
input → CIM → partial sum → accumulate
```

### 3 存储结构

```
BRAM
SRAM
寄存器
```

### 4 控制器

```
FSM
```

---

# 四、你现在最应该马上开始的事

如果我是你，我会按这个顺序做：

### 第1步

写

```
fc1 MAC模块
```

接口：

```
input[784]
weight
bias
```

输出：

```
128
```

---

### 第2步

写

```
testbench
```

使用：

```
route_b_output
```

---

### 第3步

验证

```
fc1_acc
```

---

### 第4步

加

```
relu
requant
```

---

### 第5步

写

```
fc2
```

---

### 第6步

加

```
argmax
```

---

# 五、你现在其实已经完成了 70% 的软件工作

很多做 FPGA ML 项目的人卡在：

```
数据接口
```

而你已经有：

- 量化
- 整数推理
- golden
- hex导出

这其实是 **最难的一部分**。

剩下主要是：

**RTL实现。**

---

# 六、我建议你下一步让我帮你的事情

如果你愿意，我可以直接帮你做 **三个关键东西**：

### 1  
给你画一个完整的

**FPGA + CIM 网络结构图**

（非常适合写报告）

---

### 2  
帮你写一个

**标准 Verilog testbench**

直接读取：

```
route_b_output
```

并自动对拍。

---

### 3  
帮你设计

**FC1 CIM阵列结构**

包括：

```
阵列尺寸
并行度
BRAM布局
地址生成
```

这样你这个项目基本就可以直接往完成方向走了。

用户:
请按照route_b_output的数据帮我完成阶段１的硬件设计，要求使用systemverilog设计。
AI助手:


用户:
请按照route_b_output的数据帮我完成阶段１的硬件设计，要求使用systemverilog设计。
AI助手:


AI助手:
我先按 `route_b_output` 的实际文件组织，把阶段 1 的硬件模块接口、数据位宽、存储布局和控制流程对齐好，再给你一套可直接落到 RTL 的 SystemVerilog 设计骨架。会优先保证它和你当前导出的 `fc1/fc2` 权重、bias、输入、golden 文件能一一对上。

AI助手:
我已经确认 `route_b_output` 的关键参数和文件布局：`fc1` 是 `128x784`、`fc2` 是 `10x128`，输入零点是 `-128`，两层权重零点都是 `0`，而且 `fc1_out_0.hex` 的数值范围也印证了当前第一层 requant 的目标输出就是 int8。接下来我给你的 RTL 会严格按这套数据组织来设计，不会另起一套格式。

AI助手:
我按 `route_b_output` 的实际数据组织，完成了一套阶段 1 的 SystemVerilog 硬件设计骨架，重点是先把**数据格式、位宽、量化参数、地址顺序**和你当前导出的文件完全对齐。

文件在这里：

- [SystemVerilog 设计目录](sandbox:/mnt/data/stage1_sv_design)
- [打包 ZIP](sandbox:/mnt/data/stage1_sv_design.zip)

这套设计包含：

- `mnist_mlp_pkg.sv`  
  把 `quant_config.json` 里的关键参数固化进 RTL
- `hex_rom_int8.sv` / `hex_rom_int32.sv`  
  直接读取 `route_b_output` 里的 int8 / int32 hex 文件
- `fc_layer_seq.sv`  
  顺序版全连接计算引擎，便于先验证软件/硬件接口
- `relu_vec.sv`
- `requantize_int32_to_int8.sv`
- `requant_vec.sv`
- `argmax_int8_vec.sv`
- `mnist_mlp_top.sv`  
  顶层 `784 -> 128 -> 10` 网络骨架
- `README.md`  
  说明各模块用途和下一步替换 CIM 的位置

先提醒你三点，这和你当前 `route_b_output` 很相关：

1. 这版是**阶段 1 架构版**，核心目标是先把接口和数据流跑通，不是最终高性能 CIM 宏实现。  
2. `fc_layer_seq.sv` 现在是**顺序 MAC 引擎**，后面你可以把 `fc1` 这一层替换成真正的 CIM 阵列。  
3. requant 我已经按当前 `multiplier_q31 + shift` 写了固定点实现，但**是否与 Python 完全 bit-true**，仍然建议你下一步用 `fc1_acc_0.hex / fc1_out_0.hex / logits_0.hex` 做 testbench 对拍确认。

你现在最自然的下一步就是：基于这套 RTL，我继续帮你写一个**SystemVerilog testbench**，直接读取 `route_b_output` 的 golden 文件做逐级对拍。

用户:
你所写的代码有体现CIM吗？具体在哪里？是否符合项目设计指导书？请给我解答
AI助手:


AI助手:
我先把项目指导书里对“CIM/FPGA 实现”的要求和我给你的这版 RTL 骨架逐项对一下，然后明确告诉你：哪些地方只是“为 CIM 预留接口”，哪些地方已经算真正体现了 CIM，哪些地方还没做到。

用户:

用户:
No search results matched your queries.

Note that a source being available does not imply that the user has synced data from that source yet; in case they haven't synced / connected a source yet, no documents will be returned from that source. If you get no results after a filtered search, you can also try adjusting the source_filter to query more sources.

AI助手:
我重新对照了你最开始上传的**项目设计指导书**和那份**上次对话记录**，结论很明确：

## 结论

**我给你的那版 SystemVerilog 代码，没有“真正实现 CIM 宏单元”。**  
它更准确地说是：

**一套按 `route_b_output` 数据格式对齐的、可综合的 MLP 整数推理 RTL 骨架，用来做阶段 1 的数据通路验证。**

所以答案分两层：

### 1）有没有体现 CIM？
**严格说，没有真正体现。**

### 2）是否符合项目设计指导书？
**如果按“最终作品要求”看，不符合。**  
**如果按“阶段 1 / 极简验证版”看，部分符合。**

---

# 一、为什么说“没有真正体现 CIM”

你项目指导书里对核心创新点的要求其实很清楚，重点是：

- 设计一个**包含 CIM 宏单元**的 FPGA 验证系统
- CIM 宏单元要模拟“**存算一体阵列**”行为
- 可以是**数字型实现**，例如：
  - 用 FPGA 的 BRAM 构建“存储阵列”
  - 设计周边并行乘加逻辑
  - 实现矩阵向量乘加
- 重点是：
  - **控制接口**
  - **数据流**
  - **计算时序**

而我给你的这版 RTL 里，核心计算模块是：

- `fc_layer_seq.sv`
- `mnist_mlp_top.sv`

其中 `fc_layer_seq.sv` 的本质是：

- 从 hex ROM 读权重
- 从输入向量里取一个元素
- 做一次 `int8 x int8 -> int32`
- 顺序累加
- 最后加 bias

也就是一个**顺序式 FC 引擎**，不是阵列式 CIM 宏。

---

# 二、具体“像 CIM 的地方”在哪里

如果一定要说“这版代码里有哪些地方带了 CIM 味道”，只有两个点：

## 1）权重和计算局部靠近
在 `fc_layer_seq.sv` 里：

- `hex_rom_int8` 作为权重存储
- 乘加逻辑和地址生成在同一个计算模块里

这在概念上接近：

**“把权重放在本地存储旁边计算”**

这确实有一点“memory-near-compute”的影子。

## 2）它把 `fc1` 预留成了未来可替换的 CIM 层
在 `README.md` 里我也写了：

- 先用 `fc_layer_seq`
- 后续把 `fc1` 替换成真正的 CIM datapath

所以它是一个**CIM 占位架构**，不是 CIM 本体。

---

# 三、为什么它还不能算真正的 CIM

因为真正的 CIM，至少应该体现出下面几项中的大部分，而当前代码都没有完整体现。

## 1）没有“阵列”
当前 `fc_layer_seq.sv` 是：

- 单个 MAC
- 顺序遍历 `IN_DIM`
- 顺序遍历 `OUT_DIM`

它不是：

- 多列权重阵列
- 多行输入广播
- 并行 bitline / column accumulate

也就是说没有“阵列感”。

---

## 2）没有“并行存内计算”
指导书强调可以用 FPGA 的 BRAM 模拟数字型 CIM 阵列。  
这种情况下，常见做法应该是：

- 权重按 tile 存在多个 bank
- 输入广播到多个 bank
- 多个 PE / MAC 并行工作
- 每拍产出一组 partial sum

但你现在的 `fc_layer_seq.sv` 是：

- 每拍只做一个乘法
- 顺序扫完整个向量

这本质上还是：

**普通时分复用 MAC**

不是存算阵列。

---

## 3）没有“CIM 外围控制”
CIM 宏单元通常还要有：

- weight preload / write interface
- input broadcast control
- tile select
- partial sum merge
- done / interrupt / status
- layer config

当前代码基本没有这些“宏单元外设接口”。

它只有：

- `start`
- `busy`
- `done`

这更像一个简化的运算核，不像一个可集成到 SoC 的 CIM IP。

---

## 4）没有体现“访存结构与计算单元匹配”
指导书里还有一个很重要的点：

> 对电路的访存结构进行设计和优化，使其与计算单元的性能相匹配。

当前代码里的权重读取是：

\[
addr = out\_idx \times IN\_DIM + in\_idx
\]

这是为了和 `route_b_output` 对齐，没问题。  
但它还没有做：

- bank 划分
- tile 映射
- 宽字打包
- 多端口组织
- 输入缓存/输出缓存结构优化

所以离“CIM 访存结构设计”还差一大段。

---

# 四、所以它到底符不符合项目设计指导书

## 如果按“最终项目要求”判断：
**不符合。**

因为指导书要的是：

- FPGA 上的 **CIM 宏单元**
- SoC 系统
- 软件驱动
- 外设 / 总线
- 系统级调试
- 性能/功能验证

而我给你的这版代码目前只有：

- MLP 推理 datapath 骨架
- route_b_output 对齐
- 顶层控制 FSM
- 基本 post-processing

缺少：

- 真正的 CIM array / macro
- SoC 集成
- AXI / UART / DDR
- 软核控制
- 驱动软件接口
- 系统级验证框架

---

## 如果按“阶段 1 / 极简验证版”判断：
**部分符合，而且是合理的第一步。**

因为你们之前的路线本来就是：

- 先做极简版
- 先验证单层或小网络推理
- 再逐步扩展成真正的 CIM SoC

从这个角度看，这版 RTL 的作用是：

- 先把 `route_b_output` 的数据格式跑通
- 先确认整数链路和硬件接口
- 先搭起顶层结构
- 为后续真正的 CIM 模块替换留位置

所以它适合作为：

**阶段 1 的“前置验证版 / 架构骨架”**

但不能直接宣称：

**“我们已经完成了 CIM 宏单元设计”**

这个说法站不住。

---

# 五、当前代码里最需要你警惕的地方

你如果拿这版代码去答辩，老师一问“CIM 在哪里”，你会比较被动。  
因为当前最容易被指出来的问题就是：

## 1）`fc_layer_seq` 是串行 FC，不是 CIM 阵列
老师一看就会发现：

- 没有多 bank 权重阵列
- 没有输入广播
- 没有并行列累加
- 没有 tile 结构

所以不能把它说成“CIM 核心”，最多说它是：

**用于对齐软件模型与量化数据的参考计算核**

---

## 2）顶层 `mnist_mlp_top` 是网络骨架，不是 SoC
它没有：

- AXI slave
- 寄存器映射
- DMA
- 中断
- 软核处理器接口

所以它也不能称为：

**符合指导书的 SoC 系统**

---

## 3）当前更像“验证平台中的参考 RTL”
这版 RTL 最准确的定位应该是：

**硬件设计前期的参考数据通路模型**

这个说法是成立的。

---

# 六、如果要“符合指导书”，你下一步必须补什么

你真正要补的是一个**CIM 宏单元**，而不是继续把 `fc_layer_seq` 堆复杂。

## 必补 1：真正的 `cim_core.sv`
至少要把 `fc1` 替换成一个阵列化模块，例如：

- `cim_core.sv`
- `cim_bank.sv`
- `cim_pe.sv`
- `psum_merge.sv`
- `cim_ctrl.sv`

### 这个 `cim_core` 至少要有
- 权重按 bank/tile 存储
- 输入广播到多个 bank
- 多个列/PE 并行计算
- partial sum 输出
- bias / relu / requant 接口

这时你才能说：

**我们实现了数字型 CIM 宏单元。**

---

## 必补 2：把 `fc1` 明确定位为 CIM 层
你这个 `route_b_output` 天然适合这样分工：

- `fc1 (784 -> 128)`：做成 CIM 阵列
- `fc2 (128 -> 10)`：先用普通 FPGA 逻辑实现

这非常符合项目推进节奏，也容易解释。

---

## 必补 3：设计访存结构
你现在必须开始考虑：

- 阵列并行度是多少
- 一次并行几个输出通道
- 输入一次送几维
- 权重如何切 tile
- BRAM 如何映射
- 地址如何生成

这部分才是指导书里特别看重的“存储结构与计算单元匹配”。

---

## 必补 4：后续要有 SoC 壳层
最终至少要补一个轻量版本：

- 寄存器接口
- `start/status/done`
- 权重装载
- 输入装载
- 结果读取

哪怕一开始不做完整 AXI，也要有“像 IP 核”的接口。

---

# 七、你在答辩里应该怎么描述当前这版代码

你现在最稳妥的说法不是“这就是 CIM 实现”，而是：

> 这版 RTL 是阶段 1 的硬件参考架构，主要用于对齐 `route_b_output` 的量化数据格式、位宽、地址顺序和推理数据流，验证 MLP 整数推理链路的正确性。  
> 真正的 CIM 宏单元将以 `fc1` 为目标层，在此基础上进一步替换为 bank/tile 化的并行存算阵列实现。

这个表述是专业且站得住的。

---

# 八、最终回答你的三个问题

## 1）你所写的代码有体现 CIM 吗？
**严格说没有真正体现。**

只有“局部存储 + 本地计算”的概念性影子，以及为后续替换 CIM 留出的层级结构。

---

## 2）具体在哪里？
如果勉强说“体现”，主要在：

- `fc_layer_seq.sv` 中权重 ROM 与 MAC 的局部耦合
- `mnist_mlp_top.sv` 中把 `fc1` 作为可替换的核心计算层

但这只能算：

**CIM 占位 / 预留接口**

不能算真正的 CIM 宏单元。

---

## 3）是否符合项目设计指导书？
### 按最终作品要求：
**不符合。**

### 按阶段 1 极简验证要求：
**部分符合，可作为前置验证版。**

---

你现在最该做的不是继续问“这版算不算 CIM”，而是直接进入下一步：  
**我帮你把 `fc1` 改成一个真正带 bank / tile / 并行 MAC 的 `cim_core.sv`。**

这样你就能从“参考 RTL”进入“符合指导书的 CIM 硬件设计”。

用户:
如何写出具体阵列，并行存内计算？讲解具体思路
AI助手:


AI助手:
要把“具体阵列、并行存内计算”真正写出来，核心不是先写代码，而是先把这 5 件事定死：

1. **阵列一次吃多少输入**  
2. **阵列一次并行算多少输出通道**  
3. **权重怎么切块存进阵列**  
4. **部分和怎么累加**  
5. **阵列外面谁负责 bias / ReLU / requant / 第二层**

你这个项目最适合走的是：

**把 `fc1: 784 -> 128` 做成数字型 CIM 阵列，`fc2: 128 -> 10` 先用普通 FPGA 逻辑实现。**

这样既符合指导书“CIM 宏单元”的要求，也不会一下子把复杂度拉爆。

---

# 一、先明确：你要实现的“CIM”是什么

你这个项目里，CIM 不一定是模拟电路那种真正电流求和。  
指导书也允许你做：

**数字型 CIM 宏单元**

也就是：

- 用 BRAM / ROM / 寄存器阵列存权重
- 输入向量广播到一组计算单元
- 多列并行做乘加
- 在阵列内部或阵列边缘做部分和累加

所以它的本质是：

**memory-centric parallel MAC array**

不是单个 MAC 顺序扫完整个向量。

---

# 二、从你的网络出发，先做阵列映射

你的第一层是：

\[
Y = W X + b
\]

其中：

- `X`：784 维输入
- `W`：`128 x 784`
- `Y`：128 维输出

如果直接做一个完整阵列：

- 行方向：输入维度 784
- 列方向：输出维度 128

理论上是一个 **784 x 128** 的权重矩阵阵列。

但 FPGA 上不可能一次全并行做完，所以一定要**分块 tile 化**。

---

# 三、最关键的设计：先定阵列尺寸

这是第一步。

你需要选一个阵列 tile，例如：

- 每次并行处理 **16 个输出通道**
- 每拍处理 **16 个输入元素**

那一个 tile 的逻辑就可以定义成：

\[
TIN = 16,\quad TOUT = 16
\]

意思是：

- 输入方向并行度 = 16
- 输出方向并行度 = 16

一个 tile 每拍做：

- 16 个输入广播
- 16 个输出列同时计算
- 共 `16 x 16 = 256` 个乘法

这已经很像一个数字型 CIM 阵列了。

---

# 四、为什么要这样分

因为你的第一层是 `128 x 784`。

如果设：

- `TIN = 16`
- `TOUT = 16`

那么：

- 输入 784 需要切成 `784 / 16 = 49` 个输入块
- 输出 128 需要切成 `128 / 16 = 8` 个输出块

所以整个 `fc1` 会被切成：

\[
8 \times 49 = 392 \text{ 个 tile 计算片段}
\]

每个 tile 负责：

- 16 个输出神经元
- 对应输入向量中的 16 个元素

最后把 49 次输入块的部分和加起来，就得到这 16 个输出神经元的最终结果。

---

# 五、阵列在数学上到底做什么

假设当前输出块编号是 `ob`，输入块编号是 `ib`。

那么：

- 当前输入子向量：

\[
X_{ib} \in \mathbb{Z}^{16}
\]

- 当前权重子矩阵：

\[
W_{ob,ib} \in \mathbb{Z}^{16 \times 16}
\]

- 当前输出部分和：

\[
PSUM_{ob} \in \mathbb{Z}^{16}
\]

每拍阵列做的事情就是：

\[
PSUM_{ob}[j] += \sum_{i=0}^{15} X_{ib}[i] \cdot W_{ob,ib}[j][i]
\]

其中 `j=0..15`。

这就是：

**16 路输入广播到 16 列权重阵列，得到 16 个并行输出部分和。**

这才是你要的“具体阵列”。

---

# 六、硬件结构怎么画

你可以把一个 `16x16` 数字型 CIM tile 画成这样：

```text
                 x0 x1 x2 ... x15
                  |  |  |      |
                  v  v  v      v
      +----------------------------------+
row 0 | w00 w01 w02 ... w0,15 | --> sum0 |
row 1 | w10 w11 w12 ... w1,15 | --> sum1 |
row 2 | w20 w21 w22 ... w2,15 | --> sum2 |
 ...  | ...                    |   ...   |
row15 | w15,0 ...       w15,15 | --> sum15
      +----------------------------------+
```

这里：

- 每一行对应一个输出通道
- 每一列对应一个输入元素
- 输入向量 16 个元素被广播到所有行
- 每行做 16 项乘加
- 得到 16 个输出部分和

这就是最典型的数字型 CIM tile 思路。

---

# 七、RTL 上怎么拆模块

真正写 SystemVerilog 时，不要一上来写大顶层。  
先拆成下面 5 个模块。

---

## 1. `cim_tile.sv`
这是最核心模块。

职责：

- 存一个 `TOUT x TIN` 的权重子矩阵
- 接收 `TIN` 维输入子向量
- 输出 `TOUT` 维部分和增量

输入输出可以定义成：

```systemverilog
module cim_tile #(
    parameter int TIN  = 16,
    parameter int TOUT = 16
)(
    input  logic signed [7:0] x_in   [0:TIN-1],
    input  logic signed [7:0] w_tile [0:TOUT-1][0:TIN-1],
    output logic signed [31:0] psum_out [0:TOUT-1]
);
```

内部逻辑：

```systemverilog
for each output row j:
    psum_out[j] = sum_i (x_in[i] * w_tile[j][i]);
```

这是一个**组合版** tile。

如果时序压力大，可以改成流水版。

---

## 2. `cim_bank.sv`
职责：

- 存多个 tile 的权重
- 根据 `out_block_idx` 和 `in_block_idx` 取出当前 tile 的 `16x16` 权重
- 送到 `cim_tile`

它相当于：

**阵列的存储外壳**

---

## 3. `psum_buffer.sv`
职责：

- 保存某个输出块的 16 个部分和
- 每完成一个输入块，就累加一次

也就是：

\[
psum[j] \leftarrow psum[j] + tile\_out[j]
\]

这个模块很关键，因为 784 维输入不是一拍算完，是 49 个输入块逐步累加。

---

## 4. `cim_core.sv`
职责：

- 调度输入块和输出块
- 控制 bank 取权重
- 控制 tile 开始计算
- 控制 psum 累加
- 49 个输入块结束后输出一个 16 维结果块

它是：

**真正的 CIM 宏控制器**

---

## 5. `fc1_cim_wrapper.sv`
职责：

- 在 `cim_core` 后面接：
  - bias add
  - ReLU
  - requant
- 输出 `fc1_out`

这样 `fc1` 这一层就完整了。

---

# 八、数据流到底怎么走

这里是最重要的部分。  
你一旦把这个数据流讲明白，RTL 就很好写。

---

## Step 1：输入缓存
先把 `input_0.hex` 这种 784 维 int8 输入装到 `input_buffer[0:783]`

---

## Step 2：输出块循环
因为 `TOUT = 16`，一共 128 个输出，所以要跑 8 个输出块：

- 输出块 0：输出 0~15
- 输出块 1：输出 16~31
- ...
- 输出块 7：输出 112~127

---

## Step 3：清空 psum
对当前输出块，先把 16 个部分和寄存器清零：

\[
PSUM[0..15] = 0
\]

---

## Step 4：输入块循环
因为 `TIN = 16`，784 个输入要跑 49 个输入块：

- 输入块 0：输入 0~15
- 输入块 1：输入 16~31
- ...
- 输入块 48：输入 768~783

每次做：

1. 从输入 buffer 取 16 个输入
2. 从权重 bank 取当前 `16x16` tile
3. 送入 `cim_tile`
4. 得到 16 个 `tile_psum`
5. 累加到 `PSUM`

---

## Step 5：输入块结束
49 个输入块全部完成后，当前输出块的 `PSUM[0..15]` 就是：

\[
\sum_{ib=0}^{48} W_{ob,ib} \cdot X_{ib}
\]

即完整的 16 个输出神经元的 int32 累加结果。

---

## Step 6：后处理
然后做：

1. 加 bias
2. ReLU
3. requant 到 int8

得到 `fc1_out[16]`

---

## Step 7：写回输出 buffer
写到：

- `fc1_out_buffer[out_block*16 + k]`

---

## Step 8：切换到下一个输出块
重复直到 8 个输出块全部完成。

---

# 九、这套结构为什么更像 CIM

因为它具备了几个关键特征：

## 1. 权重驻留在本地 bank
不是每拍从全局存储器乱读，而是：

- 每个 tile 的权重局部存储
- 输入广播过去参与计算

这符合 CIM 的“权重驻留”。

## 2. 一次并行算多个输出
不是单 MAC，而是 16 路输出同时算。

## 3. 输入广播到所有列
这很像阵列型计算的典型数据流。

## 4. 局部部分和累加
不是每个结果算完再回全局存储，而是在阵列侧维护 psum。

这比我之前给你的 `fc_layer_seq` 更符合指导书。

---

# 十、当前 `route_b_output` 怎么映射到这个阵列

你现在权重导出顺序是：

\[
W[j][i]
\]

也就是 row-major `[out][in]`。

这对 tile 化很友好。

---

## 当前 tile 权重地址公式

如果：

- `ob` = 输出块号
- `ib` = 输入块号
- `j` = tile 内输出索引 `0..15`
- `i` = tile 内输入索引 `0..15`

那么全局坐标：

\[
out = ob \times 16 + j
\]
\[
in = ib \times 16 + i
\]

在 `fc1_weight_int8.hex` 中的线性地址就是：

\[
addr = out \times 784 + in
\]

也就是：

\[
addr = (ob \times 16 + j)\times 784 + (ib \times 16 + i)
\]

这就是你在 `cim_bank` 里取 tile 时要用的地址公式。

---

# 十一、具体并行度怎么选

你不一定非要 `16x16`。  
要根据 FPGA 资源选。

---

## 方案 A：8x8
优点：
- 资源压力小
- 容易时序收敛

缺点：
- 拍数多

---

## 方案 B：16x16
优点：
- 比较平衡
- 很适合课设展示“阵列”

缺点：
- DSP / LUT 压力会明显上升

---

## 方案 C：32x8 或 8x32
这是一种非方阵设计。

### `TIN=32, TOUT=8`
- 每拍吞更多输入
- 输出并行少

### `TIN=8, TOUT=32`
- 每拍吞输入少
- 输出并行多

对你这个项目，我建议首版用：

\[
TIN = 16,\quad TOUT = 16
\]

最容易解释，也最像阵列。

---

# 十二、你在 RTL 里要特别注意的 6 个问题

---

## 1. 输入 zero-point
你当前输入量化是：

\[
x_q = \text{round}(x \cdot 255) - 128
\]

所以硬件乘加时要注意你是算：

\[
(x_q - z_x)(w_q - z_w)
\]

若：

- `z_x = -128`
- `z_w = 0`

则：

\[
x_q - z_x = x_q + 128
\]

这件事必须在硬件里说明白。  
否则你会发现结果和 golden 不一致。

### 更工程化的做法
你可以在输入进入阵列前就做一次：

```text
x_eff = x_q + 128
```

然后阵列内只算：

```text
x_eff * w_q
```

这样阵列逻辑更干净。

---

## 2. 权重是有符号 int8
读取 `fc1_weight_int8.hex` 时必须按 signed 解释。

---

## 3. psum 一定要 int32
因为一个输出神经元要累加 784 项，int16 不够稳。

---

## 4. bias 要在完整 psum 之后加
不要每个 tile 都加 bias。  
bias 是对完整输出神经元加一次。

---

## 5. ReLU 在 bias 之后
顺序应是：

\[
acc \rightarrow +bias \rightarrow ReLU \rightarrow requant
\]

---

## 6. requant 最好放阵列外
CIM 核只负责高吞吐乘加和部分和。  
requant 是比例缩放，更适合在外围数字逻辑做。

---

# 十三、你应该怎么开始写代码

真正动手时，我建议按这个顺序。

---

## 第一步：先写 `cim_tile.sv`
只做组合逻辑：

- 输入 16 个 int8
- 权重 16x16 int8
- 输出 16 个 int32

这一步最简单。

---

## 第二步：写 `psum_buffer.sv`
实现：

- clear
- accumulate
- read out

---

## 第三步：写 `cim_bank.sv`
给定：
- `out_block_idx`
- `in_block_idx`

输出：
- 当前 `16x16` tile 权重

先直接用 ROM 方式做，不要急着 BRAM 优化。

---

## 第四步：写 `cim_core.sv`
实现两层循环控制：

- 外层输出块
- 内层输入块

---

## 第五步：写 `fc1_postproc.sv`
实现：
- bias add
- ReLU
- requant

---

## 第六步：最后挂到 `mnist_mlp_top.sv`
把：
- `fc1` 做成 CIM
- `fc2` 继续普通 RTL

这时你就能很自然地说：

**本设计实现了数字型 CIM 宏单元，并将其用于 MLP 第一层矩阵向量乘加。**

---

# 十四、你答辩时可以怎么解释“具体思路”

你可以这样说，比较专业：

> 本设计将第一层全连接 `784×128` 权重矩阵切分为多个 `16×16` 的 tile，并在 FPGA 上构建数字型 CIM 宏单元。每个 tile 内部存储一个局部权重子矩阵，输入子向量以广播方式送入 tile，tile 内部并行完成 16 路输出神经元的乘加运算，得到局部部分和；随后对 49 个输入块的部分和进行累加，形成完整输出。阵列外部完成 bias、ReLU 和 requant 操作，第二层继续采用常规数字逻辑实现。

这个说法就已经非常接近指导书要求了。

---

# 十五、你现在最需要的不是更多解释，而是一版真正的 CIM RTL 草图

下一步最合适的是：  
我直接给你写一套 **SystemVerilog 的 `16x16 cim_tile / cim_core / psum_buffer / fc1_cim_wrapper` 代码框架**，让你从“思路”直接进入“能写 RTL”。



用户:
我的想法是这样的：应该是先把weight输入存到本地，然后分tile完成输入，和weight计算，tile内部按位广播进入，计算部分和，然后计算tile的部分和，为一次tiel计算，最后完成tile之间的计算获得最后结果，你觉得我说的对吗
AI助手:


AI助手:
对，你这个思路是对的，而且已经抓住了 **CIM/阵列化实现的主线**。  
但要稍微改得更严谨一点，不然到 RTL 设计时容易把“tile 内部分和”和“tile 间部分和”混在一起。

我先给你一句更准确的话：

> **先将权重按 tile 组织并驻留在本地阵列/本地存储中，再把输入向量按 tile 切块并广播到阵列内部，tile 内完成局部乘加得到局部部分和；随后对同一输出通道在多个输入 tile 上产生的局部部分和进行累加，最后得到完整输出。**

这句话基本就是你想表达的内容，只是把层次分清楚了。

---

# 一、你这个思路为什么是对的

你说的是：

1. 先把 weight 输入存到本地  
2. 分 tile 完成输入和 weight 计算  
3. tile 内部广播  
4. 计算部分和  
5. 再做 tile 的部分和累加  
6. 最后 tile 之间计算得到结果

这本质上就是一个典型的 **矩阵向量乘 tile 化 + 局部部分和累加** 流程。

对第一层全连接：

\[
Y = W X + b
\]

其中：

- `W`：`128 x 784`
- `X`：`784`
- `Y`：`128`

如果不 tile 化，硬件一次做不完。  
所以必须拆成：

- 输入方向切块
- 输出方向切块
- 局部计算
- 部分和累加

这就是 CIM/阵列设计最核心的计算流。

---

# 二、但你这里有两个地方要说得更准确

---

## 1）“先把 weight 输入存到本地”这句话要改一下

更准确的说法不是：

**把 weight 输入存到本地**

而是：

**把权重预先映射/加载到本地阵列存储中，并在计算过程中尽量保持驻留。**

因为在 CIM 里，权重不是“像输入一样每次送进去”。  
更像是：

- 权重先写进阵列
- 计算时输入广播过去
- 阵列直接拿本地权重参与计算

所以关键词应该是：

**权重驻留（weight stationary）**

而不是“weight 输入”。

---

## 2）“最后完成 tile 之间的计算获得最后结果”要进一步拆开

这里最容易混淆。  
tile 之间其实不是“再算一遍”，而是：

### 情况 A：输入方向 tile 之间
这些 tile 的结果要做 **累加**

因为它们对应的是同一个输出神经元在不同输入分块上的贡献。

### 情况 B：输出方向 tile 之间
这些 tile 一般不是彼此累加，而是分别对应不同输出通道。

所以更准确地说：

- **输入方向多个 tile → 部分和累加**
- **输出方向多个 tile → 结果拼接**

这点特别关键。

---

# 三、我给你画成最清楚的形式

假设你设：

- `TIN = 16`
- `TOUT = 16`

那么第一层 `128 x 784` 会变成：

- 输入维度分成 49 个 tile
- 输出维度分成 8 个 tile

---

## 1）输出方向 tile
比如第 0 个输出 tile 负责：

\[
Y[0:15]
\]

第 1 个输出 tile 负责：

\[
Y[16:31]
\]

这些 tile 之间不是相加关系，而是：

**每个 tile 负责不同的输出块。**

---

## 2）输入方向 tile
比如第 0 个输入 tile 负责：

\[
X[0:15]
\]

第 1 个输入 tile 负责：

\[
X[16:31]
\]

对于同一个输出 tile，它们会分别产生局部部分和：

\[
PSUM^{(0)}, PSUM^{(1)}, ..., PSUM^{(48)}
\]

这些要相加：

\[
Y_{tile} = \sum_{k=0}^{48} PSUM^{(k)}
\]

这才是最终这个输出 tile 的结果。

---

# 四、所以你这个流程应该严格写成这样

你现在的思路，建议整理成下面这个版本：

---

## 正确流程

### Step 1：权重预加载
将 `fc1_weight_int8.hex` 中的权重按 tile 方式映射到本地阵列/本地 BRAM 中。  
计算过程中权重尽量保持不动。

### Step 2：输入分块
将 784 维输入向量按 tile 宽度切分，比如每 16 个元素为一个输入 tile。

### Step 3：输入广播
每次取一个输入 tile，广播到当前参与计算的 CIM tile 内部。

### Step 4：tile 内并行乘加
tile 内部使用本地存储的权重子矩阵与广播输入做并行乘加，得到该 tile 对应输出块的局部部分和。

### Step 5：输入方向部分和累加
对同一输出 tile，在所有输入 tile 上得到的局部部分和进行累加，得到完整输出块的 int32 累加结果。

### Step 6：输出方向结果拼接
对所有输出 tile 重复上述过程，最终拼接成完整 128 维输出。

### Step 7：后处理
在阵列外部完成：
- bias add
- ReLU
- requant

### Step 8：送入第二层
将 `fc1_out` 输入到 `fc2`，继续计算最终 logits。

---

# 五、你现在这个想法对应的数据流，其实就是一种标准的数据流模式

这个模式通常叫：

## Weight-stationary dataflow
意思是：

- 权重固定在阵列里
- 输入在阵列间流动/广播
- 输出部分和逐步累加

这非常适合你这个项目。

因为你的 `route_b_output` 已经是：

- 权重独立导出
- 输入独立导出
- golden 可逐层验证

特别适合做这种流派。

---

# 六、你说“tile 内部按位广播进入”，这里要注意措辞

这里“按位广播”容易被理解成：

- bit-serial
- 逐 bit 输入
- 位线级广播

如果你真正想表达的是：

- 把一个 tile 的 16 个 int8 输入同时送到 tile 内部各列/各行

那更准确的说法应该是：

### 如果是普通并行 int8 输入
说：
**按元素广播**
或者
**按向量广播**

### 如果你真想做 bit-serial CIM
那才说：
**按位广播**

---

## 两者区别很大

### 按元素广播
例如一次送入：

```text
x0, x1, ..., x15
```

每个都是 int8，tile 内直接做 int8×int8。

### 按位广播
例如先送所有输入的 bit0，再送 bit1，再送 bit2...  
这是 bit-serial 乘法结构，复杂度会明显上升。

---

## 对你当前项目，我建议先不要做 bit-serial
先做：

**按元素广播的数字型 CIM**

也就是：

- 每次送一个 `TIN` 维 int8 子向量
- tile 内直接并行乘加

这样更容易和 `route_b_output` 对齐，也更容易实现。

所以你那句话建议改成：

> tile 内部按输入子向量广播进入

而不是“按位广播进入”。

---

# 七、你这个思路还差最后一个最关键的点：tile 的定义必须固定

你现在的想法是对的，但在真正进入 RTL 前，必须把下面 3 个参数定下来：

## 1）输入 tile 宽度 `TIN`
比如：
- 8
- 16
- 32

## 2）输出 tile 宽度 `TOUT`
比如：
- 8
- 16
- 32

## 3）权重在本地怎么组织
例如：
- 一个 tile = 一个 `TOUT x TIN` 权重块
- 多个 tile 存在 bank 中
- 地址按 `[out][in]` 取

没有这三个固定值，就还只是概念。

---

# 八、我帮你把你的话改写成适合写进报告/答辩的话

你现在这个想法，可以这样表述：

> 本设计采用 weight-stationary 的数字型 CIM 数据流。首先将第一层全连接权重按 tile 方式预加载到本地存储阵列中；计算时将输入向量按固定 tile 宽度切分，并逐块广播到阵列内部。每个 tile 利用本地权重子矩阵与输入子向量并行完成乘加运算，产生对应输出块的局部部分和；随后对同一输出块在不同输入 tile 上产生的局部部分和进行累加，最终得到完整的输出结果。之后在阵列外部完成 bias、ReLU 和 requant 操作。

这段话是专业的，而且是对的。

---

# 九、最后给你一个最简判断

## 你的想法对不对？
**对。**

## 哪些地方要修正？
主要两点：

1. “weight 输入存到本地”  
改成  
**权重预加载并驻留在本地**

2. “按位广播”  
若你不是做 bit-serial，改成  
**按输入子向量广播**

3. “tile 之间计算获得最后结果”  
更准确地说  
**输入 tile 间做部分和累加，输出 tile 间做结果拼接**

---

# 十、你下一步应该做什么

你现在已经不缺思路了，缺的是把这个思路变成明确的模块划分。  
下一步最合适的是先把这 4 个东西定死：

- `TIN = ?`
- `TOUT = ?`
- `fc1` 是否完全由 CIM 实现
- tile 的地址公式和 bank 组织方式

然后就可以开始写：

- `cim_tile.sv`
- `psum_buffer.sv`
- `cim_core.sv`

我可以下一条直接帮你把你这个想法整理成一版**可落地的硬件总体设计方案**，包括模块图、数据流和 SystemVerilog 模块划分。

用户:
关于：
Step 1：权重预加载

将 fc1_weight_int8.hex 中的权重按 tile 方式映射到本地阵列/本地 BRAM 中。
计算过程中权重尽量保持不动。

Step 2：输入分块

将 784 维输入向量按 tile 宽度切分，比如每 16 个元素为一个输入 tile。

Step 3：输入广播

每次取一个输入 tile，广播到当前参与计算的 CIM tile 内部。

Step 4：tile 内并行乘加

tile 内部使用本地存储的权重子矩阵与广播输入做并行乘加，得到该 tile 对应输出块的局部部分和。

Step 5：输入方向部分和累加

对同一输出 tile，在所有输入 tile 上得到的局部部分和进行累加，得到完整输出块的 int32 累加结果。

Step 6：输出方向结果拼接

对所有输出 tile 重复上述过程，最终拼接成完整 128 维输出。

Step 7：后处理

在阵列外部完成：

bias add

ReLU

requant

Step 8：送入第二层

将 fc1_out 输入到 fc2，继续计算最终 logits。

请给我具体执行步骤和代码编写流程，同时请你帮我完成tile的定义，

1）输入 tile 宽度 TIN

比如：

8

16

32

2）输出 tile 宽度 TOUT

比如：

8

16

32

3）权重在本地怎么组织

例如：

一个 tile = 一个 TOUT x TIN 权重块

多个 tile 存在 bank 中

地址按 [out][in] 取

没有这三个固定值，就还只是概念。
AI助手:


AI助手:
下面我直接把这件事定成一套**可落地的阶段 1 CIM 设计方案**，包括：

1. **具体执行步骤**
2. **代码编写流程**
3. **tile 参数最终定义**
4. **权重本地组织方式**
5. **建议的 SystemVerilog 模块划分**
6. **每一步先做什么、后做什么**

我先给结论，再展开。

---

# 先定死这 3 个关键参数

结合你的网络规模、`route_b_output` 的格式、以及 FPGA 上实现难度，我建议第一版这样定：

## tile 定义
- **输入 tile 宽度 `TIN = 16`**
- **输出 tile 宽度 `TOUT = 16`**

## 原因
你的第一层是：

- 输入维度：784
- 输出维度：128

那么：

- 输入方向分块数：
  \[
  N_{IB} = 784 / 16 = 49
  \]

- 输出方向分块数：
  \[
  N_{OB} = 128 / 16 = 8
  \]

这意味着：

- 每个输出块 16 个输出神经元
- 每个输入块 16 个输入元素
- 每个 tile 是一个 `16 x 16` 的权重子矩阵
- 一共需要访问：
  \[
  8 \times 49 = 392
  \]
  个 tile

这个规模很适合：
- 写清楚阵列概念
- 资源不至于太夸张
- 容易和报告、答辩说明统一

---

# 权重在本地怎么组织

这个也直接定死。

## 组织方式
### 1）一个 tile
一个 tile 定义为：

\[
W_{tile} \in \mathbb{Z}^{16 \times 16}
\]

也就是：

- 16 行，对应 16 个输出通道
- 16 列，对应 16 个输入元素

### 2）多个 tile 的组织
整个 `fc1_weight_int8.hex` 逻辑上按 `[out][in]` 排布。  
所以我们定义：

- 输出块索引：`ob = 0..7`
- 输入块索引：`ib = 0..48`

则某个 tile 内部元素：

- tile 内输出索引：`tr = 0..15`
- tile 内输入索引：`tc = 0..15`

对应全局坐标：

\[
out = ob \times 16 + tr
\]

\[
in = ib \times 16 + tc
\]

在 `fc1_weight_int8.hex` 中的线性地址：

\[
addr = out \times 784 + in
\]

即：

\[
addr = (ob \times 16 + tr)\times 784 + (ib \times 16 + tc)
\]

---

## 本地存储建议
第一版先不要做太复杂 bank。  
先这样组织最稳：

### 方案 A：逻辑 bank
定义 8 个输出 bank：

- `bank0` 存输出 0~15 的全部权重
- `bank1` 存输出 16~31 的全部权重
- ...
- `bank7` 存输出 112~127 的全部权重

每个 bank 的容量：

\[
16 \times 784 = 12544 \text{ 个 int8}
\]

这样每个 bank 对应一个输出块，很直观。

### bank 内地址
对于某个 bank（即某个 `ob`），地址定义：

\[
local\_addr = tr \times 784 + in
\]

或者按 tile 读取时：

\[
local\_addr = tr \times 784 + ib \times 16 + tc
\]

这样非常适合硬件控制。

---

# 具体执行步骤

下面按你那 8 步，改成真正能写代码的执行步骤。

---

## Step 0：先冻结接口和参数
先把以下常量写进 `package`：

```systemverilog
parameter int IN_DIM   = 784;
parameter int H_DIM    = 128;
parameter int OUT_DIM  = 10;

parameter int TIN      = 16;
parameter int TOUT     = 16;

parameter int N_IB     = 49;   // 784 / 16
parameter int N_OB     = 8;    // 128 / 16
```

还要把 `route_b_output/quant_config.json` 的参数固化进去：

- `input_zero_point = -128`
- `fc1_weight_zero_point = 0`
- `fc1_output_zero_point = 0`
- `fc1 multiplier/shift`
- `fc2 multiplier/shift`

这一步做完，整个设计才不再是口头概念。

---

## Step 1：权重预加载到本地阵列 / BRAM

### 目标
把 `fc1_weight_int8.hex` 中的 100352 个 int8 权重读进本地权重存储。

### 第一版建议做法
在仿真/初版综合里先用：

- `$readmemh`
- ROM/BRAM 风格数组

定义 8 个输出 bank，每个 bank 存 16×784 个 int8。

### 你要写的模块
`fc1_weight_bank.sv`

### 这个模块要做的事
输入：
- `ob`：当前输出块索引 `0..7`
- `ib`：当前输入块索引 `0..48`

输出：
- 当前 tile 的 `16x16` 权重矩阵

也就是：

```systemverilog
output logic signed [7:0] w_tile [0:TOUT-1][0:TIN-1];
```

### 实现策略
内部保留一个大 ROM：

```systemverilog
logic signed [7:0] w_mem [0:H_DIM-1][0:IN_DIM-1];
```

仿真阶段直接：

```systemverilog
$readmemh("fc1_weight_int8.hex", flat_mem);
```

然后映射到二维，或者直接一维寻址。

---

## Step 2：输入分块

### 目标
把 784 维输入向量切成 49 个 `16` 维输入 tile。

### 输入格式
`input_i.hex` 是长度 784 的 int8 向量。

### 你要写的模块
`input_buffer.sv`

### 这个模块做什么
- 保存完整 784 维输入
- 根据当前 `ib` 输出：

```systemverilog
output logic signed [7:0] x_tile [0:TIN-1];
```

### 地址关系
对于 tile 内列索引 `tc = 0..15`：

\[
x\_tile[tc] = input\_buf[ib \times 16 + tc]
\]

### 注意
你当前输入 zero-point 是 `-128`。  
为了和 Python 对齐，建议在进入乘加阵列前就做：

\[
x_{eff} = x_q - z_x = x_q + 128
\]

所以 `input_buffer` 最好再输出：

```systemverilog
logic signed [8:0] x_eff_tile [0:TIN-1];
```

因为 `x_q` 是 int8，`x_q + 128` 后范围是 `0..255`，要 9 bit。

---

## Step 3：输入广播

### 目标
把一个 16 维输入 tile 广播给 tile 内所有输出行。

### 实现方式
这一步不一定要单独写模块。  
可以在 `cim_tile.sv` 里直接用 `x_tile`。

“广播”的意思不是额外复制很多硬件，而是：

- 同一个 `x_tile[tc]`
- 被所有 16 行乘法器同时使用

### 在 RTL 上体现为
在 `generate` 或双重循环中：

- 每一行都访问同一个 `x_tile[tc]`

---

## Step 4：tile 内并行乘加

这一步是整个 CIM 的核心。

### 你要写的模块
`cim_tile.sv`

### 接口
```systemverilog
module cim_tile #(
    parameter int TIN  = 16,
    parameter int TOUT = 16
)(
    input  logic signed [8:0] x_eff_tile [0:TIN-1],         // 0..255 after zp correction
    input  logic signed [7:0] w_tile     [0:TOUT-1][0:TIN-1],
    output logic signed [31:0] tile_psum [0:TOUT-1]
);
```

### 计算内容
对每个输出行 `tr`：

\[
tile\_psum[tr] = \sum_{tc=0}^{15} x\_eff[tc] \cdot w[tr][tc]
\]

### 为什么这一步像 CIM
因为：
- 权重子矩阵驻留在本地
- 输入子向量广播
- 16 行并行产出 16 个部分和

这就是一个数字型 `16x16` CIM tile。

### 第一版实现
先做组合逻辑最简单：

```systemverilog
always_comb begin
    for (tr = 0; tr < TOUT; tr++) begin
        tile_psum[tr] = 0;
        for (tc = 0; tc < TIN; tc++) begin
            tile_psum[tr] += x_eff_tile[tc] * w_tile[tr][tc];
        end
    end
end
```

后面如果时序差，再改流水。

---

## Step 5：输入方向部分和累加

### 目标
同一个输出块 `ob`，49 个输入块都会产生一个 `16` 维 `tile_psum`，这些要累计。

### 你要写的模块
`psum_accum.sv`

### 接口
```systemverilog
module psum_accum #(
    parameter int TOUT = 16
)(
    input  logic clk,
    input  logic rst_n,
    input  logic clear,
    input  logic en,
    input  logic signed [31:0] tile_psum [0:TOUT-1],
    output logic signed [31:0] psum      [0:TOUT-1]
);
```

### 功能
- `clear=1`：清零 16 个部分和寄存器
- `en=1`：`psum += tile_psum`

### 什么时候清零
每开始一个新的输出块 `ob` 时清零。

### 什么时候累加
每完成一个输入块 `ib` 时累加一次。

### 最终结果
当 `ib = 0..48` 全部完成后：

```text
psum[0:15]
```

就是当前输出块 16 个神经元的完整 int32 累加值。

---

## Step 6：输出方向结果拼接

### 目标
把 8 个输出块的结果写回到完整的 `fc1_acc[0:127]`

### 你要写的模块
这个可以在 `fc1_cim_core.sv` 里完成，不必单独模块化。

### 写回方式
对于当前输出块 `ob`，块内索引 `tr`：

\[
fc1\_acc[ob\times 16 + tr] = psum[tr]
\]

### 结果
当 `ob = 0..7` 全部完成后，得到：

- 完整 128 维 `fc1_acc_int32`

它应当与 `route_b_output/fc1_acc_*.hex` 对齐。

---

## Step 7：阵列外后处理

这一步不要放在 tile 里，放在外围更清晰。

### 先做 bias add
你有：

- `fc1_bias_int32.hex`

对于输出块内每个元素：

\[
acc\_bias[tr] = psum[tr] + bias[ob\times16+tr]
\]

### 再做 ReLU
\[
relu[tr] = \max(acc\_bias[tr], 0)
\]

### 再做 requant
把 int32 缩放到 int8：

\[
fc1\_out_q = round(relu \cdot multiplier / 2^{shift}) + z_{out}
\]

你已经在 `route_b_output/quant_config.json` 里有：

- `multiplier_q31`
- `shift`
- `output_zero_point`

### 你要写的模块
- `fc1_bias_rom.sv`
- `relu_vec.sv`
- `requant_vec.sv`

或者统一成：
- `fc1_postproc.sv`

---

## Step 8：送入第二层

### 第一版建议
`fc2` 不做 CIM。  
直接沿用普通顺序 FC 或小并行 FC。

因为：
- `fc2` 规模只有 `10 x 128`
- 用来证明完整 MNIST 分类即可
- 把复杂度集中在 `fc1` 的 CIM 上，更符合项目重点

### 你要写的模块
`fc2_seq.sv`

输入：
- `fc1_out[0:127]` int8

权重：
- `fc2_weight_int8.hex`

偏置：
- `fc2_bias_int32.hex`

输出：
- `fc2_acc[0:9]` int32
- `logits[0:9]` int8

再接：
- `argmax`

---

# 代码编写流程

下面是我建议的**实际写代码顺序**。  
这是最重要的部分，因为很多人思路是对的，但代码顺序一乱就写崩。

---

## 第 1 阶段：先搭 package 和参数

先写：

### `mnist_cim_pkg.sv`
内容包括：
- `IN_DIM = 784`
- `H_DIM = 128`
- `OUT_DIM = 10`
- `TIN = 16`
- `TOUT = 16`
- `N_IB = 49`
- `N_OB = 8`
- 各层 requant 参数
- zero-point 常量

这一步必须先做。

---

## 第 2 阶段：先写不带控制器的小模块

顺序如下：

### 1）`input_buffer.sv`
输出 `x_tile`

### 2）`fc1_weight_bank.sv`
输出 `w_tile`

### 3）`cim_tile.sv`
组合并行乘加

### 4）`psum_accum.sv`
部分和累加

### 5）`fc1_bias_rom.sv`
读 int32 bias

### 6）`fc1_postproc.sv`
做 bias + relu + requant

这些模块都是“积木”。

---

## 第 3 阶段：写 `fc1_cim_core.sv`

这是第一个真正的大模块。

### 它内部要有两个循环控制
- `ob`：输出块计数器 `0..7`
- `ib`：输入块计数器 `0..48`

### 工作流程
对于每个 `ob`：

1. 清空 `psum`
2. `ib = 0..48`
3. 读 `x_tile`
4. 读 `w_tile`
5. `cim_tile` 得到 `tile_psum`
6. `psum_accum += tile_psum`
7. 所有 `ib` 完成后，进入后处理
8. 生成 `fc1_out[ob*16 : ob*16+15]`
9. 转下一个 `ob`

### 这个模块要输出
- `fc1_acc[0:127]`
- `fc1_relu[0:127]`
- `fc1_out[0:127]`
- `done`

---

## 第 4 阶段：先做 fc1 单层对拍

先不要接 `fc2`。  
只做：

- 载入 `input_0.hex`
- 跑 `fc1_cim_core`
- 对拍：
  - `fc1_acc_0.hex`
  - `fc1_relu_0.hex`
  - `fc1_out_0.hex`

### 这是关键
这一关过了，说明你的 CIM 主体是通的。

---

## 第 5 阶段：再写 `fc2_seq.sv`

`fc2` 不需要 tile 化，先写顺序或小并行即可。

### 流程
- 输入：`fc1_out[128]`
- 权重：`10x128`
- 累加：10 个 int32
- requant：得到 `logits[10]`
- argmax：得到 `pred`

---

## 第 6 阶段：再写 top

### `mnist_mlp_cim_top.sv`
连接：
- `fc1_cim_core`
- `fc2_seq`
- `argmax`

输入：
- 784 维 int8 向量
- start

输出：
- done
- pred
- logits

---

# tile 的最终定义

下面我直接给你这版设计的最终定义，可直接写进报告。

---

## 1）输入 tile 宽度 `TIN`
**取 `TIN = 16`**

### 含义
每次从输入向量中取 16 个元素组成一个输入 tile：

\[
X_{ib} = [x_{ib\cdot16}, ..., x_{ib\cdot16+15}]
\]

### 输入块总数
\[
N_{IB} = 784 / 16 = 49
\]

---

## 2）输出 tile 宽度 `TOUT`
**取 `TOUT = 16`**

### 含义
每个输出 tile 负责同时计算 16 个输出神经元：

\[
Y_{ob} = [y_{ob\cdot16}, ..., y_{ob\cdot16+15}]
\]

### 输出块总数
\[
N_{OB} = 128 / 16 = 8
\]

---

## 3）权重在本地怎么组织

### tile 定义
一个 tile 是一个：

\[
16 \times 16
\]

的权重子矩阵。

### tile 索引
- `ob = 0..7`
- `ib = 0..48`

### tile 内索引
- `tr = 0..15`
- `tc = 0..15`

### 全局映射
\[
out = ob \times 16 + tr
\]

\[
in = ib \times 16 + tc
\]

### 在 `fc1_weight_int8.hex` 中地址
\[
addr = out \times 784 + in
\]

### 本地存储建议
采用 8 个输出 bank，每个 bank 保存 16 个输出通道的全部权重：

- `bank[ob][tr][in]`

访问时：

```text
w_tile[tr][tc] = bank[ob][tr][ib*16 + tc]
```

这非常适合硬件实现。

---

# 给你一版完整的模块划分建议

---

## package
- `mnist_cim_pkg.sv`

## 输入相关
- `input_buffer.sv`

## 权重相关
- `fc1_weight_bank.sv`
- `fc1_bias_rom.sv`
- `fc2_weight_rom.sv`
- `fc2_bias_rom.sv`

## CIM 相关
- `cim_tile.sv`
- `psum_accum.sv`
- `fc1_cim_core.sv`

## 后处理
- `relu_vec.sv`
- `requant_vec.sv`
- `fc1_postproc.sv`

## 第二层
- `fc2_seq.sv`
- `argmax.sv`

## 顶层
- `mnist_mlp_cim_top.sv`

---

# 硬件设计上要特别注意的 6 件事

---

## 1）输入 zero-point 修正必须放清楚
你当前输入是 int8，zero-point = -128。  
因此阵列内真正参与乘法的应该是：

\[
x_{eff} = x_q + 128
\]

建议：
- 在 `input_buffer` 或 `fc1_cim_core` 里完成
- 不要一会儿在 tile 内修，一会儿在外围修

---

## 2）tile_psum 至少 int32
虽然单个 `16x16` tile 的一拍部分和未必会爆，但总共 49 次累加，所以 `psum` 必须 int32。

---

## 3）bias 只在完整输出块完成后加一次
不要每个 `ib` 都加 bias。  
bias 是对最终输出神经元加一次。

---

## 4）ReLU 和 requant 一定放在 psum 完整后
顺序是：

\[
tile\_psum \rightarrow psum \rightarrow +bias \rightarrow ReLU \rightarrow requant
\]

---

## 5）`fc1_acc` 对拍优先级最高
你第一阶段先别盯最终 `pred`。  
优先保证：

- `fc1_acc` 完全对上 `fc1_acc_0.hex`

这是最能说明 CIM 主体正确的指标。

---

## 6）第一版先别上 bit-serial
你前面提到“按位广播”，但当前项目第一版不建议做 bit-serial。  
先做：
- int8 输入 tile 广播
- int8 权重本地存储
- 并行乘加

这样最稳。

---

# 你接下来真正该怎么做

最合理的顺序是：

### 先做参数冻结
把 `TIN=16, TOUT=16` 固化进 package

### 再做四个基础模块
- `input_buffer.sv`
- `fc1_weight_bank.sv`
- `cim_tile.sv`
- `psum_accum.sv`

### 然后做 `fc1_cim_core.sv`
先把第一层跑通

### 再对拍 `fc1_acc / fc1_relu / fc1_out`
只要这一层过了，你的 CIM 设计就站住了

### 最后再接 `fc2_seq.sv`

---

你现在已经不缺思路了，下一步最有价值的是我直接继续给你输出**这套 `TIN=16, TOUT=16` 的 SystemVerilog 代码框架**，从 `package + cim_tile + psum_accum + fc1_weight_bank + fc1_cim_core` 开始。

用户:
给我一个预期文件组织架构
AI助手:


AI助手:
下面给你一套**推荐的预期文件组织架构**，按你现在这个项目的推进方式来设计：既能支撑 `route_b_output` 的测试向量验证，也能继续扩展成真正的 **FPGA + CIM + MNIST** 工程。

我按“**工程根目录 → 训练与数据 → RTL → 仿真 → 文档**”来组织。

---

# 1. 总体目录结构

```text
fpga_cim_mnist_project/
├── README.md
├── docs/
├── data/
├── model/
├── scripts/
├── route_a_output/
├── route_b_output/
├── rtl/
├── sim/
├── tb/
├── constraints/
├── build/
└── report/
```

---

# 2. 各目录详细说明

---

## `README.md`
工程总说明，建议写：

- 项目目标
- 网络结构
- 量化方式
- route_a / route_b 区别
- RTL 模块关系
- 仿真运行方式
- 上板流程

---

## `docs/`
放项目过程文档和设计说明。

建议里面放：

```text
docs/
├── project_guideline.pdf / docx
├── ai_discussion_notes.md
├── route_b_output_usage_guide.md
├── cim_architecture.md
├── quantization_design.md
└── test_plan.md
```

### 建议说明
- `project_guideline.*`：项目指导书
- `ai_discussion_notes.md`：和 AI 讨论形成的设计思路
- `route_b_output_usage_guide.md`：你前面已经整理过的使用指南
- `cim_architecture.md`：后面写 CIM 宏单元架构说明
- `quantization_design.md`：说明输入、权重、bias、requant 的量化规则
- `test_plan.md`：写验证流程

---

## `data/`
原始数据和中间数据。

```text
data/
├── mnist/
└── calib/
```

### 用途
- `mnist/`：MNIST 原始数据
- `calib/`：量化校准时用的小批样本或统计数据

---

## `model/`
放 Python 训练模型、导出模型、量化模型。

```text
model/
├── checkpoints/
├── float_model/
└── quantized_model/
```

细化可写成：

```text
model/
├── checkpoints/
│   ├── mlp_route_a.pth
│   └── mlp_route_b.pth
├── float_model/
│   ├── model_structure.txt
│   └── eval_results.txt
└── quantized_model/
    ├── quant_config.json
    ├── fc1_params.json
    └── fc2_params.json
```

---

## `scripts/`
放 Python 脚本和 notebook。

```text
scripts/
├── notebooks/
├── train/
├── export/
└── utils/
```

细化建议：

```text
scripts/
├── notebooks/
│   ├── train_simple_mlp.ipynb
│   └── export_test_vectors.ipynb
├── train/
│   ├── train_route_a.py
│   └── train_route_b.py
├── export/
│   ├── export_route_a.py
│   ├── export_route_b.py
│   └── pack_weights_for_cim.py
└── utils/
    ├── quant_utils.py
    ├── int_inference_ref.py
    └── check_hex_format.py
```

### 说明
- `train_route_b.py`：训练最终两层 MLP
- `export_route_b.py`：导出 `route_b_output`
- `pack_weights_for_cim.py`：后面把权重按 tile/bank 方式重新打包
- `int_inference_ref.py`：整数推理 golden 参考实现
- `check_hex_format.py`：检查 hex 文件位宽、行数、符号格式

---

## `route_a_output/`
保留单层验证数据。

```text
route_a_output/
├── fc1_weight_int8.hex
├── fc1_bias_int32.hex
├── input_0.hex
├── fc1_acc_golden_0.hex
├── fc1_relu_golden_0.hex
├── labels.txt
└── quant_config.json
```

这个目录主要用于：
- 单层 FC1 验证
- 前期调试
- 快速检查数据流是否正确

---

## `route_b_output/`
这是你当前最重要的目录，用于完整网络验证。

推荐最终保持成这样：

```text
route_b_output/
├── quant_config.json
├── manifest.json
├── fc1_weight_int8.hex
├── fc1_bias_int32.hex
├── fc2_weight_int8.hex
├── fc2_bias_int32.hex
├── labels.txt
├── preds.txt
├── input_0.hex
├── fc1_acc_0.hex
├── fc1_relu_0.hex
├── fc1_out_0.hex
├── fc2_acc_0.hex
├── logits_0.hex
├── pred_0.txt
├── input_1.hex
├── fc1_acc_1.hex
├── fc1_relu_1.hex
├── fc1_out_1.hex
├── fc2_acc_1.hex
├── logits_1.hex
├── pred_1.txt
└── ...
```

### 我建议额外加一个
`manifest.json`

例如：

```json
{
  "fc1_weight_int8.hex": {
    "dtype": "int8",
    "shape": [128, 784],
    "layout": "[out][in]"
  },
  "fc1_bias_int32.hex": {
    "dtype": "int32",
    "shape": [128]
  }
}
```

这样后面做仿真和上板更不容易乱。

---

## `rtl/`
这是最关键的硬件代码目录。

建议再分层：

```text
rtl/
├── pkg/
├── common/
├── mem/
├── cim/
├── postproc/
├── fc2/
└── top/
```

细化建议如下：

```text
rtl/
├── pkg/
│   └── mnist_cim_pkg.sv
├── common/
│   ├── relu_vec.sv
│   ├── requant_vec.sv
│   ├── argmax.sv
│   └── counter_fsm.sv
├── mem/
│   ├── input_buffer.sv
│   ├── fc1_weight_bank.sv
│   ├── fc1_bias_rom.sv
│   ├── fc2_weight_rom.sv
│   └── fc2_bias_rom.sv
├── cim/
│   ├── cim_tile.sv
│   ├── psum_accum.sv
│   ├── fc1_cim_core.sv
│   └── fc1_cim_wrapper.sv
├── postproc/
│   └── fc1_postproc.sv
├── fc2/
│   └── fc2_seq.sv
└── top/
    └── mnist_mlp_cim_top.sv
```

---

## `sim/`
仿真工程文件和运行脚本。

```text
sim/
├── filelist.f
├── run_vcs.sh
├── run_questa.do
├── run_xsim.tcl
└── waves/
```

### 说明
- `filelist.f`：RTL 和 testbench 文件列表
- `run_vcs.sh`：VCS 一键编译运行
- `run_questa.do`：ModelSim/Questa 的仿真脚本
- `run_xsim.tcl`：Vivado xsim 脚本
- `waves/`：波形配置

---

## `tb/`
专门放 testbench。

建议分层验证：

```text
tb/
├── tb_fc1_cim_core.sv
├── tb_fc1_postproc.sv
├── tb_fc2_seq.sv
├── tb_mnist_mlp_cim_top.sv
└── tb_utils.sv
```

### 说明
- `tb_fc1_cim_core.sv`：只对拍 `fc1_acc`
- `tb_fc1_postproc.sv`：对拍 `fc1_relu` 和 `fc1_out`
- `tb_fc2_seq.sv`：对拍 `fc2_acc` 和 `logits`
- `tb_mnist_mlp_cim_top.sv`：最终整网对拍
- `tb_utils.sv`：通用 hex 读取、比较任务

---

## `constraints/`
上板约束文件。

```text
constraints/
└── top.xdc
```

或如果不是 Xilinx，就放对应厂商约束。

---

## `build/`
综合、实现、bitstream 输出目录。

```text
build/
├── synth/
├── impl/
└── bit/
```

通常不需要手写很多内容，工具生成即可。

---

## `report/`
放课程报告和答辩材料。

```text
report/
├── figures/
├── tables/
├── draft/
└── final/
```

细化：

```text
report/
├── figures/
│   ├── cim_array_architecture.png
│   ├── dataflow.png
│   └── verification_flow.png
├── draft/
│   └── project_report_v1.docx
└── final/
    ├── project_report_final.docx
    └── slides_final.pptx
```

---

# 3. 如果按“当前阶段”来组织，最推荐的精简架构

如果你不想一开始目录太多，我建议你现在就先用下面这个版本，最实用：

```text
fpga_cim_mnist_project/
├── README.md
├── docs/
│   ├── project_guideline.docx
│   ├── ai_discussion_notes.md
│   └── route_b_output_usage_guide.md
├── scripts/
│   ├── train_route_b.py
│   ├── export_route_b.py
│   ├── quant_utils.py
│   └── int_inference_ref.py
├── route_b_output/
│   ├── quant_config.json
│   ├── fc1_weight_int8.hex
│   ├── fc1_bias_int32.hex
│   ├── fc2_weight_int8.hex
│   ├── fc2_bias_int32.hex
│   ├── input_0.hex
│   ├── fc1_acc_0.hex
│   ├── fc1_relu_0.hex
│   ├── fc1_out_0.hex
│   ├── fc2_acc_0.hex
│   ├── logits_0.hex
│   └── ...
├── rtl/
│   ├── mnist_cim_pkg.sv
│   ├── input_buffer.sv
│   ├── fc1_weight_bank.sv
│   ├── cim_tile.sv
│   ├── psum_accum.sv
│   ├── fc1_cim_core.sv
│   ├── fc1_postproc.sv
│   ├── fc2_seq.sv
│   ├── argmax.sv
│   └── mnist_mlp_cim_top.sv
├── tb/
│   ├── tb_fc1_cim_core.sv
│   └── tb_mnist_mlp_cim_top.sv
└── sim/
    ├── filelist.f
    └── run_xsim.tcl
```

这个版本已经足够你推进到：
- 阶段 1 RTL
- 单层对拍
- 完整网络对拍

---

# 4. 我建议你现在新增的几个关键文件

你当前最值得补的几个文件是：

### `rtl/mnist_cim_pkg.sv`
统一管理：
- `TIN=16`
- `TOUT=16`
- `N_IB=49`
- `N_OB=8`
- requant 参数
- 零点参数

### `route_b_output/manifest.json`
记录每个文件的：
- dtype
- shape
- layout
- meaning

### `docs/cim_architecture.md`
写清楚：
- 阵列定义
- tile 定义
- 输入广播
- psum 累加
- 后处理流程

这三个文件补上后，工程会立刻清晰很多。

---

# 5. 最后给你一句最实用的建议

你当前这个项目，最怕的不是代码不够多，而是：

**训练脚本、导出数据、RTL、testbench 四套东西不对齐。**

所以目录结构一定要体现这条链：

```text
scripts  →  route_b_output  →  rtl  →  tb  →  sim
```

只要这条链清晰，你的项目就很稳。

如果你愿意，我下一条可以继续直接给你一版**更贴合你当前 CIM 方案的实际文件树**，也就是把 `rtl/` 下面每个 SystemVerilog 文件的职责再细化成“文件名 + 模块功能 + 输入输出”。

用户:
parameter int input_dimension = 784;
parameter int hiden_dimension = 128;
parameter int output_dimension = 10;

parameter int tile_input_size = 16;
parameter int tile_output_size = 16;

parameter int n_input_blocks = 49;  // 784 / 16
parameter int n_output_blocks = 8;  // 128 / 6

parameter int input_zero_point = -128;
parameter int fc1_weight_zero_point = 0;
parameter int fc1_output_zero_point = 0;我的package.sv这样写可以吗
AI助手:


AI助手:
可以，但有几处必须改，不然后面 RTL 很容易出错。

## 先说结论
你的 `package.sv` 写法思路是对的，参数也基本合理。  
**但当前这版至少有 4 个问题：**

1. `n_output_blocks = 8;  // 128 / 6`  
   注释错了，应该是 `128 / 16`
2. `hiden_dimension` 拼写错了  
   应改为 `hidden_dimension`
3. `input_zero_point = -128` 这类参数最好声明成 **signed**
4. 仅有这些参数还不够，后面最好把位宽和一些派生参数也一起放进去

---

# 你这版修改后的推荐写法

建议直接改成下面这样：

```systemverilog
package mnist_cim_pkg;

    // -----------------------------
    // Network dimensions
    // -----------------------------
    parameter int INPUT_DIMENSION  = 784;
    parameter int HIDDEN_DIMENSION = 128;
    parameter int OUTPUT_DIMENSION = 10;

    // -----------------------------
    // Tile configuration
    // -----------------------------
    parameter int TILE_INPUT_SIZE  = 16;
    parameter int TILE_OUTPUT_SIZE = 16;

    // -----------------------------
    // Block numbers
    // -----------------------------
    parameter int N_INPUT_BLOCKS  = INPUT_DIMENSION / TILE_INPUT_SIZE;   // 784 / 16 = 49
    parameter int N_OUTPUT_BLOCKS = HIDDEN_DIMENSION / TILE_OUTPUT_SIZE; // 128 / 16 = 8

    // -----------------------------
    // Quantization zero-points
    // -----------------------------
    parameter int signed INPUT_ZERO_POINT      = -128;
    parameter int signed FC1_WEIGHT_ZERO_POINT = 0;
    parameter int signed FC1_OUTPUT_ZERO_POINT = 0;

endpackage
```

---

# 为什么要这样改

## 1. 注释必须改
你现在写的是：

```systemverilog
parameter int n_output_blocks = 8;  // 128 / 6
```

这里显然是笔误。  
应该是：

```systemverilog
parameter int N_OUTPUT_BLOCKS = HIDDEN_DIMENSION / TILE_OUTPUT_SIZE; // 128 / 16 = 8
```

虽然数值写对了，但注释错会害你后面检查 tile 逻辑时怀疑人生。

---

## 2. `hidden` 拼写要改
你现在是：

```systemverilog
parameter int hiden_dimension = 128;
```

建议改成：

```systemverilog
parameter int HIDDEN_DIMENSION = 128;
```

原因很简单：
- 少一个 `d`
- 后面文件一多，拼写不统一会很烦
- 报告和代码最好统一 terminology

---

## 3. zero-point 建议写成 `signed`
虽然很多工具对 `parameter int = -128` 也能正常处理，  
但你这里是明确有符号语义，最好写清楚：

```systemverilog
parameter int signed INPUT_ZERO_POINT = -128;
```

这样你后面写：

```systemverilog
x_eff = x_q - INPUT_ZERO_POINT;
```

语义更清晰，也更不容易在拼接、类型转换时出问题。

---

## 4. 参数最好不要手填“49、8”，尽量由表达式推出
你现在写死：

```systemverilog
parameter int n_input_blocks = 49;
parameter int n_output_blocks = 8;
```

虽然现在没错，但后面如果你改 `tile_input_size` 或 `tile_output_size`，这两个值会忘改。

所以更推荐：

```systemverilog
parameter int N_INPUT_BLOCKS  = INPUT_DIMENSION / TILE_INPUT_SIZE;
parameter int N_OUTPUT_BLOCKS = HIDDEN_DIMENSION / TILE_OUTPUT_SIZE;
```

这样 package 更稳。

---

# 还建议你继续补的参数

你这版能用，但如果你马上要写 `cim_tile.sv / fc1_cim_core.sv`，我建议再补下面这些，后面会方便很多。

## 推荐补充版

```systemverilog
package mnist_cim_pkg;

    // -----------------------------
    // Network dimensions
    // -----------------------------
    parameter int INPUT_DIMENSION  = 784;
    parameter int HIDDEN_DIMENSION = 128;
    parameter int OUTPUT_DIMENSION = 10;

    // -----------------------------
    // Tile configuration
    // -----------------------------
    parameter int TILE_INPUT_SIZE  = 16;
    parameter int TILE_OUTPUT_SIZE = 16;

    // -----------------------------
    // Block numbers
    // -----------------------------
    parameter int N_INPUT_BLOCKS  = INPUT_DIMENSION / TILE_INPUT_SIZE;   // 49
    parameter int N_OUTPUT_BLOCKS = HIDDEN_DIMENSION / TILE_OUTPUT_SIZE; // 8

    // -----------------------------
    // Data widths
    // -----------------------------
    parameter int INPUT_WIDTH   = 8;
    parameter int WEIGHT_WIDTH  = 8;
    parameter int BIAS_WIDTH    = 32;
    parameter int PSUM_WIDTH    = 32;
    parameter int OUTPUT_WIDTH  = 8;

    // 因为 x_q - zp 后范围可到 0~255，建议预留 9 bit
    parameter int X_EFF_WIDTH   = 9;

    // -----------------------------
    // Quantization zero-points
    // -----------------------------
    parameter int signed INPUT_ZERO_POINT      = -128;
    parameter int signed FC1_WEIGHT_ZERO_POINT = 0;
    parameter int signed FC1_OUTPUT_ZERO_POINT = 0;

endpackage
```

---

# 这版 package 现在够不够用

## 对阶段 1 来说
够作为起点用了，尤其如果你现在先写：

- `input_buffer.sv`
- `cim_tile.sv`
- `psum_accum.sv`

完全够。

## 但后面还要补
等你写到 requant 和 fc2 时，建议再把这些也放进去：

- `FC1_MULTIPLIER_Q31`
- `FC1_SHIFT`
- `FC2_MULTIPLIER_Q31`
- `FC2_SHIFT`

以及如果 `route_b_output/quant_config.json` 里有 scale，也可以把关键常数转成固定点参数放进去。

---

# 还有一个很重要的小建议

SystemVerilog 里 package 最好统一用**全大写参数名**，例如：

- `INPUT_DIMENSION`
- `HIDDEN_DIMENSION`
- `TILE_INPUT_SIZE`

这样你后面在模块里引用：

```systemverilog
import mnist_cim_pkg::*;

logic signed [INPUT_WIDTH-1:0] input_buf [0:INPUT_DIMENSION-1];
```

可读性会明显更好。

---

# 最后给你一句直接判断

## 你的原写法能不能用？
**能用，但不建议直接就这么定稿。**

## 最少要改哪几处？
至少改这四个：

```systemverilog
parameter int hidden_dimension = 128;
parameter int n_output_blocks = hidden_dimension / tile_output_size; // 128 / 16
parameter int signed input_zero_point = -128;
parameter int signed fc1_weight_zero_point = 0;
parameter int signed fc1_output_zero_point = 0;
```

如果你愿意，我下一条可以直接继续帮你把这个 `package.sv` 扩成一版**适合你整个 CIM 工程直接用的完整版本**，把位宽、requant 参数、地址相关常量一起补齐。

用户:
我已经写好了package.sv,如下：

package mnist_cim_pkg;

  // -----------------------------
  // Network dimensions
  // -----------------------------
  parameter int INPUT_DIMENSION = 784;
  parameter int HIDDEN_DIMENSION = 128;
  parameter int OUTPUT_DIMENSION = 10;

  // -----------------------------
  // Tile configuration
  // -----------------------------
  parameter int TILE_INPUT_SIZE = 16;
  parameter int TILE_OUTPUT_SIZE = 16;

  // -----------------------------
  // Block numbers
  // -----------------------------
  parameter int N_INPUT_BLOCKS = INPUT_DIMENSION / TILE_INPUT_SIZE;  // 49
  parameter int N_OUTPUT_BLOCKS = HIDDEN_DIMENSION / TILE_OUTPUT_SIZE;  // 8

  // -----------------------------
  // Data widths
  // -----------------------------
  parameter int INPUT_WIDTH = 8;
  parameter int WEIGHT_WIDTH = 8;
  parameter int BIAS_WIDTH = 32;
  parameter int PSUM_WIDTH = 32;
  parameter int OUTPUT_WIDTH = 8;

  // 因为 x_q - zp 后范围可到 0~255，建议预留 9 bit
  parameter int X_EFF_WIDTH = 9;

  // -----------------------------
  // Quantization zero-points
  // -----------------------------
  parameter int signed INPUT_ZERO_POINT = -128;
  parameter int signed FC1_WEIGHT_ZERO_POINT = 0;
  parameter int signed FC1_OUTPUT_ZERO_POINT = 0;

endpackage

下一步应该是预先加载weight 对吗？

AI助手:


AI助手:
对，下一步就是**预加载 `fc1` 的权重**。  
而且从工程顺序上看，最合理的是先把这三件事做出来：

1. **`fc1_weight_bank.sv`**：把 `fc1_weight_int8.hex` 读进本地权重存储  
2. **`input_buffer.sv`**：把 784 维输入按 `ib` 切成 16 维 tile  
3. **`cim_tile.sv`**：验证一个 `16x16` tile 的并行乘加

其中第一步最关键，因为后面所有 tile 读取都依赖它。

---

# 你现在这份 package 是否合适

先说结论：**可以，作为第一版是合适的。**

你这份 `package` 已经把最重要的基础参数冻结了：

- 网络维度
- tile 尺寸
- block 数量
- 位宽
- zero-point

这一步做得对。

我只建议你补两类参数，后面会更省事。

---

## 建议补充 1：fc1 / fc2 的维度常量
后面写模块时会更方便：

```systemverilog
parameter int FC1_IN_DIM  = INPUT_DIMENSION;
parameter int FC1_OUT_DIM = HIDDEN_DIMENSION;

parameter int FC2_IN_DIM  = HIDDEN_DIMENSION;
parameter int FC2_OUT_DIM = OUTPUT_DIMENSION;
```

---

## 建议补充 2：地址相关宽度
后面写计数器和 ROM 地址时很有用：

```systemverilog
parameter int FC1_WEIGHT_DEPTH = HIDDEN_DIMENSION * INPUT_DIMENSION; // 100352
parameter int FC1_BIAS_DEPTH   = HIDDEN_DIMENSION;                   // 128
parameter int FC2_WEIGHT_DEPTH = OUTPUT_DIMENSION * HIDDEN_DIMENSION; // 1280
parameter int FC2_BIAS_DEPTH   = OUTPUT_DIMENSION;                    // 10
```

如果你愿意更进一步，也可以补地址位宽函数，但第一版不是必须。

---

# 下一步为什么是“预加载 weight”

因为你现在要做的是 **weight-stationary 的 CIM 数据流**。

这意味着：

- 权重先放到本地存储
- 计算时尽量不动
- 输入 tile 逐块广播进去
- tile 内部做局部乘加

所以顺序一定是：

### 先准备权重存储
再准备输入 tile
再做 tile MAC

如果你不先把权重存储组织清楚，后面 `cim_tile` 和 `fc1_cim_core` 都会没法对地址。

---

# 下一步具体做什么

我建议你下一步不是直接写一个很大的 `fc1_cim_core.sv`，而是先写：

## 第一步：`fc1_weight_bank.sv`

它负责：

- 从 `fc1_weight_int8.hex` 读入全部权重
- 按 `[out][in]` 逻辑组织
- 根据 `ob` 和 `ib` 输出当前 `16x16` tile

这是整个 CIM 核心的“权重侧接口”。

---

# 这个模块应该长什么样

## 模块职责
输入：
- `ob`：输出块编号 `0..7`
- `ib`：输入块编号 `0..48`

输出：
- `w_tile[0:15][0:15]`

也就是输出当前 tile 的权重子矩阵。

---

## 地址关系要先定死

你当前 `fc1_weight_int8.hex` 是按：

\[
[out][in]
\]

展开的。

所以 tile 内元素：

- `tr = 0..15`
- `tc = 0..15`

对应全局：

\[
out = ob \times 16 + tr
\]
\[
in = ib \times 16 + tc
\]

线性地址：

\[
addr = out \times 784 + in
\]

这是你整个设计最关键的一条地址公式。

---

# 推荐的第一版写法

第一版先别急着真拆成 8 个物理 bank。  
先用一个**一维 ROM**把整个 `fc1_weight_int8.hex` 读进来，再在组合逻辑里切 tile。

这样最好调试。

---

## 第一版 `fc1_weight_bank.sv`

```systemverilog
module fc1_weight_bank;
```

建议先这样写：

```systemverilog
module fc1_weight_bank #(
    parameter string WEIGHT_FILE = "fc1_weight_int8.hex"
)(
    input  logic [$clog2(mnist_cim_pkg::N_OUTPUT_BLOCKS)-1:0] ob,
    input  logic [$clog2(mnist_cim_pkg::N_INPUT_BLOCKS)-1:0]  ib,
    output logic signed [mnist_cim_pkg::WEIGHT_WIDTH-1:0]
                 w_tile [0:mnist_cim_pkg::TILE_OUTPUT_SIZE-1]
                        [0:mnist_cim_pkg::TILE_INPUT_SIZE-1]
);

  import mnist_cim_pkg::*;

  logic signed [WEIGHT_WIDTH-1:0] weight_mem [0:HIDDEN_DIMENSION*INPUT_DIMENSION-1];

  integer tr, tc;
  integer out_idx, in_idx, addr;

  initial begin
    $readmemh(WEIGHT_FILE, weight_mem);
  end

  always_comb begin
    for (tr = 0; tr < TILE_OUTPUT_SIZE; tr++) begin
      for (tc = 0; tc < TILE_INPUT_SIZE; tc++) begin
        out_idx = ob * TILE_OUTPUT_SIZE + tr;
        in_idx  = ib * TILE_INPUT_SIZE + tc;
        addr    = out_idx * INPUT_DIMENSION + in_idx;
        w_tile[tr][tc] = weight_mem[addr];
      end
    end
  end

endmodule
```

---

# 这段代码在做什么

它做了三件事：

### 1）把整个 `fc1_weight_int8.hex` 读进 `weight_mem`
这是逻辑上的“本地权重存储”。

### 2）给定 `ob` 和 `ib`
它知道你当前要取哪一个 `16x16` tile。

### 3）输出 `w_tile`
这个就是后面 `cim_tile.sv` 要直接使用的权重子矩阵。

---

# 这个模块写完后怎么验证

你不要急着接完整网络。  
先写一个很小的 testbench 验证地址正确。

---

## 你应该验证什么

### case 1
`ob = 0`, `ib = 0`

这时取到的是：

- 输出 0~15
- 输入 0~15

### case 2
`ob = 0`, `ib = 1`

这时取到的是：

- 输出 0~15
- 输入 16~31

### case 3
`ob = 1`, `ib = 0`

这时取到的是：

- 输出 16~31
- 输入 0~15

只要这三个 case 对了，说明 tile 地址逻辑基本没问题。

---

# 再下一步是什么

等 `fc1_weight_bank.sv` 验证完，下一步就是：

## `input_buffer.sv`

它负责：

- 装载 `input_i.hex`
- 根据 `ib` 给出 `x_tile[16]`
- 最好顺带输出 `x_eff_tile = x_q + 128`

因为你当前 zero-point 是：

```systemverilog
INPUT_ZERO_POINT = -128;
```

所以：

\[
x_{eff} = x_q - z_x = x_q + 128
\]

这一步最好在阵列前做掉，不要留到 tile 里混着处理。

---

# 你现在的正确开发顺序

建议你严格按这个顺序写：

## 第 1 步
写并验证 `fc1_weight_bank.sv`

## 第 2 步
写并验证 `input_buffer.sv`

## 第 3 步
写 `cim_tile.sv`

## 第 4 步
写 `psum_accum.sv`

## 第 5 步
写 `fc1_cim_core.sv`

这样最稳。

---

# 你现在这一问的直接答案

## “下一步应该是预先加载 weight 对吗？”
**对。**

更准确地说：

> 下一步应该实现 `fc1_weight_bank.sv`，将 `fc1_weight_int8.hex` 预加载到本地权重存储，并能根据 `ob` 和 `ib` 输出当前 `16x16` 的权重 tile。

这是你进入真正 CIM RTL 的第一步。

---

# 我再给你一个小建议

你现在已经有 `package.sv`，下一步建议把文件名也统一规范一下：

- `mnist_cim_pkg.sv`
- `fc1_weight_bank.sv`
- `input_buffer.sv`
- `cim_tile.sv`
- `psum_accum.sv`
- `fc1_cim_core.sv`

这样工程结构会很清楚。

---

你现在最合适的下一步，是我继续直接帮你把 **`fc1_weight_bank.sv` 和 `input_buffer.sv`** 两个模块完整写出来，并顺手给一个最小 testbench。

用户:
我已经写好了package.sv,如下：



package mnist_cim_pkg;

  // -----------------------------
  // Network dimensions
  // -----------------------------
  parameter int INPUT_DIM = 784;
  parameter int HIDDEN_DIM = 128;
  parameter int OUTPUT_DIM = 10;

  // -----------------------------
  // Tile configuration
  // -----------------------------
  parameter int TILE_INPUT_SIZE = 16;
  parameter int TILE_OUTPUT_SIZE = 16;

  // -----------------------------
  // Block numbers
  // -----------------------------
  parameter int N_INPUT_BLOCKS = INPUT_DIM / TILE_INPUT_SIZE;  // 49
  parameter int N_OUTPUT_BLOCKS = HIDDEN_DIM / TILE_OUTPUT_SIZE;  // 8

  // -----------------------------
  // Data widths
  // -----------------------------
  parameter int INPUT_WIDTH = 8;
  parameter int WEIGHT_WIDTH = 8;
  parameter int BIAS_WIDTH = 32;
  parameter int PSUM_WIDTH = 32;
  parameter int OUTPUT_WIDTH = 8;

  // 因为 x_q - zp 后范围可到 0~255，建议预留 9 bit
  parameter int X_EFF_WIDTH = 9;

  // -----------------------------
  // Quantization zero-points
  // -----------------------------
  parameter int signed INPUT_ZERO_POINT = -128;
  parameter int signed FC1_WEIGHT_ZERO_POINT = 0;
  parameter int signed FC1_OUTPUT_ZERO_POINT = 0;

  // -----------------------------
  // fc1&fc2 dimension configuration
  // -----------------------------

  parameter int FC1_IN_DIM = INPUT_DIM;
  parameter int FC1_OUT_DIM = HIDDEN_DIM;

  parameter int FC2_IN_DIM = HIDDEN_DIM;
  parameter int FC2_OUT_DIM = OUTPUT_DIM;


  // -----------------------------
  // fc1&fc2 address configuration
  // -----------------------------

  parameter int FC1_WEIGHT_DEPTH = HIDDEN_DIM * INPUT_DIM;
  parameter int FC1_BIAS_DEPTH = HIDDEN_DIM;
  parameter int FC2_WEIGHT_DEPTH = OUTPUT_DIM * HIDDEN_DIM;
  parameter int FC2_BIAS_DEPTH = OUTPUT_DIM;




endpackage


下一步应该是加载weight,分析加载流程，给出正确代码
AI助手:


AI助手:
对，下一步就是**加载 `fc1_weight_int8.hex` 到本地权重存储**。  
而且这一步要先把**加载流程**讲清楚，否则后面 `tile` 取权重很容易错。

你现在的 `package` 基本没问题，可以直接往下走。

---

# 一、先明确你要加载的是什么

你当前 `route_b_output` 里的第一层权重文件是：

- `fc1_weight_int8.hex`

它的逻辑格式是：

- **每行一个 int8 权重**
- **按 `[out][in]` 的 row-major 顺序展开**
- shape 是：

\[
[128][784]
\]

也就是线性顺序：

\[
w[0][0], w[0][1], ..., w[0][783], w[1][0], ..., w[127][783]
\]

所以在硬件里，最基本的线性地址公式必须是：

\[
addr = out \times 784 + in
\]

这件事必须先固定。

---

# 二、加载流程应该怎么设计

你现在要做的不是“边算边从 hex 文件读”，而是分两层理解：

## 第 1 层：仿真/初始化层
在仿真里，用 `$readmemh` 把 `fc1_weight_int8.hex` 读到一个**线性 ROM** 里。

例如：

```systemverilog
logic signed [7:0] weight_mem [0:FC1_WEIGHT_DEPTH-1];
```

然后：

```systemverilog
initial begin
  $readmemh("fc1_weight_int8.hex", weight_mem);
end
```

---

## 第 2 层：tile 访问层
你的 `fc1_weight_bank` 模块根据：

- `ob`：输出块号 `0..7`
- `ib`：输入块号 `0..48`

把当前需要的 `16x16` 权重 tile 取出来，输出成：

```systemverilog
w_tile[tr][tc]
```

其中：

- `tr = 0..15`
- `tc = 0..15`

对应关系是：

\[
out = ob \times 16 + tr
\]

\[
in = ib \times 16 + tc
\]

\[
addr = out \times 784 + in
\]

---

# 三、为什么第一版建议先用“线性大 ROM + tile 映射”

因为这是**最稳、最不容易错**的办法。

你当然也可以一上来就做：

- 8 个 bank
- 每个 bank 存 16×784
- 更接近最终 CIM 物理组织

但第一版更容易出错，尤其是：

- bank 地址偏移
- tile 索引
- `[out][in]` 线性展开关系

所以建议：

## 第一版
先做一个**逻辑正确版**：

- 内部只有一个线性 `weight_mem`
- 对外输出 `w_tile`
- 等你 `fc1_acc` 对拍通过之后，再优化成多 bank 版本

这才是正确流程。

---

# 四、正确代码应该怎么写

下面我给你一版**推荐的第一版正确代码**：

- 先做 `fc1_weight_bank.sv`
- 内部用一维 ROM
- 输出 `16x16` tile
- 完全对齐你现在的 `package`

---

## 代码：`fc1_weight_bank.sv`

```systemverilog
module fc1_weight_bank import mnist_cim_pkg::*; #(
  parameter string WEIGHT_HEX_FILE = "fc1_weight_int8.hex"
)(
  input  logic [$clog2(N_OUTPUT_BLOCKS)-1:0] ob,  // output block index: 0..7
  input  logic [$clog2(N_INPUT_BLOCKS)-1:0]  ib,  // input block index: 0..48

  output logic signed [WEIGHT_WIDTH-1:0] w_tile [0:TILE_OUTPUT_SIZE-1][0:TILE_INPUT_SIZE-1]
);

  // ------------------------------------------------------------
  // Linear weight ROM
  // route_b_output/fc1_weight_int8.hex layout:
  //   [out][in] row-major
  // addr = out * FC1_IN_DIM + in
  // ------------------------------------------------------------
  logic signed [WEIGHT_WIDTH-1:0] weight_mem [0:FC1_WEIGHT_DEPTH-1];

  integer i, j;
  integer out_idx;
  integer in_idx;
  integer addr;

  // ------------------------------------------------------------
  // Load weights from hex file
  // ------------------------------------------------------------
  initial begin
    $display("[fc1_weight_bank] Loading weight file: %s", WEIGHT_HEX_FILE);
    $readmemh(WEIGHT_HEX_FILE, weight_mem);
  end

  // ------------------------------------------------------------
  // Tile extraction
  // w_tile[tr][tc] = weight_mem[(ob*TILE_OUTPUT_SIZE+tr)*FC1_IN_DIM + (ib*TILE_INPUT_SIZE+tc)]
  // ------------------------------------------------------------
  always_comb begin
    for (i = 0; i < TILE_OUTPUT_SIZE; i++) begin
      for (j = 0; j < TILE_INPUT_SIZE; j++) begin
        out_idx = ob * TILE_OUTPUT_SIZE + i;
        in_idx  = ib * TILE_INPUT_SIZE  + j;
        addr    = out_idx * FC1_IN_DIM + in_idx;
        w_tile[i][j] = weight_mem[addr];
      end
    end
  end

endmodule
```

---

# 五、这版代码为什么是对的

这版代码的关键点都对上了。

## 1）文件格式对上
你现在的 `fc1_weight_int8.hex` 是：

- int8
- 每行一个数
- row-major `[out][in]`

而这版代码读取就是按这个规则来的。

---

## 2）tile 维度对上
你 package 里已经定义：

- `TILE_INPUT_SIZE = 16`
- `TILE_OUTPUT_SIZE = 16`

所以每次输出的确实是：

\[
16 \times 16
\]

的权重子矩阵。

---

## 3）地址公式对上
这句最关键：

```systemverilog
addr = out_idx * FC1_IN_DIM + in_idx;
```

这就是你当前 `route_b_output` 的正确映射方式。

---

# 六、这一步的执行顺序是什么

你现在应该这样做：

## Step 1
先把这个模块单独写出来。

---

## Step 2
写一个很小的 testbench，只测试它本身。

例如固定：

- `ob = 0`
- `ib = 0`

看输出的 `w_tile[0][0] ~ w_tile[15][15]` 是否就是：

- 输出 0~15
- 输入 0~15

对应的权重块。

---

## Step 3
再测：

- `ob = 0, ib = 1`
- `ob = 1, ib = 0`

确认 tile 切换正确。

---

# 七、建议你立刻加两个辅助函数/注释

虽然不是必须，但非常建议加，能减少后面 debug 痛苦。

---

## 建议 1：把地址公式写成局部 function

这样可读性更高。

### 可选改写版

```systemverilog
module fc1_weight_bank import mnist_cim_pkg::*; #(
  parameter string WEIGHT_HEX_FILE = "fc1_weight_int8.hex"
)(
  input  logic [$clog2(N_OUTPUT_BLOCKS)-1:0] ob,
  input  logic [$clog2(N_INPUT_BLOCKS)-1:0]  ib,
  output logic signed [WEIGHT_WIDTH-1:0] w_tile [0:TILE_OUTPUT_SIZE-1][0:TILE_INPUT_SIZE-1]
);

  logic signed [WEIGHT_WIDTH-1:0] weight_mem [0:FC1_WEIGHT_DEPTH-1];

  function automatic int calc_addr(
    input int out_idx,
    input int in_idx
  );
    calc_addr = out_idx * FC1_IN_DIM + in_idx;
  endfunction

  integer tr, tc;
  integer out_idx, in_idx;

  initial begin
    $display("[fc1_weight_bank] Loading weight file: %s", WEIGHT_HEX_FILE);
    $readmemh(WEIGHT_HEX_FILE, weight_mem);
  end

  always_comb begin
    for (tr = 0; tr < TILE_OUTPUT_SIZE; tr++) begin
      for (tc = 0; tc < TILE_INPUT_SIZE; tc++) begin
        out_idx = ob * TILE_OUTPUT_SIZE + tr;
        in_idx  = ib * TILE_INPUT_SIZE  + tc;
        w_tile[tr][tc] = weight_mem[calc_addr(out_idx, in_idx)];
      end
    end
  end

endmodule
```

---

## 建议 2：明确注明“这是逻辑 bank，不是最终物理 bank”
因为这版还不是：

- 8 个独立 BRAM bank
- 真正物理分块存储

而是：

- 逻辑上抽取 tile
- 正确性优先

这个要在注释里写清楚，不然你后面容易误以为自己已经完成了“物理 bank 设计”。

---

# 八、你现在这个阶段最容易犯的错误

这一步最常见有 4 个坑。

---

## 坑 1：把 `[in][out]` 搞反
你当前文件是 `[out][in]`，不是 `[in][out]`。

所以必须：

```systemverilog
addr = out * 784 + in;
```

不能反过来。

---

## 坑 2：tile 内行列理解反了
建议统一：

- `tr`：tile row，对应输出索引
- `tc`：tile col，对应输入索引

即：

```systemverilog
w_tile[tr][tc]
```

不要写成 `w_tile[tc][tr]`。

---

## 坑 3：文件路径问题
`$readmemh` 路径是相对仿真工作目录的。  
所以 testbench 运行时，要确保：

- `fc1_weight_int8.hex` 在仿真目录下
- 或者传绝对/相对正确路径

---

## 坑 4：signed 解释问题
`$readmemh` 读进 `logic signed [7:0]` 通常没问题。  
但后面乘法时一定要确认参与运算的信号也是 signed。

---

# 九、如果你想更像“本地 bank”，下一版怎么升级

等你这版通过后，可以升级成：

## 方案
内部不再用一个线性 `weight_mem`，而是：

```systemverilog
logic signed [7:0] bank_mem [0:N_OUTPUT_BLOCKS-1][0:TILE_OUTPUT_SIZE-1][0:FC1_IN_DIM-1];
```

也就是：

- 第 0 维：输出块 bank
- 第 1 维：bank 内哪一行输出
- 第 2 维：完整 784 维输入方向权重

然后：

```systemverilog
w_tile[tr][tc] = bank_mem[ob][tr][ib*TILE_INPUT_SIZE + tc];
```

这更像真正的本地 bank。

但我建议你现在先别走这步，先把逻辑正确版跑通。

---

# 十、你下一步应该做什么

你现在最合理的顺序就是：

### 1
先把 `fc1_weight_bank.sv` 写好

### 2
写一个简单 testbench 验证：
- `ob=0, ib=0`
- `ob=0, ib=1`
- `ob=1, ib=0`

### 3
确认 `w_tile` 抽出来的内容和 Python 逻辑一致

### 4
再去写：
- `input_buffer.sv`
- `cim_tile.sv`

---

如果你愿意，我下一条可以直接继续给你写 **`input_buffer.sv` 的正确代码**，这样你就能接着进入 Step 2。

用户:
我已经写好了package.sv,如下：



package mnist_cim_pkg;

  // -----------------------------
  // Network dimensions
  // -----------------------------
  parameter int INPUT_DIM = 784;
  parameter int HIDDEN_DIM = 128;
  parameter int OUTPUT_DIM = 10;

  // -----------------------------
  // Tile configuration
  // -----------------------------
  parameter int TILE_INPUT_SIZE = 16;
  parameter int TILE_OUTPUT_SIZE = 16;

  // -----------------------------
  // Block numbers
  // -----------------------------
  parameter int N_INPUT_BLOCKS = INPUT_DIM / TILE_INPUT_SIZE;  // 49
  parameter int N_OUTPUT_BLOCKS = HIDDEN_DIM / TILE_OUTPUT_SIZE;  // 8

  // -----------------------------
  // Data widths
  // -----------------------------
  parameter int INPUT_WIDTH = 8;
  parameter int WEIGHT_WIDTH = 8;
  parameter int BIAS_WIDTH = 32;
  parameter int PSUM_WIDTH = 32;
  parameter int OUTPUT_WIDTH = 8;

  // 因为 x_q - zp 后范围可到 0~255，建议预留 9 bit
  parameter int X_EFF_WIDTH = 9;

  // -----------------------------
  // Quantization zero-points
  // -----------------------------
  parameter int signed INPUT_ZERO_POINT = -128;
  parameter int signed FC1_WEIGHT_ZERO_POINT = 0;
  parameter int signed FC1_OUTPUT_ZERO_POINT = 0;

  // -----------------------------
  // fc1&fc2 dimension configuration
  // -----------------------------

  parameter int FC1_IN_DIM = INPUT_DIM;
  parameter int FC1_OUT_DIM = HIDDEN_DIM;

  parameter int FC2_IN_DIM = HIDDEN_DIM;
  parameter int FC2_OUT_DIM = OUTPUT_DIM;


  // -----------------------------
  // fc1&fc2 address configuration
  // -----------------------------

  parameter int FC1_WEIGHT_DEPTH = HIDDEN_DIM * INPUT_DIM;
  parameter int FC1_BIAS_DEPTH = HIDDEN_DIM;
  parameter int FC2_WEIGHT_DEPTH = OUTPUT_DIM * HIDDEN_DIM;
  parameter int FC2_BIAS_DEPTH = OUTPUT_DIM;




endpackage


下一步应该是加载weight,分析加载流程，给出正确代码，要求设计module和测试testbench完全分开

AI助手:


AI助手:
你这个 `package` 已经可以作为后续 RTL 的统一参数源了。下一步确实应该先把 **fc1 权重加载链路**定下来，因为它会决定后面的 tile 读取、地址公式和 testbench 的组织方式。

AI助手:
对，下一步就该做 **weight 加载模块**。  
而且这一步要先把“**加载流程**”想清楚，再写代码，否则后面 `cim_tile` 和 `fc1_cim_core` 很容易全错。

我建议你现在先只做：

- **设计 module：`fc1_weight_bank.sv`**
- **设计 testbench：`tb_fc1_weight_bank.sv`**
- 两者完全分开
- 先只验证：**能不能从 `fc1_weight_int8.hex` 按 tile 正确读出 `16x16` 权重块**

这一步非常关键，因为后面整个 CIM 的正确性都建立在它之上。

---

# 一、先把加载流程定死

你当前 `route_b_output` 的 `fc1_weight_int8.hex` 是按：

\[
[out][in]
\]

展开的，也就是 row-major：

\[
addr = out \times 784 + in
\]

其中：

- `out = 0..127`
- `in = 0..783`

---

## 我们要读出的对象不是单个权重，而是一个 tile

你的 tile 定义已经固定：

- `TIN = 16`
- `TOUT = 16`

所以一个 tile 是：

\[
16 \times 16
\]

的子矩阵。

---

## tile 的索引定义

设：

- `ob`：输出块索引，`0..7`
- `ib`：输入块索引，`0..48`

tile 内索引：

- `tr`：tile 内输出行，`0..15`
- `tc`：tile 内输入列，`0..15`

对应全局坐标：

\[
out = ob \times 16 + tr
\]

\[
in = ib \times 16 + tc
\]

所以线性地址就是：

\[
addr = (ob \times 16 + tr) \times 784 + (ib \times 16 + tc)
\]

这就是整个 weight 加载模块最核心的地址公式。

---

# 二、模块应该做什么，不应该做什么

## 这个 module 应该做的事
输入：
- `out_block_idx`
- `in_block_idx`

输出：
- `w_tile[16][16]`

也就是直接吐出当前 tile 对应的 16×16 权重。

## 这个 module 现在不该做的事
不要一开始就做：
- AXI
- 动态写入
- 多 bank 真 BRAM 实例化
- 时钟化复杂读延迟
- CIM 控制 FSM

第一步先做成：

**一个纯读取型、可综合的 ROM 风格模块**

这样便于验证地址公式对不对。

---

# 三、推荐的设计方案

我建议你第一版 `fc1_weight_bank.sv` 用下面这种思路：

## 方案：一维 ROM + 组合抽取 tile
内部存：

```systemverilog
logic signed [7:0] weight_mem [0:FC1_WEIGHT_DEPTH-1];
```

初始化：

```systemverilog
$readmemh(WEIGHT_FILE, weight_mem);
```

然后根据 `ob`、`ib`，组合地产生：

```systemverilog
w_tile[tr][tc]
```

优点：
- 和文件完全一致
- 地址公式最直接
- testbench 最容易对拍
- 后面再重构成多 bank 也很方便

---

# 四、正确的 module 代码

下面是建议版本。

## `fc1_weight_bank.sv`

```systemverilog
module fc1_weight_bank #(
    parameter string WEIGHT_FILE = "fc1_weight_int8.hex"
) (
    input  logic [$clog2(mnist_cim_pkg::N_OUTPUT_BLOCKS)-1:0] out_block_idx,
    input  logic [$clog2(mnist_cim_pkg::N_INPUT_BLOCKS)-1:0]  in_block_idx,

    output logic signed [mnist_cim_pkg::WEIGHT_WIDTH-1:0]
        w_tile [0:mnist_cim_pkg::TILE_OUTPUT_SIZE-1]
               [0:mnist_cim_pkg::TILE_INPUT_SIZE-1]
);

  import mnist_cim_pkg::*;

  // --------------------------------------------------
  // Flat weight memory
  // Layout in file:
  //   row-major [out][in]
  // Address:
  //   addr = out * FC1_IN_DIM + in
  // --------------------------------------------------
  logic signed [WEIGHT_WIDTH-1:0] weight_mem [0:FC1_WEIGHT_DEPTH-1];

  initial begin
    $readmemh(WEIGHT_FILE, weight_mem);
  end

  integer tr, tc;
  integer out_idx, in_idx;
  integer addr;

  always_comb begin
    for (tr = 0; tr < TILE_OUTPUT_SIZE; tr = tr + 1) begin
      for (tc = 0; tc < TILE_INPUT_SIZE; tc = tc + 1) begin
        out_idx = out_block_idx * TILE_OUTPUT_SIZE + tr;
        in_idx  = in_block_idx  * TILE_INPUT_SIZE  + tc;
        addr    = out_idx * FC1_IN_DIM + in_idx;

        w_tile[tr][tc] = weight_mem[addr];
      end
    end
  end

endmodule
```

---

# 五、这段代码为什么是正确的

因为它严格对应你当前导出的文件格式。

对于文件里的第 `addr` 行：

\[
addr = out \times 784 + in
\]

而 tile 中的 `w_tile[tr][tc]` 恰好被映射到：

\[
out = ob \times 16 + tr
\]
\[
in = ib \times 16 + tc
\]

代入就得到：

\[
addr = (ob \times 16 + tr)\times 784 + (ib \times 16 + tc)
\]

这和你 `route_b_output` 的权重顺序是一致的。

---

# 六、为什么 testbench 必须独立写

因为 module 和 testbench 分开后，你可以独立验证：

1. 文件能否加载
2. 地址公式是否正确
3. tile 切块是否正确
4. `ob/ib` 是否映射正确

这比一上来把 `fc1_weight_bank` 接到 `cim_tile` 要稳得多。

---

# 七、testbench 应该验证什么

至少验证三件事：

## 1）文件是否加载成功
随机看几个 `w_tile` 是否不是 X

## 2）tile(0,0) 是否正确
对应：

- `out = 0..15`
- `in = 0..15`

也就是文件头一小块

## 3）tile(1,2) 这种非零块是否正确
这样可以验证地址公式不是偶然对了

---

# 八、推荐 testbench 代码

## `tb_fc1_weight_bank.sv`

```systemverilog
`timescale 1ns/1ps

module tb_fc1_weight_bank;

  import mnist_cim_pkg::*;

  logic [$clog2(N_OUTPUT_BLOCKS)-1:0] out_block_idx;
  logic [$clog2(N_INPUT_BLOCKS)-1:0]  in_block_idx;

  logic signed [WEIGHT_WIDTH-1:0]
      w_tile [0:TILE_OUTPUT_SIZE-1]
             [0:TILE_INPUT_SIZE-1];

  // DUT
  fc1_weight_bank #(
      .WEIGHT_FILE("fc1_weight_int8.hex")
  ) dut (
      .out_block_idx(out_block_idx),
      .in_block_idx(in_block_idx),
      .w_tile(w_tile)
  );

  // 参考内存：testbench 自己也读一份文件，用于对拍
  logic signed [WEIGHT_WIDTH-1:0] ref_weight_mem [0:FC1_WEIGHT_DEPTH-1];

  integer tr, tc;
  integer expected_addr;
  integer expected_out, expected_in;
  integer error_count;

  initial begin
    $readmemh("fc1_weight_int8.hex", ref_weight_mem);

    error_count = 0;

    // -------------------------
    // Test 1: tile (0,0)
    // -------------------------
    out_block_idx = 0;
    in_block_idx  = 0;
    #1;

    $display("Checking tile (0,0)...");
    for (tr = 0; tr < TILE_OUTPUT_SIZE; tr = tr + 1) begin
      for (tc = 0; tc < TILE_INPUT_SIZE; tc = tc + 1) begin
        expected_out  = out_block_idx * TILE_OUTPUT_SIZE + tr;
        expected_in   = in_block_idx  * TILE_INPUT_SIZE  + tc;
        expected_addr = expected_out * FC1_IN_DIM + expected_in;

        if (w_tile[tr][tc] !== ref_weight_mem[expected_addr]) begin
          $display("ERROR tile(0,0): tr=%0d tc=%0d got=%0d expected=%0d addr=%0d",
                    tr, tc, w_tile[tr][tc], ref_weight_mem[expected_addr], expected_addr);
          error_count = error_count + 1;
        end
      end
    end

    // -------------------------
    // Test 2: tile (1,2)
    // -------------------------
    out_block_idx = 1;
    in_block_idx  = 2;
    #1;

    $display("Checking tile (1,2)...");
    for (tr = 0; tr < TILE_OUTPUT_SIZE; tr = tr + 1) begin
      for (tc = 0; tc < TILE_INPUT_SIZE; tc = tc + 1) begin
        expected_out  = out_block_idx * TILE_OUTPUT_SIZE + tr;
        expected_in   = in_block_idx  * TILE_INPUT_SIZE  + tc;
        expected_addr = expected_out * FC1_IN_DIM + expected_in;

        if (w_tile[tr][tc] !== ref_weight_mem[expected_addr]) begin
          $display("ERROR tile(1,2): tr=%0d tc=%0d got=%0d expected=%0d addr=%0d",
                    tr, tc, w_tile[tr][tc], ref_weight_mem[expected_addr], expected_addr);
          error_count = error_count + 1;
        end
      end
    end

    // -------------------------
    // Test 3: last tile (7,48)
    // -------------------------
    out_block_idx = N_OUTPUT_BLOCKS - 1;
    in_block_idx  = N_INPUT_BLOCKS - 1;
    #1;

    $display("Checking tile (last,last)...");
    for (tr = 0; tr < TILE_OUTPUT_SIZE; tr = tr + 1) begin
      for (tc = 0; tc < TILE_INPUT_SIZE; tc = tc + 1) begin
        expected_out  = out_block_idx * TILE_OUTPUT_SIZE + tr;
        expected_in   = in_block_idx  * TILE_INPUT_SIZE  + tc;
        expected_addr = expected_out * FC1_IN_DIM + expected_in;

        if (w_tile[tr][tc] !== ref_weight_mem[expected_addr]) begin
          $display("ERROR tile(last,last): tr=%0d tc=%0d got=%0d expected=%0d addr=%0d",
                    tr, tc, w_tile[tr][tc], ref_weight_mem[expected_addr], expected_addr);
          error_count = error_count + 1;
        end
      end
    end

    if (error_count == 0) begin
      $display("PASS: fc1_weight_bank tile extraction is correct.");
    end else begin
      $display("FAIL: found %0d mismatches.", error_count);
    end

    $finish;
  end

endmodule
```

---

# 九、运行时需要注意的点

## 1）`fc1_weight_int8.hex` 的路径
仿真时这个文件必须在 simulator 的工作目录下，或者你写绝对/相对路径。

如果文件在 `route_b_output/` 里，那么可以写：

```systemverilog
.WEIGHT_FILE("route_b_output/fc1_weight_int8.hex")
```

以及 testbench 里也要一致：

```systemverilog
$readmemh("route_b_output/fc1_weight_int8.hex", ref_weight_mem);
```

---

## 2）signed 显示
你在 `$display` 时用 `%0d`，这样能直接看成有符号十进制，方便排错。

---

## 3）为什么 `#1`
因为 `w_tile` 是组合输出，改 `ob/ib` 后给一点时间传播更稳妥。

---

# 十、下一步正确顺序

你现在不要马上写 `cim_core`。  
先按这个顺序走：

## 第一步
完成并验证：
- `fc1_weight_bank.sv`
- `tb_fc1_weight_bank.sv`

## 第二步
再写：
- `input_buffer.sv`
- `tb_input_buffer.sv`

## 第三步
再写：
- `cim_tile.sv`
- `tb_cim_tile.sv`

## 第四步
再把它们拼成：
- `psum_accum.sv`
- `fc1_cim_core.sv`

这样最稳。

---

# 十一、如果你想更贴近真实 BRAM，后面怎么升级

现在这版是：

- 一维 ROM
- 组合抽 tile

后面可以升级成：

## 方案 A：输出 bank 化
每个 `ob` 一个 bank：

```systemverilog
logic signed [7:0] bank_mem [0:N_OUTPUT_BLOCKS-1][0:TILE_OUTPUT_SIZE-1][0:FC1_IN_DIM-1];
```

然后：

```systemverilog
w_tile[tr][tc] = bank_mem[out_block_idx][tr][in_block_idx*TILE_INPUT_SIZE+tc];
```

这会更接近本地阵列 bank 的感觉。

但第一版没必要这么快做。

---

# 十二、你这一步完成后的意义

一旦 `fc1_weight_bank` 对了，你后面就已经锁定了最关键的一件事：

**权重 tile 的地址映射完全正确。**

这样你后面写 `cim_tile` 和 `fc1_cim_core` 时，就不需要再反复怀疑文件顺序和地址公式了。

---

你现在最合理的下一步就是把这两个文件先放进工程里跑仿真。  
等你跑完，如果有报错信息或波形现象，我再继续帮你看，并带你写下一步的 `input_buffer.sv`。

用户:
为什么DUT里面有initial?
AI助手:


AI助手:
因为这里的 `initial` 不是拿来写 testbench 行为流程，而是拿来做：

**仿真/FPGA 初始化存储器内容。**

在你这个 `fc1_weight_bank` 里，`initial` 的作用是：

```systemverilog
initial begin
  $readmemh(WEIGHT_FILE, weight_mem);
end
```

意思是：

- 在仿真开始时
- 把 `fc1_weight_int8.hex` 里的数据
- 读进 `weight_mem`

所以它出现在 DUT 里，不是为了“激励”，而是为了**初始化 DUT 内部 ROM/存储器**。

---

# 一、为什么 DUT 里可以有 `initial`

很多人一开始会觉得：

- `initial` 只能写在 testbench
- DUT 里不该有 `initial`

其实不完全对。

在 RTL 设计里，`initial` 有两种常见用途：

## 1）testbench 行为控制
比如：

- 产生时钟
- 施加激励
- 打印结果

这种通常只出现在 testbench。

## 2）存储器初始化
比如：

- `$readmemh`
- `$readmemb`

这种经常会出现在 DUT 里，尤其是：

- ROM
- lookup table
- FPGA 上预初始化的 BRAM
- 权重存储器

你现在这个 `weight_mem` 就属于第二类。

---

# 二、在你的场景里，为什么这么写是合理的

你当前的 `fc1_weight_bank` 本质上是：

**一个只读权重存储模块**

它不是实时从总线写入权重，而是：

- 启动前就把权重装好
- 计算时直接读

这和 FPGA 上常见的 ROM / 预加载 BRAM 很像。

所以在 DUT 里这样写：

```systemverilog
logic signed [7:0] weight_mem [0:FC1_WEIGHT_DEPTH-1];

initial begin
  $readmemh(WEIGHT_FILE, weight_mem);
end
```

逻辑上等价于：

> 这个模块内部带了一块初始化好的 ROM，内容来自权重文件。

这在前期验证阶段非常常见，也很合适。

---

# 三、它和 testbench 里的 `initial` 有什么区别

## DUT 里的 `initial`
作用：
- 初始化内部存储器内容
- 让 DUT 一上电就“自带权重”

它更像“烧进去的系数表”。

---

## testbench 里的 `initial`
作用：
- 驱动输入
- 切换控制信号
- 检查结果

例如：

```systemverilog
initial begin
  out_block_idx = 0;
  in_block_idx  = 0;
  #1;
  ...
end
```

这是激励流程。

---

# 四、这样写有没有问题

## 对仿真来说
**没问题，而且很常用。**

---

## 对 FPGA 综合来说
要分情况看。

### 情况 1：大多数 FPGA 工具
如果是 ROM / memory init 风格，很多综合工具支持：

- `initial`
- `$readmemh`

并把它映射成：

- 初始化 BRAM
- 初始化分布式 ROM

这在 FPGA 里是常见做法。

---

### 情况 2：ASIC 流程
ASIC 一般不依赖 `initial` 来初始化硬件存储器。  
会用别的方式：

- scan/load
- boot ROM
- fuse
- 固化 memory init

但你现在是 FPGA 项目，所以前期这么写是合理的。

---

# 五、为什么我前面建议这么写

因为你当前阶段的目标不是先做“可在线写权重的最终 SoC”，而是先验证：

1. 权重文件格式正确
2. 地址映射正确
3. tile 提取正确
4. 后续 `cim_tile` 计算正确

所以最简单、最稳的做法就是：

- DUT 内部直接 `$readmemh`
- 让模块自己带权重

这样你可以把注意力放在**阵列地址和乘加**，而不是总线写入流程。

---

# 六、什么时候不该在 DUT 里用 `initial`

当你后面要进入更正式的系统集成阶段时，通常就不建议把最终设计完全依赖这种方式。

例如你后面如果要做：

- 可配置 IP
- CPU 下发权重
- AXI 接口装载模型
- DDR / BRAM 搬运权重

那时更常见的做法是：

- DUT 提供写接口
- testbench 或软件把权重写进去
- 而不是 DUT 自己读 hex

也就是说：

## 现在这个阶段
适合用 `initial + $readmemh`

## 后期系统级阶段
更适合用：
- write port
- preload controller
- DMA / AXI

---

# 七、你现在这个模块更准确的定位

你现在的 `fc1_weight_bank` 其实不是“最终的可编程权重存储器”，而是：

**用于阶段 1 验证的 ROM 风格权重 bank**

所以 DUT 里有 `initial` 很正常。

---

# 八、如果你特别想“module 和 testbench 完全职责分离”，可以怎么改

你如果坚持认为：

> DUT 不应该自己读文件，文件读取只应该由 testbench 完成

那也可以，改法是：

## 方案：DUT 提供写接口
让 DUT 只保留 memory：

```systemverilog
logic signed [7:0] weight_mem [0:FC1_WEIGHT_DEPTH-1];
```

然后加写口：

```systemverilog
input logic clk;
input logic we;
input logic [$clog2(FC1_WEIGHT_DEPTH)-1:0] wr_addr;
input logic signed [7:0] wr_data;
```

写入逻辑：

```systemverilog
always_ff @(posedge clk) begin
  if (we) begin
    weight_mem[wr_addr] <= wr_data;
  end
end
```

然后 testbench：
- 自己 `$readmemh`
- 再一拍一拍写给 DUT

这样就真正做到了：
- DUT 不依赖文件
- testbench 负责装载数据

---

## 但为什么我不先推荐这个
因为你现在还在验证：
- 地址公式
- tile 切块

如果再加一个写接口，会多出一层复杂度：

- 写入顺序对不对
- 写入时序对不对
- 写完再读有没有同步延迟

这会分散你当前最重要的问题。

所以在**前期功能验证**阶段，我更推荐 DUT 内部 `initial + $readmemh`。

---

# 九、给你一句最直接的判断

## 为什么 DUT 里面有 `initial`？
因为这里的 DUT 充当的是**预初始化权重 ROM**，需要在仿真/FPGA 启动时把 hex 文件载入内部存储器。

## 这样写对不对？
**对，在你当前这个阶段是合理且常见的。**

## 什么时候该改掉？
当你进入：
- 更真实的 SoC 系统
- 动态装载权重
- 总线写入
- 可复用 IP

那时再改成“外部写接口 + testbench/软件加载”。

---

你现在这一版先不用纠结 `initial` 是否“纯不纯”。  
更重要的是把 `fc1_weight_bank` 的 **tile 读出功能先验证对**。

用户:
  $readmemh(WEIGHT_FILE, weight_mem);
如果路径是 ../../CIM-sw-version1/sw/train&quantize/route_b_output/fc1_weight_int8.hex,怎么写

AI助手:


AI助手:
直接这样写：

```systemverilog
$readmemh("../../CIM-sw-version1/sw/train&quantize/route_b_output/fc1_weight_int8.hex", weight_mem);
```

如果你是通过参数传进去，就写成：

```systemverilog
module fc1_weight_bank #(
    parameter string WEIGHT_FILE = "../../CIM-sw-version1/sw/train&quantize/route_b_output/fc1_weight_int8.hex"
) (
    ...
);
```

然后初始化：

```systemverilog
initial begin
  $readmemh(WEIGHT_FILE, weight_mem);
end
```

---

# 但这里有一个需要特别注意的点

你的路径里有：

```text
train&quantize
```

其中包含 `&`。  
这在很多仿真器里**有可能没问题**，但也有一些环境、脚本、Makefile、shell 调用时会因为 `&` 产生麻烦。

所以更稳妥的建议是：

## 最好改目录名
把：

```text
train&quantize
```

改成：

```text
train_quantize
```

或者：

```text
train_quantize_out
```

这样最省事。

---

# 如果你不改目录名，注意两件事

## 1）SystemVerilog 字符串里直接写通常是可以的
也就是：

```systemverilog
$readmemh("../../CIM-sw-version1/sw/train&quantize/route_b_output/fc1_weight_int8.hex", weight_mem);
```

这在 HDL 代码本身里通常没问题。

## 2）但如果你在脚本里传参，就要小心 shell
比如你用仿真脚本传：

```bash
+WEIGHT_FILE=../../CIM-sw-version1/sw/train&quantize/route_b_output/fc1_weight_int8.hex
```

shell 可能会把 `&` 当后台符号处理。  
这种情况下就容易炸。

---

# 最推荐写法

如果你是直接写死在 module 里：

```systemverilog
module fc1_weight_bank #(
    parameter string WEIGHT_FILE = "../../CIM-sw-version1/sw/train&quantize/route_b_output/fc1_weight_int8.hex"
) (
    input  logic [$clog2(mnist_cim_pkg::N_OUTPUT_BLOCKS)-1:0] out_block_idx,
    input  logic [$clog2(mnist_cim_pkg::N_INPUT_BLOCKS)-1:0]  in_block_idx,
    output logic signed [mnist_cim_pkg::WEIGHT_WIDTH-1:0]
        w_tile [0:mnist_cim_pkg::TILE_OUTPUT_SIZE-1]
               [0:mnist_cim_pkg::TILE_INPUT_SIZE-1]
);

  import mnist_cim_pkg::*;

  logic signed [WEIGHT_WIDTH-1:0] weight_mem [0:FC1_WEIGHT_DEPTH-1];

  initial begin
    $readmemh(WEIGHT_FILE, weight_mem);
  end

  ...
endmodule
```

---

# 我再提醒一个更关键的问题

这个相对路径：

```text
../../CIM-sw-version1/sw/train&quantize/route_b_output/fc1_weight_int8.hex
```

是不是正确，取决于：

**仿真运行时的工作目录在哪。**

也就是说，相对路径不是相对于 `.sv` 文件位置，而是相对于**仿真器启动目录**。

---

## 举个例子
如果你是在：

```text
project/sim/
```

目录下启动仿真，那这个路径就是相对 `project/sim/` 来解析的。

所以你最好确认：

- 你是在哪个目录执行 `xrun / vcs / questa / iverilog`
- 从那个目录出发，`../../CIM-sw-version1/...` 是否真的能找到文件

---

# 最稳的调试方法

你可以先在 `initial` 里打印一下：

```systemverilog
initial begin
  $display("Loading weight file: %s", WEIGHT_FILE);
  $readmemh(WEIGHT_FILE, weight_mem);
end
```

这样至少你能确认仿真时实际尝试加载的是哪个路径。

---

# 建议

真正做工程时，最稳的方法通常是：

- 把 `.hex` 文件复制到仿真目录
- 然后直接写：

```systemverilog
$readmemh("fc1_weight_int8.hex", weight_mem);
```

这样最少踩坑。

用户:
bash run_fc1_weight_bank.sh: ../rtl/package.sv:3 syntax error
AI助手:


AI助手:
我先看一下你上传的硬件工程里 `package.sv` 的实际内容和 `run_fc1_weight_bank.sh` 用的仿真器/编译命令。这个报错很像是工具链对 SystemVerilog package 支持或编译选项的问题，也可能是 `parameter int`/`package` 语法本身被当成 Verilog 在编。

AI助手:
报错原因已经很明确了：

## 根因
你的脚本现在是：

```bash
iverilog -o ../sim/fc1_weight_bank \
	../rtl/package.sv \
	../rtl/fc1_weight_bank.sv \
	../tb/tb_fc1_weight_bank.sv
```

这里 **没有打开 SystemVerilog 2012 模式**。  
`iverilog` 默认按较老的 Verilog 语法处理，所以看到：

```systemverilog
package mnist_cim_pkg;
```

就会在第 3 行附近报 syntax error。

---

# 一、先改脚本

把 `run_fc1_weight_bank.sh` 改成下面这样：

```bash
#!/usr/bin/env bash
set -e

mkdir -p ../sim

iverilog -g2012 -o ../sim/fc1_weight_bank \
    ../rtl/package.sv \
    ../rtl/fc1_weight_bank.sv \
    ../tb/tb_fc1_weight_bank.sv

vvp ../sim/fc1_weight_bank
```

其中最关键的是：

```bash
-g2012
```

这个参数必须有。

---

# 二、但你工程里不止一个问题

除了 `package.sv` 被当成旧 Verilog 解析之外，你的 `tb_fc1_weight_bank.sv` 里还有几处明显错误，哪怕加了 `-g2012`，后面也会继续报错。

我已经帮你看出来了。

---

## 问题 1：testbench 里实例化参数名写错了

你模块里定义的是：

```systemverilog
parameter string WEIGHT_FILE = ...
```

但 testbench 里写的是：

```systemverilog
fc1_weight_bank #(
    .WEIGHT_HEX_FILE("../../CIM-sw-version1/sw/train&quantize/route_b_output/fc1_weight_int8.hex")
) dut (
```

这里的 `WEIGHT_HEX_FILE` 不存在。  
应该改成：

```systemverilog
fc1_weight_bank #(
    .WEIGHT_FILE("../../CIM-sw-version1/sw/train&quantize/route_b_output/fc1_weight_int8.hex")
) dut (
```

---

## 问题 2：testbench 里的信号名前后不一致

你声明的是：

```systemverilog
logic [$clog2(N_OUTPUT_BLOCKS)-1:0] ob;
logic [$clog2(N_INPUT_BLOCKS)-1:0] ib;
```

但后面写的是：

```systemverilog
out_block_idx = 0;
in_block_idx  = 0;
```

这两个变量根本没定义。  
应该全部改成：

```systemverilog
ob = 0;
ib = 0;
```

后面所有地方同理。

---

## 问题 3：testbench 里参考文件路径不一致

你 DUT 里给的是：

```systemverilog
"../../CIM-sw-version1/sw/train&quantize/route_b_output/fc1_weight_int8.hex"
```

但 testbench 自己读参考文件时写的是：

```systemverilog
$readmemh("fc1_weight_int8.hex", ref_weight_mem);
```

如果当前运行目录下没有这个文件，就会失败。

建议统一成同一路径。

---

# 三、给你一版修正后的 testbench

把 `tb_fc1_weight_bank.sv` 改成下面这版：

```systemverilog
`timescale 1ns / 1ps

module tb_fc1_weight_bank;
  import mnist_cim_pkg::*;

  logic [$clog2(N_OUTPUT_BLOCKS)-1:0] ob;
  logic [$clog2(N_INPUT_BLOCKS)-1:0] ib;

  logic signed [WEIGHT_WIDTH-1:0] w_tile [0:TILE_OUTPUT_SIZE-1][0:TILE_INPUT_SIZE-1];

  fc1_weight_bank #(
      .WEIGHT_FILE("../../CIM-sw-version1/sw/train&quantize/route_b_output/fc1_weight_int8.hex")
  ) dut (
      .ob(ob),
      .ib(ib),
      .w_tile(w_tile)
  );

  logic signed [WEIGHT_WIDTH-1:0] ref_weight_mem [0:FC1_WEIGHT_DEPTH-1];

  integer tr, tc;
  integer expected_addr;
  integer expected_out, expected_in;
  integer error_count;

  initial begin
    $readmemh("../../CIM-sw-version1/sw/train&quantize/route_b_output/fc1_weight_int8.hex",
              ref_weight_mem);

    error_count = 0;

    // -------------------------
    // Test 1: tile (0,0)
    // -------------------------
    ob = 0;
    ib = 0;
    #1;

    $display("Checking tile (0,0)...");
    for (tr = 0; tr < TILE_OUTPUT_SIZE; tr = tr + 1) begin
      for (tc = 0; tc < TILE_INPUT_SIZE; tc = tc + 1) begin
        expected_out  = ob * TILE_OUTPUT_SIZE + tr;
        expected_in   = ib * TILE_INPUT_SIZE + tc;
        expected_addr = expected_out * FC1_IN_DIM + expected_in;

        if (w_tile[tr][tc] !== ref_weight_mem[expected_addr]) begin
          $display("ERROR tile(0,0): tr=%0d tc=%0d got=%0d expected=%0d addr=%0d",
                   tr, tc, w_tile[tr][tc], ref_weight_mem[expected_addr], expected_addr);
          error_count = error_count + 1;
        end
      end
    end

    // -------------------------
    // Test 2: tile (1,2)
    // -------------------------
    ob = 1;
    ib = 2;
    #1;

    $display("Checking tile (1,2)...");
    for (tr = 0; tr < TILE_OUTPUT_SIZE; tr = tr + 1) begin
      for (tc = 0; tc < TILE_INPUT_SIZE; tc = tc + 1) begin
        expected_out  = ob * TILE_OUTPUT_SIZE + tr;
        expected_in   = ib * TILE_INPUT_SIZE + tc;
        expected_addr = expected_out * FC1_IN_DIM + expected_in;

        if (w_tile[tr][tc] !== ref_weight_mem[expected_addr]) begin
          $display("ERROR tile(1,2): tr=%0d tc=%0d got=%0d expected=%0d addr=%0d",
                   tr, tc, w_tile[tr][tc], ref_weight_mem[expected_addr], expected_addr);
          error_count = error_count + 1;
        end
      end
    end

    // -------------------------
    // Test 3: last tile (7,48)
    // -------------------------
    ob = N_OUTPUT_BLOCKS - 1;
    ib = N_INPUT_BLOCKS - 1;
    #1;

    $display("Checking tile (last,last)...");
    for (tr = 0; tr < TILE_OUTPUT_SIZE; tr = tr + 1) begin
      for (tc = 0; tc < TILE_INPUT_SIZE; tc = tc + 1) begin
        expected_out  = ob * TILE_OUTPUT_SIZE + tr;
        expected_in   = ib * TILE_INPUT_SIZE + tc;
        expected_addr = expected_out * FC1_IN_DIM + expected_in;

        if (w_tile[tr][tc] !== ref_weight_mem[expected_addr]) begin
          $display("ERROR tile(last,last): tr=%0d tc=%0d got=%0d expected=%0d addr=%0d",
                   tr, tc, w_tile[tr][tc], ref_weight_mem[expected_addr], expected_addr);
          error_count = error_count + 1;
        end
      end
    end

    if (error_count == 0) begin
      $display("PASS: fc1_weight_bank tile extraction is correct.");
    end else begin
      $display("FAIL: found %0d mismatches.", error_count);
    end

    $finish;
  end

endmodule
```

---

# 四、`fc1_weight_bank.sv` 本身也给你一个稳妥版

你当前 module 基本没问题，但为了减少工具兼容性问题，我建议保留这版：

```systemverilog
module fc1_weight_bank #(
    parameter string WEIGHT_FILE = "../../CIM-sw-version1/sw/train&quantize/route_b_output/fc1_weight_int8.hex"
) (
    input  logic [$clog2(mnist_cim_pkg::N_OUTPUT_BLOCKS)-1:0] ob,
    input  logic [$clog2(mnist_cim_pkg::N_INPUT_BLOCKS)-1:0]  ib,

    output logic signed [mnist_cim_pkg::WEIGHT_WIDTH-1:0]
        w_tile [0:mnist_cim_pkg::TILE_OUTPUT_SIZE-1]
               [0:mnist_cim_pkg::TILE_INPUT_SIZE-1]
);

  import mnist_cim_pkg::*;

  logic signed [WEIGHT_WIDTH-1:0] weight_mem [0:FC1_WEIGHT_DEPTH-1];

  integer tr, tc;
  integer out_idx, in_idx;
  integer addr;

  initial begin
    $display("Loading weight file: %s", WEIGHT_FILE);
    $readmemh(WEIGHT_FILE, weight_mem);
  end

  always_comb begin
    for (tr = 0; tr < TILE_OUTPUT_SIZE; tr = tr + 1) begin
      for (tc = 0; tc < TILE_INPUT_SIZE; tc = tc + 1) begin
        out_idx = ob * TILE_OUTPUT_SIZE + tr;
        in_idx  = ib * TILE_INPUT_SIZE + tc;
        addr    = out_idx * FC1_IN_DIM + in_idx;
        w_tile[tr][tc] = weight_mem[addr];
      end
    end
  end

endmodule
```

---

# 五、如果你加了 `-g2012` 之后还报 package 错

那就说明你本地 `iverilog` 版本太老，不支持 package 或对 SV 支持不完整。

你可以先检查版本：

```bash
iverilog -V
```

如果版本比较老，可能要升级。  
但多数情况下，`-g2012` 就够了。

---

# 六、你现在最该做的事

先按下面顺序改：

1. 改 `run_fc1_weight_bank.sh`，加 `-g2012`
2. 修正 `tb_fc1_weight_bank.sv` 里的：
   - 参数名 `WEIGHT_FILE`
   - 信号名 `ob/ib`
   - 参考文件路径
3. 再跑一次

如果你愿意，我下一步可以继续直接帮你检查 **第二轮编译报错**，或者直接给你写下一步的 `input_buffer.sv`。

用户:
改成synopsys运行把，帮我换一个脚本
AI助手:


AI助手:
可以，改成 **Synopsys VCS** 运行更合适。  
下面我直接给你一版可用的脚本。

---

# 一、VCS 运行脚本

把你的 `run_fc1_weight_bank.sh` 改成下面这样：

```bash
#!/usr/bin/env bash
set -e

mkdir -p ../sim
mkdir -p ../sim/log

vcs -full64 -sverilog -timescale=1ns/1ps \
  ../rtl/package.sv \
  ../rtl/fc1_weight_bank.sv \
  ../tb/tb_fc1_weight_bank.sv \
  -o ../sim/fc1_weight_bank_simv \
  -l ../sim/log/compile_fc1_weight_bank.log

../sim/fc1_weight_bank_simv -l ../sim/log/run_fc1_weight_bank.log
```

---

# 二、如果你想顺便开波形

可以用这版：

```bash
#!/usr/bin/env bash
set -e

mkdir -p ../sim
mkdir -p ../sim/log

vcs -full64 -sverilog -debug_acc -timescale=1ns/1ps \
  ../rtl/package.sv \
  ../rtl/fc1_weight_bank.sv \
  ../tb/tb_fc1_weight_bank.sv \
  -o ../sim/fc1_weight_bank_simv \
  -l ../sim/log/compile_fc1_weight_bank.log

../sim/fc1_weight_bank_simv -l ../sim/log/run_fc1_weight_bank.log
```

如果 testbench 里再加：

```systemverilog
initial begin
  $fsdbDumpfile("fc1_weight_bank.fsdb");
  $fsdbDumpvars(0, tb_fc1_weight_bank);
end
```

就能配合 Verdi 看波形。  
如果你没有 FSDB 环境，也可以先用 VCD：

```systemverilog
initial begin
  $dumpfile("fc1_weight_bank.vcd");
  $dumpvars(0, tb_fc1_weight_bank);
end
```

---

# 三、如果你的 VCS 环境不支持 string parameter 路径写法

大多数新版 VCS 支持：

```systemverilog
parameter string WEIGHT_FILE = "..."
```

如果你那边环境比较老，保险一点也可以保留现在写法先试。  
通常 `-sverilog` 就够了。

---

# 四、推荐的脚本版本

我建议你用下面这个，比较完整：

```bash
#!/usr/bin/env bash
set -e

SIM_DIR=../sim
LOG_DIR=${SIM_DIR}/log

mkdir -p ${SIM_DIR}
mkdir -p ${LOG_DIR}

vcs -full64 -sverilog -debug_acc -timescale=1ns/1ps \
  ../rtl/package.sv \
  ../rtl/fc1_weight_bank.sv \
  ../tb/tb_fc1_weight_bank.sv \
  -o ${SIM_DIR}/fc1_weight_bank_simv \
  -l ${LOG_DIR}/compile_fc1_weight_bank.log

${SIM_DIR}/fc1_weight_bank_simv -l ${LOG_DIR}/run_fc1_weight_bank.log
```

---

# 五、你还要注意两点

## 1）路径还是相对运行目录解析
你在 RTL 里写的：

```systemverilog
"../../CIM-sw-version1/sw/train&quantize/route_b_output/fc1_weight_int8.hex"
```

依然是相对于**运行仿真的当前目录**，不是相对于 `.sv` 文件目录。

所以你执行脚本的位置要稳定。  
最稳的方法是：进入 `scripts/` 目录后执行。

例如：

```bash
cd scripts
bash run_fc1_weight_bank.sh
```

---

## 2）`train&quantize` 这个目录名仍然有风险
VCS 本身读 HDL 字符串路径一般没问题，但 shell、Makefile、外部传参时 `&` 还是容易出问题。  
工程上最好还是改名成：

```text
train_quantize
```

这个更稳。

---

# 六、建议你现在这样做

先把脚本替换成上面这版 VCS 脚本，然后运行。  
如果还有报错，把：

- `compile_fc1_weight_bank.log`
- `run_fc1_weight_bank.log`

里的错误贴给我，我继续帮你往下定位。

用户:
ght_bank.sh 
/opt/Synopsys/VCS2018/vcs/O-2018.09-SP2/bin/vcs: line 6045: ../sim/log/compile_fc1_weight_bank.log: No such file or directory

Error-[NOWRITE_PERM] No write permission on file
  You don't have write permission for log file 
  '../sim/log/compile_fc1_weight_bank.log'.
  Please make sure permission exist

scripts/run_fc1_weight_bank.sh: line 8: ../sim/fc1_weight_bank_simv: No such file or directory
:
AI助手:


AI助手:
这是脚本路径和执行位置导致的，不是 VCS 本身的问题。

## 问题本质
你脚本里写的是：

```bash
mkdir -p ../sim
mkdir -p ../sim/log
...
-l ../sim/log/compile_fc1_weight_bank.log
```

但你当前很可能**不是在 `scripts/` 目录下执行的脚本**，而是在别的目录用：

```bash
bash scripts/run_fc1_weight_bank.sh
```

这样一来，`../sim/log/...` 就不是你以为的那个目录了，VCS 找不到或没有权限写。

---

# 最稳的修法

把脚本改成“**先切到脚本所在目录**”，这样无论你从哪里执行都不会错。

把 `run_fc1_weight_bank.sh` 改成下面这版：

```bash id="e0b0h7"
#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SIM_DIR="../sim"
LOG_DIR="${SIM_DIR}/log"

mkdir -p "${SIM_DIR}"
mkdir -p "${LOG_DIR}"

vcs -full64 -sverilog -debug_acc -timescale=1ns/1ps \
  ../rtl/package.sv \
  ../rtl/fc1_weight_bank.sv \
  ../tb/tb_fc1_weight_bank.sv \
  -o "${SIM_DIR}/fc1_weight_bank_simv" \
  -l "${LOG_DIR}/compile_fc1_weight_bank.log"

"${SIM_DIR}/fc1_weight_bank_simv" -l "${LOG_DIR}/run_fc1_weight_bank.log"
```

---

# 为什么这版能解决

这两行最关键：

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
```

意思是：

- 不管你从哪里运行脚本
- 脚本都会先切换到它自己所在的 `scripts/` 目录
- 后面的 `../sim`、`../rtl`、`../tb` 都会按固定工程结构解析

---

# 你现在应该怎么运行

两种都可以：

### 方式 1
在工程根目录运行：

```bash id="l527q2"
bash scripts/run_fc1_weight_bank.sh
```

### 方式 2
先进入脚本目录：

```bash id="rn8r3u"
cd scripts
bash run_fc1_weight_bank.sh
```

有了上面的修正版脚本，这两种都应该可以。

---

# 再补一个排错增强版

你也可以加几行打印，确认路径没跑偏：

```bash id="ru0g4h"
#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PWD        = $(pwd)"

SIM_DIR="../sim"
LOG_DIR="${SIM_DIR}/log"

mkdir -p "${SIM_DIR}"
mkdir -p "${LOG_DIR}"

echo "SIM_DIR    = ${SIM_DIR}"
echo "LOG_DIR    = ${LOG_DIR}"

vcs -full64 -sverilog -debug_acc -timescale=1ns/1ps \
  ../rtl/package.sv \
  ../rtl/fc1_weight_bank.sv \
  ../tb/tb_fc1_weight_bank.sv \
  -o "${SIM_DIR}/fc1_weight_bank_simv" \
  -l "${LOG_DIR}/compile_fc1_weight_bank.log"

"${SIM_DIR}/fc1_weight_bank_simv" -l "${LOG_DIR}/run_fc1_weight_bank.log"
```

---

# 还有一个你要顺手检查的点

确认 `sim/` 目录不是只读的。  
在工程根目录执行：

```bash id="9vhnzl"
ls -ld sim
ls -ld sim/log
```

如果之前目录被错误权限创建过，可以修一下：

```bash id="myj9ja"
chmod -R u+w sim
```

---

# 你这个报错的连锁关系

第一个错误：

```text
No write permission for log file '../sim/log/compile_fc1_weight_bank.log'
```

导致 VCS 根本没成功生成：

```text
../sim/fc1_weight_bank_simv
```

所以后面才会出现：

```text
No such file or directory
```

也就是说，第二个错误只是第一个错误的结果。

---

先把脚本改成上面这版再跑。跑完如果还有编译错误，把 `compile_fc1_weight_bank.log` 里的报错贴给我。

用户:
现在已经写好了第一版的大ROM weight御读取，下一步是怎么做？进入下一步还是优化本次weight预读取？更新为更贴近真实 BRAM
AI助手:


AI助手:
先不要急着把大 ROM 优化成更像真实 BRAM。

**现在最正确的选择是：进入下一步。**

原因很简单：

你现在刚完成的是**权重 tile 读取链路的第一版功能验证**。  
在整个阶段 1 里，当前最重要的目标不是“存储实现得多漂亮”，而是：

**先把整个 fc1 的 CIM 数据通路跑通。**

也就是说，优先级应该是：

1. **功能正确**
2. **数据流闭环**
3. **逐级 golden 对拍**
4. 然后才是 **BRAM 化 / bank 化 / 时序优化**

---

# 一、为什么现在不建议优先优化 weight 读取

因为你当前这个大 ROM 版本已经完成了它最核心的使命：

- 能从 `fc1_weight_int8.hex` 正确加载权重
- 能按 `(ob, ib)` 正确切出 `16x16` tile
- 能和 testbench 对拍地址公式

这就够了。

如果你现在立刻去做：

- 多 bank BRAM
- 同步读延迟
- 宽字打包
- tile cache
- 真正的 block RAM 推断

你会马上遇到很多“实现问题”，但这时你还没有验证：

- `input_buffer` 是否正确
- `x_q - zp` 是否处理正确
- `cim_tile` 是否正确
- `psum_accum` 是否正确
- `fc1_acc` 是否能对上 `fc1_acc_0.hex`

也就是说，如果现在优化存储，你会把问题源头搞混。

---

# 二、当前最正确的推进顺序

你已经完成了：

- `package.sv`
- `fc1_weight_bank.sv` 第一版大 ROM 读取
- `tb_fc1_weight_bank.sv` 对拍

下一步最应该做的是：

## 进入下一步：写 `input_buffer.sv`

因为现在整个 CIM 计算链条里，另一个关键输入已经明确了：

- 权重 tile：有了
- 输入 tile：还没有

所以接下来的顺序应该是：

### 第一步
写 `input_buffer.sv`

### 第二步
写 `tb_input_buffer.sv`

### 第三步
写 `cim_tile.sv`

### 第四步
写 `tb_cim_tile.sv`

### 第五步
写 `psum_accum.sv`

### 第六步
写 `fc1_cim_core.sv`

### 第七步
对拍：
- `fc1_acc`
- `fc1_relu`
- `fc1_out`

---

# 三、下一步为什么必须先做 `input_buffer`

因为第一层的实际乘法不是：

\[
x_q \cdot w_q
\]

而是：

\[
(x_q - z_x)(w_q - z_w)
\]

根据你当前 `route_b_output`：

- `input_zero_point = -128`
- `fc1_weight_zero_point = 0`

所以：

\[
x_{eff} = x_q - (-128) = x_q + 128
\]

这件事必须在硬件里明确实现。  
而最合适放这一步的地方，就是 `input_buffer` 或紧接输入读取的地方。

所以 `input_buffer` 不只是“读输入”，它还承担：

- 从 `input_i.hex` 读取完整 784 维输入
- 根据 `ib` 切出 16 维输入 tile
- 生成：
  - `x_tile`（原始 int8）
  - `x_eff_tile`（零点修正后，建议 9 bit）

这一步对后面和 Python golden 对齐是决定性的。

---

# 四、你现在的主线目标应该是什么

现在你不要把目标定成：

**“把 weight 读得像真实 BRAM 一样优雅”**

而要定成：

**“尽快把 fc1 的一个输出块算出来，并和 `fc1_acc_0.hex` 对上”**

这才是阶段 1 的关键里程碑。

因为一旦你能做到：

- 输入 tile 正确
- 权重 tile 正确
- tile 内乘加正确
- psum 累加正确
- `fc1_acc` 对拍正确

那你就已经可以非常有底气地说：

**第一层 CIM 宏单元的核心计算链路已经建立。**

这比“用不用 BRAM”更重要。

---

# 五、那 BRAM 化什么时候做最合适

不是不做，而是放到 **fc1 功能跑通之后**。

## 正确时机
当你完成下面这些以后，再去优化存储最合适：

- `fc1_weight_bank`
- `input_buffer`
- `cim_tile`
- `psum_accum`
- `fc1_cim_core`
- `fc1_acc / fc1_relu / fc1_out` 全部和 golden 对上

这时候你已经确认：
- 算法对
- 地址对
- tile 切块对
- 控制对

然后再把：

- 大 ROM
- 组合读

替换成：

- bank 化 BRAM
- 同步读
- 1-cycle latency

这时候改动才是“工程优化”，而不是“在不确定的基础上乱动”。

---

# 六、如果你现在硬要优化，本次 weight 读取最值得优化的方向是什么

如果你确实想顺手做一点“更贴近真实 BRAM”的改进，我建议**只做最小优化**，不要大改结构。

---

## 可做的小优化 1：改成同步读接口
你现在大概率是：

- `ob/ib` 改变
- `w_tile` 组合输出立刻变化

真实 BRAM 更接近：

- 地址输入
- 下一拍输出数据

所以你可以先在接口层面模拟这一点：

- 输入 `req_valid`
- 锁存 `ob/ib`
- 下一拍输出 `w_tile_valid`

但内部仍然可以先用数组。  
这样你后面换真 BRAM 更平滑。

---

## 可做的小优化 2：逻辑 bank 化，但底层还是大数组
例如把模块写成“概念上 8 个输出 bank”，但底层仍从 `weight_mem` 取：

```text
w_tile[tr][tc] = weight_mem[(ob*16+tr)*784 + (ib*16+tc)]
```

这样在结构上已经接近 bank 化，但不引入真实 BRAM 复杂性。

---

## 不建议现在做的事
现在先别做：

- 真双口 BRAM 拆分
- 多 always_ff 同步读 pipeline
- 宽总线打包
- 多 tile 并行预取
- 复杂 cache

这些都应该放到功能闭环之后。

---

# 七、我给你的明确建议

## 结论
**进入下一步，不优先优化 weight 预读取。**

## 下一步具体做什么
写：

### `input_buffer.sv`
功能：
- 加载 `input_i.hex`
- 保存 784 维输入
- 根据 `ib` 输出 `x_tile[16]`
- 生成 `x_eff_tile[16] = x_q + 128`

### `tb_input_buffer.sv`
验证：
- tile(0) 是否对应输入 0~15
- tile(1) 是否对应输入 16~31
- tile(48) 是否对应输入 768~783
- `x_eff_tile` 是否等于 `x_q - INPUT_ZERO_POINT`

---

# 八、你现在应该怎么决策

你可以这样定：

## 如果你当前目标是“尽快把 fc1 计算跑通”
那就：
**直接进入下一步，写 `input_buffer.sv`**

## 如果你当前目标是“让设计文档更像真实 CIM 宏”
那可以在文档里写：
- 当前 weight bank 为功能验证版 ROM 实现
- 后续将替换为 bank 化 BRAM / 同步 tile 读取

这样就够了，不一定马上改 RTL。

---

# 九、我建议你现在的实际动作

直接做下面这个选择：

**下一步：写 `input_buffer.sv`，不要先优化 `fc1_weight_bank.sv`。**

因为阶段 1 的主线，是先把：

- weight tile
- input tile
- CIM tile MAC
- psum accumulate

接起来。

---

你现在最适合的下一条，就是让我继续直接给你：
**`input_buffer.sv` + `tb_input_buffer.sv` 的完整代码和验证思路。**

用户:
先不做weight预读取的优化，进入下一步，介绍设计流程并给出样例代码。
我已经给出我现在的代码
AI助手:


AI助手:
我先看一下你当前硬件工程里已经有哪些 RTL / TB 文件，确保我给的“下一步设计流程”和样例代码能直接接在你现有命名和接口风格后面，不会又起一套新体系。

AI助手:
你现在最合理的下一步是：

**进入 `input_buffer` 设计，而不是继续动 `weight_bank`。**

因为当前链路已经有了：

- `package.sv`
- `fc1_weight_bank.sv`

还缺的是：

- 从 `input_i.hex` 读入 784 维输入
- 按 `ib` 切出 `16` 维输入 tile
- 完成输入零点修正，生成 `x_eff_tile`

这一步完成后，你就能继续写：

- `cim_tile.sv`
- `psum_accum.sv`
- `fc1_cim_core.sv`

---

# 一、先说你当前代码里要顺手修的小问题

你现有代码整体方向对，但有两个地方建议顺手改掉，不然等会儿 testbench 会出问题。

## 1. testbench 参数名不一致
你模块里是：

```systemverilog
parameter string WEIGHT_FILE = ...
```

但 testbench 里实例化写的是：

```systemverilog
.WEIGHT_HEX_FILE(...)
```

这里应该改成：

```systemverilog
.WEIGHT_FILE(...)
```

## 2. 参考文件路径不一致
你 DUT 用的是相对路径：

```text
../../CIM-sw-version1/sw/train&quantize/route_b_output/fc1_weight_int8.hex
```

但 testbench 参考内存读的是：

```systemverilog
$readmemh("fc1_weight_int8.hex", ref_weight_mem);
```

最好统一成同一路径。

---

# 二、下一步的设计流程

下面是你现在应该走的顺序。

## 第 1 步：写 `input_buffer.sv`
职责：

- 从 `input_i.hex` 加载完整 784 维 int8 输入
- 根据 `ib` 输出当前 `16` 维输入 tile
- 根据 `INPUT_ZERO_POINT=-128` 生成：
  - `x_tile`：原始 int8 输入
  - `x_eff_tile`：零点修正后输入，范围 `0..255`，建议 9 bit

## 第 2 步：写 `tb_input_buffer.sv`
验证：

- `ib=0` 是否输出输入 0~15
- `ib=1` 是否输出输入 16~31
- `ib=48` 是否输出输入 768~783
- `x_eff_tile = x_tile - INPUT_ZERO_POINT = x_tile + 128` 是否正确

## 第 3 步：再写 `cim_tile.sv`
功能：

- 输入 `x_eff_tile[16]`
- 输入 `w_tile[16][16]`
- 输出 `tile_psum[16]`

## 第 4 步：写 `psum_accum.sv`
功能：

- clear
- accumulate

## 第 5 步：写 `fc1_cim_core.sv`
控制两层循环：

- `ob = 0..7`
- `ib = 0..48`

---

# 三、为什么 `input_buffer` 是下一步

因为你现在的第一层乘法不是直接：

\[
x_q \cdot w_q
\]

而是：

\[
(x_q - z_x)(w_q - z_w)
\]

根据你的量化配置：

- `INPUT_ZERO_POINT = -128`
- `FC1_WEIGHT_ZERO_POINT = 0`

所以：

\[
x_{eff} = x_q - (-128) = x_q + 128
\]

硬件里必须明确实现这一点。  
而最合适的位置，就是 `input_buffer`。

---

# 四、建议的 `input_buffer` 接口

你现在的风格已经比较统一，我建议接口就按这个来：

```systemverilog
module input_buffer #(
    parameter string INPUT_FILE = "..."
) (
    input  logic [$clog2(mnist_cim_pkg::N_INPUT_BLOCKS)-1:0] ib,

    output logic signed [mnist_cim_pkg::INPUT_WIDTH-1:0]
        x_tile [0:mnist_cim_pkg::TILE_INPUT_SIZE-1],

    output logic [mnist_cim_pkg::X_EFF_WIDTH-1:0]
        x_eff_tile [0:mnist_cim_pkg::TILE_INPUT_SIZE-1]
);
```

这里我建议：

- `x_tile` 保持 `signed int8`
- `x_eff_tile` 直接做成**无符号** `9 bit`

因为 `x_q + 128` 的结果范围是：

\[
[-128,127] + 128 = [0,255]
\]

这更符合后面乘法直觉。

---

# 五、样例代码：`input_buffer.sv`

下面这版能直接接你现在的 `package.sv` 风格。

```systemverilog
module input_buffer #(
    parameter string INPUT_FILE = "../../CIM-sw-version1/sw/train&quantize/route_b_output/input_0.hex"
) (
    input  logic [$clog2(mnist_cim_pkg::N_INPUT_BLOCKS)-1:0] ib,

    output logic signed [mnist_cim_pkg::INPUT_WIDTH-1:0]
        x_tile [0:mnist_cim_pkg::TILE_INPUT_SIZE-1],

    output logic [mnist_cim_pkg::X_EFF_WIDTH-1:0]
        x_eff_tile [0:mnist_cim_pkg::TILE_INPUT_SIZE-1]
);

  import mnist_cim_pkg::*;

  // 完整 784 维输入缓存
  logic signed [INPUT_WIDTH-1:0] input_mem [0:INPUT_DIM-1];

  integer tc;
  integer in_idx;
  integer x_eff_tmp;

  initial begin
    $display("Loading input file: %s", INPUT_FILE);
    $readmemh(INPUT_FILE, input_mem);
  end

  always_comb begin
    for (tc = 0; tc < TILE_INPUT_SIZE; tc = tc + 1) begin
      in_idx = ib * TILE_INPUT_SIZE + tc;

      x_tile[tc] = input_mem[in_idx];

      // x_eff = x_q - INPUT_ZERO_POINT
      // 当前 INPUT_ZERO_POINT = -128
      x_eff_tmp = input_mem[in_idx] - INPUT_ZERO_POINT;

      // 限幅到 0~255，避免意外越界
      if (x_eff_tmp < 0)
        x_eff_tile[tc] = '0;
      else if (x_eff_tmp > ((1 << X_EFF_WIDTH) - 1))
        x_eff_tile[tc] = {X_EFF_WIDTH{1'b1}};
      else
        x_eff_tile[tc] = x_eff_tmp[X_EFF_WIDTH-1:0];
    end
  end

endmodule
```

---

# 六、这段代码做了什么

对于当前输入块 `ib`：

- `x_tile[0] = input_mem[ib*16 + 0]`
- `x_tile[1] = input_mem[ib*16 + 1]`
- ...
- `x_tile[15] = input_mem[ib*16 + 15]`

同时：

- `x_eff_tile[k] = x_tile[k] - INPUT_ZERO_POINT`
- 由于 `INPUT_ZERO_POINT=-128`，也就是：
  - `x_eff_tile[k] = x_tile[k] + 128`

这正好和你 `route_b_output` 的推理链路一致。

---

# 七、样例代码：`tb_input_buffer.sv`

这个 testbench 先只验证切块和零点修正。

```systemverilog
`timescale 1ns / 1ps

module tb_input_buffer;
  import mnist_cim_pkg::*;

  logic [$clog2(N_INPUT_BLOCKS)-1:0] ib;

  logic signed [INPUT_WIDTH-1:0] x_tile [0:TILE_INPUT_SIZE-1];
  logic [X_EFF_WIDTH-1:0]        x_eff_tile [0:TILE_INPUT_SIZE-1];

  input_buffer #(
      .INPUT_FILE("../../CIM-sw-version1/sw/train&quantize/route_b_output/input_0.hex")
  ) dut (
      .ib(ib),
      .x_tile(x_tile),
      .x_eff_tile(x_eff_tile)
  );

  logic signed [INPUT_WIDTH-1:0] ref_input_mem [0:INPUT_DIM-1];

  integer tc;
  integer expected_idx;
  integer expected_eff;
  integer error_count;

  initial begin
    $readmemh("../../CIM-sw-version1/sw/train&quantize/route_b_output/input_0.hex",
              ref_input_mem);

    error_count = 0;

    // -------------------------
    // Test 1: block 0
    // -------------------------
    ib = 0;
    #1;

    $display("Checking input block 0...");
    for (tc = 0; tc < TILE_INPUT_SIZE; tc = tc + 1) begin
      expected_idx = ib * TILE_INPUT_SIZE + tc;
      expected_eff = ref_input_mem[expected_idx] - INPUT_ZERO_POINT;

      if (x_tile[tc] !== ref_input_mem[expected_idx]) begin
        $display("ERROR block0 x_tile tc=%0d got=%0d expected=%0d",
                 tc, x_tile[tc], ref_input_mem[expected_idx]);
        error_count = error_count + 1;
      end

      if (x_eff_tile[tc] !== expected_eff[X_EFF_WIDTH-1:0]) begin
        $display("ERROR block0 x_eff tc=%0d got=%0d expected=%0d",
                 tc, x_eff_tile[tc], expected_eff);
        error_count = error_count + 1;
      end
    end

    // -------------------------
    // Test 2: block 1
    // -------------------------
    ib = 1;
    #1;

    $display("Checking input block 1...");
    for (tc = 0; tc < TILE_INPUT_SIZE; tc = tc + 1) begin
      expected_idx = ib * TILE_INPUT_SIZE + tc;
      expected_eff = ref_input_mem[expected_idx] - INPUT_ZERO_POINT;

      if (x_tile[tc] !== ref_input_mem[expected_idx]) begin
        $display("ERROR block1 x_tile tc=%0d got=%0d expected=%0d",
                 tc, x_tile[tc], ref_input_mem[expected_idx]);
        error_count = error_count + 1;
      end

      if (x_eff_tile[tc] !== expected_eff[X_EFF_WIDTH-1:0]) begin
        $display("ERROR block1 x_eff tc=%0d got=%0d expected=%0d",
                 tc, x_eff_tile[tc], expected_eff);
        error_count = error_count + 1;
      end
    end

    // -------------------------
    // Test 3: last block
    // -------------------------
    ib = N_INPUT_BLOCKS - 1;
    #1;

    $display("Checking last input block...");
    for (tc = 0; tc < TILE_INPUT_SIZE; tc = tc + 1) begin
      expected_idx = ib * TILE_INPUT_SIZE + tc;
      expected_eff = ref_input_mem[expected_idx] - INPUT_ZERO_POINT;

      if (x_tile[tc] !== ref_input_mem[expected_idx]) begin
        $display("ERROR last block x_tile tc=%0d got=%0d expected=%0d",
                 tc, x_tile[tc], ref_input_mem[expected_idx]);
        error_count = error_count + 1;
      end

      if (x_eff_tile[tc] !== expected_eff[X_EFF_WIDTH-1:0]) begin
        $display("ERROR last block x_eff tc=%0d got=%0d expected=%0d",
                 tc, x_eff_tile[tc], expected_eff);
        error_count = error_count + 1;
      end
    end

    if (error_count == 0)
      $display("PASS: input_buffer block extraction is correct.");
    else
      $display("FAIL: found %0d mismatches.", error_count);

    $finish;
  end

endmodule
```

---

# 八、编译脚本示例

你现在已经有 `run_fc1_weight_bank.sh` 风格了，那 `input_buffer` 也可以这样：

## `run_input_buffer.sh`

```bash
#!/usr/bin/env bash
set -e

mkdir -p ../sim

iverilog -g2012 -o ../sim/input_buffer_simv \
    ../rtl/package.sv \
    ../rtl/input_buffer.sv \
    ../tb/tb_input_buffer.sv

vvp ../sim/input_buffer_simv
```

---

# 九、这一阶段完成后，你就能做什么

只要 `input_buffer` 跑通，下一步就非常顺了：

## 下一步 1：写 `cim_tile.sv`
输入：
- `x_eff_tile[16]`
- `w_tile[16][16]`

输出：
- `tile_psum[16]`

这一步就是：

\[
tile\_psum[tr] = \sum_{tc=0}^{15} x_{eff}[tc] \cdot w[tr][tc]
\]

## 下一步 2：写 `tb_cim_tile.sv`
先手工构造一个小例子验证乘加结果。

## 下一步 3：写 `psum_accum.sv`
功能：
- clear
- en
- `psum += tile_psum`

---

# 十、你现在这一步的目标

你现在不要想着“一口气做完整 fc1”。  
当前阶段目标应该很具体：

**确认输入 tile 提取和 zero-point 修正完全正确。**

只要这一步对了，后面的 `cim_tile` 就只是在做确定性的乘加，不容易乱。

---

# 十一、我建议你现在的实际动作

你就按这个顺序落地：

1. 修一下 `tb_fc1_weight_bank.sv` 的参数名和路径
2. 新建 `rtl/input_buffer.sv`
3. 新建 `tb/tb_input_buffer.sv`
4. 跑 `run_input_buffer.sh`
5. 确认通过后，再进入 `cim_tile.sv`

如果你把你写出来的 `input_buffer.sv` 发给我，我可以直接继续帮你审并带你进入下一步 `cim_tile.sv`。

用户:
我都改成WEIGHT_HEX_FILE了，参考文件路径不一致我也已经完成了修改。我现想要把运行脚本改成iverilog版本，给出一个新的脚本
AI助手:


AI助手:
可以，下面给你一版新的 `iverilog` 运行脚本，适合你现在这种：

- RTL 和 TB 分开
- 使用 `package.sv`
- 参数名已经统一成 `WEIGHT_HEX_FILE`
- 用 `iverilog + vvp` 跑仿真

我给你三版，你按需要选。

---

# 方案 1：最直接可用版

适合你现在先跑 `fc1_weight_bank`。

## `run_fc1_weight_bank.sh`

```bash
#!/usr/bin/env bash
set -e

SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

mkdir -p ${SIM_DIR}

iverilog -g2012 -o ${SIM_DIR}/fc1_weight_bank_simv \
    ${RTL_DIR}/package.sv \
    ${RTL_DIR}/fc1_weight_bank.sv \
    ${TB_DIR}/tb_fc1_weight_bank.sv

vvp ${SIM_DIR}/fc1_weight_bank_simv
```

---

# 方案 2：带波形输出提示版

如果你后面想看波形，这版更方便。  
前提是你的 testbench 里加上：

```systemverilog
initial begin
  $dumpfile("../sim/tb_fc1_weight_bank.vcd");
  $dumpvars(0, tb_fc1_weight_bank);
end
```

## `run_fc1_weight_bank.sh`

```bash
#!/usr/bin/env bash
set -e

SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

OUT_EXE=${SIM_DIR}/fc1_weight_bank_simv

mkdir -p ${SIM_DIR}

echo "[INFO] Compiling with iverilog..."
iverilog -g2012 -o ${OUT_EXE} \
    ${RTL_DIR}/package.sv \
    ${RTL_DIR}/fc1_weight_bank.sv \
    ${TB_DIR}/tb_fc1_weight_bank.sv

echo "[INFO] Running simulation..."
vvp ${OUT_EXE}

echo "[INFO] Done."
echo "[INFO] If VCD is enabled in testbench, waveform file is in ${SIM_DIR}/"
```

---

# 方案 3：更通用的模板版

这个更适合你后面继续复用到：

- `input_buffer`
- `cim_tile`
- `fc1_cim_core`

## `run_sim.sh`

```bash
#!/usr/bin/env bash
set -e

if [ $# -lt 2 ]; then
  echo "Usage: $0 <top_name> <tb_file> [rtl_files...]"
  exit 1
fi

TOP_NAME=$1
TB_FILE=$2
shift 2

SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

OUT_EXE=${SIM_DIR}/${TOP_NAME}_simv

mkdir -p ${SIM_DIR}

echo "[INFO] Compiling ${TOP_NAME} ..."
iverilog -g2012 -o ${OUT_EXE} \
    ${RTL_DIR}/package.sv \
    "$@" \
    ${TB_DIR}/${TB_FILE}

echo "[INFO] Running ${TOP_NAME} ..."
vvp ${OUT_EXE}
```

### 用法示例

跑 `fc1_weight_bank`：

```bash
bash run_sim.sh fc1_weight_bank tb_fc1_weight_bank.sv ../rtl/fc1_weight_bank.sv
```

跑 `input_buffer`：

```bash
bash run_sim.sh input_buffer tb_input_buffer.sv ../rtl/input_buffer.sv
```

---

# 你当前最推荐用哪版

你现在阶段最适合的是：

**方案 2**

因为它：
- 简单
- 清晰
- 后面可以顺手接波形

---

# 还要注意两点

## 1. `-g2012` 必须保留
因为你用了：
- `package`
- `logic`
- `parameter string`
- 二维数组端口

这些都需要 SystemVerilog 模式。

---

## 2. 相对路径是相对“脚本执行目录”
如果你在 `scripts/` 目录下执行：

```bash
bash run_fc1_weight_bank.sh
```

那脚本里的：

```bash
../rtl/package.sv
```

就是相对 `scripts/` 来找的。

所以要保证你的目录结构真的是：

```text
project/
├── rtl/
├── tb/
├── sim/
└── scripts/
    └── run_fc1_weight_bank.sh
```

---

# 我再给你一版适配 `WEIGHT_HEX_FILE` 的 testbench 头部示例

如果你现在 testbench 已经改成参数名 `WEIGHT_HEX_FILE`，那实例化应该类似这样：

```systemverilog
fc1_weight_bank #(
    .WEIGHT_HEX_FILE("../../CIM-sw-version1/sw/train&quantize/route_b_output/fc1_weight_int8.hex")
) dut (
    .ob(ob),
    .ib(ib),
    .w_tile(w_tile)
);
```

那脚本本身不需要额外改，只要编译这几个文件就行。

---

如果你愿意，我下一步可以直接给你一版**完整可运行的 `tb_fc1_weight_bank.sv + run_fc1_weight_bank.sh` 配套版本**，按你现在 `WEIGHT_HEX_FILE` 的命名完全统一。

用户:
                         Chronologic VCS (TM)
       Version O-2018.09-SP2_Full64 -- Sat Mar  7 13:15:52 2026
               Copyright (c) 1991-2018 by Synopsys Inc.
                         ALL RIGHTS RESERVED

This program is proprietary and confidential information of Synopsys Inc.
and may be used and disclosed only as authorized in a license agreement
controlling such use and disclosure.

Parsing design file '../rtl/package.sv'
Parsing design file '../rtl/fc1_weight_bank.sv'
Parsing design file '../tb/tb_fc1_weight_bank.sv'

Error-[SE] Syntax error
  Following verilog source has syntax error :
  "../tb/tb_fc1_weight_bank.sv", 10: token is 'logic'
    logic signed [WEIGHT_WIDTH-1:0] 
  w_tile[0:TILE_OUTPUT_SIZE-1][0:TILE_INPUT_SIZE-1];
         ^
  SystemVerilog  keyword 'logic' is not expected to be used in this context.

1 error
CPU time: .082 seconds to compile
AI助手:


AI助手:
这个报错的根因不是 `w_tile` 本身，而是：

## 根因
**VCS 现在没有按 SystemVerilog 模式编译 `tb_fc1_weight_bank.sv`。**

所以它把这句：

```systemverilog
logic signed [WEIGHT_WIDTH-1:0] w_tile [0:TILE_OUTPUT_SIZE-1][0:TILE_INPUT_SIZE-1];
```

当成 Verilog 去解析，于是看到 `logic` 就报错了。

---

# 一、怎么改

你现在既然用的是 **VCS**，脚本里必须显式加：

```bash
-sverilog
```

如果没有这个选项，`.sv` 文件在一些环境下也可能没被正确当成 SystemVerilog 处理。

---

# 二、给你一版新的 VCS 运行脚本

## `run_fc1_weight_bank.sh`

```bash
#!/usr/bin/env bash
set -e

SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

mkdir -p ${SIM_DIR}

vcs -full64 -sverilog -debug_access+all \
    -o ${SIM_DIR}/fc1_weight_bank_simv \
    ${RTL_DIR}/package.sv \
    ${RTL_DIR}/fc1_weight_bank.sv \
    ${TB_DIR}/tb_fc1_weight_bank.sv

${SIM_DIR}/fc1_weight_bank_simv
```

---

# 三、如果你还想生成波形

可以用这版：

```bash
#!/usr/bin/env bash
set -e

SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

mkdir -p ${SIM_DIR}

vcs -full64 -sverilog -debug_access+all \
    -o ${SIM_DIR}/fc1_weight_bank_simv \
    ${RTL_DIR}/package.sv \
    ${RTL_DIR}/fc1_weight_bank.sv \
    ${TB_DIR}/tb_fc1_weight_bank.sv

${SIM_DIR}/fc1_weight_bank_simv -l ${SIM_DIR}/fc1_weight_bank.log
```

如果 testbench 里加了：

```systemverilog
initial begin
  $dumpfile("../sim/tb_fc1_weight_bank.vcd");
  $dumpvars(0, tb_fc1_weight_bank);
end
```

就会生成 VCD。

---

# 四、你的 testbench 代码本身大概率没问题

这句：

```systemverilog
logic signed [WEIGHT_WIDTH-1:0] w_tile [0:TILE_OUTPUT_SIZE-1][0:TILE_INPUT_SIZE-1];
```

在 SystemVerilog 里是合法的。  
报错说：

```text
SystemVerilog keyword 'logic' is not expected to be used in this context
```

这几乎就是在明确告诉你：

**当前编译器把它当 Verilog 了。**

---

# 五、再给你一个更稳的写法

虽然不是必须，但为了可读性，你也可以写成：

```systemverilog
logic signed [WEIGHT_WIDTH-1:0] 
    w_tile [0:TILE_OUTPUT_SIZE-1][0:TILE_INPUT_SIZE-1];
```

或者：

```systemverilog
logic signed [WEIGHT_WIDTH-1:0] w_tile [0:TILE_OUTPUT_SIZE-1][0:TILE_INPUT_SIZE-1];
```

都可以。  
关键不是排版，关键是 **`-sverilog`**。

---

# 六、顺手检查两件事

## 1. 文件后缀
你现在已经是：

- `package.sv`
- `fc1_weight_bank.sv`
- `tb_fc1_weight_bank.sv`

这个没问题。

## 2. `package` 导入
testbench 里要有：

```systemverilog
import mnist_cim_pkg::*;
```

这个你之前已经有了，保持就行。

---

# 七、如果加了 `-sverilog` 还报错

那我下一步会优先怀疑两件事：

## 情况 1：脚本实际没调用到你改过的命令
比如你改的是一个脚本，但实际运行的是另一个旧脚本。

## 情况 2：testbench 文件前面有隐藏字符或语法残留
例如：
- 中文全角符号
- 错误的反引号
- 上一行没结束

但从你给的报错看，当前最主要的问题还是 **没开 SystemVerilog 模式**。

---

# 八、你现在应该怎么做

直接把脚本改成我上面那版，然后重新跑。

如果还有报错，把这两样贴给我：

1. 你当前的 `run_fc1_weight_bank.sh`
2. `tb_fc1_weight_bank.sv` 前 20 行

我就能继续往下精确定位。

用户:
我已经加上-sverilog了
AI助手:


AI助手:
如果已经加了 `-sverilog`，那问题就不再是“没开 SystemVerilog 模式”这么简单了。接下来最值得怀疑的是 **testbench 前几行本身的语法上下文有问题**，导致编译器在读到 `logic` 时已经不在合法声明区了。

AI助手:
如果你已经加了 `-sverilog`，那现在最可能不是工具开关问题，而是 **`tb_fc1_weight_bank.sv` 在第 10 行之前已经把语法环境破坏了**，所以编译器读到 `logic` 时才报这个错。

这个报错最常见的真实原因有 4 类，我按概率从高到低给你排：

---

# 1. `module` / `import` / 上一行分号有问题

最典型的是前几行像这样有小错误：

```systemverilog
module tb_fc1_weight_bank
  import mnist_cim_pkg::*;
  logic ...
```

这里 `module tb_fc1_weight_bank` 后面少了分号 `;`，  
那么编译器会在下一行、再下一行开始一连串报奇怪错误，最后在 `logic` 这里炸掉。

正确写法必须是：

```systemverilog
`timescale 1ns / 1ps

module tb_fc1_weight_bank;
  import mnist_cim_pkg::*;

  logic [$clog2(N_OUTPUT_BLOCKS)-1:0] ob;
  logic [$clog2(N_INPUT_BLOCKS)-1:0]  ib;

  logic signed [WEIGHT_WIDTH-1:0]
      w_tile [0:TILE_OUTPUT_SIZE-1][0:TILE_INPUT_SIZE-1];

  ...
endmodule
```

你先重点检查：

- `module tb_fc1_weight_bank;` 这行末尾有没有 `;`
- `import mnist_cim_pkg::*;` 这行末尾有没有 `;`
- `timescale` 行有没有奇怪字符

---

# 2. `import` 放的位置不对，或者 package 名不一致

你 `package.sv` 里定义的是：

```systemverilog
package mnist_cim_pkg;
```

那么 testbench 里必须是：

```systemverilog
import mnist_cim_pkg::*;
```

不能写成别的名字。

而且建议放在 module 体内最前面：

```systemverilog
module tb_fc1_weight_bank;
  import mnist_cim_pkg::*;
```

不要先声明信号再 import。

---

# 3. 你的 testbench 前面有隐藏字符 / 中文标点 / 拷贝污染

这个问题很常见，尤其是从聊天窗口或文档里复制代码后。

比如这些都会导致很诡异的 syntax error：

- 中文分号 `；`
- 中文逗号 `，`
- 全角空格
- 非法反引号
- `logic` 前面有不可见字符

最容易排查的方法：

### 方法 A
手动把 `tb_fc1_weight_bank.sv` 前 15 行全部删掉重敲，不要复制粘贴。

### 方法 B
在 Linux 里看不可见字符：

```bash
sed -n '1,20p' ../tb/tb_fc1_weight_bank.sv | cat -A
```

如果看到奇怪字符，说明文件里混了不可见内容。

---

# 4. VCS 实际执行的命令不是你以为的那个

你说“已经加上 `-sverilog` 了”，但我还是建议确认一下**真正跑起来的命令**。

你脚本里最好先打印命令，或者直接把脚本内容检查一遍。  
一个稳妥版本应该像这样：

```bash
#!/usr/bin/env bash
set -e

SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

mkdir -p ${SIM_DIR}

vcs -full64 -sverilog -debug_access+all \
    -o ${SIM_DIR}/fc1_weight_bank_simv \
    ${RTL_DIR}/package.sv \
    ${RTL_DIR}/fc1_weight_bank.sv \
    ${TB_DIR}/tb_fc1_weight_bank.sv

${SIM_DIR}/fc1_weight_bank_simv
```

---

# 我建议你现在直接这样改

把 `tb_fc1_weight_bank.sv` 前面改成下面这个最小可用版本，先排除语法环境问题：

```systemverilog
`timescale 1ns/1ps

module tb_fc1_weight_bank;
  import mnist_cim_pkg::*;

  logic [$clog2(N_OUTPUT_BLOCKS)-1:0] ob;
  logic [$clog2(N_INPUT_BLOCKS)-1:0]  ib;

  logic signed [WEIGHT_WIDTH-1:0] w_tile [0:TILE_OUTPUT_SIZE-1][0:TILE_INPUT_SIZE-1];

  fc1_weight_bank #(
      .WEIGHT_HEX_FILE("../../CIM-sw-version1/sw/train&quantize/route_b_output/fc1_weight_int8.hex")
  ) dut (
      .ob(ob),
      .ib(ib),
      .w_tile(w_tile)
  );

endmodule
```

如果这份最小版都还在 `logic` 这里报错，那基本只剩两种可能：

1. `tb_fc1_weight_bank.sv` 文件里有隐藏字符  
2. 你实际跑的不是这份文件 / 不是这个脚本

---

# 再给你一个更稳的兼容写法

有时候为了减少工具在二维数组声明上的歧义，你可以先把前两行声明拆开：

```systemverilog
logic [$clog2(N_OUTPUT_BLOCKS)-1:0] ob;
logic [$clog2(N_INPUT_BLOCKS)-1:0]  ib;

logic signed [WEIGHT_WIDTH-1:0] w_tile [0:TILE_OUTPUT_SIZE-1][0:TILE_INPUT_SIZE-1];
```

不要断行，不要夹注释，先用一行完整声明。

---

# 现在最有效的排查顺序

1. 确认 `module tb_fc1_weight_bank;` 末尾有分号  
2. 确认 `import mnist_cim_pkg::*;` 末尾有分号  
3. 把 testbench 前 15 行手动重敲一遍  
4. 用我上面的“最小版 tb”替换后再编一次

把 `tb_fc1_weight_bank.sv` 前 20 行贴出来，我可以直接帮你把具体错误位置指出来。

用户:
我一会上传当前代码，帮我设计下一步的inputbuffer
AI助手:


AI助手:
好，等你上传。  
我会基于你现有工程风格，直接帮你把下一步的 `input_buffer` 设计出来，包括：

- 模块职责和接口定义
- 与你当前 `package.sv`、`fc1_weight_bank.sv` 的衔接方式
- `input_i.hex` 的加载方式
- `x_tile / x_eff_tile` 的输出格式
- 配套 `tb_input_buffer.sv`
- 编译运行脚本建议

你上传后我就按你现在的代码体系继续接，不另起一套。

用户:

AI助手:


AI助手:
我先快速检查你刚上传的硬件工程目录，确认现有 RTL/TB 命名、端口风格和脚本组织，然后直接按这个风格给你续上 `input_buffer`，避免接口又不一致。

AI助手:
我看过你当前工程了，风格已经比较统一，适合直接往下接。

你现在已有：

- `rtl/package.sv`
- `rtl/fc1_weight_bank.sv`
- `tb/tb_fc1_weight_bank.sv`
- `scripts/run_fc1_weight_bank.sh`

所以**下一步就按同样风格设计 `input_buffer`**，不要再换命名体系。

---

# 一、下一步要做什么

现在进入：

## `input_buffer` 设计

它的职责不是简单“读输入”，而是完成这三件事：

1. 从 `input_0.hex` 加载完整 784 维输入  
2. 按当前输入块索引 `ib` 提取一个 `16` 维 tile  
3. 根据量化参数做输入零点修正，生成 `x_eff_tile`

---

# 二、为什么这一步必须先做

你后面 `cim_tile` 里真正要算的不是：

\[
x_q \cdot w_q
\]

而是：

\[
(x_q - z_x)(w_q - z_w)
\]

根据你当前 package：

- `INPUT_ZERO_POINT = -128`
- `FC1_WEIGHT_ZERO_POINT = 0`

所以实际进入乘法阵列的输入应该是：

\[
x_{eff} = x_q - (-128) = x_q + 128
\]

这个结果范围是：

\[
[-128,127] + 128 = [0,255]
\]

所以你 package 里加的：

```systemverilog
parameter int X_EFF_WIDTH = 9;
```

是对的。  
这说明你现在的下一步就是把这个设计真正落成 RTL。

---

# 三、建议的设计流程

你现在按下面顺序做最稳：

## 第 1 步：写 `rtl/input_buffer.sv`
实现：

- 文件加载
- tile 切分
- `x_tile`
- `x_eff_tile`

## 第 2 步：写 `tb/tb_input_buffer.sv`
验证：

- `ib=0` 对应输入 0~15
- `ib=1` 对应输入 16~31
- `ib=48` 对应输入 768~783
- `x_eff_tile = x_tile - INPUT_ZERO_POINT`

## 第 3 步：写 `scripts/run_input_buffer.sh`
风格对齐你现有 `run_fc1_weight_bank.sh`

## 第 4 步：仿真确认通过
只有这一步过了，才能继续写：

- `cim_tile.sv`
- `psum_accum.sv`
- `fc1_cim_core.sv`

---

# 四、接口应该怎么定

按你当前风格，我建议 `input_buffer` 的接口这样定：

```systemverilog
module input_buffer #(
    parameter string INPUT_HEX_FILE = "..."
) (
    input  logic [$clog2(mnist_cim_pkg::N_INPUT_BLOCKS)-1:0] ib,

    output logic signed [mnist_cim_pkg::INPUT_WIDTH-1:0]
        x_tile [0:mnist_cim_pkg::TILE_INPUT_SIZE-1],

    output logic [mnist_cim_pkg::X_EFF_WIDTH-1:0]
        x_eff_tile [0:mnist_cim_pkg::TILE_INPUT_SIZE-1]
);
```

这里我建议：

- `x_tile` 保持 `signed`
- `x_eff_tile` 用**无符号 9 bit**

因为它表示的是 `0~255`。

---

# 五、样例代码：`rtl/input_buffer.sv`

下面这版是按你当前工程风格直接接的。

```systemverilog
module input_buffer #(
    parameter string INPUT_HEX_FILE = "../../CIM-sw-version1/sw/train&quantize/route_b_output/input_0.hex"
) (
    input  logic [$clog2(mnist_cim_pkg::N_INPUT_BLOCKS)-1:0] ib,

    output logic signed [mnist_cim_pkg::INPUT_WIDTH-1:0]
        x_tile [0:mnist_cim_pkg::TILE_INPUT_SIZE-1],

    output logic [mnist_cim_pkg::X_EFF_WIDTH-1:0]
        x_eff_tile [0:mnist_cim_pkg::TILE_INPUT_SIZE-1]
);

  import mnist_cim_pkg::*;

  // 完整输入缓存：784 x int8
  logic signed [INPUT_WIDTH-1:0] input_mem [0:INPUT_DIM-1];

  integer tc;
  integer in_idx;
  integer x_eff_tmp;

  initial begin
    $readmemh(INPUT_HEX_FILE, input_mem);
  end

  always_comb begin
    for (tc = 0; tc < TILE_INPUT_SIZE; tc = tc + 1) begin
      in_idx = ib * TILE_INPUT_SIZE + tc;

      // 原始 int8 输入
      x_tile[tc] = input_mem[in_idx];

      // 零点修正: x_eff = x_q - INPUT_ZERO_POINT
      // 这里 INPUT_ZERO_POINT = -128，所以等价于 x_q + 128
      x_eff_tmp = input_mem[in_idx] - INPUT_ZERO_POINT;

      // 防御性限幅到 0~255
      if (x_eff_tmp < 0)
        x_eff_tile[tc] = '0;
      else if (x_eff_tmp > ((1 << X_EFF_WIDTH) - 1))
        x_eff_tile[tc] = {X_EFF_WIDTH{1'b1}};
      else
        x_eff_tile[tc] = x_eff_tmp[X_EFF_WIDTH-1:0];
    end
  end

endmodule
```

---

# 六、这段代码在干什么

假设当前：

```systemverilog
ib = 2;
```

那么它会输出：

- `x_tile[0] = input_mem[32]`
- `x_tile[1] = input_mem[33]`
- ...
- `x_tile[15] = input_mem[47]`

同时：

- `x_eff_tile[k] = x_tile[k] - INPUT_ZERO_POINT`
- 因为 `INPUT_ZERO_POINT = -128`
- 所以：
  - `x_eff_tile[k] = x_tile[k] + 128`

这就能直接送进后面的 `cim_tile`。

---

# 七、样例代码：`tb/tb_input_buffer.sv`

这个 testbench 风格和你现在的 `tb_fc1_weight_bank.sv` 一致。

```systemverilog
`timescale 1ns / 1ps

module tb_input_buffer;
  import mnist_cim_pkg::*;

  logic [$clog2(N_INPUT_BLOCKS)-1:0] ib;

  parameter string INPUT_HEX_FILE = "../../CIM-sw-version1/sw/train&quantize/route_b_output/input_0.hex";

  logic signed [INPUT_WIDTH-1:0] x_tile [0:TILE_INPUT_SIZE-1];
  logic [X_EFF_WIDTH-1:0]        x_eff_tile [0:TILE_INPUT_SIZE-1];

  input_buffer #(
      .INPUT_HEX_FILE("../../CIM-sw-version1/sw/train&quantize/route_b_output/input_0.hex")
  ) dut (
      .ib(ib),
      .x_tile(x_tile),
      .x_eff_tile(x_eff_tile)
  );

  logic signed [INPUT_WIDTH-1:0] ref_input_mem [0:INPUT_DIM-1];

  integer tc;
  integer expected_idx;
  integer expected_eff;
  integer error_count;

  initial begin
    $readmemh(INPUT_HEX_FILE, ref_input_mem);

    error_count = 0;

    // -------------------------
    // Test 1: block 0
    // -------------------------
    ib = 0;
    #1;

    $display("Checking input block 0...");
    for (tc = 0; tc < TILE_INPUT_SIZE; tc = tc + 1) begin
      expected_idx = ib * TILE_INPUT_SIZE + tc;
      expected_eff = ref_input_mem[expected_idx] - INPUT_ZERO_POINT;

      if (x_tile[tc] !== ref_input_mem[expected_idx]) begin
        $display("ERROR block0 x_tile tc=%0d got=%0d expected=%0d",
                 tc, x_tile[tc], ref_input_mem[expected_idx]);
        error_count = error_count + 1;
      end

      if (x_eff_tile[tc] !== expected_eff[X_EFF_WIDTH-1:0]) begin
        $display("ERROR block0 x_eff tc=%0d got=%0d expected=%0d",
                 tc, x_eff_tile[tc], expected_eff);
        error_count = error_count + 1;
      end
    end

    // -------------------------
    // Test 2: block 1
    // -------------------------
    ib = 1;
    #1;

    $display("Checking input block 1...");
    for (tc = 0; tc < TILE_INPUT_SIZE; tc = tc + 1) begin
      expected_idx = ib * TILE_INPUT_SIZE + tc;
      expected_eff = ref_input_mem[expected_idx] - INPUT_ZERO_POINT;

      if (x_tile[tc] !== ref_input_mem[expected_idx]) begin
        $display("ERROR block1 x_tile tc=%0d got=%0d expected=%0d",
                 tc, x_tile[tc], ref_input_mem[expected_idx]);
        error_count = error_count + 1;
      end

      if (x_eff_tile[tc] !== expected_eff[X_EFF_WIDTH-1:0]) begin
        $display("ERROR block1 x_eff tc=%0d got=%0d expected=%0d",
                 tc, x_eff_tile[tc], expected_eff);
        error_count = error_count + 1;
      end
    end

    // -------------------------
    // Test 3: last block
    // -------------------------
    ib = N_INPUT_BLOCKS - 1;
    #1;

    $display("Checking last input block...");
    for (tc = 0; tc < TILE_INPUT_SIZE; tc = tc + 1) begin
      expected_idx = ib * TILE_INPUT_SIZE + tc;
      expected_eff = ref_input_mem[expected_idx] - INPUT_ZERO_POINT;

      if (x_tile[tc] !== ref_input_mem[expected_idx]) begin
        $display("ERROR last block x_tile tc=%0d got=%0d expected=%0d",
                 tc, x_tile[tc], ref_input_mem[expected_idx]);
        error_count = error_count + 1;
      end

      if (x_eff_tile[tc] !== expected_eff[X_EFF_WIDTH-1:0]) begin
        $display("ERROR last block x_eff tc=%0d got=%0d expected=%0d",
                 tc, x_eff_tile[tc], expected_eff);
        error_count = error_count + 1;
      end
    end

    if (error_count == 0)
      $display("PASS: input_buffer block extraction is correct.");
    else
      $display("FAIL: found %0d mismatches.", error_count);

    $finish;
  end

endmodule
```

---

# 八、脚本示例：`scripts/run_input_buffer.sh`

按你当前 `vcs` 风格，建议这样写：

```bash
SIM_DIR=../sim
RTL_DIR=../rtl
TB_DIR=../tb

mkdir -p ${SIM_DIR}
mkdir -p ${SIM_DIR}/log

vcs -full64 -sverilog -timescale=1ns/1ps \
    -debug_access+all \
    -o ${SIM_DIR}/input_buffer_simv \
    ${RTL_DIR}/package.sv \
    ${RTL_DIR}/input_buffer.sv \
    ${TB_DIR}/tb_input_buffer.sv \
    2>&1 | tee ${SIM_DIR}/log/compile_tb_input_buffer.log

${SIM_DIR}/input_buffer_simv 2>&1 | tee ${SIM_DIR}/sim_tb_input_buffer.log
```

---

# 九、你这一步完成后，下一步是什么

一旦 `input_buffer` 对拍通过，下一步就不是优化输入缓存，而是：

## 写 `cim_tile.sv`

接口大致会是：

```systemverilog
input  logic [X_EFF_WIDTH-1:0] x_eff_tile [0:15];
input  logic signed [WEIGHT_WIDTH-1:0] w_tile [0:15][0:15];
output logic signed [PSUM_WIDTH-1:0] tile_psum [0:15];
```

功能就是：

\[
tile\_psum[tr] = \sum_{tc=0}^{15} x_{eff}[tc] \cdot w[tr][tc]
\]

这一步会真正把：

- `fc1_weight_bank`
- `input_buffer`

接到一起。

---

# 十、你现在这一步的工程目标

你当前不需要追求“整个网络能跑”。  
你这一步的目标应该非常具体：

**确认输入 tile 的提取和零点修正完全正确。**

因为只要这一步错了，后面的：

- `cim_tile`
- `psum_accum`
- `fc1_acc`

全部都会错，而且很难排查。

---

# 十一、基于你当前工程，我再给两个小建议

## 建议 1：命名统一
你现在 `fc1_weight_bank.sv` 用的是：

```systemverilog
parameter string WEIGHT_HEX_FILE
```

那 `input_buffer.sv` 最好也用：

```systemverilog
parameter string INPUT_HEX_FILE
```

这样统一。

## 建议 2：`package.sv` 名字后面可以考虑统一成 `mnist_cim_pkg.sv`
不是必须，但以后模块多了会更清晰。

---

你现在可以直接把上面的 `input_buffer.sv / tb_input_buffer.sv / run_input_buffer.sh` 放进工程。  
如果你愿意，我下一步就继续直接按你当前体系，给你把 **`cim_tile.sv` 和 `tb_cim_tile.sv`** 也接着设计出来。


