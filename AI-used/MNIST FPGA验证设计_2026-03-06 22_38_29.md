### MNIST FPGA验证设计

AI助手:


用户:
项目设计”课程教学大纲项目设计题目：存算一体AI芯片的FPGA验证系统 指导教师：梁 峰1.目的和意义通过本项目设计提升学生的创新水平和实践动手能力，建立初步的工程意识，促进专业知识和工程实践的有机融合。提高学生的发现问题、分析问题和解决问题的能力。培养学生的团队精神。2.项目参与学生人数：3人3.作品的基本功能使用硬件描述语言在FPGA平台上，设计并实现一款包含存算一体（Computing-in-Memory, CIM）宏单元的SoC系统。该系统能够通过软件驱动，控制存算一体阵列完成如多层感知机（MLP）、卷积神经网络（CNN）等几种典型神经网络的推理运算，并验证其功能正确性与性能提升。4.内容简介本项目是面向前沿计算架构的综合性硬件系统设计实践。项目核心在于构建一个基于FPGA的存算一体AI芯片功能验证平台，以解决传统冯·诺依曼架构在AI计算中面临的“内存墙”瓶颈问题。项目内容主要涵盖以下四个层面：存算一体核心阵列设计： 作为项目的核心创新点，学生将设计一个模拟存算一体行为的硬件宏单元。该设计可能采用数字型实现方式（如利用FPGA内部的BRAM资源构建内存阵列，并设计周边的并行乘加计算逻辑），实现多位精度的矩阵向量乘加运算。重点在于设计其控制接口、数据流和计算时序。片上系统（SoC）集成： 采用软硬协同设计思想，构建一个完整的SoC。该系统包含：软核处理器用于运行控制软件，调度任务；自定义的存算一体阵列IP核，作为系统的协处理器或加速器；AXI等片上互联总线，用于连接处理器、存算阵列、外部内存控制器（DDR）和外围设备，实现高效的数据通信；必要的存储与外设接口，用于存储网络模型参数、输入输出数据以及与上位机进行通信（如UART, Ethernet）；开发运行在SoC软核上的轻量级嵌入式软件，该软件负责从上位机接收神经网络模型（权重）和输入数据，将训练好的权重数据部署到存算一体阵列的存储空间中，将输入数据调度至存算阵列进行计算，控制存算阵列的启动、计算和完成中断，读取计算结果并向上位机返回；最终在FPGA开发板上进行系统级调试和功能验证，选择经典的神经网络模型测试其推理功能的正确性，并通过与传统CPU/GPU或在FPGA上实现的传统加速器进行对比，定量分析存算一体架构在能效和算力密度上的潜在优势，完成
验证报告。通过本项目的完整训练，学生将亲身经历从架构设计、RTL编码、功能仿真、SoC集成、软硬件联调再到系统测试的芯片前端全流程，极大地提升在集成电路、人工智能和嵌入式系统领域的综合工程能力。实验室为学生提供FPGA硬件平台及相关外设。学生可充分发挥自己的想象力和创造力，应用该项目平台实现自己的创新思想，验证和巩固专业课程中学习的有关知识。5.基本要求要求学生在理解存算一体和神经网络原理的基础上，熟悉压缩及解压缩电路的硬件架构，完成电路设计和实现：1)要求学生根据提供的实验设备和实验条件，确认设计目标，制订出为实现该目标所需的可行性报告、设计方案、设计步骤和工作进度表。2)选用合适的元件、外设和芯片，选择开发工具和开发环境，并划分工作模块。在熟悉存算一体的硬件架构的基础上，对电路进行设计。对电路的访存结构进行设计和优化，使其与计算单元的性能相匹配。3)教师对学生提出的设计目标、设计方案和设计步骤进行审核和指导。4)项目完成后，要求学生提交总结报告。根据项目设计作品完成的实际情况、难易程度和经济性评价，进行验收和考核。按5级制给出综合评价。5)对于第一次作品演示失败的小组，允许在3周内再申请一次考核。6.评分标准项目设计课最终得分由导师评价A、项目报告B、答辩或作品演示C等3部分得分构成。其中导师评价占20%，项目报告占20%，答辩或作品演示占60%。各评分环节对课程目标的支撑如下表所示。课程目标导师评价项目报告答辩或作品演示达成度计算课程目标1A*0.25B*0.20(A*0.25+B*0.2)/45课程目标2A*0.3B*0.4C*0.2(A*0.3+B*0.4+C*0.2)/90课程目标3A*0.15B*0.2C*0.1(A*0.15+B*0.2+C*0.1)/45课程目标4A*0.2B*0.2C*0.3(A*0.2+B*0.2+C*0.3)/70课程目标5A*0.100C*0.4(A*0.1+C*0.4)/50课程总体目标ABC(A*0.2+B*0.2+C\*0.6)/100

我们预计写一个简单的mnist，int8计算，能不能先写一个极简版，然后不断添加修改最后完成要求？这个极简版如何设计?请给出极简版的设计架构，以及为了完成设计我们需要完成的所有步骤
用户:
用户在询问一个FPGA存算一体AI芯片验证系统的项目设计，目标是实现MNIST手写数字识别，使用INT8量化计算。他们希望采用迭代开发的方式，先实现一个极简版本，然后逐步完善。

让我分析一下项目需求：
1. 核心：存算一体（CIM）阵列设计
2. SoC集成：软核处理器 + CIM加速器 + 总线互联
3. 软件驱动：控制CIM阵列进行神经网络推理
4. 验证目标：MNIST分类，INT8精度

对于极简版本，我需要考虑：
- 最小可验证的功能集
- 逐步扩展的路径
- 3人团队的工作分配

极简版应该包含：
1. 最简单的CIM单元（单个MAC阵列）
2. 简化的控制接口（不需要完整AXI总线）
3. 简单的全连接层（MLP）而非CNN
4. Testbench验证而非完整SoC

然后逐步扩展到完整版本。

AI助手:
理解你们的需求了。针对MNIST INT8推理的存算一体FPGA验证系统，采用迭代开发是个明智的策略。我给你们设计一个从极简到完整的渐进式方案。

## 极简版架构（MVP - Minimum Viable Product）

**核心目标**：用纯Verilog实现一个能跑通MNIST推理的最小CIM系统

```
极简版架构：
┌─────────────────────────────────────┐
│  Testbench (Python/C生成激励)       │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Simple Controller (FSM)            │
│  - 读取输入/权重                     │
│  - 控制计算流程                      │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  CIM Core (单层全连接)               │
│  ┌────────────────────────────┐    │
│  │ Weight Memory (BRAM)       │    │
│  │ 784x128 INT8               │    │
│  └────────────────────────────┘    │
│  ┌────────────────────────────┐    │
│  │ MAC Array (16个并行MAC)    │    │
│  │ INT8 x INT8 -> INT32       │    │
│  └────────────────────────────┘    │
│  ┌────────────────────────────┐    │
│  │ Activation (ReLU)          │    │
│  └────────────────────────────┘    │
└───────────────────────────────────┘
```

**简化点**：
- 只实现单层FC（784→128），输出层用软件计算
- 固定批大小为1
- 无AXI总线，直接用简单握手协议
- 权重预加载到BRAM，不需要DDR
- 仿真验证为主，暂不上板

## 完整开发路线图

### **阶段1：极简验证版（2周）**

**目标**：仿真跑通单层推理

1. **CIM基础单元** (人员A)
   - 设计8bit×8bit MAC单元
   - 16个MAC并行阵列（处理16个输出神经元）
   - BRAM权重存储接口
   - ReLU激活函数

2. **控制逻辑** (人员B)
   - FSM状态机：IDLE → LOAD_INPUT → COMPUTE → OUTPUT
   - 地址生成器（遍历784个输入）
   - 累加器（INT32）

3. **验证环境** (人员C)
   - Python脚本：训练简化MNIST模型（784→128→10）
   - 量化为INT8并导出权重
   - 生成Verilog testbench激励文件
   - 对比软件golden结果

**交付物**：
- 单层CIM RTL代码
- 仿真通过的testbench
- 精度验证报告

---

### **阶段2：多层扩展版（2周）**

**新增功能**：
- 扩展为3层MLP（784→128→64→10）
- 添加第二个CIM实例或复用
- 实现层间数据缓存

**修改点**：
```verilog
// 添加层配置寄存器
reg [15:0] layer_input_size;
reg [15:0] layer_output_size;
reg [31:0] weight_base_addr;

// 支持多层迭代
for (layer_id = 0; layer_id < 3; layer_id++) begin
    // 配置当前层参数
    // 执行计算
end
```

---

### **阶段3：SoC集成版（3周）**

**架构升级**：
```
┌──────────────┐    AXI-Lite    ┌──────────────┐
│ MicroBlaze/  │◄──────────────►│ CIM Accel    │
│ RISC-V Core  │                │ (AXI Slave)  │
└──────┬───────┘                └──────────────┘
       │ AXI-Full
┌──────▼───────┐
│ DDR Controller│
│ (存储权重)    │
└──────────────┘
```

**新增模块**：
1. **AXI接口封装** (人员A)
   - AXI4-Lite寄存器接口（控制/状态）
   - AXI4-Full DMA接口（权重加载）

2. **软核集成** (人员B)
   - Vivado Block Design集成MicroBlaze
   - 配置DDR控制器
   - 中断连接

3. **驱动软件** (人员C)
   ```c
   // 伪代码
   void cim_inference(uint8_t* input, uint8_t* output) {
       // 1. 加载权重到CIM
       dma_transfer(weights, CIM_WEIGHT_ADDR, weight_size);
       
       // 2. 写入输入数据
       write_reg(CIM_INPUT_ADDR, input);
       
       // 3. 启动计算
       write_reg(CIM_CTRL, START_BIT);
       
       // 4. 等待中断
       wait_interrupt();
       
       // 5. 读取结果
       read_reg(CIM_OUTPUT_ADDR, output);
   }
   ```

---

### **阶段4：优化与CNN扩展（3周）**

**性能优化**：
- 流水线设计（输入加载与计算重叠）
- 增加MAC并行度（16→64）
- 片上缓存优化

**CNN支持**（可选）：
- 添加卷积计算模式
- Im2col转换逻辑
- Pooling单元

---

## 详细实施步骤

### **Step 1: 环境准备**
```bash
# 工具链
- Vivado 2020.2+
- ModelSim/Questasim
- Python 3.8+ (PyTorch/TensorFlow)

# 目录结构
project/
├── rtl/              # Verilog源码
│   ├── cim_core.v
│   ├── mac_unit.v
│   └── controller.v
├── sim/              # 仿真文件
│   ├── tb_cim.v
│   └── wave.do
├── sw/               # 软件
│   ├── train.py      # 模型训练
│   ├── quantize.py   # INT8量化
│   └── driver.c      # 驱动代码
├── data/             # 数据集
└── docs/             # 文档
```

### **Step 2: 模型准备**
```python
# train.py - 极简版
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        # 第二层先用软件实现
    
    def forward(self, x):
        return self.relu(self.fc1(x))

# 训练后量化为INT8
# 导出权重为.hex或.mem文件供Verilog读取
```

### **Step 3: 核心RTL设计**
```verilog
// mac_unit.v - INT8乘累加单元
module mac_unit (
    input clk,
    input rst_n,
    input signed [7:0] weight,
    input signed [7:0] activation,
    input acc_en,
    input acc_clear,
    output reg signed [31:0] result
);
    wire signed [15:0] product = weight * activation;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            result <= 0;
        else if (acc_clear)
            result <= 0;
        else if (acc_en)
            result <= result + product;
    end
endmodule

// cim_core.v - 存算一体核心
module cim_core #(
    parameter INPUT_SIZE = 784,
    parameter OUTPUT_SIZE = 128,
    parameter MAC_NUM = 16  // 并行度
)(
    input clk,
    input rst_n,
    input start,
    input [7:0] input_data [0:INPUT_SIZE-1],
    output reg [7:0] output_data [0:OUTPUT_SIZE-1],
    output reg done
);
    // BRAM存储权重 [OUTPUT_SIZE][INPUT_SIZE]
    reg signed [7:0] weights [0:OUTPUT_SIZE-1][0:INPUT_SIZE-1];
    
    // MAC阵列
    wire signed [31:0] mac_results [0:MAC_NUM-1];
    
    // 控制FSM
    // ...实现计算逻辑
endmodule
```

### **Step 4: 仿真验证**
```verilog
// tb_cim.v
module tb_cim;
    // 从文件读取测试向量
    initial begin
        $readmemh("input.hex", input_vector);
        $readmemh("weights.hex", golden_weights);
        $readmemh("expected.hex", expected_output);
    end
    
    // 运行测试
    initial begin
        // 激励
        // 检查结果
        if (output_data == expected_output)
            $display("PASS");
        else
            $display("FAIL");
    end
endmodule
```

### **Step 5: 上板验证**
- 生成bitstream
- UART通信接口
- 板上测试100个MNIST样本

---

## 团队分工建议

**人员A（硬件核心）**：
- CIM阵列设计
- MAC单元优化
- AXI接口

**人员B（系统集成）**：
- 控制FSM
- SoC集成
- 时序约束

**人员C（软件验证）**：
- 模型训练量化
- Testbench生成
- 驱动开发

---

## 时间规划（10周）

| 周次 | 阶段 | 里程碑 |
|------|------|--------|
| 1-2  | 阶段1 | 单层仿真通过 |
| 3-4  | 阶段2 | 多层MLP完成 |
| 5-7  | 阶段3 | SoC集成上板 |
| 8-9  | 阶段4 | 性能优化 |
| 10   | 总结  | 报告答辩 |

---

## 关键技术点

1. **INT8量化**：注意溢出处理，使用饱和运算
2. **BRAM利用**：权重矩阵按行存储，支持并行读取
3. **流水线**：输入预取、计算、输出写回三级流水
4. **调试技巧**：先用小规模参数（如10×10矩阵）验证逻辑

这个方案让你们能快速看到成果，同时保留扩展空间。极简版2周内就能跑起来，给后续优化留足时间。需要我详细展开某个模块的设计吗？

用户:
关于阶段1-3，给出详细的工作流程
用户:
用户要求详细展开阶段1-3的工作流程。我需要给出非常具体的、可执行的步骤，包括代码实现、文件组织、调试方法等。要确保每个步骤都是可操作的，让3个学生能够按照这个流程实际完成项目。

我需要覆盖：
1. 阶段1：极简验证版的详细实现
2. 阶段2：多层扩展版的详细实现
3. 阶段3：SoC集成版的详细实现

每个阶段要包括：
- 具体的代码实现
- 文件结构
- 测试方法
- 常见问题处理
- 验收标准

AI助手:
好的，我给你们详细拆解阶段1-3的完整工作流程，每个步骤都可以直接执行。

# 阶段1：极简验证版（2周）

## Week 1: 基础模块开发

### Day 1-2: 环境搭建与模型准备（人员C负责）

**1.1 创建项目结构**
```bash
mkdir cim_mnist_project
cd cim_mnist_project

# 创建目录
mkdir -p rtl/{core,utils,tb}
mkdir -p sw/{train,quantize,testgen}
mkdir -p sim/{modelsim,data}
mkdir -p docs
mkdir -p constraints
```

**1.2 训练并量化MNIST模型**

```python
# sw/train/train_simple_mlp.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128, bias=True)
        self.relu = nn.ReLU()
        # 暂时只实现第一层，后续扩展
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.relu(x)
        return x

# 训练代码
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model = SimpleMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()  # 简化损失函数
    
    # 训练10个epoch
    for epoch in range(10):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            
            # 简化：只训练特征提取
            loss = output.pow(2).mean()  # 简单的正则化损失
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    # 保存模型
    torch.save(model.state_dict(), 'simple_mlp.pth')
    return model

if __name__ == '__main__':
    model = train()
```

**1.3 INT8量化与导出**

```python
# sw/quantize/quantize_int8.py
import torch
import numpy as np

def quantize_tensor(tensor, bits=8):
    """量化为INT8"""
    qmin = -2**(bits-1)
    qmax = 2**(bits-1) - 1
    
    # 计算scale
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - min_val / scale
    
    # 量化
    q_tensor = torch.clamp(torch.round(tensor / scale + zero_point), qmin, qmax)
    return q_tensor.to(torch.int8), scale, zero_point

def export_weights_to_hex(model_path, output_dir):
    """导出权重为Verilog可读的hex文件"""
    model = SimpleMLP()
    model.load_state_dict(torch.load(model_path))
    
    # 提取第一层权重和偏置
    weight = model.fc1.weight.data  # [128, 784]
    bias = model.fc1.bias.data      # [128]
    
    # 量化
    q_weight, w_scale, w_zero = quantize_tensor(weight)
    q_bias, b_scale, b_zero = quantize_tensor(bias)
    
    # 保存为hex格式（每行一个INT8值）
    with open(f'{output_dir}/weights.hex', 'w') as f:
        for row in q_weight:
            for val in row:
                f.write(f'{val.item() & 0xFF:02x}\n')
    
    with open(f'{output_dir}/bias.hex', 'w') as f:
        for val in q_bias:
            f.write(f'{val.item() & 0xFF:02x}\n')
    
    # 保存scale信息（用于反量化验证）
    np.savez(f'{output_dir}/quant_params.npz',
             w_scale=w_scale, w_zero=w_zero,
             b_scale=b_scale, b_zero=b_zero)
    
    print(f"Weight shape: {q_weight.shape}")
    print(f"Weight scale: {w_scale}, zero_point: {w_zero}")
    print(f"Bias scale: {b_scale}, zero_point: {b_zero}")

if __name__ == '__main__':
    export_weights_to_hex('simple_mlp.pth', '../sim/data')
```

**1.4 生成测试向量**

```python
# sw/testgen/generate_testbench.py
import torch
from torchvision import datasets, transforms
import numpy as np

def generate_test_vectors(num_samples=10):
    """生成测试输入和golden输出"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # 加载量化后的模型
    model = SimpleMLP()
    model.load_state_dict(torch.load('simple_mlp.pth'))
    model.eval()
    
    # 加载量化参数
    quant_params = np.load('../sim/data/quant_params.npz')
    
    for i in range(num_samples):
        img, label = test_dataset[i]
        img_flat = img.view(-1, 784)
        
        # 量化输入
        q_img, img_scale, img_zero = quantize_tensor(img_flat)
        
        # 软件golden结果
        with torch.no_grad():
            output = model(img_flat)
        
        # 保存输入
        with open(f'../sim/data/input_{i}.hex', 'w') as f:
            for val in q_img.flatten():
                f.write(f'{val.item() & 0xFF:02x}\n')
        
        # 保存golden输出
        q_output, _, _ = quantize_tensor(output)
        with open(f'../sim/data/golden_{i}.hex', 'w') as f:
            for val in q_output.flatten():
                f.write(f'{val.item() & 0xFF:02x}\n')
        
        print(f"Generated test vector {i}, label: {label}")

if __name__ == '__main__':
    generate_test_vectors()
```

---

### Day 3-4: MAC单元设计（人员A负责）

**2.1 基础MAC单元**

```verilog
// rtl/core/mac_unit.v
module mac_unit (
    input wire clk,
    input wire rst_n,
    input wire signed [7:0] weight,      // INT8权重
    input wire signed [7:0] activation,  // INT8激活
    input wire mac_en,                   // 使能信号
    input wire acc_clear,                // 清零累加器
    output reg signed [31:0] acc_result  // INT32累加结果
);

    // 乘法结果（16位）
    wire signed [15:0] product;
    assign product = weight * activation;
    
    // 累加逻辑
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_result <= 32'd0;
        end else if (acc_clear) begin
            acc_result <= 32'd0;
        end else if (mac_en) begin
            acc_result <= acc_result + {{16{product[15]}}, product}; // 符号扩展
        end
    end

endmodule
```

**2.2 MAC单元测试**

```verilog
// rtl/tb/tb_mac_unit.v
`timescale 1ns/1ps

module tb_mac_unit;
    reg clk;
    reg rst_n;
    reg signed [7:0] weight;
    reg signed [7:0] activation;
    reg mac_en;
    reg acc_clear;
    wire signed [31:0] acc_result;
    
    // 实例化
    mac_unit uut (
        .clk(clk),
        .rst_n(rst_n),
        .weight(weight),
        .activation(activation),
        .mac_en(mac_en),
        .acc_clear(acc_clear),
        .acc_result(acc_result)
    );
    
    // 时钟生成
    initial clk = 0;
    always #5 clk = ~clk;
    
    // 测试序列
    initial begin
        rst_n = 0;
        mac_en = 0;
        acc_clear = 0;
        weight = 0;
        activation = 0;
        
        #20 rst_n = 1;
        
        // 测试1: 简单乘累加
        #10 acc_clear = 1;
        #10 acc_clear = 0;
        
        // 3 * 2 = 6
        weight = 8'd3;
        activation = 8'd2;
        mac_en = 1;
        #10;
        
        // 4 * 5 = 20, 累加 = 26
        weight = 8'd4;
        activation = 8'd5;
        #10;
        
        // 检查结果
        if (acc_result == 32'd26) begin
            $display("TEST PASS: acc_result = %d", acc_result);
        end else begin
            $display("TEST FAIL: expected 26, got %d", acc_result);
        end
        
        // 测试2: 负数
        #10 acc_clear = 1;
        #10 acc_clear = 0;
        
        weight = -8'd10;
        activation = 8'd5;
        #10;
        
        if (acc_result == -32'd50) begin
            $display("TEST PASS: negative test");
        end else begin
            $display("TEST FAIL: negative test, got %d", acc_result);
        end
        
        #50 $finish;
    end
    
    // 波形dump
    initial begin
        $dumpfile("mac_unit.vcd");
        $dumpvars(0, tb_mac_unit);
    end

endmodule
```

**运行仿真**：
```bash
cd sim/modelsim
# 使用iverilog（开源）
iverilog -o mac_unit.vvp ../../rtl/core/mac_unit.v ../../rtl/tb/tb_mac_unit.v
vvp mac_unit.vvp
gtkwave mac_unit.vcd  # 查看波形

# 或使用ModelSim
vlog ../../rtl/core/mac_unit.v ../../rtl/tb/tb_mac_unit.v
vsim -c tb_mac_unit -do "run -all; quit"
```

---

### Day 5-7: CIM核心阵列（人员A+B协作）

**3.1 CIM核心模块**

```verilog
// rtl/core/cim_core.v
module cim_core #(
    parameter INPUT_SIZE = 784,
    parameter OUTPUT_SIZE = 128,
    parameter MAC_PARALLEL = 16,  // 并行MAC数量
    parameter WEIGHT_ADDR_WIDTH = 17  // log2(784*128)
)(
    input wire clk,
    input wire rst_n,
    
    // 控制接口
    input wire start,
    output reg done,
    output reg busy,
    
    // 输入数据接口（简化：一次性加载）
    input wire [7:0] input_data [0:INPUT_SIZE-1],
    
    // 输出数据接口
    output reg [7:0] output_data [0:OUTPUT_SIZE-1],
    
    // 权重加载接口
    input wire weight_wr_en,
    input wire [WEIGHT_ADDR_WIDTH-1:0] weight_wr_addr,
    input wire [7:0] weight_wr_data
);

    // 权重存储：BRAM [OUTPUT_SIZE][INPUT_SIZE]
    reg signed [7:0] weight_mem [0:OUTPUT_SIZE*INPUT_SIZE-1];
    
    // 偏置存储
    reg signed [7:0] bias_mem [0:OUTPUT_SIZE-1];
    
    // MAC阵列
    wire signed [31:0] mac_results [0:MAC_PARALLEL-1];
    reg signed [7:0] mac_weights [0:MAC_PARALLEL-1];
    reg signed [7:0] mac_activations [0:MAC_PARALLEL-1];
    reg mac_en [0:MAC_PARALLEL-1];
    reg mac_clear [0:MAC_PARALLEL-1];
    
    genvar i;
    generate
        for (i = 0; i < MAC_PARALLEL; i = i + 1) begin : mac_array
            mac_unit mac_inst (
                .clk(clk),
                .rst_n(rst_n),
                .weight(mac_weights[i]),
                .activation(mac_activations[i]),
                .mac_en(mac_en[i]),
                .acc_clear(mac_clear[i]),
                .acc_result(mac_results[i])
            );
        end
    endgenerate
    
    // 权重写入逻辑
    always @(posedge clk) begin
        if (weight_wr_en) begin
            weight_mem[weight_wr_addr] <= weight_wr_data;
        end
    end
    
    // 控制FSM
    localparam IDLE = 3'd0;
    localparam COMPUTE = 3'd1;
    localparam ACCUMULATE = 3'd2;
    localparam ACTIVATE = 3'd3;
    localparam DONE = 3'd4;
    
    reg [2:0] state, next_state;
    reg [15:0] input_idx;   // 当前处理的输入索引 [0, 783]
    reg [15:0] output_idx;  // 当前处理的输出索引 [0, 127]
    reg [15:0] compute_cnt; // 计算周期计数
    
    // 状态转移
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end
    
    // 下一状态逻辑
    always @(*) begin
        next_state = state;
        case (state)
            IDLE: begin
                if (start)
                    next_state = COMPUTE;
            end
            
            COMPUTE: begin
                // 每个输出神经元需要784个周期完成MAC
                if (input_idx == INPUT_SIZE - 1)
                    next_state = ACTIVATE;
            end
            
            ACTIVATE: begin
                // 应用ReLU并存储
                if (output_idx == OUTPUT_SIZE - 1)
                    next_state = DONE;
                else
                    next_state = COMPUTE;
            end
            
            DONE: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // 数据通路控制
    integer j;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            input_idx <= 0;
            output_idx <= 0;
            done <= 0;
            busy <= 0;
            for (j = 0; j < MAC_PARALLEL; j = j + 1) begin
                mac_en[j] <= 0;
                mac_clear[j] <= 0;
            end
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    busy <= 0;
                    input_idx <= 0;
                    output_idx <= 0;
                end
                
                COMPUTE: begin
                    busy <= 1;
                    
                    // 为MAC_PARALLEL个输出神经元并行计算
                    for (j = 0; j < MAC_PARALLEL; j = j + 1) begin
                        if (output_idx + j < OUTPUT_SIZE) begin
                            // 清零累加器（第一个输入时）
                            if (input_idx == 0) begin
                                mac_clear[j] <= 1;
                            end else begin
                                mac_clear[j] <= 0;
                            end
                            
                            // 加载权重和激活
                            mac_weights[j] <= weight_mem[(output_idx + j) * INPUT_SIZE + input_idx];
                            mac_activations[j] <= input_data[input_idx];
                            mac_en[j] <= 1;
                        end else begin
                            mac_en[j] <= 0;
                        end
                    end
                    
                    // 递增输入索引
                    if (input_idx < INPUT_SIZE - 1) begin
                        input_idx <= input_idx + 1;
                    end else begin
                        input_idx <= 0;
                    end
                end
                
                ACTIVATE: begin
                    // 应用ReLU激活函数
                    for (j = 0; j < MAC_PARALLEL; j = j + 1) begin
                        if (output_idx + j < OUTPUT_SIZE) begin
                            // INT32 -> INT8 with ReLU
                            // 简化：直接截断高位，实际应做饱和处理
                            if (mac_results[j][31] == 1'b1) begin  // 负数
                                output_data[output_idx + j] <= 8'd0;
                            end else if (mac_results[j] > 127) begin  // 溢出
                                output_data[output_idx + j] <= 8'd127;
                            end else begin
                                output_data[output_idx + j] <= mac_results[j][7:0];
                            end
                        end
                    end
                    
                    // 递增输出索引
                    if (output_idx + MAC_PARALLEL < OUTPUT_SIZE) begin
                        output_idx <= output_idx + MAC_PARALLEL;
                    end else begin
                        output_idx <= 0;
                    end
                end
                
                DONE: begin
                    done <= 1;
                    busy <= 0;
                end
            endcase
        end
    end

endmodule
```

---

### Day 8-10: 系统级测试（人员C负责）

**4.1 顶层Testbench**

```verilog
// rtl/tb/tb_cim_system.v
`timescale 1ns/1ps

module tb_cim_system;
    parameter INPUT_SIZE = 784;
    parameter OUTPUT_SIZE = 128;
    parameter MAC_PARALLEL = 16;
    
    reg clk;
    reg rst_n;
    reg start;
    wire done;
    wire busy;
    
    reg [7:0] input_data [0:INPUT_SIZE-1];
    wire [7:0] output_data [0:OUTPUT_SIZE-1];
    
    reg weight_wr_en;
    reg [16:0] weight_wr_addr;
    reg [7:0] weight_wr_data;
    
    // 实例化CIM核心
    cim_core #(
        .INPUT_SIZE(INPUT_SIZE),
        .OUTPUT_SIZE(OUTPUT_SIZE),
        .MAC_PARALLEL(MAC_PARALLEL)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
        .busy(busy),
        .input_data(input_data),
        .output_data(output_data),
        .weight_wr_en(weight_wr_en),
        .weight_wr_addr(weight_wr_addr),
        .weight_wr_data(weight_wr_data)
    );
    
    // 时钟生成：100MHz
    initial clk = 0;
    always #5 clk = ~clk;
    
    // Golden结果
    reg [7:0] golden_output [0:OUTPUT_SIZE-1];
    integer error_count;
    
    // 测试流程
    integer i, j;
    initial begin
        $display("=== CIM System Test Start ===");
        
        // 初始化
        rst_n = 0;
        start = 0;
        weight_wr_en = 0;
        error_count = 0;
        
        #100 rst_n = 1;
        
        // 1. 加载权重
        $display("Loading weights...");
        weight_wr_en = 1;
        for (i = 0; i < OUTPUT_SIZE * INPUT_SIZE; i = i + 1) begin
            weight_wr_addr = i;
            // 从文件读取（实际应用中）
            // 这里用简单的测试数据
            weight_wr_data = (i % 256);
            @(posedge clk);
        end
        weight_wr_en = 0;
        $display("Weights loaded");
        
        // 2. 加载输入数据
        $display("Loading input data...");
        $readmemh("../data/input_0.hex", input_data);
        $readmemh("../data/golden_0.hex", golden_output);
        
        // 3. 启动计算
        $display("Starting computation...");
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        
        // 4. 等待完成
        wait(done == 1);
        $display("Computation done");
        
        // 5. 验证结果
        $display("Verifying results...");
        for (i = 0; i < OUTPUT_SIZE; i = i + 1) begin
            if (output_data[i] !== golden_output[i]) begin
                $display("ERROR at index %d: expected %h, got %h", 
                         i, golden_output[i], output_data[i]);
                error_count = error_count + 1;
            end
        end
        
        if (error_count == 0) begin
            $display("=== TEST PASSED ===");
        end else begin
            $display("=== TEST FAILED: %d errors ===", error_count);
        end
        
        #1000 $finish;
    end
    
    // 性能统计
    integer cycle_count;
    always @(posedge clk) begin
        if (busy)
            cycle_count = cycle_count + 1;
    end
    
    initial begin
        cycle_count = 0;
        wait(done == 1);
        $display("Total cycles: %d", cycle_count);
        $display("Throughput: %f GOPS", 
                 (2.0 * INPUT_SIZE * OUTPUT_SIZE) / (cycle_count * 10.0));
    end
    
    // 波形dump
    initial begin
        $dumpfile("cim_system.vcd");
        $dumpvars(0, tb_cim_system);
    end

endmodule
```

**4.2 运行完整测试**

```bash
# sim/modelsim/run_test.sh
#!/bin/bash

echo "Compiling RTL..."
vlog -work work ../../rtl/core/mac_unit.v
vlog -work work ../../rtl/core/cim_core.v
vlog -work work ../../rtl/tb/tb_cim_system.v

echo "Running simulation..."
vsim -c work.tb_cim_system -do "run -all; quit"

echo "Checking results..."
grep "TEST PASSED" transcript && echo "SUCCESS" || echo "FAILED"
```

---

## Week 2: 集成与优化

### Day 11-12: 调试与修复（全员）

**常见问题排查清单**：

1. **时序问题**
```verilog
// 检查关键路径
// 在cim_core.v中添加流水线寄存器
reg signed [7:0] mac_weights_reg [0:MAC_PARALLEL-1];
reg signed [7:0] mac_activations_reg [0:MAC_PARALLEL-1];

always @(posedge clk) begin
    for (j = 0; j < MAC_PARALLEL; j = j + 1) begin
        mac_weights_reg[j] <= mac_weights[j];
        mac_activations_reg[j] <= mac_activations[j];
    end
end
```

2. **数据对齐问题**
```python
# 检查Python生成的hex文件格式
with open('weights.hex', 'r') as f:
    lines = f.readlines()
    print(f"Total weight values: {len(lines)}")
    print(f"Expected: {784 * 128}")
```

3. **量化精度损失**
```python
# 对比浮点和INT8结果
def compare_accuracy(float_output, int8_output):
    mse = np.mean((float_output - int8_output)**2)
    print(f"MSE: {mse}")
    # 可接受范围：MSE < 0.1
```

### Day 13-14: 文档与验收（人员C负责）

**阶段1验收标准**：
- [ ] MAC单元独立测试通过
- [ ] CIM核心完成单层784→128推理
- [ ] 至少10个MNIST样本测试通过
- [ ] 仿真波形正确
- [ ] 代码有注释和文档

**提交物**：
```
stage1_deliverables/
├── rtl/                    # 所有RTL代码
├── sim/results/            # 仿真结果截图
├── docs/
│   ├── design_spec.pdf     # 设计文档
│   └── test_report.pdf     # 测试报告
└── README.md
```

---

# 阶段2：多层扩展版（2周）

## Week 3: 架构重构

### Day 15-17: 可配置层设计（人员A+B）

**1. 修改CIM核心支持可配置层**

```verilog
// rtl/core/cim_core_v2.v
module cim_core_v2 #(
    parameter MAX_INPUT_SIZE = 784,
    parameter MAX_OUTPUT_SIZE = 128,
    parameter MAC_PARALLEL = 16
)(
    input wire clk,
    input wire rst_n,
    
    // 动态配置接口
    input wire [15:0] cfg_input_size,
    input wire [15:0] cfg_output_size,
    input wire [31:0] cfg_weight_base_addr,
    input wire [31:0] cfg_bias_base_addr,
    
    // 其他接口同v1...
);

    // 使用配置参数替代硬编码
    always @(*) begin
        case (state)
            COMPUTE: begin
                // 使用cfg_input_size而不是INPUT_SIZE
                if (input_idx == cfg_input_size - 1)
                    next_state = ACTIVATE;
            end
            // ...
        endcase
    end

endmodule
```

**2. 多层控制器**

```verilog
// rtl/core/mlp_controller.v
module mlp_controller #(
    parameter NUM_LAYERS = 3
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    output reg done,
    
    // 与CIM核心的接口
    output reg cim_start,
    input wire cim_done,
    output reg [15:0] cim_input_size,
    output reg [15:0] cim_output_size,
    
    // 层间缓存接口
    output reg [31:0] buffer_wr_addr,
    output reg [7:0] buffer_wr_data,
    output reg buffer_wr_en,
    output reg [31:0] buffer_rd_addr,
    input wire [7:0] buffer_rd_data
);

    // 层配置表
    reg [15:0] layer_configs [0:NUM_LAYERS-1][0:1];  // [input_size, output_size]
    
    initial begin
        // Layer 0: 784 -> 128
        layer_configs[0][0] = 784;
        layer_configs[0][1] = 128;
        
        // Layer 1: 128 -> 64
        layer_configs[1][0] = 128;
        layer_configs[1][1] = 64;
        
        // Layer 2: 64 -> 10
        layer_configs[2][0] = 64;
        layer_configs[2][1] = 10;
    end
    
    reg [7:0] current_layer;
    
    localparam IDLE = 0;
    localparam LOAD_INPUT = 1;
    localparam COMPUTE_LAYER = 

用户:
阶段２没有讲完，请继续
用户:
用户要求我继续完成阶段2的内容。我需要从阶段2的多层控制器设计继续，然后完成阶段2的其他部分，最后完成阶段3的详细内容。

让我继续从mlp_controller.v的状态机定义开始。

AI助手:
好的，继续完成阶段2和阶段3的详细内容。

---

# 阶段2：多层扩展版（续）

## Week 3: 架构重构（续）

### Day 15-17: 可配置层设计（续）

**2. 多层控制器（续）**

```verilog
// rtl/core/mlp_controller.v (续)
    localparam IDLE = 0;
    localparam LOAD_INPUT = 1;
    localparam COMPUTE_LAYER = 2;
    localparam TRANSFER_OUTPUT = 3;
    localparam DONE_STATE = 4;
    
    reg [2:0] state, next_state;
    reg [15:0] transfer_idx;
    
    // 状态转移
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end
    
    // 下一状态逻辑
    always @(*) begin
        next_state = state;
        case (state)
            IDLE: begin
                if (start)
                    next_state = LOAD_INPUT;
            end
            
            LOAD_INPUT: begin
                // 第一层直接从外部输入，后续层从缓存读取
                next_state = COMPUTE_LAYER;
            end
            
            COMPUTE_LAYER: begin
                if (cim_done) begin
                    if (current_layer == NUM_LAYERS - 1)
                        next_state = DONE_STATE;
                    else
                        next_state = TRANSFER_OUTPUT;
                end
            end
            
            TRANSFER_OUTPUT: begin
                // 将当前层输出转移到下一层输入缓存
                if (transfer_idx == layer_configs[current_layer][1] - 1)
                    next_state = LOAD_INPUT;
            end
            
            DONE_STATE: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // 控制逻辑
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_layer <= 0;
            cim_start <= 0;
            done <= 0;
            transfer_idx <= 0;
            buffer_wr_en <= 0;
        end else begin
            case (state)
                IDLE: begin
                    current_layer <= 0;
                    done <= 0;
                    cim_start <= 0;
                end
                
                LOAD_INPUT: begin
                    // 配置CIM核心
                    cim_input_size <= layer_configs[current_layer][0];
                    cim_output_size <= layer_configs[current_layer][1];
                end
                
                COMPUTE_LAYER: begin
                    if (!cim_start && !cim_done) begin
                        cim_start <= 1;
                    end else begin
                        cim_start <= 0;
                    end
                end
                
                TRANSFER_OUTPUT: begin
                    // 将CIM输出写入缓存，作为下一层输入
                    buffer_wr_addr <= transfer_idx;
                    buffer_wr_data <= cim_output_data[transfer_idx];
                    buffer_wr_en <= 1;
                    
                    if (transfer_idx < layer_configs[current_layer][1] - 1) begin
                        transfer_idx <= transfer_idx + 1;
                    end else begin
                        transfer_idx <= 0;
                        current_layer <= current_layer + 1;
                        buffer_wr_en <= 0;
                    end
                end
                
                DONE_STATE: begin
                    done <= 1;
                end
            endcase
        end
    end

endmodule
```

**3. 层间数据缓存**

```verilog
// rtl/core/layer_buffer.v
module layer_buffer #(
    parameter DEPTH = 1024,  // 最大支持1024个元素
    parameter ADDR_WIDTH = 10
)(
    input wire clk,
    
    // 写端口
    input wire wr_en,
    input wire [ADDR_WIDTH-1:0] wr_addr,
    input wire [7:0] wr_data,
    
    // 读端口
    input wire [ADDR_WIDTH-1:0] rd_addr,
    output reg [7:0] rd_data
);

    // 使用BRAM实现
    reg [7:0] mem [0:DEPTH-1];
    
    // 写操作
    always @(posedge clk) begin
        if (wr_en) begin
            mem[wr_addr] <= wr_data;
        end
    end
    
    // 读操作（1周期延迟）
    always @(posedge clk) begin
        rd_data <= mem[rd_addr];
    end

endmodule
```

---

### Day 18-19: 顶层集成（人员B负责）

**4. 多层MLP顶层模块**

```verilog
// rtl/core/mlp_top.v
module mlp_top #(
    parameter NUM_LAYERS = 3,
    parameter MAX_LAYER_SIZE = 784,
    parameter MAC_PARALLEL = 16
)(
    input wire clk,
    input wire rst_n,
    
    // 控制接口
    input wire start,
    output wire done,
    output wire busy,
    
    // 输入接口（第一层）
    input wire [7:0] input_data [0:783],
    
    // 输出接口（最后一层）
    output wire [7:0] output_data [0:9],
    
    // 权重加载接口
    input wire weight_wr_en,
    input wire [2:0] weight_layer_sel,  // 选择哪一层
    input wire [16:0] weight_wr_addr,
    input wire [7:0] weight_wr_data
);

    // CIM核心实例
    wire cim_start, cim_done, cim_busy;
    wire [15:0] cim_input_size, cim_output_size;
    wire [7:0] cim_input [0:MAX_LAYER_SIZE-1];
    wire [7:0] cim_output [0:MAX_LAYER_SIZE-1];
    
    cim_core_v2 #(
        .MAX_INPUT_SIZE(MAX_LAYER_SIZE),
        .MAX_OUTPUT_SIZE(MAX_LAYER_SIZE),
        .MAC_PARALLEL(MAC_PARALLEL)
    ) cim_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(cim_start),
        .done(cim_done),
        .busy(cim_busy),
        .cfg_input_size(cim_input_size),
        .cfg_output_size(cim_output_size),
        .input_data(cim_input),
        .output_data(cim_output),
        .weight_wr_en(weight_wr_en && (weight_layer_sel == current_layer)),
        .weight_wr_addr(weight_wr_addr),
        .weight_wr_data(weight_wr_data)
    );
    
    // 多层控制器
    wire [31:0] buffer_wr_addr, buffer_rd_addr;
    wire [7:0] buffer_wr_data, buffer_rd_data;
    wire buffer_wr_en;
    wire [7:0] current_layer;
    
    mlp_controller #(
        .NUM_LAYERS(NUM_LAYERS)
    ) controller_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
        .cim_start(cim_start),
        .cim_done(cim_done),
        .cim_input_size(cim_input_size),
        .cim_output_size(cim_output_size),
        .buffer_wr_addr(buffer_wr_addr),
        .buffer_wr_data(buffer_wr_data),
        .buffer_wr_en(buffer_wr_en),
        .buffer_rd_addr(buffer_rd_addr),
        .buffer_rd_data(buffer_rd_data),
        .current_layer(current_layer)
    );
    
    // 层间缓存
    layer_buffer #(
        .DEPTH(1024),
        .ADDR_WIDTH(10)
    ) buffer_inst (
        .clk(clk),
        .wr_en(buffer_wr_en),
        .wr_addr(buffer_wr_addr[9:0]),
        .wr_data(buffer_wr_data),
        .rd_addr(buffer_rd_addr[9:0]),
        .rd_data(buffer_rd_data)
    );
    
    // 输入数据路由
    integer i;
    always @(*) begin
        if (current_layer == 0) begin
            // 第一层：从外部输入
            for (i = 0; i < 784; i = i + 1) begin
                cim_input[i] = input_data[i];
            end
        end else begin
            // 后续层：从缓存读取
            for (i = 0; i < MAX_LAYER_SIZE; i = i + 1) begin
                if (i < cim_input_size) begin
                    // 需要添加读取逻辑
                    cim_input[i] = buffer_rd_data;  // 简化，实际需要地址控制
                end else begin
                    cim_input[i] = 8'd0;
                end
            end
        end
    end
    
    // 输出数据路由
    always @(*) begin
        if (current_layer == NUM_LAYERS - 1 && done) begin
            for (i = 0; i < 10; i = i + 1) begin
                output_data[i] = cim_output[i];
            end
        end
    end
    
    assign busy = cim_busy;

endmodule
```

---

### Day 20-21: 多层测试（人员C负责）

**5. 更新Python训练脚本**

```python
# sw/train/train_mlp_3layer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

class ThreeLayerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    model = ThreeLayerMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 训练
    for epoch in range(10):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # 测试
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        accuracy = 100. * correct / len(test_dataset)
        print(f'Epoch {epoch}, Test Accuracy: {accuracy:.2f}%')
    
    torch.save(model.state_dict(), 'mlp_3layer.pth')
    return model

if __name__ == '__main__':
    model = train()
```

**6. 导出多层权重**

```python
# sw/quantize/export_multilayer.py
import torch
import numpy as np

def export_all_layers(model_path, output_dir):
    """导出所有层的权重"""
    model = ThreeLayerMLP()
    model.load_state_dict(torch.load(model_path))
    
    layers = [
        ('fc1', model.fc1, 784, 128),
        ('fc2', model.fc2, 128, 64),
        ('fc3', model.fc3, 64, 10)
    ]
    
    for layer_id, (name, layer, in_size, out_size) in enumerate(layers):
        weight = layer.weight.data  # [out_size, in_size]
        bias = layer.bias.data      # [out_size]
        
        # 量化
        q_weight, w_scale, w_zero = quantize_tensor(weight)
        q_bias, b_scale, b_zero = quantize_tensor(bias)
        
        # 保存
        with open(f'{output_dir}/layer{layer_id}_weights.hex', 'w') as f:
            for row in q_weight:
                for val in row:
                    f.write(f'{val.item() & 0xFF:02x}\n')
        
        with open(f'{output_dir}/layer{layer_id}_bias.hex', 'w') as f:
            for val in q_bias:
                f.write(f'{val.item() & 0xFF:02x}\n')
        
        print(f"Layer {layer_id} ({name}): {in_size} -> {out_size}")
        print(f"  Weight scale: {w_scale:.6f}, Bias scale: {b_scale:.6f}")

if __name__ == '__main__':
    export_all_layers('mlp_3layer.pth', '../sim/data')
```

**7. 多层Testbench**

```verilog
// rtl/tb/tb_mlp_top.v
`timescale 1ns/1ps

module tb_mlp_top;
    parameter NUM_LAYERS = 3;
    
    reg clk, rst_n, start;
    wire done, busy;
    
    reg [7:0] input_data [0:783];
    wire [7:0] output_data [0:9];
    
    reg weight_wr_en;
    reg [2:0] weight_layer_sel;
    reg [16:0] weight_wr_addr;
    reg [7:0] weight_wr_data;
    
    // 实例化顶层
    mlp_top #(
        .NUM_LAYERS(NUM_LAYERS),
        .MAX_LAYER_SIZE(784),
        .MAC_PARALLEL(16)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
        .busy(busy),
        .input_data(input_data),
        .output_data(output_data),
        .weight_wr_en(weight_wr_en),
        .weight_layer_sel(weight_layer_sel),
        .weight_wr_addr(weight_wr_addr),
        .weight_wr_data(weight_wr_data)
    );
    
    initial clk = 0;
    always #5 clk = ~clk;
    
    integer i, layer;
    reg [7:0] expected_label;
    reg [7:0] predicted_label;
    reg [7:0] max_val;
    
    initial begin
        $display("=== Multi-Layer MLP Test ===");
        
        rst_n = 0;
        start = 0;
        weight_wr_en = 0;
        #100 rst_n = 1;
        
        // 加载所有层的权重
        for (layer = 0; layer < NUM_LAYERS; layer = layer + 1) begin
            $display("Loading Layer %0d weights...", layer);
            weight_layer_sel = layer;
            weight_wr_en = 1;
            
            // 根据层的大小加载不同数量的权重
            case (layer)
                0: begin  // 784x128
                    for (i = 0; i < 784*128; i = i + 1) begin
                        weight_wr_addr = i;
                        // 从文件读取
                        weight_wr_data = $random & 8'hFF;  // 简化
                        @(posedge clk);
                    end
                end
                1: begin  // 128x64
                    for (i = 0; i < 128*64; i = i + 1) begin
                        weight_wr_addr = i;
                        weight_wr_data = $random & 8'hFF;
                        @(posedge clk);
                    end
                end
                2: begin  // 64x10
                    for (i = 0; i < 64*10; i = i + 1) begin
                        weight_wr_addr = i;
                        weight_wr_data = $random & 8'hFF;
                        @(posedge clk);
                    end
                end
            endcase
        end
        weight_wr_en = 0;
        $display("All weights loaded");
        
        // 测试推理
        $display("Loading test input...");
        $readmemh("../data/input_0.hex", input_data);
        expected_label = 7;  // 假设这是数字7
        
        $display("Starting inference...");
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        
        wait(done == 1);
        $display("Inference complete");
        
        // 找到最大值索引（预测类别）
        max_val = output_data[0];
        predicted_label = 0;
        for (i = 1; i < 10; i = i + 1) begin
            if (output_data[i] > max_val) begin
                max_val = output_data[i];
                predicted_label = i;
            end
        end
        
        $display("Output logits:");
        for (i = 0; i < 10; i = i + 1) begin
            $display("  Class %0d: %d", i, output_data[i]);
        end
        
        $display("Predicted: %0d, Expected: %0d", predicted_label, expected_label);
        
        if (predicted_label == expected_label) begin
            $display("=== TEST PASSED ===");
        end else begin
            $display("=== TEST FAILED ===");
        end
        
        #1000 $finish;
    end
    
    initial begin
        $dumpfile("mlp_top.vcd");
        $dumpvars(0, tb_mlp_top);
    end

endmodule
```

---

## Week 4: 验证与优化

### Day 22-24: 批量测试（全员）

**8. 批量测试脚本**

```python
# sw/testgen/batch_test.py
import torch
from torchvision import datasets, transforms
import subprocess
import os

def run_batch_test(num_samples=100):
    """运行批量测试"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # 加载模型
    model = ThreeLayerMLP()
    model.load_state_dict(torch.load('mlp_3layer.pth'))
    model.eval()
    
    correct_hw = 0
    correct_sw = 0
    
    for i in range(num_samples):
        img, label = test_dataset[i]
        
        # 软件golden
        with torch.no_grad():
            output_sw = model(img.unsqueeze(0))
            pred_sw = output_sw.argmax(dim=1).item()
        
        if pred_sw == label:
            correct_sw += 1
        
        # 生成硬件测试向量
        generate_single_test(img, label, i)
        
        # 运行硬件仿真
        result = run_hardware_sim(i)
        
        if result == label:
            correct_hw += 1
        
        if i % 10 == 0:
            print(f"Progress: {i}/{num_samples}")
    
    print(f"\n=== Batch Test Results ===")
    print(f"Software Accuracy: {100.0 * correct_sw / num_samples:.2f}%")
    print(f"Hardware Accuracy: {100.0 * correct_hw / num_samples:.2f}%")
    print(f"Match Rate: {100.0 * (correct_hw == correct_sw) / num_samples:.2f}%")

def run_hardware_sim(test_id):
    """运行单个硬件仿真"""
    # 调用ModelSim
    cmd = f"vsim -c work.tb_mlp_top -do 'run -all; quit' -gTEST_ID={test_id}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # 解析结果
    for line in result.stdout.split('\n'):
        if 'Predicted:' in line:
            pred = int(line.split(':')[1].split(',')[0].strip())
            return pred
    return -1

if __name__ == '__main__':
    run_batch_test(100)
```

### Day 25-28: 性能优化（人员A负责）

**9. 流水线优化**

```verilog
// rtl/core/cim_core_v3_pipelined.v
module cim_core_v3_pipelined #(
    parameter MAX_INPUT_SIZE = 784,
    parameter MAX_OUTPUT_SIZE = 128,
    parameter MAC_PARALLEL = 16,
    parameter PIPELINE_STAGES = 3
)(
    // 接口同v2...
);

    // 三级流水线：
    // Stage 1: 读取权重和输入
    // Stage 2: MAC计算
    // Stage 3: 累加和激活
    
    // Stage 1寄存器
    reg signed [7:0] stage1_weights [0:MAC_PARALLEL-1];
    reg signed [7:0] stage1_activations [0:MAC_PARALLEL-1];
    reg stage1_valid;
    
    // Stage 2寄存器
    reg signed [15:0] stage2_products [0:MAC_PARALLEL-1];
    reg stage2_valid;
    
    // Stage 3寄存器
    reg signed [31:0] stage3_acc [0:MAC_PARALLEL-1];
    reg stage3_valid;
    
    // Stage 1: 数据读取
    always @(posedge clk) begin
        if (state == COMPUTE) begin
            for (j = 0; j < MAC_PARALLEL; j = j + 1) begin
                stage1_weights[j] <= weight_mem[(output_idx + j) * INPUT_SIZE + input_idx];
                stage1_activations[j] <= input_data[input_idx];
            end
            stage1_valid <= 1;
        end else begin
            stage1_valid <= 0;
        end
    end
    
    // Stage 2: 乘法
    always @(posedge clk) begin
        if (stage1_valid) begin
            for (j = 0; j < MAC_PARALLEL; j = j + 1) begin
                stage2_products[j] <= stage1_weights[j] * stage1_activations[j];
            end
            stage2_valid <= 1;
        end else begin
            stage2_valid <= 0;
        end
    end
    
    // Stage 3: 累加
    always @(posedge clk) begin
        if (stage2_valid) begin
            for (j = 0; j < MAC_PARALLEL; j = j + 1) begin
                if (input_idx == 0)
                    stage3_acc[j] <= {{16{stage2_products[j][15]}}, stage2_products[j]};
                else
                    stage3_acc[j] <= stage3_acc[j] + {{16{stage2_products[j][15]}}, stage2_products[j]};
            end
            stage3_valid <= 1;
        end else begin
            stage3_valid <= 0;
        end
    end
    
    // 输出ReLU（可以再加一级流水线）
    always @(posedge clk) begin
        if (stage3_valid && input_idx == cfg_input_size - 1) begin
            for (j = 0; j < MAC_PARALLEL; j = j + 1) begin
                if (stage3_acc[j][31])  // 负数
                    output_data[output_idx + j] <= 8'd0;
                else if (stage3_acc[j] > 127)  // 溢出
                    output_data[output_idx + j] <= 8'd127;
                else
                    output_data[output_idx + j] <= stage3_acc[j][7:0];
            end
        end
    end

endmodule
```

**10. 资源利用报告**

```python
# sw/utils/resource_report.py
def parse_vivado_report(report_file):
    """解析Vivado综合报告"""
    with open(report_file, 'r') as f:
        content = f.read()
    
    # 提取资源使用
    resources = {
        'LUT': 0,
        'FF': 0,
        'BRAM': 0,
        'DSP': 0
    }
    
    # 解析逻辑（简化）
    print("=== Resource Utilization ===")
    print(f"LUTs: {resources['LUT']}")
    print(f"FFs: {resources['FF']}")
    print(f"BRAMs: {resources['BRAM']}")
    print(f"DSPs: {resources['DSP']}")
    
    # 计算理论性能
    freq_mhz = 100  # 假设100MHz
    ops_per_cycle = 16 * 2  # 16个MAC，每个MAC=1乘+1加
    gops = (freq_mhz * 1e6 * ops_per_cycle) / 1e9
    
    print(f"\n=== Performance ===")
    print(f"Frequency: {freq_mhz} MHz")
    print(f"Throughput: {gops:.2f} GOPS")

if __name__ == '__main__':
    parse_vivado_report('../../build/utilization.rpt')
```

**阶段2验收标准**：
- [ ] 完成3层MLP（784→128→64→10）
- [ ] 100个MNIST样本测试，准确率>85%
- [ ] 硬件结果与软件golden误差<5%
- [ ] 流水线优化，吞吐量提升>30%

---

# 阶段3：SoC集成版（3周）

## Week 5: AXI接口设计

### Day 29-31: AXI-Lite寄存器接口（人员A负责）

**1. AXI-Lite Slave接口**

```verilog
// rtl/soc/cim_axi_lite_slave.v
module cim_axi_lite_slave #(
    parameter C_S_AXI_DATA_WIDTH = 32,
    parameter C_S_AXI_ADDR_WIDTH = 12
)(
    // AXI时钟和复位
    input wire S_AXI_ACLK,
    input wire S_AXI_ARESETN,
    
    // AXI写地址通道
    input wire [C_S_AXI_ADDR_WIDTH-1:0] S_AXI_AWADDR,
    input wire [2:0] S_AXI_AWPROT,
    input wire S_AXI_AWVALID,
    output reg S_AXI_AWREADY,
    
    // AXI写数据通道
    input wire [C_S_AXI_DATA_WIDTH-1:0] S_AXI_WDATA,
    input wire [(C_S_AXI_DATA_WIDTH/8)-1:0] S_AXI_WSTRB,
    input wire S_AXI_WVALID,
    output reg S_AXI_WREADY,
    
    // AXI写响应通道
    output reg [1:0] S_AXI_BRESP,
    output reg S_AXI_BVALID,
    input wire S_AXI_BREADY,
    
    // AXI读地址通道
    input wire [C_S_AXI_ADDR_WIDTH-1:0] S_AXI_ARADDR,
    input wire [2:0] S_AXI_ARPROT,
    input wire S_AXI_ARVALID,
    output reg S_AXI_ARREADY,
    
    // AXI读数据通道
    output reg [C_S_AXI_DATA

