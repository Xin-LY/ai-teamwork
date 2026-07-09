# ai-teamwork
# 工业缺陷检测模型轻量化与知识蒸馏系统

## 项目简介

本项目为四川大学电子信息学院“人工智能基础”课程期中团队项目（题目五：轻量级实时目标检测优化，5C组 工业质检场景）。我们以 **NEU-DET 钢材表面缺陷检测数据集** 为场景，基于 **YOLOv8** 框架，探索了在保持模型轻量化的前提下提升检测精度的技术路径。

### 核心工作

- 训练 YOLOv8m 教师模型与 YOLOv8n 学生模型基线
- 实现基于 **知识蒸馏 (Knowledge Distillation)** 的精度提升方案，使用 KL 散度作为蒸馏损失
- 实现基于 **L1 范数的结构化通道剪枝 (Channel Pruning)**，在不同剪枝比例下分析压缩效果
- 在 416×416 和 640×640 两种输入分辨率下完成全流程实验与对比分析

### 主要结果

| 分辨率   |        模型        | mAP@0.5 | 参数量 | FPS   |
|-------- |    ------          |---------|--------|-----  |
| 640×640 | 基线 (yolov8n)     | 0.835   | 3.0M   | 20.34 |
| 640×640 | 教师 (yolov8m)     | 0.844   | 25.8M  | 2.26  |
| 640×640 | 蒸馏后 (yolov8n)   | 0.836   | 3.0M   | 21.49 |
| 640×640 | 剪枝 0.5 (yolov8n) | 0.804   | 1.06M  | 34.96 |
| 416×416 | 剪枝 0.5 (yolov8n) | 0.783   | 1.06M  | 50.94 |

**核心结论**：640×640 分辨率下，蒸馏后模型精度（0.836）超越基线（0.835），接近教师模型（0.844），参数量保持 3.0M 不变；剪枝后模型参数量降低 64.8%，FPS 最高提升 71.8%，验证了知识蒸馏与结构化剪枝在工业缺陷检测中的有效性。


## 数据集说明

本实验使用 **NEU-DET** 钢材表面缺陷检测数据集，由东北大学宋克臣团队制作。

- **总图片数**：1800 张（训练集 1770 张，验证集 30 张）
- **类别数**：6 类
  - crazing（裂纹）
  - inclusion（夹杂物）
  - patches（斑点）
  - pitted_surface（麻面）
  - rolled-in_scale（轧入氧化皮）
  - scratches（划痕）


## 环境配置

### 系统要求

- Windows / Linux / macOS
- Python 3.8 或更高版本
- 建议 8GB 以上内存

### 安装步骤

#### 1. 克隆或下载本项目

bash
git clone <repository-url>
cd yolo_project# report


#### 2. 创建并激活 conda 虚拟环境
bash
conda create -n yolov8 python=3.8 -y
conda activate yolov8

#### 3. 配置 pip 国内镜像源（加速下载）
bash
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

#### 4. 安装 PyTorch（CPU 版）
bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch
如有 NVIDIA GPU，可替换为 GPU 版：

bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

#### 5. 安装 Ultralytics
bash
pip install ultralytics

#### 6. 验证安装
bash
yolo predict model=yolov8n.pt source=https://ultralytics.com/images/bus.jpg

### 目录结构
text
yolo_project/
│
├── NEU-DET/                          # 数据集（需自行放置）
│   ├── data.yaml                     # 数据集配置文件
│   ├── train/
│   │   └── images/                   # 训练集图片
│   └── valid/
│       └── images/                   # 验证集图片
│
├── runs/                             # 训练与验证输出（自动生成）
│   └── detect/
│       ├── train/                    # 教师模型训练结果
│       ├── train-*/                  # 蒸馏/剪枝训练结果
│       └── ...
│
├── train_teacher.py                  # 教师模型训练脚本
├── train_distill.py                  # 蒸馏训练脚本
├── prune.py                          # 剪枝脚本
└── README.md                         # 本文件
注意：模型权重文件（.pt）体积较大，不纳入版本管理。各脚本训练完成后会自动保存至 runs/ 目录

### 运行方法

#### 1. 准备数据集
将 NEU-DET 数据集解压至项目根目录下的 NEU-DET/ 文件夹，确保 data.yaml 中的路径配置正确：

yaml
path: ./NEU-DET
train: train/images
val: valid/images
nc: 6
names: ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

#### 2. 训练教师模型（YOLOv8m）
bash
python train_teacher.py

#### 3. 训练学生基线（YOLOv8n）
python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='NEU-DET/data.yaml', imgsz=416, epochs=50, device='cpu')

#### 4. 知识蒸馏训练
bash
python train_distill.py

#### 5. 结构化剪枝
bash
python prune.py

#### 6. 评估模型精度
bash
# 评估任意模型
yolo val model=runs/detect/train/weights/best.pt data=NEU-DET/data.yaml imgsz=416 plots=False
实验结果汇总
模型	输入尺寸	mAP@0.5	mAP@0.5:0.95	FLOPs(G)	Params(M)	FPS
Baseline	416×416	0.827	0.502	1.71	3.01	36.65
Baseline	640×640	0.835	0.497	4.05	3.01	20.34
Distilled	416×416	0.784	0.477	1.71	3.01	27.30
Distilled	640×640	0.836	0.497	4.05	3.01	21.49
Pruned 0.3	416×416	0.754	0.436	1.04	1.68	43.01
Pruned 0.5	416×416	0.783	0.440	0.73	1.06	50.94
Pruned 0.3	640×640	0.794	0.483	2.46	1.68	22.02
Pruned 0.5	640×640	0.804	0.451	1.72	1.06	34.96
# 核心结论
输入分辨率权衡：640×640 精度更高（+0.008 mAP），但 FPS 约为 416×416 的 1/3，需按场景取舍。

知识蒸馏有效：640×640 下蒸馏后 mAP（0.836）超越基线（0.835），参数量不变，验证了“零成本”精度提升的可行性。

剪枝显著加速：0.5 剪枝使参数量降低 64.8%，FPS 最高提升 71.8%，是边缘部署的最有效手段。

综合最优：精度优先选 640×640 蒸馏模型；速度优先选 416×416 剪枝 0.5；平衡方案选 640×640 剪枝 0.5。

### 组员分工
成员	主要职责
刘宇欣	环境搭建、NEU-DET 数据集准备与预处理、416/640 双分辨率基线模型训练
刘紫涵	知识蒸馏全流程设计与实现（416/640 双分辨率蒸馏实验、超参数调优）
余玉媛	结构化通道剪枝（0.3/0.5 剪枝比例实验、剪枝后微调恢复）
王芯	评估框架搭建（mAP/FLOPs/参数量/FPS 计算、混淆矩阵与可视化分析）
李钰	报告撰写、PPT 制作

### 项目信息
课程：人工智能基础（2025-2026 春季学期）

授课教师：王君教授（四川大学电子信息学院）

题目：题目五 - 轻量级实时目标检测优化（5C 组：工业场景缺陷检测）

# 参考资料
Ultralytics YOLOv8 官方文档

NEU-DET 数据集

Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. NIPS 2014 Workshop.

Liu, Z., et al. (2017). Learning Efficient Convolutional Networks through Network Slimming. ICCV 2017.

