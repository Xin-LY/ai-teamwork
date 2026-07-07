import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, Bottleneck
from ultralytics.models.yolo.detect import DetectionTrainer


class C2f_v2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(
                self.c,
                self.c,
                shortcut,
                g,
                k=((3, 3), (3, 3)),
                e=1.0
            )
            for _ in range(n)
        )

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


PRUNED_MODEL = None


class PrunedDetectionTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        global PRUNED_MODEL
        return PRUNED_MODEL


# ===================== 640-ratio0.5 剪枝模型微调 =====================

pruned_model_path = r"D:\yolo_industrial_prune\prune_exp\official_prune_640_ratio0.5\pruned_model.pt"
data_path = r"D:\yolo_industrial_prune\prune_exp\NEU-DET\data.yaml"

print("========== 1. 加载 640-ratio0.5 剪枝模型 ==========")
model = YOLO(pruned_model_path)
PRUNED_MODEL = model.model

params = sum(p.numel() for p in PRUNED_MODEL.parameters())

print("========== 2. 检查剪枝模型参数量 ==========")
print("Params:", params)

# 640-ratio0.5 剪枝后参数量约为 1.0616M
# 如果超过 1.5M，很可能误加载了 0.3 剪枝模型或原始 baseline
if params > 1500000:
    raise RuntimeError("当前加载的不像 640-ratio0.5 剪枝模型，参数量超过 1,500,000，停止训练。请检查 pruned_model_path。")

print("========== 3. 开始 640-ratio0.5 微调训练 ==========")

model.train(
    trainer=PrunedDetectionTrainer,
    data=data_path,
    imgsz=640,
    epochs=30,
    batch=4,
    device="cpu",
    workers=0,
    lr0=0.001,
    pretrained=False,
    resume=False,
    project=r"D:\yolo_industrial_prune\prune_exp\official_prune_640_ratio0.5",
    name="finetune_real_pruned_ratio0.5",
    exist_ok=True
)

print("========== 4. 640-ratio0.5 剪枝模型微调完成 ==========")