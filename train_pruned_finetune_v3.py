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


pruned_model_path = r"D:\yolo_industrial_prune\prune_exp\official_prune_416_ratio0.3\pruned_model.pt"
data_path = r"D:\yolo_industrial_prune\prune_exp\NEU-DET\data.yaml"

model = YOLO(pruned_model_path)
PRUNED_MODEL = model.model

params = sum(p.numel() for p in PRUNED_MODEL.parameters())
print("========== V3 正式加载剪枝模型参数量 ==========")
print("Params:", params)

if params > 2000000:
    raise RuntimeError("当前加载的不是剪枝模型，参数量超过 2,000,000，停止训练。")

model.train(
    trainer=PrunedDetectionTrainer,
    data=data_path,
    imgsz=416,
    epochs=30,
    batch=8,
    device="cpu",
    workers=0,
    lr0=0.001,
    pretrained=False,
    resume=False,
    project=r"D:\yolo_industrial_prune\prune_exp\official_prune_416_ratio0.3",
    name="finetune_real_pruned_v3",
    exist_ok=True
)