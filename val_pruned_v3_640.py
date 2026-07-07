import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, Bottleneck


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


model_path = r"D:\yolo_industrial_prune\prune_exp\FINAL_C_PRUNE_RESULT\pruned_finetuned_best.pt"
data_path = r"D:\yolo_industrial_prune\prune_exp\NEU-DET\data.yaml"

model = YOLO(model_path)

params = sum(p.numel() for p in model.model.parameters())
print("========== 剪枝微调模型参数量 ==========")
print("Params:", params)

metrics = model.val(
    data=data_path,
    imgsz=640,
    batch=1,
    device="cpu",
    workers=0,
    project=r"D:\yolo_industrial_prune\prune_exp",
    name="val_pruned_finetuned_v3_640"
)

print("========== 剪枝微调模型 640 验证结果 ==========")
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
print("mAP@0.5:", metrics.box.map50)
print("mAP@0.5:0.95:", metrics.box.map)