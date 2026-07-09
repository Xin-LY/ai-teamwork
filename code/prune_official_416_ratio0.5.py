import os
import torch
import torch.nn as nn
import torch_pruning as tp

from ultralytics import YOLO
from ultralytics.nn.modules import Detect, C2f, Conv, Bottleneck


def infer_shortcut(bottleneck):
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and hasattr(bottleneck, "add") and bottleneck.add


class C2f_v2(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def transfer_weights(c2f, c2f_v2):
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m

    state_dict = c2f.state_dict()
    state_dict_v2 = c2f_v2.state_dict()

    old_weight = state_dict["cv1.conv.weight"]
    half_channels = old_weight.shape[0] // 2

    state_dict_v2["cv0.conv.weight"] = old_weight[:half_channels]
    state_dict_v2["cv1.conv.weight"] = old_weight[half_channels:]

    for bn_key in ["weight", "bias", "running_mean", "running_var"]:
        old_bn = state_dict[f"cv1.bn.{bn_key}"]
        state_dict_v2[f"cv0.bn.{bn_key}"] = old_bn[:half_channels]
        state_dict_v2[f"cv1.bn.{bn_key}"] = old_bn[half_channels:]

    for key in state_dict:
        if not key.startswith("cv1."):
            state_dict_v2[key] = state_dict[key]

    for attr_name in dir(c2f):
        attr_value = getattr(c2f, attr_name)
        if not callable(attr_value) and "_" not in attr_name:
            setattr(c2f_v2, attr_name, attr_value)

    c2f_v2.load_state_dict(state_dict_v2)


def replace_c2f_with_c2f_v2(module):
    for name, child_module in module.named_children():
        if isinstance(child_module, C2f):
            shortcut = infer_shortcut(child_module.m[0])

            c2f_v2 = C2f_v2(
                child_module.cv1.conv.in_channels,
                child_module.cv2.conv.out_channels,
                n=len(child_module.m),
                shortcut=shortcut,
                g=child_module.m[0].cv2.conv.groups,
                e=child_module.c / child_module.cv2.conv.out_channels,
            )

            transfer_weights(child_module, c2f_v2)
            setattr(module, name, c2f_v2)

        else:
            replace_c2f_with_c2f_v2(child_module)


# ===================== 主流程：416 输入尺寸，剪枝比例 0.5 =====================

BASE_DIR = r"D:\yolo_industrial_prune\prune_exp"

# 416 baseline 权重，不要改成 640 的
ORIG_WEIGHT = os.path.join(BASE_DIR, "base_weights", "yolov8n_416_baseline.pt")

# 输入尺寸：416
RESIZE_SIZE = 416

# 剪枝比例：0.5
PRUNE_RATIO = 0.5


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("使用设备:", device)

    print("========== 1. 加载模型 ==========")
    model = YOLO(ORIG_WEIGHT)
    net = model.model

    print("========== 2. 替换 C2f 为 C2f_v2 ==========")
    replace_c2f_with_c2f_v2(net)

    for p in net.parameters():
        p.requires_grad = True

    net.eval()
    net.to(device)

    example_inputs = torch.randn(1, 3, RESIZE_SIZE, RESIZE_SIZE).to(device)

    print("========== 3. 设置忽略层（Detect） ==========")
    ignored_layers = []

    for m in net.modules():
        if isinstance(m, Detect):
            ignored_layers.append(m)

    print("忽略 Detect 层数量:", len(ignored_layers))

    print("========== 4. 构建剪枝器 ==========")
    imp = tp.importance.GroupMagnitudeImportance(p=2)

    pruner = tp.pruner.GroupNormPruner(
        net,
        example_inputs=example_inputs,
        importance=imp,
        iterative_steps=1,
        ch_sparsity=PRUNE_RATIO,
        ignored_layers=ignored_layers,
    )

    print("========== 5. 统计剪枝前 ==========")
    base_macs, base_params = tp.utils.count_ops_and_params(net, example_inputs)

    print(f"剪枝前参数量: {base_params / 1e6:.4f} M")
    print(f"剪枝前MACs : {base_macs / 1e9:.4f} G")

    print("========== 6. 开始剪枝 ==========")
    pruner.step()

    print("========== 7. 统计剪枝后 ==========")
    pruned_macs, pruned_params = tp.utils.count_ops_and_params(net, example_inputs)

    param_reduce = (1 - pruned_params / base_params) * 100
    macs_reduce = (1 - pruned_macs / base_macs) * 100

    print(f"剪枝后参数量: {pruned_params / 1e6:.4f} M")
    print(f"剪枝后MACs : {pruned_macs / 1e9:.4f} G")
    print(f"参数压缩比例: {param_reduce:.2f} %")
    print(f"MACs压缩比例: {macs_reduce:.2f} %")

    print("========== 8. 保存剪枝模型 ==========")

    SAVE_DIR = os.path.join(BASE_DIR, "official_prune_416_ratio0.5")
    os.makedirs(SAVE_DIR, exist_ok=True)

    save_path = os.path.join(SAVE_DIR, "pruned_model.pt")
    torch.save({"model": net}, save_path)

    print(f"剪枝模型已保存到: {save_path}")

    print("========== 9. 剪枝结果判断 ==========")

    if pruned_params >= base_params:
        print("结果：剪枝没有真正生效。请检查剪枝器设置。")
    else:
        print("结果：剪枝已真正生效。")

    print("========== 本次实验配置 ==========")
    print(f"输入尺寸: {RESIZE_SIZE}")
    print(f"剪枝比例: {PRUNE_RATIO}")
    print(f"保存目录: {SAVE_DIR}")


if __name__ == "__main__":
    main()