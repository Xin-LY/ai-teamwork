# train_distill_640.py
from ultralytics import YOLO
import torch
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils import DEFAULT_CFG

# ---------- 1. 定义蒸馏训练器 ----------
class KDDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides=None, teacher=None, student=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        if 'model' not in overrides or overrides['model'] is None:
            if hasattr(student, 'ckpt_path') and student.ckpt_path:
                overrides['model'] = student.ckpt_path
            else:
                overrides['model'] = 'models/baseline_640.pt'
        
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self.model = student.model
        self.teacher = teacher.model
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def loss(self, batch, preds):
        criterion = v8DetectionLoss(self.model)
        det_loss, loss_items = criterion(preds, batch)

        with torch.no_grad():
            teacher_preds = self.teacher(batch['img'])

        temperature = 6.0
        alpha = 0.7

        s_logits = preds[0] / temperature
        t_logits = teacher_preds[0] / temperature

        kd_loss = F.kl_div(
            F.log_softmax(s_logits, dim=-1),
            F.softmax(t_logits, dim=-1),
            reduction='batchmean'
        ) * (temperature ** 2)

        total_loss = det_loss + alpha * kd_loss
        return total_loss, loss_items


# ---------- 2. 运行640蒸馏 ----------
if __name__ == '__main__':
    print("=" * 50)
    print("开始 640 分辨率蒸馏训练...")
    print("=" * 50)

    # ✅ 教师路径改为 train-6（你刚训练完的640教师）
    teacher_model = YOLO('runs/detect/train-6/weights/best.pt')
    print("教师模型加载成功！")

    # 学生模型（A同学的640基线）
    student_model = YOLO('models/baseline_640.pt')
    print("学生模型加载成功！")

    args = {
        'data': 'NEU-DET/data.yaml',
        'epochs': 20,
        'imgsz': 640,            # 640分辨率
        'batch': 8,              # 640用8
        'device': 'cpu',
        'workers': 0,
        'model': 'models/baseline_640.pt',
    }

    trainer = KDDetectionTrainer(
        cfg=DEFAULT_CFG,
        overrides=args,
        teacher=teacher_model,
        student=student_model
    )

    trainer.train()

    print("=" * 50)
    print("640蒸馏训练完成！")
    print("模型保存在 runs/detect/train/weights/best.pt")
    print("=" * 50)