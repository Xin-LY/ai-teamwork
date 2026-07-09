from ultralytics import YOLO
if __name__ == '__main__':
    # 基线YOLOv8m 416分辨率 50轮训练
    model = YOLO("yolov8m.pt")
    model.train(
        data="./NEU-DET/data.yaml",
        imgsz=416,
        epochs=50,
        batch=8,
        device=0,
        amp=False,
        workers=0
    )
