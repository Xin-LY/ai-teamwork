from ultralytics import YOLO

def main():
    model = YOLO("yolov8m.pt")
    model.train(
        data="NEU-DET/data.yaml",
        imgsz=416,
        epochs=20,           # 改成20
        batch=16,
        device='cpu',
        workers=0
    )

if __name__ == '__main__':
    main()