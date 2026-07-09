from ultralytics import YOLO

def main():
    model = YOLO("yolov8m.pt")
    model.train(
        data="NEU-DET/data.yaml",
        imgsz=640,
        epochs=20,
        batch=8,
        device='cpu',
        workers=0
    )

if __name__ == '__main__':
    main()