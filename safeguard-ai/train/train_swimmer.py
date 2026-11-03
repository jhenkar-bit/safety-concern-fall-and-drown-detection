"""Fine-tune YOLO to detect swimmers in YOUR pool (the biggest single accuracy gain for drowning).

1. Record 1-2 h of footage from the real camera position, different times of day (glare!).
2. Label with Roboflow / CVAT / Label Studio. Class 0 = swimmer (whole visible body or head+shoulders).
   Include hard negatives: reflections, floats, shadows, lane ropes, people on the deck.
3. Export in YOLO format and point train/swimmer.yaml at it, then:  python -m train.train_swimmer
4. Copy runs/detect/swimmer/weights/best.pt -> weights/swimmer.pt and in config.yaml set
   pool_weights: weights/swimmer.pt   and   pool_classes: null
"""
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11s.pt")
    model.train(data="train/swimmer.yaml", epochs=100, imgsz=960, batch=16, name="swimmer",
                hsv_h=0.02, hsv_v=0.5, degrees=5, mosaic=1.0, fliplr=0.5)  # strong colour aug = glare robustness
    print(model.val())
