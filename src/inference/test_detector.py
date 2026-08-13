from ultralytics import YOLO

model = YOLO("models/best.pt")

results = model.predict(
    source="data/test_videos/video_001.mp4",
    save=True,
    conf=0.25
)
