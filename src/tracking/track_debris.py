from ultralytics import YOLO

model = YOLO("models/best.pt")

results = model.track(
    source="data/test_videos/video_001.mp4",
    tracker="bytetrack.yaml",
    save=True,
    persist=True,
    conf=0.25
)
