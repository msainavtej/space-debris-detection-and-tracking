from ultralytics import YOLO
import argparse
import cv2
import csv
import os

# ==================================
# Configuration
# ==================================

MODEL_PATH = "models/best.pt"


parser = argparse.ArgumentParser()

parser.add_argument("--input", required=True)
parser.add_argument("--output_video", required=True)
parser.add_argument("--output_csv", required=True)

args = parser.parse_args()

VIDEO_PATH = args.input
OUTPUT_VIDEO_PATH = args.output_video
CSV_PATH = args.output_csv



OUTPUT_DIR = "outputs/tracking"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================
# Load Model
# ==================================

model = YOLO(MODEL_PATH)

# ==================================
# Open Video
# ==================================

cap = cv2.VideoCapture(VIDEO_PATH)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(
    OUTPUT_VIDEO_PATH,
    fourcc,
    fps,
    (width, height)
)

# ==================================
# CSV Setup
# ==================================

csv_file = open(CSV_PATH, mode="w", newline="")
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "frame",
    "object_id",
    "x_center",
    "y_center",
    "width",
    "height",
    "confidence"
])

# ==================================
# Tracking Loop
# ==================================

frame_number = 0

print("🚀 Starting tracking pipeline...")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.25,
        verbose=False
    )

    annotated_frame = results[0].plot()

    boxes = results[0].boxes

    if boxes is not None and boxes.id is not None:

        track_ids = boxes.id.cpu().numpy().astype(int)

        xywh = boxes.xywh.cpu().numpy()

        confidences = boxes.conf.cpu().numpy()

        for track_id, box, conf in zip(track_ids, xywh, confidences):

            x_center, y_center, w, h = box

            csv_writer.writerow([
                frame_number,
                int(track_id),
                round(float(x_center), 2),
                round(float(y_center), 2),
                round(float(w), 2),
                round(float(h), 2),
                round(float(conf), 4)
            ])

    video_writer.write(annotated_frame)

    frame_number += 1

cap.release()
video_writer.release()
csv_file.close()

print("✅ Tracking complete.")
print(f"🎥 Saved video: {OUTPUT_VIDEO_PATH}")
print(f"📊 Saved CSV: {CSV_PATH}")
