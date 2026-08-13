import argparse
import pandas as pd
import numpy as np
import os

# =========================
# Configuration
# =========================

parser = argparse.ArgumentParser()

parser.add_argument("--input_csv", required=True)
parser.add_argument("--output_csv", required=True)

args = parser.parse_args()

INPUT_CSV = args.input_csv
OUTPUT_CSV = args.output_csv




OUTPUT_DIR = "outputs/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Load Data
# =========================

df = pd.read_csv(INPUT_CSV)

results = []

# =========================
# Process Each Object
# =========================

for object_id in sorted(df["object_id"].unique()):

    obj_df = df[df["object_id"] == object_id].copy()

    obj_df = obj_df.sort_values("frame")

    prev_x = None
    prev_y = None

    for _, row in obj_df.iterrows():

        frame = int(row["frame"])
        x = float(row["x_center"])
        y = float(row["y_center"])

        if prev_x is None:

            velocity = 0.0
            direction = 0.0

        else:

            dx = x - prev_x
            dy = y - prev_y

            velocity = np.sqrt(dx**2 + dy**2)

            direction = np.degrees(
                np.arctan2(dy, dx)
            )

        results.append([
            frame,
            object_id,
            round(x, 2),
            round(y, 2),
            round(velocity, 3),
            round(direction, 2)
        ])

        prev_x = x
        prev_y = y

# =========================
# Save Results
# =========================

output_df = pd.DataFrame(
    results,
    columns=[
        "frame",
        "object_id",
        "x_center",
        "y_center",
        "velocity_px_per_frame",
        "direction_deg"
    ]
)

output_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("✅ Velocity analysis complete.")
print(f"📊 Saved: {OUTPUT_CSV}")
