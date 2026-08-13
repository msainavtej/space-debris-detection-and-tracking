import os
import csv
import random
import cv2
import numpy as np

# =========================
# Configuration
# =========================

OUTPUT_ROOT = "data/raw/synthetic_videos"
IMG_SIZE = 640
NUM_VIDEOS = 100
FRAMES_PER_VIDEO = 100
FPS = 20

# =========================
# Background Helpers
# =========================

def generate_starfield(size):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    num_stars = random.randint(50, 250)
    for _ in range(num_stars):
        x = random.randint(0, size - 1)
        y = random.randint(0, size - 1)
        brightness = random.randint(100, 255)
        img[y, x] = (brightness, brightness, brightness)
    return img

def add_noise(img):
    noise = np.random.normal(loc=0, scale=4, size=img.shape)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

# =========================
# Kinematics Object Logic
# =========================

class DebrisObject:
    def __init__(self, obj_id):
        self.id = obj_id
        
        # Unique start state coordinates
        self.x = float(random.randint(40, IMG_SIZE - 40))
        self.y = float(random.randint(40, IMG_SIZE - 40))
        
        # Unique velocity profiles per trajectory run (-5 to +5 px/frame)
        self.vx = random.uniform(-5, 5)
        self.vy = random.uniform(-5, 5)
        if abs(self.vx) < 0.6: self.vx = 2.0 * random.choice([-1, 1])
        if abs(self.vy) < 0.6: self.vy = 2.0 * random.choice([-1, 1])

        # Dynamic appearance signatures
        self.type = "circle" if random.random() < 0.6 else "streak"
        self.radius = random.randint(2, 6)
        self.brightness = random.randint(110, 255)
        self.streak_len = random.randint(8, 22)

        # Pre-calculate spatial box bounds for future YOLO tracking metrics
        if self.type == "circle":
            self.w_px = float(self.radius * 2)
            self.h_px = float(self.radius * 2)
        else:
            norm = np.hypot(self.vx, self.vy)
            dx = abs((self.vx / norm) * self.streak_len) if norm > 0 else 2
            dy = abs((self.vy / norm) * self.streak_len) if norm > 0 else 2
            self.w_px = float(dx + 4)  # Include margin padding 
            self.h_px = float(dy + 4)

    def update_position(self):
        self.x += self.vx
        self.y += self.vy

    def render(self, frame):
        cx, cy = int(self.x), int(self.y)
        if cx < -30 or cx > IMG_SIZE + 30 or cy < -30 or cy > IMG_SIZE + 30:
            return frame

        pad = int(max(self.w_px, self.h_px) + 10)
        ymin, ymax = max(0, cy - pad), min(IMG_SIZE, cy + pad)
        xmin, xmax = max(0, cx - pad), min(IMG_SIZE, cx + pad)
        
        if (ymax - ymin) <= 0 or (xmax - xmin) <= 0:
            return frame

        local_crop = frame[ymin:ymax, xmin:xmax].astype(np.float32)
        overlay = np.zeros_like(local_crop)
        local_cx, local_cy = cx - xmin, cy - ymin

        if self.type == "circle":
            cv2.circle(overlay, (local_cx, local_cy), self.radius, (self.brightness, self.brightness, self.brightness), -1)
        else:
            norm = np.hypot(self.vx, self.vy)
            dx = int(- (self.vx / norm) * self.streak_len) if norm > 0 else 0
            dy = int(- (self.vy / norm) * self.streak_len) if norm > 0 else 0
            cv2.line(overlay, (local_cx, local_cy), (local_cx + dx, local_cy + dy), 
                     (self.brightness, self.brightness, self.brightness), thickness=2)

        overlay = cv2.GaussianBlur(overlay, (5, 5), 0)
        frame[ymin:ymax, xmin:xmax] = np.clip(local_crop + overlay, 0, 255).astype(np.uint8)
        return frame

# =========================
# Main Execution Loop
# =========================

def main():
    print(f"🚀 Scaling Dataset Pipeline to {NUM_VIDEOS} Runs...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    total_trajectories = 0

    for v_idx in range(1, NUM_VIDEOS + 1):
        video_dir_name = f"sequence_{v_idx:03d}"
        video_folder = os.path.join(OUTPUT_ROOT, video_dir_name)
        os.makedirs(video_folder, exist_ok=True)

        video_path = os.path.join(video_folder, f"{video_dir_name}.mp4")
        csv_path = os.path.join(video_folder, f"{video_dir_name}_ground_truth.csv")

        # Randomized initialization conditions per video sequence
        base_background = generate_starfield(IMG_SIZE)
        num_objects = random.randint(3, 8)
        debris_list = [DebrisObject(obj_id=i) for i in range(1, num_objects + 1)]
        total_trajectories += num_objects

        video_writer = cv2.VideoWriter(video_path, fourcc, FPS, (IMG_SIZE, IMG_SIZE))
        csv_rows = []

        for frame_idx in range(FRAMES_PER_VIDEO):
            frame = base_background.copy()

            for obj in debris_list:
                # Log state if tracking target is observable within canvas bounds
                if 0 <= obj.x <= IMG_SIZE and 0 <= obj.y <= IMG_SIZE:
                    csv_rows.append([
                        frame_idx, obj.id, 
                        round(obj.x, 2), round(obj.y, 2), 
                        round(obj.w_px, 2), round(obj.h_px, 2)
                    ])

                frame = obj.render(frame)
                obj.update_position()

            frame = add_noise(frame)
            video_writer.write(frame)

        video_writer.release()

        # Write Upgraded Ground Truth CSV
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'id', 'x', 'y', 'width', 'height'])
            writer.writerows(csv_rows)

        if v_idx % 10 == 0 or v_idx == 1:
            print(f" └─ Generated {v_idx}/{NUM_VIDEOS} runs ({video_dir_name})")

    print(f"\n🎉 V2 Data Generation Finished!")
    print(f"📊 Extracted {NUM_VIDEOS * FRAMES_PER_VIDEO} raw video frames across {total_trajectories} unique trajectories.")
    print(f"📂 Output Root: {OUTPUT_ROOT}/")

if __name__ == "__main__":
    main()
