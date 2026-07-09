import os
import random
import cv2
import numpy as np

# =========================
# Configuration
# =========================

OUTPUT_DIR = "data/raw/synthetic"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")

IMG_SIZE = 640
NUM_IMAGES = 100
DEBRIS_CLASS_ID = 0

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)


# =========================
# Background Generation
# =========================

def generate_starfield(size):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    num_stars = random.randint(50, 300)

    for _ in range(num_stars):
        x = random.randint(0, size - 1)
        y = random.randint(0, size - 1)
        brightness = random.randint(100, 255)
        img[y, x] = (brightness, brightness, brightness)

    return img


# =========================
# Noise
# =========================

def add_noise(img):
    noise = np.random.normal(loc=0, scale=4, size=img.shape)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


# =========================
# Circular Debris
# =========================

def draw_circular_debris(img):
    radius = random.randint(2, 6)
    
    # Keep safe margins from the boundaries
    cx = random.randint(radius + 10, IMG_SIZE - radius - 10)
    cy = random.randint(radius + 10, IMG_SIZE - radius - 10)
    brightness = random.randint(120, 255)

    # Isolated Local Blur bounding box to optimize performance and prevent background bleeding
    pad = radius + 5
    ymin, ymax = cy - pad, cy + pad
    xmin, xmax = cx - pad, cx + pad

    local_crop = img[ymin:ymax, xmin:xmax].astype(np.float32)
    overlay = np.zeros_like(local_crop)
    
    # Draw circle relative to the cropped frame coordinates
    cv2.circle(overlay, (pad, pad), radius, (brightness, brightness, brightness), -1)
    overlay = cv2.GaussianBlur(overlay, (5, 5), 0)

    # Blend locally back onto main canvas
    img[ymin:ymax, xmin:xmax] = np.clip(local_crop + overlay, 0, 255).astype(np.uint8)

    # Normalized YOLO Coordinates
    x_center = cx / IMG_SIZE
    y_center = cy / IMG_SIZE
    width = (radius * 2) / IMG_SIZE
    height = (radius * 2) / IMG_SIZE

    label = f"{DEBRIS_CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    return img, label


# =========================
# Streak Debris
# =========================

def draw_streak_debris(img):
    length = random.randint(8, 25)
    x1 = random.randint(30, IMG_SIZE - 30)
    y1 = random.randint(30, IMG_SIZE - 30)

    angle = random.uniform(0, 2 * np.pi)
    x2 = int(x1 + length * np.cos(angle))
    y2 = int(y1 + length * np.sin(angle))
    brightness = random.randint(150, 255)

    # Isolate localized boundaries for blurring
    xmin_box, xmax_box = min(x1, x2), max(x1, x2)
    ymin_box, ymax_box = min(y1, y2), max(y1, y2)

    pad = 10
    ymin, ymax = ymin_box - pad, ymax_box + pad
    xmin, xmax = xmin_box - pad, xmax_box + pad

    local_crop = img[ymin:ymax, xmin:xmax].astype(np.float32)
    overlay = np.zeros_like(local_crop)

    # Draw line shifted into local coordinate systems
    cv2.line(
        overlay,
        (x1 - xmin, y1 - ymin),
        (x2 - xmin, y2 - ymin),
        (brightness, brightness, brightness),
        thickness=2
    )
    overlay = cv2.GaussianBlur(overlay, (5, 5), 0)
    
    # Merge local crop back to baseline
    img[ymin:ymax, xmin:xmax] = np.clip(local_crop + overlay, 0, 255).astype(np.uint8)

    # Calculate accurate bounding measurements
    width_px = xmax_box - xmin_box + 6
    height_px = ymax_box - ymin_box + 6

    x_center = ((xmin_box + xmax_box) / 2) / IMG_SIZE
    y_center = ((ymin_box + ymax_box) / 2) / IMG_SIZE
    width = width_px / IMG_SIZE
    height = height_px / IMG_SIZE

    label = f"{DEBRIS_CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    return img, label


# =========================
# Main Generation Loop
# =========================

def main():
    print(f"🚀 Generating {NUM_IMAGES} synthetic space frames...")

    for idx in range(NUM_IMAGES):
        img = generate_starfield(IMG_SIZE)
        labels = []
        num_objects = random.randint(1, 6)

        for _ in range(num_objects):
            if random.random() < 0.7:
                img, label = draw_circular_debris(img)
            else:
                img, label = draw_streak_debris(img)
            labels.append(label)

        img = add_noise(img)

        image_name = f"image_{idx:04d}"
        image_path = os.path.join(IMAGES_DIR, image_name + ".png")
        label_path = os.path.join(LABELS_DIR, image_name + ".txt")

        cv2.imwrite(image_path, img)

        with open(label_path, "w") as f:
            f.write("\n".join(labels))

    print("🎉 Synthetic dataset generation complete and fully optimized.")


if __name__ == "__main__":
    main()
