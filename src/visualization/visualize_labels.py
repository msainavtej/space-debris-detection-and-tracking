import os
import random
import cv2

# --- Configuration ---
DATA_DIR = "data/raw/synthetic"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_DIR = os.path.join(DATA_DIR, "labels")

# Target directory to review verified frames
OUTPUT_VERIFY_DIR = "data/raw/synthetic/visual_checks"
os.makedirs(OUTPUT_VERIFY_DIR, exist_ok=True)

def verify_and_draw():
    # Gather all image files
    all_images = [f for f in os.listdir(IMAGES_DIR) if f.endswith('.png')]
    
    if not all_images:
        print("❌ No images found. Did you run the generation script first?")
        return

    # Select up to 25 random samples for visual inspection
    num_samples = min(25, len(all_images))
    samples = random.sample(all_images, num_samples)
    
    print(f"🧐 Processing {num_samples} random images for visual verification...")

    for img_name in samples:
        base_name = os.path.splitext(img_name)[0]
        label_name = f"{base_name}.txt"
        
        img_path = os.path.join(IMAGES_DIR, img_name)
        label_path = os.path.join(LABELS_DIR, label_name)
        
        if not os.path.exists(label_path):
            print(f"⚠️ Label missing for {img_name}, skipping.")
            continue
            
        # Load the generated image
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        # Read corresponding YOLO labels
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
                
            class_id, x_center, y_center, bbox_w, bbox_h = map(float, parts)
            
            # --- Convert YOLO formats back to exact Absolute Pixel bounds ---
            # Center values
            cx_px = int(x_center * w)
            cy_px = int(y_center * h)
            # Width/Height values
            w_px = int(bbox_w * w)
            h_px = int(bbox_h * h)
            
            # Extract top-left and bottom-right corner metrics
            x1 = int(cx_px - (w_px / 2))
            y1 = int(cy_px - (h_px / 2))
            x2 = int(cx_px + (w_px / 2))
            y2 = int(cy_px + (h_px / 2))
            
            # Draw a bright high-contrast green rectangle around the debris object
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            # Optional: Add small label text above bounding edge
            cv2.putText(img, f"cls:{int(class_id)}", (x1, max(y1 - 3, 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        # Export processed verification frame to disk
        output_path = os.path.join(OUTPUT_VERIFY_DIR, f"verify_{img_name}")
        cv2.imwrite(output_path, img)

    print(f"✅ Visual check complete. Open the directory '{OUTPUT_VERIFY_DIR}/' to inspect your 25 verified images!")

if __name__ == "__main__":
    verify_and_draw()
