# 🛰️ Space Debris Detection and Tracking

An end-to-end **Computer Vision** and **Machine Learning** project for detecting, tracking, and analyzing space debris in orbital imagery and videos.

This project combines **synthetic dataset generation**, **object detection**, **multi-object tracking**, **velocity analysis**, and **interactive visualization** into a single pipeline. The goal is to demonstrate how AI and Computer Vision techniques can be used to monitor debris objects in space environments.

---

## 🌌 Project Overview

Space debris poses a significant challenge to satellites, spacecraft, and future space missions. Even small debris fragments can cause severe damage due to their extremely high orbital velocities.

This project aims to build a complete **Space Debris Detection and Tracking System** capable of:

* Detecting debris objects from images and videos
* Tracking multiple debris objects across frames
* Assigning persistent object IDs
* Computing motion statistics
* Visualizing trajectories and object behavior
* Presenting results through an interactive dashboard

---

## ✨ Features

### Synthetic Dataset Generation

* Generates synthetic space imagery
* Creates starfield backgrounds
* Simulates debris objects as points and streaks
* Automatically generates annotations

### Synthetic Video Generation

* Simulates moving debris objects
* Generates ground-truth tracking data
* Supports multiple debris objects per scene

### Object Detection

* Custom YOLOv8 model training
* Debris localization using bounding boxes
* Detection confidence scoring

### Multi-Object Tracking

* ByteTrack-based tracking pipeline
* Persistent object IDs across frames
* Tracking CSV export

### Motion Analysis

* Velocity computation
* Object trajectory extraction

### Interactive Dashboard

* Built using Gradio
* Interactive trajectory visualization using Plotly
* Detection and tracking statistics
* Video processing interface

---

## 🏗️ Project Architecture

```text
Synthetic Data Generation
            │
            ▼
YOLO Dataset Creation
            │
            ▼
YOLOv8 Training
            │
            ▼
Object Detection
            │
            ▼
ByteTrack Tracking
            │
            ▼
Tracking CSV Export
            │
            ▼
Pixel based Velocity Analysis
            │
            ▼
Interactive Dashboard
```

---

## 🛠️ Tech Stack

| Category              | Technology           |
| --------------------- | -------------------- |
| Programming Language  | Python               |
| Computer Vision       | OpenCV               |
| Deep Learning         | YOLOv8 (Ultralytics) |
| Multi-Object Tracking | ByteTrack            |
| Data Analysis         | Pandas, NumPy        |
| Visualization         | Plotly               |
| Frontend              | Gradio               |
| Training Environment  | Google Colab         |
| GPU                   | NVIDIA Tesla T4      |

---

## 📂 Dataset

### Version 1 Dataset

The first version of the project uses a fully synthetic dataset generated programmatically.

#### Dataset Characteristics

* Synthetic orbital scenes
* Random starfield backgrounds
* Circular debris objects
* Motion streak debris objects
* Automatic annotation generation
* Synthetic video sequences with ground-truth tracking data

### Dataset Structure

```text
data/
├── raw/
│   ├── synthetic/
│   └── synthetic_videos/
│
└── processed/
    └── yolo_dataset/
        ├── train/
        ├── val/
        └── test/
```

---

## 🤖 Model Training

The debris detector was trained using **YOLOv8** on the generated synthetic dataset.

### Training Pipeline

1. Generate synthetic images and annotations
2. Convert data into YOLO format
3. Split dataset into:

   * Train
   * Validation
   * Test
4. Train YOLOv8 model
5. Evaluate model performance

---

## 📊 Results

### Final Evaluation Metrics

| Metric    | Value |
| --------- | ----- |
| Precision | 0.895 |
| Recall    | 0.900 |
| mAP@50    | 0.926 |
| mAP@50-95 | 0.659 |

These results demonstrate strong detection performance on the synthetic dataset.

---

## 🎯 Tracking Pipeline

After detection, objects are tracked using **ByteTrack**.

Tracking outputs include:

* Object IDs
* Frame number
* Position coordinates
* Bounding box dimensions
* Detection confidence

### Example Output

```csv
frame,object_id,x_center,y_center,width,height,confidence
```

---

## 📈 Pixel based Velocity Analysis

The tracking output is further analyzed to estimate object motion.

### Metrics Computed

* Velocity (pixels/frame)
* Object trajectories

### Example Output

```csv
frame,object_id,x_center,y_center,velocity_px_per_frame
```

---

## 🖥️ Dashboard

The project includes an interactive dashboard built using **Gradio**.

### Dashboard Features

* Video upload
* Detection visualization
* Multi-object tracking display
* Velocity statistics
* Trajectory visualization
* Interactive analytics

---

## 📁 Project Structure

```text
space-debris-detection-and-tracking/
│
├── dashboard/app.py
├── models/best.pt
├── sample/ (sample input and output videos)
│
├── src/
│   ├── dataset_builder/
│   ├── training/
|   ├── inference/
|   ├── synthetic_data
|   ├── synthetic_video
│   ├── tracking/
│   ├── visualization/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---
## Instructions on how to use:-

### Clone the repository
 
 git clone https://github.com/msainavtej/space-debris-detection-and-tracking.git
 
 cd space-debris-detection-and-tracking 

### Create a virtual environment
 
 python3 -m venv venv
 
 source venv/bin/activate

### Install requirements
 
 pip install -r requirements.txt

### Open the Dashboard
 
 python3 dashboard/app.py
 
 (Copy the local URL and paste it in your browser)



## 🔮 Future Improvements

### Version 2

* Integrate real-world space imagery datasets
* Combine synthetic and real training data
* Improve small object detection performance

## 📚 Key Learnings

Through this project, I gained practical experience in:

* Synthetic dataset generation
* Object detection model training
* Multi-object tracking
* Computer vision pipelines
* Data analysis and visualization
* End-to-end AI system development
* Dashboard development and deployment

---

## 👨‍💻 Author

**Sai Navtej**

Aspiring Computer Science Engineer • Computer Vision Enthusiast • AI Developer

Interested in:

* Computer Vision
* Machine Learning
* Space Technology
* Data Science
* Robotics and Hardware

---
