# 🧠 Polygon Annotation Automation for Multi-Class Video Datasets

This project presents a complete, production-ready pipeline that automates polygon annotation on real-world video datasets using modern deep learning models. It bridges object detection and instance segmentation to create accurate, structured annotations from raw video — solving one of the most time-intensive bottlenecks in computer vision: **annotating dense, multi-class frames**.

---

## 🚗 Use Case

Built for scenarios like:
- Urban driving datasets
- Smart city surveillance
- Autonomous vehicle pre-processing
- Scalable labeling workflows for ML model training

---

## 📂 Dataset Overview

**Dataset**: Urban driving footage (from [Kaggle](https://www.kaggle.com/datasets/robikscube/driving-video-with-object-tracking))  
- High-resolution (1280x720) video  
- 4+ object classes: car, pedestrian, motorcycle, bus, truck  
- Visually distinct objects with consistent motion and perspective  
- Ideal for fine-grained polygon-based annotation

---

## 🎯 Key Features

- 🎯 **YOLOv8** for fast, multi-class object detection
- ✂️ **SAM (Segment Anything Model)** for generating pixel-accurate masks
- 🔁 **Frame stride sampling** for compute efficiency
- 🌀 **Automatic rotation correction** to fix sideways input videos
- 📄 **Per-frame JSON polygon output**
- 🎥 **Stitched annotated video** with overlayed masks + class labels

---

## 🧠 Pipeline Overview

1. **Preprocessing**
   - Detects and fixes orientation issues
   - Samples every 5th frame to reduce load

2. **Detection**
   - Uses YOLOv8n (lightweight, fast) to detect object classes

3. **Segmentation**
   - Uses Meta AI's SAM to convert boxes into polygon masks

4. **Output**
   - Saves polygon data in JSON format
   - Saves annotated frames + creates a stitched output video

---

## 🛠️ Technologies Used

- `YOLOv8` (Ultralytics)
- `Segment Anything Model (SAM)`
- `PyTorch`, `NumPy`, `OpenCV`, `Supervision`
- `ffmpeg` (for video stitching)

---

## 📦 Output Samples

- ✅ `output_annotated_video.mp4`  
  A stitched, annotated video showing real-time polygon tracking.
## 🖼 Pipeline Diagram  
<img width="500" height="460" alt="image" src="https://github.com/user-attachments/assets/c9e76e0e-396c-4f28-8e7c-04c1a6688fd5" />    


[▶️ Click to watch annotated output video on Google Drive](https://drive.google.com/file/d/1hKKO44qK-TR8NXGtdXRoL6c1C1ONvR7V/view?usp=drive_link)



