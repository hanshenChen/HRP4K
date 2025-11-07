# HRP4K
This repository contains the official code for the paper: "A high-resolution perspective-view road image dataset for pothole detection".

## 📖 Overview
**HRP4K** is a **high-resolution, perspective-view pothole detection dataset** designed to advance automated infrastructure monitoring and computer-vision-based road-surface analysis.  
The dataset provides **4,003 4K-resolution images** containing **7,215 annotated pothole instances** captured from real-world driving scenes across **1,100 km** of urban and rural roads in Hangzhou, Huzhou, and Jiaxing, China.
Each pothole is annotated with a **bounding box** in both **YOLO** and **COCO** formats, enabling seamless integration with major deep-learning pipelines.  

---

##  Dataset Highlights
- 📸 **High-resolution imagery** (4K) captured using mirrorless vehicle-mounted cameras  
- 🤖 **Human-in-the-loop annotation pipeline** combining AI-assisted pre-labeling and expert verification  
- 🔒 **Privacy-preserving anonymization** for faces, license plates, and traffic signs  
- 📂 **Standardized data formats**: YOLO `.txt` and COCO `.json`  

---

## 🗂️ Dataset Structure
HRP4K/  
├── Train/  
│ ├── Images/  
│ └── Labels/  
├── Valid/  
│ ├── Images/  
│ └── Labels/  
├── Test/  
│ ├── Images/  
│ └── Labels/  
└── annotations/  
├── train.json  
├── valid.json  
└── test.json

📝 1. Frame Extraction
extract frames from the recorded 4K videos at 3 frames per second:  
python extract_frames.py

🧩 2. Privacy Anonymization

Step1:Automatic Masking (YOLOv11) Automatically detects and masks faces and license plates using a YOLOv11-based detector：  
python auto_privacy_detection_anonymization.py  
Step2: LabelMe Manual annotation.  
Step3: Manual Anonymization：manually anonymizing traffic signs, faces, and license plates：  
python manual_plate_sign_face_anonymization.py

🧠 3. Model-Assisted Pre-Annotation

Semi-automated pre-annotation using YOLOv11 predictions to assist human labeling:  
python pre-annotation.py

🧠 4. Format Conversion

Convert annotations between LabelMe, YOLO, and COCO formats. For specific conversion scripts, please refer to the following resources:  
https://github.com/rooneysh/Labelme2YOLO; https://github.com/Tony607/labelme2coco
