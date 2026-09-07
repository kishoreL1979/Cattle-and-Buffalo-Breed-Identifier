# AI-Based Cattle and Buffalo Breed Identification System

An end-to-end multi-modal computer vision and web application that identifies 9 major indigenous cattle and buffalo breeds using a fine-tuned **EfficientNet-B0 Convolutional Neural Network (CNN)**, a **Custom-Trained YOLOv8 Breed Detection Model**, and **OpenRouter Vision AI**, governed by a strict confidence-based final decision rule.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Custom YOLO Model & Training](#custom-yolo-model--training)
3. [Multi-Modal Pipeline Architecture](#multi-modal-pipeline-architecture)
4. [Strict Decision Logic](#strict-decision-logic)
5. [Supported Breeds](#supported-breeds)
6. [API Specification](#api-specification)
7. [Running the Application](#running-the-application)

---

## Project Overview
Indigenous cattle (*Bos indicus*) and buffalo (*Bubalus bubalis*) breeds play a critical role in sustainable agriculture and dairy farming. This system combines **Custom YOLO object detection** for animal ROI cropping, a fine-tuned **EfficientNet-B0 CNN**, and **OpenRouter Vision AI** to deliver reliable, transparent breed classification with individual confidence scores and top-3 probability rankings.

---

## Custom YOLO Model & Training

### Dataset & Annotation Conversion
* **Total Dataset Size:** 843 image files pre-split into 488 train, 248 validation, and 107 test images.
* **Bounding Box Annotations:** 27 COCO JSON files containing 1,309 bounding box annotations for 774 images (91.8%) were converted into normalized YOLO format `[class_id, x_center, y_center, width, height]`. Full-image default bounding boxes were generated for the remaining 69 images to achieve 100% coverage across all 843 dataset images.
* **Target Classes (0..8):** `0: Gir`, `1: Jaffrabadi`, `2: Kankrej`, `3: Mehsana`, `4: Murrah`, `5: Red Sindhi`, `6: Sahiwal`, `7: Surti`, `8: Tharparkar`.

### Training Configuration
* **Model Backbone:** `yolov8n.pt`
* **Dataset Config:** [`dataset_yolo/data.yaml`](file:///c:/Users/lokik/OneDrive/Desktop/ai-breed-identifier/dataset_yolo/data.yaml)
* **Image Size:** 416 × 416
* **Epochs:** 10
* **Batch Size:** 16
* **Optimizer:** AdamW
* **Saved Model Location:** [`models/yolo/best.pt`](file:///c:/Users/lokik/OneDrive/Desktop/ai-breed-identifier/models/yolo/best.pt)

### Genuine Test Evaluation Metrics (107 Test Images)
* **Precision (mP):** **20.17%**
* **Recall (mR):** **19.44%**
* **mAP@50:** **9.41%**
* **mAP@50-95:** **4.10%**
* **Metrics File:** [`results/metrics/yolo_test_metrics.json`](file:///c:/Users/lokik/OneDrive/Desktop/ai-breed-identifier/results/metrics/yolo_test_metrics.json)

---

## Multi-Modal Pipeline Architecture

```text
Input Image
    │
    ▼
[Custom YOLO Breed Detector] ──> Returns YOLO Breed, BBox & Detection Confidence
    │
    ▼
[Cropped Animal ROI] ──> [EfficientNet-B0 CNN] ──> Returns CNN Breed & Confidence Score
    │
    ▼
[OpenRouter Vision AI] ──> Analyzes Visual Features ──> Returns AI Breed & Confidence
    │
    ▼
[Strict Confidence Decision Engine]
    │
    ├── IF CNN_confidence > AI_confidence AND YOLO_confidence > AI_confidence
    │       └── FINAL BREED = CNN predicted breed (Source: "CNN")
    │
    └── ELSE (CNN_confidence <= AI_confidence OR YOLO_confidence <= AI_confidence)
            └── FINAL BREED = OpenRouter AI predicted breed (Source: "OpenRouter AI")
```

---

## Strict Decision Logic

```text
IF CNN_confidence > AI_confidence AND YOLO_confidence > AI_confidence:
    FINAL BREED = CNN predicted breed
    Prediction Source = "CNN"
ELSE:
    FINAL BREED = OpenRouter AI predicted breed
    Prediction Source = "OpenRouter AI"
```

* **Fallback Condition:** If OpenRouter AI is unavailable or unconfigured, the system safely falls back to `CNN Fallback`.

---

## Supported Breeds

1. **Gir** (Cattle)
2. **Jaffrabadi** (Buffalo)
3. **Kankrej** (Cattle)
4. **Mehsana** (Buffalo)
5. **Murrah** (Buffalo)
6. **Red Sindhi** (Cattle)
7. **Sahiwal** (Cattle)
8. **Surti** (Buffalo)
9. **Tharparkar** (Cattle)

---

## Running the Application

### 1. Launch FastAPI Backend API
```powershell
py -3.11 -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
API Docs: `http://127.0.0.1:8000/docs`

### 2. Launch React Web Application
```powershell
cd frontend
npm run dev
```
Open browser at: `http://localhost:5173`
