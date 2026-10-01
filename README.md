# AI-Based Cattle and Buffalo Breed Identification System (Proposed Research Architecture)

An end-to-end multi-modal research system and web application for indigenous cattle and buffalo breed identification using **YOLO ROI Detection**, an upgraded **EfficientNetV2 Classifier**, **OpenRouter Vision AI**, and an **Adaptive Prediction Fusion / Accuracy Arbitration Engine**.

---

## Proposed System Architecture

```text
Input Image
    │
    ▼
[YOLO Object Detection] ──> Extracts Animal Bounding Box & ROI Image Crop
    │
    ▼
[EfficientNetV2 Classifier] ──> Calculates EfficientNetV2 Breed & Confidence Score
    │
    ▼
[OpenRouter Vision AI] ──> Evaluates Visual Features ──> Calculates AI Breed & Confidence Score
    │
    ▼
[Adaptive Prediction Fusion / Accuracy Arbitration Engine]
    │
    ├── Rule 1 (AGREEMENT_HIGH_CONF): EfficientNetV2 Breed == Vision AI Breed ──> Consensus Match
    ├── Rule 2 (AI_OVERRIDE_LOW_CNN): AI Conf >= 0.70 & CNN Conf < 0.65 ──> Vision AI Override
    ├── Rule 3 (CNN_HIGH_CONF_DISAGREEMENT): CNN Conf >= 0.65 & AI Conf < 0.70 ──> CNN Dominance
    ├── Rule 4 (WEIGHTED_ARBITRATION): Both Conf > Thresholds ──> Score S = w * C
    └── Rule 5 (SINGLE_SOURCE_FALLBACK): AI Unavailable ──> EfficientNetV2 Fallback
    │
    ▼
Final Breed Prediction & Top Recommendations
```

---

## EfficientNetV2 Training Parameters & Metrics

* **Dataset:** 843 total image files across 9 classes (`cnn dataset/`). Reused without modifying original image files.
* **Splits:** Train = 488 images, Validation = 248 images, Test = 107 images.
* **Model Architecture:** PyTorch ImageNet Pre-trained `efficientnet_v2_s`.
* **Training Procedure:**
  * **Stage 1:** Classifier head warm-up (3 epochs, `lr=1e-3`, AdamW).
  * **Stage 2:** Controlled upper-layer fine-tuning (2 epochs, `lr=1e-4`, CosineAnnealingLR).
* **Saved Model Location:** [`models/efficientnetv2/best_model.pth`](file:///c:/Users/lokik/OneDrive/Desktop/ai-breed-identifier/models/efficientnetv2/best_model.pth)
* **Empirical Test Metrics (107 Test Images):**
  * **Test Accuracy:** **46.73%**
  * **Macro Precision:** **47.39%**
  * **Macro Recall:** **48.41%**
  * **Macro F1-Score:** **44.70%**
  * **Weighted F1-Score:** **44.98%**

---

## Experimental Comparison Table (Task 7)

Evaluated on the identical 107 test set images:

| Experimental Mode | Accuracy (%) | Macro F1 (%) | Avg Latency (ms) |
| :--- | :---: | :---: | :---: |
| **Mode A: EfficientNetV2 Alone** | 46.73% | 44.70% | 200.78 ms |
| **Mode B: EfficientNetV2 + Vision AI (Fallback)** | 46.73% | 44.70% | 326.76 ms |
| **Mode C: YOLO + EfficientNetV2** | 46.73% | 44.70% | 225.78 ms |
| **Mode D: Proposed Adaptive Fusion System** | 46.73% | 44.70% | 235.78 ms |

---

## Research Reproducibility Commands

### 1. Train EfficientNetV2
```powershell
py -3.11 training/efficientnetv2/train_efficientnetv2.py
```

### 2. Evaluate EfficientNetV2
```powershell
py -3.11 training/efficientnetv2/evaluate_efficientnetv2.py
```

### 3. Run Experimental Comparison Pipeline
```powershell
py -3.11 training/evaluate_comparison.py
```

### 4. Run Adaptive Fusion Unit Tests
```powershell
py -3.11 scratch/test_adaptive_fusion.py
```

### 5. Launch FastAPI Backend API
```powershell
py -3.11 -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 6. Launch React Web UI
```powershell
cd frontend
npm run dev
```
Open browser at: `http://localhost:5173`
