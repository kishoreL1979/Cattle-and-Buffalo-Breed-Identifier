import json
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(r"c:\Users\lokik\OneDrive\Desktop\ai-breed-identifier")
DATA_YAML = BASE_DIR / "dataset_yolo" / "data.yaml"
MODEL_PATH = BASE_DIR / "models" / "yolo" / "best.pt"
RESULTS_DIR = BASE_DIR / "results" / "metrics"

def evaluate():
    print("================ CUSTOM YOLO MODEL EVALUATION (TEST SET) ================")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train the model first.")
        
    model = YOLO(str(MODEL_PATH))
    
    metrics = model.val(
        data=str(DATA_YAML),
        split='test',
        imgsz=640,
        batch=16,
        project=str(BASE_DIR / "runs" / "detect"),
        name="val_custom_yolo",
        exist_ok=True,
        verbose=True
    )
    
    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    
    eval_data = {
        "precision": round(mp, 4),
        "recall": round(mr, 4),
        "mAP_50": round(map50, 4),
        "mAP_50_95": round(map50_95, 4),
        "classes": metrics.names
    }
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "yolo_test_metrics.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, indent=2)
        
    print("\n================ CUSTOM YOLO TEST RESULTS ================")
    print(f"Precision (mP) : {mp * 100:.2f}%")
    print(f"Recall (mR)    : {mr * 100:.2f}%")
    print(f"mAP@50         : {map50 * 100:.2f}%")
    print(f"mAP@50-95      : {map50_95 * 100:.2f}%")
    print(f"Saved evaluation JSON: {out_json}")

if __name__ == '__main__':
    evaluate()
