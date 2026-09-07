import os
import shutil
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(r"c:\Users\lokik\OneDrive\Desktop\ai-breed-identifier")
DATA_YAML = BASE_DIR / "dataset_yolo" / "data.yaml"
OUTPUT_MODEL_DIR = BASE_DIR / "models" / "yolo"

def train():
    print("================ CUSTOM YOLO MODEL TRAINING ================")
    print(f"Data config: {DATA_YAML}")
    
    # 1. Initialize YOLOv8n pretrained backbone
    model = YOLO('yolov8n.pt')
    
    # 2. Train Custom Model (Optimized for CPU: imgsz=416, epochs=10)
    results = model.train(
        data=str(DATA_YAML),
        epochs=10,
        imgsz=416,
        batch=16,
        patience=5,
        project=str(BASE_DIR / "runs" / "detect"),
        name="train_custom_yolo",
        exist_ok=True,
        plots=True,
        verbose=True
    )
    
    # 3. Locate and copy best.pt
    best_weights_path = BASE_DIR / "runs" / "detect" / "train_custom_yolo" / "weights" / "best.pt"
    if not best_weights_path.exists():
        best_weights_path = BASE_DIR / "runs" / "detect" / "train_custom_yolo" / "weights" / "last.pt"
        
    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target_best_path = OUTPUT_MODEL_DIR / "best.pt"
    
    if best_weights_path.exists():
        shutil.copy2(best_weights_path, target_best_path)
        print(f"\n[Training Completed Successfully]")
        print(f"  Best Weights Saved at: {target_best_path}")
    else:
        print(f"\n[Warning] Trained weights file not found at {best_weights_path}")

if __name__ == '__main__':
    train()
