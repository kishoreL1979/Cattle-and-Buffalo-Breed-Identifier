import os
import sys
import json
from pathlib import Path
import torch
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from dataset import create_dataloaders
from model import BreedEfficientNet
from utils import calculate_metrics, plot_confusion_matrix, save_json

DATASET_ROOT = r"c:\Users\lokik\OneDrive\Desktop\ai-breed-identifier\cnn dataset"
MODEL_DIR = r"c:\Users\lokik\OneDrive\Desktop\ai-breed-identifier\models\cnn"
RESULTS_DIR = r"c:\Users\lokik\OneDrive\Desktop\ai-breed-identifier\results"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating model on test dataset using device: {device}")
    
    # 1. Load Class Mapping
    mapping_path = os.path.join(MODEL_DIR, "class_mapping.json")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Class mapping file not found at {mapping_path}. Run training first.")
        
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    class_to_idx = mapping['class_to_idx']
    idx_to_class = {int(k) if k.isdigit() else k: v for k, v in mapping['idx_to_class'].items()}
    class_names = [idx_to_class[i] if isinstance(idx_to_class, dict) else idx_to_class[str(i)] for i in range(len(class_to_idx))]
    num_classes = len(class_names)
    
    # 2. Load Dataloader for test split
    dataloaders, _ = create_dataloaders(DATASET_ROOT, batch_size=16, num_workers=0)
    test_loader = dataloaders['test']
    
    # 3. Load Trained Model
    model_path = os.path.join(MODEL_DIR, "best_model.pth")
    model = BreedEfficientNet(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    # 4. Calculate Metrics & Confusion Matrix
    metrics, cm = calculate_metrics(y_true, y_pred, class_names)
    
    # 5. Save Artifacts
    metrics_path = os.path.join(RESULTS_DIR, "metrics", "test_metrics.json")
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix", "confusion_matrix.png")
    
    save_json(metrics, metrics_path)
    plot_confusion_matrix(cm, class_names, cm_path)
    
    print("\n================ EVALUATION RESULTS (TEST SET) ================")
    print(f"Test Accuracy    : {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision  : {metrics['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall     : {metrics['macro_recall'] * 100:.2f}%")
    print(f"Macro F1 Score   : {metrics['macro_f1'] * 100:.2f}%")
    print(f"Weighted F1 Score: {metrics['weighted_f1'] * 100:.2f}%")
    print("\nPer-Class Breakdown:")
    for cls_name, p_metrics in metrics['per_class'].items():
        print(f"  - {cls_name:12s}: Precision={p_metrics['precision']*100:.1f}%, Recall={p_metrics['recall']*100:.1f}%, F1={p_metrics['f1_score']*100:.1f}% (Support={p_metrics['support']})")

if __name__ == '__main__':
    main()
