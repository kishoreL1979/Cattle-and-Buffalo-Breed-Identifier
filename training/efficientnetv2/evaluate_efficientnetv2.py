import os
import sys
import json
from pathlib import Path
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score

sys.path.append(r"c:\Users\lokik\OneDrive\Desktop\ai-breed-identifier")

from training.cnn.dataset import create_dataloaders
from training.efficientnetv2.model import BreedEfficientNetV2

DATASET_ROOT = r"c:\Users\lokik\OneDrive\Desktop\ai-breed-identifier\cnn dataset"
MODEL_PATH = r"c:\Users\lokik\OneDrive\Desktop\ai-breed-identifier\models\efficientnetv2\best_model.pth"
MAPPING_PATH = r"c:\Users\lokik\OneDrive\Desktop\ai-breed-identifier\models\efficientnetv2\class_mapping.json"
RESULTS_DIR = r"c:\Users\lokik\OneDrive\Desktop\ai-breed-identifier\results\metrics"

def evaluate_efficientnetv2():
    print("================ EFFICIENTNETV2 MODEL EVALUATION (TEST SET) ================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(MAPPING_PATH):
        raise FileNotFoundError(f"Model checkpoint or mapping not found. Please train EfficientNetV2 first.")
        
    with open(MAPPING_PATH, 'r') as f:
        mapping = json.load(f)
    class_to_idx = mapping['class_to_idx']
    idx_to_class = {int(k) if str(k).isdigit() else k: v for k, v in mapping['idx_to_class'].items()}
    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]
    
    dataloaders, _ = create_dataloaders(dataset_root=DATASET_ROOT, batch_size=16, num_workers=0)
    test_loader = dataloaders['test']
    
    model = BreedEfficientNetV2(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch[0].to(device), batch[1]
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds).tolist()
    
    metrics = {
        "model_name": "EfficientNetV2-S",
        "test_accuracy": round(float(acc), 4),
        "macro_f1": round(float(f1_macro), 4),
        "macro_precision": round(float(precision_macro), 4),
        "macro_recall": round(float(recall_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "weighted_precision": round(float(precision_weighted), 4),
        "weighted_recall": round(float(recall_weighted), 4),
        "total_test_images": len(all_labels),
        "confusion_matrix": cm,
        "classes": class_names
    }
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_json = os.path.join(RESULTS_DIR, "efficientnetv2_test_metrics.json")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
        
    print("\n================ EFFICIENTNETV2 TEST RESULTS ================")
    print(f"  Test Accuracy   : {acc * 100:.2f}%")
    print(f"  Macro Precision : {precision_macro * 100:.2f}%")
    print(f"  Macro Recall    : {recall_macro * 100:.2f}%")
    print(f"  Macro F1-Score  : {f1_macro * 100:.2f}%")
    print(f"  Weighted F1-Score: {f1_weighted * 100:.2f}%")
    print(f"Saved evaluation JSON: {out_json}")

if __name__ == '__main__':
    evaluate_efficientnetv2()
