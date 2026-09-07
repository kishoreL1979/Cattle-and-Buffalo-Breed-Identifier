import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

def compute_class_weights(dataset, num_classes: int) -> torch.Tensor:
    """
    Computes inverse class frequency weights automatically:
    W_c = N_total / (N_classes * N_c)
    """
    targets = np.array(dataset.targets)
    total_samples = len(targets)
    class_counts = np.bincount(targets, minlength=num_classes)
    
    # Avoid division by zero if any class has 0 samples
    class_counts = np.maximum(class_counts, 1)
    
    weights = total_samples / (num_classes * class_counts)
    # Normalize weights to mean of 1.0 for stability
    weights = weights / np.mean(weights)
    
    print(f"[Class Weights] Counts: {class_counts.tolist()}")
    print(f"[Class Weights] Computed Weights: {np.round(weights, 3).tolist()}")
    
    return torch.tensor(weights, dtype=torch.float32)

def calculate_metrics(y_true, y_pred, class_names):
    """
    Calculates comprehensive classification metrics:
    Accuracy, Precision, Recall, Macro F1, Weighted F1, and per-class breakdown.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    
    per_class = {}
    for i, cls in enumerate(class_names):
        per_class[cls] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i]),
            "support": int(support[i])
        }
        
    metrics = {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class
    }
    
    return metrics, cm

def plot_learning_curves(history: dict, output_path: str):
    """Plots training and validation Loss and Accuracy curves."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss Plot
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Val Loss')
    ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Accuracy Plot
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-s', label='Val Accuracy')
    ax2.set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] {output_path}")

def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: str):
    """Generates and saves a confusion matrix heatmap image."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title='Confusion Matrix — Breed Classification',
        ylabel='True Breed',
        xlabel='Predicted Breed'
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
                    
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Confusion Matrix Saved] {output_path}")

def save_json(data: dict, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"[JSON Saved] {output_path}")
