import os
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure local imports work
sys.path.append(str(Path(__file__).resolve().parent))
from dataset import create_dataloaders
from model import BreedEfficientNet
from utils import compute_class_weights, plot_learning_curves, save_json

DATASET_ROOT = r"c:\Users\lokik\OneDrive\Desktop\ai-breed-identifier\cnn dataset"
MODEL_DIR = r"c:\Users\lokik\OneDrive\Desktop\ai-breed-identifier\models\cnn"
RESULTS_DIR = r"c:\Users\lokik\OneDrive\Desktop\ai-breed-identifier\results"

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels, _ in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += images.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += images.size(0)
            
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data
    dataloaders, class_to_idx = create_dataloaders(DATASET_ROOT, batch_size=16, num_workers=0)
    num_classes = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print(f"Detected {num_classes} classes: {list(class_to_idx.keys())}")
    
    # 2. Compute Class Weights
    train_dataset = dataloaders['train'].dataset
    class_weights = compute_class_weights(train_dataset, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 3. Initialize Model
    model = BreedEfficientNet(num_classes=num_classes, dropout_rate=0.3, pretrained=True).to(device)
    
    # 4. Directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'plots'), exist_ok=True)
    
    best_model_path = os.path.join(MODEL_DIR, "best_model.pth")
    class_mapping_path = os.path.join(MODEL_DIR, "class_mapping.json")
    config_path = os.path.join(MODEL_DIR, "train_config.json")
    
    # Save class mapping
    save_json({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, class_mapping_path)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    patience = 7
    patience_counter = 0
    
    stage1_epochs = 8
    stage2_epochs = 12
    total_epochs = stage1_epochs + stage2_epochs
    
    # --- STAGE 1: Train Head Only ---
    print("\n" + "="*50)
    print("STAGE 1: Training Classification Head (Backbone Frozen)")
    print("="*50)
    model.freeze_backbone()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    
    for epoch in range(1, stage1_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, dataloaders['train'], criterion, optimizer, device)
        val_loss, val_acc = validate(model, dataloaders['valid'], criterion, device)
        elapsed = time.time() - t0
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch:02d}/{total_epochs:02d}] ({elapsed:.1f}s) - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best model (Val Acc: {best_val_acc:.2f}%)")
            
    # --- STAGE 2: Fine-tune Upper Backbone Layers ---
    print("\n" + "="*50)
    print("STAGE 2: Fine-Tuning Upper Backbone Layers")
    print("="*50)
    model.unfreeze_upper_layers(num_blocks=3)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=stage2_epochs, eta_min=1e-6)
    
    for epoch in range(stage1_epochs + 1, total_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, dataloaders['train'], criterion, optimizer, device)
        val_loss, val_acc = validate(model, dataloaders['valid'], criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        lr_curr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:02d}/{total_epochs:02d}] ({elapsed:.1f}s) [lr={lr_curr:.6f}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best model (Val Acc: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping Triggered] No validation improvement for {patience} epochs.")
                break
                
    # Save training configuration & metrics
    config_data = {
        "model_architecture": "EfficientNet-B0",
        "num_classes": num_classes,
        "classes": list(class_to_idx.keys()),
        "stage1_epochs": stage1_epochs,
        "stage2_epochs": stage2_epochs,
        "best_val_acc": float(best_val_acc),
        "batch_size": 16,
        "optimizer": "AdamW"
    }
    save_json(config_data, config_path)
    
    # Plot learning curves
    plot_learning_curves(history, os.path.join(RESULTS_DIR, 'plots', 'learning_curves.png'))
    print("\n[Training Complete] Best model saved to:", best_model_path)

if __name__ == '__main__':
    main()
