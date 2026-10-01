import os
import sys
import time
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure project root is in sys.path
sys.path.append(r"c:\Users\lokik\OneDrive\Desktop\ai-breed-identifier")

from training.cnn.dataset import create_dataloaders
from training.cnn.utils import compute_class_weights
from training.efficientnetv2.model import BreedEfficientNetV2

DATASET_ROOT = r"c:\Users\lokik\OneDrive\Desktop\ai-breed-identifier\cnn dataset"
MODEL_SAVE_DIR = r"c:\Users\lokik\OneDrive\Desktop\ai-breed-identifier\models\efficientnetv2"

def train_efficientnetv2():
    print("================ EFFICIENTNETV2 MODEL TRAINING ================", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}", flush=True)
    
    # 1. Load Data
    dataloaders, class_to_idx = create_dataloaders(
        dataset_root=DATASET_ROOT, batch_size=32, num_workers=0
    )
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    dataset_sizes = {split: len(dataloaders[split].dataset) for split in ['train', 'valid', 'test']}
    num_classes = len(class_to_idx)
    print(f"Loaded dataset: {num_classes} classes. Sizes: {dataset_sizes}", flush=True)
    
    # Calculate inverse-frequency class weights
    train_dataset = dataloaders['train'].dataset
    class_weights = compute_class_weights(train_dataset, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 2. Instantiate Model
    model = BreedEfficientNetV2(num_classes=num_classes, pretrained=True).to(device)
    
    # STAGE 1: Warm-up Classifier Head (3 Epochs)
    print("\n--- STAGE 1: Classifier Head Warm-Up (3 Epochs) ---", flush=True)
    model.freeze_backbone()
    optimizer_stage1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-2)
    
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    best_model_path = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
    best_val_acc = 0.0

    for epoch in range(1, 4):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for batch in dataloaders['train']:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            optimizer_stage1.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer_stage1.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = (running_corrects.double() / dataset_sizes['train']).item()
        
        # Validation Check
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for batch in dataloaders['valid']:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
        val_acc = (val_corrects.double() / dataset_sizes['valid']).item()
        print(f"Stage 1 - Epoch {epoch}/3 | Train Acc: {epoch_acc * 100:.2f}% | Val Acc: {val_acc * 100:.2f}%", flush=True)
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best checkpoint to {best_model_path} (Val Acc: {val_acc * 100:.2f}%)", flush=True)

    # STAGE 2: Controlled Upper Layer Fine-Tuning (2 Epochs)
    print("\n--- STAGE 2: Controlled Upper Layer Fine-Tuning (2 Epochs) ---", flush=True)
    model.unfreeze_upper_layers(unfreeze_blocks=1)
    optimizer_stage2 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer_stage2, T_max=2, eta_min=1e-6)

    for epoch in range(1, 3):
        model.train()
        train_loss = 0.0
        train_corrects = 0
        
        for batch in dataloaders['train']:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            optimizer_stage2.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer_stage2.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)
            
        scheduler.step()
        train_acc = (train_corrects.double() / dataset_sizes['train']).item()
        
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for batch in dataloaders['valid']:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
        val_acc = (val_corrects.double() / dataset_sizes['valid']).item()
        print(f"Stage 2 - Epoch {epoch:02d}/2 | Train Acc: {train_acc * 100:.2f}% | Val Acc: {val_acc * 100:.2f}%", flush=True)
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best checkpoint to {best_model_path} (Val Acc: {val_acc * 100:.2f}%)", flush=True)
                
    # Save Class Mapping
    mapping_path = os.path.join(MODEL_SAVE_DIR, "class_mapping.json")
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump({'class_to_idx': class_to_idx, 'idx_to_class': idx_to_class}, f, indent=2)
        
    print("\n[EfficientNetV2 Training Completed Successfully]", flush=True)
    print(f"  Best Validation Accuracy: {best_val_acc * 100:.2f}%", flush=True)
    print(f"  Model saved to: {best_model_path}", flush=True)

if __name__ == '__main__':
    train_efficientnetv2()
