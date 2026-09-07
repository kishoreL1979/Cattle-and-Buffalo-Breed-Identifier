import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

def get_transforms(split: str = 'train', image_size: int = 224):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])

class CattleBuffaloDataset(Dataset):
    """
    PyTorch Dataset for Cattle and Buffalo Breed Identification.
    Loads images from folder-per-class layout, ignoring non-image files (e.g. _annotations.coco.json).
    """
    def __init__(self, root_dir: str, split: str = 'train', class_to_idx: dict = None, transform=None):
        self.root_dir = Path(root_dir) / split
        self.split = split
        self.transform = transform or get_transforms(split)
        
        # Detect classes if not provided
        if class_to_idx is None:
            classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx
            
        self.classes = list(self.class_to_idx.keys())
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Gather all valid image file paths and labels
        self.samples = []
        self.targets = []
        
        for cls_name, cls_idx in self.class_to_idx.items():
            cls_folder = self.root_dir / cls_name
            if not cls_folder.is_dir():
                continue
            for file_path in cls_folder.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((str(file_path), cls_idx))
                    self.targets.append(cls_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with Image.open(path) as img:
                img = img.convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {path}: {str(e)}")

        if self.transform:
            img = self.transform(img)

        return img, label, path

def create_dataloaders(dataset_root: str, batch_size: int = 16, num_workers: int = 0):
    root = Path(dataset_root)
    
    # Establish canonical class_to_idx mapping from train directory
    train_dir = root / 'train'
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    datasets = {
        split: CattleBuffaloDataset(dataset_root, split=split, class_to_idx=class_to_idx)
        for split in ['train', 'valid', 'test']
    }
    
    dataloaders = {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True
        )
        for split in ['train', 'valid', 'test']
    }
    
    return dataloaders, class_to_idx
