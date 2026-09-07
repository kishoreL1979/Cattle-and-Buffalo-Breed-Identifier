import os
import sys
import json
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent))
from dataset import get_transforms
from model import BreedEfficientNet

MODEL_DIR = r"c:\Users\lokik\OneDrive\Desktop\ai-breed-identifier\models\cnn"

BREED_TO_ANIMAL = {
    "gir": "Cattle",
    "jaffrabadi": "Buffalo",
    "kankrej": "Cattle",
    "mehsana": "Buffalo",
    "murrah": "Buffalo",
    "red_sindhi": "Cattle",
    "sahiwal": "Cattle",
    "surti": "Buffalo",
    "tharparkar": "Cattle"
}

BREED_DISPLAY_NAMES = {
    "gir": "Gir",
    "jaffrabadi": "Jaffrabadi",
    "kankrej": "Kankrej",
    "mehsana": "Mehsana",
    "murrah": "Murrah",
    "red_sindhi": "Red Sindhi",
    "sahiwal": "Sahiwal",
    "surti": "Surti",
    "tharparkar": "Tharparkar"
}

def predict_image(image_path: str, top_k: int = 3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load class mapping
    mapping_path = os.path.join(MODEL_DIR, "class_mapping.json")
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    idx_to_class = {int(k) if str(k).isdigit() else k: v for k, v in mapping['idx_to_class'].items()}
    num_classes = len(idx_to_class)
    
    # Load model
    model_path = os.path.join(MODEL_DIR, "best_model.pth")
    model = BreedEfficientNet(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load and transform image
    transform = get_transforms(split='test')
    with Image.open(image_path) as img:
        img_rgb = img.convert('RGB')
        
    tensor_img = transform(img_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(tensor_img)
        probs = F.softmax(logits, dim=1).squeeze(0)
        
    top_probs, top_indices = torch.topk(probs, k=min(top_k, num_classes))
    
    top_probs = top_probs.cpu().numpy()
    top_indices = top_indices.cpu().numpy()
    
    top_preds = []
    for p, idx in zip(top_probs, top_indices):
        raw_breed = idx_to_class[idx] if idx in idx_to_class else idx_to_class[str(idx)]
        top_preds.append({
            "breed_key": raw_breed,
            "breed": BREED_DISPLAY_NAMES.get(raw_breed, raw_breed.title()),
            "animal_type": BREED_TO_ANIMAL.get(raw_breed, "Unknown"),
            "confidence": float(p)
        })
        
    top_1 = top_preds[0]
    result = {
        "animal_type": top_1["animal_type"],
        "breed": top_1["breed"],
        "confidence": top_1["confidence"],
        "top_predictions": [
            {"breed": item["breed"], "confidence": item["confidence"]}
            for item in top_preds
        ]
    }
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run breed inference on a single image.")
    parser.add_argument("image_path", help="Path to input cattle/buffalo image.")
    args = parser.parse_args()
    
    res = predict_image(args.image_path)
    print(json.dumps(res, indent=2))
