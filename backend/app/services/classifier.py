import json
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from fastapi import HTTPException, status

from backend.app.config import (
    MODEL_PATH, CLASS_MAPPING_PATH,
    BREED_TO_ANIMAL_TYPE, BREED_DISPLAY_NAMES
)
from training.cnn.model import BreedEfficientNet
from backend.app.services.yolo_detector import yolo_service
from backend.app.services.openrouter_service import openrouter_service

class BreedClassifierService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.transform = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(CLASS_MAPPING_PATH):
            print(f"[Warning] Class mapping file not found at {CLASS_MAPPING_PATH}.")
            return

        with open(CLASS_MAPPING_PATH, 'r') as f:
            mapping = json.load(f)
            
        self.class_to_idx = mapping['class_to_idx']
        self.idx_to_class = {int(k) if str(k).isdigit() else k: v for k, v in mapping['idx_to_class'].items()}
        num_classes = len(self.class_to_idx)
        
        if not os.path.exists(MODEL_PATH):
            print(f"[Warning] Model checkpoint not found at {MODEL_PATH}.")
            return
            
        self.model = BreedEfficientNet(num_classes=num_classes, pretrained=False).to(self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()
        
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
        print(f"[ClassifierService] CNN Model successfully loaded on {self.device} with {num_classes} classes.")

    def predict_full_pipeline(self, image: Image.Image, top_k: int = 3, force_ai_mock: dict = None) -> dict:
        """
        Full Pipeline:
        Custom YOLO Detection (ROI Crop + BBox + YOLO Confidence + YOLO Breed) -> CNN Inference -> OpenRouter AI -> Decision Engine.
        """
        if self.model is None:
            self._load_model()
            if self.model is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="CNN Model checkpoint is not available. Please train the model first."
                )

        # 1. Custom YOLO Detection
        yolo_res = yolo_service.detect_and_crop(image)
        cropped_img = yolo_res["cropped_image"]
        yolo_confidence = round(float(yolo_res["yolo_confidence"]), 4)
        yolo_breed = yolo_res.get("yolo_breed", "Bovine")
        bbox = yolo_res.get("bbox")
        class_id = yolo_res.get("class_id")

        # 2. CNN Model Inference (using cropped ROI)
        tensor_img = self.transform(cropped_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor_img)
            probs = F.softmax(logits, dim=1).squeeze(0)
            
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(self.idx_to_class)))
        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()
        
        top_predictions = []
        for p, idx in zip(top_probs, top_indices):
            raw_breed = self.idx_to_class.get(idx, self.idx_to_class.get(str(idx), "unknown"))
            display_name = BREED_DISPLAY_NAMES.get(raw_breed, raw_breed.replace("_", " ").title())
            top_predictions.append({
                "breed": display_name,
                "confidence": round(float(p), 4)
            })
            
        top_1_raw = self.idx_to_class.get(top_indices[0], self.idx_to_class.get(str(top_indices[0]), "unknown"))
        cnn_breed = BREED_DISPLAY_NAMES.get(top_1_raw, top_1_raw.replace("_", " ").title())
        cnn_confidence = round(float(top_probs[0]), 4)

        # 3. OpenRouter AI Analysis
        if force_ai_mock is not None:
            ai_res = force_ai_mock
        else:
            ai_res = openrouter_service.analyze_breed(image)

        # 4. Final Decision Engine Logic
        if ai_res is None or "ai_confidence" not in ai_res:
            final_breed = cnn_breed
            prediction_source = "CNN Fallback"
            ai_breed = None
            ai_confidence = None
            ai_reasoning = "OpenRouter AI analysis unavailable; defaulted to CNN model prediction."
        else:
            ai_breed = ai_res["ai_breed"]
            ai_confidence = round(float(ai_res["ai_confidence"]), 4)
            ai_reasoning = ai_res.get("ai_reasoning", "OpenRouter visual evaluation.")
            
            if cnn_confidence > ai_confidence and yolo_confidence > ai_confidence:
                final_breed = cnn_breed
                prediction_source = "CNN"
            else:
                final_breed = ai_breed
                prediction_source = "OpenRouter AI"

        # Determine Animal Type
        raw_final_key = final_breed.lower().replace(" ", "_")
        animal_type = BREED_TO_ANIMAL_TYPE.get(raw_final_key, BREED_TO_ANIMAL_TYPE.get(top_1_raw, "Cattle"))

        return {
            "animal_type": animal_type,
            "cnn_breed": cnn_breed,
            "cnn_confidence": cnn_confidence,
            "yolo_breed": yolo_breed,
            "yolo_confidence": yolo_confidence,
            "bbox": bbox,
            "class_id": class_id,
            "ai_breed": ai_breed,
            "ai_confidence": ai_confidence,
            "ai_reasoning": ai_reasoning,
            "final_breed": final_breed,
            "prediction_source": prediction_source,
            "top_predictions": top_predictions
        }

classifier_service = BreedClassifierService()
