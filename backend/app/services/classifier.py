import json
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from fastapi import HTTPException, status

from backend.app.config import (
    BASE_DIR, BREED_TO_ANIMAL_TYPE, BREED_DISPLAY_NAMES
)
from training.efficientnetv2.model import BreedEfficientNetV2
from backend.app.services.yolo_detector import yolo_service
from backend.app.services.openrouter_service import openrouter_service
from backend.app.services.adaptive_fusion import adaptive_fusion_engine

V2_MODEL_PATH = os.path.join(BASE_DIR, "models", "efficientnetv2", "best_model.pth")
V2_MAPPING_PATH = os.path.join(BASE_DIR, "models", "efficientnetv2", "class_mapping.json")

class BreedClassifierService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.transform = None
        self._load_model()

    def _load_model(self):
        # Fallback to CNN mapping if EfficientNetV2 mapping not found
        mapping_path = V2_MAPPING_PATH if os.path.exists(V2_MAPPING_PATH) else os.path.join(BASE_DIR, "models", "cnn", "class_mapping.json")
        model_path = V2_MODEL_PATH if os.path.exists(V2_MODEL_PATH) else os.path.join(BASE_DIR, "models", "cnn", "best_model.pth")
        
        if not os.path.exists(mapping_path):
            print(f"[Warning] Class mapping file not found at {mapping_path}.")
            return

        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
            
        self.class_to_idx = mapping['class_to_idx']
        self.idx_to_class = {int(k) if str(k).isdigit() else k: v for k, v in mapping['idx_to_class'].items()}
        num_classes = len(self.class_to_idx)
        
        if not os.path.exists(model_path):
            print(f"[Warning] EfficientNetV2 checkpoint not found at {model_path}.")
            return
            
        self.model = BreedEfficientNetV2(num_classes=num_classes, pretrained=False).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
        print(f"[BreedClassifierService] EfficientNetV2 Model loaded on {self.device} with {num_classes} classes.")

    def predict_full_pipeline(self, image: Image.Image, top_k: int = 3, force_ai_mock: dict = None) -> dict:
        """
        Full Proposed Research Pipeline:
        Input Image -> YOLO Animal ROI Crop -> EfficientNetV2 Breed Classifier -> Vision AI -> Adaptive Arbitration Engine -> Final Breed.
        """
        if self.model is None:
            self._load_model()
            if self.model is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="EfficientNetV2 model checkpoint is unavailable. Please train model first."
                )

        # 1. YOLO Animal ROI Crop & Detection
        yolo_res = yolo_service.detect_and_crop(image)
        cropped_img = yolo_res["cropped_image"]
        yolo_confidence = round(float(yolo_res["yolo_confidence"]), 4)
        yolo_breed = yolo_res.get("yolo_breed", "Bovine")
        bbox = yolo_res.get("bbox")
        class_id = yolo_res.get("class_id")

        # 2. EfficientNetV2 Classifier Inference
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

        # 3. OpenRouter Vision AI Verification Analysis
        if force_ai_mock is not None:
            ai_res = force_ai_mock
        else:
            ai_res = openrouter_service.analyze_breed(image, candidate_cnn_breed=cnn_breed)

        ai_breed = ai_res["ai_breed"] if (ai_res and "ai_breed" in ai_res) else None
        ai_confidence = round(float(ai_res["ai_confidence"]), 4) if (ai_res and "ai_confidence" in ai_res) else None
        ai_reasoning = ai_res.get("ai_reasoning") if ai_res else None

        # 4. Adaptive Prediction Fusion / Accuracy Arbitration Engine
        arbitration = adaptive_fusion_engine.arbitrate(
            cnn_breed=cnn_breed,
            cnn_confidence=cnn_confidence,
            ai_breed=ai_breed,
            ai_confidence=ai_confidence,
            ai_reasoning=ai_reasoning,
            yolo_breed=yolo_breed,
            yolo_confidence=yolo_confidence,
            bbox=bbox
        )

        final_breed = arbitration["final_breed"]
        final_confidence = arbitration.get("final_confidence", cnn_confidence)
        prediction_source = arbitration["prediction_source"]
        decision_rule = arbitration["decision_rule"]
        arbitration_reason = arbitration["arbitration_reason"]

        # Animal Type
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
            "final_confidence": final_confidence,
            "prediction_source": prediction_source,
            "decision_rule": decision_rule,
            "arbitration_reason": arbitration_reason,
            "top_predictions": top_predictions
        }

classifier_service = BreedClassifierService()
