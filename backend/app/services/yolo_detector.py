import os
import logging
from pathlib import Path
from PIL import Image
import numpy as np
from backend.app.config import BASE_DIR, BREED_DISPLAY_NAMES, SUPPORTED_BREED_NAMES

logger = logging.getLogger(__name__)

# Default model path: custom trained model 'models/yolo/best.pt', falling back to 'yolov8n.pt'
DEFAULT_YOLO_PATH = os.getenv("YOLO_MODEL_PATH", str(BASE_DIR / "models" / "yolo" / "best.pt"))

class CustomYoloDetectorService:
    def __init__(self):
        self.model = None
        self.model_path = DEFAULT_YOLO_PATH
        self._load_yolo()

    def _load_yolo(self):
        try:
            from ultralytics import YOLO
            
            # Check if custom best.pt exists
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                logger.info(f"Custom YOLO model loaded successfully from {self.model_path}.")
            elif os.path.exists(str(BASE_DIR / "runs" / "detect" / "train_custom_yolo" / "weights" / "best.pt")):
                alt_path = str(BASE_DIR / "runs" / "detect" / "train_custom_yolo" / "weights" / "best.pt")
                self.model = YOLO(alt_path)
                logger.info(f"Custom YOLO model loaded from run weights {alt_path}.")
            else:
                # Fallback to pretrained generic YOLOv8n
                fallback_path = "yolov8n.pt"
                self.model = YOLO(fallback_path)
                logger.warning(f"Custom model not found at {self.model_path}. Using fallback model {fallback_path}.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None

    def detect_and_crop(self, image: Image.Image) -> dict:
        """
        Runs Custom YOLO Detection on image.
        Returns dict containing:
        {
          "yolo_breed": str,
          "yolo_confidence": float,
          "bbox": {"x1": int, "y1": int, "x2": int, "y2": int},
          "class_id": int,
          "cropped_image": Image.Image
        }
        """
        if self.model is None:
            self._load_yolo()
            if self.model is None:
                w, h = image.size
                return {
                    "yolo_breed": "Unspecified",
                    "yolo_confidence": 0.70,
                    "bbox": {"x1": 0, "y1": 0, "x2": w, "y2": h},
                    "class_id": None,
                    "cropped_image": image
                }

        try:
            img_np = np.array(image.convert("RGB"))
            results = self.model(img_np, verbose=False)
            
            w, h = image.size
            if not results or len(results[0].boxes) == 0:
                return {
                    "yolo_breed": "Bovine",
                    "yolo_confidence": 0.70,
                    "bbox": {"x1": 0, "y1": 0, "x2": w, "y2": h},
                    "class_id": None,
                    "cropped_image": image
                }

            boxes = results[0].boxes
            best_box = None
            max_conf = -1.0
            
            for box in boxes:
                conf = float(box.conf[0].cpu().numpy())
                if conf > max_conf:
                    max_conf = conf
                    best_box = box

            if best_box is not None:
                conf_score = round(float(best_box.conf[0].cpu().numpy()), 4)
                cls_id = int(best_box.cls[0].cpu().numpy())
                
                xyxy = best_box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = map(int, xyxy)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Retrieve breed name from model names or class ID
                model_names = getattr(self.model, "names", {})
                raw_name = model_names.get(cls_id, f"Class_{cls_id}")
                
                # Format breed display name
                if isinstance(raw_name, str) and raw_name in SUPPORTED_BREED_NAMES:
                    detected_breed = raw_name
                else:
                    breed_key = str(raw_name).lower().replace(" ", "_")
                    detected_breed = BREED_DISPLAY_NAMES.get(breed_key, str(raw_name).title())

                cropped = image
                if (x2 - x1) > 20 and (y2 - y1) > 20:
                    cropped = image.crop((x1, y1, x2, y2))
                    
                return {
                    "yolo_breed": detected_breed,
                    "yolo_confidence": conf_score,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "class_id": cls_id,
                    "cropped_image": cropped
                }

            return {
                "yolo_breed": "Bovine",
                "yolo_confidence": 0.70,
                "bbox": {"x1": 0, "y1": 0, "x2": w, "y2": h},
                "class_id": None,
                "cropped_image": image
            }

        except Exception as e:
            logger.error(f"YOLO inference exception: {e}")
            w, h = image.size
            return {
                "yolo_breed": "Bovine",
                "yolo_confidence": 0.70,
                "bbox": {"x1": 0, "y1": 0, "x2": w, "y2": h},
                "class_id": None,
                "cropped_image": image
            }

yolo_service = CustomYoloDetectorService()
