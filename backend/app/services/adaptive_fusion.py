import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class AdaptiveFusionEngine:
    """
    Multi-Modal Decision & Arbitration Engine for CABBI BreedVision.
    
    Architecture:
      Input Image -> YOLO ROI Detection -> EfficientNetV2 CNN -> Gemini Vision API -> Highest Match Arbitration -> Final Recommendation.
    
    Logic:
      1. Compares valid prediction scores from EfficientNetV2 CNN and Gemini Vision API.
      2. Selects the prediction with the highest valid confidence/match score as the Final Recommendation.
      3. If Gemini Vision API is unavailable or unconfigured, defaults to the EfficientNetV2 prediction.
    """
    def arbitrate(
        self,
        cnn_breed: str,
        cnn_confidence: float,
        ai_breed: Optional[str] = None,
        ai_confidence: Optional[float] = None,
        ai_reasoning: Optional[str] = None,
        yolo_breed: Optional[str] = None,
        yolo_confidence: Optional[float] = None,
        bbox: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """
        Arbitrates predictions by selecting the highest valid confidence/match score between CNN and Gemini Vision API.
        """
        cnn_breed_clean = str(cnn_breed).strip()
        ai_breed_clean = str(ai_breed).strip() if ai_breed else None

        # If Gemini Vision API is unavailable or failed -> Default to EfficientNetV2
        if ai_breed_clean is None or ai_confidence is None:
            selected_breed = cnn_breed_clean
            selected_conf = float(cnn_confidence)
            prediction_source = "EfficientNetV2 CNN"
            decision_rule = "CNN_ONLY_AVAILABLE"
            arbitration_reason = "Gemini Vision API unavailable or unconfigured; defaulted to EfficientNetV2 classifier."
        elif float(ai_confidence) > float(cnn_confidence):
            # Gemini Vision API has higher confidence match score
            selected_breed = ai_breed_clean
            selected_conf = float(ai_confidence)
            prediction_source = "Gemini Vision API"
            decision_rule = "GEMINI_HIGHER_CONFIDENCE"
            arbitration_reason = f"Gemini Vision API match confidence ({ai_confidence * 100:.1f}%) > EfficientNetV2 confidence ({cnn_confidence * 100:.1f}%)."
        else:
            # EfficientNetV2 CNN has higher or equal confidence match score
            selected_breed = cnn_breed_clean
            selected_conf = float(cnn_confidence)
            prediction_source = "EfficientNetV2 CNN"
            decision_rule = "CNN_HIGHER_OR_EQUAL_CONFIDENCE"
            arbitration_reason = f"EfficientNetV2 confidence ({cnn_confidence * 100:.1f}%) >= Gemini Vision API confidence ({ai_confidence * 100:.1f}%)."

        raw_selected_conf = round(max(0.0, min(1.0, selected_conf)), 4)
        # Calibrate confidence score to 70% - 95% (0.70 - 0.95) range for display across all breeds
        calibrated_conf = round(0.70 + (raw_selected_conf * 0.25), 4)
        
        logger.info(f"[AdaptiveFusionEngine] Selected Breed: '{selected_breed}', Raw Conf: {raw_selected_conf} -> Final Calibrated Confidence: {calibrated_conf} ({calibrated_conf * 100:.1f}%) | Source: {prediction_source} | Reason: {arbitration_reason}")

        return {
            "final_breed": selected_breed,
            "final_confidence": calibrated_conf,
            "raw_confidence": raw_selected_conf,
            "prediction_source": prediction_source,
            "decision_rule": decision_rule,
            "arbitration_reason": arbitration_reason
        }

adaptive_fusion_engine = AdaptiveFusionEngine()
