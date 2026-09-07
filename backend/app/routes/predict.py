from fastapi import APIRouter, UploadFile, File, HTTPException, status
from backend.app.utils.image import validate_and_load_image
from backend.app.services.classifier import classifier_service

router = APIRouter()

@router.post("/predict")
async def predict_breed(file: UploadFile = File(...)):
    """
    Classifies cattle/buffalo breed using multi-modal pipeline:
    YOLO Animal Detection -> EfficientNet-B0 CNN -> OpenRouter AI -> Strict Confidence Decision Logic.
    """
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file attached to the request."
        )
        
    try:
        file_bytes = await file.read()
        image = validate_and_load_image(file_bytes, file.filename)
        result = classifier_service.predict_full_pipeline(image)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during prediction processing: {str(e)}"
        )
