import io
from PIL import Image
from fastapi import HTTPException, status
from backend.app.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB

def validate_and_load_image(file_bytes: bytes, filename: str) -> Image.Image:
    """
    Validates file extension, file size, and image integrity.
    Returns a clean RGB PIL Image object or raises HTTPException with user-friendly error.
    """
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded or file is empty."
        )
        
    # Check size
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size exceeds maximum limit of {MAX_FILE_SIZE_MB}MB."
        )
        
    # Check extension if filename provided
    if filename:
        ext = f".{filename.split('.')[-1].lower()}" if '.' in filename else ''
        if ext and ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format '{ext}'. Please upload JPG, JPEG, PNG, or WEBP images."
            )
            
    # Attempt to open image with PIL
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.verify()  # Verify image integrity
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrupted or unreadable image file. Please upload a valid image."
        )
        
    # Re-open for actual processing (verify closes the file handle)
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image = image.convert('RGB')
        
        # Memory optimization for 512MB RAM cloud tier: downscale high-res images to max 1024px
        max_dim = 1024
        if image.width > max_dim or image.height > max_dim:
            image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            
        return image
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process image: {str(e)}"
        )
