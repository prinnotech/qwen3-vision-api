from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from typing import List
import os
import tempfile
from .vision_service import VisionService
from .models import VideoResponse, ImageBatchResponse, ErrorResponse

app = FastAPI(
    title="Qwen3-Vision API", description="Video and image analysis using Qwen3-VL"
)

# Initialize vision service
vision_service = VisionService()


@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "message": "Qwen3-Vision API is running!",
        "status": "ok",
        "model": "Qwen3-VL-2B",
    }


@app.post("/analyze-video", response_model=VideoResponse)
async def analyze_video(
    video_file: UploadFile = File(..., description="Video file to analyze"),
    x_api_key: str = Header(..., alias="X-API-Key"),
):
    """Analyze single video and return detailed description"""

    # Check API key
    required_api_key = os.getenv("API_KEY")
    if not required_api_key:
        raise HTTPException(status_code=500, detail="API key not configured on server")

    if x_api_key != required_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Check file size (100MB max)
    content = await video_file.read()
    if len(content) > 100 * 1024 * 1024:  # 100MB
        raise HTTPException(status_code=413, detail="File too large. Max 100MB")

    # Validate video file type
    allowed_types = [
        "video/mp4",
        "video/avi",
        "video/mov",
        "video/webm",
        "video/quicktime",
    ]
    if video_file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, detail=f"Unsupported video type: {video_file.content_type}"
        )

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        # Analyze video
        result = await vision_service.analyze_video(temp_path, video_file.filename)

        # Clean up
        os.unlink(temp_path)

        return VideoResponse(success=True, analysis=result)

    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")


@app.post("/analyze-images", response_model=ImageBatchResponse)
async def analyze_images(
    image_files: List[UploadFile] = File(
        ..., description="Up to 60 image files to analyze"
    ),
    x_api_key: str = Header(..., alias="X-API-Key"),
):
    """Analyze up to 60 images and return detailed descriptions for each"""

    # Check API key
    required_api_key = os.getenv("API_KEY")
    if not required_api_key:
        raise HTTPException(status_code=500, detail="API key not configured on server")

    if x_api_key != required_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Check number of images
    if len(image_files) > 60:
        raise HTTPException(
            status_code=400, detail="Too many images. Maximum 60 images allowed"
        )

    if len(image_files) == 0:
        raise HTTPException(status_code=400, detail="No images provided")

    temp_files = []

    try:
        # Process each image
        for i, image_file in enumerate(image_files):
            # Check file size
            content = await image_file.read()
            if len(content) > 100 * 1024 * 1024:  # 100MB per image
                raise HTTPException(
                    status_code=413,
                    detail=f"Image {i+1} too large. Max 100MB per image",
                )

            # Validate image file type
            allowed_types = [
                "image/jpeg",
                "image/jpg",
                "image/png",
                "image/webp",
                "image/gif",
            ]
            if image_file.content_type not in allowed_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image {i+1}: Unsupported type {image_file.content_type}",
                )

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(content)
                temp_files.append((temp_file.name, image_file.filename))

        # Analyze all images
        results = await vision_service.analyze_images(temp_files)

        # Clean up temp files
        for temp_path, _ in temp_files:
            os.unlink(temp_path)

        return ImageBatchResponse(
            success=True, total_images=len(results), analyses=results
        )

    except Exception as e:
        # Clean up temp files on error
        for temp_path, _ in temp_files:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        if isinstance(e, HTTPException):
            raise e

        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")


@app.get("/health")
async def detailed_health():
    """Detailed health check with model info"""
    model_loaded = vision_service.is_model_loaded()
    return {
        "status": "healthy" if model_loaded else "loading",
        "model_loaded": model_loaded,
        "model_name": "Qwen3-VL-2B",
        "supported_video_formats": ["mp4", "avi", "mov", "webm"],
        "supported_image_formats": ["jpg", "jpeg", "png", "webp", "gif"],
        "limits": {"max_file_size": "100MB", "max_images_per_batch": 60},
    }
