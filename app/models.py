from pydantic import BaseModel
from typing import List, Optional


class ImageAnalysis(BaseModel):
    image_id: str
    filename: str
    description: str
    confidence: Optional[float] = None


class VideoAnalysis(BaseModel):
    success: bool
    filename: str
    description: str
    duration: Optional[float] = None
    key_moments: Optional[List[str]] = None


class ImageBatchResponse(BaseModel):
    success: bool
    total_images: int
    analyses: List[ImageAnalysis]


class VideoResponse(BaseModel):
    success: bool
    analysis: VideoAnalysis


class ErrorResponse(BaseModel):
    success: bool
    error: str
    details: Optional[str] = None
