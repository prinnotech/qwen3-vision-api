import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import cv2
import os
import asyncio
from typing import List, Tuple
from .models import VideoAnalysis, ImageAnalysis


class VisionService:
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_name = "Salesforce/blip2-opt-2.7b"

    def load_model(self):
        """Load the BLIP-2 model"""
        if self.model is None:
            print(f"Loading {self.model_name} model...")

            # Load processor and model
            self.processor = Blip2Processor.from_pretrained(self.model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name, torch_dtype=torch.float16, device_map="auto"
            )

            print("BLIP-2 model loaded successfully!")

    def is_model_loaded(self):
        """Check if model is loaded"""
        return self.model is not None

    def get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frame_count / fps if fps > 0 else 0

    def extract_key_frames(
        self, video_path: str, num_frames: int = 8
    ) -> List[Image.Image]:
        """Extract key frames from video"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame indices to extract
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)

        cap.release()
        return frames

    def generate_detailed_description(self, image: Image.Image) -> str:
        """Generate detailed description using BLIP-2"""
        # Create multiple prompts to get detailed information
        prompts = [
            "Describe this image in great detail, including all people, objects, colors, and setting:",
            "What are the main visual elements and composition of this image?",
            "Describe the mood, lighting, and atmosphere of this scene:",
        ]

        descriptions = []

        for prompt in prompts:
            inputs = self.processor(image, prompt, return_tensors="pt").to(
                self.model.device, torch.float16
            )

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=100)

            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            descriptions.append(generated_text)

        # Combine all descriptions
        return " ".join(descriptions)

    async def analyze_video(self, video_path: str, filename: str) -> VideoAnalysis:
        """Analyze video and return detailed description"""
        # Load model if not already loaded
        self.load_model()

        # Get video duration
        duration = self.get_video_duration(video_path)

        # Extract key frames
        frames = self.extract_key_frames(video_path, num_frames=4)

        # Analyze first frame for detailed description
        if frames:
            description = self.generate_detailed_description(frames[0])
        else:
            description = "Unable to extract frames from video"

        return VideoAnalysis(
            success=True,
            filename=filename,
            description=description,
            duration=duration,
            key_moments=None,
        )

    async def analyze_images(
        self, temp_files: List[Tuple[str, str]]
    ) -> List[ImageAnalysis]:
        """Analyze batch of images and return detailed descriptions"""
        # Load model if not already loaded
        self.load_model()

        results = []

        for i, (temp_path, original_filename) in enumerate(temp_files):
            try:
                # Load image
                image = Image.open(temp_path).convert("RGB")

                # Generate detailed description
                description = self.generate_detailed_description(image)

                results.append(
                    ImageAnalysis(
                        image_id=f"img_{i+1:03d}",
                        filename=original_filename,
                        description=description,
                        confidence=None,
                    )
                )

            except Exception as e:
                # Add error result for failed images
                results.append(
                    ImageAnalysis(
                        image_id=f"img_{i+1:03d}",
                        filename=original_filename,
                        description=f"Error analyzing image: {str(e)}",
                        confidence=0.0,
                    )
                )

        return results
