import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
import cv2
import os
import asyncio
from typing import List, Tuple
from .models import VideoAnalysis, ImageAnalysis


class VisionService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.model_name = "Qwen/Qwen2-VL-2B-Instruct"

    def load_model(self):
        """Load the Qwen3-VL model"""
        if self.model is None:
            print(f"Loading {self.model_name} model...")

            # Load model, tokenizer, and processor
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name, torch_dtype=torch.float16, device_map="auto"
            )

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.processor = AutoProcessor.from_pretrained(self.model_name)

            print("Qwen3-VL model loaded successfully!")

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

    async def analyze_video(self, video_path: str, filename: str) -> VideoAnalysis:
        """Analyze video and return detailed description"""
        # Load model if not already loaded
        self.load_model()

        # Get video duration
        duration = self.get_video_duration(video_path)

        # Extract key frames
        frames = self.extract_key_frames(video_path, num_frames=8)

        # Create detailed prompt for video analysis
        video_prompt = """Analyze this video thoroughly and provide an extremely detailed description. Include:
        
        1. SCENE & SETTING: Describe the environment, location, lighting, time of day, weather conditions
        2. PEOPLE & OBJECTS: Detail all people, their appearance, clothing, actions, expressions, and any objects present
        3. ACTIONS & EVENTS: Chronological description of what happens throughout the video
        4. VISUAL ELEMENTS: Colors, composition, camera angles, movement, visual quality
        5. ATMOSPHERE & MOOD: Overall tone, emotions conveyed, artistic style
        6. TEXT & GRAPHICS: Any visible text, signs, graphics, or overlays
        7. AUDIO INDICATORS: Visual cues about sounds, music, speech (what you can infer visually)
        
        Be extremely specific and detailed in your description. Provide a comprehensive analysis that captures every important visual element."""

        try:
            # Process frames with the model
            inputs = self.processor(
                text=video_prompt, images=frames, return_tensors="pt"
            ).to(self.model.device)

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=1024, do_sample=False
                )

            # Decode response
            response = self.processor.decode(outputs[0], skip_special_tokens=True)

            # Extract the generated description (remove the prompt part)
            description = response.split(video_prompt)[-1].strip()

            return VideoAnalysis(
                success=True,
                filename=filename,
                description=description,
                duration=duration,
                key_moments=None,  # Could be enhanced to extract timestamps
            )

        except Exception as e:
            raise Exception(f"Video analysis failed: {str(e)}")

    async def analyze_images(
        self, temp_files: List[Tuple[str, str]]
    ) -> List[ImageAnalysis]:
        """Analyze batch of images and return detailed descriptions"""
        # Load model if not already loaded
        self.load_model()

        # Detailed prompt for image analysis
        image_prompt = """Provide an extremely detailed description of this image. Include:
        
        1. MAIN SUBJECTS: People, animals, or primary objects - describe appearance, poses, expressions, clothing/features
        2. SETTING & ENVIRONMENT: Location, background, surroundings, architecture, nature elements
        3. COLORS & LIGHTING: Dominant colors, lighting conditions, shadows, brightness, contrast
        4. COMPOSITION & STYLE: Camera angle, framing, artistic style, visual techniques
        5. DETAILS & TEXTURES: Surface textures, materials, patterns, fine details
        6. MOOD & ATMOSPHERE: Emotional tone, feeling conveyed, artistic mood
        7. TEXT & GRAPHICS: Any visible text, signs, logos, or graphic elements
        8. ACTION & MOVEMENT: Any implied motion, gestures, or dynamic elements
        
        Be extremely thorough and specific. Capture every important visual element and detail."""

        results = []

        for i, (temp_path, original_filename) in enumerate(temp_files):
            try:
                # Load image
                image = Image.open(temp_path).convert("RGB")

                # Process with model
                inputs = self.processor(
                    text=image_prompt, images=[image], return_tensors="pt"
                ).to(self.model.device)

                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, max_new_tokens=512, do_sample=False
                    )

                # Decode response
                response = self.processor.decode(outputs[0], skip_special_tokens=True)

                # Extract description
                description = response.split(image_prompt)[-1].strip()

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
