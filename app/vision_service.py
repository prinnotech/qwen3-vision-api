import torch
from transformers import AutoModel, AutoTokenizer
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
        self.model_name = "Qwen/Qwen-VL-Chat"

    def load_model(self):
        """Load the Qwen-VL model"""
        if self.model is None:
            print(f"Loading {self.model_name} model...")

            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )

            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
            ).eval()

            print("Qwen-VL model loaded successfully!")

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
        frames = self.extract_key_frames(video_path, num_frames=4)

        # Save first frame temporarily for analysis
        temp_frame_path = "/tmp/temp_frame.jpg"
        frames[0].save(temp_frame_path)

        # Create detailed prompt for video analysis
        query = self.tokenizer.from_list_format(
            [
                {"image": temp_frame_path},
                {
                    "text": """Analyze this video frame thoroughly and provide an extremely detailed description. Include:
            
            1. SCENE & SETTING: Describe the environment, location, lighting, time of day, weather conditions
            2. PEOPLE & OBJECTS: Detail all people, their appearance, clothing, actions, expressions, and any objects present
            3. VISUAL ELEMENTS: Colors, composition, camera angles, visual quality
            4. ATMOSPHERE & MOOD: Overall tone, emotions conveyed, artistic style
            5. TEXT & GRAPHICS: Any visible text, signs, graphics, or overlays
            
            Be extremely specific and detailed in your description."""
                },
            ]
        )

        try:
            # Generate response
            response, _ = self.model.chat(self.tokenizer, query=query, history=None)

            # Clean up temp file
            if os.path.exists(temp_frame_path):
                os.unlink(temp_frame_path)

            return VideoAnalysis(
                success=True,
                filename=filename,
                description=response,
                duration=duration,
                key_moments=None,
            )

        except Exception as e:
            # Clean up temp file
            if os.path.exists(temp_frame_path):
                os.unlink(temp_frame_path)
            raise Exception(f"Video analysis failed: {str(e)}")

    async def analyze_images(
        self, temp_files: List[Tuple[str, str]]
    ) -> List[ImageAnalysis]:
        """Analyze batch of images and return detailed descriptions"""
        # Load model if not already loaded
        self.load_model()

        results = []

        for i, (temp_path, original_filename) in enumerate(temp_files):
            try:
                # Create detailed prompt for image analysis
                query = self.tokenizer.from_list_format(
                    [
                        {"image": temp_path},
                        {
                            "text": """Provide an extremely detailed description of this image. Include:
                    
                    1. MAIN SUBJECTS: People, animals, or primary objects - describe appearance, poses, expressions, clothing/features
                    2. SETTING & ENVIRONMENT: Location, background, surroundings, architecture, nature elements
                    3. COLORS & LIGHTING: Dominant colors, lighting conditions, shadows, brightness, contrast
                    4. COMPOSITION & STYLE: Camera angle, framing, artistic style, visual techniques
                    5. DETAILS & TEXTURES: Surface textures, materials, patterns, fine details
                    6. MOOD & ATMOSPHERE: Emotional tone, feeling conveyed, artistic mood
                    7. TEXT & GRAPHICS: Any visible text, signs, logos, or graphic elements
                    8. ACTION & MOVEMENT: Any implied motion, gestures, or dynamic elements
                    
                    Be extremely thorough and specific."""
                        },
                    ]
                )

                # Generate response
                response, _ = self.model.chat(self.tokenizer, query=query, history=None)

                results.append(
                    ImageAnalysis(
                        image_id=f"img_{i+1:03d}",
                        filename=original_filename,
                        description=response,
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
