import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import cv2
import os
from typing import List, Tuple
from .models import VideoAnalysis, ImageAnalysis


class VisionService:
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_name = "Qwen/Qwen2-VL-7B-Instruct"

    def load_model(self):
        """Load the Qwen2-VL model"""
        if self.model is None:
            print(f"Loading {self.model_name} model...")

            # Load model
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name, torch_dtype=torch.bfloat16, device_map="auto"
            )

            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_name)

            print("Qwen2-VL model loaded successfully!")

    def is_model_loaded(self):
        """Check if model is loaded"""
        return self.model is not None

    def get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            return frame_count / fps if fps > 0 else 0
        except Exception as e:
            print(f"Error getting video duration: {e}")
            return 0

    def generate_detailed_description(self, image_path: str) -> str:
        """Generate detailed description using Qwen2-VL"""
        try:
            # Create detailed prompt following Qwen2-VL format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {
                            "type": "text",
                            "text": """Provide an extremely detailed description of this image. Include:

1. MAIN SUBJECTS: People, animals, or primary objects - describe appearance, poses, expressions, clothing/features
2. SETTING & ENVIRONMENT: Location, background, surroundings, architecture, nature elements  
3. COLORS & LIGHTING: Dominant colors, lighting conditions, shadows, brightness, contrast
4. COMPOSITION & STYLE: Camera angle, framing, artistic style, visual techniques
5. DETAILS & TEXTURES: Surface textures, materials, patterns, fine details
6. MOOD & ATMOSPHERE: Emotional tone, feeling conveyed, artistic mood
7. TEXT & GRAPHICS: Any visible text, signs, logos, or graphic elements
8. ACTION & MOVEMENT: Any implied motion, gestures, or dynamic elements

Be extremely thorough and specific. Capture every important visual element and detail.""",
                        },
                    ],
                }
            ]

            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process vision info (for Qwen2-VL, use default image_patch_size=14)
            image_inputs, video_inputs = process_vision_info(messages)

            # Prepare inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            # Generate response
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            return output_text[0] if output_text else "Unable to generate description"

        except Exception as e:
            return f"Error generating description: {str(e)}"

    async def analyze_video(self, video_path: str, filename: str) -> VideoAnalysis:
        """Analyze video and return detailed description"""
        try:
            # Load model if not already loaded
            self.load_model()

            # Get video duration
            duration = self.get_video_duration(video_path)

            # Create video analysis prompt - Qwen2-VL can analyze videos directly
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "max_pixels": 360 * 420,
                            "fps": 1.0,
                        },
                        {
                            "type": "text",
                            "text": """Analyze this video thoroughly and provide an extremely detailed description. Include:

1. SCENE & SETTING: Describe the environment, location, lighting, time of day, weather conditions
2. PEOPLE & OBJECTS: Detail all people, their appearance, clothing, actions, expressions, and any objects present
3. ACTIONS & EVENTS: Chronological description of what happens throughout the video
4. VISUAL ELEMENTS: Colors, composition, camera angles, movement, visual quality
5. ATMOSPHERE & MOOD: Overall tone, emotions conveyed, artistic style
6. TEXT & GRAPHICS: Any visible text, signs, graphics, or overlays

Be extremely specific and detailed. Provide a comprehensive analysis that captures every important visual element.""",
                        },
                    ],
                }
            ]

            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process vision info
            image_inputs, video_inputs = process_vision_info(messages)

            # Prepare inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            # Generate response
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            description = output_text[0] if output_text else "Unable to analyze video"

            return VideoAnalysis(
                success=True,
                filename=filename,
                description=description,
                duration=duration,
                key_moments=None,
            )

        except Exception as e:
            raise Exception(f"Video analysis failed: {str(e)}")

    async def analyze_images(
        self, temp_files: List[Tuple[str, str]]
    ) -> List[ImageAnalysis]:
        """Analyze batch of images and return detailed descriptions"""
        try:
            # Load model if not already loaded
            self.load_model()

            results = []

            for i, (temp_path, original_filename) in enumerate(temp_files):
                try:
                    # Generate description
                    description = self.generate_detailed_description(temp_path)

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

        except Exception as e:
            raise Exception(f"Image batch analysis failed: {str(e)}")
