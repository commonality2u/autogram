"""
Image generation module supporting multiple providers.
"""

import base64
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any

import aiohttp
from openai import OpenAI
import replicate


class ImageGenerator(ABC):
    """Abstract base class for image generators."""
    
    @abstractmethod
    async def generate(self, prompt: str, output_path: Path) -> Optional[Path]:
        """Generate an image from the prompt and save it to the output path."""
        pass


class OpenAIImageGenerator(ImageGenerator):
    """OpenAI's GPT Image model implementation."""
    
    def __init__(self, api_key: str, config: Dict[str, Any]):
        self.client = OpenAI(api_key=api_key)
        self.config = config
    
    async def generate(self, prompt: str, output_path: Path) -> Optional[Path]:
        try:
            result = await self.client.images.generate(
                model=self.config["model"],
                prompt=prompt,
                size=self.config["size"],
                quality=self.config.get("quality", "medium"),
            )
            
            # Decode and save the image
            image_data = base64.b64decode(result.data[0].b64_json)
            output_path.write_bytes(image_data)
            return output_path
            
        except Exception as e:
            print(f"OpenAI image generation failed: {e}")
            return None


class ReplicateImageGenerator(ImageGenerator):
    """Replicate API implementation."""
    
    def __init__(self, api_token: str, config: Dict[str, Any]):
        self.client = replicate.Client(api_token=api_token)
        self.config = config
    
    async def generate(self, prompt: str, output_path: Path) -> Optional[Path]:
        try:
            output = await self.client.run(
                self.config["model"],
                {
                    "prompt": prompt,
                    "negative_prompt": "text, watermark, low quality",
                    "width": 1024,
                    "height": 1024,
                }
            )
            
            # Download and save the image
            async with aiohttp.ClientSession() as session:
                async with session.get(output.url) as response:
                    image_data = await response.read()
                    output_path.write_bytes(image_data)
                    return output_path
                    
        except Exception as e:
            print(f"Replicate image generation failed: {e}")
            return None


class ImageGeneratorFactory:
    """Factory class for creating image generators."""
    
    @staticmethod
    def create(provider: str, credentials: Dict[str, str], config: Dict[str, Any]) -> Optional[ImageGenerator]:
        """Create an image generator instance based on the provider."""
        if provider == "openai" and credentials.get("OPENAI_API_KEY"):
            return OpenAIImageGenerator(
                api_key=credentials["OPENAI_API_KEY"],
                config=config["providers"]["openai"]
            )
        elif provider == "replicate" and credentials.get("REPLICATE_API_TOKEN"):
            return ReplicateImageGenerator(
                api_token=credentials["REPLICATE_API_TOKEN"],
                config=config["providers"]["replicate"]
            )
        return None 
