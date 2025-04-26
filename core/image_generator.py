"""
Image generation module supporting multiple providers.
"""

import base64
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any

import aiohttp
import replicate


class ImageGenerator(ABC):
    """Abstract base class for image generators."""
    
    @abstractmethod
    async def generate(self, prompt: str, output_path: Path) -> Optional[Path]:
        """Generate an image from the prompt and save it to the output path."""
        pass


class OpenAIImageGenerator(ImageGenerator):
    """OpenAI's GPT Image model implementation using direct HTTP POST (async)."""
    
    def __init__(self, api_key: str, config: Dict[str, Any]):
        self.api_key = api_key
        self.config = config
    
    async def generate(self, prompt: str, output_path: Path) -> Optional[Path]:
        url = "https://api.openai.com/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.config["model"],
            "prompt": prompt,
            "n": 1,
            "size": self.config.get("size", "1024x1024"),
        }
        if "quality" in self.config:
            payload["quality"] = self.config["quality"]
        if "style" in self.config:
            payload["style"] = self.config["style"]
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        print(f"OpenAI image generation failed: HTTP {resp.status} - {await resp.text()}")
                        return None
                    data = await resp.json()
                    b64 = data.get("data", [{}])[0].get("b64_json")
                    if not b64:
                        print(f"OpenAI image generation failed: No image data in response: {data}")
                        return None
                    image_data = base64.b64decode(b64)
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
