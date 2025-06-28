# vision_models.py
from PIL import Image
from typing import Union, List

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, CLIPProcessor, CLIPModel

class VisionModels:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        self.device: str = device
        self.caption_model_id: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
        self.processor = AutoProcessor.from_pretrained(self.caption_model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.caption_model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(self.device)
        self.clip_model_id: str = "openai/clip-vit-base-patch16"
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_id)
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_id).to(self.device)

    def generate_caption(self, image: Union[str, Image.Image]) -> str:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "url": image},
                {"type": "text", "text": "Can you describe this image?"},
            ]
        }]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)
        generated_ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=128)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        caption: str = generated_texts[0]
        if "Assistant:" in caption:
            caption = caption.split("Assistant:", 1)[1].strip()
        return caption.strip()

    def extract_embedding(self, image: Union[str, Image.Image]) -> List[float]:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        embedding: List[float] = image_features.cpu().numpy().flatten().tolist()
        return embedding