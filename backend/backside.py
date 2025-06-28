import os
from PIL import Image
from dotenv import load_dotenv

import weaviate
from weaviate.classes.init import Auth
from weaviate.util import generate_uuid5

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig
from transformers import CLIPProcessor, CLIPModel

load_dotenv()

weaviate_url = os.environ["WEAVIATE_URL"]
weaviate_api_key = os.environ["WEAVIATE_API_KEY"]

# 1. Initialize Weaviate client
# Ensure WEAVIATE_URL and WEAVIATE_API_KEY environment variables are set
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=f"https://{weaviate_url}",
    auth_credentials=Auth.api_key(api_key=weaviate_api_key),
)

# Check if Weaviate client is ready
if client.is_ready():
    print("Weaviate client connected successfully!")
else:
    print("Failed to connect to Weaviate. Please check your URL and API key.")
    exit() # Exit if connection fails

# 2. Load SmolVLM2 model and processor for both embeddings and potential text encoding
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct" # Recommended model for vision/video tasks [4]

# Load processor and model
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)

# Load CLIP model for lightweight image embedding
clip_model_id = "openai/clip-vit-base-patch16"
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
clip_model = CLIPModel.from_pretrained(clip_model_id).to(device)

EMBEDDING_DIM = 512

def generate_caption(image) -> str:
    """Generates a caption for the given image using SmolVLM2.
    Accepts either a PIL.Image.Image or an image path as str.
    """
    # if isinstance(image, str):
    #     image = Image.open(image).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image},
                {"type": "text", "text": "Can you describe this image?"},
            ]
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=128)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    caption = generated_texts[0]
    return caption.strip()

def extract_embedding(image) -> list:
    """Extracts a vector embedding for the given image using CLIP."""
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    embedding = image_features.cpu().numpy().flatten().tolist()
    return embedding

# def encode_text_for_query(text: str) -> list:
#     """Encodes a text query into an embedding using SmolVLM2's text encoder."""
    
#     messages = [
#         {"role": "user", "content": text}
#     ]
#     inputs = processor(messages=messages, return_tensors="pt").to(device)
    
#     with torch.no_grad():
#         outputs = model.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        
#         # Assuming the embedding dimension is EMBEDDING_DIM
#         text_embedding = outputs.last_hidden_state[:, -1, :].squeeze().cpu().numpy().tolist()
        
#     return text_embedding

def push_to_weaviate(image_id: str, embedding: list, caption: str, image_url: str = ""):
    collection_name = "ImageObject"
    imgObject = client.collections.get(collection_name)
    data_object = {
        "description": caption,
        "imageUrl": image_url,
        "imageId": image_id,
    }
    uuid = imgObject.data.insert(
        properties=data_object,
        vector=embedding,
        uuid=generate_uuid5(data_object)
    )

# Example usage
if __name__ == "__main__":
    # --- Part 1: Indexing an image with caption and embedding ---
    image_path = "../images/cats/1.jpg" # <--- IMPORTANT: Change this to your image file
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}. Please update 'image_path'.")
        exit()

    try:
        # image = Image.open(image_path).convert("RGB")
        image = image_path
        image_id = "unique-image-id-123" # Make this truly unique for each image
        image_url = ""  # Optional: URL where image is stored

        # Generate caption
        caption = generate_caption(image)
        if "Assistant:" in caption:
            caption = caption.split("Assistant:", 1)[1].strip()
        print(f"Generated caption: {caption}")

        # Extract embedding
        embedding = extract_embedding(image)
        print(f"Extracted embedding with dimension: {len(embedding)}")
        # IMPORTANT: Assert that len(embedding) == EMBEDDING_DIM
        if len(embedding) != EMBEDDING_DIM:
             print(f"Warning: Expected embedding dimension {EMBEDDING_DIM}, but got {len(embedding)}. Adjust EMBEDDING_DIM.")

        # Push to Weaviate
        push_to_weaviate(image_id, embedding, caption, image_url)

    except Exception as e:
        print(f"Error during image processing or indexing: {e}")
    finally:
        client.close()