import os
from dotenv import load_dotenv
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel

from pinecone import Pinecone
from metadata_generation import run_example

load_dotenv()  # Load environment variables from .env file

# Initialize Pinecone and connect to an index
def init_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)
    index = pc.Index("pats")
    return index
    

# Load CLIP model and processor
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    return model, processor


# Embed and index images
#TODO: Add metadata extraction by LLM
def index_images(index, model, processor, image_paths):
    prompt = "<MORE_DETAILED_CAPTION>"
    text_input = "Extract metadata, such as image_id, image_title, image_description,image_date, image_resolution, \
        image_orientation, image_tags, image_keywords, image_file_creation_date, and formatted in json."
    
    vectors = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt", padding=True)
        metadata = run_example(prompt, text_input, image)
        with torch.no_grad():
            embeddings = model.get_image_features(**inputs)

        # Convert embeddings to a list and upsert to Pinecone
        vector = embeddings[0].tolist()

        vectors.append({
            "id": image_path,  # Use the image path as the ID
            "values": vector,
            "metadata": metadata
        })
        
    index.upsert(vectors=vectors)


# Query similar images
def query_images(index, model, processor, query_image_path, top_k):
    query_image = Image.open(query_image_path).convert("RGB")
    inputs = processor(images=query_image, return_tensors="pt", padding=True)

    with torch.no_grad():
        query_embedding = model.get_image_features(**inputs)

    query = query_embedding.squeeze().tolist()

    # Query Pinecone for similar images, leave "" for default namespace
    results = index.query(vector=query, top_k=top_k, include_values=True)  # Get top 5 similar images

    return results

#TODO: Combine the two scripts (upload.py) into one script
# # Main function to run the script
# if __name__ == "__main__":
#     index = init_pinecone()

#     # Load the CLIP model and processor
#     model, processor = load_clip_model()

#     # Specify paths to your images for indexing
#     image_paths = ['./data/9.jpg', './data/10.jpg']  # Add your image paths here
#     index_images(index, model, processor, image_paths)

#     # Query with a specific image
#     query_image_path = './data/6.jpg'  # Path to your query image
#     similar_images = query_images(index, model, processor, query_image_path)

#     print("Similar Images:")
#     for match in similar_images['matches']:
#         print(f"Image ID: {match['id']}, Score: {match['score']}")