import os
import pinecone
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

# Initialize Pinecone
def init_pinecone():
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),  # Set your Pinecone API key as an environment variable
        environment=os.getenv("PINECONE_ENVIRONMENT")  # Set your Pinecone environment
    )

# Create or connect to an index
def create_index(index_name):
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=512)  # CLIP model output dimension

    return pinecone.Index(index_name)

# Load CLIP model and processor
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    return model, processor

# Embed and index images
def index_images(index, model, processor, image_paths):
    for image_path in image_paths:
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            embeddings = model.get_image_features(**inputs)
        
        # Convert embeddings to a list and upsert to Pinecone
        vector = embeddings[0].tolist()
        index.upsert(vectors=[(image_path, vector)])

# Query similar images
def query_images(index, model, processor, query_image_path):
    query_image = Image.open(query_image_path)
    inputs = processor(images=query_image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        query_embedding = model.get_image_features(**inputs)

    # Query Pinecone for similar images
    results = index.query(queries=query_embedding.tolist(), top_k=5)  # Get top 5 similar images
    
    return results

# Main function to run the script
if __name__ == "__main__":
    init_pinecone()
    
    index_name = "image-search"
    index = create_index(index_name)
    
    # Load the CLIP model and processor
    model, processor = load_clip_model()
    
    # Specify paths to your images for indexing
    image_paths = ['./data/image1.jpg', './data/image2.jpg']  # Add your image paths here
    index_images(index, model, processor, image_paths)

    # Query with a specific image
    query_image_path = './data/queryImage.jpg'  # Path to your query image
    similar_images = query_images(index, model, processor, query_image_path)

    print("Similar Images:")
    for match in similar_images['matches']:
        print(f"Image ID: {match['id']}, Score: {match['score']}")