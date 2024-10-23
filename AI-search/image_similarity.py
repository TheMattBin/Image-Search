import gradio as gr
from transformers import pipeline
from PIL import Image
import torch
import os

# Load the image feature extraction model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "google/vit-base-patch16-384"  # You can choose another model if preferred
feature_extractor = pipeline(task="image-feature-extraction", model=model_name, device=DEVICE, pool=True)

# Create a directory for candidate images
CANDIDATE_IMAGES_FOLDER = 'candidate_images'
os.makedirs(CANDIDATE_IMAGES_FOLDER, exist_ok=True)

# Function to extract features from an image
def extract_features(image):
    features = feature_extractor(image)
    # Convert to tensor if necessary
    return torch.tensor(features[0])  # Assuming features[0] is a list that can be converted

# Function to calculate cosine similarity
def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def find_similar_images(uploaded_image):
    """Find similar images based on the uploaded image."""
    # Extract features from the uploaded image
    query_features = extract_features(uploaded_image)

    # Find similar images in candidate images folder
    similarities = {}

    for candidate_file in os.listdir(CANDIDATE_IMAGES_FOLDER):
        candidate_file_path = os.path.join(CANDIDATE_IMAGES_FOLDER, candidate_file)
        candidate_image = Image.open(candidate_file_path).convert("RGB")
        candidate_features = extract_features(candidate_image)
        similarity_score = cosine_similarity(query_features, candidate_features)
        similarities[candidate_file] = similarity_score

    # Sort by similarity score (highest first)
    sorted_similarities = dict(sorted(similarities.items(), key=lambda item: item[1], reverse=True))
    print(sorted_similarities)

    # Return top 5 similar images
    top_similar_images = list(sorted_similarities.keys())[:5]
    
    return [os.path.join(CANDIDATE_IMAGES_FOLDER, img) for img in top_similar_images]


# # Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Image Similarity Search")
    uploaded_image = gr.Image(type="pil", label="Upload Image")
    submit_button = gr.Button("Find Similar Images")
    
    output_gallery = gr.Gallery(label="Similar Images", show_label=True)

    submit_button.click(fn=find_similar_images, inputs=uploaded_image, outputs=output_gallery)

# Launch the app
demo.launch(share=True)