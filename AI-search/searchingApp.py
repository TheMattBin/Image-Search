import os
import json

import gradio as gr
from PIL import Image

# Load your images
image_directory = "path/to/your/images"
images = [Image.open(os.path.join(image_directory, img)) for img in os.listdir(image_directory)]
image_names = os.listdir(image_directory)


# Load image metadata from JSON file
def load_image_metadata(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Example usage
image_metadata = load_image_metadata('path/to/image_metadata.json')

def search_images(query):
    query = query.lower()  # Normalize the query to lower case
    matched_images = []
    
    for metadata in image_metadata:
        # Check if the query matches any relevant fields
        if (query in metadata["image_title"].lower() or 
            query in metadata["image_description"].lower() or 
            any(query in tag.lower() for tag in metadata["image_tags"]) or 
            any(query in keyword.lower() for keyword in metadata["image_keywords"])):
            # If match found, append corresponding image
            matched_images.append(metadata["image_id"])  # Store image ID or index
    
    # Retrieve images based on matched IDs
    result_images = [images[int(image_id) - 1] for image_id in matched_images]  # Adjust index if needed
    return result_images[:5]  # Return up to 5 matching images

def display_images(image_list):
    return [image.convert("RGB") for image in image_list]

with gr.Blocks() as demo:
    gr.Markdown("## Image Search App")
    
    with gr.Row():
        query_input = gr.Textbox(label="Enter your search query")
        submit_button = gr.Button("Search")
    
    output_gallery = gr.Gallery(label="Related Images").style(grid=[2])

    submit_button.click(fn=lambda query: display_images(search_images(query)), inputs=query_input, outputs=output_gallery)

if __name__ == "__main__":
    demo.launch()