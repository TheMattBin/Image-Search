import gradio as gr
import zipfile
import os
import shutil
from PIL import Image
import json

from metadata_generation import run_example

# Define a single directory for all processed images
PROCESSED_DIR = "processed_images"

# Create the directory if it doesn't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

#TODO: Group all uploaded files into a desinated folder
def process_files(files):
    new_metadata = []
    prompt = "<MORE_DETAILED_CAPTION>"
    text_input = "Extract metadata, such as image_id, image_title, image_description,image_date, image_resolution, \
        image_orientation, image_tags, image_keywords, image_file_creation_date, and formatted in json."

    processed_images = []
    
    for file in files:
        # Determine the file path for saving
        file_path = os.path.join(PROCESSED_DIR, os.path.basename(file.name))
        
        if file.name.endswith('.zip'):
            # Extract zip file to the same directory
            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall(PROCESSED_DIR)
                
                # Traverse the extracted directory for images
                for root, dirs, files in os.walk(PROCESSED_DIR):
                    for img_file in files:
                        if img_file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):  # Check for image formats
                            img_path = os.path.join(root, img_file)
                            processed_images.append(img_path)  # Add image path for processing
        
        else:
            # Copy individual image files directly to PROCESSED_DIR
            shutil.copy(file, file_path)  # Use shutil.copy to copy the file
            processed_images.append(file_path)  # Add individual image path for processing
    
    # Extract metadata for each image
    for image_path in processed_images:
        image = Image.open(image_path).convert("RGB")
        metadata = run_example(prompt, text_input, image)
        new_metadata.append(metadata)

    # Define the path to your JSON file
    metadata_file_path = 'images_metadata.json'

    # Check if the JSON file exists
    if os.path.exists(metadata_file_path):
        # Read existing metadata
        with open(metadata_file_path, 'r') as json_file:
            images_metadata = json.load(json_file)
    else:
        # If the file doesn't exist, start with an empty list
        images_metadata = []

    # Append new metadata to the existing list
    images_metadata.append(new_metadata)

    # Write the updated list back to the JSON file
    with open(metadata_file_path, 'w') as json_file:
        json.dump(images_metadata, json_file, indent=4)
    
    return processed_images


with gr.Blocks() as demo:
    gr.Markdown("## Upload Images or Zip Files")
    
    upload_button = gr.File(label="Upload Images or Zip File", file_count="multiple")
    output_gallery = gr.Gallery(label="Processed Images")
    
    upload_button.upload(process_files, upload_button, output_gallery)

if __name__ == "__main__":
    demo.launch()