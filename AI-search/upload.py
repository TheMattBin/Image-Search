import os
import zipfile
import gradio as gr
from PIL import Image

def process_images(files):
    # List to hold processed images
    processed_images = []
    
    for file in files:
        if file.endswith('.zip'):
            # If the uploaded file is a zip, extract it
            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall("extracted_images")
                # Process each image in the extracted folder
                for img_file in os.listdir("extracted_images"):
                    img_path = os.path.join("extracted_images", img_file)
                    processed_images.append(process_image(img_path))
        else:
            # Process individual image files
            processed_images.append(process_image(file))
    
    return processed_images

def process_image(image_path):
    # Open and process the image (example: convert to grayscale)
    img = Image.open(image_path)
    img = img.convert("L")  # Convert to grayscale (you can modify this)
    return img

with gr.Blocks() as demo:
    gr.Markdown("## Upload Images or Zip Files")
    
    with gr.Row():
        upload_button = gr.UploadButton("Upload Images or Zip File", file_types=["image", "zip"], file_count="multiple")
        output_gallery = gr.Gallery(label="Processed Images").style(grid=[2])
    
    upload_button.upload(process_images, upload_button, output_gallery)

if __name__ == "__main__":
    demo.launch()