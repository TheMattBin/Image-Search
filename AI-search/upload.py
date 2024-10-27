import gradio as gr
import zipfile
import os
import shutil

# Define a single directory for all processed images
PROCESSED_DIR = "processed_images"

# Create the directory if it doesn't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

def process_files(files):
    processed_images = []
    
    for file in files:
        print(file)
        # Determine the file path for saving
        file_path = os.path.join(PROCESSED_DIR, os.path.basename(file.name))
        print(file_path)
        
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
    
    return processed_images


with gr.Blocks() as demo:
    gr.Markdown("## Upload Images or Zip Files")
    
    upload_button = gr.File(label="Upload Images or Zip File", file_count="multiple")
    output_gallery = gr.Gallery(label="Processed Images")
    
    upload_button.upload(process_files, upload_button, output_gallery)

if __name__ == "__main__":
    demo.launch()