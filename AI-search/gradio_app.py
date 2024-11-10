import gradio as gr

from upload import upload_files
from pinecone_db import init_pinecone, load_clip_model, index_images

def image_embedding(files):
    index = init_pinecone()

    # Load the CLIP model and processor
    model, processor = load_clip_model()

    # Specify paths to your images for indexing
    image_paths = upload_files(files)
    index_images(index, model, processor, image_paths)


with gr.Blocks() as demo:
    gr.Markdown("## Upload Images or Zip Files")
    
    upload_button = gr.File(label="Upload Images or Zip File", file_count="multiple")
    output_gallery = gr.Gallery(label="Processed Images")
    
    upload_button.upload(image_embedding, upload_button, output_gallery)

if __name__ == "__main__":
    demo.launch()