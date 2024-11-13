import gradio as gr

from upload import upload_files
from pinecone_db import init_pinecone, load_clip_model, index_images, query_images

def image_embedding(files):
    index = init_pinecone()

    # Load the CLIP model and processor
    model, processor = load_clip_model()

    # Specify paths to your images for indexing
    image_paths = upload_files(files)
    index_images(index, model, processor, image_paths)

def query_similar_images(query_image):
    index = init_pinecone()

    # Load the CLIP model and processor
    model, processor = load_clip_model()

    # Query similar images
    results = query_images(index, model, processor, query_image)

    for match in results['matches']:
        print(f"Image ID: {match['id']}, Score: {match['score']}")

    return results

with gr.Blocks() as demo:
    with gr.Tab(label="Upload Images or Zip Files"):
        gr.Markdown("## Upload Images or Zip Files")
        
        upload_button = gr.File(label="Upload Images or Zip File", file_count="multiple")
        output_gallery = gr.Gallery(label="Processed Images")
        
        upload_button.upload(image_embedding, upload_button, output_gallery)

    with gr.Tab(label="Query Similar Images by Uploading an Image"):
        gr.Markdown("## Query Similar Images")
        
        query_button = gr.File(label="Upload Query Image")
        output = gr.Label(label="Similar Images")
        
        query_button.upload(query_similar_images, query_button, output)
    

if __name__ == "__main__":
    demo.launch()