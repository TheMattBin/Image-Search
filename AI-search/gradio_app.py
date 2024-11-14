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

def query_similar_images(query_image, top_k=5):
    index = init_pinecone()

    # Load the CLIP model and processor
    model, processor = load_clip_model()

    # Query similar images
    results = query_images(index, model, processor, query_image, top_k=top_k)

    # for match in results['matches']:
    #     print(f"Image ID: {match['id']}, Score: {match['score']}")

    # return results
    return [match['id'] for match in sorted(results['matches'], key=lambda x: x['score'], reverse=True)]

#TODO: Add query by key words/tags
with gr.Blocks() as demo:
    with gr.Tab(label="Upload Images or Zip Files"):
        gr.Markdown("## Upload Images or Zip Files")
        
        upload_button = gr.File(label="Upload Images or Zip File", file_count="multiple")
        output_message = gr.Textbox(label="Output Message")
        # output_gallery = gr.Gallery(label="Processed Images")
        # upload_button.upload(lambda files: "Upload succeeded", upload_button, output_message)
        
        upload_button.upload(image_embedding, upload_button, output_message)

    with gr.Tab(label="Query Similar Images by Uploading an Image"):
        gr.Markdown("## Query Similar Images")
        
        query_button = gr.File(label="Upload Query Image")
        output = gr.Label(label="Similar Images")
        
        query_button.upload(query_similar_images, query_button, output)
    

if __name__ == "__main__":
    demo.launch()