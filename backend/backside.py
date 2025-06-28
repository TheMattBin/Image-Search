import os
from PIL import Image
from dotenv import load_dotenv

import weaviate
from weaviate.classes.init import Auth
# from weaviate.collections import Property, CollectionConfig
# from weaviate.collections import Collection
from weaviate.util import generate_uuid5  # Generate a deterministic ID

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig
from transformers import CLIPProcessor, CLIPModel

load_dotenv()

weaviate_url = os.environ["WEAVIATE_URL"]
weaviate_api_key = os.environ["WEAVIATE_API_KEY"]

# 1. Initialize Weaviate client
# Ensure WEAVIATE_URL and WEAVIATE_API_KEY environment variables are set
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=f"https://{weaviate_url}",
    auth_credentials=Auth.api_key(api_key=weaviate_api_key),
)

# Check if Weaviate client is ready
if client.is_ready():
    print("Weaviate client connected successfully!")
else:
    print("Failed to connect to Weaviate. Please check your URL and API key.")
    exit() # Exit if connection fails

# 2. Load SmolVLM2 model and processor for both embeddings and potential text encoding
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct" # Recommended model for vision/video tasks [4]

# Load processor and model
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)

# Load CLIP model for lightweight image embedding
clip_model_id = "openai/clip-vit-base-patch16"
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
clip_model = CLIPModel.from_pretrained(clip_model_id).to(device)

EMBEDDING_DIM = 512

def generate_caption(image) -> str:
    """Generates a caption for the given image using SmolVLM2.
    Accepts either a PIL.Image.Image or an image path as str.
    """
    # if isinstance(image, str):
    #     image = Image.open(image).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image},
                {"type": "text", "text": "Can you describe this image?"},
            ]
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=128)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    caption = generated_texts[0]
    return caption.strip()

def extract_embedding(image) -> list:
    """Extracts a vector embedding for the given image using CLIP."""
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    embedding = image_features.cpu().numpy().flatten().tolist()
    return embedding

def encode_text_for_query(text: str) -> list:
    """Encodes a text query into an embedding using SmolVLM2's text encoder."""
    # SmolVLM2 uses SmolLM2 as its language backbone [2].
    # We need to get the text embedding from this language model.
    # This might involve passing a simple message like "query: <text>" and getting the LM's hidden state.
    
    # A common way to get text embeddings from VLMs that also have a text encoder (like SmolVLM)
    # is to use the model's text processing component.
    # We can use the processor to prepare the text and then pass it to the text backbone.
    
    # SmolVLM is a CausalLM, so direct text embedding extraction might be different from CLIP.
    # One approach: encode text, get the last hidden state of the text tokens, and pool it.
    
    messages = [
        {"role": "user", "content": text}
    ]
    inputs = processor(messages=messages, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        
        # Assuming the embedding dimension is EMBEDDING_DIM
        text_embedding = outputs.last_hidden_state[:, -1, :].squeeze().cpu().numpy().tolist()
        
    return text_embedding

def push_to_weaviate(image_id: str, embedding: list, caption: str, image_url: str = ""):
    collection_name = "ImageObject"
    imgObject = client.collections.get(collection_name)
    data_object = {
        "description": caption,
        "imageUrl": image_url,
        "imageId": image_id,
    }
    uuid = imgObject.data.insert(
        properties=data_object,
        vector=embedding,
        uuid=generate_uuid5(data_object)
    )

    # properties = [
    #     Property(name="caption", data_type="text"),
    #     Property(name="imageUrl", data_type="string"),
    # ]

    # # Define collection config without VectorIndexConfig (pass vector_index_config as dict)
    # config = CollectionConfig(
    #     properties=properties,
    #     vector_index_config={
    #         "distance": "cosine"
    #     }
    # )

    # try:
    #     # Check if collection exists
    #     if collection_name not in client.collections.list_all():
    #         # Create collection with config
    #         client.collections.create(collection_name, config)
    #         print(f"Created Weaviate collection: {collection_name}")
    #     else:
    #         print(f"Collection {collection_name} already exists.")

    #     # Get collection object
    #     collection: Collection = client.collections.get(collection_name)

    #     # Insert data object with vector embedding and properties
    #     collection.data.insert(
    #         properties={
    #             "caption": caption,
    #             "imageUrl": image_url,
    #         },
    #         vector=embedding,
    #         uuid=image_id,
    #     )
    #     print(f"Image data for ID {image_id} pushed to Weaviate successfully.")

    # except Exception as e:
    #     print(f"Error pushing data to Weaviate: {e}")


# Example usage
if __name__ == "__main__":
    # --- Part 1: Indexing an image with caption and embedding ---
    image_path = "../images/cats/1.jpg" # <--- IMPORTANT: Change this to your image file
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}. Please update 'image_path'.")
        exit()

    try:
        # image = Image.open(image_path).convert("RGB")
        image = image_path
        image_id = "unique-image-id-123" # Make this truly unique for each image
        image_url = ""  # Optional: URL where image is stored

        # Generate caption
        caption = generate_caption(image)
        if "Assistant:" in caption:
            caption = caption.split("Assistant:", 1)[1].strip()
        print(f"Generated caption: {caption}")

        # Extract embedding
        embedding = extract_embedding(image)
        print(f"Extracted embedding with dimension: {len(embedding)}")
        # IMPORTANT: Assert that len(embedding) == EMBEDDING_DIM
        if len(embedding) != EMBEDDING_DIM:
             print(f"Warning: Expected embedding dimension {EMBEDDING_DIM}, but got {len(embedding)}. Adjust EMBEDDING_DIM.")

        # Push to Weaviate
        push_to_weaviate(image_id, embedding, caption, image_url)

    except Exception as e:
        print(f"Error during image processing or indexing: {e}")
    finally:
        client.close()

    print("\n--- Part 2: Querying Weaviate ---")

    # --- Text Query Example ---
    query_text = "a cat playing in the park"
    try:
        text_query_embedding = encode_text_for_query(query_text)
        print(f"Encoded text query '{query_text}' to embedding with dimension: {len(text_query_embedding)}")

        # Perform vector similarity search in Weaviate
        # The .near_vector() method is used for vector similarity search
        # Pass the text query embedding as the vector to search for.
        
        # Weaviate's `near_vector` expects a dictionary with `vector` key
        # and optionally `distance` or `certainty`.
        
        results_text_query = (
            client.query.get("ImageObject", ["caption", "imageUrl"])
            .with_near_vector({"vector": text_query_embedding})
            .with_limit(3) # Get top 3 similar images
            .do()
        )
        print(f"\nResults for text query '{query_text}':")
        if "data" in results_text_query and "Get" in results_text_query["data"] and "ImageObject" in results_text_query["data"]["Get"]:
            for i, item in enumerate(results_text_query["data"]["Get"]["ImageObject"]):
                print(f"  Result {i+1}: Caption: '{item['caption']}', Image URL: '{item['imageUrl']}'")
        else:
            print("No results found for text query.")

    except NotImplementedError as e:
        print(f"Skipping text query: {e}")
    except Exception as e:
        print(f"Error during text query: {e}")

    # --- Image Query Example ---
    # query_image_path = "path/to/your/query_image.jpg" # <--- IMPORTANT: Change this to your query image file
    # if not os.path.exists(query_image_path):
    #     print(f"Error: Query image file not found at {query_image_path}. Please update 'query_image_path'.")
    # else:
    #     try:
    #         query_image = Image.open(query_image_path).convert("RGB")
    #         image_query_embedding = extract_embedding(query_image)
    #         print(f"Extracted embedding for query image with dimension: {len(image_query_embedding)}")

    #         results_image_query = (
    #             client.query.get("ImageObject", ["caption", "imageUrl"])
    #             .with_near_vector({"vector": image_query_embedding})
    #             .with_limit(3) # Get top 3 similar images
    #             .do()
    #         )
    #         print(f"\nResults for image query (from {query_image_path}):")
    #         if "data" in results_image_query and "Get" in results_image_query["data"] and "ImageObject" in results_image_query["data"]["Get"]:
    #             for i, item in enumerate(results_image_query["data"]["Get"]["ImageObject"]):
    #                 print(f"  Result {i+1}: Caption: '{item['caption']}', Image URL: '{item['imageUrl']}'")
    #         else:
    #             print("No results found for image query.")
    #     except NotImplementedError as e:
    #         print(f"Skipping image query: {e}")
    #     except Exception as e:
    #         print(f"Error during image query: {e}")

