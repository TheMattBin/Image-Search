import weaviate
import os
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig

# 1. Initialize Weaviate client
# Ensure WEAVIATE_URL and WEAVIATE_API_KEY environment variables are set
client = weaviate.Client(
    url=f"https://{os.getenv('WEAVIATE_URL')}",
    auth_client_secret=weaviate.AuthApiKey(api_key=os.getenv('WEAVIATE_API_KEY')),
)

# Check if Weaviate client is ready
if client.is_ready():
    print("Weaviate client connected successfully!")
else:
    print("Failed to connect to Weaviate. Please check your URL and API key.")
    exit() # Exit if connection fails

# 2. Load SmolVLM2 model and processor for both embeddings and potential text encoding
device = "cuda" if torch.cuda.is_available() else "cpu"
smolvlm_model_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct" # Recommended model for vision/video tasks [4]

# Load processor and model
smolvlm_processor = AutoProcessor.from_pretrained(smolvlm_model_id, trust_remote_code=True)
smolvlm_model = AutoModelForCausalLM.from_pretrained(smolvlm_model_id, trust_remote_code=True).to(device).eval()

# Verify the embedding dimension of SmolVLM2's vision encoder
# The vision encoder for SmolVLM2-2.2B uses SigLIP-SO400M [2]
# We need to find its output dimension. This is often 768 or 1024 or 1536 for common models.
# A common way to get the embedding dimension is to check the model's configuration
# or by running a sample inference.
# For SmolVLM, the vision config is part of the overall model config [5]
# Let's assume an output dimension, you should confirm this from SmolVLM2 documentation or by inspection.
# For many VLMs, 768 or 1024 is a common output dimension for the vision encoder.
# Let's assume 1024 for this example, but *verify this* against actual model output or documentation.
SMOLVLM_EMBEDDING_DIM = 1024 # Placeholder, *IMPORTANT: Verify this with the actual model output*

def generate_caption(image: Image.Image) -> str:
    """Generates a caption for the given image using SmolVLM2."""
    messages = [
        {"role": "user", "content": "<image><CAPTION>"}
    ]
    inputs = smolvlm_processor(messages=messages, images=image, return_tensors="pt").to(device)
    
    # Generate the caption
    generated_ids = smolvlm_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=50,
        num_beams=3,
        early_stopping=True,
    )
    caption = smolvlm_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return caption.strip()

def extract_embedding(image: Image.Image) -> list:
    """Extracts a vector embedding for the given image using SmolVLM2's vision encoder."""
    # SmolVLM2 processes images within its `apply_chat_template` or similar.
    # To get a dedicated image embedding, we need to access the vision encoder part.
    # The architecture overview for SmolVLM mentions SigLIP vision encoders [2].
    # We can try to get the raw vision features before they are combined with text.
    # This might require peeking into the model's internal structure.
    
    # A common pattern for VLMs that use a separate vision encoder is to pass the image
    # through that encoder directly.
    # SmolVLM uses SigLIP image encoders. Let's try to access it via the model's components.
    
    # This part is highly dependent on SmolVLM2's internal API.
    # Assuming `smolvlm_model.vision_model` or similar exists and outputs features.
    
    # If a direct vision_model access is not straightforward, a common workaround
    # is to get the image features used internally.
    
    # Preprocess image for the vision encoder (pixel_values)
    inputs = smolvlm_processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs["pixel_values"]

    with torch.no_grad():
        # Get vision features. This is a common way to extract image features from VLMs.
        # The exact method might vary, but many models expose a way to get the vision features.
        # SmolVLM architecture states it uses SigLIP to produce visual tokens [2].
        # We need to find the output from this part.
        # A plausible way would be to call the `get_image_features` method if it exists,
        # or pass `pixel_values` to the vision part of the model.
        
        # This is a general approach for many multimodal models to get the image features.
        # You might need to consult SmolVLM2's specific documentation/source for exact method.
        # For a model like CLIP, you would use model.get_image_features(pixel_values).
        # For SmolVLM, the visual tokens are generated after SigLIP and pixel-shuffle [2].
        # Let's assume `smolvlm_model.encode_image` or similar exists, or directly process `pixel_values`
        # to get a pooled visual representation.
        
        # Example using a common pattern for obtaining image features from a VLM:
        # Some models use the hidden states from the vision encoder, then apply a pooling layer.
        # For simplicity, let's assume the model provides a direct way to get a pooled embedding.
        
        # If smolvlm_model has a dedicated image encoding method:
        # image_features = smolvlm_model.encode_image(pixel_values)
        # If not, we might need to trace the forward pass or use a specific visual feature extractor:
        
        # A more robust approach might be to use the output of the vision part of the model
        # before it's passed to the language model.
        
        # This is a common method for VLMs where `get_image_features` is exposed or
        # by manually accessing the vision backbone's pooled output.
        # For SmolVLM2, the vision encoder is SigLIP [2].
        # The `pixel_values` are typically processed by the vision tower.
        
        # A common method in HuggingFace VLMs to get pooled vision features
        # is often through the `vision_tower` or `vision_model` and then pooling.
        
        # This is a hypothetical call based on common VLM architectures.
        # You might need to inspect SmolVLM2's `forward` method or source code.
        # As per the SmolVLM blog, visual tokens are created and concatenated [2].
        # We need the vector before concatenation.
        
        # Let's try to pass the pixel_values to the model's internal vision component.
        # This might look something like:
        
        # Use the `vision_config` from SmolVLM [5] to understand the vision part.
        # This is an approximation. The most accurate way is to find the dedicated image embedding extraction method
        # for `SmolVLM2-2.2B-Instruct` or extract features from an intermediate layer of its vision encoder.
        
        # A simpler way often provided by AutoModelForCausalLM for VLMs is to obtain image features.
        # If it's a typical VLM, the `pixel_values` are processed by an internal vision model.
        # This will produce a sequence of visual tokens. We need to average or pool them to get a single vector.
        
        # A common pattern for getting the overall image embedding (if `get_image_features` doesn't exist)
        # is to get the hidden states of the vision model and then apply pooling.
        
        # Let's assume SmolVLM2 exposes a way to get global image features.
        # This is a simplified placeholder for extracting a single image vector.
        # The SmolVLM paper talks about "visual tokens" [2]. We need a single representative vector.
        
        # A common approach for getting a single embedding from visual tokens:
        # Pass the image to the vision encoder and get the pooled output.
        # If the model integrates vision directly into the transformer without a clear `get_image_features` method,
        # we might use a dummy text input to derive the image embedding from the overall multimodal embedding.
        
        # The safest approach without knowing SmolVLM2's exact API for *just* image embedding:
        # The model's `generate` method takes `pixel_values`.
        # Its internal `vision_model` (SigLIP) would process this.
        # You would typically access the pooled output of this SigLIP model.
        
        # Placeholder for actual image feature extraction:
        # This requires detailed knowledge of SmolVLM2's internal vision architecture.
        # For now, let's assume there's a method that provides the global image embedding.
        # For models like CLIP, it's `model.get_image_features(pixel_values)`.
        # For SmolVLM, the `vision_model` component needs to be targeted.
        
        # For a VLM, typically, the `pixel_values` are processed by the vision tower,
        # and then a global representation (e.g., [CLS] token equivalent or pooled output) is extracted.
        
        # Let's simulate a direct vision encoding call for the `pixel_values`.
        # This might involve calling a specific part of the `smolvlm_model`.
        # Without explicit documentation for `SmolVLM2-2.2B-Instruct`'s vision embedding API,
        # this is the most critical part to get right.
        
        # Based on the typical usage of AutoModelForCausalLM for VLMs, we pass images
        # to the processor and then generate. To get *just* the embedding, it's trickier.
        # If the model has `vision_model` attribute (e.g., `smolvlm_model.vision_model`),
        # you might call `smolvlm_model.vision_model(pixel_values).pooler_output` or similar.
        
        # This is a common pattern for extracting image features from HuggingFace VLMs.
        # `inputs["pixel_values"]` are ready to be passed to the vision encoder.
        # You'll need to find the `vision_tower` or equivalent in `smolvlm_model`
        # For SmolVLM, the vision encoder is SigLIP [2].
        
        # Try to access the underlying vision model within SmolVLM2
        # This is a common pattern in multimodal models.
        # The `vision_model` within `SmolVLM2` will handle the `pixel_values`.
        if hasattr(smolvlm_model, 'vision_model'):
            # This is a common pattern if the vision model is directly exposed.
            # You might need to check if `vision_model` returns `last_hidden_state` and then pool.
            # Assuming `vision_model` processes `pixel_values` and gives a useful output.
            vision_outputs = smolvlm_model.vision_model(pixel_values=pixel_values)
            # Typically, get the pooled output or average last hidden states
            if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                embedding = vision_outputs.pooler_output
            else:
                # If no pooler_output, try mean pooling of the last hidden states
                embedding = vision_outputs.last_hidden_state.mean(dim=1)
        else:
            # Fallback if `vision_model` is not directly exposed.
            # This requires knowing SmolVLM2's specific API for image embedding.
            # As a last resort, one might have to run the forward pass with only image input
            # and trace where the image features are generated.
            raise NotImplementedError("Direct vision model access or image embedding method not found for SmolVLM2.")

    embedding = embedding.cpu().numpy().flatten().tolist()
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
    inputs = smolvlm_processor(messages=messages, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Pass inputs to the model. We're interested in the language model's output for the text part.
        # SmolVLM uses SmolLM2 as the language model [2].
        # We need the pooled output of the SmolLM2 for the text.
        
        # This is a common pattern for getting text embeddings from VLMs:
        # The `input_ids` will contain the tokenized text. Pass them through the language model part.
        outputs = smolvlm_model.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        
        # The last hidden state corresponding to the text tokens.
        # Typically, take the last token's hidden state or mean pool.
        # For causal LMs used in VLMs, often the last non-padding token's hidden state is used as embedding.
        # Assuming the embedding dimension is SMOLVLM_EMBEDDING_DIM
        text_embedding = outputs.last_hidden_state[:, -1, :].squeeze().cpu().numpy().tolist()
        
    return text_embedding


def push_to_weaviate(image_id: str, embedding: list, caption: str, image_url: str):
    """Pushes image embedding, caption, and URL to Weaviate."""
    class_name = "ImageObject"
    # Define schema class if not already created
    class_obj = {
        "class": class_name,
        "vectorizer": "none",  # since we provide our own embeddings
        "properties": [
            {"name": "caption", "dataType": ["text"]},
            {"name": "imageUrl", "dataType": ["string"]},
        ],
        "vectorIndexConfig": {
            "skip": False, # Ensure indexing is enabled
            "vectorCacheMaxObjects": 1000000, # Example, adjust based on memory
            "efConstruction": 128, # Example, tune for search quality
            "maxConnections": 64, # Example, tune for search quality
            "ef": -1, # Example, tune for search quality, -1 for default
            "distance": "cosine", # Use cosine distance for similarity search
        }
    }
    
    try:
        if not client.schema.contains({"class": class_name}):
            client.schema.create_class(class_obj)
            print(f"Created Weaviate schema for class: {class_name}")
        else:
            print(f"Schema for class: {class_name} already exists.")
            
        # Create object with embedding and metadata
        client.data_object.create(
            data_object={
                "caption": caption,
                "imageUrl": image_url,
            },
            class_name=class_name,
            vector=embedding,
            uuid=image_id,
        )
        print(f"Image data for ID {image_id} pushed to Weaviate successfully.")
    except Exception as e:
        print(f"Error pushing data to Weaviate: {e}")

# Example usage
if __name__ == "__main__":
    # --- Part 1: Indexing an image with caption and embedding ---
    image_path = "path/to/your/image.jpg" # <--- IMPORTANT: Change this to your image file
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}. Please update 'image_path'.")
        exit()

    try:
        image = Image.open(image_path).convert("RGB")
        image_id = "unique-image-id-123" # Make this truly unique for each image
        image_url = "https://your.storage/image.jpg"  # Optional: URL where image is stored

        # Generate caption
        caption = generate_caption(image)
        print(f"Generated caption: {caption}")

        # Extract embedding
        embedding = extract_embedding(image)
        print(f"Extracted embedding with dimension: {len(embedding)}")
        # IMPORTANT: Assert that len(embedding) == SMOLVLM_EMBEDDING_DIM
        if len(embedding) != SMOLVLM_EMBEDDING_DIM:
             print(f"Warning: Expected embedding dimension {SMOLVLM_EMBEDDING_DIM}, but got {len(embedding)}. Adjust SMOLVLM_EMBEDDING_DIM.")

        # Push to Weaviate
        push_to_weaviate(image_id, embedding, caption, image_url)

    except Exception as e:
        print(f"Error during image processing or indexing: {e}")

    print("\n--- Part 2: Querying Weaviate ---")

    # --- Text Query Example ---
    query_text = "a dog playing in the park"
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
    query_image_path = "path/to/your/query_image.jpg" # <--- IMPORTANT: Change this to your query image file
    if not os.path.exists(query_image_path):
        print(f"Error: Query image file not found at {query_image_path}. Please update 'query_image_path'.")
    else:
        try:
            query_image = Image.open(query_image_path).convert("RGB")
            image_query_embedding = extract_embedding(query_image)
            print(f"Extracted embedding for query image with dimension: {len(image_query_embedding)}")

            results_image_query = (
                client.query.get("ImageObject", ["caption", "imageUrl"])
                .with_near_vector({"vector": image_query_embedding})
                .with_limit(3) # Get top 3 similar images
                .do()
            )
            print(f"\nResults for image query (from {query_image_path}):")
            if "data" in results_image_query and "Get" in results_image_query["data"] and "ImageObject" in results_image_query["data"]["Get"]:
                for i, item in enumerate(results_image_query["data"]["Get"]["ImageObject"]):
                    print(f"  Result {i+1}: Caption: '{item['caption']}', Image URL: '{item['imageUrl']}'")
            else:
                print("No results found for image query.")
        except NotImplementedError as e:
            print(f"Skipping image query: {e}")
        except Exception as e:
            print(f"Error during image query: {e}")

