from flask import Flask, request, jsonify
from transformers import AutoModelForImageClassification, AutoProcessor
from PIL import Image
import os

# Initialize Flask app
app = Flask(__name__)

# Load the selected model (e.g., CogVLM)
model_name = "THUDM/CogVLM"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# Create a directory for uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_metadata(image_path):
    """Extract metadata from the image using the visual model."""
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Assuming outputs contain relevant metadata; adjust based on your model's output
    metadata = outputs.logits.argmax(-1).item()  
    return metadata

@app.route('/upload', methods=['POST'])
def upload_image():
    """Upload an image and return extracted metadata."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    # Extract metadata
    metadata = extract_metadata(file_path)
    
    # Optionally delete the uploaded file after processing
    os.remove(file_path)

    return jsonify({"metadata": metadata}), 200

if __name__ == '__main__':
    app.run(debug=True)