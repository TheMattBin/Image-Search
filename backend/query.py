# main.py
from fastapi import FastAPI, UploadFile, File
from weaviate_db import WeaviateImageDB
from vision_models import VisionModels

app = FastAPI()
db = WeaviateImageDB()
vision = VisionModels()

@app.post("/index-image/")
async def index_image(image: UploadFile = File(...)):
    # Save uploaded file to disk or read as bytes
    image_path = f"/tmp/{image.filename}"
    with open(image_path, "wb") as f:
        f.write(await image.read())
    caption = vision.generate_caption(image_path)
    embedding = vision.extract_embedding(image_path)
    db.insert_image(image.filename, embedding, caption)
    return {"caption": caption}

@app.get("/search-by-text/")
def search_by_text(query: str):
    results = db.query_by_text(query)
    return [obj.properties for obj in results.objects]

@app.post("/search-by-image/")
async def search_by_image(image: UploadFile = File(...)):
    image_path = f"/tmp/{image.filename}"
    with open(image_path, "wb") as f:
        f.write(await image.read())
    embedding = vision.extract_embedding(image_path)
    results = db.query_by_vector(embedding)
    return [obj.properties for obj in results.objects]