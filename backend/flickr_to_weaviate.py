import os
from pathlib import Path
from flickr_scraper import flickr_scrape
from vision_models import VisionModels
from weaviate_db import WeaviateImageDB

# --- CONFIG ---
SEARCH_TERMS = ["honeybees on flowers"]  # Change as needed
N_IMAGES = 5  # Number of images per search term
DOWNLOAD = True  # Download images
IMAGES_DIR = Path.cwd() / "images"

# --- SCRAPE FLICKR ---
flickr_scrape(search_terms=SEARCH_TERMS, n=N_IMAGES, download=DOWNLOAD)

# --- INIT MODELS & DB ---
vision = VisionModels()
db = WeaviateImageDB()

for term in SEARCH_TERMS:
    term_dir = IMAGES_DIR / term.replace(" ", "_")
    if not term_dir.exists():
        print(f"No images found for {term} in {term_dir}")
        continue
    for img_file in term_dir.iterdir():
        if not img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue
        try:
            # Caption and embedding
            caption = vision.generate_caption(str(img_file))
            embedding = vision.extract_embedding(str(img_file))
            # Insert to Weaviate
            image_id = img_file.stem
            db.insert_image(image_id=image_id, embedding=embedding, caption=caption, image_url=str(img_file))
            print(f"Inserted {img_file} with caption: {caption}")
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

db.close()
print("Done.")
