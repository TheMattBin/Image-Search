from pathlib import Path

import requests
from PIL import Image
import re

# --- Add Fernet encryption dependencies ---
from cryptography.fernet import Fernet
import os

# Fetch encryption key from environment variable, or generate/save one as needed
FERNET_KEY = os.environ.get('DOWNLOAD_ENC_KEY')
if FERNET_KEY is None:
    # Generate a new key (for demo purposes; in production, persist it securely!)
    FERNET_KEY = Fernet.generate_key()
    print("Generated new Fernet key. Store this securely for future decryption.")
fernet = Fernet(FERNET_KEY)
def download_uri(uri, dir="./"):
    """Downloads file from URI, performing checks and renaming; supports timeout and image format suffix addition."""
    # Download
    dir = Path(dir)
    # Strip Flickr 'secret' from filename if present (_[secret]_b.jpg)
    fname = Path(uri).name
    # Remove Flickr secret from filename pattern: {id}_{secret}_b.jpg -> {id}_b.jpg
    import re
    match = re.match(r"(?P<id>\d+)_(?P<secret>\w+)_b\.jpg", fname)
    if match:
        fname = f"{match.group('id')}_b.jpg"
    f = dir / fname  # sanitized filename
    # --- Download and encrypt image content before saving ---
    image_bytes = requests.get(uri, timeout=10).content
    encrypted_bytes = fernet.encrypt(image_bytes)
    with open(f, "wb") as file:
        file.write(encrypted_bytes)

    # Rename (remove wildcard characters)
    src = f  # original name
    f = Path(
        str(f)
        .replace("%20", "_")
        .replace("%", "_")
        .replace("*", "_")
        .replace("~", "_")
        .replace("(", "_")
        .replace(")", "_")
    )

    if "?" in str(f):
        f = Path(str(f)[: str(f).index("?")])

    if src != f:
        src.rename(f)  # rename

    # Add suffix (if missing)
    if not f.suffix:
        src = f  # original name
        f = f.with_suffix(f".{Image.open(f).format.lower()}")
        src.rename(f)  # rename
