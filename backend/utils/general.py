from pathlib import Path

import requests
from PIL import Image
import re

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
    with open(f, "wb") as file:
        file.write(requests.get(uri, timeout=10).content)

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
