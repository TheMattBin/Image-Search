import os
import time
from pathlib import Path
from dotenv import load_dotenv

from flickrapi import FlickrAPI

from utils.general import download_uri

# Load environment variables from .env file
load_dotenv()
key = os.getenv("FLICKR_KEY")  # Flickr API key from .env
secret = os.getenv("FLICKR_SECRET")


def get_urls(search="honeybees on flowers", n=10, download=False):
    """Fetch Flickr URLs for `search` term images, optionally downloading them; supports up to `n` images."""
    t = time.time()
    flickr = FlickrAPI(key, secret)
    license = ()  # https://www.flickr.com/services/api/explore/?method=flickr.photos.licenses.getInfo
    photos = flickr.walk(
        text=search,  # http://www.flickr.com/services/api/flickr.photos.search.html
        extras="url_o",
        per_page=500,  # 1-500
        license=license,
        sort="relevance",
    )

    if download:
        dir_path = Path.cwd() / "images" / search.replace(" ", "_")
        dir_path.mkdir(parents=True, exist_ok=True)

    urls = []
    for i, photo in enumerate(photos):
        if i <= n:
            try:
                url = photo.get("url_o")  # original size
                if url is None:
                    url = f"https://farm{photo.get('farm')}.staticflickr.com/{photo.get('server')}/{photo.get('id')}_{photo.get('secret')}_b.jpg"

                if download:
                    download_uri(url, dir_path)

                urls.append(url)
                # Avoid logging the full URL as it may contain the photo 'secret'.
                print(f"{i}/{n} photo_id={photo.get('id')}")
            except Exception:
                print(f"{i}/{n} error...")

        else:
            print(f"Done. ({time.time() - t:.1f}s)" + (f"\nAll images saved to {dir_path}" if download else ""))
            break


def flickr_scrape(search_terms=["honeybees on flowers"], n=10, download=False):
    """Wrapper function to fetch URLs for a list of search terms."""
    for search in search_terms:
        get_urls(search=search, n=n, download=download)

# Now you can import and call flickr_scrape(search_terms, n, download) from another script.
# Example usage:
# flickr_scrape(["cat", "dog"], n=5, download=True)
