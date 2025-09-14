# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Visual Search Engine that allows users to:
- Upload images and index them with AI-generated captions and embeddings
- Search for images using text queries (BM25 search on captions)
- Search for similar images using image similarity (vector search)

## Architecture

The project consists of two main components:

### Backend (FastAPI)
- **main.py**: Main FastAPI application with three endpoints:
  - `POST /index-image/`: Upload and index images
  - `GET /search-by-text/`: Text-based search using BM25
  - `POST /search-by-image/`: Image similarity search using vectors
- **vision_models.py**: Handles AI vision tasks using Hugging Face transformers:
  - Image captioning with SmolVLM2-2.2B-Instruct
  - Image embedding extraction with CLIP-ViT-Base-Patch16
- **weaviate_db.py**: Weaviate vector database integration for storing and querying image embeddings
- **flickr_scraper.py**: Utility for scraping images from Flickr API
- **utils/**: Helper utilities for image processing and downloading

### Frontend (React/TypeScript)
- **App.tsx**: Main React component with three main functions:
  - Image upload and indexing
  - Text-based search
  - Image similarity search
- **api.tsx**: Axios-based API client for backend communication

## Development Commands

### Backend Setup
```bash
# Create and activate conda environment
conda create -n myenv python=3.10
conda activate myenv

# Install Python dependencies
pip install -r requirements.txt

# Run FastAPI development server
uvicorn backend.main:app --reload
```

### Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Start development server
npm start

# Build for production
npm run build

# Run tests
npm test
```

## Key Dependencies

### Backend
- **FastAPI**: Web framework with automatic API documentation
- **Transformers**: Hugging Face library for AI models (SmolVLM2, CLIP)
- **Weaviate**: Vector database for similarity search
- **FlickrAPI**: For scraping images from Flickr
- **Pillow**: Image processing

### Frontend
- **React 19.1.0**: UI framework
- **TypeScript**: Type safety
- **Axios**: HTTP client for API calls
- **React Scripts**: Build and development tooling

## Environment Variables

The backend requires these environment variables (typically in a `.env` file):
- `WEAVIATE_URL`: Your Weaviate cluster URL
- `WEAVIATE_API_KEY`: Weaviate API key
- `FLICKR_KEY`: Flickr API key (for image scraping)
- `FLICKR_SECRET`: Flickr API secret

## File Structure

```
/
├── backend/                 # Python FastAPI backend
│   ├── main.py             # Main FastAPI application
│   ├── vision_models.py    # AI vision models (captioning, embeddings)
│   ├── weaviate_db.py      # Weaviate database integration
│   ├── flickr_scraper.py   # Flickr image scraping utility
│   └── utils/              # Utility functions
├── frontend/               # React TypeScript frontend
│   ├── src/
│   │   ├── App.tsx         # Main React component
│   │   ├── api.tsx         # API client functions
│   │   └── ...             # Other React components
│   └── package.json        # Node.js dependencies
├── requirements.txt        # Python dependencies
└── README.md              # Setup instructions
```

## Important Notes

- The backend creates a `./tmp` directory for temporary file storage during image processing
- Images are processed using two AI models: one for captioning and one for embedding extraction
- The frontend communicates with the backend on `http://localhost:8000`
- CORS is configured to allow all origins during development