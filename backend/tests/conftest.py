# conftest.py
import os
import pytest
from unittest.mock import MagicMock, Mock
from pathlib import Path
from typing import List

# Set test environment variables before importing app modules
os.environ["WEAVIATE_URL"] = "test-cluster.weaviate.cloud"
os.environ["WEAVIATE_API_KEY"] = "test-api-key"


@pytest.fixture
def mock_vision_models(monkeypatch):
    """Mock VisionModels class to avoid loading heavy AI models."""
    # Mock torch
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.bfloat16 = MagicMock()
    monkeypatch.setattr("torch.cuda.is_available", mock_torch.cuda.is_available)
    monkeypatch.setattr("torch.no_grad", MagicMock())

    # Mock transformers
    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_clip_processor = MagicMock()
    mock_clip_model = MagicMock()

    # Mock processor methods
    mock_processor.apply_chat_template.return_value = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock(),
    }
    mock_processor.batch_decode.return_value = ["A beautiful sunset over the ocean"]

    # Mock model methods
    mock_model.generate.return_value = MagicMock()
    mock_model.device = "cpu"

    # Mock CLIP processor
    mock_clip_processor.return_value = MagicMock(
        images=MagicMock(return_tensors="pt")
    )

    # Mock CLIP model
    mock_image_features = MagicMock()
    mock_image_features.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = [
        0.1, 0.2, 0.3, 0.4
    ]
    mock_clip_model.get_image_features.return_value = mock_image_features
    mock_clip_model.device = "cpu"

    # Patch the imports
    monkeypatch.setattr(
        "vision_models.AutoProcessor.from_pretrained", MagicMock(return_value=mock_processor)
    )
    monkeypatch.setattr(
        "vision_models.AutoModelForImageTextToText.from_pretrained",
        MagicMock(return_value=mock_model),
    )
    monkeypatch.setattr(
        "vision_models.CLIPProcessor.from_pretrained",
        MagicMock(return_value=mock_clip_processor),
    )
    monkeypatch.setattr(
        "vision_models.CLIPModel.from_pretrained",
        MagicMock(return_value=mock_clip_model),
    )

    return {
        "processor": mock_processor,
        "model": mock_model,
        "clip_processor": mock_clip_processor,
        "clip_model": mock_clip_model,
    }


@pytest.fixture
def mock_weaviate_client(monkeypatch):
    """Mock Weaviate client to avoid database connections."""
    # Mock the client
    mock_client = MagicMock()
    mock_client.is_ready.return_value = True

    # Mock collections
    mock_collection = MagicMock()
    mock_client.collections.get.return_value = mock_collection

    # Mock query results
    mock_result = MagicMock()
    mock_result.properties = {
        "imageId": "test-image.jpg",
        "description": "A test image",
        "imageUrl": "http://example.com/image.jpg",
    }
    mock_results = MagicMock()
    mock_results.objects = [mock_result]
    mock_collection.query.near_vector.return_value = mock_results
    mock_collection.query.bm25.return_value = mock_results

    # Mock insert
    mock_collection.data.insert.return_value = "test-uuid"

    # Patch the weaviate module
    monkeypatch.setattr("weaviate.connect_to_weaviate_cloud", MagicMock(return_value=mock_client))

    return mock_client


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("WEAVIATE_URL", "test-cluster.weaviate.cloud")
    monkeypatch.setenv("WEAVIATE_API_KEY", "test-api-key")
    return {
        "WEAVIATE_URL": "test-cluster.weaviate.cloud",
        "WEAVIATE_API_KEY": "test-api-key",
    }


@pytest.fixture
def sample_image(tmp_path):
    """Create a sample test image file."""
    import io
    from PIL import Image

    # Create a simple test image
    img = Image.new("RGB", (100, 100), color="red")
    img_path = tmp_path / "test_image.jpg"
    img.save(img_path)

    return str(img_path)


@pytest.fixture
def sample_embedding():
    """Return a sample embedding vector."""
    return [0.1, 0.2, 0.3, 0.4, 0.5]


@pytest.fixture
def sample_caption():
    """Return a sample caption."""
    return "A beautiful sunset over the ocean"
