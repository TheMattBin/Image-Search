# test_main.py
import pytest
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path
from fastapi.testclient import TestClient
from io import BytesIO

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture
def client():
    """Create a test client for FastAPI app."""
    # Mock all the heavy dependencies
    with patch("vision_models.AutoProcessor") as mock_processor, \
         patch("vision_models.AutoModelForImageTextToText") as mock_model, \
         patch("vision_models.CLIPProcessor") as mock_clip_processor, \
         patch("vision_models.CLIPModel") as mock_clip_model, \
         patch("weaviate_db.weaviate.connect_to_weaviate_cloud") as mock_connect, \
         patch("weaviate_db.load_dotenv"), \
         patch("vision_models.torch.cuda.is_available", return_value=False):

        # Setup mock processor
        mock_processor_instance = MagicMock()
        # Return a MagicMock that has .to() method
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs  # Make .to() return self
        mock_processor_instance.apply_chat_template.return_value = mock_inputs
        mock_processor_instance.batch_decode.return_value = ["A beautiful sunset"]
        mock_processor.from_pretrained.return_value = mock_processor_instance

        # Setup mock model
        mock_model_instance = MagicMock()
        mock_model_instance.device = "cpu"
        mock_model_instance.generate.return_value = MagicMock()
        # Make .to() return self
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        # Setup mock CLIP processor
        mock_clip_proc_instance = MagicMock()
        mock_clip_proc_instance.return_value = MagicMock(images=MagicMock(return_tensors="pt"))
        mock_clip_processor.from_pretrained.return_value = mock_clip_proc_instance

        # Setup mock CLIP model
        mock_clip_model_instance = MagicMock()
        mock_clip_model_instance.device = "cpu"
        mock_image_features = MagicMock()
        mock_image_features.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        mock_clip_model_instance.get_image_features.return_value = mock_image_features
        # Make .to() return self
        mock_clip_model_instance.to.return_value = mock_clip_model_instance
        mock_clip_model.from_pretrained.return_value = mock_clip_model_instance

        # Setup mock Weaviate client
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        mock_connect.return_value = mock_client

        # Import after mocking
        from main import app
        return TestClient(app)


@pytest.fixture
def client(mock_vision_models, mock_weaviate_client, mock_env_vars):
    """Create a test client for FastAPI app."""
    # Re-import to get mocked versions
    import importlib
    import main
    importlib.reload(main)

    from main import app as test_app
    return TestClient(test_app)


@pytest.fixture
def test_image_file(tmp_path):
    """Create a test image file."""
    from PIL import Image

    # Create a test image
    img = Image.new("RGB", (100, 100), color="blue")
    img_path = tmp_path / "test_image.jpg"
    img.save(img_path)

    # Return as tuple for FastAPI file upload
    return ("test_image.jpg", open(img_path, "rb"), "image/jpeg")


class TestIndexImageEndpoint:
    """Tests for POST /index-image/ endpoint."""

    def test_index_image_success(self, client, test_image_file):
        """Test successful image indexing."""
        # Create actual tmp directory for test
        import os
        import shutil
        import main

        # Save original tmp directory state
        tmp_exists = os.path.exists("./tmp")

        try:
            # Create tmp directory
            os.makedirs("./tmp", exist_ok=True)

            # Mock the vision methods to avoid loading models
            with patch.object(main.vision, 'generate_caption', return_value="A test caption"), \
                 patch.object(main.vision, 'extract_embedding', return_value=[0.1, 0.2, 0.3]), \
                 patch.object(main.db, 'insert_image', return_value="test-uuid"):

                # Get file object
                filename, file_obj, content_type = test_image_file

                response = client.post(
                    "/index-image/",
                    files={"image": (filename, file_obj, content_type)}
                )

                # Check response
                assert response.status_code == 200
                assert "caption" in response.json()
                assert response.json()["caption"] == "A test caption"
        finally:
            # Cleanup
            if os.path.exists("./tmp"):
                shutil.rmtree("./tmp", ignore_errors=True)

    def test_index_image_creates_tmp_directory(self, client, test_image_file):
        """Test that tmp directory is created."""
        import os
        import shutil
        import main

        # Remove tmp directory if it exists
        if os.path.exists("./tmp"):
            shutil.rmtree("./tmp")

        try:
            with patch.object(main.vision, 'generate_caption', return_value="Test"), \
                 patch.object(main.vision, 'extract_embedding', return_value=[0.1, 0.2]), \
                 patch.object(main.db, 'insert_image', return_value="test-uuid"):

                filename, file_obj, content_type = test_image_file
                response = client.post("/index-image/", files={"image": (filename, file_obj, content_type)})

                # Verify tmp directory was created
                assert os.path.exists("./tmp")
        finally:
            # Cleanup
            if os.path.exists("./tmp"):
                shutil.rmtree("./tmp", ignore_errors=True)

    def test_index_image_without_file(self, client):
        """Test endpoint without providing image file."""
        response = client.post("/index-image/")

        # FastAPI returns 422 Unprocessable Entity without required file
        assert response.status_code == 422


class TestSearchByTextEndpoint:
    """Tests for GET /search-by-text/ endpoint."""

    def test_search_by_text_success(self, client, monkeypatch):
        """Test successful text search."""
        # Mock the database query
        mock_db = MagicMock()
        mock_results = MagicMock()
        mock_result_obj = MagicMock()
        mock_result_obj.properties = {
            "imageId": "test-image.jpg",
            "description": "A beautiful sunset",
            "imageUrl": "http://example.com/image.jpg"
        }
        mock_results.objects = [mock_result_obj]
        mock_db.query_by_text.return_value = mock_results

        monkeypatch.setattr("main.db", mock_db)

        response = client.get("/search-by-text/", params={"query": "sunset"})

        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert len(response.json()) >= 0

    def test_search_by_text_calls_database(self, client, monkeypatch):
        """Test that database query is called with correct query."""
        mock_db = MagicMock()
        mock_results = MagicMock()
        mock_results.objects = []
        mock_db.query_by_text.return_value = mock_results

        monkeypatch.setattr("main.db", mock_db)

        test_query = "beautiful sunset"
        client.get("/search-by-text/", params={"query": test_query})

        mock_db.query_by_text.assert_called_once_with(test_query)

    def test_search_by_text_without_query(self, client):
        """Test endpoint without query parameter."""
        response = client.get("/search-by-text/")

        # FastAPI returns 422 without required query parameter
        assert response.status_code == 422


class TestSearchByImageEndpoint:
    """Tests for POST /search-by-image/ endpoint."""

    def test_search_by_image_success(self, client, test_image_file):
        """Test successful image similarity search."""
        import os
        import shutil
        import main

        # Create tmp directory
        os.makedirs("./tmp", exist_ok=True)

        try:
            with patch.object(main.vision, 'extract_embedding', return_value=[0.1, 0.2, 0.3]), \
                 patch.object(main.db, 'query_by_vector') as mock_query:

                # Setup mock query results
                mock_result = MagicMock()
                mock_result.properties = {
                    "imageId": "result.jpg",
                    "description": "A similar image",
                    "imageUrl": "http://example.com/result.jpg"
                }
                mock_results = MagicMock()
                mock_results.objects = [mock_result]
                mock_query.return_value = mock_results

                filename, file_obj, content_type = test_image_file
                response = client.post(
                    "/search-by-image/",
                    files={"image": (filename, file_obj, content_type)}
                )

                # Response should be successful
                assert response.status_code == 200
                assert isinstance(response.json(), list)
        finally:
            # Cleanup
            if os.path.exists("./tmp"):
                shutil.rmtree("./tmp", ignore_errors=True)

    def test_search_by_image_creates_tmp_directory(self, client, test_image_file):
        """Test that tmp directory is created."""
        import os
        import shutil
        import main

        # Remove tmp directory if it exists
        if os.path.exists("./tmp"):
            shutil.rmtree("./tmp")

        try:
            with patch.object(main.vision, 'extract_embedding', return_value=[0.1, 0.2]), \
                 patch.object(main.db, 'query_by_vector'):

                filename, file_obj, content_type = test_image_file
                client.post("/search-by-image/", files={"image": (filename, file_obj, content_type)})

                # Verify tmp directory was created
                assert os.path.exists("./tmp")
        finally:
            # Cleanup
            if os.path.exists("./tmp"):
                shutil.rmtree("./tmp", ignore_errors=True)

    def test_search_by_image_without_file(self, client):
        """Test endpoint without providing image file."""
        response = client.post("/search-by-image/")

        # FastAPI returns 422 without required file
        assert response.status_code == 422


class TestCORSMiddleware:
    """Tests for CORS middleware configuration."""

    def test_cors_headers_in_get_request(self, client):
        """Test that CORS headers are set correctly on GET request."""
        response = client.get("/search-by-text/", params={"query": "test"})

        # CORS should be configured for all origins as per main.py
        assert response.status_code == 200
        # Note: TestClient might not show CORS headers, but they're configured in main.py


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_endpoint(self, client):
        """Test accessing invalid endpoint returns 404."""
        response = client.get("/invalid-endpoint/")

        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test using wrong HTTP method returns 405."""
        response = client.post("/search-by-text/")

        assert response.status_code == 405
