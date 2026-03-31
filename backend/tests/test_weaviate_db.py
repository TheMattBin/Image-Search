# test_weaviate_db.py
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestWeaviateImageDBInit:
    """Tests for WeaviateImageDB.__init__ method."""

    @patch("weaviate_db.weaviate.connect_to_weaviate_cloud")
    @patch("weaviate_db.load_dotenv")
    def test_init_success(self, mock_load_dotenv, mock_connect, mock_env_vars):
        """Test successful initialization."""
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        mock_connect.return_value = mock_client

        from weaviate_db import WeaviateImageDB

        db = WeaviateImageDB()

        assert db.weaviate_url == "test-cluster.weaviate.cloud"
        assert db.weaviate_api_key == "test-api-key"
        assert db.client is not None
        mock_connect.assert_called_once()

    @patch("weaviate_db.weaviate.connect_to_weaviate_cloud")
    @patch("weaviate_db.load_dotenv")
    def test_init_failure_when_not_ready(self, mock_load_dotenv, mock_connect, mock_env_vars):
        """Test initialization fails when client is not ready."""
        mock_client = MagicMock()
        mock_client.is_ready.return_value = False
        mock_connect.return_value = mock_client

        from weaviate_db import WeaviateImageDB

        with pytest.raises(RuntimeError, match="Failed to connect to Weaviate"):
            WeaviateImageDB()

    @patch("weaviate_db.weaviate.connect_to_weaviate_cloud")
    @patch("weaviate_db.load_dotenv")
    def test_init_uses_env_vars(self, mock_load_dotenv, mock_connect, monkeypatch):
        """Test that initialization reads from environment variables."""
        monkeypatch.setenv("WEAVIATE_URL", "custom-cluster.weaviate.cloud")
        monkeypatch.setenv("WEAVIATE_API_KEY", "custom-api-key")

        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        mock_connect.return_value = mock_client

        from weaviate_db import WeaviateImageDB

        db = WeaviateImageDB()

        assert db.weaviate_url == "custom-cluster.weaviate.cloud"
        assert db.weaviate_api_key == "custom-api-key"


class TestInsertImage:
    """Tests for WeaviateImageDB.insert_image method."""

    @patch("weaviate_db.weaviate.connect_to_weaviate_cloud")
    @patch("weaviate_db.load_dotenv")
    def test_insert_image_success(self, mock_load_dotenv, mock_connect, mock_env_vars, sample_embedding, sample_caption):
        """Test successful image insertion."""
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        mock_connect.return_value = mock_client

        # Mock collection
        mock_collection = MagicMock()
        mock_collection.data.insert.return_value = "test-uuid-123"
        mock_client.collections.get.return_value = mock_collection

        from weaviate_db import WeaviateImageDB

        db = WeaviateImageDB()
        result = db.insert_image("test-image.jpg", sample_embedding, sample_caption, "http://example.com/image.jpg")

        assert result == "test-uuid-123"
        mock_collection.data.insert.assert_called_once()

        # Verify the call arguments
        call_args = mock_collection.data.insert.call_args
        assert call_args[1]["properties"]["imageId"] == "test-image.jpg"
        assert call_args[1]["properties"]["description"] == sample_caption
        assert call_args[1]["vector"] == sample_embedding

    @patch("weaviate_db.weaviate.connect_to_weaviate_cloud")
    @patch("weaviate_db.load_dotenv")
    def test_insert_image_with_default_url(self, mock_load_dotenv, mock_connect, mock_env_vars, sample_embedding):
        """Test image insertion with default empty URL."""
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        mock_connect.return_value = mock_client

        mock_collection = MagicMock()
        mock_collection.data.insert.return_value = "test-uuid"
        mock_client.collections.get.return_value = mock_collection

        from weaviate_db import WeaviateImageDB

        db = WeaviateImageDB()
        db.insert_image("test.jpg", sample_embedding, "Test caption")

        # Verify imageUrl defaults to empty string
        call_args = mock_collection.data.insert.call_args
        assert call_args[1]["properties"]["imageUrl"] == ""

    @patch("weaviate_db.weaviate.connect_to_weaviate_cloud")
    @patch("weaviate_db.load_dotenv")
    def test_insert_image_uses_correct_collection(self, mock_load_dotenv, mock_connect, mock_env_vars, sample_embedding):
        """Test that insert uses the 'ImageObject' collection."""
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        mock_connect.return_value = mock_client

        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection

        from weaviate_db import WeaviateImageDB

        db = WeaviateImageDB()
        db.insert_image("test.jpg", sample_embedding, "caption")

        mock_client.collections.get.assert_called_with("ImageObject")


class TestQueryByVector:
    """Tests for WeaviateImageDB.query_by_vector method."""

    @patch("weaviate_db.weaviate.connect_to_weaviate_cloud")
    @patch("weaviate_db.load_dotenv")
    def test_query_by_vector_success(self, mock_load_dotenv, mock_connect, mock_env_vars, sample_embedding):
        """Test successful vector query."""
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        mock_connect.return_value = mock_client

        # Mock query results
        mock_result = MagicMock()
        mock_result.properties = {
            "imageId": "result-image.jpg",
            "description": "A similar image",
            "imageUrl": "http://example.com/result.jpg"
        }
        mock_results = MagicMock()
        mock_results.objects = [mock_result]

        mock_collection = MagicMock()
        mock_collection.query.near_vector.return_value = mock_results
        mock_client.collections.get.return_value = mock_collection

        from weaviate_db import WeaviateImageDB

        db = WeaviateImageDB()
        result = db.query_by_vector(sample_embedding)

        assert result.objects == [mock_result]
        mock_collection.query.near_vector.assert_called_once()

        # Verify call arguments
        call_args = mock_collection.query.near_vector.call_args
        assert call_args[1]["near_vector"] == sample_embedding
        assert call_args[1]["limit"] == 3  # default limit

    @patch("weaviate_db.weaviate.connect_to_weaviate_cloud")
    @patch("weaviate_db.load_dotenv")
    def test_query_by_vector_with_custom_limit(self, mock_load_dotenv, mock_connect, mock_env_vars, sample_embedding):
        """Test vector query with custom limit."""
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        mock_connect.return_value = mock_client

        mock_collection = MagicMock()
        mock_collection.query.near_vector.return_value = MagicMock(objects=[])
        mock_client.collections.get.return_value = mock_collection

        from weaviate_db import WeaviateImageDB

        db = WeaviateImageDB()
        db.query_by_vector(sample_embedding, limit=10)

        call_args = mock_collection.query.near_vector.call_args
        assert call_args[1]["limit"] == 10


class TestQueryByText:
    """Tests for WeaviateImageDB.query_by_text method."""

    @patch("weaviate_db.weaviate.connect_to_weaviate_cloud")
    @patch("weaviate_db.load_dotenv")
    def test_query_by_text_success(self, mock_load_dotenv, mock_connect, mock_env_vars):
        """Test successful text query."""
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        mock_connect.return_value = mock_client

        # Mock query results
        mock_result = MagicMock()
        mock_result.properties = {
            "imageId": "result-image.jpg",
            "description": "A sunset over mountains",
            "imageUrl": "http://example.com/result.jpg"
        }
        mock_results = MagicMock()
        mock_results.objects = [mock_result]

        mock_collection = MagicMock()
        mock_collection.query.bm25.return_value = mock_results
        mock_client.collections.get.return_value = mock_collection

        from weaviate_db import WeaviateImageDB

        db = WeaviateImageDB()
        result = db.query_by_text("sunset mountains")

        assert result.objects == [mock_result]
        mock_collection.query.bm25.assert_called_once()

        # Verify call arguments
        call_args = mock_collection.query.bm25.call_args
        assert call_args[1]["query"] == "sunset mountains"
        assert call_args[1]["limit"] == 3
        assert "description" in call_args[1]["query_properties"]

    @patch("weaviate_db.weaviate.connect_to_weaviate_cloud")
    @patch("weaviate_db.load_dotenv")
    def test_query_by_text_with_custom_limit(self, mock_load_dotenv, mock_connect, mock_env_vars):
        """Test text query with custom limit."""
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        mock_connect.return_value = mock_client

        mock_collection = MagicMock()
        mock_collection.query.bm25.return_value = MagicMock(objects=[])
        mock_client.collections.get.return_value = mock_collection

        from weaviate_db import WeaviateImageDB

        db = WeaviateImageDB()
        db.query_by_text("test query", limit=5)

        call_args = mock_collection.query.bm25.call_args
        assert call_args[1]["limit"] == 5


class TestClose:
    """Tests for WeaviateImageDB.close method."""

    @patch("weaviate_db.weaviate.connect_to_weaviate_cloud")
    @patch("weaviate_db.load_dotenv")
    def test_close(self, mock_load_dotenv, mock_connect, mock_env_vars):
        """Test closing the database connection."""
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        mock_connect.return_value = mock_client

        from weaviate_db import WeaviateImageDB

        db = WeaviateImageDB()
        db.close()

        mock_client.close.assert_called_once()
