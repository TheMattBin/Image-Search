# test_vision_models.py
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestVisionModelsInit:
    """Tests for VisionModels.__init__ method."""

    @patch("vision_models.torch")
    @patch("vision_models.AutoProcessor")
    @patch("vision_models.AutoModelForImageTextToText")
    @patch("vision_models.CLIPProcessor")
    @patch("vision_models.CLIPModel")
    def test_init_with_cuda_available(
        self, mock_clip_model, mock_clip_processor, mock_model, mock_processor, mock_torch
    ):
        """Test initialization when CUDA is available."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.bfloat16 = MagicMock()

        # Mock model loading
        mock_processor.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock().to.return_value = MagicMock()
        mock_clip_processor.from_pretrained.return_value = MagicMock()
        mock_clip_model.from_pretrained.return_value = MagicMock().to.return_value = MagicMock()

        from vision_models import VisionModels

        vm = VisionModels()

        assert vm.device == "cuda"
        assert vm.caption_model_id == "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
        assert vm.clip_model_id == "openai/clip-vit-base-patch16"

    @patch("vision_models.torch")
    @patch("vision_models.AutoProcessor")
    @patch("vision_models.AutoModelForImageTextToText")
    @patch("vision_models.CLIPProcessor")
    @patch("vision_models.CLIPModel")
    def test_init_without_cuda(
        self, mock_clip_model, mock_clip_processor, mock_model, mock_processor, mock_torch
    ):
        """Test initialization when CUDA is not available."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.bfloat16 = MagicMock()

        # Mock model loading
        mock_processor.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock().to.return_value = MagicMock()
        mock_clip_processor.from_pretrained.return_value = MagicMock()
        mock_clip_model.from_pretrained.return_value = MagicMock().to.return_value = MagicMock()

        from vision_models import VisionModels

        vm = VisionModels()

        assert vm.device == "cpu"


class TestGenerateCaption:
    """Tests for VisionModels.generate_caption method."""

    @patch("vision_models.torch")
    def test_generate_caption_with_image_path(self, mock_torch, sample_image, sample_caption):
        """Test caption generation with image file path."""
        mock_torch.no_grad.return_value = MagicMock()
        mock_torch.bfloat16 = MagicMock()

        with patch("vision_models.AutoProcessor") as mock_processor_cls, \
             patch("vision_models.AutoModelForImageTextToText") as mock_model_cls, \
             patch("vision_models.CLIPProcessor") as mock_clip_processor_cls, \
             patch("vision_models.CLIPModel") as mock_clip_model_cls:

            # Mock processor
            mock_processor = MagicMock()
            mock_processor.apply_chat_template.return_value = {
                "input_ids": MagicMock(),
                "attention_mask": MagicMock(),
            }
            mock_processor.batch_decode.return_value = [f"Assistant: {sample_caption}"]
            mock_processor_cls.from_pretrained.return_value = mock_processor

            # Mock model
            mock_model = MagicMock()
            mock_model.device = "cpu"
            mock_model.generate.return_value = MagicMock()
            mock_model_cls.from_pretrained.return_value = mock_model.to.return_value = mock_model

            # Mock CLIP
            mock_clip_processor_cls.from_pretrained.return_value = MagicMock()
            mock_clip_model_cls.from_pretrained.return_value = MagicMock().to.return_value = MagicMock()

            from vision_models import VisionModels

            vm = VisionModels()
            caption = vm.generate_caption(sample_image)

            assert isinstance(caption, str)
            assert len(caption) > 0
            mock_processor.apply_chat_template.assert_called_once()
            mock_model.generate.assert_called_once()

    @patch("vision_models.torch")
    def test_generate_caption_with_pil_image(self, mock_torch):
        """Test caption generation with PIL Image object."""
        mock_torch.no_grad.return_value = MagicMock()
        mock_torch.bfloat16 = MagicMock()

        test_image = Image.new("RGB", (100, 100), color="blue")

        with patch("vision_models.AutoProcessor") as mock_processor_cls, \
             patch("vision_models.AutoModelForImageTextToText") as mock_model_cls, \
             patch("vision_models.CLIPProcessor") as mock_clip_processor_cls, \
             patch("vision_models.CLIPModel") as mock_clip_model_cls:

            # Mock processor
            mock_processor = MagicMock()
            mock_processor.apply_chat_template.return_value = {
                "input_ids": MagicMock(),
                "attention_mask": MagicMock(),
            }
            mock_processor.batch_decode.return_value = ["A blue colored image"]
            mock_processor_cls.from_pretrained.return_value = mock_processor

            # Mock model
            mock_model = MagicMock()
            mock_model.device = "cpu"
            mock_model.generate.return_value = MagicMock()
            mock_model_cls.from_pretrained.return_value = mock_model.to.return_value = mock_model

            # Mock CLIP
            mock_clip_processor_cls.from_pretrained.return_value = MagicMock()
            mock_clip_model_cls.from_pretrained.return_value = MagicMock().to.return_value = MagicMock()

            from vision_models import VisionModels

            vm = VisionModels()
            caption = vm.generate_caption(test_image)

            assert isinstance(caption, str)

    @patch("vision_models.torch")
    def test_generate_caption_strips_assistant_prefix(self, mock_torch, sample_image):
        """Test that 'Assistant:' prefix is stripped from caption."""
        mock_torch.no_grad.return_value = MagicMock()
        mock_torch.bfloat16 = MagicMock()

        with patch("vision_models.AutoProcessor") as mock_processor_cls, \
             patch("vision_models.AutoModelForImageTextToText") as mock_model_cls, \
             patch("vision_models.CLIPProcessor") as mock_clip_processor_cls, \
             patch("vision_models.CLIPModel") as mock_clip_model_cls:

            # Mock processor
            mock_processor = MagicMock()
            mock_processor.apply_chat_template.return_value = {
                "input_ids": MagicMock(),
                "attention_mask": MagicMock(),
            }
            mock_processor.batch_decode.return_value = ["Assistant: A beautiful landscape"]
            mock_processor_cls.from_pretrained.return_value = mock_processor

            # Mock model
            mock_model = MagicMock()
            mock_model.device = "cpu"
            mock_model.generate.return_value = MagicMock()
            mock_model_cls.from_pretrained.return_value = mock_model.to.return_value = mock_model

            # Mock CLIP
            mock_clip_processor_cls.from_pretrained.return_value = MagicMock()
            mock_clip_model_cls.from_pretrained.return_value = MagicMock().to.return_value = MagicMock()

            from vision_models import VisionModels

            vm = VisionModels()
            caption = vm.generate_caption(sample_image)

            assert "Assistant:" not in caption
            assert caption.strip() == "A beautiful landscape"


class TestExtractEmbedding:
    """Tests for VisionModels.extract_embedding method."""

    @patch("vision_models.torch")
    def test_extract_embedding_with_image_path(self, mock_torch, sample_image, sample_embedding):
        """Test embedding extraction with image file path."""
        mock_torch.no_grad.return_value = MagicMock()
        mock_torch.bfloat16 = MagicMock()

        with patch("vision_models.AutoProcessor") as mock_processor_cls, \
             patch("vision_models.AutoModelForImageTextToText") as mock_model_cls, \
             patch("vision_models.CLIPProcessor") as mock_clip_processor_cls, \
             patch("vision_models.CLIPModel") as mock_clip_model_cls:

            # Mock caption model
            mock_processor_cls.from_pretrained.return_value = MagicMock()
            mock_model_cls.from_pretrained.return_value = MagicMock().to.return_value = MagicMock()

            # Mock CLIP processor
            mock_clip_processor = MagicMock()
            mock_clip_inputs = MagicMock()
            mock_clip_processor.return_value = mock_clip_inputs
            mock_clip_processor_cls.from_pretrained.return_value = mock_clip_processor

            # Mock CLIP model
            mock_clip_model = MagicMock()
            mock_clip_model.device = "cpu"
            mock_image_features = MagicMock()
            mock_image_features.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = sample_embedding
            mock_clip_model.get_image_features.return_value = mock_image_features
            mock_clip_model_cls.from_pretrained.return_value = mock_clip_model.to.return_value = mock_clip_model

            from vision_models import VisionModels

            vm = VisionModels()
            embedding = vm.extract_embedding(sample_image)

            assert isinstance(embedding, list)
            assert all(isinstance(x, float) for x in embedding)
            mock_clip_processor.assert_called_once()

    @patch("vision_models.torch")
    def test_extract_embedding_with_pil_image(self, mock_torch):
        """Test embedding extraction with PIL Image object."""
        mock_torch.no_grad.return_value = MagicMock()
        mock_torch.bfloat16 = MagicMock()

        test_image = Image.new("RGB", (100, 100), color="green")

        with patch("vision_models.AutoProcessor") as mock_processor_cls, \
             patch("vision_models.AutoModelForImageTextToText") as mock_model_cls, \
             patch("vision_models.CLIPProcessor") as mock_clip_processor_cls, \
             patch("vision_models.CLIPModel") as mock_clip_model_cls:

            # Mock caption model
            mock_processor_cls.from_pretrained.return_value = MagicMock()
            mock_model_cls.from_pretrained.return_value = MagicMock().to.return_value = MagicMock()

            # Mock CLIP
            mock_clip_processor = MagicMock()
            mock_clip_processor_cls.from_pretrained.return_value = mock_clip_processor

            mock_clip_model = MagicMock()
            mock_clip_model.device = "cpu"
            mock_image_features = MagicMock()
            mock_image_features.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = [0.1, 0.2, 0.3]
            mock_clip_model.get_image_features.return_value = mock_image_features
            mock_clip_model_cls.from_pretrained.return_value = mock_clip_model.to.return_value = mock_clip_model

            from vision_models import VisionModels

            vm = VisionModels()
            embedding = vm.extract_embedding(test_image)

            assert isinstance(embedding, list)
            assert len(embedding) > 0

    @patch("vision_models.torch")
    def test_extract_embedding_converts_image_to_rgb(self, mock_torch):
        """Test that image is converted to RGB mode."""
        mock_torch.no_grad.return_value = MagicMock()
        mock_torch.bfloat16 = MagicMock()

        # Create a non-RGB image (RGBA)
        test_image = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))

        with patch("vision_models.AutoProcessor") as mock_processor_cls, \
             patch("vision_models.AutoModelForImageTextToText") as mock_model_cls, \
             patch("vision_models.CLIPProcessor") as mock_clip_processor_cls, \
             patch("vision_models.CLIPModel") as mock_clip_model_cls:

            # Mock caption model
            mock_processor_cls.from_pretrained.return_value = MagicMock()
            mock_model_cls.from_pretrained.return_value = MagicMock().to.return_value = MagicMock()

            # Mock CLIP
            mock_clip_processor = MagicMock()
            mock_clip_processor_cls.from_pretrained.return_value = mock_clip_processor

            mock_clip_model = MagicMock()
            mock_clip_model.device = "cpu"
            mock_image_features = MagicMock()
            mock_image_features.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = [0.1, 0.2, 0.3]
            mock_clip_model.get_image_features.return_value = mock_image_features
            mock_clip_model_cls.from_pretrained.return_value = mock_clip_model.to.return_value = mock_clip_model

            from vision_models import VisionModels

            vm = VisionModels()
            # Should not raise an error with RGBA image
            embedding = vm.extract_embedding(test_image)

            assert isinstance(embedding, list)
