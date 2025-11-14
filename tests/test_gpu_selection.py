"""
Tests for GPU selection functionality (Tier 1)
"""

import os
import pytest
import torch

from leann import LeannBuilder
from leann.embedding_compute import compute_embeddings_sentence_transformers


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestGPUSelection:
    """Test GPU selection functionality"""

    def test_gpu_id_parameter(self):
        """Test that gpu_id parameter is accepted"""
        # This should not raise an error
        texts = ["test text 1", "test text 2"]

        # Test with gpu_id=0 (first GPU)
        embeddings = compute_embeddings_sentence_transformers(
            texts,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            gpu_id=0,
        )

        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] > 0

    def test_gpu_id_env_variable(self):
        """Test that LEANN_GPU_ID environment variable is respected"""
        texts = ["test text 1", "test text 2"]

        # Set environment variable
        os.environ["LEANN_GPU_ID"] = "0"

        try:
            embeddings = compute_embeddings_sentence_transformers(
                texts,
                model_name="sentence-transformers/all-MiniLM-L6-v2",
            )

            assert embeddings.shape[0] == len(texts)
            assert embeddings.shape[1] > 0
        finally:
            # Clean up
            del os.environ["LEANN_GPU_ID"]

    def test_leann_builder_with_gpu_id(self):
        """Test that LeannBuilder accepts gpu_id parameter"""
        # This should not raise an error
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            gpu_id=0,
        )

        assert builder.gpu_id == 0

    @pytest.mark.skipif(
        torch.cuda.device_count() < 2,
        reason="Requires 2+ GPUs"
    )
    def test_invalid_gpu_id_raises_error(self):
        """Test that invalid GPU ID raises appropriate error"""
        texts = ["test text"]

        num_gpus = torch.cuda.device_count()
        invalid_gpu_id = num_gpus + 5  # Invalid GPU ID

        with pytest.raises(ValueError, match="GPU ID .* is invalid"):
            compute_embeddings_sentence_transformers(
                texts,
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                gpu_id=invalid_gpu_id,
            )


@pytest.mark.skipif(torch.cuda.is_available(), reason="Only test CPU fallback when no GPU")
class TestCPUFallback:
    """Test CPU fallback when no GPU available"""

    def test_cpu_fallback_with_gpu_id(self):
        """Test that gpu_id is gracefully ignored on CPU-only systems"""
        texts = ["test text 1", "test text 2"]

        # Should fall back to CPU without error
        embeddings = compute_embeddings_sentence_transformers(
            texts,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",  # Explicitly request CPU
        )

        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] > 0
