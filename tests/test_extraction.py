"""
Tests for LEANN + ContextGem extraction integration
"""

import os
import pytest

# Check if contextgem is available
try:
    import contextgem  # noqa: F401

    CONTEXTGEM_AVAILABLE = True
except ImportError:
    CONTEXTGEM_AVAILABLE = False

# Check if OpenAI API key is available for integration tests
OPENAI_API_KEY_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))

pytestmark = pytest.mark.skipif(
    not CONTEXTGEM_AVAILABLE, reason="ContextGem not installed (install with: uv pip install leann-core[extract])"
)


@pytest.fixture
def sample_index(tmp_path):
    """Create a sample LEANN index for testing."""
    from leann import LeannBuilder

    index_path = tmp_path / "test_extraction.leann"

    # Create index with sample documents
    builder = LeannBuilder(backend_name="hnsw", embedding_model="sentence-transformers/all-MiniLM-L6-v2")

    # Add sample documents about technology companies
    builder.add_text(
        "Apple Inc. reported quarterly revenue of $90 billion in Q4 2024, exceeding analyst expectations. "
        "The company's iPhone sales grew 15% year-over-year, driven by strong demand in Asia.",
        metadata={"company": "Apple", "quarter": "Q4 2024", "file_type": "earnings"},
    )

    builder.add_text(
        "Microsoft Azure cloud services grew 31% in the most recent quarter. "
        "CEO Satya Nadella emphasized the company's AI investments, particularly in OpenAI partnership.",
        metadata={"company": "Microsoft", "quarter": "Q3 2024", "file_type": "earnings"},
    )

    builder.add_text(
        "Google announced new AI features in Search and Gmail. The Gemini model will be integrated "
        "across all Google products by end of 2024. Search revenue reached $50B this quarter.",
        metadata={"company": "Google", "quarter": "Q3 2024", "file_type": "news"},
    )

    builder.add_text(
        "NVIDIA's data center revenue hit record highs due to AI chip demand. H100 GPUs remain sold out "
        "through 2025. Stock price increased 200% year-to-date.",
        metadata={"company": "NVIDIA", "quarter": "Q3 2024", "file_type": "news"},
    )

    builder.build_index(str(index_path))

    return str(index_path)


class TestLeannExtractorBasic:
    """Basic functionality tests for LeannExtractor"""

    def test_import_extractor(self):
        """Test that LeannExtractor can be imported."""
        from leann import LeannExtractor

        assert LeannExtractor is not None

    def test_extractor_initialization(self, sample_index):
        """Test basic extractor initialization."""
        from leann import LeannExtractor

        extractor = LeannExtractor(sample_index)
        assert extractor is not None
        assert extractor.searcher is not None
        assert extractor._owns_searcher is True

    def test_extractor_with_existing_searcher(self, sample_index):
        """Test initialization with existing searcher."""
        from leann import LeannExtractor, LeannSearcher

        searcher = LeannSearcher(sample_index)
        extractor = LeannExtractor(sample_index, searcher=searcher)

        assert extractor.searcher is searcher
        assert extractor._owns_searcher is False

    def test_extractor_without_contextgem_raises_error(self, tmp_path, monkeypatch):
        """Test that helpful error is raised if contextgem not installed."""
        # Mock contextgem import to fail
        import sys

        original_import = __builtins__.__import__

        def mock_import(name, *args, **kwargs):
            if name == "contextgem":
                raise ImportError("No module named 'contextgem'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(__builtins__, "__import__", mock_import)

        # Create a simple index
        from leann import LeannBuilder

        index_path = tmp_path / "test.leann"
        builder = LeannBuilder(backend_name="hnsw")
        builder.add_text("test")
        builder.build_index(str(index_path))

        # Try to create extractor - should fail with helpful message
        with pytest.raises(ImportError, match="ContextGem is not installed"):
            # Need to reimport to trigger the check
            import importlib

            import leann.extraction

            importlib.reload(leann.extraction)
            from leann.extraction import LeannExtractor

            LeannExtractor(str(index_path))

    def test_context_manager(self, sample_index):
        """Test that extractor works as context manager."""
        from leann import LeannExtractor

        with LeannExtractor(sample_index) as extractor:
            assert extractor is not None

        # Should have cleaned up
        # (searcher cleanup is called)


@pytest.mark.skipif(not OPENAI_API_KEY_AVAILABLE, reason="Requires OPENAI_API_KEY")
class TestLeannExtractorIntegration:
    """Integration tests with actual LLM calls (requires API key)"""

    def test_basic_extraction(self, sample_index):
        """Test basic search and extract workflow."""
        from contextgem import StringConcept
        from leann import LeannExtractor

        extractor = LeannExtractor(
            sample_index,
            llm_config={"model": "openai/gpt-4o-mini"},
        )

        result = extractor.search_and_extract(
            query="technology company revenue",
            concepts=[
                StringConcept(name="Companies", description="Company names mentioned in the text"),
            ],
            top_k=3,
            add_references=False,
            add_justifications=False,
        )

        # Verify structure
        assert "search_results" in result
        assert "extractions" in result
        assert "total_tokens" in result
        assert "total_cost" in result
        assert "metadata" in result

        # Should have found documents
        assert len(result["search_results"]) > 0

        # Should have extractions
        assert len(result["extractions"]) > 0

        # Each extraction should have Companies concept
        for extraction in result["extractions"]:
            assert "concepts" in extraction
            assert "Companies" in extraction["concepts"]

        # Should have used tokens
        assert result["total_tokens"] > 0

    def test_extraction_with_references(self, sample_index):
        """Test extraction with sentence-level references."""
        from contextgem import StringConcept
        from leann import LeannExtractor

        extractor = LeannExtractor(sample_index, llm_config={"model": "openai/gpt-4o-mini"})

        result = extractor.search_and_extract(
            query="AI developments",
            concepts=[
                StringConcept(name="AIFeatures", description="AI features and developments mentioned"),
            ],
            top_k=2,
            add_references=True,
            add_justifications=False,
        )

        # Check that references are included
        for extraction in result["extractions"]:
            for item in extraction["concepts"]["AIFeatures"]:
                assert "references" in item
                if item["references"]:
                    # References should have text
                    assert "text" in item["references"][0]

    def test_extraction_with_justifications(self, sample_index):
        """Test extraction with LLM justifications."""
        from contextgem import StringConcept
        from leann import LeannExtractor

        extractor = LeannExtractor(sample_index, llm_config={"model": "openai/gpt-4o-mini"})

        result = extractor.search_and_extract(
            query="revenue growth",
            concepts=[
                StringConcept(name="Growth", description="Growth metrics and percentages"),
            ],
            top_k=2,
            add_references=False,
            add_justifications=True,
        )

        # Check that justifications are included
        for extraction in result["extractions"]:
            for item in extraction["concepts"]["Growth"]:
                assert "justification" in item
                # Justification might be None if no items extracted
                if item["justification"]:
                    assert isinstance(item["justification"], str)
                    assert len(item["justification"]) > 0

    def test_json_schema_extraction(self, sample_index):
        """Test JSON schema-based extraction."""
        from contextgem import JsonObjectConcept
        from leann import LeannExtractor

        schema = JsonObjectConcept(
            name="CompanyMetrics",
            description="Company performance metrics",
            json_schema={
                "company": {"type": "string"},
                "revenue": {"type": "string"},
                "growth_metric": {"type": "string"},
            },
        )

        extractor = LeannExtractor(sample_index, llm_config={"model": "openai/gpt-4o-mini"})

        result = extractor.search_and_extract(query="quarterly earnings", concepts=[schema], top_k=2)

        # Verify schema extraction worked
        for extraction in result["extractions"]:
            if extraction["concepts"]["CompanyMetrics"]:
                metrics = extraction["concepts"]["CompanyMetrics"][0]["value"]
                # At least some fields should be populated
                assert "company" in metrics or "revenue" in metrics

    def test_combined_extraction(self, sample_index):
        """Test combining all documents into single extraction."""
        from contextgem import StringConcept
        from leann import LeannExtractor

        extractor = LeannExtractor(sample_index, llm_config={"model": "openai/gpt-4o-mini"})

        result = extractor.search_and_extract(
            query="technology companies",
            concepts=[
                StringConcept(name="AllCompanies", description="All companies mentioned"),
            ],
            top_k=4,
            combine_documents=True,  # Combine all results
        )

        # Should have single extraction (dict, not list)
        assert isinstance(result["extractions"], dict)
        assert "concepts" in result["extractions"]
        assert "AllCompanies" in result["extractions"]["concepts"]

    def test_extract_from_results(self, sample_index):
        """Test extracting from existing search results."""
        from contextgem import StringConcept
        from leann import LeannExtractor, LeannSearcher

        # First, do a search
        searcher = LeannSearcher(sample_index)
        search_results = searcher.search("AI", top_k=2)

        # Then extract from those results
        extractor = LeannExtractor(sample_index, llm_config={"model": "openai/gpt-4o-mini"})

        result = extractor.extract_from_results(
            search_results,
            concepts=[
                StringConcept(name="AITopics", description="AI-related topics"),
            ],
        )

        # Should have same number of extractions as search results
        assert len(result["extractions"]) == len(search_results)

    def test_metadata_filtering(self, sample_index):
        """Test extraction with metadata filtering."""
        from contextgem import StringConcept
        from leann import LeannExtractor

        extractor = LeannExtractor(sample_index, llm_config={"model": "openai/gpt-4o-mini"})

        result = extractor.search_and_extract(
            query="revenue",
            concepts=[
                StringConcept(name="Revenue", description="Revenue figures"),
            ],
            top_k=10,
            metadata_filters={"file_type": {"==": "earnings"}},  # Only earnings reports
        )

        # All results should be earnings reports
        for search_result in result["search_results"]:
            assert search_result.metadata.get("file_type") == "earnings"


@pytest.mark.skipif(not OPENAI_API_KEY_AVAILABLE, reason="Requires OPENAI_API_KEY")
class TestLeannExtractorEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_search_results(self, sample_index):
        """Test behavior when search returns no results."""
        from contextgem import StringConcept
        from leann import LeannExtractor

        extractor = LeannExtractor(sample_index, llm_config={"model": "openai/gpt-4o-mini"})

        result = extractor.search_and_extract(
            query="xyzabc nonexistent topic that wont match anything",
            concepts=[StringConcept(name="Test", description="Test")],
            top_k=5,
        )

        # Should handle gracefully
        assert result["search_results"] == []
        assert result["extractions"] == []
        assert result["total_tokens"] == 0
        assert result["total_cost"] == 0.0

    def test_llm_config_conversion(self, sample_index):
        """Test LEANN-style LLM config conversion."""
        from contextgem import StringConcept
        from leann import LeannExtractor

        # LEANN-style Ollama config
        extractor = LeannExtractor(
            sample_index,
            llm_config={
                "llm": "ollama",
                "model": "qwen2.5:7b",
                "host": "http://localhost:11434",
            },
        )

        # Should convert to LiteLLM format
        extractor._init_contextgem_llm()
        # Model should be prefixed with ollama/
        # (This would need actual Ollama running to fully test)

    def test_cost_tracking(self, sample_index):
        """Test that cost tracking works correctly."""
        from contextgem import StringConcept
        from leann import LeannExtractor

        extractor = LeannExtractor(sample_index, llm_config={"model": "openai/gpt-4o-mini"})

        result = extractor.search_and_extract(
            query="companies", concepts=[StringConcept(name="C", description="Companies")], top_k=2
        )

        # Cost should be tracked
        assert result["total_cost"] >= 0  # Might be 0 if using cached/free tier
        assert result["total_tokens"] > 0


class TestLeannExtractorWithoutAPIKey:
    """Tests that don't require API keys"""

    def test_extractor_initialization_without_llm_config(self, sample_index):
        """Test that extractor can be initialized without LLM config."""
        from leann import LeannExtractor

        extractor = LeannExtractor(sample_index)
        assert extractor.llm_config == {}
        # LLM should be lazily initialized
        assert extractor._contextgem_llm is None

    def test_search_only_without_extraction(self, sample_index):
        """Test that LEANN search still works without doing extraction."""
        from leann import LeannExtractor

        extractor = LeannExtractor(sample_index)

        # Should be able to use searcher directly
        results = extractor.searcher.search("revenue", top_k=3)
        assert len(results) > 0
