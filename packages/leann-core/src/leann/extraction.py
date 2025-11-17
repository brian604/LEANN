"""
LEANN + ContextGem Integration Module

This module provides structured data extraction capabilities by combining:
1. LEANN's semantic search to find relevant documents
2. ContextGem's LLM-powered extraction for structured insights

Typical workflow:
    Search (LEANN) → Extract (ContextGem) → Structure (JSON/Schema)
"""

import logging
import os
from typing import Any, List, Optional, Union

logger = logging.getLogger(__name__)


class LeannExtractor:
    """
    Combines LEANN search with ContextGem extraction for document intelligence.

    This class integrates LEANN's efficient semantic search with ContextGem's
    structured data extraction capabilities, enabling a powerful two-stage workflow:

    1. Use LEANN to find relevant documents from large collections (fast, low-storage)
    2. Use ContextGem to extract structured data from found documents (LLM-powered)

    Example:
        >>> from leann import LeannExtractor
        >>> from contextgem import StringConcept
        >>>
        >>> extractor = LeannExtractor("my_index.leann")
        >>> results = extractor.search_and_extract(
        ...     query="quarterly earnings",
        ...     concepts=[
        ...         StringConcept(name="Companies", description="Company names"),
        ...         StringConcept(name="Revenue", description="Revenue figures"),
        ...     ],
        ...     top_k=5
        ... )
        >>> print(results["extractions"])
    """

    def __init__(
        self,
        index_path: str,
        llm_config: Optional[dict[str, Any]] = None,
        searcher: Optional[Any] = None,
        gpu_id: Optional[int] = None,
        **searcher_kwargs,
    ):
        """
        Initialize extractor with LEANN index and LLM configuration.

        Args:
            index_path: Path to LEANN index file (.leann)
            llm_config: LLM configuration for extraction. Format:
                {
                    "model": "openai/gpt-4o-mini",  # LiteLLM format
                    "api_key": "your-key",
                    "api_base": "http://localhost:11434",  # Optional
                    "llm": "openai" or "ollama",  # LEANN format (converted)
                }
            searcher: Optional existing LeannSearcher instance (for reuse)
            gpu_id: GPU ID for LEANN embedding computation
            **searcher_kwargs: Additional arguments for LeannSearcher initialization

        Raises:
            ImportError: If contextgem is not installed
            FileNotFoundError: If index_path doesn't exist
        """
        # Check if ContextGem is available
        try:
            import contextgem  # noqa: F401
        except ImportError:
            raise ImportError(
                "ContextGem is not installed. Install with:\n"
                "  uv pip install leann-core[extract]\n"
                "or:\n"
                "  uv pip install contextgem"
            )

        # Import here to avoid hard dependency
        from leann.api import LeannSearcher

        # Initialize LEANN searcher
        if searcher is None:
            self.searcher = LeannSearcher(index_path, gpu_id=gpu_id, **searcher_kwargs)
            self._owns_searcher = True
        else:
            self.searcher = searcher
            self._owns_searcher = False

        # Store LLM configuration
        self.llm_config = llm_config or {}

        # Initialize ContextGem LLM (lazy initialization)
        self._contextgem_llm = None

    def _init_contextgem_llm(self):
        """
        Initialize ContextGem's DocumentLLM from LEANN llm_config.

        Converts LEANN's LLM configuration format to ContextGem's format.
        Supports OpenAI, Ollama, and other LiteLLM-compatible providers.
        """
        if self._contextgem_llm is not None:
            return  # Already initialized

        from contextgem import DocumentLLM

        # Extract configuration with defaults
        model = self.llm_config.get("model", "openai/gpt-4o-mini")
        api_key = self.llm_config.get("api_key", os.getenv("OPENAI_API_KEY"))
        api_base = self.llm_config.get("api_base")

        # Handle LEANN-style configuration (convert to LiteLLM format)
        llm_provider = self.llm_config.get("llm", "openai")
        if llm_provider == "ollama":
            # Convert to LiteLLM format
            if not model.startswith("ollama/"):
                model = f"ollama/{model}"
            api_base = self.llm_config.get("host", "http://localhost:11434")
            # Ollama doesn't require API key
            api_key = api_key or "ollama"  # Dummy key for compatibility

        logger.info(f"Initializing ContextGem with model: {model}")

        # Initialize ContextGem DocumentLLM
        kwargs = {"model": model}
        if api_key:
            kwargs["api_key"] = api_key
        if api_base:
            kwargs["api_base"] = api_base

        self._contextgem_llm = DocumentLLM(**kwargs)

    def search_and_extract(
        self,
        query: str,
        concepts: List[Any],
        aspects: Optional[List[Any]] = None,
        top_k: int = 5,
        complexity: int = 64,
        add_references: bool = True,
        add_justifications: bool = True,
        combine_documents: bool = False,
        **search_kwargs,
    ) -> dict[str, Any]:
        """
        Search for documents and extract structured data.

        This is the main method that combines LEANN search with ContextGem extraction.

        Args:
            query: Search query string
            concepts: List of ContextGem Concept objects to extract
                (StringConcept, BooleanConcept, JsonObjectConcept, etc.)
            aspects: Optional list of Aspect objects to extract
                (topics, themes, sections, etc.)
            top_k: Number of documents to retrieve via LEANN search
            complexity: LEANN search complexity (higher = more accurate, slower)
            add_references: Add sentence-level references to extracted items
            add_justifications: Add LLM explanations for each extraction
            combine_documents: If True, combine all results into single extraction.
                If False, extract from each document separately (default).
            **search_kwargs: Additional LEANN search parameters
                (beam_width, prune_ratio, metadata_filters, etc.)

        Returns:
            Dictionary containing:
            {
                "search_results": List[SearchResult],  # LEANN search results
                "extractions": List[dict] or dict,  # Extracted data
                "total_tokens": int,  # Total LLM tokens used
                "total_cost": float,  # Total cost in USD
                "metadata": {
                    "query": str,
                    "top_k": int,
                    "num_documents": int,
                    "combined": bool,
                }
            }

        Example:
            >>> from contextgem import StringConcept, JsonObjectConcept
            >>>
            >>> result = extractor.search_and_extract(
            ...     query="machine learning papers 2024",
            ...     concepts=[
            ...         StringConcept(
            ...             name="Methodologies",
            ...             description="Research methodologies used"
            ...         ),
            ...         JsonObjectConcept(
            ...             name="Metrics",
            ...             description="Performance metrics",
            ...             json_schema={
            ...                 "accuracy": {"type": "number"},
            ...                 "f1_score": {"type": "number"}
            ...             }
            ...         ),
            ...     ],
            ...     top_k=10,
            ...     add_references=True,
            ... )
            >>>
            >>> # Access results
            >>> for doc_extraction in result["extractions"]:
            ...     print(doc_extraction["concepts"]["Methodologies"])
        """
        # Ensure ContextGem LLM is initialized
        self._init_contextgem_llm()

        # Step 1: LEANN semantic search
        logger.info(f"Searching for: '{query}' (top_k={top_k})")
        search_results = self.searcher.search(query, top_k=top_k, complexity=complexity, **search_kwargs)

        if not search_results:
            logger.warning("No search results found")
            return {
                "search_results": [],
                "extractions": [] if not combine_documents else {},
                "total_tokens": 0,
                "total_cost": 0.0,
                "metadata": {
                    "query": query,
                    "top_k": top_k,
                    "num_documents": 0,
                    "combined": combine_documents,
                },
            }

        logger.info(f"Found {len(search_results)} documents")

        # Step 2: ContextGem extraction
        if combine_documents:
            # Combine all documents into one for holistic extraction
            logger.info("Combining all documents for extraction")
            combined_text = "\n\n---\n\n".join([r.text for r in search_results])
            extraction = self._extract_from_text(
                combined_text, concepts, aspects, add_references, add_justifications
            )
            extractions = extraction
        else:
            # Extract from each document separately
            logger.info("Extracting from each document separately")
            extractions = []
            for i, result in enumerate(search_results, 1):
                logger.info(f"Extracting from document {i}/{len(search_results)}")
                extraction = self._extract_from_text(
                    result.text, concepts, aspects, add_references, add_justifications
                )
                # Add source information
                extraction["source"] = {
                    "id": result.id,
                    "score": result.score,
                    "metadata": result.metadata,
                }
                extractions.append(extraction)

        # Aggregate usage statistics
        if isinstance(extractions, list):
            total_tokens = sum(e.get("tokens_used", 0) for e in extractions)
            total_cost = sum(e.get("cost", 0.0) for e in extractions)
        else:
            total_tokens = extractions.get("tokens_used", 0)
            total_cost = extractions.get("cost", 0.0)

        logger.info(f"Extraction complete. Tokens: {total_tokens}, Cost: ${total_cost:.4f}")

        return {
            "search_results": search_results,
            "extractions": extractions,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "metadata": {
                "query": query,
                "top_k": top_k,
                "num_documents": len(search_results),
                "combined": combine_documents,
            },
        }

    def _extract_from_text(
        self,
        text: str,
        concepts: List[Any],
        aspects: Optional[List[Any]],
        add_references: bool,
        add_justifications: bool,
    ) -> dict[str, Any]:
        """
        Extract structured data from text using ContextGem.

        Args:
            text: Text to extract from
            concepts: List of Concept objects
            aspects: Optional list of Aspect objects
            add_references: Add sentence-level references
            add_justifications: Add extraction justifications

        Returns:
            Dictionary with extracted concepts, aspects, and usage statistics
        """
        from contextgem import Document

        # Create Document
        document = Document(raw_text=text)

        # Configure concepts with references/justifications
        configured_concepts = []
        for concept in concepts:
            # Clone concept to avoid modifying original
            import copy

            concept_copy = copy.deepcopy(concept)

            # Add references if requested
            if add_references and not concept_copy.add_references:
                concept_copy.add_references = True
                concept_copy.reference_depth = "sentences"

            # Add justifications if requested
            if add_justifications and not concept_copy.add_justifications:
                concept_copy.add_justifications = True
                concept_copy.justification_depth = "brief"

            configured_concepts.append(concept_copy)

        document.concepts = configured_concepts
        if aspects:
            document.aspects = aspects

        # Extract using ContextGem
        logger.debug("Running ContextGem extraction...")
        document = self._contextgem_llm.extract_all(document)

        # Format results
        result = {
            "concepts": {},
            "aspects": {} if aspects else None,
            "tokens_used": getattr(document, "tokens_used", 0),
            "cost": getattr(document, "cost", 0.0),
        }

        # Extract concepts
        for concept in document.concepts:
            result["concepts"][concept.name] = [
                {
                    "value": item.value,
                    "justification": getattr(item, "justification", None) if add_justifications else None,
                    "references": (
                        [
                            {"text": ref.raw_text, "position": getattr(ref, "position", None)}
                            for ref in (getattr(item, "reference_sentences", []) or [])
                        ]
                        if add_references
                        else None
                    ),
                }
                for item in concept.extracted_items
            ]

        # Extract aspects
        if aspects and document.aspects:
            for aspect in document.aspects:
                result["aspects"][aspect.name] = [
                    {
                        "value": item.value,
                        "sub_aspects": getattr(item, "sub_aspects", None),
                    }
                    for item in aspect.extracted_items
                ]

        return result

    def extract_from_results(
        self,
        search_results: List[Any],
        concepts: List[Any],
        aspects: Optional[List[Any]] = None,
        add_references: bool = True,
        add_justifications: bool = True,
        combine_documents: bool = False,
    ) -> dict[str, Any]:
        """
        Extract structured data from existing search results.

        Useful when you already have search results and want to extract from them
        without performing another search.

        Args:
            search_results: List of SearchResult objects from previous search
            concepts: List of ContextGem Concept objects to extract
            aspects: Optional list of Aspect objects
            add_references: Add sentence-level references
            add_justifications: Add extraction justifications
            combine_documents: Combine all results into single extraction

        Returns:
            Dictionary with extractions and usage statistics (same format as search_and_extract)
        """
        # Ensure ContextGem LLM is initialized
        self._init_contextgem_llm()

        if not search_results:
            return {
                "search_results": search_results,
                "extractions": [] if not combine_documents else {},
                "total_tokens": 0,
                "total_cost": 0.0,
                "metadata": {
                    "num_documents": 0,
                    "combined": combine_documents,
                },
            }

        # Extract from results
        if combine_documents:
            combined_text = "\n\n---\n\n".join([r.text for r in search_results])
            extraction = self._extract_from_text(
                combined_text, concepts, aspects, add_references, add_justifications
            )
            extractions = extraction
        else:
            extractions = []
            for result in search_results:
                extraction = self._extract_from_text(
                    result.text, concepts, aspects, add_references, add_justifications
                )
                extraction["source"] = {
                    "id": result.id,
                    "score": result.score,
                    "metadata": result.metadata,
                }
                extractions.append(extraction)

        # Aggregate statistics
        if isinstance(extractions, list):
            total_tokens = sum(e.get("tokens_used", 0) for e in extractions)
            total_cost = sum(e.get("cost", 0.0) for e in extractions)
        else:
            total_tokens = extractions.get("tokens_used", 0)
            total_cost = extractions.get("cost", 0.0)

        return {
            "search_results": search_results,
            "extractions": extractions,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "metadata": {
                "num_documents": len(search_results),
                "combined": combine_documents,
            },
        }

    def cleanup(self):
        """Cleanup resources."""
        if self._owns_searcher and hasattr(self.searcher, "cleanup"):
            self.searcher.cleanup()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False
