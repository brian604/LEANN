"""
ContextGem processor for extracting structured data from documents.
Uses LLMs to extract concepts, aspects, and insights from documents.
"""

import os
from pathlib import Path
from typing import Any

try:
    from contextgem import (
        Document,
        DocumentLLM,
        Aspect,
        StringConcept,
        BooleanConcept,
    )

    CONTEXTGEM_AVAILABLE = True
except ImportError:
    CONTEXTGEM_AVAILABLE = False


class ContextGemProcessor:
    """
    Processor for extracting structured data from documents using ContextGem.

    This processor uses LLMs to extract:
    - Key concepts and entities
    - Document aspects and sections
    - Summaries and insights
    - Metadata for enhanced RAG
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        api_key: str | None = None,
        extract_summary: bool = True,
        extract_key_concepts: bool = True,
        extract_entities: bool = True,
        custom_aspects: list[dict[str, Any]] | None = None,
        custom_concepts: list[dict[str, Any]] | None = None,
    ):
        """
        Initialize the ContextGem processor.

        Args:
            model: LLM model to use (e.g., "openai/gpt-4o-mini", "anthropic/claude-3-haiku")
            api_key: API key for the LLM provider (defaults to env var)
            extract_summary: Whether to extract document summary
            extract_key_concepts: Whether to extract key concepts
            extract_entities: Whether to extract named entities
            custom_aspects: Custom aspects to extract (list of dicts with 'name', 'description')
            custom_concepts: Custom concepts to extract (list of dicts with 'name', 'description', 'type')
        """
        if not CONTEXTGEM_AVAILABLE:
            raise ImportError(
                "ContextGem is not installed. Install it with: pip install contextgem"
            )

        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.extract_summary = extract_summary
        self.extract_key_concepts = extract_key_concepts
        self.extract_entities = extract_entities
        self.custom_aspects = custom_aspects or []
        self.custom_concepts = custom_concepts or []

    def _create_default_aspects(self) -> list:
        """Create default aspects for extraction."""
        aspects = []

        if self.extract_summary:
            aspects.append(
                Aspect(
                    name="summary",
                    description="A concise summary of the main content and purpose of this document",
                )
            )

        return aspects

    def _create_default_concepts(self) -> list:
        """Create default concepts for extraction."""
        concepts = []

        if self.extract_key_concepts:
            concepts.append(
                StringConcept(
                    name="key_concepts",
                    description="The main concepts, topics, and themes discussed in this document",
                )
            )

        if self.extract_entities:
            concepts.append(
                StringConcept(
                    name="entities",
                    description="Named entities mentioned in the document (people, organizations, locations, products)",
                )
            )

        # Add custom concepts
        for concept_def in self.custom_concepts:
            concept_type = concept_def.get("type", "string")
            if concept_type == "boolean":
                concepts.append(
                    BooleanConcept(
                        name=concept_def["name"],
                        description=concept_def["description"],
                    )
                )
            else:
                concepts.append(
                    StringConcept(
                        name=concept_def["name"],
                        description=concept_def["description"],
                    )
                )

        return concepts

    def _create_custom_aspects(self) -> list:
        """Create custom aspects from configuration."""
        aspects = []
        for aspect_def in self.custom_aspects:
            aspects.append(
                Aspect(
                    name=aspect_def["name"],
                    description=aspect_def["description"],
                )
            )
        return aspects

    def process_document(self, text: str, filename: str = "") -> dict[str, Any]:
        """
        Process a single document and extract structured data.

        Args:
            text: Document text content
            filename: Optional filename for context

        Returns:
            Dictionary containing extracted data:
            - raw_text: Original text
            - summary: Extracted summary (if enabled)
            - key_concepts: Extracted concepts (if enabled)
            - entities: Extracted entities (if enabled)
            - aspects: Extracted aspects
            - concepts: All extracted concepts
            - enriched_text: Text enriched with extracted metadata
        """
        # Create document
        doc = Document(raw_text=text)

        # Add aspects
        aspects = self._create_default_aspects() + self._create_custom_aspects()
        for aspect in aspects:
            doc.add_aspect(aspect)

        # Add concepts
        concepts = self._create_default_concepts()
        for concept in concepts:
            doc.add_concept(concept)

        # Initialize LLM and extract
        llm = DocumentLLM(
            model=self.model,
            api_key=self.api_key,
        )

        # Perform extraction
        doc = llm.extract_all(doc)

        # Collect results
        result = {
            "raw_text": text,
            "filename": filename,
            "aspects": {},
            "concepts": {},
        }

        # Extract aspect results
        for aspect in doc.aspects:
            if hasattr(aspect, "extracted_text") and aspect.extracted_text:
                result["aspects"][aspect.name] = aspect.extracted_text

        # Extract concept results
        for concept in doc.concepts:
            if hasattr(concept, "extracted_items") and concept.extracted_items:
                result["concepts"][concept.name] = concept.extracted_items
            elif hasattr(concept, "value"):
                result["concepts"][concept.name] = concept.value

        # Create enriched text with metadata
        enriched_parts = [text]

        if result["aspects"].get("summary"):
            enriched_parts.append(f"\n\n[Summary: {result['aspects']['summary']}]")

        if result["concepts"].get("key_concepts"):
            concepts_str = result["concepts"]["key_concepts"]
            if isinstance(concepts_str, list):
                concepts_str = ", ".join(str(c) for c in concepts_str)
            enriched_parts.append(f"\n[Key Concepts: {concepts_str}]")

        if result["concepts"].get("entities"):
            entities_str = result["concepts"]["entities"]
            if isinstance(entities_str, list):
                entities_str = ", ".join(str(e) for e in entities_str)
            enriched_parts.append(f"\n[Entities: {entities_str}]")

        result["enriched_text"] = "".join(enriched_parts)

        return result

    def process_documents(
        self,
        documents: list[tuple[str, str]],
        show_progress: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Process multiple documents and extract structured data.

        Args:
            documents: List of tuples (text, filename)
            show_progress: Whether to show progress

        Returns:
            List of extraction results
        """
        results = []
        total = len(documents)

        for i, (text, filename) in enumerate(documents):
            if show_progress:
                print(f"Processing document {i + 1}/{total}: {filename}")

            try:
                result = self.process_document(text, filename)
                results.append(result)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                # Return basic result on error
                results.append({
                    "raw_text": text,
                    "filename": filename,
                    "enriched_text": text,
                    "aspects": {},
                    "concepts": {},
                    "error": str(e),
                })

        return results

    def create_enriched_chunks(
        self,
        documents: list[tuple[str, str]],
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        show_progress: bool = True,
    ) -> list[str]:
        """
        Process documents and create enriched text chunks for indexing.

        This method:
        1. Extracts structured data from each document
        2. Creates chunks with embedded metadata
        3. Returns chunks ready for LEANN indexing

        Args:
            documents: List of tuples (text, filename)
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            show_progress: Whether to show progress

        Returns:
            List of enriched text chunks
        """
        # Process all documents
        results = self.process_documents(documents, show_progress)

        # Create chunks from enriched text
        all_chunks = []

        for result in results:
            enriched_text = result.get("enriched_text", result["raw_text"])
            filename = result.get("filename", "")

            # Simple chunking with overlap
            if len(enriched_text) <= chunk_size:
                # Add filename context for small documents
                chunk = f"[Source: {filename}]\n{enriched_text}" if filename else enriched_text
                all_chunks.append(chunk)
            else:
                # Split into chunks
                start = 0
                while start < len(enriched_text):
                    end = min(start + chunk_size, len(enriched_text))
                    chunk = enriched_text[start:end]

                    # Add filename to first chunk
                    if start == 0 and filename:
                        chunk = f"[Source: {filename}]\n{chunk}"

                    all_chunks.append(chunk)
                    start = end - chunk_overlap

                    # Prevent infinite loop
                    if start >= len(enriched_text) - chunk_overlap:
                        break

        return all_chunks


def create_contextgem_processor(
    model: str = "openai/gpt-4o-mini",
    api_key: str | None = None,
    **kwargs,
) -> ContextGemProcessor:
    """
    Factory function to create a ContextGem processor.

    Args:
        model: LLM model to use
        api_key: API key for the provider
        **kwargs: Additional arguments passed to ContextGemProcessor

    Returns:
        Configured ContextGemProcessor instance
    """
    return ContextGemProcessor(model=model, api_key=api_key, **kwargs)
