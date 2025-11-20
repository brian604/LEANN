"""
ContextGem RAG example using the unified interface.
Uses LLM-based extraction to enrich documents with structured data before indexing.
Supports PDF, TXT, MD, and other document formats with intelligent extraction.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from base_rag_example import BaseRAGExample
from contextgem_data import ContextGemProcessor
from llama_index.core import SimpleDirectoryReader


class ContextGemRAG(BaseRAGExample):
    """RAG example with ContextGem-powered document extraction and enrichment."""

    def __init__(self):
        super().__init__(
            name="ContextGem",
            description="Process documents with LLM-based extraction (summary, concepts, entities) for enhanced RAG",
            default_index_name="contextgem_docs",
        )

    def _add_specific_arguments(self, parser):
        """Add ContextGem-specific arguments."""
        # Document parameters
        doc_group = parser.add_argument_group("Document Parameters")
        doc_group.add_argument(
            "--data-dir",
            type=str,
            default="data",
            help="Directory containing documents to index (default: data)",
        )
        doc_group.add_argument(
            "--file-types",
            nargs="+",
            default=None,
            help="Filter by file types (e.g., .pdf .txt .md). If not specified, all supported types are processed",
        )
        doc_group.add_argument(
            "--chunk-size",
            type=int,
            default=512,
            help="Text chunk size in characters (default: 512)",
        )
        doc_group.add_argument(
            "--chunk-overlap",
            type=int,
            default=128,
            help="Text chunk overlap in characters (default: 128)",
        )

        # ContextGem extraction parameters
        gem_group = parser.add_argument_group("ContextGem Extraction Parameters")
        gem_group.add_argument(
            "--extraction-model",
            type=str,
            default="openai/gpt-4o-mini",
            help="LLM model for extraction (default: openai/gpt-4o-mini)",
        )
        gem_group.add_argument(
            "--extraction-api-key",
            type=str,
            default=None,
            help="API key for extraction LLM (defaults to OPENAI_API_KEY)",
        )
        gem_group.add_argument(
            "--no-summary",
            action="store_true",
            help="Disable document summary extraction",
        )
        gem_group.add_argument(
            "--no-concepts",
            action="store_true",
            help="Disable key concepts extraction",
        )
        gem_group.add_argument(
            "--no-entities",
            action="store_true",
            help="Disable named entity extraction",
        )
        gem_group.add_argument(
            "--skip-extraction",
            action="store_true",
            help="Skip ContextGem extraction entirely (use plain chunking)",
        )

    async def load_data(self, args) -> list[str]:
        """Load documents, extract with ContextGem, and convert to enriched chunks."""
        print(f"Loading documents from: {args.data_dir}")
        if args.file_types:
            print(f"Filtering by file types: {args.file_types}")
        else:
            print("Processing all supported file types")

        # Check if data directory exists
        data_path = Path(args.data_dir)
        if not data_path.exists():
            raise ValueError(f"Data directory not found: {args.data_dir}")

        # Load documents
        reader_kwargs = {
            "recursive": True,
            "encoding": "utf-8",
        }
        if args.file_types:
            reader_kwargs["required_exts"] = args.file_types

        documents = SimpleDirectoryReader(args.data_dir, **reader_kwargs).load_data(
            show_progress=True
        )

        if not documents:
            print(f"No documents found in {args.data_dir} with extensions {args.file_types}")
            return []

        print(f"Loaded {len(documents)} documents")

        # Skip extraction if requested
        if args.skip_extraction:
            print("Skipping ContextGem extraction (plain chunking)")
            # Fall back to simple chunking
            all_texts = []
            for doc in documents:
                text = doc.text
                filename = getattr(doc, "metadata", {}).get("file_name", "")

                # Simple chunking
                if len(text) <= args.chunk_size:
                    chunk = f"[Source: {filename}]\n{text}" if filename else text
                    all_texts.append(chunk)
                else:
                    start = 0
                    while start < len(text):
                        end = min(start + args.chunk_size, len(text))
                        chunk = text[start:end]
                        if start == 0 and filename:
                            chunk = f"[Source: {filename}]\n{chunk}"
                        all_texts.append(chunk)
                        start = end - args.chunk_overlap
                        if start >= len(text) - args.chunk_overlap:
                            break

            return all_texts

        # Prepare documents for ContextGem processing
        doc_tuples = []
        for doc in documents:
            text = doc.text
            filename = getattr(doc, "metadata", {}).get("file_name", "unknown")
            doc_tuples.append((text, filename))

        print(f"\nInitializing ContextGem extraction...")
        print(f"  Model: {args.extraction_model}")
        print(f"  Extract summary: {not args.no_summary}")
        print(f"  Extract concepts: {not args.no_concepts}")
        print(f"  Extract entities: {not args.no_entities}")

        # Initialize ContextGem processor
        try:
            processor = ContextGemProcessor(
                model=args.extraction_model,
                api_key=args.extraction_api_key,
                extract_summary=not args.no_summary,
                extract_key_concepts=not args.no_concepts,
                extract_entities=not args.no_entities,
            )

            # Process and create enriched chunks
            print(f"\nExtracting structured data from {len(doc_tuples)} documents...")
            all_texts = processor.create_enriched_chunks(
                doc_tuples,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                show_progress=True,
            )

            print(f"Created {len(all_texts)} enriched chunks")

        except ImportError as e:
            print(f"\nWarning: ContextGem not available ({e})")
            print("Falling back to plain chunking...")

            # Fall back to simple chunking
            all_texts = []
            for text, filename in doc_tuples:
                if len(text) <= args.chunk_size:
                    chunk = f"[Source: {filename}]\n{text}" if filename else text
                    all_texts.append(chunk)
                else:
                    start = 0
                    while start < len(text):
                        end = min(start + args.chunk_size, len(text))
                        chunk = text[start:end]
                        if start == 0 and filename:
                            chunk = f"[Source: {filename}]\n{chunk}"
                        all_texts.append(chunk)
                        start = end - args.chunk_overlap
                        if start >= len(text) - args.chunk_overlap:
                            break

        except Exception as e:
            print(f"\nError during ContextGem extraction: {e}")
            print("Falling back to plain chunking...")

            # Fall back to simple chunking
            all_texts = []
            for text, filename in doc_tuples:
                if len(text) <= args.chunk_size:
                    chunk = f"[Source: {filename}]\n{text}" if filename else text
                    all_texts.append(chunk)
                else:
                    start = 0
                    while start < len(text):
                        end = min(start + args.chunk_size, len(text))
                        chunk = text[start:end]
                        if start == 0 and filename:
                            chunk = f"[Source: {filename}]\n{chunk}"
                        all_texts.append(chunk)
                        start = end - args.chunk_overlap
                        if start >= len(text) - args.chunk_overlap:
                            break

        # Apply max_items limit if specified
        if args.max_items > 0 and len(all_texts) > args.max_items:
            print(f"Limiting to {args.max_items} chunks (from {len(all_texts)})")
            all_texts = all_texts[: args.max_items]

        return all_texts


if __name__ == "__main__":
    import asyncio

    # Example usage
    print("\nContextGem RAG Example")
    print("=" * 50)
    print("\nThis example uses LLM-based extraction to enrich documents")
    print("with summaries, key concepts, and named entities before indexing.")
    print("\nExample usage:")
    print("  python contextgem_rag.py --data-dir ./my_docs")
    print("  python contextgem_rag.py --data-dir ./my_docs --extraction-model openai/gpt-4o")
    print("  python contextgem_rag.py --data-dir ./my_docs --no-entities")
    print("  python contextgem_rag.py --data-dir ./my_docs --skip-extraction")
    print("\nExample queries you can try:")
    print("- 'What are the main topics in these documents?'")
    print("- 'Summarize the key findings'")
    print("- 'What organizations are mentioned?'")
    print("- 'What concepts are discussed?'")
    print("\nOr run without --query for interactive mode\n")

    rag = ContextGemRAG()
    asyncio.run(rag.run())
