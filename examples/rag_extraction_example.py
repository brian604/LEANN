"""
RAG Application Extraction Example

This example demonstrates how to use extraction capabilities with LEANN RAG applications.
Shows both programmatic usage and CLI command examples.

Installation:
    uv pip install leann-core[extract]

Usage:
    python examples/rag_extraction_example.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add apps directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "apps"))


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def demo_cli_usage():
    """Demonstrate CLI usage examples for RAG extraction."""
    print_section("RAG Extraction - CLI Examples")

    examples = [
        {
            "title": "1. Extract from Research Papers",
            "description": "Extract methodologies, findings, and datasets from academic papers",
            "command": """python -m apps.document_rag \\
  --data-dir ./data \\
  --query "What research methodologies were used?" \\
  --extract \\
  --extract-schema examples/extraction_schemas/research_paper.json \\
  --llm openai \\
  --llm-model gpt-4o-mini""",
        },
        {
            "title": "2. Extract Financial Metrics",
            "description": "Extract structured financial data from earnings reports",
            "command": """python -m apps.document_rag \\
  --data-dir ./earnings \\
  --query "Q4 2024 financial performance" \\
  --extract \\
  --extract-schema examples/extraction_schemas/company_earnings.json \\
  --extract-output financial_metrics.json \\
  --top-k 10""",
        },
        {
            "title": "3. Extract with Inline Concepts",
            "description": "Define extraction concepts directly in the command",
            "command": """python -m apps.document_rag \\
  --query "AI technology developments" \\
  --extract \\
  --extract-concepts \\
    "Companies:Technology companies mentioned" \\
    "Technologies:AI technologies and models" \\
    "Applications:Practical applications"  \\
  --llm openai""",
        },
        {
            "title": "4. Combined Extraction for Comparison",
            "description": "Combine multiple documents for holistic analysis",
            "command": """python -m apps.document_rag \\
  --query "market trends 2024" \\
  --extract \\
  --extract-concepts "Trends:Overall market trends" \\
  --extract-combine \\
  --top-k 20""",
        },
        {
            "title": "5. Privacy-Focused Local Extraction",
            "description": "Use Ollama for completely private extraction",
            "command": """# First: ollama pull qwen2.5:7b

python -m apps.document_rag \\
  --data-dir ./confidential \\
  --query "extract sensitive data" \\
  --extract \\
  --extract-schema schema.json \\
  --llm ollama \\
  --llm-model qwen2.5:7b \\
  --embedding-mode ollama \\
  --embedding-model nomic-embed-text""",
        },
        {
            "title": "6. Contract Analysis",
            "description": "Extract key terms from legal contracts",
            "command": """python -m apps.document_rag \\
  --data-dir ./contracts \\
  --file-types .pdf .docx \\
  --query "SaaS subscription agreements" \\
  --extract \\
  --extract-schema examples/extraction_schemas/contract_terms.json \\
  --extract-output contract_analysis.json""",
        },
    ]

    for example in examples:
        print(f"\033[1m{example['title']}\033[0m")
        print(f"{example['description']}\n")
        print(f"\033[36m{example['command']}\033[0m\n")


async def demo_programmatic_usage():
    """Demonstrate programmatic usage of extraction with RAG apps."""
    print_section("RAG Extraction - Programmatic Usage")

    # Check if we have the necessary dependencies
    try:
        from contextgem import JsonObjectConcept, StringConcept
        from leann import LeannExtractor
    except ImportError:
        print("⚠️  ContextGem not installed. Install with: uv pip install leann-core[extract]")
        print("\nThis demo requires extraction capabilities.")
        return

    # Check if we have an API key for this demo
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set. This demo requires OpenAI API access.")
        print("Set it with: export OPENAI_API_KEY='your-key'")
        print("\nAlternatively, use Ollama with local models (see CLI examples)")
        return

    from document_rag import DocumentRAG

    print("Creating sample documents for extraction demo...\n")

    # Create a temporary data directory
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample documents
        doc1_path = Path(temp_dir) / "ai_report_2024.txt"
        doc1_path.write_text(
            """
        AI Industry Report 2024

        OpenAI released GPT-4o in March 2024, achieving 89% accuracy on complex reasoning tasks.
        The model uses a transformer architecture with 1.5 trillion parameters.

        Google DeepMind announced Gemini 1.5 Pro with 1 million token context window.
        Key applications include code generation, document analysis, and scientific research.

        Anthropic's Claude 3 Opus demonstrated strong performance on safety benchmarks.
        Total industry investment in AI reached $120 billion in Q3 2024.
        """
        )

        doc2_path = Path(temp_dir) / "ml_research.txt"
        doc2_path.write_text(
            """
        Machine Learning Research Findings

        Study used BERT and GPT models for sentiment analysis on 100,000 reviews.
        Methodology: Cross-validation with 80/20 train-test split.

        Key findings:
        - GPT-4 achieved 92% F1-score on sentiment classification
        - BERT performed better on shorter texts (< 100 words)
        - Fine-tuning improved accuracy by 15%

        Datasets: Stanford Sentiment Treebank, IMDB Reviews
        Limitations: English-only evaluation, limited to product reviews
        """
        )

        print(f"✅ Created sample documents in {temp_dir}\n")

        # Example 1: Extract with String Concepts
        print("\n" + "-" * 80)
        print("Example 1: String Concept Extraction")
        print("-" * 80 + "\n")

        app = DocumentRAG()

        # Manually set up the arguments (simulate CLI args)
        class Args:
            data_dir = temp_dir
            file_types = None
            chunk_size = 256
            chunk_overlap = 128
            enable_code_chunking = False
            index_dir = f"{temp_dir}/index"
            max_items = -1
            force_rebuild = True
            embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            embedding_mode = "sentence-transformers"
            embedding_host = None
            embedding_api_base = None
            embedding_api_key = None
            backend_name = "hnsw"
            graph_degree = 32
            build_complexity = 64
            no_compact = False
            no_recompute = False
            llm = "openai"
            llm_model = "gpt-4o-mini"
            llm_host = None
            llm_api_base = None
            llm_api_key = None
            thinking_budget = None
            top_k = 5
            search_complexity = 32
            use_ast_chunking = False
            query = "AI models and their performance"
            extract = True
            extract_concepts = ["Models:AI models mentioned", "Metrics:Performance metrics"]
            extract_schema = None
            extract_combine = False
            extract_no_references = True  # Simpler output for demo
            extract_no_justifications = True
            extract_output = None

        args = Args()

        # Load data and build index
        print("Loading documents...")
        texts = await app.load_data(args)
        print(f"Loaded {len(texts)} text chunks\n")

        print("Building index...")
        index_path = await app.build_index(args, texts)
        print(f"Index created at {index_path}\n")

        # Run extraction
        print("Running extraction with String Concepts...")
        await app.run_extraction(args, index_path, args.query)

        # Example 2: Extract with JSON Schema
        print("\n" + "-" * 80)
        print("Example 2: JSON Schema Extraction")
        print("-" * 80 + "\n")

        # Create a JSON schema
        schema_path = Path(temp_dir) / "ai_model_schema.json"
        import json

        schema_data = {
            "type": "json",
            "name": "AIModelInfo",
            "description": "AI model information and specifications",
            "schema": {
                "model_name": {"type": "string", "description": "Name of the AI model"},
                "release_date": {"type": "string", "description": "Release date or year"},
                "parameters": {"type": "string", "description": "Number of parameters"},
                "key_features": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key features or capabilities",
                },
                "performance_metrics": {
                    "type": "object",
                    "description": "Performance scores and metrics",
                },
            },
        }

        schema_path.write_text(json.dumps(schema_data, indent=2))

        # Update args for JSON schema extraction
        args.query = "AI model specifications and performance"
        args.extract_concepts = None
        args.extract_schema = str(schema_path)

        print("Running extraction with JSON Schema...")
        await app.run_extraction(args, index_path, args.query)

        # Example 3: Combined Extraction
        print("\n" + "-" * 80)
        print("Example 3: Combined Extraction (Holistic Analysis)")
        print("-" * 80 + "\n")

        args.query = "overall AI industry trends"
        args.extract_concepts = ["Trends:AI industry trends and patterns"]
        args.extract_schema = None
        args.extract_combine = True  # Combine all documents

        print("Running combined extraction...")
        await app.run_extraction(args, index_path, args.query)

    print("\n✅ Programmatic extraction examples complete!")


async def main():
    """Run all demos."""
    print("\n" + "🚀" + "=" * 78)
    print("  LEANN RAG Application Extraction Examples")
    print("=" * 79)

    # CLI examples (just show commands)
    demo_cli_usage()

    # Programmatic examples (actually run if possible)
    if os.getenv("OPENAI_API_KEY"):
        try:
            await demo_programmatic_usage()
        except Exception as e:
            print(f"\n❌ Error running programmatic demo: {e}")
            import traceback

            traceback.print_exc()
    else:
        print_section("Programmatic Usage Demo")
        print("⚠️  Skipping programmatic demo (OPENAI_API_KEY not set)")
        print("\nSet your API key to run the full demo:")
        print("  export OPENAI_API_KEY='your-key'")
        print("  python examples/rag_extraction_example.py")

    print("\n" + "=" * 80)
    print("  Next Steps")
    print("=" * 80)
    print("""
1. Try the CLI examples with your own documents
2. Create custom extraction schemas in examples/extraction_schemas/
3. Use extraction with any RAG app (document_rag, code_rag, email_rag, etc.)
4. Combine with metadata filtering for targeted extraction
5. Use local LLMs (Ollama) for privacy-focused extraction

Documentation:
- Extraction schemas: examples/extraction_schemas/README.md
- RAG apps: apps/README.md
- Core extraction: examples/extraction_demo.py
    """)


if __name__ == "__main__":
    asyncio.run(main())
