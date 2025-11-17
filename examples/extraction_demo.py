"""
LEANN + ContextGem Extraction Demo

This example demonstrates how to use LEANN's extraction capabilities to:
1. Search for relevant documents using semantic search
2. Extract structured data using LLM-powered extraction

Installation:
    uv pip install leann-core[extract]
    # or
    uv pip install contextgem

Usage:
    python examples/extraction_demo.py
"""

import os


def demo_basic_extraction():
    """Demo 1: Basic extraction - companies and revenue from documents."""
    print("=" * 80)
    print("Demo 1: Basic Extraction - Companies and Revenue")
    print("=" * 80)

    from contextgem import StringConcept
    from leann import LeannBuilder, LeannExtractor

    # Step 1: Build a sample index
    print("\n📚 Building sample index...")
    builder = LeannBuilder(backend_name="hnsw", embedding_model="sentence-transformers/all-MiniLM-L6-v2")

    # Add sample documents about tech companies
    builder.add_text(
        "Apple Inc. reported quarterly revenue of $90 billion in Q4 2024, "
        "exceeding analyst expectations. iPhone sales grew 15% year-over-year.",
        metadata={"company": "Apple", "quarter": "Q4 2024"},
    )

    builder.add_text(
        "Microsoft Azure cloud services grew 31% in Q3 2024. "
        "CEO Satya Nadella highlighted AI investments and OpenAI partnership.",
        metadata={"company": "Microsoft", "quarter": "Q3 2024"},
    )

    builder.add_text(
        "Google announced Gemini AI integration across all products. "
        "Search revenue reached $50 billion this quarter.",
        metadata={"company": "Google", "quarter": "Q3 2024"},
    )

    index_path = "demo_index.leann"
    builder.build_index(index_path)
    print(f"✅ Index built: {index_path}")

    # Step 2: Create extractor and search
    print("\n🔍 Searching and extracting...")
    extractor = LeannExtractor(
        index_path,
        llm_config={"model": "openai/gpt-4o-mini"},  # Uses OPENAI_API_KEY env var
    )

    # Define what to extract
    concepts = [
        StringConcept(name="Companies", description="Company names mentioned"),
        StringConcept(name="Revenue", description="Revenue figures and growth metrics"),
        StringConcept(name="Products", description="Products or services mentioned"),
    ]

    # Search and extract
    result = extractor.search_and_extract(
        query="technology company earnings revenue",
        concepts=concepts,
        top_k=3,
        add_references=True,
        add_justifications=True,
    )

    # Step 3: Display results
    print(f"\n📊 Found {len(result['search_results'])} documents")
    print(f"💰 Cost: ${result['total_cost']:.4f}")
    print(f"🔢 Tokens: {result['total_tokens']:,}")

    print("\n📄 Extracted Data:")
    for i, extraction in enumerate(result["extractions"], 1):
        print(f"\n  Document {i}:")
        print(f"  Score: {extraction['source']['score']:.3f}")
        print(f"  Company: {extraction['source']['metadata'].get('company', 'N/A')}")

        # Show extracted concepts
        for concept_name, items in extraction["concepts"].items():
            print(f"\n    {concept_name}:")
            for item in items:
                print(f"      - {item['value']}")
                if item.get("justification"):
                    print(f"        Reason: {item['justification']}")

    # Cleanup
    extractor.cleanup()
    print("\n✅ Demo 1 complete!")


def demo_json_schema_extraction():
    """Demo 2: Structured extraction using JSON schema."""
    print("\n" + "=" * 80)
    print("Demo 2: JSON Schema Extraction - Company Metrics")
    print("=" * 80)

    from contextgem import JsonObjectConcept
    from leann import LeannExtractor

    # Define JSON schema for extraction
    company_schema = JsonObjectConcept(
        name="CompanyMetrics",
        description="Company performance metrics",
        json_schema={
            "company_name": {"type": "string"},
            "revenue": {"type": "string"},
            "growth_rate": {"type": "string"},
            "key_products": {"type": "array", "items": {"type": "string"}},
            "quarter": {"type": "string"},
        },
    )

    # Use existing index
    extractor = LeannExtractor("demo_index.leann", llm_config={"model": "openai/gpt-4o-mini"})

    result = extractor.search_and_extract(
        query="quarterly performance metrics", concepts=[company_schema], top_k=3
    )

    print(f"\n📊 Extracted Structured Data:")
    for i, extraction in enumerate(result["extractions"], 1):
        print(f"\n  Document {i}:")
        for item in extraction["concepts"]["CompanyMetrics"]:
            metrics = item["value"]
            print(f"    Company: {metrics.get('company_name', 'N/A')}")
            print(f"    Revenue: {metrics.get('revenue', 'N/A')}")
            print(f"    Growth: {metrics.get('growth_rate', 'N/A')}")
            print(f"    Products: {', '.join(metrics.get('key_products', []))}")
            print(f"    Quarter: {metrics.get('quarter', 'N/A')}")

    print(f"\n💰 Cost: ${result['total_cost']:.4f}")
    extractor.cleanup()
    print("\n✅ Demo 2 complete!")


def demo_combined_extraction():
    """Demo 3: Combine all documents for holistic extraction."""
    print("\n" + "=" * 80)
    print("Demo 3: Combined Extraction - Market Overview")
    print("=" * 80)

    from contextgem import StringConcept
    from leann import LeannExtractor

    extractor = LeannExtractor("demo_index.leann", llm_config={"model": "openai/gpt-4o-mini"})

    # Extract with combined documents
    result = extractor.search_and_extract(
        query="technology industry trends",
        concepts=[
            StringConcept(name="MarketTrends", description="Overall market trends and patterns"),
            StringConcept(name="KeyPlayers", description="Major companies and their strategies"),
            StringConcept(name="FocusAreas", description="Technology focus areas (AI, cloud, etc.)"),
        ],
        top_k=3,
        combine_documents=True,  # Combine all results into one extraction
    )

    print(f"\n📊 Combined Analysis Across {len(result['search_results'])} Documents:")
    extraction = result["extractions"]  # Single dict instead of list

    for concept_name, items in extraction["concepts"].items():
        print(f"\n  {concept_name}:")
        for item in items:
            print(f"    - {item['value']}")

    print(f"\n💰 Cost: ${result['total_cost']:.4f}")
    extractor.cleanup()
    print("\n✅ Demo 3 complete!")


def demo_with_metadata_filtering():
    """Demo 4: Extraction with metadata filtering."""
    print("\n" + "=" * 80)
    print("Demo 4: Metadata Filtering - Specific Quarter Analysis")
    print("=" * 80)

    from contextgem import StringConcept
    from leann import LeannExtractor

    extractor = LeannExtractor("demo_index.leann", llm_config={"model": "openai/gpt-4o-mini"})

    # Extract only from Q3 2024 documents
    result = extractor.search_and_extract(
        query="revenue growth",
        concepts=[
            StringConcept(name="Q3Metrics", description="Q3 2024 performance metrics"),
        ],
        top_k=10,
        metadata_filters={"quarter": {"==": "Q3 2024"}},  # Filter by quarter
    )

    print(f"\n📊 Q3 2024 Analysis ({len(result['search_results'])} documents):")
    for extraction in result["extractions"]:
        company = extraction["source"]["metadata"].get("company", "Unknown")
        print(f"\n  {company}:")
        for item in extraction["concepts"]["Q3Metrics"]:
            print(f"    - {item['value']}")

    print(f"\n💰 Cost: ${result['total_cost']:.4f}")
    extractor.cleanup()
    print("\n✅ Demo 4 complete!")


def demo_ollama_local_extraction():
    """Demo 5: Local extraction using Ollama (privacy-focused)."""
    print("\n" + "=" * 80)
    print("Demo 5: Local LLM Extraction with Ollama")
    print("=" * 80)

    try:
        import requests

        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code != 200:
            print("⚠️  Ollama not running. Start with: ollama serve")
            return
    except Exception:
        print("⚠️  Ollama not available. Skipping this demo.")
        return

    from contextgem import StringConcept
    from leann import LeannExtractor

    # Use local Ollama for complete privacy
    extractor = LeannExtractor(
        "demo_index.leann",
        llm_config={
            "llm": "ollama",
            "model": "qwen2.5:7b",  # Or any model you have pulled
            "host": "http://localhost:11434",
        },
    )

    result = extractor.search_and_extract(
        query="AI developments",
        concepts=[
            StringConcept(name="AITopics", description="AI-related topics and developments"),
        ],
        top_k=2,
    )

    print(f"\n📊 Local Extraction Results:")
    for extraction in result["extractions"]:
        print(f"\n  {extraction['source']['metadata'].get('company', 'Unknown')}:")
        for item in extraction["concepts"]["AITopics"]:
            print(f"    - {item['value']}")

    print(f"\n🔐 Privacy: All processing done locally")
    print(f"🔢 Tokens: {result['total_tokens']:,}")
    extractor.cleanup()
    print("\n✅ Demo 5 complete!")


def main():
    """Run all demos."""
    # Check if OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set. Some demos will be skipped.")
        print("Set it with: export OPENAI_API_KEY='your-key'")
        print("\nYou can still run Demo 5 (Ollama) if you have it installed.")

    print("\n🚀 LEANN + ContextGem Extraction Demos")
    print("=" * 80)

    try:
        # Run demos
        if os.getenv("OPENAI_API_KEY"):
            demo_basic_extraction()
            demo_json_schema_extraction()
            demo_combined_extraction()
            demo_with_metadata_filtering()

        # Ollama demo (optional)
        demo_ollama_local_extraction()

        print("\n" + "=" * 80)
        print("✅ All demos complete!")
        print("=" * 80)

        # Cleanup demo index
        import os

        if os.path.exists("demo_index.leann"):
            import shutil

            shutil.rmtree(".leann", ignore_errors=True)
            print("\n🧹 Cleaned up demo files")

    except KeyboardInterrupt:
        print("\n\n⚠️  Demos interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
