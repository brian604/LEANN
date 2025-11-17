# LEANN Extraction Guide

**Version:** 0.3.5
**Last Updated:** 2025-11-17

## Overview

LEANN's extraction feature combines **semantic search** with **structured data extraction** using [ContextGem](https://deepwiki.com/shcherbak-ai/contextgem). This enables you to:

1. **Search** your data semantically (LEANN)
2. **Extract** structured information (ContextGem)
3. **Structure** the results as JSON

**Two-Stage Workflow:**
```
Query → LEANN Search → Retrieved Documents → ContextGem Extraction → Structured Data
```

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Extraction Concepts](#extraction-concepts)
4. [Python API](#python-api)
5. [RAG App Integration](#rag-app-integration)
6. [Pre-Built Schemas](#pre-built-schemas)
7. [Creating Custom Schemas](#creating-custom-schemas)
8. [Best Practices](#best-practices)
9. [Privacy & Cost](#privacy--cost)
10. [Troubleshooting](#troubleshooting)

---

## Installation

Install LEANN with extraction support:

```bash
# Using pip
pip install leann[extract]

# Using uv
uv pip install leann[extract]

# From source
cd LEANN
uv sync --extra extract
```

**Dependencies:**
- `contextgem>=0.3.0` - Structured extraction framework
- `litellm>=1.0.0` - LLM provider abstraction

---

## Quick Start

### Python API

```python
from leann import LeannBuilder, LeannExtractor
from contextgem import StringConcept

# 1. Build index
builder = LeannBuilder(backend_name="hnsw")
builder.add_text("LEANN achieves 97% storage reduction through graph-based recomputation.")
builder.add_text("ContextGem extracts structured data using LLMs.")
builder.build_index("my_index.leann")

# 2. Extract structured data
extractor = LeannExtractor("my_index.leann")

concepts = [
    StringConcept("StorageSavings", "Percentage of storage reduction achieved"),
    StringConcept("TechnicalApproach", "Methods and techniques used"),
]

result = extractor.search_and_extract(
    query="How does LEANN save storage?",
    concepts=concepts,
    top_k=5
)

# 3. Access results
for extraction in result["extractions"]:
    print(f"{extraction['concept']}: {extraction['value']}")
    print(f"  Reference: {extraction['reference']}")
    print(f"  Justification: {extraction['justification']}\n")
```

**Output:**
```
StorageSavings: 97%
  Reference: Chunk 0
  Justification: The document explicitly states "97% storage reduction"

TechnicalApproach: Graph-based recomputation
  Reference: Chunk 0
  Justification: The approach is described as "graph-based recomputation"
```

### RAG App Integration

```bash
# Extract from documents using pre-built schema
python -m apps.document_rag \
  --query "What are the main findings?" \
  --extract \
  --extract-schema examples/extraction_schemas/research_paper.json \
  --extract-output results.json

# Extract inline concepts
python -m apps.email_rag \
  --query "food orders" \
  --extract \
  --extract-concepts "OrderDate" "Restaurant" "TotalAmount" "Items"

# Use local LLM for privacy
python -m apps.code_rag \
  --repo-dir ./myproject \
  --query "authentication flow" \
  --extract \
  --extract-concepts "EntryPoint" "Dependencies" "SecurityMeasures" \
  --llm ollama \
  --llm-model qwen2.5:7b
```

---

## Extraction Concepts

ContextGem supports two types of extraction concepts:

### 1. String Concepts

Extract free-form text information:

```python
from contextgem import StringConcept

concepts = [
    StringConcept(
        name="KeyFinding",
        description="The main research finding or conclusion"
    ),
    StringConcept(
        name="Methodology",
        description="Research methods and experimental design used"
    ),
]
```

**Use Cases:** Research papers, articles, reports, emails

### 2. JSON Object Concepts

Extract structured data with schema validation:

```python
from contextgem import JsonObjectConcept

concept = JsonObjectConcept(
    name="CompanyMetrics",
    description="Financial performance metrics",
    schema={
        "company_name": {"type": "string"},
        "quarter": {"type": "string"},
        "revenue": {"type": "string"},
        "revenue_growth": {"type": "string"},
        "net_income": {"type": "string"},
        "eps": {"type": "string"},
        "key_products": {"type": "array", "items": {"type": "string"}},
        "guidance": {"type": "string"},
        "risks": {"type": "array", "items": {"type": "string"}},
    }
)
```

**Use Cases:** Financial reports, contracts, structured documents

---

## Python API

### LeannExtractor Class

```python
from leann import LeannExtractor

extractor = LeannExtractor(
    index_path="my_index.leann",
    llm_config={
        "llm_type": "openai",  # or "ollama", "hf"
        "llm_model": "gpt-4o-mini",
        "thinking_budget": None,  # For reasoning models (o3, o3-mini)
    },
    gpu_id=0,  # Optional: GPU selection
)
```

### search_and_extract()

Main extraction method:

```python
result = extractor.search_and_extract(
    query="Your search query",
    concepts=[...],  # List of StringConcept or JsonObjectConcept
    aspects=None,  # Optional: List of aspects to focus on
    top_k=5,  # Number of documents to retrieve
    complexity=64,  # Search complexity (higher = more accurate)
    add_references=True,  # Include source references
    add_justifications=True,  # Include reasoning
    combine_documents=False,  # Merge all docs before extraction
    metadata_filters={},  # Optional metadata filtering
)
```

**Returns:**
```python
{
    "search_results": [...],  # LEANN search results
    "extractions": [
        {
            "concept": "ConceptName",
            "value": "Extracted value",
            "reference": "Chunk 3",
            "justification": "Reasoning...",
            "metadata": {...},
        },
        ...
    ],
    "total_tokens": 1234,
    "total_cost": 0.0056,
}
```

### extract_from_results()

Extract from existing search results:

```python
from leann import LeannSearcher

searcher = LeannSearcher("my_index.leann")
results = searcher.search("query", top_k=10)

extractions = extractor.extract_from_results(
    results=results,
    concepts=[...],
    combine_documents=False,
)
```

### Context Manager

Automatic cleanup:

```python
with LeannExtractor("my_index.leann") as extractor:
    result = extractor.search_and_extract(...)
    # Process results
# Searcher automatically cleaned up
```

---

## RAG App Integration

All RAG apps (`document_rag`, `code_rag`, `email_rag`, etc.) support extraction via command-line flags:

### Extraction Flags

```bash
--extract                    # Enable extraction
--extract-concepts C1 C2 ... # Inline concept names (string concepts)
--extract-schema FILE        # Load schema from JSON file
--extract-combine            # Combine all documents before extraction
--extract-no-references      # Disable source references
--extract-no-justifications  # Disable reasoning explanations
--extract-output FILE        # Save results to JSON file
```

### Examples

**1. Research Paper Analysis:**
```bash
python -m apps.document_rag \
  --docs ./papers/ \
  --query "machine learning techniques" \
  --extract \
  --extract-schema examples/extraction_schemas/research_paper.json \
  --extract-output findings.json
```

**2. Financial Analysis:**
```bash
python -m apps.document_rag \
  --docs ./earnings_reports/ \
  --query "Q3 2024 performance" \
  --extract \
  --extract-schema examples/extraction_schemas/company_earnings.json \
  --top-k 3 \
  --extract-combine
```

**3. Code Analysis:**
```bash
python -m apps.code_rag \
  --repo-dir ./myproject \
  --query "error handling" \
  --extract \
  --extract-concepts "ErrorTypes" "HandlingStrategies" "LoggingApproach"
```

**4. Email Search:**
```bash
python -m apps.email_rag \
  --query "project deadlines" \
  --extract \
  --extract-concepts "ProjectName" "Deadline" "Owner" "Status"
```

---

## Pre-Built Schemas

LEANN includes 4 production-ready schemas in `examples/extraction_schemas/`:

### 1. research_paper.json

Extract key information from academic papers:

**Concepts:**
- `Methodologies` - Research methods and experimental designs
- `KeyFindings` - Main research findings and contributions
- `Datasets` - Datasets used for experiments
- `Limitations` - Acknowledged limitations
- `FutureWork` - Suggested future research directions

**Usage:**
```bash
python -m apps.document_rag \
  --docs ./papers/ \
  --query "deep learning research" \
  --extract \
  --extract-schema examples/extraction_schemas/research_paper.json
```

### 2. company_earnings.json

Extract financial metrics from earnings reports:

**Schema Fields:**
- `company_name`, `quarter`, `revenue`, `revenue_growth`
- `net_income`, `eps`, `key_products`, `guidance`, `risks`

**Usage:**
```bash
python -m apps.document_rag \
  --docs ./earnings/ \
  --query "Q4 2024 results" \
  --extract \
  --extract-schema examples/extraction_schemas/company_earnings.json \
  --extract-combine
```

### 3. contract_terms.json

Extract key terms from legal contracts:

**Schema Fields:**
- `contract_type`, `parties`, `effective_date`, `termination_date`
- `payment_terms`, `key_obligations`, `termination_clauses`, `governing_law`

**Usage:**
```bash
python -m apps.document_rag \
  --docs ./contracts/ \
  --query "vendor agreements" \
  --extract \
  --extract-schema examples/extraction_schemas/contract_terms.json
```

### 4. technical_doc.json

Extract information from technical documentation:

**Concepts:**
- `APIs` - API endpoints and interfaces
- `Configuration` - Configuration options
- `Dependencies` - Required dependencies
- `Installation` - Installation steps
- `Examples` - Usage examples
- `Troubleshooting` - Common issues and solutions

**Usage:**
```bash
python -m apps.document_rag \
  --docs ./docs/ \
  --query "getting started" \
  --extract \
  --extract-schema examples/extraction_schemas/technical_doc.json
```

---

## Creating Custom Schemas

### String Concept Schema

Create a JSON file with an array of concepts:

```json
[
  {
    "name": "ConceptName",
    "description": "What to extract (be specific and detailed)"
  },
  {
    "name": "AnotherConcept",
    "description": "Another piece of information to extract"
  }
]
```

**Example: Product Review Schema**
```json
[
  {
    "name": "ProductName",
    "description": "Name and model of the product being reviewed"
  },
  {
    "name": "OverallRating",
    "description": "Overall rating or score given to the product (e.g., 4/5 stars)"
  },
  {
    "name": "Pros",
    "description": "Positive aspects and advantages of the product"
  },
  {
    "name": "Cons",
    "description": "Negative aspects and disadvantages of the product"
  },
  {
    "name": "Recommendation",
    "description": "Reviewer's recommendation on whether to buy the product"
  }
]
```

### JSON Object Schema

Create a JSON file with schema definition:

```json
{
  "type": "json",
  "name": "SchemaName",
  "description": "What this schema extracts",
  "schema": {
    "field_name": {"type": "string"},
    "numeric_field": {"type": "number"},
    "array_field": {
      "type": "array",
      "items": {"type": "string"}
    },
    "nested_object": {
      "type": "object",
      "properties": {
        "sub_field": {"type": "string"}
      }
    }
  }
}
```

**Example: Event Information Schema**
```json
{
  "type": "json",
  "name": "EventDetails",
  "description": "Information about events, meetings, or conferences",
  "schema": {
    "event_name": {"type": "string"},
    "date": {"type": "string"},
    "location": {"type": "string"},
    "organizer": {"type": "string"},
    "attendees": {
      "type": "array",
      "items": {"type": "string"}
    },
    "agenda_items": {
      "type": "array",
      "items": {"type": "string"}
    },
    "key_decisions": {
      "type": "array",
      "items": {"type": "string"}
    },
    "action_items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "task": {"type": "string"},
          "assignee": {"type": "string"},
          "due_date": {"type": "string"}
        }
      }
    }
  }
}
```

### Schema Design Tips

1. **Be Specific:** Clear descriptions improve extraction quality
   - ❌ "Information about the company"
   - ✅ "Company revenue growth percentage for the reported quarter"

2. **Match Data Structure:** Use string concepts for free-form text, JSON schemas for structured data

3. **Test Iteratively:** Start with a few concepts, refine based on results

4. **Consider Token Costs:** More concepts = more tokens = higher cost

5. **Use Combine for Multi-Document:** Set `--extract-combine` when extracting from multiple related documents

---

## Best Practices

### 1. Optimize Search First

Good extraction starts with good search results:

```python
result = extractor.search_and_extract(
    query="specific, targeted query",
    top_k=5,  # Start small, increase if needed
    complexity=64,  # Higher for better accuracy
    metadata_filters={"type": {"==": "research_paper"}},  # Filter by type
)
```

### 2. Use Appropriate Combine Mode

**Individual Extraction (`combine_documents=False`):**
- Default mode
- Extracts from each document separately
- Best for: Multiple distinct items (e.g., multiple papers, emails)

**Combined Extraction (`combine_documents=True`):**
- Merges all documents first
- Single extraction from combined text
- Best for: Single entity across multiple chunks (e.g., multi-page contract)

### 3. Control Output Verbosity

```python
result = extractor.search_and_extract(
    query="...",
    concepts=[...],
    add_references=True,  # Include source chunk IDs
    add_justifications=True,  # Include LLM reasoning
)
```

**Production Mode:**
```python
add_references=False,
add_justifications=False,
```
Reduces token usage and speeds up extraction.

### 4. Handle Costs

```python
result = extractor.search_and_extract(...)

print(f"Tokens used: {result['total_tokens']}")
print(f"Estimated cost: ${result['total_cost']:.4f}")
```

**Cost Reduction:**
- Use smaller LLMs: `gpt-4o-mini` instead of `gpt-4o`
- Reduce `top_k` value
- Disable references and justifications
- Use local models (Ollama)

### 5. Save and Reuse Results

```python
import json

# Save results
with open("extractions.json", "w") as f:
    json.dump(result["extractions"], f, indent=2)

# Load and process later
with open("extractions.json") as f:
    extractions = json.load(f)
```

---

## Privacy & Cost

### Local Extraction (100% Private)

Use Ollama for completely local extraction:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull qwen2.5:7b

# Extract locally
python -m apps.document_rag \
  --docs ./sensitive_docs/ \
  --query "confidential information" \
  --extract \
  --extract-concepts "KeyPoints" "ActionItems" \
  --llm ollama \
  --llm-model qwen2.5:7b
```

**Supported Ollama Models:**
- `qwen2.5:7b` - Recommended for extraction
- `llama3.1:8b` - Good general-purpose
- `mistral:7b` - Fast and efficient
- `mixtral:8x7b` - High quality (requires more RAM)

### Cost Estimation

**OpenAI Pricing (as of 2025-01):**
- `gpt-4o-mini`: $0.15/1M input tokens, $0.60/1M output tokens
- `gpt-4o`: $2.50/1M input tokens, $10.00/1M output tokens

**Example Cost:**
```
Query: "What are the main findings?"
Top-K: 5 (5 chunks, ~500 tokens each)
Concepts: 5 string concepts

Input tokens: ~2,500 (documents) + ~200 (prompt) = 2,700
Output tokens: ~300 (extractions)

Cost (gpt-4o-mini): (2,700 * $0.15 + 300 * $0.60) / 1,000,000 = $0.0006
```

**Typical Costs:**
- Single extraction: $0.0005 - $0.002
- Batch processing (100 documents): $0.05 - $0.20
- Daily usage (50 extractions): $0.025 - $0.10

---

## Troubleshooting

### Import Error: No module named 'contextgem'

**Solution:** Install extraction dependencies
```bash
pip install leann[extract]
```

### Import Error: No module named 'litellm'

**Solution:** Install LiteLLM
```bash
pip install litellm>=1.0.0
```

### Extraction Quality Issues

**Problem:** Extracted information is inaccurate or incomplete

**Solutions:**
1. Improve concept descriptions (be more specific)
2. Increase `top_k` to retrieve more context
3. Increase `complexity` for better search accuracy
4. Use `combine_documents=True` for multi-chunk entities
5. Try a better LLM (gpt-4o instead of gpt-4o-mini)
6. Refine your search query to be more targeted

### High Token Usage

**Problem:** Extraction uses too many tokens

**Solutions:**
1. Reduce `top_k` value
2. Use shorter concept descriptions
3. Disable references: `add_references=False`
4. Disable justifications: `add_justifications=False`
5. Switch to smaller model (gpt-4o-mini)
6. Use local models (Ollama)

### Ollama Connection Issues

**Problem:** Cannot connect to Ollama server

**Solutions:**
```bash
# 1. Check Ollama is running
ollama list

# 2. Set host explicitly
export LEANN_OLLAMA_HOST=http://localhost:11434

# 3. Verify model is available
ollama pull qwen2.5:7b
```

### Empty Extractions

**Problem:** Extraction returns empty or null values

**Solutions:**
1. Verify search returns relevant results first
2. Check if the information exists in your documents
3. Improve search query specificity
4. Increase `top_k` to get more context
5. Try different concept descriptions

### GPU Memory Issues

**Problem:** CUDA out of memory during extraction

**Solutions:**
```bash
# Use specific GPU
--gpu-id 0

# Or use CPU for embedding
CUDA_VISIBLE_DEVICES="" python -m apps.document_rag ...
```

---

## Advanced Topics

### Multi-Language Extraction

ContextGem supports multiple languages:

```python
# Extract from Chinese documents
concepts = [
    StringConcept("研究方法", "论文中使用的研究方法和实验设计"),
    StringConcept("主要发现", "论文的主要研究发现和贡献"),
]

result = extractor.search_and_extract(
    query="深度学习",
    concepts=concepts,
    top_k=5
)
```

### Batch Extraction

Process multiple queries efficiently:

```python
queries = [
    "What are the main findings?",
    "What methodology was used?",
    "What are the limitations?",
]

all_results = []
for query in queries:
    result = extractor.search_and_extract(
        query=query,
        concepts=concepts,
        top_k=5
    )
    all_results.append(result)
```

### Custom LLM Configuration

Use specific LLM settings:

```python
extractor = LeannExtractor(
    index_path="my_index.leann",
    llm_config={
        "llm_type": "openai",
        "llm_model": "gpt-4o",
        "thinking_budget": 10000,  # For o3-mini reasoning
        "temperature": 0.0,  # Deterministic extraction
        "max_tokens": 2000,
    }
)
```

### Aspect-Based Extraction

Focus extraction on specific aspects:

```python
from contextgem import Aspect

aspects = [
    Aspect("Technical", "Focus on technical details and implementation"),
    Aspect("Business", "Focus on business impact and ROI"),
]

result = extractor.search_and_extract(
    query="system architecture",
    concepts=concepts,
    aspects=aspects,  # Extract from both perspectives
)
```

---

## API Reference

### LeannExtractor

```python
class LeannExtractor:
    def __init__(
        self,
        index_path: str,
        llm_config: Optional[dict[str, Any]] = None,
        searcher: Optional[Any] = None,
        gpu_id: Optional[int] = None,
        **searcher_kwargs
    )

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
        **search_kwargs
    ) -> dict[str, Any]

    def extract_from_results(
        self,
        results: List[Any],
        concepts: List[Any],
        aspects: Optional[List[Any]] = None,
        add_references: bool = True,
        add_justifications: bool = True,
        combine_documents: bool = False,
    ) -> List[dict[str, Any]]

    def cleanup(self) -> None
    def __enter__(self) -> "LeannExtractor"
    def __exit__(self, *args) -> None
```

### RAG App CLI Flags

```
--extract                       Enable extraction
--extract-concepts C1 C2 ...    Inline string concepts
--extract-schema FILE           JSON schema file path
--extract-combine               Combine all documents
--extract-no-references         Disable source references
--extract-no-justifications     Disable reasoning
--extract-output FILE           Save to JSON file
```

---

## Examples

See `examples/rag_extraction_example.py` for comprehensive examples:
- Basic extraction
- JSON schema extraction
- Combined document extraction
- Metadata filtering
- Local extraction with Ollama
- Programmatic usage

---

## Further Reading

- [ContextGem Documentation](https://deepwiki.com/shcherbak-ai/contextgem)
- [LEANN Paper](https://arxiv.org/abs/2506.08276)
- [LEANN README](../README.md)
- [Configuration Guide](configuration-guide.md)
- [Metadata Filtering](metadata_filtering.md)

---

## Support

- **Issues:** https://github.com/yichuan-w/LEANN/issues
- **Discussions:** https://github.com/yichuan-w/LEANN/discussions
- **Slack:** See README for invite link

---

**Happy Extracting! 🚀**
