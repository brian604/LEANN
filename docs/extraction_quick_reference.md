# LEANN Extraction Quick Reference

## Installation

```bash
pip install leann[extract]  # or uv pip install leann[extract]
```

## Python API - 3 Lines to Extract

```python
from leann import LeannExtractor
from contextgem import StringConcept

extractor = LeannExtractor("my_index.leann")
concepts = [StringConcept("Finding", "Main research finding")]
result = extractor.search_and_extract("query", concepts, top_k=5)
```

## RAG App CLI - Add 2 Flags

```bash
# Inline concepts
python -m apps.document_rag --query "..." --extract \
  --extract-concepts "Finding1" "Finding2" "Finding3"

# JSON schema
python -m apps.document_rag --query "..." --extract \
  --extract-schema examples/extraction_schemas/research_paper.json
```

## Pre-Built Schemas

| Schema | Use Case | Type | Fields/Concepts |
|--------|----------|------|-----------------|
| `research_paper.json` | Academic papers | String | 5 concepts (Methodologies, KeyFindings, Datasets, Limitations, FutureWork) |
| `company_earnings.json` | Financial reports | JSON | 9 fields (company_name, quarter, revenue, growth, etc.) |
| `contract_terms.json` | Legal contracts | JSON | 8 fields (parties, dates, payment_terms, obligations, etc.) |
| `technical_doc.json` | Tech documentation | String | 6 concepts (APIs, Configuration, Dependencies, etc.) |

## Concept Types

### String Concepts (Free-Form Text)
```python
from contextgem import StringConcept

concepts = [
    StringConcept("ConceptName", "Detailed description of what to extract"),
    StringConcept("AnotherConcept", "Another piece of information"),
]
```

### JSON Object Concepts (Structured Data)
```python
from contextgem import JsonObjectConcept

concept = JsonObjectConcept(
    name="MetricsData",
    description="Financial metrics from the report",
    schema={
        "revenue": {"type": "string"},
        "growth": {"type": "string"},
        "products": {"type": "array", "items": {"type": "string"}},
    }
)
```

## CLI Flags Reference

| Flag | Description | Example |
|------|-------------|---------|
| `--extract` | Enable extraction | `--extract` |
| `--extract-concepts C1 C2...` | Inline string concepts | `--extract-concepts "Finding" "Method"` |
| `--extract-schema FILE` | Load schema from JSON | `--extract-schema schema.json` |
| `--extract-combine` | Combine docs before extraction | `--extract-combine` |
| `--extract-no-references` | Disable source references | `--extract-no-references` |
| `--extract-no-justifications` | Disable reasoning | `--extract-no-justifications` |
| `--extract-output FILE` | Save to JSON file | `--extract-output results.json` |

## Common Use Cases

### Extract from Research Papers
```bash
python -m apps.document_rag --docs ./papers/ \
  --query "machine learning techniques" \
  --extract \
  --extract-schema examples/extraction_schemas/research_paper.json
```

### Extract from Financial Reports
```bash
python -m apps.document_rag --docs ./earnings/ \
  --query "Q4 2024 results" \
  --extract \
  --extract-schema examples/extraction_schemas/company_earnings.json \
  --extract-combine
```

### Extract from Emails
```bash
python -m apps.email_rag --query "project deadlines" \
  --extract \
  --extract-concepts "ProjectName" "Deadline" "Owner" "Status"
```

### Extract from Code
```bash
python -m apps.code_rag --repo-dir ./myproject \
  --query "authentication flow" \
  --extract \
  --extract-concepts "EntryPoint" "Dependencies" "SecurityMeasures"
```

## Privacy Mode (Local LLMs)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b

# Extract locally (100% private)
python -m apps.document_rag --query "..." --extract \
  --extract-concepts "..." \
  --llm ollama --llm-model qwen2.5:7b
```

## Create Custom Schema

**String Concepts:**
```json
[
  {
    "name": "ConceptName",
    "description": "What to extract (be specific!)"
  }
]
```

**JSON Schema:**
```json
{
  "type": "json",
  "name": "SchemaName",
  "description": "What this schema extracts",
  "schema": {
    "field1": {"type": "string"},
    "field2": {"type": "number"},
    "field3": {"type": "array", "items": {"type": "string"}}
  }
}
```

## Python API Reference

### LeannExtractor

```python
# Initialize
extractor = LeannExtractor(
    index_path="my_index.leann",
    llm_config={
        "llm_type": "openai",  # or "ollama", "hf"
        "llm_model": "gpt-4o-mini",
    },
    gpu_id=0,  # Optional GPU selection
)

# Extract
result = extractor.search_and_extract(
    query="Your query",
    concepts=[...],
    top_k=5,
    complexity=64,
    add_references=True,
    add_justifications=True,
    combine_documents=False,
)

# Result structure
{
    "search_results": [...],
    "extractions": [
        {
            "concept": "ConceptName",
            "value": "Extracted value",
            "reference": "Chunk 3",
            "justification": "Reasoning...",
            "metadata": {...},
        }
    ],
    "total_tokens": 1234,
    "total_cost": 0.0056,
}

# Cleanup
extractor.cleanup()

# Or use context manager
with LeannExtractor("my_index.leann") as extractor:
    result = extractor.search_and_extract(...)
```

## Cost Estimation

**Typical costs with gpt-4o-mini ($0.15/1M input, $0.60/1M output):**
- Single extraction: $0.0005 - $0.002
- Batch (100 docs): $0.05 - $0.20
- Daily usage (50 queries): $0.025 - $0.10

**Cost reduction:**
- Use `gpt-4o-mini` instead of `gpt-4o`
- Reduce `top_k` value
- Disable references/justifications
- Use local models (Ollama) - **$0.00**

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `No module named 'contextgem'` | `pip install leann[extract]` |
| Poor extraction quality | Improve concept descriptions, increase top_k, use better LLM |
| High token usage | Reduce top_k, disable references/justifications, use smaller model |
| Empty extractions | Verify search results first, improve query, increase top_k |
| Ollama connection issues | Check `ollama list`, set `LEANN_OLLAMA_HOST` |

## Full Documentation

- [Complete Extraction Guide](extraction_guide.md)
- [Pre-Built Schemas Guide](../examples/extraction_schemas/README.md)
- [RAG Extraction Examples](../examples/rag_extraction_example.py)
- [ContextGem Documentation](https://deepwiki.com/shcherbak-ai/contextgem)

---

**Quick Start in 30 Seconds:**

```bash
# 1. Install
pip install leann[extract]

# 2. Build index
leann build docs --docs ./my_documents/

# 3. Extract
python -m apps.document_rag --query "key findings" --extract \
  --extract-schema examples/extraction_schemas/research_paper.json
```

**Done!** 🚀
