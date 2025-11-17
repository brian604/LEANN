# LEANN Extraction Schemas

This directory contains pre-built extraction schemas for common use cases. These schemas can be used with LEANN RAG applications to extract structured data from documents.

## Available Schemas

### 1. Research Papers (`research_paper.json`)
Extract key information from academic papers:
- Methodologies and experimental designs
- Key findings and contributions
- Datasets used
- Limitations
- Future work directions

**Usage:**
```bash
python -m apps.document_rag \
  --query "What methodologies were used?" \
  --extract \
  --extract-schema examples/extraction_schemas/research_paper.json
```

### 2. Company Earnings Reports (`company_earnings.json`)
Extract financial metrics from earnings reports (JSON schema):
- Company name and quarter
- Revenue and growth
- Net income and EPS
- Key products
- Forward guidance
- Risk factors

**Usage:**
```bash
python -m apps.document_rag \
  --data-dir ./earnings_reports \
  --query "Q4 2024 financial performance" \
  --extract \
  --extract-schema examples/extraction_schemas/company_earnings.json \
  --extract-output results.json
```

### 3. Contract Terms (`contract_terms.json`)
Extract key terms from legal contracts (JSON schema):
- Parties involved
- Effective and termination dates
- Contract value
- Payment terms
- Obligations
- Termination clauses
- Confidentiality terms

**Usage:**
```bash
python -m apps.document_rag \
  --data-dir ./contracts \
  --query "SaaS subscription agreements" \
  --extract \
  --extract-schema examples/extraction_schemas/contract_terms.json \
  --extract-combine  # Combine all contracts for comparative analysis
```

### 4. Technical Documentation (`technical_doc.json`)
Extract technical information from documentation:
- API endpoints and methods
- Configuration parameters
- Dependencies and versions
- Installation steps
- Usage examples
- Troubleshooting guides

**Usage:**
```bash
python -m apps.document_rag \
  --data-dir ./docs \
  --query "API authentication setup" \
  --extract \
  --extract-schema examples/extraction_schemas/technical_doc.json
```

## Schema Format

### String Concept Format
For simple text extraction:
```json
[
  {
    "name": "ConceptName",
    "description": "Description of what to extract"
  }
]
```

### JSON Schema Format
For structured data extraction with validation:
```json
{
  "type": "json",
  "name": "SchemaName",
  "description": "Description of the structured data",
  "schema": {
    "field_name": {
      "type": "string|number|array|object",
      "description": "Field description"
    }
  }
}
```

## Custom Schemas

You can create custom schemas for your specific needs:

### Example: Product Reviews
```json
{
  "type": "json",
  "name": "ProductReview",
  "description": "Extract product review information",
  "schema": {
    "product_name": {"type": "string"},
    "rating": {"type": "number"},
    "pros": {"type": "array", "items": {"type": "string"}},
    "cons": {"type": "array", "items": {"type": "string"}},
    "recommendation": {"type": "string"}
  }
}
```

Save as `product_review.json` and use:
```bash
python -m apps.document_rag \
  --query "product reviews" \
  --extract \
  --extract-schema product_review.json
```

## CLI Options

### Extraction Flags
- `--extract`: Enable extraction mode
- `--extract-schema FILE`: Load schema from JSON file
- `--extract-concepts "Name:Description"`: Define concepts inline
- `--extract-combine`: Combine all results for holistic extraction
- `--extract-no-references`: Disable sentence-level references
- `--extract-no-justifications`: Disable LLM justifications
- `--extract-output FILE`: Save results to JSON file

### Example: Multiple Inline Concepts
```bash
python -m apps.document_rag \
  --query "AI developments" \
  --extract \
  --extract-concepts \
    "Companies:Company names" \
    "Technologies:AI technologies mentioned" \
    "Applications:Practical applications"
```

### Example: With Cost Control
```bash
# Use cheaper model for extraction
python -m apps.document_rag \
  --query "extract data" \
  --extract \
  --extract-schema schema.json \
  --llm openai \
  --llm-model gpt-4o-mini  # Cheaper model
  --top-k 5  # Limit documents to reduce cost
```

## Tips

1. **Start Small**: Test with `--top-k 2` first to verify extraction quality
2. **Choose Right Model**: Use `gpt-4o-mini` for simple extraction, `gpt-4o` for complex schemas
3. **Combine When Appropriate**: Use `--extract-combine` for comparative analysis
4. **Save Results**: Always use `--extract-output` to save structured data
5. **Cost Awareness**: Monitor `total_cost` in output, each extraction uses LLM tokens
6. **Privacy**: Use Ollama with `--llm ollama --llm-model qwen2.5:7b` for local extraction

## Privacy-Focused Local Extraction

For sensitive documents, use local LLMs with Ollama:

```bash
# First, pull a model
ollama pull qwen2.5:7b

# Run extraction locally
python -m apps.document_rag \
  --data-dir ./confidential \
  --query "sensitive information" \
  --extract \
  --extract-schema schema.json \
  --llm ollama \
  --llm-model qwen2.5:7b \
  --embedding-mode ollama \
  --embedding-model nomic-embed-text
```

All processing stays on your machine!

## Schema Design Best Practices

1. **Clear Descriptions**: Be specific about what each field should contain
2. **Appropriate Types**: Use correct JSON types (string, number, array, object)
3. **Nested Objects**: Use for complex related data (e.g., payment_terms)
4. **Arrays for Lists**: Use arrays when extracting multiple items
5. **Optional Fields**: Not all fields need to be extracted from every document

## Error Handling

If extraction fails:
1. Check that ContextGem is installed: `uv pip install leann-core[extract]`
2. Verify API keys are set: `echo $OPENAI_API_KEY`
3. Simplify schema if getting empty results
4. Increase `--top-k` if not finding relevant documents
5. Try different search queries

## Examples in Action

See `examples/extraction_demo.py` for complete working examples with these schemas.
