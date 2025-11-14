# LEANN Codebase Guide for AI Assistants

**Last Updated:** 2025-11-14  
**Version:** 0.3.5  
**Purpose:** Comprehensive guide for AI assistants working on the LEANN codebase

---

## Project Overview

**LEANN** (Low-Storage Vector Index) is an innovative vector database that achieves **97% storage reduction** compared to traditional vector databases through graph-based selective recomputation. It enables personal AI assistants to run entirely on laptops with complete privacy.

**Key Innovation:** Instead of storing all embeddings, LEANN stores a pruned graph structure and recomputes embeddings on-demand during search, using high-degree preserving pruning to maintain accuracy.

**Repository:** https://github.com/yichuan-w/LEANN  
**Paper:** https://arxiv.org/abs/2506.08276

---

## 1. Directory Structure

```
LEANN/
├── apps/                    # RAG applications for various data sources
│   ├── base_rag_example.py # Base class for all RAG apps
│   ├── document_rag.py     # Generic document RAG (PDF, MD, TXT)
│   ├── code_rag.py         # Code-specific RAG with AST chunking
│   ├── email_rag.py        # Apple Mail RAG
│   ├── browser_rag.py      # Chrome browser history RAG
│   ├── wechat_rag.py       # WeChat messages RAG
│   ├── chatgpt_rag.py      # ChatGPT conversation RAG
│   ├── claude_rag.py       # Claude conversation RAG
│   ├── imessage_rag.py     # iMessage RAG
│   ├── slack_rag.py        # Slack MCP integration RAG
│   ├── twitter_rag.py      # Twitter bookmarks MCP RAG
│   └── */                  # Supporting modules for each app
│
├── packages/               # Monorepo package structure
│   ├── leann-core/        # Core API and plugin system
│   │   └── src/leann/
│   │       ├── __init__.py           # Package initialization, backend autodiscovery
│   │       ├── api.py                # LeannBuilder, LeannSearcher, LeannChat (1291 lines)
│   │       ├── cli.py                # CLI implementation (1658 lines)
│   │       ├── mcp.py                # MCP server for Claude Code integration
│   │       ├── chat.py               # LLM integration (OpenAI, Ollama, HF)
│   │       ├── embedding_compute.py  # Embedding computation (1122 lines)
│   │       ├── embedding_server_manager.py # Server lifecycle management
│   │       ├── chunking_utils.py     # Text chunking utilities
│   │       ├── metadata_filter.py    # Metadata filtering engine
│   │       ├── searcher_base.py      # Base searcher class
│   │       ├── interface.py          # Backend interface definitions
│   │       ├── registry.py           # Backend plugin registry
│   │       ├── settings.py           # Environment variable handling
│   │       └── interactive_utils.py  # Interactive session helpers
│   │
│   ├── leann-backend-hnsw/    # HNSW backend (FAISS-based)
│   │   ├── leann_backend_hnsw/
│   │   │   ├── __init__.py
│   │   │   ├── hnsw_backend.py        # HNSW builder and searcher
│   │   │   ├── hnsw_embedding_server.py # Embedding server for HNSW
│   │   │   └── convert_to_csr.py      # Graph pruning and CSR conversion
│   │   └── faiss/                     # Custom FAISS C++ code
│   │
│   ├── leann-backend-diskann/  # DiskANN backend with PQ support
│   │   ├── leann_backend_diskann/
│   │   │   ├── __init__.py
│   │   │   ├── diskann_backend.py     # DiskANN builder and searcher
│   │   │   ├── diskann_embedding_server.py # Embedding server for DiskANN
│   │   │   ├── graph_partition.py     # Graph partitioning utilities
│   │   │   └── embedding_pb2.py       # Protobuf definitions
│   │   └── third_party/DiskANN/       # DiskANN C++ library (submodule)
│   │
│   ├── leann/             # Meta package (combines all backends)
│   ├── astchunk-leann/    # AST-aware code chunking (empty placeholder)
│   └── wechat-exporter/   # WeChat data export tool
│
├── benchmarks/            # Performance benchmarking
│   ├── run_evaluation.py          # Main evaluation script
│   ├── compare_faiss_vs_leann.py  # FAISS vs LEANN comparison
│   ├── diskann_vs_hnsw_speed_comparison.py
│   ├── micro_tpt.py               # Throughput benchmarking
│   ├── llm_utils.py               # LLM evaluation utilities
│   └── */                         # Benchmark-specific data and scripts
│
├── tests/                # pytest test suite
│   ├── test_basic.py               # Basic functionality tests
│   ├── test_ci_minimal.py          # Minimal CI tests
│   ├── test_metadata_filtering.py  # Metadata filter tests
│   ├── test_mcp_integration.py     # MCP integration tests
│   ├── test_document_rag.py        # Document RAG tests
│   ├── test_cli_ask.py             # CLI tests
│   ├── test_astchunk_integration.py # AST chunking tests
│   └── test_*.py                   # Other test modules
│
├── docs/                 # Documentation
│   ├── CONTRIBUTING.md            # Contribution guidelines
│   ├── configuration-guide.md     # Configuration best practices
│   ├── metadata_filtering.md      # Metadata filtering guide
│   ├── grep_search.md             # Grep search feature
│   ├── ast_chunking_guide.md      # AST chunking documentation
│   ├── slack-setup-guide.md       # Slack MCP setup
│   ├── features.md                # Feature list
│   ├── faq.md                     # FAQ
│   └── roadmap.md                 # Project roadmap
│
├── examples/             # Example scripts
│   ├── basic_demo.py              # Simple usage example
│   ├── mcp_integration_demo.py    # MCP integration example
│   ├── grep_search_example.py     # Grep search example
│   ├── mlx_demo.py                # MLX backend example (Apple Silicon)
│   └── *.py                       # Other examples
│
├── scripts/              # Build and release automation
│   ├── build_and_test.sh          # Build and test script
│   ├── release.sh                 # Release automation
│   ├── bump_version.sh            # Version bumping
│   └── upload_to_pypi.sh          # PyPI upload
│
├── data/                 # Sample data for testing
│   ├── 2506.08276v1.pdf           # LEANN paper
│   ├── 2501.14312v1 (1).pdf       # Another paper
│   ├── PrideandPrejudice.txt      # Classic text
│   └── huawei_pangu.md            # Chinese document
│
├── .github/workflows/    # CI/CD configuration
│   ├── build-and-publish.yml      # Main CI workflow
│   ├── build-reusable.yml         # Reusable build workflow
│   ├── release-manual.yml         # Manual release trigger
│   └── link-check.yml             # Link validation
│
├── pyproject.toml        # Root project configuration
├── uv.lock              # Dependency lock file
├── .pre-commit-config.yaml # Pre-commit hooks
├── .python-version      # Python version (3.9)
├── llms.txt             # MCP integration metadata
├── demo.ipynb           # Jupyter demo notebook
├── README.md            # Main documentation
└── LICENSE              # MIT License
```

---

## 2. Core Architecture

### 2.1 Package Organization (Monorepo)

LEANN uses a **monorepo structure** with multiple editable packages:

1. **leann-core** (v0.3.5) - Core functionality
   - Main API: `LeannBuilder`, `LeannSearcher`, `LeannChat`
   - CLI implementation
   - MCP server
   - Embedding computation and server management
   - Metadata filtering
   - Entry points: `leann`, `leann_mcp`

2. **leann-backend-hnsw** (v0.3.5) - HNSW backend
   - Custom FAISS-based HNSW implementation
   - C++ extensions built with scikit-build-core
   - CSR graph storage with pruning
   - Embedding recomputation support

3. **leann-backend-diskann** (v0.3.5) - DiskANN backend
   - PQ-based graph traversal
   - Real-time reranking
   - C++ extensions via DiskANN library (submodule)
   - Superior search performance vs HNSW

4. **leann** (v0.3.5) - Meta package
   - Combines core + both backends
   - Main PyPI distribution

5. **astchunk-leann** - AST-aware code chunking
   - Preserves semantic boundaries (functions, classes)
   - Supports Python, Java, C#, TypeScript
   - Uses tree-sitter for parsing

### 2.2 Key Modules and Responsibilities

#### Core API (`packages/leann-core/src/leann/api.py`)
**Purpose:** Main entry point for building indexes and searching

**Key Classes:**
```python
class LeannBuilder:
    """Build LEANN indexes with minimal storage"""
    - add_text(text, metadata={}) -> str  # Add text chunk
    - add_texts(texts, metadatas=[]) -> list[str]  # Batch add
    - build_index(index_path, **kwargs)  # Build and save index
    
class LeannSearcher:
    """Search LEANN indexes with on-demand embedding recomputation"""
    - search(query, top_k=5, complexity=32, metadata_filters={}) -> list[SearchResult]
    - cleanup()  # Shutdown background servers
    
class LeannChat:
    """RAG-powered chat using LEANN search"""
    - ask(question, top_k=20, **kwargs) -> str  # Ask question with context
```

**Important Functions:**
- `compute_embeddings()` - Handles embedding computation with multiple backends
- `get_registered_backends()` - List available backends

#### CLI (`packages/leann-core/src/leann/cli.py`)
**Purpose:** Command-line interface for LEANN

**Commands:**
- `leann build <name> --docs <paths...>` - Build index from files/directories
- `leann search <name> <query>` - Search an index
- `leann ask <name> [--interactive]` - RAG Q&A
- `leann list` - List all indexes
- `leann remove <name>` - Remove an index

**Index Storage:** `.leann/indexes/` (project-local, like `.git`)

#### MCP Server (`packages/leann-core/src/leann/mcp.py`)
**Purpose:** Model Context Protocol server for Claude Code integration

**Tools:**
- `leann_search` - Semantic code search
- `leann_list` - List available indexes

**Protocol:** JSON-RPC 2.0, stdio-based

#### Backend Registry (`packages/leann-core/src/leann/registry.py`)
**Purpose:** Plugin system for vector index backends

**Key Functions:**
- `register_backend(name)` - Decorator to register backends
- `autodiscover_backends()` - Auto-import all `leann-backend-*` packages
- `register_project_directory()` - Register project for `leann list`

**Design Pattern:** Backends register themselves on import using `@register_backend` decorator

#### Embedding Computation (`packages/leann-core/src/leann/embedding_compute.py`)
**Purpose:** Compute embeddings with multiple backends

**Supported Modes:**
- `sentence-transformers` - Local models via sentence-transformers
- `openai` - OpenAI embedding API
- `ollama` - Ollama local inference
- `mlx` - Apple Silicon MLX backend

**Key Features:**
- Automatic batching and GPU utilization
- Token truncation handling
- Normalization for cosine similarity

#### Chat/LLM Integration (`packages/leann-core/src/leann/chat.py`)
**Purpose:** LLM backends for RAG generation

**Supported LLMs:**
- OpenAI (GPT-4, GPT-3.5, etc.)
- Ollama (local models)
- HuggingFace Transformers
- Simulated mode (for testing)

**Special Features:**
- Thinking budget support for reasoning models (o3, o3-mini)
- Automatic model availability checking
- Fuzzy model name matching

#### Metadata Filtering (`packages/leann-core/src/leann/metadata_filter.py`)
**Purpose:** Filter search results by metadata

**Operators:**
- Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Membership: `in`, `not_in`
- String: `contains`, `starts_with`, `ends_with`
- Boolean: `is_true`, `is_false`

### 2.3 Backend Architecture

Both backends implement the interface defined in `packages/leann-core/src/leann/interface.py`:

```python
class LeannBackendBuilderInterface(ABC):
    @abstractmethod
    def build(self, data: np.ndarray, ids: list[str], index_path: str, **kwargs)
    
class LeannBackendSearcherInterface(ABC):
    @abstractmethod
    def search(self, query: np.ndarray, top_k: int, complexity: int, 
               recompute_embeddings: bool, **kwargs) -> dict
    @abstractmethod
    def compute_query_embedding(self, query: str, **kwargs) -> np.ndarray
    
class LeannBackendFactoryInterface(ABC):
    @staticmethod
    def builder(**kwargs) -> LeannBackendBuilderInterface
    @staticmethod
    def searcher(index_path: str, **kwargs) -> LeannBackendSearcherInterface
```

**HNSW Backend Features:**
- FAISS-based HNSW graph construction
- High-degree preserving pruning
- CSR (Compressed Sparse Row) storage format
- Full embedding recomputation during search

**DiskANN Backend Features:**
- PQ-based approximate graph traversal
- Real-time reranking with fresh embeddings
- Beam search for better parallelism
- Superior search performance

---

## 3. Development Workflow

### 3.1 Build System

**Package Manager:** [uv](https://docs.astral.sh/uv/) (fast Python package manager)

**Build Commands:**
```bash
# Development setup (with DiskANN)
git clone https://github.com/yichuan-w/LEANN.git
cd LEANN
git submodule update --init --recursive
uv sync --extra diskann

# Install from PyPI (users)
uv venv
source .venv/bin/activate
uv pip install leann
```

**C++ Extensions:**
- Built with `scikit-build-core` + CMake
- HNSW: Custom FAISS with SWIG bindings
- DiskANN: pybind11 bindings to DiskANN C++ library

**System Dependencies:**
- macOS: `brew install libomp boost protobuf zeromq`
- Linux: `apt-get install libomp-dev libboost-all-dev protobuf-compiler libzmq3-dev libmkl-full-dev`

### 3.2 Testing Framework

**Test Runner:** pytest with plugins
- `pytest-cov` - Coverage reporting
- `pytest-xdist` - Parallel testing
- `pytest-timeout` - Timeout protection

**Test Organization:**
```
tests/
├── test_basic.py               # Basic functionality
├── test_ci_minimal.py          # Fast CI tests
├── test_metadata_filtering.py  # Metadata features
├── test_mcp_integration.py     # MCP server tests
├── test_document_rag.py        # Document RAG tests
└── test_*.py                   # Other modules
```

**Running Tests:**
```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_basic.py -v

# With coverage
pytest tests/ --cov=leann --cov-report=html

# Parallel execution
pytest tests/ -n auto
```

**CI Testing:**
- Multi-OS: Ubuntu (x86_64, ARM64), macOS (Intel, ARM64)
- Multi-Python: 3.9, 3.10, 3.11, 3.12, 3.13
- Arch Linux smoke test for compatibility

### 3.3 Code Quality Tools

**Linting and Formatting:** Ruff (v0.12.7 - fixed version)

**Configuration** (pyproject.toml):
```toml
[tool.ruff]
target-version = "py39"
line-length = 100

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "C4", "UP", "N", "RUF"]
ignore = ["E501", "B008", "B904", "N812", "N806", "RUF012"]
```

**Pre-commit Hooks:**
```bash
# Install hooks
uv run --only-group lint pre-commit install

# Run manually
uv run --only-group lint pre-commit run --all-files
```

**Hooks:**
- Trailing whitespace removal
- End-of-file fixer
- YAML validation
- Large file detection
- Merge conflict detection
- Debug statement detection
- Ruff linting and formatting

### 3.4 CI/CD Pipeline

**Workflow Files:**
1. `.github/workflows/build-and-publish.yml` - Main CI (triggers on push/PR)
2. `.github/workflows/build-reusable.yml` - Reusable build workflow
3. `.github/workflows/release-manual.yml` - Manual release trigger
4. `.github/workflows/link-check.yml` - Link validation

**Build Process:**
1. **Lint Stage** - Pre-commit checks on Python 3.11
2. **Build Stage** - Matrix build across OS/Python versions
   - Install system dependencies
   - Build C++ extensions
   - Create wheels
   - Repair wheels (auditwheel/delocate)
3. **Test Stage** - pytest on installed wheels
4. **Arch Smoke Test** - Verify installability on Arch Linux
5. **Artifact Upload** - Store wheels for release

**Release Process:**
```bash
# Automated via scripts/release.sh
scripts/bump_version.sh <new_version>
scripts/build_and_test.sh
scripts/upload_to_pypi.sh
```

---

## 4. Dependencies

### 4.1 Core Dependencies (leann-core)

**Vector/ML Libraries:**
- `numpy>=1.20.0` - Array operations
- `torch>=2.0.0` - Deep learning framework
- `sentence-transformers>=3.0.0` - Embedding models
- `transformers>=4.30.0,<4.46` - HuggingFace models (pinned for Py3.9)
- `accelerate>=0.20.0` - Model acceleration

**Document Processing:**
- `llama-index-core>=0.12.0` - Document loading
- `llama-index-readers-file>=0.4.0` - File readers
- `llama-index-embeddings-huggingface>=0.5.5` - Embeddings
- `PyPDF2>=3.0.0`, `pymupdf>=1.23.0`, `pdfplumber>=0.10.0` - PDF parsing
- `nbconvert>=7.0.0` - Jupyter notebook support
- `gitignore-parser>=0.1.12` - .gitignore handling

**LLM Integration:**
- `openai>=1.0.0` - OpenAI API
- `requests>=2.25.0` - HTTP requests

**System:**
- `tqdm>=4.60.0` - Progress bars
- `psutil>=5.8.0` - System monitoring
- `pyzmq>=23.0.0` - ZeroMQ messaging
- `msgpack>=1.0.0` - Binary serialization
- `python-dotenv>=1.0.0` - Environment variables

**Apple Silicon Only:**
- `mlx>=0.26.3` - MLX framework
- `mlx-lm>=0.26.0` - MLX language models

### 4.2 Backend Dependencies

**leann-backend-hnsw:**
- Custom FAISS build (C++)
- `numpy`, `pyzmq`, `msgpack`
- Build: `scikit-build-core`, `swig`

**leann-backend-diskann:**
- DiskANN library (C++ submodule)
- `numpy`, `protobuf>=3.19.0`
- Build: `scikit-build-core`, `pybind11`

### 4.3 Optional Dependencies

**DiskANN Backend:**
```bash
uv sync --extra diskann  # Or uv pip install leann-core[diskann]
```

**Document Processing:**
```bash
uv sync --extra documents
# Includes: beautifulsoup4, python-docx, openpyxl, pandas
```

**Colab Environment:**
```bash
uv pip install leann-core[colab]
# Pinned versions for Colab compatibility
```

### 4.4 Development Dependencies

**Dependency Groups** (pyproject.toml):
```toml
[dependency-groups]
lint = ["pre-commit>=3.5.0", "ruff==0.12.7"]
test = ["pytest>=7.0", "pytest-cov>=4.0", "pytest-xdist>=3.0", "pytest-timeout>=2.0"]
dev = ["matplotlib", "huggingface-hub>=0.20.0"]
```

---

## 5. Entry Points and Usage Patterns

### 5.1 Python API

**Basic Usage:**
```python
from leann import LeannBuilder, LeannSearcher, LeannChat

# Build index
builder = LeannBuilder(backend_name="hnsw")
builder.add_text("LEANN saves 97% storage")
builder.add_text("Graph-based recomputation")
builder.build_index("my_index.leann")

# Search
searcher = LeannSearcher("my_index.leann")
results = searcher.search("storage savings", top_k=5)

# Chat
chat = LeannChat("my_index.leann")
response = chat.ask("How much storage does LEANN save?")
```

**Advanced Features:**
```python
# Metadata filtering
builder.add_text("auth code", metadata={"lang": "python", "loc": 25})
results = searcher.search(
    "authentication",
    metadata_filters={"lang": {"==": "python"}, "loc": {"<": 100}}
)

# Grep search (exact matching)
results = searcher.search("banana-crocodile", use_grep=True)

# Backend selection
builder = LeannBuilder(backend_name="diskann", num_neighbors=64)

# Custom embedding models
builder = LeannBuilder(
    embedding_model="text-embedding-3-small",
    embedding_mode="openai"
)
```

### 5.2 CLI Interface

**Building Indexes:**
```bash
# From directory
leann build docs --docs ./documentation/

# Multiple sources
leann build codebase --docs ./src/ ./tests/ ./config/

# File type filtering
leann build slides --docs ./ --file-types .pptx,.pdf

# Custom settings
leann build large-corpus --docs ./data/ \
  --backend-name diskann \
  --embedding-model facebook/contriever \
  --graph-degree 64
```

**Searching:**
```bash
# Simple search
leann search docs "installation guide"

# Advanced search
leann search docs "error handling" \
  --top-k 10 \
  --complexity 64 \
  --show-metadata
```

**Interactive Chat:**
```bash
# Interactive mode (type 'quit' to exit)
leann ask docs --interactive

# Single question
leann ask docs "How do I configure the system?"
```

**Index Management:**
```bash
# List all indexes
leann list

# Remove index
leann remove docs
leann remove docs --force  # Skip confirmation
```

### 5.3 RAG Applications

**Structure:** All apps inherit from `BaseRAGExample` for consistent interface

**Common Parameters:**
```bash
--index-dir DIR              # Index storage location
--query "QUESTION"           # Single query (omit for interactive mode)
--max-items N                # Limit data processing
--force-rebuild              # Rebuild existing index
--embedding-model MODEL      # Embedding model selection
--embedding-mode MODE        # sentence-transformers, openai, mlx, ollama
--llm TYPE                   # openai, ollama, hf
--llm-model MODEL            # Model name
--top-k N                    # Search result count
--backend-name NAME          # hnsw or diskann
```

**Example Applications:**

1. **Document RAG:**
```bash
python -m apps.document_rag --query "What are LEANN's main techniques?"
```

2. **Code RAG:**
```bash
python -m apps.code_rag --repo-dir ./my_project \
  --query "How does authentication work?"
```

3. **Email RAG:**
```bash
python -m apps.email_rag --query "food orders from DoorDash"
```

4. **Browser History RAG:**
```bash
python -m apps.browser_rag --query "machine learning papers"
```

5. **MCP-Based RAG (Slack):**
```bash
python -m apps.slack_rag \
  --mcp-server "slack-mcp-server" \
  --workspace-name "my-team" \
  --channels general dev-team \
  --query "product launch decisions"
```

### 5.4 MCP Server for Claude Code

**Installation:**
```bash
# Install globally
uv tool install leann-core --with leann

# Register with Claude Code
claude mcp add --scope user leann-server -- leann_mcp

# Verify
claude mcp list | cat
```

**Usage in Claude Code:**
1. Build index: `leann build my-code --docs ./src/`
2. Claude Code automatically uses `leann_search` tool for semantic code search
3. Natural language queries: "How does authentication work?"

**MCP Tools:**
- `leann_search` - Semantic search with configurable top_k and complexity
- `leann_list` - List available indexes

---

## 6. Configuration and Settings

### 6.1 Environment Variables

**OpenAI:**
- `OPENAI_API_KEY` - API key
- `OPENAI_BASE_URL` - Custom endpoint URL

**Ollama:**
- `LEANN_OLLAMA_HOST` or `OLLAMA_HOST` - Server URL (default: http://localhost:11434)

**System:**
- `OMP_NUM_THREADS=1` - OpenMP threading (macOS)
- `TOKENIZERS_PARALLELISM=false` - Disable tokenizer parallelism
- `HF_HUB_DISABLE_SYMLINKS=1` - Disable symlinks (Windows compatibility)

**GPU Selection:**
- `LEANN_GPU_ID` - Specific GPU ID to use for embedding computation (0, 1, 2, ...)

### 6.2 GPU Selection (Multi-GPU Systems)

LEANN supports specifying which GPU to use for embedding computation on multi-GPU systems.

**Method 1: Command-Line**
```bash
# Use GPU 0
leann build docs --docs ./data/ --gpu-id 0

# Use GPU 1
leann search docs "query" --gpu-id 1
```

**Method 2: Environment Variable**
```bash
export LEANN_GPU_ID=1
leann build docs --docs ./data/  # Uses GPU 1
```

**Method 3: Python API**
```python
builder = LeannBuilder(backend_name="hnsw", gpu_id=0)
searcher = LeannSearcher("index.leann", gpu_id=1)
chat = LeannChat("index.leann", gpu_id=2)
```

**GPU Selection Priority:**
1. Explicit `gpu_id` parameter or `--gpu-id` flag
2. `LEANN_GPU_ID` environment variable
3. Auto-detection (first available GPU or MPS on Apple Silicon)

### 6.3 Backend Configuration

**HNSW Parameters:**
```python
builder = LeannBuilder(
    backend_name="hnsw",
    M=32,                    # Graph degree (higher = better accuracy, more memory)
    efConstruction=200,      # Build complexity
    is_compact=True,         # Use CSR storage
    is_recompute=True,       # Enable recomputation
    distance_metric="mips"   # mips, l2, or cosine
)
```

**DiskANN Parameters:**
```python
builder = LeannBuilder(
    backend_name="diskann",
    num_neighbors=64,        # Graph degree
    search_list_size=100,    # Build complexity
    distance_metric="mips"
)
```

**Search Parameters:**
```python
results = searcher.search(
    query="...",
    top_k=20,                # Number of results
    complexity=64,           # Search complexity (higher = more accurate, slower)
    beam_width=2,            # DiskANN: parallel search paths
    prune_ratio=0.5,         # DiskANN: PQ pruning ratio
    recompute_embeddings=True  # Enable recomputation
)
```

### 6.3 Chunking Configuration

**Document Chunking:**
```python
from llama_index.core.node_parser import SentenceSplitter

parser = SentenceSplitter(
    chunk_size=256,
    chunk_overlap=128,
    separator=" ",
    paragraph_separator="\n\n"
)
```

**Code Chunking (AST-aware):**
```python
# Automatic in code_rag.py
# Preserves function/class boundaries
# Supports: Python, Java, C#, TypeScript
```

---

## 7. Common Development Tasks

### 7.1 Adding a New Backend

1. Create package: `packages/leann-backend-mybackend/`
2. Implement interfaces from `leann.interface`:
   - `LeannBackendBuilderInterface`
   - `LeannBackendSearcherInterface`
   - `LeannBackendFactoryInterface`
3. Register backend:
```python
from leann.registry import register_backend

@register_backend("mybackend")
class MyBackend(LeannBackendFactoryInterface):
    @staticmethod
    def builder(**kwargs):
        return MyBackendBuilder(**kwargs)
    
    @staticmethod
    def searcher(index_path: str, **kwargs):
        return MyBackendSearcher(index_path, **kwargs)
```
4. Add to `pyproject.toml`:
```toml
[tool.uv.sources]
leann-backend-mybackend = { path = "packages/leann-backend-mybackend", editable = true }
```

### 7.2 Adding a New RAG Application

1. Create `apps/mydata_rag.py` inheriting from `BaseRAGExample`:
```python
from apps.base_rag_example import BaseRAGExample

class MyDataRAG(BaseRAGExample):
    def __init__(self):
        super().__init__(
            name="MyData RAG",
            description="RAG on my data source",
            default_index_name="mydata_index"
        )
    
    def add_custom_args(self, parser):
        # Add data source specific arguments
        parser.add_argument("--data-path", help="Path to data")
    
    def prepare_data(self, args, builder):
        # Load and chunk data
        # builder.add_text(text, metadata={...})
        pass

if __name__ == "__main__":
    app = MyDataRAG()
    app.run()
```

2. Create data reader in `apps/mydata_data/` if needed

### 7.3 Adding a New MCP Integration

1. Create MCP reader: `apps/myplatform_data/myplatform_mcp_reader.py`
2. Create RAG app: `apps/myplatform_rag.py`
3. Follow patterns from `apps/slack_rag.py` and `apps/slack_data/slack_mcp_reader.py`

**Key Components:**
- MCP client connection
- Data fetching via MCP tools
- Retry logic for async operations
- Connection testing (`--test-connection` flag)

### 7.4 Running Benchmarks

```bash
# Full evaluation (auto-downloads data)
uv run benchmarks/run_evaluation.py

# Specific benchmark
uv run benchmarks/compare_faiss_vs_leann.py

# DiskANN vs HNSW speed comparison
uv run benchmarks/diskann_vs_hnsw_speed_comparison.py
```

### 7.5 Building and Testing Locally

```bash
# Clean build from source
rm -rf .venv build/ dist/ packages/*/dist/
uv sync --extra diskann

# Run tests
pytest tests/ -v

# Build wheels
cd packages/leann-core && uv build && cd ../..
cd packages/leann-backend-hnsw && uv build && cd ../..
cd packages/leann-backend-diskann && uv build && cd ../..

# Test installation
uv venv test-env
source test-env/bin/activate
uv pip install packages/leann-core/dist/*.whl
uv pip install packages/leann-backend-hnsw/dist/*.whl
uv pip install packages/leann-backend-diskann/dist/*.whl
```

### 7.6 Releasing a New Version

```bash
# 1. Update version
./scripts/bump_version.sh 0.4.0

# 2. Build and test
./scripts/build_and_test.sh

# 3. Create release (manual via GitHub UI)
# 4. Upload to PyPI (automated via GitHub Actions)
```

---

## 8. Architecture Patterns and Best Practices

### 8.1 Plugin Architecture

**Backend Registry Pattern:**
- Backends auto-register on import
- Discovery via `importlib.metadata.distributions()`
- Zero-config plugin system
- Consistent interface via ABC

### 8.2 Embedding Server Architecture

**Design:**
- ZeroMQ-based client-server
- Persistent server for multiple queries
- Automatic lifecycle management
- Background process monitoring

**Benefits:**
- Model loaded once (expensive)
- Reused across multiple searches
- Automatic cleanup on exit

### 8.3 Metadata Storage

**Format:**
```json
{
  "embedding_model": "facebook/contriever",
  "embedding_mode": "sentence-transformers",
  "dimensions": 768,
  "backend_name": "hnsw",
  "backend_kwargs": {
    "distance_metric": "mips",
    "M": 32,
    "is_compact": true,
    "is_recompute": true
  },
  "chunk_count": 1000,
  "created_at": "2025-11-14T00:00:00"
}
```

**Location:** `<index_path>.meta.json`

### 8.4 Error Handling Patterns

**Graceful Degradation:**
- Fallback embedding backends
- Retry logic for MCP operations
- Model availability checking
- Fuzzy model name matching

**Cleanup:**
- Server shutdown on exceptions
- Temporary file cleanup
- Resource release in `finally` blocks

### 8.5 Testing Patterns

**Test Isolation:**
- Use `tempfile.TemporaryDirectory()` for indexes
- Mock external APIs (OpenAI, Ollama)
- Skip expensive tests in CI (`@pytest.mark.skipif`)

**Markers:**
```python
@pytest.mark.slow          # Long-running tests
@pytest.mark.openai        # Requires API key
```

---

## 9. Common Pitfalls and Solutions

### 9.1 Build Issues

**Problem:** C++ extension build failures  
**Solution:** 
- macOS: Install Homebrew dependencies
- Linux: Install system packages (MKL for x86_64, OpenBLAS for ARM64)
- Check CMake can find dependencies: `cmake --find-package ...`

**Problem:** Python 3.13 compatibility  
**Solution:** PyTorch 2.5+ required for Py3.13, not available on Intel macOS

### 9.2 Runtime Issues

**Problem:** "Embedding model not found in meta.json"  
**Solution:** Rebuild index with current version (old indexes missing metadata)

**Problem:** "users cache is not ready yet" (Slack MCP)  
**Solution:** Increase `--max-retries` and `--retry-delay`, wait for cache sync

**Problem:** MPS memory issues on macOS  
**Solution:** Set `PYTORCH_ENABLE_MPS_FALLBACK=0` or use CPU

### 9.3 Performance Issues

**Problem:** Slow embedding computation  
**Solution:**
- Use smaller models (e.g., `Qwen3-Embedding-0.6B-8bit`)
- Enable batching (automatic in `compute_embeddings`)
- Use MLX on Apple Silicon
- Use Ollama for local inference

**Problem:** Poor search quality  
**Solution:**
- Increase search complexity (default: 32 → 64+)
- Increase graph degree (M) during build
- Use better embedding models
- Tune chunking parameters

---

## 10. Key Files for AI Assistants

### 10.1 Must-Read Files

1. **README.md** - User-facing documentation, feature overview
2. **packages/leann-core/src/leann/api.py** - Core API implementation
3. **packages/leann-core/src/leann/interface.py** - Backend interfaces
4. **packages/leann-core/src/leann/registry.py** - Plugin system
5. **apps/base_rag_example.py** - RAG application base class
6. **docs/CONTRIBUTING.md** - Contribution guidelines

### 10.2 Architecture Reference Files

1. **packages/leann-backend-hnsw/leann_backend_hnsw/hnsw_backend.py** - HNSW implementation
2. **packages/leann-backend-diskann/leann_backend_diskann/diskann_backend.py** - DiskANN implementation
3. **packages/leann-core/src/leann/embedding_server_manager.py** - Server management
4. **packages/leann-core/src/leann/metadata_filter.py** - Filtering logic

### 10.3 Configuration Files

1. **pyproject.toml** - Root project config, dependencies
2. **packages/leann-core/pyproject.toml** - Core package config
3. **.pre-commit-config.yaml** - Code quality hooks
4. **.github/workflows/build-reusable.yml** - CI pipeline

---

## 11. Project Conventions

### 11.1 Code Style

- **Line length:** 100 characters
- **String quotes:** Double quotes
- **Imports:** Sorted by ruff (isort)
- **Type hints:** Preferred but not required for Python 3.9 compatibility
- **Docstrings:** Google style

### 11.2 Naming Conventions

- **Packages:** `leann-backend-<name>` for backends
- **Modules:** Snake_case (`embedding_compute.py`)
- **Classes:** PascalCase (`LeannBuilder`)
- **Functions:** Snake_case (`compute_embeddings`)
- **Constants:** UPPER_SNAKE_CASE (`BACKEND_REGISTRY`)

### 11.3 File Organization

- **Apps:** One file per data source (`<source>_rag.py`)
- **Data readers:** Subdirectory per source (`apps/<source>_data/`)
- **Tests:** Mirror source structure (`tests/test_<module>.py`)
- **Docs:** One file per feature (`docs/<feature>.md`)

### 11.4 Git Workflow

- **Branch:** Feature branches from main
- **Commits:** Conventional commits preferred
- **PR:** Required for all changes
- **CI:** Must pass before merge

---

## 12. External Resources

### 12.1 Documentation

- **GitHub:** https://github.com/yichuan-w/LEANN
- **Paper:** https://arxiv.org/abs/2506.08276
- **PyPI:** https://pypi.org/project/leann/
- **Slack:** Join via README link

### 12.2 Related Projects

- **FAISS:** https://github.com/facebookresearch/faiss
- **DiskANN:** https://github.com/microsoft/DiskANN
- **LlamaIndex:** https://github.com/run-llama/llama_index
- **uv:** https://docs.astral.sh/uv/

### 12.3 MCP Resources

- **MCP Specification:** https://modelcontextprotocol.io/
- **Claude Code:** https://www.anthropic.com/claude/code
- **Example Servers:** https://github.com/modelcontextprotocol/servers

---

## 13. Statistics

- **Total Python files:** 106
- **Core module lines:** 7,052 lines
- **Supported Python versions:** 3.9, 3.10, 3.11, 3.12, 3.13
- **Supported platforms:** Ubuntu (x86_64, ARM64), macOS (Intel, ARM64), Arch Linux
- **Test coverage:** Core functionality + integration tests
- **Build time:** ~5-10 minutes (with C++ compilation)
- **Storage savings:** 91-97% vs traditional vector DBs

---

## 14. Quick Reference Commands

```bash
# Development setup
git clone https://github.com/yichuan-w/LEANN.git && cd LEANN
git submodule update --init --recursive
uv sync --extra diskann

# Run tests
pytest tests/ -v

# Format code
uv run --only-group lint pre-commit run --all-files

# Build wheels
cd packages/leann-core && uv build && cd ../..

# Install globally
uv tool install leann-core --with leann

# MCP setup
claude mcp add --scope user leann-server -- leann_mcp

# Run examples
python -m apps.document_rag
python examples/basic_demo.py

# Benchmarks
uv run benchmarks/run_evaluation.py
```

---

**Document Version:** 1.0  
**Generated:** 2025-11-14  
**For:** AI assistants working on LEANN codebase

This guide provides comprehensive information for AI assistants to understand, navigate, and contribute to the LEANN codebase effectively.
