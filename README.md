# codebase-copilot
# Local Codebase Copilot

A local-first AI assistant that understands your GitHub repository. Ask natural-language questions, get answers grounded in your actual code — no cloud, no API keys, no data leaving your machine.

> "Where is authentication handled?" → cited file paths + line ranges + explanation.
> "Find dead code." → list of unreferenced symbols with confidence scores.
> "Suggest tests for `Chunker._split_oversized`." → ready-to-paste pytest cases.

Built with **Ollama** (LLM), **FastAPI** (backend), **Tree-sitter** (parsing), **Chroma** (vectors), and a **VS Code** extension.

---

## Why this exists

LLM coding assistants today are great at writing snippets but bad at reasoning over *your* repo. They hallucinate file paths, invent functions, and miss the architectural context that actually matters. This project is a small, modular reference implementation of a code-aware agent that:

1. **Indexes your repo** with Tree-sitter (function/class boundaries, not arbitrary 500-char windows).
2. **Retrieves** with hybrid vector + BM25 search and reciprocal rank fusion.
3. **Reasons** through a tool-calling agent loop — `search_code`, `open_file`, `run_tests`, `static_analysis`, `get_dependency_graph`, plus composed tools like `find_dead_code` and `suggest_tests`.
4. **Grounds** every answer in cited `path:line` ranges. Strict prompt refuses when grounding is weak.

It runs on a laptop. The default config uses `qwen2.5-coder:7b` + `nomic-embed-text` — about 5 GB of model weights total.

---

## Architecture

```
┌──────────────────┐    HTTP/SSE    ┌──────────────────────────────────────┐
│  VS Code         │ ─────────────▶ │  FastAPI server (localhost:8000)     │
│  extension       │                │                                      │
│  (chat panel,    │ ◀───────────── │  /index  /search  /chat  /tools/call │
│   commands)      │                └──────────────┬───────────────────────┘
└──────────────────┘                               │
                                                   ▼
                              ┌────────────────────┴────────────────────┐
                              │                                         │
                              ▼                                         ▼
              ┌──────────────────────────┐           ┌─────────────────────────────┐
              │  Indexing pipeline       │           │  Agent (ReAct loop)         │
              │  ─ Tree-sitter parser    │           │  ─ Strict-grounding prompt  │
              │  ─ Symbol-aware chunker  │           │  ─ Tool registry dispatch   │
              │  ─ Ollama embeddings     │           │  ─ Cited answer assembly    │
              │  ─ Incremental (sha1)    │           └────────────────┬────────────┘
              └────────────┬─────────────┘                            │
                           │                                          │
                           ▼                                          ▼
            ┌────────────────────────────┐         ┌──────────────────────────────┐
            │  Chroma (vectors)          │ ◀────── │  Hybrid retriever            │
            │  + SQLite metadata         │         │  vector + BM25, RRF fusion   │
            └────────────────────────────┘         └──────────────────────────────┘
                                                                      │
                                                                      ▼
                                                        ┌──────────────────────────┐
                                                        │  Ollama (localhost:11434)│
                                                        │  qwen2.5-coder:7b        │
                                                        │  nomic-embed-text        │
                                                        └──────────────────────────┘
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for class diagrams, data-flow traces, and the model/prompt comparison that produced the default config.

---

## Quickstart

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- ~5 GB free disk for default models
- Node 18+ (only if you want the VS Code extension)

### 2. Install

```bash
git clone https://github.com/<you>/codebase-copilot
cd codebase-copilot
bash scripts/setup.sh        # pulls Ollama models + installs Python deps
```

Or manually:

```bash
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
pip install -e .
```

### 3. Index a repo

```bash
copilot index /path/to/your/repo
```

First run takes a few minutes for a medium repo (10k-50k LOC). Subsequent runs are incremental — only changed files are re-embedded.

### 4. Ask questions

```bash
copilot chat /path/to/your/repo "Where is authentication handled?"
```

Or run the server and use the VS Code extension:

```bash
bash scripts/run.sh
# then in VS Code: Cmd+Shift+P → "Codebase Copilot: Open Chat"
```

---

## Default configuration

After running the eval harness on the included sample cases, the recommended default is:

| Setting              | Value                  | Why                                                      |
|----------------------|------------------------|----------------------------------------------------------|
| LLM                  | `qwen2.5-coder:7b`     | Best small model for code + reliable tool calling        |
| Embeddings           | `nomic-embed-text`     | 768-dim, fast, strong on code/symbol semantics           |
| System prompt        | `strict` (P3)          | Forces path:line citations, refuses on weak grounding    |
| Temperature          | `0.2`                  | Deterministic enough for code, leaves room for synthesis |
| `num_ctx`            | `8192`                 | Fits ~6 retrieved chunks + tool outputs comfortably      |
| Retrieval `k`        | `16` → rerank to `6`   | High recall first, then precision                        |
| BM25 weight (RRF)    | `0.4`                  | Vector dominates, but BM25 catches exact symbol hits     |
| Max agent steps      | `6`                    | Prevents tool-loop runaway                               |

Other configs benchmarked in [`docs/ARCHITECTURE.md#model--prompt-comparison`](docs/ARCHITECTURE.md):

- **Laptop / CPU-only:** `phi3:mini` + minimal prompt + `num_ctx=4096`
- **Workstation:** `deepseek-coder-v2:16b` + strict prompt + `num_ctx=16384`

Run your own benchmark on your repo:

```bash
copilot eval /path/to/repo --cases app/eval/sample_cases.json
```

---

## CLI

```
copilot index <path>                    # build / update index
copilot search <path> "query"           # raw retrieval, no LLM
copilot chat <path> "question"          # full agent loop
copilot serve                           # start FastAPI on :8000
copilot eval <path> --cases <file>      # run benchmark suite
```

All commands accept `--model`, `--prompt`, `--temperature` to override defaults.

---

## API

`POST /api/index` — index or refresh a repo
`POST /api/search` — hybrid retrieval (no LLM)
`POST /api/chat` — agent loop, optional SSE streaming of intermediate steps
`GET /api/tools` — list available tools and schemas
`POST /api/tools/call` — invoke a single tool directly (debugging)
`GET /api/health` — Ollama reachability + index stats

OpenAPI docs at `http://localhost:8000/docs` once running.

---

## Tools available to the agent

| Tool                    | What it does                                                         |
|-------------------------|----------------------------------------------------------------------|
| `search_code`           | Hybrid retrieval over indexed chunks                                 |
| `open_file`             | Read a file (or line range) with traversal protection                |
| `run_tests`             | Run pytest / unittest, return summary + failures                     |
| `static_analysis`       | Run `ruff` and return JSON diagnostics                               |
| `get_dependency_graph`  | Build identifier-level call graph for a module/symbol                |
| `find_dead_code`        | Composed: graph + reachability ⇒ unreferenced public symbols         |
| `suggest_tests`         | Composed: open_file + analysis ⇒ pytest skeletons                    |
| `propose_patch`         | Composed: search + open_file + ruff ⇒ unified diff suggestion        |

Each tool returns a JSON envelope `{ok: bool, data: ..., error?: str}` so the agent can recover from failures.

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check app/ tests/
```

Tests do not require a running Ollama instance — `tests/test_agent.py` uses a `FakeLLM`, and `tests/test_chunker.py` exercises Tree-sitter directly.

CI runs on every push (see `.github/workflows/ci.yml`).

---

## Project layout

```
codebase-copilot/
├── app/
│   ├── api/          # FastAPI routes, schemas, app state
│   ├── agent/        # ReAct loop + tool dispatch
│   ├── indexing/     # parser → chunker → embedder → vector store
│   ├── retrieval/    # hybrid vector + BM25 with RRF
│   ├── llm/          # Ollama client + system prompts
│   ├── tools/        # search_code, open_file, run_tests, ...
│   └── eval/         # benchmark harness + sample cases
├── cli/              # Typer-based CLI (entry point: `copilot`)
├── extension/        # VS Code extension (TypeScript)
├── tests/            # pytest suites (no network deps)
├── docs/             # ARCHITECTURE.md and design notes
└── scripts/          # setup.sh, run.sh
```

---

## Optional improvements (roadmap)

- **FAISS backend.** `VectorStore` is an ABC; `FaissStore` would only need to implement `add_chunks`, `query`, `delete_by_path`. Useful if you outgrow Chroma's per-collection overhead at >1M chunks.
- **Smarter incremental indexing.** Currently SHA1-per-file; could move to AST-diff per symbol so editing one function doesn't re-embed the whole file.
- **Embedding cache.** Memoize `OllamaEmbedder.embed()` by chunk content hash — saves ~30% on re-indexes after rebases.
- **BM25 persistence.** Rebuild on every server start today; pickle the corpus for faster cold start.
- **Multi-repo workspaces.** One Chroma collection per repo (already namespaced by `repo_id`); add a UI selector.
- **Streaming patches.** SSE-stream `propose_patch` so the user sees the diff being constructed.
- **Reranker model.** Drop a small cross-encoder (e.g. `bge-reranker-base` via Ollama-compatible endpoint) between retrieval and the agent.

See [`docs/ARCHITECTURE.md#optional-improvements`](docs/ARCHITECTURE.md) for tradeoffs.

---

## License

MIT — see [LICENSE](LICENSE).