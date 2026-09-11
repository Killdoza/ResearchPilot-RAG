# ResearchPilot-RAG

<p align="center">
  <img src="resources/full_logo.png" alt="ResearchPilot-RAG" width="520">
</p>

[![CI](https://github.com/Killdoza/ResearchPilot-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/Killdoza/ResearchPilot-RAG/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/Killdoza/ResearchPilot-RAG)](https://github.com/Killdoza/ResearchPilot-RAG/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[简体中文](README.md) | English

ResearchPilot is an evaluation-driven, traceable RAG system for reading AI and computer-science papers. It combines structure-aware parsing, hybrid retrieval, page-level citations, answer contracts, reproducible ablations, a FastAPI service, a Streamlit workspace, Docker, CI, and a lightweight research-note workflow.

> The current public version is `v1.1.0`, based on the internally validated `product-rc2` milestone. It is a local research tool and reproducible portfolio project—not a production multi-tenant SaaS with authentication or an SLA.

## What makes it more than a ChatPDF demo

- Docling-based structured parsing with page, section, block-type, and extraction-quality metadata;
- structure-aware Parent–Child chunks and Dense + persistent BM25 + RRF + CrossEncoder retrieval;
- explicit routing for metadata, summaries, experimental setup/results, multi-section questions, detailed QA, general chat, and cross-document requests;
- normalized citations, claim/citation boundaries, numeric-faithfulness checks, structured-table replay, summary validation, and deterministic full-document field scans;
- paper-scoped working memory, long-term conclusions, and user annotations;
- direct PDF-page inspection, research notes, and Markdown export;
- E0–E5 retrieval ablations, long-context comparisons, and a query-aware experiment designed to prevent Oracle leakage;
- a shared service core used by Streamlit and FastAPI, plus CPU Docker Compose, redacted traces, GitHub Actions, and 274 automated tests.

## Verified results

| Evaluation | Result | Scope |
| --- | ---: | --- |
| E5 test MRR | 0.6767 | paper-held-out retrieval test |
| E5 test Recall@5 | 0.8000 | E4 fixed-chunk baseline: 0.5500 |
| E5 test nDCG@5 | 0.6744 | evidence-page ranking |
| A0 single-paper Citation F1 | 0.6067 | answer/citation evaluation |
| `product-rc2` regression | 18/18 | frozen real-product cases only |
| expected-page coverage | 36/42 (85.7%) | six expected pages remained missing |
| automated tests | 274 passed | code contracts, not semantic accuracy |

The P2 query-aware/evidence-constrained strategy improved source boundaries and prevented unsupported comparisons, but did **not** establish an overall Citation-F1 gain. Strict cross-document comparison therefore remains disabled by default.

## Architecture

```text
Streamlit / FastAPI
        │
RuntimeOrchestrator
  ├── product_rules_v2
  ├── paper-scoped memory
  ├── annotation policy
  └── retry / refusal boundary
        │
ResearchPilotService
  ├── parsing and ingestion
  ├── scoped Chroma + persistent BM25
  ├── task-aware hybrid retrieval
  └── evidence-grounded generation and validation
```

## Quick start on Windows

Requirements: Python 3.13 and Git. The first real run downloads local parsing, embedding, and reranking models.

```powershell
git clone https://github.com/Killdoza/ResearchPilot-RAG.git
cd ResearchPilot_RAG
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
# Put DEEPSEEK_API_KEY in the local .env file. Never commit it.

$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Start the API instead:

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m uvicorn researchpilot.api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`.

## Docker

```powershell
Copy-Item .env.example .env
docker compose up --build
```

The CPU Compose path has been built and smoke-tested on Windows with WSL2. See [`docs/deployment_linux.md`](docs/deployment_linux.md) for boundaries and troubleshooting.

## Quality gates

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff format --check .
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m compileall -q src scripts streamlit_app.py
.venv\Scripts\python.exe scripts\check_no_secrets.py
.venv\Scripts\python.exe -m pip check
```

## Known limitations

- Text PDFs are the supported path; OCR for scanned papers and visual understanding of tables, plots, diagrams, and workflows are not productized.
- Strict cross-document comparison is experimental and disabled by default.
- Claim/Citation v1 validates citation scope, structure, and numbers; it is not a full natural-language entailment prover.
- There is no public authentication, authorization, rate limiting, async job queue, multi-tenant governance, or production SLA.
- Raw PDFs, model caches, and runtime databases are not distributed with the repository.

See the [documentation map](docs/README.md), [v1.1.0 release notes](docs/release_v1.1.0.md), and [contribution guide](CONTRIBUTING.md).

## License

Project code is released under the [MIT License](LICENSE). Papers, datasets, and third-party models retain their respective licenses and terms.
