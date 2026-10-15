# Telecom RAG Agent

A domain-focused RAG QA agent for communication engineering materials, including 6G survey papers and 3GPP specifications.

## Project Overview

This RAG agent is tailored for communication engineering documents. It implements section-aware chunking to preserve document hierarchy, metadata filtering for section reference queries, and source verification to reduce LLM hallucination.

## Tech Stack

- **Vector store**: Chroma
- **LLM**: DeepSeek API (`deepseek-chat`)
- **Embedding**: Sentence-Transformer (`BAAI/bge-small-zh-v1.5`)
- **Interface**: Streamlit
- **PDF parsing**: pypdf

## Evaluation

Test set: 8 domain-specific queries from a 6G spectrum management survey paper (6 retrieval questions + 2 hallucination tests).

Metric: Recall@5 (whether the expected section appears in top-5 retrieved chunks) and hallucination refusal rate.

| Version | Embedding Model | Recall@5 | Refusal Rate |
|---|---|---|---|
| v1 | all-MiniLM-L6-v2 (English) | 0.50 | 0.50 |
| v2 | BAAI/bge-small-zh-v1.5 (Chinese) | **0.67** | **1.00** |

Key fixes in v2:
- Switched to a Chinese-optimized embedding model for Chinese technical papers
- Filtered false section headers (e.g., years like "2030" misidentified as section IDs)
- Tightened the system prompt to refuse unsupported questions

Remaining failure mode: open-ended questions ("what are the limitations") where the answer is distributed across multiple sections.

## Setup

1. Create a Python virtual environment:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and fill in your `DEEPSEEK_API_KEY`.

3. Put your PDF files into `data/pdf/`.

4. Build the vector database:

```
python -c "from pathlib import Path; from src.retrieval.indexer import build_vector_db; build_vector_db(Path('data/pdf'))"
```

5. Run the app:

```
streamlit run app.py
```

## Evaluation Script

To reproduce the evaluation results:

```
python evaluate_rag.py
```

This outputs `rag_eval_result.csv` and `rag_eval_result.txt`.