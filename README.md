\# Telecom RAG Agent

A domain-focused RAG QA agent for communication engineering materials, especially 3GPP specifications.



\## Project Overview

This RAG agent is tailored for 3GPP technical documents. It implements clause-aware chunking to preserve specification hierarchy, metadata filtering for clause reference queries, and source verification to reduce LLM hallucination.



\## Tech Stack

LangChain, Chroma, OpenAI API, Streamlit, pypdf



\## Setup

1\. Create python virtual environment

```



python -m venv venv

venv\\Scripts\\activate

pip install -r requirements.txt



```

2\. Copy `.env.example` to `.env` and fill your OpenAI key

3\. Put 3GPP PDF specs into `data/raw/`

4\. Run app:

```



streamlit run app.py



```



```

