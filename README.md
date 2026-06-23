# 🧠 Self-Healing RAG Pipeline

> AI that detects its own weak answers, re-queries with smarter strategies, and heals itself — in real time.

**Day 7 of 7 — AI Build Series by Vaishnavi Gahoi**

## What makes this different from normal RAG?

| Normal RAG | Self-Healing RAG |
|---|---|
| Returns answer regardless of quality | Evaluates its own confidence (0-10) |
| Fails silently | Detects weak answers before returning |
| Fixed retrieval strategy | Re-queries with rephrase / expand / decompose strategies |
| Hallucination-prone | Reduces hallucination by verifying context relevance |

## How it works

```
Question → Retrieve → Generate → Evaluate Confidence
                                        ↓
                               Below threshold?
                                YES ↓        NO ↓
                          Re-query (heal)   Return answer
                                ↓
                         New retrieval → New answer → Return
```

## Self-healing strategies

- **Rephrase** — rewrites the question with different keywords
- **Expand** — broadens the query to find related content  
- **Decompose** — extracts the single most important concept

## Stack

- Python 3.11
- Streamlit (UI)
- Groq API — llama-3.1-8b-instant (LLM)
- TF-IDF cosine similarity (vector store — no external DB needed)
- 100% free

## Run locally

```bash
pip install -r requirements.txt
# Add your Groq key to .env
cp .env.example .env
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push to GitHub
2. Go to share.streamlit.io
3. Add `GROQ_API_KEY` in Secrets
4. Deploy
