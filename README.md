# rag-with-sources

Retrieval-augmented question answering over technical documentation (a fictional heat pump manual)
that **cites its sources, verifies the citations and says "I don't know"** instead of guessing.

```
docs/*.md ─► section chunks ─► embeddings ─► Qdrant
question ─► retrieve top-k (score threshold) ─► LLM with numbered sources ─► citation check ─► answer + sources
                       └── Langfuse traces (optional)
```

## What it demonstrates

| Concern | How |
|---|---|
| Traceable answers | Every statement cites `[n]`; the response returns `file#section` for each cited source |
| No hallucinated citations | `validate_citations` rejects answers citing sources that were not retrieved |
| Knowing when not to answer | Score threshold + explicit refusal path; evaluated on unanswerable questions |
| Vector DB | Qdrant (in-memory for tests, server URL for production) |
| Observability | Langfuse `@observe` on retrieval and answering when `LANGFUSE_*` keys are set |
| Runs offline | Hashing embedder + extractive fallback → tests and evals need no API key; OpenAI embeddings / Claude for real use |
| Evals | `hit@3`, MRR and correct refusals (`python -m evals.run`), enforced in CI |

## Run

```bash
pip install -r requirements.txt
pytest -q
python -m evals.run          # hit@3 7/7 | MRR 1.00 | correct refusals 1/1 (offline embedder)
```

Use a real model: `pip install anthropic openai`, set `OPENAI_API_KEY` (embeddings) and
`ANTHROPIC_API_KEY`, then `answer(index, question, llm=anthropic_llm())`.
