# rag-docs

Async document RAG pipeline: ingest PDFs → queue → workers (parse/chunk/embed) →
Postgres + pgvector → query API with source citations.

Built to show production-shaped engineering around an LLM, not a notebook demo:
distributed processing, durable storage, and answers traced back to the exact
page they came from.

## Stack
- **FastAPI** — upload + query
- **RabbitMQ** — async ingestion queue
- **Postgres + pgvector** — chunks & embeddings, cosine search (no separate vector DB)
- **Redis** — embedding cache / dedup
- **OpenAI** — `text-embedding-3-small` + chat
- **Docker Compose** — one command brings it all up

## Run
```bash
export OPENAI_API_KEY=sk-...
docker compose up
```

## Status
- [x] `docker-compose.yml`, `schema.sql`
- [x] `worker/pipeline.py` — parse / chunk / embed (self-checked, no network)
- [ ] `worker/worker.py` — consume queue, persist
- [ ] `api/main.py` — `POST /documents`, `POST /query` with citations
- [ ] Dockerfiles + requirements

## Architecture
```
upload ──> API ──> RabbitMQ ──> worker ──> [parse → chunk → embed] ──> Postgres(pgvector)
                                                                            │
query ──> API ── embed(question) ── cosine search ─────────────────────────┘
              └─> LLM answer + citations (document, page)
```
