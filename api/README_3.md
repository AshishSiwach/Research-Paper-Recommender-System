# ArXiv Recommender — REST API

FastAPI + Docker wrapper for the embedding-based ArXiv paper recommender system.
136,238 papers · all-MiniLM-L6-v2 · cosine similarity · KeyBERT keyword extraction

## Setup

### 1. Prepare data (run once)

Copy `arXiv_scientific_dataset.csv` into a `data/` folder, then:

```bash
pip install pandas sentence-transformers torch numpy
python prepare_api_data.py
```

Encodes all 136k papers and saves `data/arxiv_embeddings.pkl` (~2 min on GPU, ~12 min on CPU).

### 2. Build and run

```bash
docker compose up --build
```

API starts at **http://localhost:8000**. First startup loads the model and embeddings (~30s).

### 3. Try it

```bash
# Health check
curl http://localhost:8000/health

# Get recommendations
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "transformer models for information retrieval", "top_k": 5}'
```

Or open **http://localhost:8000/docs** for the interactive Swagger UI.

## API

### `POST /recommend`

Request:
```json
{ "query": "neural networks for text classification", "top_k": 5 }
```

Response:
```json
{
  "query": "neural networks for text classification",
  "keywords": ["text classification", "neural networks"],
  "results": [
    {
      "rank": 1,
      "paper_id": "cs-9308101v1",
      "title": "...",
      "abstract": "...",
      "category": "Artificial Intelligence",
      "first_author": "...",
      "score": 0.847,
      "keywords": ["keyword1", "keyword2", "keyword3"]
    }
  ],
  "latency_ms": 42.3
}
```

### `GET /health`

Returns model status, device, and number of papers loaded.

## Without Docker

```bash
pip install -r requirements.txt
python prepare_api_data.py
uvicorn app:app --reload
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformers model name |
| `DATA_PATH` | `data/arxiv_embeddings.pkl` | Path to prepared embeddings file |
| `TOP_K_MAX` | `20` | Maximum results per request |
