"""
app.py — ArXiv Research Paper Recommender API

Ports the original Colab notebook (RecSyS_Code.ipynb) to a FastAPI service:
  - Same model: all-MiniLM-L6-v2
  - Same text field: title + category + summary
  - Same similarity: cosine similarity on L2-normalised embeddings
  - Same keyword extraction: KeyBERT
"""

import os
import pickle
import time
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from keybert import KeyBERT
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ArXiv Paper Recommender API",
    description=(
        "Embedding-based semantic recommender over 136k ArXiv papers. "
        "Uses all-MiniLM-L6-v2 + cosine similarity + KeyBERT keyword extraction."
    ),
    version="1.0.0",
)

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DATA_PATH  = os.getenv("DATA_PATH",       "data/arxiv_embeddings.pkl")
TOP_K_MAX  = int(os.getenv("TOP_K_MAX",   "20"))

# ── State ──────────────────────────────────────────────────────────────────────

class _State:
    model:      SentenceTransformer = None
    kw_model:   KeyBERT             = None
    embeddings: torch.Tensor        = None   # (N, 384) on CPU
    papers:     list                = None   # list of dicts from CSV

state = _State()
device = "cuda" if torch.cuda.is_available() else "cpu"


# ── Startup ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
def load_resources():
    print(f"Loading model: {MODEL_NAME}  |  device: {device}")
    state.model    = SentenceTransformer(MODEL_NAME)
    state.model.to(device)
    state.kw_model = KeyBERT(model=state.model)

    print(f"Loading data: {DATA_PATH}")
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    # Store embeddings as a torch tensor (same as original notebook)
    state.embeddings = torch.tensor(data["embeddings"], dtype=torch.float32)
    state.papers     = data["papers"]
    print(f"Ready — {len(state.papers):,} papers  |  embeddings: {state.embeddings.shape}")


# ── Schemas ────────────────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    query:  str
    top_k:  Optional[int] = 5


class PaperResult(BaseModel):
    rank:         int
    paper_id:     str
    title:        str
    abstract:     str
    category:     str
    first_author: str
    score:        float
    keywords:     List[str]


class RecommendResponse(BaseModel):
    query:      str
    keywords:   List[str]
    results:    List[PaperResult]
    latency_ms: float


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":        "ok",
        "papers_loaded": len(state.papers) if state.papers else 0,
        "model":         MODEL_NAME,
        "device":        device,
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if state.model is None or state.embeddings is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    top_k = min(req.top_k, TOP_K_MAX)
    if top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be >= 1")

    t0 = time.perf_counter()

    # Encode query — mirrors original notebook's get_top_n_recommendations()
    query_embedding = state.model.encode(
        req.query,
        convert_to_tensor=True,
        device=device,
        normalize_embeddings=True,
    )

    # Cosine similarity — util.pytorch_cos_sim matches original notebook exactly
    cos_scores = util.pytorch_cos_sim(query_embedding, state.embeddings)[0]
    top_indices = torch.topk(cos_scores, k=top_k).indices.tolist()

    # Query keywords via KeyBERT — matches original extract_keywords()
    query_keywords = [
        kw for kw, _ in state.kw_model.extract_keywords(
            req.query,
            stop_words="english",
            top_n=5,
        )
    ]

    # Build results
    results = []
    for rank, idx in enumerate(top_indices, 1):
        paper = state.papers[idx]
        abstract = paper["summary"]

        # Paper keywords
        paper_keywords = [
            kw for kw, _ in state.kw_model.extract_keywords(
                abstract,
                stop_words="english",
                top_n=3,
            )
        ]

        results.append(PaperResult(
            rank=rank,
            paper_id=str(paper.get("id", idx)),
            title=paper["title"],
            abstract=abstract[:400] + "..." if len(abstract) > 400 else abstract,
            category=paper.get("category", ""),
            first_author=paper.get("first_author", ""),
            score=float(cos_scores[idx]),
            keywords=paper_keywords,
        ))

    latency_ms = (time.perf_counter() - t0) * 1000

    return RecommendResponse(
        query=req.query,
        keywords=query_keywords,
        results=results,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/")
def root():
    return {
        "message": "ArXiv Recommender API",
        "docs":    "/docs",
        "health":  "/health",
    }
