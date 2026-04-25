"""
prepare_api_data.py

Builds arxiv_embeddings.pkl needed by the FastAPI app.
Mirrors the exact preprocessing from the original notebook:
    df["text"] = df["title"] + " " + df["category"] + " " + df["summary"]

Run once before docker compose up:
    python prepare_api_data.py

Requires: pandas sentence-transformers torch numpy
    pip install pandas sentence-transformers torch numpy
"""

import pickle
import numpy as np
import pandas as pd
import torch
import os
from sentence_transformers import SentenceTransformer

# ── Config ─────────────────────────────────────────────────────────────────────
CSV_PATH    = os.getenv("CSV_PATH",    "data/arXiv_scientific dataset.csv")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "data/arxiv_embeddings.pkl")
MODEL_NAME  = os.getenv("MODEL_NAME",  "all-MiniLM-L6-v2")
BATCH_SIZE  = int(os.getenv("BATCH_SIZE", "512"))
# ──────────────────────────────────────────────────────────────────────────────


def prepare():
    os.makedirs("data", exist_ok=True)

    # ── Load CSV ───────────────────────────────────────────────────────────────
    print(f"Loading: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df):,} rows  |  columns: {list(df.columns)}")

    # ── Build text field ──────────────────
    df["text"] = (
        df["title"].fillna("") + " " +
        df["category"].fillna("") + " " +
        df["summary"].fillna("")
    )

    # ── Encode ─────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nEncoding {len(df):,} papers with '{MODEL_NAME}' on {device}")
    

    model = SentenceTransformer(MODEL_NAME)
    model.to(device)

    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalise: dot product == cosine similarity
        device=device,
    )
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"  Embeddings shape: {embeddings.shape}")

    # ── Build paper records — keep relevant columns ────────────────────────────
    papers = (
        df[["id", "title", "summary", "category", "category_code", "first_author"]]
        .to_dict(orient="records")
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    print(f"\nSaving to {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump({"papers": papers, "embeddings": embeddings}, f)

    size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
    print(f"Done — {size_mb:.1f} MB")


if __name__ == "__main__":
    prepare()