"""
evaluate.py — ArXiv Recommender Evaluation

Evaluates the recommender system using two approaches:

1. Category-based evaluation (matches original notebook methodology):
   - Relevance proxy: returned paper shares the same category as query paper
   - Computes Precision@K and mean Precision@K (mP@K)
   - Note: the original notebook called this "MAP@5" but it is actually
     mean Precision@5 — this script uses the correct terminology.

2. Self-retrieval evaluation (stronger sanity check):
   - Query = abstract of a paper in the corpus
   - Relevant = that exact paper appears in top-K results
   - A working recommender should retrieve the source paper at rank 1

Run:
    python evaluate.py
    python evaluate.py --num-queries 200 --k 5

Requires: pandas sentence-transformers torch numpy scikit-learn tqdm
    pip install pandas sentence-transformers torch numpy scikit-learn tqdm
"""

import argparse
import pickle
import random
import time
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
DEFAULT_DATA_PATH  = "data/arxiv_embeddings.pkl"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_K          = 5
DEFAULT_NUM_QUERIES = 100
SEED               = 42
# ──────────────────────────────────────────────────────────────────────────────


def load_data(data_path: str, model_name: str, device: str):
    print(f"Loading data from {data_path}...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    papers     = data["papers"]
    embeddings = torch.tensor(data["embeddings"], dtype=torch.float32)
    model      = SentenceTransformer(model_name)
    model.to(device)

    print(f"  {len(papers):,} papers  |  embeddings: {embeddings.shape}  |  device: {device}")
    return papers, embeddings, model


def get_recommendations(query_text: str, model, embeddings, k: int, device: str):
    """Encode query and return top-k indices + scores."""
    query_emb = model.encode(
        query_text,
        convert_to_tensor=True,
        device=device,
        normalize_embeddings=True,
    )
    scores     = util.pytorch_cos_sim(query_emb, embeddings)[0]
    top_result = torch.topk(scores, k=k + 1)   # +1 to exclude self if present
    return top_result.indices.tolist(), top_result.values.tolist()


# ── Evaluation 1: Category-based (matches original notebook) ──────────────────

def evaluate_category(papers, embeddings, model, num_queries: int, k: int, device: str):
    """
    Precision@K using same-category as relevance proxy.
    Reproduces the original notebook's methodology.
    Note: The original notebook labelled this MAP@5 but it is mean Precision@5.
    """
    print(f"\n{'='*60}")
    print(f"EVALUATION 1 — Category-based Precision@{k}")
    print(f"Relevance proxy: returned paper shares query paper's category")
    print(f"Queries: {num_queries}  |  K: {k}")
    print(f"{'='*60}")

    random.seed(SEED)
    indices = random.sample(range(5, len(papers)), num_queries)

    precision_scores = []
    results_list     = []

    for idx in tqdm(indices, desc="Evaluating"):
        query_paper    = papers[idx]
        query_text     = (query_paper["title"] + " " +
                         query_paper.get("category", "") + " " +
                         query_paper["summary"])
        query_category = query_paper.get("category", "")

        top_indices, top_scores = get_recommendations(
            query_text, model, embeddings, k + 1, device
        )

        # Exclude the query paper itself from results
        top_indices = [i for i in top_indices if i != idx][:k]

        # Relevance: same category
        relevant = sum(
            1 for i in top_indices
            if papers[i].get("category", "") == query_category
        )
        p_at_k = relevant / k
        precision_scores.append(p_at_k)

        results_list.append({
            "Query Title":    query_paper["title"][:60] + "...",
            "Category":       query_category,
            f"Relevant@{k}":  relevant,
            f"P@{k}":         round(p_at_k, 4),
        })

    mean_p_at_k = np.mean(precision_scores)
    results_df  = pd.DataFrame(results_list)

    print(f"\nResults (first 10 queries):")
    print(results_df.head(10).to_string(index=False))
    print(f"\n{'─'*60}")
    print(f"Mean Precision@{k} (mP@{k}): {mean_p_at_k:.4f}")
    print(f"  (Original notebook terminology: 'MAP@{k}' = {mean_p_at_k:.4f})")
    print(f"  Evaluated on {num_queries} queries (original notebook: 10)")
    print(f"{'─'*60}")

    return mean_p_at_k, results_df


# ── Evaluation 2: Full-text self-retrieval (sanity check) ─────────────────────

def evaluate_self_retrieval(papers, embeddings, model, num_queries: int, k: int, device: str):
    """
    Self-retrieval: query with a paper's own full text, check if it appears in top-K.
    Rank 1 = perfect. A good embedder should score near 100% on this.
    Easy test — the query text closely matches what was indexed.
    """
    print(f"\n{'='*60}")
    print(f"EVALUATION 2 — Full-text self-retrieval (sanity check)")
    print(f"Query = paper's own title+category+summary  |  Relevant = that paper in top-{k}")
    print(f"Queries: {num_queries}  |  K: {k}")
    print(f"{'='*60}")

    random.seed(SEED)
    indices = random.sample(range(len(papers)), num_queries)

    hits_at_k = 0
    rank_list  = []
    latencies  = []

    for idx in tqdm(indices, desc="Full-text self-retrieval"):
        paper      = papers[idx]
        query_text = (paper["title"] + " " +
                      paper.get("category", "") + " " +
                      paper["summary"])

        t0 = time.perf_counter()
        top_indices, _ = get_recommendations(query_text, model, embeddings, k + 1, device)
        latencies.append((time.perf_counter() - t0) * 1000)

        if idx in top_indices:
            rank = top_indices.index(idx) + 1
            hits_at_k += 1
        else:
            rank = k + 1

        rank_list.append(rank)

    recall_at_k = hits_at_k / num_queries
    mrr         = np.mean([1 / r for r in rank_list])
    avg_latency = np.mean(latencies)

    print(f"\n{'─'*60}")
    print(f"Recall@{k}: {recall_at_k:.4f}  ({hits_at_k}/{num_queries})")
    print(f"MRR:        {mrr:.4f}")
    print(f"Avg latency:{avg_latency:.1f}ms")
    print(f"{'─'*60}")

    return recall_at_k, mrr, avg_latency


# ── Evaluation 3: Title-only self-retrieval (main evaluation metric) ───────────

def evaluate_title_only(papers, embeddings, model, num_queries: int, k: int, device: str):
    """
    Title-only self-retrieval: query with just the paper title, check if the
    correct paper appears in top-K results against all 136k papers.

    This is a stronger and more realistic test than full-text self-retrieval:
    - Titles are short and ambiguous — similar to real user queries
    - The indexed text includes category + summary, so the model must bridge
      the gap between a short title query and a rich indexed document
    - Ground truth is unambiguous: exactly one correct answer per query
    - No human annotation needed

    Metrics:
    - Recall@K: fraction of queries where correct paper is in top-K
    - MRR: Mean Reciprocal Rank (rewards finding paper at rank 1)
    - Recall@1: fraction where correct paper is the top result
    """
    print(f"\n{'='*60}")
    print(f"EVALUATION 3 — Title-only self-retrieval (primary metric)")
    print(f"Query = title only  |  Relevant = that paper in top-{k}")
    print(f"Corpus size: {len(papers):,} papers  |  Queries: {num_queries}  |  K: {k}")
    print(f"{'='*60}")

    random.seed(SEED)
    indices = random.sample(range(len(papers)), num_queries)

    hits_at_1 = 0
    hits_at_k = 0
    rank_list  = []
    latencies  = []

    for idx in tqdm(indices, desc="Title-only retrieval"):
        paper      = papers[idx]
        query_text = paper["title"]   # title only — no category, no summary

        t0 = time.perf_counter()
        top_indices, _ = get_recommendations(query_text, model, embeddings, k + 1, device)
        latencies.append((time.perf_counter() - t0) * 1000)

        if idx in top_indices:
            rank = top_indices.index(idx) + 1
            hits_at_k += 1
            if rank == 1:
                hits_at_1 += 1
        else:
            rank = k + 1

        rank_list.append(rank)

    recall_at_1 = hits_at_1 / num_queries
    recall_at_k = hits_at_k / num_queries
    mrr         = np.mean([1 / r for r in rank_list])
    avg_latency = np.mean(latencies)

    print(f"\n{'─'*60}")
    print(f"Recall@1 (exact top result):  {recall_at_1:.4f}  ({hits_at_1}/{num_queries})")
    print(f"Recall@{k}:                    {recall_at_k:.4f}  ({hits_at_k}/{num_queries})")
    print(f"MRR:                          {mrr:.4f}")
    print(f"Avg query latency:            {avg_latency:.1f}ms")
    print(f"{'─'*60}")
    print(f"Interpretation: given only a paper's title, the model retrieves")
    print(f"the correct paper in the top {k} results {recall_at_k*100:.1f}% of the time")
    print(f"across {len(papers):,} candidates.")
    print(f"{'─'*60}")

    return recall_at_1, recall_at_k, mrr, avg_latency


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate ArXiv Recommender")
    parser.add_argument("--data-path",   default=DEFAULT_DATA_PATH)
    parser.add_argument("--model",       default=DEFAULT_MODEL_NAME)
    parser.add_argument("--num-queries", type=int, default=DEFAULT_NUM_QUERIES)
    parser.add_argument("--k",           type=int, default=DEFAULT_K)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    papers, embeddings, model = load_data(args.data_path, args.model, device)

    # Evaluation 1 — category-based (reproduces original notebook)
    mp_at_k, _ = evaluate_category(
        papers, embeddings, model, args.num_queries, args.k, device
    )

    # Evaluation 2 — full-text self-retrieval (sanity check)
    recall_full, mrr_full, lat_full = evaluate_self_retrieval(
        papers, embeddings, model, args.num_queries, args.k, device
    )

    # Evaluation 3 — title-only self-retrieval (primary metric)
    recall_1, recall_title, mrr_title, lat_title = evaluate_title_only(
        papers, embeddings, model, args.num_queries, args.k, device
    )

    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY  ({args.num_queries} queries, K={args.k}, corpus={len(papers):,})")
    print(f"{'='*60}")
    print(f"Eval 1 — Category mP@{args.k} (original notebook proxy): {mp_at_k:.4f}")
    print(f"Eval 2 — Full-text self-retrieval Recall@{args.k}:        {recall_full:.4f}")
    print(f"Eval 3 — Title-only Recall@1:                          {recall_1:.4f}  ← primary")
    print(f"Eval 3 — Title-only Recall@{args.k}:                      {recall_title:.4f}  ← primary")
    print(f"Eval 3 — Title-only MRR:                               {mrr_title:.4f}  ← primary")
    print(f"Avg query latency (title-only):                        {lat_title:.1f}ms")
    print(f"{'='*60}")
    print(f"\nRecommended CV metric:")
    print(f"  Title-only Recall@{args.k} = {recall_title:.4f} against {len(papers):,} papers")


if __name__ == "__main__":
    main()