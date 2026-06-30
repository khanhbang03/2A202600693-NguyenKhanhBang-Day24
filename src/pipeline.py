from __future__ import annotations

"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4."""

import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K


_LATENCY = {
    "chunking_s": 0.0,
    "enrichment_s": 0.0,
    "indexing_s": 0.0,
    "reranker_load_s": 0.0,
    "retrieval_ms": [],
    "reranking_ms": [],
    "generation_ms": [],
    "evaluation_s": 0.0,
}


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    _LATENCY["chunking_s"] = time.time() - t0
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({_LATENCY['chunking_s']:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        _LATENCY["enrichment_s"] = time.time() - t0
        print(f"  ✓ Enriched {len(enriched)} chunks ({_LATENCY['enrichment_s']:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    _LATENCY["indexing_s"] = time.time() - t0
    print(f"  ✓ Indexed ({_LATENCY['indexing_s']:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    reranker._load_model()
    _LATENCY["reranker_load_s"] = time.time() - t0
    print(f"  ✓ Reranker ready ({_LATENCY['reranker_load_s']:.1f}s)", flush=True)

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    started = time.perf_counter()
    results = search.search(query)
    _LATENCY["retrieval_ms"].append((time.perf_counter() - started) * 1000)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    started = time.perf_counter()
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    _LATENCY["reranking_ms"].append((time.perf_counter() - started) * 1000)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    started = time.perf_counter()
    from config import OPENAI_API_KEY
    use_openai = bool(OPENAI_API_KEY) and os.getenv("RAG_USE_OPENAI", "").lower() in {
        "1", "true", "yes",
    }
    if use_openai and contexts:
        try:
            from openai import OpenAI
            client = OpenAI()
            context_str = "\n\n".join(contexts)
            resp = client.chat.completions.create(model="gpt-4o-mini", messages=[
                {"role": "system", "content": "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ])
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        # Offline extractive answer: retain all top evidence instead of silently
        # pretending that the first chunk alone is a generated response.
        answer = "\n\n".join(contexts) if contexts else "Không tìm thấy thông tin."
    _LATENCY["generation_ms"].append((time.perf_counter() - started) * 1000)
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    _LATENCY["evaluation_s"] = time.time() - t0
    print(f"  ✓ RAGAS done ({_LATENCY['evaluation_s']:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    os.makedirs("reports", exist_ok=True)
    save_report(results, failures, path="reports/ragas_report.json")
    _save_latency_report()
    return results


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _save_latency_report() -> None:
    summary = {
        "chunking_s": round(float(_LATENCY["chunking_s"]), 4),
        "enrichment_s": round(float(_LATENCY["enrichment_s"]), 4),
        "indexing_s": round(float(_LATENCY["indexing_s"]), 4),
        "reranker_load_s": round(float(_LATENCY["reranker_load_s"]), 4),
        "retrieval_avg_ms": round(_average(_LATENCY["retrieval_ms"]), 4),
        "reranking_avg_ms": round(_average(_LATENCY["reranking_ms"]), 4),
        "generation_avg_ms": round(_average(_LATENCY["generation_ms"]), 4),
        "evaluation_s": round(float(_LATENCY["evaluation_s"]), 4),
        "num_queries": len(_LATENCY["retrieval_ms"]),
    }
    with open("reports/latency_breakdown.json", "w", encoding="utf-8") as handle:
        import json
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    os.makedirs("analysis", exist_ok=True)
    rows = "\n".join(
        f"| {name} | {value:.4f} |"
        for name, value in [
            ("Chunking", summary["chunking_s"]),
            ("Enrichment", summary["enrichment_s"]),
            ("Indexing", summary["indexing_s"]),
            ("Reranker load", summary["reranker_load_s"]),
            ("Retrieval/query (ms)", summary["retrieval_avg_ms"]),
            ("Reranking/query (ms)", summary["reranking_avg_ms"]),
            ("Generation/query (ms)", summary["generation_avg_ms"]),
            ("Evaluation", summary["evaluation_s"]),
        ]
    )
    with open("analysis/latency_report.md", "w", encoding="utf-8") as handle:
        handle.write(
            "# Latency Breakdown\n\n"
            f"Đo trên {summary['num_queries']} truy vấn của test set.\n\n"
            "| Bước | Thời gian |\n|---|---:|\n"
            f"{rows}\n"
        )


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
