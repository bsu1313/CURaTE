#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import time
from pathlib import Path
from typing import List, Iterable, Optional

import numpy as np
from datasets import load_dataset

import torch
from sentence_transformers import SentenceTransformer

import faiss


def load_texts_from_json(path: str, key: str) -> List[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for ex in data:
        if key in ex and isinstance(ex[key], str) and ex[key].strip():
            out.append(ex[key].strip())
    if not out:
        raise ValueError(f"No texts extracted from {path} using key='{key}'")
    return out


def iter_unrelated_texts(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    text_field: str,
    streaming: bool,
    max_docs: int,
    seed: int = 0,
):
    ds = load_dataset(dataset_name, dataset_config, split=split, streaming=streaming)
    if streaming:
        try:
            ds = ds.shuffle(seed=seed, buffer_size=10_000)
        except Exception:
            pass
        n = 0
        for row in ds:
            if n >= max_docs:
                break
            txt = row.get(text_field, None)
            if isinstance(txt, str):
                txt = txt.strip()
                if txt:
                    yield txt
                    n += 1
    else:
        ds = ds.shuffle(seed=seed)
        ds = ds.select(range(min(max_docs, len(ds))))
        for row in ds:
            txt = row.get(text_field, None)
            if isinstance(txt, str):
                txt = txt.strip()
                if txt:
                    yield txt


def batched(it: Iterable[str], bs: int):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= bs:
            yield buf
            buf = []
    if buf:
        yield buf


def truncate_text(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return s
    s = s.strip().replace("\n", " ")
    return s[:max_chars]


@torch.no_grad()
def encode_batch_gpu(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
    fp16: bool,
) -> np.ndarray:
    """
    SentenceTransformer.encode already batches internally, but we still pass batch_size
    and we use autocast for fp16 speed.
    """
    # ST encode returns numpy by default; we ensure float32 for FAISS.
    # normalize_embeddings=True makes cosine=IP.
    if fp16 and torch.cuda.is_available():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            emb = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
    else:
        emb = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    emb = np.asarray(emb, dtype=np.float32)
    return emb


def recall_at_k(I_ann: np.ndarray, I_gt: np.ndarray, k: int) -> float:
    hits = 0.0
    for a, g in zip(I_ann, I_gt):
        hits += len(set(a[:k]).intersection(set(g[:k]))) / float(k)
    return hits / float(len(I_gt))


def time_search(index, query_emb: np.ndarray, k: int, runs: int = 5, warmup: int = 1):
    for _ in range(warmup):
        _ = index.search(query_emb[: min(64, len(query_emb))], k)

    times = []
    last = None
    for _ in range(runs):
        t0 = time.perf_counter()
        last = index.search(query_emb, k)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times, dtype=np.float64)
    return {
        "p50_s": float(np.percentile(times, 50)),
        "p95_s": float(np.percentile(times, 95)),
        "mean_s": float(times.mean()),
        "runs_s": times.tolist(),
        "last": last,
    }


def try_make_gpu_flat(index_flat_cpu: faiss.IndexFlatIP):
    """
    Optional: move FlatIP to GPU for faster ground-truth search.
    Requires faiss-gpu.
    """
    if not hasattr(faiss, "StandardGpuResources"):
        raise RuntimeError("This FAISS build does not have GPU support (faiss-gpu not installed).")

    res = faiss.StandardGpuResources()
    # 0 = GPU id
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index_flat_cpu)
    return index_gpu


def main():
    ap = argparse.ArgumentParser()

    # Your data
    ap.add_argument("--forget_json", required=True)
    ap.add_argument("--forget_key", default="paraphrased_instruction")

    # Unrelated corpus
    ap.add_argument("--unrel_dataset", default="wikipedia")
    ap.add_argument("--unrel_config", default="20220301.en")
    ap.add_argument("--unrel_split", default="train")
    ap.add_argument("--unrel_text_field", default="text")
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--unrel_docs", type=int, default=1_000_000)

    # Speed knobs
    ap.add_argument("--max_chars", type=int, default=400,
                    help="Truncate unrelated docs to this many characters (big speed win).")
    ap.add_argument("--encode_bs", type=int, default=512)
    ap.add_argument("--fp16", action="store_true", help="Use autocast fp16 during encoding (GPU).")

    # Embedder
    ap.add_argument("--model_name", default="multi-qa-mpnet-base-dot-v1")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    # Queries/eval
    ap.add_argument("--query_n", type=int, default=2000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--runs", type=int, default=5)

    # IVF
    ap.add_argument("--nlist", type=int, default=4096)
    ap.add_argument("--nprobe", type=int, default=16)
    ap.add_argument("--train_size", type=int, default=200_000)

    # Optional FAISS GPU for exact baseline
    ap.add_argument("--use_faiss_gpu_flat", action="store_true")

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    unrel_config = args.unrel_config.strip() or None

    print("Loading forget texts...")
    forget_texts = load_texts_from_json(args.forget_json, args.forget_key)
    print(f"Forget texts: {len(forget_texts):,}")

    # Queries sampled from forget texts
    qn = min(args.query_n, len(forget_texts))
    q_idx = rng.choice(len(forget_texts), size=qn, replace=False)
    query_texts = [forget_texts[i] for i in q_idx]
    print(f"Queries: {len(query_texts):,}")

    # Load embedder on GPU
    print(f"Loading embedder: {args.model_name} on {args.device}")
    model = SentenceTransformer(args.model_name, device=args.device)

    print("Encoding query embeddings...")
    query_emb = encode_batch_gpu(model, query_texts, batch_size=args.encode_bs, fp16=args.fp16)
    d = query_emb.shape[1]
    print(f"Embedding dim: {d}")

    # Prepare unrelated iterator
    print("Preparing unrelated corpus iterator...")
    unrel_iter = iter_unrelated_texts(
        dataset_name=args.unrel_dataset,
        dataset_config=unrel_config,
        split=args.unrel_split,
        text_field=args.unrel_text_field,
        streaming=args.streaming,
        max_docs=args.unrel_docs,
        seed=args.seed,
    )

    # -----------------------
    # IVF training pool
    # -----------------------
    print(f"\n[Step A] Building IVF training pool: {args.train_size:,} vectors")
    train_texts: List[str] = []

    # Include some forget texts first
    for t in forget_texts[: min(len(forget_texts), args.train_size)]:
        train_texts.append(t)
        if len(train_texts) >= args.train_size:
            break

    # Fill rest with unrelated docs (truncated)
    if len(train_texts) < args.train_size:
        need = args.train_size - len(train_texts)
        tmp = []
        for doc in unrel_iter:
            tmp.append(truncate_text(doc, args.max_chars))
            if len(tmp) >= need:
                break
        train_texts.extend(tmp)

    print("Encoding IVF training embeddings (GPU)...")
    train_emb = encode_batch_gpu(model, train_texts, batch_size=args.encode_bs, fp16=args.fp16)

    # Build indexes (CPU)
    print("\n[Step A] Training IVF index (CPU FAISS)...")
    quantizer = faiss.IndexFlatIP(d)
    index_ivf = faiss.IndexIVFFlat(quantizer, d, args.nlist, faiss.METRIC_INNER_PRODUCT)
    index_ivf.train(train_emb)
    index_ivf.nprobe = args.nprobe

    # Exact baseline index (CPU)
    index_flat_cpu = faiss.IndexFlatIP(d)

    # -----------------------
    # Add corpus vectors
    # -----------------------
    print("\n[Step B] Adding vectors to indexes...")

    print("Adding forget vectors...")
    for batch in batched(forget_texts, bs=args.encode_bs):
        emb = encode_batch_gpu(model, batch, batch_size=args.encode_bs, fp16=args.fp16)
        index_flat_cpu.add(emb)
        index_ivf.add(emb)

    # Recreate unrelated iterator (because we consumed some for training)
    unrel_iter2 = iter_unrelated_texts(
        dataset_name=args.unrel_dataset,
        dataset_config=unrel_config,
        split=args.unrel_split,
        text_field=args.unrel_text_field,
        streaming=args.streaming,
        max_docs=args.unrel_docs,
        seed=args.seed,
    )

    print(f"Adding unrelated vectors: {args.unrel_docs:,} docs (truncated to {args.max_chars} chars)")
    added = 0
    for batch in batched((truncate_text(x, args.max_chars) for x in unrel_iter2), bs=args.encode_bs):
        emb = encode_batch_gpu(model, batch, batch_size=args.encode_bs, fp16=args.fp16)
        index_flat_cpu.add(emb)
        index_ivf.add(emb)
        added += len(batch)
        if added % (args.encode_bs * 50) == 0:
            print(f"  added {added:,}/{args.unrel_docs:,}")

    N = index_flat_cpu.ntotal
    print(f"\nTotal indexed vectors: N={N:,}  nlist={args.nlist}  nprobe={args.nprobe}")

    # Optional: move exact index to GPU for faster ground-truth
    index_flat = index_flat_cpu
    if args.use_faiss_gpu_flat:
        print("\nMoving exact FlatIP to GPU (requires faiss-gpu)...")
        index_flat = try_make_gpu_flat(index_flat_cpu)

    # -----------------------
    # Benchmark
    # -----------------------
    print("\n[Step C] Benchmark search...")
    print("Timing exact search (FlatIP)...")
    flat_stats = time_search(index_flat, query_emb, k=args.k, runs=args.runs, warmup=1)
    D_gt, I_gt = flat_stats["last"]

    print("Timing ANN search (IVF)...")
    ivf_stats = time_search(index_ivf, query_emb, k=args.k, runs=args.runs, warmup=1)
    D_ann, I_ann = ivf_stats["last"]

    qps_flat = len(query_emb) / flat_stats["mean_s"]
    qps_ivf = len(query_emb) / ivf_stats["mean_s"]
    speedup = flat_stats["mean_s"] / ivf_stats["mean_s"]
    r_at_k = recall_at_k(I_ann, I_gt, k=args.k)

    print("\n==== RESULTS ====")
    print(f"Corpus N={N:,}, dim={d}, queries Q={len(query_emb):,}, top-k={args.k}")
    print(f"Exact FlatIP: mean={flat_stats['mean_s']:.4f}s  p50={flat_stats['p50_s']:.4f}s  p95={flat_stats['p95_s']:.4f}s  QPS={qps_flat:.1f}")
    print(f"ANN  IVF   : mean={ivf_stats['mean_s']:.4f}s  p50={ivf_stats['p50_s']:.4f}s  p95={ivf_stats['p95_s']:.4f}s  QPS={qps_ivf:.1f}")
    print(f"Recall@{args.k} (IVF vs exact): {r_at_k:.4f}")
    print(f"Speedup (mean): {speedup:.2f}x")


if __name__ == "__main__":
    main()