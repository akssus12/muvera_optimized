# -*- coding: utf-8 -*-
"""
ANN + TPack(B^T, F-order) + io_uring 병렬 pread 재랭크 (패널링 + 자동튜닝)
- TPack(열-주 저장: (D, W) F-order)에서 여러 문서의 열 범위를 io_uring으로 병렬 pread
- iovec/readv 미사용: liburing 바인딩 차이를 피하기 위해 io_uring_prep_read(단일 버퍼)만 사용
- 문서 패널링(panel vstack 복사 有) + 단일 GEMM로 처리 → tail latency(p99) 개선
- (1) 경량 비용모델 초기치, (2) Contextual LinUCB 밴딧, (3) 가드레일(비율/히스테리시스/쿨다운)
- Rerank(CP), Rerank(IO), Rerank(VS) 타이밍 분리 + 패널 단위 상세 계측
"""

import os, json, time, hashlib, logging, pathlib, csv, math, heapq, random, threading, argparse, sys, resource
from collections import OrderedDict, deque
from dataclasses import dataclass, replace
from typing import Optional, List, Tuple, Dict
from queue import Queue
from statistics import mean
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice

import numpy as np
import torch
import joblib
import nltk

# --- Single-thread BLAS by default (문서/패널 단위 병렬화를 우선) ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ===== liburing (io_uring) detection =====
_LIBURING_OK = False
try:
    import liburing  # pip install liburing
    from liburing import (
        io_uring, io_uring_cqe, io_uring_sqe,
        io_uring_queue_init, io_uring_queue_exit,
        io_uring_get_sqe, io_uring_submit,
        io_uring_wait_cqe, io_uring_cqe_seen, io_uring_peek_cqe,
        io_uring_prep_read,
        io_uring_sqe_set_data64
    )
    _LIBURING_OK = True
except Exception:
    _LIBURING_OK = False

def _trap_error(res: int):
    if isinstance(res, int) and res < 0:
        err = -int(res)
        raise OSError(err, os.strerror(err))

# ========== Faiss (CPU) ==========
try:
    import faiss
    _FAISS_OK = True
except Exception:
    _FAISS_OK = False
    faiss = None

# ======================
# --- Configuration ----
# ======================
DATASET_REPO_ID = "treccovid"   # 출력용 명칭
dataset = "trec-covid"          # BEIR 내부 이름
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"

COLBERT_MODEL_NAME = "raphaelsty/neural-cherche-colbert"
TOP_K = 10

RERANK_WORKERS = 1
TARGET_NUM_QUERIES = 100
RANDOM_SEED = 42

FAISS_NLIST = 1000
FAISS_NPROBE = 50
FAISS_CANDIDATES = 100
FAISS_NUM_THREADS = 1

ANN_BATCH_SIZE = 12
RERANK_BATCH_QUERIES = 12
BATCH_RERANK_SIZE = 12

RERANK_TOPN = 0  # 전체 후보 재랭크(0이면 모든 ANN 후보)
BATCH_RERANK_MODE = "batch"   # "immediate" or "batch"

# Doc 임베딩 차원(FDE)
FDE_DIM = 128
FDE_NUM_REPETITIONS = 2
FDE_NUM_SIMHASH = 3

# TPack io_uring QDepth / submit batch
TPACK_IOURING_QD = 64
TPACK_IOURING_SUBMIT = 64

# ===== 패널링 & 오토튜너 기본값 =====
DEFAULT_PANEL_INIT_COLS = 25600
PANEL_MIN_COLS = 1024
PANEL_MAX_COLS = 65536
COPY_GEMM_RATIO_CAP = 0.25   # copy_time/gemm_time 상한 (가드레일)
HYSTERESIS_DOWN = 0.67       # 급변 방지 (하한 배율)
HYSTERESIS_UP   = 1.50       # 급변 방지 (상한 배율)
COOLDOWN_STEPS  = 3          # 연속 변경 억제

# Device 선택
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

CACHE_ROOT = os.path.join(pathlib.Path(__file__).parent.absolute(), "cache_muvera")
os.makedirs(CACHE_ROOT, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info(f"Using device: {DEVICE}  |  FAISS={'on' if _FAISS_OK else 'off'}")
if _LIBURING_OK:
    logging.info("[io_uring] liburing detected -> async pread enabled (single-buffer)")
else:
    logging.info("[io_uring] liburing not available -> fallback to synchronous pread")

# ===== 전역 집계(평균/분포) =====
avg_search_time_list = []
avg_ann_time_list = []
avg_rerank_time_list = []
avg_rerank_cp_list = []
avg_rerank_io_list = []
avg_rerank_wait_list = []
avg_vstack_time_list = []
avg_dup_ratio_list = []

# ======================
# ===== Dataset ========
# ======================
from beir import util
from beir.datasets.data_loader import GenericDataLoader

def load_nanobeir_dataset():
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    logging.info(f"Loading dataset '{DATASET_REPO_ID}' from {data_path} ...")
    corpus, queries, _ = GenericDataLoader(data_folder=data_path).load(split="test")
    target_queries = dict(islice(queries.items(), TARGET_NUM_QUERIES))

    # qrels 로드
    candidates = [
        os.path.join(data_path, "qrels", "test.tsv"),
        os.path.join(data_path, "test.tsv"),
    ]
    qrels_pos = {}
    tsv_path = next((p for p in candidates if os.path.exists(p)), None)
    if tsv_path is None:
        logging.warning("[qrels] test.tsv not found; fallback to BEIR loader qrels.")
        _, _, qrels_beir = GenericDataLoader(data_folder=data_path).load(split="test")
        qrels_pos = qrels_beir
    else:
        with open(tsv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            def _get(row, *keys):
                for k in keys:
                    if k in row:
                        return row[k]
                raise KeyError(f"Missing keys {keys} in row: {row}")
            for row in reader:
                try:
                    qid   = _get(row, "query-id", "qid", "query_id")
                    docid = _get(row, "corpus-id", "docid", "doc_id")
                    score = int(_get(row, "score", "label"))
                except Exception:
                    continue
                if score > 0:
                    qrels_pos.setdefault(str(qid), {})[str(docid)] = 1
    logging.info(f"Dataset loaded: {len(corpus)} docs, {len(target_queries)} queries, "
                 f"{sum(len(v) for v in qrels_pos.values())} positive qrels.")
    return data_path, corpus, target_queries, qrels_pos

def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy().astype(np.float32)
    elif isinstance(x, np.ndarray):
        return x.astype(np.float32)
    else:
        raise TypeError(type(x))

# =====================================
# FDE stubs (외부 모듈)
# =====================================
from fde_generator_optimized_stream import (
    FixedDimensionalEncodingConfig,
    generate_query_fde,
    generate_document_fde_batch,
)

# =====================================
# TPack (col-major B^T) Builder & Reader
# =====================================
class TPackBuilder:
    def __init__(self, tpack_dir: str, dim: int):
        os.makedirs(tpack_dir, exist_ok=True)
        self.tokensF = os.path.join(tpack_dir, "tokensF.bin")       # col-major B^T bytes
        self.doc_col_ptrs = os.path.join(tpack_dir, "doc_col_ptrs.npy")
        self.meta = os.path.join(tpack_dir, "tpack_meta.json")
        self.D = int(dim)  # build()에서 실제 문서 차원으로 덮어씀

    def build(self, retriever: "ColbertFdeRetrieverNaive", doc_ids: List[str]):
        logging.info(f"[TPackBuilder] building col-major B^T...")
        ptrs = [0]
        total_cols = 0
        true_D = None
        with open(self.tokensF, "wb", buffering=0) as f:
            for did in doc_ids:
                X = retriever._get_doc_embeddings(did, allow_build=True).astype(np.float32, copy=False)  # [n_i, D]
                if true_D is None:
                    true_D = int(X.shape[1])
                elif int(X.shape[1]) != true_D:
                    raise ValueError(f"[TPackBuilder] doc dim mismatch: got {X.shape[1]} vs expected {true_D}")
                XT = np.asfortranarray(X.T)  # (D, n_i), F-contig
                f.write(XT.tobytes(order="F"))
                total_cols += int(X.shape[0])
                ptrs.append(total_cols)
        if true_D is None:
            true_D = int(self.D)
        np.save(self.doc_col_ptrs, np.asarray(ptrs, dtype=np.int64))
        with open(self.meta, "w", encoding="utf-8") as f:
            json.dump({"dtype": "float32", "order": "F", "D": int(true_D), "total_cols": int(total_cols),
                       "tokensF": "tokensF.bin", "doc_col_ptrs": "doc_col_ptrs.npy"}, f, indent=2)
        logging.info(f"[TPackBuilder] done: W={total_cols}, D={true_D}")

class TPackReader:
    """
    tokensF.bin stores B^T as (D, W) in F-contiguous layout.
    - pread_cols(): 단일 연속 열 범위를 읽어 (D, n) F-order view 반환
    - pread_cols_many(): (doc_id, s, e) 스팬들을 모아 io_uring 병렬 pread (폴백: ThreadPool+pread)
    """
    def __init__(self,
                 tpack_dir: str,
                 use_iouring: bool = True,
                 iouring_qd: int = 64,
                 iouring_submit_batch: int = 64):
        meta_path = os.path.join(tpack_dir, "tpack_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["dtype"] == "float32" and meta["order"] == "F"
        self.D = int(meta["D"])
        self.total_cols = int(meta["total_cols"])
        self.tokensF_path = os.path.join(tpack_dir, meta["tokensF"])
        self.doc_col_ptrs = np.load(os.path.join(tpack_dir, meta["doc_col_ptrs"]), mmap_mode='r')
        self.fd = os.open(self.tokensF_path, os.O_RDONLY)
        self.itemsize = 4

        self.use_iouring = bool(use_iouring)
        self.iouring_qd = max(1, int(iouring_qd))
        self.iouring_submit_batch = max(1, int(iouring_submit_batch))

        self._liburing_ready = _LIBURING_OK and self.use_iouring
        if self._liburing_ready:
            logging.info(f"[TPack/liburing] enabled (qd={self.iouring_qd}, submit_batch={self.iouring_submit_batch}, mode=read)")
        else:
            if self.use_iouring and not _LIBURING_OK:
                logging.warning("[TPack/liburing] liburing not available; fallback to threaded pread.")
            logging.info("[TPack] using threaded pread fallback")

    def close(self):
        try: os.close(self.fd)
        except Exception: pass

    def doc_col_span(self, doc_idx: int) -> Tuple[int, int]:
        s = int(self.doc_col_ptrs[doc_idx]); e = int(self.doc_col_ptrs[doc_idx+1])
        return s, e  # [s, e) over columns of B^T

    def pread_cols(self, start_col: int, n_cols: int) -> np.ndarray:
        if n_cols <= 0:
            return np.empty((self.D, 0), dtype=np.float32, order="F")
        byte_off = start_col * self.D * self.itemsize
        byte_len = n_cols   * self.D * self.itemsize
        buf = os.pread(self.fd, byte_len, byte_off)
        arr = np.frombuffer(buf, dtype=np.float32, count=n_cols * self.D)
        return np.ndarray(shape=(self.D, n_cols), dtype=np.float32, buffer=arr, order="F")

    def _col_off_len(self, start_col: int, n_cols: int) -> Tuple[int, int]:
        return (start_col * self.D * self.itemsize,
                n_cols   * self.D * self.itemsize)

    def pread_cols_many_uring(self,
                              doc_spans: List[Tuple[str, int, int]],
                              qd: Optional[int] = None,
                              submit_batch: Optional[int] = None,
                              verify: bool = False) -> Dict[str, np.ndarray]:
        """
        doc_spans: [(doc_id, s, e), ...]
        returns: {doc_id: ndarray (D, ncols), F-order view}
        """
        if not doc_spans:
            return {}
        qd = self.iouring_qd if qd is None else max(1, int(qd))
        submit_batch = self.iouring_submit_batch if submit_batch is None else max(1, int(submit_batch))

        # normalize & build reqs
        reqs: List[Tuple[str, int, int, int, int]] = []
        for did, s, e in doc_spans:
            ncols = max(0, e - s)
            if ncols <= 0: continue
            off, nbytes = self._col_off_len(s, ncols)
            reqs.append((did, s, e, off, nbytes))
        if not reqs:
            return {}

        if not self._liburing_ready:
            return self._pread_cols_many_threaded(reqs, max_workers=qd)

        out: Dict[str, np.ndarray] = {}
        total = len(reqs)
        completed = 0
        issued = 0
        in_flight = 0

        ring = io_uring()
        cqe  = io_uring_cqe()
        ret = io_uring_queue_init(qd, ring, 0)
        _trap_error(ret)

        try:
            bufs: List[Optional[bytearray]] = [None] * total
            next_idx = 0

            def _submit_one(idx: int) -> bool:
                did, s, e, off, nbytes = reqs[idx]
                buf = bytearray(nbytes)
                bufs[idx] = buf
                sqe = io_uring_get_sqe(ring)
                if not sqe:
                    return False
                io_uring_prep_read(sqe, self.fd, buf, nbytes, off)
                io_uring_sqe_set_data64(sqe, idx)
                return True

            while completed < total:
                batch = 0
                while next_idx < total and in_flight < qd and batch < submit_batch:
                    if not _submit_one(next_idx):
                        break
                    next_idx += 1
                    issued += 1
                    in_flight += 1
                    batch += 1

                if batch > 0:
                    sub = io_uring_submit(ring)
                    _trap_error(sub)

                io_uring_wait_cqe(ring, cqe)
                res = cqe.res
                _trap_error(res)
                idx = cqe.user_data

                did, s, e, off, nbytes = reqs[idx]
                arr = np.frombuffer(bufs[idx], dtype=np.float32, count=(e - s) * self.D)
                viewF = np.ndarray(shape=(self.D, (e - s)), dtype=np.float32, buffer=arr, order="F")
                out[did] = viewF

                io_uring_cqe_seen(ring, cqe)
                completed += 1
                in_flight -= 1

            return out
        finally:
            io_uring_queue_exit(ring)

    def _pread_cols_many_threaded(self, reqs: List[Tuple[str, int, int, int, int]], max_workers: int) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        def _one(r):
            did, s, e, off, nbytes = r
            buf = os.pread(self.fd, nbytes, off)
            arr = np.frombuffer(buf, dtype=np.float32, count=(e - s) * self.D)
            viewF = np.ndarray(shape=(self.D, (e - s)), dtype=np.float32, buffer=arr, order="F")
            return did, viewF
        with ThreadPoolExecutor(max_workers=max(1, RERANK_WORKERS), thread_name_prefix="tpack-pread") as ex:
            futures = [ex.submit(_one, r) for r in reqs]
            for fut in as_completed(futures):
                did, arrF = fut.result()
                out[did] = arrF
        return out

# =====================================
# 간단한 BRUTE-FORCE 상한선(평가용)
# =====================================
BF_WORKERS = max(1, (os.cpu_count() or 4) // 2)
BF_CHUNK_SIZE = 256
_DOC_BUILD_LOCK = threading.Lock()

def _bf_chunk_worker(retriever, q_emb, doc_ids, k):
    local_heap: List[Tuple[float, str]] = []
    push = heapq.heappush; replace = heapq.heapreplace
    for did in doc_ids:
        d_tok = retriever._get_doc_embeddings(did, allow_build=True)
        score = float((q_emb @ d_tok.T).max(axis=1).sum()) if d_tok.size else -1e9
        if len(local_heap) < k: push(local_heap, (score, did))
        else:
            if score > local_heap[0][0]: replace(local_heap, (score, did))
    return local_heap

def _compute_bf_topk_for_query(retriever, qid, qtext, k, workers=None, chunk_size=256):
    if workers is None: workers = max(1, (os.cpu_count() or 4) // 2)
    key = retriever._query_key(qtext, qid)
    qemb, qfde = retriever._load_query_cache(key)
    if qemb is None:
        qmap = retriever.ranker.encode_queries(queries=[qtext])
        qemb = to_numpy(next(iter(qmap.values())))
        qcfg = replace(retriever.doc_config, fill_empty_partitions=False)
        qfde = generate_query_fde(qemb, qcfg)
        retriever._save_query_cache(key, qemb, qfde)

    doc_ids = retriever.doc_ids
    chunks: List[List[str]] = [doc_ids[i:i+chunk_size] for i in range(0, len(doc_ids), chunk_size)]

    global_heap: List[Tuple[float, str]] = []
    push = heapq.heappush; replace = heapq.heapreplace

    with ThreadPoolExecutor(max_workers=max(1, int(workers)), thread_name_prefix="bf-doc") as ex:
        futures = [ex.submit(_bf_chunk_worker, retriever, qemb, ch, k) for ch in chunks]
        for fut in as_completed(futures):
            local_heap = fut.result()
            for sc, did in local_heap:
                if len(global_heap) < k: push(global_heap, (sc, did))
                else:
                    if sc > global_heap[0][0]: replace(global_heap, (sc, did))

    top_sorted = sorted(((did, sc) for sc, did in global_heap), key=lambda x: x[1], reverse=True)
    return top_sorted

def _append_bf_topk(path: str, qid: str, topk: List[Tuple[str, float]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for rank, (docid, score) in enumerate(topk, start=1):
            f.write(f"{qid}\t{docid}\t{score:.8f}\t{rank}\n")

def compute_and_persist_bf_topk(retriever, queries, k, outfile):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    seen = set()
    if os.path.exists(outfile):
        with open(outfile, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                qid = line.split("\t", 1)[0]
                seen.add(qid)
    will = [(qid, qtext) for qid, qtext in queries.items() if str(qid) not in seen]
    logging.info(f"[BF] compute Top-{k} for {len(will)} queries (append -> {outfile})")
    for qid, qtext in will:
        t0 = time.perf_counter()
        topk = _compute_bf_topk_for_query(retriever, str(qid), qtext, k, workers=BF_WORKERS, chunk_size=BF_CHUNK_SIZE)
        _append_bf_topk(outfile, str(qid), topk)
        logging.info(f"[BF] qid={qid} done in {time.perf_counter()-t0:.3f}s")

def load_bf_truth(outfile: str) -> Dict[str, List[Tuple[str, float]]]:
    truth: Dict[str, List[Tuple[str, float]]] = {}
    if not os.path.exists(outfile): return truth
    with open(outfile, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            qid, docid, score, rank = line.rstrip("\n").split("\t")
            score = float(score)
            truth.setdefault(qid, []).append((docid, score))
    for qid in truth.keys():
        truth[qid] = sorted(truth[qid], key=lambda x: x[1], reverse=True)[:TOP_K]
    return truth

def system_topk_from_results(results: Dict[str, OrderedDict], k: int) -> Dict[str, List[str]]:
    out = {}
    for qid, ranked in results.items():
        out[str(qid)] = list(islice(ranked.keys(), k))
    return out

def recall_at_k_wrt_bf(results_topk, bf_truth, k):
    hits = 0; total = 0
    for qid, bf_list in bf_truth.items():
        bf_set = {doc for doc, _ in bf_list[:k]}
        sys_set = set(results_topk.get(qid, [])[:k])
        if not bf_set: continue
        total += 1
        hits += len(bf_set.intersection(sys_set)) / len(bf_set)
    return hits / total if total > 0 else 0.0

def hit_at_k_wrt_bf(results_topk, bf_truth, k):
    hits = 0; total = 0
    for qid, bf_list in bf_truth.items():
        bf_set = {doc for doc, _ in bf_list[:k]}
        if not bf_set: continue
        total += 1
        sys_set = set(results_topk.get(qid, [])[:k])
        hits += 1 if len(bf_set.intersection(sys_set)) > 0 else 0
    return hits / total if total > 0 else 0.0

def ndcg_at_k_wrt_bf(sys_topk, bf_truth, k):
    def _discount(i: int) -> float:
        return 1.0 / math.log2(i + 1)
    per_q: List[float] = []
    for qid, sys_docs in sys_topk.items():
        ideal = bf_truth.get(str(qid), [])
        if not ideal: continue
        ideal_k = ideal[:k]
        if not ideal_k: continue
        ideal_gains = [max(s, 0.0) for _, s in ideal_k]
        idcg = sum(g * _discount(i+1) for i, g in enumerate(ideal_gains))
        if idcg <= 0:
            per_q.append(0.0); continue
        ideal_map = {doc: max(sc, 0.0) for doc, sc in ideal_k}
        dcg = 0.0
        for i, d in enumerate(sys_docs[:k], start=1):
            g = ideal_map.get(d, 0.0)
            dcg += g * _discount(i)
        per_q.append(dcg / idcg)
    mean_ndcg = sum(per_q) / len(per_q) if per_q else 0.0
    return mean_ndcg, per_q

# =====================================
# 오토튜너 구성요소: 최근 통계, LinUCB, 튜너 본체
# =====================================
@dataclass
class RecentStats:
    p99_gemm: float = 0.0
    p50_gemm: float = 0.0
    p99_pread: float = 0.0
    p50_pread: float = 0.0
    dup_ratio: float = 0.0
    seq_ratio: float = 0.0
    F_peak: float = 6e11    # FLOPs/s (초기치; 런타임 보정)
    B_peak: float = 2.5e10  # Bytes/s (초기치; 런타임 보정)

class EWMA:
    def __init__(self, alpha=0.2, init=None):
        self.alpha = float(alpha)
        self.val = init
    def update(self, x):
        if self.val is None:
            self.val = float(x)
        else:
            self.val = self.alpha * float(x) + (1-self.alpha) * self.val
        return self.val
    def get(self, default=0.0):
        return self.val if self.val is not None else default

class PercentileWindow:
    def __init__(self, cap=256):
        self.cap = cap
        self.buf = deque(maxlen=cap)
    def add(self, x):
        self.buf.append(float(x))
    def p(self, q):
        if not self.buf: return 0.0
        arr = np.array(self.buf, dtype=np.float64)
        return float(np.percentile(arr, q))

class LinUCB:
    """
    Global linear UCB (하나의 선형 모델; 피처에 action 포함)
    x = [1, Tq, log(Tq+1), Np, log(Np+1), Tq*Np, p99_gemm, p99_pread, dup_ratio, seq_ratio]
    """
    def __init__(self, d: int, alpha: float = 1.5, l2: float = 1e-3):
        self.d = d
        self.alpha = float(alpha)
        self.A = np.eye(d) * l2
        self.b = np.zeros((d, 1), dtype=np.float64)

    def _theta(self):
        A_inv = np.linalg.inv(self.A)
        return (A_inv @ self.b).reshape(-1), A_inv

    def select(self, X_list: List[np.ndarray]) -> int:
        theta, A_inv = self._theta()
        best_i, best_val = 0, -1e30
        for i, x in enumerate(X_list):
            x = x.reshape(-1, 1)
            mu = float(theta @ x.flatten())
            sigma = float(np.sqrt(x.T @ A_inv @ x))
            val = mu + self.alpha * sigma
            if val > best_val:
                best_val = val; best_i = i
        return best_i

    def update(self, x: np.ndarray, reward: float):
        x = x.reshape(-1, 1)
        self.A += (x @ x.T)
        self.b += (reward * x)

class PanelAutoTuner:
    def __init__(self,
                 D: int,
                 init_cols: int = DEFAULT_PANEL_INIT_COLS,
                 min_cols: int = PANEL_MIN_COLS,
                 max_cols: int = PANEL_MAX_COLS,
                 copy_gemm_ratio_cap: float = COPY_GEMM_RATIO_CAP,
                 hysteresis_dn: float = HYSTERESIS_DOWN,
                 hysteresis_up: float = HYSTERESIS_UP,
                 cooldown_steps: int = COOLDOWN_STEPS,
                 w99: float = 0.8,
                 w50: float = 0.2,
                 eps_acc: float = 0.01):
        self.D = int(D)
        self.min_cols = int(min_cols)
        self.max_cols = int(max_cols)
        self.copy_gemm_ratio_cap = float(copy_gemm_ratio_cap)
        self.h_dn = float(hysteresis_dn)
        self.h_up = float(hysteresis_up)
        self.cooldown_steps = int(cooldown_steps)
        self.w99 = float(w99)
        self.w50 = float(w50)
        self.eps_acc = float(eps_acc)

        self.alpha_g = 1.0
        self.alpha_c = 1.0
        self.last_Np = int(init_cols)
        self.cooldown = 0

        self.pwin_panel = PercentileWindow(256)
        self.pwin_copy  = PercentileWindow(256)
        self.pwin_gemm  = PercentileWindow(256)
        self.pwin_pread = PercentileWindow(256)

        self.ew_F = EWMA(alpha=0.2, init=6e11)   # FLOPs/s
        self.ew_B = EWMA(alpha=0.2, init=2.5e10) # Bytes/s

        self.bandit = LinUCB(d=10, alpha=1.5, l2=1e-3)

    def _roofline(self, Tq, Np, F_peak, B_peak):
        # GEMM FLOPs ~ 2*Tq*D*Np ; Copy Bytes ~ 4*D*Np
        t_gemm = self.alpha_g * (2.0 * Tq * self.D * Np) / max(F_peak, 1e6)
        t_copy = self.alpha_c * (4.0 * self.D * Np) / max(B_peak, 1.0)
        return t_gemm, t_copy

    def _features(self, Tq, Np, stats: RecentStats):
        return np.array([
            1.0,
            float(Tq),
            math.log1p(float(Tq)),
            float(Np),
            math.log1p(float(Np)),
            float(Tq) * float(Np),
            float(stats.p99_gemm),
            float(stats.p99_pread),
            float(stats.dup_ratio),
            float(stats.seq_ratio),
        ], dtype=np.float64)

    def propose(self, Tq: int, stats: RecentStats) -> int:
        # 1) 비용모델 기반 초기치
        Np = int(max(self.min_cols, min(self.last_Np, self.max_cols)))
        t_gemm, t_copy = self._roofline(Tq, Np, stats.F_peak, stats.B_peak)
        ratio = t_copy / max(t_gemm, 1e-9)
        if ratio > 0.20: Np = int(Np * 0.75)
        elif ratio < 0.08: Np = int(Np * 1.25)
        Np = int(max(self.min_cols, min(Np, self.max_cols)))

        # 2) 후보 생성
        cand = sorted(set([
            Np,
            max(self.min_cols, min(self.max_cols, Np // 2)),
            max(self.min_cols, min(self.max_cols, int(Np * 1.5))),
        ]))

        # 3) 가드레일(모델 상 비율 필터)
        safe = []
        for c in cand:
            tg, tc = self._roofline(Tq, c, stats.F_peak, stats.B_peak)
            if (tc / max(tg, 1e-9)) <= self.copy_gemm_ratio_cap:
                safe.append(c)
        if not safe:
            safe = [self.min_cols]

        # 4) LinUCB로 선택 (쿨다운/히스테리시스)
        x_list = [self._features(Tq, c, stats) for c in safe]
        idx = self.bandit.select(x_list)
        act = int(safe[idx])

        if self.cooldown > 0:
            lim_low  = max(self.min_cols, int(self.last_Np * self.h_dn))
            lim_high = min(self.max_cols, int(self.last_Np * self.h_up))
            act = int(max(lim_low, min(lim_high, act)))
            self.cooldown -= 1

        return act

    def observe(self, Tq: int, Np: int, stats: RecentStats,
                panel_ms: float, p50_ms: float, p99_ms: float,
                copy_ms: float, gemm_ms: float, pread_ms: float,
                acc_drop: float = 0.0):
        # 보상(작을수록 좋음 → 음수)
        reward = - (self.w99 * p99_ms + self.w50 * p50_ms)
        if acc_drop > self.eps_acc:
            reward -= 100.0

        x = self._features(Tq, Np, stats)
        self.bandit.update(x, reward)

        # 비용모델 보정(EWMA 기반 상대 오차 보정)
        est_tg, est_tc = self._roofline(Tq, Np, stats.F_peak, stats.B_peak)
        k = 0.2
        if est_tg > 1e-9:
            self.alpha_g = float(np.clip(self.alpha_g * (1.0 + k*((gemm_ms/1000.0)/est_tg - 1.0)), 0.2, 5.0))
        if est_tc > 1e-9:
            self.alpha_c = float(np.clip(self.alpha_c * (1.0 + k*((copy_ms/1000.0)/est_tc - 1.0)), 0.2, 5.0))

        # 실효 F_peak/B_peak 추정 갱신(관측값으로 역추정)
        # FLOPs/s ≈ (2*Tq*D*Np)/t_gemm , Bytes/s ≈ (4*D*Np)/t_copy
        if gemm_ms > 0:
            F_obs = (2.0 * Tq * self.D * Np) / (gemm_ms/1000.0)
            self.ew_F.update(F_obs)
        if copy_ms > 0:
            B_obs = (4.0 * self.D * Np) / (copy_ms/1000.0)
            self.ew_B.update(B_obs)

        # 분포창 갱신
        self.pwin_panel.add(panel_ms)
        self.pwin_copy.add(copy_ms)
        self.pwin_gemm.add(gemm_ms)
        self.pwin_pread.add(pread_ms)

        # 다음 제안 안정화
        self.last_Np = int(Np)
        self.cooldown = max(self.cooldown, 3)

    def recent_stats(self, base: RecentStats) -> RecentStats:
        out = RecentStats(
            p99_gemm = self.pwin_gemm.p(99),
            p50_gemm = self.pwin_gemm.p(50),
            p99_pread = self.pwin_pread.p(99),
            p50_pread = self.pwin_pread.p(50),
            dup_ratio = base.dup_ratio,
            seq_ratio = base.seq_ratio,
            F_peak = self.ew_F.get(base.F_peak),
            B_peak = self.ew_B.get(base.B_peak),
        )
        return out

# =====================================
# Retriever
# =====================================
import neural_cherche.models as neural_cherche_models
import neural_cherche.rank as neural_cherche_rank

class ColbertFdeRetrieverNaive:
    def __init__(self,
        model_name: str = COLBERT_MODEL_NAME,
        rerank_candidates: int = RERANK_TOPN,
        enable_rerank: bool = True,
        save_doc_embeds: bool = True,
        latency_log_path: Optional[str] = None,
        external_doc_embeds_dir: Optional[str] = None,
        use_faiss_ann: bool = True,
        faiss_nlist: int = FAISS_NLIST,
        faiss_nprobe: int = FAISS_NPROBE,
        faiss_candidates: int = FAISS_CANDIDATES,
        faiss_num_threads: int = FAISS_NUM_THREADS,
        fde_dim: int = FDE_DIM,
        fde_reps: int = FDE_NUM_REPETITIONS,
        fde_simhash: int = FDE_NUM_SIMHASH,
        # TPack 옵션
        use_tpack: bool = True,
        build_tpack_if_missing: bool = False,
        tpack_iouring_qd: int = TPACK_IOURING_QD,
        tpack_iouring_submit: int = TPACK_IOURING_SUBMIT,
        # Auto Tuner
        autotune_panel: bool = True,
        panel_init_cols: int = DEFAULT_PANEL_INIT_COLS,
        panel_min_cols: int = PANEL_MIN_COLS,
        panel_max_cols: int = PANEL_MAX_COLS,
        copy_gemm_ratio_cap: float = COPY_GEMM_RATIO_CAP,
    ):
        self.faiss_num_threads = max(1, int(faiss_num_threads))
        model = neural_cherche_models.ColBERT(model_name_or_path=model_name, device=DEVICE)
        self.ranker = neural_cherche_rank.ColBERT(key="id", on=["title", "text"], model=model)

        self.doc_config = FixedDimensionalEncodingConfig(
            dimension=fde_dim, num_repetitions=fde_reps, num_simhash_projections=fde_simhash,
            seed=42, fill_empty_partitions=True,
        )

        self.fde_index: Optional[np.ndarray] = None
        self.doc_ids: List[str] = []
        self._doc_pos = {}
        self._corpus = None

        self.enable_rerank = enable_rerank
        self.rerank_candidates = rerank_candidates
        self.save_doc_embeds = save_doc_embeds
        self.external_doc_embeds_dir = external_doc_embeds_dir

        self.use_faiss_ann = use_faiss_ann and _FAISS_OK
        self.faiss_nlist = faiss_nlist
        self.faiss_nprobe = faiss_nprobe
        self.faiss_candidates = faiss_candidates
        self.faiss_index = None

        self._model_name = model_name
        self._cache_dir = os.path.join(CACHE_ROOT, DATASET_REPO_ID)

        self._fde_path = os.path.join(self._cache_dir, "fde_index.pkl")
        self._ids_path = os.path.join(self._cache_dir, "doc_ids.json")
        self._meta_path = os.path.join(self._cache_dir, "meta.json")
        self._queries_dir = os.path.join(self._cache_dir, "queries")
        self._doc_emb_dir = os.path.join(self._cache_dir, "doc_embeds")
        self._faiss_path = os.path.join(self._cache_dir, "faiss.index")

        os.makedirs(self._cache_dir, exist_ok=True)
        os.makedirs(self._queries_dir, exist_ok=True)
        if self.save_doc_embeds:
            os.makedirs(self._doc_emb_dir, exist_ok=True)

        self._latency_log_path = latency_log_path or os.path.join(self._cache_dir, "latency.tsv")
        try:
            if not os.path.exists(self._latency_log_path):
                with open(self._latency_log_path, "a", encoding="utf-8") as f:
                    f.write("qid\tann_ms\trerank_ms\trerank_compute_ms\trerank_io_ms\twait_ms\trerank_vstack_ms\n")
        except Exception:
            pass

        # tpack 상태
        self.use_tpack = bool(use_tpack)
        self.build_tpack_if_missing = bool(build_tpack_if_missing)
        self._tpack_dir = os.path.join(self._cache_dir, "tpack")
        self.tpack: Optional[TPackReader] = None
        self._tpack_qd = int(tpack_iouring_qd)
        self._tpack_submit = int(tpack_iouring_submit)

        # Auto-tuner
        self.autotune_panel = bool(autotune_panel)
        self._panel_min_cols = int(panel_min_cols)
        self._panel_max_cols = int(panel_max_cols)
        self._copy_gemm_ratio_cap = float(copy_gemm_ratio_cap)
        self._panel_tuner: Optional[PanelAutoTuner] = None

    def _set_faiss_threads(self):
        if not self.use_faiss_ann:
            return
        try:
            faiss.omp_set_num_threads(self.faiss_num_threads)
        except Exception:
            pass

    def _query_key(self, query_text: str, query_id: Optional[str]) -> str:
        base = (query_id or "") + "||" + query_text
        return hashlib.sha1(base.encode("utf-8")).hexdigest()

    def _query_paths(self, key: str) -> Tuple[str, str]:
        return (
            os.path.join(self._queries_dir, f"{key}.emb.npy"),
            os.path.join(self._queries_dir, f"{key}.fde.npy"),
        )

    def _external_doc_emb_path(self, doc_id: str) -> Optional[str]:
        if not self.external_doc_embeds_dir:
            return None
        pos = self._doc_pos.get(doc_id)
        if pos is None:
            return None
        return os.path.join(self._doc_emb_dir, f"{pos:08d}.npy")

    def _internal_doc_emb_path(self, doc_id: str) -> str:
        pos = self._doc_pos[doc_id]
        return os.path.join(self._doc_emb_dir, f"{pos:08d}.npy")

    def ensure_tpack(self):
        if not self.use_tpack:
            return
        os.makedirs(self._tpack_dir, exist_ok=True)
        need = not all(os.path.exists(os.path.join(self._tpack_dir, p)) for p in
                       ["tokensF.bin", "doc_col_ptrs.npy", "tpack_meta.json"])
        if need:
            if not self.build_tpack_if_missing:
                raise FileNotFoundError("[TPack] missing; pass --build_tpack_if_missing or prebuild.")
            builder = TPackBuilder(self._tpack_dir, dim=int(self.doc_config.dimension))
            builder.build(self, self.doc_ids)
        self.tpack = TPackReader(self._tpack_dir,
                                 use_iouring=True,
                                 iouring_qd=self._tpack_qd,
                                 iouring_submit_batch=self._tpack_submit)
        logging.info("[TPack] ready.")

    def _save_query_cache(self, key: str, query_embeddings: np.ndarray, query_fde: np.ndarray):
        emb_path, fde_path = self._query_paths(key)
        np.save(emb_path, query_embeddings)
        np.save(fde_path, query_fde)

    def _load_query_cache(self, key: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        emb_path, fde_path = self._query_paths(key)
        emb = np.load(emb_path) if os.path.exists(emb_path) else None
        fde = np.load(fde_path) if os.path.exists(fde_path) else None
        return emb, fde

    @staticmethod
    def _chamfer(query_tok: np.ndarray, doc_tok_BT: np.ndarray) -> float:
        if doc_tok_BT.size == 0:
            return -1e9
        S = query_tok @ doc_tok_BT  # (Tq, ncols)
        return float(S.max(axis=1).sum())

    def _get_doc_embeddings(self, doc_id: str, allow_build: bool = True) -> np.ndarray:
        ext_path = self._external_doc_emb_path(doc_id)
        if ext_path and os.path.exists(ext_path):
            arr = np.load(ext_path)
        else:
            int_path = self._internal_doc_emb_path(doc_id)
            if os.path.exists(int_path):
                arr = np.load(int_path)
            else:
                if not allow_build:
                    raise FileNotFoundError(ext_path or int_path)
                if self._corpus is None:
                    raise RuntimeError("Corpus not set.")
                doc = {"id": doc_id, **self._corpus[doc_id]}
                emap = self.ranker.encode_documents(documents=[doc])
                arr = to_numpy(emap[doc_id])
                np.save(int_path, arr)
        return arr

    def index(self, corpus: dict, panel_init_cols: int = DEFAULT_PANEL_INIT_COLS):
        self._corpus = corpus
        with open(self._ids_path, "r", encoding="utf-8") as f:
            self.doc_ids = json.load(f)
        self._doc_pos = {d: i for i, d in enumerate(self.doc_ids)}
        self.fde_index = joblib.load(self._fde_path)
        logging.info(f"[{self.__class__.__name__}] Loaded FDE index cache: {self.fde_index.shape} for {len(self.doc_ids)} docs")

        if self.use_faiss_ann and os.path.exists(self._faiss_path):
            try:
                self.faiss_index = faiss.read_index(self._faiss_path)
                self.faiss_index.nprobe = FAISS_NPROBE
            except Exception:
                self.faiss_index = None

        if self.use_tpack:
            self.ensure_tpack()

        # 오토튜너 준비
        if self.autotune_panel:
            self._panel_tuner = PanelAutoTuner(
                D=int(self.tpack.D if self.tpack else self.doc_config.dimension),
                init_cols=panel_init_cols,
                min_cols=self._panel_min_cols,
                max_cols=self._panel_max_cols,
                copy_gemm_ratio_cap=self._copy_gemm_ratio_cap,
            )

    def _build_or_load_faiss_index(self):
        if not self.use_faiss_ann:
            return
        if self.faiss_index is not None and os.path.exists(self._fde_path):
            return
        self._set_faiss_threads()
        dim   = int(self.fde_index.shape[1])
        nvecs = int(self.fde_index.shape[0])
        logging.info(f"[FAISS] Building IVFFlat(IP) nlist={FAISS_NLIST} for {nvecs} (dim={dim})")
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, FAISS_NLIST, faiss.METRIC_INNER_PRODUCT)
        index.train(self.fde_index.astype(np.float32, copy=False))
        index.add(self.fde_index.astype(np.float32, copy=False))
        faiss.write_index(index, self._faiss_path)
        index.nprobe = FAISS_NPROBE
        self.faiss_index = index

    def ann_search_batch(self, XQ_batch: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, float]:
        assert XQ_batch.ndim == 2
        if self.faiss_index is None:
            self._build_or_load_faiss_index()
        self._set_faiss_threads()
        t0 = time.perf_counter()
        D, I = self.faiss_index.search(XQ_batch, k)
        ann_time = time.perf_counter() - t0
        return D, I, ann_time

    def _log_latency(self, qid: str, search_s: float, ann_s: float, rerank_s: float,
                     rerank_compute_s: float, rerank_io_s: float, wait_s: float, vstack_s: float):
        try:
            with open(os.path.join(CACHE_ROOT, "latency.tsv"), "a", encoding="utf-8") as f:
                f.write(
                    f"{qid}\t{(ann_s/ANN_BATCH_SIZE)*1000:.3f}\t{rerank_s*1000:.3f}\t"
                    f"{rerank_compute_s*1000:.3f}\t{rerank_io_s*1000:.3f}\t{wait_s*1000:.3f}\t{vstack_s*1000:.3f}\n"
                )
        except Exception:
            pass

# ================= 오케스트레이션 =================
@dataclass
class AnnItem:
    qid: str
    qtext: str
    t_enqueue: float

@dataclass
class RerankTask:
    qid: str
    qtext: str
    query_embeddings: np.ndarray
    initial_candidates: List[Tuple[str, float]]
    search_time_s: float
    ann_time_s: float
    enqueued_time_s: float

def ann_aggregator_loop(retriever: ColbertFdeRetrieverNaive,
                        in_q: Queue, out_q: Queue,
                        k: int,
                        batch_size: int = ANN_BATCH_SIZE):
    exp_dim = int(retriever.fde_index.shape[1])
    XQ_list: List[np.ndarray] = []
    metas: List[Tuple[str, str, np.ndarray, float]] = []

    def flush():
        if not XQ_list: return
        XQb = np.vstack(XQ_list)
        D, I, ann_time = retriever.ann_search_batch(XQb, k)
        t_now = time.perf_counter()
        for i, (qid, qtext, qemb, t_enq) in enumerate(metas):
            mask = I[i] >= 0
            cand_ids = [retriever.doc_ids[idx] for idx in I[i][mask]]
            cand_scores = D[i][mask].tolist()
            initial_candidates = list(zip(cand_ids, cand_scores))
            task = RerankTask(qid=qid, qtext=qtext, query_embeddings=qemb,
                              initial_candidates=initial_candidates,
                              search_time_s=(t_now - t_enq), ann_time_s=ann_time,
                              enqueued_time_s=t_now)
            out_q.put(task)
        XQ_list.clear(); metas.clear()

    while True:
        item = in_q.get()
        if item == "__STOP__":
            flush()
            out_q.put("__STOP__")
            break

        qid, qtext, t_enq = item.qid, item.qtext, item.t_enqueue
        key = retriever._query_key(qtext, qid)
        qemb, qfde = retriever._load_query_cache(key)
        if (qemb is None) or (qfde is None) or (qfde.shape[0] != exp_dim):
            qmap = retriever.ranker.encode_queries(queries=[qtext])
            qemb = to_numpy(next(iter(qmap.values())))
            qcfg = replace(retriever.doc_config, fill_empty_partitions=False)
            qfde = generate_query_fde(qemb, qcfg)
            retriever._save_query_cache(key, qemb, qfde)

        XQ_list.append(np.ascontiguousarray(qfde.reshape(1, -1).astype(np.float32)))
        metas.append((qid, qtext, qemb, t_enq))
        if len(XQ_list) >= batch_size:
            flush()

# =====================================
# 핵심: 패널링 + 오토튜너가 들어간 재랭크 태스크
# =====================================
def _rerank_task_with_paneling_autotune(retriever: ColbertFdeRetrieverNaive, task: "RerankTask", top_k: int):
    """
    단계:
    1) 후보 문서 span 수집 → io_uring 병렬 pread
    2) 패널링(vstack 복사 有)으로 (D, panel_cols) 만든 뒤 단일 GEMM
    3) 패널 단위 메트릭 계측 → 오토튜너 observe()
    """
    start_time = time.perf_counter()
    q_emb = task.query_embeddings                      # [Tq, D]
    Tq, Dq = int(q_emb.shape[0]), int(q_emb.shape[1])

    N_compute = min(
        retriever.rerank_candidates if retriever.rerank_candidates > 0 else len(task.initial_candidates),
        len(task.initial_candidates)
    )
    compute_ids = [did for (did, _) in task.initial_candidates[:N_compute]]

    # TPack 없으면 메모리 경로
    if retriever.tpack is None:
        t_io0 = time.perf_counter()
        reranked_pairs: List[Tuple[str, float]] = []
        for did in compute_ids:
            Ddoc = retriever._get_doc_embeddings(did, allow_build=True).astype(np.float32, copy=False)  # (n, D)
            score = float((q_emb @ Ddoc.T).max(axis=1).sum()) if Ddoc.size else -1e9
            reranked_pairs.append((did, score))
        io_s = time.perf_counter() - t_io0
        compute_s = 0.0
        vstack_s = 0.0
        reranked_pairs.sort(key=lambda x: x[1], reverse=True)
        out = OrderedDict((did, float(sc)) for did, sc in reranked_pairs)
        total_s = io_s + compute_s
        meta = {}
        return out, total_s, compute_s, io_s, meta, vstack_s

    tp = retriever.tpack
    if int(tp.D) != Dq:
        logging.warning(f"[TPack] D mismatch (tp.D={tp.D} vs query.D={Dq}) → memory path fallback.")
        return _rerank_task_with_paneling_autotune(retriever=replace(retriever, tpack=None), task=task, top_k=top_k)

    # 1) span 수집 및 pread
    t_io0 = time.perf_counter()
    doc_spans: List[Tuple[str, int, int]] = []
    for did in compute_ids:
        di = retriever._doc_pos[did]
        s, e = tp.doc_col_span(di)
        doc_spans.append((did, s, e))

    doc_blocks: Dict[str, np.ndarray] = tp.pread_cols_many_uring(
        doc_spans, qd=retriever._tpack_qd, submit_batch=retriever._tpack_submit, verify=False
    ) # io_uring 병렬 read로 후보 문서들을 토큰 벡터 읽어옴
    io_s = time.perf_counter() - t_io0

    # 2) 패널링 + GEMM
    vstack_total = 0.0
    gemm_total   = 0.0
    panel_total  = 0.0
    pread_ms_for_panels = 0.0  # 이미 완료(=0)로 간주, residual은 0

    # 최근 통계/튜너
    tuner = retriever._panel_tuner
    use_tuner = (tuner is not None)
    base_stats = RecentStats(dup_ratio=0.0, seq_ratio=0.0)  # dup_ratio는 batch 수준에서 계산 가능
    if use_tuner:
        stats_for_prop = tuner.recent_stats(base_stats)
        target_panel_cols = tuner.propose(Tq=Tq, stats=stats_for_prop)
        # print(f"[Tuner] initial target_panel_cols={target_panel_cols}")
    else:
        target_panel_cols = DEFAULT_PANEL_INIT_COLS
        # print(f"[Tuner] disabled; using fixed target_panel_cols={target_panel_cols}")

    # 문서를 순회하며 패널 쌓기
    reranked_pairs: List[Tuple[str, float]] = []
    pending: List[Tuple[str, np.ndarray]] = []
    total_cols_in_panel = 0

    def _flush_panel():
        nonlocal vstack_total, gemm_total, panel_total, pread_ms_for_panels, pending, total_cols_in_panel
        if not pending:
            return
        # 패널 복사(vstack)
        t_v0 = time.perf_counter()
        panel_cols = sum(int(B.shape[1]) for _, B in pending)
        D = int(pending[0][1].shape[0])
        panelBT = np.empty((D, panel_cols), dtype=np.float32, order="F")  # 복사 有
        c = 0
        offsets = []
        for did, B in pending:
            n = int(B.shape[1])
            panelBT[:, c:c+n] = B
            offsets.append((did, c, c+n))
            c += n
        vstack_ms = (time.perf_counter() - t_v0) * 1000.0

        # GEMM
        t_g0 = time.perf_counter()
        S = q_emb @ panelBT  # (Tq, panel_cols)
        gemm_ms = (time.perf_counter() - t_g0) * 1000.0

        # 패널 전체 시간(복사+GEMM, I/O residual은 없음)
        panel_ms = vstack_ms + gemm_ms

        # 문서 스코어 집계
        for did, s0, s1 in offsets:
            score = float(S[:, s0:s1].max(axis=1).sum())
            reranked_pairs.append((did, score))

        # 누계
        vstack_total += vstack_ms/1000.0
        gemm_total   += gemm_ms/1000.0
        panel_total  += panel_ms/1000.0

        print(f"[Rerank] qid={task.qid} panel_cols={panel_cols} panel_ms={panel_ms:.3f}ms (vstack_ms={vstack_ms:.3f}ms, gemm_ms={gemm_ms:.3f}ms)")

        # 오토튜너 갱신
        if use_tuner:
            # 관측치 보고(정확도 하락은 여기선 0으로 가정)
            tuner.observe(
                Tq=Tq, Np=panel_cols, stats=stats_for_prop,
                panel_ms=panel_ms, p50_ms=panel_ms, p99_ms=panel_ms,  # 패널 1개 기준 → 동일
                copy_ms=vstack_ms, gemm_ms=gemm_ms, pread_ms=pread_ms_for_panels,
                acc_drop=0.0
            )

        # 리셋
        pending.clear()
        total_cols_in_panel = 0

    # doc_blocks 순서대로 패널 구성(순차성)
    for did in compute_ids:
        Bdoc = doc_blocks.get(did, None) # 128KB per Bdoc
        # print(f"Bdoc bytes: {Bdoc.nbytes if Bdoc is not None else 'None'} for did={did}")
        if Bdoc is None or Bdoc.size == 0:
            reranked_pairs.append((did, -1e9))
            continue
        n = int(Bdoc.shape[1])
        # 새 패널 시작 시점이면 튜너로 재결정(옵션)
        if use_tuner and total_cols_in_panel == 0:
            stats_for_prop = tuner.recent_stats(base_stats)
            target_panel_cols = tuner.propose(Tq=Tq, stats=stats_for_prop)
            # print(f"[Tuner] proposed target_panel_cols={target_panel_cols}")
        pending.append((did, Bdoc))
        total_cols_in_panel += n
        # print(f"[Rerank] qid={task.qid} pending panel_cols={total_cols_in_panel}")
        if total_cols_in_panel >= target_panel_cols:
            _flush_panel()
    # 남은 패널 처리
    _flush_panel()

    compute_s = gemm_total
    vstack_s  = vstack_total
    io_s = io_s  # pread 단계
    reranked_pairs.sort(key=lambda x: x[1], reverse=True)
    out = OrderedDict((did, float(sc)) for did, sc in reranked_pairs)
    meta = {"target_panel_cols": target_panel_cols, "autotune": use_tuner}
    total_s = time.perf_counter() - start_time
    # print(f"[Rerank] qid={task.qid} done rerank: total_s={total_s:.3f}s (io_s={io_s:.3f}s, compute_s={compute_s:.3f}s, vstack_s={vstack_s:.3f}s)")
    return out, total_s, compute_s, io_s, meta, vstack_s

# ================= 배치 Rerank 오케스트레이션 =================
def rerank_aggregator_loop(retriever: ColbertFdeRetrieverNaive,
                           in_q: Queue,
                           out_dict: Dict[str, OrderedDict],
                           batch_queries: int = RERANK_BATCH_QUERIES,
                           top_k: int = TOP_K,
                           num_workers: int = RERANK_WORKERS):
    results_lock = threading.Lock()

    def _commit_result(task: RerankTask, out_pairs: OrderedDict,
                       rerank_time: float, compute_rerank_time: float, io_rerank_time: float,
                       wait_s: float, vstack_s: float, dup_ratio: Optional[float] = None, meta: Optional[dict] = None):
        with results_lock:
            out_dict[task.qid] = out_pairs
        total_search_time = task.ann_time_s + rerank_time
        avg_search_time_list.append(total_search_time)
        avg_ann_time_list.append(task.ann_time_s/ANN_BATCH_SIZE)
        avg_rerank_time_list.append(rerank_time)
        avg_rerank_cp_list.append(compute_rerank_time)
        avg_rerank_io_list.append(io_rerank_time)
        avg_rerank_wait_list.append(wait_s)
        avg_vstack_time_list.append(vstack_s)
        if dup_ratio is not None:
            avg_dup_ratio_list.append(dup_ratio)
        retriever._log_latency(task.qid, total_search_time, task.ann_time_s,
                               rerank_time, compute_rerank_time, io_rerank_time, wait_s, vstack_s)

    def _process_task(task: RerankTask, dup_ratio_for_batch: Optional[float]):
        topN = retriever.rerank_candidates if retriever.rerank_candidates > 0 else len(task.initial_candidates)
        task = replace(task, initial_candidates=task.initial_candidates[:topN])
        t_start = time.perf_counter()
        wait_s = t_start - task.enqueued_time_s
        t0 = time.perf_counter()
        out_pairs, total_rerank_s, compute_s, io_s, meta, vstack_s = _rerank_task_with_paneling_autotune(retriever, task, top_k)
        rerank_time = time.perf_counter() - t0
        _commit_result(task, out_pairs, rerank_time, compute_s, io_s, wait_s, vstack_s,
                       dup_ratio=dup_ratio_for_batch, meta=meta)

    if BATCH_RERANK_MODE == "immediate":
        workers = []
        def worker_loop():
            while True:
                item = in_q.get()
                if item == "__STOP__":
                    in_q.put("__STOP__"); break
                _process_task(item, dup_ratio_for_batch=None)
        for _ in range(max(1, int(num_workers))):
            t = threading.Thread(target=worker_loop, daemon=True)
            t.start(); workers.append(t)
        for t in workers: t.join()
        return

    elif BATCH_RERANK_MODE == "batch":
        buffer: List[RerankTask] = []

        def _flush_batch(buf: List[RerankTask]):
            if not buf: return
            # 배치 중복률(캐시/재사용 근사)
            all_doc_ids: List[str] = []
            for task in buf:
                N_compute = min(top_k if top_k>0 else len(task.initial_candidates),
                                retriever.rerank_candidates if retriever.rerank_candidates>0 else len(task.initial_candidates),
                                len(task.initial_candidates))
                all_doc_ids.extend([did for (did, _) in task.initial_candidates[:N_compute]])
            total_docs = len(all_doc_ids)
            dup_ratio_for_batch = 0.0 if total_docs == 0 else 1.0 - (len(set(all_doc_ids)) / total_docs)

            parallelism = min(max(1, int(num_workers)), len(buf))
            with ThreadPoolExecutor(max_workers=parallelism, thread_name_prefix="rerank-batch") as ex:
                futures = [ex.submit(_process_task, t, dup_ratio_for_batch) for t in buf]
                for f in as_completed(futures):
                    _ = f.result()

        while True:
            item = in_q.get()
            if item == "__STOP__":
                _flush_batch(buffer)
                break
            buffer.append(item)
            if len(buffer) >= max(1, int(BATCH_RERANK_SIZE)):
                _flush_batch(buffer); buffer.clear()
        return
    else:
        raise ValueError(f"Unknown BATCH_RERANK_MODE={BATCH_RERANK_MODE}")

# ======================
# --- Main Script ------
# ======================
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        pass

    def _positive_int(x: str) -> int:
        v = int(x)
        if v <= 0:
            raise argparse.ArgumentTypeError("must be > 0")
        return v

    parser = argparse.ArgumentParser(description="ANN + TPack(io_uring) paneling rerank with auto-tuner")
    parser.add_argument("--num_simhash_projections", "--p", type=_positive_int, required=True)
    parser.add_argument("--num_repetitions", "--r", type=_positive_int, required=True)
    parser.add_argument("--nlist", "-nl", type=_positive_int, default=1000)
    parser.add_argument("--num_rerank_cand", "-rc", type=int, required=True)
    parser.add_argument("--topk", "-tk", type=int, required=True)

    # TPack 옵션
    parser.add_argument("--use_tpack", action="store_true")
    parser.add_argument("--build_tpack_if_missing", action="store_true")
    parser.add_argument("--tpack_iouring_qd", type=int, default=TPACK_IOURING_QD)
    parser.add_argument("--tpack_iouring_submit", type=int, default=TPACK_IOURING_SUBMIT)

    # Auto Tuner 옵션
    parser.add_argument("--autotune_panel", action="store_true")
    parser.add_argument("--panel_init_cols", type=int, default=DEFAULT_PANEL_INIT_COLS)
    parser.add_argument("--panel_min_cols", type=int, default=PANEL_MIN_COLS)
    parser.add_argument("--panel_max_cols", type=int, default=PANEL_MAX_COLS)
    parser.add_argument("--copy_gemm_ratio_cap", type=float, default=COPY_GEMM_RATIO_CAP)

    args, _ = parser.parse_known_args()

    P = args.num_simhash_projections
    R = args.num_repetitions
    argslist = args.nlist
    num_rank_candidates = args.num_rerank_cand
    number_of_topk = args.topk

    # 캐시 경로 준비
    data_path, corpus, queries, qrels = load_nanobeir_dataset()
    retriever = ColbertFdeRetrieverNaive(
        model_name=COLBERT_MODEL_NAME,
        rerank_candidates=num_rank_candidates,
        enable_rerank=True,
        save_doc_embeds=True,
        latency_log_path=os.path.join(CACHE_ROOT, "latency.tsv"),
        external_doc_embeds_dir=None,
        use_faiss_ann=True,
        faiss_nlist=FAISS_NLIST,
        faiss_nprobe=FAISS_NPROBE,
        faiss_candidates=FAISS_CANDIDATES,
        faiss_num_threads=FAISS_NUM_THREADS,
        fde_dim=FDE_DIM,
        fde_reps=FDE_NUM_REPETITIONS,
        fde_simhash=FDE_NUM_SIMHASH,
        # TPack
        use_tpack=args.use_tpack,
        build_tpack_if_missing=args.build_tpack_if_missing,
        tpack_iouring_qd=args.tpack_iouring_qd,
        tpack_iouring_submit=args.tpack_iouring_submit,
        # Auto Tuner
        autotune_panel=True,
        panel_init_cols=args.panel_init_cols,
        panel_min_cols=args.panel_min_cols,
        panel_max_cols=args.panel_max_cols,
        copy_gemm_ratio_cap=args.copy_gemm_ratio_cap,
    )

    # 캐시 파일명 (P,R에 맞춤)
    in_default = f"fde_index_{P}_{R}.pkl"
    faiss_default = f"ivf{argslist}_ip_{P}_{R}.faiss"
    meta_default = f"meta_{P}_{R}.json"

    retriever._fde_path  = os.path.join(retriever._cache_dir, in_default)
    retriever._faiss_path = os.path.join(retriever._cache_dir, faiss_default)
    retriever._meta_path = os.path.join(retriever._cache_dir, meta_default)

    t_ready0 = time.perf_counter()
    retriever.index(corpus, panel_init_cols=args.panel_init_cols)   # TPack/튜너 준비
    t_ready = time.perf_counter() - t_ready0
    logging.info(f"Retriever ready in {t_ready:.3f}s (tpack_used={retriever.tpack is not None}, D={retriever.tpack.D if retriever.tpack else 'NA'}, autotune={retriever.autotune_panel})")

    # 쿼리 사전계산
    missing = 0
    exp_dim = int(retriever.fde_index.shape[1])
    for qid, qtext in queries.items():
        key = retriever._query_key(qtext, str(qid))
        emb, fde = retriever._load_query_cache(key)
        if emb is None or fde is None or fde.shape[0] != exp_dim:
            qmap = retriever.ranker.encode_queries(queries=[qtext])
            qemb = to_numpy(next(iter(qmap.values())))
            qcfg = replace(retriever.doc_config, fill_empty_partitions=False)
            qfde = generate_query_fde(qemb, qcfg)
            retriever._save_query_cache(key, qemb, qfde)
            missing += 1
    logging.info(f"Precomputed queries: {missing} rebuilt")

    # BF Top-K 상한선 (평가용)
    BF_OUTFILE = os.path.join(CACHE_ROOT, f"{DATASET_REPO_ID}_bruteforce_top{number_of_topk}.tsv")
    compute_and_persist_bf_topk(retriever, queries, number_of_topk, BF_OUTFILE)
    bf_truth = load_bf_truth(BF_OUTFILE)

    # 파이프라인 스레드
    ann_in_q: Queue = Queue(maxsize=4096)
    rerank_in_q: Queue = Queue(maxsize=4096)
    results: Dict[str, OrderedDict] = {}

    ann_thr = threading.Thread(target=ann_aggregator_loop, args=(retriever, ann_in_q, rerank_in_q,
                                                                 max(FAISS_CANDIDATES, RERANK_TOPN),
                                                                 ANN_BATCH_SIZE),
                               daemon=True)
    rr_thr = threading.Thread(target=rerank_aggregator_loop, args=(retriever, rerank_in_q, results, RERANK_BATCH_QUERIES, number_of_topk),
                              daemon=True)

    start_time = time.perf_counter()
    ann_thr.start()
    rr_thr.start()

    for qid, qtext in queries.items():
        ann_in_q.put(AnnItem(qid=str(qid), qtext=qtext, t_enqueue=time.perf_counter()))

    ann_in_q.put("__STOP__")
    ann_thr.join()
    rerank_in_q.put("__STOP__")
    rr_thr.join()

    end_time = time.perf_counter() - start_time
    total_search_s = mean(avg_ann_time_list) + mean(avg_rerank_time_list) if avg_ann_time_list and avg_rerank_time_list else 0.0

    # BF 기반 평가
    sys_topk = system_topk_from_results(results, number_of_topk)
    bf_recall = recall_at_k_wrt_bf(sys_topk, bf_truth, number_of_topk)
    bf_hit = hit_at_k_wrt_bf(sys_topk, bf_truth, number_of_topk)
    bf_ndcg, _ = ndcg_at_k_wrt_bf(sys_topk, bf_truth, number_of_topk)

    _per_experiment_log_path = os.path.join(CACHE_ROOT, f"per_experiment_{DATASET_REPO_ID}")
    try:
        with open(_per_experiment_log_path, "a", encoding="utf-8") as f:
            f.write(
            f"Dataset: {DATASET_REPO_ID}, Queries: {len(queries)}, FirstCand: {FAISS_CANDIDATES} | "
            f"ANN_BATCH:{ANN_BATCH_SIZE}, RERANK_BATCH_Q:{RERANK_BATCH_QUERIES}, "
            f"RERANK_TOTAL: {mean(avg_rerank_time_list)*1000:.3f} | "
            f"RERANK_CAND:{num_rank_candidates}, Search: {total_search_s*1000:.3f} | "
            f"ANN_AVG: {mean(avg_ann_time_list)*1000:.3f} Rerank(CP)_AVG: {mean(avg_rerank_cp_list)*1000:.3f} | "
            f"Rerank(VS)_AVG: {mean(avg_vstack_time_list)*1000:.3f} | "
            f"Rerank(IO)_AVG: {mean(avg_rerank_io_list)*1000:.3f} | "
            f"RERANK(CP)_99P: {np.percentile(avg_rerank_cp_list, 99)*1000:.3f} | "
            f"RERANK(IO)_99P: {np.percentile(avg_rerank_io_list, 99)*1000:.3f} | "
            f"Recall@{number_of_topk}(BF): {bf_recall:.3f}, nDCG@{number_of_topk}(BF): {bf_ndcg:.3f}\n"
        )
    except Exception as e:
        logging.warning(f"Failed to write per-experiment log: {e}")