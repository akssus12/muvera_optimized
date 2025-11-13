# -*- coding: utf-8 -*-
"""
Naive-baseline pipeline (deadline 제거, 고정 배치 크기 100):
  - ANN: 100개 모이면 XQ를 묶어 faiss_index.search(XQ, k)
  - Rerank: 100개 모이면 배치 시작(내부는 문서별 순차 Chamfer; 대형 GEMM/세그먼트 리듀스 없음)

실험 파라미터:
  TARGET_NUM_QUERIES, RANDOM_SEED
  ANN_BATCH_SIZE = 100
  RERANK_BATCH_QUERIES = 100
  RERANK_TOPN (= rerank_candidates)
  OMP/BLAS thread caps (기본 1)
"""
import os, json, time, hashlib, logging, pathlib, random, threading, argparse, sys, resource, csv, math, heapq
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Optional, List, Tuple, Dict
from queue import Queue
from statistics import mean
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice

import nltk
import numpy as np
import torch
import joblib

import neural_cherche.models as neural_cherche_models
import neural_cherche.rank as neural_cherche_rank

from beir import util
from beir.datasets.data_loader import GenericDataLoader

from threading import Lock
try:
    # 선택: BLAS 스레드 과다 생성 방지 (NumPy/OpenBLAS/MKL)
    from threadpoolctl import threadpool_limits
    # BLAS thread 비활성화
    _TPCTL = threadpool_limits(limits=1)  # 프로그램 시작 시
    _TPCTL_OK = True
except Exception:
    _TPCTL_OK = False

# ---- 환경 제한(권장: 내부 라이브러리 멀티스레딩 억제) ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ========== Faiss (CPU) ==========
try:
    import faiss  # pip install faiss-cpu
    _FAISS_OK = True
except Exception as _e:
    _FAISS_OK = False
    faiss = None

# ======================
# --- Configuration ----
# ======================
DATASET_REPO_ID = "treccovid"
COLBERT_MODEL_NAME = "raphaelsty/neural-cherche-colbert"
TOP_K = 10

# 데이터셋 경로
dataset = "trec-covid" # fiqa, arguana, scidocs, treccovid, quora
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

# ----- Rerank 워커 개수 -----
RERANK_WORKERS = 1  # 코어/메모리/I/O 상황에 맞춰 조정

# ----- 실험 스케일 -----
TARGET_NUM_QUERIES = 100
RANDOM_SEED = 42

# ----- ANN 배치 (고정 크기) -----
FAISS_NLIST = 1000
FAISS_NPROBE = 50
FAISS_CANDIDATES = 100        # over-fetch; rerank보다 크거나 같게 권장
FAISS_NUM_THREADS = 1         # OpenMP 스레드 수(권장: 1 또는 소수)

# ----- Rerank 배치(나이브, 고정 크기) -----
ANN_BATCH_SIZE = 12          # ← 100개 모이면 배치 검색 (지금은 4로 운영)
RERANK_BATCH_QUERIES = 12      # ← 100개 모이면 배치 시작(현 코드에서는 즉시 처리였음)
BATCH_RERANK_SIZE = 12         # 배치 모을 크기
RERANK_TOPN = 0               # top-N만 재랭크 (0이면 재랭크 없음)

# ====== Rerank Batch Mode Switches ======
# 'immediate' : 쿼리 도착 즉시 Rerank (워커 병렬, mega GEMM)
# 'batch'     : Rerank 작업을 BATCH_RERANK_SIZE 개 모아 한 번에 멀티스레드 나이브 rerank
BATCH_RERANK_MODE = "batch"   # "immediate" or "batch"

# ----- 캐시(기본 끔: 정말 나이브) -----
ENABLE_DOC_EMB_LRU_CACHE = False
DOC_EMB_LRU_SIZE = 0          # 0이면 캐시 안씀

# ----- FDE 설정(예: 1024 차원) -----
FDE_DIM = 128
FDE_NUM_REPETITIONS = 2
FDE_NUM_SIMHASH = 3

# ----- 디바이스 -----
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# 캐시 루트
CACHE_ROOT = os.path.join(pathlib.Path(__file__).parent.absolute(), "cache_muvera")
os.makedirs(CACHE_ROOT, exist_ok=True)

# ======================
# --- Logging Setup ----
# ======================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info(f"Using device: {DEVICE}  |  FAISS={'on' if _FAISS_OK else 'off'}")
logging.info(f"[RERANK MODE] {BATCH_RERANK_MODE} (batch_size={BATCH_RERANK_SIZE})")

# ======================
# --- Metric Setup ----
# ======================
avg_search_time_list = []
avg_ann_time_list = []
avg_rerank_time_list = []
avg_rerank_cp_list = []
avg_rerank_io_list = []
avg_rerank_wait_list = []
avg_dup_ratio_list = []
avg_vstack_time_list = []

# ===========================
# --- Helper Functions  -----
# ===========================
def load_nanobeir_dataset(repo_id: str):
    logging.info(f"Loading dataset from local path (BEIR): '{repo_id}'...")
    corpus, queries, _ = GenericDataLoader(data_folder=data_path).load(split="test")
    target_queries = dict(islice(queries.items(), TARGET_NUM_QUERIES))

    # ---- qrels: test.tsv에서 score>0만 포함 ----
    candidates = [
        os.path.join(data_path, "qrels", "test.tsv"),
        os.path.join(data_path, "test.tsv"),
    ]
    qrels_pos = {}
    tsv_path = next((p for p in candidates if os.path.exists(p)), None)

    if tsv_path is None:
        logging.warning("[qrels] test.tsv not found; falling back to BEIR loader qrels (may already be filtered).")
        _, _, qrels_beir = GenericDataLoader(data_folder=data_path).load(split="test")
        qrels_pos = qrels_beir
    else:
        with open(tsv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            def _get(row, *keys):
                for k in keys:
                    if k in row:
                        return row[k]
                raise KeyError(f"Missing any of keys {keys} in row: {row}")

            kept, skipped = 0, 0
            for row in reader:
                try:
                    qid   = _get(row, "query-id", "qid", "query_id")
                    docid = _get(row, "corpus-id", "docid", "doc_id")
                    score = int(_get(row, "score", "label"))
                except Exception as e:
                    logging.warning(f"[qrels] skip malformed row: {e}")
                    continue

                if score > 0:
                    qrels_pos.setdefault(str(qid), {})[str(docid)] = 1
                    kept += 1
                else:
                    skipped += 1

        logging.info(f"[qrels] loaded from {tsv_path}: kept positives={kept}, skipped non-positives={skipped}")

    logging.info(f"Dataset loaded: {len(corpus)} documents, {len(target_queries)} queries, "
                 f"{sum(len(v) for v in qrels_pos.values())} positive qrels.")
    return corpus, target_queries, qrels_pos

# === (NEW) Per-query Recall@K ===
def per_query_recall_at_k(results: dict, qrels: dict, k: int) -> Dict[str, float]:
    from itertools import islice
    recalls: Dict[str, float] = {}
    for qid, ranked_docs in results.items():
        qid_str = str(qid)
        rel = set(qrels.get(qid_str, {}).keys())
        if not rel:
            continue

        if isinstance(ranked_docs, OrderedDict):
            topk_ids = set(islice(ranked_docs.keys(), k))
        else:
            topk_ids = set(
                doc for doc, _ in sorted(
                    ranked_docs.items(), key=lambda x: x[1], reverse=True
                )[:k]
            )

        hit_rel = rel.intersection(topk_ids)
        recall = len(hit_rel) / len(rel)
        recalls[qid_str] = recall
    return recalls

def evaluate_hit_k(results: dict, qrels: dict, k: int) -> float:
    hits, total_queries = 0, 0
    for query_id, ranked_docs in results.items():
        relevant_docs = set(qrels.get(str(query_id), {}).keys())
        if not relevant_docs:
            continue
        total_queries += 1
        top_k_docs = set(list(ranked_docs.keys())[:k])
        if not relevant_docs.isdisjoint(top_k_docs):
            hits += 1
    return hits / total_queries if total_queries > 0 else 0.0

# === (nDCG with qrels) ===
def _dcg_at_k(ranked_docids: List[str], qrels_q: Dict[str, int], k: int) -> float:
    dcg = 0.0
    for i, did in enumerate(ranked_docids[:k], start=1):
        rel = int(qrels_q.get(str(did), 0))
        if rel <= 0:
            continue
        gain = (2 ** rel) - 1
        dcg += gain / math.log2(i + 1)
    return dcg

def per_query_ndcg_at_k(results: Dict[str, OrderedDict], qrels: Dict[str, Dict[str, int]], k: int) -> Dict[str, float]:
    ndcgs: Dict[str, float] = {}
    for qid, ranked_docs in results.items():
        qrels_q = qrels.get(str(qid), {})
        if not any(int(v) > 0 for v in qrels_q.values()):
            continue

        if isinstance(ranked_docs, OrderedDict):
            topk_ids = list(islice(ranked_docs.keys(), k))
        else:
            topk_ids = [doc for doc, _ in sorted(ranked_docs.items(), key=lambda x: x[1], reverse=True)[:k]]

        dcg = _dcg_at_k(topk_ids, qrels_q, k)
        ideal_gains = sorted([int(v) for v in qrels_q.values() if int(v) > 0], reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal_gains[:k], start=1):
            gain = (2 ** rel) - 1
            idcg += gain / math.log2(i + 1)
        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        ndcgs[qid] = ndcg
    return ndcgs

def to_numpy(tensor_or_array) -> np.ndarray:
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.cpu().detach().numpy().astype(np.float32)
    elif isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array.astype(np.float32)
    else:
        raise TypeError(f"Unsupported type for conversion: {type(tensor_or_array)}")

# 랜덤 I/O 계측 헬퍼
def _read_proc_io_bytes() -> Optional[int]:
    try:
        with open("/proc/self/io", "r") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    return int(line.split()[1])
    except Exception:
        return None
    return None

def _get_rusage_faults():
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return int(ru.ru_minflt), int(ru.ru_majflt)
    except Exception:
        return None, None

# =====================================
# --- FDE Query/Doc Generator Stubs  ---
# =====================================
from fde_generator_optimized_stream import (
    FixedDimensionalEncodingConfig,
    generate_query_fde,
    generate_document_fde_batch,
)

# =====================================
# --- Retriever (ANN 배치 + Rerank) ---
# =====================================
class ColbertFdeRetrieverNaive:
    def __init__(
        self,
        model_name: str = COLBERT_MODEL_NAME,
        rerank_candidates: int = RERANK_TOPN,
        enable_rerank: bool = True,
        save_doc_embeds: bool = True,
        latency_log_path: Optional[str] = None,
        external_doc_embeds_dir: Optional[str] = None,
        # FAISS params
        use_faiss_ann: bool = True,
        faiss_nlist: int = FAISS_NLIST,
        faiss_nprobe: int = FAISS_NPROBE,
        faiss_candidates: int = FAISS_CANDIDATES,
        faiss_num_threads: int = FAISS_NUM_THREADS,
        # FDE config
        fde_dim: int = FDE_DIM,
        fde_reps: int = FDE_NUM_REPETITIONS,
        fde_simhash: int = FDE_NUM_SIMHASH,
    ):
        self.faiss_num_threads = max(1, int(faiss_num_threads))
        model = neural_cherche_models.ColBERT(model_name_or_path=model_name, device=DEVICE)
        self.ranker = neural_cherche_rank.ColBERT(key="id", on=["title", "text"], model=model)

        self.doc_config = FixedDimensionalEncodingConfig(
            dimension=fde_dim,
            num_repetitions=fde_reps,
            num_simhash_projections=fde_simhash,
            seed=42,
            fill_empty_partitions=True,
        )

        self.fde_index: Optional[np.ndarray] = None
        self.doc_ids: List[str] = []
        self._doc_pos = {}
        self._corpus = None

        self.enable_rerank = enable_rerank
        self.rerank_candidates = num_rank_candidates
        self.save_doc_embeds = save_doc_embeds
        self.external_doc_embeds_dir = external_doc_embeds_dir

        # FAISS
        self.use_faiss_ann = use_faiss_ann and _FAISS_OK
        self.faiss_nlist = faiss_nlist
        self.faiss_nprobe = faiss_nprobe
        self.faiss_candidates = faiss_candidates
        self.faiss_index = None

        # cache paths
        self._model_name = model_name
        self._cache_dir = self._compute_cache_dir(dataset=DATASET_REPO_ID)

        # (주의) in_default/meta_default/faiss_default는 main에서 설정됨
        self._fde_path = os.path.join(self._cache_dir, in_default)
        self._ids_path = os.path.join(self._cache_dir, "doc_ids.json")
        self._meta_path = os.path.join(self._cache_dir, meta_default)
        self._queries_dir = os.path.join(self._cache_dir, "queries")
        self._doc_emb_dir = os.path.join(self._cache_dir, "doc_embeds")
        self._faiss_path = os.path.join(self._cache_dir, faiss_default)

        os.makedirs(self._cache_dir, exist_ok=True)
        os.makedirs(self._queries_dir, exist_ok=True)
        if self.save_doc_embeds:
            os.makedirs(self._doc_emb_dir, exist_ok=True)

        self._latency_log_path = latency_log_path or os.path.join(self._cache_dir, "latency.tsv")
        self._log_lock = threading.Lock()

        # (선택) 작은 LRU 캐시 – 기본 비활성
        self._lru_enabled = ENABLE_DOC_EMB_LRU_CACHE and DOC_EMB_LRU_SIZE > 0
        if self._lru_enabled:
            from collections import OrderedDict as _OD
            self._lru = _OD()
            self._lru_lock = threading.Lock()
            self._lru_cap = DOC_EMB_LRU_SIZE
        
        # 헤더 기록
        try:
            with self._log_lock:
                if not os.path.exists(self._latency_log_path):
                    with open(self._latency_log_path, "a", encoding="utf-8") as f:
                        f.write(
                            "qid\tann_ms\trerank_ms\trerank_compute_ms\trerank_io_ms\t"
                            "wait_ms\trerank_vstack_ms\n"
                        )
        except Exception as e:
            logging.warning(f"[{self.__class__.__name__}] Failed to write latency header: {e}")
        
        # per-query recall 로깅 파일
        self._per_query_log_path = os.path.join(CACHE_ROOT, f"per_query_{TOP_K}.tsv")
        try:
            with self._log_lock:                
                if not os.path.exists(self._per_query_log_path):                    
                    with open(self._per_query_log_path, "a", encoding="utf-8") as f:                        
                        f.write("qid\trecall_at_k\n")
        except Exception as e:
            logging.warning(f"[{self.__class__.__name__}] Failed to write per-query header: {e}")

        # 실험 로깅파일
        # self._per_experiment_log_path = os.path.join(CACHE_ROOT, f"per_experiment_{DATASET_REPO_ID}")
        # try:
        #     with self._log_lock:                
        #         if not os.path.exists(self._per_experiment_log_path):                    
        #             with open(self._per_experiment_log_path, "a", encoding="utf-8") as f:                        
        #                 f.write("qid\trecall_at_k\n")
        # except Exception as e:
        #     logging.warning(f"[{self.__class__.__name__}] Failed to write per-query header: {e}")        

    def _compute_cache_dir(self, dataset: str) -> str:
        return os.path.join(CACHE_ROOT, dataset)

    def _set_faiss_threads(self):
        if not self.use_faiss_ann:
            return
        try:
            faiss.omp_set_num_threads(self.faiss_num_threads)
        except Exception as e:
            logging.warning(f"[FAISS] omp_set_num_threads failed: {e}")

    def _query_key(self, query_text: str, query_id: Optional[str]) -> str:
        base = (query_id or "") + "||" + query_text
        return hashlib.sha1(base.encode("utf-8")).hexdigest()

    def _query_paths(self, key: str) -> Tuple[str, str]:
        return (
            os.path.join(self._queries_dir, f"{key}.emb.npy"),
            os.path.join(self._queries_dir, f"{key}.fde.npy"),
        )

    def _doc_emb_path(self) -> str:
        raise NotImplementedError  # 내부 pos 기반 파일 이름 사용

    def _external_doc_emb_path(self, doc_id: str) -> Optional[str]:
        if not self.external_doc_embeds_dir:
            return None
        pos = self._doc_pos.get(doc_id)
        if pos is None:
            return None
        return os.path.join(self.external_doc_embeds_dir, f"{pos:08d}.npy")

    def _internal_doc_emb_path(self, doc_id: str) -> str:
        pos = self._doc_pos[doc_id]        
        return os.path.join(self._doc_emb_dir, f"{pos:08d}.npy")

    def _load_cache(self) -> bool:
        self.fde_index = joblib.load(self._fde_path)
        with open(self._ids_path, "r", encoding="utf-8") as f:
            self.doc_ids = json.load(f)
        self._doc_pos = {d: i for i, d in enumerate(self.doc_ids)}
        logging.info(f"[{self.__class__.__name__}] Loaded FDE index cache: "
                     f"{self.fde_index.shape} for {len(self.doc_ids)} docs")

        # meta로 실제 FDE 설정 동기화
        try:
            with open(self._meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            cfg = meta.get("config", {})
            self.doc_config = FixedDimensionalEncodingConfig(
                dimension=cfg.get("dimension", self.doc_config.dimension),
                num_repetitions=cfg.get("num_repetitions", self.doc_config.num_repetitions),
                num_simhash_projections=cfg.get("num_simhash_projections", self.doc_config.num_simhash_projections),
                seed=cfg.get("seed", self.doc_config.seed),
                fill_empty_partitions=cfg.get("fill_empty_partitions", self.doc_config.fill_empty_partitions),
            )
            logging.info(f"[FDE] Synchronized doc_config to cache meta: {self.doc_config}")
        except Exception as e:
            logging.warning(f"[FDE] Could not read meta.json; keep current doc_config. ({e})")

        # FAISS 로드 & 차원 검증
        if self.use_faiss_ann and os.path.exists(self._faiss_path):
            try:
                self.faiss_index = faiss.read_index(self._faiss_path)
                self.faiss_index.nprobe = self.faiss_nprobe
                if hasattr(self.faiss_index, "d") and self.faiss_index.d != int(self.fde_index.shape[1]):
                    logging.warning(f"[FAISS] dim mismatch: index.d={self.faiss_index.d} vs FDE={self.fde_index.shape[1]} ⇒ rebuild")
                    self.faiss_index = None
            except Exception as e:
                logging.warning(f"[FAISS] load failed: {e}")
                self.faiss_index = None
        return True

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
    def _chamfer(query_tok: np.ndarray, doc_tok: np.ndarray) -> float:
        sim = query_tok @ doc_tok.T
        return float(sim.max(axis=1).sum())

    def _get_doc_embeddings(self, doc_id: str, allow_build: bool = True) -> np.ndarray:
        # if self._lru_enabled:
        #     with self._lru_lock:
        #         if doc_id in self._lru:
        #             arr = self._lru.pop(doc_id)
        #             self._lru[doc_id] = arr
        #             return arr

        ext_path = self._external_doc_emb_path(doc_id)
        if ext_path and os.path.exists(ext_path):
            # read_start = time.perf_counter()
            arr = np.load(ext_path, mmap_mode='r')
            # print(f"[Doc Emb] Loaded external embed for doc_id={doc_id} from {ext_path} in {time.perf_counter() - read_start:.3f}s")
        else:
            int_path = self._internal_doc_emb_path(doc_id)            
            if os.path.exists(int_path):
                arr = np.load(int_path, mmap_mode='r')
            else:
                if not allow_build:
                    raise FileNotFoundError(ext_path or int_path)
                if self._corpus is None:
                    raise RuntimeError("Corpus not set.")
                doc = {"id": doc_id, **self._corpus[doc_id]}
                emap = self.ranker.encode_documents(documents=[doc])
                arr = to_numpy(emap[doc_id])
                np.save(int_path, arr)

        # if self._lru_enabled:
        #     with self._lru_lock:
        #         self._lru[doc_id] = arr
        #         if len(self._lru) > self._lru_cap:
        #             self._lru.popitem(last=False)
        return arr

    # per_query logging
    def _log_per_query(self, qid: str, recall_at_k: int):
        try:
            with self._log_lock:
                with open(self._per_query_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{qid}\t{recall_at_k}\n")
        except Exception as e:
            logging.warning(f"[{self.__class__.__name__}] Failed to write per-query row: {e}")
    
    def _log_latency(self, qid: str, search_s: float, ann_s: float, rerank_s: float,
                     rerank_compute_s: float, rerank_io_s: float, wait_s: float, vstack_s: float,
                     dup_ratio: Optional[float] = None):
        try:
            divided_ann_s = ann_s / ANN_BATCH_SIZE
            dr = -1.0 if (dup_ratio is None) else float(dup_ratio)            
            with self._log_lock:
                with open(self._latency_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{qid}\t{divided_ann_s*1000:.3f}\t{rerank_s*1000:.3f}\t"
                        f"{rerank_compute_s*1000:.3f}\t{rerank_io_s*1000:.3f}\t{wait_s*1000:.3f}\t{vstack_s*1000:.3f}\n"
                    )
        except Exception as e:
            logging.warning(f"[{self.__class__.__name__}] Failed to write latency log: {e}")

    def _build_or_load_faiss_index(self):
        if not self.use_faiss_ann:
            return
        if self.faiss_index is not None and os.path.exists(self._fde_path):
            return
        self._set_faiss_threads()
        dim   = int(self.fde_index.shape[1])
        nvecs = int(self.fde_index.shape[0])

        logging.info(f"[FAISS] Building IVFFlat(IP) nlist={self.faiss_nlist} for {nvecs} vectors (dim={dim})")
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, self.faiss_nlist, faiss.METRIC_INNER_PRODUCT)

        # Train
        try:
            x_train = self.fde_index
            if x_train.dtype != np.float32 or not x_train.flags["C_CONTIGUOUS"]:
                raise RuntimeError("Training would copy; fallback.")
            index.train(x_train)
        except Exception as e:
            logging.warning(f"[FAISS] train all failed: {e}; sampling.")
            min_per_list = 100
            target_train = max(self.faiss_nlist * min_per_list, 50_000)
            train_size = min(nvecs, target_train)
            rng = np.random.default_rng(42)
            train_x = np.empty((train_size, dim), dtype=np.float32, order="C")
            filled = 0
            block = 65536
            for start in range(0, nvecs, block):
                end = min(start + block, nvecs)
                Xblk = self.fde_index[start:end]
                need = train_size - filled
                if need <= 0:
                    break
                p = need / (nvecs - start)
                sel = np.nonzero(rng.random(end - start) < p)[0]
                k = min(need, sel.size)
                if k > 0:
                    train_x[filled:filled + k] = Xblk[sel[:k]]
                    filled += k
            if filled < train_size:
                need = train_size - filled
                step = max(1, nvecs // need)
                pos = 0
                for i in range(need):
                    train_x[filled + i] = self.fde_index[pos]
                    pos = (pos + step) % nvecs
            index.train(train_x)

        # Add
        add_bs = 100_000
        for start in range(0, nvecs, add_bs):
            end = min(start + add_bs, nvecs)
            xb = np.ascontiguousarray(self.fde_index[start:end], dtype=np.float32)
            index.add(xb)

        faiss.write_index(index, self._faiss_path)
        index.nprobe = self.faiss_nprobe
        self.faiss_index = index

    # --------- Public API ---------
    def index(self, corpus: dict):
        self._corpus = corpus
        if self._load_cache():
            if self.use_faiss_ann and self.faiss_index is None:
                try:
                    self._build_or_load_faiss_index()
                except Exception as e:
                    logging.warning(f"[FAISS] Build/load skipped: {e}")
            return
        raise RuntimeError("fde_index cache missing; create fde_index.pkl first.")

    def precompute_queries(self, queries: dict):
        missing = 0
        exp_dim = int(self.fde_index.shape[1])
        for qid, qtext in queries.items():
            key = self._query_key(qtext, str(qid))
            emb, fde = self._load_query_cache(key)
            need = (emb is None or fde is None or (fde is not None and fde.shape[0] != exp_dim))
            if need:
                if fde is not None and fde.shape[0] != exp_dim:
                    logging.warning(f"[QueryCache] dim mismatch qid={qid}: {fde.shape[0]} vs {exp_dim}. Rebuild.")
                qmap = self.ranker.encode_queries(queries=[qtext])
                qemb = to_numpy(next(iter(qmap.values())))
                qcfg = replace(self.doc_config, fill_empty_partitions=False)
                qfde = generate_query_fde(qemb, qcfg)
                self._save_query_cache(key, qemb, qfde)
                missing += 1
        logging.info(f"[{self.__class__.__name__}] Precomputed {missing} queries.")

    # ----- 배치 ANN (XQ 한 번에) -----
    def ann_search_batch(self, XQ_batch: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, float]:
        assert XQ_batch.ndim == 2
        if self.faiss_index is None:
            self._build_or_load_faiss_index()
        self._set_faiss_threads()
        t0 = time.perf_counter()
        D, I = self.faiss_index.search(XQ_batch, k)
        ann_time = time.perf_counter() - t0
        return D, I, ann_time

# ============== 헬퍼: 병렬 문서 읽기 후 vstack + BigGEMM ==============
def _rerank_task_with_mega_gemm_parallel(
    retriever: ColbertFdeRetrieverNaive,
    task: "RerankTask",
    top_k: int,
) -> Tuple[OrderedDict, float, float, float, float, dict, float]:
    """
    병렬 후보 문서 읽기 -> np.vstack -> big GEMM(MaxSim) -> 정렬
    """
    q_emb = task.query_embeddings

    # 몇 개를 재랭크할지 결정(외부 인자 사용)
    N_compute = num_rank_candidates
    compute_ids = [did for (did, _) in task.initial_candidates[:N_compute]]
    max_workers = max(1, RERANK_WORKERS)
    print(f"[biggemm] N_compute={N_compute}, max_workers={max_workers}")

    # -------- 병렬 문서 임베딩 로드 --------
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: List[Tuple[int, str, np.ndarray, int, float]] = []
    
    def _load_one(idx_did: Tuple[int, str]):
        idx, did = idx_did        
        d_tok = retriever._get_doc_embeddings(did, allow_build=True)        
        # 메모리 레이아웃에 민감하므로 float32 보장만 하고 복사는 vstack에서 일괄 수행
        n_i = int(d_tok.shape[0])
        return idx, did, d_tok, n_i
    
    io_start_s = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="doc-io") as ex:
        futures = [ex.submit(_load_one, (i, did)) for i, did in enumerate(compute_ids)]
        for fut in as_completed(futures):
            idx, did, arr, n_i = fut.result()                
            results.append((idx, did, arr, n_i))
    io_s = time.perf_counter() - io_start_s

    # 원래 순서를 보존하며 blocks / doc_spans 구성
    t_vstack0 = time.perf_counter()

    results.sort(key=lambda x: x[0])
    blocks: List[np.ndarray] = []
    doc_spans: List[Tuple[str, int, int]] = []
    col_start = 0
    for _, did, d_tok, n_i in results:
        blocks.append(d_tok)
        doc_spans.append((did, col_start, col_start + n_i))
        col_start += n_i

    # -------- vstack (연속 메모리 확보) --------
    D_all = None
    if blocks:
        # [sum(n_i), d], C-contiguous float32
        D_all = np.ascontiguousarray(np.vstack(blocks).astype(np.float32))
    vstack_s = time.perf_counter() - t_vstack0

    # -------- big GEMM + segment max(sum of per-query-token max) --------
    reranked_pairs: List[Tuple[str, float]] = []
    t_c0 = time.perf_counter()
    if D_all is not None and D_all.size > 0:
        # q_emb: [Q_tokens, d], D_all: [sum(n_i), d]
        S = q_emb @ D_all.T
        for did, s, e in doc_spans:
            if e - s == 0:
                score = -1e9
            else:
                score = float(S[:, s:e].max(axis=1).sum())
            reranked_pairs.append((did, score))
    compute_s = time.perf_counter() - t_c0

    # -------- 정렬 및 꼬리 처리 --------
    t_sort0 = time.perf_counter()
    reranked_pairs.sort(key=lambda x: x[1], reverse=True)
    computed_set = {did for (did, _) in reranked_pairs}
    tail_pairs = [(did, sc) for (did, sc) in task.initial_candidates if did not in computed_set]

    out = OrderedDict()
    for did, sc in reranked_pairs:
        out[did] = float(sc)
    for did, sc in tail_pairs:
        out[did] = float(sc)

    sort_s = time.perf_counter() - t_sort0
    total_s = io_s + compute_s + sort_s

    meta = dict(
        io_workers=min(max(1, int(RERANK_WORKERS)), len(compute_ids)),
        loaded_docs=len(results),
        stacked_rows=int(D_all.shape[0]) if D_all is not None else 0,
    )
    return out, total_s, compute_s, io_s, sort_s, meta, vstack_s

# ============== 공통: 하나의 task를 대형 GEMM + 세그먼트-리듀스로 재랭크 ==============
def _rerank_task_with_mega_gemm(
    retriever: ColbertFdeRetrieverNaive,
    task: "RerankTask",
    top_k: int,
) -> Tuple[OrderedDict, float, float, float, float, dict]:
    q_emb = task.query_embeddings
    # N_compute = min(top_k if top_k > 0 else len(task.initial_candidates),
    #                 retriever.rerank_candidates if retriever.rerank_candidates > 0 else len(task.initial_candidates),
    #                 len(task.initial_candidates))
    # compute_ids = [did for (did, _) in task.initial_candidates[:N_compute]]
    N_compute = num_rank_candidates
    compute_ids = [did for (did, _) in task.initial_candidates[:N_compute]]    

    io_before = _read_proc_io_bytes()
    mf0, M0   = _get_rusage_faults()
    pos_seq: List[int] = []

    doc_spans: List[Tuple[str, int, int]] = []
    blocks = []
    col_start = 0
    t_io0 = time.perf_counter()
    for did in compute_ids:        
        d_tok = retriever._get_doc_embeddings(did, allow_build=True)
        n_i = int(d_tok.shape[0])
        blocks.append(d_tok)
        doc_spans.append((did, col_start, col_start + n_i))
        col_start += n_i
    io_s = time.perf_counter() - t_io0
    
    D_all = None    
    
    t_vstack0 = time.perf_counter()
    if blocks:
        # [sum(n_i), d], C-contiguous
        D_all = np.ascontiguousarray(np.vstack(blocks).astype(np.float32))
    io_s = time.perf_counter() - t_io0
    vstack_s = time.perf_counter() - t_vstack0

    reranked_pairs: List[Tuple[str, float]] = []

    t_c0 = time.perf_counter()
    if D_all is not None and D_all.size > 0:
        S = q_emb @ D_all.T
        for did, s, e in doc_spans:
            if e - s == 0:
                score = -1e9
            else:
                score = float(S[:, s:e].max(axis=1).sum())
            reranked_pairs.append((did, score))
    compute_s = time.perf_counter() - t_c0

    t_sort0 = time.perf_counter()
    reranked_pairs.sort(key=lambda x: x[1], reverse=True)
    computed_set = {did for (did, _) in reranked_pairs}
    tail_pairs = [(did, sc) for (did, sc) in task.initial_candidates if did not in computed_set]
    out = OrderedDict()
    for did, sc in reranked_pairs:
        out[did] = float(sc)
    for did, sc in tail_pairs:
        out[did] = float(sc)
    sort_s = time.perf_counter() - t_sort0

    total_s = io_s + compute_s + sort_s

    meta = dict()
    return out, total_s, compute_s, io_s, sort_s, meta, vstack_s

# ============== 나이브 per-task rerank (멀티스레드 배치용) ==============
def _rerank_task_naive(
    retriever: ColbertFdeRetrieverNaive,
    task: "RerankTask",
    top_k: int,
) -> Tuple[OrderedDict, float, float, float, float, dict]:
    q_emb = task.query_embeddings
    N_compute = num_rank_candidates
    compute_ids = [did for (did, _) in task.initial_candidates[:N_compute]]

    io_before = _read_proc_io_bytes()
    mf0, M0   = _get_rusage_faults()
    pos_seq: List[int] = []

    io_s = 0.0
    compute_s = 0.0
    vstack_s = 0.0
    reranked_pairs: List[Tuple[str, float]] = []

    for did in compute_ids:
        try:
            pos_seq.append(int(retriever._doc_pos[did]))
        except Exception:
            pass

        t_io = time.perf_counter()
        d_tok = retriever._get_doc_embeddings(did, allow_build=True)
        io_s += time.perf_counter() - t_io

        t_cp = time.perf_counter()
        score = retriever._chamfer(q_emb, d_tok)
        compute_s += time.perf_counter() - t_cp
        reranked_pairs.append((did, float(score)))

    t_sort0 = time.perf_counter()
    reranked_pairs.sort(key=lambda x: x[1], reverse=True)
    computed_set = {did for (did, _) in reranked_pairs}
    tail_pairs = [(did, sc) for (did, sc) in task.initial_candidates if did not in computed_set]

    out = OrderedDict()
    for did, sc in reranked_pairs:
        out[did] = sc
    for did, sc in tail_pairs:
        out[did] = float(sc)

    sort_s = time.perf_counter() - t_sort0
    total_s = io_s + compute_s + sort_s

    io_after = _read_proc_io_bytes()
    mf1, M1  = _get_rusage_faults()
    read_bytes = (io_after - io_before) if (io_before is not None and io_after is not None) else None
    minflt_delta = (mf1 - mf0) if (mf0 is not None and mf1 is not None) else None
    majflt_delta = (M1 - M0) if (M0 is not None and M1 is not None) else None

    mean_pos_delta = None
    p95_pos_delta = None
    if len(pos_seq) >= 2:
        deltas = [abs(pos_seq[i] - pos_seq[i-1]) for i in range(1, len(pos_seq))]
        deltas_sorted = sorted(deltas)
        mean_pos_delta = float(sum(deltas) / len(deltas))
        p95_pos_delta = float(deltas_sorted[int(0.95*(len(deltas_sorted)-1))])

    meta = dict(
        docloads=len(compute_ids),
        uniqdocs=len(set(compute_ids)),
        read_bytes=read_bytes,
        minflt_delta=minflt_delta,
        majflt_delta=majflt_delta,
        mean_pos_delta=mean_pos_delta,
        p95_pos_delta=p95_pos_delta,
    )
    return out, total_s, compute_s, io_s, sort_s, meta, vstack_s

# ============== 배치 오케스트레이션(나이브, 고정 크기) ==============
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
    metas: List[Tuple[str, str, np.ndarray, float]] = [] # qid, qtext, qemb, t_enq

    def flush():
        if not XQ_list:
            return
        XQb = np.vstack(XQ_list)
        D, I, ann_time = retriever.ann_search_batch(XQb, k)
        t_now = time.perf_counter()
        for i, (qid, qtext, qemb, t_enq) in enumerate(metas):
            mask = I[i] >= 0
            cand_ids = [retriever.doc_ids[idx] for idx in I[i][mask]]
            cand_scores = D[i][mask].tolist()
            initial_candidates = list(zip(cand_ids, cand_scores))
            task = RerankTask(
                qid=qid, qtext=qtext, query_embeddings=qemb,
                initial_candidates=initial_candidates,
                search_time_s=(t_now - t_enq),
                ann_time_s=ann_time, enqueued_time_s=t_now
            )
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
        if (qemb is None) or (qfde is None) or (fde := qfde) is None or (fde.shape[0] != exp_dim):
            qmap = retriever.ranker.encode_queries(queries=[qtext])
            qemb = to_numpy(next(iter(qmap.values())))
            qcfg = replace(retriever.doc_config, fill_empty_partitions=False)
            qfde = generate_query_fde(qemb, qcfg)
            retriever._save_query_cache(key, qemb, qfde)

        XQ_list.append(np.ascontiguousarray(qfde.reshape(1, -1).astype(np.float32)))
        metas.append((qid, qtext, qemb, t_enq))

        if len(XQ_list) >= batch_size:
            flush()

def rerank_aggregator_loop(retriever: ColbertFdeRetrieverNaive,
                           in_q: Queue,
                           out_dict: Dict[str, OrderedDict],
                           top_k: int,
                           batch_queries: int = RERANK_BATCH_QUERIES,
                           num_workers: int = RERANK_WORKERS):
    results_lock = Lock()
    stop_token = "__STOP__"

    def _commit_result(
        task: RerankTask,
        out_pairs: OrderedDict,
        rerank_time: float,
        compute_rerank_time: float,
        io_rerank_time: float,
        wait_s: float,
        vstack_s: float,
        dup_ratio: Optional[float] = None,
        meta: Optional[dict] = None,
    ):
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
                               rerank_time, compute_rerank_time, io_rerank_time, wait_s, vstack_s,
                               dup_ratio=dup_ratio)

    # Immediate 모드
    if BATCH_RERANK_MODE == "immediate":
        def process_one(task: RerankTask):
            t_start = time.perf_counter()
            wait_s = t_start - task.enqueued_time_s
            t0 = time.perf_counter()
            out_pairs, total_rerank_s, compute_s, io_s, sort_s, meta, vstack_s = _rerank_task_with_mega_gemm(
                retriever, task, top_k)
            rerank_time = time.perf_counter() - t0
            _commit_result(task, out_pairs, rerank_time, compute_s, io_s, wait_s, vstack_s, dup_ratio=None, meta=meta)

        workers = []
        def worker_loop():
            while True:
                item = in_q.get()
                if item == "__STOP__":
                    in_q.put("__STOP__")
                    break
                process_one(item)

        for _ in range(max(1, int(num_workers))):
            t = threading.Thread(target=worker_loop, daemon=True)
            t.start()
            workers.append(t)
        for t in workers:
            t.join()
        return

    # Batch 모드
    elif BATCH_RERANK_MODE == "batch":
        buffer: List[RerankTask] = []

        def _process_task(task: RerankTask, dup_ratio_for_batch: Optional[float]): # task is equal to query
            t_start = time.perf_counter()
            wait_s = t_start - task.enqueued_time_s
            t0 = time.perf_counter()
            out_pairs, total_rerank_s, compute_s, io_s, sort_s, meta, vstack_s = _rerank_task_with_mega_gemm_parallel( # _rerank_task_with_mega_gemm
                retriever, task, top_k
            )
            rerank_time = time.perf_counter() - t0
            # print(f"[BigGEMM] Completed qid={task.qid} in {rerank_time*1000:.2f} ms")
            _commit_result(
                task, out_pairs, rerank_time, compute_s, io_s, wait_s, vstack_s,
                dup_ratio=dup_ratio_for_batch, meta=meta
            )

        def _flush_batch(buf: List[RerankTask]):
            if not buf:
                return
            # 배치 내 중복율
            all_doc_ids: List[str] = []
            for task in buf:
                N_compute = min(top_k, retriever.rerank_candidates, len(task.initial_candidates))
                all_doc_ids.extend([did for (did, _) in task.initial_candidates[:N_compute]])
            total_docs = len(all_doc_ids)
            if total_docs > 0:
                unique_docs = len(set(all_doc_ids))
                dup_ratio_for_batch = 1.0 - (unique_docs / total_docs)
            else:
                dup_ratio_for_batch = 0.0

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
                _flush_batch(buffer)
                buffer.clear()
        return

    else:
        raise ValueError(f"Unknown BATCH_RERANK_MODE={BATCH_RERANK_MODE}")

# ============================
# --- NEW: Bruteforce Top-K ---
# ============================
# 병렬 브루트포스 설정
BF_WORKERS = max(1, (os.cpu_count() or 4) // 2)
BF_CHUNK_SIZE = 256

# 전역 빌드 락(문서 임베딩이 없을 때 생성 구간 직렬화)
_DOC_BUILD_LOCK = threading.Lock()

def _load_existing_bf_qids(path: str) -> set:
    """이미 저장된 qid 집합을 반환 (파일 없으면 빈 집합)."""
    if not os.path.exists(path):
        return set()
    seen = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                qid = line.split("\t", 1)[0]
                seen.add(qid)
    except Exception as e:
        logging.warning(f"[BF] failed to read existing file: {e}")
    return seen

def _append_bf_topk(path: str, qid: str, topk: List[Tuple[str, float]]):
    """브루트포스 Top-K를 파일에 append (qid당 K줄)."""
    with open(path, "a", encoding="utf-8") as f:
        for rank, (docid, score) in enumerate(topk, start=1):
            f.write(f"{qid}\t{docid}\t{score:.8f}\t{rank}\n")

def _safe_get_doc_embeddings(retriever: ColbertFdeRetrieverNaive, did: str) -> np.ndarray:
    """
    문서 임베딩 로드. 없어서 생성이 필요하면 생성 구간만 전역 락으로 직렬화하여
    write 충돌을 방지한다.
    """
    int_path = retriever._internal_doc_emb_path(did)
    if os.path.exists(int_path):
        return np.load(int_path)
    # 생성이 필요할 수 있으니 락
    with _DOC_BUILD_LOCK:
        # 다른 스레드가 방금 생성했을 수 있으니 재확인
        if os.path.exists(int_path):
            return np.load(int_path)
        return retriever._get_doc_embeddings(did, allow_build=True)

def _bf_chunk_worker(retriever: ColbertFdeRetrieverNaive,
                     q_emb: np.ndarray,
                     doc_ids: List[str],
                     k: int) -> List[Tuple[float, str]]:
    """
    한 청크의 문서들에 대한 로컬 Top-K 반환: [(score, docid), ...] (min-heap 유지)
    """
    local_heap: List[Tuple[float, str]] = []
    push = heapq.heappush
    replace = heapq.heapreplace

    for did in doc_ids:
        d_tok = _safe_get_doc_embeddings(retriever, did)
        score = retriever._chamfer(q_emb, d_tok)
        if len(local_heap) < k:
            push(local_heap, (score, did))
        else:
            if score > local_heap[0][0]:
                replace(local_heap, (score, did))
    return local_heap

def _compute_bf_topk_for_query(retriever: ColbertFdeRetrieverNaive,
                               qid: str,
                               qtext: str,
                               k: int,
                               workers: int = BF_WORKERS,
                               chunk_size: int = BF_CHUNK_SIZE) -> List[Tuple[str, float]]:
    """쿼리 하나에 대해 Chamfer 정확 점수로 전 코퍼스를 병렬 브루트포스하고 Top-K 반환."""
    # 쿼리 임베딩 준비(토큰)
    key = retriever._query_key(qtext, qid)
    qemb, qfde = retriever._load_query_cache(key)
    if qemb is None:
        qmap = retriever.ranker.encode_queries(queries=[qtext])
        qemb = to_numpy(next(iter(qmap.values())))
        qcfg = replace(retriever.doc_config, fill_empty_partitions=False)
        qfde = generate_query_fde(qemb, qcfg)
        retriever._save_query_cache(key, qemb, qfde)

    # 문서 id를 청크로 분할
    doc_ids = retriever.doc_ids
    chunks: List[List[str]] = [doc_ids[i:i+chunk_size] for i in range(0, len(doc_ids), chunk_size)]

    # 각 청크를 병렬로 처리하여 로컬 top-k 반환 → 전역 병합
    global_heap: List[Tuple[float, str]] = []
    push = heapq.heappush
    replace = heapq.heapreplace

    with ThreadPoolExecutor(max_workers=max(1, int(workers)), thread_name_prefix="bf-doc") as ex:
        futures = [ex.submit(_bf_chunk_worker, retriever, qemb, ch, k) for ch in chunks]
        for fut in as_completed(futures):
            local_heap = fut.result()
            for sc, did in local_heap:
                if len(global_heap) < k:
                    push(global_heap, (sc, did))
                else:
                    if sc > global_heap[0][0]:
                        replace(global_heap, (sc, did))

    # 큰 점수 우선 내림차순 정렬
    top_sorted = sorted(((did, sc) for sc, did in global_heap), key=lambda x: x[1], reverse=True)
    return top_sorted

def compute_and_persist_bf_topk(retriever: ColbertFdeRetrieverNaive,
                                queries: Dict[str, str],
                                k: int,
                                outfile: str):
    """ANN 전: 각 쿼리에 대해 브루트포스 Top-K를 계산해 outfile에 append 저장(이미 있으면 스킵)."""
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    seen_qids = _load_existing_bf_qids(outfile)
    will_process = [ (qid, qtext) for qid, qtext in queries.items() if str(qid) not in seen_qids ]
    if not will_process:
        logging.info(f"[BF] Nothing to compute. All {len(queries)} queries already exist in {outfile}.")
        return

    logging.info(f"[BF] Computing brute-force Top-{k} for {len(will_process)} queries (append to {outfile}) | workers={BF_WORKERS}, chunk={BF_CHUNK_SIZE}")
    for qid, qtext in will_process:
        t0 = time.perf_counter()
        topk = _compute_bf_topk_for_query(retriever, str(qid), qtext, k,
                                          workers=BF_WORKERS, chunk_size=BF_CHUNK_SIZE)
        _append_bf_topk(outfile, str(qid), topk)
        logging.info(f"[BF] qid={qid} done in {time.perf_counter()-t0:.2f}s")

def load_bf_truth(outfile: str) -> Dict[str, List[Tuple[str, float]]]:
    """파일에서 브루트포스 Top-K 진리값을 로드: {qid: [(docid, score), ...] (desc)}"""
    truth: Dict[str, List[Tuple[str, float]]] = {}
    if not os.path.exists(outfile):
        return truth
    with open(outfile, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            qid, docid, score, rank = line.rstrip("\n").split("\t")
            score = float(score)
            truth.setdefault(qid, []).append((docid, score))
    # rank가 보장되긴 하지만, 안전하게 점수 내림차순 정렬
    for qid in truth.keys():
        truth[qid] = sorted(truth[qid], key=lambda x: x[1], reverse=True)
        # 상위 K만 유지(파일이 중복 append되더라도 방어)
        truth[qid] = truth[qid][:TOP_K]
    return truth

# === (NEW) Metrics w.r.t. Brute-force Top-K ===
def system_topk_from_results(results: Dict[str, OrderedDict], k: int) -> Dict[str, List[str]]:
    out = {}
    for qid, ranked in results.items():
        out[str(qid)] = list(islice(ranked.keys(), k))
    return out

def recall_at_k_wrt_bf(results_topk: Dict[str, List[str]],
                       bf_truth: Dict[str, List[Tuple[str, float]]],
                       k: int) -> float:
    hits = 0
    total = 0
    for qid, bf_list in bf_truth.items():
        bf_set = {doc for doc, _ in bf_list[:k]}
        sys_set = set(results_topk.get(qid, [])[:k])
        if not bf_set:
            continue
        total += 1
        hits += len(bf_set.intersection(sys_set)) / len(bf_set)
    return hits / total if total > 0 else 0.0

def hit_at_k_wrt_bf(results_topk: Dict[str, List[str]],
                    bf_truth: Dict[str, List[Tuple[str, float]]],
                    k: int) -> float:
    hits = 0
    total = 0
    for qid, bf_list in bf_truth.items():
        bf_set = {doc for doc, _ in bf_list[:k]}
        if not bf_set:
            continue
        total += 1
        sys_set = set(results_topk.get(qid, [])[:k])
        hits += 1 if len(bf_set.intersection(sys_set)) > 0 else 0
    return hits / total if total > 0 else 0.0

def ndcg_at_k_wrt_bf(sys_topk: Dict[str, List[str]],
                              bf_truth: Dict[str, List[Tuple[str, float]]],
                              k: int) -> Tuple[float, List[float]]:
    """
    sys_topk를 기준으로 순회하며 qid별 nDCG@k 계산.
    - bf_truth[qid]에 정답이 없거나 비어 있으면 해당 qid는 스킵.
    - 이상적 이득(idcg)은 bf_truth의 상위 k개 점수(>=0으로 컷)로 계산.
    - 시스템 이득(dcg)은 sys_topk[qid]의 순서에 따라 bf_truth 점수를 매핑해 계산.
    """
    def _discount(i: int) -> float:
        return 1.0 / math.log2(i + 1)

    per_q: List[float] = []
    for qid, sys_docs in sys_topk.items():
        ideal = bf_truth.get(str(qid), [])
        if not ideal:
            continue

        ideal_k = ideal[:k]
        if not ideal_k:
            continue

        # IDCG: bf_truth의 점수 기반
        ideal_gains = [max(s, 0.0) for _, s in ideal_k]
        idcg = sum(g * _discount(i+1) for i, g in enumerate(ideal_gains))
        if idcg <= 0:
            per_q.append(0.0)
            continue

        # DCG: 시스템 결과 순서에 bf_truth 점수를 매핑
        ideal_map = {doc: max(sc, 0.0) for doc, sc in ideal_k}
        dcg = 0.0
        for i, d in enumerate(sys_docs[:k], start=1):
            g = ideal_map.get(d, 0.0)
            dcg += g * _discount(i)

        per_q.append(dcg / idcg)

    mean_ndcg = sum(per_q) / len(per_q) if per_q else 0.0
    return mean_ndcg, per_q

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

    parser = argparse.ArgumentParser(description="Build FAISS IVF-Flat (IP) from fde_index_{P}_{R}.pkl")
    parser.add_argument("--num_simhash_projections", "--p", type=_positive_int, required=True,
                        help="Number of simhash projections (P)")
    parser.add_argument("--num_repetitions", "--r", type=_positive_int, required=True,
                        help="Number of repetitions (R)")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Optional explicit input pickle (defaults to fde_index_{P}_{R}.pkl)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Optional explicit FAISS output path (defaults to ivf1000_ip_{P}_{R}.faiss)")
    parser.add_argument("--nlist", "-nl", type=_positive_int, default=1000,
                        help="IVF list count (default: 1000)")
    parser.add_argument("--num_rerank_cand", "-rc", type=int, required=True,
                        help="number of rerank candidates")
    parser.add_argument("--topk", "-tk", type=int, required=True,
                        help="number of tok-k")
    
    args, _ = parser.parse_known_args()

    P = args.num_simhash_projections
    R = args.num_repetitions
    argslist = args.nlist
    num_rank_candidates = args.num_rerank_cand
    number_of_topk = args.topk

    in_default = f"fde_index_{P}_{R}.pkl"
    faiss_default = f"ivf{argslist}_ip_{P}_{R}.faiss"
    meta_default = f"meta_{P}_{R}.json"

    corpus, queries, qrels = load_nanobeir_dataset(DATASET_REPO_ID)

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
    )

    # 인덱스/캐시 로드
    t_ready0 = time.perf_counter()
    retriever.index(corpus)
    t_ready = time.perf_counter() - t_ready0
    logging.info(f"Retriever ready in {t_ready:.2f}s")

    # --- 쿼리 캐시 선계산
    retriever.precompute_queries(queries)

    # --- (NEW) ANN 전: 브루트포스 Top-K 진리 생성 & 저장(append, overwrite 금지) [병렬]
    BF_OUTFILE = os.path.join(CACHE_ROOT, f"{DATASET_REPO_ID}_bruteforce_top{number_of_topk}.tsv")
    compute_and_persist_bf_topk(retriever, queries, number_of_topk, BF_OUTFILE)
    bf_truth = load_bf_truth(BF_OUTFILE)

    # 파이프 큐
    ann_in_q: Queue = Queue(maxsize=4096)
    rerank_in_q: Queue = Queue(maxsize=4096)

    # 출력
    results: Dict[str, OrderedDict] = {}

    # 스레드: ANN Aggregator, Rerank Aggregator
    start_time = time.perf_counter()
    ann_thr = threading.Thread(target=ann_aggregator_loop, args=(retriever, ann_in_q, rerank_in_q,
                                                                 max(FAISS_CANDIDATES, RERANK_TOPN),
                                                                 ANN_BATCH_SIZE),
                               daemon=True)
    rr_thr = threading.Thread(target=rerank_aggregator_loop, args=(retriever, rerank_in_q, results, number_of_topk, RERANK_BATCH_QUERIES),
                              daemon=True)
    ann_thr.start()
    rr_thr.start()

    # 프론트: 쿼리 주입
    q_start_times: Dict[str, float] = {}
    for qid, qtext in queries.items():
        q_start_times[qid] = time.perf_counter()
        ann_in_q.put(AnnItem(qid=qid, qtext=qtext, t_enqueue=q_start_times[qid]))

    # 종료 시그널
    ann_in_q.put("__STOP__")
    ann_thr.join()
    rerank_in_q.put("__STOP__")
    rr_thr.join()

    end_time = time.perf_counter() - start_time

    total_search_s = mean(avg_ann_time_list) + mean(avg_rerank_time_list) if avg_ann_time_list and avg_rerank_time_list else 0.0
        
    # --- (NEW) 브루트포스 Top-K 기반 평가 ---
    sys_topk = system_topk_from_results(results, number_of_topk)
    bf_recall = recall_at_k_wrt_bf(sys_topk, bf_truth, number_of_topk)
    bf_hit = hit_at_k_wrt_bf(sys_topk, bf_truth, number_of_topk)
    bf_ndcg, ndcg_list = ndcg_at_k_wrt_bf(sys_topk, bf_truth, number_of_topk)
    
    _per_experiment_log_path = os.path.join(CACHE_ROOT, f"per_experiment_{DATASET_REPO_ID}")
    _per_ndcg_log_path = os.path.join(CACHE_ROOT, f"per_ndcg_{DATASET_REPO_ID}")
    
    try:
        if not os.path.exists(_per_experiment_log_path):
            with open(_per_experiment_log_path, "a", encoding="utf-8") as f:
                f.write(
                f"Dataset: {DATASET_REPO_ID}, Queries: {len(queries)}, FirstCand: {FAISS_CANDIDATES} | "
                f"ANN_BATCH:{ANN_BATCH_SIZE}, RERANK_BATCH_Q:{RERANK_BATCH_QUERIES}, "
                f"RERANK_TOTAL: {mean(avg_rerank_time_list)*1000:.2f} | "
                f"RERANK_CAND:{num_rank_candidates}, AvgSearch: {mean(avg_search_time_list)*1000:.2f} | "
                f"ANN: {mean(avg_ann_time_list)*1000:.2f} Rerank(CP): {mean(avg_rerank_cp_list)*1000:.2f} | "
                f"Rerank(VS): {mean(avg_vstack_time_list)*1000:.2f} | "
                f"Rerank(IO): {mean(avg_rerank_io_list)*1000:.2f} | "
                f"RERANK(CP)_99P: {np.percentile(avg_rerank_cp_list, 99)*1000:.3f} | "
                f"RERANK(IO)_99P: {np.percentile(avg_rerank_io_list, 99)*1000:.3f} | "                
                f"Recall@{number_of_topk}(BF): {bf_recall:.4f}, nDCG@{number_of_topk}(BF): {bf_ndcg:.4f}\n"                
            )
        else:
            with open(_per_experiment_log_path, "a", encoding="utf-8") as f:
                f.write(
                f"Dataset: {DATASET_REPO_ID}, Queries: {len(queries)}, FirstCand: {FAISS_CANDIDATES} | "
                f"ANN_BATCH:{ANN_BATCH_SIZE}, RERANK_BATCH_Q:{RERANK_BATCH_QUERIES}, "
                f"RERANK_TOTAL: {mean(avg_rerank_time_list)*1000:.2f} | "
                f"RERANK_CAND:{num_rank_candidates}, AvgSearch: {mean(avg_search_time_list)*1000:.2f} | "
                f"ANN: {mean(avg_ann_time_list)*1000:.2f} Rerank(CP): {mean(avg_rerank_cp_list)*1000:.2f} | "
                f"Rerank(VS): {mean(avg_vstack_time_list)*1000:.2f} | "
                f"Rerank(IO): {mean(avg_rerank_io_list)*1000:.2f} | "
                f"RERANK(CP)_99P: {np.percentile(avg_rerank_cp_list, 99)*1000:.3f} | "
                f"RERANK(IO)_99P: {np.percentile(avg_rerank_io_list, 99)*1000:.3f} | "
                f"Recall@{number_of_topk}(BF): {bf_recall:.4f}, nDCG@{number_of_topk}(BF): {bf_ndcg:.4f}\n"                
            )
    except Exception as e:
        logging.warning(f"Failed to write per-query header: {e}")

    for i in range(len(ndcg_list)):
        try:            
            with open(_per_ndcg_log_path, "a", encoding="utf-8") as f:
                f.write(
                f"Dataset: {DATASET_REPO_ID}, RERANK_CAND:{num_rank_candidates}, [{i}] nDCG@{number_of_topk}: {ndcg_list[i]}\n"
            )            
        except Exception as e:
            logging.warning(f"Failed to write per-query header: {e}")