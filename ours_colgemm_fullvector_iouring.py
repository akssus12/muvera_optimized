# -*- coding: utf-8 -*-
"""
ANN + TPack(B^T, F-order) + io_uring 병렬 pread 재랭크 (그룹 없음)
- TPack(열-주 저장: (D, W) F-order)에서 여러 문서의 열 범위를 io_uring으로 병렬 pread
- iovec/readv 미사용: liburing 바인딩 차이를 피하기 위해 io_uring_prep_read(단일 버퍼)만 사용
- vstack 제거(문서별 바로 GEMM) → Rerank(VS)~0
- Rerank(CP), Rerank(IO), Rerank(VS) 타이밍 분리
"""

import os, json, time, hashlib, logging, pathlib, csv, math, heapq, random, threading, argparse, sys, resource
from collections import OrderedDict
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

import neural_cherche.models as neural_cherche_models
import neural_cherche.rank as neural_cherche_rank

from beir import util
from beir.datasets.data_loader import GenericDataLoader

# --- Single-thread BLAS by default ---
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
        io_uring_prep_read,  # 단일 버퍼 read만 사용 (readv/iovec 사용 X)
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

avg_search_time_list = []
avg_ann_time_list = []
avg_rerank_time_list = []
avg_rerank_cp_list = []
avg_rerank_io_list = []
avg_rerank_wait_list = []
avg_vstack_time_list = []
avg_dup_ratio_list = []

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

    # -------- io_uring: 여러 (did, s, e) 병렬 pread --------
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

        try:
            return self._pread_cols_many_liburing(reqs, qd=qd, submit_batch=submit_batch, verify=verify)
        except Exception as e:
            logging.warning(f"[TPack/liburing] failed ({e}); fallback to threaded pread.")
            return self._pread_cols_many_threaded(reqs, max_workers=qd)

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

    def _pread_cols_many_liburing(self,
                                  reqs: List[Tuple[str, int, int, int, int]],
                                  qd: int,
                                  submit_batch: int,
                                  verify: bool = False) -> Dict[str, np.ndarray]:
        """
        Robust liburing path that only uses io_uring_prep_read (no readv / iovec).
        """
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
                # 단일 버퍼 read로 통일
                io_uring_prep_read(sqe, self.fd, buf, nbytes, off)
                io_uring_sqe_set_data64(sqe, idx)
                return True

            while completed < total:
                # issue up to qd
                batch = 0
                while next_idx < total and in_flight < qd and batch < submit_batch: # next_idx < total and in_flight < qd and batch < submit_batch
                    if not _submit_one(next_idx):
                        break
                    next_idx += 1
                    issued += 1
                    in_flight += 1
                    batch += 1

                if batch > 0:
                    sub = io_uring_submit(ring)
                    _trap_error(sub)

                # wait for completion
                # time.sleep(0.00001)
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
                
                # if res != nbytes:
                #     # short read 방어
                #     raw = os.pread(self.fd, nbytes, off)
                #     if len(raw) != nbytes:
                #         raise IOError(f"short read persists: want={nbytes}, got={len(raw)}")
                #     bufs[idx][:] = raw
                

            # if verify and len(reqs) > 0:
            #     rid = random.randrange(len(reqs))
            #     did, s, e, off, nbytes = reqs[rid]
            #     gold = self.pread_cols(s, e - s)
            #     if not np.allclose(gold, out[did], atol=1e-6):
            #         logging.warning("[TPack.IOURING] mismatch detected; falling back to synchronous for this batch")
            #         return self._pread_cols_many_threaded(reqs, max_workers=qd)

            return out
        finally:
            io_uring_queue_exit(ring)

# =====================================
# Retriever
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
        """
        query_tok: (Tq, D)
        doc_tok_BT: (D, ncols)  [TPack F-order view]
        """
        if doc_tok_BT.size == 0:
            return -1e9
        S = query_tok @ doc_tok_BT  # (Tq, ncols)
        return float(S.max(axis=1).sum())

    def _get_doc_embeddings(self, doc_id: str, allow_build: bool = True) -> np.ndarray:
        # 원본 임베딩(행-주, C-order) 생성/캐시용 (TPack build 시 사용)
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

    def index(self, corpus: dict):
        self._corpus = corpus
        # FDE 인덱스/ID 로드
        with open(self._ids_path, "r", encoding="utf-8") as f:
            self.doc_ids = json.load(f)
        self._doc_pos = {d: i for i, d in enumerate(self.doc_ids)}
        self.fde_index = joblib.load(self._fde_path)
        logging.info(f"[{self.__class__.__name__}] Loaded FDE index cache: {self.fde_index.shape} for {len(self.doc_ids)} docs")

        # FAISS 준비
        if self.use_faiss_ann and os.path.exists(self._faiss_path):
            try:
                self.faiss_index = faiss.read_index(self._faiss_path)
                self.faiss_index.nprobe = FAISS_NPROBE
            except Exception:
                self.faiss_index = None

        if self.use_tpack:
            self.ensure_tpack()

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

# ================= 브루트포스 (상한선) =================
BF_WORKERS = max(1, (os.cpu_count() or 4) // 2)
BF_CHUNK_SIZE = 256
_DOC_BUILD_LOCK = threading.Lock()

def _bf_chunk_worker(retriever: ColbertFdeRetrieverNaive, q_emb: np.ndarray, doc_ids: List[str], k: int):
    local_heap: List[Tuple[float, str]] = []
    push = heapq.heappush; replace = heapq.heapreplace
    for did in doc_ids:
        # BF는 원본 임베딩 로딩(행-주)로 유지
        d_tok = retriever._get_doc_embeddings(did, allow_build=True)
        score = float((q_emb @ d_tok.T).max(axis=1).sum()) if d_tok.size else -1e9
        if len(local_heap) < k: push(local_heap, (score, did))
        else:
            if score > local_heap[0][0]: replace(local_heap, (score, did))
    return local_heap

def _compute_bf_topk_for_query(retriever: ColbertFdeRetrieverNaive, qid: str, qtext: str, k: int,
                               workers: int = None, chunk_size: int = 256):
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
    with open(path, "a", encoding="utf-8") as f:
        for rank, (docid, score) in enumerate(topk, start=1):
            f.write(f"{qid}\t{docid}\t{score:.8f}\t{rank}\n")

def compute_and_persist_bf_topk(retriever: ColbertFdeRetrieverNaive, queries: Dict[str, str], k: int, outfile: str):
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

def recall_at_k_wrt_bf(results_topk: Dict[str, List[str]], bf_truth: Dict[str, List[Tuple[str, float]]], k: int) -> float:
    hits = 0; total = 0
    for qid, bf_list in bf_truth.items():
        bf_set = {doc for doc, _ in bf_list[:k]}
        sys_set = set(results_topk.get(qid, [])[:k])
        if not bf_set: continue
        total += 1
        hits += len(bf_set.intersection(sys_set)) / len(bf_set)
    return hits / total if total > 0 else 0.0

def hit_at_k_wrt_bf(results_topk: Dict[str, List[str]], bf_truth: Dict[str, List[Tuple[str, float]]], k: int) -> float:
    hits = 0; total = 0
    for qid, bf_list in bf_truth.items():
        bf_set = {doc for doc, _ in bf_list[:k]}
        if not bf_set: continue
        total += 1
        sys_set = set(results_topk.get(qid, [])[:k])
        hits += 1 if len(bf_set.intersection(sys_set)) > 0 else 0
    return hits / total if total > 0 else 0.0

def ndcg_at_k_wrt_bf(sys_topk: Dict[str, List[str]],
                     bf_truth: Dict[str, List[Tuple[str, float]]],
                     k: int) -> Tuple[float, List[float]]:
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

# ============== io_uring 기반 재랭크 (TPack, 그룹 없음) ==============
def _rerank_task_with_grouped_gemm(retriever: ColbertFdeRetrieverNaive, task: "RerankTask", top_k: int):
    """
    [TPack, 그룹 없음] compute_ids에 대해:
      - 각 문서의 (s,e) 열 범위를 계산
      - TPackReader.pread_cols_many_uring()으로 병렬 pread
      - 문서별로 즉시 GEMM (q_emb @ Bdoc), vstack 불필요
    """
    start_time = time.perf_counter()
    q_emb = task.query_embeddings                      # [Tq, D]
    Tq, Dq = int(q_emb.shape[0]), int(q_emb.shape[1])

    # 재랭크할 후보 집합
    N_compute = num_rank_candidates if num_rank_candidates > 0 else len(task.initial_candidates)
    N_compute = min(N_compute, len(task.initial_candidates))
    compute_ids = [did for (did, _) in task.initial_candidates[:N_compute]]

    # TPack 없으면 메모리 경로 폴백
    if retriever.tpack is None:
        t_io0 = time.perf_counter()
        reranked_pairs: List[Tuple[str, float]] = []
        for did in compute_ids:
            Ddoc = retriever._get_doc_embeddings(did, allow_build=True).astype(np.float32, copy=False)  # (n, D)
            score = float((q_emb @ Ddoc.T).max(axis=1).sum()) if Ddoc.size else -1e9
            reranked_pairs.append((did, score))
        io_s = time.perf_counter() - t_io0
        compute_s = 0.0  # 위에서 함께 계산
        vstack_s = 0.0
        reranked_pairs.sort(key=lambda x: x[1], reverse=True)
        out = OrderedDict((did, float(sc)) for did, sc in reranked_pairs)
        total_s = io_s + compute_s
        meta = {}
        return out, total_s, compute_s, io_s, meta, vstack_s

    t_io0 = time.perf_counter()
    tp = retriever.tpack
    if int(tp.D) != Dq:
        logging.warning(f"[TPack] D mismatch (tp.D={tp.D} vs query.D={Dq}) → memory path fallback.")
        return _rerank_task_with_grouped_gemm(retriever=replace(retriever, tpack=None), task=task, top_k=top_k)

    doc_spans: List[Tuple[str, int, int]] = []
    for did in compute_ids:
        di = retriever._doc_pos[did]
        s, e = tp.doc_col_span(di)
        doc_spans.append((did, s, e))

    # 2) 병렬 pread
    doc_blocks: Dict[str, np.ndarray] = tp.pread_cols_many_uring(
        doc_spans, qd=retriever._tpack_qd, submit_batch=retriever._tpack_submit, verify=False
    )
    io_s = time.perf_counter() - t_io0

    # 3) 문서별 GEMM(즉시 스코어), vstack 불필요
    vstack_s = 0.0
    compute_start_time = time.perf_counter()
    
    reranked_pairs: List[Tuple[str, float]] = []    
    
    ########## ------ sequential version ------ ##########
    # for did in compute_ids:
    #     doc_block_get_start_time = time.perf_counter()
    #     Bdoc = doc_blocks.get(did, None)
    #     io_s += time.perf_counter() - doc_block_get_start_time        
        
    #     compute_start_time = time.perf_counter()
    #     Sblk = q_emb @ Bdoc                  # (Tq, ncols)        
    #     score = float(Sblk.max(axis=1).sum())        
    #     compute_s += time.perf_counter() - compute_start_time
        
    #     reranked_pairs.append((did, score))
    ########## ------ sequential version ------ ##########

    ########## ------ multi-threaded version ------ ##########
    def _score_one(did: str) -> Tuple[str, float, float, float]:
        """한 문서에 대해 (did, score, compute_dt, io_dt)를 반환."""        
        Bdoc = doc_blocks.get(did, None)
        Sblk = q_emb @ Bdoc                  # (Tq, ncols)
        score = float(Sblk.max(axis=1).sum())
        return did, score #, gemm_dt, {"elapsed_s": r_gemm.elapsed_s, **r_gemm.counts}
    
    parallelism = max(1, int(RERANK_WORKERS))
    with ThreadPoolExecutor(max_workers=parallelism, thread_name_prefix="rerank-doc") as ex:
        futs = [ex.submit(_score_one, did) for did in compute_ids]
        for fut in as_completed(futs):
            did, sc = fut.result()
            reranked_pairs.append((did, sc))            
    compute_s = time.perf_counter() - compute_start_time

    reranked_pairs.sort(key=lambda x: x[1], reverse=True)
    out = OrderedDict((did, float(sc)) for did, sc in reranked_pairs)
    
    meta = {}
    total_s = time.perf_counter() - start_time # io_s + compute_s

    print(f"[iouring] Total time: {total_s:.3f}, Rerank IO time: {io_s:.3f}s, Compute time: {compute_s:.3f}s, cp_ratio: {compute_s/total_s:.3f}, io_ratio: {io_s/total_s:.3f}")
    return out, total_s, compute_s, io_s, meta, vstack_s
 ########## ------ multi-threaded version ------ ##########

# ============== 배치 오케스트레이션 ==============
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
        # 상위 N만 재랭크(0이면 전부)
        topN = retriever.rerank_candidates if retriever.rerank_candidates > 0 else len(task.initial_candidates)
        task = replace(task, initial_candidates=task.initial_candidates[:topN])

        t_start = time.perf_counter()
        wait_s = t_start - task.enqueued_time_s
        t0 = time.perf_counter()
        out_pairs, total_rerank_s, compute_s, io_s, meta, vstack_s = _rerank_task_with_grouped_gemm(retriever, task, top_k)
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
            all_doc_ids: List[str] = []
            for task in buf:
                N_compute = min(top_k if top_k>0 else len(task.initial_candidates),
                                retriever.rerank_candidates if retriever.rerank_candidates>0 else len(task.initial_candidates),
                                len(task.initial_candidates))
                all_doc_ids.extend([did for (did, _) in task.initial_candidates[:N_compute]])
            total_docs = len(all_doc_ids)
            dup_ratio_for_batch = 0.0 if total_docs == 0 else 1.0 - (len(set(all_doc_ids)) / total_docs)

            parallelism = min(max(1, int(num_workers)), len(buf))
            # print(f"[iouring batch rerank] processing {len(buf)} tasks with {parallelism} workers")
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

    parser = argparse.ArgumentParser(description="ANN + TPack(io_uring) parallel pread rerank (no grouping)")
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
    )

    # 캐시 파일명 (P,R에 맞춤)
    in_default = f"fde_index_{P}_{R}.pkl"
    faiss_default = f"ivf{argslist}_ip_{P}_{R}.faiss"
    meta_default = f"meta_{P}_{R}.json"

    retriever._fde_path  = os.path.join(retriever._cache_dir, in_default)
    retriever._faiss_path = os.path.join(retriever._cache_dir, faiss_default)
    retriever._meta_path = os.path.join(retriever._cache_dir, meta_default)

    t_ready0 = time.perf_counter()
    retriever.index(corpus)   # TPack 준비는 index() 내부에서
    t_ready = time.perf_counter() - t_ready0
    logging.info(f"Retriever ready in {t_ready:.3f}s (tpack_used={retriever.tpack is not None}, D={retriever.tpack.D if retriever.tpack else 'NA'})")

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
    # compute_and_persist_bf_topk(retriever, queries, number_of_topk, BF_OUTFILE)
    bf_truth = load_bf_truth(BF_OUTFILE)

    # 파이프라인 스레드
    ann_in_q: Queue = Queue(maxsize=4096)
    rerank_in_q: Queue = Queue(maxsize=4096)
    results: Dict[str, OrderedDict] = {}

    ann_thr = threading.Thread(target=ann_aggregator_loop, args=(retriever, ann_in_q, rerank_in_q,
                                                                 max(FAISS_CANDIDATES, number_of_topk),
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
    # sys_topk = system_topk_from_results(results, number_of_topk)
    # bf_recall = recall_at_k_wrt_bf(sys_topk, bf_truth, number_of_topk)
    # bf_hit = hit_at_k_wrt_bf(sys_topk, bf_truth, number_of_topk)
    # bf_ndcg, _ = ndcg_at_k_wrt_bf(sys_topk, bf_truth, number_of_topk)

    _per_experiment_log_path = os.path.join(CACHE_ROOT, f"per_experiment_{DATASET_REPO_ID}")
        
    try:
        if not os.path.exists(_per_experiment_log_path):
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
                f"RERANK(IO)_99P: {np.percentile(avg_rerank_io_list, 99)*1000:.3f}\n"
                # f"Recall@{number_of_topk}(BF): {bf_recall:.3f}, nDCG@{number_of_topk}(BF): {bf_ndcg:.3f}\n"                
            )
        else:
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
                f"RERANK(IO)_99P: {np.percentile(avg_rerank_io_list, 99)*1000:.3f}\n"
                # f"Recall@{number_of_topk}(BF): {bf_recall:.3f}, nDCG@{number_of_topk}(BF): {bf_ndcg:.3f}\n"                
            )
    except Exception as e:
        logging.warning(f"Failed to write per-query header: {e}")