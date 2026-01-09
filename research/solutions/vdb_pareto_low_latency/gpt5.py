import numpy as np
from typing import Tuple
try:
    import faiss
except Exception:
    faiss = None


class LowLatencyIVFPQIndex:
    def __init__(self, dim: int, **kwargs):
        self.dim = dim
        self.nlist = int(kwargs.get("nlist", 8192))
        self.m = int(kwargs.get("m", 16))
        self.nbits = int(kwargs.get("nbits", 8))
        self.nprobe = int(kwargs.get("nprobe", 8))
        self.use_opq = bool(kwargs.get("use_opq", True))
        self.refine_k = int(kwargs.get("refine_k", 64))
        self.train_size = int(kwargs.get("train_size", 120000))
        self.seed = int(kwargs.get("seed", 123))
        self.threads = int(kwargs.get("threads", 8))
        self.coarse_M = int(kwargs.get("coarse_M", 32))
        self.coarse_efSearch = int(kwargs.get("coarse_efSearch", 64))
        self.coarse_efConstruction = int(kwargs.get("coarse_efConstruction", 80))
        self.metric = faiss.METRIC_L2 if faiss is not None else None

        self.index = None
        self.quantizer = None
        self._is_trained = False
        self._added = 0

        if faiss is not None:
            try:
                max_thr = faiss.omp_get_max_threads()
                faiss.omp_set_num_threads(min(self.threads, max_thr if max_thr > 0 else self.threads))
            except Exception:
                pass

    def _build_index(self):
        if faiss is None:
            raise RuntimeError("faiss is required for this index.")
        # HNSW coarse quantizer for fast coarse search
        self.quantizer = faiss.IndexHNSWFlat(self.dim, self.coarse_M, self.metric)
        self.quantizer.hnsw.efConstruction = self.coarse_efConstruction
        self.quantizer.hnsw.efSearch = self.coarse_efSearch

        ivfpq = faiss.IndexIVFPQ(self.quantizer, self.dim, self.nlist, self.m, self.nbits, self.metric)
        ivfpq.by_residual = True
        # precomputed table speeds up search at memory cost; beneficial for low-latency
        ivfpq.use_precomputed_table = 1

        base_index = ivfpq
        if self.use_opq:
            # OPQ dimension must be divisible by m
            opq_m = self.m
            opq = faiss.OPQMatrix(self.dim, opq_m)
            opq.niter = 20
            opq.verbose = False
            base_index = faiss.IndexPreTransform(opq, ivfpq)

        # Refine with exact L2 on a small candidate set
        refine = faiss.IndexRefineFlat(base_index)
        refine.k_factor = max(1, self.refine_k)

        # Set nprobe
        try:
            ivf = faiss.extract_index_ivf(base_index)
            ivf.nprobe = max(1, self.nprobe)
        except Exception:
            pass

        self.index = refine

    def add(self, xb: np.ndarray) -> None:
        if faiss is None:
            raise RuntimeError("faiss is required for this index.")
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32, copy=False)
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError("xb must have shape (N, dim) with dtype float32")

        if self.index is None:
            self._build_index()

        # Train on a random subset
        if not self._is_trained:
            n = xb.shape[0]
            ntrain = min(self.train_size, n)
            if ntrain < self.nlist:
                # Ensure enough training vectors compared to nlist to avoid poor training
                ntrain = min(n, max(self.nlist * 4, self.nlist + 1000))
            rs = np.random.RandomState(self.seed)
            perm = rs.permutation(xb.shape[0])[:ntrain]
            xtrain = xb[perm].copy()
            self.index.train(xtrain)
            # Ensure nprobe on trained index
            try:
                base_index = self.index.base_index
                ivf = faiss.extract_index_ivf(base_index)
                ivf.nprobe = max(1, self.nprobe)
                # set coarse efSearch again in case quantizer recreated internals
                if isinstance(ivf.quantizer, faiss.IndexHNSWFlat):
                    ivf.quantizer.hnsw.efSearch = self.coarse_efSearch
            except Exception:
                pass
            self._is_trained = True

        self.index.add(xb)
        self._added += xb.shape[0]

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if faiss is None:
            raise RuntimeError("faiss is required for this index.")
        if self.index is None or not self._is_trained or self._added == 0:
            raise RuntimeError("Index not ready. Call add() with data before searching.")
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32, copy=False)
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError("xq must have shape (nq, dim) with dtype float32")
        if k <= 0:
            raise ValueError("k must be positive")

        # Ensure runtime parameters are set
        try:
            base_index = self.index.base_index
            ivf = faiss.extract_index_ivf(base_index)
            ivf.nprobe = max(1, self.nprobe)
            if isinstance(ivf.quantizer, faiss.IndexHNSWFlat):
                ivf.quantizer.hnsw.efSearch = self.coarse_efSearch
        except Exception:
            pass

        D, I = self.index.search(xq, k)
        if not isinstance(D, np.ndarray):
            D = np.array(D)
        if not isinstance(I, np.ndarray):
            I = np.array(I)
        return D, I