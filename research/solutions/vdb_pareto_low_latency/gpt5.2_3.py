import os
import numpy as np

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None


def _cpu_threads_default():
    try:
        c = os.cpu_count() or 1
    except Exception:
        c = 1
    return max(1, min(8, int(c)))


def _downcast(idx):
    if faiss is None:
        return idx
    try:
        return faiss.downcast_index(idx)
    except Exception:
        return idx


def _extract_base_index(idx):
    idx = _downcast(idx)
    for _ in range(8):
        name = type(idx).__name__
        if "IndexRefine" in name and hasattr(idx, "base_index"):
            idx = _downcast(idx.base_index)
            continue
        if "IndexPreTransform" in name and hasattr(idx, "index"):
            idx = _downcast(idx.index)
            continue
        break
    return idx


def _extract_ivf(idx):
    idx = _extract_base_index(idx)
    if hasattr(idx, "nprobe") and hasattr(idx, "quantizer"):
        return idx
    return None


def _extract_ivfpq(idx):
    idx = _extract_base_index(idx)
    name = type(idx).__name__
    if "IndexIVFPQ" in name:
        return idx
    return None


def _choose_m(dim, m_pref):
    m = int(m_pref)
    if m <= 0:
        m = 1
    m = min(m, dim)
    while m > 1 and (dim % m) != 0:
        m -= 1
    return max(1, m)


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = int(dim)

        self.nlist = int(kwargs.get("nlist", 8192))
        self.m = _choose_m(self.dim, int(kwargs.get("m", 16)))
        self.nbits = int(kwargs.get("nbits", 8))

        self.nprobe = int(kwargs.get("nprobe", 32))
        self.k_factor = int(kwargs.get("k_factor", 256))
        self.max_candidates = int(kwargs.get("max_candidates", 512))

        self.hnsw_m = int(kwargs.get("hnsw_m", 32))
        self.quantizer_ef_search = int(kwargs.get("quantizer_ef_search", 64))

        self.train_size = int(kwargs.get("train_size", 200000))
        self.seed = int(kwargs.get("seed", 123))
        self.use_opq = bool(kwargs.get("use_opq", True))

        self.niter = int(kwargs.get("niter", 15))
        self.max_points_per_centroid = int(kwargs.get("max_points_per_centroid", 256))

        self._threads = int(kwargs.get("threads", _cpu_threads_default()))
        if faiss is not None:
            try:
                faiss.omp_set_num_threads(self._threads)
            except Exception:
                pass

        if faiss is None:  # pragma: no cover
            self.index = None
            self._xb = None
            return

        descr_parts = []
        if self.use_opq:
            descr_parts.append(f"OPQ{self.m}")
        descr_parts.append(f"IVF{self.nlist}_HNSW{self.hnsw_m}")
        descr_parts.append(f"PQ{self.m}x{self.nbits}")
        descr = ",".join(descr_parts)

        base = faiss.index_factory(self.dim, descr, faiss.METRIC_L2)
        self.index = faiss.IndexRefineFlat(base)
        try:
            self.index.k_factor = max(1, int(self.k_factor))
        except Exception:
            pass

        self._configure_index()

    def _configure_index(self):
        if faiss is None or self.index is None:
            return

        ivf = _extract_ivf(self.index)
        if ivf is not None:
            try:
                ivf.nprobe = max(1, int(self.nprobe))
            except Exception:
                pass

            try:
                if hasattr(ivf, "cp") and ivf.cp is not None:
                    ivf.cp.niter = max(5, int(self.niter))
                    ivf.cp.max_points_per_centroid = max(32, int(self.max_points_per_centroid))
            except Exception:
                pass

            try:
                q = _downcast(ivf.quantizer)
                if hasattr(q, "hnsw") and q.hnsw is not None:
                    q.hnsw.efSearch = max(8, int(self.quantizer_ef_search))
            except Exception:
                pass

        ivfpq = _extract_ivfpq(self.index)
        if ivfpq is not None:
            try:
                if hasattr(ivfpq, "use_precomputed_table"):
                    ivfpq.use_precomputed_table = 0
            except Exception:
                pass

    def add(self, xb: np.ndarray) -> None:
        if xb is None or len(xb) == 0:
            return

        if faiss is None or self.index is None:  # pragma: no cover
            xb = np.ascontiguousarray(xb, dtype=np.float32)
            if self._xb is None:
                self._xb = xb.copy()
            else:
                self._xb = np.vstack([self._xb, xb])
            return

        xb = np.ascontiguousarray(xb, dtype=np.float32)
        if xb.shape[1] != self.dim:
            raise ValueError(f"xb dim mismatch: expected {self.dim}, got {xb.shape[1]}")

        if not self.index.is_trained:
            n = xb.shape[0]
            ts = min(self.train_size, n)
            if ts < n:
                rng = np.random.default_rng(self.seed)
                sel = rng.choice(n, size=ts, replace=False)
                trainset = np.ascontiguousarray(xb[sel], dtype=np.float32)
            else:
                trainset = xb
            self._configure_index()
            self.index.train(trainset)

        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int):
        k = int(k)
        if k <= 0:
            raise ValueError("k must be >= 1")
        if xq is None or len(xq) == 0:
            return (np.empty((0, k), dtype=np.float32), np.empty((0, k), dtype=np.int64))

        if faiss is None or self.index is None:  # pragma: no cover
            xq = np.ascontiguousarray(xq, dtype=np.float32)
            xb = self._xb
            if xb is None:
                return (np.full((len(xq), k), np.inf, dtype=np.float32), np.full((len(xq), k), -1, dtype=np.int64))
            xq2 = (xq * xq).sum(axis=1, keepdims=True)
            xb2 = (xb * xb).sum(axis=1)[None, :]
            dots = xq @ xb.T
            d2 = xq2 + xb2 - 2.0 * dots
            idx = np.argpartition(d2, kth=min(k - 1, d2.shape[1] - 1), axis=1)[:, :k]
            row = np.arange(len(xq))[:, None]
            dd = d2[row, idx]
            ord2 = np.argsort(dd, axis=1)
            idx = idx[row, ord2].astype(np.int64, copy=False)
            dd = dd[row, ord2].astype(np.float32, copy=False)
            return dd, idx

        xq = np.ascontiguousarray(xq, dtype=np.float32)
        if xq.shape[1] != self.dim:
            raise ValueError(f"xq dim mismatch: expected {self.dim}, got {xq.shape[1]}")

        if hasattr(self.index, "k_factor"):
            eff_kf = max(1, min(int(self.k_factor), int(self.max_candidates // max(1, k))))
            try:
                if int(self.index.k_factor) != eff_kf:
                    self.index.k_factor = eff_kf
            except Exception:
                pass

        D, I = self.index.search(xq, k)

        if D.dtype != np.float32:
            D = D.astype(np.float32, copy=False)
        if I.dtype != np.int64:
            I = I.astype(np.int64, copy=False)

        if D.shape != (xq.shape[0], k) or I.shape != (xq.shape[0], k):
            D = np.ascontiguousarray(D, dtype=np.float32).reshape(xq.shape[0], k)
            I = np.ascontiguousarray(I, dtype=np.int64).reshape(xq.shape[0], k)

        return D, I