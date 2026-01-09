import os
from typing import Tuple, Optional

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    faiss = None


def _largest_divisor_leq(n: int, x: int) -> int:
    x = max(1, min(x, n))
    while x > 1 and (n % x) != 0:
        x -= 1
    return max(1, x)


def _set_nprobe_recursive(index, nprobe: int) -> bool:
    if index is None:
        return False
    if hasattr(index, "nprobe"):
        try:
            index.nprobe = int(nprobe)
            return True
        except Exception:
            pass
    for attr in ("base_index", "index"):
        if hasattr(index, attr):
            try:
                sub = getattr(index, attr)
            except Exception:
                sub = None
            if _set_nprobe_recursive(sub, nprobe):
                return True
    return False


class YourIndexClass:
    def __init__(
        self,
        dim: int,
        **kwargs,
    ):
        self.dim = int(dim)

        self.threads = int(kwargs.get("threads", 0) or (os.cpu_count() or 8))
        if self.threads <= 0:
            self.threads = 8

        self.exact_threshold = int(kwargs.get("exact_threshold", 50000))

        self.nlist = int(kwargs.get("nlist", 4096))
        self.nprobe = int(kwargs.get("nprobe", 24))

        self.use_pq = bool(kwargs.get("use_pq", True))
        self.m = int(kwargs.get("m", 32))
        self.nbits = int(kwargs.get("nbits", 8))

        self.use_opq = bool(kwargs.get("use_opq", True))
        self.opq_m = int(kwargs.get("opq_m", self.m))

        self.refine = bool(kwargs.get("refine", True))
        self.k_factor = int(kwargs.get("k_factor", 128))

        self.train_size = int(kwargs.get("train_size", 200000))
        self.seed = int(kwargs.get("seed", 123))

        self._index = None
        self._trained = False
        self._ntotal_added = 0

        if faiss is not None:
            try:
                faiss.omp_set_num_threads(self.threads)
            except Exception:
                pass

    def _build_index(self, N_first_add: int):
        if faiss is None:
            raise RuntimeError("faiss is required but not available")

        if N_first_add <= self.exact_threshold:
            self._index = faiss.IndexFlatL2(self.dim)
            self._trained = True
            return

        nlist = max(1, min(self.nlist, N_first_add))
        m = _largest_divisor_leq(self.dim, self.m)
        opq_m = _largest_divisor_leq(self.dim, self.opq_m)

        quantizer = faiss.IndexFlatL2(self.dim)

        if self.use_pq:
            base_ivf = faiss.IndexIVFPQ(quantizer, self.dim, nlist, m, self.nbits)
            try:
                base_ivf.use_precomputed_table = 1
            except Exception:
                pass
        else:
            base_ivf = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_L2)

        try:
            base_ivf.cp.niter = int(kwargs_niter)  # noqa: F821
        except Exception:
            pass

        base = base_ivf
        if self.use_opq and self.use_pq:
            opq = faiss.OPQMatrix(self.dim, opq_m)
            try:
                opq.niter = int(kwargs_opq_niter)  # noqa: F821
            except Exception:
                pass
            base = faiss.IndexPreTransform(opq, base)

        if self.refine and self.use_pq:
            idx = faiss.IndexRefineFlat(base)
            try:
                idx.k_factor = int(max(1, self.k_factor))
            except Exception:
                pass
            self._index = idx
        else:
            self._index = base

        _set_nprobe_recursive(self._index, self.nprobe)

    def add(self, xb: np.ndarray) -> None:
        if xb is None or xb.size == 0:
            return
        xb = np.ascontiguousarray(xb, dtype=np.float32)
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError(f"xb must have shape (N, {self.dim})")

        if self._index is None:
            # Allow passing extra build-time params via kwargs names (optional)
            # without storing them on self; default niter values are FAISS defaults.
            global kwargs_niter, kwargs_opq_niter
            kwargs_niter = 20
            kwargs_opq_niter = 25
            self._build_index(int(xb.shape[0]))

        if not self._trained and hasattr(self._index, "is_trained") and not self._index.is_trained:
            ts = int(min(max(1, self.train_size), xb.shape[0]))
            xtrain = xb[:ts]
            self._index.train(xtrain)
            self._trained = True
        else:
            self._trained = True

        self._index.add(xb)
        self._ntotal_added += int(xb.shape[0])

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._index is None or self._ntotal_added == 0:
            nq = 0 if xq is None else int(xq.shape[0])
            D = np.full((nq, k), np.inf, dtype=np.float32)
            I = np.full((nq, k), -1, dtype=np.int64)
            return D, I

        k = int(k)
        if k <= 0:
            raise ValueError("k must be >= 1")

        xq = np.ascontiguousarray(xq, dtype=np.float32)
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError(f"xq must have shape (nq, {self.dim})")

        _set_nprobe_recursive(self._index, self.nprobe)

        if self.refine and hasattr(self._index, "k_factor"):
            try:
                self._index.k_factor = int(max(1, self.k_factor))
            except Exception:
                pass

        D, I = self._index.search(xq, k)

        if D.dtype != np.float32:
            D = D.astype(np.float32, copy=False)
        if I.dtype != np.int64:
            I = I.astype(np.int64, copy=False)
        return D, I