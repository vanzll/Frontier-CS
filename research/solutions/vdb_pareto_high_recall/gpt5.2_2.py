import os
from typing import Tuple, Optional, List

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    faiss = None


def _set_thread_env(n_threads: int) -> None:
    n_threads = int(n_threads) if n_threads is not None else 0
    if n_threads <= 0:
        return
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if var not in os.environ:
            os.environ[var] = str(n_threads)
    if faiss is not None:
        try:
            faiss.omp_set_num_threads(n_threads)
        except Exception:
            pass


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = int(dim)

        self._requested_nlist = int(kwargs.get("nlist", 4096))
        self._nprobe = int(kwargs.get("nprobe", 512))
        self._train_size = int(kwargs.get("train_size", 100000))
        self._n_threads = int(kwargs.get("n_threads", 8))

        self._index = None
        self._pending: List[np.ndarray] = []
        self._pending_total = 0

        _set_thread_env(self._n_threads)

        if faiss is None:
            raise RuntimeError("faiss is required but could not be imported")

    def _choose_nlist(self, n_total: int) -> int:
        n_total = int(n_total)
        if n_total <= 0:
            return 1
        heuristic = int(4.0 * (n_total ** 0.5))
        nlist = min(self._requested_nlist, max(1, heuristic))
        nlist = min(nlist, n_total)
        nlist = max(1, nlist)
        return nlist

    def _sample_train_vectors_from_pending(self, m: int) -> np.ndarray:
        m = int(m)
        if m <= 0:
            return np.empty((0, self.dim), dtype=np.float32)

        nb = len(self._pending)
        if nb == 1:
            x = self._pending[0]
            n = x.shape[0]
            if n <= m:
                return np.ascontiguousarray(x, dtype=np.float32)
            idx = np.linspace(0, n - 1, m, dtype=np.int64)
            return np.ascontiguousarray(x[idx], dtype=np.float32)

        sizes = np.array([b.shape[0] for b in self._pending], dtype=np.int64)
        cum = np.cumsum(sizes)
        total = int(cum[-1])
        if total <= m:
            return np.ascontiguousarray(np.vstack(self._pending), dtype=np.float32)

        gidx = np.linspace(0, total - 1, m, dtype=np.int64)
        bidx = np.searchsorted(cum, gidx, side="right")
        starts = np.concatenate(([0], cum[:-1]))
        local = gidx - starts[bidx]

        out = np.empty((m, self.dim), dtype=np.float32)
        pos = 0
        for bi in range(nb):
            mask = (bidx == bi)
            if not np.any(mask):
                continue
            locs = local[mask]
            block = self._pending[bi]
            out[pos : pos + locs.shape[0]] = block[locs]
            pos += locs.shape[0]
        return np.ascontiguousarray(out, dtype=np.float32)

    def _build_from_pending(self) -> None:
        if self._index is not None:
            return
        if self._pending_total <= 0:
            raise ValueError("No vectors to build index")

        nlist = self._choose_nlist(self._pending_total)
        quantizer = faiss.IndexFlatL2(self.dim)
        index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_L2)

        m = min(self._train_size, self._pending_total)
        train_x = self._sample_train_vectors_from_pending(m)
        if not index.is_trained:
            index.train(train_x)

        for block in self._pending:
            xb = np.ascontiguousarray(block, dtype=np.float32)
            if xb.shape[1] != self.dim:
                raise ValueError(f"xb dim mismatch: got {xb.shape[1]}, expected {self.dim}")
            index.add(xb)

        index.nprobe = max(1, min(int(self._nprobe), int(nlist)))
        self._index = index
        self._pending.clear()
        self._pending_total = 0

    def add(self, xb: np.ndarray) -> None:
        xb = np.asarray(xb, dtype=np.float32)
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError(f"xb must have shape (N, {self.dim}), got {xb.shape}")
        if xb.shape[0] == 0:
            return

        if self._index is None:
            self._pending.append(xb)
            self._pending_total += int(xb.shape[0])
            self._build_from_pending()
            return

        xb = np.ascontiguousarray(xb, dtype=np.float32)
        self._index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._index is None:
            if self._pending_total > 0:
                self._build_from_pending()
            else:
                raise ValueError("Index is empty; add vectors before searching")

        k = int(k)
        if k <= 0:
            raise ValueError("k must be >= 1")

        xq = np.asarray(xq, dtype=np.float32)
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError(f"xq must have shape (nq, {self.dim}), got {xq.shape}")

        xq = np.ascontiguousarray(xq, dtype=np.float32)
        D, I = self._index.search(xq, k)

        if D.dtype != np.float32:
            D = D.astype(np.float32, copy=False)
        if I.dtype != np.int64:
            I = I.astype(np.int64, copy=False)

        return D, I