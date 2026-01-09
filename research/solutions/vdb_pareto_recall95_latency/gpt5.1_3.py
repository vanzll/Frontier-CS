import numpy as np
import faiss
from typing import Tuple


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        """
        Faiss IVF-Flat index optimized for high recall and good latency.
        """
        self.dim = int(dim)

        # Hyperparameters with sensible defaults for SIFT1M
        self.target_nlist = int(kwargs.get("nlist", 4096))
        if self.target_nlist <= 0:
            self.target_nlist = 1

        self.nprobe = int(kwargs.get("nprobe", 64))
        if self.nprobe <= 0:
            self.nprobe = 1

        self.max_train_points = int(kwargs.get("max_train_points", 100000))
        if self.max_train_points <= 0:
            self.max_train_points = 100000

        num_threads = kwargs.get("num_threads", None)
        if num_threads is not None:
            try:
                faiss.omp_set_num_threads(int(num_threads))
            except Exception:
                pass

        self.index = None
        self.nlist = None

    def _build_index(self, xb: np.ndarray) -> None:
        n, d = xb.shape
        if d != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {d}")

        # Choose nlist adapted to data size: roughly at most N/40
        max_nlist_based_on_data = max(1, n // 40)
        nlist = min(self.target_nlist, max_nlist_based_on_data)
        if nlist < 1:
            nlist = 1
        self.nlist = nlist

        quantizer = faiss.IndexFlatL2(self.dim)
        index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_L2)
        index.nprobe = self.nprobe

        # Training
        n_train = min(n, self.max_train_points)
        if n_train < nlist:
            n_train = n  # ensure at least nlist training points

        if n_train < n:
            rng = np.random.default_rng(123)
            train_idx = rng.choice(n, size=n_train, replace=False)
            train_x = xb[train_idx].copy()
        else:
            train_x = xb.copy()

        index.train(train_x)
        self.index = index

    def add(self, xb: np.ndarray) -> None:
        if xb is None or xb.size == 0:
            return

        xb = np.ascontiguousarray(xb, dtype=np.float32)
        n, d = xb.shape
        if d != self.dim:
            raise ValueError(f"Vector dimensionality mismatch: expected {self.dim}, got {d}")

        if self.index is None:
            self._build_index(xb)

        if not self.index.is_trained:
            # Fallback: (re)build and train if for some reason still untrained
            self._build_index(xb)

        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None or self.index.ntotal == 0:
            nq = xq.shape[0]
            D = np.empty((nq, k), dtype=np.float32)
            I = np.full((nq, k), -1, dtype=np.int64)
            return D, I

        xq = np.ascontiguousarray(xq, dtype=np.float32)
        D, I = self.index.search(xq, k)

        if D.dtype != np.float32:
            D = D.astype(np.float32)
        if I.dtype != np.int64:
            I = I.astype(np.int64)

        return D, I