import numpy as np
from typing import Tuple

try:
    import faiss
except Exception as e:
    faiss = None


class YourIndexClass:
    def __init__(
        self,
        dim: int,
        nlist: int = 8192,
        nprobe: int = 3,
        hnsw_M: int = 32,
        hnsw_efSearch: int = 64,
        hnsw_efConstruction: int = 40,
        train_size: int = 250000,
        random_seed: int = 1234,
        threads: int = 0,
        **kwargs
    ):
        self.dim = dim
        self.nlist = int(nlist)
        self.nprobe = int(nprobe)
        self.hnsw_M = int(hnsw_M)
        self.hnsw_efSearch = int(hnsw_efSearch)
        self.hnsw_efConstruction = int(hnsw_efConstruction)
        self.train_size = int(train_size)
        self.random_seed = int(random_seed)
        self.threads = int(threads) if int(threads) > 0 else None

        self.index = None
        self._is_trained = False

        # Buffer to support multiple add() calls pre-training if needed
        self._xb_buffer = []
        self._ntotal = 0

        # Initialize FAISS threading if available
        if faiss is not None:
            try:
                max_threads = faiss.omp_get_max_threads()
                use_threads = max_threads if self.threads is None else min(self.threads, max_threads)
                faiss.omp_set_num_threads(use_threads)
            except Exception:
                pass

        # Seeding numpy for reproducibility in sampling
        np.random.seed(self.random_seed)

    def _build_index(self):
        if faiss is None:
            raise RuntimeError("FAISS is required for this index implementation.")
        # HNSW quantizer for fast coarse assignment
        quantizer = faiss.IndexHNSWFlat(self.dim, self.hnsw_M)
        quantizer.hnsw.efConstruction = self.hnsw_efConstruction
        quantizer.hnsw.efSearch = max(self.hnsw_efSearch, self.nprobe)

        index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_L2)
        index.nprobe = self.nprobe
        self.index = index

    def _train_if_needed(self, xb: np.ndarray):
        if self._is_trained:
            return

        if self.index is None:
            self._build_index()

        # Determine training set
        n = xb.shape[0]
        train_n = min(self.train_size, n)
        if train_n < self.nlist:
            # Ensure at least nlist training points
            train_n = min(max(self.nlist, 2 * self.nlist), n)

        if train_n < self.nlist:
            # If still not enough, will fallback to training on all available
            train_n = n

        if train_n == n:
            train_x = xb
        else:
            # Sample without replacement
            sel = np.random.choice(n, train_n, replace=False)
            train_x = xb[sel]

        self.index.train(train_x)
        # After training, set quantizer's efSearch again (some faiss versions may reset)
        if hasattr(self.index, "quantizer") and hasattr(self.index.quantizer, "hnsw"):
            self.index.quantizer.hnsw.efSearch = max(self.hnsw_efSearch, self.nprobe)
        self._is_trained = True

    def add(self, xb: np.ndarray) -> None:
        if not isinstance(xb, np.ndarray):
            xb = np.asarray(xb, dtype=np.float32)
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32, copy=False)
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError("xb must have shape (N, dim) with dtype float32")

        # If not trained, train on this batch (or subset)
        if not self._is_trained:
            self._train_if_needed(xb)

        # Add to index directly
        self.index.add(xb)
        self._ntotal += xb.shape[0]

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if not isinstance(xq, np.ndarray):
            xq = np.asarray(xq, dtype=np.float32)
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32, copy=False)
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError("xq must have shape (nq, dim) with dtype float32")
        if k <= 0:
            raise ValueError("k must be positive")
        if self.index is None or not self._is_trained or self._ntotal == 0:
            # Fallback to brute-force if index is not ready
            xb = None
            if self.index is not None and hasattr(self.index, "reconstruct_n"):
                try:
                    xb = self.index.reconstruct_n(0, self._ntotal)
                except Exception:
                    xb = None
            if xb is None:
                # No data to search
                nq = xq.shape[0]
                return np.full((nq, k), np.inf, dtype=np.float32), np.full((nq, k), -1, dtype=np.int64)

            diffs = xq[:, None, :] - xb[None, :, :]
            dists = np.einsum('qnd,qnd->qn', diffs, diffs, optimize=True)
            if k >= dists.shape[1]:
                idx = np.argsort(dists, axis=1)[:, :k]
            else:
                part = np.argpartition(dists, k - 1, axis=1)[:, :k]
                # sort the top-k partitions
                rows = np.arange(dists.shape[0])[:, None]
                sorted_idx = np.argsort(dists[rows, part], axis=1)
                idx = part[rows, sorted_idx]
            rows = np.arange(dists.shape[0])[:, None]
            final_d = dists[rows, idx].astype(np.float32, copy=False)
            final_i = idx.astype(np.int64, copy=False)
            return final_d, final_i

        # Ensure nprobe and efSearch are set
        self.index.nprobe = self.nprobe
        if hasattr(self.index, "quantizer") and hasattr(self.index.quantizer, "hnsw"):
            self.index.quantizer.hnsw.efSearch = max(self.hnsw_efSearch, self.nprobe)

        D, I = self.index.search(xq, k)
        if D is None or I is None:
            nq = xq.shape[0]
            return np.full((nq, k), np.inf, dtype=np.float32), np.full((nq, k), -1, dtype=np.int64)

        # Ensure correct dtypes
        if D.dtype != np.float32:
            D = D.astype(np.float32, copy=False)
        if I.dtype != np.int64:
            I = I.astype(np.int64, copy=False)
        return D, I