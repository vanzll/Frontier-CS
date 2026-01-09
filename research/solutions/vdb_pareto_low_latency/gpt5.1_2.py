import numpy as np
from typing import Tuple

try:
    import faiss
except ImportError:
    faiss = None


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        """
        Vector index using Faiss IVF-Flat, optimized for low-latency approximate NN search.
        """
        self.dim = dim

        # Parameters with sensible defaults for SIFT1M and latency constraint
        self.nlist = int(kwargs.get("nlist", 4096))          # number of IVF lists
        self.nprobe = int(kwargs.get("nprobe", 64))          # probes at search time
        self.train_size = int(kwargs.get("train_size", 100000))
        self.seed = int(kwargs.get("seed", 123))

        self.index = None          # Faiss index (IVF-Flat) when faiss is available
        self._xb = None            # Fallback storage when faiss is not available
        self.rng = np.random.default_rng(self.seed)

        if faiss is not None:
            try:
                # Use all available threads for best performance
                max_threads = faiss.omp_get_max_threads()
                faiss.omp_set_num_threads(max_threads)
            except Exception:
                # If OpenMP control is not available, silently ignore
                pass

    def _build_faiss_index(self, xb: np.ndarray) -> None:
        """
        Internal helper to build and train the IVF-Flat index on first add().
        """
        assert faiss is not None, "_build_faiss_index requires faiss"

        # Quantizer for IVF (L2 metric)
        quantizer = faiss.IndexFlatL2(self.dim)

        # IVF-Flat index
        index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_L2)

        # Select training samples
        n_train = min(self.train_size, xb.shape[0])
        if n_train < xb.shape[0]:
            train_idx = self.rng.choice(xb.shape[0], size=n_train, replace=False)
            train_x = xb[train_idx].copy()
        else:
            train_x = xb.copy()

        # Ensure proper dtype/contiguity
        if train_x.dtype != np.float32:
            train_x = train_x.astype(np.float32)
        train_x = np.ascontiguousarray(train_x)

        index.train(train_x)
        index.nprobe = self.nprobe

        self.index = index

    def add(self, xb: np.ndarray) -> None:
        """
        Add base vectors to the index. Can be called multiple times (cumulative).
        """
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError(f"xb must have shape (N, {self.dim}), got {xb.shape}")

        if xb.dtype != np.float32:
            xb = xb.astype(np.float32)
        xb = np.ascontiguousarray(xb)

        if faiss is None:
            # Fallback: just store the vectors for later brute-force search
            if self._xb is None:
                self._xb = xb.copy()
            else:
                self._xb = np.vstack((self._xb, xb))
            return

        # Faiss-based index
        if self.index is None:
            # First add: build and train the IVF index, then add vectors
            self._build_faiss_index(xb)

        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of query vectors.
        Returns (distances, indices) with shapes (nq, k).
        """
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError(f"xq must have shape (nq, {self.dim}), got {xq.shape}")

        if k <= 0:
            raise ValueError("k must be positive")

        if xq.dtype != np.float32:
            xq = xq.astype(np.float32)
        xq = np.ascontiguousarray(xq)

        nq = xq.shape[0]

        if faiss is None:
            # Fallback: brute-force search in NumPy (intended only for small datasets)
            if self._xb is None or self._xb.shape[0] == 0:
                # No data added, return empty results
                D = np.full((nq, k), np.float32(np.inf), dtype=np.float32)
                I = np.full((nq, k), -1, dtype=np.int64)
                return D, I

            xb = self._xb
            # Compute full pairwise distances (may be expensive for large N)
            # distances: (nq, N)
            diff = xq[:, None, :] - xb[None, :, :]
            distances = np.einsum('qni,qni->qn', diff, diff, dtype=np.float32)

            if k >= xb.shape[0]:
                # If k >= N, just sort all
                idx_sorted = np.argsort(distances, axis=1)
                D = np.take_along_axis(distances, idx_sorted, axis=1)
                I = idx_sorted.astype(np.int64)
                return D[:, :k].astype(np.float32), I[:, :k]

            # Partial sort to get k smallest
            idx_part = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
            part_dist = np.take_along_axis(distances, idx_part, axis=1)
            # Now sort these k
            order = np.argsort(part_dist, axis=1)
            I = np.take_along_axis(idx_part, order, axis=1).astype(np.int64)
            D = np.take_along_axis(part_dist, order, axis=1).astype(np.float32)
            return D, I

        # Faiss-based search
        if self.index is None or self.index.ntotal == 0:
            # No data added, return empty results
            D = np.full((nq, k), np.float32(np.inf), dtype=np.float32)
            I = np.full((nq, k), -1, dtype=np.int64)
            return D, I

        # Ensure nprobe is set (in case it was modified externally)
        try:
            self.index.nprobe = self.nprobe
        except AttributeError:
            pass

        D, I = self.index.search(xq, k)
        # Faiss returns float32 distances and int64 indices already
        return D, I