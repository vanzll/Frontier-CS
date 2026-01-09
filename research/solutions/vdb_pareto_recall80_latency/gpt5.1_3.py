import numpy as np
from typing import Tuple

try:
    import faiss
except ImportError:
    faiss = None


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.

        Args:
            dim: Vector dimensionality (e.g., 128 for SIFT1M)
            **kwargs: Optional parameters
                - M: HNSW graph degree (default: 16)
                - ef_construction: HNSW construction parameter (default: 80)
                - ef_search: HNSW search parameter (default: 64)
                - num_threads: number of threads for Faiss (default: faiss.omp_get_max_threads())
        """
        self.dim = dim
        self.use_faiss = faiss is not None
        self.xb = None  # used only in fallback (no-faiss) mode

        if self.use_faiss:
            m = int(kwargs.get("M", 16))
            ef_construction = int(kwargs.get("ef_construction", 80))
            ef_search = int(kwargs.get("ef_search", 64))

            num_threads = kwargs.get("num_threads", None)
            try:
                if num_threads is None:
                    num_threads = faiss.omp_get_max_threads()
                faiss.omp_set_num_threads(int(num_threads))
            except Exception:
                pass

            index = faiss.IndexHNSWFlat(dim, m)
            index.hnsw.efConstruction = ef_construction
            index.hnsw.efSearch = ef_search
            self.index = index
        else:
            self.index = None

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            xb: Base vectors, shape (N, dim), dtype float32
        """
        xb = np.ascontiguousarray(xb, dtype="float32")
        if self.use_faiss:
            if self.index is None:
                raise RuntimeError("Faiss index is not initialized.")
            if self.index.ntotal == 0 and self.index.is_trained is False and self.index.requires_training:
                self.index.train(xb)
            self.index.add(xb)
        else:
            if self.xb is None:
                self.xb = xb.copy()
            else:
                self.xb = np.vstack((self.xb, xb))

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of query vectors.

        Args:
            xq: Query vectors, shape (nq, dim), dtype float32
            k: Number of nearest neighbors to return

        Returns:
            (distances, indices):
                - distances: shape (nq, k), dtype float32, L2 distances (or L2-squared)
                - indices: shape (nq, k), dtype int64, indices into base vectors
        """
        xq = np.ascontiguousarray(xq, dtype="float32")
        if self.use_faiss:
            if self.index is None or self.index.ntotal == 0:
                raise RuntimeError("Index is empty or not initialized.")
            distances, indices = self.index.search(xq, k)
            if indices.dtype != np.int64:
                indices = indices.astype(np.int64)
            if distances.dtype != np.float32:
                distances = distances.astype(np.float32)
            return distances, indices
        else:
            if self.xb is None or self.xb.shape[0] == 0:
                raise RuntimeError("Index is empty.")
            # Brute-force fallback (slow, used only if faiss is unavailable)
            xb = self.xb
            # Compute squared L2 distances efficiently: ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x·y
            xq_sq = np.sum(xq ** 2, axis=1, keepdims=True)  # (nq, 1)
            xb_sq = np.sum(xb ** 2, axis=1, keepdims=True).T  # (1, N)
            cross = xq @ xb.T  # (nq, N)
            distances = xq_sq + xb_sq - 2.0 * cross  # (nq, N)
            # Select k smallest distances
            if k == 1:
                indices = np.argmin(distances, axis=1, keepdims=True)
                d_min = distances[np.arange(distances.shape[0])[:, None], indices]
                return d_min.astype(np.float32), indices.astype(np.int64)
            else:
                idx_part = np.argpartition(distances, k - 1, axis=1)[:, :k]
                row_indices = np.arange(distances.shape[0])[:, None]
                part_dist = distances[row_indices, idx_part]
                order = np.argsort(part_dist, axis=1)
                final_idx = idx_part[row_indices, order]
                final_dist = distances[row_indices, final_idx]
                return final_dist.astype(np.float32), final_idx.astype(np.int64)