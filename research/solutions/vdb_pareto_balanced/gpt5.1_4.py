import numpy as np
import faiss
from typing import Tuple


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.

        Optional kwargs:
            num_threads: int, number of threads for FAISS (defaults to FAISS default)
        """
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)

        num_threads = kwargs.get("num_threads", None)
        if num_threads is not None:
            try:
                faiss.omp_set_num_threads(int(num_threads))
            except Exception:
                pass

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            xb: Base vectors, shape (N, dim), dtype float32
        """
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError(f"xb must have shape (N, {self.dim})")

        if xb.dtype != np.float32:
            xb = xb.astype(np.float32, copy=False)

        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of query vectors.

        Args:
            xq: Query vectors, shape (nq, dim), dtype float32
            k: Number of nearest neighbors to return

        Returns:
            (distances, indices):
                - distances: shape (nq, k), dtype float32, L2-squared distances
                - indices: shape (nq, k), dtype int64, indices into base vectors
        """
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError(f"xq must have shape (nq, {self.dim})")

        if xq.dtype != np.float32:
            xq = xq.astype(np.float32, copy=False)

        if k <= 0:
            raise ValueError("k must be positive")

        nq = xq.shape[0]
        n_db = self.index.ntotal

        if n_db == 0:
            distances = np.full((nq, k), np.inf, dtype=np.float32)
            indices = np.full((nq, k), -1, dtype=np.int64)
            return distances, indices

        k_search = min(k, n_db)
        D, I = self.index.search(xq, k_search)

        if k_search == k:
            return D, I

        # If requested k > number of database vectors, pad results
        distances = np.full((nq, k), np.inf, dtype=np.float32)
        indices = np.full((nq, k), -1, dtype=np.int64)
        distances[:, :k_search] = D
        indices[:, :k_search] = I

        return distances, indices