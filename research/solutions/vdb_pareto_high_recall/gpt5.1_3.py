import numpy as np
from typing import Tuple
import faiss


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.
        Uses exact L2 search with Faiss IndexFlatL2 for maximum recall.
        """
        self.dim = int(dim)
        self.index = faiss.IndexFlatL2(self.dim)

        # Optional: allow overriding number of threads if provided
        num_threads = kwargs.get("num_threads", None)
        if num_threads is not None and hasattr(faiss, "omp_set_num_threads"):
            try:
                n = int(num_threads)
                if n > 0:
                    faiss.omp_set_num_threads(n)
            except Exception:
                pass

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.
        """
        if xb is None:
            return

        if xb.ndim != 2 or xb.shape[1] != self.dim:
            xb = xb.reshape(-1, self.dim)

        if xb.dtype != np.float32:
            xb = xb.astype(np.float32, copy=False)

        xb = np.ascontiguousarray(xb)
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of query vectors.
        """
        k = int(k)

        if xq is None:
            return (
                np.empty((0, k), dtype=np.float32),
                np.empty((0, k), dtype=np.int64),
            )

        if xq.ndim != 2 or xq.shape[1] != self.dim:
            xq = xq.reshape(-1, self.dim)

        if xq.dtype != np.float32:
            xq = xq.astype(np.float32, copy=False)

        xq = np.ascontiguousarray(xq)
        D, I = self.index.search(xq, k)
        return D, I