import numpy as np
from typing import Tuple

try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.

        Args:
            dim: Vector dimensionality
            **kwargs:
                - M: HNSW connectivity parameter (default: 16)
                - ef_construction / efConstruction: HNSW construction parameter (default: 200)
                - ef_search / efSearch: HNSW search parameter (default: 128)
        """
        self.dim = dim
        self.ntotal = 0

        if _FAISS_AVAILABLE:
            M = int(kwargs.get("M", 16))
            ef_construction = int(
                kwargs.get(
                    "ef_construction",
                    kwargs.get("efConstruction", 200),
                )
            )
            self.ef_search = int(
                kwargs.get(
                    "ef_search",
                    kwargs.get("efSearch", 128),
                )
            )

            # HNSW index with L2 metric
            self.index = faiss.IndexHNSWFlat(dim, M)
            self.index.hnsw.efConstruction = ef_construction
            self.index.hnsw.efSearch = self.ef_search
        else:
            # Fallback: brute-force index (not suitable for large datasets, but keeps API functional)
            self.index = None
            self._xb = None  # type: ignore

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            xb: Base vectors, shape (N, dim), dtype float32
        """
        if xb is None:
            return

        xb = np.ascontiguousarray(xb, dtype=np.float32)
        if xb.shape[1] != self.dim:
            raise ValueError(f"Input dimension {xb.shape[1]} does not match index dim {self.dim}")

        if _FAISS_AVAILABLE:
            self.index.add(xb)
            self.ntotal = self.index.ntotal
        else:  # pragma: no cover
            if self._xb is None:
                self._xb = xb.copy()
            else:
                self._xb = np.concatenate([self._xb, xb], axis=0)
            self.ntotal = self._xb.shape[0]

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
        if self.ntotal == 0:
            raise ValueError("The index is empty. Call add() before search().")

        xq = np.ascontiguousarray(xq, dtype=np.float32)
        if xq.shape[1] != self.dim:
            raise ValueError(f"Query dimension {xq.shape[1]} does not match index dim {self.dim}")

        if k <= 0:
            raise ValueError("k must be >= 1")

        if _FAISS_AVAILABLE:
            # Ensure efSearch is set (in case user modified self.ef_search)
            self.index.hnsw.efSearch = self.ef_search
            D, I = self.index.search(xq, k)
            # Ensure numpy dtypes
            D = np.asarray(D, dtype=np.float32)
            I = np.asarray(I, dtype=np.int64)
            return D, I

        # Fallback brute-force search (slow, for environments without Faiss)  # pragma: no cover
        xb = self._xb
        nq = xq.shape[0]
        N = xb.shape[0]
        k_eff = min(k, N)

        # Initialize best distances and indices
        best_dist = np.full((nq, k_eff), np.inf, dtype=np.float32)
        best_idx = np.full((nq, k_eff), -1, dtype=np.int64)

        # Precompute query norms
        xq_norms = (xq ** 2).sum(axis=1, keepdims=True).astype(np.float32)

        block_size = 512
        for start in range(0, N, block_size):
            end = min(start + block_size, N)
            xb_block = xb[start:end]
            nb = xb_block.shape[0]

            xb_block_norms = (xb_block ** 2).sum(axis=1, keepdims=True).T.astype(np.float32)  # (1, nb)
            # Distances: ||xq - xb||^2 = ||xq||^2 + ||xb||^2 - 2 xq·xb^T
            cross = np.dot(xq, xb_block.T).astype(np.float32)
            dist_block = xq_norms + xb_block_norms - 2.0 * cross  # (nq, nb)

            # Combine current best and new block
            cand_dist = np.concatenate([best_dist, dist_block], axis=1)  # (nq, k_eff + nb)
            idx_block = np.arange(start, end, dtype=np.int64)
            cand_idx_block = np.broadcast_to(idx_block[None, :], (nq, nb))
            cand_idx = np.concatenate([best_idx, cand_idx_block], axis=1)

            # Partial sort to get k_eff smallest distances
            part_idx = np.argpartition(cand_dist, k_eff - 1, axis=1)[:, :k_eff]
            row_idx = np.arange(nq)[:, None]
            best_dist = cand_dist[row_idx, part_idx]
            best_idx = cand_idx[row_idx, part_idx]

            # Sort within k_eff
            order = np.argsort(best_dist, axis=1)
            best_dist = best_dist[row_idx, order]
            best_idx = best_idx[row_idx, order]

        # If k > N, pad with -1 and inf to match requested k
        if k_eff < k:
            pad_cols = k - k_eff
            pad_dist = np.full((nq, pad_cols), np.inf, dtype=np.float32)
            pad_idx = np.full((nq, pad_cols), -1, dtype=np.int64)
            best_dist = np.concatenate([best_dist, pad_dist], axis=1)
            best_idx = np.concatenate([best_idx, pad_idx], axis=1)

        return best_dist[:, :k], best_idx[:, :k]