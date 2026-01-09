import numpy as np
from typing import Tuple

try:
    import faiss
except ImportError:
    faiss = None


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        """
        High-recall HNSW-based index using FAISS.

        Args:
            dim: Vector dimensionality.
            M: HNSW parameter (number of neighbors per layer). Default: 32.
            ef_construction / efConstruction: HNSW construction parameter. Default: 200.
            ef_search / efSearch: HNSW search parameter. Default: 512.
            metric: 'l2' or 'ip'. Default: 'l2'.
            num_threads: Optional, number of OpenMP threads for FAISS.
        """
        self.dim = int(dim)

        # Hyperparameters with sensible defaults for high recall within latency budget
        self.M = int(kwargs.get("M", 32))
        self.ef_construction = int(
            kwargs.get("ef_construction", kwargs.get("efConstruction", 200))
        )
        self.ef_search = int(
            kwargs.get("ef_search", kwargs.get("efSearch", 512))
        )
        self.metric = kwargs.get("metric", "l2")

        self.index = None
        self.xb = None  # used only as a fallback if faiss is unavailable

        if faiss is not None:
            # Configure FAISS threading if requested
            num_threads = kwargs.get("num_threads", None)
            if num_threads is not None:
                try:
                    faiss.omp_set_num_threads(int(num_threads))
                except Exception:
                    pass

            # Determine metric type
            metric_type = faiss.METRIC_L2
            if isinstance(self.metric, str):
                m = self.metric.lower()
                if m in ("l2", "euclidean", "l2sqr", "l2sq"):
                    metric_type = faiss.METRIC_L2
                elif m in ("ip", "dot", "inner_product"):
                    metric_type = faiss.METRIC_INNER_PRODUCT
            elif isinstance(self.metric, int):
                metric_type = self.metric

            # Create HNSW index
            try:
                self.index = faiss.IndexHNSWFlat(self.dim, self.M, metric_type)
            except TypeError:
                # Older FAISS versions without metric argument
                self.index = faiss.IndexHNSWFlat(self.dim, self.M)

            # Configure HNSW parameters before any additions
            hnsw = self.index.hnsw
            hnsw.efConstruction = self.ef_construction
            hnsw.efSearch = max(self.ef_search, 1)

    def add(self, xb: np.ndarray) -> None:
        if xb is None or xb.size == 0:
            return

        xb = np.ascontiguousarray(xb, dtype="float32")
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError(f"Expected xb shape (N, {self.dim}), got {xb.shape}")

        if self.index is not None:
            self.index.add(xb)
        else:
            # Fallback: store vectors for brute-force search
            if self.xb is None:
                self.xb = xb.copy()
            else:
                self.xb = np.vstack((self.xb, xb))

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if k <= 0:
            raise ValueError("k must be >= 1")

        xq = np.ascontiguousarray(xq, dtype="float32")
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError(f"Expected xq shape (nq, {self.dim}), got {xq.shape}")
        nq = xq.shape[0]

        # If using FAISS HNSW
        if self.index is not None:
            if self.index.ntotal == 0:
                D = np.full((nq, k), np.inf, dtype="float32")
                I = np.full((nq, k), -1, dtype="int64")
                return D, I

            # Ensure efSearch is at least k for good quality
            if hasattr(self.index, "hnsw"):
                desired_ef = max(self.ef_search, k)
                if self.index.hnsw.efSearch != desired_ef:
                    self.index.hnsw.efSearch = desired_ef

            D, I = self.index.search(xq, k)
            return D, I

        # Fallback: brute-force L2 search (very slow for large N, but always correct)
        if self.xb is None or self.xb.shape[0] == 0:
            D = np.full((nq, k), np.inf, dtype="float32")
            I = np.full((nq, k), -1, dtype="int64")
            return D, I

        xb = self.xb
        N = xb.shape[0]

        # Precompute squared norms
        xq_norms = np.sum(xq ** 2, axis=1, keepdims=True)  # (nq, 1)
        xb_norms = np.sum(xb ** 2, axis=1, keepdims=True).T  # (1, N)

        D_out = np.empty((nq, k), dtype="float32")
        I_out = np.empty((nq, k), dtype="int64")

        # Blocked computation over queries to control memory usage
        # This code path is only used if FAISS is unavailable.
        block_q = 128
        for i0 in range(0, nq, block_q):
            i1 = min(i0 + block_q, nq)
            sub_xq = xq[i0:i1]
            sub_norms = xq_norms[i0:i1]

            # Compute squared L2 distances using the identity:
            # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x·y
            dots = sub_xq @ xb.T  # (nb, N)
            dists = sub_norms + xb_norms - 2.0 * dots  # (nb, N)

            # Partial sort to get top-k
            idx_part = np.argpartition(dists, k - 1, axis=1)[:, :k]
            dists_part = np.take_along_axis(dists, idx_part, axis=1)
            order = np.argsort(dists_part, axis=1)
            final_idx = np.take_along_axis(idx_part, order, axis=1)
            final_dists = np.take_along_axis(dists_part, order, axis=1)

            D_out[i0:i1] = final_dists.astype("float32")
            I_out[i0:i1] = final_idx.astype("int64")

        return D_out, I_out