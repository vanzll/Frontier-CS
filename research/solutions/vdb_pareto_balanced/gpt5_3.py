import numpy as np
import os

try:
    import faiss
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = int(dim)
        self._use_faiss = _FAISS_AVAILABLE
        self.ntotal = 0

        # Parameters with sensible defaults for high recall under latency constraint
        M = int(kwargs.get("M", 24))
        ef_construction = int(kwargs.get("ef_construction", 200))
        ef_search = int(kwargs.get("ef_search", 128))
        num_threads = kwargs.get("num_threads", None)

        if self._use_faiss:
            # Configure threads
            try:
                if num_threads is None:
                    # Use available CPUs but cap at a reasonable number to avoid oversubscription
                    num_threads = max(1, min(os.cpu_count() or 1, 16))
                faiss.omp_set_num_threads(int(num_threads))
            except Exception:
                pass

            # Build HNSW Flat index for L2 metric
            self.index = faiss.IndexHNSWFlat(self.dim, M)
            # Set construction/search parameters
            try:
                self.index.hnsw.efConstruction = ef_construction
            except Exception:
                pass
            try:
                self.index.hnsw.efSearch = ef_search
            except Exception:
                pass
        else:
            # Fallback to simple storage and brute-force search (for environments without faiss)
            self.index = None
            self._xb = None

    def add(self, xb: np.ndarray) -> None:
        if not isinstance(xb, np.ndarray):
            xb = np.asarray(xb, dtype=np.float32)
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32, copy=False)
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError("xb must have shape (N, dim) with dim matching initialized dimension")

        n = xb.shape[0]
        if self._use_faiss:
            self.index.add(xb)
            self.ntotal = self.index.ntotal
        else:
            if self._xb is None:
                self._xb = xb.copy()
            else:
                self._xb = np.vstack((self._xb, xb))
            self.ntotal = self._xb.shape[0]

    def search(self, xq: np.ndarray, k: int):
        if not isinstance(xq, np.ndarray):
            xq = np.asarray(xq, dtype=np.float32)
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32, copy=False)
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError("xq must have shape (nq, dim) with dim matching initialized dimension")
        if k <= 0:
            raise ValueError("k must be positive")
        if self.ntotal == 0:
            # No data added; return empty-like results
            nq = xq.shape[0]
            return np.full((nq, k), np.inf, dtype=np.float32), np.full((nq, k), -1, dtype=np.int64)
        k = min(k, max(1, self.ntotal))  # ensure k does not exceed ntotal

        if self._use_faiss:
            D, I = self.index.search(xq, k)
            # Ensure types
            if D.dtype != np.float32:
                D = D.astype(np.float32, copy=False)
            if I.dtype != np.int64:
                I = I.astype(np.int64, copy=False)
            return D, I
        else:
            # Brute-force fallback: compute squared L2 distances in chunks to save memory
            nq = xq.shape[0]
            xb = self._xb
            N = xb.shape[0]

            # Precompute norms
            xq_sq = np.sum(xq ** 2, axis=1, keepdims=True)  # (nq, 1)
            xb_sq = np.sum(xb ** 2, axis=1, keepdims=True).ravel()  # (N,)

            # Choose a chunk size for xb to fit in memory and speed
            # Target around ~128MB per chunk for safety
            bytes_per_vec = xb.shape[1] * 4
            target_chunk_bytes = 128 * 1024 * 1024
            chunk_size = max(1, min(N, target_chunk_bytes // max(bytes_per_vec, 1)))

            best_D = np.full((nq, k), np.inf, dtype=np.float32)
            best_I = np.full((nq, k), -1, dtype=np.int64)

            for start in range(0, N, chunk_size):
                end = min(N, start + chunk_size)
                xb_chunk = xb[start:end]
                # Compute distances: ||xq - xb||^2 = ||xq||^2 + ||xb||^2 - 2 xq xb^T
                # Use matrix multiplication for speed
                cross = xq @ xb_chunk.T  # (nq, chunk)
                D_chunk = xq_sq + xb_sq[start:end][None, :] - 2.0 * cross  # (nq, chunk)
                # Partial top-k merge
                # Get top-k from combined existing best and this chunk
                cand_D = np.hstack((best_D, D_chunk))
                cand_I = np.hstack((best_I, np.arange(start, end, dtype=np.int64)[None, :].repeat(nq, axis=0)))
                # Argpartition to get k smallest
                idx_part = np.argpartition(cand_D, kth=k - 1, axis=1)[:, :k]
                row_indices = np.arange(nq)[:, None]
                part_D = cand_D[row_indices, idx_part]
                part_I = cand_I[row_indices, idx_part]
                # Sort within the k
                order = np.argsort(part_D, axis=1)
                best_D = part_D[row_indices, order]
                best_I = part_I[row_indices, order]

            return best_D, best_I