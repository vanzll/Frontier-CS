import numpy as np
import os

try:
    import faiss
except ImportError as e:
    faiss = None


class YourIndexClass:
    def __init__(
        self,
        dim: int,
        **kwargs,
    ):
        if faiss is None:
            raise RuntimeError("faiss is required for this index.")
        self.dim = dim

        # Parameters with sensible defaults for high recall within latency budget
        M = int(kwargs.get("M", 32))
        ef_search = int(kwargs.get("ef_search", 640))
        ef_construction = int(kwargs.get("ef_construction", 120))
        use_refine = bool(kwargs.get("use_refine", True))
        kfactor = int(kwargs.get("kfactor", 64))
        threads = kwargs.get("threads", None)

        # Set FAISS threads to available CPUs unless overridden
        if threads is None:
            threads = max(1, os.cpu_count() or 1)
        try:
            faiss.omp_set_num_threads(int(threads))
        except Exception:
            pass

        # Build HNSW Flat index (L2)
        base_index = faiss.IndexHNSWFlat(dim, M)
        base_index.hnsw.efConstruction = ef_construction
        base_index.hnsw.efSearch = ef_search

        # Optional refine step to boost recall with minimal overhead (k=1 -> refine a small candidate set)
        if use_refine:
            index = faiss.IndexRefineFlat(base_index)
            try:
                index.kfactor = kfactor
            except Exception:
                # Older FAISS may not expose kfactor attribute
                pass
            self.index = index
        else:
            self.index = base_index

        self.ef_search = ef_search

    def _set_ef_search(self):
        try:
            if hasattr(self.index, "base_index") and hasattr(self.index.base_index, "hnsw"):
                self.index.base_index.hnsw.efSearch = self.ef_search
            elif hasattr(self.index, "hnsw"):
                self.index.hnsw.efSearch = self.ef_search
        except Exception:
            pass

    def add(self, xb: np.ndarray) -> None:
        if not isinstance(xb, np.ndarray):
            xb = np.asarray(xb, dtype=np.float32)
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32, copy=False)
        xb = np.ascontiguousarray(xb)

        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int):
        if not isinstance(xq, np.ndarray):
            xq = np.asarray(xq, dtype=np.float32)
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32, copy=False)
        xq = np.ascontiguousarray(xq)

        self._set_ef_search()
        D, I = self.index.search(xq, k)
        # Ensure types/shapes
        if D.dtype != np.float32:
            D = D.astype(np.float32, copy=False)
        if I.dtype != np.int64:
            I = I.astype(np.int64, copy=False)
        return D, I