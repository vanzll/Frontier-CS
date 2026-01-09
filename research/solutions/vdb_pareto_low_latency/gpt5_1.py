import numpy as np
from typing import Tuple

try:
    import faiss
except Exception:
    faiss = None


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = dim
        self.nlist = int(kwargs.get("nlist", 16384))
        self.m = int(kwargs.get("m", 16))  # PQ subquantizers
        self.nbits = int(kwargs.get("nbits", 8))  # bits per subquantizer
        self.use_opq = bool(kwargs.get("opq", True))
        self.opq_m = int(kwargs.get("opq_m", self.m))
        self.use_hnsw_quantizer = bool(kwargs.get("hnsw_quantizer", True))
        self.hnsw_m = int(kwargs.get("hnsw_m", 32))
        self.hnsw_ef_search = int(kwargs.get("hnsw_ef_search", 64))
        self.hnsw_ef_construction = int(kwargs.get("hnsw_ef_construction", 200))
        self.nprobe = int(kwargs.get("nprobe", 8))
        self.train_size = int(kwargs.get("train_size", 120000))
        self.random_seed = int(kwargs.get("seed", 123))
        self.num_threads = kwargs.get("num_threads", None)

        self.index = None
        self._trained = False
        self._added = 0

        if faiss is not None and self.num_threads is not None:
            try:
                faiss.omp_set_num_threads(int(self.num_threads))
            except Exception:
                pass

    def _build_index(self):
        if faiss is None:
            self.index = None
            return

        if self.use_hnsw_quantizer:
            quantizer = faiss.IndexHNSWFlat(self.dim, self.hnsw_m)
            quantizer.hnsw.efSearch = self.hnsw_ef_search
            quantizer.hnsw.efConstruction = self.hnsw_ef_construction
        else:
            quantizer = faiss.IndexFlatL2(self.dim)

        ivfpq = faiss.IndexIVFPQ(quantizer, self.dim, self.nlist, self.m, self.nbits)

        if self.use_opq:
            # OPQ rotation before IVF-PQ
            opq = faiss.OPQMatrix(self.dim, self.opq_m)
            # Slightly fewer iterations for speed without major recall loss
            opq.niter = 20
            self.index = faiss.IndexPreTransform(opq, ivfpq)
        else:
            self.index = ivfpq

        # Set nprobe after build (will be valid after training)
        if isinstance(self.index, faiss.IndexPreTransform):
            base = faiss.downcast_index(self.index.index)
            if isinstance(base, faiss.IndexIVF):
                base.nprobe = self.nprobe
        else:
            if isinstance(self.index, faiss.IndexIVF):
                self.index.nprobe = self.nprobe

    def _ensure_trained(self, x: np.ndarray):
        if self._trained or faiss is None:
            return
        if self.index is None:
            self._build_index()

        # Sample a subset for training
        rs = np.random.RandomState(self.random_seed)
        n = x.shape[0]
        train_n = min(self.train_size, n)
        if train_n < n:
            idx = rs.choice(n, train_n, replace=False)
            xtrain = np.ascontiguousarray(x[idx], dtype=np.float32)
        else:
            xtrain = np.ascontiguousarray(x, dtype=np.float32)

        # Train index (trains OPQ, coarse kmeans centroids, and PQ codebooks)
        self.index.train(xtrain)

        # If using HNSW quantizer, make sure its search params are set
        # and nprobe configured on IVF
        if isinstance(self.index, faiss.IndexPreTransform):
            base = faiss.downcast_index(self.index.index)
            if isinstance(base, faiss.IndexIVF):
                # Re-set HNSW params just in case train rebuilt the quantizer data
                if isinstance(base.quantizer, faiss.IndexHNSWFlat):
                    base.quantizer.hnsw.efSearch = self.hnsw_ef_search
                    base.quantizer.hnsw.efConstruction = self.hnsw_ef_construction
                base.nprobe = self.nprobe
        else:
            if isinstance(self.index, faiss.IndexIVF):
                if isinstance(self.index.quantizer, faiss.IndexHNSWFlat):
                    self.index.quantizer.hnsw.efSearch = self.hnsw_ef_search
                    self.index.quantizer.hnsw.efConstruction = self.hnsw_ef_construction
                self.index.nprobe = self.nprobe

        self._trained = True

    def add(self, xb: np.ndarray) -> None:
        if faiss is None:
            # Fallback: store raw data (will be extremely slow in search)
            if not hasattr(self, "_xb"):
                self._xb = np.ascontiguousarray(xb, dtype=np.float32)
            else:
                self._xb = np.vstack((self._xb, np.ascontiguousarray(xb, dtype=np.float32)))
            self._added = self._xb.shape[0]
            return

        xb = np.ascontiguousarray(xb, dtype=np.float32)
        if self.index is None:
            self._build_index()

        if not self._trained:
            self._ensure_trained(xb)

        self.index.add(xb)
        self._added += xb.shape[0]

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        xq = np.ascontiguousarray(xq, dtype=np.float32)

        if faiss is None:
            # Extremely slow fallback linear search
            if not hasattr(self, "_xb") or self._xb is None:
                N = xq.shape[0]
                return np.full((N, k), np.inf, dtype=np.float32), -np.ones((N, k), dtype=np.int64)
            # Compute distances
            # Using broadcasting; memory heavy but fallback path only
            xb = self._xb
            q_norms = np.sum(xq ** 2, axis=1, keepdims=True)
            b_norms = np.sum(xb ** 2, axis=1, keepdims=True).T
            distances = q_norms + b_norms - 2 * xq.dot(xb.T)
            idx = np.argpartition(distances, k - 1, axis=1)[:, :k]
            part_vals = distances[np.arange(distances.shape[0])[:, None], idx]
            order = np.argsort(part_vals, axis=1)
            final_idx = idx[np.arange(idx.shape[0])[:, None], order]
            final_dist = distances[np.arange(distances.shape[0])[:, None], final_idx]
            return final_dist.astype(np.float32), final_idx.astype(np.int64)

        if self.index is None or (isinstance(self.index, faiss.Index) and not self.index.is_trained):
            # If index was not properly initialized/trained, return empty results
            N = xq.shape[0]
            return np.full((N, k), np.inf, dtype=np.float32), -np.ones((N, k), dtype=np.int64)

        # Ensure IVF nprobe parameter is set before search
        if isinstance(self.index, faiss.IndexPreTransform):
            base = faiss.downcast_index(self.index.index)
            if isinstance(base, faiss.IndexIVF):
                base.nprobe = self.nprobe
                if isinstance(base.quantizer, faiss.IndexHNSWFlat):
                    base.quantizer.hnsw.efSearch = self.hnsw_ef_search
        else:
            if isinstance(self.index, faiss.IndexIVF):
                self.index.nprobe = self.nprobe
                if isinstance(self.index.quantizer, faiss.IndexHNSWFlat):
                    self.index.quantizer.hnsw.efSearch = self.hnsw_ef_search

        D, I = self.index.search(xq, k)
        if D is None or I is None:
            N = xq.shape[0]
            return np.full((N, k), np.inf, dtype=np.float32), -np.ones((N, k), dtype=np.int64)
        return D.astype(np.float32), I.astype(np.int64)