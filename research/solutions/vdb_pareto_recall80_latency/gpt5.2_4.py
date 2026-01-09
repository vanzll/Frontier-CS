import numpy as np
from typing import Tuple

import faiss


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = int(dim)

        self.nlist = int(kwargs.get("nlist", 4096))
        self.m = int(kwargs.get("m", 16))
        self.nbits = int(kwargs.get("nbits", 8))
        self.nprobe = int(kwargs.get("nprobe", 16))

        self.train_size = int(kwargs.get("train_size", 200_000))
        self.nthreads = int(kwargs.get("nthreads", 8))

        self.use_opq = bool(kwargs.get("use_opq", True))
        self.opq_niter = int(kwargs.get("opq_niter", 10))
        self.kmeans_niter = int(kwargs.get("kmeans_niter", 15))
        self.max_points_per_centroid = int(kwargs.get("max_points_per_centroid", 256))

        try:
            faiss.omp_set_num_threads(self.nthreads)
        except Exception:
            pass

        self._rng = np.random.RandomState(int(kwargs.get("seed", 12345)))

        self.index = None
        self._ivf = None
        self._built = False

        self._create_index()

    def _create_index(self) -> None:
        quantizer = faiss.IndexFlatL2(self.dim)
        ivfpq = faiss.IndexIVFPQ(quantizer, self.dim, self.nlist, self.m, self.nbits)

        try:
            ivfpq.cp.niter = self.kmeans_niter
            ivfpq.cp.max_points_per_centroid = self.max_points_per_centroid
        except Exception:
            pass

        try:
            pq = ivfpq.pq
            if hasattr(pq, "cp"):
                pq.cp.niter = max(10, self.kmeans_niter)
                pq.cp.max_points_per_centroid = max(256, self.max_points_per_centroid)
        except Exception:
            pass

        try:
            ivfpq.use_precomputed_table = 1
        except Exception:
            pass

        if self.use_opq:
            opq = faiss.OPQMatrix(self.dim, self.m)
            try:
                opq.niter = self.opq_niter
            except Exception:
                pass
            self.index = faiss.IndexPreTransform(opq, ivfpq)
        else:
            self.index = ivfpq

        self._ivf = faiss.extract_index_ivf(self.index)
        if self._ivf is not None:
            self._ivf.nprobe = self.nprobe

        self._built = True

    def add(self, xb: np.ndarray) -> None:
        if xb is None or xb.size == 0:
            return

        xb = np.ascontiguousarray(xb, dtype=np.float32)
        if xb.shape[1] != self.dim:
            raise ValueError(f"xb has dim {xb.shape[1]} but index dim is {self.dim}")

        if not self._built or self.index is None:
            self._create_index()

        if not self.index.is_trained:
            n = xb.shape[0]
            ts = min(self.train_size, n)
            if ts <= 0:
                raise ValueError("Cannot train on empty data")
            idx = self._rng.randint(0, n, size=ts, dtype=np.int64)
            train = np.ascontiguousarray(xb[idx], dtype=np.float32)
            self.index.train(train)

            self._ivf = faiss.extract_index_ivf(self.index)
            if self._ivf is not None:
                self._ivf.nprobe = self.nprobe

            try:
                ivf2 = faiss.extract_index_ivf(self.index)
                if ivf2 is not None:
                    ivf2.nprobe = self.nprobe
            except Exception:
                pass

        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None or not self._built:
            raise RuntimeError("Index not initialized")

        if not self.index.is_trained or self.index.ntotal == 0:
            nq = 0 if xq is None else int(xq.shape[0])
            D = np.full((nq, k), np.inf, dtype=np.float32)
            I = np.full((nq, k), -1, dtype=np.int64)
            return D, I

        k = int(k)
        if k <= 0:
            raise ValueError("k must be >= 1")

        xq = np.ascontiguousarray(xq, dtype=np.float32)
        if xq.shape[1] != self.dim:
            raise ValueError(f"xq has dim {xq.shape[1]} but index dim is {self.dim}")

        ivf = self._ivf if self._ivf is not None else faiss.extract_index_ivf(self.index)
        if ivf is not None:
            ivf.nprobe = self.nprobe

        D, I = self.index.search(xq, k)
        if D.dtype != np.float32:
            D = D.astype(np.float32, copy=False)
        if I.dtype != np.int64:
            I = I.astype(np.int64, copy=False)
        return D, I