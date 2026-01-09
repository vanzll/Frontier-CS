import os
import numpy as np
from typing import Tuple

try:
    import faiss  # type: ignore
except Exception as e:
    faiss = None


class YourIndexClass:
    def __init__(
        self,
        dim: int,
        **kwargs,
    ):
        self.dim = int(dim)

        self.nlist = int(kwargs.get("nlist", 4096))
        self.nprobe = int(kwargs.get("nprobe", 32))
        self.hnsw_m = int(kwargs.get("hnsw_m", 32))
        self.quantizer_ef_search = int(kwargs.get("quantizer_ef_search", 64))
        self.quantizer_ef_construction = int(kwargs.get("quantizer_ef_construction", 200))

        self.opq_m = int(kwargs.get("opq_m", 32))
        self.pq_m = int(kwargs.get("pq_m", 32))
        self.pq_bits = int(kwargs.get("pq_bits", 4))  # try fast-scan 4 bits
        self.train_size = int(kwargs.get("train_size", 100000))

        self.nthreads = int(kwargs.get("nthreads", 8))

        if self.nthreads > 0:
            os.environ.setdefault("OMP_NUM_THREADS", str(self.nthreads))
            os.environ.setdefault("OPENBLAS_NUM_THREADS", str(self.nthreads))
            os.environ.setdefault("MKL_NUM_THREADS", str(self.nthreads))
            os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(self.nthreads))
            os.environ.setdefault("NUMEXPR_NUM_THREADS", str(self.nthreads))

        if faiss is None:
            raise RuntimeError("faiss is required but not available")

        faiss.omp_set_num_threads(max(1, self.nthreads))

        self.index = None
        self._ivf = None
        self._is_trained = False

        self._build_index()

    def _build_index(self) -> None:
        d = self.dim

        # Prefer fast-scan IVF-PQ with OPQ and HNSW quantizer
        desc_candidates = []
        if self.pq_bits == 4:
            desc_candidates.append(
                f"OPQ{self.opq_m}_{d},IVF{self.nlist}_HNSW{self.hnsw_m},PQ{self.pq_m}x4fs"
            )
        desc_candidates.append(
            f"OPQ{self.opq_m}_{d},IVF{self.nlist}_HNSW{self.hnsw_m},PQ{self.pq_m}x8"
        )
        desc_candidates.append(
            f"IVF{self.nlist}_HNSW{self.hnsw_m},PQ{self.pq_m}x8"
        )
        desc_candidates.append(
            f"IVF{self.nlist},PQ{self.pq_m}x8"
        )

        last_err = None
        for desc in desc_candidates:
            try:
                idx = faiss.index_factory(d, desc, faiss.METRIC_L2)
                self.index = idx
                break
            except Exception as e:
                last_err = e
                self.index = None

        if self.index is None:
            raise RuntimeError(f"Failed to create FAISS index, last error: {last_err}")

        self._ivf = faiss.extract_index_ivf(self.index)
        if self._ivf is not None:
            self._ivf.nprobe = int(self.nprobe)

            q = self._ivf.quantizer
            if hasattr(q, "hnsw") and q is not None:
                try:
                    q.hnsw.efSearch = int(self.quantizer_ef_search)
                    q.hnsw.efConstruction = int(self.quantizer_ef_construction)
                except Exception:
                    pass

            try:
                self._ivf.cp.niter = 20
            except Exception:
                pass

    def add(self, xb: np.ndarray) -> None:
        if xb is None or xb.size == 0:
            return
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32, copy=False)
        xb = np.ascontiguousarray(xb)

        if not self._is_trained:
            n = xb.shape[0]
            ts = min(max(10000, self.train_size), n)
            train_x = xb[:ts]
            self.index.train(train_x)
            self._is_trained = True
            if self._ivf is not None:
                self._ivf.nprobe = int(self.nprobe)

        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32, copy=False)
        xq = np.ascontiguousarray(xq)

        if self._ivf is not None:
            self._ivf.nprobe = int(self.nprobe)

        D, I = self.index.search(xq, int(k))
        if D.dtype != np.float32:
            D = D.astype(np.float32, copy=False)
        if I.dtype != np.int64:
            I = I.astype(np.int64, copy=False)
        return D, I