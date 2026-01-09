import os
from typing import Tuple
import numpy as np

try:
    import faiss
except Exception:
    faiss = None


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = int(dim)
        self.nlist = int(kwargs.get("nlist", 16384))
        self.nprobe = int(kwargs.get("nprobe", 16))
        self.train_samples = int(kwargs.get("train_samples", 200000))
        self.metric = kwargs.get("metric", "l2")
        self.use_opq = bool(kwargs.get("use_opq", False))
        self.pq_m = int(kwargs.get("pq_m", 16))
        self.pq_nbits = int(kwargs.get("pq_nbits", 8))
        self.index_type = kwargs.get("index_type", "IVFFlat")  # "IVFFlat" or "IVFPQ"

        self._index = None
        self._trained = False

        if faiss is not None:
            # Set FAISS to use available cores
            try:
                nthread = int(os.environ.get("FAISS_NUM_THREADS", os.cpu_count() or 1))
                faiss.omp_set_num_threads(nthread)
            except Exception:
                pass

    def _build_index(self):
        if faiss is None:
            raise RuntimeError("faiss is required for this index")

        metric_type = faiss.METRIC_L2 if self.metric.lower() == "l2" else faiss.METRIC_INNER_PRODUCT

        quantizer = faiss.IndexFlatL2(self.dim) if metric_type == faiss.METRIC_L2 else faiss.IndexFlatIP(self.dim)

        if self.index_type.upper() == "IVFPQ":
            base_index = faiss.IndexIVFPQ(quantizer, self.dim, self.nlist, self.pq_m, self.pq_nbits, metric_type)
            # Optional fast-scan settings, guard for version differences
            try:
                base_index.use_precomputed_table = 1
            except Exception:
                pass
        else:
            base_index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, metric_type)

        if self.use_opq:
            opq = faiss.OPQMatrix(self.dim, self.pq_m)
            self._index = faiss.IndexPreTransform(opq, base_index)
            self._ivf = base_index  # keep handle to IVF for parameters like nprobe
        else:
            self._index = base_index
            self._ivf = base_index

        try:
            self._ivf.nprobe = self.nprobe
        except Exception:
            pass

    def add(self, xb: np.ndarray) -> None:
        if not isinstance(xb, np.ndarray):
            xb = np.asarray(xb, dtype=np.float32)
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32, copy=False)
        xb = np.ascontiguousarray(xb)

        if self._index is None:
            self._build_index()

        if not self._trained:
            # Sample training data from xb
            n_train = min(self.train_samples, xb.shape[0])
            if n_train < xb.shape[0]:
                rng = np.random.RandomState(123)
                idx = rng.choice(xb.shape[0], size=n_train, replace=False)
                xtrain = xb[idx]
            else:
                xtrain = xb

            self._index.train(xtrain)
            self._trained = True

        self._index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if not isinstance(xq, np.ndarray):
            xq = np.asarray(xq, dtype=np.float32)
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32, copy=False)
        xq = np.ascontiguousarray(xq)

        try:
            self._ivf.nprobe = self.nprobe
        except Exception:
            pass

        D, I = self._index.search(xq, int(k))
        if D.dtype != np.float32:
            D = D.astype(np.float32, copy=False)
        if I.dtype != np.int64:
            I = I.astype(np.int64, copy=False)
        return D, I