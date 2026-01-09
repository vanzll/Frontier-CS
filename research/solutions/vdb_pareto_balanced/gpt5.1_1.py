import numpy as np
import faiss
from typing import Tuple


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = int(dim)

        # IVF parameters
        self.nlist = int(kwargs.get("nlist", 4096))
        self.nprobe = int(kwargs.get("nprobe", 256))
        if self.nprobe <= 0:
            self.nprobe = 64
        self.nprobe = min(self.nprobe, self.nlist)

        # Training parameters
        self.max_training_points = int(kwargs.get("max_training_points", 200000))
        seed = int(kwargs.get("seed", 123))
        self.rng = np.random.RandomState(seed)

        # Threading
        num_threads = kwargs.get("num_threads", 0)
        try:
            num_threads = int(num_threads)
        except Exception:
            num_threads = 0
        if num_threads <= 0:
            try:
                if hasattr(faiss, "omp_get_max_threads"):
                    num_threads = faiss.omp_get_max_threads()
            except Exception:
                num_threads = 0
        if num_threads > 0 and hasattr(faiss, "omp_set_num_threads"):
            faiss.omp_set_num_threads(num_threads)

        # Build IVF-Flat index with L2 metric
        self.quantizer = faiss.IndexFlatL2(self.dim)
        self.index = faiss.IndexIVFFlat(self.quantizer, self.dim, self.nlist, faiss.METRIC_L2)
        self.index.nprobe = self.nprobe

    def add(self, xb: np.ndarray) -> None:
        if xb is None:
            return

        xb = np.asarray(xb, dtype=np.float32)
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError(f"Input xb must have shape (N, {self.dim}), got {xb.shape}")
        xb = np.ascontiguousarray(xb, dtype=np.float32)

        # Train IVF coarse quantizer on a subset if not yet trained
        if not self.index.is_trained:
            n_train = min(self.max_training_points, xb.shape[0])
            if xb.shape[0] > n_train:
                indices = self.rng.choice(xb.shape[0], n_train, replace=False)
                train_x = xb[indices]
            else:
                train_x = xb
            self.index.train(train_x)

        # Add all provided vectors
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        xq = np.asarray(xq, dtype=np.float32)
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError(f"Input xq must have shape (nq, {self.dim}), got {xq.shape}")
        xq = np.ascontiguousarray(xq, dtype=np.float32)

        if self.index.ntotal == 0:
            nq = xq.shape[0]
            D = np.full((nq, k), np.inf, dtype=np.float32)
            I = np.full((nq, k), -1, dtype=np.int64)
            return D, I

        # Ensure nprobe is set (in case user changed it)
        self.index.nprobe = min(self.nprobe, self.nlist)

        D, I = self.index.search(xq, k)
        # Ensure correct dtypes
        D = np.asarray(D, dtype=np.float32)
        I = np.asarray(I, dtype=np.int64)
        return D, I