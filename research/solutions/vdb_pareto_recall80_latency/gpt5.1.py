import numpy as np
from typing import Tuple
import faiss


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = int(dim)

        self.nlist = int(kwargs.get("nlist", 4096))
        self.m = int(kwargs.get("m", 16))
        self.nprobe = int(kwargs.get("nprobe", 64))
        self.train_size = int(kwargs.get("train_size", 100000))

        num_threads = kwargs.get("num_threads", None)
        if num_threads is None:
            try:
                max_th = faiss.omp_get_max_threads()
            except AttributeError:
                import multiprocessing
                max_th = multiprocessing.cpu_count()
            num_threads = min(8, max_th)
        self.num_threads = int(num_threads)

        try:
            faiss.omp_set_num_threads(self.num_threads)
        except AttributeError:
            pass

        index_str = f"IVF{self.nlist},PQ{self.m}"
        self.index = faiss.index_factory(self.dim, index_str, faiss.METRIC_L2)
        self._set_nprobe(self.nprobe)

    def _set_nprobe(self, nprobe: int) -> None:
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe
        else:
            try:
                ivf = faiss.extract_index_ivf(self.index)
                ivf.nprobe = nprobe
            except Exception:
                pass

    def _train(self, xb: np.ndarray) -> None:
        if xb.shape[0] == 0:
            return
        n_train = min(self.train_size, xb.shape[0])
        if n_train <= 0:
            return
        train_x = xb[:n_train].copy()
        self.index.train(train_x)

    def add(self, xb: np.ndarray) -> None:
        if xb is None:
            return
        xb = np.asarray(xb, dtype=np.float32)
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            xb = xb.reshape(-1, self.dim)
        xb = np.ascontiguousarray(xb, dtype=np.float32)

        if not self.index.is_trained:
            self._train(xb)

        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        xq = np.asarray(xq, dtype=np.float32)
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            xq = xq.reshape(-1, self.dim)
        xq = np.ascontiguousarray(xq, dtype=np.float32)
        nq = xq.shape[0]

        if not self.index.is_trained or self.index.ntotal == 0:
            distances = np.full((nq, k), np.inf, dtype=np.float32)
            indices = np.full((nq, k), -1, dtype=np.int64)
            return distances, indices

        k_eff = min(k, self.index.ntotal)
        if k_eff <= 0:
            distances = np.full((nq, k), np.inf, dtype=np.float32)
            indices = np.full((nq, k), -1, dtype=np.int64)
            return distances, indices

        D, I = self.index.search(xq, k_eff)

        if k_eff == k:
            if D.dtype != np.float32:
                D = D.astype(np.float32)
            if I.dtype != np.int64:
                I = I.astype(np.int64)
            return D, I

        distances = np.full((nq, k), np.inf, dtype=np.float32)
        indices = np.full((nq, k), -1, dtype=np.int64)
        distances[:, :k_eff] = D
        indices[:, :k_eff] = I
        return distances, indices