import numpy as np
from typing import Tuple
import faiss
import multiprocessing


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = dim

        # Set number of threads for Faiss
        n_threads = kwargs.get("n_threads", None)
        if n_threads is None:
            try:
                n_threads = min(8, multiprocessing.cpu_count())
            except (ImportError, NotImplementedError):
                n_threads = 8
        try:
            faiss.omp_set_num_threads(int(n_threads))
        except Exception:
            pass

        # IVF parameters tuned for SIFT1M
        self.nlist = int(kwargs.get("nlist", 4096))
        self.nprobe = int(kwargs.get("nprobe", 256))

        # Build IVF-Flat index (coarse quantizer + exact search in selected lists)
        quantizer = faiss.IndexFlatL2(dim)
        self.quantizer = quantizer
        self.index = faiss.IndexIVFFlat(quantizer, dim, self.nlist, faiss.METRIC_L2)
        self.index.nprobe = self.nprobe

    def add(self, xb: np.ndarray) -> None:
        if xb is None:
            return
        xb = np.asarray(xb, dtype="float32")
        if xb.size == 0:
            return
        if not xb.flags["C_CONTIGUOUS"]:
            xb = np.ascontiguousarray(xb)

        if not self.index.is_trained:
            # Use up to 100k vectors for training the coarse quantizer
            n_train = min(100000, xb.shape[0])
            train_x = xb[:n_train]
            self.index.train(train_x)

        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        xq = np.asarray(xq, dtype="float32")
        if not xq.flags["C_CONTIGUOUS"]:
            xq = np.ascontiguousarray(xq)

        nq = xq.shape[0]

        if self.index is None or self.index.ntotal == 0 or k <= 0:
            D = np.full((nq, k), np.inf, dtype="float32")
            I = np.full((nq, k), -1, dtype="int64")
            return D, I

        k_eff = min(k, self.index.ntotal)
        D, I = self.index.search(xq, k_eff)

        if k_eff < k:
            D_pad = np.full((nq, k), np.inf, dtype="float32")
            I_pad = np.full((nq, k), -1, dtype="int64")
            D_pad[:, :k_eff] = D
            I_pad[:, :k_eff] = I
            D, I = D_pad, I_pad

        return D, I