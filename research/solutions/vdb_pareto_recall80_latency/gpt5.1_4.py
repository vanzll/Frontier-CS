import numpy as np
import faiss
from typing import Tuple


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize an IVF-Flat index using FAISS, tuned for SIFT1M-like data.
        """
        self.dim = dim

        # IVF parameters (can be overridden via kwargs)
        self.nlist = int(kwargs.get("nlist", 100))       # number of coarse centroids
        self.nprobe = int(kwargs.get("nprobe", 8))       # number of lists to probe at search time
        self.train_size = int(kwargs.get("train_size", 100000))

        self.metric = faiss.METRIC_L2
        self._index = None

        # Configure FAISS threading (default: use all available CPU cores)
        try:
            n_threads = kwargs.get("faiss_n_threads", None)
            if n_threads is None:
                try:
                    import multiprocessing

                    n_threads = multiprocessing.cpu_count()
                except Exception:
                    n_threads = 1
            faiss.omp_set_num_threads(int(n_threads))
        except Exception:
            # If anything goes wrong, FAISS will use its default threading
            pass

    def _build_index(self, xb: np.ndarray) -> None:
        """
        Internal helper to construct and train the IVF index on first add().
        """
        # Coarse quantizer
        quantizer = faiss.IndexFlatL2(self.dim)

        # IVF-Flat index (stores full vectors in inverted lists)
        index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, self.metric)

        # Select training data
        n_train = min(self.train_size, xb.shape[0])
        if xb.shape[0] > n_train:
            rs = np.random.RandomState(1234)
            train_idx = rs.choice(xb.shape[0], n_train, replace=False)
            train_x = xb[train_idx]
        else:
            train_x = xb

        index.train(train_x)
        index.nprobe = self.nprobe
        self._index = index

    def add(self, xb: np.ndarray) -> None:
        """
        Add base vectors to the index. Can be called multiple times.
        """
        if xb is None or xb.size == 0:
            return

        xb = np.ascontiguousarray(xb, dtype="float32")

        if self._index is None:
            self._build_index(xb)

        self._index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search k nearest neighbors for query vectors.
        """
        xq = np.ascontiguousarray(xq, dtype="float32")
        nq = xq.shape[0]

        if self._index is None or self._index.ntotal == 0:
            # No data in index: return empty results
            D = np.full((nq, k), np.inf, dtype="float32")
            I = np.full((nq, k), -1, dtype="int64")
            return D, I

        # Ensure correct nprobe
        self._index.nprobe = self.nprobe

        D, I = self._index.search(xq, k)

        if not isinstance(D, np.ndarray):
            D = np.array(D)
        if not isinstance(I, np.ndarray):
            I = np.array(I)

        if D.dtype != np.float32:
            D = D.astype("float32")
        if I.dtype != np.int64:
            I = I.astype("int64")

        return D, I