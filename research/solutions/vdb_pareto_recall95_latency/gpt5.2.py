import os
import numpy as np
from typing import Tuple

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


class YourIndexClass:
    def __init__(
        self,
        dim: int,
        **kwargs,
    ):
        self.dim = int(dim)

        self.nlist = int(kwargs.get("nlist", 4096))
        self.nprobe = int(kwargs.get("nprobe", kwargs.get("ef_search", 24)))
        self.training_size = int(kwargs.get("training_size", 100000))
        self.seed = int(kwargs.get("seed", 123))
        self.n_threads = int(kwargs.get("n_threads", 0))

        self._ntotal = 0
        self._buffer = []
        self._buffer_rows = 0

        if faiss is None:
            self._faiss_index = None
            self._xb = None
            return

        if self.n_threads <= 0:
            self.n_threads = min(8, (os.cpu_count() or 1))
        try:
            faiss.omp_set_num_threads(self.n_threads)
        except Exception:
            pass

        self._quantizer = faiss.IndexFlatL2(self.dim)
        self._faiss_index = faiss.IndexIVFFlat(self._quantizer, self.dim, self.nlist, faiss.METRIC_L2)
        self._faiss_index.nprobe = self.nprobe

    def _as_float32_contig(self, x: np.ndarray) -> np.ndarray:
        if x is None:
            return x
        if x.dtype != np.float32 or not x.flags["C_CONTIGUOUS"]:
            return np.ascontiguousarray(x, dtype=np.float32)
        return x

    def _train_if_needed(self, xb: np.ndarray) -> None:
        if faiss is None or self._faiss_index is None:
            return
        if self._faiss_index.is_trained:
            return

        n = xb.shape[0]
        if n <= 0:
            return

        ntrain = self.training_size
        min_train = max(self.nlist * 20, self.nlist + 1)
        if ntrain < min_train:
            ntrain = min_train
        if ntrain > n:
            ntrain = n

        rs = np.random.RandomState(self.seed)
        if ntrain == n:
            xtrain = xb
        else:
            idx = rs.randint(0, n, size=ntrain, dtype=np.int64)
            xtrain = xb[idx]

        xtrain = self._as_float32_contig(xtrain)
        self._faiss_index.train(xtrain)

    def add(self, xb: np.ndarray) -> None:
        xb = self._as_float32_contig(xb)
        if xb is None or xb.size == 0:
            return
        if xb.shape[1] != self.dim:
            raise ValueError(f"Expected xb with dim={self.dim}, got {xb.shape[1]}")

        if faiss is None or self._faiss_index is None:
            if self._xb is None:
                self._xb = xb.copy()
            else:
                self._xb = np.vstack([self._xb, xb])
            self._ntotal = int(self._xb.shape[0])
            return

        if not self._faiss_index.is_trained:
            self._train_if_needed(xb)

        if not self._faiss_index.is_trained:
            self._buffer.append(xb.copy())
            self._buffer_rows += xb.shape[0]
            if self._buffer_rows >= max(self.training_size, self.nlist * 20):
                allbuf = np.vstack(self._buffer)
                self._buffer = []
                self._buffer_rows = 0
                self._train_if_needed(allbuf)
                if self._faiss_index.is_trained:
                    self._faiss_index.add(allbuf)
                    self._ntotal += int(allbuf.shape[0])
            return

        if self._buffer:
            allbuf = np.vstack(self._buffer)
            self._buffer = []
            self._buffer_rows = 0
            self._faiss_index.add(allbuf)
            self._ntotal += int(allbuf.shape[0])

        self._faiss_index.add(xb)
        self._ntotal += int(xb.shape[0])

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        k = int(k)
        if k <= 0:
            raise ValueError("k must be >= 1")

        xq = self._as_float32_contig(xq)
        nq = 0 if xq is None else int(xq.shape[0])

        if nq == 0:
            return (np.empty((0, k), dtype=np.float32), np.empty((0, k), dtype=np.int64))

        if xq.shape[1] != self.dim:
            raise ValueError(f"Expected xq with dim={self.dim}, got {xq.shape[1]}")

        if faiss is None or self._faiss_index is None:
            if self._xb is None or self._xb.shape[0] == 0:
                D = np.full((nq, k), np.inf, dtype=np.float32)
                I = np.full((nq, k), -1, dtype=np.int64)
                return D, I
            xb = self._xb
            xq2 = (xq * xq).sum(axis=1, keepdims=True)
            xb2 = (xb * xb).sum(axis=1)[None, :]
            dist = xq2 + xb2 - 2.0 * (xq @ xb.T)
            idx = np.argpartition(dist, kth=min(k - 1, dist.shape[1] - 1), axis=1)[:, :k]
            row = np.arange(nq)[:, None]
            dsel = dist[row, idx]
            ord_ = np.argsort(dsel, axis=1)
            I = idx[row, ord_].astype(np.int64, copy=False)
            D = dsel[row, ord_].astype(np.float32, copy=False)
            return D, I

        if self._buffer:
            allbuf = np.vstack(self._buffer)
            self._buffer = []
            self._buffer_rows = 0
            if not self._faiss_index.is_trained:
                self._train_if_needed(allbuf)
            if self._faiss_index.is_trained and allbuf.shape[0] > 0:
                self._faiss_index.add(self._as_float32_contig(allbuf))
                self._ntotal += int(allbuf.shape[0])

        if self._ntotal <= 0:
            D = np.full((nq, k), np.inf, dtype=np.float32)
            I = np.full((nq, k), -1, dtype=np.int64)
            return D, I

        D, I = self._faiss_index.search(xq, k)
        if D.dtype != np.float32:
            D = D.astype(np.float32, copy=False)
        if I.dtype != np.int64:
            I = I.astype(np.int64, copy=False)
        return D, I