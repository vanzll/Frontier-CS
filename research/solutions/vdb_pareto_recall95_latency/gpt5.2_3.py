import os
import numpy as np
from typing import Tuple

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = int(dim)

        self.nlist = int(kwargs.get("nlist", 4096))
        self.nprobe = int(kwargs.get("nprobe", kwargs.get("ef_search", 64)))
        self.train_size = int(kwargs.get("train_size", 200000))
        self.seed = int(kwargs.get("seed", 12345))

        n_threads = kwargs.get("n_threads", None)
        if n_threads is None:
            n_threads = os.cpu_count() or 1
        self.n_threads = int(n_threads)

        self._use_faiss = faiss is not None
        self._xb = None  # fallback

        if self._use_faiss:
            try:
                faiss.omp_set_num_threads(self.n_threads)
            except Exception:
                pass

            self._build_faiss_index(self.nlist)
        else:
            self._xb = None

    def _build_faiss_index(self, nlist: int) -> None:
        nlist = max(1, int(nlist))
        quantizer = faiss.IndexFlatL2(self.dim)
        index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_L2)
        try:
            index.cp.niter = 20
            index.cp.max_points_per_centroid = 256
            index.cp.seed = self.seed
        except Exception:
            pass
        index.nprobe = max(1, int(self.nprobe))
        self.index = index

    def add(self, xb: np.ndarray) -> None:
        xb = np.asarray(xb, dtype=np.float32, order="C")
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError(f"xb must have shape (N, {self.dim})")

        if not self._use_faiss:
            if self._xb is None:
                self._xb = xb.copy()
            else:
                self._xb = np.vstack((self._xb, xb))
            return

        if not self.index.is_trained:
            n = xb.shape[0]
            if n < self.nlist:
                new_nlist = max(1, min(self.nlist, int(max(1, np.sqrt(n)))))
                if new_nlist != self.index.nlist:
                    self._build_faiss_index(new_nlist)

            n_train = min(n, self.train_size)
            rng = np.random.default_rng(self.seed)
            if n_train <= 0:
                raise ValueError("Cannot train index on empty input")
            idx = rng.integers(0, n, size=n_train, endpoint=False)
            x_train = xb[idx].copy(order="C")
            self.index.train(x_train)

        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        k = int(k)
        if k <= 0:
            raise ValueError("k must be >= 1")

        xq = np.asarray(xq, dtype=np.float32, order="C")
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError(f"xq must have shape (nq, {self.dim})")

        if not self._use_faiss:
            if self._xb is None or self._xb.shape[0] == 0:
                nq = xq.shape[0]
                D = np.full((nq, k), np.inf, dtype=np.float32)
                I = np.full((nq, k), -1, dtype=np.int64)
                return D, I
            return self._numpy_search(xq, k)

        try:
            self.index.nprobe = max(1, int(self.nprobe))
        except Exception:
            pass

        D, I = self.index.search(xq, k)
        if D.dtype != np.float32:
            D = D.astype(np.float32, copy=False)
        if I.dtype != np.int64:
            I = I.astype(np.int64, copy=False)
        return D, I

    def _numpy_search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        xb = self._xb
        nq = xq.shape[0]
        nb = xb.shape[0]

        xq_norm = (xq * xq).sum(axis=1, keepdims=True)  # (nq, 1)
        xb_norm = (xb * xb).sum(axis=1)  # (nb,)

        block = 16384
        best_D = np.full((nq, k), np.inf, dtype=np.float32)
        best_I = np.full((nq, k), -1, dtype=np.int64)

        for start in range(0, nb, block):
            end = min(nb, start + block)
            xb_blk = xb[start:end]
            dots = xq @ xb_blk.T  # (nq, b)
            d2 = xq_norm + xb_norm[start:end][None, :] - 2.0 * dots
            d2 = np.maximum(d2, 0.0).astype(np.float32, copy=False)

            # merge current block candidates with existing best
            b = end - start
            cand_D = np.concatenate([best_D, d2], axis=1)  # (nq, k+b)
            cand_I = np.concatenate([best_I, (np.arange(start, end, dtype=np.int64)[None, :]).repeat(nq, axis=0)], axis=1)

            sel = np.argpartition(cand_D, kth=k - 1, axis=1)[:, :k]
            row = np.arange(nq)[:, None]
            best_D = cand_D[row, sel]
            best_I = cand_I[row, sel]

            order = np.argsort(best_D, axis=1)
            best_D = best_D[row, order]
            best_I = best_I[row, order]

        return best_D, best_I