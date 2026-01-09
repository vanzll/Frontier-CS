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

        self.threads = int(kwargs.get("threads", min(8, (os.cpu_count() or 1))))
        self.nlist = int(kwargs.get("nlist", 4096))
        self.m = int(kwargs.get("m", 32))
        self.nbits = int(kwargs.get("nbits", 8))
        self.nprobe = int(kwargs.get("nprobe", 32))
        self.train_size = int(kwargs.get("train_size", 200_000))
        self.use_opq = bool(kwargs.get("use_opq", True))
        self.opq_m = int(kwargs.get("opq_m", self.m))
        self.k_factor = int(kwargs.get("k_factor", 64))

        self._index = None
        self._ivf = None
        self._pending = []
        self._pending_total = 0

        if faiss is not None:
            try:
                faiss.omp_set_num_threads(self.threads)
            except Exception:
                pass
            try:
                faiss.cvar.rand_seed = 12345
            except Exception:
                pass

    def _as_float32_contig(self, x: np.ndarray) -> np.ndarray:
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)
        return x

    def _min_train_points(self) -> int:
        return max(self.nlist, 50_000)

    def _build_faiss_index(self) -> None:
        if faiss is None:
            return
        quantizer = faiss.IndexFlatL2(self.dim)
        ivf = faiss.IndexIVFPQ(quantizer, self.dim, self.nlist, self.m, self.nbits)
        try:
            ivf.nprobe = self.nprobe
        except Exception:
            pass
        try:
            ivf.use_precomputed_table = 1
        except Exception:
            pass
        try:
            ivf.scan_table_threshold = 0
        except Exception:
            pass

        base = ivf
        if self.use_opq:
            opq = faiss.OPQMatrix(self.dim, self.opq_m)
            try:
                opq.niter = 25
            except Exception:
                pass
            base = faiss.IndexPreTransform(opq, ivf)

        idx = faiss.IndexRefineFlat(base)
        try:
            idx.k_factor = self.k_factor
        except Exception:
            pass

        self._index = idx
        try:
            self._ivf = faiss.extract_index_ivf(self._index)
        except Exception:
            self._ivf = None
        if self._ivf is not None:
            try:
                self._ivf.nprobe = self.nprobe
            except Exception:
                pass

    def _train_on_sample(self, xb: np.ndarray) -> None:
        if self._index is None or faiss is None:
            return

        n = xb.shape[0]
        ntrain = min(self.train_size, n)
        if ntrain < max(self.nlist, 1000):
            ntrain = n

        if ntrain == n:
            xt = xb
        else:
            rng = np.random.default_rng(12345)
            idx = rng.choice(n, size=ntrain, replace=False)
            xt = xb[idx]

        xt = self._as_float32_contig(xt)
        self._index.train(xt)

        if self._ivf is not None:
            try:
                self._ivf.nprobe = self.nprobe
            except Exception:
                pass

    def _brute_search(self, xb: np.ndarray, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        xb = self._as_float32_contig(xb)
        xq = self._as_float32_contig(xq)
        nq = xq.shape[0]
        nb = xb.shape[0]
        if nb == 0:
            D = np.full((nq, k), np.inf, dtype=np.float32)
            I = np.full((nq, k), -1, dtype=np.int64)
            return D, I

        xq_norm = (xq * xq).sum(axis=1, keepdims=True).astype(np.float32, copy=False)
        xb_norm = (xb * xb).sum(axis=1, keepdims=True).T.astype(np.float32, copy=False)
        sims = xq @ xb.T
        dists = xq_norm + xb_norm - 2.0 * sims
        if k >= nb:
            I = np.argsort(dists, axis=1)[:, :k].astype(np.int64, copy=False)
        else:
            I = np.argpartition(dists, kth=k - 1, axis=1)[:, :k].astype(np.int64, copy=False)
            row = np.arange(nq)[:, None]
            sub = dists[row, I]
            ord2 = np.argsort(sub, axis=1)
            I = I[row, ord2]
        D = dists[np.arange(nq)[:, None], I].astype(np.float32, copy=False)
        return D, I

    def add(self, xb: np.ndarray) -> None:
        if xb is None:
            return
        xb = self._as_float32_contig(xb)
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError(f"xb must have shape (N, {self.dim})")

        if faiss is None:
            self._pending.append(xb)
            self._pending_total += xb.shape[0]
            return

        if self._index is None:
            n = xb.shape[0]
            if n >= self._min_train_points():
                self._build_faiss_index()
                self._train_on_sample(xb)
                self._index.add(xb)
                return

            self._pending.append(xb)
            self._pending_total += n
            if self._pending_total < self._min_train_points():
                return

            all_xb = np.vstack(self._pending)
            self._pending.clear()
            self._pending_total = 0
            self._build_faiss_index()
            self._train_on_sample(all_xb)
            self._index.add(all_xb)
            return

        self._index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        k = int(k)
        if k <= 0:
            nq = 0 if xq is None else int(xq.shape[0])
            return np.empty((nq, 0), dtype=np.float32), np.empty((nq, 0), dtype=np.int64)

        if xq is None:
            return np.empty((0, k), dtype=np.float32), np.empty((0, k), dtype=np.int64)

        xq = self._as_float32_contig(xq)
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError(f"xq must have shape (nq, {self.dim})")

        if faiss is None:
            if self._pending_total == 0:
                nq = xq.shape[0]
                return np.full((nq, k), np.inf, dtype=np.float32), np.full((nq, k), -1, dtype=np.int64)
            xb = np.vstack(self._pending) if len(self._pending) > 1 else self._pending[0]
            return self._brute_search(xb, xq, k)

        if self._index is None:
            if self._pending_total == 0:
                nq = xq.shape[0]
                return np.full((nq, k), np.inf, dtype=np.float32), np.full((nq, k), -1, dtype=np.int64)
            xb = np.vstack(self._pending) if len(self._pending) > 1 else self._pending[0]
            return self._brute_search(xb, xq, k)

        D, I = self._index.search(xq, k)
        if D.dtype != np.float32:
            D = D.astype(np.float32, copy=False)
        if I.dtype != np.int64:
            I = I.astype(np.int64, copy=False)
        return D, I