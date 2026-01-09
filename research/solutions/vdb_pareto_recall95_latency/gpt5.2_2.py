import os
from typing import Tuple, Optional

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    faiss = None  # type: ignore


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = int(dim)

        self.threads = int(kwargs.get("threads", os.cpu_count() or 1))
        if faiss is not None:
            try:
                faiss.omp_set_num_threads(self.threads)
            except Exception:
                pass

        # Index configuration
        self.index_type = str(kwargs.get("index_type", "ivfpq")).lower()  # "ivfpq" or "ivfflat" or "flat"
        self.nlist = int(kwargs.get("nlist", 4096))
        self.nprobe = int(kwargs.get("nprobe", 64))
        self.train_size = int(kwargs.get("train_size", 200_000))

        # IVFPQ parameters
        self.m = int(kwargs.get("m", 32))
        self.nbits = int(kwargs.get("nbits", 8))

        # Exact rerank count (0 disables rerank, otherwise reranks exact among top-R approximate)
        self.rerank = int(kwargs.get("rerank", 128))

        # Internal storage for rerank
        self._xb_chunks = []
        self.xb: Optional[np.ndarray] = None
        self.xb_norms: Optional[np.ndarray] = None

        self._pending_train = []
        self._index = None
        self._trained = False

        if faiss is None:  # pragma: no cover
            self.index_type = "flat"

        self._init_faiss_index()

    def _init_faiss_index(self):
        if faiss is None:
            self._index = None
            return

        if self.index_type == "flat":
            self._index = faiss.IndexFlatL2(self.dim)
            self._trained = True
            return

        if self.index_type == "ivfflat":
            quantizer = faiss.IndexFlatL2(self.dim)
            index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_L2)
            index.nprobe = self.nprobe
            try:
                index.parallel_mode = 3
            except Exception:
                pass
            try:
                index.cp.niter = int(getattr(index.cp, "niter", 20))
            except Exception:
                pass
            self._index = index
            self._trained = False
            return

        # default ivfpq
        quantizer = faiss.IndexFlatL2(self.dim)
        index = faiss.IndexIVFPQ(quantizer, self.dim, self.nlist, self.m, self.nbits)
        index.nprobe = self.nprobe
        try:
            index.parallel_mode = 3
        except Exception:
            pass
        try:
            index.use_precomputed_table = 1
        except Exception:
            pass
        try:
            index.cp.niter = int(getattr(index.cp, "niter", 20))
        except Exception:
            pass
        self._index = index
        self._trained = False

    def _ensure_xb_ready(self):
        if self.xb is not None:
            return
        if not self._xb_chunks:
            self.xb = np.empty((0, self.dim), dtype=np.float32)
            self.xb_norms = np.empty((0,), dtype=np.float32)
            return
        if len(self._xb_chunks) == 1:
            self.xb = self._xb_chunks[0]
        else:
            self.xb = np.ascontiguousarray(np.vstack(self._xb_chunks), dtype=np.float32)
            self._xb_chunks = [self.xb]
        self.xb_norms = np.sum(self.xb * self.xb, axis=1, dtype=np.float32)

    def _maybe_train(self, xb: np.ndarray):
        if faiss is None or self._trained:
            return
        if self._index is None:
            return

        # Accumulate for training if needed
        self._pending_train.append(xb)
        total = sum(a.shape[0] for a in self._pending_train)
        if total < max(self.train_size, self.nlist * 5):
            return

        train_mat = np.ascontiguousarray(np.vstack(self._pending_train), dtype=np.float32)
        self._pending_train.clear()

        ntrain = train_mat.shape[0]
        if ntrain <= 0:
            return

        # Subsample
        tsize = min(self.train_size, ntrain)
        if tsize < ntrain:
            rng = np.random.default_rng(12345)
            idx = rng.choice(ntrain, size=tsize, replace=False)
            train_mat = np.ascontiguousarray(train_mat[idx], dtype=np.float32)

        try:
            self._index.train(train_mat)
            self._trained = True
        except Exception:
            # Fallback: reduce nlist by rebuilding index if kmeans fails
            # (should not happen in SIFT1M setting)
            self.nlist = max(1, min(self.nlist, max(1, train_mat.shape[0] // 20)))
            self._init_faiss_index()
            if hasattr(self._index, "train"):
                self._index.train(train_mat)
                self._trained = True

    def add(self, xb: np.ndarray) -> None:
        xb = np.ascontiguousarray(xb, dtype=np.float32)
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError("xb must have shape (N, dim)")

        self._xb_chunks.append(xb)
        self.xb = None
        self.xb_norms = None

        if faiss is None:
            return

        if self._index is None:
            self._init_faiss_index()

        if not self._trained:
            self._maybe_train(xb)
            if not self._trained:
                return

            # If we trained after accumulating pending, we still need to add all stored chunks so far
            all_xb = np.ascontiguousarray(np.vstack(self._xb_chunks), dtype=np.float32)
            self._index.add(all_xb)
            return

        self._index.add(xb)

    def _rerank_exact(self, xq: np.ndarray, I0: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_xb_ready()
        xb = self.xb
        xb_norms = self.xb_norms
        if xb is None or xb_norms is None or xb.shape[0] == 0:
            nq = xq.shape[0]
            return np.full((nq, k), np.inf, dtype=np.float32), np.full((nq, k), -1, dtype=np.int64)

        nq = xq.shape[0]
        R = I0.shape[1]
        q_norms = np.sum(xq * xq, axis=1, dtype=np.float32)

        D_out = np.empty((nq, k), dtype=np.float32)
        I_out = np.empty((nq, k), dtype=np.int64)

        block = 256
        ar_b = np.arange(block, dtype=np.int64)

        for s in range(0, nq, block):
            e = min(nq, s + block)
            b = e - s
            qb = xq[s:e]
            cn = I0[s:e].astype(np.int64, copy=False)

            flat = cn.reshape(-1)
            vecs = xb[flat].reshape(b, R, self.dim)

            dot = np.einsum("bd,brd->br", qb, vecs, optimize=True)
            dist = xb_norms[flat].reshape(b, R) + q_norms[s:e, None] - 2.0 * dot
            dist = dist.astype(np.float32, copy=False)
            np.maximum(dist, 0.0, out=dist)

            if k == 1:
                best = np.argmin(dist, axis=1)
                I_out[s:e, 0] = cn[np.arange(b), best]
                D_out[s:e, 0] = dist[np.arange(b), best]
            else:
                kk = k if k <= R else R
                part = np.argpartition(dist, kk - 1, axis=1)[:, :kk]
                sel = dist[np.arange(b)[:, None], part]
                order = np.argsort(sel, axis=1)
                best_pos = part[np.arange(b)[:, None], order]
                I_out[s:e, :kk] = cn[np.arange(b)[:, None], best_pos]
                D_out[s:e, :kk] = dist[np.arange(b)[:, None], best_pos]
                if kk < k:
                    I_out[s:e, kk:k] = -1
                    D_out[s:e, kk:k] = np.inf

        return D_out, I_out

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        xq = np.ascontiguousarray(xq, dtype=np.float32)
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError("xq must have shape (nq, dim)")
        k = int(k)
        if k <= 0:
            raise ValueError("k must be >= 1")

        if faiss is None or self._index is None:
            self._ensure_xb_ready()
            xb = self.xb
            if xb is None or xb.shape[0] == 0:
                nq = xq.shape[0]
                return np.full((nq, k), np.inf, dtype=np.float32), np.full((nq, k), -1, dtype=np.int64)
            # Exact fallback in numpy (not optimized, but only used if faiss missing)
            q_norms = np.sum(xq * xq, axis=1, dtype=np.float32)
            b_norms = np.sum(xb * xb, axis=1, dtype=np.float32)
            D = q_norms[:, None] + b_norms[None, :] - 2.0 * (xq @ xb.T)
            D = np.maximum(D, 0.0).astype(np.float32, copy=False)
            if k == 1:
                I = np.argmin(D, axis=1).astype(np.int64)[:, None]
                D1 = D[np.arange(D.shape[0]), I[:, 0]].astype(np.float32)[:, None]
                return D1, I
            I = np.argpartition(D, k - 1, axis=1)[:, :k]
            sel = D[np.arange(D.shape[0])[:, None], I]
            order = np.argsort(sel, axis=1)
            I = I[np.arange(D.shape[0])[:, None], order].astype(np.int64, copy=False)
            Dk = D[np.arange(D.shape[0])[:, None], I].astype(np.float32, copy=False)
            return Dk, I

        if not self._trained:
            # Try to train on whatever we have stored
            self._ensure_xb_ready()
            if self.xb is not None and self.xb.shape[0] > 0 and hasattr(self._index, "train"):
                self._maybe_train(self.xb)
                if self._trained and self._index.ntotal == 0:
                    self._index.add(self.xb)

        if hasattr(self._index, "nprobe"):
            try:
                self._index.nprobe = self.nprobe
            except Exception:
                pass

        if self.rerank and self.index_type in ("ivfpq", "ivfflat"):
            R = max(self.rerank, k)
            D0, I0 = self._index.search(xq, R)
            D, I = self._rerank_exact(xq, I0, k)
            return D, I

        D, I = self._index.search(xq, k)
        return D, I.astype(np.int64, copy=False)