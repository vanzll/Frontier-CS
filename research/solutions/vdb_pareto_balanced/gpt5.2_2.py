import os
import time
import gc
from typing import Tuple, Optional, List

import numpy as np

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

        self.index_type = kwargs.get("index_type", "auto")  # auto | flat | ivf
        self.target_ms = float(kwargs.get("target_ms", 5.35))
        self.threads = int(kwargs.get("threads", min(8, os.cpu_count() or 1)))

        self.nlist = int(kwargs.get("nlist", 4096))
        self.nprobe = kwargs.get("nprobe", None)
        self.max_nprobe = int(kwargs.get("max_nprobe", 512))
        self.nprobe_candidates = kwargs.get(
            "nprobe_candidates",
            [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512],
        )

        self._faiss_index = None
        self._ntotal = 0
        self._trained = False

        self._rng = np.random.default_rng(int(kwargs.get("seed", 12345)))

        if faiss is not None:
            try:
                faiss.omp_set_num_threads(self.threads)
            except Exception:
                pass

    def _as_f32_contig(self, x: np.ndarray) -> np.ndarray:
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)
        return x

    def _sample_rows(self, x: np.ndarray, n: int) -> np.ndarray:
        n = int(n)
        if n <= 0:
            return x[:0]
        N = x.shape[0]
        if n >= N:
            return x
        idx = self._rng.choice(N, size=n, replace=False)
        return x[idx]

    def _time_search_ms_per_query(self, index, xq: np.ndarray, k: int = 1, reps: int = 1) -> float:
        xq = self._as_f32_contig(xq)
        best = float("inf")
        for _ in range(max(1, int(reps))):
            t0 = time.perf_counter()
            index.search(xq, int(k))
            t1 = time.perf_counter()
            ms = (t1 - t0) * 1000.0 / max(1, xq.shape[0])
            if ms < best:
                best = ms
        return best

    def _build_flat(self, xb: np.ndarray):
        idx = faiss.IndexFlatL2(self.dim)
        idx.add(xb)
        return idx

    def _build_ivf(self, xb: np.ndarray):
        N = xb.shape[0]
        nlist = int(self.nlist)
        if nlist <= 0:
            nlist = 1
        if nlist > N:
            nlist = max(1, int(np.sqrt(N)))
        self.nlist = nlist

        quantizer = faiss.IndexFlatL2(self.dim)
        idx = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_L2)

        min_train = max(nlist * 40, 200_000)
        train_n = min(N, min_train)
        train_n = max(train_n, min(N, nlist * 5))
        train_n = min(N, train_n)

        xt = self._sample_rows(xb, train_n)
        idx.train(xt)
        idx.add(xb)
        return idx

    def _autotune_ivf_nprobe(self, idx, xb: np.ndarray):
        if self.nprobe is not None:
            try:
                idx.nprobe = int(self.nprobe)
            except Exception:
                pass
            return

        qn = min(1500, xb.shape[0])
        qn = max(200, qn)
        xq_cal = self._sample_rows(xb, qn)

        candidates: List[int] = []
        for v in self.nprobe_candidates:
            iv = int(v)
            if iv <= 0:
                continue
            if iv > self.max_nprobe:
                continue
            candidates.append(iv)
        if not candidates:
            candidates = [1]

        best_nprobe = candidates[0]
        for npb in candidates:
            try:
                idx.nprobe = int(npb)
            except Exception:
                continue
            ms = self._time_search_ms_per_query(idx, xq_cal, k=1, reps=1)
            if ms <= self.target_ms:
                best_nprobe = int(npb)
            else:
                break

        try:
            idx.nprobe = int(best_nprobe)
        except Exception:
            pass
        self.nprobe = int(best_nprobe)

    def add(self, xb: np.ndarray) -> None:
        if faiss is None:
            if self._faiss_index is None:
                self._faiss_index = self._as_f32_contig(xb).copy()
                self._ntotal = self._faiss_index.shape[0]
            else:
                arr = self._as_f32_contig(xb)
                self._faiss_index = np.vstack([self._faiss_index, arr])
                self._ntotal = self._faiss_index.shape[0]
            return

        xb = self._as_f32_contig(xb)
        if xb.shape[1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got xb.shape={xb.shape}")

        if self._faiss_index is not None:
            self._faiss_index.add(xb)
            self._ntotal += xb.shape[0]
            return

        try:
            faiss.omp_set_num_threads(self.threads)
        except Exception:
            pass

        N = xb.shape[0]
        if self.index_type == "flat" or (self.index_type == "auto" and N <= 200_000):
            self._faiss_index = self._build_flat(xb)
            self._ntotal = N
            self._trained = True
            return

        chosen = None

        if self.index_type == "auto":
            flat_idx = self._build_flat(xb)
            qn = min(200, N)
            qn = max(50, qn)
            xq_cal = self._sample_rows(xb, qn)
            ms_flat = self._time_search_ms_per_query(flat_idx, xq_cal, k=1, reps=1)
            if ms_flat <= self.target_ms:
                chosen = flat_idx
                self.index_type = "flat"
            else:
                del flat_idx
                gc.collect()

        if chosen is None:
            if self.index_type not in ("ivf", "auto"):
                self.index_type = "ivf"
            ivf_idx = self._build_ivf(xb)
            self._autotune_ivf_nprobe(ivf_idx, xb)
            chosen = ivf_idx
            self.index_type = "ivf"

        self._faiss_index = chosen
        self._ntotal = N
        self._trained = True

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        k = int(k)
        if k <= 0:
            raise ValueError("k must be >= 1")

        if self._faiss_index is None:
            nq = int(xq.shape[0])
            D = np.full((nq, k), np.inf, dtype=np.float32)
            I = np.full((nq, k), -1, dtype=np.int64)
            return D, I

        if faiss is None:
            xb = self._faiss_index
            xq = self._as_f32_contig(xq)
            xb = self._as_f32_contig(xb)

            xq_norm = np.sum(xq * xq, axis=1, keepdims=True)
            xb_norm = np.sum(xb * xb, axis=1, keepdims=True).T
            G = xq @ xb.T
            dist = xq_norm + xb_norm - 2.0 * G

            if k == 1:
                I = np.argmin(dist, axis=1).astype(np.int64)[:, None]
                D = dist[np.arange(dist.shape[0]), I[:, 0]].astype(np.float32)[:, None]
                return D, I

            part = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
            row = np.arange(dist.shape[0])[:, None]
            pd = dist[row, part]
            order = np.argsort(pd, axis=1)
            I = part[row, order].astype(np.int64)
            D = pd[row, order].astype(np.float32)
            return D, I

        xq = self._as_f32_contig(xq)
        if xq.shape[1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got xq.shape={xq.shape}")

        D, I = self._faiss_index.search(xq, k)
        if D.dtype != np.float32:
            D = D.astype(np.float32, copy=False)
        if I.dtype != np.int64:
            I = I.astype(np.int64, copy=False)
        return D, I