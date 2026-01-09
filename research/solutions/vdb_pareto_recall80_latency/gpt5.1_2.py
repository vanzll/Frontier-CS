import numpy as np
from typing import Tuple

try:
    import faiss
    _FAISS_AVAILABLE = True
except Exception:
    faiss = None
    _FAISS_AVAILABLE = False


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = int(dim)
        if self.dim <= 0:
            raise ValueError("dim must be positive")

        self.use_faiss = _FAISS_AVAILABLE

        nlist = int(kwargs.get("nlist", 4096))
        if nlist <= 0:
            nlist = 4096
        self.nlist = nlist

        nprobe = int(kwargs.get("nprobe", 64))
        if nprobe <= 0:
            nprobe = 64
        self.nprobe = nprobe

        default_max_train_points = min(1000000, self.nlist * 128)
        max_train_points = int(kwargs.get("max_train_points", default_max_train_points))
        if max_train_points <= 0:
            max_train_points = default_max_train_points
        self.max_train_points = max_train_points

        self._index = None  # faiss index or None
        self._xb = None     # fallback storage when faiss is unavailable

    def add(self, xb: np.ndarray) -> None:
        xb = np.ascontiguousarray(xb, dtype="float32")
        if xb.ndim != 2:
            raise ValueError("xb must be a 2D array")
        if xb.shape[1] != self.dim:
            raise ValueError(f"xb dimension {xb.shape[1]} does not match index dim {self.dim}")

        if xb.shape[0] == 0:
            return

        if self.use_faiss:
            if self._index is None:
                n_vectors = xb.shape[0]

                # For very small datasets, use exact Flat index
                if n_vectors < self.nlist:
                    index = faiss.IndexFlatL2(self.dim)
                    index.add(xb)
                    self._index = index
                else:
                    quantizer = faiss.IndexFlatL2(self.dim)
                    index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_L2)
                    index.nprobe = min(self.nprobe, self.nlist)

                    train_size = min(self.max_train_points, n_vectors)
                    train_x = xb[:train_size]
                    index.train(train_x)
                    index.add(xb)
                    self._index = index
            else:
                self._index.add(xb)
        else:
            if self._xb is None:
                self._xb = xb.copy()
            else:
                self._xb = np.vstack((self._xb, xb))

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        xq = np.ascontiguousarray(xq, dtype="float32")
        if xq.ndim != 2:
            raise ValueError("xq must be a 2D array")
        if xq.shape[1] != self.dim:
            raise ValueError(f"xq dimension {xq.shape[1]} does not match index dim {self.dim}")

        k_req = int(k)
        if k_req <= 0:
            raise ValueError("k must be positive")

        nq = xq.shape[0]

        if self.use_faiss:
            if self._index is None or self._index.ntotal == 0:
                D = np.full((nq, k_req), np.inf, dtype="float32")
                I = np.full((nq, k_req), -1, dtype="int64")
                return D, I

            ntotal = self._index.ntotal
            k_eff = min(k_req, ntotal)

            D, I = self._index.search(xq, k_eff)

            if D.dtype != np.float32:
                D = D.astype("float32")
            if I.dtype != np.int64:
                I = I.astype("int64")

            if k_eff < k_req:
                D_pad = np.full((nq, k_req), np.inf, dtype="float32")
                I_pad = np.full((nq, k_req), -1, dtype="int64")
                D_pad[:, :k_eff] = D
                I_pad[:, :k_eff] = I
                D, I = D_pad, I_pad

            return D, I
        else:
            if self._xb is None or self._xb.shape[0] == 0:
                D = np.full((nq, k_req), np.inf, dtype="float32")
                I = np.full((nq, k_req), -1, dtype="int64")
                return D, I

            xb = self._xb
            N = xb.shape[0]
            k_eff = min(k_req, N)

            D = np.empty((nq, k_eff), dtype="float32")
            I = np.empty((nq, k_eff), dtype="int64")

            for i in range(nq):
                diff = xb - xq[i]
                dist = np.einsum("ij,ij->i", diff, diff, optimize=True)

                if k_eff == 1:
                    idx = int(np.argmin(dist))
                    D[i, 0] = dist[idx]
                    I[i, 0] = idx
                else:
                    idx = np.argpartition(dist, k_eff - 1)[:k_eff]
                    part = dist[idx]
                    order = np.argsort(part)
                    idx = idx[order]
                    D[i, :] = dist[idx]
                    I[i, :] = idx.astype("int64")

            if k_eff < k_req:
                D_pad = np.full((nq, k_req), np.inf, dtype="float32")
                I_pad = np.full((nq, k_req), -1, dtype="int64")
                D_pad[:, :k_eff] = D
                I_pad[:, :k_eff] = I
                D, I = D_pad, I_pad

            return D, I