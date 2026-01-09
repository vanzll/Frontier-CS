import numpy as np
import faiss
from typing import Tuple


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = dim

        # Configuration parameters (tuned for SIFT1M)
        self.max_nlist = int(kwargs.get("nlist", 8192))          # maximum number of IVF clusters
        self.nprobe = int(kwargs.get("nprobe", 256))             # number of clusters to search
        self.train_size = int(kwargs.get("train_size", 200000))  # training subset size for k-means
        self.cluster_size = int(kwargs.get("cluster_size", 128)) # target vectors per cluster
        self.random_seed = int(kwargs.get("random_seed", 123))

        # FAISS index, created lazily on first add()
        self.index = None
        self.ntotal = 0

        # RNG for training sampling
        self._rng = np.random.RandomState(self.random_seed)

        # Configure FAISS threads (use all available by default)
        try:
            if hasattr(faiss, "omp_get_max_threads") and hasattr(faiss, "omp_set_num_threads"):
                max_threads = faiss.omp_get_max_threads()
                num_threads = kwargs.get("num_threads", max_threads)
                if not isinstance(num_threads, int) or num_threads <= 0 or num_threads > max_threads:
                    num_threads = max_threads
                faiss.omp_set_num_threads(num_threads)
        except Exception:
            # If anything goes wrong, just proceed with FAISS defaults
            pass

    def _create_ivf_index(self, xb: np.ndarray) -> None:
        """Create and train an IVF-Flat index based on the first batch of vectors."""
        n_vectors = xb.shape[0]
        if n_vectors == 0:
            return

        # Determine training size
        n_train = min(self.train_size, n_vectors)

        # Choose nlist to keep ~constant candidate count across dataset sizes
        # Target: about `cluster_size` vectors per list
        nlist = max(1, n_vectors // self.cluster_size)
        nlist = min(nlist, self.max_nlist)

        # Ensure nlist does not exceed number of training points
        if nlist > n_train:
            nlist = max(1, n_train // 2)

        # Final safety clamp
        nlist = max(1, int(nlist))

        index_str = f"IVF{nlist},Flat"
        self.index = faiss.index_factory(self.dim, index_str, faiss.METRIC_L2)

        # Prepare training data
        if n_train < n_vectors:
            train_idx = self._rng.choice(n_vectors, n_train, replace=False)
            train_xb = xb[train_idx]
        else:
            train_xb = xb

        if not self.index.is_trained:
            self.index.train(train_xb)

        # Set nprobe (cannot exceed nlist)
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = min(self.nprobe, nlist)

    def add(self, xb: np.ndarray) -> None:
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError(f"xb must have shape (N, {self.dim}), got {xb.shape}")

        xb = np.ascontiguousarray(xb, dtype="float32")

        if self.index is None:
            # First call: create and train IVF index
            self._create_ivf_index(xb)

        if self.index is None:
            # If index could not be created (e.g., empty xb), nothing to add
            return

        self.index.add(xb)
        self.ntotal = int(self.index.ntotal)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError(f"xq must have shape (nq, {self.dim}), got {xq.shape}")

        xq = np.ascontiguousarray(xq, dtype="float32")
        nq = xq.shape[0]

        if self.index is None or self.ntotal == 0:
            # No data: return empty results
            distances = np.full((nq, k), np.inf, dtype="float32")
            indices = np.full((nq, k), -1, dtype="int64")
            return distances, indices

        # Ensure nprobe is set appropriately before search (in case index was deserialized or modified)
        if hasattr(self.index, "nprobe") and hasattr(self.index, "nlist"):
            nlist = int(getattr(self.index, "nlist"))
            self.index.nprobe = min(self.nprobe, nlist)

        distances, indices = self.index.search(xq, k)
        return distances.astype("float32"), indices.astype("int64")