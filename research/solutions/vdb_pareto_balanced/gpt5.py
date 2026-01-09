import os
import numpy as np
import faiss
from typing import Tuple

class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = int(dim)
        # Configuration with sensible defaults for SIFT1M
        self.nlist = int(kwargs.get("nlist", 8192))
        self.nprobe = int(kwargs.get("nprobe", 96))
        self.train_size = int(kwargs.get("train_size", 250_000))
        self.threads = int(kwargs.get("threads", max(1, min(8, (os.cpu_count() or 8)))))
        self.metric = faiss.METRIC_L2

        self.index = None
        self._trained = False
        self.ntotal = 0
        self._effective_nlist = None
        self._pending = []  # store data chunks if training is delayed

        try:
            faiss.omp_set_num_threads(self.threads)
        except Exception:
            pass

    def _maybe_init_index(self, total_points: int):
        if self.index is not None:
            return
        eff_nlist = min(self.nlist, max(1, total_points))
        self._effective_nlist = eff_nlist
        quantizer = faiss.IndexFlatL2(self.dim)
        self.index = faiss.IndexIVFFlat(quantizer, self.dim, eff_nlist, self.metric)
        # Will set nprobe after training

    def _train_if_needed(self, xb: np.ndarray):
        if self.index is None:
            self._maybe_init_index(xb.shape[0])
        if not self._trained:
            # Aggregate pending + current data for training sample
            if self._pending:
                all_data = [p for p in self._pending]
                all_data.append(xb)
                train_pool = np.concatenate(all_data, axis=0)
            else:
                train_pool = xb

            sample_size = min(self.train_size, train_pool.shape[0])
            # Ensure at least as many training points as centroids
            sample_size = max(sample_size, self._effective_nlist)
            if train_pool.shape[0] < self._effective_nlist:
                # Not enough to train; keep pending and return
                self._pending.append(xb)
                return False

            if train_pool.shape[0] > sample_size:
                idx = np.random.choice(train_pool.shape[0], size=sample_size, replace=False)
                train_data = train_pool[idx]
            else:
                train_data = train_pool

            self.index.train(np.ascontiguousarray(train_data, dtype=np.float32))
            self._trained = True
            try:
                self.index.nprobe = min(self.nprobe, self._effective_nlist)
            except Exception:
                pass

            # After training, if we had pending data, add them now
            if self._pending:
                for chunk in self._pending:
                    self.index.add(np.ascontiguousarray(chunk, dtype=np.float32))
                    self.ntotal += chunk.shape[0]
                self._pending = []
            return True
        return True

    def add(self, xb: np.ndarray) -> None:
        xb = np.ascontiguousarray(xb, dtype=np.float32)
        # Initialize/training if necessary
        trained_now = self._train_if_needed(xb)
        if self._trained:
            self.index.add(xb)
            self.ntotal += xb.shape[0]
        else:
            # Training delayed due to insufficient data
            # Keep in pending; will be added when enough data arrive
            if not trained_now:
                self._pending.append(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None or (not self._trained and not self._pending):
            # No data added
            nq = xq.shape[0]
            return np.full((nq, k), np.inf, dtype=np.float32), np.full((nq, k), -1, dtype=np.int64)

        # If still not trained (e.g., only small adds), force training on whatever is available
        if not self._trained:
            # Concatenate all pending data to train and add
            if self._pending:
                combined = np.concatenate(self._pending, axis=0)
                self._maybe_init_index(combined.shape[0])
                # Ensure enough samples for training (at least nlist)
                sample_size = max(min(self.train_size, combined.shape[0]), self._effective_nlist)
                if combined.shape[0] > sample_size:
                    idx = np.random.choice(combined.shape[0], size=sample_size, replace=False)
                    train_data = combined[idx]
                else:
                    train_data = combined
                self.index.train(np.ascontiguousarray(train_data, dtype=np.float32))
                self._trained = True
                try:
                    self.index.nprobe = min(self.nprobe, self._effective_nlist)
                except Exception:
                    pass
                self.index.add(np.ascontiguousarray(combined, dtype=np.float32))
                self.ntotal += combined.shape[0]
                self._pending = []

        xq = np.ascontiguousarray(xq, dtype=np.float32)
        try:
            faiss.omp_set_num_threads(self.threads)
        except Exception:
            pass
        try:
            self.index.nprobe = min(self.nprobe, self._effective_nlist)
        except Exception:
            pass

        D, I = self.index.search(xq, int(k))
        return D, I