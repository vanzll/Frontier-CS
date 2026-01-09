import numpy as np
import faiss
from typing import Tuple

class LowLatencyIndex:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize HNSW index optimized for strict latency constraint (< 2.31ms).
        Strategy: Use HNSW with tuned parameters to balance recall and speed.
        - M=32: Standard graph connectivity for SIFT1M.
        - ef_search=60: Selected to ensure runtime is well within 2.31ms (target ~1.5ms)
          while maintaining high recall (~0.95+).
        """
        self.dim = dim
        self.M = 32
        self.ef_construction = 128
        self.ef_search = 60
        
        # Use HNSW with Flat storage (no quantization) for best recall/speed ratio in RAM
        self.index = faiss.IndexHNSWFlat(dim, self.M, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = self.ef_construction
        
        # Set number of threads to match the 8 vCPU environment
        faiss.omp_set_num_threads(8)

    def add(self, xb: np.ndarray) -> None:
        """
        Add base vectors to the HNSW index.
        """
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors with latency-optimized settings.
        """
        self.index.hnsw.efSearch = self.ef_search
        return self.index.search(xq, k)