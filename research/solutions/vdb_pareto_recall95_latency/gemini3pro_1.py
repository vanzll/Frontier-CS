import numpy as np
import faiss
from typing import Tuple

class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.
        Uses HNSW graph index for optimal latency/recall tradeoff on CPU.
        """
        self.dim = dim
        
        # Configuration for SIFT1M to balance high recall and low latency
        # M=32: Good connectivity for 128D vectors
        self.M = 32
        # ef_construction=128: Sufficient for building a high-quality graph
        self.ef_construction = 128
        # ef_search=64: Tuned to safely exceed 0.95 recall (typically ~0.98+)
        # while keeping latency well under the 7.7ms limit (typically <1ms).
        self.ef_search_base = 64
        
        # Initialize Faiss HNSW Flat index
        # We use Flat storage (no quantization) to maximize recall.
        # 1M x 128 float32 vectors fit comfortably in 16GB RAM.
        self.index = faiss.IndexHNSWFlat(dim, self.M, faiss.METRIC_L2)
        
        # Apply construction parameters
        self.index.hnsw.efConstruction = self.ef_construction
        
        # Optimize threading for the evaluation environment (8 vCPUs)
        faiss.omp_set_num_threads(8)

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.
        """
        # Faiss requires float32 and C-contiguous memory layout
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32, copy=False)
        if not xb.flags.c_contiguous:
            xb = np.ascontiguousarray(xb)
            
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of query vectors.
        """
        # Faiss requires float32 and C-contiguous memory layout
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32, copy=False)
        if not xq.flags.c_contiguous:
            xq = np.ascontiguousarray(xq)
            
        # Adjust efSearch to be at least k, or the baseline parameter
        # This ensures we search enough nodes to find k neighbors while maintaining recall
        self.index.hnsw.efSearch = max(self.ef_search_base, k)
        
        D, I = self.index.search(xq, k)
        
        return D, I