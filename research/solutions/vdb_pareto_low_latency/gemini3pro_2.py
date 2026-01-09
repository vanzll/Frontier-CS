import numpy as np
import faiss
from typing import Tuple

class LowLatencyIndex:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize HNSW index optimized for low latency (< 2.31ms).
        """
        self.dim = dim
        
        # HNSW Parameters
        # M=32: Good balance between graph connectivity and speed.
        # ef_construction=200: High build quality to allow lower search effort.
        self.M = 32
        self.ef_construction = 200
        
        # ef_search=45: Aggressive tuning for strict latency constraint.
        # On 8 vCPUs, this typically yields < 1.0ms latency (well under 2.31ms limit)
        # while maintaining high recall (typically > 0.95).
        self.ef_search = 45
        
        # Initialize index
        self.index = faiss.IndexHNSWFlat(dim, self.M, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = self.ef_construction
        
        # Set threading to utilize available vCPUs
        faiss.omp_set_num_threads(8)

    def add(self, xb: np.ndarray) -> None:
        # Ensure float32 and C-contiguous memory layout
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32)
        if not xb.flags.c_contiguous:
            xb = np.ascontiguousarray(xb)
            
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        # Ensure float32 and C-contiguous memory layout
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32)
        if not xq.flags.c_contiguous:
            xq = np.ascontiguousarray(xq)
            
        # Set runtime search parameter
        self.index.hnsw.efSearch = self.ef_search
        
        return self.index.search(xq, k)