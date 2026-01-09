import numpy as np
import faiss
from typing import Tuple

class Recall95Index:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.
        
        Optimized for SIFT1M to achieve recall@1 >= 0.95 while minimizing latency.
        Uses HNSW graph with M=32 and tuned ef_search.
        """
        self.dim = dim
        
        # HNSW configuration
        # M=32 provides sufficient graph connectivity for high recall on 128D vectors
        # without excessive memory overhead or search latency.
        self.index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_L2)
        
        # Construction parameters
        # High efConstruction ensures a high-quality graph structure, which is critical
        # for achieving high recall with lower efSearch values during queries.
        self.index.hnsw.efConstruction = 200
        
        # Search parameters
        # efSearch controls the trade-off between recall and latency.
        # For SIFT1M with M=32, efSearch=80 typically yields >98% recall, providing
        # a safe buffer above the 0.95 requirement while maintaining sub-millisecond
        # batch latency on modern CPUs.
        self.index.hnsw.efSearch = 80

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.
        """
        # Faiss handles batch addition efficiently
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        """
        # Faiss automatically utilizes OpenMP for parallelizing batch queries
        # Returns (distances, indices)
        return self.index.search(xq, k)