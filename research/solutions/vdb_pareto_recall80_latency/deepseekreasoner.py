import numpy as np
import faiss
import time
from typing import Tuple

class LatencyOptimizedIndex:
    def __init__(self, dim: int, **kwargs):
        """
        IVF index optimized for latency with recall constraint.
        Uses IVF with 256 centroids and nprobe=1 for extreme speed.
        """
        self.dim = dim
        self.nlist = 256  # Fixed for SIFT1M to balance speed/recall
        self.nprobe = 1   # Minimal probes for maximum speed
        
        # Allow override via kwargs
        if 'nlist' in kwargs:
            self.nlist = kwargs['nlist']
        if 'nprobe' in kwargs:
            self.nprobe = kwargs['nprobe']
            
        # Create the IVF index
        quantizer = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIVFFlat(quantizer, dim, self.nlist, faiss.METRIC_L2)
        self.index.nprobe = self.nprobe
        
        # Track if trained
        self.is_trained = False
        self.vectors_added = 0

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors with training on first batch if needed.
        """
        if not self.is_trained:
            # Train on first 100k vectors (enough for good centroids)
            n_train = min(100000, len(xb))
            self.index.train(xb[:n_train])
            self.is_trained = True
            
        # Add all vectors
        self.index.add(xb)
        self.vectors_added += len(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search with minimal overhead. Returns L2 distances and indices.
        """
        # Ensure index is ready
        if not self.is_trained or self.vectors_added == 0:
            raise RuntimeError("Index not trained or no vectors added")
            
        # Set search parameters
        self.index.nprobe = self.nprobe
        
        # Perform search
        return self.index.search(xq, k)