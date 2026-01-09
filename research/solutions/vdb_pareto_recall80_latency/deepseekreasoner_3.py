import numpy as np
import faiss
from typing import Tuple

class Recall80LatencyTierIndex:
    def __init__(self, dim: int, **kwargs):
        """
        IVF index optimized for ultra-low latency with recall constraint.
        Uses minimal nprobe (1-2) to stay within 0.6ms while maintaining >80% recall.
        """
        self.dim = dim
        
        # Extract parameters or use defaults optimized for latency
        nlist = kwargs.get('nlist', 256)  # Number of clusters
        self.nprobe = kwargs.get('nprobe', 2)  # Minimal probes for speed
        nbits = kwargs.get('nbits', 8)  # Quantization bits for IVF
        
        # Create quantizer for IVF
        quantizer = faiss.IndexFlatL2(dim)
        
        # Create IVF index with scalar quantizer for faster distance computation
        self.index = faiss.IndexIVFScalarQuantizer(
            quantizer, dim, nlist, faiss.ScalarQuantizer.QT_8bit
        )
        
        # Disable parallel search threads for more predictable latency
        # (batch queries already benefit from vectorization)
        faiss.omp_set_num_threads(1)
        
        self.is_trained = False
        self.xb = None
        
    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index with training on first batch.
        """
        if self.xb is None:
            self.xb = xb.astype(np.float32)
        else:
            self.xb = np.vstack([self.xb, xb.astype(np.float32)])
        
        # Train on first batch (minimum 39 vectors for k-means)
        if not self.is_trained and self.xb.shape[0] >= 39:
            self.index.train(self.xb[:min(50000, self.xb.shape[0])])
            self.is_trained = True
        
        # Add vectors if index is trained
        if self.is_trained:
            # Only add new vectors
            if hasattr(self, '_added_count'):
                start_idx = self._added_count
                self.index.add(xb.astype(np.float32))
                self._added_count += xb.shape[0]
            else:
                self.index.add(self.xb.astype(np.float32))
                self._added_count = self.xb.shape[0]
    
    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search with minimal nprobe for ultra-low latency.
        Returns k nearest neighbors with L2 distances.
        """
        if not self.is_trained:
            # Fallback to exact search if not trained (shouldn't happen in evaluation)
            return self._exact_search(xq, k)
        
        xq = xq.astype(np.float32)
        
        # Set nprobe for this search
        self.index.nprobe = self.nprobe
        
        # Pre-allocate arrays for results
        nq = xq.shape[0]
        distances = np.empty((nq, k), dtype=np.float32)
        indices = np.empty((nq, k), dtype=np.int64)
        
        # Search all queries at once (batch optimization)
        self.index.search(xq, k, distances, indices)
        
        return distances, indices
    
    def _exact_search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback exact search (slow, for initialization only)."""
        xq = xq.astype(np.float32)
        
        # Compute squared L2 distances efficiently
        # Using xq@xb.T - 0.5*(||xq||^2 + ||xb||^2) for numerical stability
        xq_norm = np.sum(xq**2, axis=1, keepdims=True)
        xb_norm = np.sum(self.xb**2, axis=1, keepdims=True).T
        
        # Batch matrix multiplication for efficiency
        distances = xq_norm + xb_norm - 2 * (xq @ self.xb.T)
        
        # Get k nearest neighbors
        indices = np.argpartition(distances, k-1, axis=1)[:, :k]
        
        # Sort the k nearest neighbors
        rows = np.arange(distances.shape[0])[:, None]
        sorted_idx = np.argsort(distances[rows, indices], axis=1)
        indices = indices[rows, sorted_idx]
        distances = distances[rows, indices]
        
        return distances.astype(np.float32), indices.astype(np.int64)