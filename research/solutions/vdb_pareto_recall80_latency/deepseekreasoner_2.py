import numpy as np
import faiss
import time

class Recall80LatencyIndex:
    def __init__(self, dim: int, **kwargs):
        self.dim = dim
        self.nlist = kwargs.get('nlist', 2048)
        self.nprobe = kwargs.get('nprobe', 2)
        self.quantizer = None
        self.index = None
        self.vectors_added = 0
        
        # Create the quantizer first
        self.quantizer = faiss.IndexFlatL2(dim)
        
        # Create IVF index with the quantizer
        self.index = faiss.IndexIVFFlat(self.quantizer, dim, self.nlist)
        
        # Set nprobe for search
        self.index.nprobe = self.nprobe
        
        # Enable parallel computation
        faiss.omp_set_num_threads(8)

    def add(self, xb: np.ndarray) -> None:
        if not self.index.is_trained:
            # Train on first batch
            n_train = min(100000, len(xb))
            self.index.train(xb[:n_train])
        
        # Add vectors in batches to manage memory
        batch_size = 50000
        for i in range(0, len(xb), batch_size):
            end_idx = min(i + batch_size, len(xb))
            self.index.add(xb[i:end_idx])
        
        self.vectors_added += len(xb)

    def search(self, xq: np.ndarray, k: int):
        # Ensure k doesn't exceed number of vectors
        k = min(k, self.vectors_added)
        
        # Perform search
        distances, indices = self.index.search(xq, k)
        
        return distances.astype(np.float32), indices.astype(np.int64)