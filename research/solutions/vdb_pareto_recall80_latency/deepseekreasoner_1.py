import numpy as np
import faiss

class Recall80LatencyTierIndex:
    def __init__(self, dim: int, **kwargs):
        """
        Optimized for latency with recall constraint of 80%.
        Uses IVF with product quantization and minimal nprobe.
        """
        self.dim = dim
        
        # Extract parameters or use defaults optimized for latency
        nlist = kwargs.get('nlist', 4096)  # Number of Voronoi cells
        nprobe = kwargs.get('nprobe', 1)   # Minimal search budget
        m = kwargs.get('m', 16)            # Number of subquantizers (128/16=8)
        bits = kwargs.get('bits', 8)       # Bits per subquantizer
        
        # Create coarse quantizer
        quantizer = faiss.IndexFlatL2(dim)
        
        # Create IVF + Product Quantization index for maximum speed
        self.index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, bits)
        
        # Set minimal nprobe for lowest latency
        self.index.nprobe = nprobe
        
        # Enable parallelism for batch queries
        self.index.parallel_mode = 3  # Parallelize over queries
        
        # Track if index is trained
        self.is_trained = False
        self.train_threshold = min(10000, nlist * 40)  # Enough for training
        
    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors with training on first batch if needed.
        """
        # Train on first batch if not trained
        if not self.is_trained:
            # Ensure we have enough data for training
            if len(xb) >= self.train_threshold:
                # Use subset for faster training
                train_size = min(100000, len(xb))
                train_data = xb[:train_size]
                self.index.train(train_data)
                self.is_trained = True
        
        # Add vectors in batches to avoid memory issues
        batch_size = 50000
        for i in range(0, len(xb), batch_size):
            batch = xb[i:i+batch_size]
            if not self.is_trained and len(batch) >= self.train_threshold:
                # Fallback training if first batch was too small
                self.index.train(batch)
                self.is_trained = True
            self.index.add(batch)
            
        # Ensure index is trained even with small datasets
        if not self.is_trained and len(xb) > 0:
            self.index.train(xb)
            self.is_trained = True
    
    def search(self, xq: np.ndarray, k: int) -> tuple:
        """
        Search with minimal overhead.
        Returns distances and indices.
        """
        # Pre-allocate results
        nq = xq.shape[0]
        distances = np.empty((nq, k), dtype=np.float32)
        indices = np.empty((nq, k), dtype=np.int64)
        
        # Perform search
        self.index.search(xq, k, distances, indices)
        
        return distances, indices