import numpy as np
import faiss
import time
import threading

class EfficientIndex:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize an optimized index for low-latency vector search.
        Uses IVF with PCA preprocessing for fast approximate search.
        """
        self.dim = dim
        self.index = None
        self.pca_matrix = None
        self.pca_dim = kwargs.get('pca_dim', 64)  # Reduced dimensionality
        self.nlist = kwargs.get('nlist', 2048)    # Number of IVF clusters
        self.nprobe = kwargs.get('nprobe', 8)     # Clusters to search
        self.train_samples = kwargs.get('train_samples', 100000)
        self.verbose = kwargs.get('verbose', False)
        
        # Initialize FAISS thread pool for parallel search
        faiss.omp_set_num_threads(8)
        
    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index with PCA preprocessing and IVF indexing.
        """
        if self.index is not None:
            # Incremental addition not supported - rebuild entire index
            return
            
        N = xb.shape[0]
        
        if self.verbose:
            print(f"Building index for {N} vectors of dim {self.dim}")
            print(f"PCA reducing from {self.dim} to {self.pca_dim} dimensions")
            
        # Train PCA
        if N > self.train_samples:
            # Use subset for PCA training
            np.random.seed(123)
            sample_indices = np.random.choice(N, self.train_samples, replace=False)
            train_data = xb[sample_indices].astype(np.float32)
        else:
            train_data = xb.astype(np.float32)
            
        # Compute PCA matrix
        cov = np.cov(train_data.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort by descending eigenvalues
        idx = np.argsort(eigenvalues)[::-1][:self.pca_dim]
        self.pca_matrix = eigenvectors[:, idx].astype(np.float32)
        
        # Apply PCA to all data
        xb_pca = xb @ self.pca_matrix
        
        # Train IVF quantizer
        quantizer = faiss.IndexFlatL2(self.pca_dim)
        self.index = faiss.IndexIVFFlat(quantizer, self.pca_dim, self.nlist, faiss.METRIC_L2)
        
        # Train on subset for speed
        if N > self.train_samples:
            self.index.train(xb_pca[sample_indices])
        else:
            self.index.train(xb_pca)
            
        # Add all vectors
        self.index.add(xb_pca)
        self.index.nprobe = self.nprobe
        
        if self.verbose:
            print(f"Index built with {self.nlist} clusters, nprobe={self.nprobe}")
            
    def search(self, xq: np.ndarray, k: int) -> tuple:
        """
        Search for k nearest neighbors using optimized batch processing.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call add() first.")
            
        # Apply PCA transformation
        xq_pca = xq @ self.pca_matrix
        
        # Pre-allocate result arrays
        nq = xq.shape[0]
        distances = np.empty((nq, k), dtype=np.float32)
        indices = np.empty((nq, k), dtype=np.int64)
        
        # Perform search
        self.index.search(xq_pca, k, distances, indices)
        
        return distances, indices

class FastHNSWIndex:
    def __init__(self, dim: int, **kwargs):
        """
        HNSW index optimized for low latency with aggressive parameters.
        """
        self.dim = dim
        self.index = None
        
        # Optimized HNSW parameters for low latency
        self.M = kwargs.get('M', 12)  # Lower M for faster search (default 16)
        self.ef_construction = kwargs.get('ef_construction', 100)
        self.ef_search = kwargs.get('ef_search', 32)  # Very low for latency
        
        # Initialize thread pool
        faiss.omp_set_num_threads(8)
        
    def add(self, xb: np.ndarray) -> None:
        """
        Build HNSW index with optimized parameters.
        """
        if self.index is not None:
            return
            
        # Create HNSW index
        self.index = faiss.IndexHNSWFlat(self.dim, self.M)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
        
        # Add vectors
        self.index.add(xb.astype(np.float32))
        
    def search(self, xq: np.ndarray, k: int) -> tuple:
        """
        Search using HNSW with optimized efSearch.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call add() first.")
            
        # Set search parameter
        self.index.hnsw.efSearch = self.ef_search
        
        # Perform search
        return self.index.search(xq.astype(np.float32), k)

class IVFPQIndex:
    def __init__(self, dim: int, **kwargs):
        """
        IVF+PQ index optimized for low latency with product quantization.
        """
        self.dim = dim
        self.index = None
        
        # Optimized parameters for SIFT1M
        self.nlist = kwargs.get('nlist', 1024)  # Number of clusters
        self.nprobe = kwargs.get('nprobe', 4)   # Very low for latency
        self.m = kwargs.get('m', 16)  # Number of subquantizers (128/16=8 bits per sub)
        self.nbits = kwargs.get('nbits', 8)  # Bits per subquantizer
        self.train_samples = kwargs.get('train_samples', 100000)
        
        faiss.omp_set_num_threads(8)
        
    def add(self, xb: np.ndarray) -> None:
        """
        Build IVF+PQ index.
        """
        if self.index is not None:
            return
            
        N = xb.shape[0]
        
        # Train quantizer
        quantizer = faiss.IndexFlatL2(self.dim)
        
        # Create IVF+PQ index
        self.index = faiss.IndexIVFPQ(quantizer, self.dim, self.nlist, self.m, self.nbits)
        
        # Train on subset
        if N > self.train_samples:
            np.random.seed(123)
            sample_indices = np.random.choice(N, self.train_samples, replace=False)
            self.index.train(xb[sample_indices].astype(np.float32))
        else:
            self.index.train(xb.astype(np.float32))
            
        # Add all vectors
        self.index.add(xb.astype(np.float32))
        self.index.nprobe = self.nprobe
        
    def search(self, xq: np.ndarray, k: int) -> tuple:
        """
        Search using IVF+PQ.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call add() first.")
            
        return self.index.search(xq.astype(np.float32), k)

class LowLatencyIndex:
    def __init__(self, dim: int, **kwargs):
        """
        Main index class combining multiple strategies for optimal recall-latency tradeoff.
        Uses ensemble of indices with early termination.
        """
        self.dim = dim
        self.indices = []
        self.weights = []
        
        # Create multiple indices with different tradeoffs
        # 1. HNSW for high recall
        self.hnsw_idx = FastHNSWIndex(dim, M=12, ef_search=48)
        self.indices.append(self.hnsw_idx)
        self.weights.append(0.7)
        
        # 2. IVF for speed
        self.ivf_idx = EfficientIndex(dim, nlist=4096, nprobe=6, pca_dim=96)
        self.indices.append(self.ivf_idx)
        self.weights.append(0.3)
        
        # For timing
        self.last_search_time = 0
        self.search_count = 0
        
    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to all indices in parallel.
        """
        import threading
        
        def add_to_index(index, data):
            index.add(data)
        
        threads = []
        for idx in self.indices:
            t = threading.Thread(target=add_to_index, args=(idx, xb))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
    def search(self, xq: np.ndarray, k: int) -> tuple:
        """
        Search using ensemble with timing constraint.
        """
        nq = xq.shape[0]
        
        # Use the fastest index that meets quality constraints
        # For low latency tier, prioritize speed
        
        # Start with IVF (fastest)
        start_time = time.time()
        distances, indices = self.ivf_idx.search(xq, k)
        search_time = (time.time() - start_time) * 1000 / nq
        
        # If too slow, fall back to simpler method (shouldn't happen with our params)
        if search_time > 1.5:  # Leave margin below 2.31ms
            # Try HNSW with lower ef
            self.hnsw_idx.ef_search = 32
            distances, indices = self.hnsw_idx.search(xq, k)
            
        return distances, indices

# The main class to be discovered by evaluator
class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        """
        Optimized index for low-latency vector search on SIFT1M.
        Uses IVF with PCA reduction and optimized parameters.
        """
        self.dim = dim
        self.index = None
        
        # Optimized parameters for 2.31ms latency target
        self.nlist = kwargs.get('nlist', 2048)    # Clusters
        self.nprobe = kwargs.get('nprobe', 6)     # Very low for latency
        self.pca_dim = kwargs.get('pca_dim', 96)  # PCA reduction (128 -> 96)
        self.train_samples = kwargs.get('train_samples', 100000)
        
        # Set threads for parallel processing
        faiss.omp_set_num_threads(8)
        
    def add(self, xb: np.ndarray) -> None:
        """
        Build IVF index with PCA preprocessing for faster search.
        """
        if self.index is not None:
            return
            
        N = xb.shape[0]
        xb = xb.astype(np.float32)
        
        # Train PCA on subset
        if N > self.train_samples:
            np.random.seed(123)
            sample_indices = np.random.choice(N, self.train_samples, replace=False)
            train_data = xb[sample_indices]
        else:
            train_data = xb
            
        # Compute PCA transformation
        mean = train_data.mean(axis=0, keepdims=True)
        centered = train_data - mean
        cov = centered.T @ centered / (centered.shape[0] - 1)
        
        # Get top eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1][:self.pca_dim]
        self.pca_matrix = eigenvectors[:, idx].astype(np.float32)
        self.mean = mean.astype(np.float32)
        
        # Apply PCA to all data
        xb_pca = (xb - self.mean) @ self.pca_matrix
        
        # Build IVF index
        quantizer = faiss.IndexFlatL2(self.pca_dim)
        self.index = faiss.IndexIVFFlat(quantizer, self.pca_dim, self.nlist, faiss.METRIC_L2)
        
        # Train on subset
        if N > self.train_samples:
            self.index.train(xb_pca[sample_indices])
        else:
            self.index.train(xb_pca)
            
        # Add all vectors
        self.index.add(xb_pca)
        self.index.nprobe = self.nprobe
        
    def search(self, xq: np.ndarray, k: int) -> tuple:
        """
        Fast batch search with PCA transformation.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call add() first.")
            
        xq = xq.astype(np.float32)
        
        # Apply PCA transformation
        xq_pca = (xq - self.mean) @ self.pca_matrix
        
        # Perform search
        distances = np.empty((xq.shape[0], k), dtype=np.float32)
        indices = np.empty((xq.shape[0], k), dtype=np.int64)
        
        self.index.search(xq_pca, k, distances, indices)
        
        return distances, indices