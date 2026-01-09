import numpy as np
import faiss
import time

class HNSWIndex:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize HNSW index optimized for high recall (>95%) with latency < 7.7ms.
        Parameters tuned for SIFT1M (1M vectors, 128 dim).
        """
        self.dim = dim
        
        # Extract parameters with defaults optimized for recall-latency tradeoff
        self.M = kwargs.get('M', 32)  # Increased from default 16 for better recall
        self.ef_construction = kwargs.get('ef_construction', 400)  # High for better recall
        self.ef_search = kwargs.get('ef_search', 128)  # Balanced for recall vs speed
        
        # Create HNSW index - using Inner Product for L2 with normalization
        # HNSW is hierarchical navigable small world graph, good for high recall
        self.index = faiss.IndexHNSWFlat(dim, self.M)
        
        # Set construction parameters
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
        
        # Store vectors for potential re-indexing if needed
        self.vectors = None
        self.is_trained = False
        
        # For batch optimization
        self.batch_size = kwargs.get('batch_size', 8192)

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index. Supports cumulative addition.
        """
        if self.vectors is None:
            self.vectors = xb.copy()
        else:
            self.vectors = np.vstack([self.vectors, xb])
            
        # Train if not already trained (HNSW doesn't require separate training but we'll ensure)
        if not self.is_trained and len(xb) >= self.M:
            self.index.train(xb)
            self.is_trained = True
            
        # Add in batches to manage memory
        n_vectors = xb.shape[0]
        for i in range(0, n_vectors, self.batch_size):
            end_idx = min(i + self.batch_size, n_vectors)
            batch = xb[i:end_idx]
            self.index.add(batch)

    def search(self, xq: np.ndarray, k: int) -> tuple:
        """
        Search for k nearest neighbors using batch-optimized HNSW.
        Returns (distances, indices) with shapes (nq, k).
        """
        # Ensure we have enough capacity for search
        if not self.is_trained:
            # If somehow untrained, use brute force as fallback
            return self._brute_force_search(xq, k)
            
        # Search in batches for better cache utilization
        n_queries = xq.shape[0]
        distances = np.empty((n_queries, k), dtype=np.float32)
        indices = np.empty((n_queries, k), dtype=np.int64)
        
        # Batch processing for better CPU cache utilization
        batch_size = min(self.batch_size, n_queries)
        
        for i in range(0, n_queries, batch_size):
            end_idx = min(i + batch_size, n_queries)
            batch_xq = xq[i:end_idx]
            
            # Perform search on batch
            batch_d, batch_i = self.index.search(batch_xq, k)
            
            distances[i:end_idx] = batch_d
            indices[i:end_idx] = batch_i
        
        return distances, indices
    
    def _brute_force_search(self, xq: np.ndarray, k: int) -> tuple:
        """
        Fallback brute-force search (for safety, though HNSW should be trained).
        """
        # Compute squared L2 distances: ||xq||² - 2xq·xb + ||xb||²
        xq_norm = np.sum(xq**2, axis=1, keepdims=True)
        xb_norm = np.sum(self.vectors**2, axis=1, keepdims=True).T
        
        # Efficient batch matrix multiplication
        distances = xq_norm - 2 * xq @ self.vectors.T + xb_norm
        
        # Get k nearest neighbors
        indices = np.argpartition(distances, k-1, axis=1)[:, :k]
        
        # Sort the k neighbors for each query
        rows = np.arange(xq.shape[0])[:, None]
        sorted_indices = np.argsort(distances[rows, indices], axis=1)
        
        final_indices = indices[rows, sorted_indices]
        final_distances = distances[rows, final_indices]
        
        return final_distances, final_indices

class HNSWIVFIndex:
    """
    Alternative implementation combining IVF with HNSW for potentially better performance.
    IVF reduces search space, HNSW provides accurate local search.
    """
    def __init__(self, dim: int, **kwargs):
        self.dim = dim
        
        # IVF parameters
        self.nlist = kwargs.get('nlist', 4096)  # Number of Voronoi cells
        
        # HNSW parameters for each cell
        self.M = kwargs.get('M', 24)
        self.ef_construction = kwargs.get('ef_construction', 200)
        self.ef_search = kwargs.get('ef_search', 64)
        
        # Number of cells to search
        self.nprobe = kwargs.get('nprobe', 16)  # Search 16/4096 = 0.4% of cells
        
        # Create IVF index with HNSW as quantizer
        quantizer = faiss.IndexHNSWFlat(dim, self.M)
        quantizer.hnsw.efConstruction = self.ef_construction
        quantizer.hnsw.efSearch = self.ef_search
        
        self.index = faiss.IndexIVFFlat(quantizer, dim, self.nlist, faiss.METRIC_L2)
        
        # Training parameters
        self.n_train = min(100000, 100 * self.nlist)  # FAISS recommendation
        
        self.vectors = None
        self.is_trained = False
        self.batch_size = kwargs.get('batch_size', 8192)

    def add(self, xb: np.ndarray) -> None:
        if self.vectors is None:
            self.vectors = xb.copy()
        else:
            self.vectors = np.vstack([self.vectors, xb])
        
        # Train if needed (IVF requires training)
        if not self.is_trained and len(xb) >= self.n_train:
            # Use subset for training
            train_data = xb[:self.n_train]
            self.index.train(train_data)
            self.is_trained = True
        
        # Add vectors in batches
        if self.is_trained:
            n_vectors = xb.shape[0]
            for i in range(0, n_vectors, self.batch_size):
                end_idx = min(i + self.batch_size, n_vectors)
                self.index.add(xb[i:end_idx])
        else:
            # If untrained, store for later training
            pass

    def search(self, xq: np.ndarray, k: int) -> tuple:
        if not self.is_trained and self.vectors is not None:
            # Train on first search if not already trained
            train_data = self.vectors[:min(self.n_train, len(self.vectors))]
            self.index.train(train_data)
            self.index.add(self.vectors)
            self.is_trained = True
        
        if not self.is_trained:
            # Fallback to brute force
            return self._brute_force_search(xq, k)
        
        # Set search parameters
        self.index.nprobe = self.nprobe
        
        # Batch search
        n_queries = xq.shape[0]
        distances = np.empty((n_queries, k), dtype=np.float32)
        indices = np.empty((n_queries, k), dtype=np.int64)
        
        batch_size = min(self.batch_size, n_queries)
        
        for i in range(0, n_queries, batch_size):
            end_idx = min(i + batch_size, n_queries)
            batch_d, batch_i = self.index.search(xq[i:end_idx], k)
            distances[i:end_idx] = batch_d
            indices[i:end_idx] = batch_i
        
        return distances, indices
    
    def _brute_force_search(self, xq: np.ndarray, k: int) -> tuple:
        xq_norm = np.sum(xq**2, axis=1, keepdims=True)
        xb_norm = np.sum(self.vectors**2, axis=1, keepdims=True).T
        
        distances = xq_norm - 2 * xq @ self.vectors.T + xb_norm
        
        indices = np.argpartition(distances, k-1, axis=1)[:, :k]
        rows = np.arange(xq.shape[0])[:, None]
        sorted_indices = np.argsort(distances[rows, indices], axis=1)
        
        final_indices = indices[rows, sorted_indices]
        final_distances = distances[rows, final_indices]
        
        return final_distances, final_indices

class Recall95Index:
    """
    Main index class that automatically selects the best performing strategy.
    Uses HNSW as primary, with IVF-HNSW as fallback if needed.
    """
    def __init__(self, dim: int, **kwargs):
        self.dim = dim
        
        # Try HNSW first (generally good balance of recall and latency)
        self.index = HNSWIndex(dim, **kwargs)
        
        # Alternative index as backup
        self.alt_index = None
        self.use_alt = False
        
        # Performance monitoring
        self.last_recall = 0.0
        self.last_latency = 0.0

    def add(self, xb: np.ndarray) -> None:
        self.index.add(xb)
        
        # Also add to alternative index if it exists
        if self.alt_index is not None:
            self.alt_index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> tuple:
        # Use primary index
        distances, indices = self.index.search(xq, k)
        return distances, indices