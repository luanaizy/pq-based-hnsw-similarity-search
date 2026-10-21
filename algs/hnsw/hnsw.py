import numpy as np
import faiss

class HNSW:
    """
    Índice HNSW baseado na biblioteca Faiss com suporte a CPU e GPU.
    """
    def __init__(self, dim: int = 128, M: int = 32, ef_construction: int = 64, ef_search: int = 32, use_gpu: bool = False):
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        
        # Cria o índice base em CPU
        cpu_index = faiss.IndexHNSWFlat(self.dim, self.M)
        cpu_index.hnsw.efConstruction = self.ef_construction
        cpu_index.hnsw.efSearch = self.ef_search

        if use_gpu:
            try:
                print("Tentando carregar o índice HNSW para a GPU NVIDIA...")
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                print("Sucesso: Índice transferido para a GPU NVIDIA!")
            except Exception as e:
                print(f"Aviso: Não foi possível usar a GPU ({e}). Usando CPU como fallback.")
                self.index = cpu_index
        else:
            self.index = cpu_index

    def insert(self, vectors: np.ndarray):
        """Adiciona vetores ao índice."""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index.add(vectors)

    def knn_search(self, queries: np.ndarray, k: int = 5) -> tuple:
        """Busca os k vizinhos mais próximos."""
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        
        if hasattr(self.index, 'hnsw'):
            self.index.hnsw.efSearch = self.ef_search
            
        distances, indices = self.index.search(queries, k)
        return distances, indices