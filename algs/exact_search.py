import numpy as np

class ExactSearch:
    """
    Realiza a busca exata (varredura linear com distância Euclidiana) 
    para servir como baseline de precisão máxima (Recall = 1.0).
    """
    def __init__(self):
        self.vectors = None
        self.is_fitted = False

    def fit(self, X: np.ndarray):
        """Armazena o dataset de referência para a busca exata."""
        self.vectors = X
        self.is_fitted = True
        return self

    def search(self, query: np.ndarray, k: int = 5):
        """Calcula a distância exata para todos os vetores e retorna os k mais próximos."""
        if not self.is_fitted:
            raise ValueError("O modelo precisa ser ajustado com 'fit' primeiro.")
            
        query_array = np.asarray(query, dtype=np.float32)
        if query_array.ndim == 1:
            query_array = query_array.reshape(1, -1)
            
        n_queries = query_array.shape[0]
        distances_out = np.zeros((n_queries, k), dtype=np.float32)
        indices_out = np.zeros((n_queries, k), dtype=np.int32)

        for i, q in enumerate(query_array):
            dists = np.linalg.norm(self.vectors - q, axis=1)
            top_k_idx = np.argpartition(dists, k - 1)[:k]
            sorted_top_k = top_k_idx[np.argsort(dists[top_k_idx])]
            
            indices_out[i] = sorted_top_k
            distances_out[i] = dists[sorted_top_k]

        return distances_out, indices_out