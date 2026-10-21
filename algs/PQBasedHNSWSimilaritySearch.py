import numpy as np

class PQBasedHNSWSimilaritySearch:
    """
    Classe de busca de similaridade baseada em HNSW (via Faiss) combinada com Product Quantization (PQ),
    suportando o uso de KMeans tradicional ou DP-KMeans (Privacidade Diferencial).
    """
    def __init__(self, hnsw_index, product_quantizer):
        """
        Parâmetros:
        - hnsw_index: Instância da classe HNSW atualizada (baseada no Faiss).
        - product_quantizer: Instância da classe ProductQuantization (configurada com KMeans ou DPKMeans).
        """
        self.hnsw = hnsw_index
        self.pq = product_quantizer
        self.is_fitted = False

    def fit(self, X: np.ndarray):
        """
        1. Treina o Product Quantization (KMeans / DPKMeans) nos dados X.
        2. Codifica e reconstrói os vetores no espaço quantizado (PQ).
        3. Insere os vetores resultantes no índice HNSW do Faiss.
        """
        print("Iniciando o treinamento do Product Quantization (PQ)...")
        self.pq.fit(X)
        
        print("Codificando e reconstruindo os vetores do dataset...")
        codes = self.pq.encode(X)
        X_reconstructed = self.pq.decode(codes)
        
        print("Inserindo os vetores no índice HNSW (Faiss)...")
        self.hnsw.insert(X_reconstructed)
        
        self.is_fitted = True
        return self

    def search(self, query: np.ndarray, k: int = 5):
        """
        Realiza a busca aproximada dos K vizinhos mais próximos para uma dada consulta (query).
        Retorna as distâncias e os índices dos vizinhos mais próximos.
        """
        if not self.is_fitted:
            raise ValueError("O índice precisa ser treinado com o método 'fit' antes de realizar buscas.")
        
        query_array = np.asarray(query, dtype=np.float32)
        if query_array.ndim == 1:
            query_array = query_array.reshape(1, -1)
            
        # Executa a busca diretamente no HNSW do Faiss
        distances, indices = self.hnsw.knn_search(query_array, k=k)
        
        return distances, indices