import numpy as np

class PQSimilaritySearch:
    """
    Classe de busca de similaridade baseada APENAS em Product Quantization (PQ) 
    ou DP-PQ, realizando a busca por varredura linear (Asymmetric Distance Computation - ADC) 
    sem o auxílio de um índice de grafos como o HNSW.
    """
    def __init__(self, product_quantizer):
        """
        Parâmetros:
        - product_quantizer: Instância da classe ProductQuantization (configurada com KMeans ou DPKMeans).
        """
        self.pq = product_quantizer
        self.codes = None
        self.is_fitted = False

    def fit(self, X: np.ndarray):
        """
        1. Treina o Product Quantization (KMeans / DPKMeans) nos dados X.
        2. Codifica o dataset em códigos compactos e os armazena para busca por ADC.
        """
        print("Iniciando o treinamento do Product Quantization (PQ) puro...")
        self.pq.fit(X)
        
        print("Codificando o dataset para busca linear baseada em PQ...")
        self.codes = self.pq.encode(X)
        
        self.is_fitted = True
        return self

    def search(self, query: np.ndarray, k: int = 5):
        """
        Realiza a busca dos K vizinhos mais próximos utilizando Computação de Distância 
        Assimétrica (ADC) entre a query original e os vetores quantizados armazenados.
        Retorna as distâncias e os índices dos vizinhos mais próximos.
        """
        if not self.is_fitted:
            raise ValueError("O índice precisa ser treinado com o método 'fit' antes de realizar buscas.")
        
        query_array = np.asarray(query, dtype=np.float32)
        if query_array.ndim == 1:
            query_array = query_array.reshape(1, -1)
            
        n_queries = query_array.shape[0]
        n_samples = self.codes.shape[0]
        
        distances = np.zeros((n_queries, k), dtype=np.float32)
        indices = np.zeros((n_queries, k), dtype=np.int32)

        # ADC (Asymmetric Distance Computation):
        # Calculamos a distância entre a query e os centróides de cada subespaço
        sub_dim = query_array.shape[1] // self.pq.M
        
        for q_idx in range(n_queries):
            q = query_array[q_idx]
            # Matriz para acumular as distâncias parciais para cada amostra do dataset: (n_samples,)
            total_distances = np.zeros(n_samples, dtype=np.float32)
            
            for m in range(self.pq.M):
                q_sub = q[m * sub_dim : (m + 1) * sub_dim]
                centroids = self.pq.codebooks[m].centroids  # (K, sub_dim)
                
                # Distância Euclidiana ao quadrado entre a sub-query e todos os centróides deste subespaço
                # Formato resultante: (K,)
                dists_to_centroids = np.sum((centroids - q_sub) ** 2, axis=1)
                
                # Recupera os códigos (índices dos centróides) do subespaço m para todas as amostras
                sub_codes = self.codes[:, m]  # (n_samples,)
                
                # Soma a distância correspondente à distância total acumulada
                total_distances += dists_to_centroids[sub_codes]
            
            # Pega os k menores valores (menores distâncias)
            top_k_idx = np.argpartition(total_distances, k)[:k]
            # Ordena os k melhores de fato
            sorted_top_k = top_k_idx[np.argsort(total_distances[top_k_idx])]
            
            indices[q_idx] = sorted_top_k
            distances[q_idx] = np.sqrt(total_distances[sorted_top_k]) # Raiz para retornar a distância euclidiana real
            
        return distances, indices