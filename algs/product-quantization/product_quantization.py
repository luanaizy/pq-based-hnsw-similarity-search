import numpy as np

class ProductQuantization:
    """
    Implementação de Product Quantization (PQ).
    Divide os vetores de alta dimensão em M subvetores e aplica um quantizador 
    (KMeans ou DPKMeans) em cada subespaço para gerar códigos compactos.
    """

    def __init__(
        self,
        n_subvectors: int = 8,
        n_clusters_per_subvector: int = 256,
        quantizer_cls = None,
        quantizer_kwargs: dict = None
    ):
        """
        Parâmetros:
        - n_subvectors (M): Número de subespaços em que o vetor será dividido.
        - n_clusters_per_subvector (K): Tamanho do vocabulário/centróides por subespaço (ex: 256 para 1 byte).
        - quantizer_cls: A classe do quantizador a ser utilizada (ex: KMeans ou DPKMeans).
        - quantizer_kwargs: Dicionário com parâmetros adicionais para o quantizador (ex: epsilon, max_iter).
        """
        self.M = n_subvectors
        self.K = n_clusters_per_subvector
        self.quantizer_cls = quantizer_cls
        self.quantizer_kwargs = quantizer_kwargs if quantizer_kwargs is not None else {}
        
        self.codebooks = []  # Armazenará os M quantizadores treinados para cada subespaço

    def fit(self, X: np.ndarray):
        """
        Treina os M quantizadores independentes em cada subespaço dos dados X.
        X deve ter formato (n_samples, n_features).
        """
        if self.quantizer_cls is None:
            raise ValueError("Nenhuma classe de quantizador foi fornecida (quantizer_cls).")

        n_samples, d = X.shape
        if d % self.M != 0:
            raise ValueError(f"A dimensionalidade dos vetores ({d}) deve ser divisível pelo número de subvetores ({self.M}).")

        sub_dim = d // self.M
        self.codebooks = []

        print(f"Treinando Product Quantization: {self.M} subespaços de {sub_dim} dimensões cada...")

        for m in range(self.M):
            # Extrai o sub-vetor correspondente ao m-ésimo subespaço para todas as amostras
            X_sub = X[:, m * sub_dim : (m + 1) * sub_dim]

            # Configura os parâmetros do quantizador para este subespaço
            kwargs = self.quantizer_kwargs.copy()
            kwargs['n_clusters'] = self.K  # Garante que o número de centróides seja o definido no PQ

            # Instancia e treina o quantizador (KMeans ou DPKMeans)
            quantizer = self.quantizer_cls(**kwargs)
            quantizer.fit(X_sub)

            self.codebooks.append(quantizer)

        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Codifica um conjunto de vetores X em códigos compactos de tamanho M.
        Retorna uma matriz de inteiros com formato (n_samples, M).
        """
        if not self.codebooks:
            raise ValueError("O modelo de Product Quantization ainda não foi treinado. Execute 'fit' primeiro.")

        n_samples, d = X.shape
        sub_dim = d // self.M
        codes = np.zeros((n_samples, self.M), dtype=np.int32)

        for m in range(self.M):
            X_sub = X[:, m * sub_dim : (m + 1) * sub_dim]
            # Usa o método predict do quantizador treinado para achar o centróide mais próximo
            codes[:, m] = self.codebooks[m].predict(X_sub)

        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        Reconstrói os vetores aproximados a partir dos códigos compactos (tabela de consulta / Asymmetric Distance Computation).
        codes formato: (n_samples, M)
        """
        if not self.codebooks:
            raise ValueError("O modelo não foi treinado.")

        n_samples, M = codes.shape
        sub_dim = self.codebooks[0].centroids.shape[1]
        d = M * sub_dim
        X_reconstructed = np.zeros((n_samples, d), dtype=np.float32)

        for m in range(self.M):
            sub_codes = codes[:, m]
            centroids = self.codebooks[m].centroids
            # Substitui cada índice de código pelo seu respectivo centróide treinado
            X_reconstructed[:, m * sub_dim : (m + 1) * sub_dim] = centroids[sub_codes]

        return X_reconstructed