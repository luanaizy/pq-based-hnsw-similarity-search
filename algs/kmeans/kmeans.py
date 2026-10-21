import numpy as np

class KMeans:
    """
    Implementação do algoritmo K-Means usando NumPy.
    Projetado para suportar o treinamento de vocabulários em Quantização de Produto (PQ).
    """

    def __init__(self, n_clusters: int = 256, max_iter: int = 100, tol: float = 1e-4, random_state: int = None):
        """
        Atributos do K-Means:
        - n_clusters (int): Número de centroides (tamanho do vocabulário, ex: 256 para 1 byte).
        - max_iter (int): Número máximo de iterações para convergência.
        - tol (float): Tolerância para critério de parada baseada no deslocamento dos centroides.
        - random_state (int): Semente aleatória para reprodutibilidade.
        - centroids (np.ndarray): Matriz com os centroides finais [n_clusters, n_features].
        - labels_ (np.ndarray): Rótulos de atribuição dos dados de treino.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None

    def fit(self, X: np.ndarray):
        """
        Treina o K-Means encontrando os centroides ideais para a matriz de dados X.
        X deve ter o formato (n_samples, n_features).
        """
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        if n_samples < self.n_clusters:
            raise ValueError("O número de amostras é menor que o número de clusters (n_clusters).")

        # Inicialização aleatória dos centroides escolhendo amostras únicas de X
        random_indices = rng.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices].copy()

        for iteration in range(self.max_iter):
            # 1. Etapa de Atribuição: Calcula as distâncias euclidianas entre pontos e centroides
            # Distância vetorizada: X [n_samples, 1, n_features] vs centroids [1, n_clusters, n_features]
            distances = np.linalg.norm(X[:, np.newaxis, :] - self.centroids, axis=2)
            self.labels_ = np.argmin(distances, axis=1)

            # 2. Etapa de Atualização: Recalcula os centroides como a média dos pontos associados
            new_centroids = np.zeros_like(self.centroids)
            for i in range(self.n_clusters):
                cluster_points = X[self.labels_ == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = np.mean(cluster_points, axis=0)
                else:
                    # Caso um cluster ficasse vazio, reatribuímos aleatoriamente para evitar NaN
                    new_centroids[i] = X[rng.choice(n_samples)]

            # 3. Critério de Parada: Verifica a convergência pelo deslocamento dos centroides
            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids

            if shift < self.tol:
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Atribui novos vetores de entrada X aos centroides (códigos) mais próximos.
        Retorna os índices dos centroides correspondentes.
        """
        if self.centroids is None:
            raise ValueError("O modelo KMeans ainda não foi treinado. Execute 'fit' primeiro.")
        
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Treina o modelo em X e retorna os rótulos de atribuição dos dados de treino.
        """
        self.fit(X)
        return self.labels_