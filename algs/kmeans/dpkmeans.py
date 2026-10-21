import numpy as np
import math
from typing import Optional, Tuple
from kmeans import KMeans  # Importa a classe KMeans que você criou anteriormente

class DPKMeans:
    """
    Implementação do DP-KMeans baseada no algoritmo DP Lloyd / SuLQ 
    (Blum et al., PODS 2005), utilizando a classe KMeans padrão como base.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 10,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        total_queries: Optional[int] = None,
        random_state: Optional[int] = None
    ):
        """
        Parâmetros:
        - n_clusters: Número de centroides k.
        - max_iter: Número máximo de iterações do K-Means.
        - epsilon: Orçamento de privacidade global (e).
        - delta: Parâmetro de relaxamento da privacidade diferencial (delta).
        - total_queries: Número total estimado de consultas T ao longo do algoritmo.
                         No K-Means do SuLQ, cada iteração gasta k queries de contagem 
                         e k*d queries de somas vetoriais. Se não informado, é estimado.
        - random_state: Semente para reprodutibilidade.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.delta = delta
        self.total_queries = total_queries
        self.random_state = random_state
        
        self.centroids = None
        self.labels_ = None
        self.base_kmeans = None

    def _calculate_suql_noise_variance(self, d: int) -> float:
        """
        Calcula a variância R do ruído Gaussiano N(0, R) com base no Teorema 1 do SuLQ (Blum et al., 2005).
        R >= (2 * T * log(1 / delta)) / epsilon^2
        """
        if self.total_queries is None:
            # Estimativa padrão de queries para o K-Means: max_iter * (k + k * d)
            T = self.max_iter * self.n_clusters * (1 + d)
        else:
            T = self.total_queries

        # Tratamento para epsilon conforme o Teorema 1 do artigo
        eps = max(self.epsilon, 1e-6)
        delta = max(self.delta, 1e-12)
        
        # Variância R baseada na fórmula do Teorema 1 do DPLloyd
        R = (2 * T * math.log(1 / delta)) / (eps ** 2)
        return R

    def fit(self, X: np.ndarray):
        """
        Executa o treinamento do DP-KMeans utilizando consultas ruidosas estilo SuLQ 
        para atualizar os centroides a cada iteração.
        """
        rng = np.random.default_rng(self.random_state)
        n_samples, n_dim = X.shape

        if n_samples < self.n_clusters:
            raise ValueError("O número de amostras é menor que o número de clusters.")

        # 1. Inicializa os centroides usando a classe KMeans normal (ou seleção aleatória segura)
        self.base_kmeans = KMeans(
            n_clusters=self.n_clusters, 
            max_iter=1, 
            random_state=self.random_state
        )
        # Inicializa os centroides de forma idêntica ao KMeans puro
        self.base_kmeans.fit(X)
        self.centroids = self.base_kmeans.centroids.copy()

        # Calcula a variância do ruído SuLQ para esta execução
        R = self._calculate_suql_noise_variance(n_dim)
        noise_std = math.sqrt(R)

        for iteration in range(self.max_iter):
            # Etapa de Atribuição (Calculada com os centroides atuais)
            distances = np.linalg.norm(X[:, np.newaxis, :] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.zeros_like(self.centroids)

            # Etapa de Atualização Privada (SuLQ / DP Lloyd)
            for j in range(self.n_clusters):
                cluster_mask = (labels == j)
                cluster_points = X[cluster_mask]
                
                # Contagem exata de pontos no cluster j
                true_count = len(cluster_points)
                
                # Soma exata dos pontos no cluster j (assume dados limitados em [0, 1]^d conforme o artigo)
                if true_count > 0:
                    true_sum = np.sum(cluster_points, axis=0)
                else:
                    true_sum = np.zeros(n_dim)

                # Adiciona ruído Gaussiano calibrado do SuLQ (DP Lloyd)
                # N(0, R) para a contagem
                noisy_count = true_count + rng.normal(0, noise_std)
                
                # N(0, R) independente para cada dimensão da soma vetorial
                noisy_sum = true_sum + rng.normal(0, noise_std, size=n_dim)

                # Evita divisão por zero ou contagens negativas absurdas
                if noisy_count > 1.0:
                    new_centroids[j] = noisy_sum / noisy_count
                else:
                    # Se o cluster estiver vazio ou com ruído muito baixo, mantém o anterior ou reatribui
                    new_centroids[j] = self.centroids[j]

            # Critério de convergência simples baseado na mudança dos centroides
            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids

            if shift < 1e-4:
                break

        # Atribuição final com os centroides privatizados
        final_distances = np.linalg.norm(X[:, np.newaxis, :] - self.centroids, axis=2)
        self.labels_ = np.argmin(final_distances, axis=1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Atribui novos dados aos centroides privatizados gerados."""
        if self.centroids is None:
            raise ValueError("O modelo DPKMeans não foi treinado.")
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)