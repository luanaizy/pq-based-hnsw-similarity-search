import matplotlib.pyplot as plt
import numpy as np

class ExperimentVisualizer:
    """
    Classe dedicada a plotar os gráficos analíticos do trabalho.
    """
    def __init__(
        self, 
        base_results: list = None, 
        dp_epsilon_results: list = None, 
        k_variation_results: list = None,
        scalability_results: list = None,
    ):
        self.base_results = base_results or []
        self.dp_epsilon_results = dp_epsilon_results or []
        self.k_variation_results = k_variation_results or []
        self.scalability_results = scalability_results or []

    def plot_pareto_recall_vs_latency(self):
        """Gráfico 1: Curva de Pareto - Recall vs Latência para todos os métodos."""
        plt.figure(figsize=(8, 5))
        
        # Coleta dados
        names = [r.name for r in self.base_results]
        recalls = [r.recall_at_k for r in self.base_results]
        latencies = [r.average_search_time_ms for r in self.base_results]

        plt.scatter(latencies, recalls, color='blue', s=100, zorder=3)
        for i, txt in enumerate(names):
            plt.annotate(txt, (latencies[i], recalls[i]), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')

        plt.xscale('log')
        plt.xlabel('Latência Média (ms) [Escala Log]', fontweight='bold')
        plt.ylabel(f'Recall@K', fontweight='bold')
        plt.title('Gráfico 1: Fronteira de Pareto (Recall vs. Latência)', fontweight='bold')
        plt.grid(True, which="both", linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_epsilon_impact_utility(self):
        """Gráfico 2: Impacto do Epsilon na utilidade (Recall)."""
        plt.figure(figsize=(8, 5))
        
        epsilons = sorted(list(set([r.epsilon for r in self.dp_epsilon_results if r.epsilon is not None])))
        
        dp_pq_recalls = [r.recall_at_k for r in self.dp_epsilon_results if "DP-PQ (" in r.name]
        dp_hnsw_recalls = [r.recall_at_k for r in self.dp_epsilon_results if "DP-PQ + HNSW" in r.name]

        plt.plot(epsilons, dp_pq_recalls, marker='o', label='DP-PQ (Linear)', color='orange')
        plt.plot(epsilons, dp_hnsw_recalls, marker='s', label='DP-PQ + HNSW', color='green')

        plt.xlabel('Orçamento de Privacidade ($\epsilon$)', fontweight='bold')
        plt.ylabel('Recall@K', fontweight='bold')
        plt.title('Gráfico 2: Impacto do Ruído Diferencial (Epsilon) na Qualidade', fontweight='bold')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_k_vs_latency(self):
        """Gráfico 3: Custo computacional variando K vs Latência."""
        plt.figure(figsize=(8, 5))
        
        k_values = sorted(list(set([r.k for r in self.k_variation_results])))
        
        pq_latencies = [r.average_search_time_ms for r in self.k_variation_results if r.name == "PQ"]
        hnsw_latencies = [r.average_search_time_ms for r in self.k_variation_results if r.name == "PQ + HNSW"]

        plt.plot(k_values, pq_latencies, marker='o', label='PQ (Varredura Linear)', color='red')
        plt.plot(k_values, hnsw_latencies, marker='^', label='PQ + HNSW (Grafo)', color='blue')

        plt.xlabel('Número de Vizinhos (K)', fontweight='bold')
        plt.ylabel('Latência Média (ms)', fontweight='bold')
        plt.title('Gráfico 3: Escalabilidade da Latência ao Variar K', fontweight='bold')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_epsilon_utility_vs_latency(self):
        """Gráfico 4: Subgráficos (a) Recall vs Epsilon | (b) Latência vs Epsilon."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epsilons = sorted(list(set([r.epsilon for r in self.dp_epsilon_results if r.epsilon is not None])))
        
        dp_pq_recalls = [r.recall_at_k for r in self.dp_epsilon_results if "DP-PQ (" in r.name]
        dp_hnsw_recalls = [r.recall_at_k for r in self.dp_epsilon_results if "DP-PQ + HNSW" in r.name]
        
        dp_pq_lat = [r.average_search_time_ms for r in self.dp_epsilon_results if "DP-PQ (" in r.name]
        dp_hnsw_lat = [r.average_search_time_ms for r in self.dp_epsilon_results if "DP-PQ + HNSW" in r.name]

        # Subgráfico A: Recall
        ax1.plot(epsilons, dp_pq_recalls, marker='o', label='DP-PQ', color='orange')
        ax1.plot(epsilons, dp_hnsw_recalls, marker='s', label='DP-PQ + HNSW', color='green')
        ax1.set_xlabel('Orçamento de Privacidade ($\epsilon$)', fontweight='bold')
        ax1.set_ylabel('Recall@K', fontweight='bold')
        ax1.set_title('(a) Utilidade (Recall) vs. $\epsilon$', fontweight='bold')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)

        # Subgráfico B: Latência
        ax2.plot(epsilons, dp_pq_lat, marker='o', label='DP-PQ', color='orange')
        ax2.plot(epsilons, dp_hnsw_lat, marker='s', label='DP-PQ + HNSW', color='green')
        ax2.set_xlabel('Orçamento de Privacidade ($\epsilon$)', fontweight='bold')
        ax2.set_ylabel('Latência Média (ms)', fontweight='bold')
        ax2.set_title('(b) Latência vs. $\epsilon$ (Invariância)', fontweight='bold')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.suptitle('Gráfico 4: Isolamento de Trade-offs do DP-KMeans', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_fit_time_vs_n_vectors(self):
        """Gráfico de Escalabilidade - Tempo de Treinamento vs N_VECTORS."""
        plt.figure(figsize=(8, 5))
        methods = set([r.name for r in self.scalability_results])
        for method in methods:
            subset = [r for r in self.scalability_results if r.name == method]
            subset = sorted(subset, key=lambda x: x.n_vectors)
            n_vecs = [r.n_vectors for r in subset]
            times = [r.fit_time_seconds for r in subset]
            plt.plot(n_vecs, times, marker='o', label=method, linewidth=2)

        plt.xlabel('Tamanho do Dataset (Número de Vetores)', fontweight='bold')
        plt.ylabel('Tempo de Treinamento [Fit Time] (segundos)', fontweight='bold')
        plt.title('Escalabilidade do Treinamento (KMeans vs. DP-KMeans)', fontweight='bold')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()