import subprocess
import sys
from pathlib import Path

# --- HIPERPARÂMETROS GLOBAIS 
N_VECTORS = 10_000  
DIMENSIONS = 16   #128
N_SUBVECTORS = 4  
N_CLUSTERS = 32   #256
N_QUERIES = 15    
K_DEFAULT = 5
DP_EPSILON = 1.5
DP_DELTA = 1e-5
N_REPETITIONS = 5
USE_GPU = False  # Mude para True quando estiver em maquina com GPU NVIDIA

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.extend(
    [
        str(PROJECT_ROOT / "datasets"),
        str(PROJECT_ROOT / "algs"),
        str(PROJECT_ROOT / "algs" / "hnsw"),
        str(PROJECT_ROOT / "algs" / "kmeans"),
        str(PROJECT_ROOT / "algs" / "product-quantization"),
        str(PROJECT_ROOT / "utils"),
    ]
)
def check_and_install_dependencies():
    """Verifica e instala o requirements.txt dinamicamente e gerencia o faiss-cpu vs faiss-gpu."""
    print("Verificando dependências do ambiente...")
    
    req_path = PROJECT_ROOT / "requirements.txt"
    if req_path.exists():
        print("Sincronizando dependências do requirements.txt...")
        # O pip já verifica internamente se algo novo foi adicionado e instala apenas o que falta
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_path)])
    
    # Gerencia dinamicamente Faiss CPU vs GPU com base na variável USE_GPU
    try:
        import faiss
        has_faiss = True
    except ImportError:
        has_faiss = False

    if USE_GPU:
        # Se quer usar GPU, garante que o faiss-gpu está instalado e o cpu removido
        print("Configurado para GPU: Garantindo instalação do faiss-gpu...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "faiss-cpu"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Instala o faiss-gpu (o pip pula se já estiver instalado e atualizado)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-gpu"])
    else:
        # Se quer usar CPU, garante que o faiss-cpu está instalado e o gpu removido
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "faiss-gpu"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if not has_faiss:
            print("Instalando faiss-cpu...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
# Executa a verificação antes de importar os módulos que usam o faiss
check_and_install_dependencies()

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np

from SIFT import SIFT
from PQBasedHNSWSimilaritySearch import PQBasedHNSWSimilaritySearch
from PQSimilaritySearch import PQSimilaritySearch
from dpkmeans import DPKMeans
from hnsw import HNSW
from kmeans import KMeans
from product_quantization import ProductQuantization
from visualizer import ExperimentVisualizer
from exact_search import ExactSearch


@dataclass
class SearchMetrics:
    """Métricas agregadas de um experimento unificadas."""
    name: str
    epsilon: Optional[float]
    k: int
    recall_at_k: float
    average_search_time_ms: float
    fit_time_seconds: Optional[float] = None  # Opcional para métricas de escalabilidade de treino
    n_vectors: Optional[int] = None           # Opcional para escalabilidade de dados
    n_subvectors: Optional[int] = None        # Opcional para sensibilidade de M


def build_search_engine(method_name: str, quantizer_cls, quantizer_kwargs):
    """Fábrica para instanciar o motor de busca correto (Puro PQ ou PQ+HNSW)."""
    product_quantizer = ProductQuantization(
        n_subvectors=N_SUBVECTORS,
        n_clusters_per_subvector=N_CLUSTERS,
        quantizer_cls=quantizer_cls,
        quantizer_kwargs=quantizer_kwargs,
    )
    
    if "HNSW" in method_name:
        hnsw_index = HNSW(dim=DIMENSIONS, M=32, use_gpu=USE_GPU)
        return PQBasedHNSWSimilaritySearch(hnsw_index=hnsw_index, product_quantizer=product_quantizer)
    else:
        return PQSimilaritySearch(product_quantizer=product_quantizer)


def exact_neighbors(vectors: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Retorna os k vizinhos exatos usando distância Euclidiana (Busca Exata)."""
    distances = np.linalg.norm(vectors - query, axis=1)
    return np.argpartition(distances, k - 1)[:k]


def evaluate_single(
    name: str,
    search_engine,
    vectors: np.ndarray,
    query_ids: np.ndarray,
    epsilon: Optional[float],
    k: int,
) -> SearchMetrics:
    """
    Treina o modelo e calcula a média do Recall@K e da latência 
    ao longo de múltiplas repetições e queries.
    """
    start_fit = time.perf_counter()
    search_engine.fit(vectors)
    fit_duration = time.perf_counter() - start_fit

    all_recalls = []
    all_latencies = []

    rng = np.random.default_rng(42)
    # Utiliza a variável global N_REPETITIONS aqui
    for _ in range(N_REPETITIONS):
        current_query_ids = rng.choice(len(vectors), size=min(len(query_ids), len(vectors)), replace=False)
        
        recalls_run = []
        search_times_run = []

        for query_id in current_query_ids:
            query = vectors[query_id]
            ground_truth = set(exact_neighbors(vectors, query, k).tolist())

            n_runs_internal = 5
            query_latencies = []
            
            for _ in range(n_runs_internal):
                started_at = time.perf_counter()
                _, indices = search_engine.search(query, k=k)
                elapsed = (time.perf_counter() - started_at) * 1000
                query_latencies.append(elapsed)
                
            search_times_run.append(np.mean(query_latencies))
            neighbors = set(indices[0].tolist())

            found = len(ground_truth.intersection(neighbors))
            recalls_run.append(found / k)

        all_recalls.append(np.mean(recalls_run))
        all_latencies.append(np.mean(search_times_run))

    return SearchMetrics(
        name=name,
        epsilon=epsilon,
        k=k,
        recall_at_k=float(np.mean(all_recalls)),
        average_search_time_ms=float(np.mean(all_latencies)),
        fit_time_seconds=fit_duration,
        n_vectors=len(vectors),
    )

def main() -> None:
    """Executa os cenários de avaliação para alimentar todos os gráficos da pesquisa."""
    dataset = SIFT(
        n_vectors=N_VECTORS,
        dim=DIMENSIONS,
        data_dir=str(PROJECT_ROOT / "data"),
    )
    vectors = dataset.vectors
    if vectors is None or len(vectors) == 0:
        raise RuntimeError("O dataset SIFT não contém vetores para indexação.")

    rng = np.random.default_rng(42)
    query_ids = rng.choice(len(vectors), size=min(N_QUERIES, len(vectors)), replace=False)

    print("\n[Experimento 1 & 3] Avaliando os 5 métodos principais (com k fixo e epsilon fixo)...")
    base_results = [
        evaluate_single(
            "Busca Exata", ExactSearch(), vectors, query_ids, epsilon=None, k=K_DEFAULT
        ),
        evaluate_single(
            "PQ", build_search_engine("PQ", KMeans, {"max_iter": 10, "random_state": 42}),
            vectors, query_ids, epsilon=None, k=K_DEFAULT
        ),
        evaluate_single(
            "DP-PQ", build_search_engine("DP-PQ", DPKMeans, {"epsilon": DP_EPSILON, "delta": DP_DELTA, "max_iter": 10, "random_state": 42}),
            vectors, query_ids, epsilon=DP_EPSILON, k=K_DEFAULT
        ),
        evaluate_single(
            "PQ + HNSW", build_search_engine("PQ + HNSW", KMeans, {"max_iter": 10, "random_state": 42}),
            vectors, query_ids, epsilon=None, k=K_DEFAULT
        ),
        evaluate_single(
            "DP-PQ + HNSW", build_search_engine("DP-PQ + HNSW", DPKMeans, {"epsilon": DP_EPSILON, "delta": DP_DELTA, "max_iter": 10, "random_state": 42}),
            vectors, query_ids, epsilon=DP_EPSILON, k=K_DEFAULT
        ),
    ]

    print("\n[Experimento 2 & 4] Avaliando impacto da variação de Epsilon...")
    epsilon_values = [0.5, 1.0, 1.5, 2.0, 5.0]
    dp_epsilon_results = []
    for eps in epsilon_values:
        dp_epsilon_results.append(
            evaluate_single(
                f"DP-PQ (eps={eps})", 
                build_search_engine("DP-PQ", DPKMeans, {"epsilon": eps, "delta": DP_DELTA, "max_iter": 10, "random_state": 42}),
                vectors, query_ids, epsilon=eps, k=K_DEFAULT
            )
        )
        dp_epsilon_results.append(
            evaluate_single(
                f"DP-PQ + HNSW (eps={eps})", 
                build_search_engine("DP-PQ + HNSW", DPKMeans, {"epsilon": eps, "delta": DP_DELTA, "max_iter": 10, "random_state": 42}),
                vectors, query_ids, epsilon=eps, k=K_DEFAULT
            )
        )

    print("\n[Experimento 3] Avaliando o impacto da variação de K...")
    k_values = [1, 5, 10, 20, 50]
    k_variation_results = []
    for test_k in k_values:
        k_variation_results.append(
            evaluate_single(
                "PQ", build_search_engine("PQ", KMeans, {"max_iter": 10, "random_state": 42}),
                vectors, query_ids, epsilon=None, k=test_k
            )
        )
        k_variation_results.append(
            evaluate_single(
                "PQ + HNSW", build_search_engine("PQ + HNSW", KMeans, {"max_iter": 10, "random_state": 42}),
                vectors, query_ids, epsilon=None, k=test_k
            )
        )

    print("\n[Experimento Escalabilidade] Avaliando tempo de treino por tamanho do dataset...")
    vector_sizes = [500, 1000, 2000]
    scalability_results = []
    for size in vector_sizes:
        sub_vectors = vectors[:size]
        sub_queries = query_ids[query_ids < size]
        if len(sub_queries) == 0:
            sub_queries = np.array([0])

        # KMeans Scalability
        res_k = evaluate_single(
            "PQ (KMeans)", build_search_engine("PQ", KMeans, {"max_iter": 5, "random_state": 42}),
            sub_vectors, sub_queries, epsilon=None, k=K_DEFAULT
        )
        scalability_results.append(res_k)

        # DP-KMeans Scalability
        res_dp = evaluate_single(
            "DP-PQ (DP-KMeans)", build_search_engine("DP-PQ", DPKMeans, {"epsilon": DP_EPSILON, "delta": DP_DELTA, "max_iter": 5, "random_state": 42}),
            sub_vectors, sub_queries, epsilon=DP_EPSILON, k=K_DEFAULT
        )
        scalability_results.append(res_dp)

    print("\nGerando todos os gráficos da pesquisa...")
    visualizer = ExperimentVisualizer(
        base_results=base_results, 
        dp_epsilon_results=dp_epsilon_results, 
        k_variation_results=k_variation_results,
        scalability_results=scalability_results,
    )
    
    visualizer.plot_pareto_recall_vs_latency()      # Gráfico 1
    visualizer.plot_epsilon_impact_utility()        # Gráfico 2
    visualizer.plot_k_vs_latency()                  # Gráfico 3
    visualizer.plot_epsilon_utility_vs_latency()    # Gráfico 4 (Subgráficos)
    visualizer.plot_fit_time_vs_n_vectors()         # Gráfico de Escalabilidade


if __name__ == "__main__":
    main()