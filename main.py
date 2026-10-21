import subprocess
import sys
from pathlib import Path

N_VECTORS = 2000  # 10_000
DIMENSIONS = 16   # 128 
N_SUBVECTORS = 8
N_CLUSTERS = 5    # 256
N_QUERIES = 10
K = 5
DP_EPSILON = 1.5
DP_DELTA = 1e-5

USE_GPU = False  # Mude para True quando estiver em maquina com GPU NVIDIA

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.extend(
    [
        str(PROJECT_ROOT / "datasets"),
        str(PROJECT_ROOT / "algs"),
        str(PROJECT_ROOT / "algs" / "hnsw"),
        str(PROJECT_ROOT / "algs" / "kmeans"),
        str(PROJECT_ROOT / "algs" / "product-quantization"),
    ]
)

def check_and_install_dependencies():
    """Verifica e instala o requirements.txt e gerencia o faiss-cpu vs faiss-gpu automaticamente."""
    print("Verificando dependências do ambiente...")
    
    # 1. Instala o requirements.txt base se existir
    req_path = PROJECT_ROOT / "requirements.txt"
    if req_path.exists():
        try:
            # Verifica se o numpy está instalado como teste rápido
            import numpy
        except ImportError:
            print("Instalando dependências do requirements.txt...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_path)])
    
    # 2. Gerencia dinamicamente Faiss CPU vs GPU com base na variável USE_GPU
    try:
        import faiss
        has_faiss = True
    except ImportError:
        has_faiss = False

    if USE_GPU:
        # Se quer usar GPU, garante que o faiss-gpu está instalado e o cpu removido
        try:
            import faiss
            # O faiss-gpu nativo possui suporte a GPU, mas podemos forçar a reinstalação se necessário
            # Uma forma simples é checar se o pacote GPU está ativo
            print("Configurado para GPU: Garantindo instalação do faiss-gpu...")
        except Exception:
            pass
        
        # Desinstala cpu se houver e instala gpu
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "faiss-cpu"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-gpu"])
    else:
        # Se quer usar CPU, garante que o faiss-cpu está instalado e o gpu removido
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "faiss-gpu"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if not has_faiss:
            print("Instalando faiss-cpu...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])

# Executa a verificação antes de importar os módulos que usam o faiss
check_and_install_dependencies()


import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from SIFT import SIFT
from PQBasedHNSWSimilaritySearch import PQBasedHNSWSimilaritySearch
from dpkmeans import DPKMeans
from hnsw import HNSW
from kmeans import KMeans
from product_quantization import ProductQuantization


@dataclass
class SearchMetrics:
    """Métricas agregadas de um experimento."""

    name: str
    epsilon: Optional[float]
    recall_at_5: float
    average_search_time_ms: float


def build_similarity_search(quantizer_cls, quantizer_kwargs) -> PQBasedHNSWSimilaritySearch:
    """Cria uma busca PQ-HNSW usando o quantizador informado e o HNSW do Faiss."""
    product_quantizer = ProductQuantization(
        n_subvectors=N_SUBVECTORS,
        n_clusters_per_subvector=N_CLUSTERS,
        quantizer_cls=quantizer_cls,
        quantizer_kwargs=quantizer_kwargs,
    )
    return PQBasedHNSWSimilaritySearch(
        # Passa a dimensão correta e a flag de GPU (False por padrão, mude para True na máquina do trabalho se desejar)
        hnsw_index=HNSW(dim=DIMENSIONS, M=32, use_gpu=USE_GPU),
        product_quantizer=product_quantizer,
    )


def exact_neighbors(vectors: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Retorna os k vizinhos exatos usando distância Euclidiana."""
    distances = np.linalg.norm(vectors - query, axis=1)
    return np.argpartition(distances, k - 1)[:k]


def evaluate(
    name: str,
    similarity_search: PQBasedHNSWSimilaritySearch,
    vectors: np.ndarray,
    query_ids: np.ndarray,
    epsilon: Optional[float],
) -> SearchMetrics:
    """Treina um índice e calcula Recall@5 e tempo médio de busca."""
    similarity_search.fit(vectors)
    recalls = []
    search_times = []

    for query_id in query_ids:
        query = vectors[query_id]
        ground_truth = set(exact_neighbors(vectors, query, K).tolist())

        started_at = time.perf_counter()
        # O search da classe unificada retorna (distances, indices)
        _, indices = similarity_search.search(query, k=K)
        search_times.append((time.perf_counter() - started_at) * 1000)

        # Extrai os índices da primeira (e única) query enviada
        neighbors = set(indices[0].tolist())

        found = len(ground_truth.intersection(neighbors))
        recalls.append(found / K)

    return SearchMetrics(
        name=name,
        epsilon=epsilon,
        recall_at_5=float(np.mean(recalls)),
        average_search_time_ms=float(np.mean(search_times)),
    )


def print_results(results: list[SearchMetrics]) -> None:
    """Exibe as métricas dos dois quantizadores em formato tabular."""
    print("\nComparação KMeans vs DP-KMeans (com HNSW/Faiss)")
    print("-" * 76)
    print(f"{'Método':<14} {'Epsilon':>10} {'Recall@5':>12} {'Tempo médio (ms)':>20}")
    print("-" * 76)
    for result in results:
        epsilon = "N/A" if result.epsilon is None else f"{result.epsilon:.2f}"
        print(
            f"{result.name:<14} {epsilon:>10} "
            f"{result.recall_at_5:>11.3f} {result.average_search_time_ms:>19.3f}"
        )
    print("-" * 76)


def main() -> None:
    """Compara busca aproximada com KMeans e DP-KMeans."""
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

    results = [
        evaluate(
            "KMeans",
            build_similarity_search(
                KMeans,
                {"max_iter": 10, "random_state": 42},
            ),
            vectors,
            query_ids,
            epsilon=None,
        ),
        evaluate(
            "DP-KMeans",
            build_similarity_search(
                DPKMeans,
                {
                    "epsilon": DP_EPSILON,
                    "delta": DP_DELTA,
                    "max_iter": 10,
                    "random_state": 42,
                },
            ),
            vectors,
            query_ids,
            epsilon=DP_EPSILON,
        ),
    ]
    print(f"Consultas avaliadas: {len(query_ids)} | k={K} | distância: Euclidiana (Faiss)")
    print_results(results)


if __name__ == "__main__":
    main()