import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algs.hnsw.hnsw import HNSW
from datasets.sift.SIFT import SIFT


def load_lshrr():
    module_path = PROJECT_ROOT / "algs" / "lshrr" / "LSH-RR.py"
    spec = importlib.util.spec_from_file_location("lsh_rr", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Nao foi possivel carregar {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LSHRR_MODULE = load_lshrr()
BitwiseRandomizedResponse = LSHRR_MODULE.BitwiseRandomizedResponse
LSHRR = LSHRR_MODULE.LSHRR


def read_fvecs(filename, max_vectors, expected_dim):
    return SIFT(
        n_vectors=1,
        dim=expected_dim,
        data_dir=str(PROJECT_ROOT / "datasets"),
    )._read_fvecs(
        str(filename), max_vectors, expected_dim
    )


def exact_knn(vectors, queries, k):
    distances = ((queries[:, None, :] - vectors[None, :, :]) ** 2).sum(axis=2)
    return np.argpartition(distances, kth=k - 1, axis=1)[:, :k]


class MESSIndex:
    """Implementacao do protocolo MESS sobre multiplos HNSW."""

    def __init__(
        self,
        vectors,
        n_shards=4,
        route_degree=2,
        n_bits=128,
        storage_epsilon=2.442,
        prr_epsilon=2.442,
        irr_epsilon=2.442,
        m=16,
        m0=32,
        ef_construction=300,
        route_seed=42,
        rng_seed=43,
    ):
        if not 1 <= route_degree <= n_shards:
            raise ValueError("route_degree deve estar entre 1 e n_shards")
        if n_bits <= 0 or storage_epsilon <= 0:
            raise ValueError("n_bits e storage_epsilon devem ser positivos")

        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.n_shards = n_shards
        self.route_degree = route_degree
        self.route_seed = route_seed
        self.shards = []
        self.routes = []
        self.prr_cache = {}

        for shard_id in range(n_shards):
            shard_rng = np.random.default_rng(rng_seed + shard_id)
            mechanism = LSHRR(
                n_bits=n_bits,
                dim=self.vectors.shape[1],
                epsilon=storage_epsilon,
                rng=shard_rng,
            )
            codes = mechanism.privatize_batch(self.vectors)

            def code_distance(first, second, codes=codes):
                second_code = codes[second] if isinstance(second, (int, np.integer)) else second
                return int(np.count_nonzero(codes[first] != second_code))

            graph = HNSW(
                M=m,
                M0=m0,
                efConstruction=ef_construction,
                distance_func=code_distance,
            )
            self.shards.append(
                {
                    "mechanism": mechanism,
                    "codes": codes,
                    "graph": graph,
                    "prr": BitwiseRandomizedResponse(prr_epsilon, shard_rng),
                    "irr": BitwiseRandomizedResponse(irr_epsilon, shard_rng),
                }
            )

        for item_id in range(len(self.vectors)):
            shard_ids = self._route(item_id)
            self.routes.append(shard_ids)
            for shard_id in shard_ids:
                self.shards[shard_id]["graph"].insert(item_id)

    def _route(self, item_id):
        rng = np.random.default_rng(self.route_seed + item_id)
        return rng.choice(self.n_shards, self.route_degree, replace=False)

    def _query_code(self, query, query_id, shard):
        cache_key = (query_id, id(shard))
        if cache_key not in self.prr_cache:
            clean_code = shard["mechanism"].lsh.hash(query)
            self.prr_cache[cache_key] = shard["prr"].apply(clean_code)
        return shard["irr"].apply(self.prr_cache[cache_key])

    def search(self, query, query_id, k=10, candidates_per_shard=200):
        candidate_ids = set()
        for shard in self.shards:
            query_code = self._query_code(query, query_id, shard)
            ids = shard["graph"].knn_search(
                query_code, candidates_per_shard, candidates_per_shard
            )
            candidate_ids.update(ids)

        if not candidate_ids:
            return []
        candidate_ids = np.fromiter(candidate_ids, dtype=np.int64)
        differences = self.vectors[candidate_ids] - query
        distances = np.einsum("ij,ij->i", differences, differences)
        order = np.argsort(distances)[:k]
        return [(int(candidate_ids[i]), float(distances[i])) for i in order]


def run_experiment(args):
    print(
        f"Carregando SIFT ({args.index_size} vetores, {args.queries} consultas, "
        f"{args.dim}d)..."
    )
    dataset = SIFT(
        n_vectors=args.index_size,
        dim=args.dim,
        data_dir=str(PROJECT_ROOT / "datasets"),
    )
    index_vectors = dataset.vectors
    query_path = PROJECT_ROOT / "datasets" / "sift" / "sift_query.fvecs"
    queries = read_fvecs(query_path, args.queries, args.dim)
    if len(index_vectors) < args.index_size or len(queries) < args.queries:
        raise RuntimeError("Nao foram encontrados vetores suficientes no SIFT")

    build_start = time.perf_counter()
    index = MESSIndex(
        index_vectors,
        n_shards=args.shards,
        route_degree=args.route_degree,
        n_bits=args.bits,
        storage_epsilon=args.storage_epsilon,
        prr_epsilon=args.prr_epsilon,
        irr_epsilon=args.irr_epsilon,
        m=args.m,
        m0=args.m0,
        ef_construction=args.ef_construction,
    )
    build_seconds = time.perf_counter() - build_start

    exact_start = time.perf_counter()
    exact_neighbors = exact_knn(index_vectors, queries, args.k)
    exact_seconds = time.perf_counter() - exact_start

    search_start = time.perf_counter()
    results = [
        index.search(query, query_id, args.k, args.candidates)
        for query_id, query in enumerate(queries)
    ]
    search_seconds = time.perf_counter() - search_start

    recalls = [
        len({item_id for item_id, _ in result}.intersection(expected)) / args.k
        for result, expected in zip(results, exact_neighbors)
    ]
    print(f"Shards: {args.shards} | roteamento: {args.route_degree} shards/item")
    print(f"Bits: {args.bits} | candidatos/shard: {args.candidates}")
    print(f"Construcao MESS: {build_seconds:.3f}s")
    print(f"Busca exata: {exact_seconds:.3f}s")
    print(f"Busca MESS ({args.queries} consultas): {search_seconds:.3f}s")
    print(f"Recall@{args.k}: {np.mean(recalls):.4f}")
    print(f"Latencia MESS por consulta: {search_seconds / args.queries:.6f}s")


def parse_args():
    parser = argparse.ArgumentParser(description="MESS com LSHRR e multi-grafo HNSW.")
    parser.add_argument("--index-size", type=int, default=10000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--route-degree", type=int, default=2)
    parser.add_argument("--bits", type=int, default=128)
    parser.add_argument("--storage-epsilon", type=float, default=2.442)
    parser.add_argument("--prr-epsilon", type=float, default=2.442)
    parser.add_argument("--irr-epsilon", type=float, default=2.442)
    parser.add_argument("--candidates", type=int, default=200)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--m0", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=300)
    args = parser.parse_args()
    if min(args.index_size, args.queries, args.dim, args.k, args.shards) <= 0:
        parser.error("tamanhos, k e shards devem ser positivos")
    if not 1 <= args.route_degree <= args.shards:
        parser.error("route-degree deve estar entre 1 e shards")
    if args.k > args.index_size or args.candidates < args.k:
        parser.error("candidates deve ser >= k e k <= index-size")
    return args


if __name__ == "__main__":
    run_experiment(parse_args())
