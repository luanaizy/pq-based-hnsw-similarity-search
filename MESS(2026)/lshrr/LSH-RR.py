"""
LSHRR: LSH-then-Randomized-Response privacy mechanism.

Implements the mechanisms and privacy accounting from:

    Natasha Fernandes, Yusuke Kawamoto, Takao Murakami,
    "Locality Sensitive Hashing with Extended Differential Privacy",
    ESORICS 2021 (arXiv:2010.09393).

Provided building blocks
------------------------
- ``RandomProjectionLSH``   : kappa-bit random-projection LSH (SimHash) for
                              the angular distance (Section 3.2, h_proj).
- ``BitwiseRandomizedResponse`` : Warner's randomized response applied
                              independently to every bit (Definition 5).
- ``LSHRR``                 : the composed LSH-then-RR mechanism
                              Q_LSHRR = Q_brr o H* (Definition 6).
- Privacy accounting helpers implementing Propositions 2-5 and Theorem 1
  (Section 6), which bound the extended differential privacy (XDP) that
  LSHRR provides w.r.t. the angular distance.
- Utility helpers (Definition 1/2) to evaluate approximate k-NN / friend
  matching quality, mirroring the paper's experimental setup (Section 7).

Running this file directly reproduces a small friend-matching experiment on
synthetic data, printing both the achieved privacy guarantee and utility
loss, analogous to Figure 2 of the paper.
"""

from __future__ import annotations

import argparse
import math
from typing import Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Distances (Section 3.1-3.2)
# --------------------------------------------------------------------------


def angular_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Angular distance d_theta(x, y) = (1/pi) * arccos(x^T y / (||x|| ||y||)).

    d_theta(x, y) = 0 iff x == y, and d_theta(x, y) = 1 iff x == -y.
    Orthogonal vectors (including the case where either vector is zero) are
    assigned distance 0.5, by convention (Eq. 3).
    """
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    if nx == 0.0 or ny == 0.0:
        return 0.5
    cos_sim = float(np.dot(x, y) / (nx * ny))
    cos_sim = max(-1.0, min(1.0, cos_sim))  # guard against fp round-off
    return math.acos(cos_sim) / math.pi


def hamming_distance(v: np.ndarray, w: np.ndarray) -> int:
    """Hamming distance d_V(v, w) = sum_i |v_i - w_i| for bit-vectors v, w."""
    return int(np.sum(np.abs(v.astype(np.int64) - w.astype(np.int64))))


# --------------------------------------------------------------------------
# Random-projection LSH for the angular distance (Section 3.2, h_proj)
# --------------------------------------------------------------------------


class RandomProjectionLSH:
    """A kappa-bit LSH built from kappa independent random hyperplanes.

    Each one-bit hash is h_proj(x) = 0 if r^T x < 0 else 1, with each element
    of the normal vector r drawn i.i.d. from the standard normal N(0, 1).
    This family is an LSH w.r.t. the angular distance d_theta (Charikar's
    SimHash): Pr[h_proj(x) != h_proj(x')] = d_theta(x, x').
    """

    def __init__(self, n_bits: int, dim: int, rng: Optional[np.random.Generator] = None):
        if n_bits <= 0 or dim <= 0:
            raise ValueError("n_bits and dim must be positive")
        self.n_bits = n_bits
        self.dim = dim
        rng = rng if rng is not None else np.random.default_rng()
        # kappa independent hyperplane normals, one per row -> shape (kappa, dim)
        self.hyperplanes = rng.standard_normal(size=(n_bits, dim))

    def hash(self, x: np.ndarray) -> np.ndarray:
        """Return the kappa-bit hash H(x) in {0,1}^kappa of a single vector x."""
        projections = self.hyperplanes @ x
        return (projections >= 0).astype(np.uint8)

    def hash_batch(self, X: np.ndarray) -> np.ndarray:
        """Vectorized hash for a batch of row-vectors X, shape (N, dim)."""
        projections = X @ self.hyperplanes.T
        return (projections >= 0).astype(np.uint8)


# --------------------------------------------------------------------------
# Warner's randomized response, applied bit-wise (Definition 5)
# --------------------------------------------------------------------------


class BitwiseRandomizedResponse:
    """The (epsilon, kappa)-bitwise randomized response Q_brr.

    Applies epsilon-RR independently to each bit of a kappa-bit string: a bit
    is reported truthfully with probability e^eps / (e^eps + 1), and flipped
    with probability 1 / (e^eps + 1).
    """

    def __init__(self, epsilon: float, rng: Optional[np.random.Generator] = None):
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        self.epsilon = epsilon
        self.rng = rng if rng is not None else np.random.default_rng()
        self.keep_prob = math.exp(epsilon) / (math.exp(epsilon) + 1.0)

    def apply(self, bits: np.ndarray) -> np.ndarray:
        """Randomize a single kappa-bit vector."""
        keep = self.rng.random(bits.shape[-1]) < self.keep_prob
        return np.where(keep, bits, 1 - bits).astype(np.uint8)

    def apply_batch(self, bits: np.ndarray) -> np.ndarray:
        """Randomize a batch of kappa-bit vectors, shape (N, kappa)."""
        keep = self.rng.random(bits.shape) < self.keep_prob
        return np.where(keep, bits, 1 - bits).astype(np.uint8)


# --------------------------------------------------------------------------
# LSHRR mechanism (Definition 6): Q_LSHRR = Q_brr o H*
# --------------------------------------------------------------------------


class LSHRR:
    """The epsilon-LSH-then-RR privacy mechanism (LSHRR).

    Given an n-dimensional input vector x, LSHRR:
      1. hashes x with a kappa-bit LSH function H, randomly chosen once and
         then shared across users (as is typical in NNS applications, where
         the service provider fixes a single hash seed), and
      2. applies epsilon-randomized response independently to every bit of
         H(x) before releasing it.
    """

    def __init__(
        self,
        n_bits: int,
        dim: int,
        epsilon: float,
        rng: Optional[np.random.Generator] = None,
    ):
        rng = rng if rng is not None else np.random.default_rng()
        self.n_bits = n_bits
        self.dim = dim
        self.epsilon = epsilon
        self.lsh = RandomProjectionLSH(n_bits, dim, rng=rng)
        self.rr = BitwiseRandomizedResponse(epsilon, rng=rng)

    def privatize(self, x: np.ndarray) -> np.ndarray:
        """Return one LSHRR-privatized kappa-bit sample for a single input x."""
        return self.rr.apply(self.lsh.hash(x))

    def privatize_batch(self, X: np.ndarray) -> np.ndarray:
        """Return LSHRR-privatized kappa-bit samples for a batch of inputs X."""
        return self.rr.apply_batch(self.lsh.hash_batch(X))


# --------------------------------------------------------------------------
# Privacy accounting (Section 6)
# --------------------------------------------------------------------------


def xdp_for_hash_pair(epsilon: float, hash_hamming_dist: int) -> float:
    """Proposition 2: (d^eps_H, 0)-XDP, where d^eps_H(x,x') = eps * d_V(H(x),H(x')).

    Returns the realized privacy-loss bound for a *given* LSH function H
    (i.e., once the Hamming distance between the two hash values is known).
    """
    return epsilon * hash_hamming_dist


def worst_case_dp(epsilon: float, n_bits: int) -> float:
    """Proposition 3: LSHRR provides (kappa * epsilon)-DP in the worst case,
    independently of the actual choice of LSH function/seed.
    """
    return epsilon * n_bits


def cxdp_params(epsilon: float, n_bits: int) -> Tuple[float, float]:
    """Proposition 4: LSHRR provides (mu, tau, d_theta)-CXDP with
    mu = eps * kappa and tau = eps * kappa / 2.
    """
    mu = epsilon * n_bits
    tau = epsilon * n_bits / 2.0
    return mu, tau


def pxdp_theorem1(epsilon: float, n_bits: int, d_theta: float, delta: float) -> float:
    """Theorem 1: (xi, delta)-PXDP (hence XDP) bound for LSHRR, derived from
    the (mu, tau)-CXDP of Proposition 4 via a Chernoff-Hoeffding argument:

        xi(x, x') = eps * kappa * d_theta(x, x') + eps' * sqrt(kappa)
        eps'      = eps * sqrt(-ln(delta) / 2)

    This bound holds uniformly for all delta in (0, 1] without requiring
    knowledge of d_theta(x, x') in advance of choosing alpha (cf. Prop. 5).
    """
    if not (0.0 < delta <= 1.0):
        raise ValueError("delta must be in (0, 1]")
    eps_prime = epsilon * math.sqrt(-math.log(delta) / 2.0)
    return epsilon * n_bits * d_theta + eps_prime * math.sqrt(n_bits)


def _kl_bernoulli(a: float, b: float) -> float:
    """KL divergence D_KL(a || b) between Bernoulli(a) and Bernoulli(b)."""
    eps = 1e-12
    a = min(max(a, eps), 1 - eps)
    b = min(max(b, eps), 1 - eps)
    return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))


def pxdp_prop5(epsilon: float, n_bits: int, d_theta: float, alpha: float) -> Tuple[float, float]:
    """Proposition 5 (tighter, input-dependent bound). Returns (xi_alpha, delta_alpha):

        xi_alpha(x, x')    = eps * kappa * (d_theta(x, x') + alpha)
        delta_alpha(x, x') = exp( -kappa * D_KL(d_theta(x,x')+alpha || d_theta(x,x')) )

    Requires 0 < alpha and d_theta(x, x') + alpha <= 1.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    xi_alpha = epsilon * n_bits * (d_theta + alpha)
    delta_alpha = math.exp(-n_bits * _kl_bernoulli(d_theta + alpha, d_theta))
    return xi_alpha, delta_alpha


# --------------------------------------------------------------------------
# Utility: approximate k-NN via Hamming distance + utility loss (Definition 2)
# --------------------------------------------------------------------------


def knn_by_distance(distances: np.ndarray, query_idx: int, k: int) -> np.ndarray:
    """Indices of the k smallest distances to query_idx, excluding itself."""
    distances = distances.astype(np.float64, copy=True)
    distances[query_idx] = np.inf
    return np.argsort(distances)[:k]


def true_knn(X: np.ndarray, query_idx: int, k: int) -> np.ndarray:
    """Exact k-NN of X[query_idx] among the rows of X, w.r.t. angular distance."""
    dists = np.array([angular_distance(X[query_idx], X[j]) for j in range(X.shape[0])])
    return knn_by_distance(dists, query_idx, k)


def approx_knn_hamming(codes: np.ndarray, query_idx: int, k: int) -> np.ndarray:
    """Approximate k-NN of codes[query_idx] among the rows of codes, w.r.t.
    the Hamming distance on (possibly privatized) LSH codes.
    """
    dists = np.array([hamming_distance(codes[query_idx], codes[j]) for j in range(codes.shape[0])])
    return knn_by_distance(dists, query_idx, k)


def utility_loss(
    X: np.ndarray, query_idx: int, approx_neighbors: np.ndarray, true_neighbors: np.ndarray
) -> float:
    """Definition 2: average utility loss of approximate neighbours relative
    to the true nearest neighbours, measured in angular distance.
    """
    q = X[query_idx]
    approx_avg = float(np.mean([angular_distance(q, X[j]) for j in approx_neighbors]))
    true_avg = float(np.mean([angular_distance(q, X[j]) for j in true_neighbors]))
    return approx_avg - true_avg


# --------------------------------------------------------------------------
# Demo: privacy-preserving friend matching on synthetic data (cf. Section 7)
# --------------------------------------------------------------------------


def _make_synthetic_dataset(
    n_users: int, dim: int, sparsity: float, rng: np.random.Generator
) -> np.ndarray:
    """Sparse, non-negative "rating/visit-count"-like vectors, akin to the
    MovieLens/Foursquare vectors used in the paper's experiments.
    """
    mask = rng.random((n_users, dim)) < sparsity
    values = rng.integers(1, 6, size=(n_users, dim)).astype(np.float64)
    return mask * values


def run_demo(
    n_users: int = 300,
    dim: int = 200,
    sparsity: float = 0.05,
    n_bits: int = 20,
    epsilon: float = 2.0,
    k: int = 5,
    n_queries: int = 30,
    delta: float = 0.01,
    alpha: float = 0.05,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    X = _make_synthetic_dataset(n_users, dim, sparsity, rng)

    lshrr = LSHRR(n_bits=n_bits, dim=dim, epsilon=epsilon, rng=rng)
    vanilla_codes = lshrr.lsh.hash_batch(X)
    private_codes = lshrr.rr.apply_batch(vanilla_codes)

    query_idxs = rng.choice(n_users, size=n_queries, replace=False)
    losses_lsh, losses_lshrr = [], []
    for q in query_idxs:
        true_nn = true_knn(X, q, k)
        lsh_nn = approx_knn_hamming(vanilla_codes, q, k)
        lshrr_nn = approx_knn_hamming(private_codes, q, k)
        losses_lsh.append(utility_loss(X, q, lsh_nn, true_nn))
        losses_lshrr.append(utility_loss(X, q, lshrr_nn, true_nn))

    print(f"Dataset: {n_users} users, dim={dim}, sparsity={sparsity}")
    print(f"LSH: kappa={n_bits} bits | LSHRR: epsilon={epsilon}\n")
    print(f"Mean utility loss (vanilla LSH):  {np.mean(losses_lsh):.4f}")
    print(f"Mean utility loss (LSHRR):        {np.mean(losses_lshrr):.4f}\n")

    print("Privacy guarantees for LSHRR:")
    print(f"  Proposition 3 (worst-case DP):        kappa*epsilon = {worst_case_dp(epsilon, n_bits):.3f}")
    mu, tau = cxdp_params(epsilon, n_bits)
    print(f"  Proposition 4 (CXDP params):           mu={mu:.3f}, tau={tau:.3f}")
    for d_theta in (0.1, 0.3, 0.5):
        xi1 = pxdp_theorem1(epsilon, n_bits, d_theta, delta)
        xi5, delta5 = pxdp_prop5(epsilon, n_bits, d_theta, alpha)
        print(
            f"  d_theta={d_theta:.1f}: "
            f"Theorem 1 xi={xi1:.3f} (delta={delta}) | "
            f"Prop. 5 xi_alpha={xi5:.3f} (delta_alpha={delta5:.2e}, alpha={alpha})"
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LSHRR demo: privacy-preserving friend matching")
    p.add_argument("--n-users", type=int, default=300)
    p.add_argument("--dim", type=int, default=200)
    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--n-bits", type=int, default=20, help="kappa, the LSH hash length")
    p.add_argument("--epsilon", type=float, default=2.0, help="per-bit RR privacy budget")
    p.add_argument("--k", type=int, default=5, help="number of nearest neighbours")
    p.add_argument("--n-queries", type=int, default=30)
    p.add_argument("--delta", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_demo(
        n_users=args.n_users,
        dim=args.dim,
        sparsity=args.sparsity,
        n_bits=args.n_bits,
        epsilon=args.epsilon,
        k=args.k,
        n_queries=args.n_queries,
        delta=args.delta,
        alpha=args.alpha,
        seed=args.seed,
    )
