"""
biodiversity_metrics.py
=======================

Real ecological biodiversity metrics for the Sea-quenced pipeline.

The original prototype reported a Shannon index drawn from
``np.random.uniform(1.5, 3.5)`` -- i.e. a fake random number.  This module
replaces that with proper, peer-reviewed ecological estimators computed from
the *observed cluster abundances* (each cluster / OTU is treated as a taxon and
its read count as its abundance).

All functions are pure and dependency-light (numpy only) so they are trivial to
unit-test.  ``compute_all`` returns a single dict suitable for a JSON report.

References
----------
* Shannon, C.E. (1948) A Mathematical Theory of Communication.
* Simpson, E.H. (1949) Measurement of Diversity. Nature 163:688.
* Pielou, E.C. (1966) The measurement of diversity in different types of
  biological collections.
* Chao, A. (1984) Non-parametric estimation of the number of classes in a
  population. Scandinavian Journal of Statistics 11:265-270.
* Good, I.J. (1953) The population frequencies of species and the estimation of
  population parameters. Biometrika 40:237-264.
* Margalef, R. (1958) Information theory in ecology.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "as_counts",
    "species_richness",
    "shannon_index",
    "simpson_index",
    "inverse_simpson",
    "pielou_evenness",
    "chao1",
    "goods_coverage",
    "margalef_richness",
    "berger_parker_dominance",
    "rank_abundance",
    "rarefaction_curve",
    "compute_all",
]


def as_counts(abundances: Iterable[float]) -> np.ndarray:
    """Return a clean 1-D array of strictly positive integer-ish counts.

    Zero / negative entries are dropped (a taxon with zero reads is not present).
    """
    arr = np.asarray(list(abundances), dtype=float)
    arr = arr[arr > 0]
    return arr


def species_richness(counts: Sequence[float]) -> int:
    """S -- number of observed taxa (clusters) with at least one read."""
    return int(np.count_nonzero(as_counts(counts)))


def shannon_index(counts: Sequence[float], base: float | None = None) -> float:
    """Shannon diversity H' = -sum(p_i * log(p_i)).

    ``base=None`` uses natural log (nats), the ecological convention.
    """
    c = as_counts(counts)
    if c.size == 0:
        return 0.0
    p = c / c.sum()
    h = -np.sum(p * np.log(p))
    if base is not None:
        h /= math.log(base)
    return float(h)


def simpson_index(counts: Sequence[float]) -> float:
    """Gini-Simpson index (1 - D): probability two random reads are *different* taxa."""
    c = as_counts(counts)
    if c.size == 0:
        return 0.0
    p = c / c.sum()
    return float(1.0 - np.sum(p ** 2))


def inverse_simpson(counts: Sequence[float]) -> float:
    """Inverse Simpson 1/D -- the "effective number of species"."""
    c = as_counts(counts)
    if c.size == 0:
        return 0.0
    p = c / c.sum()
    d = np.sum(p ** 2)
    return float(1.0 / d) if d > 0 else 0.0


def pielou_evenness(counts: Sequence[float]) -> float:
    """Pielou's evenness J' = H' / ln(S), in [0, 1].  1.0 == perfectly even."""
    c = as_counts(counts)
    s = c.size
    if s <= 1:
        return 1.0 if s == 1 else 0.0
    return float(shannon_index(c) / math.log(s))


def chao1(counts: Sequence[float]) -> float:
    """Chao1 estimator of *true* richness (observed + estimated unseen taxa).

    Uses the bias-corrected formula, which is well-defined even when there are
    no doubletons:

        S_chao1 = S_obs + F1 * (F1 - 1) / (2 * (F2 + 1))

    where F1 = singletons (taxa seen once) and F2 = doubletons (seen twice).
    """
    c = as_counts(counts).astype(int)
    if c.size == 0:
        return 0.0
    s_obs = c.size
    f1 = int(np.count_nonzero(c == 1))
    f2 = int(np.count_nonzero(c == 2))
    return float(s_obs + (f1 * (f1 - 1)) / (2.0 * (f2 + 1)))


def goods_coverage(counts: Sequence[float]) -> float:
    """Good's coverage estimate C = 1 - F1 / N -- fraction of the community sampled."""
    c = as_counts(counts).astype(int)
    n = c.sum()
    if n == 0:
        return 0.0
    f1 = int(np.count_nonzero(c == 1))
    return float(1.0 - f1 / n)


def margalef_richness(counts: Sequence[float]) -> float:
    """Margalef's index D = (S - 1) / ln(N) -- richness adjusted for sample size."""
    c = as_counts(counts)
    s = c.size
    n = c.sum()
    if n <= 1 or s <= 1:
        return 0.0
    return float((s - 1) / math.log(n))


def berger_parker_dominance(counts: Sequence[float]) -> float:
    """Berger-Parker dominance: relative abundance of the most abundant taxon."""
    c = as_counts(counts)
    if c.size == 0:
        return 0.0
    return float(c.max() / c.sum())


def rank_abundance(counts: Sequence[float]) -> list[dict]:
    """Sorted (descending) abundances with rank and relative abundance.

    Useful for plotting a Whittaker rank-abundance curve in the dashboard.
    """
    c = np.sort(as_counts(counts))[::-1]
    total = c.sum() if c.size else 1.0
    return [
        {"rank": i + 1, "abundance": int(v), "relative_abundance": float(v / total)}
        for i, v in enumerate(c)
    ]


def rarefaction_curve(
    counts: Sequence[float], n_points: int = 25, random_state: int = 42
) -> list[dict]:
    """Expected number of taxa observed as sampling depth increases.

    Uses the *analytical* (Hurlbert) rarefaction expectation rather than slow
    Monte-Carlo resampling, so it is exact and fast:

        E[S_n] = sum_i ( 1 - C(N - N_i, n) / C(N, n) )

    Returns a list of ``{"reads": n, "expected_species": E}`` sample points.
    """
    c = as_counts(counts).astype(int)
    n_total = int(c.sum())
    s_obs = c.size
    if n_total == 0:
        return []

    # log-gamma based log binomial for numerical stability with large N
    def log_choose(n: int, k: int) -> float:
        if k < 0 or k > n:
            return -math.inf
        return (
            math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        )

    depths = np.unique(
        np.linspace(1, n_total, num=min(n_points, n_total), dtype=int)
    )
    log_choose_total = {n: log_choose(n_total, n) for n in depths}

    curve = []
    for n in depths:
        lc_total = log_choose_total[int(n)]
        expected = 0.0
        for ni in c:
            remaining = n_total - int(ni)
            if remaining < n:
                # taxon is guaranteed to be sampled at this depth
                expected += 1.0
            else:
                # probability taxon i is *missed* at depth n
                p_absent = math.exp(log_choose(remaining, int(n)) - lc_total)
                expected += 1.0 - p_absent
        curve.append({"reads": int(n), "expected_species": round(expected, 3)})
    # anchor the final point at the observed richness
    if curve:
        curve[-1]["expected_species"] = float(s_obs)
    return curve


def compute_all(counts: Sequence[float]) -> dict:
    """Compute every metric and return a JSON-serialisable summary dict."""
    c = as_counts(counts)
    return {
        "species_richness": species_richness(c),
        "total_reads": int(c.sum()),
        "shannon_index": round(shannon_index(c), 4),
        "shannon_index_log2": round(shannon_index(c, base=2), 4),
        "simpson_index": round(simpson_index(c), 4),
        "inverse_simpson": round(inverse_simpson(c), 4),
        "pielou_evenness": round(pielou_evenness(c), 4),
        "chao1_estimated_richness": round(chao1(c), 2),
        "goods_coverage": round(goods_coverage(c), 4),
        "margalef_richness": round(margalef_richness(c), 4),
        "berger_parker_dominance": round(berger_parker_dominance(c), 4),
    }


if __name__ == "__main__":
    # Quick self-demonstration on a toy community.
    demo = [50, 30, 12, 5, 2, 1, 1]
    import json

    print("Toy abundances:", demo)
    print(json.dumps(compute_all(demo), indent=2))
    print("\nRank-abundance:")
    print(json.dumps(rank_abundance(demo), indent=2))
    print("\nRarefaction (5 points):")
    print(json.dumps(rarefaction_curve(demo, n_points=5), indent=2))
