"""
run_clustering.py
=================

Unsupervised discovery stage.  Clusters the latent-space embeddings so that
groups of near-identical ASVs collapse into candidate taxa (OTUs) -- including
taxa that exist in no reference database.

Improvements over the original prototype:
* a single ``run_clustering`` entry point supporting both **DBSCAN** (density
  based, finds noise) and **KMeans** (when you want a fixed number of taxa),
* a **silhouette score** so cluster quality is quantified rather than assumed,
* helper plots (cluster scatter + dendrogram) saved as PNGs,
* ``asv_id`` and ``original_label`` are preserved end-to-end.

``run_dbscan_clustering`` is kept as a thin wrapper for backwards compatibility.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def latent_columns(df: pd.DataFrame) -> list[str]:
    """Return the latent feature columns (``z0..zk``) if present, else ``x``/``y``."""
    zcols = sorted([c for c in df.columns if c.startswith("z") and c[1:].isdigit()],
                   key=lambda c: int(c[1:]))
    return zcols if zcols else ["x", "y"]


def estimate_eps(coords, min_samples: int = 10, percentile: float = 90.0) -> float:
    """Pick a DBSCAN ``eps`` from the data via the k-distance heuristic.

    For every point we measure the distance to its ``min_samples``-th nearest
    neighbour; ``eps`` is then a high percentile of that distribution (the
    classic "knee" of the sorted k-distance curve).  This makes clustering
    robust to the latent space's absolute scale, which varies with dataset size
    and embedding method -- so the same defaults work on 1k or 1M reads.
    """
    coords = np.asarray(coords)
    k = max(2, min(min_samples, len(coords) - 1))
    if len(coords) <= k:
        return 1.0
    nn = NearestNeighbors(n_neighbors=k).fit(coords)
    dist, _ = nn.kneighbors(coords)
    return float(np.percentile(dist[:, -1], percentile))


def run_clustering(
    latent_space_data: pd.DataFrame,
    algorithm: str = "hdbscan",
    eps: float | None = None,
    min_samples: int = 10,
    min_cluster_size: int = 15,
    n_clusters: int = 8,
    eps_percentile: float = 90.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Cluster latent-space points and return the frame with a ``cluster_id`` column.

    Clustering runs on the **full latent space** (``z0..zk``) when available,
    falling back to the 2-D view.  ``hdbscan`` (default) needs no distance
    threshold and finds variable-density taxa, exposing a per-read
    ``cluster_probability``.  For DBSCAN, ``eps=None`` auto-selects via
    :func:`estimate_eps`.
    """
    cols = latent_columns(latent_space_data)
    coords = latent_space_data[cols].to_numpy()
    algorithm = algorithm.lower()
    probs = None

    if algorithm == "hdbscan":
        model = HDBSCAN(min_cluster_size=max(2, min_cluster_size),
                        min_samples=min_samples, copy=False)
        labels = model.fit_predict(coords)
        probs = getattr(model, "probabilities_", None)
    elif algorithm == "dbscan":
        if eps is None:
            eps = estimate_eps(coords, min_samples, eps_percentile)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(coords)
    elif algorithm == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = model.fit_predict(coords)
    else:
        raise ValueError(
            f"Unknown algorithm {algorithm!r}; use 'hdbscan', 'dbscan' or 'kmeans'.")

    out = latent_space_data.copy()
    out["cluster_id"] = labels
    if probs is not None:
        out["cluster_probability"] = np.round(probs, 4)
    return out


def run_dbscan_clustering(latent_space_data, eps: float = 1.5, min_samples: int = 2):
    """Backwards-compatible DBSCAN wrapper."""
    return run_clustering(latent_space_data, algorithm="dbscan",
                          eps=eps, min_samples=min_samples)


def clustering_quality(clustered_data: pd.DataFrame) -> dict:
    """Compute silhouette score (excluding noise) and basic cluster stats."""
    non_noise = clustered_data[clustered_data["cluster_id"] != -1]
    n_clusters = int(non_noise["cluster_id"].nunique())
    n_noise = int((clustered_data["cluster_id"] == -1).sum())

    silhouette = None
    if n_clusters >= 2 and len(non_noise) > n_clusters:
        try:
            cols = latent_columns(clustered_data)
            # subsample for speed on very large datasets
            sample = non_noise
            if len(non_noise) > 10000:
                sample = non_noise.sample(10000, random_state=42)
            silhouette = float(
                silhouette_score(sample[cols], sample["cluster_id"])
            )
        except Exception:
            silhouette = None

    return {
        "n_clusters": n_clusters,
        "n_noise_points": n_noise,
        "noise_fraction": round(n_noise / max(1, len(clustered_data)), 4),
        "silhouette_score": round(silhouette, 4) if silhouette is not None else None,
    }


def plot_clusters(clustered_data: pd.DataFrame, path: str = "cluster_plot.png") -> str:
    """Save a scatter plot of the latent space coloured by cluster."""
    fig, ax = plt.subplots(figsize=(10, 8))
    noise = clustered_data[clustered_data["cluster_id"] == -1]
    signal = clustered_data[clustered_data["cluster_id"] != -1]
    if len(noise):
        ax.scatter(noise["x"], noise["y"], c="lightgray", s=6, label="Noise", alpha=0.5)
    sc = ax.scatter(signal["x"], signal["y"], c=signal["cluster_id"],
                    cmap="viridis", s=10, alpha=0.8)
    ax.set_title("Latent-space clusters (candidate taxa)")
    ax.set_xlabel("Latent dim 1")
    ax.set_ylabel("Latent dim 2")
    ax.grid(True, alpha=0.2)
    fig.colorbar(sc, ax=ax, label="cluster id")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def generate_dendrogram(latent_space_data: pd.DataFrame,
                        path: str = "dendrogram.png",
                        max_clusters: int = 40) -> str | None:
    """Hierarchical dendrogram of cluster *centroids* (clearer than per-point)."""
    clustered = latent_space_data[latent_space_data["cluster_id"] != -1]
    if clustered["cluster_id"].nunique() < 2:
        print("Not enough clusters to generate a dendrogram.")
        return None

    centroids = (
        clustered.groupby("cluster_id")[["x", "y"]].mean()
        .head(max_clusters)
    )
    linked = linkage(centroids.to_numpy(), method="ward")
    plt.figure(figsize=(12, 7))
    dendrogram(linked, orientation="top",
               labels=[f"C{c}" for c in centroids.index])
    plt.title("Hierarchical relationships between candidate taxa")
    plt.xlabel("Cluster")
    plt.ylabel("Ward distance")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


if __name__ == "__main__":
    latent_space_data = pd.read_csv("latent_space_data.csv")
    clustered_data = run_clustering(latent_space_data, eps=0.5, min_samples=10)
    print("Clusters generated (first 5 rows):")
    print(clustered_data.head())
    print("\nQuality:", clustering_quality(clustered_data))

    plot_clusters(clustered_data)
    print("Scatter plot saved to 'cluster_plot.png'")
    generate_dendrogram(clustered_data)
    print("Dendrogram saved to 'dendrogram.png'")

    cols = [c for c in ["asv_id", "x", "y", "cluster_id"] if c in clustered_data.columns]
    clustered_data[cols].to_json("cluster_results.json", orient="records")
    print("Cluster results saved to 'cluster_results.json'")
