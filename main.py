"""
main.py
=======

Sea-quenced pipeline orchestrator with a real command-line interface.

The pipeline:
    1. featurise + autoencode ASV sequences into a learned latent space,
    2. discover candidate taxa with density clustering (HDBSCAN by default),
    3. classify clusters against a reference DB via Mahalanobis novelty
       detection and compute real biodiversity metrics.

Examples
--------
Full synthetic demo (learned autoencoder + HDBSCAN)::

    python main.py

Analyse a real FASTA of ASVs::

    python main.py --input my_asvs.fasta

Fast smoke test::

    python main.py --quick
"""
from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd

from generate_asv_data import generate_asv_data
from autoencoder_simulation import simulate_autoencoder_output
from run_clustering import (run_clustering, clustering_quality,
                            plot_clusters, generate_dendrogram, estimate_eps)
from conceptual_transformer import classify_and_report


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="sea-quenced",
        description="AI pipeline for deep-sea eDNA biodiversity assessment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    io = p.add_argument_group("input / output")
    io.add_argument("--input", help="FASTA or CSV of ASV sequences. "
                    "If omitted, a synthetic dataset is generated.")
    io.add_argument("--output-dir", default="data_output",
                    help="Directory for all generated artefacts.")
    io.add_argument("--no-plots", action="store_true",
                    help="Skip writing the PNG figures.")

    emb = p.add_argument_group("representation (self-supervised encoder)")
    emb.add_argument("--method", choices=["contrastive", "pca", "tsne"],
                     default="contrastive", help="Latent-space embedding method.")
    emb.add_argument("--latent-dim", type=int, default=16,
                     help="Encoder latent dimensionality.")
    emb.add_argument("--kmer", type=int, default=4, help="k-mer size for features.")
    emb.add_argument("--epochs", type=int, default=40, help="Autoencoder epochs.")

    clu = p.add_argument_group("clustering")
    clu.add_argument("--algorithm", choices=["hdbscan", "dbscan", "kmeans"],
                     default="hdbscan")
    clu.add_argument("--min-cluster-size", type=int, default=15,
                     help="HDBSCAN minimum cluster size.")
    clu.add_argument("--min-samples", type=int, default=10)
    clu.add_argument("--eps", type=float, default=None,
                     help="DBSCAN eps (auto-selected from the data if omitted).")
    clu.add_argument("--eps-percentile", type=float, default=90.0)
    clu.add_argument("--n-clusters", type=int, default=12, help="KMeans cluster count.")
    clu.add_argument("--classify-threshold", type=float, default=None,
                     help="Override the Mahalanobis known/novel threshold.")

    syn = p.add_argument_group("synthetic data (when --input is not given)")
    syn.add_argument("--known-species", type=int, default=300)
    syn.add_argument("--novel-species", type=int, default=200)
    syn.add_argument("--known-asvs", type=int, default=100)
    syn.add_argument("--novel-asvs", type=int, default=50)
    syn.add_argument("--abundance-skew", type=float, default=0.4)

    p.add_argument("--quick", action="store_true",
                   help="Tiny, fast configuration for smoke tests.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Print autoencoder training progress.")
    return p.parse_args(argv)


def _log(step: str, msg: str) -> None:
    print(f"  [{step}] {msg}")


def run_pipeline(args: argparse.Namespace) -> dict:
    os.makedirs(args.output_dir, exist_ok=True)
    out = lambda name: os.path.join(args.output_dir, name)
    timings: dict[str, float] = {}
    t0 = time.time()

    print("\n=== Sea-quenced pipeline ===")

    # ---- Step 1: data ----
    if args.input:
        from io_utils import load_asv_table
        _log("1/4", f"Loading ASVs from {args.input} ...")
        asv_data = load_asv_table(args.input)
    else:
        if args.quick:
            args.known_species, args.novel_species = 20, 15
            args.known_asvs, args.novel_asvs = 40, 25
            args.epochs = min(args.epochs, 25)
        _log("1/4", "Generating synthetic ASV data ...")
        asv_data = generate_asv_data(
            n_known_species=args.known_species, n_novel_species=args.novel_species,
            known_asvs_per_species=args.known_asvs,
            novel_asvs_per_species=args.novel_asvs,
            abundance_skew=args.abundance_skew, random_state=args.seed)
    asv_data.to_csv(out("synthetic_asv_data.csv"), index=False)
    _log("1/4", f"{len(asv_data):,} ASVs ready.")

    # ---- Step 2: representation learning ----
    _log("2/4", f"Embedding sequences ({args.method}"
                + (f", latent_dim={args.latent_dim}" if args.method == "contrastive" else "")
                + ") ...")
    t = time.time()
    latent = simulate_autoencoder_output(
        asv_data, method=args.method, latent_dim=args.latent_dim,
        k=args.kmer, epochs=args.epochs, random_state=args.seed,
        verbose=args.verbose)
    timings["embedding_s"] = round(time.time() - t, 2)
    latent.to_csv(out("latent_space_data.csv"), index=False)
    _log("2/4", f"latent space ready in {timings['embedding_s']}s")

    # ---- Step 3: clustering ----
    _log("3/4", f"Discovering taxa ({args.algorithm}) ...")
    t = time.time()
    resolved_eps = args.eps
    if args.algorithm == "dbscan" and resolved_eps is None:
        from run_clustering import latent_columns
        resolved_eps = estimate_eps(latent[latent_columns(latent)].to_numpy(),
                                    args.min_samples, args.eps_percentile)
        _log("3/4", f"auto-selected eps={resolved_eps:.3f}")
    clustered = run_clustering(latent, algorithm=args.algorithm, eps=resolved_eps,
                               min_samples=args.min_samples,
                               min_cluster_size=args.min_cluster_size,
                               n_clusters=args.n_clusters, random_state=args.seed)
    timings["clustering_s"] = round(time.time() - t, 2)
    quality = clustering_quality(clustered)
    clustered.to_json(out("cluster_results.json"), orient="records")
    if not args.no_plots:
        plot_clusters(clustered, out("cluster_plot.png"))
        generate_dendrogram(clustered, out("dendrogram.png"))
    _log("3/4", f"{quality['n_clusters']} taxa, silhouette={quality['silhouette_score']}, "
                f"noise={quality['noise_fraction']:.1%} ({timings['clustering_s']}s)")

    # ---- Step 4: classification + report ----
    _log("4/4", "Classifying (Mahalanobis novelty) & computing metrics ...")
    params = {
        "method": args.method, "latent_dim": args.latent_dim, "kmer": args.kmer,
        "algorithm": args.algorithm, "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "eps": round(resolved_eps, 4) if resolved_eps is not None else None,
        "n_clusters": args.n_clusters, "seed": args.seed,
        "input": args.input or "synthetic", "timings": timings,
    }
    report = classify_and_report(clustered, classify_threshold=args.classify_threshold,
                                 quality=quality, params=params)
    with open(out("biodiversity_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    s, met = report["summary"], report["biodiversity_metrics"]
    _log("4/4", f"Known={s['known_species_count']}  Novel={s['novel_taxa_count']}  "
                f"Shannon={met['shannon_index']}  Chao1={met['chao1_estimated_richness']}")

    print(f"\nDone in {time.time() - t0:.1f}s. Artefacts in '{args.output_dir}/'.")
    print(f"  - biodiversity_report.json   (metrics, taxonomy, novelty)")
    print(f"  - latent_space_data.csv      (learned embedding)")
    print(f"  - cluster_results.json       (per-read cluster assignment)")
    if not args.no_plots:
        print(f"  - cluster_plot.png / dendrogram.png\n")
    return report


if __name__ == "__main__":
    run_pipeline(parse_args())
