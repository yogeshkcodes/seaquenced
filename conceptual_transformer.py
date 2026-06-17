"""
conceptual_transformer.py
=========================

Classification & reporting stage.

Upgraded from the original hard-coded ``{0: 'Bathynomus giganteus', ...}`` dict
(which mislabelled real species as "Noise" and faked the Shannon index) to a
principled **Gaussian novelty detector**:

1.  Build a reference database where every known taxon is modelled as a Gaussian
    in the autoencoder's latent space (mean + regularised covariance).
2.  Classify each discovered cluster by its **Mahalanobis distance** to the
    nearest reference Gaussian.  Mahalanobis (not raw Euclidean) accounts for
    the shape/spread of each taxon, so the known/novel decision uses a
    statistically meaningful cut-off:

        d_M <= sqrt(chi2_{0.975, latent_dim})   -> Known taxon (named)
        otherwise                               -> Novel candidate (discovery)

    The distance also yields a calibrated confidence and a ``novelty_score``.
3.  Report **real** ecological diversity metrics from cluster abundances.

``run_conceptual_transformer`` is preserved as a backwards-compatible wrapper.
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd

import biodiversity_metrics as bm
from io_utils import REFERENCE_TAXA
from run_clustering import latent_columns

try:
    from scipy.stats import chi2 as _chi2
except Exception:  # pragma: no cover
    _chi2 = None


def _reg_inv_cov(X: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Inverse covariance with ridge + shrinkage, robust to tiny/singular groups."""
    d = X.shape[1]
    if len(X) < 2:
        return np.eye(d)
    cov = np.cov(X, rowvar=False)
    cov = np.atleast_2d(cov)
    # Ledoit-Wolf style shrinkage towards a scaled identity
    shrink = 0.1
    target = np.trace(cov) / d * np.eye(d)
    cov = (1 - shrink) * cov + shrink * target + ridge * np.eye(d)
    try:
        return np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(cov)


def build_reference_db(clustered_data: pd.DataFrame) -> list[dict]:
    """Model each 'Known Species' as a Gaussian (mean + inverse covariance).

    Returns ``[]`` when no ground-truth labels are present (e.g. real FASTA
    input), in which case every cluster is treated as a novel candidate.
    """
    if "original_label" not in clustered_data.columns:
        return []
    known = clustered_data[clustered_data["original_label"]
                           .astype(str).str.startswith("Known Species")]
    if known.empty:
        return []

    cols = latent_columns(clustered_data)
    refs: list[dict] = []
    for i, (_, grp) in enumerate(known.groupby("original_label")):
        taxon = REFERENCE_TAXA[i % len(REFERENCE_TAXA)]
        Z = grp[cols].to_numpy()
        refs.append({
            "name": taxon["name"], "phylum": taxon["phylum"], "group": taxon["group"],
            "mean": Z.mean(axis=0),
            "inv_cov": _reg_inv_cov(Z),
            "x": float(grp["x"].mean()), "y": float(grp["y"].mean()),
        })
    return refs


def _mahalanobis(point: np.ndarray, ref: dict) -> float:
    delta = point - ref["mean"]
    return float(np.sqrt(max(0.0, delta @ ref["inv_cov"] @ delta)))


def classify_and_report(
    clustered_data: pd.DataFrame,
    classify_threshold: Optional[float] = None,
    quality: Optional[dict] = None,
    params: Optional[dict] = None,
) -> dict:
    """Classify clusters against the reference DB and build a biodiversity report."""
    data = clustered_data.copy()
    cols = latent_columns(data)
    d = len(cols)
    refs = build_reference_db(data)

    # statistical novelty cut-off: 97.5th percentile of the chi distribution
    if classify_threshold is not None:
        threshold = classify_threshold
    elif _chi2 is not None:
        threshold = float(np.sqrt(_chi2.ppf(0.975, df=d)))
    else:
        threshold = float(np.sqrt(d) + 2.0)

    centroids_latent = data.groupby("cluster_id")[cols].mean()
    centroids_xy = data.groupby("cluster_id")[["x", "y"]].mean()
    sizes = data.groupby("cluster_id").size()
    has_prob = "cluster_probability" in data.columns

    novel_counter = 0
    cluster_meta: dict[int, dict] = {}
    for cid in sorted(sizes.index, key=lambda c: -sizes[c]):
        if cid == -1:
            cluster_meta[cid] = {
                "name": "Noise", "status": "Noise", "confidence": None,
                "lineage": "Unassigned / low-density reads",
                "nearest_reference": None, "mahalanobis": None, "novelty_score": None,
            }
            continue

        point = centroids_latent.loc[cid].to_numpy()
        if refs:
            dists = [_mahalanobis(point, r) for r in refs]
            j = int(np.argmin(dists))
            dist, nearest = dists[j], refs[j]
        else:
            dist, nearest = np.inf, None

        if nearest is not None and dist <= threshold:
            similarity = max(0.0, 1.0 - dist / threshold) if threshold > 0 else 1.0
            cluster_meta[cid] = {
                "name": nearest["name"], "status": "Known",
                "confidence": round(0.70 + 0.29 * similarity, 4),
                "lineage": f"Phylum {nearest['phylum']} ({nearest['group']})",
                "nearest_reference": nearest["name"],
                "mahalanobis": round(dist, 4),
                "novelty_score": round(dist / threshold, 3),
            }
        else:
            novel_counter += 1
            closest_phylum = nearest["phylum"] if nearest else "unknown"
            cluster_meta[cid] = {
                "name": f"Candidate Taxon {novel_counter}", "status": "Novel",
                "confidence": None,
                "lineage": f"No reference match; closest lineage: Phylum {closest_phylum}",
                "nearest_reference": nearest["name"] if nearest else None,
                "mahalanobis": round(dist, 4) if np.isfinite(dist) else None,
                "novelty_score": round(dist / threshold, 3) if np.isfinite(dist) else None,
            }

    data["assigned_name"] = data["cluster_id"].map(lambda c: cluster_meta[c]["name"])
    data["status"] = data["cluster_id"].map(lambda c: cluster_meta[c]["status"])

    taxa = sizes.drop(index=-1, errors="ignore")
    abundances = taxa.to_numpy()
    known_clusters = [c for c, m in cluster_meta.items() if m["status"] == "Known"]
    novel_clusters = [c for c, m in cluster_meta.items() if m["status"] == "Novel"]

    total_reads = int(len(data))
    noise_reads = int((data["cluster_id"] == -1).sum())
    known_reads = int(sizes.loc[known_clusters].sum()) if known_clusters else 0
    novel_reads = int(sizes.loc[novel_clusters].sum()) if novel_clusters else 0

    profile = []
    for cid in sorted(sizes.index):
        m = cluster_meta[cid]
        cx, cy = (float(centroids_xy.loc[cid, "x"]), float(centroids_xy.loc[cid, "y"])) \
            if cid in centroids_xy.index else (None, None)
        entry = {
            "cluster_id": int(cid), "name": m["name"], "status": m["status"],
            "read_count": int(sizes[cid]),
            "relative_abundance": round(sizes[cid] / total_reads, 5),
            "confidence": m["confidence"], "lineage": m["lineage"],
            "nearest_reference": m["nearest_reference"],
            "mahalanobis_distance": m["mahalanobis"],
            "novelty_score": m["novelty_score"],
            "centroid": {"x": cx, "y": cy},
        }
        if has_prob:
            entry["mean_cluster_probability"] = round(
                float(data.loc[data["cluster_id"] == cid, "cluster_probability"].mean()), 4)
        profile.append(entry)
    profile.sort(key=lambda r: (r["status"] == "Noise", -r["read_count"]))

    phylum_counts: dict[str, int] = {}
    for cid in known_clusters:
        phylum = cluster_meta[cid]["lineage"].replace("Phylum ", "").split(" (")[0]
        phylum_counts[phylum] = phylum_counts.get(phylum, 0) + int(sizes[cid])

    report = {
        "summary": {
            "total_asvs_processed": total_reads,
            "candidate_taxa_total": int(len(taxa)),
            "known_species_count": len(known_clusters),
            "novel_taxa_count": len(novel_clusters),
            "noise_reads_count": noise_reads,
        },
        "composition_reads": {"known": known_reads, "novel": novel_reads,
                              "noise": noise_reads},
        "biodiversity_metrics": bm.compute_all(abundances),
        "phylum_breakdown": [{"phylum": k, "read_count": v}
                             for k, v in sorted(phylum_counts.items(),
                                                key=lambda kv: -kv[1])],
        "rank_abundance": bm.rank_abundance(abundances),
        "rarefaction_curve": bm.rarefaction_curve(abundances, n_points=30),
        "reference_database": [{"name": r["name"], "phylum": r["phylum"],
                                "group": r["group"]} for r in refs],
        "taxonomic_profile": profile,
        "novelty_detection": {
            "latent_dim": d,
            "mahalanobis_threshold": round(threshold, 4),
            "rule": "chi-squared 97.5% cut-off on latent Mahalanobis distance",
        },
    }
    if quality:
        report["clustering_quality"] = quality
    if params:
        report["parameters"] = params
    return report


def run_conceptual_transformer(clustered_data: pd.DataFrame) -> dict:
    """Backwards-compatible entry point returning the biodiversity report."""
    return classify_and_report(clustered_data)


if __name__ == "__main__":
    clustered_data = pd.read_json("cluster_results.json")
    final_report = classify_and_report(clustered_data)
    with open("biodiversity_report.json", "w") as f:
        json.dump(final_report, f, indent=2)
    s = final_report["summary"]
    print("Final biodiversity report saved to 'biodiversity_report.json'")
    print(f"  Known species : {s['known_species_count']}")
    print(f"  Novel taxa    : {s['novel_taxa_count']}")
    print(f"  Shannon index : {final_report['biodiversity_metrics']['shannon_index']}")
