"""Fast, torch-free tests for the Sea-quenced pipeline."""
import math
import os
import tempfile

import numpy as np
import pandas as pd

import biodiversity_metrics as bm
from autoencoder_simulation import kmer_features, simulate_autoencoder_output
from generate_asv_data import generate_asv_data
from io_utils import read_fasta, write_fasta, load_asv_table
from run_clustering import run_clustering, clustering_quality
from conceptual_transformer import classify_and_report


# ---- biodiversity metrics ----
def test_shannon_and_evenness_uniform():
    counts = [10, 10, 10, 10]
    assert math.isclose(bm.shannon_index(counts), math.log(4), rel_tol=1e-9)
    assert math.isclose(bm.pielou_evenness(counts), 1.0, rel_tol=1e-9)


def test_simpson_and_richness():
    counts = [10, 10, 10, 10]
    assert bm.species_richness(counts) == 4
    assert math.isclose(bm.simpson_index(counts), 0.75, rel_tol=1e-9)


def test_metrics_handle_empty():
    assert bm.shannon_index([]) == 0.0
    assert bm.species_richness([0, 0]) == 0


def test_chao1_accounts_for_singletons():
    # observed 3 taxa, 2 singletons, 1 doubleton -> chao1 > observed
    assert bm.chao1([5, 1, 1]) > bm.species_richness([5, 1, 1])


def test_compute_all_keys():
    out = bm.compute_all([5, 3, 2, 1])
    for key in ("shannon_index", "simpson_index", "chao1_estimated_richness",
                "pielou_evenness", "species_richness"):
        assert key in out


# ---- features ----
def test_kmer_features_are_normalised():
    X = kmer_features(["ACGTACGTACGT", "TTTTAAAACCCC"], k=3)
    assert X.shape == (2, 4 ** 3)
    assert np.allclose(X.sum(axis=1), 1.0, atol=1e-5)


# ---- IO ----
def test_fasta_roundtrip():
    df = generate_asv_data(n_known_species=2, n_novel_species=1,
                           known_asvs_per_species=3, novel_asvs_per_species=2)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.fasta")
        write_fasta(df, path)
        records = dict(read_fasta(path))
    assert len(records) == len(df)
    first = df.iloc[0]
    assert records[first["asv_id"]] == first["asv_sequence"]


def test_load_asv_table_csv():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "asv.csv")
        pd.DataFrame({"ASV_id": ["a", "b"],
                      "Sequence": ["ACGT", "TGCA"]}).to_csv(path, index=False)
        df = load_asv_table(path)
    assert list(df.columns[:2]) == ["asv_id", "asv_sequence"]
    assert len(df) == 2


# ---- end-to-end (PCA path keeps it fast / torch-free) ----
def test_end_to_end_recovers_known_and_novel():
    asv = generate_asv_data(n_known_species=6, n_novel_species=5,
                            known_asvs_per_species=30, novel_asvs_per_species=30,
                            abundance_skew=0.0, random_state=0)
    latent = simulate_autoencoder_output(asv, method="pca", latent_dim=8, random_state=0)
    clustered = run_clustering(latent, algorithm="hdbscan", min_cluster_size=10)
    quality = clustering_quality(clustered)
    report = classify_and_report(clustered, quality=quality)

    # ~11 true species should be largely recovered and split known/novel
    assert quality["n_clusters"] >= 8
    assert report["summary"]["known_species_count"] >= 1
    assert report["summary"]["novel_taxa_count"] >= 1
    assert report["biodiversity_metrics"]["shannon_index"] > 1.0
    # every known assignment names a real reference taxon
    for entry in report["taxonomic_profile"]:
        if entry["status"] == "Known":
            assert entry["nearest_reference"] is not None
            assert entry["confidence"] is not None
