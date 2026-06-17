"""
generate_asv_data.py
=====================

Generate a synthetic dataset of ASV (Amplicon Sequence Variant) sequences for
the Sea-quenced prototype.

Each *species* is built from a core "barcode" pattern; individual ASVs are that
pattern with a small per-read mutation rate, mimicking intraspecific variation
in a real metabarcoding run.  "Known" species are meant to resemble taxa that
exist in a reference database; "Novel" species use deliberately divergent
patterns so the unsupervised stage can discover them.

Improvements over the original prototype:
* every ASV gets a stable ``asv_id`` (needed to join results across stages and
  to export FASTA),
* configurable abundance skew so the community looks realistic (a few dominant
  taxa, a long tail of rare ones) instead of every species having identical
  read counts,
* optional FASTA export for interoperability with real tools.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd

BASE_PAIRS = ("A", "C", "G", "T")


def _mutate(pattern: list[str], rate: float, rng: random.Random,
            np_rng: np.random.Generator) -> str:
    """Return a copy of ``pattern`` with a fraction ``rate`` of bases mutated."""
    seq = list(pattern)
    n = len(seq)
    num_mutations = max(1, int(n * rate))
    idxs = np_rng.choice(n, size=min(num_mutations, n), replace=False)
    for idx in idxs:
        original = seq[idx]
        seq[idx] = rng.choice([b for b in BASE_PAIRS if b != original])
    return "".join(seq)


def generate_asv_data(
    n_known_species: int = 300,
    n_novel_species: int = 200,
    known_asvs_per_species: int = 100,
    novel_asvs_per_species: int = 50,
    asv_length: int = 250,
    mutation_rate: float = 0.01,
    abundance_skew: float = 0.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic ASV dataset.

    Parameters
    ----------
    abundance_skew:
        0.0 keeps the original fixed counts.  Values > 0 draw each species'
        read count from a log-normal distribution (sigma = ``abundance_skew``)
        scaled around the nominal per-species count, producing a realistic
        long-tailed rank-abundance curve.

    Returns a DataFrame with columns ``asv_id``, ``asv_sequence`` and
    ``original_label`` (kept for backwards compatibility / validation).
    """
    rng = random.Random(random_state)
    np_rng = np.random.default_rng(random_state)

    asv_sequences: list[str] = []
    labels: list[str] = []

    def _count_for(nominal: int) -> int:
        if abundance_skew <= 0:
            return nominal
        factor = float(np_rng.lognormal(mean=0.0, sigma=abundance_skew))
        return max(2, int(round(nominal * factor)))

    for i in range(n_known_species):
        pattern = list(rng.choices(BASE_PAIRS, k=asv_length))
        for _ in range(_count_for(known_asvs_per_species)):
            asv_sequences.append(_mutate(pattern, mutation_rate, rng, np_rng))
            labels.append(f"Known Species {i}")

    for i in range(n_novel_species):
        pattern = list(rng.choices(BASE_PAIRS, k=asv_length))
        for _ in range(_count_for(novel_asvs_per_species)):
            asv_sequences.append(_mutate(pattern, mutation_rate, rng, np_rng))
            labels.append(f"Novel Species {i}")

    df = pd.DataFrame(
        {
            "asv_id": [f"ASV_{i + 1}" for i in range(len(asv_sequences))],
            "asv_sequence": asv_sequences,
            "original_label": labels,
        }
    )
    return df


if __name__ == "__main__":
    asv_data = generate_asv_data()
    print("Generated ASV dataset (first 5 rows):")
    print(asv_data.head())
    print(f"\nTotal ASVs: {len(asv_data):,} "
          f"across {asv_data['original_label'].nunique()} species")
    asv_data.to_csv("synthetic_asv_data.csv", index=False)
    print("Dataset saved to 'synthetic_asv_data.csv'")

    try:
        from io_utils import write_fasta
        write_fasta(asv_data, "synthetic_asv_data.fasta")
        print("FASTA saved to 'synthetic_asv_data.fasta'")
    except Exception as exc:  # pragma: no cover - convenience only
        print(f"(FASTA export skipped: {exc})")
