"""
io_utils.py
===========

Input/output helpers that move Sea-quenced from a synthetic-only prototype
towards real-world eDNA datasets.

* FASTA reading/writing so the pipeline can ingest real amplicon sequences
  (e.g. exported from DADA2 / QIIME2 or downloaded from NCBI SRA).
* A small curated reference database of real deep-sea eukaryotic taxa, used by
  the classifier to give clusters believable taxonomic names instead of the old
  hard-coded ``{0: 'Bathynomus giganteus', ...}`` dict.
"""
from __future__ import annotations

import os
from typing import Iterator

import pandas as pd

__all__ = [
    "REFERENCE_TAXA",
    "read_fasta",
    "write_fasta",
    "load_asv_table",
]

# ---------------------------------------------------------------------------
# Curated reference taxa
# ---------------------------------------------------------------------------
# A compact, realistic reference list of deep-sea / pelagic eukaryotic taxa.
# In a production system this would be a BLAST/SILVA/PR2 database; here it gives
# the conceptual classifier real names with phylum-level lineage so the report
# reads like genuine biodiversity output.
REFERENCE_TAXA: list[dict] = [
    {"name": "Bathynomus giganteus",       "phylum": "Arthropoda",     "group": "Giant isopod"},
    {"name": "Halobates micans",           "phylum": "Arthropoda",     "group": "Sea skater"},
    {"name": "Riftia pachyptila",          "phylum": "Annelida",       "group": "Giant tube worm"},
    {"name": "Grimpoteuthis bathynectes",  "phylum": "Mollusca",       "group": "Dumbo octopus"},
    {"name": "Vampyroteuthis infernalis",  "phylum": "Mollusca",       "group": "Vampire squid"},
    {"name": "Bathymodiolus thermophilus", "phylum": "Mollusca",       "group": "Deep-sea mussel"},
    {"name": "Munidopsis subsquamosa",     "phylum": "Arthropoda",     "group": "Squat lobster"},
    {"name": "Caulophryne jordani",        "phylum": "Chordata",       "group": "Fanfin anglerfish"},
    {"name": "Macrourus berglax",          "phylum": "Chordata",       "group": "Roughhead grenadier"},
    {"name": "Anoplogaster cornuta",       "phylum": "Chordata",       "group": "Fangtooth"},
    {"name": "Atolla wyvillei",            "phylum": "Cnidaria",       "group": "Crown jellyfish"},
    {"name": "Periphylla periphylla",      "phylum": "Cnidaria",       "group": "Helmet jellyfish"},
    {"name": "Praya dubia",                "phylum": "Cnidaria",       "group": "Giant siphonophore"},
    {"name": "Pheronema carpenteri",       "phylum": "Porifera",       "group": "Glass sponge"},
    {"name": "Freyastera benthophila",     "phylum": "Echinodermata",  "group": "Deep-sea starfish"},
    {"name": "Scotoplanes globosa",        "phylum": "Echinodermata",  "group": "Sea pig"},
    {"name": "Enypniastes eximia",         "phylum": "Echinodermata",  "group": "Swimming sea cucumber"},
    {"name": "Radiolaria spumellaria",     "phylum": "Retaria",        "group": "Radiolarian"},
    {"name": "Globigerina bulloides",      "phylum": "Foraminifera",   "group": "Planktonic foram"},
    {"name": "Calanus hyperboreus",        "phylum": "Arthropoda",     "group": "Copepod"},
]


# ---------------------------------------------------------------------------
# FASTA I/O
# ---------------------------------------------------------------------------
def read_fasta(path: str) -> Iterator[tuple[str, str]]:
    """Yield ``(header, sequence)`` pairs from a FASTA file.

    Minimal, dependency-free parser (no Biopython required).  Sequences spread
    over multiple lines are concatenated; whitespace is stripped.
    """
    header: str | None = None
    chunks: list[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            yield header, "".join(chunks)


def write_fasta(asv_data: pd.DataFrame, path: str,
                id_col: str = "asv_id", seq_col: str = "asv_sequence",
                wrap: int = 70) -> str:
    """Write an ASV table to a FASTA file (one record per ASV)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for _, row in asv_data.iterrows():
            fh.write(f">{row[id_col]}\n")
            seq = str(row[seq_col])
            if wrap and wrap > 0:
                for i in range(0, len(seq), wrap):
                    fh.write(seq[i:i + wrap] + "\n")
            else:
                fh.write(seq + "\n")
    return path


def load_asv_table(path: str) -> pd.DataFrame:
    """Load ASVs from FASTA or CSV into the canonical pipeline DataFrame.

    Returns columns ``asv_id``, ``asv_sequence`` and (if present) ``original_label``.
    A CSV is accepted if it contains a sequence column named one of
    ``asv_sequence`` / ``Sequence`` / ``sequence``.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in {".fasta", ".fa", ".fna", ".fas"}:
        records = list(read_fasta(path))
        df = pd.DataFrame(records, columns=["asv_id", "asv_sequence"])
        return df

    # CSV / TSV
    sep = "\t" if ext in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep)
    # normalise the sequence column name
    seq_aliases = {"asv_sequence", "Sequence", "sequence", "seq", "Seq"}
    seq_col = next((c for c in df.columns if c in seq_aliases), None)
    if seq_col is None:
        raise ValueError(
            f"No sequence column found in {path}. "
            f"Expected one of {sorted(seq_aliases)}; got {list(df.columns)}."
        )
    df = df.rename(columns={seq_col: "asv_sequence"})
    id_aliases = {"asv_id", "ASV_id", "ASV_ID", "id", "ID"}
    id_col = next((c for c in df.columns if c in id_aliases), None)
    if id_col is None:
        df.insert(0, "asv_id", [f"ASV_{i + 1}" for i in range(len(df))])
    else:
        df = df.rename(columns={id_col: "asv_id"})
    return df


if __name__ == "__main__":
    print(f"Reference database contains {len(REFERENCE_TAXA)} curated deep-sea taxa:")
    for t in REFERENCE_TAXA:
        print(f"  - {t['name']:32s} ({t['phylum']}, {t['group']})")
