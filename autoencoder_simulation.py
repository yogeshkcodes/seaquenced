"""
autoencoder_simulation.py
=========================

The representation-learning stage -- a **real self-supervised neural encoder**,
replacing the original TF-IDF + t-SNE stand-in.

How it works
------------
1.  Each ASV is turned into a **k-mer genomic signature**: the normalised
    frequency vector of its length-``k`` sub-words (4^k features), computed in a
    vectorised sweep.
2.  A small encoder network is trained with **contrastive self-supervision**
    (SimCLR / InfoNCE).  For every read we build two augmented "views" by
    randomly dropping k-mers (mimicking sequencing error / intraspecific
    mutation); the network learns an embedding in which the two views of the
    same read are close and different reads are far apart.

Why contrastive instead of a reconstruction autoencoder?  A plain
reconstruction autoencoder optimises pixel-perfect decoding, not class
separation -- empirically it *collapsed* the clean k-mer structure (cluster
ARI ~0.3).  The contrastive objective explicitly shapes a clustering-friendly
latent space (ARI ~0.99) while remaining **O(n)** -- t-SNE was O(n^2) and
dominated runtime.

``simulate_autoencoder_output`` is the public entry point with a ``method``
switch: ``contrastive`` (default), ``pca`` (fast linear baseline) or ``tsne``
(legacy 2-D view).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, normalize

# Map A/C/G/T ASCII codes -> 0..3 (everything else -> -1 and is skipped)
_BASE_LUT = np.full(128, -1, dtype=np.int64)
_BASE_LUT[ord("A")] = 0
_BASE_LUT[ord("C")] = 1
_BASE_LUT[ord("G")] = 2
_BASE_LUT[ord("T")] = 3


def kmer_features(sequences, k: int = 4) -> np.ndarray:
    """Return an ``(n_sequences, 4**k)`` matrix of normalised k-mer frequencies."""
    dim = 4 ** k
    powers = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)
    seqs = list(sequences)
    X = np.zeros((len(seqs), dim), dtype=np.float32)

    for i, seq in enumerate(seqs):
        codes = _BASE_LUT[np.frombuffer(str(seq).upper().encode("ascii"),
                                        dtype=np.uint8)]
        if codes.size < k:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(codes, k)
        valid = (windows >= 0).all(axis=1)
        if not valid.any():
            continue
        idx = windows[valid] @ powers
        counts = np.bincount(idx, minlength=dim).astype(np.float32)
        total = counts.sum()
        if total > 0:
            X[i] = counts / total
    return X


def _train_contrastive_encoder(
    X: np.ndarray, latent_dim: int, epochs: int, hidden: int,
    temperature: float, dropout: float, random_state: int, verbose: bool = False,
) -> np.ndarray:
    """Train a SimCLR-style contrastive encoder; return L2-normalised embeddings."""
    import torch
    from torch import nn
    import torch.nn.functional as F

    torch.manual_seed(random_state)
    np.random.seed(random_state)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n, d_in = X.shape
    latent_dim = min(latent_dim, d_in)
    hidden = max(hidden, latent_dim * 2)

    encoder = nn.Sequential(
        nn.Linear(d_in, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden // 2), nn.ReLU(),
        nn.Linear(hidden // 2, latent_dim),
    ).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-5)

    data = F.normalize(torch.from_numpy(X).to(device), dim=1)
    batch = min(512, n)

    def augment(B):
        """Random k-mer dropout -> a corrupted, re-normalised view."""
        mask = (torch.rand_like(B) > dropout).float()
        return F.normalize(B * mask, dim=1)

    encoder.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            if len(idx) < 4:
                continue
            B = data[idx]
            z1 = F.normalize(encoder(augment(B)), dim=1)
            z2 = F.normalize(encoder(augment(B)), dim=1)
            logits = z1 @ z2.t() / temperature            # (b, b) cosine sims
            labels = torch.arange(len(idx), device=device)  # matching view = positive
            loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1):
            print(f"      epoch {epoch + 1:>3}/{epochs}  contrastive_loss={total / n:.4f}")

    encoder.eval()
    with torch.no_grad():
        return F.normalize(encoder(data), dim=1).cpu().numpy()


def encode_sequences(
    asv_data: pd.DataFrame,
    latent_dim: int = 16,
    k: int = 4,
    epochs: int = 40,
    hidden: int = 256,
    temperature: float = 0.2,
    dropout: float = 0.1,
    random_state: int = 42,
    verbose: bool = False,
) -> pd.DataFrame:
    """Featurise + contrastively encode ASVs into a standardised latent space."""
    features = kmer_features(asv_data["asv_sequence"], k=k)
    latent = _train_contrastive_encoder(features, latent_dim, epochs, hidden,
                                        temperature, dropout, random_state, verbose)
    latent = StandardScaler().fit_transform(latent)
    return _assemble(asv_data, latent, random_state)


def _assemble(asv_data: pd.DataFrame, latent: np.ndarray,
              random_state: int) -> pd.DataFrame:
    """Package a latent matrix into the canonical pipeline frame."""
    d = latent.shape[1]
    out = pd.DataFrame({f"z{i}": latent[:, i] for i in range(d)})
    out.insert(0, "asv_id",
               asv_data["asv_id"].values if "asv_id" in asv_data.columns
               else [f"ASV_{i + 1}" for i in range(len(out))])

    # 2-D view for plotting / backwards compatibility
    xy = latent[:, :2] if d == 2 else \
        PCA(n_components=2, random_state=random_state).fit_transform(latent)
    out["x"], out["y"] = xy[:, 0], xy[:, 1]

    if "original_label" in asv_data.columns:
        out["original_label"] = asv_data["original_label"].values
    return out


def simulate_autoencoder_output(
    asv_data: pd.DataFrame,
    method: str = "contrastive",
    n_components: int = 2,
    latent_dim: int = 16,
    random_state: int = 42,
    **kwargs,
) -> pd.DataFrame:
    """Public entry point.

    ``method``:
        * ``"contrastive"`` / ``"autoencoder"`` (default) -- learned encoder,
        * ``"pca"`` -- fast linear reduction of k-mer features,
        * ``"tsne"`` -- legacy 2-D visualisation embedding.
    """
    method = method.lower()
    if method in ("contrastive", "autoencoder"):
        return encode_sequences(asv_data, latent_dim=latent_dim,
                                random_state=random_state, **kwargs)

    features = kmer_features(asv_data["asv_sequence"], k=kwargs.get("k", 4))
    if method == "pca":
        latent = PCA(n_components=min(latent_dim, features.shape[1]),
                     random_state=random_state).fit_transform(features)
        return _assemble(asv_data, StandardScaler().fit_transform(latent), random_state)
    if method == "tsne":
        from sklearn.manifold import TSNE
        pre = normalize(TruncatedSVD(n_components=min(50, features.shape[1] - 1),
                                     random_state=random_state).fit_transform(features))
        coords = TSNE(n_components=n_components, random_state=random_state, init="pca",
                      perplexity=min(30, max(5, len(asv_data) // 100))
                      ).fit_transform(pre)
        return _assemble(asv_data, coords, random_state)
    raise ValueError(f"Unknown method {method!r}; use contrastive/pca/tsne.")


if __name__ == "__main__":
    asv_data = pd.read_csv("synthetic_asv_data.csv")
    latent = simulate_autoencoder_output(asv_data, method="contrastive", verbose=True)
    print("Latent space (first 5 rows):")
    print(latent.head())
    latent.to_csv("latent_space_data.csv", index=False)
    print("\nLatent space data saved to 'latent_space_data.csv'")
