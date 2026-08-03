"""Sequence space analysis (spec section 15).

Two stages:

1. Embed sequences into fixed-length vectors.
   - ``kmer``: normalized k-mer frequency vector over the canonical
     alphabet (no external dependency).
   - ``minhash``: reuses similarity.py's MinHash sketch as the vector
     itself (no external dependency).
   - ``esm``: mean-pooled ESM (protein language model) embeddings, via the
     ``fair-esm`` package + PyTorch (heavy, optional).
   - ``prott5``: mean-pooled ProtT5 embeddings, via HuggingFace
     ``transformers`` + PyTorch + sentencepiece (heavy, optional).
2. Reduce to 2-3 dimensions for visualization: PCA / t-SNE / UMAP
   (scikit-learn for the first two, umap-learn for the third -- all in the
   optional ``cluster``/``embed`` extras).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from .core import BioRecord, SeqType
from .profile import default_alphabet
from .similarity import minhash_signature

_EMBED_METHODS = ("kmer", "minhash", "esm", "prott5")
_REDUCE_METHODS = ("pca", "tsne", "umap")
_PROTEIN_ONLY_EMBED_METHODS = ("esm", "prott5")


def _require_numpy():
    import numpy as np

    return np


# -- embedding: k-mer frequency --------------------------------------------


def build_kmer_vocabulary(alphabet: list[str], k: int) -> list[str]:
    return ["".join(p) for p in itertools.product(alphabet, repeat=k)]


def kmer_frequency_vector(sequence: str, vocabulary: list[str], k: int) -> list[float]:
    index = {kmer: i for i, kmer in enumerate(vocabulary)}
    counts = [0] * len(vocabulary)
    total = 0
    for i in range(len(sequence) - k + 1):
        idx = index.get(sequence[i : i + k])
        if idx is not None:
            counts[idx] += 1
            total += 1
    if total == 0:
        return [0.0] * len(vocabulary)
    return [c / total for c in counts]


def kmer_embed(records: list[BioRecord], k: int = 3):
    np = _require_numpy()
    alphabet = default_alphabet(records[0].seq_type)
    vocab_size = len(alphabet) ** k
    if vocab_size > 20000:
        raise ValueError(
            f"k={k} over a {len(alphabet)}-letter alphabet gives {vocab_size} "
            f"dimensions -- too large. Use a smaller k, or --method minhash "
            f"for a fixed-size embedding regardless of k."
        )
    vocabulary = build_kmer_vocabulary(alphabet, k)
    return np.array([kmer_frequency_vector(r.sequence, vocabulary, k) for r in records])


# -- embedding: MinHash -----------------------------------------------------


def minhash_embed(records: list[BioRecord], k: int = 9, num_hashes: int = 64):
    np = _require_numpy()
    vectors = [minhash_signature(r.sequence, k=k, num_hashes=num_hashes) for r in records]
    return np.array(vectors, dtype=float)


# -- embedding: protein language models (optional, heavy) -------------------


def esm_embed(records: list[BioRecord], model_name: str = "esm2_t6_8M_UR50D"):
    """Mean-pooled per-residue representations from an ESM-2 model."""
    try:
        import esm
        import torch
    except ImportError as e:
        raise RuntimeError(
            "ESM embedding requires the 'fair-esm' package and PyTorch "
            "(pip install fair-esm torch). Use --method kmer/minhash for a "
            "dependency-free embedding instead."
        ) from e

    np = _require_numpy()
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    data = [(r.name, r.sequence) for r in records]
    _, _, tokens = batch_converter(data)
    with torch.no_grad():
        out = model(tokens, repr_layers=[model.num_layers])
    reprs = out["representations"][model.num_layers]
    vectors = [reprs[i, 1 : len(r.sequence) + 1].mean(0).numpy() for i, r in enumerate(records)]
    return np.array(vectors)


def prott5_embed(records: list[BioRecord], model_name: str = "Rostlab/prot_t5_xl_half_uniref50-enc"):
    """Mean-pooled per-residue representations from a ProtT5 encoder."""
    try:
        import torch
        from transformers import T5EncoderModel, T5Tokenizer
    except ImportError as e:
        raise RuntimeError(
            "ProtT5 embedding requires 'transformers', PyTorch, and "
            "sentencepiece (pip install transformers torch sentencepiece). "
            "Use --method kmer/minhash for a dependency-free embedding instead."
        ) from e

    np = _require_numpy()
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name)
    model.eval()
    spaced = [" ".join(r.sequence) for r in records]
    encoded = tokenizer.batch_encode_plus(spaced, add_special_tokens=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        out = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
    vectors = [out.last_hidden_state[i, : len(r.sequence)].mean(0).numpy() for i, r in enumerate(records)]
    return np.array(vectors)


def embed_sequences(
    records: list[BioRecord],
    method: str = "kmer",
    k: int | None = None,
    num_hashes: int = 64,
    model_name: str | None = None,
):
    if method not in _EMBED_METHODS:
        raise ValueError(f"unknown embedding method: {method} (choose from {_EMBED_METHODS})")
    if method in _PROTEIN_ONLY_EMBED_METHODS and records[0].seq_type != SeqType.PROTEIN:
        raise ValueError(f"--method {method} is for protein sequences only")

    if method == "kmer":
        return kmer_embed(records, k=k or 3)
    if method == "minhash":
        return minhash_embed(records, k=k or 9, num_hashes=num_hashes)
    if method == "esm":
        return esm_embed(records, model_name=model_name or "esm2_t6_8M_UR50D")
    return prott5_embed(records, model_name=model_name or "Rostlab/prot_t5_xl_half_uniref50-enc")


# -- dimensionality reduction ------------------------------------------------


def reduce_pca(vectors, n_components: int = 2, random_state: int = 0):
    try:
        from sklearn.decomposition import PCA
    except ImportError as e:
        raise RuntimeError("PCA requires scikit-learn (pip install -e '.[cluster]')") from e
    n_components = min(n_components, vectors.shape[0], vectors.shape[1])
    return PCA(n_components=n_components, random_state=random_state).fit_transform(vectors)


def reduce_tsne(vectors, n_components: int = 2, perplexity: float = 30.0, random_state: int = 0):
    try:
        from sklearn.manifold import TSNE
    except ImportError as e:
        raise RuntimeError("t-SNE requires scikit-learn (pip install -e '.[cluster]')") from e
    n = vectors.shape[0]
    if n < 4:
        raise ValueError(f"t-SNE needs at least 4 sequences, got {n}")
    perplexity = min(perplexity, (n - 1) / 3)
    return TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state, init="pca").fit_transform(vectors)


def reduce_umap(vectors, n_components: int = 2, n_neighbors: int = 15, random_state: int = 0):
    try:
        import umap
    except ImportError as e:
        raise RuntimeError("UMAP requires umap-learn (pip install -e '.[embed]')") from e
    n = vectors.shape[0]
    if n < 3:
        raise ValueError(f"UMAP needs at least 3 sequences, got {n}")
    n_neighbors = min(n_neighbors, n - 1)
    return umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state).fit_transform(vectors)


def reduce_sequence_space(vectors, method: str = "pca", **kwargs):
    if method not in _REDUCE_METHODS:
        raise ValueError(f"unknown reduction method: {method} (choose from {_REDUCE_METHODS})")
    if method == "pca":
        return reduce_pca(vectors, **kwargs)
    if method == "tsne":
        return reduce_tsne(vectors, **kwargs)
    return reduce_umap(vectors, **kwargs)


# -- pulling it together ------------------------------------------------


@dataclass
class SequenceSpaceResult:
    seq_ids: list[str]
    names: list[str]
    coordinates: "object"  # np.ndarray, n_records x n_components
    embed_method: str
    reduce_method: str


def build_sequence_space(
    records: list[BioRecord],
    embed_method: str = "kmer",
    reduce_method: str = "pca",
    embed_kwargs: dict | None = None,
    reduce_kwargs: dict | None = None,
) -> SequenceSpaceResult:
    if len(records) < 2:
        raise ValueError("sequence space analysis needs at least 2 records")
    vectors = embed_sequences(records, method=embed_method, **(embed_kwargs or {}))
    coords = reduce_sequence_space(vectors, method=reduce_method, **(reduce_kwargs or {}))
    return SequenceSpaceResult(
        seq_ids=[r.seq_id for r in records],
        names=[r.name for r in records],
        coordinates=coords,
        embed_method=embed_method,
        reduce_method=reduce_method,
    )
