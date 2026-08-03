"""Similarity search (spec section 8).

Two tiers, matching the spec:

- Fast, dependency-free: k-mer Jaccard and MinHash, implemented in pure
  Python so ``bio search`` works with no external tools installed.
- High-precision: BLAST / DIAMOND / MMseqs2, invoked as subprocesses. These
  require the corresponding binary on PATH; if it's missing we raise a
  clear, actionable error rather than silently falling back, the same way
  ChemExplorer's external 3D viewer integration behaves.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .core import BioCollection, BioRecord
from .io import write_fasta

# -- fast, in-process methods -------------------------------------------


def kmer_set(sequence: str, k: int = 4) -> set[str]:
    if len(sequence) < k:
        return {sequence}
    return {sequence[i : i + k] for i in range(len(sequence) - k + 1)}


def jaccard_similarity(seq_a: str, seq_b: str, k: int = 4) -> float:
    set_a, set_b = kmer_set(seq_a, k), kmer_set(seq_b, k)
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


# MinHash: a fixed family of universal hash functions h_i(x) = (a_i*x + b_i) mod p,
# applied to a 64-bit hash of each k-mer. Deterministic (fixed seeds) so
# signatures are reproducible across runs -- important for the workflow log
# / replay to be meaningful.
_MERSENNE_PRIME = (1 << 61) - 1
_DEFAULT_NUM_HASHES = 64


def _hash_coeffs(num_hashes: int) -> list[tuple[int, int]]:
    # Deterministic LCG-derived coefficients, not cryptographic -- fine for
    # sketching, and avoids pulling in numpy/random-state plumbing here.
    coeffs = []
    a, b = 1103515245, 12345
    state = 42
    for _ in range(num_hashes):
        state = (a * state + b) % (2**31)
        coeff_a = (state % (_MERSENNE_PRIME - 1)) + 1
        state = (a * state + b) % (2**31)
        coeff_b = state % _MERSENNE_PRIME
        coeffs.append((coeff_a, coeff_b))
    return coeffs


def minhash_signature(
    sequence: str, k: int = 9, num_hashes: int = _DEFAULT_NUM_HASHES
) -> tuple[int, ...]:
    kmers = kmer_set(sequence, k)
    if not kmers:
        return tuple([0] * num_hashes)
    coeffs = _hash_coeffs(num_hashes)
    hashed = [hash(km) & ((1 << 61) - 1) for km in kmers]
    sig = []
    for a, b in coeffs:
        sig.append(min((a * x + b) % _MERSENNE_PRIME for x in hashed))
    return tuple(sig)


def minhash_similarity(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
    if not sig_a or len(sig_a) != len(sig_b):
        return 0.0
    matches = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return matches / len(sig_a)


@dataclass
class SimilarityHit:
    record: BioRecord
    score: float


def search_fast(
    collection: BioCollection,
    query_sequence: str,
    method: str = "kmer",
    k: int | None = None,
    top_n: int = 10,
    min_score: float = 0.0,
) -> list[SimilarityHit]:
    """k-mer Jaccard or MinHash similarity search against every record in
    the collection. O(n) in the collection size; fine for the scale this
    in-process method is meant for (larger libraries should use
    BLAST/DIAMOND/MMseqs2 below)."""
    if method == "kmer":
        kk = k or 4
        query_set = kmer_set(query_sequence, kk)
        hits = []
        for rec in collection:
            rec_set = kmer_set(rec.sequence, kk)
            union = len(query_set | rec_set)
            score = len(query_set & rec_set) / union if union else 0.0
            if score >= min_score:
                hits.append(SimilarityHit(rec, score))
    elif method == "minhash":
        kk = k or 9
        query_sig = minhash_signature(query_sequence, kk)
        hits = []
        for rec in collection:
            rec_sig = minhash_signature(rec.sequence, kk)
            score = minhash_similarity(query_sig, rec_sig)
            if score >= min_score:
                hits.append(SimilarityHit(rec, score))
    else:
        raise ValueError(f"unknown fast similarity method: {method}")

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_n]


# -- external high-precision tools ---------------------------------------


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"'{name}' was not found on PATH. Install it, or use a "
            f"built-in, dependency-free alternative where available "
            f"(e.g. --method kmer / --method minhash for similarity search)."
        )
    return path


def search_blast(
    collection: BioCollection,
    query_sequence: str,
    program: str = "blastn",
    top_n: int = 10,
    evalue: float = 10.0,
) -> list[SimilarityHit]:
    _require_tool(program)
    _require_tool("makeblastdb")
    dbtype = "nucl" if program == "blastn" else "prot"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_fasta = tmp_path / "db.fasta"
        query_fasta = tmp_path / "query.fasta"
        write_fasta(collection, db_fasta)
        query_fasta.write_text(f">query\n{query_sequence}\n")

        subprocess.run(
            ["makeblastdb", "-in", str(db_fasta), "-dbtype", dbtype, "-out", str(tmp_path / "db")],
            check=True,
            capture_output=True,
        )
        result = subprocess.run(
            [
                program,
                "-query", str(query_fasta),
                "-db", str(tmp_path / "db"),
                "-outfmt", "6 sseqid bitscore pident",
                "-evalue", str(evalue),
                "-max_target_seqs", str(top_n),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    by_name: dict[str, BioRecord] = {}
    for rec in collection:
        by_name.setdefault(rec.name, rec)

    hits = []
    for line in result.stdout.strip().splitlines():
        sseqid, bitscore, pident = line.split("\t")
        rec = by_name.get(sseqid)
        if rec is not None:
            hits.append(SimilarityHit(rec, float(pident) / 100.0))
    return hits[:top_n]


def search_diamond(
    collection: BioCollection, query_sequence: str, top_n: int = 10
) -> list[SimilarityHit]:
    _require_tool("diamond")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_fasta = tmp_path / "db.fasta"
        query_fasta = tmp_path / "query.fasta"
        write_fasta(collection, db_fasta)
        query_fasta.write_text(f">query\n{query_sequence}\n")

        subprocess.run(
            ["diamond", "makedb", "--in", str(db_fasta), "-d", str(tmp_path / "db")],
            check=True,
            capture_output=True,
        )
        result = subprocess.run(
            [
                "diamond", "blastp",
                "-q", str(query_fasta),
                "-d", str(tmp_path / "db"),
                "-k", str(top_n),
                "--outfmt", "6", "sseqid", "pident",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    by_name = {rec.name: rec for rec in collection}
    hits = []
    for line in result.stdout.strip().splitlines():
        sseqid, pident = line.split("\t")
        rec = by_name.get(sseqid)
        if rec is not None:
            hits.append(SimilarityHit(rec, float(pident) / 100.0))
    return hits[:top_n]


def search_mmseqs(
    collection: BioCollection, query_sequence: str, top_n: int = 10
) -> list[SimilarityHit]:
    _require_tool("mmseqs")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_fasta = tmp_path / "db.fasta"
        query_fasta = tmp_path / "query.fasta"
        write_fasta(collection, db_fasta)
        query_fasta.write_text(f">query\n{query_sequence}\n")

        subprocess.run(
            ["mmseqs", "createdb", str(db_fasta), str(tmp_path / "targetDB")],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["mmseqs", "createdb", str(query_fasta), str(tmp_path / "queryDB")],
            check=True, capture_output=True,
        )
        subprocess.run(
            [
                "mmseqs", "search",
                str(tmp_path / "queryDB"), str(tmp_path / "targetDB"),
                str(tmp_path / "resultDB"), str(tmp_path / "tmp"),
            ],
            check=True, capture_output=True,
        )
        subprocess.run(
            [
                "mmseqs", "convertalis",
                str(tmp_path / "queryDB"), str(tmp_path / "targetDB"),
                str(tmp_path / "resultDB"), str(tmp_path / "result.tsv"),
                "--format-output", "target,pident",
            ],
            check=True, capture_output=True,
        )
        result_text = (tmp_path / "result.tsv").read_text()

    by_name = {rec.name: rec for rec in collection}
    hits = []
    for line in result_text.strip().splitlines():
        target, pident = line.split("\t")
        rec = by_name.get(target)
        if rec is not None:
            hits.append(SimilarityHit(rec, float(pident) / 100.0))
    return hits[:top_n]


_EXTERNAL_METHODS = {"blast", "diamond", "mmseqs"}


def search_similar(
    collection: BioCollection,
    query_sequence: str,
    method: str = "kmer",
    k: int | None = None,
    top_n: int = 10,
    min_score: float = 0.0,
) -> list[SimilarityHit]:
    """Dispatch to the requested similarity method."""
    if method in ("kmer", "minhash"):
        return search_fast(collection, query_sequence, method=method, k=k, top_n=top_n, min_score=min_score)
    if method == "blast":
        return search_blast(collection, query_sequence, top_n=top_n)
    if method == "diamond":
        return search_diamond(collection, query_sequence, top_n=top_n)
    if method == "mmseqs":
        return search_mmseqs(collection, query_sequence, top_n=top_n)
    raise ValueError(f"unknown similarity method: {method}")
