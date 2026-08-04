"""Database download helpers.

BLAST/DIAMOND/MMseqs2 search against a local index -- there's no reason to
reimplement a downloader when the tools already ship official ones:

- BLAST: ``update_blastdb.pl`` (part of blast+), which knows how to fetch
  and decompress any DB in NCBI's public list (nr, nt, swissprot, ...).
- MMseqs2: ``mmseqs databases``, which fetches and formats a curated list
  (UniRef50/90/100, UniProtKB, PDB seqres, Pfam-A, ...) directly into an
  mmseqs DB ready for ``bio search --method mmseqs --db``.

DIAMOND has no downloader of its own -- build a DIAMOND DB from any FASTA
you already have (e.g. one fetched via update_blastdb.pl --decompress, or
downloaded from UniProt directly) with ``diamond makedb``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .similarity import _require_tool

# A handful of commonly-used names, for `bio db list` -- not exhaustive;
# both tools' full catalogs are large and change over time.
COMMON_BLAST_DBS = ("nr", "nt", "swissprot", "pdbaa", "pdbnt", "refseq_protein", "refseq_rna")
COMMON_MMSEQS_DBS = ("UniRef50", "UniRef90", "UniRef100", "UniProtKB", "PDB", "Pfam-A.full")


def fetch_blast_db(name: str, output_dir: Path, decompress: bool = True) -> Path:
    """Download a pre-formatted BLAST DB via update_blastdb.pl. `name` is
    an NCBI DB name (see COMMON_BLAST_DBS, or `update_blastdb.pl --showall`
    for the full list). Returns the DB prefix to pass as --db."""
    binary = _require_tool("update_blastdb.pl")
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [binary, name]
    if decompress:
        cmd.append("--decompress")
    subprocess.run(cmd, check=True, cwd=str(output_dir), capture_output=True, text=True)
    return output_dir / name


def fetch_mmseqs_db(name: str, output_prefix: Path, tmp_dir: Path | None = None) -> Path:
    """Download and format an MMseqs2 DB via `mmseqs databases`. `name` is
    one of mmseqs's curated DB names (see COMMON_MMSEQS_DBS, or `mmseqs
    databases` with no args for the full list). Returns the DB prefix to
    pass as --db."""
    binary = _require_tool("mmseqs")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    resolved_tmp = tmp_dir or (output_prefix.parent / f".{output_prefix.name}_tmp")
    resolved_tmp.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [binary, "databases", name, str(output_prefix), str(resolved_tmp)],
        check=True, capture_output=True, text=True,
    )
    return output_prefix


def fetch_db(tool: str, name: str, output_path: Path, **kwargs) -> Path:
    if tool == "blast":
        return fetch_blast_db(name, output_path, **kwargs)
    if tool == "mmseqs":
        return fetch_mmseqs_db(name, output_path, **kwargs)
    raise ValueError(f"unknown tool: {tool} (choose from 'blast', 'mmseqs')")
