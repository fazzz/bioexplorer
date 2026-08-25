"""File I/O for BioExplorer.

Phase-1 scope: the record-oriented formats Bio.SeqIO handles natively
(FASTA / FASTQ / GenBank / EMBL). Alignment formats (Stockholm, Clustal),
tree formats (Newick, Nexus) and structure formats (PDB, mmCIF) are read
directly by the modules that need them (align.py, tree.py, structure.py)
rather than funnelled through BioCollection, since they aren't flat
sequence-record collections.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from Bio import SeqIO

from .core import BioCollection, BioRecord, SeqType

# Maps a --format flag / file suffix to a Bio.SeqIO format name.
_SEQIO_FORMATS = {
    "fasta": "fasta",
    "fa": "fasta",
    "fna": "fasta",
    "faa": "fasta",
    "fastq": "fastq",
    "fq": "fastq",
    "genbank": "genbank",
    "gb": "genbank",
    "gbk": "genbank",
    "embl": "embl",
}

SUPPORTED_IMPORT_FORMATS = sorted(set(_SEQIO_FORMATS.values()))


def guess_format(path: Path) -> str:
    suffix = path.suffix.lstrip(".").lower()
    # handle .fasta.gz-style double suffixes minimally
    if suffix in ("gz", "bz2", "zip"):
        suffix = path.with_suffix("").suffix.lstrip(".").lower()
    if suffix not in _SEQIO_FORMATS:
        raise ValueError(
            f"cannot infer format from suffix '.{suffix}' for {path}; "
            f"pass --format explicitly (one of {SUPPORTED_IMPORT_FORMATS})"
        )
    return _SEQIO_FORMATS[suffix]


def read_file(
    path: Path, fmt: str | None = None, seq_type: str | None = None
) -> list[BioRecord]:
    """Read a single sequence file into a list of BioRecord."""
    resolved_fmt = _SEQIO_FORMATS.get(fmt, fmt) if fmt else guess_format(path)
    records = []
    for rec in SeqIO.parse(str(path), resolved_fmt):
        records.append(BioRecord.from_seqrecord(rec, seq_type=seq_type))
        records[-1].set("source_file", str(path))
        records[-1].set("source_format", resolved_fmt)
    return records


def iter_input_paths(path: Path, recursive: bool = False) -> Iterable[Path]:
    """Expand a file-or-directory input into a flat list of files.

    Supports the "one sequence per file" layout some pipelines produce
    (e.g. a directory of single-record FASTA/GenBank files), mirroring
    ChemExplorer's ``chem import`` directory handling.
    """
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise FileNotFoundError(path)
    pattern = "**/*" if recursive else "*"
    for p in sorted(path.glob(pattern)):
        if p.is_file() and not p.name.startswith("."):
            yield p


def load_collection(
    paths: list[Path],
    fmt: str | None = None,
    seq_type: str | None = None,
    recursive: bool = False,
) -> BioCollection:
    collection = BioCollection()
    for path in paths:
        for file_path in iter_input_paths(path, recursive=recursive):
            try:
                recs = read_file(file_path, fmt=fmt, seq_type=seq_type)
            except ValueError:
                if fmt is None and path.is_dir():
                    # skip files we can't recognize when scanning a directory
                    continue
                raise
            for rec in recs:
                collection.add(rec)
    return collection


def write_fasta(collection: BioCollection, path: Path) -> None:
    with open(path, "w") as fh:
        for rec in collection:
            header = rec.name
            desc = rec.description
            if desc.startswith(rec.name + " "):
                desc = desc[len(rec.name) + 1 :]
            if desc and desc != rec.name:
                header = f"{rec.name} {desc}"
            fh.write(f">{header}\n")
            seq = rec.sequence
            for i in range(0, len(seq), 70):
                fh.write(seq[i : i + 70] + "\n")


def _record_row(rec: BioRecord) -> dict:
    row = {
        "seq_id": rec.seq_id,
        "name": rec.name,
        "seq_type": rec.seq_type.value,
        "length": rec.length,
        "description": rec.description,
        "tags": ";".join(sorted(rec.tags)),
    }
    for k, v in rec.metadata.items():
        if isinstance(v, (list, set, tuple)):
            v = ";".join(str(x) for x in v)
        row[f"meta_{k}"] = v
    return row


def write_table(collection: BioCollection, path: Path, fmt: str) -> None:
    rows = [_record_row(rec) for rec in collection]
    if fmt in ("csv", "tsv"):
        delimiter = "," if fmt == "csv" else "\t"
        fieldnames: list[str] = []
        for row in rows:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(rows)
    elif fmt == "json":
        with open(path, "w") as fh:
            json.dump(rows, fh, indent=2, default=str)
    elif fmt == "parquet":
        try:
            import pandas as pd
        except ImportError as e:
            raise RuntimeError(
                "parquet export requires pandas + pyarrow "
                "(uv sync --extra parquet, or pip install -e '.[parquet]')"
            ) from e
        pd.DataFrame(rows).to_parquet(path)
    else:
        raise ValueError(f"unsupported table format: {fmt}")


def write_collection(collection: BioCollection, path: Path, fmt: str) -> None:
    if fmt == "fasta":
        write_fasta(collection, path)
    elif fmt in ("csv", "tsv", "json", "parquet"):
        write_table(collection, path, fmt)
    else:
        raise ValueError(f"unsupported export format: {fmt}")
