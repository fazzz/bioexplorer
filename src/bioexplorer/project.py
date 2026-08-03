"""Per-directory project state for the ``bio`` CLI.

Each CLI invocation is a fresh process, so state (the current record
collection, plus a log of the commands that produced it) is persisted under
``.bioexplorer/`` in the working directory -- the same pattern ChemExplorer
uses under ``.chemexplorer/`` for ``chem replay``.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .core import BioCollection, BioRecord, SeqType

PROJECT_DIR = ".bioexplorer"
STATE_FILE = "records.json"
LOG_FILE = "log.json"
ALIGNMENTS_DIR = "alignments"
TREES_DIR = "trees"


def project_dir(cwd: Path | None = None) -> Path:
    return (cwd or Path.cwd()) / PROJECT_DIR


def _state_path(cwd: Path | None = None) -> Path:
    return project_dir(cwd) / STATE_FILE


def _log_path(cwd: Path | None = None) -> Path:
    return project_dir(cwd) / LOG_FILE


def exists(cwd: Path | None = None) -> bool:
    return _state_path(cwd).exists()


def save_collection(collection: BioCollection, cwd: Path | None = None) -> None:
    pdir = project_dir(cwd)
    pdir.mkdir(exist_ok=True)
    rows = []
    for rec in collection:
        rows.append(
            {
                "seq_id": rec.seq_id,
                "name": rec.name,
                "sequence": rec.sequence,
                "seq_type": rec.seq_type.value,
                "description": rec.description,
                "tags": sorted(rec.tags),
                "metadata": rec.metadata,
            }
        )
    with open(_state_path(cwd), "w") as fh:
        json.dump(rows, fh, indent=2, default=str)


def load_collection(cwd: Path | None = None) -> BioCollection:
    path = _state_path(cwd)
    if not path.exists():
        raise FileNotFoundError(
            "no BioExplorer project found in this directory "
            "(run `bio import ...` first)"
        )
    with open(path) as fh:
        rows = json.load(fh)
    collection = BioCollection()
    for row in rows:
        rec = BioRecord(
            name=row["name"],
            sequence=row["sequence"],
            seq_type=SeqType(row["seq_type"]),
            seq_id=row["seq_id"],
            description=row.get("description", ""),
            tags=set(row.get("tags", [])),
            metadata=row.get("metadata", {}),
        )
        collection.add(rec)
    return collection


_current_argv: list[str] | None = None


def set_current_argv(argv: list[str] | None) -> None:
    """Called once per CLI invocation (see cli._RecordingGroup.main) with
    the exact argv Click was invoked with -- works whether that's the real
    process's sys.argv or the args list Click's CliRunner was given."""
    global _current_argv
    _current_argv = argv


def read_log(cwd: Path | None = None) -> list[dict]:
    """Read the recorded command history (for `bio replay`, spec section 19)."""
    log_path = _log_path(cwd)
    if not log_path.exists():
        return []
    with open(log_path) as fh:
        return json.load(fh)


def log_command(cwd: Path | None = None) -> None:
    """Append the current CLI invocation to the workflow log (for
    `bio replay`, spec section 19)."""
    pdir = project_dir(cwd)
    pdir.mkdir(exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "argv": _current_argv if _current_argv is not None else sys.argv[1:],
    }
    log_path = _log_path(cwd)
    history = []
    if log_path.exists():
        with open(log_path) as fh:
            history = json.load(fh)
    history.append(entry)
    with open(log_path, "w") as fh:
        json.dump(history, fh, indent=2)


def alignment_path(name: str, cwd: Path | None = None) -> Path:
    return project_dir(cwd) / ALIGNMENTS_DIR / f"{name}.fasta"


def save_alignment(records, name: str, cwd: Path | None = None) -> Path:
    """Persist an aligned record set (list[BioRecord]) as a named MSA under
    .bioexplorer/alignments/<name>.fasta, for later use by profile/tree."""
    from .core import BioCollection
    from .io import write_fasta

    path = alignment_path(name, cwd)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_fasta(BioCollection(records), path)
    return path


def load_alignment(name: str, cwd: Path | None = None) -> list[BioRecord]:
    path = alignment_path(name, cwd)
    if not path.exists():
        raise FileNotFoundError(
            f"no alignment named '{name}' found "
            f"(run `bio align ... --name {name}` first)"
        )
    from Bio import SeqIO

    records = []
    for seqrec in SeqIO.parse(str(path), "fasta"):
        records.append(
            BioRecord(
                name=seqrec.id,
                sequence=str(seqrec.seq),
                seq_type=SeqType.guess(str(seqrec.seq).replace("-", "")),
                description=seqrec.description,
            )
        )
    return records


def list_alignments(cwd: Path | None = None) -> list[str]:
    adir = project_dir(cwd) / ALIGNMENTS_DIR
    if not adir.exists():
        return []
    return sorted(p.stem for p in adir.glob("*.fasta"))


def tree_path(name: str, cwd: Path | None = None) -> Path:
    return project_dir(cwd) / TREES_DIR / f"{name}.nwk"


def save_tree(tree, name: str, cwd: Path | None = None) -> Path:
    from .tree import write_newick

    path = tree_path(name, cwd)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_newick(tree, path)
    return path


def load_tree(name: str, cwd: Path | None = None):
    from .tree import read_newick

    path = tree_path(name, cwd)
    if not path.exists():
        raise FileNotFoundError(
            f"no tree named '{name}' found (run `bio tree ... --name {name}` first)"
        )
    return read_newick(path)


def list_trees(cwd: Path | None = None) -> list[str]:
    tdir = project_dir(cwd) / TREES_DIR
    if not tdir.exists():
        return []
    return sorted(p.stem for p in tdir.glob("*.nwk"))
