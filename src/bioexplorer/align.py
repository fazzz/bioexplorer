"""Alignment (spec section 11).

Pairwise: Bio.Align.PairwiseAligner in global (Needleman-Wunsch) or local
(Smith-Waterman) mode -- no external tool needed.

Multiple: MAFFT / MUSCLE / Clustal Omega, invoked as subprocesses, same
pattern as the BLAST/DIAMOND/MMseqs2 wrappers in similarity.py (missing
binary -> clear RuntimeError, not a silent fallback).
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from Bio import Align, SeqIO

from .core import BioRecord, SeqType
from .io import write_fasta
from .similarity import _require_tool

# -- pairwise (in-process) ------------------------------------------------


@dataclass
class PairwiseResult:
    mode: str
    score: float
    aligned_a: str
    aligned_b: str
    target_id: str = ""
    query_id: str = ""


def _build_aligner(
    mode: str,
    seq_type: SeqType,
    match: float,
    mismatch: float,
    gap_open: float,
    gap_extend: float,
) -> Align.PairwiseAligner:
    if mode not in ("global", "local"):
        raise ValueError("mode must be 'global' (Needleman-Wunsch) or 'local' (Smith-Waterman)")
    aligner = Align.PairwiseAligner()
    aligner.mode = mode
    aligner.open_gap_score = gap_open
    aligner.extend_gap_score = gap_extend
    if seq_type == SeqType.PROTEIN:
        aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
    else:
        aligner.match_score = match
        aligner.mismatch_score = mismatch
    return aligner


def pairwise_align(
    seq_a: str,
    seq_b: str,
    mode: str = "global",
    seq_type: SeqType = SeqType.DNA,
    match: float = 2.0,
    mismatch: float = -1.0,
    gap_open: float = -10.0,
    gap_extend: float = -0.5,
    target_id: str = "",
    query_id: str = "",
) -> PairwiseResult:
    """Align two raw sequences. Global = Needleman-Wunsch, local =
    Smith-Waterman. Uses BLOSUM62 for protein, match/mismatch scores for
    nucleotide sequences."""
    aligner = _build_aligner(mode, seq_type, match, mismatch, gap_open, gap_extend)
    alignment = aligner.align(seq_a, seq_b)[0]
    lines = str(alignment).splitlines()
    # Bio.Align's str() interleaves target/query blocks; extract the two
    # aligned strings via the alignment's own indices instead of parsing text.
    aligned_a, aligned_b = alignment[0], alignment[1]
    return PairwiseResult(
        mode=mode,
        score=alignment.score,
        aligned_a=aligned_a,
        aligned_b=aligned_b,
        target_id=target_id,
        query_id=query_id,
    )


def format_alignment(result: PairwiseResult, width: int = 60) -> str:
    """Render a PairwiseResult as a readable, wrapped alignment block with a
    match-line between the two sequences."""
    match_line = "".join(
        "|" if a == b and a != "-" else (" " if a == "-" or b == "-" else ".")
        for a, b in zip(result.aligned_a, result.aligned_b)
    )
    a_label = result.target_id or "target"
    b_label = result.query_id or "query"
    label_width = max(len(a_label), len(b_label))
    out = [f"mode={result.mode}  score={result.score}"]
    for i in range(0, len(result.aligned_a), width):
        out.append(f"{a_label.ljust(label_width)}  {result.aligned_a[i:i + width]}")
        out.append(f"{' ' * label_width}  {match_line[i:i + width]}")
        out.append(f"{b_label.ljust(label_width)}  {result.aligned_b[i:i + width]}")
        out.append("")
    return "\n".join(out).rstrip()


# -- multiple alignment (external tools) -----------------------------------

_MSA_TOOLS = ("mafft", "muscle", "clustalo")


def multiple_align(
    records: list[BioRecord], tool: str = "mafft", extra_args: list[str] | None = None
) -> list[BioRecord]:
    """Align records with an external MSA tool and return new BioRecord
    copies whose ``sequence`` includes gap characters ('-'). The originals
    (in the project) are left untouched."""
    if tool not in _MSA_TOOLS:
        raise ValueError(f"unknown MSA tool: {tool} (choose from {_MSA_TOOLS})")
    if len(records) < 2:
        raise ValueError("multiple alignment needs at least 2 sequences")

    from .core import BioCollection

    collection = BioCollection(records)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_fasta = tmp_path / "input.fasta"
        output_fasta = tmp_path / "aligned.fasta"
        write_fasta(collection, input_fasta)

        if tool == "mafft":
            binary = _require_tool("mafft")
            cmd = [binary, "--auto", *(extra_args or []), str(input_fasta)]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            output_fasta.write_text(result.stdout)
        elif tool == "muscle":
            binary = _require_tool("muscle")
            cmd = [binary, "-align", str(input_fasta), "-output", str(output_fasta), *(extra_args or [])]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        else:  # clustalo
            binary = _require_tool("clustalo")
            cmd = [binary, "-i", str(input_fasta), "-o", str(output_fasta), "--force", *(extra_args or [])]
            subprocess.run(cmd, check=True, capture_output=True, text=True)

        by_name = {rec.name: rec for rec in records}
        aligned: list[BioRecord] = []
        for seqrec in SeqIO.parse(str(output_fasta), "fasta"):
            original = by_name.get(seqrec.id)
            seq_type = original.seq_type if original else SeqType.guess(str(seqrec.seq).replace("-", ""))
            new_rec = BioRecord(
                name=seqrec.id,
                sequence=str(seqrec.seq),
                seq_type=seq_type,
                description=original.description if original else seqrec.description,
                tags=set(original.tags) if original else set(),
                metadata=dict(original.metadata) if original else {},
            )
            new_rec.set("aligned_with", tool)
            aligned.append(new_rec)
        return aligned
