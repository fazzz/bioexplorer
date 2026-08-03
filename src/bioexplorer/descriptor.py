"""Sequence descriptors (spec section 6).

DNA/RNA: length, GC%, AT%, N content, codon usage, GC3
Protein: length, MW, pI, amino acid composition, aromaticity,
instability index, GRAVY

Descriptors are written into ``BioRecord.metadata`` under a ``descriptor``
sub-dict so they can be used directly as search/filter fields (spec
section 7) and exported flat via ``bio export``.
"""

from __future__ import annotations

from collections import Counter

from Bio.SeqUtils import gc_fraction
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from .core import BioRecord, SeqType

_CODON_TABLE_ALPHABET = set("ACGT")


def _nucleotide_descriptors(rec: BioRecord) -> dict:
    seq = rec.sequence.replace("U", "T") if rec.seq_type == SeqType.RNA else rec.sequence
    length = len(seq)
    n_count = seq.count("N")
    counted = length - n_count
    gc = gc_fraction(seq, ambiguous="ignore") if length else 0.0
    at = 1.0 - gc if counted else 0.0

    out: dict = {
        "length": length,
        "gc_percent": round(gc * 100, 3),
        "at_percent": round(at * 100, 3),
        "n_content": round(n_count / length, 4) if length else 0.0,
    }

    if length % 3 == 0 and length > 0 and set(seq) <= (_CODON_TABLE_ALPHABET | {"N"}):
        codons = [seq[i : i + 3] for i in range(0, length, 3)]
        clean_codons = [c for c in codons if "N" not in c]
        codon_counts = Counter(clean_codons)
        total = sum(codon_counts.values())
        out["codon_usage"] = (
            {codon: round(count / total, 4) for codon, count in codon_counts.most_common()}
            if total
            else {}
        )
        third_positions = "".join(c[2] for c in clean_codons if len(c) == 3)
        out["gc3_percent"] = (
            round(gc_fraction(third_positions, ambiguous="ignore") * 100, 3)
            if third_positions
            else None
        )
    else:
        out["codon_usage"] = None
        out["gc3_percent"] = None

    return out


def _protein_descriptors(rec: BioRecord) -> dict:
    seq = rec.sequence.replace("*", "").replace("-", "")
    if not seq:
        return {
            "length": 0,
            "molecular_weight": None,
            "pi": None,
            "aa_composition": {},
            "aromaticity": None,
            "instability_index": None,
            "gravy": None,
        }
    analysis = ProteinAnalysis(seq)
    return {
        "length": len(seq),
        "molecular_weight": round(analysis.molecular_weight(), 2),
        "pi": round(analysis.isoelectric_point(), 3),
        "aa_composition": {
            aa: round(frac, 4) for aa, frac in analysis.amino_acids_percent.items()
        },
        "aromaticity": round(analysis.aromaticity(), 4),
        "instability_index": round(analysis.instability_index(), 3),
        "gravy": round(analysis.gravy(), 4),
    }


def compute_descriptors(rec: BioRecord) -> dict:
    """Compute and return the descriptor dict for a single record (does not
    mutate the record)."""
    if rec.seq_type == SeqType.PROTEIN:
        return _protein_descriptors(rec)
    return _nucleotide_descriptors(rec)


def annotate_descriptors(rec: BioRecord) -> dict:
    """Compute descriptors and store them on the record's metadata."""
    desc = compute_descriptors(rec)
    rec.set("descriptor", desc)
    return desc
