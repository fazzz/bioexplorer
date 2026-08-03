"""Evolutionary analysis (spec section 13).

- dN/dS (Ka/Ks): implemented here, via Biopython's Bio.codonalign (Nei-Gojobori
  1986 by default; no external tool needed) with an optional PAML yn00
  backend for the ML-based estimate, same defensive external-tool pattern
  as elsewhere in the package.
- Conservation Analysis: already covered by profile.py's conservation_score
  / relative_entropy (built from an MSA). ``conservation_summary`` below is
  a thin re-export so it's reachable from this section-13 surface too,
  rather than a second implementation.
- Bootstrap Support: already covered by tree.py's ``build_distance_tree(...,
  bootstrap=N)``. Re-exported here for the same reason.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from Bio import BiopythonExperimentalWarning

warnings.filterwarnings("ignore", category=BiopythonExperimentalWarning, module="Bio.codonalign")

from Bio.codonalign.codonseq import CodonSeq, cal_dn_ds

from .core import BioRecord
from .profile import build_profile
from .similarity import _require_tool
from .tree import build_distance_tree

_STOP_CODONS = {"TAA", "TAG", "TGA"}
_DN_DS_METHODS = ("NG86", "LWL85", "YN00")


def _strip_trailing_stop(seq_a: str, seq_b: str) -> tuple[str, str]:
    """dN/dS conventionally excludes the stop codon (and Biopython's codon
    table has no entry for one, so leaving it in raises a KeyError)."""
    if seq_a[-3:] in _STOP_CODONS and seq_b[-3:] in _STOP_CODONS:
        return seq_a[:-3], seq_b[:-3]
    return seq_a, seq_b


@dataclass
class DnDsResult:
    seq_a_id: str
    seq_b_id: str
    method: str
    dn: float
    ds: float
    omega: float | None  # dN/dS; None when dS == 0 (undefined/infinite selection signal)


def _check_codon_aligned(sequence: str, label: str) -> None:
    if len(sequence) % 3 != 0:
        raise ValueError(
            f"{label} length ({len(sequence)}) is not a multiple of 3 -- "
            f"dN/dS needs codon-aligned coding sequences (equal length, "
            f"gaps in multiples of 3). Align coding sequences with `bio "
            f"align` first if needed."
        )


def pairwise_dn_ds(
    seq_a: str,
    seq_b: str,
    seq_a_id: str = "a",
    seq_b_id: str = "b",
    method: str = "NG86",
) -> DnDsResult:
    """dN/dS between two codon-aligned coding sequences (equal length,
    multiple of 3; '-' gap codons are handled by the underlying method).
    NG86 (Nei & Gojobori 1986, counting-based) is the default and needs no
    external tool; LWL85 is likewise pure Python. Use method='YN00' for
    the ML-based PAML estimate (requires the `yn00` binary)."""
    if method not in _DN_DS_METHODS:
        raise ValueError(f"unknown dN/dS method: {method} (choose from {_DN_DS_METHODS})")
    if len(seq_a) != len(seq_b):
        raise ValueError("sequences must be the same (codon-aligned) length")
    seq_a, seq_b = _strip_trailing_stop(seq_a.upper().replace("U", "T"), seq_b.upper().replace("U", "T"))
    _check_codon_aligned(seq_a, seq_a_id)
    _check_codon_aligned(seq_b, seq_b_id)

    if method == "YN00":
        dn, ds = _yn00_dn_ds(seq_a, seq_b, seq_a_id, seq_b_id)
    else:
        codon_a = CodonSeq(seq_a)
        codon_b = CodonSeq(seq_b)
        try:
            dn, ds = cal_dn_ds(codon_a, codon_b, method=method)
        except (ValueError, ZeroDivisionError) as e:
            raise RuntimeError(
                f"{method} could not be computed for this pair (sequence too "
                f"short or too divergent for the method's approximations); "
                f"try method='NG86' or a longer alignment."
            ) from e

    omega = (dn / ds) if ds and ds > 0 else None
    return DnDsResult(seq_a_id=seq_a_id, seq_b_id=seq_b_id, method=method, dn=dn, ds=ds, omega=omega)


def _yn00_dn_ds(seq_a: str, seq_b: str, id_a: str, id_b: str) -> tuple[float, float]:
    _require_tool("yn00")
    import tempfile
    from pathlib import Path

    from Bio.Phylo.PAML import yn00

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        aln_path = tmp_path / "pair.phy"
        # yn00 wants relaxed PHYLIP with codon-aligned sequences
        aln_path.write_text(
            f" 2 {len(seq_a)}\n{id_a[:10]:<10}{seq_a}\n{id_b[:10]:<10}{seq_b}\n"
        )
        runner = yn00.Yn00(alignment=str(aln_path), working_dir=str(tmp_path), out_file=str(tmp_path / "yn00.out"))
        results = runner.run()
        pair = results[id_a][id_b]["YN00"] if id_a in results else results[id_b][id_a]["YN00"]
        return pair["dN"], pair["dS"]


def dn_ds_matrix(records: list[BioRecord], method: str = "NG86") -> list[DnDsResult]:
    """All-pairs dN/dS across a set of codon-aligned coding sequences."""
    results = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            a, b = records[i], records[j]
            results.append(pairwise_dn_ds(a.sequence, b.sequence, a.name, b.name, method=method))
    return results


# -- re-exports tying the other two spec-13 topics into this module ---------


def conservation_summary(records: list[BioRecord]) -> dict:
    """Conservation Analysis (spec section 13), built on profile.py's MSA
    profile: mean/most/least conserved positions."""
    profile = build_profile(records)
    scores = profile.conservation_score
    most = max(range(len(scores)), key=lambda i: scores[i])
    least = min(range(len(scores)), key=lambda i: scores[i])
    return {
        "mean_conservation": sum(scores) / len(scores) if scores else 0.0,
        "most_conserved_position": most + 1,
        "most_conserved_score": scores[most] if scores else None,
        "least_conserved_position": least + 1,
        "least_conserved_score": scores[least] if scores else None,
        "consensus": profile.consensus,
    }


def bootstrap_tree(records: list[BioRecord], method: str = "nj", replicates: int = 100, model: str | None = None, seed: int = 0):
    """Bootstrap Support (spec section 13): a distance tree with branch
    confidence values from bootstrap resampling. Thin wrapper around
    tree.build_distance_tree(..., bootstrap=replicates)."""
    return build_distance_tree(records, method=method, model=model, bootstrap=replicates, seed=seed)
