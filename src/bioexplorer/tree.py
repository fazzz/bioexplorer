"""Phylogenetic analysis (spec section 14).

Distance-based methods (Neighbor Joining, UPGMA) run in-process via
Bio.Phylo.TreeConstruction -- no external tool needed, since they only need
the alignment plus a substitution model to build a distance matrix.

Maximum-likelihood/maximum-parsimony tools (IQ-TREE, FastTree, RAxML) are
external binaries, wrapped as subprocesses with the same
found-on-PATH-or-clear-error pattern used throughout the package.

Trees are Bio.Phylo tree objects; ``write_newick``/``read_newick`` persist
them, matching the spec's "saved as Newick" requirement.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from Bio import Phylo
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.Consensus import bootstrap_trees, get_support, majority_consensus
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

from .core import BioRecord, SeqType
from .io import write_fasta
from .similarity import _require_tool

_DISTANCE_METHODS = ("nj", "upgma")
_EXTERNAL_TOOLS = ("iqtree", "fasttree", "raxml")

# Reasonable default substitution models per alphabet for
# Bio.Phylo.TreeConstruction.DistanceCalculator.
_DEFAULT_MODEL = {
    SeqType.DNA: "identity",
    SeqType.RNA: "identity",
    SeqType.PROTEIN: "blosum62",
}


def _to_msa(records: list[BioRecord]) -> MultipleSeqAlignment:
    lengths = {r.length for r in records}
    if len(lengths) != 1:
        raise ValueError(
            "phylogenetic tree building needs an alignment (equal-length "
            "sequences) -- run `bio align` first"
        )
    seqrecords = [SeqRecord(Seq(r.sequence), id=r.name) for r in records]
    return MultipleSeqAlignment(seqrecords)


def build_distance_tree(
    records: list[BioRecord],
    method: str = "nj",
    model: str | None = None,
    bootstrap: int = 0,
    seed: int = 0,
):
    """Build a tree with Neighbor Joining or UPGMA from an alignment.

    With ``bootstrap > 0``, builds that many bootstrap-resampled trees and
    returns the majority-rule consensus tree with branch ``confidence``
    values set (spec section 13's Bootstrap Support, applied here since
    it's naturally a tree-building option).
    """
    if method not in _DISTANCE_METHODS:
        raise ValueError(f"unknown distance method: {method} (choose from {_DISTANCE_METHODS})")
    msa = _to_msa(records)
    resolved_model = model or _DEFAULT_MODEL[records[0].seq_type]
    calculator = DistanceCalculator(resolved_model)
    constructor = DistanceTreeConstructor(calculator, method=method)

    if bootstrap and bootstrap > 0:
        import random

        random.seed(seed)
        replicate_trees = list(bootstrap_trees(msa, bootstrap, constructor))
        tree = majority_consensus(replicate_trees)
        tree = get_support(tree, replicate_trees)
        return tree

    dm = calculator.get_distance(msa)
    return constructor.nj(dm) if method == "nj" else constructor.upgma(dm)


# -- external ML/MP tools ---------------------------------------------------


def build_tree_external(
    records: list[BioRecord],
    tool: str = "fasttree",
    seq_type: SeqType | None = None,
    model: str | None = None,
    extra_args: list[str] | None = None,
):
    if tool not in _EXTERNAL_TOOLS:
        raise ValueError(f"unknown external tree tool: {tool} (choose from {_EXTERNAL_TOOLS})")
    resolved_type = seq_type or records[0].seq_type

    from .core import BioCollection

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_fasta = tmp_path / "aligned.fasta"
        write_fasta(BioCollection(records), input_fasta)

        if tool == "fasttree":
            binary = _require_tool("fasttree") if _which("fasttree") else _require_tool("FastTree")
            cmd = [binary]
            if resolved_type != SeqType.PROTEIN:
                cmd.append("-nt")
            cmd += [*(extra_args or []), str(input_fasta)]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            newick_text = result.stdout

        elif tool == "iqtree":
            binary = _require_tool("iqtree2") if _which("iqtree2") else _require_tool("iqtree")
            cmd = [binary, "-s", str(input_fasta), "-m", model or "MFP", "-quiet", *(extra_args or [])]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            treefile = input_fasta.with_suffix(input_fasta.suffix + ".treefile")
            newick_text = treefile.read_text()

        else:  # raxml
            binary = _require_tool("raxml-ng") if _which("raxml-ng") else _require_tool("raxmlHPC")
            run_name = "bioexplorer"
            data_type = "DNA" if resolved_type != SeqType.PROTEIN else "AA"
            cmd = [
                binary, "--all" if "raxmlHPC" in binary else "--search",
                "--msa", str(input_fasta), "--model", model or (data_type + "GTR" if data_type == "DNA" else "LG"),
                *(extra_args or []),
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=tmp_path)
            candidates = list(tmp_path.glob("*bestTree*")) + list(tmp_path.glob("RAxML_bestTree*"))
            if not candidates:
                raise RuntimeError("raxml did not produce a best-tree output file")
            newick_text = candidates[0].read_text()

    from io import StringIO

    return Phylo.read(StringIO(newick_text), "newick")


def _which(name: str) -> bool:
    import shutil

    return shutil.which(name) is not None


# -- I/O ----------------------------------------------------------------


def write_newick(tree, path: Path) -> Path:
    Phylo.write(tree, str(path), "newick")
    return path


def read_newick(path: Path):
    return Phylo.read(str(path), "newick")


def tree_summary(tree) -> dict:
    terminals = tree.get_terminals()
    depths = tree.depths()
    return {
        "n_taxa": len(terminals),
        "n_internal_nodes": len(tree.get_nonterminals()),
        "total_branch_length": tree.total_branch_length(),
        "max_depth": max(depths.values()) if depths else 0.0,
        "taxa": [t.name for t in terminals],
    }
