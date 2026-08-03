import pytest

from bioexplorer.core import BioRecord, SeqType
from bioexplorer.tree import (
    build_distance_tree,
    build_tree_external,
    read_newick,
    tree_summary,
    write_newick,
)

ALIGNED = [
    BioRecord(name="s1", sequence="ACGTACGTACGT", seq_type=SeqType.DNA),
    BioRecord(name="s2", sequence="ACGTACGTACGT", seq_type=SeqType.DNA),
    BioRecord(name="s3", sequence="ACGTACGTAAGT", seq_type=SeqType.DNA),
    BioRecord(name="s4", sequence="TTGTACGTAAGT", seq_type=SeqType.DNA),
]


def test_build_distance_tree_nj_has_all_taxa():
    tree = build_distance_tree(ALIGNED, method="nj")
    summary = tree_summary(tree)
    assert summary["n_taxa"] == 4
    assert set(summary["taxa"]) == {"s1", "s2", "s3", "s4"}


def test_build_distance_tree_upgma_has_all_taxa():
    tree = build_distance_tree(ALIGNED, method="upgma")
    summary = tree_summary(tree)
    assert summary["n_taxa"] == 4


def test_build_distance_tree_unequal_length_raises():
    unaligned = [
        BioRecord(name="s1", sequence="ACGT", seq_type=SeqType.DNA),
        BioRecord(name="s2", sequence="ACGTAA", seq_type=SeqType.DNA),
    ]
    with pytest.raises(ValueError):
        build_distance_tree(unaligned, method="nj")


def test_build_distance_tree_unknown_method_raises():
    with pytest.raises(ValueError):
        build_distance_tree(ALIGNED, method="not-a-method")


def test_build_distance_tree_bootstrap_sets_confidence():
    tree = build_distance_tree(ALIGNED, method="nj", bootstrap=10, seed=1)
    summary = tree_summary(tree)
    assert summary["n_taxa"] == 4
    confidences = [c.confidence for c in tree.get_nonterminals() if c.confidence is not None]
    assert len(confidences) > 0
    assert all(0 <= c <= 100 for c in confidences)


def test_newick_roundtrip(tmp_path):
    tree = build_distance_tree(ALIGNED, method="nj")
    path = tmp_path / "tree.nwk"
    write_newick(tree, path)
    reloaded = read_newick(path)
    assert tree_summary(reloaded)["n_taxa"] == 4
    assert set(tree_summary(reloaded)["taxa"]) == {"s1", "s2", "s3", "s4"}


def test_build_tree_external_unknown_tool_raises():
    with pytest.raises(ValueError):
        build_tree_external(ALIGNED, tool="not-a-tool")


def test_build_tree_external_missing_binary_raises_clear_error():
    with pytest.raises(RuntimeError, match="not found on PATH"):
        build_tree_external(ALIGNED, tool="fasttree")
