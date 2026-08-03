from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")

from bioexplorer.core import BioRecord, SeqType
from bioexplorer.tree import build_distance_tree
from bioexplorer import viz

ALIGNED = [
    BioRecord(name="s1", sequence="ACGTACGTACGT", seq_type=SeqType.DNA),
    BioRecord(name="s2", sequence="ACGTACGTACGT", seq_type=SeqType.DNA),
    BioRecord(name="s3", sequence="ACGTACGTAAGT", seq_type=SeqType.DNA),
    BioRecord(name="s4", sequence="TTGTACGTAAGT", seq_type=SeqType.DNA),
]


def test_plot_alignment_viewer_creates_file(tmp_path):
    out = tmp_path / "aln.png"
    result = viz.plot_alignment_viewer(ALIGNED, out)
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_alignment_viewer_color_by_conservation(tmp_path):
    out = tmp_path / "aln_cons.png"
    viz.plot_alignment_viewer(ALIGNED, out, color_by="conservation")
    assert out.exists() and out.stat().st_size > 0


def test_plot_alignment_viewer_unequal_length_raises(tmp_path):
    bad = [
        BioRecord(name="s1", sequence="ACGT", seq_type=SeqType.DNA),
        BioRecord(name="s2", sequence="ACGTA", seq_type=SeqType.DNA),
    ]
    with pytest.raises(ValueError):
        viz.plot_alignment_viewer(bad, tmp_path / "out.png")


def test_plot_alignment_viewer_truncates_long_alignment(tmp_path):
    long_seq = "ACGT" * 100  # 400 positions
    records = [
        BioRecord(name="s1", sequence=long_seq, seq_type=SeqType.DNA),
        BioRecord(name="s2", sequence=long_seq, seq_type=SeqType.DNA),
    ]
    out = tmp_path / "long.png"
    viz.plot_alignment_viewer(records, out, max_positions=50)
    assert out.exists() and out.stat().st_size > 0


def test_plot_tree_creates_file(tmp_path):
    tree = build_distance_tree(ALIGNED, method="nj")
    out = tmp_path / "tree.png"
    result = viz.plot_tree(tree, out)
    assert result == out
    assert out.exists() and out.stat().st_size > 0


def test_plot_tree_with_bootstrap_confidence(tmp_path):
    tree = build_distance_tree(ALIGNED, method="nj", bootstrap=10, seed=1)
    out = tmp_path / "tree_bs.png"
    viz.plot_tree(tree, out, show_confidence=True)
    assert out.exists() and out.stat().st_size > 0
