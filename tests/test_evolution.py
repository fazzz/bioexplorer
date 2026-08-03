import pytest

from bioexplorer.core import BioRecord, SeqType
from bioexplorer.evolution import (
    bootstrap_tree,
    conservation_summary,
    dn_ds_matrix,
    pairwise_dn_ds,
)

# Two codon-aligned CDSs: identical except one synonymous change (codon 2:
# GGA -> GGC, both Gly) and one nonsynonymous change (codon 4: ATT[Ile] -> ACT[Thr]).
SEQ_A = "ATG" "GGA" "AAA" "ATT" "TAA"
SEQ_B = "ATG" "GGC" "AAA" "ACT" "TAA"


def test_pairwise_dn_ds_ng86_basic():
    result = pairwise_dn_ds(SEQ_A, SEQ_B, "a", "b", method="NG86")
    assert result.method == "NG86"
    assert result.dn > 0  # the Ile->Thr change should register as nonsynonymous
    assert result.ds >= 0


def test_pairwise_dn_ds_identical_sequences_zero_divergence():
    result = pairwise_dn_ds(SEQ_A, SEQ_A, "a", "a2", method="NG86")
    assert result.dn == pytest.approx(0.0)
    assert result.ds == pytest.approx(0.0)
    assert result.omega is None  # dS == 0 -> undefined


def test_pairwise_dn_ds_lwl85_runs():
    seq_a = "ATG" + "GGA" * 20 + "AAA" * 10 + "TAA"
    seq_b = "ATG" + ("GGC" + "GGA" * 19) + ("ACT" + "AAA" * 9) + "TAA"
    result = pairwise_dn_ds(seq_a, seq_b, "a", "b", method="LWL85")
    assert result.method == "LWL85"
    assert result.dn >= 0


def test_pairwise_dn_ds_unequal_length_raises():
    with pytest.raises(ValueError):
        pairwise_dn_ds(SEQ_A, SEQ_A[:-3], "a", "b")


def test_pairwise_dn_ds_not_multiple_of_three_raises():
    with pytest.raises(ValueError, match="multiple of 3"):
        pairwise_dn_ds("ATGA", "ATGC", "a", "b")


def test_pairwise_dn_ds_unknown_method_raises():
    with pytest.raises(ValueError):
        pairwise_dn_ds(SEQ_A, SEQ_B, method="not-a-method")


def test_pairwise_dn_ds_yn00_missing_binary_raises_clear_error():
    with pytest.raises(RuntimeError, match="not found on PATH"):
        pairwise_dn_ds(SEQ_A, SEQ_B, "a", "b", method="YN00")


def test_dn_ds_matrix_all_pairs():
    records = [
        BioRecord(name="s1", sequence=SEQ_A, seq_type=SeqType.DNA),
        BioRecord(name="s2", sequence=SEQ_B, seq_type=SeqType.DNA),
        BioRecord(name="s3", sequence=SEQ_A, seq_type=SeqType.DNA),
    ]
    results = dn_ds_matrix(records, method="NG86")
    assert len(results) == 3  # 3 choose 2
    pairs = {(r.seq_a_id, r.seq_b_id) for r in results}
    assert ("s1", "s2") in pairs and ("s1", "s3") in pairs and ("s2", "s3") in pairs


def test_conservation_summary_reexport():
    records = [
        BioRecord(name="s1", sequence="ACGT", seq_type=SeqType.DNA),
        BioRecord(name="s2", sequence="ACGT", seq_type=SeqType.DNA),
        BioRecord(name="s3", sequence="ACAT", seq_type=SeqType.DNA),
    ]
    summary = conservation_summary(records)
    assert summary["consensus"] == "ACGT"
    assert 0 <= summary["mean_conservation"] <= 1


def test_bootstrap_tree_reexport():
    records = [
        BioRecord(name="s1", sequence="ACGTACGT", seq_type=SeqType.DNA),
        BioRecord(name="s2", sequence="ACGTACGT", seq_type=SeqType.DNA),
        BioRecord(name="s3", sequence="ACGTATGT", seq_type=SeqType.DNA),
        BioRecord(name="s4", sequence="TCGTATGT", seq_type=SeqType.DNA),
    ]
    tree = bootstrap_tree(records, method="nj", replicates=10, seed=1)
    confidences = [c.confidence for c in tree.get_nonterminals() if c.confidence is not None]
    assert len(confidences) > 0
