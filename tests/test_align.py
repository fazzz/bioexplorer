import pytest

from bioexplorer.align import format_alignment, multiple_align, pairwise_align
from bioexplorer.core import BioRecord, SeqType


def test_pairwise_global_identical_sequences():
    result = pairwise_align("ACGTACGT", "ACGTACGT", mode="global", seq_type=SeqType.DNA)
    assert result.aligned_a == "ACGTACGT"
    assert result.aligned_b == "ACGTACGT"
    assert result.score > 0


def test_pairwise_global_introduces_gap_for_indel():
    result = pairwise_align("ACGTACGT", "ACGTCGT", mode="global", seq_type=SeqType.DNA)
    assert "-" in result.aligned_a or "-" in result.aligned_b
    assert len(result.aligned_a) == len(result.aligned_b)


def test_pairwise_local_finds_shared_subsequence():
    result = pairwise_align("AAAAACGTACGTAAAA", "GGGGACGTACGTGGGG", mode="local", seq_type=SeqType.DNA)
    assert "ACGTACGT" in result.aligned_a.replace("-", "")


def test_pairwise_protein_uses_blosum62():
    result = pairwise_align("MKTAYIAK", "MKTAYIAK", mode="global", seq_type=SeqType.PROTEIN)
    assert result.score > 0
    assert result.aligned_a == "MKTAYIAK"


def test_pairwise_invalid_mode_raises():
    with pytest.raises(ValueError):
        pairwise_align("ACGT", "ACGT", mode="banana", seq_type=SeqType.DNA)


def test_format_alignment_contains_score_and_sequences():
    result = pairwise_align("ACGTACGT", "ACGTACGT", mode="global", seq_type=SeqType.DNA, target_id="a", query_id="b")
    text = format_alignment(result)
    assert "score=" in text
    assert "a" in text and "b" in text
    assert "ACGTACGT" in text.replace("\n", "")


def test_multiple_align_requires_at_least_two_records():
    recs = [BioRecord(name="only", sequence="ACGT", seq_type=SeqType.DNA)]
    with pytest.raises(ValueError):
        multiple_align(recs, tool="mafft")


def test_multiple_align_unknown_tool_raises():
    recs = [
        BioRecord(name="a", sequence="ACGT", seq_type=SeqType.DNA),
        BioRecord(name="b", sequence="ACGA", seq_type=SeqType.DNA),
    ]
    with pytest.raises(ValueError):
        multiple_align(recs, tool="not-a-tool")


def test_multiple_align_missing_binary_raises_clear_error():
    recs = [
        BioRecord(name="a", sequence="ACGT", seq_type=SeqType.DNA),
        BioRecord(name="b", sequence="ACGA", seq_type=SeqType.DNA),
    ]
    with pytest.raises(RuntimeError, match="not found on PATH"):
        multiple_align(recs, tool="mafft")
