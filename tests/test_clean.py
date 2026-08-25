import pytest

from bioexplorer.clean import (
    ambiguous_fraction,
    clean_records,
    strip_gaps,
    trim_adapter,
    trim_ambiguous_ends,
    trim_by_quality,
)
from bioexplorer.core import BioRecord, SeqType


# -- pure helper functions ---------------------------------------------------


def test_strip_gaps_removes_dashes_and_dots():
    assert strip_gaps("AC-GT.AC") == "ACGTAC"


def test_strip_gaps_no_gaps_is_noop():
    assert strip_gaps("ACGT") == "ACGT"


def test_trim_ambiguous_ends_dna():
    seq, lead, trail = trim_ambiguous_ends("NNACGTNN", SeqType.DNA)
    assert seq == "ACGT"
    assert lead == 2
    assert trail == 2


def test_trim_ambiguous_ends_leaves_internal_ambiguous():
    seq, lead, trail = trim_ambiguous_ends("ACNGT", SeqType.DNA)
    assert seq == "ACNGT"
    assert lead == 0
    assert trail == 0


def test_trim_ambiguous_ends_protein_uses_x():
    seq, lead, trail = trim_ambiguous_ends("XXMKTAYXX", SeqType.PROTEIN)
    assert seq == "MKTAY"
    assert lead == 2
    assert trail == 2


def test_trim_ambiguous_ends_all_ambiguous():
    seq, lead, trail = trim_ambiguous_ends("NNNN", SeqType.DNA)
    assert seq == ""
    assert lead == 4
    assert trail == 0  # start meets end before any trailing trim happens


def test_ambiguous_fraction_basic():
    assert ambiguous_fraction("NNAACC", SeqType.DNA) == pytest.approx(2 / 6)


def test_ambiguous_fraction_empty_sequence_is_zero():
    assert ambiguous_fraction("", SeqType.DNA) == 0.0


def test_trim_adapter_5_prime():
    seq, lead, trail = trim_adapter("AAAACGTACGT", "AAAA", end="5")
    assert seq == "CGTACGT"
    assert lead == 4
    assert trail == 0


def test_trim_adapter_3_prime():
    seq, lead, trail = trim_adapter("ACGTACGTTTTT", "TTTT", end="3")
    assert seq == "ACGTACGT"
    assert lead == 0
    assert trail == 4


def test_trim_adapter_both_ends():
    seq, lead, trail = trim_adapter("AAAAACGTTTTT", "AAAA", end="both")
    # only the 5' adapter matches here; 3' end doesn't match "AAAA"
    assert seq == "ACGTTTTT"
    assert lead == 4
    assert trail == 0


def test_trim_adapter_no_match_is_noop():
    seq, lead, trail = trim_adapter("ACGTACGT", "GGGG", end="both")
    assert seq == "ACGTACGT"
    assert lead == 0
    assert trail == 0


def test_trim_by_quality_trims_low_quality_ends():
    seq = "ACGTACGT"
    quality = [5, 5, 40, 40, 40, 40, 5, 5]
    trimmed_seq, trimmed_q, lead, trail = trim_by_quality(seq, quality, min_quality=20, window=1)
    assert trimmed_seq == "GTAC"
    assert trimmed_q == [40, 40, 40, 40]
    assert lead == 2
    assert trail == 2


def test_trim_by_quality_all_high_quality_is_noop():
    seq = "ACGT"
    quality = [40, 40, 40, 40]
    trimmed_seq, trimmed_q, lead, trail = trim_by_quality(seq, quality, min_quality=20, window=1)
    assert trimmed_seq == seq
    assert lead == 0 and trail == 0


def test_trim_by_quality_empty_quality_is_noop():
    seq, q, lead, trail = trim_by_quality("ACGT", [], min_quality=20)
    assert seq == "ACGT"
    assert lead == 0 and trail == 0


# -- clean_records: individual operations ------------------------------------


def test_clean_records_dedup_sequence_keeps_first():
    records = [
        BioRecord(name="a", sequence="ACGT", seq_type=SeqType.DNA),
        BioRecord(name="b", sequence="ACGT", seq_type=SeqType.DNA),  # dup of a
        BioRecord(name="c", sequence="TTTT", seq_type=SeqType.DNA),
    ]
    report = clean_records(records, dedup_sequence=True)
    assert report.kept == 2
    assert report.dropped_duplicate_sequence == 1
    assert [r.name for r in report.kept_records] == ["a", "c"]


def test_clean_records_dedup_sequence_case_insensitive():
    records = [
        BioRecord(name="a", sequence="acgt", seq_type=SeqType.DNA),
        BioRecord(name="b", sequence="ACGT", seq_type=SeqType.DNA),
    ]
    report = clean_records(records, dedup_sequence=True)
    assert report.kept == 1
    assert report.dropped_duplicate_sequence == 1


def test_clean_records_dedup_name_keeps_first():
    records = [
        BioRecord(name="dup", sequence="ACGT", seq_type=SeqType.DNA),
        BioRecord(name="dup", sequence="TTTT", seq_type=SeqType.DNA),
    ]
    report = clean_records(records, dedup_name=True)
    assert report.kept == 1
    assert report.dropped_duplicate_name == 1
    assert report.kept_records[0].sequence == "ACGT"


def test_clean_records_strip_gaps():
    records = [BioRecord(name="a", sequence="AC-GT.AC", seq_type=SeqType.DNA)]
    report = clean_records(records, strip_gaps_flag=True)
    assert report.trimmed_gaps == 1
    assert report.kept_records[0].sequence == "ACGTAC"


def test_clean_records_trim_ambiguous_ends():
    records = [BioRecord(name="a", sequence="NNACGTNN", seq_type=SeqType.DNA)]
    report = clean_records(records, trim_ambiguous=True)
    assert report.trimmed_ambiguous_ends == 1
    assert report.kept_records[0].sequence == "ACGT"


def test_clean_records_max_ambiguous_fraction_drops():
    records = [
        BioRecord(name="clean", sequence="ACGTACGT", seq_type=SeqType.DNA),
        BioRecord(name="dirty", sequence="NNNNACGT", seq_type=SeqType.DNA),
    ]
    report = clean_records(records, max_ambiguous_fraction=0.3)
    assert report.kept == 1
    assert report.dropped_ambiguous == 1
    assert report.kept_records[0].name == "clean"


def test_clean_records_length_filters():
    records = [
        BioRecord(name="short", sequence="AC", seq_type=SeqType.DNA),
        BioRecord(name="mid", sequence="ACGTACGT", seq_type=SeqType.DNA),
        BioRecord(name="long", sequence="ACGT" * 10, seq_type=SeqType.DNA),
    ]
    report = clean_records(records, min_length=4, max_length=20)
    assert report.kept == 1
    assert report.dropped_length == 2
    assert report.kept_records[0].name == "mid"


def test_clean_records_adapter_trim():
    records = [BioRecord(name="a", sequence="AAAACGTACGT", seq_type=SeqType.DNA)]
    report = clean_records(records, adapter="AAAA", adapter_end="5")
    assert report.trimmed_adapter == 1
    assert report.kept_records[0].sequence == "CGTACGT"


def test_clean_records_drop_empty_after_trim():
    records = [BioRecord(name="allgap", sequence="----", seq_type=SeqType.DNA)]
    report = clean_records(records, strip_gaps_flag=True)
    assert report.dropped_empty == 1
    assert report.kept == 0


def test_clean_records_no_operations_keeps_everything_unchanged():
    records = [
        BioRecord(name="a", sequence="ACGT", seq_type=SeqType.DNA),
        BioRecord(name="b", sequence="ACGT", seq_type=SeqType.DNA),
    ]
    report = clean_records(records)
    assert report.kept == 2
    assert [r.sequence for r in report.kept_records] == ["ACGT", "ACGT"]


# -- clean_records: quality handling ------------------------------------------


def test_clean_records_quality_trim():
    rec = BioRecord(name="read1", sequence="ACGTACGT", seq_type=SeqType.DNA, quality=[5, 5, 40, 40, 40, 40, 5, 5])
    report = clean_records([rec], min_quality=20, quality_window=1)
    assert report.trimmed_quality == 1
    kept = report.kept_records[0]
    assert kept.sequence == "GTAC"
    assert kept.quality == [40, 40, 40, 40]
    assert len(kept.sequence) == len(kept.quality)


def test_clean_records_quality_stays_aligned_through_multiple_steps():
    # gap-stripping then quality-trimming: quality must track the sequence
    # through both transformations, not just the last one. Low quality is
    # at the (post-gap-strip) leading edge, where end-trimming is expected
    # to act -- an internal dip would correctly be left alone.
    rec = BioRecord(
        name="read1", sequence="-ACGTACGT", seq_type=SeqType.DNA,
        quality=[0, 5, 5, 40, 40, 40, 40, 40, 40],
    )
    report = clean_records([rec], strip_gaps_flag=True, min_quality=20, quality_window=1)
    kept = report.kept_records[0]
    assert len(kept.sequence) == len(kept.quality)
    assert kept.sequence == "GTACGT"
    assert kept.quality == [40, 40, 40, 40, 40, 40]


def test_clean_records_record_without_quality_unaffected_by_min_quality():
    rec = BioRecord(name="a", sequence="ACGTACGT", seq_type=SeqType.DNA)  # no quality (e.g. FASTA import)
    report = clean_records([rec], min_quality=30)
    assert report.trimmed_quality == 0
    assert report.kept_records[0].sequence == "ACGTACGT"


# -- clean_records: combined pipeline -----------------------------------------


def test_clean_records_combined_pipeline_real_world_shape():
    records = [
        BioRecord(name="good", sequence="ACGTACGTACGT", seq_type=SeqType.DNA),
        BioRecord(name="dup_of_good", sequence="ACGTACGTACGT", seq_type=SeqType.DNA),
        BioRecord(name="too_ambiguous", sequence="NNNNNNNNNNNN", seq_type=SeqType.DNA),
        BioRecord(name="too_short", sequence="AC", seq_type=SeqType.DNA),
        BioRecord(name="needs_trim", sequence="NNACGTACGTNN", seq_type=SeqType.DNA),
    ]
    report = clean_records(
        records,
        dedup_sequence=True,
        trim_ambiguous=True,
        max_ambiguous_fraction=0.5,
        min_length=4,
    )
    kept_names = {r.name for r in report.kept_records}
    assert kept_names == {"good", "needs_trim"}
    assert report.kept_records[1].sequence == "ACGTACGT"  # needs_trim, trimmed
