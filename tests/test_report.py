import pytest

from bioexplorer.core import BioRecord, SeqType
from bioexplorer.report import build_report, resolve_axis_value


def _rec(name, seq_type, tags=(), metadata=None):
    r = BioRecord(name=name, sequence="ACGT", seq_type=seq_type)
    for t in tags:
        r.add_tag(t)
    if metadata:
        for k, v in metadata.items():
            r.set(k, v)
    return r


@pytest.fixture
def records():
    return [
        _rec("s1", SeqType.DNA, tags=["cluster_0", "cluster_representative"]),
        _rec("s2", SeqType.DNA, tags=["cluster_0"]),
        _rec("s3", SeqType.PROTEIN, tags=["cluster_1"]),
        _rec("s4", SeqType.PROTEIN, tags=["cluster_1", "signal_peptide"]),
        _rec("s5", SeqType.PROTEIN, tags=[]),
    ]


def test_resolve_axis_value_type():
    r = _rec("x", SeqType.PROTEIN)
    assert resolve_axis_value(r, "type") == "protein"


def test_resolve_axis_value_tag_present():
    r = _rec("x", SeqType.DNA, tags=["has_orf"])
    assert resolve_axis_value(r, "tag:has_orf") == "yes"


def test_resolve_axis_value_tag_absent():
    r = _rec("x", SeqType.DNA)
    assert resolve_axis_value(r, "tag:has_orf") == "no"


def test_resolve_axis_value_tag_prefix_match():
    r = _rec("x", SeqType.DNA, tags=["cluster_3", "cluster_representative"])
    assert resolve_axis_value(r, "tag_prefix:cluster_") == "3,representative"


def test_resolve_axis_value_tag_prefix_no_match():
    r = _rec("x", SeqType.DNA, tags=["signal_peptide"])
    assert resolve_axis_value(r, "tag_prefix:cluster_") == "(none)"


def test_resolve_axis_value_field_categorical():
    r = _rec("x", SeqType.DNA, metadata={"cluster_id": 2})
    assert resolve_axis_value(r, "field:cluster_id") == "2"


def test_resolve_axis_value_field_missing():
    r = _rec("x", SeqType.DNA)
    assert resolve_axis_value(r, "field:cluster_id") == "(none)"


def test_resolve_axis_value_field_binned():
    r = _rec("x", SeqType.DNA, metadata={"descriptor": {"gc_percent": 42.3}})
    assert resolve_axis_value(r, "field:descriptor.gc_percent:bin10") == "[40,50)"


def test_resolve_axis_value_field_binned_negative_and_zero_edges():
    r0 = _rec("x", SeqType.DNA, metadata={"v": 0.0})
    r1 = _rec("y", SeqType.DNA, metadata={"v": 9.999})
    assert resolve_axis_value(r0, "field:v:bin10") == "[0,10)"
    assert resolve_axis_value(r1, "field:v:bin10") == "[0,10)"


def test_resolve_axis_value_invalid_bin_width_raises():
    r = _rec("x", SeqType.DNA, metadata={"v": 5})
    with pytest.raises(ValueError):
        resolve_axis_value(r, "field:v:binabc")


def test_resolve_axis_value_unknown_spec_raises():
    r = _rec("x", SeqType.DNA)
    with pytest.raises(ValueError):
        resolve_axis_value(r, "bogus:thing")


def test_build_report_single_axis(records):
    rows = build_report(records, ["type"])
    counts = {row["type"]: row["count"] for row in rows}
    assert counts == {"dna": 2, "protein": 3}


def test_build_report_two_axes_crosstab(records):
    rows = build_report(records, ["type", "tag_prefix:cluster_"])
    lookup = {(row["type"], row["tag_prefix:cluster_"]): row["count"] for row in rows}
    # s1 has both cluster_0 and cluster_representative -> combined category
    assert lookup[("dna", "0,representative")] == 1
    assert lookup[("dna", "0")] == 1
    assert lookup[("protein", "1")] == 2
    assert lookup[("protein", "(none)")] == 1


def test_build_report_three_axes(records):
    rows = build_report(records, ["type", "tag:signal_peptide", "tag_prefix:cluster_"])
    assert all(len(row) == 4 for row in rows)  # 3 axes + count
    total = sum(row["count"] for row in rows)
    assert total == len(records)


def test_build_report_no_axes_raises(records):
    with pytest.raises(ValueError):
        build_report(records, [])


def test_build_report_empty_records_returns_empty():
    assert build_report([], ["type"]) == []


def test_build_report_rows_sorted_stably(records):
    rows = build_report(records, ["type"])
    keys = [row["type"] for row in rows]
    assert keys == sorted(keys)
