import pytest

from bioexplorer.core import BioCollection, BioRecord, SeqType
from bioexplorer.descriptor import annotate_descriptors
from bioexplorer.search import (
    filter_by_id_exclude,
    filter_by_metadata_not_equals,
    filter_by_motif_absent,
    filter_by_tag_absent,
    run_filters,
)


@pytest.fixture
def collection():
    c = BioCollection()
    a = BioRecord(name="a", sequence="ACGTACGT", seq_type=SeqType.DNA)
    a.add_tag("bridged")
    b = BioRecord(name="b", sequence="ACGTAAAA", seq_type=SeqType.DNA)
    b.add_tag("open_chain")
    c_rec = BioRecord(name="c", sequence="TTTTTTTT", seq_type=SeqType.DNA)
    c_rec.add_tag("bridged")
    for r in (a, b, c_rec):
        c.add(r)
    return c


def test_filter_by_tag_absent(collection):
    result = filter_by_tag_absent(collection, "bridged")
    assert {r.name for r in result} == {"b"}


def test_filter_by_id_exclude(collection):
    target = next(r for r in collection if r.name == "a")
    result = filter_by_id_exclude(collection, [target.seq_id])
    assert {r.name for r in result} == {"b", "c"}


def test_filter_by_id_exclude_multiple(collection):
    ids = [r.seq_id for r in collection if r.name in ("a", "b")]
    result = filter_by_id_exclude(collection, ids)
    assert {r.name for r in result} == {"c"}


def test_filter_by_motif_absent_regex(collection):
    result = filter_by_motif_absent(collection, "ACGT", regex=False)
    assert {r.name for r in result} == {"c"}


def test_filter_by_metadata_not_equals(collection):
    for rec in collection:
        annotate_descriptors(rec)
    # all three are dna length 8, gc_percent differs; exclude a's exact value
    a_gc = next(r for r in collection if r.name == "a").metadata["descriptor"]["gc_percent"]
    result = filter_by_metadata_not_equals(collection, "descriptor.gc_percent", str(a_gc))
    assert "a" not in {r.name for r in result}


def test_filter_by_metadata_not_equals_missing_field_passes(collection):
    # records with no value at all for the field count as "not equal"
    result = filter_by_metadata_not_equals(collection, "descriptor.gc_percent", "50.0")
    assert len(result) == 3


def test_run_filters_exclude_tag(collection):
    result = run_filters(collection, exclude_tag="bridged")
    assert {r.name for r in result} == {"b"}


def test_run_filters_exclude_motif(collection):
    result = run_filters(collection, exclude_motif="ACGT", )
    # exclude_motif uses regex by default in run_filters via filter_by_motif_absent(regex=True default arg is regex=True there too)
    assert "c" in {r.name for r in result}


def test_run_filters_exclude_ids(collection):
    target = next(r for r in collection if r.name == "b")
    result = run_filters(collection, exclude_ids=[target.seq_id])
    assert {r.name for r in result} == {"a", "c"}


def test_run_filters_field_not_equals(collection):
    for rec in collection:
        annotate_descriptors(rec)
    a_gc = next(r for r in collection if r.name == "a").metadata["descriptor"]["gc_percent"]
    result = run_filters(collection, field="descriptor.gc_percent", field_not_equals=str(a_gc))
    assert "a" not in {r.name for r in result}


def test_run_filters_combined_positive_and_negative(collection):
    # require dna type, but exclude the 'bridged' tag
    result = run_filters(collection, seq_type="dna", exclude_tag="bridged")
    assert {r.name for r in result} == {"b"}
