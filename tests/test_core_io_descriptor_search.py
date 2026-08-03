from pathlib import Path

import pytest

from bioexplorer.core import BioCollection, BioRecord, SeqType
from bioexplorer.descriptor import annotate_descriptors, compute_descriptors
from bioexplorer.io import load_collection, write_collection
from bioexplorer.search import (
    filter_by_length,
    filter_by_metadata_range,
    filter_by_motif,
    filter_by_name,
    filter_by_tag,
    run_filters,
)

DNA_MULTIPLE_OF_3 = "ATGGCGATCGATCGATCGGCTAGCTAGCTAGCATGCATGCATGCTAGCTAG"  # len 51 wait
PROTEIN_SEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEV"


@pytest.fixture
def sample_collection() -> BioCollection:
    c = BioCollection()
    c.add(BioRecord(name="dna1", sequence=DNA_MULTIPLE_OF_3, seq_type=SeqType.DNA))
    c.add(BioRecord(name="dna2", sequence="ATGCGT", seq_type=SeqType.DNA))
    c.add(BioRecord(name="prot1", sequence=PROTEIN_SEQ, seq_type=SeqType.PROTEIN))
    return c


# -- core ---------------------------------------------------------------


def test_seqtype_guess():
    assert SeqType.guess("ACGTACGT") == SeqType.DNA
    assert SeqType.guess("ACGUACGU") == SeqType.RNA
    assert SeqType.guess("MKTAYIAK") == SeqType.PROTEIN


def test_biorecord_upper_and_type_coercion():
    r = BioRecord(name="x", sequence="acgt", seq_type="dna")
    assert r.sequence == "ACGT"
    assert r.seq_type == SeqType.DNA
    assert r.length == 4


def test_collection_add_duplicate_id_raises():
    c = BioCollection()
    r = BioRecord(name="x", sequence="ACGT", seq_type=SeqType.DNA, seq_id="fixed")
    c.add(r)
    with pytest.raises(ValueError):
        c.add(BioRecord(name="y", sequence="ACGT", seq_type=SeqType.DNA, seq_id="fixed"))


def test_collection_type_counts(sample_collection):
    counts = sample_collection.type_counts()
    assert counts == {"dna": 2, "protein": 1}


# -- io -------------------------------------------------------------------


def test_fasta_roundtrip(tmp_path: Path, sample_collection: BioCollection):
    out = tmp_path / "out.fasta"
    write_collection(sample_collection, out, "fasta")
    reloaded = load_collection([out], fmt="fasta")
    assert len(reloaded) == len(sample_collection)
    names = sorted(r.name for r in reloaded)
    assert names == ["dna1", "dna2", "prot1"]


def test_csv_export(tmp_path: Path, sample_collection: BioCollection):
    out = tmp_path / "out.csv"
    write_collection(sample_collection, out, "csv")
    text = out.read_text()
    assert "seq_id" in text.splitlines()[0]
    assert text.count("\n") >= 4  # header + 3 records (+ trailing newline)


def test_directory_import(tmp_path: Path):
    (tmp_path / "a.fasta").write_text(">a\nACGTACGT\n")
    (tmp_path / "b.fasta").write_text(">b\nACGTTTTT\n")
    (tmp_path / "notes.txt").write_text("not a sequence file")
    c = load_collection([tmp_path])
    assert len(c) == 2


# -- descriptor -------------------------------------------------------------


def test_dna_descriptor_basic():
    r = BioRecord(name="x", sequence="GCGCAAAA", seq_type=SeqType.DNA)
    d = compute_descriptors(r)
    assert d["length"] == 8
    assert d["gc_percent"] == pytest.approx(50.0)
    assert d["at_percent"] == pytest.approx(50.0)


def test_dna_codon_usage_only_when_multiple_of_3():
    r3 = BioRecord(name="x", sequence="ATGATG", seq_type=SeqType.DNA)
    d3 = compute_descriptors(r3)
    assert d3["codon_usage"] is not None
    assert d3["codon_usage"]["ATG"] == pytest.approx(1.0)

    r_not3 = BioRecord(name="y", sequence="ATGA", seq_type=SeqType.DNA)
    d_not3 = compute_descriptors(r_not3)
    assert d_not3["codon_usage"] is None
    assert d_not3["gc3_percent"] is None


def test_protein_descriptor_fields():
    r = BioRecord(name="p", sequence=PROTEIN_SEQ, seq_type=SeqType.PROTEIN)
    d = compute_descriptors(r)
    assert d["length"] == len(PROTEIN_SEQ)
    assert d["molecular_weight"] > 0
    assert 0 <= d["pi"] <= 14
    assert -5 <= d["gravy"] <= 5


def test_annotate_descriptors_mutates_metadata():
    r = BioRecord(name="p", sequence=PROTEIN_SEQ, seq_type=SeqType.PROTEIN)
    annotate_descriptors(r)
    assert "descriptor" in r.metadata
    assert r.metadata["descriptor"]["length"] == len(PROTEIN_SEQ)


# -- search ------------------------------------------------------------------


def test_filter_by_name_substring(sample_collection):
    result = filter_by_name(sample_collection, "dna")
    assert {r.name for r in result} == {"dna1", "dna2"}


def test_filter_by_name_regex(sample_collection):
    result = filter_by_name(sample_collection, r"^prot\d$", regex=True)
    assert {r.name for r in result} == {"prot1"}


def test_filter_by_length(sample_collection):
    result = filter_by_length(sample_collection, min_len=10)
    assert "dna2" not in {r.name for r in result}


def test_filter_by_motif_regex(sample_collection):
    result = filter_by_motif(sample_collection, "GCG.T")
    assert "dna1" in {r.name for r in result}


def test_filter_by_tag(sample_collection):
    rec = next(r for r in sample_collection if r.name == "dna1")
    rec.add_tag("interesting")
    result = filter_by_tag(sample_collection, "interesting")
    assert {r.name for r in result} == {"dna1"}


def test_filter_by_metadata_range_after_descriptor(sample_collection):
    for rec in sample_collection:
        annotate_descriptors(rec)
    result = filter_by_metadata_range(sample_collection, "descriptor.gc_percent", min_value=50)
    assert len(result) >= 1


def test_run_filters_combination(sample_collection):
    for rec in sample_collection:
        annotate_descriptors(rec)
    result = run_filters(sample_collection, seq_type="dna", min_length=5)
    assert all(r.seq_type == SeqType.DNA for r in result)
    assert all(r.length >= 5 for r in result)
