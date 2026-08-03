import pytest

from bioexplorer.core import BioCollection, BioRecord, SeqType
from bioexplorer.similarity import (
    jaccard_similarity,
    kmer_set,
    minhash_signature,
    minhash_similarity,
    search_fast,
    search_similar,
)


def test_kmer_set_basic():
    assert kmer_set("ACGT", k=2) == {"AC", "CG", "GT"}


def test_kmer_set_shorter_than_k():
    assert kmer_set("AC", k=4) == {"AC"}


def test_jaccard_identical_sequences_is_one():
    assert jaccard_similarity("ACGTACGT", "ACGTACGT", k=3) == pytest.approx(1.0)


def test_jaccard_unrelated_sequences_lower_than_related():
    related = jaccard_similarity("ACGTACGTACGT", "ACGTACGTACGA", k=4)
    unrelated = jaccard_similarity("ACGTACGTACGT", "TTTTGGGGCCCC", k=4)
    assert related > unrelated


def test_minhash_signature_deterministic():
    sig1 = minhash_signature("ACGTACGTACGTACGT", k=4, num_hashes=16)
    sig2 = minhash_signature("ACGTACGTACGTACGT", k=4, num_hashes=16)
    assert sig1 == sig2
    assert len(sig1) == 16


def test_minhash_similarity_identical_is_one():
    sig = minhash_signature("ACGTACGTACGTACGTGGGG", k=4, num_hashes=32)
    assert minhash_similarity(sig, sig) == pytest.approx(1.0)


def test_minhash_similarity_related_higher_than_unrelated():
    base = "ACGTACGTACGTACGTGGGGCCCCAAAA"
    close = "ACGTACGTACGTACGTGGGGCCCCAAAT"  # one base changed
    far = "TTTTTTTTTTTTTTTTTTTTTTTTTTTT"
    sig_base = minhash_signature(base, k=4, num_hashes=64)
    sig_close = minhash_signature(close, k=4, num_hashes=64)
    sig_far = minhash_signature(far, k=4, num_hashes=64)
    assert minhash_similarity(sig_base, sig_close) > minhash_similarity(sig_base, sig_far)


@pytest.fixture
def similarity_collection() -> BioCollection:
    c = BioCollection()
    c.add(BioRecord(name="close", sequence="ACGTACGTACGTACGTGGGG", seq_type=SeqType.DNA))
    c.add(BioRecord(name="far", sequence="TTTTGGGGCCCCAAAATTTT", seq_type=SeqType.DNA))
    return c


def test_search_fast_ranks_closer_sequence_higher(similarity_collection):
    hits = search_fast(similarity_collection, "ACGTACGTACGTACGTGGGA", method="kmer", top_n=2)
    assert hits[0].record.name == "close"
    assert hits[0].score >= hits[1].score


def test_search_fast_min_score_filters(similarity_collection):
    hits = search_fast(similarity_collection, "ACGTACGTACGTACGTGGGA", method="kmer", min_score=0.99)
    assert all(h.score >= 0.99 for h in hits)


def test_search_similar_dispatch_kmer(similarity_collection):
    hits = search_similar(similarity_collection, "ACGTACGTACGTACGTGGGA", method="kmer")
    assert len(hits) == 2


def test_search_similar_unknown_method_raises(similarity_collection):
    with pytest.raises(ValueError):
        search_similar(similarity_collection, "ACGT", method="not-a-method")


def test_search_similar_missing_external_tool_raises_clear_error(similarity_collection):
    with pytest.raises(RuntimeError, match="not found on PATH"):
        search_similar(similarity_collection, "ACGT", method="blast")
