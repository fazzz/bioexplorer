import pytest

from bioexplorer.cluster import (
    Cluster,
    annotate_clusters,
    assign_centroid,
    cluster_cdhit,
    cluster_greedy,
    cluster_mmseqs,
    compute_consensus,
)
from bioexplorer.core import BioRecord, SeqType

SEQ_A = "ACGTACGTACGTACGTGGGGCCCCAAAA"
SEQ_A_VARIANT = "ACGTACGTACGTACGTGGGGCCCCAAAT"  # 1 base different from A
SEQ_B = "TTTTTTTTTTTTTTTTTTTTTTTTTTTT"  # unrelated


@pytest.fixture
def records():
    return [
        BioRecord(name="a1", sequence=SEQ_A, seq_type=SeqType.DNA),
        BioRecord(name="a2", sequence=SEQ_A_VARIANT, seq_type=SeqType.DNA),
        BioRecord(name="b1", sequence=SEQ_B, seq_type=SeqType.DNA),
    ]


def test_cluster_greedy_groups_similar_separates_unrelated(records):
    clusters = cluster_greedy(records, method="kmer", threshold=0.5)
    assert len(clusters) == 2
    sizes = sorted(len(c.members) for c in clusters)
    assert sizes == [1, 2]
    big_cluster = max(clusters, key=lambda c: len(c.members))
    names = {m.name for m in big_cluster.members}
    assert names == {"a1", "a2"}


def test_cluster_greedy_representative_is_longest_first_processed():
    # a1 and a2 same length; longest-first order means the first one
    # encountered after sorting becomes the seed representative.
    records = [
        BioRecord(name="short", sequence="ACGTACGT", seq_type=SeqType.DNA),
        BioRecord(name="long", sequence=SEQ_A, seq_type=SeqType.DNA),
    ]
    clusters = cluster_greedy(records, threshold=0.99)
    assert len(clusters) == 2  # unrelated lengths/content, no merge
    reps = {c.representative.name for c in clusters}
    assert reps == {"short", "long"}


def test_cluster_greedy_empty_input():
    assert cluster_greedy([]) == []


def test_assign_centroid_singleton_matches_representative():
    rec = BioRecord(name="only", sequence=SEQ_A, seq_type=SeqType.DNA)
    cluster = Cluster(cluster_id=0, members=[rec], representative=rec)
    centroid = assign_centroid(cluster)
    assert centroid is rec


def test_assign_centroid_picks_most_central_member():
    a = BioRecord(name="a", sequence=SEQ_A, seq_type=SeqType.DNA)
    a_var = BioRecord(name="a_var", sequence=SEQ_A_VARIANT, seq_type=SeqType.DNA)
    outlier = BioRecord(name="outlier", sequence="ACGTACGTACGTACGTGGGGCCCCTTTT", seq_type=SeqType.DNA)
    cluster = Cluster(cluster_id=0, members=[a, a_var, outlier], representative=a)
    centroid = assign_centroid(cluster, method="kmer")
    assert centroid.name in {"a", "a_var"}  # not the outlier


def test_compute_consensus_equal_length_majority_vote():
    members = [
        BioRecord(name="s1", sequence="ACGT", seq_type=SeqType.DNA),
        BioRecord(name="s2", sequence="ACGT", seq_type=SeqType.DNA),
        BioRecord(name="s3", sequence="ACAT", seq_type=SeqType.DNA),
    ]
    cluster = Cluster(cluster_id=0, members=members, representative=members[0])
    consensus = compute_consensus(cluster, msa_tool=None)
    assert consensus == "ACGT"
    assert cluster.consensus_is_approximate is False


def test_compute_consensus_unequal_length_no_msa_tool_falls_back_to_representative():
    members = [
        BioRecord(name="s1", sequence="ACGTACGT", seq_type=SeqType.DNA),
        BioRecord(name="s2", sequence="ACGT", seq_type=SeqType.DNA),
    ]
    cluster = Cluster(cluster_id=0, members=members, representative=members[0])
    consensus = compute_consensus(cluster, msa_tool=None)
    assert consensus == members[0].sequence
    assert cluster.consensus_is_approximate is True


def test_annotate_clusters_tags_members(records):
    clusters = cluster_greedy(records, method="kmer", threshold=0.5)
    annotate_clusters(clusters, compute_consensus_seqs=True, msa_tool=None)
    for cluster in clusters:
        assert cluster.centroid is not None
        assert cluster.consensus is not None
        for member in cluster.members:
            assert member.get("cluster_id") == cluster.cluster_id
            assert f"cluster_{cluster.cluster_id}" in member.tags
        assert "cluster_representative" in cluster.representative.tags


def test_cluster_cdhit_missing_binary_raises_clear_error(records):
    with pytest.raises(RuntimeError, match="not found on PATH"):
        cluster_cdhit(records, seq_type=SeqType.DNA)


def test_cluster_mmseqs_missing_binary_raises_clear_error(records):
    with pytest.raises(RuntimeError, match="not found on PATH"):
        cluster_mmseqs(records)
