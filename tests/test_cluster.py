import pytest

from bioexplorer.cluster import (
    Cluster,
    annotate_clusters,
    assign_centroid,
    cluster_cdhit,
    cluster_greedy,
    cluster_hierarchical,
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


try:
    import scipy  # noqa: F401

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

needs_scipy = pytest.mark.skipif(not HAS_SCIPY, reason="requires scipy (the 'cluster' extra)")


@needs_scipy
def test_cluster_hierarchical_groups_similar_separates_unrelated(records):
    clusters = cluster_hierarchical(records, method="kmer", distance_threshold=0.5)
    assert len(clusters) == 2
    sizes = sorted(len(c.members) for c in clusters)
    assert sizes == [1, 2]
    big_cluster = max(clusters, key=lambda c: len(c.members))
    names = {m.name for m in big_cluster.members}
    assert names == {"a1", "a2"}


@needs_scipy
def test_cluster_hierarchical_n_clusters_exact_count(records):
    clusters = cluster_hierarchical(records, n_clusters=2)
    assert len(clusters) == 2
    total_members = sum(len(c.members) for c in clusters)
    assert total_members == len(records)


@needs_scipy
def test_cluster_hierarchical_n_clusters_one_group(records):
    clusters = cluster_hierarchical(records, n_clusters=1)
    assert len(clusters) == 1
    assert len(clusters[0].members) == len(records)


@needs_scipy
def test_cluster_hierarchical_representative_is_longest_member():
    short = BioRecord(name="short", sequence="ACGTACGT", seq_type=SeqType.DNA)
    long_ = BioRecord(name="long", sequence=SEQ_A, seq_type=SeqType.DNA)
    clusters = cluster_hierarchical([short, long_], n_clusters=1)
    assert clusters[0].representative.name == "long"


def test_cluster_hierarchical_empty_input():
    # short-circuits before ever touching scipy -- must pass with or without it
    assert cluster_hierarchical([]) == []


def test_cluster_hierarchical_single_record():
    # likewise short-circuits before scipy is needed
    rec = BioRecord(name="only", sequence=SEQ_A, seq_type=SeqType.DNA)
    clusters = cluster_hierarchical([rec])
    assert len(clusters) == 1
    assert clusters[0].members == [rec]
    assert clusters[0].representative is rec


def test_cluster_hierarchical_both_threshold_and_n_clusters_raises(records):
    # validated before scipy is imported -- must raise with or without it
    with pytest.raises(ValueError, match="either"):
        cluster_hierarchical(records, distance_threshold=0.3, n_clusters=2)


def test_cluster_hierarchical_n_clusters_out_of_range_raises(records):
    with pytest.raises(ValueError, match="between"):
        cluster_hierarchical(records, n_clusters=10)


def test_cluster_hierarchical_unknown_linkage_raises(records):
    with pytest.raises(ValueError, match="linkage"):
        cluster_hierarchical(records, linkage_method="ward")


@needs_scipy
def test_cluster_hierarchical_linkage_methods_all_run(records):
    for linkage_method in ("single", "complete", "average", "weighted"):
        clusters = cluster_hierarchical(records, linkage_method=linkage_method, n_clusters=2)
        assert sum(len(c.members) for c in clusters) == len(records)


@needs_scipy
def test_cluster_hierarchical_minhash_similarity(records):
    clusters = cluster_hierarchical(records, method="minhash", distance_threshold=0.5)
    total_members = sum(len(c.members) for c in clusters)
    assert total_members == len(records)


def test_cluster_hierarchical_missing_scipy_raises_clear_error(records, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "scipy.cluster.hierarchy" or name.startswith("scipy"):
            raise ImportError("simulated: scipy not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="scipy"):
        cluster_hierarchical(records, n_clusters=2)


@needs_scipy
def test_annotate_clusters_works_with_hierarchical(records):
    clusters = cluster_hierarchical(records, distance_threshold=0.5)
    annotate_clusters(clusters, compute_consensus_seqs=True, msa_tool=None)
    for cluster in clusters:
        assert cluster.centroid is not None
        assert cluster.consensus is not None
        for member in cluster.members:
            assert member.get("cluster_id") == cluster.cluster_id


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
