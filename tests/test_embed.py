import pytest

np = pytest.importorskip("numpy")

from bioexplorer.core import BioRecord, SeqType
from bioexplorer.embed import (
    build_kmer_vocabulary,
    build_sequence_space,
    embed_sequences,
    esm_embed,
    kmer_embed,
    kmer_frequency_vector,
    minhash_embed,
    prott5_embed,
    reduce_pca,
    reduce_sequence_space,
    reduce_tsne,
    reduce_umap,
)


def _dna_records(n=8):
    import random

    rng = random.Random(42)
    bases = "ACGT"
    records = []
    base_seq = "".join(rng.choice(bases) for _ in range(60))
    for i in range(n):
        seq = list(base_seq)
        # sprinkle a few mutations so points aren't identical
        for _ in range(3):
            pos = rng.randrange(len(seq))
            seq[pos] = rng.choice(bases)
        records.append(BioRecord(name=f"s{i}", sequence="".join(seq), seq_type=SeqType.DNA))
    return records


def _protein_records(n=6):
    seqs = [
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQ",
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVA",
        "MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVYLLPRRGPRLGVRATRKTSERSQPRGRRQPIPKARRPEGRTWAQPGYPWPLYGNEGCGWAGWLLSPRGSRPSWGPTDPRR",
    ] * 2
    return [BioRecord(name=f"p{i}", sequence=s, seq_type=SeqType.PROTEIN) for i, s in enumerate(seqs[:n])]


def test_build_kmer_vocabulary_size():
    vocab = build_kmer_vocabulary(["A", "C", "G", "T"], k=2)
    assert len(vocab) == 16
    assert "AC" in vocab


def test_kmer_frequency_vector_sums_to_one():
    vocab = build_kmer_vocabulary(["A", "C", "G", "T"], k=2)
    vec = kmer_frequency_vector("ACGTACGT", vocab, k=2)
    assert sum(vec) == pytest.approx(1.0)


def test_kmer_frequency_vector_empty_sequence_is_zero():
    vocab = build_kmer_vocabulary(["A", "C", "G", "T"], k=2)
    vec = kmer_frequency_vector("A", vocab, k=2)  # too short for any 2-mer
    assert sum(vec) == 0.0


def test_kmer_embed_shape():
    records = _dna_records(5)
    vectors = kmer_embed(records, k=2)
    assert vectors.shape == (5, 16)


def test_kmer_embed_too_large_k_raises():
    records = _dna_records(5)
    with pytest.raises(ValueError, match="too large"):
        kmer_embed(records, k=10)


def test_minhash_embed_shape():
    records = _dna_records(5)
    vectors = minhash_embed(records, k=6, num_hashes=32)
    assert vectors.shape == (5, 32)


def test_embed_sequences_dispatch_kmer_and_minhash():
    records = _dna_records(4)
    assert embed_sequences(records, method="kmer", k=2).shape[0] == 4
    assert embed_sequences(records, method="minhash").shape[0] == 4


def test_embed_sequences_unknown_method_raises():
    with pytest.raises(ValueError):
        embed_sequences(_dna_records(4), method="not-a-method")


def test_embed_sequences_protein_only_methods_reject_dna():
    with pytest.raises(ValueError, match="protein"):
        embed_sequences(_dna_records(4), method="esm")


def test_esm_embed_missing_dependency_raises_clear_error():
    with pytest.raises(RuntimeError, match="fair-esm"):
        esm_embed(_protein_records(3))


def test_prott5_embed_missing_dependency_raises_clear_error():
    with pytest.raises(RuntimeError, match="transformers"):
        prott5_embed(_protein_records(3))


def test_reduce_pca_shape():
    records = _dna_records(8)
    vectors = kmer_embed(records, k=2)
    coords = reduce_pca(vectors, n_components=2)
    assert coords.shape == (8, 2)


def test_reduce_pca_component_capped_by_n_samples():
    vectors = np.random.RandomState(0).rand(2, 16)  # only 2 samples
    coords = reduce_pca(vectors, n_components=2)
    assert coords.shape[0] == 2
    assert coords.shape[1] <= 2


def test_reduce_tsne_shape():
    records = _dna_records(10)
    vectors = kmer_embed(records, k=2)
    coords = reduce_tsne(vectors, n_components=2, perplexity=30)
    assert coords.shape == (10, 2)


def test_reduce_tsne_too_few_samples_raises():
    vectors = np.random.RandomState(0).rand(2, 16)
    with pytest.raises(ValueError, match="at least 4"):
        reduce_tsne(vectors)


def test_reduce_umap_shape():
    records = _dna_records(10)
    vectors = kmer_embed(records, k=2)
    coords = reduce_umap(vectors, n_components=2, n_neighbors=15)
    assert coords.shape == (10, 2)


def test_reduce_sequence_space_dispatch():
    records = _dna_records(8)
    vectors = kmer_embed(records, k=2)
    assert reduce_sequence_space(vectors, method="pca").shape == (8, 2)


def test_reduce_sequence_space_unknown_method_raises():
    with pytest.raises(ValueError):
        reduce_sequence_space(np.random.rand(5, 4), method="not-a-method")


def test_build_sequence_space_end_to_end():
    records = _dna_records(8)
    result = build_sequence_space(records, embed_method="kmer", reduce_method="pca", embed_kwargs={"k": 2})
    assert result.coordinates.shape == (8, 2)
    assert result.names == [r.name for r in records]
    assert result.embed_method == "kmer"
    assert result.reduce_method == "pca"


def test_build_sequence_space_too_few_records_raises():
    with pytest.raises(ValueError):
        build_sequence_space(_dna_records(1))
