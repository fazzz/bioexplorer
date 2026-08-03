import math

import pytest

from bioexplorer.core import BioRecord, SeqType
from bioexplorer.profile import (
    build_profile,
    compute_pfm,
    compute_ppm,
    compute_pssm,
    compute_pwm,
    conservation_score,
    consensus_sequence,
    default_alphabet,
    default_background,
    relative_entropy,
    shannon_entropy,
)

DNA_ALIGNMENT = ["ACGT", "ACGT", "ACGT", "ACAT"]  # position 2 has one variant


def test_compute_pfm_counts():
    pfm = compute_pfm(DNA_ALIGNMENT, default_alphabet(SeqType.DNA))
    assert pfm[0] == {"A": 4, "C": 0, "G": 0, "T": 0, "gap": 0}
    assert pfm[2]["G"] == 3
    assert pfm[2]["A"] == 1


def test_compute_pfm_rejects_unequal_lengths():
    with pytest.raises(ValueError):
        compute_pfm(["ACGT", "ACG"], default_alphabet(SeqType.DNA))


def test_compute_ppm_sums_to_one():
    alphabet = default_alphabet(SeqType.DNA)
    pfm = compute_pfm(DNA_ALIGNMENT, alphabet)
    ppm = compute_ppm(pfm, alphabet, pseudocount=0.5)
    for row in ppm:
        assert sum(row.values()) == pytest.approx(1.0)


def test_compute_pwm_zero_at_uniform_background_when_ppm_matches():
    alphabet = default_alphabet(SeqType.DNA)
    background = default_background(SeqType.DNA)
    # a fully-uniform column should give near-zero log-odds everywhere
    uniform_ppm = [{s: 0.25 for s in alphabet}]
    pwm = compute_pwm(uniform_ppm, background)
    for score in pwm[0].values():
        assert score == pytest.approx(0.0, abs=1e-9)


def test_compute_pssm_scales_pwm():
    pwm = [{"A": 1.0, "C": -1.0, "G": 0.0, "T": 2.0}]
    pssm = compute_pssm(pwm, scale=100)
    assert pssm[0]["A"] == 100
    assert pssm[0]["C"] == -100
    assert pssm[0]["T"] == 200


def test_consensus_sequence_majority_vote():
    alphabet = default_alphabet(SeqType.DNA)
    pfm = compute_pfm(DNA_ALIGNMENT, alphabet)
    consensus = consensus_sequence(pfm, alphabet)
    assert consensus == "ACGT"


def test_shannon_entropy_zero_for_fully_conserved_column():
    alphabet = default_alphabet(SeqType.DNA)
    ppm = [{"A": 1.0, "C": 0.0, "G": 0.0, "T": 0.0}]
    h = shannon_entropy(ppm)
    assert h[0] == pytest.approx(0.0)


def test_shannon_entropy_max_for_uniform_column():
    alphabet = default_alphabet(SeqType.DNA)
    ppm = [{s: 0.25 for s in alphabet}]
    h = shannon_entropy(ppm)
    assert h[0] == pytest.approx(math.log2(4))


def test_conservation_score_range_and_extremes():
    alphabet = default_alphabet(SeqType.DNA)
    conserved_ppm = [{"A": 1.0, "C": 0.0, "G": 0.0, "T": 0.0}]
    uniform_ppm = [{s: 0.25 for s in alphabet}]
    assert conservation_score(conserved_ppm, alphabet)[0] == pytest.approx(1.0)
    assert conservation_score(uniform_ppm, alphabet)[0] == pytest.approx(0.0)


def test_relative_entropy_zero_when_matches_background():
    alphabet = default_alphabet(SeqType.DNA)
    background = default_background(SeqType.DNA)
    ppm = [dict(background)]
    d = relative_entropy(ppm, background)
    assert d[0] == pytest.approx(0.0, abs=1e-9)


def test_relative_entropy_positive_when_diverges_from_background():
    alphabet = default_alphabet(SeqType.DNA)
    background = default_background(SeqType.DNA)
    ppm = [{"A": 0.97, "C": 0.01, "G": 0.01, "T": 0.01}]
    d = relative_entropy(ppm, background)
    assert d[0] > 1.0


def test_build_profile_end_to_end():
    records = [
        BioRecord(name=f"s{i}", sequence=seq, seq_type=SeqType.DNA)
        for i, seq in enumerate(DNA_ALIGNMENT)
    ]
    profile = build_profile(records)
    assert profile.length == 4
    assert profile.n_sequences == 4
    assert profile.consensus == "ACGT"
    assert len(profile.shannon_entropy) == 4
    assert len(profile.pssm) == 4


def test_build_profile_requires_records():
    with pytest.raises(ValueError):
        build_profile([])
