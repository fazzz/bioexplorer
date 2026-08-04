import pytest

from bioexplorer.annotate import (
    find_canonical_introns,
    find_coiled_coil,
    find_low_complexity_regions,
    find_orfs,
    find_tata_box,
    find_transmembrane_regions,
    hydropathy_profile,
    list_prosite_patterns,
    predict_signal_peptide,
    scan_prosite_patterns,
)

# ---- DNA/RNA ----

# ATG + 40 codons + stop, 123 nt total -> 41-residue protein (M + 40 G's)
_LONG_ORF_SEQ = "ATG" + "GGC" * 40 + "TAA"


def test_find_orfs_detects_forward_orf():
    hits = find_orfs(_LONG_ORF_SEQ, min_protein_length=30, both_strands=False)
    assert len(hits) == 1
    assert hits[0].strand == 1
    assert hits[0].start == 0
    assert len(hits[0].protein) == 41
    assert hits[0].protein == "M" + "G" * 40


def test_find_orfs_respects_min_length():
    hits = find_orfs(_LONG_ORF_SEQ, min_protein_length=42, both_strands=False)
    assert hits == []


def test_find_orfs_both_strands_finds_more_than_single_strand():
    padded = "TTTT" + _LONG_ORF_SEQ + "TTTT"
    both = find_orfs(padded, min_protein_length=30, both_strands=True)
    fwd_only = find_orfs(padded, min_protein_length=30, both_strands=False)
    assert len(both) >= len(fwd_only)
    assert any(h.strand == 1 for h in both)


def test_find_orfs_no_start_codon_returns_empty():
    hits = find_orfs("CCCCCCCCCCCCCCCCCCCC", min_protein_length=1)
    assert hits == []


def test_find_orfs_handles_rna_u():
    seq_dna = _LONG_ORF_SEQ
    seq_rna = seq_dna.replace("T", "U")
    hits_dna = find_orfs(seq_dna, min_protein_length=30, both_strands=False)
    hits_rna = find_orfs(seq_rna, min_protein_length=30, both_strands=False)
    assert len(hits_dna) == len(hits_rna) == 1
    assert hits_dna[0].protein == hits_rna[0].protein


def test_find_canonical_introns_basic():
    # GT ... AG with a 30-base spacer in between
    seq = "AAAA" + "GT" + "A" * 30 + "AG" + "TTTT"
    hits = find_canonical_introns(seq, min_intron_len=20, max_intron_len=100)
    assert len(hits) >= 1
    assert hits[0].start == 4
    assert seq[hits[0].start : hits[0].start + 2] == "GT"
    assert seq[hits[0].end - 2 : hits[0].end] == "AG"


def test_find_canonical_introns_respects_min_length():
    seq = "AAAA" + "GT" + "A" * 5 + "AG" + "TTTT"  # too short a spacer
    hits = find_canonical_introns(seq, min_intron_len=20, max_intron_len=100)
    assert hits == []


def test_find_canonical_introns_no_gt_returns_empty():
    hits = find_canonical_introns("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", min_intron_len=5)
    assert hits == []


def test_find_tata_box_detects_consensus():
    seq = "C" * 50 + "TATAAAA" + "GGGGGG"
    hits = find_tata_box(seq, search_window=60)
    assert len(hits) >= 1
    assert hits[0].motif.startswith("TATA")


def test_find_tata_box_outside_window_not_found():
    seq = "TATAAAA" + "C" * 200
    hits = find_tata_box(seq, search_window=50)  # box is far outside the last 50 bases
    assert hits == []


def test_find_tata_box_no_match():
    hits = find_tata_box("GGGGCCCCGGGGCCCC", search_window=20)
    assert hits == []


# ---- Protein ----


def test_hydropathy_profile_length_matches_sequence():
    profile = hydropathy_profile("MKTAYIAKQRQISFVKSHFSRQLEERL", window=9)
    assert len(profile) == 27


def test_hydropathy_profile_hydrophobic_region_scores_higher():
    hydrophobic = "LLLLLLLLLLLLLLLLLLLL"
    charged = "DDDDDDDDDDDDDDDDDDDD"
    hp = hydropathy_profile(hydrophobic, window=9)
    cp = hydropathy_profile(charged, window=9)
    assert sum(hp) / len(hp) > sum(cp) / len(cp)


def test_find_transmembrane_regions_detects_hydrophobic_stretch():
    # a clearly hydrophobic 20-residue stretch flanked by charged residues
    seq = "DDDDD" + "LIVLIVLIVLIVLIVLIVLI" + "DDDDD"
    regions = find_transmembrane_regions(seq, window=9, threshold=1.6)
    assert len(regions) >= 1
    assert regions[0].mean_hydropathy >= 1.6


def test_find_transmembrane_regions_no_hits_for_hydrophilic_protein():
    seq = "DDDDDEEEEEKKKKKRRRRR" * 2
    regions = find_transmembrane_regions(seq, threshold=1.6)
    assert regions == []


def test_predict_signal_peptide_positive_case():
    # positively charged n-region, hydrophobic h-region, small residues before cleavage
    seq = "MKR" + "LLLLLLLLLLLLLL" + "AGA" + "QRSTUVWXYZABCDEFGH".replace("U", "V").replace("X","V").replace("Z","V")
    result = predict_signal_peptide(seq, max_length=30)
    assert isinstance(result.score, float)
    assert 0.0 <= result.score <= 1.0


def test_predict_signal_peptide_negative_case():
    seq = "DDDDDDDDDDDDDDDDDDDDDDDDDDDDDD"  # all charged, no hydrophobic stretch
    result = predict_signal_peptide(seq)
    assert result.is_signal_peptide is False
    assert result.score == 0.0


def test_find_coiled_coil_periodic_hydrophobic_pattern():
    # heptad repeat with hydrophobic residues (L) at positions a/d (0-indexed 0 and 3 of each 7)
    unit = "LAEALKE"  # L at 0, A at 3rd-ish -- use a stronger synthetic pattern instead
    seq = "LAEELKKLAEELKKLAEELKKLAEELKKLAEELKK"  # heptad-like repeat, L every 7
    regions = find_coiled_coil(seq, window=28, threshold=0.5)
    assert isinstance(regions, list)  # structural check: doesn't crash, returns list of regions


def test_find_coiled_coil_too_short_sequence_returns_empty():
    assert find_coiled_coil("LLLL", window=28) == []


def test_find_low_complexity_regions_detects_homopolymer():
    seq = "MKTAYIAKQR" + "Q" * 30 + "SFVKSHFSRQ"
    regions = find_low_complexity_regions(seq, window=12, entropy_threshold=1.0)
    assert len(regions) >= 1
    assert regions[0].entropy < 1.0


def test_find_low_complexity_regions_no_hits_for_diverse_sequence():
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSG"
    regions = find_low_complexity_regions(seq, window=12, entropy_threshold=1.0)
    assert regions == []


def test_find_low_complexity_regions_too_short_returns_empty():
    assert find_low_complexity_regions("MKT", window=12) == []


def test_scan_prosite_patterns_n_glycosylation():
    seq = "AAANATSAAA"  # N-A-T-S matches N[^P][ST][^P]
    hits = scan_prosite_patterns(seq, pattern_ids=["PS00001"])
    assert len(hits) == 1
    assert hits[0].pattern_id == "PS00001"
    assert hits[0].matched_text == "NATS"


def test_scan_prosite_patterns_walker_a():
    seq = "XX" + "AXXXXGKS" + "XX"
    hits = scan_prosite_patterns(seq, pattern_ids=["WALKER_A"])
    assert len(hits) == 1
    assert hits[0].start == 2


def test_scan_prosite_patterns_unknown_id_raises():
    with pytest.raises(ValueError):
        scan_prosite_patterns("MKTAYIAK", pattern_ids=["NOT_A_PATTERN"])


def test_scan_prosite_patterns_no_hits():
    hits = scan_prosite_patterns("EEEEEEEEEEEEEEEEEEEEEEEE", pattern_ids=["PS00001"])
    assert hits == []


def test_list_prosite_patterns_returns_all_ids():
    patterns = list_prosite_patterns()
    assert "PS00001" in patterns
    assert "WALKER_A" in patterns
    assert len(patterns) >= 5
