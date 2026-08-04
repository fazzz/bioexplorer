"""Sequence annotation -- classic, dependency-free algorithms (spec section 9).

Everything here runs in-process with nothing but Biopython: no external
database, no network call, no external binary. It covers the "automatic
recognition" half of the spec (ORF/CDS, canonical intron boundaries, TATA
box, signal peptide, transmembrane regions, coiled coil, low-complexity
regions, and a small built-in PROSITE-style pattern set for "Motif").

What it is NOT: a replacement for SignalP/TMHMM/COILS/SEG or PROSITE/Pfam/
InterPro themselves. These are simplified, classic-textbook heuristics
(hydropathy windows, heptad periodicity, composition entropy, canonical
splice-site consensus) -- useful for a first pass and for understanding
*why* a region looks interesting, not for a publication-grade call. The
real Domain/Pfam/InterPro/UniProt integration (which needs an external DB
or network service) lives in annotate_external.py.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from Bio.Seq import Seq
from Bio.SeqUtils.ProtParamData import kd as _KD_SCALE

_STOP_CODONS = {"TAA", "TAG", "TGA"}
_START_CODON = "ATG"


# ============================================================
# DNA/RNA
# ============================================================


@dataclass
class ORFHit:
    start: int  # 0-based, nucleotide coordinate on the given strand's sequence
    end: int  # exclusive
    strand: int  # +1 or -1
    frame: int  # 0, 1, or 2
    protein: str  # translation, stop codon excluded


def find_orfs(
    sequence: str,
    min_protein_length: int = 30,
    both_strands: bool = True,
) -> list[ORFHit]:
    """Naive ORF finder: scan all 3 (or 6, with both_strands) reading
    frames for ATG...stop, keeping ORFs whose translated protein is at
    least min_protein_length residues. This is the classic first-pass CDS
    candidate detector -- it has no concept of splicing, so on genomic
    (as opposed to already-spliced mRNA/CDS) input it will only find
    intron-free ORFs."""
    sequence = sequence.upper().replace("U", "T")
    hits: list[ORFHit] = []
    strands = (1, -1) if both_strands else (1,)

    for strand in strands:
        strand_seq = sequence if strand == 1 else str(Seq(sequence).reverse_complement())
        for frame in range(3):
            i = frame
            start_pos = None
            while i + 3 <= len(strand_seq):
                codon = strand_seq[i : i + 3]
                if start_pos is None and codon == _START_CODON:
                    start_pos = i
                elif start_pos is not None and codon in _STOP_CODONS:
                    nt_len = i - start_pos
                    protein_len = nt_len // 3
                    if protein_len >= min_protein_length:
                        protein = str(Seq(strand_seq[start_pos:i]).translate())
                        hits.append(ORFHit(start=start_pos, end=i + 3, strand=strand, frame=frame, protein=protein))
                    start_pos = None
                i += 3
    return sorted(hits, key=lambda h: (-len(h.protein), h.start))


@dataclass
class IntronCandidate:
    start: int  # 0-based, inclusive (first base of the GT)
    end: int  # exclusive (one past the last base of the AG)
    length: int


def find_canonical_introns(
    sequence: str,
    min_intron_len: int = 20,
    max_intron_len: int = 5000,
) -> list[IntronCandidate]:
    """Candidate introns bounded by the canonical GT...AG splice-site
    consensus (the "GT-AG rule" -- true for the vast majority of
    eukaryotic introns, but this is a boundary-consensus scan, not a
    splice-site strength model or real gene prediction: expect many
    false-positive candidates on real genomic sequence, especially at
    short min_intron_len."""
    sequence = sequence.upper().replace("U", "T")
    candidates: list[IntronCandidate] = []
    for match in re.finditer("GT", sequence):
        start = match.start()
        window_end = min(len(sequence), start + max_intron_len)
        window_start = start + min_intron_len
        if window_start >= window_end:
            continue
        ag_pos = sequence.find("AG", window_start - 2, window_end)
        if ag_pos != -1:
            end = ag_pos + 2
            candidates.append(IntronCandidate(start=start, end=end, length=end - start))
    return candidates


@dataclass
class PromoterHit:
    position: int  # 0-based start of the motif match
    motif: str  # the matched sequence
    offset_from_end: int  # position - len(sequence); typically negative, i.e. upstream of a presumed TSS at the sequence's end


_TATA_PATTERN = re.compile("TATA[AT]A[AT]?")


def find_tata_box(sequence: str, search_window: int = 100) -> list[PromoterHit]:
    """Scan the last `search_window` bases of `sequence` for a TATA-box-like
    motif (consensus TATAWAW), on the assumption the sequence's end
    approximates a transcription start site -- i.e. pass in the region
    immediately upstream of a gene, not an arbitrary internal fragment.
    Real core promoters also involve the Inr, BRE, DPE etc.; this covers
    only the single most recognizable element."""
    sequence = sequence.upper().replace("U", "T")
    region_start = max(0, len(sequence) - search_window)
    region = sequence[region_start:]
    hits = []
    for match in _TATA_PATTERN.finditer(region):
        pos = region_start + match.start()
        hits.append(PromoterHit(position=pos, motif=match.group(), offset_from_end=pos - len(sequence)))
    return hits


# ============================================================
# Protein
# ============================================================


def hydropathy_profile(sequence: str, window: int = 19) -> list[float]:
    """Kyte & Doolittle (1982) hydropathy, averaged over a centered
    sliding window, one value per residue position."""
    sequence = sequence.upper()
    n = len(sequence)
    half = window // 2
    scores = [_KD_SCALE.get(aa, 0.0) for aa in sequence]
    profile = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        segment = scores[lo:hi]
        profile.append(sum(segment) / len(segment))
    return profile


@dataclass
class TMRegion:
    start: int
    end: int  # exclusive
    mean_hydropathy: float


def find_transmembrane_regions(sequence: str, window: int = 19, threshold: float = 1.6) -> list[TMRegion]:
    """Contiguous stretches whose Kyte-Doolittle hydropathy average stays
    above `threshold` -- the classic single-scale TM-helix heuristic
    (real predictors like TMHMM add an HMM over multiple signals; this is
    the textbook first pass)."""
    profile = hydropathy_profile(sequence, window=window)
    regions = []
    start = None
    for i, score in enumerate(profile):
        above = score >= threshold
        if above and start is None:
            start = i
        elif not above and start is not None:
            regions.append(TMRegion(start=start, end=i, mean_hydropathy=sum(profile[start:i]) / (i - start)))
            start = None
    if start is not None:
        regions.append(TMRegion(start=start, end=len(profile), mean_hydropathy=sum(profile[start:]) / (len(profile) - start)))
    return [r for r in regions if r.end - r.start >= 6]  # a membrane-spanning helix is ~20 residues; drop noise blips


_SMALL_RESIDUES = set("AGSCT")
_CHARGED_POSITIVE = set("KR")


@dataclass
class SignalPeptideResult:
    is_signal_peptide: bool
    cleavage_site: int | None  # residue index (0-based) right after which cleavage is predicted
    score: float  # 0-1, higher = more confident


def predict_signal_peptide(sequence: str, max_length: int = 30) -> SignalPeptideResult:
    """Simplified 3-region signal-peptide heuristic (n-region: positive
    charge near the N-terminus; h-region: a hydrophobic stretch; c-region:
    small residues at the -3/-1 positions before a candidate cleavage
    site) -- the classic von Heijne (1986) rule-of-thumb that SignalP's
    early versions were built on, without the trained weight matrices."""
    sequence = sequence.upper()
    n_region = sequence[:5]
    has_positive_charge = any(c in _CHARGED_POSITIVE for c in n_region)

    search_region = sequence[: max_length + 5]
    profile = hydropathy_profile(search_region, window=7)
    best_h_start, best_h_score = None, 0.0
    i = 0
    while i < len(profile):
        if profile[i] >= 1.5:
            j = i
            while j < len(profile) and profile[j] >= 1.0:
                j += 1
            span_score = sum(profile[i:j]) / (j - i)
            if 6 <= (j - i) <= 20 and span_score > best_h_score:
                best_h_start, best_h_score = i, span_score
            i = j
        else:
            i += 1

    if best_h_start is None:
        return SignalPeptideResult(is_signal_peptide=False, cleavage_site=None, score=0.0)

    cleavage_site = None
    search_from = best_h_start + 6
    for pos in range(search_from, min(len(sequence) - 1, max_length)):
        if sequence[pos] in _SMALL_RESIDUES and pos + 2 < len(sequence) and sequence[pos + 2] in _SMALL_RESIDUES:
            cleavage_site = pos + 3
            break

    score = 0.0
    score += 0.3 if has_positive_charge else 0.0
    score += min(best_h_score / 4.0, 0.4)
    score += 0.3 if cleavage_site is not None else 0.0
    is_signal = has_positive_charge and best_h_score >= 1.5 and cleavage_site is not None

    return SignalPeptideResult(is_signal_peptide=is_signal, cleavage_site=cleavage_site, score=round(min(score, 1.0), 3))


@dataclass
class CoiledCoilRegion:
    start: int
    end: int  # exclusive
    score: float


def find_coiled_coil(sequence: str, window: int = 28, threshold: float = 1.0) -> list[CoiledCoilRegion]:
    """Heptad-periodicity hydrophobicity scan: coiled coils bury
    hydrophobic residues at the 'a' and 'd' positions of a 7-residue
    (heptad) repeat. For each window, try all 7 phase offsets and score
    how much more hydrophobic the a/d positions are than the rest -- the
    core idea behind COILS/PairCoil, without their statistical background
    model."""
    sequence = sequence.upper()
    n = len(sequence)
    scores = [_KD_SCALE.get(aa, 0.0) for aa in sequence]
    if n < window:
        return []

    window_scores = [0.0] * n
    for start in range(0, n - window + 1):
        best_phase_score = float("-inf")
        for phase in range(7):
            ad_positions = [start + k for k in range(window) if (k - phase) % 7 in (0, 3)]
            other_positions = [start + k for k in range(window) if (k - phase) % 7 not in (0, 3)]
            ad_mean = sum(scores[p] for p in ad_positions) / len(ad_positions)
            other_mean = sum(scores[p] for p in other_positions) / len(other_positions)
            phase_score = ad_mean - other_mean
            best_phase_score = max(best_phase_score, phase_score)
        mid = start + window // 2
        window_scores[mid] = max(window_scores[mid], best_phase_score)

    regions = []
    start = None
    for i, s in enumerate(window_scores):
        above = s >= threshold
        if above and start is None:
            start = i
        elif not above and start is not None:
            regions.append(CoiledCoilRegion(start=start, end=i, score=max(window_scores[start:i])))
            start = None
    if start is not None:
        regions.append(CoiledCoilRegion(start=start, end=n, score=max(window_scores[start:])))
    return regions


@dataclass
class LowComplexityRegion:
    start: int
    end: int  # exclusive
    entropy: float  # bits; lower = less complex


def find_low_complexity_regions(sequence: str, window: int = 12, entropy_threshold: float = 3.0) -> list[LowComplexityRegion]:
    """Sliding-window Shannon entropy over amino acid composition (SEG's
    core idea, without its dynamic-programming boundary trimming): a
    window is flagged when its local compositional entropy drops well
    below the ~4.32-bit max for the 20-letter alphabet, i.e. it's
    dominated by a few residue types (poly-Q runs, Ser/Pro-rich linkers,
    etc.)."""
    sequence = sequence.upper()
    n = len(sequence)
    if n < window:
        return []

    def entropy_of(segment: str) -> float:
        counts: dict[str, int] = {}
        for c in segment:
            counts[c] = counts.get(c, 0) + 1
        total = len(segment)
        return -sum((c / total) * math.log2(c / total) for c in counts.values())

    flags = [entropy_of(sequence[i : i + window]) < entropy_threshold for i in range(n - window + 1)]

    regions = []
    start = None
    for i, flagged in enumerate(flags):
        if flagged and start is None:
            start = i
        elif not flagged and start is not None:
            end = i + window - 1
            regions.append(LowComplexityRegion(start=start, end=end, entropy=entropy_of(sequence[start:end])))
            start = None
    if start is not None:
        end = n
        regions.append(LowComplexityRegion(start=start, end=end, entropy=entropy_of(sequence[start:end])))
    return regions


# ============================================================
# Motif scanning (built-in PROSITE-style patterns)
# ============================================================

# A small, illustrative subset of classic PROSITE patterns, translated to
# Python regex. NOT the PROSITE database -- for the real, current, full
# pattern set (and proper documentation of each), query PROSITE/InterPro
# directly (see annotate_external.py).
_PROSITE_PATTERNS: dict[str, tuple[str, str]] = {
    "PS00001": ("N-glycosylation site", r"N[^P][ST][^P]"),
    "PS00004": ("cAMP/cGMP-dependent protein kinase phosphorylation site", r"[RK]{2}.[ST]"),
    "PS00006": ("Casein kinase II phosphorylation site", r"[ST].{2}[DE]"),
    "PS00007": ("Protein kinase C phosphorylation site", r"[ST].[RK]"),
    "PS00008": ("N-myristoylation site", r"G[^EDRKHPFYW].{2}[STAGCN][^P]"),
    "WALKER_A": ("Walker A motif (P-loop NTPase)", r"[AG].{4}GK[ST]"),
    "ZINC_FINGER_C2H2": ("C2H2-type zinc finger", r"C.{2,4}C.{3}[LIVMFYWC].{8}H.{3,5}H"),
}


@dataclass
class MotifHit:
    pattern_id: str
    name: str
    start: int
    end: int  # exclusive
    matched_text: str


def scan_prosite_patterns(sequence: str, pattern_ids: list[str] | None = None) -> list[MotifHit]:
    """Scan a protein sequence against the built-in pattern set (or a
    subset of it, via pattern_ids). Overlapping matches of the same
    pattern are all reported (PROSITE patterns can legitimately recur)."""
    sequence = sequence.upper()
    ids = pattern_ids or list(_PROSITE_PATTERNS)
    hits = []
    for pid in ids:
        if pid not in _PROSITE_PATTERNS:
            raise ValueError(f"unknown pattern id: {pid} (choose from {list(_PROSITE_PATTERNS)})")
        name, pattern = _PROSITE_PATTERNS[pid]
        for match in re.finditer(f"(?={pattern})", sequence):  # (?=...) lookahead to allow overlaps
            real_match = re.match(pattern, sequence[match.start():])
            if real_match:
                hits.append(MotifHit(pattern_id=pid, name=name, start=match.start(), end=match.start() + len(real_match.group()), matched_text=real_match.group()))
    return sorted(hits, key=lambda h: h.start)


def list_prosite_patterns() -> dict[str, str]:
    return {pid: name for pid, (name, _) in _PROSITE_PATTERNS.items()}
