"""Sequence profile analysis (spec section 10).

Built from a multiple sequence alignment (gapped, equal-length sequences --
see align.py / project.save_alignment).

Matrix stack, in the order the spec lists them:

- PFM (Position Frequency Matrix): raw per-position symbol counts.
- PPM (Position Probability Matrix): PFM normalized to probabilities, with
  a pseudocount to avoid zeros.
- PWM (Position Weight Matrix): log2(PPM / background) -- the log-odds
  matrix used for motif scanning.
- PSSM (Position-Specific Scoring Matrix): PWM scaled to integers, the way
  PSI-BLAST/JASPAR-style tools store scoring matrices for fast lookup. Some
  literature treats PWM and PSSM as synonyms; here they're kept distinct
  (float log-odds vs. integer-scaled score) since the spec lists both.

Analysis: consensus sequence, Shannon entropy, conservation score
(1 - H/H_max), relative entropy (KL divergence from background) --
the last three all quantify "how conserved is this column" from different
angles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .core import BioRecord, SeqType

GAP = "-"

_DNA_ALPHABET = list("ACGT")
_RNA_ALPHABET = list("ACGU")
_PROTEIN_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")


def default_alphabet(seq_type: SeqType) -> list[str]:
    if seq_type == SeqType.DNA:
        return list(_DNA_ALPHABET)
    if seq_type == SeqType.RNA:
        return list(_RNA_ALPHABET)
    return list(_PROTEIN_ALPHABET)


def default_background(seq_type: SeqType) -> dict[str, float]:
    """Uniform background over the canonical alphabet. Swap in observed
    genome/proteome-wide frequencies here if a non-uniform null model is
    wanted -- accepted as an override on build_profile()."""
    alphabet = default_alphabet(seq_type)
    p = 1.0 / len(alphabet)
    return {symbol: p for symbol in alphabet}


def _check_aligned(sequences: list[str]) -> int:
    if not sequences:
        raise ValueError("at least one aligned sequence is required")
    length = len(sequences[0])
    if any(len(s) != length for s in sequences):
        raise ValueError(
            "sequences are not the same length -- they must come from an "
            "alignment (bio align), not raw unaligned sequences"
        )
    return length


def compute_pfm(sequences: list[str], alphabet: list[str]) -> list[dict[str, int]]:
    """Raw per-position counts over `alphabet`, plus a 'gap' key for '-'."""
    length = _check_aligned(sequences)
    pfm = []
    for pos in range(length):
        counts = {symbol: 0 for symbol in alphabet}
        counts["gap"] = 0
        for seq in sequences:
            char = seq[pos].upper()
            if char == GAP:
                counts["gap"] += 1
            elif char in counts:
                counts[char] += 1
            # symbols outside the alphabet (e.g. ambiguity codes) are
            # ignored, consistent with spec's core-alphabet descriptors
        pfm.append(counts)
    return pfm


def compute_ppm(
    pfm: list[dict[str, int]], alphabet: list[str], pseudocount: float = 0.5
) -> list[dict[str, float]]:
    """Normalize PFM counts to probabilities over `alphabet` (gaps
    excluded from the probability mass), with an additive pseudocount."""
    ppm = []
    for counts in pfm:
        non_gap_total = sum(counts[s] for s in alphabet)
        denom = non_gap_total + pseudocount * len(alphabet)
        if denom == 0:
            probs = {s: 1.0 / len(alphabet) for s in alphabet}
        else:
            probs = {s: (counts[s] + pseudocount) / denom for s in alphabet}
        ppm.append(probs)
    return ppm


def compute_pwm(
    ppm: list[dict[str, float]], background: dict[str, float]
) -> list[dict[str, float]]:
    """Log-odds matrix: log2(P(symbol) / background(symbol))."""
    pwm = []
    for probs in ppm:
        row = {}
        for symbol, p in probs.items():
            q = background.get(symbol, 1e-9)
            row[symbol] = math.log2(p / q) if p > 0 and q > 0 else float("-inf")
        pwm.append(row)
    return pwm


def compute_pssm(pwm: list[dict[str, float]], scale: int = 100) -> list[dict[str, int]]:
    """Integer-scaled version of the PWM for fast lookup-table scanning."""
    pssm = []
    for row in pwm:
        scaled = {}
        for symbol, score in row.items():
            scaled[symbol] = int(round(score * scale)) if math.isfinite(score) else -(scale * 20)
        pssm.append(scaled)
    return pssm


def consensus_sequence(pfm: list[dict[str, int]], alphabet: list[str]) -> str:
    """Per-position majority symbol (ties broken by alphabet order). A
    position is called as a gap only if gaps outnumber every residue."""
    out = []
    for counts in pfm:
        best_symbol, best_count = GAP, counts["gap"]
        for symbol in alphabet:
            if counts[symbol] > best_count:
                best_symbol, best_count = symbol, counts[symbol]
        out.append(best_symbol)
    return "".join(out)


def shannon_entropy(ppm: list[dict[str, float]]) -> list[float]:
    """H = -sum(p * log2(p)) per position, in bits."""
    out = []
    for probs in ppm:
        h = -sum(p * math.log2(p) for p in probs.values() if p > 0)
        out.append(h)
    return out


def conservation_score(ppm: list[dict[str, float]], alphabet: list[str]) -> list[float]:
    """1 - H/H_max, so 1.0 = fully conserved, 0.0 = uniform over the
    alphabet at that position."""
    h_max = math.log2(len(alphabet))
    entropies = shannon_entropy(ppm)
    if h_max == 0:
        return [1.0 for _ in entropies]
    return [1.0 - (h / h_max) for h in entropies]


def relative_entropy(
    ppm: list[dict[str, float]], background: dict[str, float]
) -> list[float]:
    """KL divergence per position: sum(p * log2(p/q)). This is the
    information-content term sequence logos scale letter heights by."""
    out = []
    for probs in ppm:
        d = 0.0
        for symbol, p in probs.items():
            q = background.get(symbol, 1e-9)
            if p > 0:
                d += p * math.log2(p / q)
        out.append(d)
    return out


@dataclass
class Profile:
    seq_type: SeqType
    alphabet: list[str]
    length: int
    n_sequences: int
    background: dict[str, float]
    pfm: list[dict[str, int]]
    ppm: list[dict[str, float]]
    pwm: list[dict[str, float]]
    pssm: list[dict[str, int]]
    consensus: str
    shannon_entropy: list[float] = field(repr=False)
    conservation_score: list[float] = field(repr=False)
    relative_entropy: list[float] = field(repr=False)


def build_profile(
    records: list[BioRecord],
    pseudocount: float = 0.5,
    background: dict[str, float] | None = None,
    alphabet: list[str] | None = None,
) -> Profile:
    if not records:
        raise ValueError("build_profile needs at least one aligned record")
    seq_type = records[0].seq_type
    sequences = [r.sequence for r in records]
    length = _check_aligned(sequences)

    resolved_alphabet = alphabet or default_alphabet(seq_type)
    resolved_background = background or default_background(seq_type)

    pfm = compute_pfm(sequences, resolved_alphabet)
    ppm = compute_ppm(pfm, resolved_alphabet, pseudocount=pseudocount)
    pwm = compute_pwm(ppm, resolved_background)
    pssm = compute_pssm(pwm)
    consensus = consensus_sequence(pfm, resolved_alphabet)

    return Profile(
        seq_type=seq_type,
        alphabet=resolved_alphabet,
        length=length,
        n_sequences=len(records),
        background=resolved_background,
        pfm=pfm,
        ppm=ppm,
        pwm=pwm,
        pssm=pssm,
        consensus=consensus,
        shannon_entropy=shannon_entropy(ppm),
        conservation_score=conservation_score(ppm, resolved_alphabet),
        relative_entropy=relative_entropy(ppm, resolved_background),
    )


def profile_table(profile: Profile) -> list[dict]:
    """Flatten the profile into one row per alignment position, suitable
    for CSV/JSON export (position-wise residue frequency + the three
    conservation metrics + consensus call)."""
    rows = []
    for pos in range(profile.length):
        row = {
            "position": pos + 1,
            "consensus": profile.consensus[pos],
            "shannon_entropy": round(profile.shannon_entropy[pos], 4),
            "conservation_score": round(profile.conservation_score[pos], 4),
            "relative_entropy": round(profile.relative_entropy[pos], 4),
        }
        for symbol in profile.alphabet:
            row[f"freq_{symbol}"] = round(profile.ppm[pos][symbol], 4)
        row["freq_gap"] = profile.pfm[pos]["gap"]
        rows.append(row)
    return rows
