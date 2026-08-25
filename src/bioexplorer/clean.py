"""Sequence cleanup / QC (not in the spec -- ports ChemExplorer's ``chem
standardize`` and ProteinExplorer's ``prot fix`` finishing touch to
sequence data: a dedicated pass to normalize/trim/deduplicate/filter
records before doing real analysis on them, since raw imports -- especially
FASTQ reads -- routinely need this before anything else is meaningful).

Operations, each independently optional:

- Deduplication: identical sequence, or identical name (first-wins).
- Gap stripping: remove '-'/'.' from the raw sequence (e.g. someone
  imported an already-aligned FASTA as if it were raw).
- Adapter trimming: exact-match trim from the 5', 3', or both ends.
- Quality trimming: sliding-window Phred-quality trim from both ends
  (FASTQ imports only -- see core.BioRecord.quality).
- Ambiguous-end trimming: strip leading/trailing ambiguous symbols (N and
  IUPAC ambiguity codes for DNA/RNA, X for protein).
- Length/ambiguity filtering: drop records outside a length range, or
  whose ambiguous-symbol fraction exceeds a threshold, after the above.

All trimming operations that touch the sequence keep any attached quality
scores aligned to the trimmed sequence (sliced the same way), rather than
silently dropping or misaligning them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .core import BioRecord, SeqType

_GAP_CHARS = set("-.")
_IUPAC_AMBIGUOUS_NT = set("RYSWKMBDHVN")
_AMBIGUOUS_PROTEIN = set("X")


def _ambiguous_chars(seq_type: SeqType) -> set[str]:
    return _AMBIGUOUS_PROTEIN if seq_type == SeqType.PROTEIN else _IUPAC_AMBIGUOUS_NT


def strip_gaps(sequence: str) -> str:
    return "".join(c for c in sequence if c not in _GAP_CHARS)


def trim_ambiguous_ends(sequence: str, seq_type: SeqType) -> tuple[str, int, int]:
    """Returns (trimmed_sequence, n_trimmed_from_start, n_trimmed_from_end)."""
    ambiguous = _ambiguous_chars(seq_type)
    start, end = 0, len(sequence)
    while start < end and sequence[start] in ambiguous:
        start += 1
    while end > start and sequence[end - 1] in ambiguous:
        end -= 1
    return sequence[start:end], start, len(sequence) - end


def ambiguous_fraction(sequence: str, seq_type: SeqType) -> float:
    if not sequence:
        return 0.0
    ambiguous = _ambiguous_chars(seq_type)
    return sum(1 for c in sequence if c in ambiguous) / len(sequence)


def trim_adapter(sequence: str, adapter: str, end: str = "both") -> tuple[str, int, int]:
    """Exact-match adapter trim. Returns (trimmed_sequence,
    n_trimmed_from_start, n_trimmed_from_end)."""
    adapter_upper = adapter.upper()
    lead = 0
    if end in ("5", "both") and sequence.startswith(adapter_upper):
        lead = len(adapter_upper)
    trailing = 0
    remaining = sequence[lead:]
    if end in ("3", "both") and remaining.endswith(adapter_upper):
        trailing = len(adapter_upper)
    trimmed = sequence[lead: len(sequence) - trailing]
    return trimmed, lead, trailing


def trim_by_quality(
    sequence: str, quality: list[int], min_quality: int, window: int = 4
) -> tuple[str, list[int], int, int]:
    """Trim from both ends while the sliding-window average Phred quality
    stays below min_quality -- a simplified version of the sliding-window
    trim fastp/Trimmomatic use. Returns (trimmed_sequence,
    trimmed_quality, n_trimmed_from_start, n_trimmed_from_end)."""
    n = len(sequence)
    if n == 0 or not quality:
        return sequence, quality, 0, 0

    def window_mean(i: int, forward: bool) -> float:
        seg = quality[i : i + window] if forward else quality[max(0, i - window + 1) : i + 1]
        return sum(seg) / len(seg) if seg else 0.0

    start = 0
    while start < n and window_mean(start, forward=True) < min_quality:
        start += 1
    end = n
    while end > start and window_mean(end - 1, forward=False) < min_quality:
        end -= 1
    return sequence[start:end], quality[start:end], start, n - end


@dataclass
class CleanReport:
    kept: int = 0
    dropped_duplicate_sequence: int = 0
    dropped_duplicate_name: int = 0
    dropped_length: int = 0
    dropped_ambiguous: int = 0
    dropped_empty: int = 0
    trimmed_gaps: int = 0
    trimmed_ambiguous_ends: int = 0
    trimmed_adapter: int = 0
    trimmed_quality: int = 0
    kept_records: list[BioRecord] = field(default_factory=list, repr=False)


def clean_records(
    records: list[BioRecord],
    dedup_sequence: bool = False,
    dedup_name: bool = False,
    strip_gaps_flag: bool = False,
    trim_ambiguous: bool = False,
    max_ambiguous_fraction: float | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    adapter: str | None = None,
    adapter_end: str = "both",
    min_quality: int | None = None,
    quality_window: int = 4,
    drop_empty: bool = True,
) -> CleanReport:
    """Run the requested cleaning steps over `records`, in a fixed order
    (gap strip -> adapter trim -> quality trim -> ambiguous-end trim ->
    length/ambiguity filters -> dedup), and return a CleanReport with the
    kept records (mutated in place, same objects) plus counts of what
    happened. Steps are independently opt-in; an unset option is a no-op."""
    report = CleanReport()
    seen_sequences: set[str] = set()
    seen_names: set[str] = set()

    for rec in records:
        seq = rec.sequence
        quality = list(rec.quality) if rec.quality else None

        if strip_gaps_flag and any(c in _GAP_CHARS for c in seq):
            keep_mask = [c not in _GAP_CHARS for c in seq]
            new_seq = "".join(c for c, keep in zip(seq, keep_mask) if keep)
            if quality is not None:
                quality = [q for q, keep in zip(quality, keep_mask) if keep]
            if new_seq != seq:
                report.trimmed_gaps += 1
            seq = new_seq

        if adapter:
            new_seq, lead, trail = trim_adapter(seq, adapter, end=adapter_end)
            if lead or trail:
                report.trimmed_adapter += 1
                if quality is not None:
                    quality = quality[lead: len(quality) - trail]
            seq = new_seq

        if min_quality is not None and quality:
            new_seq, quality, lead, trail = trim_by_quality(seq, quality, min_quality, window=quality_window)
            if lead or trail:
                report.trimmed_quality += 1
            seq = new_seq

        if trim_ambiguous:
            new_seq, lead, trail = trim_ambiguous_ends(seq, rec.seq_type)
            if lead or trail:
                report.trimmed_ambiguous_ends += 1
                if quality is not None:
                    quality = quality[lead: len(quality) - trail]
            seq = new_seq

        if drop_empty and not seq:
            report.dropped_empty += 1
            continue

        if min_length is not None and len(seq) < min_length:
            report.dropped_length += 1
            continue
        if max_length is not None and len(seq) > max_length:
            report.dropped_length += 1
            continue

        if max_ambiguous_fraction is not None and ambiguous_fraction(seq, rec.seq_type) > max_ambiguous_fraction:
            report.dropped_ambiguous += 1
            continue

        if dedup_sequence:
            key = seq.upper()
            if key in seen_sequences:
                report.dropped_duplicate_sequence += 1
                continue
            seen_sequences.add(key)

        if dedup_name:
            if rec.name in seen_names:
                report.dropped_duplicate_name += 1
                continue
            seen_names.add(rec.name)

        if quality is not None and len(quality) != len(seq):
            raise ValueError(
                f"internal error: quality/sequence length mismatch for "
                f"'{rec.name}' after cleaning ({len(quality)} vs {len(seq)})"
            )

        rec.sequence = seq
        rec.quality = quality
        report.kept_records.append(rec)
        report.kept += 1

    return report
