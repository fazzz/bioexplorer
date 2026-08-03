"""Core data model for BioExplorer.

BioRecord wraps a Biopython SeqRecord and adds the bookkeeping BioExplorer
needs on top of it: a molecule-type tag (DNA/RNA/Protein), a free-form tag
set (populated by ``bio annotate`` / recognition steps), and a metadata dict
(populated by ``bio descriptor`` and friends). This mirrors ChemExplorer's
Molecule wrapper around RDKit Mol objects.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Iterator

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


class SeqType(str, Enum):
    """Biological alphabet a record belongs to."""

    DNA = "dna"
    RNA = "rna"
    PROTEIN = "protein"

    @classmethod
    def guess(cls, sequence: str) -> "SeqType":
        """Guess the sequence type from its alphabet.

        Heuristic, not a validator: BioExplorer never silently "corrects" a
        user-declared type, this is only used when the type is not given
        explicitly (e.g. plain FASTA import without a --type flag).
        """
        letters = set(sequence.upper()) - {"-", "*", "."}
        if not letters:
            return cls.DNA
        dna_alphabet = set("ACGTN")
        rna_alphabet = set("ACGUN")
        if letters <= dna_alphabet:
            return cls.DNA
        if letters <= rna_alphabet and "U" in letters:
            return cls.RNA
        return cls.PROTEIN


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class BioRecord:
    """A single sequence record managed by BioExplorer.

    Parameters
    ----------
    seq_id:
        Stable internal identifier. Auto-generated if not supplied, distinct
        from ``name`` (which usually comes from the FASTA header / accession).
    name:
        Human-readable identifier (accession, locus tag, header id, ...).
    sequence:
        Raw sequence string, upper-cased.
    seq_type:
        DNA / RNA / PROTEIN.
    description:
        Free-text description (FASTA header remainder, GenBank DEFINITION, ...).
    tags:
        Set of short labels, e.g. added by ``bio annotate`` (``signal_peptide``,
        ``transmembrane``) or by clustering/representative-selection steps.
    metadata:
        Arbitrary key/value store: descriptors, cluster assignments, source
        file, organism, taxonomy, etc. Values should be JSON-serialisable so
        the record round-trips through ``bio export``.
    """

    name: str
    sequence: str
    seq_type: SeqType
    seq_id: str = field(default_factory=_new_id)
    description: str = ""
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sequence = self.sequence.upper()
        if isinstance(self.seq_type, str):
            self.seq_type = SeqType(self.seq_type)

    # -- convenience -----------------------------------------------------

    @property
    def length(self) -> int:
        return len(self.sequence)

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags

    def add_tag(self, tag: str) -> None:
        self.tags.add(tag)

    def get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    # -- Biopython interop -------------------------------------------------

    def to_seqrecord(self) -> SeqRecord:
        rec = SeqRecord(
            Seq(self.sequence),
            id=self.name,
            name=self.name,
            description=self.description,
        )
        rec.annotations["bioexplorer_seq_id"] = self.seq_id
        rec.annotations["bioexplorer_seq_type"] = self.seq_type.value
        rec.annotations["bioexplorer_tags"] = sorted(self.tags)
        return rec

    @classmethod
    def from_seqrecord(
        cls, rec: SeqRecord, seq_type: SeqType | str | None = None
    ) -> "BioRecord":
        sequence = str(rec.seq)
        resolved_type = (
            SeqType(seq_type) if seq_type is not None else SeqType.guess(sequence)
        )
        tags = set(rec.annotations.get("bioexplorer_tags", []))
        seq_id = rec.annotations.get("bioexplorer_seq_id") or _new_id()
        return cls(
            name=rec.id,
            sequence=sequence,
            seq_type=resolved_type,
            seq_id=seq_id,
            description=rec.description,
            tags=tags,
        )


class BioCollection:
    """An ordered, ID-indexed collection of BioRecord objects.

    Counterpart to ChemExplorer's MoleculeCollection: the unit that ``bio
    import``, ``bio descriptor``, ``bio search`` etc. all operate on, and
    what gets persisted between CLI invocations (see io.py / project state).
    """

    def __init__(self, records: Iterable[BioRecord] | None = None) -> None:
        self._records: dict[str, BioRecord] = {}
        if records:
            for r in records:
                self.add(r)

    def add(self, record: BioRecord) -> None:
        if record.seq_id in self._records:
            raise ValueError(f"duplicate seq_id: {record.seq_id}")
        self._records[record.seq_id] = record

    def get(self, seq_id: str) -> BioRecord:
        return self._records[seq_id]

    def remove(self, seq_id: str) -> None:
        del self._records[seq_id]

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[BioRecord]:
        return iter(self._records.values())

    def __contains__(self, seq_id: str) -> bool:
        return seq_id in self._records

    def by_name(self, name: str) -> list[BioRecord]:
        return [r for r in self if r.name == name]

    def filter(self, predicate) -> "BioCollection":
        return BioCollection(r for r in self if predicate(r))

    def type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self:
            counts[r.seq_type.value] = counts.get(r.seq_type.value, 0) + 1
        return counts
