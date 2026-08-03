"""Search & filter (spec section 7).

Each ``filter_by_*`` function takes a BioCollection and returns a new,
filtered BioCollection, so they compose by chaining. ``run_filters`` wires
these together for the CLI (``bio search``) from a flat set of options.

Organism / Taxonomy / Domain fields are metadata-driven (populated by
GenBank import or by ``bio annotate`` / external DB lookups once those land)
rather than dedicated filter functions -- they go through
``filter_by_metadata`` using dotted keys, e.g. ``organism`` or
``descriptor.gc_percent``.
"""

from __future__ import annotations

import re
from typing import Any

from .core import BioCollection, BioRecord


def _get_nested(rec: BioRecord, dotted_key: str) -> Any:
    value: Any = rec.metadata
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def filter_by_id(collection: BioCollection, seq_ids: list[str]) -> BioCollection:
    wanted = set(seq_ids)
    return collection.filter(lambda r: r.seq_id in wanted)


def filter_by_name(
    collection: BioCollection, pattern: str, regex: bool = False
) -> BioCollection:
    if regex:
        rx = re.compile(pattern)
        return collection.filter(lambda r: rx.search(r.name) is not None)
    return collection.filter(lambda r: pattern.lower() in r.name.lower())


def filter_by_tag(collection: BioCollection, tag: str) -> BioCollection:
    return collection.filter(lambda r: r.has_tag(tag))


def filter_by_seq_type(collection: BioCollection, seq_type: str) -> BioCollection:
    return collection.filter(lambda r: r.seq_type.value == seq_type)


def filter_by_length(
    collection: BioCollection, min_len: int | None = None, max_len: int | None = None
) -> BioCollection:
    def pred(r: BioRecord) -> bool:
        if min_len is not None and r.length < min_len:
            return False
        if max_len is not None and r.length > max_len:
            return False
        return True

    return collection.filter(pred)


def filter_by_motif(
    collection: BioCollection, pattern: str, regex: bool = True
) -> BioCollection:
    """Filter by a sequence motif. Regex by default so IUPAC-style
    alternations (e.g. ``N[AG]G``) work; pass regex=False for a plain
    substring search."""
    if regex:
        rx = re.compile(pattern.upper())
        return collection.filter(lambda r: rx.search(r.sequence) is not None)
    needle = pattern.upper()
    return collection.filter(lambda r: needle in r.sequence)


def filter_by_metadata_range(
    collection: BioCollection,
    dotted_key: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> BioCollection:
    """Numeric range filter over a metadata field, e.g.
    ``descriptor.gc_percent`` or ``descriptor.pi``."""

    def pred(r: BioRecord) -> bool:
        value = _get_nested(r, dotted_key)
        if value is None:
            return False
        if min_value is not None and value < min_value:
            return False
        if max_value is not None and value > max_value:
            return False
        return True

    return collection.filter(pred)


def filter_by_metadata_equals(
    collection: BioCollection, dotted_key: str, expected: str
) -> BioCollection:
    return collection.filter(lambda r: str(_get_nested(r, dotted_key)) == expected)


def run_filters(
    collection: BioCollection,
    name: str | None = None,
    name_regex: bool = False,
    tag: str | None = None,
    seq_type: str | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    motif: str | None = None,
    field: str | None = None,
    field_min: float | None = None,
    field_max: float | None = None,
    field_equals: str | None = None,
) -> BioCollection:
    """Apply whichever filters were given, in a fixed, cheap-first order."""
    result = collection
    if seq_type:
        result = filter_by_seq_type(result, seq_type)
    if name:
        result = filter_by_name(result, name, regex=name_regex)
    if tag:
        result = filter_by_tag(result, tag)
    if min_length is not None or max_length is not None:
        result = filter_by_length(result, min_length, max_length)
    if motif:
        result = filter_by_motif(result, motif)
    if field and (field_min is not None or field_max is not None):
        result = filter_by_metadata_range(result, field, field_min, field_max)
    if field and field_equals is not None:
        result = filter_by_metadata_equals(result, field, field_equals)
    return result
