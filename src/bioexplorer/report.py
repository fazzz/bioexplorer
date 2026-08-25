"""Generic crosstab aggregation (not in the spec -- ported from
ChemExplorer's ``chem report`` finishing touch). Counts records grouped by
any combination of axes over tag/type/metadata, so you can answer
"how many of each X, broken down by Y (and Z, ...)" without writing a
one-off script.

Axis spec grammar (each ``--by`` on the CLI is one of these):

- ``type``                          -- seq_type (dna/rna/protein)
- ``tag:<name>``                    -- boolean: has this exact tag? ("yes"/"no")
- ``tag_prefix:<prefix>``           -- categorical: which tag(s) starting with
                                        `prefix` are present, prefix stripped
                                        (e.g. tag_prefix:cluster_ on tags
                                        {"cluster_2", "cluster_representative"}
                                        gives "2"); "(none)" if no match
- ``field:<dotted.key>``            -- raw metadata value as a category
- ``field:<dotted.key>:bin<width>`` -- numeric metadata value, bucketed into
                                        fixed-width bins (e.g. bin10 on
                                        descriptor.gc_percent=42.3 -> "[40,50)")
"""

from __future__ import annotations

import math

from .core import BioRecord
from .search import _get_nested


def resolve_axis_value(record: BioRecord, spec: str) -> str:
    if spec == "type":
        return record.seq_type.value

    if spec.startswith("tag_prefix:"):
        prefix = spec[len("tag_prefix:"):]
        if not prefix:
            raise ValueError("tag_prefix: needs a non-empty prefix")
        matches = sorted(t for t in record.tags if t.startswith(prefix))
        if not matches:
            return "(none)"
        return ",".join(m[len(prefix):] or m for m in matches)

    if spec.startswith("tag:"):
        tag = spec[len("tag:"):]
        if not tag:
            raise ValueError("tag: needs a non-empty tag name")
        return "yes" if record.has_tag(tag) else "no"

    if spec.startswith("field:"):
        rest = spec[len("field:"):]
        if not rest:
            raise ValueError("field: needs a dotted metadata key")
        if ":bin" in rest:
            dotted_key, bin_part = rest.split(":bin", 1)
            try:
                width = float(bin_part)
            except ValueError:
                raise ValueError(f"invalid bin width in --by spec: {spec!r}")
            if width <= 0:
                raise ValueError(f"bin width must be positive: {spec!r}")
            value = _get_nested(record, dotted_key)
            if value is None:
                return "(none)"
            try:
                v = float(value)
            except (TypeError, ValueError):
                return str(value)
            lo = math.floor(v / width) * width
            hi = lo + width
            return f"[{lo:g},{hi:g})"
        value = _get_nested(record, rest)
        return "(none)" if value is None else str(value)

    raise ValueError(
        f"unknown --by spec: {spec!r} -- use 'type', 'tag:<name>', "
        f"'tag_prefix:<prefix>', 'field:<dotted.key>', or "
        f"'field:<dotted.key>:bin<width>'"
    )


def build_report(records: list[BioRecord], by_specs: list[str]) -> list[dict]:
    """Group records by the tuple of axis values across all by_specs and
    count members per combination. Rows are sorted by axis values for
    stable, diffable output."""
    if not by_specs:
        raise ValueError("build_report needs at least one --by spec")

    groups: dict[tuple, int] = {}
    for rec in records:
        key = tuple(resolve_axis_value(rec, spec) for spec in by_specs)
        groups[key] = groups.get(key, 0) + 1

    rows = []
    for key in sorted(groups):
        row = dict(zip(by_specs, key))
        row["count"] = groups[key]
        rows.append(row)
    return rows
