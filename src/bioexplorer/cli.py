"""BioExplorer CLI (``bio``).

Command coverage in this phase: import, export, status, descriptor, search.
Everything else in the spec (align, profile, logo, cluster, tree, embed,
structure, plot, annotate, replay) lands in later phases, following the
same order ChemExplorer was implemented in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from . import annotate as bio_annotate
from . import annotate_external
from . import db as bio_db
from . import io as bio_io
from . import project
from . import report as bio_report
from . import structure as bio_structure
from .align import format_alignment, multiple_align, pairwise_align
from .clean import clean_records
from .cluster import annotate_clusters, cluster_cdhit, cluster_greedy, cluster_hierarchical, cluster_mmseqs
from .core import BioCollection, BioRecord, SeqType
from .descriptor import annotate_descriptors
from .embed import build_sequence_space
from .evolution import dn_ds_matrix, pairwise_dn_ds
from .profile import build_profile, profile_table
from .replay import replay as run_replay
from .search import run_filters
from .similarity import search_similar
from .tree import build_distance_tree, build_tree_external, tree_summary


class _RecordingGroup(click.Group):
    """Captures the argv Click was actually invoked with (via `args=` when
    called programmatically -- e.g. by Click's CliRunner, which is what
    `bio replay` and the test suite use -- or via sys.argv for the real
    console script) so project.log_command records the true invocation
    instead of the host process's own sys.argv."""

    def main(self, args=None, prog_name=None, **kwargs):
        project.set_current_argv(list(args) if args is not None else sys.argv[1:])
        return super().main(args=args, prog_name=prog_name, **kwargs)


@click.group(cls=_RecordingGroup)
def main() -> None:
    """BioExplorer -- CLI-based bioinformatics workbench."""


def _load_alignment(alignment_file: Path | None, name: str) -> list:
    """Load aligned records either from an explicit FASTA file or from a
    named alignment saved by `bio align --name ...`."""
    if alignment_file is not None:
        return bio_io.read_file(alignment_file, fmt="fasta")
    return project.load_alignment(name)


# -- shared non-destructive selection vocabulary -----------------------------
#
# Not in the spec -- ported from ChemExplorer's finishing touches: the same
# --tag/--field/--motif/--name vocabulary as `bio search`'s filter mode,
# available on output-producing commands (export/align/profile/tree/plot/
# embed/dnds/report) so a different slice of the project can be produced
# without `bio search --save-as project` destructively replacing it (which
# would mean re-importing to get back to the full set). Includes the
# exclusion counterparts (--exclude-tag, --exclude-motif, --exclude-id,
# --field-not-equals) for '!='-style conditions.

_SHARED_FILTER_OPTIONS = [
    click.option("--name", "sel_name", default=None, help="Substring or (with --name-regex) regex match on name."),
    click.option("--name-regex", "sel_name_regex", is_flag=True),
    click.option("--tag", "sel_tag", default=None, help="Require this tag."),
    click.option("--exclude-tag", "sel_exclude_tag", default=None, help="Exclude records with this tag."),
    click.option("--type", "sel_seq_type", default=None, type=click.Choice(["dna", "rna", "protein"])),
    click.option("--min-length", "sel_min_length", type=int, default=None),
    click.option("--max-length", "sel_max_length", type=int, default=None),
    click.option("--motif", "sel_motif", default=None, help="Regex sequence motif, e.g. 'N[AG]G'."),
    click.option("--exclude-motif", "sel_exclude_motif", default=None, help="Exclude records whose sequence matches this regex motif."),
    click.option("--field", "sel_field", default=None, help="Dotted metadata field, e.g. descriptor.gc_percent or cluster_id."),
    click.option("--field-min", "sel_field_min", type=float, default=None),
    click.option("--field-max", "sel_field_max", type=float, default=None),
    click.option("--field-equals", "sel_field_equals", default=None),
    click.option("--field-not-equals", "sel_field_not_equals", default=None, help="'!=' exclusion on --field's value."),
    click.option("--exclude-id", "sel_exclude_ids", multiple=True, help="Exclude this seq_id (repeatable)."),
]


def shared_filter_options(func):
    """Stack the shared selection options onto a command. Reads back via
    the ``sel_*`` kwargs Click injects (prefixed to avoid clashing with a
    command's own same-named options, e.g. `bio align --tag` for restricting
    multiple-alignment input)."""
    for opt in reversed(_SHARED_FILTER_OPTIONS):
        func = opt(func)
    return func


def _apply_shared_filters(collection: BioCollection, **sel) -> BioCollection:
    return run_filters(
        collection,
        name=sel.get("sel_name"),
        name_regex=sel.get("sel_name_regex", False),
        tag=sel.get("sel_tag"),
        exclude_tag=sel.get("sel_exclude_tag"),
        seq_type=sel.get("sel_seq_type"),
        min_length=sel.get("sel_min_length"),
        max_length=sel.get("sel_max_length"),
        motif=sel.get("sel_motif"),
        exclude_motif=sel.get("sel_exclude_motif"),
        field=sel.get("sel_field"),
        field_min=sel.get("sel_field_min"),
        field_max=sel.get("sel_field_max"),
        field_equals=sel.get("sel_field_equals"),
        field_not_equals=sel.get("sel_field_not_equals"),
        exclude_ids=list(sel["sel_exclude_ids"]) if sel.get("sel_exclude_ids") else None,
    )


def _enrich_with_project_metadata(records: list, project_collection: BioCollection) -> list:
    """Records loaded from an alignment/tree file carry only name+sequence
    (no tags/metadata) -- copy them over from the matching project record
    (by name) so --tag/--field selection has something to match against.
    Records with no match in the project keep empty tags/metadata, which
    just means tag/field filters naturally exclude them."""
    by_name: dict[str, BioRecord] = {}
    for rec in project_collection:
        by_name.setdefault(rec.name, rec)
    for r in records:
        match = by_name.get(r.name)
        if match is not None:
            r.tags = set(match.tags)
            r.metadata = dict(match.metadata)
    return records


def _select_alignment_records(records: list, **sel) -> list:
    """Apply the shared filters to a list of (possibly tag-enriched)
    alignment records, preserving order."""
    if not any(v for k, v in sel.items() if k != "sel_name_regex"):
        return records
    collection = BioCollection(records)
    filtered = _apply_shared_filters(collection, **sel)
    wanted = {r.seq_id for r in filtered}
    return [r for r in records if r.seq_id in wanted]


@main.command("import")
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--format", "fmt", default=None, help="Force input format (fasta/fastq/genbank/embl).")
@click.option("--type", "seq_type", default=None, type=click.Choice(["dna", "rna", "protein"]), help="Force sequence type instead of guessing.")
@click.option("--recursive", is_flag=True, help="Recurse into subdirectories when a path is a directory.")
@click.option("--append", is_flag=True, help="Add to the existing project instead of replacing it.")
def import_cmd(paths, fmt, seq_type, recursive, append) -> None:
    """Import sequence file(s) or a directory of sequence files into the project."""
    new_collection = bio_io.load_collection(list(paths), fmt=fmt, seq_type=seq_type, recursive=recursive)

    if append and project.exists():
        collection = project.load_collection()
        for rec in new_collection:
            collection.add(rec)
    else:
        collection = new_collection

    project.save_collection(collection)
    project.log_command()
    counts = collection.type_counts()
    counts_str = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    click.echo(f"imported {len(new_collection)} record(s); project total {len(collection)} ({counts_str})")


@main.command("status")
def status_cmd() -> None:
    """Show a summary of the current project."""
    if not project.exists():
        click.echo("no BioExplorer project in this directory yet (run `bio import ...`).")
        return
    collection = project.load_collection()
    counts = collection.type_counts()
    click.echo(f"records: {len(collection)}")
    for seq_type, n in sorted(counts.items()):
        click.echo(f"  {seq_type}: {n}")
    tag_counts: dict[str, int] = {}
    for rec in collection:
        for tag in rec.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    if tag_counts:
        click.echo("tags:")
        for tag, n in sorted(tag_counts.items(), key=lambda kv: -kv[1]):
            click.echo(f"  {tag}: {n}")


@main.command("export")
@click.argument("output", type=click.Path(path_type=Path))
@click.option("--format", "fmt", default=None, help="Output format (fasta/csv/tsv/json/parquet); inferred from suffix if omitted.")
@shared_filter_options
def export_cmd(output: Path, fmt: str | None, **sel) -> None:
    """Export the current project to a file. Accepts the shared selection
    options (--tag/--field/--motif/... and their --exclude-* counterparts)
    to export just a slice, without touching the project itself."""
    collection = project.load_collection()
    selected = _apply_shared_filters(collection, **sel)
    resolved_fmt = fmt or output.suffix.lstrip(".").lower()
    if resolved_fmt in ("fa", "fna", "faa"):
        resolved_fmt = "fasta"
    bio_io.write_collection(selected, output, resolved_fmt)
    project.log_command()
    click.echo(f"exported {len(selected)}/{len(collection)} record(s) to {output} ({resolved_fmt})")


@main.command("descriptor")
@click.option("--type", "seq_type", default=None, type=click.Choice(["dna", "rna", "protein"]), help="Only compute for records of this type.")
def descriptor_cmd(seq_type: str | None) -> None:
    """Compute sequence descriptors (spec section 6) for the current project."""
    collection = project.load_collection()
    targets = collection if seq_type is None else collection.filter(lambda r: r.seq_type.value == seq_type)
    n = 0
    for rec in targets:
        annotate_descriptors(rec)
        n += 1
    project.save_collection(collection)
    project.log_command()
    click.echo(f"computed descriptors for {n} record(s)")


@main.command("search")
@click.argument("query_file", required=False, type=click.Path(exists=True, path_type=Path))
@click.option("--method", default="kmer", type=click.Choice(["kmer", "minhash", "blast", "diamond", "mmseqs"]), help="Similarity method, only used when QUERY_FILE is given (spec section 8).")
@click.option("--k", "kmer_k", type=int, default=None, help="k-mer length for kmer/minhash methods.")
@click.option("--top-n", type=int, default=10, help="Max hits to return for similarity search.")
@click.option("--min-score", type=float, default=0.0, help="Minimum similarity score (0-1) for kmer/minhash methods.")
@click.option("--db", "db_path", type=click.Path(), default=None, help="Search an existing prebuilt DB instead of the project (blast: DB prefix; diamond: .dmnd file; mmseqs: DB prefix). See `bio db fetch`. Ignored for --method kmer/minhash.")
@click.option("--program", default="blastn", type=click.Choice(["blastn", "blastp", "blastx", "tblastn", "tblastx"]), help="BLAST program (--method blast only).")
@click.option("--export", "export_path", type=click.Path(path_type=Path), default=None, help="Write hits (id, score) to this CSV -- mainly useful with --db, where hits may not be in the project.")
@click.option("--name", default=None, help="Substring or (with --name-regex) regex match on name.")
@click.option("--name-regex", is_flag=True)
@click.option("--tag", default=None, help="Require this tag.")
@click.option("--exclude-tag", default=None, help="Exclude records with this tag.")
@click.option("--type", "seq_type", default=None, type=click.Choice(["dna", "rna", "protein"]))
@click.option("--min-length", type=int, default=None)
@click.option("--max-length", type=int, default=None)
@click.option("--motif", default=None, help="Regex sequence motif, e.g. 'N[AG]G'.")
@click.option("--exclude-motif", default=None, help="Exclude records whose sequence matches this regex motif.")
@click.option("--field", default=None, help="Dotted metadata field, e.g. descriptor.gc_percent or descriptor.pi.")
@click.option("--field-min", type=float, default=None)
@click.option("--field-max", type=float, default=None)
@click.option("--field-equals", default=None)
@click.option("--field-not-equals", default=None, help="'!=' exclusion on --field's value.")
@click.option("--exclude-id", "exclude_ids", multiple=True, help="Exclude this seq_id (repeatable).")
@click.option("--save-as", default=None, help="If set, replace the project with the result instead of just printing it.")
def search_cmd(query_file, method, kmer_k, top_n, min_score, db_path, program, export_path, name, name_regex, tag, exclude_tag, seq_type, min_length, max_length, motif, exclude_motif, field, field_min, field_max, field_equals, field_not_equals, exclude_ids, save_as) -> None:
    """Search the current project.

    With no QUERY_FILE: metadata/tag/motif filtering (spec section 7).
    With QUERY_FILE: similarity search (spec section 8) -- against the
    project by default, or against an existing DB with --db (e.g. a
    downloaded UniRef/nr/PDB DB; see `bio db fetch`). Filter options apply
    only to hits resolvable in the current project; --db hits from outside
    the project are shown/exported as bare id+score pairs.
    """
    collection = project.load_collection()
    exclude_ids = list(exclude_ids) if exclude_ids else None

    if query_file is None:
        result = run_filters(
            collection, name=name, name_regex=name_regex, tag=tag, exclude_tag=exclude_tag, seq_type=seq_type,
            min_length=min_length, max_length=max_length, motif=motif, exclude_motif=exclude_motif,
            field=field, field_min=field_min, field_max=field_max, field_equals=field_equals,
            field_not_equals=field_not_equals, exclude_ids=exclude_ids,
        )
        for rec in result:
            click.echo(f"{rec.seq_id}\t{rec.name}\t{rec.seq_type.value}\t{rec.length}")
        click.echo(f"-- {len(result)}/{len(collection)} record(s) matched", err=True)
        if save_as == "project":
            project.save_collection(result)
            project.log_command()
            click.echo("project replaced with search result", err=True)
        return

    query_records = bio_io.read_file(query_file)
    if not query_records:
        raise click.ClickException(f"no sequences found in {query_file}")
    query_seq = query_records[0].sequence
    if len(query_records) > 1:
        click.echo(f"note: {query_file} has {len(query_records)} sequences; using the first ('{query_records[0].name}') as query", err=True)

    try:
        hits = search_similar(
            collection, query_seq, method=method, k=kmer_k, top_n=top_n, min_score=min_score,
            db_path=db_path, program=program,
        )
    except RuntimeError as e:
        raise click.ClickException(str(e))

    in_project = [h for h in hits if h.record is not None]
    external = [h for h in hits if h.record is None]

    if db_path is not None:
        # Searching an external DB: hits generally won't resolve to project
        # records, so skip the metadata-filter path entirely and just show
        # id+score (the project-filter options don't mean anything here).
        ordered = sorted(hits, key=lambda h: -h.score)
        for hit in ordered:
            tag_note = "" if hit.record is None else "\t(in project)"
            click.echo(f"{hit.hit_id}\tscore={hit.score:.4f}{tag_note}")
        click.echo(f"-- {len(ordered)} hit(s) (method={method}, db={db_path})", err=True)
        if export_path is not None:
            import csv as csv_mod
            with open(export_path, "w", newline="") as fh:
                writer = csv_mod.writer(fh)
                writer.writerow(["hit_id", "score", "in_project"])
                for hit in ordered:
                    writer.writerow([hit.hit_id, hit.score, hit.record is not None])
            click.echo(f"exported hits to {export_path}", err=True)
        return

    result = BioCollection()
    for hit in in_project:
        hit.record.set("similarity_score", round(hit.score, 4))
        if hit.record.seq_id not in result:
            result.add(hit.record)
    result = run_filters(
        result, name=name, name_regex=name_regex, tag=tag, exclude_tag=exclude_tag, seq_type=seq_type,
        min_length=min_length, max_length=max_length, motif=motif, exclude_motif=exclude_motif,
        field=field, field_min=field_min, field_max=field_max, field_equals=field_equals,
        field_not_equals=field_not_equals, exclude_ids=exclude_ids,
    )
    ordered = sorted(result, key=lambda r: -r.get("similarity_score", 0.0))
    for rec in ordered:
        click.echo(f"{rec.seq_id}\t{rec.name}\t{rec.seq_type.value}\t{rec.length}\tscore={rec.get('similarity_score')}")
    click.echo(f"-- {len(ordered)} hit(s) (method={method})", err=True)
    if external:
        click.echo(f"-- (also {len(external)} hit(s) outside the project, not shown -- pass --db explicitly to search a DB and see raw hits)", err=True)

    if export_path is not None:
        import csv as csv_mod
        with open(export_path, "w", newline="") as fh:
            writer = csv_mod.writer(fh)
            writer.writerow(["seq_id", "name", "score"])
            for rec in ordered:
                writer.writerow([rec.seq_id, rec.name, rec.get("similarity_score")])
        click.echo(f"exported hits to {export_path}", err=True)

    if save_as == "project":
        project.save_collection(result)
        project.log_command()
        click.echo("project replaced with search result", err=True)


@main.command("align")
@click.option("--pairwise", "pairwise_ids", nargs=2, default=None, help="Two seq_ids for pairwise alignment (Needleman-Wunsch/Smith-Waterman).")
@click.option("--mode", default="global", type=click.Choice(["global", "local"]), help="Pairwise mode: global=Needleman-Wunsch, local=Smith-Waterman.")
@click.option("--tool", default="mafft", type=click.Choice(["mafft", "muscle", "clustalo"]), help="Multiple alignment tool (used when --pairwise is not given).")
@click.option("--save-as", "save_name", default="default", help="Name to save the multiple alignment under (.bioexplorer/alignments/<name>.fasta).")
@shared_filter_options
def align_cmd(pairwise_ids, mode, tool, save_name, **sel) -> None:
    """Align sequences (spec section 11).

    --pairwise ID1 ID2: pairwise alignment between two records in the
    project (no external tool required).
    Without --pairwise: multiple alignment of the whole project via
    mafft/muscle/clustalo, saved as a named alignment for later use by
    `bio profile` / `bio tree`. Restrict the input with the shared
    selection options (--tag/--type/--field/... and their --exclude-*
    counterparts).
    """
    collection = project.load_collection()

    if pairwise_ids:
        id_a, id_b = pairwise_ids
        try:
            rec_a, rec_b = collection.get(id_a), collection.get(id_b)
        except KeyError as e:
            raise click.ClickException(f"seq_id not found in project: {e}")
        if rec_a.seq_type != rec_b.seq_type:
            click.echo(f"note: aligning across types ({rec_a.seq_type.value} vs {rec_b.seq_type.value})", err=True)
        result = pairwise_align(
            rec_a.sequence, rec_b.sequence, mode=mode, seq_type=rec_a.seq_type,
            target_id=rec_a.name, query_id=rec_b.name,
        )
        click.echo(format_alignment(result))
        project.log_command()
        return

    records = list(_apply_shared_filters(collection, **sel))
    if len(records) < 2:
        raise click.ClickException(
            f"only {len(records)} record(s) matched the given filters; "
            f"multiple alignment needs at least 2"
        )

    try:
        aligned = multiple_align(records, tool=tool)
    except RuntimeError as e:
        raise click.ClickException(str(e))

    path = project.save_alignment(aligned, name=save_name)
    project.log_command()
    click.echo(f"aligned {len(aligned)} record(s) with {tool}, saved as alignment '{save_name}' -> {path}")


@main.command("profile")
@click.argument("alignment_file", required=False, type=click.Path(exists=True, path_type=Path))
@click.option("--alignment-name", "alignment_name", default="default", help="Named alignment to use if ALIGNMENT_FILE is not given (from `bio align --save-as`).")
@click.option("--pseudocount", type=float, default=0.5, help="Pseudocount added to each symbol count before normalizing to probabilities.")
@click.option("--export", "export_path", type=click.Path(path_type=Path), default=None, help="Write the position-wise profile table (csv/tsv/json) to this path.")
@click.option("--plot", "plot_kind", default=None, type=click.Choice(["logo", "heatmap", "conservation"]), help="Also render a plot.")
@click.option("--plot-output", type=click.Path(path_type=Path), default=None, help="Path for --plot output (default: profile_<kind>.png).")
@shared_filter_options
def profile_cmd(alignment_file, alignment_name, pseudocount, export_path, plot_kind, plot_output, **sel) -> None:
    """Compute a sequence profile from an alignment (spec section 10):
    PFM/PPM/PWM/PSSM, consensus sequence, Shannon entropy, conservation
    score, and relative entropy, per alignment position. The shared
    selection options (--tag/--field/... and --exclude-*) restrict the
    profile to a subset of the alignment (e.g. just one cluster)."""
    try:
        records = _load_alignment(alignment_file, alignment_name)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    records = _select_alignment_records(_enrich_with_project_metadata(records, project.load_collection()), **sel)
    if len(records) < 1:
        raise click.ClickException("no alignment records matched the given selection")

    profile = build_profile(records, pseudocount=pseudocount)

    click.echo(f"{profile.n_sequences} sequences x {profile.length} positions ({profile.seq_type.value})")
    click.echo(f"consensus: {profile.consensus}")
    mean_conservation = sum(profile.conservation_score) / profile.length
    click.echo(f"mean conservation score: {mean_conservation:.4f}")
    total_info = sum(profile.relative_entropy)
    click.echo(f"total information content: {total_info:.2f} bits")

    if export_path is not None:
        rows = profile_table(profile)
        fmt = export_path.suffix.lstrip(".").lower() or "csv"
        if fmt == "csv":
            import csv as csv_mod
            with open(export_path, "w", newline="") as fh:
                writer = csv_mod.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        elif fmt == "json":
            import json as json_mod
            with open(export_path, "w") as fh:
                json_mod.dump(rows, fh, indent=2)
        else:
            raise click.ClickException(f"unsupported export format: {fmt} (use .csv or .json)")
        click.echo(f"exported profile table to {export_path}")

    if plot_kind is not None:
        from . import viz
        out = plot_output or Path(f"profile_{plot_kind}.png")
        try:
            if plot_kind == "logo":
                viz.plot_sequence_logo(profile, out)
            elif plot_kind == "heatmap":
                viz.plot_heatmap(profile, out)
            else:
                viz.plot_conservation(profile, out)
        except RuntimeError as e:
            raise click.ClickException(str(e))
        click.echo(f"saved {plot_kind} plot to {out}")

    project.log_command()


@main.command("logo")
@click.argument("alignment_file", required=False, type=click.Path(exists=True, path_type=Path))
@click.option("--alignment-name", "alignment_name", default="default", help="Named alignment to use if ALIGNMENT_FILE is not given.")
@click.option("--output", type=click.Path(path_type=Path), default=Path("logo.png"), help="Output image path (.png/.svg/.pdf).")
@shared_filter_options
def logo_cmd(alignment_file, alignment_name, output, **sel) -> None:
    """Render a sequence logo from an alignment (shortcut for
    `bio profile --plot logo`). Accepts the shared selection options to
    render just a subset of the alignment."""
    try:
        records = _load_alignment(alignment_file, alignment_name)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    records = _select_alignment_records(_enrich_with_project_metadata(records, project.load_collection()), **sel)
    if len(records) < 1:
        raise click.ClickException("no alignment records matched the given selection")
    profile = build_profile(records)
    from . import viz
    try:
        viz.plot_sequence_logo(profile, output)
    except RuntimeError as e:
        raise click.ClickException(str(e))
    project.log_command()
    click.echo(f"saved sequence logo to {output}")


@main.command("cluster")
@click.option("--method", default="greedy", type=click.Choice(["greedy", "hierarchical", "cdhit", "mmseqs"]), help="Clustering algorithm.")
@click.option("--similarity", "similarity_method", default="kmer", type=click.Choice(["kmer", "minhash"]), help="Similarity proxy for --method greedy/hierarchical and for centroid selection.")
@click.option("--k", "kmer_k", type=int, default=None)
@click.option("--threshold", type=float, default=0.8, help="Similarity threshold for --method greedy (0-1).")
@click.option("--linkage", "linkage_method", default="average", type=click.Choice(["single", "complete", "average", "weighted"]), help="Linkage criterion for --method hierarchical.")
@click.option("--distance-threshold", type=float, default=None, help="Cut the dendrogram at this distance (--method hierarchical; default 0.3, mutually exclusive with --n-clusters).")
@click.option("--n-clusters", type=int, default=None, help="Cut the dendrogram to exactly this many clusters (--method hierarchical; mutually exclusive with --distance-threshold).")
@click.option("--identity", type=float, default=0.9, help="Sequence identity for --method cdhit (0-1).")
@click.option("--min-seq-id", type=float, default=0.9, help="Minimum sequence identity for --method mmseqs (0-1).")
@click.option("--no-consensus", is_flag=True, help="Skip consensus computation (faster; representative/centroid still computed).")
@click.option("--msa-tool", default="mafft", type=click.Choice(["mafft", "muscle", "clustalo", "none"]), help="MSA tool for consensus of variable-length clusters; 'none' disables the fallback alignment step.")
@click.option("--save-as", default=None, help="If 'project', write cluster tags/metadata back into the project.")
@shared_filter_options
def cluster_cmd(method, similarity_method, kmer_k, threshold, linkage_method, distance_threshold, n_clusters, identity, min_seq_id, no_consensus, msa_tool, save_as, **sel) -> None:
    """Cluster the current project's sequences (spec section 12) and pick a
    representative/centroid/consensus for each cluster. Restrict the input
    with the shared selection options (--tag/--type/--field/... and their
    --exclude-* counterparts)."""
    collection = project.load_collection()
    records = list(_apply_shared_filters(collection, **sel))
    if not records:
        raise click.ClickException("no records matched the given filters")

    try:
        if method == "greedy":
            clusters = cluster_greedy(records, method=similarity_method, k=kmer_k, threshold=threshold)
        elif method == "hierarchical":
            clusters = cluster_hierarchical(
                records, method=similarity_method, k=kmer_k, linkage_method=linkage_method,
                distance_threshold=distance_threshold, n_clusters=n_clusters,
            )
        elif method == "cdhit":
            clusters = cluster_cdhit(records, seq_type=records[0].seq_type, identity=identity)
        else:
            clusters = cluster_mmseqs(records, min_seq_id=min_seq_id)
    except (RuntimeError, ValueError) as e:
        raise click.ClickException(str(e))

    annotate_clusters(
        clusters,
        centroid_method=similarity_method,
        compute_consensus_seqs=not no_consensus,
        msa_tool=None if msa_tool == "none" else msa_tool,
    )

    sizes = sorted((len(c.members) for c in clusters), reverse=True)
    click.echo(f"{len(records)} record(s) -> {len(clusters)} cluster(s) (method={method})")
    click.echo(f"cluster sizes: {sizes[:10]}{' ...' if len(sizes) > 10 else ''}")
    for c in clusters:
        approx = " (approx)" if c.consensus_is_approximate else ""
        click.echo(
            f"  cluster {c.cluster_id}: n={len(c.members)} "
            f"rep={c.representative.name} centroid={c.centroid.name if c.centroid else '-'}"
            f"{f' consensus_len={len(c.consensus)}{approx}' if c.consensus else ''}"
        )

    if save_as == "project":
        project.save_collection(collection)
        project.log_command()
        click.echo("cluster tags/metadata written back to project", err=True)


@main.command("tree")
@click.argument("alignment_file", required=False, type=click.Path(exists=True, path_type=Path))
@click.option("--alignment-name", "alignment_name", default="default", help="Named alignment to use if ALIGNMENT_FILE is not given (from `bio align --save-as`).")
@click.option("--method", default="nj", type=click.Choice(["nj", "upgma", "iqtree", "fasttree", "raxml"]), help="Tree-building method.")
@click.option("--model", default=None, help="Substitution model (nj/upgma: Bio.Phylo model name, e.g. identity/blastn/blosum62; iqtree/raxml: model string).")
@click.option("--bootstrap", type=int, default=0, help="Bootstrap replicates for --method nj/upgma (0 disables).")
@click.option("--save-as", "save_name", default="default", help="Name to save the tree under (.bioexplorer/trees/<name>.nwk).")
@click.option("--output", type=click.Path(path_type=Path), default=None, help="Also write the Newick tree to this path.")
@shared_filter_options
def tree_cmd(alignment_file, alignment_name, method, model, bootstrap, save_name, output, **sel) -> None:
    """Build a phylogenetic tree from an alignment (spec section 14) and
    save it as Newick. The shared selection options (--tag/--field/... and
    --exclude-*) restrict the tree to a subset of the alignment (e.g. one
    cluster) without editing the alignment file itself."""
    try:
        records = _load_alignment(alignment_file, alignment_name)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    records = _select_alignment_records(_enrich_with_project_metadata(records, project.load_collection()), **sel)
    if len(records) < 2:
        raise click.ClickException(f"only {len(records)} alignment record(s) matched the given selection; a tree needs at least 2")

    try:
        if method in ("nj", "upgma"):
            tree = build_distance_tree(records, method=method, model=model, bootstrap=bootstrap)
        else:
            tree = build_tree_external(records, tool=method, model=model)
    except (RuntimeError, ValueError) as e:
        raise click.ClickException(str(e))

    summary = tree_summary(tree)
    click.echo(f"built {method} tree: {summary['n_taxa']} taxa, {summary['n_internal_nodes']} internal nodes")
    click.echo(f"total branch length: {summary['total_branch_length']:.4f}")

    saved_path = project.save_tree(tree, name=save_name)
    click.echo(f"saved tree as '{save_name}' -> {saved_path}")
    if output is not None:
        from .tree import write_newick
        write_newick(tree, output)
        click.echo(f"also wrote {output}")

    project.log_command()


@main.group("structure")
def structure_group() -> None:
    """Structure integration (spec section 16): prediction/viewer wrappers
    plus in-process structure analysis. BioExplorer doesn't implement
    structure prediction itself -- these subcommands are a unified
    interface to external tools."""


@structure_group.command("predict")
@click.argument("fasta_path", type=click.Path(exists=True, path_type=Path))
@click.option("--engine", default="colabfold", type=click.Choice(["colabfold", "alphafold", "modeller"]))
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("structure_prediction"))
@click.option("--template", type=click.Path(exists=True, path_type=Path), default=None, help="Template PDB (--engine modeller only).")
@click.option("--alignment", "alignment_file", type=click.Path(exists=True, path_type=Path), default=None, help="Target-template alignment (--engine modeller only).")
def structure_predict_cmd(fasta_path, engine, output_dir, template, alignment_file) -> None:
    """Predict a structure with an external engine (AlphaFold/ColabFold/MODELLER)."""
    kwargs = {}
    if engine == "modeller":
        if template is None or alignment_file is None:
            raise click.ClickException("--engine modeller needs --template and --alignment")
        kwargs = {"template_pdb": template, "alignment_file": alignment_file}
    try:
        out = bio_structure.predict_structure(fasta_path, output_dir, engine=engine, **kwargs)
    except RuntimeError as e:
        raise click.ClickException(str(e))
    click.echo(f"structure prediction ({engine}) output in {out}")


@structure_group.command("view")
@click.argument("pdb_path", type=click.Path(exists=True, path_type=Path))
@click.option("--viewer", default="vmd", type=click.Choice(["vmd", "chimerax", "pymol"]))
def structure_view_cmd(pdb_path, viewer) -> None:
    """Launch an external 3D structure viewer (non-blocking)."""
    try:
        bio_structure.view_structure(pdb_path, viewer=viewer)
    except RuntimeError as e:
        raise click.ClickException(str(e))
    click.echo(f"launched {viewer} for {pdb_path}")


@structure_group.command("rmsd")
@click.argument("struct_a_path", type=click.Path(exists=True, path_type=Path))
@click.argument("struct_b_path", type=click.Path(exists=True, path_type=Path))
@click.option("--chain-a", default="A")
@click.option("--chain-b", default="A")
def structure_rmsd_cmd(struct_a_path, struct_b_path, chain_a, chain_b) -> None:
    """Sequence-guided structural superposition + RMSD between two chains."""
    struct_a = bio_structure.read_structure(struct_a_path)
    struct_b = bio_structure.read_structure(struct_b_path)
    try:
        result = bio_structure.superimpose_structures(struct_a, struct_b, chain_a, chain_b)
    except ValueError as e:
        raise click.ClickException(str(e))
    click.echo(f"RMSD: {result.rmsd:.3f} A over {result.n_atoms} equivalent CA atoms")
    click.echo(f"sequence identity guiding the superposition: {format_alignment(result.sequence_alignment)}")


@structure_group.command("ss")
@click.argument("pdb_path", type=click.Path(exists=True, path_type=Path))
def structure_ss_cmd(pdb_path) -> None:
    """Per-residue secondary structure assignment (DSSP)."""
    try:
        ss = bio_structure.secondary_structure(pdb_path)
    except RuntimeError as e:
        raise click.ClickException(str(e))
    for (chain_id, resnum), code in ss.items():
        click.echo(f"{chain_id}\t{resnum}\t{code}")


@structure_group.command("map-conservation")
@click.argument("pdb_path", type=click.Path(exists=True, path_type=Path))
@click.option("--chain", default="A", help="Chain whose residues line up with the alignment/profile.")
@click.option("--alignment-file", type=click.Path(exists=True, path_type=Path), default=None, help="Aligned FASTA to compute the profile from.")
@click.option("--name", default="default", help="Named project alignment to use if --alignment-file is not given.")
@click.option("--output", type=click.Path(path_type=Path), required=True, help="Output PDB with conservation scores written into the B-factor column.")
def structure_map_conservation_cmd(pdb_path, chain, alignment_file, name, output) -> None:
    """Map per-position conservation scores (spec section 10) onto a
    structure's B-factor column for coloring in PyMOL/ChimeraX/VMD."""
    try:
        records = _load_alignment(alignment_file, name)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    profile = build_profile(records)
    structure = bio_structure.read_structure(pdb_path)
    try:
        bio_structure.map_conservation_to_bfactor(structure, profile.conservation_score, chain_id=chain)
    except ValueError as e:
        raise click.ClickException(str(e))
    bio_structure.write_structure(structure, output)
    click.echo(f"wrote conservation-mapped structure to {output}")


@main.command("dnds")
@click.argument("ids", nargs=-1, required=False)
@click.option("--method", default="NG86", type=click.Choice(["NG86", "LWL85", "YN00"]), help="NG86/LWL85 are pure Python (Nei-Gojobori / Li-Wu-Luo); YN00 uses PAML's yn00 binary.")
@click.option("--all", "all_pairs", is_flag=True, help="Compute all-pairs dN/dS across the (optionally filtered) project instead of a single pair.")
@shared_filter_options
def dnds_cmd(ids, method, all_pairs, **sel) -> None:
    """dN/dS (Ka/Ks) between codon-aligned coding sequences (spec section 13).

    `bio dnds ID1 ID2` for a single pair, or `bio dnds --all` for every
    pair in the project (restrict with the shared selection options --
    --tag/--type/--field/... and their --exclude-* counterparts).
    Sequences must be codon-aligned: equal length, a multiple of 3.
    """
    collection = project.load_collection()

    try:
        if all_pairs:
            records = list(_apply_shared_filters(collection, **sel))
            if len(records) < 2:
                raise click.ClickException("need at least 2 records for --all")
            results = dn_ds_matrix(records, method=method)
        else:
            if len(ids) != 2:
                raise click.ClickException("give exactly two seq_ids, or use --all")
            try:
                rec_a, rec_b = collection.get(ids[0]), collection.get(ids[1])
            except KeyError as e:
                raise click.ClickException(f"seq_id not found in project: {e}")
            results = [pairwise_dn_ds(rec_a.sequence, rec_b.sequence, rec_a.name, rec_b.name, method=method)]
    except ValueError as e:
        raise click.ClickException(str(e))
    except RuntimeError as e:
        raise click.ClickException(str(e))

    for r in results:
        omega_str = f"{r.omega:.4f}" if r.omega is not None else "undefined (dS=0)"
        click.echo(f"{r.seq_a_id}\t{r.seq_b_id}\tdN={r.dn:.4f}\tdS={r.ds:.4f}\tomega(dN/dS)={omega_str}")

    project.log_command()


@main.group("plot")
def plot_group() -> None:
    """Visualization (spec section 17): alignment viewer and phylogenetic
    tree rendering. (Sequence logo/heatmap/conservation live under `bio
    profile --plot` / `bio logo`; sequence-space and domain-architecture
    plots will land alongside embedding/annotation.)"""


@plot_group.command("alignment")
@click.argument("alignment_file", required=False, type=click.Path(exists=True, path_type=Path))
@click.option("--alignment-name", "alignment_name", default="default", help="Named alignment to use if ALIGNMENT_FILE is not given.")
@click.option("--color-by", default="residue", type=click.Choice(["residue", "conservation"]))
@click.option("--max-positions", type=int, default=200, help="Truncate to this many positions (0 = no limit).")
@click.option("--output", type=click.Path(path_type=Path), default=Path("alignment.png"))
@shared_filter_options
def plot_alignment_cmd(alignment_file, alignment_name, color_by, max_positions, output, **sel) -> None:
    """Render a Jalview-style colored alignment viewer. Accepts the shared
    selection options to show just a subset of the alignment."""
    try:
        records = _load_alignment(alignment_file, alignment_name)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    records = _select_alignment_records(_enrich_with_project_metadata(records, project.load_collection()), **sel)
    if len(records) < 1:
        raise click.ClickException("no alignment records matched the given selection")
    from . import viz
    try:
        viz.plot_alignment_viewer(
            records, output, color_by=color_by,
            max_positions=None if max_positions == 0 else max_positions,
        )
    except (RuntimeError, ValueError) as e:
        raise click.ClickException(str(e))
    project.log_command()
    click.echo(f"saved alignment viewer to {output}")


@plot_group.command("tree")
@click.argument("tree_file", required=False, type=click.Path(exists=True, path_type=Path))
@click.option("--tree-name", "tree_name", default="default", help="Named project tree to use if TREE_FILE is not given (from `bio tree --save-as`).")
@click.option("--no-confidence", is_flag=True, help="Hide bootstrap/confidence branch labels.")
@click.option("--output", type=click.Path(path_type=Path), default=Path("tree.png"))
@shared_filter_options
def plot_tree_cmd(tree_file, tree_name, no_confidence, output, **sel) -> None:
    """Render a phylogenetic tree as a dendrogram. The shared selection
    options (--tag/--field/... and --exclude-*) prune the tree to a subset
    of taxa (matched against the project by name) without touching the
    saved tree file."""
    from .tree import prune_tree_to_names, read_newick
    if tree_file is not None:
        tree = read_newick(tree_file)
    else:
        try:
            tree = project.load_tree(tree_name)
        except FileNotFoundError as e:
            raise click.ClickException(str(e))

    if any(v for k, v in sel.items() if k != "sel_name_regex"):
        taxa_names = [t.name for t in tree.get_terminals()]
        stub_records = _enrich_with_project_metadata(
            [BioRecord(name=n, sequence="", seq_type=SeqType.DNA, seq_id=n) for n in taxa_names],
            project.load_collection(),
        )
        keep = {r.name for r in _select_alignment_records(stub_records, **sel)}
        try:
            tree = prune_tree_to_names(tree, keep)
        except ValueError as e:
            raise click.ClickException(str(e))

    from . import viz
    try:
        viz.plot_tree(tree, output, show_confidence=not no_confidence)
    except RuntimeError as e:
        raise click.ClickException(str(e))
    project.log_command()
    click.echo(f"saved tree plot to {output}")


@main.command("replay")
@click.option("--dry-run", is_flag=True, help="Print what would run without executing anything.")
@click.option("--from", "from_index", type=int, default=None, help="1-based step number to start from.")
@click.option("--to", "to_index", type=int, default=None, help="1-based step number to stop at (inclusive).")
@click.option("--skip", "skip_str", default=None, help="Comma-separated top-level commands to skip (default: 'structure,annotate,replay' -- interactive viewer, external DB/network calls, and self-recursion).")
@click.option("--continue-on-error", is_flag=True, help="Keep going past a failed step instead of stopping.")
@click.option("--no-reset", is_flag=True, help="Replay on top of the current project state instead of rebuilding from scratch.")
def replay_cmd(dry_run, from_index, to_index, skip_str, continue_on_error, no_reset) -> None:
    """Replay the recorded command history (spec section 19) to rebuild
    the project from scratch, verifying the pipeline is reproducible."""
    if skip_str is not None:
        skip_commands = {s.strip() for s in skip_str.split(",") if s.strip()}
    else:
        from .replay import DEFAULT_SKIP
        skip_commands = DEFAULT_SKIP | {"structure", "annotate"}

    try:
        report = run_replay(
            skip_commands=skip_commands,
            from_index=from_index,
            to_index=to_index,
            dry_run=dry_run,
            continue_on_error=continue_on_error,
            reset_state=not no_reset,
        )
    except FileNotFoundError as e:
        raise click.ClickException(str(e))

    if report.backup_path is not None:
        click.echo(f"backed up previous project state to {report.backup_path}", err=True)

    for step in report.steps:
        cmd_str = " ".join(step.argv)
        rewritten_note = " (id rewritten)" if getattr(step, "ids_rewritten", False) else ""
        if step.status == "would_execute":
            click.echo(f"[{step.index}] would run: bio {cmd_str}")
        elif step.status == "skipped":
            click.echo(f"[{step.index}] skipped: bio {cmd_str}")
        elif step.status == "executed":
            click.echo(f"[{step.index}] ok: bio {cmd_str}{rewritten_note}")
        else:
            last_line = step.error.strip().splitlines()[-1] if step.error and step.error.strip() else "(no output captured)"
            click.echo(f"[{step.index}] FAILED: bio {cmd_str}{rewritten_note}\n    {last_line}")

    if not dry_run:
        click.echo(
            f"-- {report.n_executed} executed, {report.n_skipped} skipped, {report.n_failed} failed",
            err=True,
        )
        if report.n_failed:
            raise SystemExit(1)


@main.command("embed")
@click.option("--method", "embed_method", default="kmer", type=click.Choice(["kmer", "minhash", "esm", "prott5"]), help="Vectorization method.")
@click.option("--k", "kmer_k", type=int, default=None, help="k-mer length (kmer: default 3; minhash: default 9).")
@click.option("--model", "model_name", default=None, help="Model name/checkpoint for --method esm/prott5.")
@click.option("--reduce", "reduce_method", default="pca", type=click.Choice(["pca", "tsne", "umap"]), help="Dimensionality reduction.")
@click.option("--n-components", type=int, default=2)
@click.option("--export", "export_path", type=click.Path(path_type=Path), default=None, help="Write coordinates (csv/json) to this path.")
@click.option("--plot", "plot_output", type=click.Path(path_type=Path), default=None, help="Also render a Sequence Space scatter plot to this path.")
@click.option("--color-by", default=None, help="Metadata/tag field to color the plot by, e.g. seq_type or cluster_id. Defaults to seq_type when available.")
@shared_filter_options
def embed_cmd(embed_method, kmer_k, model_name, reduce_method, n_components, export_path, plot_output, color_by, **sel) -> None:
    """Sequence space analysis (spec section 15): embed sequences into
    vectors (k-mer/MinHash/ESM/ProtT5) and reduce to 2-3D (PCA/t-SNE/UMAP)
    for visualization. Restrict the input with the shared selection
    options (--tag/--type/--field/... and their --exclude-* counterparts)."""
    collection = project.load_collection()
    records = list(_apply_shared_filters(collection, **sel))
    if len(records) < 2:
        raise click.ClickException(f"only {len(records)} record(s) matched -- sequence space analysis needs at least 2")

    try:
        result = build_sequence_space(
            records,
            embed_method=embed_method,
            reduce_method=reduce_method,
            embed_kwargs={"k": kmer_k, "model_name": model_name},
            reduce_kwargs={"n_components": n_components},
        )
    except (RuntimeError, ValueError) as e:
        raise click.ClickException(str(e))

    click.echo(f"{len(records)} sequences -> {embed_method} embedding -> {reduce_method} ({result.coordinates.shape[1]}D)")
    for name, coord in zip(result.names, result.coordinates):
        click.echo(f"{name}\t" + "\t".join(f"{c:.4f}" for c in coord))

    if export_path is not None:
        rows = [
            {"seq_id": sid, "name": name, **{f"dim{i+1}": float(c) for i, c in enumerate(coord)}}
            for sid, name, coord in zip(result.seq_ids, result.names, result.coordinates)
        ]
        fmt = export_path.suffix.lstrip(".").lower() or "csv"
        if fmt == "csv":
            import csv as csv_mod
            with open(export_path, "w", newline="") as fh:
                writer = csv_mod.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        elif fmt == "json":
            import json as json_mod
            with open(export_path, "w") as fh:
                json_mod.dump(rows, fh, indent=2)
        else:
            raise click.ClickException(f"unsupported export format: {fmt} (use .csv or .json)")
        click.echo(f"exported coordinates to {export_path}")

    if plot_output is not None:
        from . import viz
        if color_by is None:
            labels = [r.seq_type.value for r in records]
        else:
            labels = [
                str(r.get(color_by)) if r.get(color_by) is not None
                else ("yes" if r.has_tag(color_by) else "no")
                for r in records
            ]
        try:
            viz.plot_sequence_space(result, plot_output, labels=labels)
        except RuntimeError as e:
            raise click.ClickException(str(e))
        click.echo(f"saved sequence space plot to {plot_output}")

    project.log_command()


@main.group("db")
def db_group() -> None:
    """Download public sequence databases for `bio search --db` (spec
    section 8's high-precision similarity search against a real DB,
    rather than just the current project)."""


@db_group.command("fetch")
@click.argument("name", required=False)
@click.option("--tool", required=True, type=click.Choice(["blast", "mmseqs", "pfam"]), help="Which downloader to use.")
@click.option("--output", "output_path", type=click.Path(path_type=Path), required=True, help="Output DB prefix/directory (blast/mmseqs) or .hmm file path (pfam).")
def db_fetch_cmd(name, tool, output_path) -> None:
    """Download a public DB (e.g. `bio db fetch nr --tool blast --output
    ./blastdb/nr`, `bio db fetch UniRef50 --tool mmseqs --output
    ./mmseqs_db/uniref50`, or `bio db fetch --tool pfam --output
    ./pfam/Pfam-A.hmm` -- NAME is ignored for --tool pfam, there's only
    one Pfam-A.hmm). See `bio db list` for common names; each tool's full
    catalog is larger and changes over time."""
    if tool != "pfam" and not name:
        raise click.ClickException(f"--tool {tool} needs a NAME argument (see `bio db list`)")
    try:
        result_path = bio_db.fetch_db(tool, name or "", output_path)
    except RuntimeError as e:
        raise click.ClickException(str(e))
    click.echo(f"downloaded {name or 'Pfam-A.hmm'} ({tool}) -> {result_path}")
    if tool == "pfam":
        click.echo(f"use it with: bio annotate pfam --hmm-db {result_path}")
    else:
        click.echo(f"use it with: bio search QUERY --method {tool} --db {result_path}")


@db_group.command("list")
def db_list_cmd() -> None:
    """Show a few commonly-used DB names for each downloader."""
    click.echo("blast (via update_blastdb.pl):")
    for name in bio_db.COMMON_BLAST_DBS:
        click.echo(f"  {name}")
    click.echo("mmseqs (via `mmseqs databases`):")
    for name in bio_db.COMMON_MMSEQS_DBS:
        click.echo(f"  {name}")
    click.echo("(not exhaustive -- run `update_blastdb.pl --showall` or `mmseqs databases` with no args for the full catalog)")


@main.group("annotate")
def annotate_group() -> None:
    """Sequence annotation (spec section 9). Classic in-process algorithms
    (orf/intron/promoter for DNA-RNA; protein-features/motif for protein)
    need nothing beyond Biopython. pfam/uniprot/interpro talk to a
    downloaded DB or a remote service -- see docs/TUTORIAL.md."""


def _annotate_targets(seq_type: str | None, tag: str | None):
    collection = project.load_collection()
    targets = collection
    if seq_type:
        targets = targets.filter(lambda r: r.seq_type.value == seq_type)
    if tag:
        targets = targets.filter(lambda r: r.has_tag(tag))
    records = list(targets)
    if not records:
        raise click.ClickException("no records matched the given filters")
    return collection, records


@annotate_group.command("orf")
@click.option("--tag", default=None, help="Restrict to records with this tag.")
@click.option("--min-length", type=int, default=30, help="Minimum ORF protein length (residues).")
@click.option("--single-strand", is_flag=True, help="Only scan the given strand (default: both).")
def annotate_orf_cmd(tag, min_length, single_strand) -> None:
    """Find candidate ORFs/CDS in DNA/RNA records (ATG...stop scan, all
    reading frames)."""
    collection, records = _annotate_targets(None, tag)
    records = [r for r in records if r.seq_type.value in ("dna", "rna")]
    if not records:
        raise click.ClickException("no dna/rna records matched")
    n_with_orf = 0
    for rec in records:
        hits = bio_annotate.find_orfs(rec.sequence, min_protein_length=min_length, both_strands=not single_strand)
        rec.set("orfs", [{"start": h.start, "end": h.end, "strand": h.strand, "frame": h.frame, "protein_length": len(h.protein), "protein": h.protein} for h in hits])
        if hits:
            rec.add_tag("has_orf")
            n_with_orf += 1
        click.echo(f"{rec.name}\t{len(hits)} orf(s)" + (f"\tlongest={len(hits[0].protein)}aa" if hits else ""))
    project.save_collection(collection)
    project.log_command()
    click.echo(f"-- {n_with_orf}/{len(records)} record(s) have at least one ORF", err=True)


@annotate_group.command("intron")
@click.option("--tag", default=None, help="Restrict to records with this tag.")
@click.option("--min-len", "min_len", type=int, default=20)
@click.option("--max-len", "max_len", type=int, default=5000)
def annotate_intron_cmd(tag, min_len, max_len) -> None:
    """Find candidate introns via the canonical GT...AG splice-site
    consensus (a boundary heuristic, not real gene prediction -- expect
    many candidates on real genomic sequence)."""
    collection, records = _annotate_targets(None, tag)
    records = [r for r in records if r.seq_type.value in ("dna", "rna")]
    if not records:
        raise click.ClickException("no dna/rna records matched")
    n_with_hit = 0
    for rec in records:
        hits = bio_annotate.find_canonical_introns(rec.sequence, min_intron_len=min_len, max_intron_len=max_len)
        rec.set("introns", [{"start": h.start, "end": h.end, "length": h.length} for h in hits])
        if hits:
            rec.add_tag("has_intron_candidate")
            n_with_hit += 1
        click.echo(f"{rec.name}\t{len(hits)} candidate(s)")
    project.save_collection(collection)
    project.log_command()
    click.echo(f"-- {n_with_hit}/{len(records)} record(s) have at least one candidate", err=True)


@annotate_group.command("promoter")
@click.option("--tag", default=None, help="Restrict to records with this tag.")
@click.option("--search-window", type=int, default=100, help="Bases from the sequence's end to search (pass in the region upstream of a presumed TSS).")
def annotate_promoter_cmd(tag, search_window) -> None:
    """Scan for a TATA-box-like motif near the end of DNA/RNA records."""
    collection, records = _annotate_targets(None, tag)
    records = [r for r in records if r.seq_type.value in ("dna", "rna")]
    if not records:
        raise click.ClickException("no dna/rna records matched")
    n_with_hit = 0
    for rec in records:
        hits = bio_annotate.find_tata_box(rec.sequence, search_window=search_window)
        rec.set("promoter_hits", [{"position": h.position, "motif": h.motif, "offset_from_end": h.offset_from_end} for h in hits])
        if hits:
            rec.add_tag("has_tata_box")
            n_with_hit += 1
        click.echo(f"{rec.name}\t{len(hits)} hit(s)")
    project.save_collection(collection)
    project.log_command()
    click.echo(f"-- {n_with_hit}/{len(records)} record(s) have a candidate TATA box", err=True)


@annotate_group.command("protein-features")
@click.option("--tag", default=None, help="Restrict to records with this tag.")
@click.option("--tm-threshold", type=float, default=1.6)
@click.option("--cc-threshold", type=float, default=1.0)
@click.option("--lc-entropy", type=float, default=3.0)
def annotate_protein_features_cmd(tag, tm_threshold, cc_threshold, lc_entropy) -> None:
    """Signal peptide + transmembrane regions + coiled coil + low-complexity
    regions, in one pass over protein records."""
    collection, records = _annotate_targets("protein", tag)
    for rec in records:
        sig = bio_annotate.predict_signal_peptide(rec.sequence)
        tm = bio_annotate.find_transmembrane_regions(rec.sequence, threshold=tm_threshold)
        cc = bio_annotate.find_coiled_coil(rec.sequence, threshold=cc_threshold)
        lc = bio_annotate.find_low_complexity_regions(rec.sequence, entropy_threshold=lc_entropy)

        rec.set("signal_peptide", {"is_signal_peptide": sig.is_signal_peptide, "cleavage_site": sig.cleavage_site, "score": sig.score})
        rec.set("tm_regions", [{"start": r.start, "end": r.end, "mean_hydropathy": round(r.mean_hydropathy, 3)} for r in tm])
        rec.set("coiled_coil_regions", [{"start": r.start, "end": r.end, "score": round(r.score, 3)} for r in cc])
        rec.set("low_complexity_regions", [{"start": r.start, "end": r.end, "entropy": round(r.entropy, 3)} for r in lc])

        if sig.is_signal_peptide:
            rec.add_tag("signal_peptide")
        if tm:
            rec.add_tag("transmembrane")
        if cc:
            rec.add_tag("coiled_coil")
        if lc:
            rec.add_tag("low_complexity")

        click.echo(f"{rec.name}\tsignal={sig.is_signal_peptide}\ttm={len(tm)}\tcoiled_coil={len(cc)}\tlow_complexity={len(lc)}")
    project.save_collection(collection)
    project.log_command()


@annotate_group.command("motif")
@click.option("--tag", default=None, help="Restrict to records with this tag.")
@click.option("--patterns", default=None, help="Comma-separated pattern IDs (default: all built-in patterns). See `bio annotate motif --list`.")
@click.option("--list", "list_patterns", is_flag=True, help="List the built-in pattern IDs and exit.")
def annotate_motif_cmd(tag, patterns, list_patterns) -> None:
    """Scan protein records against a small built-in PROSITE-style pattern
    set (N-glycosylation, kinase phosphorylation sites, Walker A, C2H2
    zinc finger, ...). Not the real PROSITE database -- see `bio annotate
    interpro` for the real thing."""
    if list_patterns:
        for pid, name in bio_annotate.list_prosite_patterns().items():
            click.echo(f"{pid}\t{name}")
        return

    collection, records = _annotate_targets("protein", tag)
    pattern_ids = [p.strip() for p in patterns.split(",")] if patterns else None
    n_with_hit = 0
    for rec in records:
        try:
            hits = bio_annotate.scan_prosite_patterns(rec.sequence, pattern_ids=pattern_ids)
        except ValueError as e:
            raise click.ClickException(str(e))
        rec.set("motifs", [{"pattern_id": h.pattern_id, "name": h.name, "start": h.start, "end": h.end, "matched_text": h.matched_text} for h in hits])
        for h in hits:
            rec.add_tag(f"motif:{h.pattern_id}")
        if hits:
            n_with_hit += 1
        click.echo(f"{rec.name}\t{len(hits)} hit(s)")
    project.save_collection(collection)
    project.log_command()
    click.echo(f"-- {n_with_hit}/{len(records)} record(s) have at least one motif hit", err=True)


@annotate_group.command("pfam")
@click.option("--hmm-db", "hmm_db_path", type=click.Path(exists=True, path_type=Path), required=True, help="Path to a pressed Pfam-A.hmm (see `bio db fetch --tool pfam`).")
@click.option("--tag", default=None, help="Restrict to records with this tag.")
@click.option("--evalue", type=float, default=1e-3)
def annotate_pfam_cmd(hmm_db_path, tag, evalue) -> None:
    """Domain annotation via a local Pfam-A.hmm scan (HMMER's hmmscan).
    Untested from this environment -- HMMER + a real Pfam-A.hmm aren't
    available here; the subprocess/parsing follows hmmscan's documented
    --domtblout format."""
    collection, records = _annotate_targets("protein", tag)
    with __import__("tempfile").TemporaryDirectory() as tmp:
        fasta_path = Path(tmp) / "query.fasta"
        bio_io.write_fasta(BioCollection(records), fasta_path)
        try:
            hits = annotate_external.run_hmmscan(fasta_path, hmm_db_path, evalue=evalue)
        except RuntimeError as e:
            raise click.ClickException(str(e))

    by_query: dict[str, list] = {}
    for h in hits:
        by_query.setdefault(h.query_id, []).append(h)
    for rec in records:
        rec_hits = by_query.get(rec.name, [])
        rec.set("pfam_domains", [{"name": h.domain_name, "accession": h.domain_accession, "evalue": h.evalue, "score": h.score, "start": h.start, "end": h.end} for h in rec_hits])
        for h in rec_hits:
            rec.add_tag(f"domain:{h.domain_accession}")
        click.echo(f"{rec.name}\t{len(rec_hits)} domain(s)")
    project.save_collection(collection)
    project.log_command()


@annotate_group.command("uniprot")
@click.argument("accession")
@click.option("--export", "export_path", type=click.Path(path_type=Path), default=None, help="Write the full raw JSON record to this path.")
def annotate_uniprot_cmd(accession, export_path) -> None:
    """Look up a single UniProtKB entry by accession. Standalone (not
    project-based) -- untested from this environment (no network route to
    rest.uniprot.org here)."""
    try:
        entry = annotate_external.fetch_uniprot(accession)
    except RuntimeError as e:
        raise click.ClickException(str(e))
    summary = annotate_external.summarize_uniprot_entry(entry)
    click.echo(f"accession: {summary['accession']}")
    click.echo(f"name: {summary['name']}")
    click.echo(f"organism: {summary['organism']}")
    click.echo(f"genes: {', '.join(g for g in summary['genes'] if g)}")
    click.echo(f"length: {summary['sequence_length']}")
    click.echo(f"Pfam domains: {', '.join(summary['pfam_domains']) or '(none)'}")
    if export_path is not None:
        import json as json_mod
        with open(export_path, "w") as fh:
            json_mod.dump(entry, fh, indent=2)
        click.echo(f"exported full record to {export_path}")


@annotate_group.command("interpro")
@click.option("--tag", default=None, help="Restrict to records with this tag.")
@click.option("--email", required=True, help="Contact email (required by EBI's job dispatcher).")
@click.option("--timeout", type=float, default=300.0, help="Max seconds to wait per sequence.")
@click.option("--interval", type=float, default=10.0, help="Polling interval in seconds.")
def annotate_interpro_cmd(tag, email, timeout, interval) -> None:
    """Domain/family annotation via the EBI InterProScan5 REST service, one
    project protein record at a time (submit -> poll -> fetch per
    sequence; can take minutes per sequence). Untested from this
    environment (no network route to ebi.ac.uk here)."""
    collection, records = _annotate_targets("protein", tag)
    for rec in records:
        click.echo(f"{rec.name}: submitting...")
        try:
            result = annotate_external.run_interproscan(rec.sequence, email=email, title=rec.name, timeout=timeout, interval=interval)
        except (RuntimeError, TimeoutError, ValueError) as e:
            click.echo(f"{rec.name}: FAILED -- {e}", err=True)
            continue
        rec.set("interpro", result)
        rec.add_tag("interpro_annotated")
        n_matches = len(result.get("results", [{}])[0].get("matches", [])) if result.get("results") else 0
        click.echo(f"{rec.name}: {n_matches} match(es)")
    project.save_collection(collection)
    project.log_command()


@main.command("report")
@click.option("--by", "by_specs", multiple=True, required=True, help="Axis spec, repeatable for an N-dimensional crosstab: type / tag:<name> / tag_prefix:<prefix> / field:<dotted.key> / field:<dotted.key>:bin<width>.")
@click.option("--export", "export_path", type=click.Path(path_type=Path), default=None, help="Write the crosstab to this CSV.")
@shared_filter_options
def report_cmd(by_specs, export_path, **sel) -> None:
    """Crosstab record counts across any combination of tag/type/metadata
    axes (not in the spec -- e.g. `bio report --by type --by
    tag_prefix:cluster_` to see how many of each seq_type fell into each
    cluster). Restrict the scope first with the shared selection options."""
    collection = project.load_collection()
    selected = list(_apply_shared_filters(collection, **sel))
    try:
        rows = bio_report.build_report(selected, list(by_specs))
    except ValueError as e:
        raise click.ClickException(str(e))

    header = list(by_specs) + ["count"]
    click.echo("\t".join(header))
    for row in rows:
        click.echo("\t".join(str(row[h]) for h in header))
    click.echo(f"-- {len(rows)} combination(s) over {len(selected)} record(s)", err=True)

    if export_path is not None:
        import csv as csv_mod
        with open(export_path, "w", newline="") as fh:
            writer = csv_mod.DictWriter(fh, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        click.echo(f"exported report to {export_path}", err=True)

    project.log_command()


@main.command("clean")
@click.option("--dedup-sequence", is_flag=True, help="Drop records whose sequence is identical to one already kept (first-wins).")
@click.option("--dedup-name", is_flag=True, help="Drop records whose name is identical to one already kept (first-wins).")
@click.option("--strip-gaps", "strip_gaps_flag", is_flag=True, help="Remove '-'/'.' gap characters from the raw sequence.")
@click.option("--trim-ambiguous-ends", is_flag=True, help="Trim leading/trailing ambiguous symbols (N/IUPAC codes for DNA/RNA, X for protein).")
@click.option("--max-ambiguous-fraction", type=float, default=None, help="Drop records whose ambiguous-symbol fraction (after other steps) exceeds this (0-1).")
@click.option("--result-min-length", "clean_min_length", type=int, default=None, help="Drop records shorter than this after other cleaning steps (distinct from the pre-selection --min-length above).")
@click.option("--result-max-length", "clean_max_length", type=int, default=None, help="Drop records longer than this after other cleaning steps (distinct from the pre-selection --max-length above).")
@click.option("--adapter", default=None, help="Adapter sequence to trim (exact match).")
@click.option("--adapter-end", default="both", type=click.Choice(["5", "3", "both"]), help="Which end(s) to trim --adapter from.")
@click.option("--min-quality", type=int, default=None, help="Sliding-window Phred quality trim from both ends (FASTQ imports only -- records without quality scores are left alone).")
@click.option("--quality-window", type=int, default=4, help="Window size for --min-quality's sliding average.")
@click.option("--dry-run", is_flag=True, help="Report what would change without saving to the project.")
@shared_filter_options
def clean_cmd(dedup_sequence, dedup_name, strip_gaps_flag, trim_ambiguous_ends, max_ambiguous_fraction, clean_min_length, clean_max_length, adapter, adapter_end, min_quality, quality_window, dry_run, **sel) -> None:
    """Sequence cleanup/QC (not in the spec): dedup, gap/adapter/ambiguous-end
    trimming, length/ambiguity filtering, and FASTQ quality trimming.
    Restrict scope with the shared selection options (--tag/--type/--field/...
    and their --exclude-* counterparts) -- unselected records are left
    untouched. By default the project is cleaned and saved back; use
    --dry-run to preview the counts first."""
    requested = any([
        dedup_sequence, dedup_name, strip_gaps_flag, trim_ambiguous_ends,
        max_ambiguous_fraction is not None, clean_min_length is not None,
        clean_max_length is not None, adapter, min_quality is not None,
    ])
    if not requested:
        raise click.ClickException(
            "no cleaning operation given -- pass at least one of "
            "--dedup-sequence/--dedup-name/--strip-gaps/--trim-ambiguous-ends/"
            "--max-ambiguous-fraction/--result-min-length/--result-max-length/--adapter/--min-quality"
        )

    collection = project.load_collection()
    targets = list(_apply_shared_filters(collection, **sel))
    if not targets:
        raise click.ClickException("no records matched the given filters")
    target_ids = {r.seq_id for r in targets}
    untouched = [r for r in collection if r.seq_id not in target_ids]

    try:
        report = clean_records(
            targets,
            dedup_sequence=dedup_sequence, dedup_name=dedup_name,
            strip_gaps_flag=strip_gaps_flag, trim_ambiguous=trim_ambiguous_ends,
            max_ambiguous_fraction=max_ambiguous_fraction,
            min_length=clean_min_length, max_length=clean_max_length,
            adapter=adapter, adapter_end=adapter_end,
            min_quality=min_quality, quality_window=quality_window,
        )
    except ValueError as e:
        raise click.ClickException(str(e))

    click.echo(f"{len(targets)} record(s) selected -> {report.kept} kept")
    for label, count in (
        ("gap-stripped", report.trimmed_gaps),
        ("adapter-trimmed", report.trimmed_adapter),
        ("quality-trimmed", report.trimmed_quality),
        ("ambiguous-end-trimmed", report.trimmed_ambiguous_ends),
        ("dropped (duplicate sequence)", report.dropped_duplicate_sequence),
        ("dropped (duplicate name)", report.dropped_duplicate_name),
        ("dropped (length out of range)", report.dropped_length),
        ("dropped (too ambiguous)", report.dropped_ambiguous),
        ("dropped (empty after trimming)", report.dropped_empty),
    ):
        if count:
            click.echo(f"  {label}: {count}")

    if dry_run:
        click.echo("(dry run -- project not modified)", err=True)
        return

    new_collection = BioCollection(untouched + report.kept_records)
    project.save_collection(new_collection)
    project.log_command()
    click.echo(f"project updated: {len(new_collection)} record(s) total")


if __name__ == "__main__":
    main()
