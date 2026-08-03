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

from . import io as bio_io
from . import project
from . import structure as bio_structure
from .align import format_alignment, multiple_align, pairwise_align
from .cluster import annotate_clusters, cluster_cdhit, cluster_greedy, cluster_mmseqs
from .core import BioCollection
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
def export_cmd(output: Path, fmt: str | None) -> None:
    """Export the current project to a file."""
    collection = project.load_collection()
    resolved_fmt = fmt or output.suffix.lstrip(".").lower()
    if resolved_fmt in ("fa", "fna", "faa"):
        resolved_fmt = "fasta"
    bio_io.write_collection(collection, output, resolved_fmt)
    project.log_command()
    click.echo(f"exported {len(collection)} record(s) to {output} ({resolved_fmt})")


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
@click.option("--name", default=None, help="Substring or (with --name-regex) regex match on name.")
@click.option("--name-regex", is_flag=True)
@click.option("--tag", default=None, help="Require this tag.")
@click.option("--type", "seq_type", default=None, type=click.Choice(["dna", "rna", "protein"]))
@click.option("--min-length", type=int, default=None)
@click.option("--max-length", type=int, default=None)
@click.option("--motif", default=None, help="Regex sequence motif, e.g. 'N[AG]G'.")
@click.option("--field", default=None, help="Dotted metadata field, e.g. descriptor.gc_percent or descriptor.pi.")
@click.option("--field-min", type=float, default=None)
@click.option("--field-max", type=float, default=None)
@click.option("--field-equals", default=None)
@click.option("--save-as", default=None, help="If set, replace the project with the result instead of just printing it.")
def search_cmd(query_file, method, kmer_k, top_n, min_score, name, name_regex, tag, seq_type, min_length, max_length, motif, field, field_min, field_max, field_equals, save_as) -> None:
    """Search the current project.

    With no QUERY_FILE: metadata/tag/motif filtering (spec section 7).
    With QUERY_FILE: similarity search against the project (spec section 8),
    then optionally narrowed further by the same filter options.
    """
    collection = project.load_collection()

    if query_file is not None:
        query_records = bio_io.read_file(query_file)
        if not query_records:
            raise click.ClickException(f"no sequences found in {query_file}")
        query_seq = query_records[0].sequence
        if len(query_records) > 1:
            click.echo(f"note: {query_file} has {len(query_records)} sequences; using the first ('{query_records[0].name}') as query", err=True)
        try:
            hits = search_similar(collection, query_seq, method=method, k=kmer_k, top_n=top_n, min_score=min_score)
        except RuntimeError as e:
            raise click.ClickException(str(e))
        result = BioCollection()
        for hit in hits:
            hit.record.set("similarity_score", round(hit.score, 4))
            if hit.record.seq_id not in result:
                result.add(hit.record)
        result = run_filters(
            result, name=name, name_regex=name_regex, tag=tag, seq_type=seq_type,
            min_length=min_length, max_length=max_length, motif=motif,
            field=field, field_min=field_min, field_max=field_max, field_equals=field_equals,
        )
        ordered = sorted(result, key=lambda r: -r.get("similarity_score", 0.0))
        for rec in ordered:
            click.echo(f"{rec.seq_id}\t{rec.name}\t{rec.seq_type.value}\t{rec.length}\tscore={rec.get('similarity_score')}")
        click.echo(f"-- {len(ordered)} hit(s) (method={method})", err=True)
    else:
        result = run_filters(
            collection, name=name, name_regex=name_regex, tag=tag, seq_type=seq_type,
            min_length=min_length, max_length=max_length, motif=motif,
            field=field, field_min=field_min, field_max=field_max, field_equals=field_equals,
        )
        for rec in result:
            click.echo(f"{rec.seq_id}\t{rec.name}\t{rec.seq_type.value}\t{rec.length}")
        click.echo(f"-- {len(result)}/{len(collection)} record(s) matched", err=True)

    if save_as == "project":
        project.save_collection(result)
        project.log_command()
        click.echo("project replaced with search result", err=True)


@main.command("align")
@click.option("--pairwise", "pairwise_ids", nargs=2, default=None, help="Two seq_ids for pairwise alignment (Needleman-Wunsch/Smith-Waterman).")
@click.option("--mode", default="global", type=click.Choice(["global", "local"]), help="Pairwise mode: global=Needleman-Wunsch, local=Smith-Waterman.")
@click.option("--tool", default="mafft", type=click.Choice(["mafft", "muscle", "clustalo"]), help="Multiple alignment tool (used when --pairwise is not given).")
@click.option("--type", "seq_type", default=None, type=click.Choice(["dna", "rna", "protein"]), help="Restrict multiple alignment to records of this type.")
@click.option("--tag", default=None, help="Restrict multiple alignment to records with this tag.")
@click.option("--name", default="default", help="Name to save the multiple alignment under (.bioexplorer/alignments/<name>.fasta).")
def align_cmd(pairwise_ids, mode, tool, seq_type, tag, name) -> None:
    """Align sequences (spec section 11).

    --pairwise ID1 ID2: pairwise alignment between two records in the
    project (no external tool required).
    Without --pairwise: multiple alignment of the whole project (optionally
    filtered by --type/--tag) via mafft/muscle/clustalo, saved as a named
    alignment for later use by `bio profile` / `bio tree`.
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

    targets = collection
    if seq_type:
        targets = targets.filter(lambda r: r.seq_type.value == seq_type)
    if tag:
        targets = targets.filter(lambda r: r.has_tag(tag))
    records = list(targets)
    if len(records) < 2:
        raise click.ClickException(
            f"only {len(records)} record(s) matched the given filters; "
            f"multiple alignment needs at least 2"
        )

    try:
        aligned = multiple_align(records, tool=tool)
    except RuntimeError as e:
        raise click.ClickException(str(e))

    path = project.save_alignment(aligned, name=name)
    project.log_command()
    click.echo(f"aligned {len(aligned)} record(s) with {tool}, saved as alignment '{name}' -> {path}")


@main.command("profile")
@click.argument("alignment_file", required=False, type=click.Path(exists=True, path_type=Path))
@click.option("--name", default="default", help="Named alignment to use if ALIGNMENT_FILE is not given (from `bio align --name`).")
@click.option("--pseudocount", type=float, default=0.5, help="Pseudocount added to each symbol count before normalizing to probabilities.")
@click.option("--export", "export_path", type=click.Path(path_type=Path), default=None, help="Write the position-wise profile table (csv/tsv/json) to this path.")
@click.option("--plot", "plot_kind", default=None, type=click.Choice(["logo", "heatmap", "conservation"]), help="Also render a plot.")
@click.option("--plot-output", type=click.Path(path_type=Path), default=None, help="Path for --plot output (default: profile_<kind>.png).")
def profile_cmd(alignment_file, name, pseudocount, export_path, plot_kind, plot_output) -> None:
    """Compute a sequence profile from an alignment (spec section 10):
    PFM/PPM/PWM/PSSM, consensus sequence, Shannon entropy, conservation
    score, and relative entropy, per alignment position."""
    try:
        records = _load_alignment(alignment_file, name)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))

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
@click.option("--name", default="default", help="Named alignment to use if ALIGNMENT_FILE is not given.")
@click.option("--output", type=click.Path(path_type=Path), default=Path("logo.png"), help="Output image path (.png/.svg/.pdf).")
def logo_cmd(alignment_file, name, output) -> None:
    """Render a sequence logo from an alignment (shortcut for
    `bio profile --plot logo`)."""
    try:
        records = _load_alignment(alignment_file, name)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    profile = build_profile(records)
    from . import viz
    try:
        viz.plot_sequence_logo(profile, output)
    except RuntimeError as e:
        raise click.ClickException(str(e))
    project.log_command()
    click.echo(f"saved sequence logo to {output}")


@main.command("cluster")
@click.option("--method", default="greedy", type=click.Choice(["greedy", "cdhit", "mmseqs"]), help="Clustering algorithm.")
@click.option("--type", "seq_type", default=None, type=click.Choice(["dna", "rna", "protein"]), help="Restrict clustering to records of this type.")
@click.option("--tag", default=None, help="Restrict clustering to records with this tag.")
@click.option("--similarity", "similarity_method", default="kmer", type=click.Choice(["kmer", "minhash"]), help="Similarity proxy for --method greedy and for centroid selection.")
@click.option("--k", "kmer_k", type=int, default=None)
@click.option("--threshold", type=float, default=0.8, help="Similarity threshold for --method greedy (0-1).")
@click.option("--identity", type=float, default=0.9, help="Sequence identity for --method cdhit (0-1).")
@click.option("--min-seq-id", type=float, default=0.9, help="Minimum sequence identity for --method mmseqs (0-1).")
@click.option("--no-consensus", is_flag=True, help="Skip consensus computation (faster; representative/centroid still computed).")
@click.option("--msa-tool", default="mafft", type=click.Choice(["mafft", "muscle", "clustalo", "none"]), help="MSA tool for consensus of variable-length clusters; 'none' disables the fallback alignment step.")
@click.option("--save-as", default=None, help="If 'project', write cluster tags/metadata back into the project.")
def cluster_cmd(method, seq_type, tag, similarity_method, kmer_k, threshold, identity, min_seq_id, no_consensus, msa_tool, save_as) -> None:
    """Cluster the current project's sequences (spec section 12) and pick a
    representative/centroid/consensus for each cluster."""
    collection = project.load_collection()
    targets = collection
    if seq_type:
        targets = targets.filter(lambda r: r.seq_type.value == seq_type)
    if tag:
        targets = targets.filter(lambda r: r.has_tag(tag))
    records = list(targets)
    if not records:
        raise click.ClickException("no records matched the given filters")

    try:
        if method == "greedy":
            clusters = cluster_greedy(records, method=similarity_method, k=kmer_k, threshold=threshold)
        elif method == "cdhit":
            clusters = cluster_cdhit(records, seq_type=records[0].seq_type, identity=identity)
        else:
            clusters = cluster_mmseqs(records, min_seq_id=min_seq_id)
    except RuntimeError as e:
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
@click.option("--name", default="default", help="Named alignment to use if ALIGNMENT_FILE is not given (from `bio align --name`).")
@click.option("--method", default="nj", type=click.Choice(["nj", "upgma", "iqtree", "fasttree", "raxml"]), help="Tree-building method.")
@click.option("--model", default=None, help="Substitution model (nj/upgma: Bio.Phylo model name, e.g. identity/blastn/blosum62; iqtree/raxml: model string).")
@click.option("--bootstrap", type=int, default=0, help="Bootstrap replicates for --method nj/upgma (0 disables).")
@click.option("--save-as", default="default", help="Name to save the tree under (.bioexplorer/trees/<name>.nwk).")
@click.option("--output", type=click.Path(path_type=Path), default=None, help="Also write the Newick tree to this path.")
def tree_cmd(alignment_file, name, method, model, bootstrap, save_as, output) -> None:
    """Build a phylogenetic tree from an alignment (spec section 14) and
    save it as Newick."""
    try:
        records = _load_alignment(alignment_file, name)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))

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

    saved_path = project.save_tree(tree, name=save_as)
    click.echo(f"saved tree as '{save_as}' -> {saved_path}")
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
@click.option("--type", "seq_type", default=None, type=click.Choice(["dna", "rna"]), help="Restrict --all to records of this type.")
@click.option("--tag", default=None, help="Restrict --all to records with this tag.")
def dnds_cmd(ids, method, all_pairs, seq_type, tag) -> None:
    """dN/dS (Ka/Ks) between codon-aligned coding sequences (spec section 13).

    `bio dnds ID1 ID2` for a single pair, or `bio dnds --all` for every
    pair in the project (optionally filtered by --type/--tag). Sequences
    must be codon-aligned: equal length, a multiple of 3.
    """
    collection = project.load_collection()

    try:
        if all_pairs:
            targets = collection
            if seq_type:
                targets = targets.filter(lambda r: r.seq_type.value == seq_type)
            if tag:
                targets = targets.filter(lambda r: r.has_tag(tag))
            records = list(targets)
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
@click.option("--name", default="default", help="Named alignment to use if ALIGNMENT_FILE is not given.")
@click.option("--color-by", default="residue", type=click.Choice(["residue", "conservation"]))
@click.option("--max-positions", type=int, default=200, help="Truncate to this many positions (0 = no limit).")
@click.option("--output", type=click.Path(path_type=Path), default=Path("alignment.png"))
def plot_alignment_cmd(alignment_file, name, color_by, max_positions, output) -> None:
    """Render a Jalview-style colored alignment viewer."""
    try:
        records = _load_alignment(alignment_file, name)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
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
@click.option("--name", default="default", help="Named project tree to use if TREE_FILE is not given (from `bio tree --save-as`).")
@click.option("--no-confidence", is_flag=True, help="Hide bootstrap/confidence branch labels.")
@click.option("--output", type=click.Path(path_type=Path), default=Path("tree.png"))
def plot_tree_cmd(tree_file, name, no_confidence, output) -> None:
    """Render a phylogenetic tree as a dendrogram."""
    from .tree import read_newick
    if tree_file is not None:
        tree = read_newick(tree_file)
    else:
        try:
            tree = project.load_tree(name)
        except FileNotFoundError as e:
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
@click.option("--skip", "skip_str", default=None, help="Comma-separated top-level commands to skip (default: 'structure,replay' -- interactive viewer + self-recursion).")
@click.option("--continue-on-error", is_flag=True, help="Keep going past a failed step instead of stopping.")
@click.option("--no-reset", is_flag=True, help="Replay on top of the current project state instead of rebuilding from scratch.")
def replay_cmd(dry_run, from_index, to_index, skip_str, continue_on_error, no_reset) -> None:
    """Replay the recorded command history (spec section 19) to rebuild
    the project from scratch, verifying the pipeline is reproducible."""
    if skip_str is not None:
        skip_commands = {s.strip() for s in skip_str.split(",") if s.strip()}
    else:
        from .replay import DEFAULT_SKIP
        skip_commands = DEFAULT_SKIP | {"structure"}

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
        if step.status == "would_execute":
            click.echo(f"[{step.index}] would run: bio {cmd_str}")
        elif step.status == "skipped":
            click.echo(f"[{step.index}] skipped: bio {cmd_str}")
        elif step.status == "executed":
            click.echo(f"[{step.index}] ok: bio {cmd_str}")
        else:
            last_line = step.error.strip().splitlines()[-1] if step.error and step.error.strip() else "(no output captured)"
            click.echo(f"[{step.index}] FAILED: bio {cmd_str}\n    {last_line}")

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
@click.option("--type", "seq_type", default=None, type=click.Choice(["dna", "rna", "protein"]), help="Restrict to records of this type.")
@click.option("--tag", default=None, help="Restrict to records with this tag.")
@click.option("--export", "export_path", type=click.Path(path_type=Path), default=None, help="Write coordinates (csv/json) to this path.")
@click.option("--plot", "plot_output", type=click.Path(path_type=Path), default=None, help="Also render a Sequence Space scatter plot to this path.")
@click.option("--color-by", default=None, help="Metadata/tag field to color the plot by, e.g. seq_type or cluster_id. Defaults to seq_type when available.")
def embed_cmd(embed_method, kmer_k, model_name, reduce_method, n_components, seq_type, tag, export_path, plot_output, color_by) -> None:
    """Sequence space analysis (spec section 15): embed sequences into
    vectors (k-mer/MinHash/ESM/ProtT5) and reduce to 2-3D (PCA/t-SNE/UMAP)
    for visualization."""
    collection = project.load_collection()
    targets = collection
    if seq_type:
        targets = targets.filter(lambda r: r.seq_type.value == seq_type)
    if tag:
        targets = targets.filter(lambda r: r.has_tag(tag))
    records = list(targets)
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


if __name__ == "__main__":
    main()
