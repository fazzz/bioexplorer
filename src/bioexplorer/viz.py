"""Visualization (spec section 17, with Sequence Logo/Heatmap/Conservation
Plot doubling as section 10's profile visualizations, and Sequence Space
rendering embed.py's PCA/t-SNE/UMAP output).

Covered here: Alignment Viewer, Sequence Logo, Conservation Plot,
Phylogenetic Tree, Sequence Space. Domain Architecture is still deferred --
it renders the output of annotation (spec section 9), which doesn't exist
yet.

All plotting requires matplotlib (``pip install -e '.[viz]'``); missing it
raises a clear RuntimeError rather than failing on some baffling ImportError
deep in the call stack.
"""

from __future__ import annotations

from pathlib import Path

from .profile import Profile


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless, no display needed
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError(
            "plotting requires matplotlib (pip install -e '.[viz]')"
        ) from e
    return plt


# Rough letter widths so stacked-letter glyphs aren't visually lopsided;
# good enough for a proportional-ish logo without a dedicated font metrics dep.
_LOGO_COLORS = {
    # DNA/RNA
    "A": "#2ca02c", "C": "#1f77b4", "G": "#ff7f0e", "T": "#d62728", "U": "#d62728",
}
_PROTEIN_LOGO_PALETTE = [
    "#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _logo_color(symbol: str, alphabet: list[str]) -> str:
    if symbol in _LOGO_COLORS:
        return _LOGO_COLORS[symbol]
    return _PROTEIN_LOGO_PALETTE[alphabet.index(symbol) % len(_PROTEIN_LOGO_PALETTE)]


def plot_sequence_logo(profile: Profile, output: Path, width_per_pos: float = 0.4) -> Path:
    """Information-content-scaled sequence logo: at each position, letters
    are stacked tallest-to-shortest, each letter's height proportional to
    its frequency times the position's relative entropy (bits)."""
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(max(6, profile.length * width_per_pos), 3))

    for pos in range(profile.length):
        info = max(profile.relative_entropy[pos], 0.0)
        freqs = sorted(
            ((symbol, profile.ppm[pos][symbol]) for symbol in profile.alphabet),
            key=lambda kv: kv[1],
        )
        y = 0.0
        for symbol, freq in freqs:
            height = freq * info
            if height <= 0:
                continue
            ax.text(
                pos + 0.5, y + height / 2, symbol,
                ha="center", va="center",
                fontsize=max(6, height * 40),
                fontweight="bold",
                color=_logo_color(symbol, profile.alphabet),
                family="monospace",
            )
            y += height

    max_info = max(profile.relative_entropy) if profile.relative_entropy else 1.0
    ax.set_xlim(0, profile.length)
    ax.set_ylim(0, max(max_info, 0.1) * 1.1)
    ax.set_xlabel("Position")
    ax.set_ylabel("Bits")
    ax.set_title("Sequence Logo")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_heatmap(profile: Profile, output: Path) -> Path:
    """Position x residue frequency heatmap (the PPM, visualized)."""
    plt = _require_matplotlib()
    import numpy as np

    matrix = np.array([[row[s] for row in profile.ppm] for s in profile.alphabet])

    fig, ax = plt.subplots(figsize=(max(6, profile.length * 0.3), max(3, len(profile.alphabet) * 0.3)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(profile.alphabet)))
    ax.set_yticklabels(profile.alphabet)
    ax.set_xlabel("Position")
    ax.set_ylabel("Residue")
    ax.set_title("Position-wise Residue Frequency")
    fig.colorbar(im, ax=ax, label="Frequency")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_conservation(profile: Profile, output: Path) -> Path:
    """Per-position conservation score (1 - H/H_max) as a bar plot, with
    the consensus residue labelled along the top."""
    plt = _require_matplotlib()

    fig, ax = plt.subplots(figsize=(max(6, profile.length * 0.25), 3))
    positions = list(range(1, profile.length + 1))
    ax.bar(positions, profile.conservation_score, color="#4c72b0", width=0.9)
    ax.set_xlabel("Position")
    ax.set_ylabel("Conservation score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Conservation Plot")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


# -- Alignment Viewer ---------------------------------------------------


def plot_alignment_viewer(
    records,
    output: Path,
    color_by: str = "residue",
    cell_width: float = 0.25,
    cell_height: float = 0.3,
    max_positions: int | None = 200,
) -> Path:
    """Jalview-style colored-grid view of an alignment: one row per
    sequence, one cell per position, letter drawn on a residue-colored
    background. ``color_by='residue'`` colors every cell by its own
    residue; ``color_by='conservation'`` colors cells by how conserved
    that column is (via profile.py), independent of which residue.

    ``max_positions`` truncates very long alignments to keep the image a
    sane size; pass None to disable.
    """
    plt = _require_matplotlib()
    from .profile import build_profile, default_alphabet

    length = records[0].length
    if any(r.length != length for r in records):
        raise ValueError("plot_alignment_viewer needs an alignment (equal-length sequences)")

    display_length = length if max_positions is None else min(length, max_positions)
    alphabet = default_alphabet(records[0].seq_type)

    conservation = None
    if color_by == "conservation":
        conservation = build_profile(records).conservation_score

    n_seqs = len(records)
    fig, ax = plt.subplots(figsize=(max(6, display_length * cell_width), max(2, n_seqs * cell_height)))

    for row, rec in enumerate(records):
        y = n_seqs - row - 1
        for pos in range(display_length):
            char = rec.sequence[pos]
            if color_by == "conservation" and conservation is not None:
                score = conservation[pos]
                bg = plt.cm.viridis(score)
            elif char == "-":
                bg = "#f0f0f0"
            else:
                bg = _logo_color(char, alphabet) if char in alphabet or char in _LOGO_COLORS else "#cccccc"
                # soften the logo palette for a background fill rather than bold letter color
                import matplotlib.colors as mcolors

                r, g, b = mcolors.to_rgb(bg)
                bg = (r * 0.4 + 0.6, g * 0.4 + 0.6, b * 0.4 + 0.6)
            ax.add_patch(plt.Rectangle((pos, y), 1, 1, facecolor=bg, edgecolor="none"))
            if display_length <= 120:  # letters get unreadable beyond this
                ax.text(pos + 0.5, y + 0.5, char, ha="center", va="center", fontsize=7, family="monospace")

    ax.set_xlim(0, display_length)
    ax.set_ylim(0, n_seqs)
    ax.set_yticks([n_seqs - i - 0.5 for i in range(n_seqs)])
    ax.set_yticklabels([r.name for r in records], fontsize=8)
    ax.set_xlabel("Position" + (f" (showing 1-{display_length} of {length})" if display_length < length else ""))
    ax.set_title("Alignment Viewer")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


# -- Phylogenetic Tree ----------------------------------------------------


def plot_tree(tree, output: Path, show_confidence: bool = True) -> Path:
    """Render a Bio.Phylo tree (from tree.py) as a dendrogram."""
    plt = _require_matplotlib()
    from Bio import Phylo

    n_taxa = tree.count_terminals()
    fig, ax = plt.subplots(figsize=(8, max(3, n_taxa * 0.3)))

    def branch_label(clade):
        if show_confidence and clade.confidence is not None and not clade.is_terminal():
            return f"{clade.confidence:.0f}"
        return None

    Phylo.draw(
        tree, axes=ax, do_show=False,
        branch_labels=branch_label if show_confidence else None,
    )
    ax.set_title("Phylogenetic Tree")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


# -- Sequence Space ---------------------------------------------------------


def plot_sequence_space(
    result,
    output: Path,
    labels: list[str] | None = None,
    label_points: bool = False,
    point_size: float = 30.0,
) -> Path:
    """2D scatter of a SequenceSpaceResult (spec section 15's PCA/t-SNE/UMAP
    output). ``labels`` (one per record, e.g. cluster_id or seq_type)
    colors points by category if given; otherwise every point is one color."""
    plt = _require_matplotlib()

    coords = result.coordinates
    fig, ax = plt.subplots(figsize=(7, 6))

    if labels is not None:
        unique = sorted(set(labels))
        cmap = plt.get_cmap("tab10" if len(unique) <= 10 else "tab20")
        color_of = {label: cmap(i % cmap.N) for i, label in enumerate(unique)}
        for label in unique:
            idx = [i for i, l in enumerate(labels) if l == label]
            ax.scatter(coords[idx, 0], coords[idx, 1], s=point_size, color=color_of[label], label=str(label))
        ax.legend(fontsize=7, loc="best", framealpha=0.8)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=point_size, color="#4c72b0")

    if label_points:
        for i, name in enumerate(result.names):
            ax.annotate(name, (coords[i, 0], coords[i, 1]), fontsize=6, alpha=0.8)

    ax.set_xlabel(f"{result.reduce_method.upper()} 1")
    ax.set_ylabel(f"{result.reduce_method.upper()} 2")
    ax.set_title(f"Sequence Space ({result.embed_method} + {result.reduce_method})")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output
