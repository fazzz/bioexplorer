"""Structure integration (spec section 16).

BioExplorer does not implement structure prediction itself -- this module
is a unified interface to external prediction/viewer tools, plus
in-process structure *analysis* built on Bio.PDB (parsing, sequence
extraction, RMSD/superposition, secondary structure via DSSP, and mapping
conservation scores from profile.py onto a structure's B-factor column for
external viewers to color by).

Formats: PDB and mmCIF, both via Bio.PDB.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from Bio.PDB import MMCIFIO, MMCIFParser, PDBIO, PDBParser, Superimposer
from Bio.PDB.Structure import Structure
from Bio.SeqUtils import seq1

from .align import PairwiseResult, pairwise_align
from .core import SeqType
from .similarity import _require_tool

# -- I/O --------------------------------------------------------------------


def read_structure(path: Path, structure_id: str | None = None) -> Structure:
    suffix = path.suffix.lower()
    sid = structure_id or path.stem
    if suffix in (".cif", ".mmcif"):
        parser = MMCIFParser(QUIET=True)
    elif suffix in (".pdb", ".ent"):
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError(f"unsupported structure format: {suffix} (use .pdb or .cif)")
    return parser.get_structure(sid, str(path))


def write_structure(structure: Structure, path: Path) -> Path:
    suffix = path.suffix.lower()
    io = MMCIFIO() if suffix in (".cif", ".mmcif") else PDBIO()
    io.set_structure(structure)
    io.save(str(path))
    return path


# -- sequence extraction ----------------------------------------------------


def structure_sequence(structure: Structure, chain_id: str | None = None, model_index: int = 0) -> dict[str, str]:
    """Extract the single-letter amino-acid sequence per chain (only
    standard residues with a CA atom are counted)."""
    model = structure[model_index]
    sequences: dict[str, str] = {}
    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        letters = []
        for residue in chain:
            if residue.id[0] != " " or "CA" not in residue:
                continue  # skip heteroatoms/waters and incomplete residues
            try:
                letters.append(seq1(residue.get_resname()))
            except KeyError:
                letters.append("X")
        if letters:
            sequences[chain.id] = "".join(letters)
    return sequences


def chain_residues(structure: Structure, chain_id: str, model_index: int = 0) -> list:
    """CA-bearing standard residues of a chain, in sequence order --
    parallel to structure_sequence()'s output for that chain."""
    model = structure[model_index]
    chain = model[chain_id]
    return [r for r in chain if r.id[0] == " " and "CA" in r]


# -- secondary structure (DSSP) ----------------------------------------------


def secondary_structure(pdb_path: Path, structure: Structure | None = None, model_index: int = 0) -> dict[tuple[str, int], str]:
    """Per-residue secondary structure assignment via DSSP. Requires the
    `mkdssp` (or `dssp`) binary. Returns {(chain_id, residue_seq_num): ss_code}."""
    from Bio.PDB.DSSP import DSSP

    binary = shutil.which("mkdssp") or shutil.which("dssp")
    if binary is None:
        raise RuntimeError(
            "secondary structure assignment requires the DSSP binary "
            "('mkdssp' or 'dssp') on PATH."
        )
    struct = structure if structure is not None else read_structure(pdb_path)
    model = struct[model_index]
    dssp = DSSP(model, str(pdb_path), dssp=binary)
    result = {}
    for key in dssp.keys():
        chain_id, res_id = key
        ss = dssp[key][2]
        result[(chain_id, res_id[1])] = ss
    return result


# -- structural alignment / RMSD --------------------------------------------


@dataclass
class SuperpositionResult:
    rmsd: float
    n_atoms: int
    sequence_alignment: PairwiseResult


def superimpose_structures(
    struct_a: Structure,
    struct_b: Structure,
    chain_a: str,
    chain_b: str,
) -> SuperpositionResult:
    """Sequence-guided structural superposition: pairwise-align the two
    chains' sequences (Needleman-Wunsch, BLOSUM62), take CA atoms at
    positions aligned without a gap on either side, and superimpose with
    Bio.PDB.Superimposer. This is the practical alternative to a dedicated
    structural aligner (TM-align/CE) when only Biopython is available --
    for very divergent structures a real structural aligner will do
    better, since it doesn't depend on sequence similarity."""
    seq_a_map = structure_sequence(struct_a, chain_id=chain_a)
    seq_b_map = structure_sequence(struct_b, chain_id=chain_b)
    if chain_a not in seq_a_map or chain_b not in seq_b_map:
        raise ValueError("requested chain not found (or has no CA-bearing residues)")

    residues_a = chain_residues(struct_a, chain_a)
    residues_b = chain_residues(struct_b, chain_b)

    alignment = pairwise_align(
        seq_a_map[chain_a], seq_b_map[chain_b], mode="global", seq_type=SeqType.PROTEIN,
        target_id=chain_a, query_id=chain_b,
    )

    atoms_a, atoms_b = [], []
    idx_a = idx_b = 0
    for ca, cb in zip(alignment.aligned_a, alignment.aligned_b):
        has_a, has_b = ca != "-", cb != "-"
        if has_a and has_b:
            atoms_a.append(residues_a[idx_a]["CA"])
            atoms_b.append(residues_b[idx_b]["CA"])
        if has_a:
            idx_a += 1
        if has_b:
            idx_b += 1

    if len(atoms_a) < 3:
        raise ValueError(
            f"only {len(atoms_a)} equivalent CA atom(s) found from the sequence "
            f"alignment -- too few to superimpose"
        )

    superimposer = Superimposer()
    superimposer.set_atoms(atoms_a, atoms_b)
    superimposer.apply(atom for res in struct_b[0][chain_b] for atom in res if res.id[0] == " ")

    return SuperpositionResult(
        rmsd=superimposer.rms,
        n_atoms=len(atoms_a),
        sequence_alignment=alignment,
    )


def structural_alignment_external(pdb_a: Path, pdb_b: Path, tool: str = "tmalign") -> str:
    """Real structure-based (sequence-independent) alignment via an
    external tool (TM-align by default). Returns the tool's raw stdout;
    callers can parse the TM-score / RMSD lines they need."""
    binary_name = {"tmalign": "TMalign", "usalign": "USalign"}.get(tool, tool)
    binary = _require_tool(binary_name)
    result = subprocess.run([binary, str(pdb_a), str(pdb_b)], check=True, capture_output=True, text=True)
    return result.stdout


# -- conservation mapping ---------------------------------------------------


def map_conservation_to_bfactor(
    structure: Structure,
    position_scores: list[float],
    chain_id: str,
    model_index: int = 0,
) -> Structure:
    """Write conservation scores (spec section 10's per-position
    conservation_score, aligned to this chain's residue order) into the
    B-factor column, so external viewers (PyMOL/ChimeraX/VMD) can color the
    structure by conservation with their built-in b-factor coloring."""
    residues = chain_residues(structure, chain_id, model_index=model_index)
    if len(position_scores) != len(residues):
        raise ValueError(
            f"got {len(position_scores)} score(s) for {len(residues)} residue(s) "
            f"in chain {chain_id} -- scores must be pre-mapped to this chain's "
            f"residue order (e.g. via a structure-aware alignment)"
        )
    for residue, score in zip(residues, position_scores):
        for atom in residue:
            atom.set_bfactor(float(score))
    return structure


# -- external prediction tools -----------------------------------------------


def predict_structure_colabfold(
    fasta_path: Path, output_dir: Path, extra_args: list[str] | None = None
) -> Path:
    binary = _require_tool("colabfold_batch")
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [binary, str(fasta_path), str(output_dir), *(extra_args or [])],
        check=True, capture_output=True, text=True,
    )
    return output_dir


def predict_structure_alphafold(
    fasta_path: Path, output_dir: Path, binary: str = "run_alphafold.sh", extra_args: list[str] | None = None
) -> Path:
    resolved = _require_tool(binary)
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [resolved, f"--fasta_paths={fasta_path}", f"--output_dir={output_dir}", *(extra_args or [])],
        check=True, capture_output=True, text=True,
    )
    return output_dir


def predict_structure_modeller(
    sequence_fasta: Path, template_pdb: Path, alignment_file: Path, output_dir: Path
) -> Path:
    """Homology modeling via MODELLER's Python API. Requires MODELLER
    (a licensed, separately-installed Python package named `modeller`) and
    a pre-built target-template alignment (e.g. from `bio align`)."""
    try:
        import modeller
        from modeller.automodel import automodel
    except ImportError as e:
        raise RuntimeError(
            "MODELLER is not installed (it's a separate, licensed package -- "
            "see https://salilab.org/modeller/). Use --engine colabfold or "
            "--engine alphafold instead if you don't have a license."
        ) from e

    output_dir.mkdir(parents=True, exist_ok=True)
    env = modeller.Environ()
    env.io.atom_files_directory = [str(template_pdb.parent)]
    a = automodel(env, alnfile=str(alignment_file), knowns=template_pdb.stem, sequence=sequence_fasta.stem)
    a.starting_model = 1
    a.ending_model = 1
    a.make()
    return output_dir


def predict_structure(
    fasta_path: Path, output_dir: Path, engine: str = "colabfold", **kwargs
) -> Path:
    if engine == "colabfold":
        return predict_structure_colabfold(fasta_path, output_dir, **kwargs)
    if engine == "alphafold":
        return predict_structure_alphafold(fasta_path, output_dir, **kwargs)
    if engine == "modeller":
        return predict_structure_modeller(fasta_path, output_dir=output_dir, **kwargs)
    raise ValueError(f"unknown structure prediction engine: {engine}")


# -- external viewers ---------------------------------------------------------

_VIEWER_BINARIES = {"vmd": "vmd", "chimerax": "chimerax", "pymol": "pymol"}


def view_structure(pdb_path: Path, viewer: str = "vmd", extra_args: list[str] | None = None) -> subprocess.Popen:
    """Launch an external 3D structure viewer (non-blocking), mirroring
    ChemExplorer's `chem view` external-viewer integration."""
    if viewer not in _VIEWER_BINARIES:
        raise ValueError(f"unknown viewer: {viewer} (choose from {list(_VIEWER_BINARIES)})")
    binary = _require_tool(_VIEWER_BINARIES[viewer])
    return subprocess.Popen([binary, str(pdb_path), *(extra_args or [])])
