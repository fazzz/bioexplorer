import math

import pytest

from bioexplorer import structure as struct_mod

pytest.importorskip("Bio.PDB")


def _pdb_atom_line(atom_num, atom_name, resname, chain, resseq, x, y, z):
    return (
        f"ATOM  {atom_num:>5} {atom_name:<4} {resname:>3} {chain}{resseq:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.00:>6.2f}{0.00:>6.2f}          "
        f"{atom_name[0]:>2}\n"
    )


def _write_synthetic_pdb(path, chain="A", resnames=("ALA", "GLY", "SER", "LEU", "VAL"), offset=(0.0, 0.0, 0.0)):
    lines = []
    atom_num = 1
    for i, resname in enumerate(resnames):
        x, y, z = i * 3.8 + offset[0], offset[1], offset[2]
        lines.append(_pdb_atom_line(atom_num, "CA", resname, chain, i + 1, x, y, z))
        atom_num += 1
    lines.append("END\n")
    path.write_text("".join(lines))


@pytest.fixture
def struct_a_path(tmp_path):
    path = tmp_path / "a.pdb"
    _write_synthetic_pdb(path, chain="A")
    return path


@pytest.fixture
def struct_b_path(tmp_path):
    # identical relative geometry, translated -- should superimpose to ~0 RMSD
    path = tmp_path / "b.pdb"
    _write_synthetic_pdb(path, chain="A", offset=(10.0, 5.0, -2.0))
    return path


def test_read_structure_pdb(struct_a_path):
    s = struct_mod.read_structure(struct_a_path)
    assert s is not None
    assert len(list(s.get_chains())) == 1


def test_read_structure_unsupported_suffix_raises(tmp_path):
    bad = tmp_path / "structure.xyz"
    bad.write_text("not a structure")
    with pytest.raises(ValueError):
        struct_mod.read_structure(bad)


def test_structure_sequence_extraction(struct_a_path):
    s = struct_mod.read_structure(struct_a_path)
    seqs = struct_mod.structure_sequence(s)
    assert "A" in seqs
    assert seqs["A"] == "AGSLV"  # ALA GLY SER LEU VAL


def test_chain_residues_count(struct_a_path):
    s = struct_mod.read_structure(struct_a_path)
    residues = struct_mod.chain_residues(s, "A")
    assert len(residues) == 5


def test_write_structure_pdb_roundtrip(tmp_path, struct_a_path):
    s = struct_mod.read_structure(struct_a_path)
    out = tmp_path / "out.pdb"
    struct_mod.write_structure(s, out)
    reloaded = struct_mod.read_structure(out)
    assert struct_mod.structure_sequence(reloaded)["A"] == "AGSLV"


def test_superimpose_identical_geometry_near_zero_rmsd(struct_a_path, struct_b_path):
    struct_a = struct_mod.read_structure(struct_a_path, structure_id="a")
    struct_b = struct_mod.read_structure(struct_b_path, structure_id="b")
    result = struct_mod.superimpose_structures(struct_a, struct_b, "A", "A")
    assert result.n_atoms == 5
    assert result.rmsd < 1e-3


def test_superimpose_missing_chain_raises(struct_a_path, struct_b_path):
    struct_a = struct_mod.read_structure(struct_a_path, structure_id="a")
    struct_b = struct_mod.read_structure(struct_b_path, structure_id="b")
    with pytest.raises(ValueError):
        struct_mod.superimpose_structures(struct_a, struct_b, "A", "Z")


def test_map_conservation_to_bfactor(struct_a_path):
    s = struct_mod.read_structure(struct_a_path)
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    struct_mod.map_conservation_to_bfactor(s, scores, chain_id="A")
    residues = struct_mod.chain_residues(s, "A")
    for residue, expected in zip(residues, scores):
        for atom in residue:
            assert atom.get_bfactor() == pytest.approx(expected)


def test_map_conservation_wrong_length_raises(struct_a_path):
    s = struct_mod.read_structure(struct_a_path)
    with pytest.raises(ValueError):
        struct_mod.map_conservation_to_bfactor(s, [0.1, 0.2], chain_id="A")


def test_secondary_structure_missing_dssp_raises_clear_error(struct_a_path):
    with pytest.raises(RuntimeError, match="DSSP"):
        struct_mod.secondary_structure(struct_a_path)


def test_predict_structure_colabfold_missing_binary_raises(tmp_path):
    fasta = tmp_path / "q.fasta"
    fasta.write_text(">q\nACDEFG\n")
    with pytest.raises(RuntimeError, match="not found on PATH"):
        struct_mod.predict_structure(fasta, tmp_path / "out", engine="colabfold")


def test_predict_structure_modeller_missing_package_raises(tmp_path):
    fasta = tmp_path / "q.fasta"
    fasta.write_text(">q\nACDEFG\n")
    template = tmp_path / "template.pdb"
    template.write_text("")
    aln = tmp_path / "aln.pir"
    aln.write_text("")
    with pytest.raises(RuntimeError, match="MODELLER"):
        struct_mod.predict_structure(
            fasta, tmp_path / "out", engine="modeller",
            template_pdb=template, alignment_file=aln,
        )


def test_predict_structure_unknown_engine_raises(tmp_path):
    fasta = tmp_path / "q.fasta"
    fasta.write_text(">q\nACDEFG\n")
    with pytest.raises(ValueError):
        struct_mod.predict_structure(fasta, tmp_path / "out", engine="not-an-engine")


def test_view_structure_missing_binary_raises(struct_a_path):
    with pytest.raises(RuntimeError, match="not found on PATH"):
        struct_mod.view_structure(struct_a_path, viewer="vmd")


def test_view_structure_unknown_viewer_raises(struct_a_path):
    with pytest.raises(ValueError):
        struct_mod.view_structure(struct_a_path, viewer="not-a-viewer")
