import json

import pytest
from click.testing import CliRunner

from bioexplorer import project
from bioexplorer.cli import main


@pytest.fixture
def project_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run(runner, args):
    result = runner.invoke(main, args, catch_exceptions=True)
    assert result.exit_code == 0, result.output
    return result


def _records():
    return json.loads((project.project_dir() / "records.json").read_text())


def test_clean_cli_no_operation_given_errors(project_dir):
    (project_dir / "seqs.fasta").write_text(">a\nACGT\n")
    runner = CliRunner()
    _run(runner, ["import", "seqs.fasta"])
    result = runner.invoke(main, ["clean"], catch_exceptions=True)
    assert result.exit_code != 0
    assert "no cleaning operation" in result.output


def test_clean_cli_dedup_sequence_updates_project(project_dir):
    (project_dir / "seqs.fasta").write_text(">a\nACGT\n>b\nACGT\n>c\nTTTT\n")
    runner = CliRunner()
    _run(runner, ["import", "seqs.fasta"])
    result = _run(runner, ["clean", "--dedup-sequence"])
    assert "3 record(s) selected -> 2 kept" in result.output
    assert "duplicate sequence" in result.output

    records = _records()
    assert len(records) == 2
    assert {r["name"] for r in records} == {"a", "c"}


def test_clean_cli_dry_run_does_not_modify_project(project_dir):
    (project_dir / "seqs.fasta").write_text(">a\nACGT\n>b\nACGT\n")
    runner = CliRunner()
    _run(runner, ["import", "seqs.fasta"])
    result = _run(runner, ["clean", "--dedup-sequence", "--dry-run"])
    assert "dry run" in result.output
    assert len(_records()) == 2  # unchanged


def test_clean_cli_strip_gaps(project_dir):
    (project_dir / "seqs.fasta").write_text(">a\nAC-GT\n")
    runner = CliRunner()
    _run(runner, ["import", "seqs.fasta"])
    _run(runner, ["clean", "--strip-gaps"])
    records = _records()
    assert records[0]["sequence"] == "ACGT"


def test_clean_cli_restricted_by_shared_selection(project_dir):
    (project_dir / "seqs.fasta").write_text(">a\nAC-GT\n>b\nAC-GT\n")
    runner = CliRunner()
    _run(runner, ["import", "seqs.fasta"])
    _run(runner, ["clean", "--strip-gaps", "--name", "a"])  # only touch 'a'
    records = {r["name"]: r["sequence"] for r in _records()}
    assert records["a"] == "ACGT"
    assert records["b"] == "AC-GT"  # untouched


def test_clean_cli_no_matching_records_errors(project_dir):
    (project_dir / "seqs.fasta").write_text(">a\nACGT\n")
    runner = CliRunner()
    _run(runner, ["import", "seqs.fasta"])
    result = runner.invoke(main, ["clean", "--strip-gaps", "--tag", "nonexistent"], catch_exceptions=True)
    assert result.exit_code != 0


def test_clean_cli_min_quality_on_fastq_import(project_dir):
    (project_dir / "reads.fastq").write_text(
        "@read1\nACGTACGT\n+\n!!IIIIII\n"  # leading 2 bases at phred 0, rest at 40
    )
    runner = CliRunner()
    _run(runner, ["import", "reads.fastq", "--format", "fastq"])
    before = _records()[0]
    assert before["quality"] == [0, 0, 40, 40, 40, 40, 40, 40]

    _run(runner, ["clean", "--min-quality", "20", "--quality-window", "1"])
    after = _records()[0]
    assert after["sequence"] == "GTACGT"
    assert after["quality"] == [40, 40, 40, 40, 40, 40]


def test_clean_cli_result_length_filter_distinct_from_selection_length_filter(project_dir):
    """--result-min-length/--result-max-length (post-trim) must not collide
    with the shared --min-length/--max-length (pre-selection) -- they used
    to share a flag name, which made Click warn and silently misbehave."""
    (project_dir / "seqs.fasta").write_text(">a\nNNACGTNN\n>b\nNNNNNNNN\n")
    runner = CliRunner()
    _run(runner, ["import", "seqs.fasta"])
    # pre-selection --min-length passes both (8 bases each); --result-max-length
    # then drops whatever is left shorter than 4 post-trim.
    result = _run(runner, ["clean", "--trim-ambiguous-ends", "--result-min-length", "4", "--min-length", "8"])
    assert "1 kept" in result.output
    records = _records()
    assert len(records) == 1
    assert records[0]["name"] == "a"


def test_clean_cli_replay_reproduces_result(project_dir):
    (project_dir / "seqs.fasta").write_text(">a\nACGT\n>b\nACGT\n>c\nTTTT\n")
    runner = CliRunner()
    _run(runner, ["import", "seqs.fasta"])
    _run(runner, ["clean", "--dedup-sequence"])
    before = sorted(r["name"] for r in _records())

    from bioexplorer.replay import replay

    report = replay()
    assert report.n_failed == 0, [s.error for s in report.steps if s.status == "failed"]
    after = sorted(r["name"] for r in _records())
    assert before == after == ["a", "c"]
