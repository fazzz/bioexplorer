import json

import pytest
from click.testing import CliRunner

from bioexplorer import project
from bioexplorer.cli import main
from bioexplorer.replay import replay


@pytest.fixture
def project_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "seqs.fasta").write_text(">s1\nACGTACGT\n>s2\nACGTACGA\n")
    return tmp_path


def _run(runner, args):
    result = runner.invoke(main, args, catch_exceptions=True)
    assert result.exit_code == 0, result.output
    return result


def test_log_is_recorded_by_normal_commands(project_dir):
    runner = CliRunner()
    _run(runner, ["import", "seqs.fasta"])
    _run(runner, ["descriptor"])
    log = project.read_log()
    assert len(log) == 2
    assert log[0]["argv"] == ["import", "seqs.fasta"]
    assert log[1]["argv"] == ["descriptor"]


def test_replay_rebuilds_project_from_scratch(project_dir):
    runner = CliRunner()
    _run(runner, ["import", "seqs.fasta"])
    _run(runner, ["descriptor"])

    before = json.loads((project_dir / ".bioexplorer" / "records.json").read_text())
    assert before[0]["metadata"]["descriptor"]["length"] == 8

    report = replay()
    assert report.n_executed == 2
    assert report.n_failed == 0
    assert report.backup_path is not None
    assert report.backup_path.exists()

    after = json.loads((project_dir / ".bioexplorer" / "records.json").read_text())
    assert len(after) == len(before)
    assert after[0]["metadata"]["descriptor"]["length"] == 8


def test_replay_does_not_duplicate_log_entries(project_dir):
    runner = CliRunner()
    _run(runner, ["import", "seqs.fasta"])
    _run(runner, ["descriptor"])
    replay()
    log_after = project.read_log()
    assert len(log_after) == 2  # replay's sub-invocations must not re-append


def test_replay_dry_run_executes_nothing(project_dir):
    runner = CliRunner()
    _run(runner, ["import", "seqs.fasta"])
    _run(runner, ["descriptor"])

    report = replay(dry_run=True)
    assert all(s.status in ("would_execute", "skipped") for s in report.steps)
    # project state untouched: descriptor metadata still present
    records = json.loads((project_dir / ".bioexplorer" / "records.json").read_text())
    assert records[0]["metadata"]["descriptor"]["length"] == 8


def test_replay_skip_commands(project_dir):
    runner = CliRunner()
    _run(runner, ["import", "seqs.fasta"])
    _run(runner, ["descriptor"])

    report = replay(skip_commands={"descriptor"})
    statuses = {" ".join(s.argv): s.status for s in report.steps}
    assert statuses["descriptor"] == "skipped"
    assert statuses["import seqs.fasta"] == "executed"

    records = json.loads((project_dir / ".bioexplorer" / "records.json").read_text())
    assert "descriptor" not in records[0]["metadata"]


def test_replay_from_to_slicing(project_dir):
    runner = CliRunner()
    _run(runner, ["import", "seqs.fasta"])
    _run(runner, ["descriptor"])
    _run(runner, ["search", "--type", "dna"])

    report = replay(from_index=2, to_index=2)
    assert len(report.steps) == 1
    assert report.steps[0].argv == ["descriptor"]


def test_replay_stops_on_failure_by_default(project_dir):
    runner = CliRunner()
    _run(runner, ["import", "seqs.fasta"])
    # manually inject a bogus step into the log
    log_path = project_dir / ".bioexplorer" / "log.json"
    log = json.loads(log_path.read_text())
    log.append({"timestamp": "x", "argv": ["descriptor", "--type", "not-a-type"]})
    log.append({"timestamp": "x", "argv": ["descriptor"]})
    log_path.write_text(json.dumps(log))

    report = replay()
    assert report.n_failed == 1
    assert report.steps[-1].status == "failed"
    assert len(report.steps) == 2  # stopped before the 3rd (valid) step


def test_replay_continue_on_error(project_dir):
    runner = CliRunner()
    _run(runner, ["import", "seqs.fasta"])
    log_path = project_dir / ".bioexplorer" / "log.json"
    log = json.loads(log_path.read_text())
    log.append({"timestamp": "x", "argv": ["descriptor", "--type", "not-a-type"]})
    log.append({"timestamp": "x", "argv": ["descriptor"]})
    log_path.write_text(json.dumps(log))

    report = replay(continue_on_error=True)
    assert len(report.steps) == 3
    assert report.steps[-1].status == "executed"


def test_replay_no_log_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        replay()


def test_replay_cli_dry_run(project_dir):
    runner = CliRunner()
    _run(runner, ["import", "seqs.fasta"])
    result = runner.invoke(main, ["replay", "--dry-run"], catch_exceptions=True)
    assert result.exit_code == 0
    assert "would run" in result.output
