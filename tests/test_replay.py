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


# -- seq_id rewriting (ports ProteinExplorer's replay-id-remapping approach) --


def _get_seq_id_by_name(name: str) -> str:
    records = json.loads((project.project_dir() / "records.json").read_text())
    return next(r["seq_id"] for r in records if r["name"] == name)


def test_replay_rewrites_stale_seq_id_in_pairwise_align(project_dir):
    """The exact scenario that used to break replay: `bio align --pairwise
    ID1 ID2` logs the IDs from the original import, but a fresh `bio
    import` during replay assigns brand-new random IDs to the same-named
    records. Replay must rewrite the stale IDs rather than fail."""
    runner = CliRunner()
    _run(runner, ["import", "seqs.fasta"])
    old_id_s1 = _get_seq_id_by_name("s1")
    old_id_s2 = _get_seq_id_by_name("s2")
    _run(runner, ["align", "--pairwise", old_id_s1, old_id_s2])

    report = replay()
    assert report.n_failed == 0, [s.error for s in report.steps if s.status == "failed"]
    assert report.n_executed == 2

    align_step = report.steps[1]
    assert align_step.status == "executed"
    assert align_step.ids_rewritten is True
    # the rewritten argv should reference the *new* IDs, not the stale ones
    new_id_s1 = _get_seq_id_by_name("s1")
    new_id_s2 = _get_seq_id_by_name("s2")
    assert old_id_s1 not in align_step.argv
    assert old_id_s2 not in align_step.argv
    assert new_id_s1 in align_step.argv
    assert new_id_s2 in align_step.argv


def test_replay_no_rewrite_needed_flag_is_false(project_dir):
    runner = CliRunner()
    _run(runner, ["import", "seqs.fasta"])
    _run(runner, ["descriptor"])  # no seq_id arguments at all

    report = replay()
    assert all(not s.ids_rewritten for s in report.steps)


def test_replay_id_rewrite_no_match_leaves_token_and_fails_naturally(project_dir):
    """If the referenced record genuinely isn't there under any name this
    time (e.g. renamed, or excluded by --from skipping its import), the
    stale ID can't be resolved -- it should be left alone and fail exactly
    as a literal replay would, not silently swallowed."""
    runner = CliRunner()
    _run(runner, ["import", "seqs.fasta"])
    old_id_s1 = _get_seq_id_by_name("s1")
    old_id_s2 = _get_seq_id_by_name("s2")
    _run(runner, ["align", "--pairwise", old_id_s1, old_id_s2])

    # Corrupt the logged step to reference IDs that were never actually
    # assigned to anything -- neither is in old_id_to_name, so neither
    # can be rewritten.
    log_path = project_dir / ".bioexplorer" / "log.json"
    log = json.loads(log_path.read_text())
    fake_id_a, fake_id_b = "0123456789ab", "abcdef012345"
    log[1]["argv"] = ["align", "--pairwise", fake_id_a, fake_id_b]
    log_path.write_text(json.dumps(log))

    report = replay()
    assert report.n_failed == 1
    assert report.steps[-1].ids_rewritten is False
    assert fake_id_a in report.steps[-1].argv
    assert fake_id_b in report.steps[-1].argv


def test_replay_id_rewrite_across_multiple_imports(project_dir):
    """Two separate `bio import` calls (e.g. import then --append) each
    mint fresh IDs; a later step referencing IDs from either batch must
    still resolve correctly after replay reruns both imports."""
    (project_dir / "more.fasta").write_text(">s3\nTTTTAAAA\n")
    runner = CliRunner()
    _run(runner, ["import", "seqs.fasta"])
    _run(runner, ["import", "more.fasta", "--append"])
    old_id_s1 = _get_seq_id_by_name("s1")
    old_id_s3 = _get_seq_id_by_name("s3")
    _run(runner, ["align", "--pairwise", old_id_s1, old_id_s3])

    report = replay()
    assert report.n_failed == 0, [s.error for s in report.steps if s.status == "failed"]
    align_step = report.steps[-1]
    assert align_step.ids_rewritten is True
    new_id_s1 = _get_seq_id_by_name("s1")
    new_id_s3 = _get_seq_id_by_name("s3")
    assert new_id_s1 in align_step.argv
    assert new_id_s3 in align_step.argv


def test_replay_dry_run_does_not_rewrite_ids(project_dir):
    """Dry-run is a preview of the logged intent, not a simulation of a
    live rebuild -- it shows the original argv as recorded."""
    runner = CliRunner()
    _run(runner, ["import", "seqs.fasta"])
    old_id_s1 = _get_seq_id_by_name("s1")
    old_id_s2 = _get_seq_id_by_name("s2")
    _run(runner, ["align", "--pairwise", old_id_s1, old_id_s2])

    report = replay(dry_run=True)
    align_step = report.steps[-1]
    assert old_id_s1 in align_step.argv
    assert old_id_s2 in align_step.argv
    assert align_step.ids_rewritten is False


def test_rewrite_ids_helper_unit(project_dir):
    from bioexplorer.replay import _rewrite_ids

    runner = CliRunner()
    _run(runner, ["import", "seqs.fasta"])
    old_id_s1 = _get_seq_id_by_name("s1")
    old_id_to_name = {old_id_s1: "s1"}

    # Simulate what replay does before re-executing: reset, then re-import
    # (which assigns s1 a brand-new random id).
    (project.project_dir() / "records.json").unlink()
    _run(runner, ["import", "seqs.fasta"])
    new_id_s1 = _get_seq_id_by_name("s1")
    assert new_id_s1 != old_id_s1  # sanity: ids really did change

    argv = ["align", "--pairwise", old_id_s1, "unrelated-token"]
    rewritten, changed = _rewrite_ids(argv, old_id_to_name)
    assert changed is True
    assert rewritten[2] == new_id_s1
    assert rewritten[3] == "unrelated-token"  # never touched -- not a known old id


def test_rewrite_ids_helper_empty_mapping_is_noop(project_dir):
    from bioexplorer.replay import _rewrite_ids

    argv = ["align", "--pairwise", "abc123def456", "abc123def456"]
    rewritten, changed = _rewrite_ids(argv, {})
    assert rewritten == argv
    assert changed is False
