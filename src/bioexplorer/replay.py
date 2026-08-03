"""Workflow recording & replay (spec section 19).

Every ``bio`` command already appends its argv to ``.bioexplorer/log.json``
(see project.log_command) as it runs. ``replay`` reads that log back and
re-executes the recorded commands in order, in-process, via Click's test
runner -- rebuilding the project deterministically from the same inputs.

Notes/limitations, stated up front rather than discovered the hard way:

- Commands reference files by the path the user originally typed (e.g.
  ``bio import data/seqs.fasta``), so replay must be run from the same
  working directory the original session used.
- Randomized elements (record ``seq_id``s, e.g.) will differ between runs
  by design; everything derived from the data (descriptors, clusters,
  consensus sequences, tree topology) should match.
- ``bio structure view`` launches an interactive GUI viewer -- replaying it
  headlessly doesn't mean much, so it (and ``replay`` itself, to avoid
  accidental recursion) are skipped by default; pass ``--skip ""`` to force
  everything through.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import project

DEFAULT_SKIP = {"replay"}


@dataclass
class ReplayStep:
    index: int
    argv: list[str]
    status: str  # "would_execute" | "executed" | "skipped" | "failed"
    error: str | None = None


@dataclass
class ReplayReport:
    steps: list[ReplayStep] = field(default_factory=list)
    backup_path: Path | None = None

    @property
    def n_executed(self) -> int:
        return sum(1 for s in self.steps if s.status == "executed")

    @property
    def n_skipped(self) -> int:
        return sum(1 for s in self.steps if s.status == "skipped")

    @property
    def n_failed(self) -> int:
        return sum(1 for s in self.steps if s.status == "failed")


def _backup_and_reset(cwd: Path | None = None) -> Path | None:
    """Snapshot the current .bioexplorer/ before wiping the regenerable
    parts (records/alignments/trees), so a failed replay doesn't destroy
    the only copy of the project. The log itself is left in place -- we
    already have its contents in memory for this replay run."""
    pdir = project.project_dir(cwd)
    if not pdir.exists():
        return None
    backup = pdir.parent / f".bioexplorer_prereplay_{int(time.time())}"
    shutil.copytree(pdir, backup)
    for name in (project.STATE_FILE, project.ALIGNMENTS_DIR, project.TREES_DIR):
        target = pdir / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    return backup


def replay(
    cwd: Path | None = None,
    skip_commands: set[str] | None = None,
    from_index: int | None = None,
    to_index: int | None = None,
    dry_run: bool = False,
    continue_on_error: bool = False,
    reset_state: bool = True,
) -> ReplayReport:
    """Replay the recorded workflow log.

    ``from_index``/``to_index`` are 1-based, inclusive, matching the step
    numbers this function reports (so a user can re-run ``--from 5 --to 5``
    after fixing whatever failed at step 5).
    """
    entries = project.read_log(cwd)
    if not entries:
        raise FileNotFoundError(
            "no workflow log found (.bioexplorer/log.json) -- nothing to replay"
        )

    skip = set(skip_commands) if skip_commands is not None else set(DEFAULT_SKIP)
    start = (from_index - 1) if from_index else 0
    end = to_index if to_index else len(entries)
    selected = list(enumerate(entries[start:end], start=start + 1))

    report = ReplayReport()

    if dry_run:
        for index, entry in selected:
            argv = entry["argv"]
            top = argv[0] if argv else ""
            status = "skipped" if top in skip else "would_execute"
            report.steps.append(ReplayStep(index=index, argv=argv, status=status))
        return report

    if reset_state:
        report.backup_path = _backup_and_reset(cwd)

    from click.testing import CliRunner

    from .cli import main as cli_main

    # Sub-invocations would otherwise each append their own log entry,
    # making every replay double the log. Suppress logging for the
    # duration of the replay; the fact that a replay happened is evident
    # from the (unmodified) log itself.
    original_log_command = project.log_command
    project.log_command = lambda cwd=None: None
    runner = CliRunner()
    try:
        for index, entry in selected:
            argv = entry["argv"]
            top = argv[0] if argv else ""
            if top in skip:
                report.steps.append(ReplayStep(index=index, argv=argv, status="skipped"))
                continue
            result = runner.invoke(cli_main, argv, catch_exceptions=True)
            if result.exit_code != 0:
                error_text = (result.output or str(result.exception) or "").strip()
                report.steps.append(
                    ReplayStep(index=index, argv=argv, status="failed", error=error_text)
                )
                if not continue_on_error:
                    break
            else:
                report.steps.append(ReplayStep(index=index, argv=argv, status="executed"))
    finally:
        project.log_command = original_log_command

    return report
