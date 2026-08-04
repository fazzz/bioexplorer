from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bioexplorer import db as db_mod
from bioexplorer.core import BioCollection, BioRecord, SeqType
from bioexplorer.similarity import SimilarityHit, search_blast, search_diamond, search_mmseqs


@pytest.fixture
def collection():
    c = BioCollection()
    c.add(BioRecord(name="local1", sequence="ACGTACGTACGT", seq_type=SeqType.DNA))
    return c


def _fake_completed(stdout: str):
    result = MagicMock()
    result.stdout = stdout
    return result


class TestSimilarityHit:
    def test_hit_id_defaults_to_record_name(self):
        rec = BioRecord(name="foo", sequence="ACGT", seq_type=SeqType.DNA)
        hit = SimilarityHit(rec, 0.9)
        assert hit.hit_id == "foo"

    def test_hit_id_explicit_when_no_record(self):
        hit = SimilarityHit(None, 0.5, hit_id="UniRef50_ABC123")
        assert hit.record is None
        assert hit.hit_id == "UniRef50_ABC123"


class TestSearchBlastDbPath:
    @patch("bioexplorer.similarity.shutil.which", return_value="/usr/bin/blastn")
    @patch("bioexplorer.similarity.subprocess.run")
    def test_db_path_skips_makeblastdb(self, mock_run, mock_which, collection):
        mock_run.return_value = _fake_completed("UniRef50_XYZ\t150.0\t92.50\n")
        hits = search_blast(collection, "ACGT", db_path="/data/blastdb/nr")

        # only one subprocess call (the search itself) -- no makeblastdb call
        assert mock_run.call_count == 1
        called_cmd = mock_run.call_args[0][0]
        assert "-db" in called_cmd
        assert called_cmd[called_cmd.index("-db") + 1] == "/data/blastdb/nr"
        assert "makeblastdb" not in " ".join(called_cmd)

        assert len(hits) == 1
        assert hits[0].hit_id == "UniRef50_XYZ"
        assert hits[0].record is None  # not in the local collection
        assert hits[0].score == pytest.approx(0.925)

    @patch("bioexplorer.similarity.shutil.which", return_value="/usr/bin/blastn")
    @patch("bioexplorer.similarity.subprocess.run")
    def test_no_db_path_builds_ephemeral_db(self, mock_run, mock_which, collection):
        mock_run.return_value = _fake_completed("local1\t150.0\t100.00\n")
        hits = search_blast(collection, "ACGT")

        # two calls: makeblastdb, then the search
        assert mock_run.call_count == 2
        makeblastdb_call = mock_run.call_args_list[0][0][0]
        assert "makeblastdb" in makeblastdb_call[0]

        assert hits[0].record is not None
        assert hits[0].record.name == "local1"

    @patch("bioexplorer.similarity.shutil.which", return_value="/usr/bin/blastn")
    @patch("bioexplorer.similarity.subprocess.run")
    def test_hit_resolved_against_collection_even_with_db_path(self, mock_run, mock_which, collection):
        # a hit id that happens to match something already in the project
        # should still resolve to that record even when searching an
        # external db (e.g. the project was itself built from that db).
        mock_run.return_value = _fake_completed("local1\t150.0\t100.00\n")
        hits = search_blast(collection, "ACGT", db_path="/data/blastdb/nr")
        assert hits[0].record is not None
        assert hits[0].record.name == "local1"


class TestSearchDiamondDbPath:
    @patch("bioexplorer.similarity.shutil.which", return_value="/usr/bin/diamond")
    @patch("bioexplorer.similarity.subprocess.run")
    def test_db_path_skips_makedb(self, mock_run, mock_which, collection):
        mock_run.return_value = _fake_completed("sprot|P01308\t88.10\n")
        hits = search_diamond(collection, "ACGT", db_path="/data/sprot.dmnd")

        assert mock_run.call_count == 1
        called_cmd = mock_run.call_args[0][0]
        assert "-d" in called_cmd
        assert called_cmd[called_cmd.index("-d") + 1] == "/data/sprot.dmnd"
        assert hits[0].hit_id == "sprot|P01308"
        assert hits[0].record is None


class TestSearchMmseqsDbPath:
    @patch("bioexplorer.similarity.shutil.which", return_value="/usr/bin/mmseqs")
    @patch("bioexplorer.similarity.subprocess.run")
    @patch("pathlib.Path.read_text", return_value="UniRef90_ABC\t76.30\n")
    def test_db_path_skips_target_createdb(self, mock_read_text, mock_run, mock_which, collection):
        hits = search_mmseqs(collection, "ACGT", db_path="/data/uniref90_db")

        # only createdb(query) + search + convertalis = 3 calls (no target createdb)
        assert mock_run.call_count == 3
        commands = [call[0][0] for call in mock_run.call_args_list]
        assert not any(cmd[:2] == ["mmseqs", "createdb"] and "/data/uniref90_db" in cmd for cmd in commands)
        search_cmd = commands[1]
        assert "/data/uniref90_db" in search_cmd
        assert hits[0].hit_id == "UniRef90_ABC"
        assert hits[0].record is None


class TestDbFetch:
    def test_fetch_blast_db_missing_binary_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="not found on PATH"):
            db_mod.fetch_blast_db("nr", tmp_path / "blastdb")

    def test_fetch_mmseqs_db_missing_binary_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="not found on PATH"):
            db_mod.fetch_mmseqs_db("UniRef50", tmp_path / "mmseqs_db" / "uniref50")

    def test_fetch_db_unknown_tool_raises(self, tmp_path):
        with pytest.raises(ValueError):
            db_mod.fetch_db("not-a-tool", "nr", tmp_path / "out")

    @patch("bioexplorer.similarity.shutil.which", return_value="/usr/bin/update_blastdb.pl")
    @patch("bioexplorer.db.subprocess.run")
    def test_fetch_blast_db_calls_update_blastdb(self, mock_run, mock_which, tmp_path):
        out_dir = tmp_path / "blastdb"
        result = db_mod.fetch_blast_db("swissprot", out_dir, decompress=True)
        assert result == out_dir / "swissprot"
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/usr/bin/update_blastdb.pl"
        assert "swissprot" in cmd
        assert "--decompress" in cmd

    @patch("bioexplorer.similarity.shutil.which", return_value="/usr/bin/mmseqs")
    @patch("bioexplorer.db.subprocess.run")
    def test_fetch_mmseqs_db_calls_mmseqs_databases(self, mock_run, mock_which, tmp_path):
        out_prefix = tmp_path / "mmseqs_db" / "uniref50"
        result = db_mod.fetch_mmseqs_db("UniRef50", out_prefix)
        assert result == out_prefix
        cmd = mock_run.call_args[0][0]
        assert cmd[:2] == ["/usr/bin/mmseqs", "databases"]
        assert "UniRef50" in cmd
