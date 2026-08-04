import json
from unittest.mock import MagicMock, patch

import pytest

from bioexplorer import annotate_external, db as db_mod


# ---- UniProt ----


class TestFetchUniprot:
    @patch("bioexplorer.annotate_external.urllib.request.urlopen")
    def test_fetch_uniprot_returns_parsed_json(self, mock_urlopen):
        payload = json.dumps({"primaryAccession": "P01308", "sequence": {"length": 110}}).encode()
        cm = MagicMock()
        cm.__enter__.return_value.read.return_value = payload
        mock_urlopen.return_value = cm

        result = annotate_external.fetch_uniprot("P01308")
        assert result["primaryAccession"] == "P01308"
        called_url = mock_urlopen.call_args[0][0]
        assert "P01308.json" in called_url

    @patch("bioexplorer.annotate_external.urllib.request.urlopen")
    def test_fetch_uniprot_http_error_raises_clear_message(self, mock_urlopen):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError("url", 404, "Not Found", None, None)
        with pytest.raises(RuntimeError, match="404"):
            annotate_external.fetch_uniprot("NOTAREALACCESSION")


class TestSummarizeUniprotEntry:
    def test_summarize_pulls_expected_fields(self):
        entry = {
            "primaryAccession": "P01308",
            "proteinDescription": {"recommendedName": {"fullName": {"value": "Insulin"}}},
            "organism": {"scientificName": "Homo sapiens"},
            "genes": [{"geneName": {"value": "INS"}}],
            "sequence": {"length": 110},
            "features": [{"type": "Signal", "description": "", "location": {"start": {"value": 1}, "end": {"value": 24}}}],
            "uniProtKBCrossReferences": [{"database": "Pfam", "id": "PF00049"}, {"database": "PDB", "id": "1XYZ"}],
        }
        summary = annotate_external.summarize_uniprot_entry(entry)
        assert summary["accession"] == "P01308"
        assert summary["name"] == "Insulin"
        assert summary["organism"] == "Homo sapiens"
        assert summary["genes"] == ["INS"]
        assert summary["sequence_length"] == 110
        assert summary["pfam_domains"] == ["PF00049"]
        assert len(summary["features"]) == 1


class TestSearchUniprot:
    @patch("bioexplorer.annotate_external.urllib.request.urlopen")
    def test_search_uniprot_returns_results_list(self, mock_urlopen):
        payload = json.dumps({"results": [{"primaryAccession": "P01308"}]}).encode()
        cm = MagicMock()
        cm.__enter__.return_value.read.return_value = payload
        mock_urlopen.return_value = cm

        results = annotate_external.search_uniprot("insulin", size=5)
        assert len(results) == 1
        assert results[0]["primaryAccession"] == "P01308"


# ---- Pfam / hmmscan ----


class TestRunHmmscan:
    def test_missing_hmmscan_binary_raises_clear_error(self, tmp_path):
        fasta = tmp_path / "q.fasta"
        fasta.write_text(">q\nMKTAYIAK\n")
        hmm_db = tmp_path / "Pfam-A.hmm"
        hmm_db.write_text("")
        with pytest.raises(RuntimeError, match="not found on PATH"):
            annotate_external.run_hmmscan(fasta, hmm_db)

    @patch("bioexplorer.similarity.shutil.which", return_value="/usr/bin/hmmscan")
    @patch("bioexplorer.annotate_external.subprocess.run")
    def test_parses_domtblout_format(self, mock_run, mock_which, tmp_path):
        domtbl_content = (
            "#\n"
            "PF00069.wrong_col_test\n"  # a comment-ish stray line should be skipped by the '#' check only
        )
        # simulate hmmscan writing a domtblout file as a side effect of subprocess.run
        def fake_run(cmd, **kwargs):
            domtblout_path = cmd[cmd.index("--domtblout") + 1]
            line = "PF00069 PF00069.20 250 query1 - 100 1.2e-30 105.3 0.1 1 1 1.2e-30 1.2e-30 105.3 0.1 3 120 10 128 8 130 0.98 Protein_kinase_domain\n"
            with open(domtblout_path, "w") as fh:
                fh.write("# comment\n")
                fh.write(line)
            return MagicMock()

        mock_run.side_effect = fake_run
        fasta = tmp_path / "q.fasta"
        fasta.write_text(">query1\nMKTAYIAK\n")
        hmm_db = tmp_path / "Pfam-A.hmm"
        hmm_db.write_text("")

        hits = annotate_external.run_hmmscan(fasta, hmm_db, evalue=1e-3)
        assert len(hits) == 1
        assert hits[0].query_id == "query1"
        assert hits[0].domain_name == "PF00069"
        assert hits[0].domain_accession == "PF00069.20"
        assert hits[0].start == 10
        assert hits[0].end == 128


# ---- InterPro ----


class TestSubmitInterproscan:
    def test_missing_email_raises(self):
        with pytest.raises(ValueError, match="email"):
            annotate_external.submit_interproscan("MKTAYIAK", email="")

    def test_invalid_email_raises(self):
        with pytest.raises(ValueError, match="email"):
            annotate_external.submit_interproscan("MKTAYIAK", email="not-an-email")

    @patch("bioexplorer.annotate_external.urllib.request.urlopen")
    def test_valid_submission_returns_job_id(self, mock_urlopen):
        cm = MagicMock()
        cm.__enter__.return_value.read.return_value = b"iprscan5-R20240101-123456-1234-5678"
        mock_urlopen.return_value = cm

        job_id = annotate_external.submit_interproscan("MKTAYIAK", email="test@example.com")
        assert job_id == "iprscan5-R20240101-123456-1234-5678"


class TestPollInterproscan:
    @patch("bioexplorer.annotate_external.time.sleep", return_value=None)
    @patch("bioexplorer.annotate_external.urllib.request.urlopen")
    def test_poll_returns_result_when_finished(self, mock_urlopen, mock_sleep):
        status_cm = MagicMock()
        status_cm.__enter__.return_value.read.return_value = b"FINISHED"
        result_cm = MagicMock()
        result_cm.__enter__.return_value.read.return_value = json.dumps({"results": [{"matches": []}]}).encode()
        mock_urlopen.side_effect = [status_cm, result_cm]

        result = annotate_external.poll_interproscan("job123", timeout=60, interval=1)
        assert result == {"results": [{"matches": []}]}

    @patch("bioexplorer.annotate_external.time.sleep", return_value=None)
    @patch("bioexplorer.annotate_external.urllib.request.urlopen")
    def test_poll_raises_on_failure_status(self, mock_urlopen, mock_sleep):
        status_cm = MagicMock()
        status_cm.__enter__.return_value.read.return_value = b"FAILURE"
        mock_urlopen.return_value = status_cm

        with pytest.raises(RuntimeError, match="FAILURE"):
            annotate_external.poll_interproscan("job123", timeout=60, interval=1)

    @patch("bioexplorer.annotate_external.time.sleep", return_value=None)
    @patch("bioexplorer.annotate_external.urllib.request.urlopen")
    def test_poll_times_out(self, mock_urlopen, mock_sleep):
        status_cm = MagicMock()
        status_cm.__enter__.return_value.read.return_value = b"RUNNING"
        mock_urlopen.return_value = status_cm

        with pytest.raises(TimeoutError):
            annotate_external.poll_interproscan("job123", timeout=5, interval=2)


# ---- db.py: Pfam-A.hmm fetch ----


class TestFetchPfamHmm:
    def test_missing_hmmpress_binary_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="not found on PATH"):
            db_mod.fetch_pfam_hmm(tmp_path / "Pfam-A.hmm")

    @patch("bioexplorer.db.subprocess.run")
    @patch("bioexplorer.similarity.shutil.which", return_value="/usr/bin/hmmpress")
    @patch("bioexplorer.db.urllib.request.urlretrieve")
    def test_fetch_pfam_hmm_downloads_decompresses_and_presses(self, mock_urlretrieve, mock_which, mock_run, tmp_path):
        import gzip

        output_path = tmp_path / "pfam" / "Pfam-A.hmm"

        def fake_urlretrieve(url, filename):
            with gzip.open(filename, "wb") as fh:
                fh.write(b"FAKE HMM CONTENT")

        mock_urlretrieve.side_effect = fake_urlretrieve

        result = db_mod.fetch_pfam_hmm(output_path)
        assert result == output_path
        assert output_path.exists()
        assert output_path.read_bytes() == b"FAKE HMM CONTENT"
        assert not output_path.with_suffix(".hmm.gz").exists()
        hmmpress_cmd = mock_run.call_args[0][0]
        assert "hmmpress" in hmmpress_cmd[0]
