"""Sequence annotation via external databases (spec section 9's other
half): UniProt, Pfam, InterPro. Unlike annotate.py, everything here either
needs a downloaded database or a network call, and none of it has been
exercised against the real services from this environment (no network
route to rest.uniprot.org / ebi.ac.uk here) -- the HTTP/subprocess
plumbing follows each service's documented API exactly, but treat it as
unverified until you've run it once yourself.

- UniProt: REST lookup (accession -> record) and search, via
  rest.uniprot.org. No local download needed for single lookups.
- Pfam: local HMMER (hmmscan) against a downloaded Pfam-A.hmm (see
  db.fetch_pfam_hmm) -- this is the domain-annotation equivalent of BLAST
  against a downloaded DB elsewhere in this package.
- InterPro: the EBI InterProScan REST job dispatcher (submit -> poll ->
  fetch). EBI requires a contact email for job submission -- that's not
  optional on their end, so it isn't optional here either.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .similarity import _require_tool

_UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"
_INTERPRO_BASE = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"


def _http_get(url: str, timeout: float = 15.0) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from {url}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"could not reach {url}: {e.reason}") from e


def _http_post(url: str, data: bytes, headers: dict[str, str], timeout: float = 15.0) -> bytes:
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {e.reason} -- {body[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"could not reach {url}: {e.reason}") from e


# ============================================================
# UniProt
# ============================================================


def fetch_uniprot(accession: str, timeout: float = 15.0) -> dict:
    """Look up a single UniProtKB entry by accession (e.g. 'P01308')."""
    url = f"{_UNIPROT_BASE}/{accession}.json"
    raw = _http_get(url, timeout=timeout)
    return json.loads(raw)


def search_uniprot(query: str, size: int = 25, reviewed: bool = True, timeout: float = 20.0) -> list[dict]:
    """Search UniProtKB (e.g. query='globin AND organism_id:9606')."""
    full_query = f"({query}) AND reviewed:{'true' if reviewed else 'false'}" if reviewed else query
    from urllib.parse import urlencode

    url = f"{_UNIPROT_BASE}/search?" + urlencode({"query": full_query, "format": "json", "size": str(size)})
    raw = _http_get(url, timeout=timeout)
    return json.loads(raw).get("results", [])


def summarize_uniprot_entry(entry: dict) -> dict:
    """Pull out the handful of fields most annotation workflows want, from
    the (large, deeply nested) raw UniProt JSON record."""
    protein_desc = entry.get("proteinDescription", {})
    rec_name = protein_desc.get("recommendedName", {}).get("fullName", {}).get("value")
    organism = entry.get("organism", {}).get("scientificName")
    genes = [g.get("geneName", {}).get("value") for g in entry.get("genes", []) if g.get("geneName")]
    features = [
        {"type": f.get("type"), "description": f.get("description"), "start": f.get("location", {}).get("start", {}).get("value"), "end": f.get("location", {}).get("end", {}).get("value")}
        for f in entry.get("features", [])
    ]
    xrefs_pfam = [x.get("id") for x in entry.get("uniProtKBCrossReferences", []) if x.get("database") == "Pfam"]
    return {
        "accession": entry.get("primaryAccession"),
        "name": rec_name,
        "organism": organism,
        "genes": genes,
        "sequence_length": entry.get("sequence", {}).get("length"),
        "features": features,
        "pfam_domains": xrefs_pfam,
    }


# ============================================================
# Pfam (local HMMER)
# ============================================================


@dataclass
class DomainHit:
    query_id: str
    domain_name: str
    domain_accession: str
    evalue: float
    score: float
    start: int  # 1-based, as hmmscan reports it
    end: int


def run_hmmscan(fasta_path: Path, hmm_db_path: Path, evalue: float = 1e-3) -> list[DomainHit]:
    """Scan protein sequences against a local HMM database (e.g.
    Pfam-A.hmm from db.fetch_pfam_hmm) with HMMER's hmmscan, parsed from
    --domtblout. Requires the hmm_db_path to already be hmmpress'd
    (fetch_pfam_hmm does this)."""
    binary = _require_tool("hmmscan")
    with tempfile.TemporaryDirectory() as tmp:
        domtbl = Path(tmp) / "domtblout.txt"
        subprocess.run(
            [binary, "--domtblout", str(domtbl), "-E", str(evalue), "--noali", str(hmm_db_path), str(fasta_path)],
            check=True, capture_output=True, text=True,
        )
        text = domtbl.read_text()

    hits = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        # domtblout columns: target_name, target_accession, tlen, query_name, query_accession,
        # qlen, E-value(full), score(full), bias(full), #, of, c-Evalue, i-Evalue, score(domain),
        # bias(domain), hmm_from, hmm_to, ali_from, ali_to, env_from, env_to, acc, description...
        hits.append(
            DomainHit(
                query_id=fields[3],
                domain_name=fields[0],
                domain_accession=fields[1],
                evalue=float(fields[12]),
                score=float(fields[13]),
                start=int(fields[17]),
                end=int(fields[18]),
            )
        )
    return hits


# ============================================================
# InterPro (EBI InterProScan REST job dispatcher)
# ============================================================


def submit_interproscan(sequence: str, email: str, title: str = "bioexplorer") -> str:
    """Submit a protein sequence to the EBI InterProScan5 REST service.
    An email is required by EBI's job dispatcher (not optional on their
    end -- jobs submitted without a valid contact address are routinely
    rejected/rate-limited). Returns a job id for poll_interproscan."""
    if not email or "@" not in email:
        raise ValueError("submit_interproscan requires a valid contact email (EBI job dispatcher policy)")
    fasta = f">{title}\n{sequence}\n"
    from urllib.parse import urlencode

    data = urlencode({"email": email, "title": title, "sequence": fasta}).encode()
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "text/plain"}
    raw = _http_post(f"{_INTERPRO_BASE}/run", data=data, headers=headers)
    return raw.decode().strip()


def poll_interproscan(job_id: str, timeout: float = 300.0, interval: float = 10.0) -> dict:
    """Poll a submitted InterProScan job until it finishes (or `timeout`
    seconds elapse), then fetch and return the JSON result."""
    status_url = f"{_INTERPRO_BASE}/status/{job_id}"
    elapsed = 0.0
    while elapsed < timeout:
        status = _http_get(status_url).decode().strip()
        if status == "FINISHED":
            result_url = f"{_INTERPRO_BASE}/result/{job_id}/json"
            return json.loads(_http_get(result_url))
        if status in ("FAILURE", "NOT_FOUND", "ERROR"):
            raise RuntimeError(f"InterProScan job {job_id} ended with status {status}")
        time.sleep(interval)
        elapsed += interval
    raise TimeoutError(f"InterProScan job {job_id} did not finish within {timeout}s (last status poll pending)")


def run_interproscan(sequence: str, email: str, title: str = "bioexplorer", timeout: float = 300.0, interval: float = 10.0) -> dict:
    """Convenience: submit + poll + fetch in one call."""
    job_id = submit_interproscan(sequence, email=email, title=title)
    return poll_interproscan(job_id, timeout=timeout, interval=interval)
