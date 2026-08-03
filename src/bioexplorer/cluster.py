"""Clustering (spec section 12).

Methods:

- ``greedy``: a pure-Python re-implementation of CD-HIT's greedy
  incremental algorithm (sort by length, walk records, join the first
  existing cluster whose seed is similar enough, else start a new cluster),
  using the k-mer/MinHash similarity from similarity.py as the identity
  proxy. No external tool required.
- ``cdhit``: the real CD-HIT / CD-HIT-EST binary (exact sequence identity
  via its own greedy+alignment algorithm), via subprocess.
- ``mmseqs``: MMseqs2's ``easy-cluster``, via subprocess.

Representative selection (spec: Representative / Consensus / Centroid) is a
separate, method-agnostic step applied to whatever members ended up in a
cluster:

- Representative: the cluster seed (longest member -- the CD-HIT
  convention, and what ``greedy``/``cdhit`` naturally produce anyway).
- Centroid: the member with the highest average similarity to every other
  member (medoid) -- can differ from the seed once a cluster has grown.
- Consensus: majority-vote sequence. Exact when all members share a length
  (no indels to resolve); otherwise falls back to an external MSA tool if
  one is available, and to the representative sequence (flagged as
  approximate) if not.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .core import BioRecord, SeqType
from .io import write_fasta
from .similarity import jaccard_similarity, minhash_signature, minhash_similarity
from .similarity import _require_tool


@dataclass
class Cluster:
    cluster_id: int
    members: list[BioRecord]
    representative: BioRecord
    centroid: BioRecord | None = None
    consensus: str | None = None
    consensus_is_approximate: bool = field(default=False)


# -- in-process greedy clustering ------------------------------------------


def _fast_similarity(seq_a: str, seq_b: str, method: str, k: int) -> float:
    if method == "kmer":
        return jaccard_similarity(seq_a, seq_b, k=k)
    if method == "minhash":
        sig_a = minhash_signature(seq_a, k=k, num_hashes=64)
        sig_b = minhash_signature(seq_b, k=k, num_hashes=64)
        return minhash_similarity(sig_a, sig_b)
    raise ValueError(f"unknown fast similarity method: {method}")


def cluster_greedy(
    records: list[BioRecord],
    method: str = "kmer",
    k: int | None = None,
    threshold: float = 0.8,
) -> list[Cluster]:
    """CD-HIT-style greedy incremental clustering: process longest-first,
    join the first cluster whose representative is similar enough, else
    start a new cluster with this record as its representative."""
    if not records:
        return []
    kk = k or (4 if method == "kmer" else 9)
    ordered = sorted(records, key=lambda r: -r.length)

    clusters: list[Cluster] = []
    for rec in ordered:
        joined = False
        for cluster in clusters:
            score = _fast_similarity(rec.sequence, cluster.representative.sequence, method, kk)
            if score >= threshold:
                cluster.members.append(rec)
                joined = True
                break
        if not joined:
            clusters.append(Cluster(cluster_id=len(clusters), members=[rec], representative=rec))
    return clusters


# -- external tools ---------------------------------------------------------


def cluster_cdhit(
    records: list[BioRecord],
    seq_type: SeqType,
    identity: float = 0.9,
    word_size: int | None = None,
    extra_args: list[str] | None = None,
) -> list[Cluster]:
    """Run CD-HIT (nucleotide: cd-hit-est, protein: cd-hit) and parse the
    resulting .clstr file into Cluster objects."""
    binary = "cd-hit" if seq_type == SeqType.PROTEIN else "cd-hit-est"
    _require_tool(binary)

    if word_size is None:
        # CD-HIT's own recommended word size given the identity threshold
        word_size = 5 if identity >= 0.7 else (4 if identity >= 0.6 else 3)

    by_name = {rec.name: rec for rec in records}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_fasta = tmp_path / "input.fasta"
        output_prefix = tmp_path / "clustered"
        from .core import BioCollection

        write_fasta(BioCollection(records), input_fasta)

        cmd = [
            binary, "-i", str(input_fasta), "-o", str(output_prefix),
            "-c", str(identity), "-n", str(word_size), "-d", "0",
            *(extra_args or []),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        clstr_text = (Path(str(output_prefix) + ".clstr")).read_text()

    return _parse_cdhit_clstr(clstr_text, by_name)


def _parse_cdhit_clstr(clstr_text: str, by_name: dict[str, BioRecord]) -> list[Cluster]:
    clusters: list[Cluster] = []
    current_members: list[BioRecord] = []
    current_rep: BioRecord | None = None

    def flush():
        if current_members:
            rep = current_rep or current_members[0]
            clusters.append(Cluster(cluster_id=len(clusters), members=list(current_members), representative=rep))

    for line in clstr_text.splitlines():
        if line.startswith(">Cluster"):
            flush()
            current_members = []
            current_rep = None
            continue
        # e.g. "0    330aa, >prot1... *" or "1    250aa, >prot2... at 95.20%"
        name_part = line.split(">", 1)[1].split("...")[0] if ">" in line else None
        if name_part and name_part in by_name:
            rec = by_name[name_part]
            current_members.append(rec)
            if line.rstrip().endswith("*"):
                current_rep = rec
    flush()
    return clusters


def cluster_mmseqs(
    records: list[BioRecord],
    min_seq_id: float = 0.9,
    coverage: float = 0.8,
    extra_args: list[str] | None = None,
) -> list[Cluster]:
    """Run `mmseqs easy-cluster` and parse the *_cluster.tsv output
    (representative<TAB>member per line) into Cluster objects."""
    _require_tool("mmseqs")
    by_name = {rec.name: rec for rec in records}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_fasta = tmp_path / "input.fasta"
        out_prefix = tmp_path / "result"
        from .core import BioCollection

        write_fasta(BioCollection(records), input_fasta)

        cmd = [
            "mmseqs", "easy-cluster", str(input_fasta), str(out_prefix), str(tmp_path / "tmp"),
            "--min-seq-id", str(min_seq_id), "-c", str(coverage),
            *(extra_args or []),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        tsv_text = Path(str(out_prefix) + "_cluster.tsv").read_text()

    groups: dict[str, list[str]] = {}
    for line in tsv_text.strip().splitlines():
        rep_name, member_name = line.split("\t")
        groups.setdefault(rep_name, []).append(member_name)

    clusters = []
    for i, (rep_name, member_names) in enumerate(groups.items()):
        members = [by_name[n] for n in member_names if n in by_name]
        representative = by_name.get(rep_name, members[0] if members else None)
        if members:
            clusters.append(Cluster(cluster_id=i, members=members, representative=representative))
    return clusters


# -- representative selection: centroid + consensus -------------------------


def assign_centroid(cluster: Cluster, method: str = "kmer", k: int | None = None) -> BioRecord:
    """Medoid: the member with the highest average similarity to every
    other member. For singleton/pair clusters this trivially matches the
    representative."""
    members = cluster.members
    if len(members) == 1:
        cluster.centroid = members[0]
        return cluster.centroid
    kk = k or (4 if method == "kmer" else 9)
    best_rec, best_avg = members[0], -1.0
    for candidate in members:
        total = 0.0
        for other in members:
            if other is candidate:
                continue
            total += _fast_similarity(candidate.sequence, other.sequence, method, kk)
        avg = total / (len(members) - 1)
        if avg > best_avg:
            best_rec, best_avg = candidate, avg
    cluster.centroid = best_rec
    return best_rec


def compute_consensus(cluster: Cluster, msa_tool: str | None = "mafft") -> str:
    """Majority-vote consensus. Exact (equal-length, ungapped vote) when
    every member shares a length; otherwise aligns with an external MSA
    tool if available, else falls back to the representative sequence and
    flags the result as approximate."""
    members = cluster.members
    lengths = {m.length for m in members}

    if len(lengths) == 1:
        length = lengths.pop()
        consensus_chars = []
        for pos in range(length):
            counts: dict[str, int] = {}
            for m in members:
                c = m.sequence[pos]
                counts[c] = counts.get(c, 0) + 1
            consensus_chars.append(max(counts.items(), key=lambda kv: kv[1])[0])
        cluster.consensus = "".join(consensus_chars)
        cluster.consensus_is_approximate = False
        return cluster.consensus

    if msa_tool is not None:
        try:
            from .align import multiple_align
            from .profile import build_profile

            aligned = multiple_align(members, tool=msa_tool)
            profile = build_profile(aligned)
            cluster.consensus = profile.consensus.replace("-", "")
            cluster.consensus_is_approximate = False
            return cluster.consensus
        except (RuntimeError, ValueError):
            pass  # fall through to the approximate fallback below

    cluster.consensus = cluster.representative.sequence
    cluster.consensus_is_approximate = True
    return cluster.consensus


def annotate_clusters(
    clusters: list[Cluster],
    centroid_method: str = "kmer",
    compute_consensus_seqs: bool = True,
    msa_tool: str | None = "mafft",
) -> None:
    """Fill in centroid + consensus for every cluster, and tag member
    records with cluster_id/representative/centroid metadata so results
    round-trip through `bio export`."""
    for cluster in clusters:
        assign_centroid(cluster, method=centroid_method)
        if compute_consensus_seqs:
            compute_consensus(cluster, msa_tool=msa_tool)
        for member in cluster.members:
            member.set("cluster_id", cluster.cluster_id)
            member.add_tag(f"cluster_{cluster.cluster_id}")
            if member is cluster.representative:
                member.add_tag("cluster_representative")
            if cluster.centroid is not None and member is cluster.centroid:
                member.add_tag("cluster_centroid")
