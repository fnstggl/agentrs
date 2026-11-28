#!/usr/bin/env python3
"""Secret scan of every blob introduced by the commits BASE..SHA (used by tools/push_milestone.sh).

G2-integration G2I-1 / G2I-2: scanning only the net diff misses a credential added in one unpushed commit and removed
in a later one (GitHub receives every commit), and a text diff never shows binary or compressed files. This scans
each blob that any commit of the range adds or modifies, as raw bytes, after decompressing gzip / xz / bzip2 content
(nested up to three levels; tar archives are scanned as their bytes), for

1. the real credentials held on this machine (exact match; their values are read here and never printed), and
2. credential-shaped patterns, tolerated only in the paths listed in tools/secret_scan_allow.txt.

Output names paths and commit ids only. Exit status: 0 clean, 4 a real credential, 3 a pattern hit, 2 usage / git
error.

    python3 cam_live/tools/secret_scan.py BASE SHA
"""
from __future__ import annotations

import bz2
import gzip
import io
import json
import lzma
import os
import re
import subprocess
import sys
import zipfile
from typing import Dict, Iterator, List, Optional, Set, Tuple

PAT = re.compile(rb"ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|gho_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|"
                 rb"AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY|xox[baprs]-[A-Za-z0-9-]{10,}|"
                 rb"[MN][A-Za-z0-9_-]{23,25}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27,}|hf_[A-Za-z0-9]{30,}")
SECRET_FILES = ("/root/.config/camlive/discord_bot_token", "/root/.config/camlive/gh_token",
                "/root/.config/agentrs/gh_token")
ACCOUNT_FILE = "/root/.config/camlive/discord_account.json"
MAX_EXPANDED = 512 * 1024 * 1024
HERE = os.path.dirname(os.path.abspath(__file__))


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def real_secrets() -> List[bytes]:
    out: List[bytes] = []
    for f in SECRET_FILES:
        try:
            with open(f, "rb") as fh:
                v = fh.read().strip()
        except OSError:
            continue
        if len(v) >= 8:
            out.append(v)
    try:
        with open(ACCOUNT_FILE) as fh:
            pw = str(json.load(fh).get("password") or "").encode()
        if len(pw) >= 8:
            out.append(pw)
    except (OSError, ValueError):
        pass
    return out


def introduced_blobs(base: str, sha: str) -> Iterator[Tuple[str, str, str]]:
    """(commit, blob, path) for every blob a commit of base..sha adds or modifies (merges included per parent)."""
    raw = git("log", "--format=commit %H", "--raw", "--no-abbrev", "--no-renames", "-m", f"{base}..{sha}")
    commit = ""
    for line in raw.decode("utf-8", "surrogateescape").splitlines():
        if line.startswith("commit "):
            commit = line.split()[1]
        elif line.startswith(":"):
            meta, path = line.split("\t", 1)
            parts = meta.split()
            status, new = parts[4], parts[3]
            if status[0] in "AMT" and not set(new) <= {"0"} and parts[1] != "160000":
                yield commit, new, path


def expansions(data: bytes, depth: int = 0) -> Iterator[bytes]:
    yield data
    if depth >= 3:
        return
    if data.startswith(b"PK\x03\x04"):                    # zip (G2I-16): every member, expanded in turn
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                total = 0
                for info in z.infolist():
                    total += info.file_size
                    if total > MAX_EXPANDED:
                        break
                    yield from expansions(z.read(info), depth + 1)
        except Exception:  # noqa: BLE001 - a corrupt archive is scanned as its bytes only
            pass
        return
    for magic, fn in ((b"\x1f\x8b", gzip.decompress), (b"\xfd7zXZ\x00", lzma.decompress), (b"BZh", bz2.decompress)):
        if data.startswith(magic):
            try:
                inner = fn(data)
            except Exception:  # noqa: BLE001 - a corrupt archive is scanned as its bytes only
                return
            if len(inner) <= MAX_EXPANDED:
                yield from expansions(inner, depth + 1)
            return


def allowed_paths() -> Set[str]:
    p = os.path.join(HERE, "secret_scan_allow.txt")
    try:
        with open(p) as fh:
            return {ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")}
    except OSError:
        return set()


def main(argv: Optional[List[str]] = None) -> int:
    a = sys.argv[1:] if argv is None else argv
    if len(a) != 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    base, sha = a
    try:
        blobs = list(introduced_blobs(base, sha))
    except subprocess.CalledProcessError as e:
        print(f"secret scan: git failed ({e.returncode})", file=sys.stderr)
        return 2
    secrets, allow = real_secrets(), allowed_paths()
    real_hits: Set[Tuple[str, str]] = set()
    pat_hits: Set[Tuple[str, str]] = set()
    seen: Dict[str, Tuple[bool, bool]] = {}           # blob -> (real credential, credential pattern)
    for commit, blob, path in blobs:
        if blob not in seen:                            # each blob is scanned once ...
            real = pat = False
            for chunk in expansions(git("cat-file", "blob", blob)):
                real = real or any(s in chunk for s in secrets)
                pat = pat or bool(PAT.search(chunk))
            seen[blob] = (real, pat)
        real, pat = seen[blob]                          # ... but judged at every path it occurs at (G2I-17)
        if real:
            real_hits.add((commit[:12], path))
        if pat and path not in allow:
            pat_hits.add((commit[:12], path))
    print(f"secret scan: {len(seen)} blobs of {base[:12]}..{sha[:12]} scanned (every commit, decompressed)")
    if real_hits:
        print("secret scan: a REAL credential of this machine occurs in: "
              + ", ".join(f"{p} @{c}" for c, p in sorted(real_hits)) + " — not pushing")
        return 4
    if pat_hits:
        print("secret scan: credential-like text in: " + ", ".join(f"{p} @{c}" for c, p in sorted(pat_hits))
              + " — not pushing")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
