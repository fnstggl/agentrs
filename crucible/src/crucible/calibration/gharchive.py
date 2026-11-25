"""GH Archive acquisition and compact extraction for software-adoption calibration (workstream E2).

Offline and evaluator side only: world code never imports this module (it depends only on fitted checkpoints through
the registry). Each hourly dump of the public GitHub events timeline is downloaded, its SHA-256 checked against the
committed manifest (or recorded on first acquisition), reduced to a compact numeric event table, and deleted. No
logins, names, e-mail addresses or text are kept: only numeric actor and repository ids as distributed, the event
type and action, and a few flags derived at extraction time.

Event table columns (one row per event, in archive order within an hour):
    t       int32   seconds since the window start (2024-10-07T00:00:00Z)
    type    uint8   index+1 into TYPES (0 = other)
    action  uint8   index+1 into ACTIONS (0 = none / other)
    flags   uint8   FLAG_ORG | FLAG_BOT | FLAG_OWNER | FLAG_REPO_CREATE
    size    uint16  PushEvent payload.size (commits), capped at 65535; 0 otherwise
    actor   int64   GitHub numeric actor id
    repo    int64   GitHub numeric repository id
"""
from __future__ import annotations

import datetime as _dt
import gzip
import hashlib
import json
import os
import time
import urllib.request

import numpy as np

BASE_URL = "https://data.gharchive.org/"
DATASET = "gharchive_2024-10-07_96h"
WINDOW_START = _dt.datetime(2024, 10, 7, 0, 0, 0, tzinfo=_dt.timezone.utc)
HOURS = 96
EXTRACTOR_VERSION = "1"

TYPES = ("PushEvent", "CreateEvent", "DeleteEvent", "PullRequestEvent", "IssueCommentEvent", "WatchEvent",
         "PullRequestReviewEvent", "PullRequestReviewCommentEvent", "IssuesEvent", "ForkEvent", "ReleaseEvent",
         "MemberEvent", "CommitCommentEvent", "GollumEvent", "PublicEvent", "DiscussionEvent")
TYPE_CODE = {name: i + 1 for i, name in enumerate(TYPES)}
ACTIONS = ("started", "opened", "closed", "reopened", "created", "published", "edited", "deleted", "labeled",
           "added", "synchronize", "assigned", "submitted", "released", "prereleased", "unlabeled")
ACTION_CODE = {name: i + 1 for i, name in enumerate(ACTIONS)}
FLAG_ORG, FLAG_BOT, FLAG_OWNER, FLAG_REPO_CREATE = 1, 2, 4, 8
COLUMNS = (("t", np.int32), ("type", np.uint8), ("action", np.uint8), ("flags", np.uint8), ("size", np.uint16),
           ("actor", np.int64), ("repo", np.int64))


def hour_names(start: _dt.datetime = WINDOW_START, hours: int = HOURS) -> list[str]:
    """GH Archive file stems, e.g. '2024-10-07-0' (hour without leading zero)."""
    out = []
    for h in range(hours):
        d = start + _dt.timedelta(hours=h)
        out.append(f"{d:%Y-%m-%d}-{d.hour}")
    return out


def data_root() -> str:
    return os.environ.get("CRUCIBLE_DATA", "/workspace/data/crucible")


def dataset_dir(dest: str | None = None) -> str:
    return dest or os.path.join(data_root(), DATASET)


def content_digest(arrays: dict[str, np.ndarray]) -> str:
    """SHA-256 over the columns in COLUMNS order (dtype, shape, bytes); independent of the npz container."""
    h = hashlib.sha256()
    for name, dtype in COLUMNS:
        a = np.ascontiguousarray(arrays[name], dtype=dtype)
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def _download(url: str, path: str, retries: int = 4) -> tuple[str, int]:
    last = None
    for attempt in range(retries):
        try:
            h = hashlib.sha256(); n = 0
            req = urllib.request.Request(url, headers={"User-Agent": "crucible-calibration/1"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(path, "wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    h.update(chunk); out.write(chunk); n += len(chunk)
            return h.hexdigest(), n
        except Exception as exc:  # network errors: retry with backoff
            last = exc
            time.sleep(2 + 4 * attempt)
    raise RuntimeError(f"download failed after {retries} attempts: {url}: {last}")


def extract_file(path: str, hour_index: int) -> tuple[dict[str, np.ndarray], dict]:
    """Parse one hourly dump into the compact event table. Returns (arrays, stats)."""
    t0 = int((WINDOW_START + _dt.timedelta(hours=hour_index)).timestamp())
    base = int(WINDOW_START.timestamp())
    cols = {name: [] for name, _ in COLUMNS}
    bad = 0; out_of_hour = 0
    hour_cache: dict[str, int] = {}

    def epoch(s: str) -> int:   # 'YYYY-MM-DDTHH:MM:SSZ'
        key = s[:13]
        b = hour_cache.get(key)
        if b is None:
            b = int(_dt.datetime.strptime(key, "%Y-%m-%dT%H").replace(tzinfo=_dt.timezone.utc).timestamp())
            hour_cache[key] = b
        if len(s) != 20 or s[13] != ":" or s[16] != ":" or s[19] != "Z":
            raise ValueError(s)
        return b + 60 * int(s[14:16]) + int(s[17:19])

    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            try:
                e = json.loads(line)
                actor = e["actor"]; repo = e["repo"]; typ = e["type"]
                ts = epoch(e["created_at"])
                aid = int(actor["id"]); rid = int(repo["id"])
            except (ValueError, KeyError, TypeError):
                bad += 1
                continue
            if not (t0 - 3600 <= ts < t0 + 7200):
                out_of_hour += 1   # kept: GH Archive files can hold slightly late events; time is what we use
            payload = e.get("payload") or {}
            login = str(actor.get("login") or "")
            owner = str(repo.get("name") or "").split("/", 1)[0]
            flags = 0
            if e.get("org"):
                flags |= FLAG_ORG
            if login.endswith("[bot]"):
                flags |= FLAG_BOT
            if login and owner and login.lower() == owner.lower():
                flags |= FLAG_OWNER
            if typ == "CreateEvent" and payload.get("ref_type") == "repository":
                flags |= FLAG_REPO_CREATE
            size = 0
            if typ == "PushEvent":
                try:
                    size = min(int(payload.get("size") or 0), 65535)
                except (TypeError, ValueError):
                    size = 0
            cols["t"].append(ts - base)
            cols["type"].append(TYPE_CODE.get(typ, 0))
            cols["action"].append(ACTION_CODE.get(payload.get("action"), 0) if isinstance(payload, dict) else 0)
            cols["flags"].append(flags)
            cols["size"].append(size)
            cols["actor"].append(aid)
            cols["repo"].append(rid)
    arrays = {name: np.asarray(cols[name], dtype=dtype) for name, dtype in COLUMNS}
    stats = {"events": int(arrays["t"].size), "unparseable_lines": bad, "outside_hour_events": out_of_hour}
    return arrays, stats


def acquire_hour(name: str, hour_index: int, dest: str, expected_sha256: str | None = None) -> dict:
    """Download, verify, extract and delete one hourly dump. Idempotent: an existing extract is reused."""
    ext_dir = os.path.join(dest, "extracted"); raw_dir = os.path.join(dest, "raw")
    os.makedirs(ext_dir, exist_ok=True); os.makedirs(raw_dir, exist_ok=True)
    out = os.path.join(ext_dir, f"{name}.npz")
    meta_path = os.path.join(ext_dir, f"{name}.meta.json")
    if os.path.exists(out) and os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
        if expected_sha256 is None or meta.get("raw_sha256") == expected_sha256:
            return meta
    url = f"{BASE_URL}{name}.json.gz"
    raw = os.path.join(raw_dir, f"{name}.json.gz")
    sha, nbytes = _download(url, raw)
    if expected_sha256 is not None and sha != expected_sha256:
        os.remove(raw)
        raise RuntimeError(f"checksum mismatch for {url}: {sha} != {expected_sha256}")
    arrays, stats = extract_file(raw, hour_index)
    os.remove(raw)
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out)
    meta = {"file": f"{name}.json.gz", "url": url, "raw_sha256": sha, "raw_bytes": nbytes,
            "extract_digest": content_digest(arrays), "extractor_version": EXTRACTOR_VERSION, **stats}
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=1)
    return meta


def _acquire_one(args):
    return acquire_hour(*args)


def acquire(dest: str | None = None, workers: int = 1, expected: dict[str, str] | None = None) -> str:
    """Acquire the whole window. `expected` maps file name -> SHA-256 (from the committed manifest)."""
    dest = dataset_dir(dest)
    os.makedirs(dest, exist_ok=True)
    jobs = []
    for i, name in enumerate(hour_names()):
        exp = (expected or {}).get(f"{name}.json.gz")
        jobs.append((name, i, dest, exp))
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as pool:
            metas = pool.map(_acquire_one, jobs, chunksize=1)
    else:
        metas = [_acquire_one(j) for j in jobs]
    summary = {"dataset": DATASET, "window_start": WINDOW_START.isoformat(), "hours": HOURS,
               "extractor_version": EXTRACTOR_VERSION, "files": metas,
               "events": int(sum(m["events"] for m in metas))}
    with open(os.path.join(dest, "acquisition.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    raw_dir = os.path.join(dest, "raw")
    if os.path.isdir(raw_dir) and not os.listdir(raw_dir):
        os.rmdir(raw_dir)
    return dest


LICENSE = {
    "summary": "Public GitHub events as archived by GH Archive. Use is governed by GitHub's Terms of Service and the "
               "Acceptable Use Policies' information-usage restrictions (research and archival use allowed; personal "
               "information must not be used for spamming, sold, or used to identify people). GH Archive's own code is "
               "MIT licensed.",
    "text_locations": ["https://docs.github.com/en/site-policy/github-terms/github-terms-of-service",
                       "https://docs.github.com/en/site-policy/acceptable-use-policies/github-acceptable-use-policies",
                       "https://github.com/igrigorik/gharchive.org/blob/master/LICENSE"],
}


def manifest(dest: str | None = None) -> dict:
    """The committed dataset manifest, built from a completed acquisition (acquisition.json)."""
    with open(os.path.join(dataset_dir(dest), "acquisition.json")) as fh:
        acq = json.load(fh)
    return {
        "schema": "crucible.v5.calibration.dataset/1", "name": DATASET, "workstream": "E2", "synthetic": False,
        "url": BASE_URL, "homepage": "https://www.gharchive.org/",
        "license": LICENSE,
        "files": [{"file": f["file"], "url": f["url"], "sha256": f["raw_sha256"], "bytes": f["raw_bytes"],
                   "events": f["events"], "extract_digest": f["extract_digest"]} for f in acq["files"]],
        "size": {"raw_bytes_streamed": int(sum(f["raw_bytes"] for f in acq["files"])), "events": acq["events"],
                 "kept_on_disk": "compact event tables only (about 160 MB); raw files deleted after extraction"},
        "schema_extracted": {name: str(np.dtype(dt)) for name, dt in COLUMNS},
        "extractor": {"module": "crucible.calibration.gharchive", "version": EXTRACTOR_VERSION,
                      "types": list(TYPES), "actions": list(ACTIONS),
                      "flags": {"org": FLAG_ORG, "bot": FLAG_BOT, "owner": FLAG_OWNER, "repo_create": FLAG_REPO_CREATE}},
        "provenance": {
            "source": "GH Archive hourly JSON dumps of the GitHub public events API (https://data.gharchive.org/)",
            "window_utc": [WINDOW_START.isoformat(), (WINDOW_START + _dt.timedelta(hours=HOURS)).isoformat()],
            "acquired_by": "crucible.calibration.gharchive.acquire (streamed one hour at a time)",
            "population": "public GitHub accounts and public repositories whose events reached the public events API",
            "privacy": "only numeric actor and repository ids as distributed, event type/action, derived flags and push "
                       "size are kept; no logins, names, e-mails, commit messages or text",
        },
    }


def load_events(dest: str | None = None, hours: int = HOURS) -> dict[str, np.ndarray]:
    """Concatenate the extracted hours in order. Adds `pos` (global archive order)."""
    dest = dataset_dir(dest)
    parts = {name: [] for name, _ in COLUMNS}
    for name in hour_names(hours=hours):
        p = os.path.join(dest, "extracted", f"{name}.npz")
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} missing: run crucible.calibration.registry_e2.acquire('{DATASET}')")
        with np.load(p) as z:
            for col, _ in COLUMNS:
                parts[col].append(z[col])
    ev = {col: np.concatenate(parts[col]) for col, _ in COLUMNS}
    ev["pos"] = np.arange(ev["t"].size, dtype=np.int64)
    return ev


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Acquire the GH Archive calibration window (streamed, raw files deleted)")
    ap.add_argument("--dest", default=None)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    print(acquire(a.dest, workers=a.workers))
