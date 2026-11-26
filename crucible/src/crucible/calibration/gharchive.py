"""GH Archive acquisition and compact extraction for software-adoption calibration (workstream E2).

Offline and evaluator side only: world code never imports this module (it depends only on fitted checkpoints through
the registry). Each hourly dump of the public GitHub events timeline is downloaded, its SHA-256 checked against the
committed manifest (or recorded on first acquisition), reduced to a compact numeric event table, and deleted.

Privacy (MASTER_SPEC 5.1: no reconstruction of individual real-world identities). No logins, names, e-mail
addresses or text are kept, and since extractor version 2 no GitHub id either: actor and repository ids are replaced
at extraction by keyed pseudonyms, blake2b-64 of the id under a 32-byte secret salt (`pseudonym_key`), created once
per dataset directory with mode 0600 and never committed, logged or copied into any repository file. The pseudonyms
keep only equality (same developer, same repository) inside this window, which is all the features use; they cannot
be inverted or looked up through GitHub's API without the salt. Deleting the salt file makes the pseudonyms
unlinkable even for the operator (a later re-extraction then creates new pseudonyms; verification does not depend
on them, see `content_digest`). The repository hash bucket that the software fit uses for its train/validation split
is computed from the raw id at extraction and kept as its own column (`repo_bucket`), so no fit needs a raw id.
Residual risk, recorded in DATASETS.md: exact event timestamps and types can still be matched against the public GH
Archive by someone who downloads it, so the extracted tables stay local (never committed) and only aggregate model
parameters are published.

Event table columns (one row per event, in archive order within an hour):
    t           int32   seconds since the window start (2024-10-07T00:00:00Z)
    type        uint8   index+1 into TYPES (0 = other)
    action      uint8   index+1 into ACTIONS (0 = none / other)
    flags       uint8   FLAG_ORG | FLAG_BOT | FLAG_OWNER | FLAG_REPO_CREATE
    size        uint16  PushEvent payload.size (commits), capped at 65535; 0 otherwise
    actor       int64   keyed pseudonym of the GitHub actor (63 bits)
    repo        int64   keyed pseudonym of the GitHub repository (63 bits)
    repo_bucket uint8   splitmix64(raw repository id ^ 20261125) % 100 (the software fit's repository split)
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
EXTRACTOR_VERSION = "2"           # 1: raw GitHub ids (superseded); 2: keyed pseudonyms

TYPES = ("PushEvent", "CreateEvent", "DeleteEvent", "PullRequestEvent", "IssueCommentEvent", "WatchEvent",
         "PullRequestReviewEvent", "PullRequestReviewCommentEvent", "IssuesEvent", "ForkEvent", "ReleaseEvent",
         "MemberEvent", "CommitCommentEvent", "GollumEvent", "PublicEvent", "DiscussionEvent")
TYPE_CODE = {name: i + 1 for i, name in enumerate(TYPES)}
ACTIONS = ("started", "opened", "closed", "reopened", "created", "published", "edited", "deleted", "labeled",
           "added", "synchronize", "assigned", "submitted", "released", "prereleased", "unlabeled")
ACTION_CODE = {name: i + 1 for i, name in enumerate(ACTIONS)}
FLAG_ORG, FLAG_BOT, FLAG_OWNER, FLAG_REPO_CREATE = 1, 2, 4, 8
COLUMNS = (("t", np.int32), ("type", np.uint8), ("action", np.uint8), ("flags", np.uint8), ("size", np.uint16),
           ("actor", np.int64), ("repo", np.int64), ("repo_bucket", np.uint8))
IDENTITY_COLUMNS = ("actor", "repo")
SALT_FILE = "pseudonym_key"
REPO_SPLIT_SEED = 20261125            # fit_software's repository split (kept identical across extractor versions)


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


def canonical_ids(a: np.ndarray) -> np.ndarray:
    """Ids replaced by the order of their first appearance (0, 1, 2, ...): the equality structure of a column, the
    same for raw ids and for any injective pseudonym of them."""
    a = np.asarray(a)
    _, first, inv = np.unique(a, return_index=True, return_inverse=True)
    order = np.empty(first.size, dtype=np.int64)
    order[np.argsort(first, kind="stable")] = np.arange(first.size)
    return order[inv.reshape(-1)]


def content_digest(arrays: dict[str, np.ndarray]) -> str:
    """SHA-256 over the columns in COLUMNS order (dtype, shape, bytes), with the identity columns in canonical form
    (`canonical_ids`), so the digest does not depend on the pseudonym salt; independent of the npz container."""
    h = hashlib.sha256()
    for name, dtype in COLUMNS:
        a = np.ascontiguousarray(arrays[name], dtype=dtype)
        if name in IDENTITY_COLUMNS:
            a = np.ascontiguousarray(canonical_ids(a), dtype=np.int64)
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def content_digest_v1(arrays: dict[str, np.ndarray]) -> str:
    """The extractor-1 digest (raw ids, seven columns), kept to check a raw extract before it is pseudonymised."""
    h = hashlib.sha256()
    for name, dtype in COLUMNS[:7]:
        a = np.ascontiguousarray(arrays[name], dtype=dtype)
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def pseudonym_key(dest: str | None = None, create: bool = True) -> bytes:
    """The dataset's secret salt (32 bytes, file mode 0600, outside the repository)."""
    p = os.path.join(dataset_dir(dest), SALT_FILE)
    if not os.path.exists(p):
        if not create:
            raise FileNotFoundError(p)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(os.urandom(32))
    with open(p, "rb") as fh:
        key = fh.read()
    if len(key) != 32:
        raise ValueError(f"{p}: the pseudonym key must be 32 bytes")
    return key


def pseudonyms(ids: np.ndarray, key: bytes) -> np.ndarray:
    """Keyed 63-bit pseudonyms (blake2b-64 under `key`) of integer ids; equal ids give equal pseudonyms."""
    ids = np.asarray(ids, dtype=np.int64)
    u, inv = np.unique(ids, return_inverse=True)
    out = np.fromiter((int.from_bytes(hashlib.blake2b(str(int(x)).encode(), key=key, digest_size=8).digest(),
                                      "little") & ((1 << 63) - 1) for x in u), dtype=np.int64, count=u.size)
    if np.unique(out).size != u.size:
        raise ValueError("pseudonym collision; delete the key file and re-extract")
    return out[inv.reshape(-1)]


def repo_buckets(raw_repo: np.ndarray) -> np.ndarray:
    """fit_software's repository split bucket of raw repository ids (splitmix64(id ^ seed) % 100)."""
    z = np.asarray(raw_repo, np.int64).astype(np.uint64) ^ np.uint64(REPO_SPLIT_SEED)
    z = z + np.uint64(0x9E3779B97F4A7C15)
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return ((z ^ (z >> np.uint64(31))) % np.uint64(100)).astype(np.uint8)


def pseudonymise(raw: dict[str, np.ndarray], key: bytes) -> dict[str, np.ndarray]:
    """An extractor-2 table from a raw-id table: identity columns pseudonymised, repo_bucket added."""
    out = {name: np.asarray(raw[name]) for name, _ in COLUMNS[:5]}
    out["repo_bucket"] = repo_buckets(raw["repo"])
    out["actor"] = pseudonyms(raw["actor"], key)
    out["repo"] = pseudonyms(raw["repo"], key)
    return {name: np.asarray(out[name], dtype=dt) for name, dt in COLUMNS}


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


def extract_file(path: str, hour_index: int, key: bytes) -> tuple[dict[str, np.ndarray], dict]:
    """Parse one hourly dump into the compact event table, ids pseudonymised under `key` (the dataset's salt). Raw
    ids exist only in memory while the hour is parsed. Returns (arrays, stats)."""
    t0 = int((WINDOW_START + _dt.timedelta(hours=hour_index)).timestamp())
    base = int(WINDOW_START.timestamp())
    cols = {name: [] for name, _ in COLUMNS[:7]}
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
    raw = {name: np.asarray(cols[name], dtype=dtype) for name, dtype in COLUMNS[:7]}
    del cols
    arrays = pseudonymise(raw, key)
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
        if meta.get("extractor_version") != EXTRACTOR_VERSION:
            meta = pseudonymise_hour(name, dest)
        if expected_sha256 is None or meta.get("raw_sha256") == expected_sha256:
            return meta
    url = f"{BASE_URL}{name}.json.gz"
    raw = os.path.join(raw_dir, f"{name}.json.gz")
    sha, nbytes = _download(url, raw)
    if expected_sha256 is not None and sha != expected_sha256:
        os.remove(raw)
        raise RuntimeError(f"checksum mismatch for {url}: {sha} != {expected_sha256}")
    arrays, stats = extract_file(raw, hour_index, pseudonym_key(dest))
    os.remove(raw)
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out)
    meta = {"file": f"{name}.json.gz", "url": url, "raw_sha256": sha, "raw_bytes": nbytes,
            "extract_digest": content_digest(arrays), "extractor_version": EXTRACTOR_VERSION, **stats}
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=1)
    return meta


def pseudonymise_hour(name: str, dest: str | None = None) -> dict:
    """Convert an extractor-1 hour (raw ids) in place to extractor 2: the raw table must still match its recorded
    extractor-1 digest; the pseudonymised table is written atomically over the raw-id file."""
    dest = dataset_dir(dest)
    ext_dir = os.path.join(dest, "extracted")
    out = os.path.join(ext_dir, f"{name}.npz")
    meta_path = os.path.join(ext_dir, f"{name}.meta.json")
    with open(meta_path) as fh:
        meta = json.load(fh)
    if meta.get("extractor_version") == EXTRACTOR_VERSION:
        return meta
    with np.load(out) as z:
        raw = {k: z[k] for k in z.files}
    if content_digest_v1(raw) != meta.get("extract_digest"):
        raise RuntimeError(f"{out}: raw extract does not match its recorded digest; re-acquire the hour")
    arrays = pseudonymise(raw, pseudonym_key(dest))
    del raw
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out)
    meta = dict(meta, extract_digest=content_digest(arrays), extract_digest_raw_v1=meta.get("extract_digest"),
                extractor_version=EXTRACTOR_VERSION)
    with open(meta_path + ".tmp", "w") as fh:
        json.dump(meta, fh, indent=1)
    os.replace(meta_path + ".tmp", meta_path)
    return meta


def pseudonymise_window(dest: str | None = None) -> dict:
    """Convert every hour of an extractor-1 acquisition and rewrite acquisition.json (raw ids are deleted)."""
    dest = dataset_dir(dest)
    metas = [pseudonymise_hour(n, dest) for n in hour_names()]
    p = os.path.join(dest, "acquisition.json")
    with open(p) as fh:
        acq = json.load(fh)
    acq.update(files=metas, extractor_version=EXTRACTOR_VERSION)
    with open(p + ".tmp", "w") as fh:
        json.dump(acq, fh, indent=1)
    os.replace(p + ".tmp", p)
    return acq


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
    path = os.path.join(dest, "acquisition.json")
    with open(path + ".tmp", "w") as fh:     # atomic: readers (verify_dataset, tests) never see a partial record
        json.dump(summary, fh, indent=1)
    os.replace(path + ".tmp", path)
    raw_dir = os.path.join(dest, "raw")
    if os.path.isdir(raw_dir) and not os.listdir(raw_dir):
        os.rmdir(raw_dir)
    return dest


LICENSE = {
    "name": "GitHub Terms of Service and Acceptable Use Policies (public events archived by GH Archive; GH Archive "
            "code MIT)",
    "summary": "Public GitHub events as archived by GH Archive. Use is governed by GitHub's Terms of Service and the "
               "Acceptable Use Policies' information-usage restrictions (research and archival use allowed; personal "
               "information must not be used for spamming, sold, or used to identify people). GH Archive's own code is "
               "MIT licensed.",
    "text_locations": ["https://docs.github.com/en/site-policy/github-terms/github-terms-of-service",
                       "https://docs.github.com/en/site-policy/acceptable-use-policies/github-acceptable-use-policies",
                       "https://github.com/igrigorik/gharchive.org/blob/master/LICENSE"],
}


TITLE = "GH Archive: public GitHub events, 2024-10-07 to 2024-10-10 UTC (96 hourly dumps)"
DESCRIPTION = ("Hourly dumps of the GitHub public events API archived by GH Archive, 2024-10-07T00 to 2024-10-11T00 "
               "UTC, streamed one hour at a time and reduced to a numeric event table (time, event type and action, "
               "keyed pseudonyms of actor and repository ids, four derived flags, push size, repository split bucket).")
# Provenance facts that the calibration registry requires of every dataset (datasets.py, audit.provenance_complete).
PROVENANCE = {
    "publisher": "GH Archive (Ilya Grigorik), archiving the GitHub public events API",
    "citation": "GH Archive, https://www.gharchive.org/ (Ilya Grigorik): hourly dumps 2024-10-07-0 to 2024-10-10-23",
    "platform": "GitHub (github.com): public repositories and public accounts",
    "period": "2024-10-07T00:00 to 2024-10-11T00:00 UTC (Monday to Thursday, 96 hours)",
    "identification": "observational",
    "retrieved": "2026-11-25",
}


def manifest(dest: str | None = None) -> dict:
    """The committed dataset manifest, built from a completed acquisition (acquisition.json)."""
    with open(os.path.join(dataset_dir(dest), "acquisition.json")) as fh:
        acq = json.load(fh)
    return {
        "schema": "crucible.v5.calibration.dataset/1", "name": DATASET, "workstream": "E2", "synthetic": False,
        "title": TITLE, "description": DESCRIPTION, "acquirer": __name__,
        "url": BASE_URL, "homepage": "https://www.gharchive.org/",
        "license": LICENSE,
        "files": [{"file": f["file"], "url": f["url"], "sha256": f["raw_sha256"], "bytes": f["raw_bytes"],
                   "events": f["events"], "extract_digest": f["extract_digest"],
                   **({"extract_digest_raw_v1": f["extract_digest_raw_v1"]} if f.get("extract_digest_raw_v1") else {})}
                  for f in acq["files"]],
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
            "privacy": "no GitHub id, login, name, e-mail, commit message or text is kept: actor and repository ids are "
                       "replaced at extraction by keyed 63-bit pseudonyms (blake2b under a 32-byte secret salt kept "
                       "outside the repository, mode 0600); only equality inside the window survives; event "
                       "type/action, derived flags, push size and the repository split bucket are kept. Exact "
                       "timestamps could still be matched against the public archive, so extracted tables are never "
                       "committed",
            **PROVENANCE,
        },
    }


# ---------------------------------------------------------------- registry hooks (crucible.calibration.datasets)
def acquire_dataset(entry: dict, dest: str | None = None) -> str:
    """`datasets.acquire` hook: stream the window, checking each hour against the manifest's SHA-256."""
    expected = {f["file"]: f["sha256"] for f in entry.get("files", [])}
    return acquire(dest, workers=int(os.environ.get("CRUCIBLE_ACQUIRE_WORKERS", "1")), expected=expected)


def _hour_files(entry: dict, dest: str | None):
    d = dataset_dir(dest)
    p = os.path.join(d, "acquisition.json")
    acq = {}
    if os.path.exists(p):
        with open(p) as fh:
            acq = {f["file"]: f for f in json.load(fh)["files"]}
    for f in entry.get("files", []):
        yield f, acq.get(f["file"]), os.path.join(d, "extracted", f"{f['file'][:-len('.json.gz')]}.npz")


def verify_dataset(entry: dict, dest: str | None = None) -> dict[str, str]:
    """`datasets.verify` hook, {hour file: ok|missing|mismatch}. The raw hourly files are deleted after extraction,
    so an hour is ok when the SHA-256 of its raw file recorded while it was streamed (acquisition.json) and the
    content digest of its extracted table both equal the manifest's."""
    out = {}
    for f, rec, npz in _hour_files(entry, dest):
        if rec is None or not os.path.exists(npz):
            out[f["file"]] = "missing"
            continue
        with np.load(npz) as z:
            dig = content_digest({k: z[k] for k in z.files})
        out[f["file"]] = "ok" if rec.get("raw_sha256") == f["sha256"] and dig == f.get("extract_digest") else "mismatch"
    return out


def validate_dataset(entry: dict, dest: str | None = None) -> list[str]:
    """`datasets.validate_schema` hook: every extracted hour has the manifest's columns and dtypes and the number of
    events recorded for it."""
    want = entry.get("schema_extracted") or {name: str(np.dtype(dt)) for name, dt in COLUMNS}
    errs = []
    for f, _, npz in _hour_files(entry, dest):
        if not os.path.exists(npz):
            errs.append(f"{f['file']}: extracted table missing")
            continue
        with np.load(npz) as z:
            got = {k: str(z[k].dtype) for k in z.files}
            n = {int(z[k].shape[0]) for k in z.files}
        if any(got.get(k) != v for k, v in want.items()):
            errs.append(f"{f['file']}: columns {got} differ from {want}")
        if n != {int(f.get("events", -1))}:
            errs.append(f"{f['file']}: {sorted(n)} rows, manifest records {f.get('events')} events")
    return errs


def load_events(dest: str | None = None, hours: int = HOURS) -> dict[str, np.ndarray]:
    """Concatenate the extracted hours in order. Adds `pos` (global archive order)."""
    dest = dataset_dir(dest)
    parts = {name: [] for name, _ in COLUMNS}
    for name in hour_names(hours=hours):
        p = os.path.join(dest, "extracted", f"{name}.npz")
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} missing: run crucible.calibration.acquire('{DATASET}')")
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
