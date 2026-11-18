"""Complete, resumable neural checkpoints on GitHub Releases (``python -m agentrs.gh_backup``).

GitHub is a mandatory checkpoint destination for the overnight run. Each upload is one GitHub
Release in a (private) repository, holding one *bundle*:

  payload/ckpt/<name>/      the checkpoint itself: policy LoRA (Qwen3-4B), world-model and critic
                            LoRA weights, optimizer + LR-scheduler state (``trainer.pt``), RNG state
                            (``rng.pt``), model ids + revisions + git commit (``meta.json``) and the
                            run state at checkpoint time (``state.json``: curriculum, counters,
                            task indices, policy version, evals, promotions)
  payload/run/              run-level state: state.json, registry.json, logs, status, configs
  payload/run/experience/   replay: every stored trajectory (agent, demonstrations, evaluations)
  payload/run/live/         HN/Bluesky observer: delayed-outcome ledger, forecasts, cursors,
                            prequential scores, observer model checkpoint (SQLite copied with the
                            online-backup API, so each DB is a consistent snapshot)
  payload/run/github/       GitHub-archive observer model + state
  BUNDLE_INFO.json          per-file sha256 + sizes, versions, step, timestamp, git commit

Public repositories: release assets of a public repo are public, so third-party content collected by
the HN/Bluesky observer (post text, author pseudonyms, post/item references, the pseudonymisation key)
is REDACTED from the copied live databases before bundling. The observer's learned weights, optimizer
state, forecasts, numeric outcomes and prequential scores are kept. Everything else (all AGENTRS
weights, optimizer state, synthetic task experience) is complete. Use a private repository the token
can access (``AGENTRS_GH_CKPT_REPO``) for unredacted backups.

The bundle is streamed into an uncompressed tar that is split on the fly into parts of at most
``--part-mib`` MiB (GitHub's per-asset limit is 2 GiB). ``MANIFEST.json`` lists every part with its
size and sha256, the tar's sha256, model versions, training step, timestamp, the release URL and
reconstruction instructions. A release is created as a *draft*, all assets are uploaded, every asset
is verified (GitHub's server-side sha256 ``digest`` where available, otherwise by downloading it
again), and only then is the release published. ``restore`` only ever uses published releases.

    python -m agentrs.gh_backup upload  --run runs/overnight --ckpt latest
    python -m agentrs.gh_backup watch   --run runs/overnight          # the scheduled service
    python -m agentrs.gh_backup list
    python -m agentrs.gh_backup restore --dest /tmp/r [--tag ckpt-...] [--install-run runs/overnight]

Credentials: a GitHub token with Contents read/write on the repository, read from the file named
by ``AGENTRS_GITHUB_TOKEN_FILE`` (default ``~/.config/agentrs/github_token``) or ``GITHUB_TOKEN``.
The token is never written into the bundle, the repository, logs or process arguments.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

DEFAULT_REPO = os.environ.get("AGENTRS_GH_CKPT_REPO", "fnstggl/agentrs")
TOKEN_FILE = os.environ.get("AGENTRS_GITHUB_TOKEN_FILE", str(Path.home() / ".config" / "agentrs" / "github_token"))
API = "https://api.github.com"
UPLOADS = "https://uploads.github.com"
TAG_PREFIX = "ckpt-"
BUNDLE_SCHEMA = "agentrs-gh-ckpt-v1"


def log(msg: str) -> None:
    print(f"[gh_backup {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def token() -> str:
    t = os.environ.get("GITHUB_TOKEN", "").strip()
    if not t and Path(TOKEN_FILE).exists():
        t = Path(TOKEN_FILE).read_text().strip()
    if not t:
        raise RuntimeError(f"no GitHub token: put one in {TOKEN_FILE} (chmod 600) or set GITHUB_TOKEN")
    return t


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------- GitHub API
class GitHub:
    def __init__(self, repo: str = DEFAULT_REPO):
        self.repo = repo
        self._tok = token()
        hdr = tempfile.NamedTemporaryFile("w", delete=False, prefix="ghhdr_", suffix=".txt")
        os.chmod(hdr.name, 0o600)
        hdr.write(f"Authorization: Bearer {self._tok}\n")
        hdr.close()
        self._hdr_file = hdr.name  # curl reads the header from a 0600 file: the token never appears in argv

    def __del__(self):
        try:
            os.unlink(self._hdr_file)
        except (OSError, AttributeError):
            pass

    def api(self, method: str, path: str, body: Optional[dict] = None, retries: int = 4):
        url = path if path.startswith("http") else f"{API}/repos/{self.repo}{path}"
        data = json.dumps(body).encode() if body is not None else None
        for attempt in range(retries):
            req = urllib.request.Request(url, data=data, method=method, headers={
                "Authorization": f"Bearer {self._tok}", "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "agentrs-backup"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    raw = r.read()
                    return json.loads(raw) if raw else None
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")[:500]
                if e.code in (500, 502, 503, 504) or (e.code == 403 and "rate" in msg.lower()):
                    time.sleep(5 * (attempt + 1))
                    continue
                raise RuntimeError(f"GitHub API {method} {path}: HTTP {e.code}: {msg}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == retries - 1:
                    raise RuntimeError(f"GitHub API {method} {path}: {e}") from None
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"GitHub API {method} {path}: retries exhausted")

    def _curl(self, args: list[str], timeout: int = 7200) -> subprocess.CompletedProcess:
        cmd = ["curl", "-sS", "--fail-with-body", "--retry", "5", "--retry-all-errors", "--retry-delay", "5",
               "-H", f"@{self._hdr_file}", "-H", "User-Agent: agentrs-backup", *args]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def releases(self) -> list[dict]:
        out, page = [], 1
        while True:
            r = self.api("GET", f"/releases?per_page=100&page={page}")
            out += r
            if len(r) < 100:
                return out
            page += 1

    def create_release(self, tag: str, name: str, body: str) -> dict:
        repo = self.api("GET", "")
        return self.api("POST", "/releases", {"tag_name": tag, "name": name, "body": body, "draft": True,
                                               "target_commitish": repo["default_branch"]})

    def upload_asset(self, rel: dict, path: Path, name: str) -> dict:
        url = f"{UPLOADS}/repos/{self.repo}/releases/{rel['id']}/assets?name={urllib.parse.quote(name)}"
        for attempt in range(3):
            # -T streams the file (``--data-binary @file`` would load a 1.9 GiB part into memory)
            r = self._curl(["-X", "POST", "-H", "Content-Type: application/octet-stream", "-T", str(path), url])
            if r.returncode == 0:
                return json.loads(r.stdout)
            # a failed upload can leave a broken asset with the same name behind: delete and retry
            for a in self.api("GET", f"/releases/{rel['id']}/assets?per_page=100"):
                if a["name"] == name:
                    self.api("DELETE", f"/releases/assets/{a['id']}")
            log(f"upload of {name} failed (attempt {attempt + 1}): {r.stderr.strip()[-300:]} {r.stdout[-300:]}")
            time.sleep(10 * (attempt + 1))
        raise RuntimeError(f"upload of {name} failed")

    def download_asset(self, asset: dict, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = self._curl(["-L", "-H", "Accept: application/octet-stream", "-o", str(dest),
                        f"{API}/repos/{self.repo}/releases/assets/{asset['id']}"])
        if r.returncode != 0:
            raise RuntimeError(f"download of {asset['name']} failed: {r.stderr.strip()[-300:]}")


# ---------------------------------------------------------------------------- bundling
class SplitWriter(io.RawIOBase):
    """File-like sink that splits a stream into numbered parts, hashing each part and the whole."""

    def __init__(self, out_dir: Path, stem: str, part_bytes: int):
        self.out_dir, self.stem, self.part_bytes = out_dir, stem, part_bytes
        self.parts: list[dict] = []
        self.total = hashlib.sha256()
        self.size = 0
        self._f = None
        self._h = None
        self._n = 0

    def writable(self) -> bool:
        return True

    def _open(self):
        name = f"{self.stem}.tar.part{len(self.parts):03d}"
        self._f = open(self.out_dir / name, "wb")
        self._h = hashlib.sha256()
        self._n = 0
        self.parts.append({"name": name})

    def _close_part(self):
        if self._f is not None:
            self._f.close()
            self.parts[-1].update({"size": self._n, "sha256": self._h.hexdigest()})
            self._f = None

    def write(self, b) -> int:
        b = memoryview(b)
        n = len(b)
        self.total.update(b)
        self.size += n
        while len(b):
            if self._f is None or self._n >= self.part_bytes:
                self._close_part()
                self._open()
            k = min(len(b), self.part_bytes - self._n)
            self._f.write(b[:k])
            self._h.update(b[:k])
            self._n += k
            b = b[k:]
        return n

    def close(self):
        self._close_part()
        super().close()


def _link_or_copy(src: Path, dst: Path, copy: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not copy:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _snapshot_ckpt(src: Path, dst: Path, retries: int = 20) -> None:
    """Checkpoint dirs are written to <name>.tmp and swapped in atomically; files are never modified
    in place. Hard-link a completed one and make sure it was not swapped while we linked it."""
    for _ in range(retries):
        if not (src / "DONE").exists():
            time.sleep(3)
            continue
        try:
            tok = (src.stat().st_ino, (src / "DONE").read_text())
            if dst.exists():
                shutil.rmtree(dst)
            for f in src.rglob("*"):
                if f.is_file():
                    _link_or_copy(f, dst / f.relative_to(src))
            if (src.stat().st_ino, (src / "DONE").read_text()) == tok:
                return
        except FileNotFoundError:
            pass
        time.sleep(3)
    raise RuntimeError(f"checkpoint {src} kept changing or is incomplete")


def _copy_sqlite(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=60)
    d = sqlite3.connect(dst)
    try:
        s.backup(d)
    finally:
        d.close()
        s.close()


def _copy_tree(src: Path, dst: Path, skip_dirs: tuple = ()) -> None:
    """Copy a directory written to concurrently by another process: SQLite via the backup API,
    other files copied (not linked — they may be appended to)."""
    if not src.exists():
        return
    for f in sorted(src.rglob("*")):
        rel = f.relative_to(src)
        if any(p in skip_dirs or p.endswith((".tmp", ".old")) for p in rel.parts):
            continue
        if not f.is_file() or f.name.endswith(("-wal", "-shm", "-journal", ".lock", ".tmp")):
            continue
        try:
            if f.suffix in (".sqlite", ".db", ".sqlite3"):
                _copy_sqlite(f, dst / rel)
            else:
                _link_or_copy(f, dst / rel, copy=True)
        except FileNotFoundError:
            continue  # rotated away while copying


# third-party content columns removed from public bundles (table -> columns set to NULL)
REDACT = {
    "ledger.sqlite": {"actions": ["goal", "context_text", "context_json", "action_str", "action_json", "semantic_json",
                                  "actor_ref", "provenance_json"],
                      "events": ["actor_ref", "payload_json", "provider_event_id"]},
}
PUBLIC_EXCLUDE = ("pseudonym.key", "dedup.sqlite")  # re-identification key; raw item references


def redact_public(live_dst: Path) -> list[str]:
    done = []
    for f in PUBLIC_EXCLUDE:
        for p in live_dst.rglob(f):
            p.unlink()
            done.append(f"removed {p.name}")
    for db, tables in REDACT.items():
        p = live_dst / db
        if not p.exists():
            continue
        c = sqlite3.connect(p)
        try:
            for t, cols in tables.items():
                have = {r[1]: bool(r[3]) for r in c.execute(f"PRAGMA table_info({t})")}  # name -> NOT NULL
                sets = ", ".join(f"{col}={chr(39) * 2 if have[col] else 'NULL'}" for col in cols if col in have)
                if sets:
                    c.execute(f"UPDATE {t} SET {sets}")
                    done.append(f"{db}:{t}:{','.join(x for x in cols if x in have)}")
            c.commit()
            c.execute("VACUUM")  # drop the old pages that still held the text
        finally:
            c.close()
    return done


def git_commit(repo_root: Path) -> dict:
    def g(*a):
        return subprocess.run(["git", "-C", str(repo_root), *a], capture_output=True, text=True).stdout.strip()
    return {"commit": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(g("status", "--porcelain")), "remote": "https://github.com/fnstggl/agentrs"}


def build_bundle(run: Path, ckpt: str, work: Path, part_mib: int, kind: str, public: bool = True) -> dict:
    """Stage everything, stream it into split tar parts, return the manifest (without release info)."""
    run = run.resolve()
    repo_root = Path(__file__).resolve().parents[2]
    stage = work / "stage"
    parts_dir = work / "parts"
    for d in (stage, parts_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    ck_src = run / "ckpt" / ckpt
    _snapshot_ckpt(ck_src, stage / "payload" / "ckpt" / ckpt)
    meta = json.loads((stage / "payload" / "ckpt" / ckpt / "meta.json").read_text())
    ck_state = json.loads((stage / "payload" / "ckpt" / ckpt / "state.json").read_text())
    rdst = stage / "payload" / "run"
    for name in ("state.json", "registry.json", "trainer.log", "trainer_status.json", "latest_status.json",
                 "latest_status.md", "status_history.jsonl", "report.md", "report.json", "gh_backup_status.json"):
        if (run / name).exists():
            _link_or_copy(run / name, rdst / name, copy=True)
    _copy_tree(run / "experience", rdst / "experience")
    _copy_tree(run / "live", rdst / "live", skip_dirs=("snapshots",))
    redactions = redact_public(rdst / "live") if public else []
    _copy_tree(run / "github", rdst / "github")
    _copy_tree(run / "factory", rdst / "factory")
    for cfg in (repo_root / "configs" / "overnight").glob("*.yaml"):
        _link_or_copy(cfg, rdst / "configs" / cfg.name, copy=True)
    files = {}
    for f in sorted(stage.rglob("*")):
        if f.is_file():
            files[str(f.relative_to(stage))] = {"size": f.stat().st_size, "sha256": sha256_file(f)}
    c = ck_state.get("counters", {})
    info = {
        "schema": BUNDLE_SCHEMA, "kind": kind, "checkpoint": ckpt, "created": time.time(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": socket.gethostname(),
        "policy_version": meta.get("version"), "cycle": meta.get("cycle"), "phase": meta.get("phase"),
        "checkpoint_time": meta.get("time"), "models": meta.get("models"), "git_at_checkpoint": meta.get("git"),
        "git_now": git_commit(repo_root), "rl_lr": meta.get("rl_lr"),
        "training_step": {"policy_updates": c.get("policy_updates"), "bc_updates": c.get("bc_updates"),
                          "wm_updates": c.get("wm_updates"), "critic_updates": c.get("critic_updates"),
                          "cycles": c.get("cycles"), "attempts": c.get("attempts"), "successes": c.get("successes")},
        "best": ck_state.get("best"), "promoted": ck_state.get("promoted"),
        "contents": {
            "policy": f"payload/ckpt/{ckpt}/policy (LoRA r=16 on {meta.get('models', {}).get('policy')} "
                      f"@ {meta.get('models', {}).get('policy_revision')}; weights.safetensors + trainer.pt "
                      "= AdamW + LR scheduler state)",
            "world_model": f"payload/ckpt/{ckpt}/world_model (LoRA on {meta.get('models', {}).get('small')})",
            "critic": f"payload/ckpt/{ckpt}/critic (LoRA + value head on {meta.get('models', {}).get('small')})",
            "rng": f"payload/ckpt/{ckpt}/rng.pt (python/numpy/torch/cuda RNG + sampling generator)",
            "run_state": f"payload/ckpt/{ckpt}/state.json (curriculum, task indices, counters, evals, version)",
            "replay": "payload/run/experience/*.jsonl.gz",
            "delayed_outcome_ledger": "payload/run/live/ledger.sqlite (+ jobs/prequential/dedup DBs, cursors.json)",
            "grad_scaler": "none (bf16 training without loss scaling)",
        },
        "public_redactions": redactions,
        "files": files,
    }
    (stage / "BUNDLE_INFO.json").write_text(json.dumps(info, indent=1, default=str))
    stem = f"agentrs-{kind}-v{meta.get('version')}"
    sw = SplitWriter(parts_dir, stem, part_mib * 2**20)
    with tarfile.open(fileobj=sw, mode="w|") as tf:
        tf.add(stage / "BUNDLE_INFO.json", arcname="BUNDLE_INFO.json")
        tf.add(stage / "payload", arcname="payload")
    sw.close()
    shutil.rmtree(stage)
    info.pop("files")
    return {**info, "n_files": len(files), "payload_bytes": sum(v["size"] for v in files.values()),
            "tar": {"name": f"{stem}.tar", "size": sw.size, "sha256": sw.total.hexdigest()},
            "parts": sw.parts, "parts_dir": str(parts_dir)}


RESTORE_TXT = """Restore this checkpoint (needs the agentrs repo at the commit in MANIFEST.json, and a GitHub token):

    python -m agentrs.gh_backup restore --tag {tag} --dest /tmp/agentrs_restore --install-run runs/overnight

Manual reconstruction:  download every *.tar.partNNN asset, check each sha256 against MANIFEST.json,
    cat {stem}.tar.part* > {stem}.tar && sha256sum {stem}.tar   # must equal MANIFEST tar.sha256
    tar xf {stem}.tar        # -> BUNDLE_INFO.json (per-file sha256) + payload/
Then copy payload/ckpt/<name> to runs/overnight/ckpt/latest, payload/ckpt/<name>/state.json to
runs/overnight/state.json and payload/run/{{experience,live,github}} to runs/overnight/, and start the
trainer: python -m agentrs.econ.overnight --config configs/overnight/a6000.yaml --new-session.
The policy is base model {policy} @ revision {rev} + the LoRA in payload/ckpt/<name>/policy.
"""


def upload(run: Path, ckpt: str, kind: str, repo: str = DEFAULT_REPO, part_mib: int = 1900,
           work: Optional[Path] = None) -> dict:
    t0 = time.time()
    gh = GitHub(repo)
    work = work or (run.resolve().parent / ".gh_backup_work")
    public = not gh.api("GET", "").get("private", False)
    man = build_bundle(run, ckpt, work, part_mib, kind, public=public)
    parts_dir = Path(man.pop("parts_dir"))
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    tag = f"{TAG_PREFIX}{stamp}-{kind}-v{man['policy_version']}"
    stem = man["tar"]["name"][:-4]
    rel = gh.create_release(tag, f"AGENTRS {kind} checkpoint v{man['policy_version']} (cycle {man['cycle']}, {stamp} UTC)",
                            "Upload in progress (draft). Not usable until published.")
    man.update({"release_tag": tag, "release_url": rel["html_url"].replace("/releases/tag/untagged-", f"/releases/tag/{tag}#"),
                "repo": repo, "complete": True,
                "restore": RESTORE_TXT.format(tag=tag, stem=stem, policy=(man.get("models") or {}).get("policy"),
                                              rev=(man.get("models") or {}).get("policy_revision"))})
    man["release_url"] = f"https://github.com/{repo}/releases/tag/{tag}"
    (parts_dir / "MANIFEST.json").write_text(json.dumps(man, indent=1, default=str))
    (parts_dir / "RESTORE.md").write_text(man["restore"])
    try:
        assets = {}
        for p in man["parts"]:
            t1 = time.time()
            assets[p["name"]] = gh.upload_asset(rel, parts_dir / p["name"], p["name"])
            log(f"uploaded {p['name']} ({p['size'] / 2**20:.0f} MiB) in {time.time() - t1:.0f}s")
        for n in ("MANIFEST.json", "RESTORE.md"):
            assets[n] = gh.upload_asset(rel, parts_dir / n, n)
        # ---- verify every asset before publishing
        remote = {a["name"]: a for a in gh.api("GET", f"/releases/{rel['id']}/assets?per_page=100")}
        expected = {p["name"]: (p["size"], p["sha256"]) for p in man["parts"]}
        for n in ("MANIFEST.json", "RESTORE.md"):
            expected[n] = ((parts_dir / n).stat().st_size, sha256_file(parts_dir / n))
        method = {}
        for name, (size, sha) in expected.items():
            a = remote.get(name)
            if a is None or a.get("state") != "uploaded" or a["size"] != size:
                raise RuntimeError(f"asset {name} missing or wrong size on GitHub: {a and (a.get('state'), a['size'])} vs {size}")
            dig = (a.get("digest") or "")
            if dig.startswith("sha256:"):
                if dig[7:] != sha:
                    raise RuntimeError(f"asset {name}: GitHub sha256 {dig[7:]} != local {sha}")
                method[name] = "github-digest"
            else:
                tmp = work / "verify" / name
                gh.download_asset(a, tmp)
                got = sha256_file(tmp)
                tmp.unlink()
                if got != sha:
                    raise RuntimeError(f"asset {name}: downloaded sha256 {got} != local {sha}")
                method[name] = "re-download"
        body = (f"**Complete, verified AGENTRS checkpoint** `{ckpt}` ({kind}).\n\n"
                f"- policy version v{man['policy_version']}, cycle {man['cycle']}, phase {man['phase']}\n"
                f"- training step: `{json.dumps(man['training_step'])}`\n"
                f"- base models: `{json.dumps(man['models'])}`\n- code: `{json.dumps(man['git_now'])}`\n"
                f"- {len(man['parts'])} part(s), tar {man['tar']['size'] / 2**30:.2f} GiB, sha256 `{man['tar']['sha256']}`\n"
                f"- every asset verified ({', '.join(sorted(set(method.values())))}) before publishing\n"
                f"- public repository: third-party observer content redacted ({len(man['public_redactions'])} operations)\n\n"
                f"```\n{man['restore']}```\n")
        pub = gh.api("PATCH", f"/releases/{rel['id']}", {"draft": False, "body": body, "tag_name": tag})
        man["release_url"] = pub["html_url"]
    except BaseException:
        try:
            gh.api("DELETE", f"/releases/{rel['id']}")  # never leave a half-uploaded release behind
        except Exception:  # noqa: BLE001
            pass
        raise
    finally:
        shutil.rmtree(parts_dir, ignore_errors=True)
    rec = {"tag": tag, "url": man["release_url"], "kind": kind, "checkpoint": ckpt, "version": man["policy_version"],
           "cycle": man["cycle"], "checkpoint_time": man["checkpoint_time"], "uploaded_at": time.time(),
           "tar_bytes": man["tar"]["size"], "parts": len(man["parts"]), "verified": True,
           "verify_method": sorted(set(method.values())), "seconds": round(time.time() - t0),
           "training_step": man["training_step"]}
    log(f"published {tag}: {man['tar']['size'] / 2**30:.2f} GiB in {len(man['parts'])} part(s), verified, "
        f"{time.time() - t0:.0f}s -> {man['release_url']}")
    return rec


# ---------------------------------------------------------------------------- restore
def latest_complete(gh: GitHub, tag: Optional[str] = None, kind: Optional[str] = None) -> dict:
    rels = [r for r in gh.releases() if not r["draft"] and r["tag_name"].startswith(TAG_PREFIX)
            and any(a["name"] == "MANIFEST.json" for a in r["assets"])]
    if tag:
        rels = [r for r in rels if r["tag_name"] == tag]
    if kind:
        rels = [r for r in rels if f"-{kind}-" in r["tag_name"]]
    if not rels:
        raise RuntimeError("no complete checkpoint release found" + (f" for tag {tag}" if tag else ""))
    return max(rels, key=lambda r: r["created_at"])


def restore(dest: Path, repo: str = DEFAULT_REPO, tag: Optional[str] = None, kind: Optional[str] = None) -> dict:
    gh = GitHub(repo)
    rel = latest_complete(gh, tag, kind)
    assets = {a["name"]: a for a in rel["assets"]}
    dest.mkdir(parents=True, exist_ok=True)
    pdir = dest / "parts"
    gh.download_asset(assets["MANIFEST.json"], pdir / "MANIFEST.json")
    man = json.loads((pdir / "MANIFEST.json").read_text())
    log(f"restoring {rel['tag_name']}: {len(man['parts'])} part(s), {man['tar']['size'] / 2**30:.2f} GiB")
    for p in man["parts"]:
        f = pdir / p["name"]
        gh.download_asset(assets[p["name"]], f)
        got = sha256_file(f)
        if got != p["sha256"] or f.stat().st_size != p["size"]:
            raise RuntimeError(f"part {p['name']} failed verification ({got} != {p['sha256']})")

    class Cat(io.RawIOBase):
        def __init__(self, files):
            self.files, self.i, self.f, self.h = list(files), 0, None, hashlib.sha256()

        def readable(self):
            return True

        def readinto(self, b):
            while True:
                if self.f is None:
                    if self.i >= len(self.files):
                        return 0
                    self.f = open(self.files[self.i], "rb")
                    self.i += 1
                n = self.f.readinto(b)
                if n:
                    self.h.update(memoryview(b)[:n])
                    return n
                self.f.close()
                self.f = None

    out = dest / "bundle"
    if out.exists():
        shutil.rmtree(out)
    cat = Cat([pdir / p["name"] for p in man["parts"]])
    with tarfile.open(fileobj=io.BufferedReader(cat, 1 << 20), mode="r|") as tf:
        tf.extractall(out, filter="data")
    if cat.h.hexdigest() != man["tar"]["sha256"]:
        raise RuntimeError("reassembled tar sha256 mismatch")
    info = json.loads((out / "BUNDLE_INFO.json").read_text())
    bad = [p for p, e in info["files"].items()
           if not (out / p).exists() or (out / p).stat().st_size != e["size"] or sha256_file(out / p) != e["sha256"]]
    if bad:
        raise RuntimeError(f"{len(bad)} restored files failed sha256 verification, e.g. {bad[:3]}")
    shutil.rmtree(pdir)
    log(f"restored and verified {len(info['files'])} files into {out}")
    return {"tag": rel["tag_name"], "dir": str(out), "checkpoint": info["checkpoint"], "files": len(info["files"]),
            "policy_version": info["policy_version"], "verified": True}


def install_run(bundle_dir: Path, run: Path, force: bool = False) -> None:
    """Lay a restored bundle out as a resumable run directory (trainer resumes from ckpt/latest)."""
    if run.exists() and any(run.iterdir()) and not force:
        raise RuntimeError(f"{run} exists and is not empty; refusing to overwrite (use --force)")
    info = json.loads((bundle_dir / "BUNDLE_INFO.json").read_text())
    ck = bundle_dir / "payload" / "ckpt" / info["checkpoint"]
    run.mkdir(parents=True, exist_ok=True)
    (run / "ckpt").mkdir(exist_ok=True)
    for name in {info["checkpoint"], "latest"}:
        if (run / "ckpt" / name).exists():
            shutil.rmtree(run / "ckpt" / name)
        shutil.copytree(ck, run / "ckpt" / name)
    shutil.copy2(ck / "state.json", run / "state.json")  # the run state that matches these weights
    src = bundle_dir / "payload" / "run"
    for sub in ("experience", "live", "github", "factory", "configs"):
        if (src / sub).exists():
            shutil.copytree(src / sub, run / sub, dirs_exist_ok=True)
    for name in ("registry.json", "trainer.log"):
        if (src / name).exists():
            shutil.copy2(src / name, run / name)
    log(f"installed checkpoint {info['checkpoint']} (v{info['policy_version']}) as a resumable run in {run}")


# ---------------------------------------------------------------------------- scheduled service
class Watcher:
    """Uploads: an initial checkpoint, then at most every ``every_h`` hours, whenever a new best
    validated policy appears (rate-limited), and the final checkpoint when the run finishes."""

    def __init__(self, run: Path, repo: str, every_h: float, best_min_gap_min: float, part_mib: int):
        self.run, self.repo, self.every_h, self.best_gap, self.part_mib = run, repo, every_h, best_min_gap_min, part_mib
        self.status_path = run / "gh_backup_status.json"
        self.status = self._load()
        self.stop = False
        signal.signal(signal.SIGTERM, self._sig)

    def _sig(self, *_):
        self.stop = True

    def _load(self) -> dict:
        try:
            return json.loads(self.status_path.read_text())
        except (OSError, ValueError):
            return {"repo": self.repo, "uploads": [], "failures": [], "run_started_at": None}

    def _save(self) -> None:
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.status, indent=1, default=str))
        os.replace(tmp, self.status_path)

    def _state(self) -> dict:
        try:
            return json.loads((self.run / "state.json").read_text())
        except (OSError, ValueError):
            return {}

    def _do(self, ckpt: str, kind: str) -> bool:
        try:
            self.status["in_progress"] = {"ckpt": ckpt, "kind": kind, "since": time.time()}
            self._save()
            rec = upload(self.run, ckpt, kind, self.repo, self.part_mib)
            self.status["uploads"].append(rec)
            self.status["last_success"] = rec
            return True
        except Exception as e:  # noqa: BLE001 - keep the service alive, report loudly
            self.status["failures"].append({"time": time.time(), "ckpt": ckpt, "kind": kind, "error": str(e)[:800]})
            log(f"ERROR uploading {ckpt}: {e}")
            return False
        finally:
            self.status.pop("in_progress", None)
            self._save()

    def _ckpt_ok(self, name: str) -> bool:
        return (self.run / "ckpt" / name / "DONE").exists()

    def tick(self) -> bool:
        """One scheduling decision; returns False when the service should exit."""
        st = self._state()
        ups = self.status["uploads"]
        session = st.get("session_started_at") or st.get("started_at")
        mine = [u for u in ups if u["uploaded_at"] >= (session or 0) - 60]
        last = max((u["uploaded_at"] for u in mine), default=None)
        newest_ck = "latest" if self._ckpt_ok("latest") else ("after_bc" if self._ckpt_ok("after_bc") else None)
        if st.get("phase") == "done":
            if self._ckpt_ok("final") and not any(u["kind"] == "final" for u in mine):
                self._do("final", "final")
            if any(u["kind"] == "final" for u in mine):
                log("run finished and final checkpoint is on GitHub: exiting")
                return False
            return True
        if newest_ck is None:
            return True
        if last is None:
            # the newest checkpoint may already be on GitHub (uploaded and verified before this session)
            try:
                ct = json.loads((self.run / "ckpt" / newest_ck / "meta.json").read_text())["time"]
            except (OSError, ValueError, KeyError):
                ct = None
            prev = [u for u in ups if ct is not None and u.get("checkpoint_time") == ct and u.get("verified")]
            if prev:
                last = prev[-1]["uploaded_at"]
                mine = prev[-1:]
        if last is None:
            self._do(newest_ck, "initial")
            return True
        best = st.get("best") or {}
        up_best = max((u["version"] for u in mine if u["kind"] in ("best", "initial", "periodic", "final")
                       and u.get("version") == best.get("version")), default=None)
        last_best = max((u["uploaded_at"] for u in mine if u["kind"] == "best"), default=0)
        if (best.get("version") is not None and up_best is None and (best.get("cycle") or 0) > 0
                and self._ckpt_ok("best") and time.time() - last_best > self.best_gap * 60):
            self._do("best", "best")
            return True
        if time.time() - last > self.every_h * 3600:
            self._do(newest_ck, "periodic")
        return True

    def loop(self, interval: int) -> int:
        log(f"watching {self.run} -> github.com/{self.repo} (every {self.every_h} h, on new best, at the end)")
        while not self.stop:
            if not self.tick():
                return 0
            for _ in range(interval):
                if self.stop:
                    break
                time.sleep(1)
        # graceful shutdown: if the trainer's latest checkpoint is newer than anything uploaded, upload it
        st = self._state()
        last = max((u.get("checkpoint_time") or 0 for u in self.status["uploads"]), default=0)
        lc = (st.get("latest_checkpoint") or {}).get("time") or 0
        if self._ckpt_ok("latest") and lc > last + 600:
            log("SIGTERM: uploading the newest checkpoint before exiting")
            self._do("latest", "shutdown")
        return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agentrs.gh_backup", description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=["upload", "watch", "list", "restore"])
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--run", default="runs/overnight")
    ap.add_argument("--ckpt", default="latest")
    ap.add_argument("--kind", default="manual")
    ap.add_argument("--part-mib", type=int, default=1900)
    ap.add_argument("--every-h", type=float, default=float(os.environ.get("AGENTRS_GH_BACKUP_EVERY_H", "3")))
    ap.add_argument("--best-gap-min", type=float, default=45.0)
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--tag")
    ap.add_argument("--dest", default="/tmp/agentrs_restore")
    ap.add_argument("--install-run")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.action == "upload":
        rec = upload(Path(a.run), a.ckpt, a.kind, a.repo, a.part_mib)
        sp = Path(a.run) / "gh_backup_status.json"
        st = json.loads(sp.read_text()) if sp.exists() else {"repo": a.repo, "uploads": [], "failures": []}
        st["uploads"].append(rec)
        st["last_success"] = rec
        sp.write_text(json.dumps(st, indent=1, default=str))
        print(json.dumps(rec, indent=1))
    elif a.action == "watch":
        return Watcher(Path(a.run), a.repo, a.every_h, a.best_gap_min, a.part_mib).loop(a.interval)
    elif a.action == "list":
        gh = GitHub(a.repo)
        for r in sorted(gh.releases(), key=lambda r: r["created_at"]):
            size = sum(x["size"] for x in r["assets"])
            print(f"{r['created_at']}  {'DRAFT ' if r['draft'] else ''}{r['tag_name']:<48} {size / 2**30:6.2f} GiB  {r['html_url']}")
    elif a.action == "restore":
        res = restore(Path(a.dest), a.repo, a.tag)
        if a.install_run:
            install_run(Path(res["dir"]), Path(a.install_run), a.force)
        print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
