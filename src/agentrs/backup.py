"""Off-instance backups of learned weights and run state.

A cloud GPU instance can disappear (destroyed / recycled) and take its disk with it. This module
copies everything needed to keep or resume training — checkpoints (weights + optimizer state),
the RL crash-safety state, per-seed stage caches, configs, metrics and results — to an external
target, and restores it onto a fresh machine.

Targets
  hf://<user>/<repo>           private Hugging Face Hub model repo (versioned; token from
                               ``hf auth login`` / ``HF_TOKEN``, never from this repo)
  rclone:<remote>:<path>       any rclone remote (S3, Backblaze B2, Cloudflare R2, GDrive, SFTP, ...)
                               configured with ``rclone config``

Consistency: checkpoints are written to ``<dir>.tmp`` and atomically renamed, so the backup takes
a hard-link snapshot of each *completed* checkpoint directory (skipping ``*.tmp`` / ``*.old`` and
RL states without a ``DONE`` marker) and re-checks that the directory was not swapped while it was
being linked. Files are never modified in place by the trainers, so a linked file is a consistent
copy. After upload, the remote copy is verified against a sha256 manifest (rclone: ``rclone check``;
HF: LFS sha256 / size of every file). Nothing is ever deleted remotely.

    python -m agentrs.backup once   --target rclone:b2:my-bucket/agentrs
    python -m agentrs.backup watch  --target hf://me/agentrs-runs --interval 600
    python -m agentrs.backup restore --target hf://me/agentrs-runs [--only runs/final/v2_D_s0]
    python -m agentrs.backup verify --target ...

The target may also come from ``AGENTRS_BACKUP_TARGET`` (e.g. in ${WORKSPACE}/.env).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

SKIP_SUFFIXES = (".tmp", ".old")
SKIP_FILE_SUFFIXES = (".lock", ".tmp")
MANIFEST = "BACKUP_MANIFEST.json"


def log(msg: str) -> None:
    print(f"[backup {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------- selection + snapshot
def _skipped_dir(p: Path, root: Path) -> bool:
    rel = p.relative_to(root)
    return any(part.endswith(SKIP_SUFFIXES) or part.startswith(".") for part in rel.parts)


def _incomplete_rl_state(d: Path) -> bool:
    return d.name == "rl_state" and not (d / "DONE").exists()


def _dir_token(d: Path) -> tuple:
    st = os.stat(d)
    return (st.st_ino, st.st_mtime_ns)


def select_files(root: Path, include_experience: bool = True) -> dict[Path, list[Path]]:
    """Map each directory under ``root`` to the files to back up in it (completed state only)."""
    out: dict[Path, list[Path]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        if d != root and (_skipped_dir(d, root) or _incomplete_rl_state(d)):
            dirnames[:] = []
            continue
        if not include_experience and "experience" in d.relative_to(root).parts:
            dirnames[:] = []
            continue
        dirnames[:] = [n for n in dirnames if not n.endswith(SKIP_SUFFIXES) and not n.startswith(".")]
        files = [d / f for f in filenames if not f.endswith(SKIP_FILE_SUFFIXES) and not f.startswith(".")]
        if files:
            out[d] = files
    return out


def snapshot(root: Path, staging: Path, prefix: str, include_experience: bool = True,
             retries: int = 5) -> list[Path]:
    """Hard-link (or copy) every selected file into ``staging/prefix/...``; returns staged files."""
    if staging.exists():
        shutil.rmtree(staging)
    staged: list[Path] = []
    for d, files in select_files(root, include_experience).items():
        for attempt in range(retries):
            try:
                tok = _dir_token(d)
                dest_dir = staging / prefix / d.relative_to(root.parent)
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                dest_dir.mkdir(parents=True, exist_ok=True)
                cur = []
                for f in files:
                    dst = dest_dir / f.name
                    try:
                        os.link(f, dst)
                    except OSError:
                        shutil.copy2(f, dst)
                    cur.append(dst)
                if _dir_token(d) != tok:  # the checkpoint directory was replaced under us: redo it
                    raise FileNotFoundError(str(d))
                staged += cur
                break
            except FileNotFoundError:
                time.sleep(0.5 * (attempt + 1))
                files = [f for f in (d.iterdir() if d.exists() else []) if f.is_file()
                         and not f.name.endswith(SKIP_FILE_SUFFIXES) and not f.name.startswith(".")]
                if not d.exists():
                    break
        else:
            log(f"warning: {d} kept changing; skipped this round")
    return staged


# ---------------------------------------------------------------------------- hashing + manifest
@dataclass
class HashCache:
    path: Path
    data: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "HashCache":
        try:
            return cls(path, json.loads(path.read_text()))
        except (OSError, ValueError):
            return cls(path, {})

    def sha256(self, f: Path) -> str:
        st = f.stat()
        key = f"{st.st_ino}:{st.st_size}:{st.st_mtime_ns}"
        hit = self.data.get(key)
        if hit:
            return hit
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        self.data[key] = h.hexdigest()
        return self.data[key]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data))
        os.replace(tmp, self.path)


def build_manifest(staging: Path, files: list[Path], cache: HashCache) -> dict:
    entries = {str(f.relative_to(staging)): {"size": f.stat().st_size, "sha256": cache.sha256(f)} for f in files}
    return {"host": socket.gethostname(), "time": time.time(), "files": entries}


# ---------------------------------------------------------------------------- targets
class Target:
    def upload(self, staging: Path, message: str) -> None:
        raise NotImplementedError

    def remote_files(self) -> dict[str, dict]:
        """path -> {"size": int, "sha256": Optional[str]} for every file on the target."""
        raise NotImplementedError

    def download(self, dest: Path, only: Optional[str] = None) -> None:
        raise NotImplementedError

    def verify(self, manifest: dict) -> list[str]:
        """Paths from the manifest that are missing or differ on the target."""
        remote = self.remote_files()
        bad = []
        for p, e in manifest["files"].items():
            r = remote.get(p)
            if r is None or r["size"] != e["size"] or (r.get("sha256") and r["sha256"] != e["sha256"]):
                bad.append(p)
        return bad


class RcloneTarget(Target):
    def __init__(self, spec: str, run: Callable = subprocess.run):
        self.dest = spec  # "<remote>:<path>"
        self.run = run

    def _rc(self, *args: str, capture: bool = False) -> subprocess.CompletedProcess:
        cmd = ["rclone", *args]
        r = self.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd[:3])} ... failed ({r.returncode}): {r.stderr.strip()[-800:]}")
        return r

    def upload(self, staging: Path, message: str) -> None:
        self._rc("copy", str(staging), self.dest, "--checksum", "--transfers", "8", "--retries", "5",
                 "--low-level-retries", "20")
        self._staging = staging

    def remote_files(self) -> dict[str, dict]:
        r = self._rc("lsjson", "-R", "--files-only", "--hash", self.dest)
        out = {}
        for e in json.loads(r.stdout or "[]"):
            hashes = {k.lower(): v for k, v in (e.get("Hashes") or {}).items()}
            out[e["Path"]] = {"size": e["Size"], "sha256": hashes.get("sha256")}
        return out

    def verify(self, manifest: dict) -> list[str]:
        bad = super().verify(manifest)  # size (+ sha256 where the backend exposes it)
        staging = getattr(self, "_staging", None)
        if not bad and staging is not None and staging.exists():
            # content check with the strongest hash both sides support (md5/sha1/...); --download
            # falls back to byte comparison for backends without a common hash
            r = self.run(["rclone", "check", str(staging), self.dest, "--one-way"], capture_output=True, text=True)
            if r.returncode != 0:
                r = self.run(["rclone", "check", str(staging), self.dest, "--one-way", "--download"],
                             capture_output=True, text=True)
                if r.returncode != 0:
                    bad.append(f"rclone check failed: {r.stderr.strip()[-300:]}")
        return bad

    def download(self, dest: Path, only: Optional[str] = None) -> None:
        src = self.dest.rstrip("/") + ("/" + only.strip("/") if only else "")
        dst = dest / only.strip("/") if only else dest
        self._rc("copy", src, str(dst), "--checksum", "--transfers", "8", "--retries", "5")


class HFTarget(Target):
    def __init__(self, repo_id: str, api=None):
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi()
        self.api, self.repo_id = api, repo_id
        self._ensured = False

    def _ensure(self) -> None:
        if not self._ensured:
            self.api.create_repo(self.repo_id, repo_type="model", private=True, exist_ok=True)
            self._ensured = True

    def upload(self, staging: Path, message: str) -> None:
        self._ensure()
        self.api.upload_folder(repo_id=self.repo_id, repo_type="model", folder_path=str(staging),
                               commit_message=message)

    def remote_files(self) -> dict[str, dict]:
        self._ensure()
        out = {}
        for e in self.api.list_repo_tree(self.repo_id, repo_type="model", recursive=True, expand=True):
            if not hasattr(e, "size") or getattr(e, "type", "file") == "directory" or e.__class__.__name__ == "RepoFolder":
                continue
            lfs = getattr(e, "lfs", None)
            sha = getattr(lfs, "sha256", None) if lfs is not None else None
            out[e.path] = {"size": e.size, "sha256": sha}
        return out

    def download(self, dest: Path, only: Optional[str] = None) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(self.repo_id, repo_type="model", local_dir=str(dest),
                          allow_patterns=[only.strip("/") + "/**"] if only else None)


def make_target(spec: str) -> Target:
    if spec.startswith("hf://"):
        return HFTarget(spec[len("hf://"):])
    if spec.startswith("rclone:"):
        return RcloneTarget(spec[len("rclone:"):])
    raise ValueError(f"unknown backup target {spec!r} (use hf://<repo> or rclone:<remote>:<path>)")


# ---------------------------------------------------------------------------- operations
def backup_once(target: Target, runs_dir: Path, prefix: str = "", include_experience: bool = True,
                state_dir: Optional[Path] = None) -> dict:
    runs_dir = runs_dir.resolve()
    state_dir = state_dir or Path(os.environ.get("AGENTRS_CACHE", Path.home() / ".cache" / "agentrs")) / "backup"
    staging = runs_dir.parent / ".agentrs_backup_staging"
    cache = HashCache.load(state_dir / "hashes.json")
    t0 = time.time()
    files = snapshot(runs_dir, staging, prefix, include_experience)
    if not files:
        log(f"nothing to back up under {runs_dir}")
        return {"files": 0}
    manifest = build_manifest(staging, files, cache)
    cache.save()
    last_path = state_dir / "last_manifest.json"
    try:
        last = json.loads(last_path.read_text())["files"]
    except (OSError, ValueError, KeyError):
        last = {}
    changed = [p for p, e in manifest["files"].items() if last.get(p) != e]
    total = sum(e["size"] for e in manifest["files"].values())
    if not changed:
        shutil.rmtree(staging, ignore_errors=True)
        log(f"up to date ({len(files)} files, {total / 2**20:.1f} MiB)")
        return {"files": len(files), "changed": 0, "bytes": total}
    mpath = staging / (prefix + "/" if prefix else "") / MANIFEST
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=1))
    target.upload(staging, f"agentrs backup: {len(changed)} changed files from {manifest['host']}")
    bad = target.verify(manifest)
    shutil.rmtree(staging, ignore_errors=True)
    if bad:
        raise RuntimeError(f"verification failed for {len(bad)} files, e.g. {bad[:3]}")
    last_path.parent.mkdir(parents=True, exist_ok=True)
    last_path.write_text(json.dumps(manifest))
    log(f"uploaded+verified {len(changed)} changed of {len(files)} files "
        f"({total / 2**20:.1f} MiB total) in {time.time() - t0:.0f}s")
    return {"files": len(files), "changed": len(changed), "bytes": total, "verified": True}


def watch(target: Target, runs_dir: Path, interval: int, prefix: str = "", include_experience: bool = True) -> None:
    log(f"watching {runs_dir} every {interval}s")
    failures = 0
    while True:
        try:
            if runs_dir.exists():
                backup_once(target, runs_dir, prefix, include_experience)
            failures = 0
        except Exception as e:  # keep the daemon alive; report loudly
            failures += 1
            log(f"ERROR (attempt {failures}): {e}")
        time.sleep(min(interval, 60 * failures) if failures else interval)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m agentrs.backup", description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=["once", "watch", "restore", "verify", "ls"])
    ap.add_argument("--target", default=os.environ.get("AGENTRS_BACKUP_TARGET", ""))
    ap.add_argument("--runs", default=os.environ.get("AGENTRS_RUNS_DIR", "runs"))
    ap.add_argument("--interval", type=int, default=int(os.environ.get("AGENTRS_BACKUP_INTERVAL", "600")))
    ap.add_argument("--no-experience", action="store_true", help="skip experience parquet shards")
    ap.add_argument("--only", help="restore: a sub-path such as runs/final/v2_D_s0")
    ap.add_argument("--dest", default=".", help="restore: directory that will contain runs/")
    a = ap.parse_args(argv)
    if not a.target:
        sys.exit("no backup target: pass --target or set AGENTRS_BACKUP_TARGET "
                 "(hf://<user>/<repo> or rclone:<remote>:<path>)")
    tgt = make_target(a.target)
    runs = Path(a.runs)
    if a.action == "once":
        print(json.dumps(backup_once(tgt, runs, include_experience=not a.no_experience)))
    elif a.action == "watch":
        watch(tgt, runs, a.interval, include_experience=not a.no_experience)
    elif a.action == "restore":
        dest = Path(a.dest)
        tgt.download(dest, a.only)
        log(f"restored {a.only or 'everything'} into {dest.resolve()}")
    elif a.action == "ls":
        for p, e in sorted(tgt.remote_files().items()):
            print(f"{e['size']:>14,d}  {p}")
    elif a.action == "verify":
        state = Path(os.environ.get("AGENTRS_CACHE", Path.home() / ".cache" / "agentrs")) / "backup"
        manifest = json.loads((state / "last_manifest.json").read_text())
        bad = tgt.verify(manifest)
        print(json.dumps({"checked": len(manifest["files"]), "bad": bad[:20], "ok": not bad}))
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
