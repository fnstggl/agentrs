"""Fresh-process, weights-only evaluation (MASTER_SPEC Part 15.4; U.3, SP-15.4).

    ck = save_checkpoint(learner, directory)            # arch.json + params.npz (+ memory/optimizer, never loaded)
    out = weights_only_eval(ck, worlds)                  # a new `python -I` process scores the parameters alone

The child process receives exactly two files - the immutable architecture configuration (`arch.json`) and the
parameter arrays (`params.npz`) - copied read-only into an empty private directory, plus the expected hashes, and
nothing else of the parent: an empty HOME, a private TMPDIR and cache, one thread, no PYTHONPATH, no experience
store, no replay buffer, no memory file. In the child (all checks are computed there, by the evaluator code, not
reported by the learner):

    fresh_process            the child's pid is new and no crucible module was imported before its boot code ran
    isolated_interpreter     ``sys.flags.isolated``; HOME and the working directory are empty private directories
    architecture_immutable   SHA-256 of arch.json equals the hash recorded when the checkpoint was written, and the
                             architecture code (`learners.code_digest`) equals the code the checkpoint was bound to
    safe_loading             parameters are read with ``numpy.load(allow_pickle=False)``: no code is executed
    authorised_parameters    the array names equal the architecture's declared parameters, every array is numeric
                             (a memory table smuggled into the parameter file is refused)
    parameter_hash_verified  the evaluator's own hash of the loaded arrays equals the expected hash
    histories_cleared        no episode history or task context remains (`reset_state`)
    recurrent_state_cleared  no within-episode or temporary adaptation state remains
    memories_cleared         written memories and per-task states are empty (`clear_memory`)
    retrieval_cleared        retrieval tables are empty and no memory file was supplied
    replay_unavailable       no experience/replay store is reachable (no store paths in the environment, nothing but
                             the two checkpoint files in the input directory)
    optimizer_reset          optimizer state is the architecture's initial state
    frozen_verified          the parameter hash after evaluation equals the hash before (measured, not self-reported)
    authorised_configuration the architecture configuration holds only the keys the architecture declares
                             (`learners.CONFIG`), each a JSON scalar: no memory table, path or payload rides in it
    filesystem_confined      the child runs under the Landlock jail of `crucible.sandbox.isolation` (own uid, no
                             sockets, read-only standard library, numpy and crucible source, read-write only its own
                             private directory) and cannot open a canary file the parent wrote outside it: a stash
                             written beside or outside the checkpoint is unreadable (SP-15.8)

Within-episode adaptation runs (`in_process_eval`) intentionally keep local state and are reported separately; the
protocol compares the two, and a gain in process with an unchanged checkpoint is *not* a weights-only result
(`invalid_weights_claim`).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Mapping

import numpy as np

from .evaluator import isolated_env, src_root
from .learners import ARCHITECTURES, CONFIG, code_digest, params_hash

CHECKS = ("fresh_process", "isolated_interpreter", "architecture_immutable", "safe_loading", "authorised_parameters",
          "parameter_hash_verified", "histories_cleared", "recurrent_state_cleared", "memories_cleared",
          "retrieval_cleared", "replay_unavailable", "optimizer_reset", "frozen_verified", "authorised_configuration",
          "filesystem_confined")


def _sha_file(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def save_checkpoint(learner, directory: str) -> dict:
    """Write a checkpoint: arch.json (immutable config bound to the architecture code), params.npz (authorised
    parameters), memory.json and optimizer.json (non-parameter state, never loaded by weights-only evaluation)."""
    os.makedirs(directory, exist_ok=True)
    arch = dict(learner.config(), code_digest=code_digest())
    ap = os.path.join(directory, "arch.json")
    with open(ap, "w") as fh:
        json.dump(arch, fh, sort_keys=True)
    params = {k: np.asarray(v, dtype=np.float64) for k, v in learner.parameters().items()}
    pp = os.path.join(directory, "params.npz")
    with open(pp, "wb") as fh:
        np.savez(fh, **params)
    json.dump(learner.memory_state(), open(os.path.join(directory, "memory.json"), "w"))
    json.dump(learner.optimizer_state(), open(os.path.join(directory, "optimizer.json"), "w"))
    man = {"dir": directory, "arch_sha256": _sha_file(ap), "params_hash": params_hash(params),
           "arch": arch["arch"], "n_params": int(sum(a.size for a in params.values()))}
    json.dump(man, open(os.path.join(directory, "manifest.json"), "w"), sort_keys=True)
    return man


def _others_can_read(path: str) -> bool:
    """Whether a process of another, unprivileged uid can reach `path` (every ancestor searchable by others)."""
    p = os.path.abspath(path)
    while True:
        try:
            if not os.stat(p).st_mode & stat.S_IXOTH:
                return False
        except OSError:
            return False
        parent = os.path.dirname(p)
        if parent == p:
            return True
        p = parent


def _tmp_root() -> str:
    """A temporary directory root the jail uid can traverse (TMPDIR may sit inside a private 0700 tree)."""
    t = tempfile.gettempdir()
    return t if _others_can_read(t) else "/var/tmp"


def _source_for_jail(dest: str) -> str:
    """The crucible source root the jailed child imports: the checkout itself when the jail uid can reach it, else a
    read-only copy of the package (code and small data, no caches) under `dest` (e.g. a checkout inside a private
    0700 scratch directory)."""
    root = src_root()
    if _others_can_read(os.path.join(root, "crucible", "research", "v5", "fresh.py")):
        return root
    out = os.path.join(dest, "src")
    shutil.copytree(os.path.join(root, "crucible"), os.path.join(out, "crucible"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for dp, dirs, files in os.walk(out):
        os.chmod(dp, 0o755)
        for f in files:
            os.chmod(os.path.join(dp, f), 0o644)
    return out


def _jailed(argv: list[str], work: str, src: str | None = None) -> tuple[list[str], str]:
    """`argv` (a python command line) under the sandbox jail: own uid, Landlock with read-only system libraries,
    standard library, numpy and the crucible source tree, read-write `work` only, no sockets, no new processes.
    Raises `isolation.JailUnavailable` where the jail cannot be applied."""
    import sysconfig
    from ...sandbox import isolation as ISO
    ok, why = ISO.available()
    if not ok:
        raise ISO.JailUnavailable(why)
    py = os.path.realpath(sys.executable)
    paths = sysconfig.get_paths()
    np_dir = os.path.dirname(os.path.abspath(np.__file__))
    site = os.path.dirname(np_dir)
    ro = ISO.RO_SYSTEM + [paths["stdlib"], paths["platstdlib"], os.path.join(paths["stdlib"], "lib-dynload"), np_dir,
                          os.path.join(site, "numpy.libs"), os.path.dirname(py), os.path.join(sys.base_prefix, "lib"),
                          src or src_root()]
    uid = ISO.uid_for("weights-only-eval")
    cfg = ISO._write_cfg({"uid": uid, "gid": uid, "mode": "tool", "ro": sorted({p for p in ro if os.path.exists(p)}),
                          "exec": [py] + ISO._loaders(), "rw": [work], "chown": [work]})
    return [sys.executable, "-I", "-S", "-B", ISO.JAIL, cfg, py, *argv[1:]], cfg


def weights_only_eval(manifest: Mapping, worlds: list[Mapping], *, timeout: float = 900.0,
                      params_file: str | None = None, jail: bool = True) -> dict:
    """Score the checkpoint's parameters alone, in a fresh isolated process, on `worlds` ([{entry, seed,
    difficulty?}], scored by `episodes.score_episode`). Returns {checks, ok, scores, refused?}. ``jail=False``
    exists only for the adversarial demonstrations of `adversarial` (it shows what the jail prevents); the check
    `filesystem_confined` then fails, so such a run is never ``ok``."""
    import numpy
    tmp_root = _tmp_root()
    base = tempfile.mkdtemp(prefix="crucible-v5-fresh-", dir=tmp_root)
    outside = tempfile.mkdtemp(prefix="crucible-v5-canary-", dir=tmp_root)
    cfg_path = src_dir = None
    try:
        inp = os.path.join(base, "in")
        os.makedirs(inp)
        for name, src in (("arch.json", os.path.join(manifest["dir"], "arch.json")),
                          ("params.npz", params_file or os.path.join(manifest["dir"], "params.npz"))):
            dst = os.path.join(inp, name)
            shutil.copyfile(src, dst)
            os.chmod(dst, stat.S_IRUSR | stat.S_IRGRP)
        canary = os.path.join(outside, "canary.txt")
        with open(canary, "w") as fh:
            fh.write("outside the checkpoint")
        os.chmod(canary, 0o644)
        os.chmod(outside, 0o755)
        home, tmp = os.path.join(base, "home"), os.path.join(base, "tmp")
        os.makedirs(home)
        os.makedirs(tmp)
        env = isolated_env(home)
        shutil.rmtree(os.path.dirname(env["TMPDIR"]), ignore_errors=True)
        env.update(TMPDIR=tmp, CRUCIBLE_CACHE=os.path.join(tmp, "cache"))
        from .distribution import build_registry, thaw
        reg = build_registry()
        entries = {w["entry"]: thaw(reg[w["entry"]]) for w in worlds}
        req = {"input": inp, "arch_sha256": manifest["arch_sha256"], "params_hash": manifest["params_hash"],
               "worlds": [dict(w) for w in worlds], "entries": entries, "parent_pid": os.getpid(),
               "canary": canary}
        site = os.path.dirname(os.path.dirname(os.path.abspath(numpy.__file__)))
        boot = ("import sys; _pre = sorted(m for m in sys.modules if m.startswith('crucible')); "
                "sys.path.insert(0, sys.argv[1]); sys.path.append(sys.argv[2]); "
                "from crucible.research.v5.fresh import _child; _child(_pre)")
        src = src_root()
        if jail:
            src_dir = tempfile.mkdtemp(prefix="crucible-v5-fresh-src-", dir=tmp_root)
            os.chmod(src_dir, 0o755)
            src = _source_for_jail(src_dir)
        cmd = [sys.executable, "-I", "-c", boot, src, site]
        if jail:
            cmd, cfg_path = _jailed(cmd, base, src)
        p = subprocess.run(cmd, input=json.dumps(req).encode(), capture_output=True, env=env, cwd=home,
                           timeout=timeout)
        if p.returncode != 0:
            return {"ok": False, "checks": {}, "error": p.stderr.decode()[-2000:]}
        out = json.loads(p.stdout.decode().strip().splitlines()[-1])
        out["jailed"] = bool(jail)
        return out
    finally:
        shutil.rmtree(base, ignore_errors=True)
        shutil.rmtree(outside, ignore_errors=True)
        if src_dir:
            shutil.rmtree(src_dir, ignore_errors=True)
        if cfg_path:
            try:
                os.unlink(cfg_path)
            except OSError:
                pass


def _child(pre_imported: list[str]) -> None:
    req = json.loads(sys.stdin.read())
    checks: dict[str, bool] = {}
    out = {"pid": os.getpid(), "checks": checks, "refused": None, "scores": []}

    def done():
        out["ok"] = bool(checks) and all(checks.values()) and out["refused"] is None
        print(json.dumps(out))

    checks["fresh_process"] = os.getpid() != int(req["parent_pid"]) and not pre_imported
    home = os.environ.get("HOME", "")
    checks["isolated_interpreter"] = bool(sys.flags.isolated) and "PYTHONPATH" not in os.environ and \
        os.path.isdir(home) and not os.listdir(home) and os.getcwd() == home
    inp = req["input"]
    files = sorted(os.listdir(inp))
    ap, pp = os.path.join(inp, "arch.json"), os.path.join(inp, "params.npz")
    checks["architecture_immutable"] = _sha_file(ap) == req["arch_sha256"]
    arch = json.load(open(ap))
    checks["architecture_immutable"] = checks["architecture_immutable"] and arch.get("code_digest") == code_digest() \
        and arch.get("arch") in ARCHITECTURES
    if not checks["architecture_immutable"]:
        out["refused"] = "architecture configuration or code changed"
        return done()
    try:
        with np.load(pp, allow_pickle=False) as z:
            params = {k: np.array(z[k]) for k in z.files}
        checks["safe_loading"] = True
    except (ValueError, OSError) as e:
        checks["safe_loading"] = False
        out["refused"] = f"parameters could not be loaded without executing code: {type(e).__name__}"
        return done()
    from . import learners as LR
    cfg = arch.get("cfg") or {}
    allowed = CONFIG.get(arch["arch"], {})
    checks["authorised_configuration"] = isinstance(cfg, dict) and set(cfg) <= set(allowed) and all(
        isinstance(v, allowed[k]) and not (isinstance(v, str) and len(v) > 64) for k, v in cfg.items()
        if k in allowed)
    if not checks["authorised_configuration"]:
        out["refused"] = f"unauthorised architecture configuration {sorted(set(cfg) - set(allowed)) or sorted(cfg)}"
        return done()
    try:
        with open(req["canary"]) as fh:
            fh.read()
        checks["filesystem_confined"] = False
    except OSError:
        checks["filesystem_confined"] = True
    learner = LR.make(arch["arch"], int(arch.get("seed", 0)), **dict(cfg))
    fresh_params = {k: np.asarray(v) for k, v in learner.parameters().items()}
    declared = set(fresh_params)
    # every entry has exactly the shape AND dtype the freshly constructed architecture gives it (DEF-O-12): a
    # checkpoint with widened layers, or with the declared shapes in a wider type (complex128 carries two reals per
    # entry; float16/float32/int64 change what the declared architecture computes), under a manifest the adversary
    # rehashed, is refused (SP-15.8)
    wrong = sorted(k for k in set(params) & declared if params[k].shape != fresh_params[k].shape)
    wrong_type = sorted(k for k in set(params) & declared if params[k].dtype != fresh_params[k].dtype)
    checks["authorised_parameters"] = set(params) == declared and not wrong and not wrong_type
    if not checks["authorised_parameters"]:
        out["refused"] = (f"parameter shapes differ from the declared architecture {wrong}" if wrong else
                          f"parameter dtypes differ from the declared architecture {wrong_type}" if wrong_type else
                          f"unauthorised parameter entries {sorted(set(params) ^ declared)}")
        return done()
    h0 = params_hash(params)
    checks["parameter_hash_verified"] = h0 == req["params_hash"]
    if not checks["parameter_hash_verified"]:
        out["refused"] = "parameter hash mismatch"
        return done()
    learner.load_parameters(params)
    opt0 = LR.make(arch["arch"], int(arch.get("seed", 0)), **dict(cfg)).optimizer_state()
    learner.clear_memory()
    learner.reset_optimizer()
    learner.reset_state()
    learner.frozen = True
    checks["histories_cleared"] = not learner.ctx
    checks["recurrent_state_cleared"] = not learner.state_summary()["within_episode"]
    mem = learner.memory_state()
    checks["memories_cleared"] = all(not v for v in mem.values())
    checks["retrieval_cleared"] = checks["memories_cleared"] and "memory.json" not in files
    checks["replay_unavailable"] = files == ["arch.json", "params.npz"] and not any(
        k.startswith("CRUCIBLE_") and k != "CRUCIBLE_CACHE" for k in os.environ)
    checks["optimizer_reset"] = learner.optimizer_state() == opt0
    from .episodes import score_episode
    reg = dict(req["entries"])            # the world entries to score (data, not code)
    rng = np.random.default_rng(12345)
    for w in req["worlds"]:
        r = score_episode(reg, w["entry"], int(w["seed"]), learner, learn=False, rng=rng,
                          difficulty=w.get("difficulty"))
        b = score_episode(reg, w["entry"], int(w["seed"]), learner, learn=False, rng=rng,
                          difficulty=w.get("difficulty"), blind=True)
        out["scores"].append({"entry": w["entry"], "seed": int(w["seed"]), "score": r["score"],
                              "blind_score": b["score"], "early_rate": r["early_rate"], "late_rate": r["late_rate"],
                              "interactions": r["interactions"]})
    checks["frozen_verified"] = params_hash(learner.parameters()) == h0
    out["params_hash"] = h0
    done()


def in_process_eval(learner, worlds: list[Mapping], *, rng_seed: int = 12345) -> dict:
    """Within-episode / in-process evaluation: the learner object as it is (memory and process state kept),
    frozen. Reported separately from weights-only results."""
    from .distribution import build_registry
    from .episodes import score_episode
    reg = build_registry()
    rng = np.random.default_rng(rng_seed)
    was = learner.frozen
    learner.frozen = True
    h0 = params_hash(learner.parameters())
    scores = []
    for w in worlds:
        r = score_episode(reg, w["entry"], int(w["seed"]), learner, learn=False, rng=rng,
                          difficulty=w.get("difficulty"))
        scores.append({"entry": w["entry"], "seed": int(w["seed"]), "score": r["score"],
                       "early_rate": r["early_rate"], "late_rate": r["late_rate"], "interactions": r["interactions"]})
    learner.frozen = was
    return {"scores": scores, "params_hash": h0, "params_unchanged": params_hash(learner.parameters()) == h0}


def tampered_copy(manifest: Mapping, kind: str, directory: str) -> dict:
    """Adversarial checkpoints for the gate: ``memory_in_params`` (a memory table added to the parameter file),
    ``pickle`` (an object array that needs pickle to load), ``arch_edit`` (architecture configuration changed after
    the hash was recorded), ``widened_layers`` (every array of the parameter file doubled along its last axis) or
    ``retyped:<dtype>`` (the declared shapes in another type: ``complex128`` carries a second real per entry,
    ``float32``/``float16``/``int64`` change what the declared architecture computes; DEF-O-12); for the last two
    the manifest's parameter hash is recomputed by the adversary, which controls its own checkpoint. Returns a
    manifest pointing at the copy (with the original expected hashes, except where the kind rehashes)."""
    os.makedirs(directory, exist_ok=True)
    shutil.copyfile(os.path.join(manifest["dir"], "arch.json"), os.path.join(directory, "arch.json"))
    with np.load(os.path.join(manifest["dir"], "params.npz"), allow_pickle=False) as z:
        params = {k: np.array(z[k]) for k in z.files}
    if kind == "memory_in_params":
        params["memory_table"] = np.arange(12, dtype=np.float64)
    elif kind == "pickle":
        params["payload"] = np.array([{"stash": [1, 2, 3]}], dtype=object)
    elif kind == "arch_edit":
        a = json.load(open(os.path.join(directory, "arch.json")))
        a["cfg"] = dict(a.get("cfg") or {}, lr=9.0)
        json.dump(a, open(os.path.join(directory, "arch.json"), "w"), sort_keys=True)
    elif kind == "widened_layers":
        if {"W1", "b1", "W2"} <= set(params):          # a consistent wider hidden layer (the network still runs)
            W1, b1, W2 = params["W1"], params["b1"], params["W2"]
            params = dict(params, W1=np.concatenate([W1, 0.5 * W1], axis=1), b1=np.concatenate([b1, b1]),
                          W2=np.concatenate([0.5 * W2, 0.5 * W2], axis=0))
        else:
            params = {k: (np.concatenate([v, v], axis=-1) if v.ndim else v) for k, v in params.items()}
    elif kind.startswith("retyped:"):
        dt = np.dtype(kind.split(":", 1)[1])
        if dt.kind == "c":                              # a hidden second real per entry (twice the capacity)
            params = {k: v.astype(dt) + 1j * 0.25 for k, v in params.items()}
        elif dt.kind in "iu":
            params = {k: np.round(v * 1000).astype(dt) for k, v in params.items()}
        else:
            params = {k: v.astype(dt) for k, v in params.items()}
    else:
        raise ValueError(kind)
    with open(os.path.join(directory, "params.npz"), "wb") as fh:
        np.savez(fh, **params)
    if kind == "widened_layers" or kind.startswith("retyped:"):
        return dict(manifest, dir=directory, params_hash=params_hash(params))
    return dict(manifest, dir=directory)
