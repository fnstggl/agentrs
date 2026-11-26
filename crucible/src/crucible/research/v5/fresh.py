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
from .learners import ARCHITECTURES, code_digest, params_hash

CHECKS = ("fresh_process", "isolated_interpreter", "architecture_immutable", "safe_loading", "authorised_parameters",
          "parameter_hash_verified", "histories_cleared", "recurrent_state_cleared", "memories_cleared",
          "retrieval_cleared", "replay_unavailable", "optimizer_reset", "frozen_verified")


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


def weights_only_eval(manifest: Mapping, worlds: list[Mapping], *, timeout: float = 900.0,
                      params_file: str | None = None) -> dict:
    """Score the checkpoint's parameters alone, in a fresh isolated process, on `worlds` ([{entry, seed,
    difficulty?}], scored by `episodes.score_episode`). Returns {checks, ok, scores, refused?}."""
    base = tempfile.mkdtemp(prefix="crucible-v5-fresh-")
    try:
        inp = os.path.join(base, "in")
        os.makedirs(inp)
        for name, src in (("arch.json", os.path.join(manifest["dir"], "arch.json")),
                          ("params.npz", params_file or os.path.join(manifest["dir"], "params.npz"))):
            dst = os.path.join(inp, name)
            shutil.copyfile(src, dst)
            os.chmod(dst, stat.S_IRUSR | stat.S_IRGRP)
        env = isolated_env()
        from .distribution import build_registry, thaw
        reg = build_registry()
        entries = {w["entry"]: thaw(reg[w["entry"]]) for w in worlds}
        req = {"input": inp, "arch_sha256": manifest["arch_sha256"], "params_hash": manifest["params_hash"],
               "worlds": [dict(w) for w in worlds], "entries": entries, "parent_pid": os.getpid()}
        boot = ("import sys; _pre = sorted(m for m in sys.modules if m.startswith('crucible')); "
                "sys.path.insert(0, sys.argv[1]); from crucible.research.v5.fresh import _child; _child(_pre)")
        p = subprocess.run([sys.executable, "-I", "-c", boot, src_root()], input=json.dumps(req).encode(),
                           capture_output=True, env=env, cwd=env["HOME"], timeout=timeout)
        if p.returncode != 0:
            return {"ok": False, "checks": {}, "error": p.stderr.decode()[-2000:]}
        return json.loads(p.stdout.decode().strip().splitlines()[-1])
    finally:
        shutil.rmtree(base, ignore_errors=True)


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
    learner = LR.make(arch["arch"], int(arch.get("seed", 0)), **dict(arch.get("cfg") or {}))
    declared = set(learner.parameters())
    numeric = all(np.issubdtype(a.dtype, np.number) for a in params.values())
    checks["authorised_parameters"] = set(params) == declared and numeric
    if not checks["authorised_parameters"]:
        out["refused"] = f"unauthorised parameter entries {sorted(set(params) ^ declared)}"
        return done()
    h0 = params_hash(params)
    checks["parameter_hash_verified"] = h0 == req["params_hash"]
    if not checks["parameter_hash_verified"]:
        out["refused"] = "parameter hash mismatch"
        return done()
    learner.load_parameters(params)
    opt0 = LR.make(arch["arch"], int(arch.get("seed", 0)), **dict(arch.get("cfg") or {})).optimizer_state()
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
        out["scores"].append({"entry": w["entry"], "seed": int(w["seed"]), "score": r["score"],
                              "early_rate": r["early_rate"], "late_rate": r["late_rate"],
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
    ``pickle`` (an object array that needs pickle to load) or ``arch_edit`` (architecture configuration changed after
    the hash was recorded). Returns a manifest pointing at the copy (with the original expected hashes)."""
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
    else:
        raise ValueError(kind)
    with open(os.path.join(directory, "params.npz"), "wb") as fh:
        np.savez(fh, **params)
    return dict(manifest, dir=directory)
