"""Instrument integrity (SP-15.8): adversarial learners against the research instrument's evaluator
(`crucible.research.v5`: the evaluator process and its participant channel, fresh-process weights-only evaluation,
the generator of Tier 1 worlds).

Each control runs an adversarial learner of `crucible.redteam.v5.adversaries` (or an adversarial checkpoint, or
a fixed rule that would exploit the generator) through the instrument's own interfaces and passes only if the
instrument's measurement is not fooled - and, where the attack hides something, only if the attack demonstrably had
something to hide (a control whose attack did nothing is not passed):

    process_memory_stash        learned state kept in process memory with a constant checkpoint is not credited as
                                weight learning by the fresh-process weights-only evaluation (while the in-process
                                gain it hides is real)
    files_outside_checkpoint    the same with the learned state in files outside the checkpoint, in the checkpoint
                                directory's memory file and in the architecture configuration
    checkpoint_tampering        checkpoints changed after they were written or carrying more than the declared
                                architecture (extra arrays, pickled objects, widened layers, an edited
                                configuration, parameters swapped after hashing) are refused
    evaluator_internals         a learner driven by the evaluator's own driver (`evaluator.run_remote_episode`)
                                reaches no evaluator internals: no evaluator handle or channel in its process (its
                                verified score, its partition), no evaluator environment, memory or sealed salt
                                through the operating system
    evaluator_internals_hosted  the same learner hosted by `crucible.sandbox.learnerhost` (the jailed learner process)
    hosted_state_outlives_the_process
                                a jailed learner process leaves its state outside its checkpoint (shared and parent
                                directories, System V IPC, POSIX queues) and a FRESH learner process cannot recover it
    evaluator_survives_a_stalled_participant
                                a learner that sends part of a request and goes quiet does not freeze the evaluator
                                (the harness's channel keeps getting answers)
    interaction_budget          a learner granted k informative interactions cannot make more by calling its
                                participant environment itself (the evaluator refuses, or marks the run)
    timing_channel              the latency of participant operations reveals neither the episode's partition nor its
                                verified outcome (evidence of absence: interval bounds with stated power, one fixed
                                setting so that the learner's own choice does not confound latency and score)
    partition_awareness         nothing a participant receives tells a validation, an evaluation or a sealed episode
                                apart (no seed, alias, partition or ticket in its observations, tools or events, and
                                the same structure in all three)
    calibration_shortcut        no fixed rule of the public defaults (the audit's anti-default 25/75 rule, mirror,
                                centre, extremes, a fixed step away from the default) solves generated worlds: its
                                median share of the best constant setting's improvement stays below one half

`run_controls()` returns {name: {"passed", "summary", "evidence"}}; `tests/v5/test_gate_SP_15_8.py` records them.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from typing import Callable

import numpy as np

CONTROLS = ("process_memory_stash", "files_outside_checkpoint", "checkpoint_tampering", "evaluator_internals",
            "evaluator_internals_hosted", "hosted_state_outlives_the_process", "evaluator_survives_a_stalled_participant",
            "interaction_budget", "timing_channel", "partition_awareness", "calibration_shortcut")
LEARNER_SEEDS = (0, 1, 2, 3)     # fallbacks; the stash battery uses the instrument's own design sizes (_design)
TRAIN_FAMILIES = 4
TRAIN_TASKS = 24                 # as the instrument's own battery (instrument.Design: 24 tasks x 3 repeats)
REPEATS = 3
EVAL_SEEDS = (1_000_301, 1_000_302, 1_000_303)


def _reg():
    from ...research.v5.distribution import build_registry
    return build_registry()


def _families(reg, n: int = TRAIN_FAMILIES) -> list[str]:
    from ...research.v5.episodes import lever_entries
    return sorted(lever_entries(reg, "train"))[:n]


def _learnable_families(reg, n: int = TRAIN_FAMILIES) -> list[str]:
    """The families on which the instrument itself established that its `weights` learner gains on new seeds after
    training (`research.v5.instrument.establish`, independent seeds, cached and shared with Gate U): there a learner
    that keeps the same learning outside its parameters has something real to hide."""
    from ...research.v5 import instrument as IN
    from ...research.v5.episodes import lever_entries
    est = IN.establish(reg, lever_entries(reg, "train"))
    ok = sorted((f for f, v in est.items() if v["weights_gain"] >= IN.ESTABLISH_MIN), key=lambda f: -est[f]["weights_gain"])
    if not ok:
        raise RuntimeError("no family on which weight learning is established; the stash controls cannot be run")
    return sorted(ok[:n])


def _design() -> dict:
    """The stash battery's sizes follow the instrument's own design (`research.v5.instrument.Design`: learner seeds,
    training tasks x repeats, unfamiliar worlds per family), so the adversary's hidden gain is measured with the
    power the instrument itself needs for the honest learner's effects (N raised it to ten worlds per family and five
    seeds when real effects of 0.13-0.49 had intervals reaching below zero); evaluation world seeds stay this
    control's own."""
    from ...research.v5 import instrument as IN
    f = getattr(getattr(IN, "Design", None), "__dataclass_fields__", {})

    def get(name, default):
        v = getattr(f.get(name), "default", None)
        return default if v is None or isinstance(v, type) else v
    n_eval = len(tuple(get("eval_seeds", EVAL_SEEDS)))
    return {"seeds": tuple(range(len(tuple(get("seeds", LEARNER_SEEDS))))),
            "train_tasks": int(get("train_tasks", TRAIN_TASKS)), "repeats": int(get("repeats", REPEATS)),
            "eval_seeds": tuple(EVAL_SEEDS[0] + i for i in range(max(n_eval, len(EVAL_SEEDS))))}


def _result(passed: bool, summary: str, evidence: dict) -> dict:
    return {"passed": bool(passed), "summary": summary, "evidence": evidence}


# ====================================================================== stash controls (weights-only evaluation)
def _stash_battery(make: Callable[[int], object], reg) -> dict:
    """Untrained, in-process-after-training and fresh-process weights-only scores of one adversary per learner seed
    on unfamiliar worlds (new seeds of its training families), with N's own drivers, scorer and fresh-process
    evaluator."""
    from ...research.v5 import fresh as FR, learners as LR
    from ...research.v5.episodes import run_episode, score_episode
    from ...research.v5.sampler import build_env
    fams = _learnable_families(reg)
    D = _design()
    worlds = [(f, s) for f in fams for s in D["eval_seeds"]]
    out = {"u": [], "ip": [], "wo": [], "params_changed": [], "fresh_checks": [], "reports": [], "worlds": worlds,
           "refused": [], "design": {k: (len(v) if isinstance(v, tuple) else v) for k, v in D.items()}}
    work = tempfile.mkdtemp(prefix="crucible-rt-instrument-")
    try:
        for seed in D["seeds"]:
            L = make(seed)
            rng = np.random.default_rng(50_000 + seed)

            def scores():
                was = L.frozen
                L.frozen = True
                v = [score_episode(reg, f, s, L, learn=False, rng=rng)["score"] for f, s in worlds]
                L.frozen = was
                return v
            h0 = LR.params_hash(L.parameters())
            out["u"].append(scores())
            for i in range(D["train_tasks"]):
                f = fams[i % len(fams)]
                for _ in range(D["repeats"]):
                    run_episode(build_env(reg[f], 300 + i, {}, {"mode": "assigned"}), L, learn=True, rng=rng)
            out["ip"].append(scores())
            out["params_changed"].append(LR.params_hash(L.parameters()) != h0)
            man = FR.save_checkpoint(L, os.path.join(work, f"adv-{seed}"))
            fo = FR.weights_only_eval(man, [{"entry": f, "seed": s} for f, s in worlds])
            if not fo.get("ok") and fo.get("refused"):
                # the fresh evaluation refused the checkpoint (e.g. state smuggled into the architecture
                # configuration): nothing of it is credited - recorded, and the seed carries no weights-only score
                out["refused"].append(str(fo["refused"])[:300])
                out["wo"].append(None)
                out["fresh_checks"].append({"refused": True})
            elif not fo.get("ok"):
                raise RuntimeError(f"weights-only evaluation failed: {fo.get('error')}")
            else:
                out["wo"].append([x["score"] for x in fo["scores"]])
                out["fresh_checks"].append(fo["checks"])
            out["reports"].append(L.report() if hasattr(L, "report") else {})
            if hasattr(L, "cleanup"):
                L.cleanup()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out


def _stash_verdict(name: str, b: dict) -> dict:
    from ...research.v5 import instrument as IN
    u, ip = (np.array(b[k], dtype=float) for k in ("u", "ip"))
    keep = [i for i, w in enumerate(b["wo"]) if w is not None]          # seeds whose checkpoint was not refused
    in_process_gain = IN._one(ip - u)
    # "the attack had something to hide" is this control's own single question, tested at the conventional 95% level
    # (the instrument's labels use a family-wise level across its seven explanations); the credit verdict below
    # (labelled_weights) is the instrument's own labelling rule, unchanged
    from ...research.v5 import hstats as HS
    hidden = HS.crossed_diff(np.zeros_like(ip - u), ip - u, level=0.95)
    hid_something = bool(hidden["lo"] > 0 and hidden["estimate"] >= IN.EFFECTS["persistent_state"]
                         and min(hidden["seeds"]) >= HS.MIN_SEEDS)
    if keep:
        wo = np.array([b["wo"][i] for i in keep], dtype=float)
        persistent = IN._one(ip[keep] - wo)
        weight = IN._one(wo - u[keep])
        labelled_weights = IN._label(weight, IN.EFFECTS["weight_learning"]) and \
            all(b["params_changed"][i] for i in keep)
        labelled_persistent = IN._label(persistent, IN.EFFECTS["persistent_state"])
    else:                                       # every checkpoint refused: nothing credited at all
        persistent = weight = {"estimate": None}
        labelled_weights, labelled_persistent = False, None
    fresh_ok = all(all(c.values()) for c in b["fresh_checks"] if not c.get("refused"))
    ev = {"weights_only_gain": _stat(weight), "in_process_gain": _stat(in_process_gain),
          "in_process_gain_95": _stat(hidden), "design": b.get("design"),
          "persistent_state": _stat(persistent), "labelled_weight_learning": bool(labelled_weights),
          "labelled_persistent_state": bool(labelled_persistent), "params_changed": b["params_changed"],
          "fresh_checks_all_passed": fresh_ok, "learner_seeds": len(b["u"]), "worlds": len(b["worlds"]),
          "checkpoints_refused": b.get("refused", []), "adversary_reports": b["reports"]}
    if labelled_weights or not fresh_ok:
        return _result(False, f"{name}: the weights-only evaluation credited state kept outside the parameters", ev)
    if not hid_something:
        return _result(False, f"{name}: inconclusive - the adversary's in-process gain was not significant, so the "
                              "control shows nothing", ev)
    wtxt = "every checkpoint refused" if weight.get("estimate") is None else f"weights-only gain {weight['estimate']:.3f}"
    return _result(True, f"{name}: in-process gain {in_process_gain['estimate']:.3f} (hidden outside the parameters) "
                         f"not credited; {wtxt}, persistent state labelled {labelled_persistent}"
                         + (f"; {len(b.get('refused', []))} checkpoint(s) refused" if b.get("refused") else ""), ev)


def _stat(r: dict) -> dict:
    return {k: (round(float(r[k]), 4) if isinstance(r.get(k), (int, float, np.floating)) else r.get(k))
            for k in ("estimate", "lo", "hi", "se") if k in r}


def control_process_memory_stash(reg=None) -> dict:
    from .adversaries import ProcessStash, _hidden
    reg = reg or _reg()
    try:
        b = _stash_battery(lambda s: ProcessStash(seed=s), reg)
    finally:
        _hidden().clear()
    return _stash_verdict("process_memory_stash", b)


def control_files_outside_checkpoint(reg=None) -> dict:
    """The file channels (home cache, /tmp, /var/tmp, /dev/shm, the checkpoint's memory file) through the full
    battery, then the architecture-configuration channel on its own: a fresh evaluation that refuses a configuration
    with extra keys would otherwise refuse every checkpoint and hide whether the file channels are credited."""
    from ...research.v5 import fresh as FR
    from .adversaries import FileStash
    reg = reg or _reg()
    b = _stash_battery(lambda s: FileStash(seed=s, smuggle_cfg=False), reg)
    r = _stash_verdict("files_outside_checkpoint", b)
    written = sorted({p for rep in b["reports"] for p, v in rep.get("files", {}).items() if v == "written"})
    r["evidence"]["stash_files_written"] = written
    if r["passed"] and not written:
        r["passed"] = False
        r["summary"] = "files_outside_checkpoint: inconclusive - no stash file could be written"
    cfg = _cfg_smuggle(reg, FR, FileStash)
    r["evidence"]["architecture_config_channel"] = cfg
    if not cfg["not_credited"]:
        r["passed"] = False
        r["summary"] = (f"files_outside_checkpoint: state smuggled into the architecture configuration was credited "
                        f"(weights-only {cfg['weights_only']:.3f} vs untrained {cfg['untrained']:.3f})")
    else:
        r["summary"] += "; architecture-configuration channel: " + cfg["how"]
    return r


def _cfg_smuggle(reg, FR, FileStash) -> dict:
    """One trained FileStash whose network also sits in arch.json `cfg`: refused, or scored like the untrained
    network (the configuration does not carry the learning into the fresh child)."""
    from ...research.v5.episodes import run_episode, score_episode
    from ...research.v5.sampler import build_env
    fams = _learnable_families(reg)
    D = _design()
    worlds = [(f, s) for f in fams for s in D["eval_seeds"]]
    L = FileStash(seed=0, smuggle_cfg=True)
    rng = np.random.default_rng(60_000)
    work = tempfile.mkdtemp(prefix="crucible-rt-cfg-")
    try:
        L.frozen = True
        untrained = float(np.mean([score_episode(reg, f, s, L, learn=False, rng=rng)["score"] for f, s in worlds]))
        L.frozen = False
        for i in range(D["train_tasks"]):
            for _ in range(D["repeats"]):
                run_episode(build_env(reg[fams[i % len(fams)]], 300 + i, {}, {"mode": "assigned"}), L, learn=True,
                            rng=rng)
        man = FR.save_checkpoint(L, os.path.join(work, "cfg"))
        for p in L._paths():                       # only the configuration may carry the network into the child
            try:
                os.unlink(p)
            except OSError:
                pass
        fo = FR.weights_only_eval(man, [{"entry": f, "seed": s} for f, s in worlds])
    finally:
        L.cleanup()
        shutil.rmtree(work, ignore_errors=True)
    if not fo.get("ok"):
        return {"not_credited": bool(fo.get("refused")), "how": f"refused ({str(fo.get('refused') or fo.get('error'))[:120]})",
                "untrained": untrained, "weights_only": None}
    wo = float(np.mean([x["score"] for x in fo["scores"]]))
    return {"not_credited": wo <= untrained + 0.02, "how": f"accepted, not used (weights-only {wo:.3f}, untrained "
                                                           f"{untrained:.3f})", "untrained": untrained, "weights_only": wo}


# ====================================================================== checkpoint tampering
def control_checkpoint_tampering(reg=None) -> dict:
    """A trained honest `weights` checkpoint, then each tampering; each must be refused by the fresh evaluation."""
    from ...research.v5 import fresh as FR, learners as LR
    from ...research.v5.episodes import run_episode
    from ...research.v5.sampler import build_env
    reg = reg or _reg()
    fams = _families(reg, 1)
    worlds = [{"entry": fams[0], "seed": EVAL_SEEDS[0]}]
    L = LR.make("weights", 0)
    rng = np.random.default_rng(7)
    for i in range(4):
        run_episode(build_env(reg[fams[0]], 300 + i, {}, {"mode": "assigned"}), L, learn=True, rng=rng)
    work = tempfile.mkdtemp(prefix="crucible-rt-tamper-")
    out: dict[str, dict] = {}
    try:
        base = FR.save_checkpoint(L, os.path.join(work, "base"))
        honest = FR.weights_only_eval(base, worlds)
        p = {k: np.asarray(v) for k, v in L.parameters().items()}

        def variant(name: str, edit: Callable[[str, dict], dict]) -> None:
            d = os.path.join(work, name)
            shutil.copytree(base["dir"], d)
            man = dict(base, dir=d)
            man = edit(d, man) or man
            r = FR.weights_only_eval(man, worlds)
            out[name] = {"accepted": bool(r.get("ok")), "refused": r.get("refused"),
                         "error": (r.get("error") or "")[-160:] or None,
                         "score": (r.get("scores") or [{}])[0].get("score")}

        def write_params(d, params):
            with open(os.path.join(d, "params.npz"), "wb") as fh:
                np.savez(fh, **params)

        def rehash(d, man):                       # the adversary also controls the manifest of its own checkpoint
            with np.load(os.path.join(d, "params.npz"), allow_pickle=True) as z:
                params = {k: z[k] for k in z.files}
            try:
                man["params_hash"] = LR.params_hash(params)
            except (TypeError, ValueError):
                pass
            return man

        variant("extra_array", lambda d, m: (write_params(d, {**p, "memory": np.arange(64.0)}), rehash(d, m))[1])
        variant("object_array", lambda d, m: (write_params(d, {**p, "W1": np.array([{"x": 1}] * 4, dtype=object)}),
                                              m)[1])
        H = 64
        wide = {"W1": np.random.default_rng(1).normal(0, 0.5, (p["W1"].shape[0], H)), "b1": np.zeros(H),
                "W2": np.random.default_rng(2).normal(0, 0.1, (H, p["W2"].shape[1])), "b2": p["b2"]}
        variant("widened_layers", lambda d, m: (write_params(d, wide), rehash(d, m))[1])

        def edit_arch(d, m):
            a = json.load(open(os.path.join(d, "arch.json")))
            a["cfg"] = dict(a.get("cfg") or {}, lr=0.5)
            json.dump(a, open(os.path.join(d, "arch.json"), "w"), sort_keys=True)
            return m
        variant("arch_edited_after_manifest", edit_arch)
        variant("params_swapped_after_manifest",
                lambda d, m: (write_params(d, {k: v + 0.01 for k, v in p.items()}), m)[1])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    accepted = sorted(k for k, v in out.items() if v["accepted"])
    ev = {"honest_accepted": bool(honest.get("ok")), "variants": out, "accepted_tampered": accepted}
    if not honest.get("ok"):
        return _result(False, f"checkpoint_tampering: the honest checkpoint was refused ({honest.get('refused')})", ev)
    if accepted:
        return _result(False, f"checkpoint_tampering: tampered checkpoints accepted and scored: {accepted}", ev)
    return _result(True, f"checkpoint_tampering: all {len(out)} tampered checkpoints refused; the honest one scored", ev)


# ====================================================================== the evaluator process from a learner
def _evaluator(sealed: str):
    from ...research.v5 import evaluator as E
    return E.Evaluator.start(sealed_dir=sealed)


def _internals_verdict(name: str, rep: dict, how: str) -> dict:
    bad = {k: v for k, v in rep.get("found", {}).items()
           if k in ("verified_result_read", "verified_result_read_via_channel", "registry_read", "sealed_dir_named",
                    "sealed_salt_read", "evaluator_memory_read", "evaluator_server_in_process", "gc:evaluator",
                    "other_process_environ_read")}
    ev = {"how": how, "reach": bad, "found": rep.get("found"), "attempts": rep.get("attempts"),
          "verified_values_read_mid_episode": rep.get("verified_values_read_mid_episode")}
    if bad:
        return _result(False, f"{name}: the learner reached evaluator internals ({how}): {sorted(bad)}", ev)
    return _result(True, f"{name}: no evaluator handle, channel, environment, memory or salt in reach ({how}; "
                         f"{len(rep.get('attempts', {}))} attempts)", ev)


def control_evaluator_internals(reg=None) -> dict:
    """The adversary driven by the evaluator's own driver, as the instrument drives learners today."""
    from ...research.v5 import evaluator as E
    from .adversaries import Prober
    reg = reg or _reg()
    f = _families(reg, 1)[0]
    sealed = tempfile.mkdtemp(prefix="crucible-rt-sealed-")
    ev = _evaluator(sealed)
    try:
        L = Prober(seed=0)
        t = ev.open_episode(f, 1_000_401, purpose="dev")
        E.run_remote_episode(ev, t, L, np.random.default_rng(0))
        rep = L.report()
    finally:
        ev.close()
        shutil.rmtree(sealed, ignore_errors=True)
    return _internals_verdict("evaluator_internals", rep, "evaluator.run_remote_episode in the harness process")


def control_evaluator_internals_hosted(reg=None) -> dict:
    """The same adversary hosted by the jailed learner process."""
    from ...sandbox.learnerhost import LearnerHost
    reg = reg or _reg()
    f = _families(reg, 1)[0]
    sealed = tempfile.mkdtemp(prefix="crucible-rt-sealed-")
    ev = _evaluator(sealed)
    try:
        with LearnerHost("crucible.redteam.v5.adversaries:make", {"name": "prober", "seed": 0},
                         participant=ev.part.sock) as h:
            t = ev.open_episode(f, 1_000_402, purpose="dev")
            h.episode(t, learn=False, seed=0)
            rep = h.call("report")
            score = ev.result(t).get("score")
            uid = h.jail_uid
    finally:
        ev.close()
        shutil.rmtree(sealed, ignore_errors=True)
    r = _internals_verdict("evaluator_internals_hosted", rep, "crucible.sandbox.learnerhost (jailed process)")
    r["evidence"].update(verified_score_from_evaluator=score, learner_uid=uid)
    if r["passed"] and score is None:
        r["passed"], r["summary"] = False, "evaluator_internals_hosted: the hosted episode produced no verified result"
    return r


def control_hosted_state_outlives_the_process(reg=None) -> dict:
    """A jailed learner process that will be replaced by a fresh one (a protocol that restarts the learner to clear
    state outside its checkpoint) leaves a canary in every place it can write - shared and parent directories, its
    own working directory, System V IPC and POSIX message queues - and the fresh learner process looks for it (also
    in the old process's working directory, whose path it is told). Passes when the old process demonstrably held
    the state (its own directory write succeeded) and the fresh one recovers none of it."""
    import zlib
    from ...sandbox.learnerhost import LearnerHost
    from .crossworld import _ipc_cleanup
    key = 0x5D000000 | (zlib.crc32(f"carry|{os.getpid()}|{time.monotonic_ns()}".encode()) & 0x00FFFFF0)
    canary = f"carried-state-{key:x}"
    uids = []
    try:
        with LearnerHost(None) as a:
            uids.append(a.jail_uid)
            left = a.function("crucible.redteam.v5.adversaries:stash_everywhere", {"key": key, "canary": canary})
        with LearnerHost(None) as b:
            uids.append(b.jail_uid)
            got = b.function("crucible.redteam.v5.adversaries:recall_everywhere",
                             {"key": key, "previous_cwd": left.get("cwd", "")})
    finally:
        removed = _ipc_cleanup(key, uids)
    carried = sorted([f"file:{d}" for d, v in (got.get("files") or {}).items() if canary in str(v)] +
                     [f"ipc:{k}" for k, v in ((got.get("ipc") or {}).get("read") or {}).items()
                      if canary in str(v) or (k == "sem_value" and v == len(canary))])
    escaped = list(left.get("outside_cwd") or [])
    ev = {"left": left, "recalled": got, "carried": carried, "writes_outside_private_dir": escaped,
          "uids": uids, "left_in_kernel": removed}
    if (left.get("files") or {}).get(".") != "written":
        return _result(False, "hosted_state_outlives_the_process: the learner could not even write its own directory "
                              "(the attack had nothing to carry; inconclusive)", ev)
    if carried:
        return _result(False, f"hosted_state_outlives_the_process: a fresh learner process recovered state the old "
                              f"one left outside its checkpoint: {carried}", ev)
    return _result(True, "hosted_state_outlives_the_process: the fresh learner process recovered nothing (writes "
                         f"outside the private directory: {escaped or 'none'}; kernel IPC refused)", ev)


def control_evaluator_survives_a_stalled_participant(reg=None) -> dict:
    """A jailed learner holding only its participant channel sends part of a request and never ends the line. The
    evaluator process serves the participant and the harness's evaluator channel; it must keep answering the
    harness (here: an episode's verified result) instead of waiting on the participant forever - otherwise a
    learner decides when (and whether) its episodes are measured."""
    import threading
    from ...sandbox.learnerhost import LearnerHost
    reg = reg or _reg()
    f = _families(reg, 1)[0]
    sealed = tempfile.mkdtemp(prefix="crucible-rt-sealed-")
    ev = _evaluator(sealed)
    out, hung = {}, None
    try:
        t = ev.open_episode(f, 1_000_403, purpose="dev")
        with LearnerHost(None, participant=ev.part.sock) as h:
            out["stall"] = h.function("crucible.redteam.v5.adversaries:stall_channel", {})
            res = []

            def ask():
                try:
                    res.append(ev.call("result", t))
                except Exception as e:  # noqa: BLE001
                    res.append({"error": repr(e)[:200]})
            th = threading.Thread(target=ask, daemon=True)
            th.start()
            th.join(20)
            hung = th.is_alive()
            out["harness_answer"] = None if hung else res[0]
    finally:
        if hung:
            ev.proc.kill()                       # the evaluator never answered: stop it (it is this control's)
            try:
                ev.proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                pass
        else:
            ev.close()
        shutil.rmtree(sealed, ignore_errors=True)
    ev_ = {"attack": out, "harness_blocked_s": 20 if hung else 0}
    if not out.get("stall", {}).get("sent"):
        return _result(False, "evaluator_survives_a_stalled_participant: the stall was not sent (inconclusive)", ev_)
    if hung:
        return _result(False, "evaluator_survives_a_stalled_participant: a partial participant request froze the "
                              "evaluator: the harness got no answer in 20 s (it reads whole lines from the participant "
                              "in one thread)", ev_)
    return _result(True, "evaluator_survives_a_stalled_participant: the evaluator kept answering the harness while the "
                         "participant stalled", ev_)


# ====================================================================== interaction budget
def control_interaction_budget(reg=None) -> dict:
    """The driver grants one informative interaction; the adversary executes lever settings on its own each tick."""
    import inspect
    from ...research.v5 import evaluator as E
    from .adversaries import BudgetBypass
    reg = reg or _reg()
    f = _families(reg, 1)[0]
    sealed = tempfile.mkdtemp(prefix="crucible-rt-sealed-")
    ev = _evaluator(sealed)
    budget = 1
    try:
        params = inspect.signature(ev.open_episode).parameters
        kw = {"budget": budget} if "budget" in params else {}
        t = ev.open_episode(f, 1_000_403, purpose="dev", **kw)
        L = BudgetBypass(seed=0)
        E.run_remote_episode(ev, t, L, np.random.default_rng(0), budget=budget)
        res = ev.result(t)
        rep = L.report()
    finally:
        ev.close()
        shutil.rmtree(sealed, ignore_errors=True)
    n = int(res.get("interactions") or 0)
    marked = any(k in res for k in ("budget_exceeded", "invalid")) and bool(res.get("budget_exceeded") or
                                                                              res.get("invalid"))
    evd = {"budget": budget, "evaluator_supports_budget": bool(kw), "evaluator_counted_interactions": n,
           "adversary": rep, "result_marked": marked}
    if rep.get("extra_interactions_executed", 0) == 0 and n <= budget:
        return _result(False, "interaction_budget: inconclusive - the adversary made no extra interaction", evd)
    if n > budget and not marked:
        return _result(False, f"interaction_budget: a learner granted {budget} informative interaction made {n} and "
                              "the evaluator's result is unmarked (the budget is enforced only by the learner-side "
                              "driver)", evd)
    return _result(True, f"interaction_budget: extra interactions refused or the run marked ({n} counted)", evd)


# ====================================================================== timing channel
def _spearman(a, b) -> tuple[float, float]:
    from scipy import stats
    r = stats.spearmanr(a, b)
    return float(r.statistic), float(r.pvalue)


TIMING_PER_PARTITION = 80      # episodes per partition (validation, sealed), interleaved
TIMING_SETTING = 0.5           # ONE fixed setting for every Timer: the learner's own setting moved latency and score
TIMING_RHO_BOUND = 0.35        # largest latency-score association the control accepts as "no channel" (95% interval)
TIMING_RHO_DETECT = 0.20       # a significant association at least this large is reported as a channel
TIMING_RATIO_BOUND = 0.10      # partitions' median latencies within +-10% (95% bootstrap interval of the ratio)


def _fisher_ci(rho: float, n: int, level: float = 0.95) -> tuple[float, float]:
    from scipy import stats
    z, se = float(np.arctanh(np.clip(rho, -0.999999, 0.999999))), 1.0 / np.sqrt(max(n - 3, 1))
    q = float(stats.norm.ppf(0.5 + level / 2))
    return float(np.tanh(z - q * se)), float(np.tanh(z + q * se))


def _timing_power(n: int, rho: float = TIMING_RHO_BOUND) -> float:
    """Probability that the score test reports a channel (significant and |r| >= TIMING_RHO_DETECT) when the true
    latency-score correlation is `rho` (Fisher z normal approximation)."""
    from scipy import stats
    se = 1.0 / np.sqrt(max(n - 3, 1))
    thr = max(float(np.arctanh(TIMING_RHO_DETECT)), 1.96 * se)
    return float(1.0 - stats.norm.cdf((thr - float(np.arctanh(rho))) / se))


def control_timing_channel(reg=None) -> dict:
    """Timer adversaries play one fixed setting in validation and sealed episodes of the same family, interleaved;
    per-episode median latency of the participant round trip is compared across partitions and against the verified
    score. The control passes only on evidence of absence (wave-3b verifier: 24 episodes and a 'p < 0.01 and
    |rho| > 0.5' leak rule could not detect a channel; the learner's own six settings confounded latency and score):
    the 95% interval of the latency-score Spearman correlation must lie inside +-TIMING_RHO_BOUND and the 95%
    bootstrap interval of the ratio of the partitions' median latencies inside 1 +- TIMING_RATIO_BOUND. A
    significant association beyond the detection size is a channel; anything else is inconclusive (not passed)."""
    from scipy import stats
    from ...research.v5 import evaluator as E
    from .adversaries import Timer
    reg = reg or _reg()
    f = _families(reg, 1)[0]
    sealed = tempfile.mkdtemp(prefix="crucible-rt-sealed-")
    ev = _evaluator(sealed)
    rows = []
    n = TIMING_PER_PARTITION
    try:
        dev = [ev.open_episode(f, 1_000_500 + i, purpose="dev") for i in range(n)]
        ev.must("declare_sealed", "rt-timing", [f], n, 1)
        pr = ev.preregister({"control": "timing_channel", "metric": "latency", "episodes_per_partition": n,
                             "setting": TIMING_SETTING})
        tk = [x["ticket"] for x in ev.must("open_sealed", "rt-timing", ack=E.SEALED_ACK, purpose="red team timing",
                                           prereg=pr)["tickets"]]
        order = [(p, i, t) for i, pair in enumerate(zip(dev, tk)) for p, t in zip(("validation", "sealed"), pair)]
        for j, (part, i, t) in enumerate(order):          # interleaved: drift over time hits both partitions alike
            L = Timer(seed=i, setting=TIMING_SETTING)
            E.run_remote_episode(ev, t, L, np.random.default_rng(i), budget=1)
            lat = L.report()["latencies_s"]
            res = ev.result(t)
            rows.append({"partition": part, "lat": float(np.median(lat)), "score": res.get("score")})
    finally:
        ev.close()
        shutil.rmtree(sealed, ignore_errors=True)
    a = np.array([r["lat"] for r in rows if r["partition"] == "validation"])
    b = np.array([r["lat"] for r in rows if r["partition"] == "sealed"])
    mw = stats.mannwhitneyu(a, b)
    rng = np.random.default_rng(0)
    ratios = [float(np.median(rng.choice(b, len(b)))) / max(float(np.median(rng.choice(a, len(a)))), 1e-12)
              for _ in range(2000)]
    r_lo, r_hi = (float(x) for x in np.quantile(ratios, [0.025, 0.975]))
    rel = abs(np.median(a) - np.median(b)) / max(np.median(a), 1e-9)
    sc = [(r["lat"], r["score"]) for r in rows if r["score"] is not None]
    rho, prho = _spearman([x for x, _ in sc], [y for _, y in sc])
    c_lo, c_hi = _fisher_ci(rho, len(sc))
    part_absent = r_lo >= 1 - TIMING_RATIO_BOUND and r_hi <= 1 + TIMING_RATIO_BOUND
    score_absent = c_lo > -TIMING_RHO_BOUND and c_hi < TIMING_RHO_BOUND
    leak_part = mw.pvalue < 0.01 and rel > TIMING_RATIO_BOUND
    leak_score = (c_lo > 0 or c_hi < 0) and abs(rho) >= TIMING_RHO_DETECT
    ev_ = {"episodes": len(rows), "episodes_scored": len(sc), "setting": TIMING_SETTING,
           "median_latency_validation_s": round(float(np.median(a)), 6),
           "median_latency_sealed_s": round(float(np.median(b)), 6), "partition_pvalue": round(float(mw.pvalue), 4),
           "partition_relative_difference": round(float(rel), 4),
           "partition_ratio_ci95": [round(r_lo, 4), round(r_hi, 4)], "latency_score_spearman": round(rho, 4),
           "latency_score_pvalue": round(prho, 4), "latency_score_ci95": [round(c_lo, 4), round(c_hi, 4)],
           "power_to_detect_rho_bound": round(_timing_power(len(sc)), 3),
           "bounds": {"rho": TIMING_RHO_BOUND, "rho_detect": TIMING_RHO_DETECT, "ratio": TIMING_RATIO_BOUND}}
    if leak_part or leak_score:
        return _result(False, f"timing_channel: latency reveals {'the partition' if leak_part else 'the outcome'} "
                              f"(rho={rho:.2f}, CI {c_lo:.2f}..{c_hi:.2f}; ratio CI {r_lo:.3f}..{r_hi:.3f})", ev_)
    if not (part_absent and score_absent):
        return _result(False, f"timing_channel: inconclusive - the intervals do not exclude a channel (score rho "
                              f"CI {c_lo:.2f}..{c_hi:.2f}, bound +-{TIMING_RHO_BOUND}; partition ratio CI "
                              f"{r_lo:.3f}..{r_hi:.3f}, bound 1+-{TIMING_RATIO_BOUND})", ev_)
    return _result(True, f"timing_channel: no channel within the bounds (score rho={rho:.2f}, 95% CI {c_lo:.2f}.."
                         f"{c_hi:.2f} inside +-{TIMING_RHO_BOUND}; partition ratio CI {r_lo:.3f}..{r_hi:.3f}; power "
                         f"{ev_['power_to_detect_rho_bound']:.2f} to detect |rho|={TIMING_RHO_BOUND}; {len(rows)} "
                         "episodes, one setting)", ev_)


# ====================================================================== partition awareness
def control_partition_awareness(reg=None) -> dict:
    """Everything a participant receives at the start of validation, evaluation and sealed episodes (agents, tools,
    observation, events), searched for identifying values and compared structurally."""
    from ...research.v5 import evaluator as E
    from ...research.v5.episodes import lever_entries
    reg = reg or _reg()
    f = _families(reg, 1)[0]
    ef = sorted(lever_entries(reg, "evaluation"))
    sealed = tempfile.mkdtemp(prefix="crucible-rt-sealed-")
    ev = _evaluator(sealed)
    seen: dict[str, list] = {}
    secrets: dict[str, list] = {}
    try:
        pr = ev.preregister({"control": "partition_awareness"})
        bare = f.split(":")[-1]
        items = [("validation", ev.open_episode(f, 1_000_601, purpose="dev"), [str(1_000_601), bare]),
                 ("validation", ev.open_episode(f, 1_000_602, purpose="dev"), [str(1_000_602), bare])]
        if ef:
            items.append(("evaluation", ev.open_episode(ef[0], 1_000_603, purpose="confirmation", prereg=pr),
                          [str(1_000_603), ef[0].split(":")[-1]]))
        ev.must("declare_sealed", "rt-aware", [f], 2, 1)
        for x in ev.must("open_sealed", "rt-aware", ack=E.SEALED_ACK, purpose="red team awareness",
                         prereg=pr)["tickets"]:
            items.append(("sealed", x["ticket"], [x["item"], "rt-aware", bare]))
        for part, t, sec in items:
            env = ev.participant_env(t)
            a = env.agents()[0]
            obs = env.observe(a)
            tools = env.tools(a)
            evs, _ = env.observe_since(a, 0)
            seen.setdefault(part, []).append({"agents": env.agents(), "obs": obs, "tools": tools, "events": evs})
            secrets.setdefault(part, []).extend(sec + [t])
        status = ev.must("sealed_status")
    finally:
        ev.close()
        shutil.rmtree(sealed, ignore_errors=True)

    def keys(x):
        """The structure a participant sees: dict keys recursively, lists and scalars by type only."""
        if isinstance(x, dict):
            return {k: keys(v) for k, v in sorted(x.items())}
        return "list" if isinstance(x, list) else type(x).__name__
    blobs = {p: [json.dumps(v, default=str, sort_keys=True) for v in vs] for p, vs in seen.items()}
    # identifying values (seeds, sealed aliases and set names, tickets, family names) anywhere in what was received
    hits = [{"partition": p, "value": s_[:40]} for p, bs in blobs.items() for b in bs for s_ in secrets[p]
            if s_ and s_ in b]
    # partition words as JSON tokens: telling only if they appear in one partition's episodes and not another's
    words = ('"sealed"', '"validation"', '"evaluation"', '"partition"', '"confirmation"', '"holdout"', '"test"')
    present = {w: sorted(p for p, bs in blobs.items() if any(w in b for b in bs)) for w in words}
    differential = {w: ps for w, ps in present.items() if ps and len(ps) < len(blobs)}
    same = keys(seen["validation"][0]["obs"]) == keys(seen["sealed"][0]["obs"]) and \
        sorted(t["name"] for t in seen["validation"][0]["tools"]) == sorted(t["name"] for t in seen["sealed"][0]["tools"])
    evd = {"episodes": {p: len(b) for p, b in seen.items()}, "identifying_values_seen": hits[:10],
           "partition_words_by_partition": differential, "same_structure_validation_vs_sealed": same,
           "sealed_status": status.get("sets")}
    if hits or differential or not same:
        return _result(False, f"partition_awareness: a participant can tell episodes apart: values {hits[:3]}, words "
                              f"{differential}, same structure {same}", evd)
    return _result(True, "partition_awareness: no seed, alias, ticket, family name or partition word reached the "
                         "participant, and validation and sealed episodes of one family look alike", evd)


# ====================================================================== generator calibration shortcut
RULES = {"anti_default_25_75": lambda d: np.where(d < 0.5, 0.75, 0.25), "mirror": lambda d: 1.0 - d,
         "centre": lambda d: np.full_like(d, 0.5), "all_max": lambda d: np.ones_like(d),
         "all_min": lambda d: np.zeros_like(d),
         "step_away_035": lambda d: np.clip(np.where(d < 0.5, d + 0.35, d - 0.35), 0.0, 1.0)}
SHORTCUT_WORLDS = 40


def control_calibration_shortcut(reg=None) -> dict:
    """Generated worlds exactly as the sampler builds them (`compose` with the family stream of each seed), from
    uniform draws of the grammar; each rule's share of the best constant setting's improvement over the defaults,
    on the noise-free mechanism (`compose.schedule_ledgers`, the world's own engine)."""
    from ...rapid.engines import build
    from ...rapid.families import family_rng
    from ...research.v5 import compose as C, curriculum as CU
    rng = np.random.default_rng(20_261_126)
    rows = []
    for _ in range(SHORTCUT_WORLDS):
        s = CU.random_structure(rng)
        name = C.structure_id(s)
        for seed in (0, 1):
            cfg = C.compose(family_rng(name, seed), s, {})
            eng = build(cfg)
            pub = eng.public_levers()
            lo = np.array([x["min"] for x in pub])
            hi = np.array([x["max"] for x in pub])
            span = np.where(hi > lo, hi - lo, 1.0)
            d0 = np.array([(x.get("default", x["min"]) - x["min"]) / max(x["max"] - x["min"], 1e-12) for x in pub])
            F = C._grid(len(pub))
            cands = [d0] + [fn(d0) for fn in RULES.values()]
            X = np.vstack([F] + [c[None, :] for c in cands])
            rev, cost = C.schedule_ledgers(eng, eng.init_state(np.random.default_rng(0)),
                                           np.repeat((lo + X * span)[:, None, :], eng.horizon, axis=1))
            net = rev - cost
            best, nd = float(net[: len(F)].max()), float(net[len(F)])
            if best - nd <= 1e-9:
                continue
            rows.append({"calibrated": bool(cfg.get("calibrated")),
                         **{k: float((net[len(F) + 1 + j] - nd) / (best - nd)) for j, k in enumerate(RULES)}})
    stats_ = {}
    for k in RULES:
        v = np.array([r[k] for r in rows])
        vc = np.array([r[k] for r in rows if r["calibrated"]]) if any(r["calibrated"] for r in rows) else v
        stats_[k] = {"median": round(float(np.median(v)), 4), "median_calibrated": round(float(np.median(vc)), 4),
                     "share_of_worlds_at_least_0.75": round(float(np.mean(v >= 0.75)), 4)}
    worst = max(stats_, key=lambda k: max(stats_[k]["median"], stats_[k]["median_calibrated"]))
    wv = max(stats_[worst]["median"], stats_[worst]["median_calibrated"])
    evd = {"worlds": len(rows), "calibrated_worlds": sum(r["calibrated"] for r in rows), "rules": stats_,
           "worst_rule": worst, "worst_median_share": wv}
    if len(rows) < 20:
        return _result(False, f"calibration_shortcut: inconclusive - only {len(rows)} generated worlds", evd)
    if wv >= 0.5:
        return _result(False, f"calibration_shortcut: the fixed rule {worst} reaches a median {wv:.2f} of the best "
                              "constant improvement on generated worlds", evd)
    return _result(True, f"calibration_shortcut: no fixed rule of the defaults solves generated worlds (worst "
                         f"{worst}: median share {wv:.2f} over {len(rows)} worlds)", evd)


FUNCS = {"process_memory_stash": control_process_memory_stash,
         "files_outside_checkpoint": control_files_outside_checkpoint,
         "checkpoint_tampering": control_checkpoint_tampering, "evaluator_internals": control_evaluator_internals,
         "evaluator_internals_hosted": control_evaluator_internals_hosted,
         "hosted_state_outlives_the_process": control_hosted_state_outlives_the_process,
         "evaluator_survives_a_stalled_participant": control_evaluator_survives_a_stalled_participant,
         "interaction_budget": control_interaction_budget, "timing_channel": control_timing_channel,
         "partition_awareness": control_partition_awareness, "calibration_shortcut": control_calibration_shortcut}


def run_controls(names=CONTROLS, progress=None) -> dict:
    """Run the controls; a control that raises is reported as not passed with its error (never skipped)."""
    reg = _reg()
    out = {}
    for n in names:
        t = time.time()
        try:
            out[n] = FUNCS[n](reg)
        except Exception as e:  # noqa: BLE001 - a control that cannot run counts as not passed, with the reason
            out[n] = _result(False, f"{n}: error {type(e).__name__}: {str(e)[:300]}", {"error": repr(e)[:600]})
        out[n]["seconds"] = round(time.time() - t, 1)
        if progress:
            progress(n, out[n])
    return out
