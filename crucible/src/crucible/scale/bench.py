"""The Crucible 4.0 benchmark suite. Every number it writes was measured on the machine it ran on.

    python examples/bench_openworld.py --quick                # a few minutes
    python examples/bench_openworld.py                        # the complete table
    python examples/bench_openworld.py --only kernel,scaling  # any subset; merged into the existing file

Sections (each independently selectable):

``kernel``        kernel transactions per second: bank transfers, a mixed tool-call workload, reads; replay speed
``families``      simulated days per wall-clock (and per CPU) second for every world family discovered at run time
                  from `crucible.worldgen.families()` - idle, and with scripted agents acting; events per day
``storage``       events and SQLite bytes per simulated day, snapshot size and time, fork time, replay speed
``memory``        resident memory per world as a function of population size (one fresh process per world)
``scaling``       aggregate simulated days per second for 1, 2, 4, 8 parallel worlds through `WorldPool`
``rollout``       agent steps per second through `RolloutFarm` as workers are added
``browser``       actions per second in one real Chromium session (skipped, with the reason, if unavailable)
``gpu``           GPU utilisation and memory read from ``nvidia-smi``: other tenants' load (the suite uses no GPU)
``episode_cost``  CPU-seconds and peak memory of one experimental episode per family (30 simulated days in the
                  full run), and dollars if a price per CPU-hour is supplied
``llm``           LLM inference throughput: copied from ``--llm-file`` if given, otherwise "not measured in this run"

Every row carries its conditions: load average when it was measured, core count and the container's CPU quota,
worker count, world parameters, and the number of repetitions with their spread (median, min, max, standard
deviation, the raw samples). Wall-clock rates depend on what else runs in the container; rates per CPU-second are
reported next to them because they do not.

A family that cannot be built or that fails while running becomes an error row; it never aborts the run.
The result goes to ``docs/benchmarks/openworld.json`` (schema `SCHEMA`); the 3.0 files there are never touched.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterable

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from .pool import DAY, WorldPool, default_workers, ensure_registered, pid_budget, world_job  # noqa: E402
from .rollout import RolloutFarm  # noqa: E402

SCHEMA = "crucible-bench-openworld/1"
SECTIONS = ("kernel", "families", "storage", "memory", "scaling", "rollout", "browser", "gpu", "episode_cost", "llm")
SECTION_KEYS = ("status", "reason", "generated", "mode", "wall_s", "conditions", "params", "rows")
TOP_KEYS = ("schema", "generated", "mode", "argv", "machine", "caveat", "discovery", "run", "sections")
STANDARD_EPISODE_DAYS = 30
PROTECTED_OUTPUTS = ("bench.json", "internet.json")      # the recorded 3.0 results
QUICK_BASE_FAMILIES = ("tiny", "bazaar", "open_digital", "open_industry")
KERNEL_MODULES = ("identity", "bank", "admin")                 # the world the 3.0 kernel benchmark used
KERNEL_MODULES_MIXED = ("identity", "bank", "admin", "forum", "mail", "docs", "search")
CAVEAT = ("Measured on a shared machine. Wall-clock rates are pessimistic whenever other tenants use the "
          "container's CPU quota (see machine.cgroup_cpu_quota_cores and each section's "
          "conditions.cgroup_throttled_fraction and loadavg). Rates per CPU-second remove the time spent waiting "
          "for a core, but not the slowdown from sharing caches and hyper-threads with other tenants, so they are "
          "pessimistic too, only less. Nothing in this file is extrapolated.")


# ====================================================================== conditions
def loadavg() -> list[float]:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except OSError:
        return []


def cgroup_cpu() -> dict:
    """CPU accounting of this container's cgroup (v1 or v2): quota in cores, CPU seconds used by *all* its
    processes so far, and how many scheduler periods were throttled. Missing values are None."""
    out: dict[str, Any] = {"quota_cores": None, "usage_s": None, "nr_periods": None, "nr_throttled": None,
                           "throttled_s": None}

    def read(p):
        with open(p) as f:
            return f.read().strip()
    try:                                             # cgroup v1
        q, per = int(read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")), int(read("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
        out["quota_cores"] = round(q / per, 3) if q > 0 else None
        for line in read("/sys/fs/cgroup/cpu/cpu.stat").splitlines():
            k, _, v = line.partition(" ")
            if k in ("nr_periods", "nr_throttled"):
                out[k] = int(v)
            elif k == "throttled_time":
                out["throttled_s"] = int(v) / 1e9
        out["usage_s"] = int(read("/sys/fs/cgroup/cpuacct/cpuacct.usage")) / 1e9
        return out
    except (OSError, ValueError):
        pass
    try:                                             # cgroup v2
        q, per = read("/sys/fs/cgroup/cpu.max").split()[:2]
        out["quota_cores"] = None if q == "max" else round(int(q) / int(per), 3)
        for line in read("/sys/fs/cgroup/cpu.stat").splitlines():
            k, _, v = line.partition(" ")
            if k in ("nr_periods", "nr_throttled"):
                out[k] = int(v)
            elif k == "throttled_usec":
                out["throttled_s"] = int(v) / 1e6
            elif k == "usage_usec":
                out["usage_s"] = int(v) / 1e6
    except (OSError, ValueError):
        pass
    return out


def _cpu_now() -> float:
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


class Probe:
    """Conditions around one measurement: take one before, call `done()` after."""

    def __init__(self):
        self.t = time.perf_counter()
        self.load = loadavg()
        self.cg = cgroup_cpu()
        self.cpu = _cpu_now()

    def done(self, **extra) -> dict:
        wall = max(1e-9, time.perf_counter() - self.t)
        cg = cgroup_cpu()
        cpu = _cpu_now() - self.cpu
        used = thr = None
        if cg["usage_s"] is not None and self.cg["usage_s"] is not None:
            used = round((cg["usage_s"] - self.cg["usage_s"]) / wall, 2)
        if cg["nr_periods"] is not None and self.cg["nr_periods"] is not None and cg["nr_periods"] > self.cg["nr_periods"]:
            thr = round((cg["nr_throttled"] - self.cg["nr_throttled"]) / (cg["nr_periods"] - self.cg["nr_periods"]), 3)
        d = {"loadavg_start": self.load, "loadavg_end": loadavg(), "cpu_count": os.cpu_count(),
             "cgroup_cpu_quota_cores": cg["quota_cores"], "cgroup_cores_used_all_tenants": used,
             "cgroup_throttled_fraction": thr, "bench_cpu_s": round(cpu, 3), "bench_cores_used": round(cpu / wall, 3),
             "wall_s": round(wall, 3)}
        d.update(extra)
        return d


def stat(samples: Iterable[float], nd: int = 4) -> dict:
    """Repetition count with spread. ``median`` is the headline value."""
    xs = [float(x) for x in samples]
    if not xs:
        return {"n": 0, "median": None, "min": None, "max": None, "mean": None, "stdev": None, "samples": []}
    return {"n": len(xs), "median": round(statistics.median(xs), nd), "min": round(min(xs), nd),
            "max": round(max(xs), nd), "mean": round(statistics.fmean(xs), nd),
            "stdev": round(statistics.stdev(xs), nd) if len(xs) > 1 else 0.0, "samples": [round(x, nd) for x in xs]}


def machine_info() -> dict:
    mem = None
    try:
        with open("/proc/meminfo") as f:
            mem = round(int(f.readline().split()[1]) / 2**20, 1)
    except (OSError, ValueError, IndexError):
        pass
    return {"cpu_count": os.cpu_count(), "cgroup_cpu_quota_cores": cgroup_cpu()["quota_cores"],
            "memory_total_gb": mem, "pid_budget": pid_budget(), "python": platform.python_version(),
            "platform": platform.platform(), "loadavg": loadavg()}


# ====================================================================== world helpers (run inside workers)
class SweepPolicy:
    """Scripted reader: calls, in turn, every read-only tool that needs no arguments. Unlike `RandomPolicy`, its
    commands are valid, so a benchmark world also processes accepted commands. (Duck-typed `agents.base.Policy`.)"""
    name, kind, version = "sweep", "scripted", "1"

    def reset(self, actor, rng) -> None:
        self.actor, self.rng, self.i = actor, rng, 0

    def act(self, obs, tools):
        cands = [t["name"] for t in tools
                 if t["annotations"].get("readOnlyHint") and not t["inputSchema"].get("required")]
        if not cands:
            return None
        self.i += 1
        return {"tool": cands[(self.i - 1) % len(cands)], "args": {}}

    def observe_result(self, action, result) -> None:
        pass

    def usage(self) -> dict:
        return {}

    def state_dict(self) -> dict:
        return {"i": self.i}

    def load_state_dict(self, d: dict) -> None:
        self.i = int(d["i"])


def spec_facts(spec) -> dict:
    """Declared size of a world, read from its spec (no world needed)."""
    cfg = {m["name"]: m.get("config") or {} for m in spec.modules}
    civ = cfg.get("civ", {})
    return {"modules": len(spec.modules), "cost_model": spec.params.get("cost_model", "measured"),
            "consumers": int(cfg.get("consumers", {}).get("n", 0) or 0),
            "shoppers": int(cfg.get("shoppers", {}).get("n", 0) or 0),
            "npc_workers": int(civ.get("workers", 0) or 0), "firms": len(civ.get("firms", []) or []),
            "rigs": len(cfg.get("rig", {}).get("rigs", []) or [])}


def world_facts(world) -> dict:
    """Summariser: actual size of a finished world."""
    kinds: dict[str, int] = {}
    for _, p in world.state.scan("principals"):
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
    return {"principals": sum(kinds.values()), "principal_kinds": kinds,
            "table_rows": sum(len(t) for t in world.state.tables.values()),
            "array_bytes": int(sum(a.nbytes for a in world.state.arrays.values())),
            "snapshot_bytes": len(world.snapshot_blob()), **spec_facts(world.spec)}


def world_costs(world, reps: int = 3) -> dict:
    """Summariser: snapshot size and time, fork time, replay speed of a finished world (with digest checks)."""
    from ..core.kernel import World
    world.flush()
    out = world_facts(world)
    digest = world.digest()
    ms = []
    blob = b""
    for _ in range(reps):
        t = time.perf_counter()
        blob = world.snapshot_blob()
        ms.append((time.perf_counter() - t) * 1e3)
    out["snapshot_bytes"], out["snapshot_ms"] = len(blob), ms
    fork_ms, equal = [], True
    for i in range(reps):
        t = time.perf_counter()
        f = world.fork(f"bench-fork-{i}")
        fork_ms.append((time.perf_counter() - t) * 1e3)
        equal = equal and f.digest() == digest
        f.close()
    out["fork_ms"], out["fork_digest_equal"] = fork_ms, equal
    t = time.perf_counter()
    if world.store is not None:
        events = world.store.read_events(world.world_id)
        out["replay_source"] = "sqlite"
    else:
        events = [e.to_dict() for e in world.events_range(0)]
        out["replay_source"] = "memory"
    read_s = time.perf_counter() - t
    out["replay_events"] = len(events)
    if events and events[0]["seq"] == 1:
        t = time.perf_counter()
        c = time.process_time()
        w2 = World.replay(world.spec, events)
        replay_s, replay_cpu = time.perf_counter() - t, time.process_time() - c
        out.update(read_events_per_s=len(events) / max(read_s, 1e-9), replay_events_per_s=len(events) / max(replay_s, 1e-9),
                   replay_events_per_cpu_s=len(events) / max(replay_cpu, 1e-9),
                   replay_digest_equal=w2.digest() == digest)
        w2.close()
    else:
        out.update(read_events_per_s=None, replay_events_per_s=None, replay_events_per_cpu_s=None,
                   replay_digest_equal=None)
    return out


def family_group(name: str) -> str:
    if name.startswith("mech_"):
        return "mech"
    if name.startswith("ow_"):
        return "ow"
    if name.startswith("open_"):
        return "open"
    if name.startswith("engineering_lab"):
        return "lab"
    return "legacy"


def discover(imports: Iterable[str] = ()) -> dict:
    """World families present right now (after run-time discovery of newly added packages)."""
    reg = ensure_registered(imports)
    from ..worldgen import FAMILIES, families
    names = families()
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(family_group(n), []).append(n)
    return {"families": names, "groups": groups, "splits": {n: FAMILIES[n].split for n in names},
            "imported": reg["imported"], "import_errors": reg["errors"]}


def select_families(disc: dict, mode: str, wanted: Iterable[str] | None = None,
                    max_per_group: int | None = None) -> tuple[list[str], dict[str, str]]:
    """Families to run and, for the rest, why not. Full mode: everything. Quick mode: a small base set of the 3.0
    families plus the ``mech_*`` and ``ow_*`` families, at most `max_per_group` of each group; under a cap the
    sample is spread over the sub-groups (``mech_abm_*``, ``mech_scm_*``, ...), one plain variant of each first,
    then one ``_tools`` variant of each."""
    names = list(disc["families"])
    if wanted:
        wanted = list(wanted)
        if wanted == ["all"]:
            return names, {}
        return wanted, {n: "not requested (--families)" for n in names if n not in wanted}
    if mode != "quick":
        return names, {}
    chosen, why = [], {}
    for g in ("mech", "ow"):
        members = [n for n in names if family_group(n) == g]
        keep = members
        if max_per_group is not None and len(members) > max_per_group:
            buckets: dict[tuple, list[str]] = {}       # (is a _tools variant, sub-group) -> names
            for n in members:
                buckets.setdefault((n.endswith("_tools"), n.split("_")[1] if n.count("_") else n), []).append(n)
            keep = []
            while len(keep) < max_per_group:
                for k in sorted(buckets):
                    if buckets[k] and len(keep) < max_per_group:
                        keep.append(buckets[k].pop(0))
        chosen += keep
        for n in members:
            if n not in keep:
                why[n] = f"quick mode runs at most {max_per_group} {g}_* families (--max-per-group); run the full suite"
    for n in names:
        if family_group(n) in ("mech", "ow"):
            continue
        if n in QUICK_BASE_FAMILIES:
            chosen.append(n)
        else:
            why[n] = "not part of the quick set of 3.0 families; run the full suite"
    return sorted(chosen), why


def _as_spec(f: str | dict, seed: int) -> dict:
    s = {"family": f} if isinstance(f, str) else dict(f)
    s.setdefault("seed", seed)
    return s


def agent_mix(module_names: Iterable[str]) -> list[tuple]:
    """The standard scripted agents of a benchmark episode: two random actors and a reader everywhere, plus a
    scripted entrepreneur (`Builder`: writes, deploys and operates a real service) where the family has a market."""
    mods = set(module_names)
    mix = [("bench-r1", "crucible.agents.scripted:RandomPolicy", {}),
           ("bench-r2", "crucible.agents.scripted:RandomPolicy", {}),
           ("bench-s1", "crucible.scale.bench:SweepPolicy", {})]
    if {"market", "cloud", "code", "business", "forum", "ads", "consumers"} <= mods:
        mix.append(("bench-b1", "crucible.agents.scripted:Builder", {}))
    return mix


# ====================================================================== kernel
def bench_kernel(n_actors: int = 200, n_cmds: int = 20_000, reps: int = 3, seed: int = 0) -> dict:
    """Kernel throughput in one process. In the world the 3.0 benchmark used (identity, bank, admin): fully
    transactional `bank.transfer` commands, recorded read commands, and replay from genesis with a digest check.
    In a wider world (plus forum, mail, docs, search): a mixed workload of nine tools."""
    ensure_registered()
    import random

    from ..core.kernel import World, WorldSpec
    acc = {k: {"wall": [], "cpu": [], "ok": []} for k in ("transfer", "mixed", "read")}
    replay, replay_cpu, events_per_cmd, mix_counts, mix_ok = [], [], [], {}, {}
    invariants_ok = replay_equal = True
    n_read = max(1, n_cmds // 4)

    def make_world(mods, s):
        spec = WorldSpec("bench_kernel", s, [{"name": m, "config": {}} for m in mods])
        w = World.create(spec)
        for i in range(n_actors):
            w.system_call("admin.join", {"handle": f"a{i}", "endowment": 10_000_000})
        return spec, w, [w.modules["bank"].primary_account(w.state, f"p:a{i}") for i in range(n_actors)]

    for rep in range(reps):
        rng = random.Random(seed * 1000 + rep)

        def run(w, make, n, key):
            pairs = [(rng.randrange(n_actors), rng.randrange(n_actors - 1)) for _ in range(n)]
            acts = [(i, make(i, j if j < i else j + 1, k)) for k, (i, j) in enumerate(pairs)]
            seq0 = w.seq
            t, c = time.perf_counter(), time.process_time()
            oks = [w.execute(f"p:a{i}", a).ok for i, a in acts]
            dt, dc = time.perf_counter() - t, time.process_time() - c
            acc[key]["wall"].append(n / dt)
            acc[key]["cpu"].append(n / max(dc, 1e-9))
            acc[key]["ok"].append(sum(oks) / n)
            return oks, acts, w.seq - seq0

        spec, w, accts = make_world(KERNEL_MODULES, seed + rep)

        def transfer(i, j, k=0, accts=accts):
            return {"tool": "bank.transfer", "args": {"from_account": accts[i], "to": f"a{j}", "amount": 100}}

        _, _, n_ev = run(w, transfer, n_cmds, "transfer")
        events_per_cmd.append(n_ev / n_cmds)
        run(w, lambda i, j, k: {"tool": "bank.accounts", "args": {}}, n_read, "read")
        evs = [e.to_dict() for e in w.log]
        t, c = time.perf_counter(), time.process_time()
        w2 = World.replay(spec, evs)
        replay.append(len(evs) / (time.perf_counter() - t))
        replay_cpu.append(len(evs) / max(time.process_time() - c, 1e-9))
        replay_equal = replay_equal and w2.digest() == w.digest()
        invariants_ok = invariants_ok and w.check_invariants() == []
        w2.close()
        w.close()

        _, w, accts = make_world(KERNEL_MODULES_MIXED, seed + rep)

        def mixed(i, j, k, accts=accts):
            r = rng.random()
            if r < 0.30:
                return {"tool": "bank.transfer", "args": {"from_account": accts[i], "to": f"a{j}", "amount": 100}}
            if r < 0.45:
                return {"tool": "bank.accounts", "args": {}}
            if r < 0.55:
                return {"tool": "bank.balance", "args": {"account": accts[i]}}
            if r < 0.65:
                return {"tool": "world.clock", "args": {}}
            if r < 0.75:
                return {"tool": "forum.post", "args": {"board": "general", "title": f"note {k}",
                                                       "body": f"benchmark post {k} by a{i}"}}
            if r < 0.85:
                return {"tool": "forum.list", "args": {"board": "general"}}
            if r < 0.90:
                return {"tool": "mail.send", "args": {"to": [f"a{j}"], "subject": f"s{k}", "body": "benchmark mail"}}
            if r < 0.95:
                return {"tool": "mail.inbox", "args": {}}
            return {"tool": "docs.create", "args": {"title": f"doc {k}", "body": "benchmark document " * 4}}

        oks, acts, _ = run(w, mixed, n_cmds, "mixed")
        for (_, a), ok in zip(acts, oks):
            mix_counts[a["tool"]] = mix_counts.get(a["tool"], 0) + 1
            mix_ok[a["tool"]] = mix_ok.get(a["tool"], 0) + int(ok)
        invariants_ok = invariants_ok and w.check_invariants() == []
        w.close()
    base = {"modules": list(KERNEL_MODULES), "actors": n_actors}
    wide = {"modules": list(KERNEL_MODULES_MIXED), "actors": n_actors}
    rows = []
    for key, label, n, world in (("transfer", "bank.transfer (journaled transaction)", n_cmds, base),
                                 ("mixed", "mixed tool calls (9 tools)", n_cmds, wide),
                                 ("read", "bank.accounts (recorded read)", n_read, base)):
        rows.append({"measure": label, "unit": "commands/s", "per_wall_s": stat(acc[key]["wall"], 1),
                     "per_cpu_s": stat(acc[key]["cpu"], 1), "ok_fraction": round(min(acc[key]["ok"]), 4),
                     "commands_per_rep": n, "reps": reps, "workers": 1, "world": world, "loadavg": loadavg()})
    rows[0]["events_per_command"] = round(statistics.fmean(events_per_cmd), 2)
    rows[1]["tool_mix"] = {t: {"calls": mix_counts[t], "ok": mix_ok[t]} for t in sorted(mix_counts)}
    rows.append({"measure": "replay from genesis (digest checked)", "unit": "events/s", "per_wall_s": stat(replay, 1),
                 "per_cpu_s": stat(replay_cpu, 1), "ok_fraction": 1.0 if replay_equal else 0.0, "reps": reps,
                 "workers": 1, "world": base, "loadavg": loadavg(), "invariants_ok": invariants_ok})
    return {"params": {"n_actors": n_actors, "n_cmds": n_cmds, "reps": reps, "seed": seed}, "rows": rows}


# ====================================================================== families (rates, episode cost)
def bench_families(families: Iterable[str | dict], idle_days: float = 1.0, agent_days: float = 2.0, reps: int = 2,
                   workers: int = 4, seed: int = 1, dt_s: int = 3600, timeout_s: float = 1800.0,
                   log: Callable[[str], None] | None = None) -> dict:
    """Per family: simulated days per second idle (`reps` runs of `idle_days`) and with the standard scripted
    agents acting every `dt_s` simulated seconds (one run of `agent_days`: the benchmark episode). Each run is a
    `WorldPool` job, so a family that breaks is an error row and the rest still run. ``idle_days=0`` or
    ``agent_days=0`` skips that part."""
    ensure_registered()
    from ..worldgen import FAMILIES, generate
    rows: dict[str, dict] = {}
    mixes: dict[str, list[tuple]] = {}
    jobs = []
    for f in families:
        spec = _as_spec(f, seed)
        name = spec["family"]
        row = {"family": name, "group": family_group(name), "split": None, "status": "ok", "error": None,
               "world": {"seed": spec["seed"], "overrides": spec.get("overrides") or {}}, "workers": workers,
               "reps": reps, "loadavg": loadavg(), "idle": None, "agents": None}
        rows[name] = row
        try:
            ws = generate({k: v for k, v in spec.items() if k != "seed"}, spec["seed"])
            row["split"] = FAMILIES[name].split
            row["world"].update(spec_facts(ws))
            mix = mixes[name] = agent_mix(m["name"] for m in ws.modules)
        except Exception as e:  # noqa: BLE001 - the family does not build: report it, keep going
            row["status"], row["error"] = "error", f"build failed: {type(e).__name__}: {str(e)[:300]}"
            continue
        if idle_days > 0:
            for rep in range(reps):
                jobs.append(world_job(spec, days=idle_days, tag=(name, "idle", rep), store=False,
                                      summarise="crucible.scale.bench:world_facts", timeout_s=timeout_s))
        if agent_days > 0:
            jobs.append(world_job(spec, days=agent_days, agents=mix, dt_s=dt_s, tag=(name, "agents", 0), store=False,
                                  summarise="crucible.scale.bench:world_facts", timeout_s=timeout_s))
    got: dict[str, dict[str, list]] = {}
    start_s = None
    if jobs:
        with WorldPool(workers, max_jobs_per_worker=8) as pool:
            start_s = round(pool.wait_ready(), 2)
            for j in jobs:
                pool.submit(j)
            for res in pool.as_completed():
                name, kind, _ = res["tag"]
                got.setdefault(name, {}).setdefault(kind, []).append(res)
                rows[name]["loadavg"] = loadavg()
                if log:
                    log(f"  {name} {kind}: " + (f"{res['run_wall_s']:.2f}s wall, {res['cpu_s']:.2f}s cpu"
                                              if res["ok"] else f"FAILED {res['error']['message'][:120]}"))
    for name, row in rows.items():
        if row["status"] == "error":
            continue
        errs = []
        idle = [r for r in got.get(name, {}).get("idle", [])]
        ok = [r for r in idle if r["ok"]]
        errs += [f"idle: {r['error']['type']}: {r['error']['message'][:200]}" for r in idle if not r["ok"]]
        if ok:
            row["world"].update({k: ok[0]["summary"][k] for k in ("principals", "table_rows", "array_bytes")})
            row["idle"] = {
                "days": idle_days, "reps": len(ok),
                "sim_days_per_wall_s": stat(r["sim_days_run"] / max(r["run_wall_s"], 1e-9) for r in ok),
                "sim_days_per_cpu_s": stat(r["sim_days_run"] / max(r["run_cpu_s"], 1e-9) for r in ok),
                "events_per_sim_day": stat((r["counts"]["events"] / r["sim_days_run"] for r in ok), 1),
                "process_runs_per_sim_day": round(statistics.fmean(r["counts"]["process_runs"] / r["sim_days_run"]
                                                                   for r in ok), 1),
                "process_failures": max(r["counts"]["process_failures"] for r in ok),
                "create_s": stat(r["setup_wall_s"] for r in ok),
                "peak_rss_mb": stat((r["peak_rss_mb"] for r in ok), 1),
                "same_digest_across_reps": len({r["digest"] for r in ok}) == 1 if len(ok) > 1 else None}
        ag = got.get(name, {}).get("agents", [])
        errs += [f"agents: {r['error']['type']}: {r['error']['message'][:200]}" for r in ag if not r["ok"]]
        if ag and ag[0]["ok"]:
            r = ag[0]
            c = r["counts"]
            row["agents"] = {
                "days": agent_days, "reps": 1, "dt_s": dt_s,
                "mix": [a[1].split(":")[1] for a in mixes[name]],
                "agent_steps": c["agent_steps"], "commands": c["commands"], "rejected": c["rejected"],
                "sim_days_per_wall_s": stat([r["sim_days_run"] / max(r["run_wall_s"], 1e-9)]),
                "sim_days_per_cpu_s": stat([r["sim_days_run"] / max(r["run_cpu_s"], 1e-9)]),
                "events_per_sim_day": stat([c["events"] / r["sim_days_run"]], 1),
                "process_failures": c["process_failures"], "wall_s": r["wall_s"], "cpu_s": r["cpu_s"],
                "cpu_children_s": r["cpu_children_s"], "peak_rss_mb": r["peak_rss_mb"],
                "peak_rss_scope": r["peak_rss_scope"]}
        if errs:
            row["error"] = "; ".join(errs)
            row["status"] = "error" if row["idle"] is None and row["agents"] is None else "partial"
    return {"params": {"idle_days": idle_days, "agent_days": agent_days, "reps": reps, "workers": workers,
                       "seed": seed, "dt_s": dt_s, "pool_start_s": start_s}, "rows": list(rows.values())}


def episode_cost_rows(family_rows: list[dict], price_per_cpu_hour: float | None = None) -> list[dict]:
    """Cost of one experimental episode per family, from the agent runs of `bench_families`: CPU-seconds (world
    process plus the sandboxed programs it ran), peak resident memory, wall time; dollars only from a supplied
    price. The policy's own compute (a learner, an LLM) is not part of the world's cost and is not included."""
    out = []
    for r in family_rows:
        a = r.get("agents")
        row = {"family": r["family"], "group": r["group"], "status": "ok" if a else "error",
               "error": None if a else (r.get("error") or "no agent run"), "episode_days": a["days"] if a else None,
               "standard_30_day_episode": bool(a) and a["days"] == STANDARD_EPISODE_DAYS, "cpu_s": None,
               "peak_rss_mb": None, "wall_s": None, "agent_steps": None, "usd": None,
               "price_per_cpu_hour": price_per_cpu_hour, "reps": 1 if a else 0, "workers": r["workers"],
               "world": r["world"], "loadavg": r["loadavg"], "agents": a["mix"] if a else None}
        if a:
            cpu = a["cpu_s"] + a["cpu_children_s"]
            row.update(cpu_s=round(cpu, 3), peak_rss_mb=a["peak_rss_mb"], wall_s=a["wall_s"],
                       agent_steps=a["agent_steps"],
                       usd=round(cpu / 3600.0 * price_per_cpu_hour, 6) if price_per_cpu_hour is not None else None)
        out.append(row)
    return out


# ====================================================================== storage, snapshot, fork, replay
def bench_storage(families: Iterable[str | dict], days: int = 2, workers: int = 2, seed: int = 1,
                  store_dir: str | None = None, timeout_s: float = 1800.0) -> dict:
    """Events and SQLite bytes per simulated day (measured after every day), snapshot size and time, fork time and
    replay speed of the finished world, per family. Fork and replay digests are compared with the original."""
    ensure_registered()
    own = store_dir is None
    store_dir = store_dir or tempfile.mkdtemp(prefix="crucible-bench-store-")
    rows = []
    specs = [_as_spec(f, seed) for f in families]
    try:
        with WorldPool(workers, store_dir=store_dir, max_jobs_per_worker=8) as pool:
            res = pool.map([world_job(s, days=days, tag=s["family"], name=f"storage-{i:03d}-{s['family']}",
                                      summarise="crucible.scale.bench:world_costs", timeout_s=timeout_s)
                            for i, s in enumerate(specs)])
        for s, r in zip(specs, res):
            row = {"family": s["family"], "group": family_group(s["family"]), "status": "ok" if r["ok"] else "error",
                   "error": None if r["ok"] else f"{r['error']['type']}: {r['error']['message'][:300]}",
                   "world": {"seed": s["seed"], "overrides": s.get("overrides") or {}}, "workers": workers,
                   "days": days, "reps": 1, "loadavg": loadavg()}
            if r["ok"]:
                c = r["summary"]
                tl = r["timeline"]
                cum = [{"day": round(x["t"] / DAY, 3), "events": x["seq"], "store_bytes": x["store_bytes"]} for x in tl]
                d_ev = [b["events"] - a["events"] for a, b in zip(cum, cum[1:])]
                d_by = [b["store_bytes"] - a["store_bytes"] for a, b in zip(cum, cum[1:])]
                row["world"].update({k: c[k] for k in ("principals", "table_rows", "array_bytes", "modules",
                                                       "consumers", "shoppers", "cost_model")})
                row.update(
                    cumulative=cum, events_total=r["seq"], store_bytes_total=r["store_bytes"],
                    events_per_sim_day=stat(d_ev or [r["seq"] / max(r["sim_days_run"], 1e-9)], 1),
                    store_bytes_per_sim_day=stat(d_by or [r["store_bytes"] / max(r["sim_days_run"], 1e-9)], 0),
                    bytes_per_event=round(r["store_bytes"] / max(r["seq"], 1), 1),
                    snapshot_bytes=c["snapshot_bytes"], snapshot_ms=stat(c["snapshot_ms"], 2),
                    snapshot_file_write_ms=r.get("snapshot_write_ms"),
                    fork_ms=stat(c["fork_ms"], 2), fork_digest_equal=c["fork_digest_equal"],
                    replay_events=c["replay_events"], replay_source=c["replay_source"],
                    replay_events_per_s=None if c["replay_events_per_s"] is None else round(c["replay_events_per_s"], 1),
                    replay_events_per_cpu_s=None if c["replay_events_per_cpu_s"] is None
                    else round(c["replay_events_per_cpu_s"], 1),
                    read_events_per_s=None if c["read_events_per_s"] is None else round(c["read_events_per_s"], 1),
                    replay_digest_equal=c["replay_digest_equal"])
            rows.append(row)
    finally:
        if own:
            shutil.rmtree(store_dir, ignore_errors=True)
    return {"params": {"days": days, "workers": workers, "seed": seed, "store": "sqlite (WAL, synchronous=NORMAL)"},
            "rows": rows}


# ====================================================================== memory vs population
def bench_memory(points: Iterable[dict], days: float = 1.0, workers: int = 2, seed: int = 1,
                 timeout_s: float = 1800.0) -> dict:
    """Resident memory of one world as its population grows. Every world runs in a fresh process (workers are
    recycled after one job), so ``world_rss_mb`` = peak resident memory of the job minus the resident memory of
    the process before the world was built. `points` are world specs (``{"family", "overrides"}``)."""
    specs = [_as_spec(p, seed) for p in points]
    with WorldPool(workers, max_jobs_per_worker=1) as pool:
        res = pool.map([world_job(s, days=days, tag=i, store=False, summarise="crucible.scale.bench:world_facts",
                                  timeout_s=timeout_s) for i, s in enumerate(specs)])
    rows = []
    for s, r in zip(specs, res):
        row = {"family": s["family"], "group": family_group(s["family"]), "status": "ok" if r["ok"] else "error",
               "error": None if r["ok"] else f"{r['error']['type']}: {r['error']['message'][:300]}",
               "world": {"seed": s["seed"], "overrides": s.get("overrides") or {}}, "workers": workers, "days": days,
               "reps": 1, "loadavg": loadavg()}
        if r["ok"]:
            f = r["summary"]
            row["world"].update(f)
            row.update(population=f["consumers"] + f["shoppers"] + f["npc_workers"], principals=f["principals"],
                       rss_before_world_mb=r["rss_start_mb"], rss_end_mb=r["rss_end_mb"],
                       peak_rss_mb=r["peak_rss_mb"], world_rss_mb=round(r["peak_rss_mb"] - r["rss_start_mb"], 1),
                       state_array_mb=round(f["array_bytes"] / 2**20, 2),
                       snapshot_mb=round(f["snapshot_bytes"] / 2**20, 3),
                       sim_days_per_wall_s=round(r["sim_days_run"] / max(r["run_wall_s"], 1e-9), 4))
        rows.append(row)
    return {"params": {"days": days, "workers": workers, "seed": seed, "fresh_process_per_world": True}, "rows": rows}


# ====================================================================== scaling through WorldPool
def bench_scaling(counts: Iterable[int] = (1, 2, 4, 8), spec: dict | str = "tiny", days: float = 1.0, reps: int = 1,
                  seed: int = 0, timeout_s: float = 1800.0) -> dict:
    """Aggregate simulated days per wall second when n independent worlds run at once. One pool with max(counts)
    workers is started once (its start-up time is reported separately and is not part of the rates); for each n,
    n jobs are submitted together and timed until all have returned."""
    counts = sorted(set(int(c) for c in counts))
    base = _as_spec(spec, seed)
    rows = []
    with WorldPool(max(counts), max_jobs_per_worker=None) as pool:
        start_s = pool.wait_ready()
        pool.map([world_job(dict(base, seed=seed + 1000 + i), days=min(days, 0.1), store=False)
                  for i in range(max(counts))])            # warm every worker (lazy imports, allocator)
        for n in counts:
            agg, per, cpu, failed = [], [], [], 0
            p = Probe()
            for rep in range(reps):
                jobs = [world_job(dict(base, seed=seed + i), days=days, store=False, timeout_s=timeout_s, tag=(n, rep, i))
                        for i in range(n)]
                t = time.perf_counter()
                res = pool.map(jobs)
                wall = time.perf_counter() - t
                ok = [r for r in res if r["ok"]]
                failed += len(res) - len(ok)
                agg.append(sum(r["sim_days_run"] for r in ok) / wall)
                per.extend(r["run_wall_s"] for r in ok)
                cpu.append(sum(r["cpu_s"] + r["cpu_children_s"] for r in ok))
            cond = p.done()
            rows.append({"parallel_worlds": n, "workers": pool.n_workers, "days_each": days, "reps": reps,
                         "aggregate_sim_days_per_wall_s": stat(agg), "per_world_run_wall_s": stat(per),
                         "cpu_s_per_batch": stat(cpu, 3), "failed_jobs": failed,
                         "status": "ok" if not failed else "error", "error": None if not failed else f"{failed} jobs failed",
                         "world": {"family": base["family"], "overrides": base.get("overrides") or {}},
                         "loadavg": cond["loadavg_end"], "cgroup_throttled_fraction": cond["cgroup_throttled_fraction"],
                         "cgroup_cores_used_all_tenants": cond["cgroup_cores_used_all_tenants"]})
    one = next((r["aggregate_sim_days_per_wall_s"]["median"] for r in rows if r["parallel_worlds"] == 1), None)
    for r in rows:
        m = r["aggregate_sim_days_per_wall_s"]["median"]
        r["speedup_vs_1"] = round(m / one, 3) if one and m else None
        r["efficiency"] = round(m / one / r["parallel_worlds"], 3) if one and m else None
    return {"params": {"counts": counts, "days": days, "reps": reps, "pool_start_s": round(start_s, 2),
                       "start_method": "spawn"}, "rows": rows}


# ====================================================================== rollout farm
def bench_rollout(worker_counts: Iterable[int] = (1, 4), tasks_per_worker: int = 4, steps: int = 16,
                  think_s: int = 1800, spec: dict | str = "tiny", seed: int = 0) -> dict:
    """Agent steps per wall second through `RolloutFarm`: every worker owns one persistent world, a random policy
    acts, each step advances the world by `think_s` simulated seconds plus the action's own duration. Start-up and
    one warm-up task per world are excluded."""
    base = _as_spec(spec, seed)
    rows = []
    for n in sorted(set(int(c) for c in worker_counts)):
        p = Probe()
        with RolloutFarm(n, policy_kwargs={"cls": "crucible.agents.scripted:RandomPolicy"},
                         world_kwargs={"think_s": think_s}, record_obs=False) as farm:
            start_s = farm.wait_ready()
            for k in range(n):
                farm.submit(dict(base, seed=seed + k, key=f"w{k}"), steps=2)
            warm = [r for r in farm]
            t = time.perf_counter()
            for i in range(n * tasks_per_worker):
                farm.submit(dict(base, seed=seed + i % n, key=f"w{i % n}"), steps=steps)
            got = [r for r in farm]
            wall = time.perf_counter() - t
            st = farm.stats()
        ok = [r for r in got if r.ok]
        n_steps = sum(r.steps for r in ok)
        cond = p.done()
        rows.append({"workers": n, "tasks": len(got), "failed": len(got) - len(ok) + sum(not r.ok for r in warm),
                     "steps": n_steps, "steps_per_task": steps, "reps": 1,
                     "steps_per_wall_s": stat([n_steps / wall], 2),
                     "sim_days_per_wall_s": stat([sum(r.sim_seconds for r in ok) / DAY / wall]),
                     "task_wall_s": stat(r.wall_s for r in ok), "farm_start_s": round(start_s, 2),
                     "accounting_exact": st["submitted"] == st["ok"] + st["failed"] and st["steps"] ==
                     n_steps + sum(r.steps for r in warm if r.ok),
                     "status": "ok" if len(ok) == len(got) else "error", "error": None,
                     "world": {"family": base["family"], "overrides": base.get("overrides") or {}, "think_s": think_s},
                     "loadavg": cond["loadavg_end"], "cgroup_throttled_fraction": cond["cgroup_throttled_fraction"]})
    one = next((r["steps_per_wall_s"]["median"] for r in rows if r["workers"] == 1), None)
    for r in rows:
        r["speedup_vs_1"] = round(r["steps_per_wall_s"]["median"] / one, 3) if one else None
    return {"params": {"worker_counts": [r["workers"] for r in rows], "tasks_per_worker": tasks_per_worker,
                       "steps": steps, "think_s": think_s, "policy": "RandomPolicy"}, "rows": rows}


# ====================================================================== browser (one Chromium, never more)
def chromium_running() -> list[int]:
    """Pids of Chromium processes visible in this container (any owner)."""
    out = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/comm") as f:
                comm = f.read().strip().lower()
        except OSError:
            continue
        if comm.startswith(("chrome", "chromium", "headless_shell")):
            out.append(int(name))
    return out


def bench_browser(n: int = 20, seed: int = 1, wait_s: float = 30.0) -> dict:
    """Actions per second in one real Chromium session browsing a generated site of an `open_digital` world,
    next to the cost of the same page through the API (`net.fetch`). Skipped, with the reason, when Playwright
    or its Chromium build is missing, when another Chromium is already running in this container (the thread
    limit allows one), or when the pid budget is too low."""
    def skipped(reason):
        return {"status": "skipped", "reason": reason, "params": {"n": n}, "rows": []}
    end = time.monotonic() + wait_s
    while chromium_running():
        if time.monotonic() > end:
            return skipped(f"another Chromium is running in this container (pids {chromium_running()[:5]}); "
                           f"waited {wait_s:.0f}s; only one may run at a time here")
        time.sleep(2.0)
    b = pid_budget()
    if b is not None and b["free"] < 400:
        return skipped(f"pid budget too low for a Chromium session ({b['current']}/{b['max']} threads in use)")
    try:
        from ..web.app import make_web_app
        from ..web.browser import BrowserSession, playwright_available, serve_in_thread
    except Exception as e:  # noqa: BLE001
        return skipped(f"web stack not importable: {type(e).__name__}: {e}")
    if not playwright_available():
        return skipped("Playwright or its Chromium build is not installed")
    ensure_registered()
    from ..core.kernel import World
    from ..worldgen import generate
    w = World.create(generate({"family": "open_digital", "overrides": {
        "n_consumers": 500, "module_config": {"webgen": {"sites": 2}}}}, seed))
    try:
        site = sorted(h for h, _ in w.state.scan("hosts") if w.modules["net"].web_dep(w.state, h)[1] is not None)[0]
        url = f"http://{site}.crucible/"
        w.system_call("admin.join", {"handle": "bench-web", "endowment": 100_000})
        api = []
        for _ in range(3):
            t = time.perf_counter()
            for _ in range(n):
                assert w.execute("p:bench-web", {"tool": "net.fetch", "args": {"url": url}}).ok
                w.advance_time(w.now + 20)       # stay below the 240 actions per simulated hour limit
            api.append(n / (time.perf_counter() - t))
        wapp, stop = serve_in_thread(make_web_app(w))
        session = BrowserSession(wapp, timeout_ms=30_000)
        try:
            t = time.perf_counter()
            first = session.act({"action": "goto", "url": url})
            first_ms = (time.perf_counter() - t) * 1e3
            if not first["ok"]:
                return {"status": "error", "reason": f"first page load failed: {first['error']}", "params": {"n": n},
                        "rows": []}
            goto, observe = [], []
            for _ in range(n):
                t = time.perf_counter()
                o = session.act({"action": "goto", "url": url})
                goto.append(time.perf_counter() - t)
                assert o["ok"], o["error"]
            for _ in range(n):
                t = time.perf_counter()
                session.act({"action": "observe"})
                observe.append(time.perf_counter() - t)
        finally:
            session.close()
            stop()
    finally:
        w.close()
    world = {"family": "open_digital", "overrides": {"n_consumers": 500, "module_config": {"webgen": {"sites": 2}}},
             "seed": seed, "site": site}
    g_ms, o_ms = stat((x * 1e3 for x in goto), 2), stat((x * 1e3 for x in observe), 2)
    rows = [
        {"measure": "Chromium goto (navigate + observe)", "unit": "actions/s", "actions_per_s": round(n / sum(goto), 2),
         "ms_per_action": g_ms, "first_load_ms": round(first_ms, 1), "reps": n, "workers": 1, "world": world,
         "loadavg": loadavg()},
        {"measure": "Chromium observe (read the current page)", "unit": "actions/s",
         "actions_per_s": round(n / sum(observe), 2), "ms_per_action": o_ms, "reps": n, "workers": 1, "world": world,
         "loadavg": loadavg()},
        {"measure": "API net.fetch of the same page", "unit": "actions/s", "actions_per_s": stat(api, 1)["median"],
         "per_s": stat(api, 1), "reps": len(api), "workers": 1, "world": world, "loadavg": loadavg()}]
    rows[0]["cost_vs_api"] = round((sum(goto) / n) * rows[2]["actions_per_s"], 1)
    return {"status": "ok", "reason": None, "params": {"n": n, "sessions": 1}, "rows": rows}


# ====================================================================== GPU (read-only) and LLM placeholder
def sample_gpu() -> dict:
    """One read-only `nvidia-smi` query. The suite does not use the GPU; this is other tenants' load."""
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
    except FileNotFoundError:
        return {"ok": False, "reason": "nvidia-smi not found"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "nvidia-smi timed out"}
    if p.returncode != 0:
        return {"ok": False, "reason": f"nvidia-smi exit {p.returncode}: {p.stderr.strip()[:200]}"}
    gpus = []
    for line in p.stdout.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        try:
            gpus.append({"index": int(f[0]), "name": f[1], "util_pct": float(f[2]), "mem_used_mb": float(f[3]),
                         "mem_total_mb": float(f[4])})
        except (ValueError, IndexError):
            continue
    return {"ok": True, "t": time.strftime("%H:%M:%S"), "loadavg": loadavg(), "gpus": gpus}


def gpu_section(samples: list[dict]) -> dict:
    good = [s for s in samples if s.get("ok")]
    if not good:
        return {"status": "skipped", "reason": (samples[-1].get("reason") if samples else "no samples"),
                "params": {"samples": len(samples)}, "rows": []}
    rows = []
    for idx in sorted({g["index"] for s in good for g in s["gpus"]}):
        gs = [g for s in good for g in s["gpus"] if g["index"] == idx]
        rows.append({"gpu": idx, "name": gs[0]["name"], "label": "other tenants' load (this suite does not use the GPU)",
                     "util_pct": stat((g["util_pct"] for g in gs), 1), "mem_used_mb": stat((g["mem_used_mb"] for g in gs), 0),
                     "mem_total_mb": gs[0]["mem_total_mb"], "reps": len(gs), "workers": 0,
                     "sampled_at": [s["t"] for s in good], "loadavg": good[-1]["loadavg"]})
    return {"status": "ok", "reason": None, "params": {"samples": len(good), "source": "nvidia-smi, read-only"},
            "rows": rows}


def llm_section(path: str | None) -> dict:
    """LLM inference throughput is measured elsewhere. With a file, its JSON content is carried over verbatim."""
    if not path:
        return {"status": "not_measured", "reason": "not measured in this run", "params": {"input_file": None},
                "rows": []}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        return {"status": "error", "reason": f"could not read {path}: {type(e).__name__}: {e}",
                "params": {"input_file": path}, "rows": []}
    rows = data if isinstance(data, list) else data.get("rows") if isinstance(data, dict) and isinstance(
        data.get("rows"), list) else [data]
    return {"status": "ok", "reason": f"copied from {path}; not measured by this script",
            "params": {"input_file": path}, "rows": [r if isinstance(r, dict) else {"value": r} for r in rows]}


# ====================================================================== the suite
def _memory_points(mode: str, fams: list[str]) -> list[dict]:
    pts = []
    if "bazaar" in fams:
        for n in ((1000, 10_000, 100_000) if mode == "quick" else (1000, 6000, 25_000, 100_000, 400_000)):
            pts.append({"family": "bazaar", "overrides": {"n_consumers": n}})
    if mode != "quick" and "open_mixed" in fams:
        for c, s in ((2000, 20_000), (8000, 80_000)):
            pts.append({"family": "open_mixed", "overrides": {"n_consumers": c, "n_shoppers": s}})
    if mode != "quick" and "tiny" in fams:
        for n in (50, 2000):
            pts.append({"family": "tiny", "overrides": {"workers": n, "n_consumers": 2000}})
    return pts


def run_suite(mode: str = "quick", only: Iterable[str] | None = None, skip: Iterable[str] = (),
              families: Iterable[str] | None = None, workers: int | None = None, out: str | None = None,
              price_per_cpu_hour: float | None = None, llm_file: str | None = None, seed: int = 1,
              max_per_group: int | None = None, merge: bool = True, imports: Iterable[str] = (),
              argv: list[str] | None = None, log: Callable[[str], None] | None = print,
              section_kwargs: dict[str, dict] | None = None) -> dict:
    """Run the selected sections and return the result document (also written to `out` if given).
    `section_kwargs` overrides the arguments a section's benchmark function is called with, e.g.
    ``{"kernel": {"n_cmds": 500}}``."""
    if mode not in ("quick", "full"):
        raise ValueError("mode is 'quick' or 'full'")
    if out and os.path.basename(out) in PROTECTED_OUTPUTS:
        raise ValueError(f"{out}: the recorded 3.0 results are never overwritten; choose another file")
    selected = [s for s in (list(only) if only else SECTIONS) if s not in set(skip)]
    bad = sorted(set(selected) - set(SECTIONS))
    if bad:
        raise ValueError(f"unknown sections {bad}; known: {list(SECTIONS)}")
    quick = mode == "quick"
    say = log or (lambda s: None)
    kw = {k: dict(v) for k, v in (section_kwargs or {}).items()}

    def args(name: str, **defaults) -> dict:
        return {**defaults, **kw.get(name, {})}
    workers = workers or min(4, default_workers())
    if max_per_group is None and quick:
        max_per_group = 6
    t_run = time.perf_counter()
    whole = Probe()
    disc = discover(imports)
    chosen, why_not = select_families(disc, mode, families, max_per_group)
    say(f"crucible openworld benchmark ({mode}); load {loadavg()}, {os.cpu_count()} cpus, "
        f"cgroup quota {cgroup_cpu()['quota_cores']} cores; {len(disc['families'])} families, running {len(chosen)}")
    if disc["import_errors"]:
        say(f"  import errors during discovery: {disc['import_errors']}")
    gpu_samples = [sample_gpu()]
    sections: dict[str, dict] = {}
    fam_rows: list[dict] | None = None
    fam_cond: dict | None = None

    def not_run_rows() -> list[dict]:
        return [{"family": n, "group": family_group(n), "status": "not_run", "error": r} for n, r in sorted(why_not.items())]

    def measure(name: str, fn: Callable[[], dict]) -> dict:
        say(f"[{name}] ...")
        p = Probe()
        try:
            r = fn()
        except Exception as e:  # noqa: BLE001 - one section failing must not lose the others
            import traceback
            r = {"status": "error", "reason": f"{type(e).__name__}: {str(e)[:500]}", "params": {}, "rows": [],
                 "traceback": traceback.format_exc()[-3000:]}
        cond = p.done()
        wall = cond["wall_s"]
        if r.get("conditions"):                      # derived from another section: its conditions apply
            cond = r["conditions"]
        sec = {"status": r.get("status", "ok"), "reason": r.get("reason"),
               "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "mode": mode, "wall_s": wall,
               "conditions": cond, "params": r.get("params", {}), "rows": r.get("rows", [])}
        if "traceback" in r:
            sec["traceback"] = r["traceback"]
        gpu_samples.append(sample_gpu())
        say(f"[{name}] {sec['status']} in {sec['wall_s']:.1f}s" + (f" ({sec['reason']})" if sec["reason"] else ""))
        return sec

    def do(name: str, fn: Callable[[], dict]) -> None:
        if name in selected:
            sections[name] = measure(name, fn)

    do("kernel", lambda: bench_kernel(**args("kernel", n_cmds=4000 if quick else 20_000, reps=2 if quick else 3)))

    def families_fn():
        nonlocal fam_rows
        r = bench_families(chosen, **args("families", idle_days=1 if quick else 3,
                                          agent_days=2 if quick else STANDARD_EPISODE_DAYS, reps=2 if quick else 3,
                                          workers=workers, seed=seed, timeout_s=900 if quick else 6 * 3600, log=say))
        fam_rows = r["rows"]
        r["rows"] = r["rows"] + not_run_rows()
        return r
    if "families" in selected:
        do("families", families_fn)
        fam_cond = sections["families"]["conditions"]
    elif "episode_cost" in selected:                 # episode cost alone: only the agent runs are needed
        def agents_only():
            nonlocal fam_rows
            r = bench_families(chosen, **args("families", agent_days=2 if quick else STANDARD_EPISODE_DAYS, reps=1,
                                              workers=workers, seed=seed, timeout_s=900 if quick else 6 * 3600,
                                              log=say, idle_days=0))
            fam_rows = r["rows"]
            return r
        fam_cond = measure("families (agent runs only, for episode_cost)", agents_only)["conditions"]

    def storage_fn():
        fams = chosen
        if quick and not families:
            fams = [n for n in chosen if n in ("tiny", "open_digital") or family_group(n) in ("mech", "ow")]
        r = bench_storage(fams, **args("storage", days=2 if quick else 5, workers=min(workers, 2 if quick else workers),
                                       seed=seed, timeout_s=900 if quick else 6 * 3600))
        r["rows"] += [{"family": n, "group": family_group(n), "status": "not_run",
                       "error": "quick mode measures storage for a subset; run the full suite"}
                      for n in chosen if n not in fams] + not_run_rows()
        return r
    do("storage", storage_fn)
    do("memory", lambda: bench_memory(**args("memory", points=_memory_points(mode, disc["families"]), days=1,
                                             workers=min(workers, 2), seed=seed)))
    do("scaling", lambda: bench_scaling(**args("scaling", counts=(1, 2, 4, 8), days=2, reps=2 if quick else 3,
                                               spec="tiny" if quick else {"family": "bazaar",
                                                                          "overrides": {"n_consumers": 3000}})))
    do("rollout", lambda: bench_rollout(**args("rollout", worker_counts=(1, 4) if quick else (1, 2, 4, 8),
                                               tasks_per_worker=4 if quick else 8, steps=12 if quick else 32)))
    do("browser", lambda: bench_browser(**args("browser", n=10 if quick else 30, seed=seed,
                                               wait_s=20 if quick else 120)))

    def episode_fn():
        rows = episode_cost_rows(fam_rows or [], price_per_cpu_hour)
        days = sorted({r["episode_days"] for r in rows if r["episode_days"] is not None})
        note = None
        if days and days != [STANDARD_EPISODE_DAYS]:
            note = (f"episodes of {days} simulated days were measured in this run; the standard "
                    f"{STANDARD_EPISODE_DAYS}-day figure is produced by the full run and is not extrapolated here")
        return {"status": "ok" if rows else "error", "reason": note if rows else "no family runs available",
                "conditions": fam_cond,
                "params": {"episode_days": days, "standard_episode_days": STANDARD_EPISODE_DAYS,
                           "price_per_cpu_hour": price_per_cpu_hour,
                           "includes": "world process CPU + sandboxed programs; excludes the policy's own compute"},
                "rows": rows + not_run_rows()}
    do("episode_cost", episode_fn)
    do("llm", lambda: llm_section(llm_file))
    if "gpu" in selected:
        g = gpu_section(gpu_samples)
        sections["gpu"] = {"status": g["status"], "reason": g["reason"], "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                           "mode": mode, "wall_s": 0.0, "conditions": whole.done(), "params": g["params"],
                           "rows": g["rows"]}

    if "kernel" in sections and out:
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(out)), "bench.json")) as f:
                old = json.load(f)
            sections["kernel"]["params"]["recorded_3_0"] = {
                "source": "bench.json", "t": old.get("t"), "transfers_per_wall_s": old["kernel"]["transfers_per_s"],
                "read_cmds_per_wall_s": old["kernel"]["read_cmds_per_s"],
                "replay_events_per_wall_s": old["kernel"]["replay_events_per_s"],
                "note": "same world and workload as the first, third and fourth rows; measured at another time "
                        "under another load"}
        except (OSError, ValueError, KeyError):
            pass
    cond = whole.done()
    doc: dict[str, Any] = {
        "schema": SCHEMA, "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": mode if set(selected) == set(SECTIONS) else f"{mode} (partial)",
        "argv": list(argv if argv is not None else sys.argv[1:]), "machine": machine_info(), "caveat": CAVEAT,
        "discovery": {**disc, "selected": chosen, "not_run": why_not},
        "run": {"wall_s": round(time.perf_counter() - t_run, 2), "sections_run": [s for s in SECTIONS if s in sections],
                "workers": workers, "seed": seed, "conditions": cond,
                "cpu_utilisation": {"bench_cpu_s": cond["bench_cpu_s"], "bench_cores_used": cond["bench_cores_used"],
                                    "cgroup_cores_used_all_tenants": cond["cgroup_cores_used_all_tenants"],
                                    "cgroup_cpu_quota_cores": cond["cgroup_cpu_quota_cores"],
                                    "cgroup_throttled_fraction": cond["cgroup_throttled_fraction"]}},
        "sections": {}}
    prev = None
    if out and merge and os.path.exists(out) and set(selected) != set(SECTIONS):
        try:
            with open(out) as f:
                prev = json.load(f)
            if prev.get("schema") != SCHEMA:
                prev = None
        except (OSError, ValueError):
            prev = None
    for name in SECTIONS:
        if name in sections:
            doc["sections"][name] = sections[name]
        elif prev is not None and name in prev.get("sections", {}):
            doc["sections"][name] = prev["sections"][name]          # kept from the earlier run, with its own stamp
        else:
            doc["sections"][name] = {"status": "not_run", "reason": "not selected in this run", "generated": None,
                                     "mode": None, "wall_s": 0.0, "conditions": {}, "params": {}, "rows": []}
    problems = validate(doc)
    if problems:
        raise AssertionError(f"benchmark document violates its schema: {problems}")
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(doc, f, indent=1, default=_json_default)
            f.write("\n")
        os.replace(tmp, out)
        say(f"wrote {out}")
    return doc


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def validate(doc: dict) -> list[str]:
    """Schema check of a result document. Returns the list of problems (empty = well-formed)."""
    errs = []
    for k in TOP_KEYS:
        if k not in doc:
            errs.append(f"missing top-level key {k!r}")
    if doc.get("schema") != SCHEMA:
        errs.append(f"schema is {doc.get('schema')!r}, expected {SCHEMA!r}")
    secs = doc.get("sections", {})
    if sorted(secs) != sorted(SECTIONS):
        errs.append(f"sections are {sorted(secs)}, expected {sorted(SECTIONS)}")
    for name, s in secs.items():
        for k in SECTION_KEYS:
            if k not in s:
                errs.append(f"section {name}: missing {k!r}")
        if s.get("status") not in ("ok", "error", "skipped", "not_run", "not_measured"):
            errs.append(f"section {name}: bad status {s.get('status')!r}")
        if not isinstance(s.get("rows"), list) or not all(isinstance(r, dict) for r in s.get("rows", [])):
            errs.append(f"section {name}: rows must be a list of objects")
        if s.get("status") == "ok":
            c = s.get("conditions", {})
            for k in ("loadavg_start", "loadavg_end", "cpu_count", "wall_s"):
                if k not in c:
                    errs.append(f"section {name}: conditions lack {k!r}")
            if name not in ("llm",) and not s["rows"]:
                errs.append(f"section {name}: ok but no rows")
            for i, r in enumerate(s["rows"]):
                if r.get("status") in ("not_run", "error"):
                    continue
                for k in ("reps", "workers", "loadavg"):
                    if name != "llm" and k not in r:
                        errs.append(f"section {name} row {i}: lacks {k!r}")
    return errs


# ====================================================================== text table
def _m(s: dict | None, nd: int = 3) -> str:
    """median (min-max) of a `stat` record."""
    if not s or s.get("median") is None:
        return "-"
    if s["n"] <= 1 or s["min"] == s["max"]:
        return f"{s['median']:.{nd}g}"
    return f"{s['median']:.{nd}g} ({s['min']:.{nd}g}-{s['max']:.{nd}g})"


def _not_run(rows: list[dict], show: int = 6) -> list[str]:
    nr = [r["family"] for r in rows if r.get("status") == "not_run"]
    if not nr:
        return []
    more = f", ... (+{len(nr) - show} more, listed in the JSON)" if len(nr) > show else ""
    return [f"  not run in this mode ({len(nr)}): {', '.join(nr[:show])}{more}"]


def _row(cells: list, widths: list[int]) -> str:
    out = []
    for i, (c, w) in enumerate(zip(cells, widths)):
        c = str(c)
        out.append(c.ljust(w) if i == 0 else c.rjust(w))
    return "  " + "  ".join(out).rstrip()


def format_table(doc: dict) -> str:
    """Human-readable summary of a result document. Ranges in parentheses are min-max over the repetitions."""
    L: list[str] = []
    m, run = doc["machine"], doc["run"]
    c = run["conditions"]
    L.append(f"Crucible OpenWorld benchmark - {doc['mode']} - {doc['generated']}")
    L.append(f"machine: {m['cpu_count']} cpus, cgroup CPU quota {m['cgroup_cpu_quota_cores']} cores, "
             f"{m['memory_total_gb']} GB RAM, python {m['python']}")
    L.append(f"load average: {c['loadavg_start']} at the start, {c['loadavg_end']} at the end; all tenants together "
             f"used {c['cgroup_cores_used_all_tenants']} cores; share of scheduler periods in which the container "
             f"was throttled: {c['cgroup_throttled_fraction']}")
    L.append(f"this run: {run['wall_s']} s wall, {run['cpu_utilisation']['bench_cpu_s']} CPU-s "
             f"({run['cpu_utilisation']['bench_cores_used']} cores used on average by the benchmark processes), "
             f"{run['workers']} worlds at a time")
    L.append("NOTE: " + doc["caveat"])
    S = doc["sections"]

    def head(name, title):
        s = S[name]
        extra = f" - {s['reason']}" if s.get("reason") else ""
        cnd = s.get("conditions") or {}
        load = ""
        if cnd and s["status"] not in ("not_run", "not_measured"):
            load = (f" [load {cnd.get('loadavg_start')} -> {cnd.get('loadavg_end')}, throttled "
                    f"{cnd.get('cgroup_throttled_fraction')}, {cnd.get('wall_s')} s]")
        L.append("")
        L.append(f"== {title}: {s['status']}{extra}{load}")
        return s

    s = head("kernel", "kernel transactions per second (1 process)")
    W = [38, 26, 26, 6, 4]
    if s["rows"]:
        L.append(_row(["measure", "per wall-second", "per CPU-second", "ok", "n"], W))
    for r in s["rows"]:
        L.append(_row([r["measure"], _m(r["per_wall_s"], 5), _m(r["per_cpu_s"], 5), r["ok_fraction"], r["reps"]], W))
    old = s["params"].get("recorded_3_0") if s.get("params") else None
    if old:
        L.append(f"  recorded for 3.0 ({old['t']}, other load): {old['transfers_per_wall_s']} transfers, "
                 f"{old['read_cmds_per_wall_s']} reads, {old['replay_events_per_wall_s']} replayed events per wall-s")
    s = head("families", "simulated days per second, per world family")
    W = [30, 24, 20, 9, 9, 10, 8, 8]
    if s["rows"]:
        p = s["params"]
        L.append(f"  idle: {p.get('idle_days')} simulated day(s) x {p.get('reps')} repetitions; agents: one run of "
                 f"{p.get('agent_days')} day(s), every agent acting each {p.get('dt_s')} simulated s; "
                 f"{p.get('workers')} worlds at a time")
        L.append(_row(["family", "idle d/wall-s", "idle d/cpu-s", "agents", "agents", "events/d", "peak MB", "cost"], W))
        L.append(_row(["", "", "", "d/wall-s", "d/cpu-s", "(idle)", "(idle)", "model"], W))
    for r in s["rows"]:
        if r["status"] == "not_run":
            continue
        i, a = r.get("idle"), r.get("agents")
        if r["status"] == "error":
            L.append(f"  {r['family']:<30}  ERROR: {r['error']}")
            continue
        L.append(_row([r["family"], _m(i and i["sim_days_per_wall_s"]), _m(i and i["sim_days_per_cpu_s"]),
                       _m(a and a["sim_days_per_wall_s"]), _m(a and a["sim_days_per_cpu_s"]),
                       f"{i['events_per_sim_day']['median']:.0f}" if i else "-",
                       f"{i['peak_rss_mb']['median']:.0f}" if i else "-", r["world"].get("cost_model", "-")[:8]], W)
                 + (f"  PARTIAL: {r['error']}" if r["status"] != "ok" else ""))
    L.extend(_not_run(s["rows"]))
    s = head("storage", "events and storage per simulated day (SQLite), snapshot, fork, replay")
    W = [30, 9, 9, 8, 8, 18, 18, 11, 7]
    if s["rows"]:
        L.append(_row(["family", "events/d", "kB/d", "B/event", "snap kB", "snapshot ms", "fork ms", "replay ev/s",
                       "digests"], W))
    for r in s["rows"]:
        if r["status"] == "not_run":
            continue
        if r["status"] != "ok":
            L.append(f"  {r['family']:<30}  ERROR: {r['error']}")
            continue
        ok = "equal" if r["fork_digest_equal"] and r["replay_digest_equal"] in (True, None) else "DIFFER"
        L.append(_row([r["family"], f"{r['events_per_sim_day']['median']:.0f}",
                       f"{r['store_bytes_per_sim_day']['median'] / 1024:.0f}", r["bytes_per_event"],
                       f"{r['snapshot_bytes'] / 1024:.0f}", _m(r["snapshot_ms"]), _m(r["fork_ms"]),
                       f"{(r['replay_events_per_s'] or 0):.0f}", ok], W))
    L.extend(_not_run(s["rows"]))
    s = head("memory", "resident memory per world vs population (a fresh process per world)")
    W = [24, 11, 11, 10, 13, 12, 12]
    if s["rows"]:
        L.append(_row(["family", "population", "principals", "world MB", "process peak", "arrays MB", "snapshot MB"], W))
    for r in s["rows"]:
        if r["status"] != "ok":
            L.append(f"  {r['family']:<24}  ERROR: {r['error']}")
            continue
        L.append(_row([r["family"], r["population"], r["principals"], r["world_rss_mb"], r["peak_rss_mb"],
                       r["state_array_mb"], r["snapshot_mb"]], W))
    s = head("scaling", "parallel worlds through WorldPool")
    W = [16, 26, 9, 11, 24, 10]
    if s["rows"]:
        L.append(f"  world: {s['rows'][0]['world']}, {s['rows'][0]['days_each']} simulated day(s) each, "
                 f"{s['rows'][0]['reps']} repetition(s); pool start-up {s['params'].get('pool_start_s')} s (not included)")
        L.append(_row(["parallel worlds", "sim-days/wall-s (total)", "speedup", "efficiency", "per-world wall s",
                       "throttled"], W))
    for r in s["rows"]:
        L.append(_row([f"{r['parallel_worlds']} of {r['workers']} workers", _m(r["aggregate_sim_days_per_wall_s"]),
                       r["speedup_vs_1"], r["efficiency"], _m(r["per_world_run_wall_s"]),
                       r["cgroup_throttled_fraction"]], W))
    s = head("rollout", "asynchronous rollout workers (RolloutFarm)")
    W = [10, 20, 18, 9, 7, 10]
    if s["rows"]:
        L.append(_row(["workers", "agent steps/wall-s", "sim-days/wall-s", "speedup", "tasks", "exact"], W))
    for r in s["rows"]:
        L.append(_row([r["workers"], _m(r["steps_per_wall_s"]), _m(r["sim_days_per_wall_s"]), r["speedup_vs_1"],
                       r["tasks"], r["accounting_exact"]], W))
    s = head("browser", "one real Chromium session")
    for r in s["rows"]:
        L.append(f"  {r['measure']:<44} {r['actions_per_s']:>9} actions/s"
                 + (f"  {_m(r['ms_per_action'])} ms each" if "ms_per_action" in r else "")
                 + (f"  ({r['cost_vs_api']}x the API cost)" if "cost_vs_api" in r else ""))
    s = head("gpu", "GPU (read-only nvidia-smi: other tenants' load, the suite uses no GPU)")
    for r in s["rows"]:
        L.append(f"  gpu {r['gpu']} {r['name']}: utilisation {_m(r['util_pct'])} %, memory {_m(r['mem_used_mb'], 5)} of "
                 f"{r['mem_total_mb']:.0f} MB, {r['reps']} samples")
    s = head("episode_cost", "cost per experimental episode (world only; the policy's own compute is extra)")
    W = [30, 6, 10, 9, 9, 7, 10]
    if s["rows"]:
        L.append(_row(["family", "days", "CPU-s", "peak MB", "wall s", "steps", "USD"], W))
    for r in s["rows"]:
        if r["status"] == "not_run":
            continue
        if r["status"] != "ok":
            L.append(f"  {r['family']:<30}  ERROR: {r['error']}")
            continue
        L.append(_row([r["family"], r["episode_days"], f"{r['cpu_s']:.2f}", f"{r['peak_rss_mb']:.1f}",
                       f"{r['wall_s']:.1f}", r["agent_steps"], "-" if r["usd"] is None else f"{r['usd']:.5f}"], W))
    s = head("llm", "LLM inference throughput")
    for r in s["rows"]:
        L.append(f"  {json.dumps(r, default=str)[:200]}")
    return "\n".join(L)


# ====================================================================== command line
def main(argv: list[str] | None = None, default_out: str | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Crucible 4.0 benchmark suite (measured, with conditions).")
    ap.add_argument("--quick", action="store_true", help="a few minutes: short horizons, a subset of the 3.0 families")
    ap.add_argument("--only", default="", help=f"comma-separated sections to run (of: {', '.join(SECTIONS)})")
    ap.add_argument("--skip", default="", help="comma-separated sections to leave out")
    ap.add_argument("--families", default="", help="comma-separated world families, or 'all' (default: by mode)")
    ap.add_argument("--max-per-group", type=int, default=None, help="quick mode: cap on mech_* / ow_* families")
    ap.add_argument("--workers", type=int, default=None, help="worlds measured at a time (default: up to 4)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=default_out or os.path.join("docs", "benchmarks", "openworld.json"))
    ap.add_argument("--no-merge", action="store_true", help="with --only: do not keep other sections of --out")
    ap.add_argument("--price-per-cpu-hour", type=float, default=None,
                    help="your price in dollars per CPU-hour; without it no dollar figure is reported")
    ap.add_argument("--llm-file", default=None, help="JSON with LLM throughput measured elsewhere, copied into the result")
    ap.add_argument("--import", dest="imports", default="", help="extra python modules to import before discovery")
    ap.add_argument("--list", action="store_true", help="list sections and discovered families, then exit")
    a = ap.parse_args(argv)
    split = lambda s: [x for x in s.split(",") if x]  # noqa: E731
    if a.list:
        d = discover(split(a.imports))
        print("sections:", ", ".join(SECTIONS))
        for g, names in sorted(d["groups"].items()):
            print(f"{g}: {', '.join(names)}")
        if d["import_errors"]:
            print("import errors:", d["import_errors"])
        return d
    doc = run_suite(mode="quick" if a.quick else "full", only=split(a.only) or None, skip=split(a.skip),
                    families=split(a.families) or None, workers=a.workers, out=a.out,
                    price_per_cpu_hour=a.price_per_cpu_hour, llm_file=a.llm_file, seed=a.seed,
                    max_per_group=a.max_per_group, merge=not a.no_merge, imports=split(a.imports),
                    argv=list(argv) if argv is not None else sys.argv[1:])
    print()
    print(format_table(doc))
    return doc


if __name__ == "__main__":
    main()
