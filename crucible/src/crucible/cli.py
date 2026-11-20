"""`crucible` command line: worlds, actors, experiments, self-play, evaluation, tracing, dashboards, serving.

    crucible world create --family tiny --seed 1 [--id W] [--deterministic] [--param k=v] [--store URL]
    crucible world list | inspect W | run W --days N [--snapshot-every-days K] | snapshot W | fork SNAP [--id F]
    crucible world replay W [--verify] | verify W [--no-regen]
    crucible actor join W alice [--endowment 1000000] | run W alice --policy builder --days N [--trajectory f]
    crucible experiment run --goal revenue --world tiny --level 0 --seeds 1,2 --policies idle,builder [--out r.json]
    crucible selfplay run --world tiny --seeds 1 --days 2 --policies "builder:undercut=0.9,builder:undercut=0.7"
    crucible evaluate --policy builder [--split val] [--world-families ...] [--max-days D] [--out f]
    crucible trace W ENTITY [--chain SEQ] | report RESULTS.json | counterfactual W --undeploy HOST --days N
    crucible emergence W | dashboard W [--port 8050] | serve [--port 8000]

Store URLs: memory://, sqlite:///path/to.db, postgresql://... Default: $CRUCIBLE_STORE or
sqlite:///./crucible-data/crucible.db (the directory is created). Policy specs: ``name[:key=value...]`` with name in
builder | employer | composer | bounty | random | idle (e.g. ``builder:need=route.shortest:undercut=0.8``).

Persistent runs (`world run`, `actor run`) hold the world's single-writer lease (`--lease-ttl`), flush events to
the store continuously, snapshot periodically and at the end, and stop cleanly on Ctrl-C (SIGINT); a later run
resumes from the latest snapshot plus the events after it (`World.load`). Scripted policies keep their plan in
memory only: a new `actor run` starts the policy's script afresh on the persisted world.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from typing import Any

DAY = 86400
DEFAULT_STORE = "sqlite:///./crucible-data/crucible.db"


# ====================================================================== helpers
def _store_url(a) -> str:
    url = getattr(a, "store", None) or os.environ.get("CRUCIBLE_STORE") or DEFAULT_STORE
    if url.startswith("sqlite:"):
        path = url.split("sqlite://", 1)[1] if "sqlite://" in url else url[len("sqlite:"):]
        if path and path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    return url


def _server(a, readonly: bool = False):
    import crucible
    crucible.load_all_modules()
    from .api.local import WorldServer
    ttl = None if readonly or not getattr(a, "lease_ttl", 0) else float(a.lease_ttl)
    return WorldServer(_store_url(a), lease_ttl_s=ttl)


def _out(a, obj: Any, text: str | None = None) -> None:
    if getattr(a, "json", False) or text is None:
        print(json.dumps(obj, indent=1, sort_keys=True, default=str))
    else:
        print(text)


def _kv(items) -> dict:
    from .training.league import _val
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"expected key=value, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = _val(v.strip())
    return out


def _ints(s: str) -> list[int]:
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            lo, hi = part.split("-", 1)
            out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def _fmt_t(t: int) -> str:
    return f"day {t // DAY} {(t % DAY) // 3600:02d}:{(t % 3600) // 60:02d}"


class _Stop:
    """SIGINT -> stop flag (finish the current step, snapshot, exit). A second Ctrl-C aborts immediately."""

    def __init__(self):
        self.flag = False
        self._old = None

    def __enter__(self):
        try:
            self._old = signal.signal(signal.SIGINT, self._handle)
        except ValueError:      # not in the main thread
            self._old = None
        return self

    def _handle(self, signum, frame):
        if self.flag:
            raise KeyboardInterrupt
        self.flag = True
        print("\n[crucible] interrupt received: finishing the current step, then snapshot and exit "
              "(Ctrl-C again to abort)", file=sys.stderr, flush=True)

    def __exit__(self, *exc):
        if self._old is not None:
            signal.signal(signal.SIGINT, self._old)
        return False


def _table(rows: list[list], header: list[str]) -> str:
    cols = [header] + [[("" if c is None else str(c)) for c in r] for r in rows]
    w = [max(len(r[i]) for r in cols) for i in range(len(header))]
    lines = ["  ".join(h.ljust(w[i]) for i, h in enumerate(header)), "  ".join("-" * x for x in w)]
    lines += ["  ".join(c.ljust(w[i]) for i, c in enumerate(r)) for r in cols[1:]]
    return "\n".join(lines)


# ====================================================================== world
def cmd_world_create(a) -> int:
    ov: dict = {}
    params = _kv(a.param)
    if a.deterministic:
        params["cost_model"] = "deterministic"
    if params:
        ov["params"] = params
    if a.n_consumers:
        ov["n_consumers"] = int(a.n_consumers)
    ov.update(_kv(a.override))
    with _server(a) as srv:
        wid = srv.create_world({"family": a.family, "overrides": ov}, a.seed, world_id=a.id)
        if a.snapshot:
            srv.snapshot_world(wid)
        info = srv.world_info(wid)
    _out(a, info, f"created world {wid} (family {a.family}, seed {a.seed}, {info.get('seq')} events)")
    return 0


def cmd_world_list(a) -> int:
    with _server(a, readonly=True) as srv:
        rows = srv.list_worlds()
        store = srv.store
        table = []
        for r in rows:
            if r.get("alias"):
                table.append([r["name"], "alias ->", r["world_id"], "", "", "", ""])
                continue
            snaps = store.list_snapshots(r["world_id"])
            par = r.get("parent")
            table.append([r["world_id"], r["family"], r["seed"], r.get("status"), store.last_seq(r["world_id"]),
                          len(snaps), f"{par['world']}@{par['seq']}" if par else ""])
            r["events"] = store.last_seq(r["world_id"])
            r["snapshots"] = [s["id"] for s in snaps]
    _out(a, rows, _table(table, ["world", "family", "seed", "status", "last_seq", "snapshots", "parent"])
         if table else "no worlds")
    return 0


def inspect_world(w, recent: int = 15) -> dict:
    from .analytics import NOISE_TYPES, accounts_of, hosts_of, world_events
    from .analytics.trace import describe_event
    st = w.state
    out: dict = {"world_id": w.world_id, "family": w.spec.family, "seed": w.spec.seed, "time": w.now,
                 "clock": _fmt_t(w.now), "seq": w.seq, "hash": w.last_hash, "spec_hash": w.spec.hash()[:16]}
    cons = w.modules.get("consumers")
    shares = cons.market_shares(w) if cons is not None and "c_prov" in st.arrays else {}

    def cash(owner):
        return int(sum(st.tables["accounts"][x]["balance"] for x in accounts_of(w, owner)))
    if "persons" in st.tables:
        out["actors"] = [{"id": pid, "kind": p["kind"], "status": st.tables["principals"][pid].get("status"),
                          "actions": st.tables["principals"][pid].get("actions", 0), "cash": cash(pid)}
                         for pid, p in st.scan("persons") if p["kind"] != "npc" or p["info"].get("role") not in
                         ("consumer-voice",)]
    if "orgs" in st.tables:
        out["orgs"] = [{"id": oid, "status": o["status"], "kind": o["kind"],
                        "members": len(st.find("members", "org", oid)), "cash": cash(oid),
                        "hosts": hosts_of(w, oid)} for oid, o in st.scan("orgs")]
    if "accounts" in st.tables:
        accs = sorted(((aid, x) for aid, x in st.scan("accounts") if x["owner"] != "sys"),
                      key=lambda kv: -kv[1]["balance"])
        out["top_balances"] = [{"account": aid, "owner": x["owner"], "balance": x["balance"]} for aid, x in accs[:10]]
        hh = st.get("accounts", "acct_households")
        if hh:
            out["household_cash"] = hh["balance"]
    if "deployments" in st.tables:
        out["services"] = [{"host": d["host"], "owner": d["owner"], "status": d["status"], "requests": d["requests"],
                            "customers": shares.get(d["host"], 0), "trusted": d["trusted"]}
                           for _, d in st.scan("deployments") if d["status"] in ("live", "stopped", "failed")]
    evs = []
    lo = max(0, w.seq - 4000)
    for ev in world_events(w, lo):
        if ev.type not in NOISE_TYPES:
            evs.append(ev)
    out["recent_events"] = [{"seq": e.seq, "t": e.t, "type": e.type, "text": describe_event(e)}
                            for e in evs[-recent:]]
    out["invariant_violations"] = w.check_invariants()
    return out


def _render_inspect(d: dict) -> str:
    from .analytics import money
    L = [f"world {d['world_id']}  family={d['family']} seed={d['seed']}  {d['clock']} (t={d['time']})  "
         f"seq={d['seq']}  hash={d['hash'][:12]}"]
    if d.get("actors"):
        L += ["", "actors:", _table([[x["id"], x["kind"], x["status"], x["actions"], money(x["cash"])]
                                     for x in d["actors"]], ["principal", "kind", "status", "actions", "cash"])]
    if d.get("orgs"):
        L += ["", "organisations:", _table([[x["id"], x["status"], x["members"], money(x["cash"]), ",".join(x["hosts"])]
                                            for x in d["orgs"]], ["org", "status", "members", "cash", "hosts"])]
    if d.get("top_balances"):
        L += ["", "top balances:", _table([[x["account"], x["owner"], money(x["balance"])] for x in d["top_balances"]],
                                          ["account", "owner", "balance"])]
    if d.get("services"):
        L += ["", "services:", _table([[x["host"], x["owner"], x["status"], x["requests"], x["customers"]]
                                       for x in d["services"]], ["host", "owner", "status", "requests", "customers"])]
    if d.get("recent_events"):
        L += ["", "recent events:"] + [f"  #{e['seq']} {_fmt_t(e['t'])} {e['text']}" for e in d["recent_events"]]
    L += ["", f"invariants: {'ok' if not d['invariant_violations'] else d['invariant_violations']}"]
    return "\n".join(L)


def cmd_world_inspect(a) -> int:
    with _server(a, readonly=True) as srv:
        w = srv.world(a.world)
        d = inspect_world(w, a.recent)
    _out(a, d, _render_inspect(d))
    return 0


def cmd_world_run(a) -> int:
    with _server(a) as srv, _Stop() as stop:
        w = srv.world(a.world)
        wid = w.world_id
        start = w.now
        end = start + int(round(a.days * DAY))
        step = int(a.step_hours * 3600)
        snap_every = int(round(a.snapshot_every_days * DAY)) if a.snapshot_every_days else 0
        next_snap = start + snap_every if snap_every else None
        snaps = []
        t_wall = time.perf_counter()
        last_report = start
        while w.now < end and not stop.flag:
            srv.advance_time(wid, until=min(end, w.now + step))
            if next_snap is not None and w.now >= next_snap:
                snaps.append(srv.snapshot_world(wid))
                next_snap += snap_every
            if not a.quiet and (w.now - last_report >= DAY or w.now >= end):
                last_report = w.now
                print(f"[crucible] {wid}: {_fmt_t(w.now)} seq={w.seq} "
                      f"({(time.perf_counter() - t_wall):.1f}s wall)", flush=True)
        if not a.no_final_snapshot:
            sid = srv.snapshot_world(wid)       # idempotent: same id if a periodic snapshot was just taken
            if not snaps or snaps[-1] != sid:
                snaps.append(sid)
        res = {"world_id": wid, "from": start, "to": w.now, "days": round((w.now - start) / DAY, 4), "seq": w.seq,
               "interrupted": stop.flag, "snapshots": snaps, "wall_s": round(time.perf_counter() - t_wall, 3)}
    _out(a, res, f"{'interrupted' if res['interrupted'] else 'ran'} {wid} from {_fmt_t(start)} to {_fmt_t(res['to'])} "
                 f"({res['days']} days, seq {res['seq']}); snapshots: {', '.join(snaps) or 'none'}")
    return 130 if res["interrupted"] and a.exit_code_on_interrupt else 0


def cmd_world_snapshot(a) -> int:
    with _server(a) as srv:
        sid = srv.snapshot_world(a.world)
        w = srv.world(a.world)
        res = {"snapshot": sid, "world_id": w.world_id, "seq": w.seq, "time": w.now}
    _out(a, res, sid)
    return 0


def cmd_world_fork(a) -> int:
    with _server(a) as srv:
        src = a.snapshot
        if srv.store.world_info(srv.resolve(src)) is not None:     # a world id: fork its current state
            src = srv.snapshot_world(src)
        new = srv.fork_world(src, a.id)
        info = srv.world_info(new)
    _out(a, dict(info, snapshot=src), f"forked {src} -> {new}")
    return 0


def _lineage_events(store, wid: str) -> list[dict]:
    from .analytics import _read_lineage
    return _read_lineage(store, wid, 0, None)


def cmd_world_replay(a) -> int:
    import crucible
    crucible.load_all_modules()
    from collections import Counter

    from .core.kernel import World, WorldSpec
    with _server(a, readonly=True) as srv:
        wid = srv.resolve(a.world)
        info = srv.store.world_info(wid)
        if info is None:
            raise SystemExit(f"no world {a.world!r}")
        spec = WorldSpec.from_dict(info["spec"])
        t0 = time.perf_counter()
        events = _lineage_events(srv.store, wid)
        res: dict = {"world_id": wid, "events": len(events), "last_seq": events[-1]["seq"] if events else 0,
                     "types": dict(Counter(e["type"] for e in events).most_common(12))}
        ok = True
        try:
            rep = World.replay(spec, events, world_id=f"{wid}~replay", verify=a.verify)
            res["replay_hash"] = rep.last_hash
            res["replay_digest"] = rep.digest()
            res["replay_time"] = rep.now
        except Exception as e:  # noqa: BLE001
            ok = False
            res["error"] = f"{type(e).__name__}: {e}"
            rep = None
        if a.verify and rep is not None:
            fresh = World.load(srv.store, wid)
            res["hash_chain_ok"] = True
            res["store_hash"] = events[-1]["hash"] if events else None
            res["loaded_digest"] = fresh.digest()
            res["digest_match"] = fresh.digest() == rep.digest()
            res["hash_match"] = fresh.last_hash == rep.last_hash == res["store_hash"]
            ok = ok and res["digest_match"] and res["hash_match"]
            fresh.close()
        if rep is not None:
            rep.close()
        res["ok"] = ok
        res["wall_s"] = round(time.perf_counter() - t0, 3)
    txt = (f"replayed {res['events']} events of {wid} (last seq {res['last_seq']}) in {res['wall_s']}s; "
           + ("" if not a.verify else f"hash chain {'verified' if res.get('hash_chain_ok') else 'BROKEN'}, "
              f"digest vs fresh load {'match' if res.get('digest_match') else 'MISMATCH'}, final hash "
              f"{'match' if res.get('hash_match') else 'MISMATCH'}")
           + (f" ERROR {res['error']}" if "error" in res else ""))
    _out(a, res, txt)
    return 0 if ok else 1


def cmd_world_verify(a) -> int:
    import crucible
    crucible.load_all_modules()
    from .core.kernel import World, WorldSpec
    with _server(a, readonly=True) as srv:
        wid = srv.resolve(a.world)
        info = srv.store.world_info(wid)
        if info is None:
            raise SystemExit(f"no world {a.world!r}")
        t0 = time.perf_counter()
        w = World.load(srv.store, wid)       # snapshot + replay, hash chain verified
        res: dict = {"world_id": wid, "seq": w.seq, "time": w.now, "load_ok": True}
        res["invariant_violations"] = w.check_invariants()
        ok = not res["invariant_violations"]
        if not a.no_regen:
            events = _lineage_events(srv.store, wid)
            try:
                g = World.regenerate(WorldSpec.from_dict(info["spec"]), events, world_id=f"{wid}~regen")
                res["regen_hash_match"] = g.last_hash == w.last_hash
                res["regen_digest_match"] = g.digest() == w.digest()
                res["regen_seq"] = g.seq
                ok = ok and res["regen_hash_match"] and res["regen_digest_match"]
                g.close()
            except Exception as e:  # noqa: BLE001
                res["regen_error"] = f"{type(e).__name__}: {e}"
                ok = False
        w.close()
        res["ok"] = ok
        res["wall_s"] = round(time.perf_counter() - t0, 3)
    txt = (f"{wid}: load+hash chain ok, invariants "
           f"{'ok' if not res['invariant_violations'] else res['invariant_violations']}"
           + ("" if a.no_regen else f", regeneration {'identical' if res.get('regen_hash_match') and res.get('regen_digest_match') else 'DIFFERS ' + str(res.get('regen_error', ''))}")
           + f" ({res['wall_s']}s)")
    _out(a, res, txt)
    return 0 if ok else 1


# ====================================================================== actors
def cmd_actor_join(a) -> int:
    with _server(a) as srv:
        ident = srv.join_world(a.world, a.handle, kind=a.kind, endowment=int(a.endowment))
        res = ident.to_dict(include_token=True)
    _out(a, res, f"joined {res['principal']} (account {res['account']}) in {res['world_id']}\n"
                 f"token (shown once, keep it secret): {res['token']}")
    return 0


def cmd_actor_run(a) -> int:
    from .training.league import policy_from_spec
    from .training.trajectories import RecordingRunner, export_jsonl, export_npz
    with _server(a) as srv, _Stop() as stop:
        w = srv.world(a.world)
        wid = w.world_id
        actor = a.handle if a.handle.startswith("p:") else f"p:{a.handle}"
        if w.principal(actor) is None:
            if not a.join:
                raise SystemExit(f"no actor {actor} in {wid}; run `crucible actor join {wid} {a.handle}` first "
                                 f"(or pass --join)")
            srv.join_world(wid, actor.split(":", 1)[1], endowment=int(a.endowment))
        pol = policy_from_spec(a.policy)
        run = RecordingRunner(w, [(actor, pol)], dt=int(a.dt), seed=int(a.seed), record=bool(a.trajectory))
        start, seq0 = w.now, w.seq
        end = start + int(round(a.days * DAY))
        snap_every = int(round(a.snapshot_every_days * DAY)) if a.snapshot_every_days else 0
        next_snap = start + snap_every if snap_every else None
        actions0 = (w.principal(actor) or {}).get("actions", 0)
        t_wall = time.perf_counter()
        n_rounds = 0
        while w.now + int(a.dt) <= end and not stop.flag:
            run.round()
            n_rounds += 1
            srv.advance_time(wid, until=w.now)     # flush to the store and renew the writer lease
            if next_snap is not None and w.now >= next_snap:
                srv.snapshot_world(wid)
                next_snap += snap_every
        if w.now < end and not stop.flag:
            srv.advance_time(wid, until=end)
        snap = srv.snapshot_world(wid)
        rejected = sum(1 for e in w.log if e.type == "cmd.rejected" and e.payload.get("actor") == actor
                       and e.seq > seq0)
        from .goals.metrics import m_cash
        res = {"world_id": wid, "actor": actor, "policy": a.policy, "rounds": n_rounds, "from": start, "to": w.now,
               "actions": (w.principal(actor) or {}).get("actions", 0) - actions0, "rejected": rejected,
               "cash": int(m_cash(w, actor)), "snapshot": snap, "interrupted": stop.flag,
               "wall_s": round(time.perf_counter() - t_wall, 3), "log": list(getattr(pol, "log", []))[-10:]}
        if a.trajectory:
            n = (export_npz(run, a.trajectory)["n"] if a.trajectory.endswith(".npz")
                 else export_jsonl(run, a.trajectory))
            res["trajectory"] = {"path": a.trajectory, "records": n}
    _out(a, res, f"{actor} ran {a.policy} for {round((res['to'] - start) / DAY, 3)} days in {wid}: "
                 f"{res['actions']} actions ({res['rejected']} rejected), cash {res['cash']}c; snapshot {snap}"
                 + (f"; trajectory -> {a.trajectory} ({res['trajectory']['records']} steps)" if a.trajectory else ""))
    return 0


# ====================================================================== experiments
def _curve(traj, max_points: int = 200) -> list:
    pts, cum = [], 0.0
    for s in traj.steps:
        cum += s.reward
        pts.append([round(s.t / DAY, 4), round(cum, 6)])
    if len(pts) > max_points:
        idx = sorted({int(i) for i in [round(k * (len(pts) - 1) / (max_points - 1)) for k in range(max_points)]})
        pts = [pts[i] for i in idx]
    return pts


def run_experiment(goal_family: str, world_family: str, level: int, seeds, policies, *, max_days=None, dt=3600,
                   stop_on_success=True, overrides=None, split=None) -> dict:
    import crucible
    crucible.load_all_modules()
    from .evaluation.harness import paired, run_goal, summarize
    from .goals.contracts import generate_goal
    from .training.league import policy_from_spec
    t0 = time.perf_counter()
    runs: dict[str, list] = {p: [] for p in policies}
    curves: dict[str, list] = {p: [] for p in policies}
    goals = []
    for seed in seeds:
        g = generate_goal(goal_family, world_family, int(seed), int(level), split=split, overrides=overrides)
        goals.append(g.to_dict())
        for spec in policies:
            pol = policy_from_spec(spec)
            r, run = run_goal(g, pol, dt=dt, max_days=max_days, stop_on_success=stop_on_success, record=True)
            r.policy = spec
            runs[spec].append(r)
            traj = run.trajs[run.agents[0][0]]
            curves[spec].append({"goal_id": g.id, "seed": int(seed), "success": r.success, "points": _curve(traj)})
            run.world.close()
    out = {"format": "crucible-experiment/1", "config": {"goal": goal_family, "world": world_family,
                                                         "level": int(level), "seeds": [int(s) for s in seeds],
                                                         "policies": list(policies), "max_days": max_days, "dt": dt,
                                                         "overrides": overrides or {}},
           "goals": goals, "policies": {p: summarize(rs) for p, rs in runs.items()}, "curves": curves,
           "paired": {}, "wall_s": 0.0}
    base = policies[0]
    for p in policies[1:]:
        out["paired"][f"{p} - {base}"] = {"progress": paired(runs[p], runs[base], "progress"),
                                          "success": paired(runs[p], runs[base], "success"),
                                          "reward": paired(runs[p], runs[base], "reward")}
    out["wall_s"] = round(time.perf_counter() - t0, 3)
    return out


def cmd_experiment_run(a) -> int:
    from .training.league import split_specs
    res = run_experiment(a.goal, a.world, a.level, _ints(a.seeds), split_specs(a.policies), max_days=a.max_days,
                         dt=a.dt, stop_on_success=not a.no_stop, overrides={"params": _kv(a.param)} if a.param else None)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=str)
    _out(a, res if a.json else None, report_markdown(res) + (f"\n\nwrote {a.out}" if a.out else ""))
    return 0


def cmd_selfplay_run(a) -> int:
    from .training.league import League, split_specs
    ov = {"params": _kv(a.param)} if a.param else {}
    if a.n_consumers:
        ov["n_consumers"] = int(a.n_consumers)
    lg = League(world=a.world, days=a.days, seed=a.seed, mode=a.mode, overrides=ov, endowment=int(a.endowment),
                dt=int(a.dt), deterministic=not a.measured_cost)
    specs = split_specs(a.policies)
    for i, s in enumerate(specs):
        lg.add_policy(s, spec=s, tags=("scripted",))
    for s in split_specs(a.heldout or ""):
        lg.add_policy(f"heldout:{s}", spec=s, heldout=True, tags=("heldout",))
    seeds = _ints(a.seeds)
    t0 = time.perf_counter()
    if a.schedule == "round_robin":
        lg.round_robin(seeds, both_seatings=not a.one_seating)
    else:
        learner = a.learner or specs[0]
        lg.train_schedule(learner, seeds, int(a.matches), method=a.schedule)
    heldout = {}
    if lg.heldout_pool():
        for p in lg.training_pool():
            heldout[p] = lg.evaluate_heldout(p, seeds[:1], both_seatings=False)
    res = lg.to_dict()
    res["heldout_eval"] = heldout
    res["wall_s"] = round(time.perf_counter() - t0, 3)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=str)
    txt = lg.format_tables()
    if heldout:
        txt += "\n\nheld-out evaluation (never used for training):\n" + "\n".join(
            f"  {p}: mean win rate {v['mean_win_rate']}" for p, v in heldout.items())
    _out(a, res if a.json else None, txt + f"\n\n{len(lg.records)} matches in {res['wall_s']}s"
         + (f"; wrote {a.out}" if a.out else ""))
    return 0


def cmd_evaluate(a) -> int:
    import crucible
    crucible.load_all_modules()
    from .evaluation.harness import evaluate, train_goals, validation_goals
    from .training.league import policy_factory
    kw = {"n_per": a.n_per, "levels": tuple(_ints(a.levels))}
    if a.goal_families:
        kw["families"] = [x.strip() for x in a.goal_families.split(",") if x.strip()]
    if a.world_families:
        kw["world_families"] = [x.strip() for x in a.world_families.split(",") if x.strip()]
    goals = (validation_goals if a.split == "val" else train_goals)(**kw)
    t0 = time.perf_counter()
    res = evaluate(policy_factory(a.policy), goals, dt=a.dt, max_days=a.max_days)
    res = {"format": "crucible-evaluation/1", "policy": a.policy, "split": a.split, "goals": [g.id for g in goals],
           **res, "wall_s": round(time.perf_counter() - t0, 3)}
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=str)
    s, p = res["success"], res["progress"]
    _out(a, res if a.json else None, f"{a.policy} on {res['n']} {a.split} goals: success {s[0]:.3f} "
                                     f"[{s[1]:.3f}, {s[2]:.3f}], progress {p[0]:.3f} [{p[1]:.3f}, {p[2]:.3f}], "
                                     f"integrity flags {res['integrity_flags']}" + (f"; wrote {a.out}" if a.out else ""))
    return 0


# ====================================================================== analysis
def cmd_trace(a) -> int:
    from .analytics.trace import causal_chain, explain, timeline
    with _server(a, readonly=True) as srv:
        w = srv.world(a.world)
        if a.chain is not None or a.entity.startswith("#"):
            seq = int(a.chain if a.chain is not None else a.entity[1:])
            res = causal_chain(w, seq)
            _out(a, res, res["text"])
            return 0
        res = explain(w, a.entity, since=a.since)
        tl = timeline(w, a.entity, since=a.since, limit=a.timeline) if a.timeline else []
        res["timeline"] = tl
    txt = res["text"]
    if tl:
        txt += "\n\nTimeline (latest):\n" + "\n".join(f"  #{r['seq']} {_fmt_t(r['t'])} {r['text']}" for r in tl)
    _out(a, res, txt)
    return 0


def _ci(x) -> str:
    if isinstance(x, (list, tuple)) and len(x) == 3:
        return f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"
    return str(x)


def report_markdown(d: dict) -> str:
    fmt = d.get("format", "")
    if fmt.startswith("crucible-experiment"):
        c = d["config"]
        L = [f"### Experiment: goal `{c['goal']}` level {c['level']} on `{c['world']}` (seeds {c['seeds']})", "",
             "| policy | runs | success [95% CI] | progress [95% CI] | progress AUC | actions | integrity flags |",
             "|---|---|---|---|---|---|---|"]
        for p, s in d["policies"].items():
            L.append(f"| {p} | {s['n']} | {_ci(s['success'])} | {_ci(s['progress'])} | {_ci(s['progress_auc'])} | "
                     f"{s['actions_mean']:.1f} | {s['integrity_flags']} |")
        if d.get("paired"):
            L += ["", "| paired comparison | n | progress diff | success diff | reward diff |", "|---|---|---|---|---|"]
            for k, v in d["paired"].items():
                L.append(f"| {k} | {v['progress']['n']} | {_ci(v['progress']['mean_diff'])} | "
                         f"{_ci(v['success']['mean_diff'])} | {_ci(v['reward']['mean_diff'])} |")
        return "\n".join(L)
    if fmt.startswith("crucible-league"):
        from .training.league import League
        return League.load(d).format_tables()
    if fmt.startswith("crucible-evaluation"):
        return (f"### Evaluation of `{d['policy']}` on {d['n']} {d['split']} goals\n\n| metric | value |\n|---|---|\n"
                f"| success | {_ci(d['success'])} |\n| progress | {_ci(d['progress'])} |\n"
                f"| progress AUC | {_ci(d['progress_auc'])} |\n| actions (mean) | {d['actions_mean']:.1f} |\n"
                f"| integrity flags | {d['integrity_flags']} |")
    if "delta" in d and "intervention" in d:
        return "```\n" + d.get("text", "") + "\n```"
    rows = [(k, v) for k, v in d.items() if isinstance(v, (int, float, str))]
    return "| key | value |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)


def cmd_report(a) -> int:
    with open(a.results) as f:
        d = json.load(f)
    print(report_markdown(d))
    return 0


def cmd_counterfactual(a) -> int:
    from .evaluation import counterfactual as CF
    from .training.league import policy_factory
    ivs = [CF.undeploy_host(h) for h in a.undeploy or []] + [CF.close_listings(host=h) for h in a.close_listings or []]
    ivs += [CF.suspend_principal(p) for p in a.suspend or []]
    if a.set_param:
        ivs.append(CF.set_params(**_kv(a.set_param)))
    if not ivs:
        raise SystemExit("give an intervention: --undeploy HOST, --close-listings HOST, --suspend PRINCIPAL, "
                         "--set-param k=v")
    iv = ivs[0] if len(ivs) == 1 else CF.combine(*ivs)
    agents = None
    if a.agent:
        agents = []
        for spec in a.agent:
            handle, pol = spec.split("=", 1)
            agents.append((handle if handle.startswith("p:") else f"p:{handle}", policy_factory(pol)))
    with _server(a, readonly=True) as srv:
        w = srv.world(a.world)
        res = CF.fork_compare(w, iv, int(round(a.days * DAY)), agents, dt=int(a.dt),
                              check_determinism=a.check_determinism)
    res["format"] = "crucible-counterfactual/1"
    for k in ("before",):
        res.pop(k, None)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=str)
    _out(a, res if a.json else None, res["text"] + (f"\nwrote {a.out}" if a.out else ""))
    return 0


def cmd_emergence(a) -> int:
    from .evaluation.emergence import emergence_report
    with _server(a, readonly=True) as srv:
        w = srv.world(a.world)
        rep = emergence_report(w, since=a.since)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1, default=str)
    s = rep["strategy_diversity"]
    txt = (f"tool reuse: {rep['tool_reuse']['n_reused']}/{rep['tool_reuse']['n_agent_hosts']} agent hosts have "
           f"independent callers\nconcentration (HHI now): {rep['concentration']['current']}\n"
           f"specialization: {rep['specialization']['summary']}\nstrategy diversity: {s['n_clusters']} cluster(s), "
           f"mean pairwise JS {s['mean_pairwise_js']}\ndependency graph: {rep['dependencies']['n_nodes']} nodes, "
           f"{rep['dependencies']['n_edges']} edges, longest service chain {rep['dependencies']['max_service_chain']}")
    _out(a, rep if a.json else None, txt)
    return 0


# ====================================================================== servers
def cmd_dashboard(a) -> int:
    import uvicorn

    from .dashboard.app import make_app
    if a.host not in ("127.0.0.1", "localhost", "::1") and not a.allow_remote:
        raise SystemExit("the dashboard binds to loopback only; pass --allow-remote to bind elsewhere "
                         "(it has no authentication)")
    app = make_app(store_url=_store_url(a), worlds=[a.world] if a.world else None,
                   experiments=a.experiments or (), trajectories=a.trajectories or (), operator=a.operator)
    print(f"[crucible] dashboard for {a.world or 'all worlds'} at http://{a.host}:{a.port}/", flush=True)
    uvicorn.run(app, host=a.host, port=int(a.port), log_level="warning")
    return 0


def cmd_serve(a) -> int:
    from .api import http_server
    argv = ["--store", _store_url(a), "--host", a.host, "--port", str(a.port), "--lease-ttl", str(a.lease_ttl)]
    if a.world:
        argv += ["--world", a.world]
    return http_server.main(argv)


# ====================================================================== parser
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--store", default=None, help=f"store URL (default $CRUCIBLE_STORE or {DEFAULT_STORE})")
    common.add_argument("--json", action="store_true", help="machine-readable JSON output")
    common.add_argument("--lease-ttl", type=float, default=60.0,
                        help="single-writer lease seconds for commands that write (0 disables)")
    ap = argparse.ArgumentParser(prog="crucible", description="Crucible world simulator: researcher tooling.")
    sub = ap.add_subparsers(dest="cmd")

    # world
    pw = sub.add_parser("world", help="create, run, snapshot, fork, replay and verify persistent worlds")
    ws = pw.add_subparsers(dest="sub")
    p = ws.add_parser("create", parents=[common], help="create a world from a generator family")
    p.add_argument("--family", default="tiny")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--id", default=None)
    p.add_argument("--deterministic", action="store_true", help="params.cost_model=deterministic (exact forks)")
    p.add_argument("--param", action="append", help="world param override key=value (repeatable)")
    p.add_argument("--override", action="append", help="generator override key=value (repeatable)")
    p.add_argument("--n-consumers", type=int, default=None)
    p.add_argument("--snapshot", action="store_true", help="snapshot right after genesis")
    p.set_defaults(func=cmd_world_create)
    p = ws.add_parser("list", parents=[common])
    p.set_defaults(func=cmd_world_list)
    p = ws.add_parser("inspect", parents=[common], help="time, actors, orgs, balances, services, recent events")
    p.add_argument("world")
    p.add_argument("--recent", type=int, default=15)
    p.set_defaults(func=cmd_world_inspect)
    p = ws.add_parser("run", parents=[common], help="advance a persistent world (resumable, Ctrl-C safe)")
    p.add_argument("world")
    p.add_argument("--days", type=float, required=True)
    p.add_argument("--snapshot-every-days", type=float, default=0.0)
    p.add_argument("--step-hours", type=float, default=1.0)
    p.add_argument("--no-final-snapshot", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--exit-code-on-interrupt", action="store_true", help="exit 130 when interrupted")
    p.set_defaults(func=cmd_world_run)
    p = ws.add_parser("snapshot", parents=[common])
    p.add_argument("world")
    p.set_defaults(func=cmd_world_snapshot)
    p = ws.add_parser("fork", parents=[common], help="fork a snapshot (or a world's current state)")
    p.add_argument("snapshot")
    p.add_argument("--id", default=None)
    p.set_defaults(func=cmd_world_fork)
    p = ws.add_parser("replay", parents=[common], help="replay the event log from the store")
    p.add_argument("world")
    p.add_argument("--verify", action="store_true", help="verify the hash chain and the digest vs a fresh load")
    p.set_defaults(func=cmd_world_replay)
    p = ws.add_parser("verify", parents=[common], help="invariants + regeneration check")
    p.add_argument("world")
    p.add_argument("--no-regen", action="store_true")
    p.set_defaults(func=cmd_world_verify)

    # actor
    pa = sub.add_parser("actor", help="admit actors and run policies for them")
    as_ = pa.add_subparsers(dest="sub")
    p = as_.add_parser("join", parents=[common])
    p.add_argument("world")
    p.add_argument("handle")
    p.add_argument("--endowment", type=int, default=1_000_000)
    p.add_argument("--kind", default="agent", choices=["agent", "npc", "human"])
    p.set_defaults(func=cmd_actor_join)
    p = as_.add_parser("run", parents=[common])
    p.add_argument("world")
    p.add_argument("handle")
    p.add_argument("--policy", required=True, help="policy spec, e.g. builder:need=route.shortest")
    p.add_argument("--days", type=float, required=True)
    p.add_argument("--dt", type=int, default=3600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--join", action="store_true", help="admit the actor first if missing")
    p.add_argument("--endowment", type=int, default=1_000_000)
    p.add_argument("--snapshot-every-days", type=float, default=0.0)
    p.add_argument("--trajectory", default=None, help="export the trajectory (.jsonl or .npz)")
    p.set_defaults(func=cmd_actor_run)

    # experiment
    pe = sub.add_parser("experiment", help="goal experiments with the evaluation harness")
    es = pe.add_subparsers(dest="sub")
    p = es.add_parser("run", parents=[common])
    p.add_argument("--goal", required=True)
    p.add_argument("--world", required=True)
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--seeds", default="1")
    p.add_argument("--policies", required=True, help="comma-separated policy specs")
    p.add_argument("--max-days", type=float, default=None)
    p.add_argument("--dt", type=int, default=3600)
    p.add_argument("--no-stop", action="store_true", help="keep running after success")
    p.add_argument("--param", action="append", help="world param override key=value")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_experiment_run)

    # selfplay
    pl = sub.add_parser("selfplay", help="league / self-play among scripted or parametrised policies")
    ls = pl.add_subparsers(dest="sub")
    p = ls.add_parser("run", parents=[common])
    p.add_argument("--world", default="tiny")
    p.add_argument("--seeds", default="1")
    p.add_argument("--days", type=float, default=3.0)
    p.add_argument("--policies", default="builder:need=route.shortest:undercut=0.9,"
                                         "builder:need=route.shortest:undercut=0.7:advertise=0,"
                                         "composer:need=route.shortest")
    p.add_argument("--heldout", default="", help="held-out evaluation policies (never trained against)")
    p.add_argument("--mode", default="competition", choices=["competition", "cooperation", "mixed"])
    p.add_argument("--schedule", default="round_robin", choices=["round_robin", "pfsp", "uniform", "recent"])
    p.add_argument("--learner", default=None)
    p.add_argument("--seed", type=int, default=0, help="league sampling seed")
    p.add_argument("--matches", type=int, default=4)
    p.add_argument("--one-seating", action="store_true", help="do not swap seats in paired worlds")
    p.add_argument("--endowment", type=int, default=2_000_000)
    p.add_argument("--dt", type=int, default=3600)
    p.add_argument("--n-consumers", type=int, default=None)
    p.add_argument("--param", action="append")
    p.add_argument("--measured-cost", action="store_true", help="do not force the deterministic cost model")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_selfplay_run)

    # evaluate
    p = sub.add_parser("evaluate", parents=[common], help="evaluate a policy on the validation (or train) suite")
    p.add_argument("--policy", required=True)
    p.add_argument("--split", default="val", choices=["val", "train"],
                   help="the locked test split is only reachable through crucible.evaluation.harness")
    p.add_argument("--n-per", type=int, default=1)
    p.add_argument("--levels", default="0")
    p.add_argument("--goal-families", default=None)
    p.add_argument("--world-families", default=None)
    p.add_argument("--max-days", type=float, default=None)
    p.add_argument("--dt", type=int, default=3600)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_evaluate)

    # trace / report / counterfactual / emergence
    p = sub.add_parser("trace", parents=[common], help="explain an entity (org, host, account, person) or #seq")
    p.add_argument("world")
    p.add_argument("entity")
    p.add_argument("--chain", type=int, default=None, help="causal chain of event SEQ")
    p.add_argument("--since", type=int, default=0)
    p.add_argument("--timeline", type=int, default=15, help="latest timeline entries to show (0: none)")
    p.set_defaults(func=cmd_trace)
    p = sub.add_parser("report", help="markdown table from an experiment / league / evaluation JSON")
    p.add_argument("results")
    p.set_defaults(func=cmd_report)
    p = sub.add_parser("counterfactual", parents=[common], help="fork twice, intervene in one, compare")
    p.add_argument("world")
    p.add_argument("--undeploy", action="append")
    p.add_argument("--close-listings", action="append")
    p.add_argument("--suspend", action="append")
    p.add_argument("--set-param", action="append")
    p.add_argument("--agent", action="append", help="HANDLE=POLICY to run in both forks")
    p.add_argument("--days", type=float, default=2.0)
    p.add_argument("--dt", type=int, default=3600)
    p.add_argument("--check-determinism", action="store_true")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_counterfactual)
    p = sub.add_parser("emergence", parents=[common], help="emergence metrics report")
    p.add_argument("world")
    p.add_argument("--since", type=int, default=0)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_emergence)

    # servers
    p = sub.add_parser("dashboard", parents=[common], help="read-only researcher dashboard (loopback)")
    p.add_argument("world", nargs="?", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--allow-remote", action="store_true")
    p.add_argument("--experiments", nargs="*", help="experiment / league JSON files")
    p.add_argument("--trajectories", nargs="*", help="trajectory JSONL / NPZ files")
    p.add_argument("--operator", action="store_true", help="show private communication payloads")
    p.set_defaults(func=cmd_dashboard)
    p = sub.add_parser("serve", parents=[common], help="start the agent API HTTP server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--world", default=None)
    p.set_defaults(func=cmd_serve)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    if not getattr(a, "func", None):
        if a.cmd:
            ap.parse_args([a.cmd, "--help"])
        ap.print_help()
        return 2
    try:
        return int(a.func(a) or 0)
    except KeyboardInterrupt:
        print("[crucible] aborted", file=sys.stderr)
        return 130
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
