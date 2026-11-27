"""In-situ routing measurement of class-4 decisions (DECISION_LOG D-37, D-38 and its amendment; MASTER_SPEC 3.2
class 4: "use the smallest model or combination of models that meets each capability requirement").

What is measured: the participants' own decision pipeline (`mind.decide`: the next-step phrase, the service, the tool)
on the class-4 bench (`generalists.bench`), DEV templates only - the TEST templates stay held out for Gate H.4.
Two samples of the DEV templates: the selection sample (seed 0, the one D-37 used) and a fresh confirmation sample
(seed 1: another civilisation, persona and goals), 6 items per template each.

Rule (`routing.derive` with ``confirm``), fixed before this procedure ran for the first time (D-38 amendment):

1. **tool choice**: variants Qwen3-0.6B (options scored), Qwen3-0.6B reasoning first in its thinking mode for at
   most 192 / 384 tokens, Qwen3-1.7B, with the next-step phrase on its current route (situation interpretation's,
   Qwen3-1.7B), all measured on the selection sample. Candidates effective there (accuracy >= 0.6) are confirmed on
   the confirmation sample in cost order until one reaches 0.6 there too; that one is routed (a pick among four
   variants on 72 items is optimistic: winner's curse).
2. **next-step phrase** (`routing.USES["next_step"]`, the only part of situation interpretation this bench
   measures): with tool choice as routed in 1, Qwen3-0.6B vs Qwen3-1.7B by the same rule.
3. Goal formation and the choice among known identifiers stay on situation interpretation's SP-3.2b route
   (held-out templates: Qwen3-0.6B 0.28, Qwen3-1.7B 0.65); this bench does not measure them.

Every other route is the committed decision's. `measure` writes the measurements (JSON); `apply` writes a routing
decision file from them (`routing.DECISION_FILE` by default; in a snapshot when run unattended) - the decision is
then committed with the measurement's SHA-256.

    PYTHONPATH=src python -m crucible.slots ... -- python -m crucible.semantic.generalists.insitu measure --out M.json
    PYTHONPATH=src python -m crucible.semantic.generalists.insitu apply M.json [--decision routing.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time

SMALL, BIG = "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"
TOOL_VARIANTS = (SMALL, f"{SMALL}#think192", f"{SMALL}#think384", BIG)
NEXT_VARIANTS = (SMALL, BIG)
PER_TEMPLATE = 6
SELECTION_SEED, CONFIRM_SEED = 0, 1


def brain_for(tool_variant: str, next_model: str):
    """The participants' brain with tool choice and the next-step phrase on the given variants, every other route as
    committed."""
    from .. import routing
    from .brain import LMBrain
    routes = {c: routing.route(c) for c in routing.TABLE}
    routes["tool_choice"] = routing.base_model(tool_variant)
    routes["next_step"] = next_model
    k = routing.think_of(tool_variant)
    return LMBrain(routes, think={"tool_choice": k} if k else {})


def _eval(items, tool_variant: str, next_model: str, log=print) -> dict:
    from . import bench, mind as M
    t0 = time.time()
    res = bench.evaluate(items, brain_for(tool_variant, next_model), M.Config())
    d = res["splits"]["dev"]
    log(f"  tool={tool_variant} next_step={next_model}: {d['accuracy']} {d['ci95']} n={d['n']} "
        f"({time.time() - t0:.0f} s)")
    return {"accuracy": d["accuracy"], "ci95": d["ci95"], "n": d["n"], "host_recall": d["host_recall"],
            "by_template": d["by_template"], "wall_s": round(time.time() - t0, 1),
            "items": [{k: o[k] for k in ("id", "intent", "service", "pick", "correct")} for o in res["items"]]}


def _cost(v: str):
    from .. import routing
    return routing.PARAMS[routing.base_model(v)], routing.think_of(v)


def _confirm(sel: dict, conf_items, run) -> dict:
    """Confirmation accuracies of the effective candidates, in cost order, until one reaches the bar."""
    from .. import routing
    out = {}
    for v in sorted((v for v, m in sel.items() if m["accuracy"] >= routing.EFFECTIVE_ACCURACY), key=_cost):
        out[v] = run(conf_items, v)
        if out[v]["accuracy"] >= routing.EFFECTIVE_ACCURACY:
            break
    return out


def measure(log=print) -> dict:
    import torch
    torch.set_num_threads(1)
    from .. import routing
    from . import bench
    t0 = time.time()
    sel_items = bench.build(SELECTION_SEED, PER_TEMPLATE, ("dev",))
    conf_items = bench.build(CONFIRM_SEED, PER_TEMPLATE, ("dev",))
    log(f"items: selection {len(sel_items)}, confirmation {len(conf_items)} ({time.time() - t0:.0f} s)")
    dec = routing.decision() or {}
    nxt_route = routing.route("situation_interpretation")
    log("tool choice, selection sample")
    tool_sel = {v: _eval(sel_items, v, nxt_route, log) for v in TOOL_VARIANTS}
    log("tool choice, confirmation sample")
    tool_conf = _confirm(tool_sel, conf_items, lambda items, v: _eval(items, v, nxt_route, log))
    acc = lambda d: {v: m["accuracy"] for v, m in d.items()}          # noqa: E731
    pick = routing.derive(dec["report"], {"tool_choice": acc(tool_sel)}, {"tool_choice": acc(tool_conf)})["tool_choice"]
    tv = pick["model"] + (f"#think{pick['think_tokens']}" if pick.get("think_tokens") else "")
    log(f"tool choice routed: {tv} ({pick['reason']})")
    log("next-step phrase, selection sample")
    nxt_sel = {v: (tool_sel[tv] if v == nxt_route else _eval(sel_items, tv, v, log)) for v in NEXT_VARIANTS}
    known = {nxt_route: tool_conf[tv]} if tv in tool_conf else {}
    log("next-step phrase, confirmation sample")
    nxt_conf = {}
    for v in sorted((v for v, m in nxt_sel.items() if m["accuracy"] >= routing.EFFECTIVE_ACCURACY), key=_cost):
        nxt_conf[v] = known.get(v) or _eval(conf_items, tv, v, log)
        if nxt_conf[v]["accuracy"] >= routing.EFFECTIVE_ACCURACY:
            break
    return {"procedure": "crucible.semantic.generalists.insitu (D-38 amendment)", "date": time.strftime("%Y-%m-%d"),
            "samples": {"selection": {"seed": SELECTION_SEED, "n": len(sel_items)},
                        "confirmation": {"seed": CONFIRM_SEED, "n": len(conf_items)}},
            "routes_fixed": {c: routing.route(c) for c in routing.TABLE if c != "tool_choice"},
            "tool_choice": {"next_step_route": nxt_route, "selection": tool_sel, "confirmation": tool_conf,
                            "routed": tv},
            "next_step": {"tool_choice_route": tv, "selection": nxt_sel, "confirmation": nxt_conf},
            "wall_s": round(time.time() - t0, 1)}


def apply(raw: bytes, path: str | None = None) -> dict:
    """Write the routing decision derived from a measurement (`measure`'s JSON bytes) to `path`."""
    from .. import routing
    m = json.loads(raw)
    sha = hashlib.sha256(raw).hexdigest()
    dec = dict(routing.decision() or {})
    in_situ, confirm = dict(dec.get("in_situ") or {}), dict(dec.get("in_situ_confirm") or {})
    src = dict(dec.get("in_situ_source") or {})
    acc = lambda d: {v: x["accuracy"] for v, x in d.items()}          # noqa: E731
    ci = lambda d: {v: x["ci95"] for v, x in d.items()}               # noqa: E731
    n = {k: v["n"] for k, v in m["samples"].items()}
    for cap, what in (("tool_choice", "tool choice varied over Qwen3-0.6B (options scored), Qwen3-0.6B reasoning "
                                      "first in its thinking mode (at most 192 / 384 tokens), Qwen3-1.7B; next-step "
                                      f"phrase on {m['tool_choice']['next_step_route']}"),
                      ("next_step", "the next-step phrase (what the service directory is searched for) varied over "
                                    f"Qwen3-0.6B, Qwen3-1.7B; tool choice as routed ({m['next_step']['tool_choice_route']})"
                                    "; goal formation and argument choice are not part of this bench and stay on "
                                    "situation interpretation's SP-3.2b route")):
        in_situ[cap] = acc(m[cap]["selection"])
        confirm[cap] = acc(m[cap]["confirmation"])
        src[cap] = {"what": "class-4 decisions in a civilisation (crucible.semantic.generalists.bench, DEV templates "
                            f"only, {PER_TEMPLATE} items per template): {what}; candidates effective on the selection "
                            f"sample (seed {m['samples']['selection']['seed']}) confirmed in cost order on a fresh "
                            f"sample (seed {m['samples']['confirmation']['seed']})",
                    "n_items": n, "ci95": {"selection": ci(m[cap]["selection"]),
                                           "confirmation": ci(m[cap]["confirmation"])},
                    "sha256": sha, "date": m["date"], "decision": "D-38 (amendment)",
                    "procedure": "python -m crucible.semantic.generalists.insitu measure"}
    dec.update(in_situ=in_situ, in_situ_confirm=confirm, in_situ_source=src)
    dec["routes"] = routing.derive_decision(dec)
    with open(path or routing.DECISION_FILE, "w") as fh:
        json.dump(dec, fh, indent=1)
        fh.write("\n")
    return {c: (r["model"], r.get("think_tokens", 0), r["reason"]) for c, r in dec["routes"].items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m crucible.semantic.generalists.insitu")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("measure")
    a.add_argument("--out", required=True)
    b = sub.add_parser("apply")
    b.add_argument("measurement")
    b.add_argument("--decision", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "measure":
        import crucible
        crucible.load_all_modules()
        from .. import routing
        routing.limit_gpu_memory()
        res = measure(lambda s: print(s, flush=True))
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print("wrote", args.out)
        return 0
    with open(args.measurement, "rb") as fh:
        raw = fh.read()
    for c, r in apply(raw, args.decision).items():
        print(c, *r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
