"""Run the SP-3.2b benchmark: every candidate on every capability; statistics and recommendations."""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

from . import CAPABILITIES, tasks
from ..service import binom_sf, peak_vram_mb, poisson_binom_sf, wilson

DEFAULT_LMS = ("HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct", "Qwen/Qwen3-0.6B",
               "Qwen/Qwen3-1.7B")
# capabilities whose use in the world needs generation; the others are decisions among options
GENERATIVE_PRIMARY = ("communication", "programming")
EFFECTIVE = {"accuracy": 0.6, "p_vs_chance": 0.01, "gen_wilson_lo": 0.4}
METHODS = ("mean", "sum", "label")


def data_dir() -> str:
    d = os.path.join(os.environ.get("CRUCIBLE_DATA", "/workspace/data/crucible"), "semantic", "capbench")
    os.makedirs(d, exist_ok=True)
    return d


def load_sets(seed: int = 0, caps=CAPABILITIES) -> dict[str, list[dict]]:
    """Build (or load the cached copy of) the evaluation sets. The cache key covers the generator's source."""
    src = open(tasks.__file__, "rb").read()
    key = hashlib.sha256(src + json.dumps([seed, list(caps)]).encode()).hexdigest()[:16]
    path = os.path.join(data_dir(), f"sets_{key}.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    sets = tasks.build_all(seed, caps)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(sets, fh)
    os.replace(tmp, path)
    return sets


def mc_stats(items: list[dict], preds: list[int]) -> dict:
    n = len(items)
    if n == 0:
        return {"n": 0}
    hits = [int(p == it["answer"]) for p, it in zip(preds, items)]
    k = sum(hits)
    lo, hi = wilson(k, n)
    chance = [it["chance"] for it in items]
    out = {"n": n, "correct": k, "accuracy": round(k / n, 4), "ci95": [round(lo, 4), round(hi, 4)],
           "chance": round(float(np.mean(chance)), 4), "p_vs_chance": float(f"{poisson_binom_sf(k, chance):.3g}")}
    held = [h for h, it in zip(hits, items) if it.get("held_out_template")]
    seen = [h for h, it in zip(hits, items) if not it.get("held_out_template")]
    if held and seen:
        out["held_out_templates"] = {"n": len(held), "accuracy": round(sum(held) / len(held), 4)}
        out["seen_templates"] = {"n": len(seen), "accuracy": round(sum(seen) / len(seen), 4)}
    pairs: dict[str, list[int]] = {}
    for h, it in zip(hits, items):
        if it.get("pair"):
            pairs.setdefault(it["pair"], []).append(h)
    if pairs:
        both = [int(all(v)) for v in pairs.values() if len(v) == 2]
        out["pairs_both_correct"] = {"n": len(both), "rate": round(sum(both) / max(1, len(both)), 4),
                                     "chance": round(float(np.mean([items[0]["chance"] ** 2])), 4)}
    return out


def gen_stats(items: list[dict], checks: list[dict]) -> dict:
    n = len(items)
    if n == 0:
        return {"n": 0}
    k = sum(1 for c in checks if c["ok"])
    lo, hi = wilson(k, n)
    out = {"n": n, "passed": k, "pass_rate": round(k / n, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    by: dict[str, list[int]] = {}
    for it, c in zip(items, checks):
        by.setdefault(it["template"], []).append(int(c["ok"]))
    out["by_template"] = {t: round(sum(v) / len(v), 4) for t, v in sorted(by.items())}
    return out


def _split(items, split, fmt):
    return [i for i in items if i["split"] == split and i["format"] == fmt]


def evaluate_lm(cand, sets: dict, caps=CAPABILITIES, log=print) -> dict:
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    res = {"model": cand.info(), "capabilities": {}}
    total_items, total_s, gen_tok, gen_s = 0, 0.0, 0, 0.0
    for cap in caps:
        items = sets[cap]
        r: dict = {}
        dev_mc, test_mc = _split(items, "dev", "mc"), _split(items, "test", "mc")
        if test_mc:
            dev_acc = {}
            for m in METHODS:
                p, s = cand.mc(dev_mc, m)
                dev_acc[m] = round(float(np.mean([int(a == it["answer"]) for a, it in zip(p, dev_mc)])), 4) if dev_mc else 0.0
                total_items += len(dev_mc)
                total_s += s
            method = max(METHODS, key=lambda m: (dev_acc[m], -METHODS.index(m)))
            preds, s = cand.mc(test_mc, method)
            total_items += len(test_mc)
            total_s += s
            r["mc"] = {**mc_stats(test_mc, preds), "method": method, "dev_accuracy_by_method": dev_acc,
                       "ms_per_item": round(1000 * s / max(1, len(test_mc)), 1)}
            if cap == "tool_choice":
                lex = [it["lexical_choice"] for it in test_mc]
                r["lexical_baseline"] = mc_stats(test_mc, lex)
        test_gen = _split(items, "test", "gen")
        if test_gen:
            texts, s, ntok = cand.gen(test_gen)
            gen_tok += ntok
            gen_s += s
            checks = [tasks.check_generation(it, t) for it, t in zip(test_gen, texts)]
            r["gen"] = {**gen_stats(test_gen, checks), "ms_per_item": round(1000 * s / len(test_gen), 1),
                        "tokens_per_s": round(ntok / s, 1) if s else None,
                        "examples": [{"id": it["id"], "ok": c["ok"], "text": t[:400]}
                                     for it, c, t in list(zip(test_gen, checks, texts))[:3]]}
        res["capabilities"][cap] = r
        log(f"  {cand.name} {cap}: " + json.dumps({k: (v.get('accuracy', v.get('pass_rate')) if isinstance(v, dict) else v)
                                                    for k, v in r.items()}))
    res["peak_vram_mb"] = peak_vram_mb("cuda")
    res["mc_ms_per_item"] = round(1000 * total_s / max(1, total_items), 1)
    res["gen_tokens_per_s"] = round(gen_tok / gen_s, 1) if gen_s else None
    res["service_usage"] = cand.svc.usage()
    return res


def evaluate_trained(cand, sets: dict, caps=CAPABILITIES, log=print) -> dict:
    import resource
    res = {"model": None, "capabilities": {}}
    for cap in caps:
        items = sets[cap]
        info = cand.fit(cap, [i for i in items if i["split"] == "train"], [i for i in items if i["split"] == "dev"])
        test_mc = _split(items, "test", "mc")
        r: dict = {"training": info}
        if test_mc and info.get("trained", True) is not False:
            preds, s = cand.mc(test_mc, cap)
            r["mc"] = {**mc_stats(test_mc, preds), "ms_per_item": round(1000 * s / max(1, len(test_mc)), 1)}
        if _split(items, "test", "gen"):
            r["gen"] = {"n": len(_split(items, "test", "gen")), "pass_rate": None,
                        "not_applicable": "this candidate cannot generate text or programs"}
        res["capabilities"][cap] = r
        log(f"  {cand.name} {cap}: mc {r.get('mc', {}).get('accuracy')} (train {info})")
    res["model"] = cand.info()
    res["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    return res


def primary(cap: str, r: dict) -> dict:
    """The capability's primary measurement for one model: generation where the capability needs it."""
    if cap in GENERATIVE_PRIMARY:
        g = r.get("gen") or {}
        if g.get("pass_rate") is None:
            return {"metric": "generation_pass_rate", "value": None, "effective": False,
                    "reason": g.get("not_applicable", "not measured")}
        eff = g["pass_rate"] >= EFFECTIVE["accuracy"] and g["ci95"][0] >= EFFECTIVE["gen_wilson_lo"]
        return {"metric": "generation_pass_rate", "value": g["pass_rate"], "ci95": g["ci95"], "effective": eff}
    m = r.get("mc") or {}
    if not m.get("n"):
        return {"metric": "accuracy", "value": None, "effective": False}
    eff = m["accuracy"] >= EFFECTIVE["accuracy"] and m["p_vs_chance"] <= EFFECTIVE["p_vs_chance"]
    return {"metric": "accuracy", "value": m["accuracy"], "ci95": m["ci95"], "p_vs_chance": m["p_vs_chance"],
            "effective": eff}


def recommend(results: list[dict], caps=CAPABILITIES) -> dict:
    """Per capability: the smallest effective model, and whether it is statistically indistinguishable from the best
    (overlapping 95% intervals)."""
    out = {}
    order = sorted(results, key=lambda r: r["model"]["params"])
    for cap in caps:
        rows = [(r["model"]["name"], r["model"]["params"], primary(cap, r["capabilities"].get(cap, {}))) for r in order]
        eff = [x for x in rows if x[2]["effective"]]
        best = max((x for x in rows if x[2]["value"] is not None), key=lambda x: x[2]["value"], default=None)
        if eff:
            s = eff[0]
            close = best is not None and s[2].get("ci95") and best[2].get("ci95") and s[2]["ci95"][1] >= best[2]["ci95"][0]
            out[cap] = {"recommended": s[0], "params": s[1], "value": s[2]["value"], "metric": s[2]["metric"],
                        "best": best[0] if best else None, "best_value": best[2]["value"] if best else None,
                        "within_best_interval": bool(close)}
        else:
            out[cap] = {"recommended": None, "reason": "no candidate met the effectiveness threshold",
                        "best": best[0] if best else None, "best_value": best[2]["value"] if best else None,
                        "metric": rows[0][2]["metric"] if rows else None}
    return out


def run(models=DEFAULT_LMS, caps=CAPABILITIES, seed: int = 0, device: str = "auto", encoders: bool = True,
        out: str | None = None, log=print) -> dict:
    from .models import EncoderCandidate, LMCandidate, NgramCandidate
    t0 = time.time()
    sets = load_sets(seed)
    sizes = {c: {sp: sum(1 for i in sets[c] if i["split"] == sp) for sp in tasks.SPLITS} for c in caps}
    log(f"sets: {json.dumps(sizes)} ({time.time() - t0:.0f}s)")
    results = []
    if encoders:
        for C in (NgramCandidate, EncoderCandidate):
            c = C(seed=seed)
            t1 = time.time()
            r = evaluate_trained(c, sets, caps, log)
            r["wall_s"] = round(time.time() - t1, 1)
            results.append(r)
            c.close()
    for mid in models:
        c = LMCandidate(mid, device=device)
        t1 = time.time()
        r = evaluate_lm(c, sets, caps, log)
        r["wall_s"] = round(time.time() - t1, 1)
        results.append(r)
        c.close()
    rep = {"schema": "crucible.v5.sp32b/1", "seed": seed, "capabilities": list(caps), "set_sizes": sizes,
           "effective_threshold": EFFECTIVE, "generative_primary": list(GENERATIVE_PRIMARY), "results": results,
           "recommendation": recommend(results, caps), "wall_s": round(time.time() - t0, 1)}
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(rep, fh, indent=1)
    return rep


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="SP-3.2b semantic model benchmark")
    ap.add_argument("--models", default=",".join(DEFAULT_LMS))
    ap.add_argument("--caps", default=",".join(CAPABILITIES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-encoders", action="store_true")
    ap.add_argument("--out", default=os.path.join(data_dir(), "sp32b_report.json"))
    a = ap.parse_args(argv)
    rep = run([m for m in a.models.split(",") if m], a.caps.split(","), a.seed, a.device, not a.no_encoders, a.out)
    print(json.dumps(rep["recommendation"], indent=1))
    print("report:", a.out)
    return 0
