"""Audit of the held-out scoring rule (``temporal_evaluation.event_truth``).

The rule used before this audit skipped an example at horizon h when its outcome was unknown at h
from its *label alone*: an interval label straddling h, or a right-censored label ending before h.
Both make being scored depend on the outcome, so the scored rows are biased:

* interval straddling h (polled events): occurrences between the last check before h and h are
  skipped, examples still absent at the next check are counted as negatives -> too few positives;
* right-censored before h (follow-up ended early, e.g. a check jittered to 2.85 d for h = 3 d):
  examples still pending are skipped, examples whose event happened before follow-up ended are kept
  -> too many positives.

The current rule skips an example whenever its *observation record* cannot settle the outcome at h,
whatever the outcome (``followed_s`` / ``checks_s``). This script compares the two:

  labels    per corpus, split, event and horizon: n scored and positive rate, old vs new rule
  pollonly  one lab run observed by two networks at once (event feed + polls, and polls only);
            on the same posts, the feed network's exact labels give the reference rate, the
            poll-only network shows what each rule does with interval-censored labels
  expA      Experiment A at reduced size: every arm's predictions scored with both rules
            (the per-horizon baseline is trained with each rule's targets)

    python scripts/temporal/audit_event_truth.py labels
    python scripts/temporal/audit_event_truth.py pollonly
    python scripts/temporal/audit_event_truth.py expA --corpus 0 --steps 300
    python scripts/temporal/audit_event_truth.py summary   # tables from all audit_event_truth_* files
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    EVENTS,
    HELD_HORIZONS,
    HORIZONS,
    RUNS,
    Timer,
    save_json,
    threads,
)

import agentrs.evaluation.temporal_baselines as tb  # noqa: E402
import agentrs.temporal.temporal_evaluation as te  # noqa: E402
from agentrs.sensors.socialweblab_source import lab_temporal_spec  # noqa: E402
from agentrs.temporal.delayed_outcome_ledger import DelayedOutcomeLedger  # noqa: E402
from agentrs.temporal.time_encoding import describe_duration  # noqa: E402
from agentrs.training.temporal_world_model_trainer import (
    TemporalTrainingDataBuilder,  # noqa: E402
)

NEW_RULE = te.event_truth


def old_event_truth(ex, name, h):
    """The rule before the fix: decided from the label alone (verbatim)."""
    lab = ex.events.get(name)
    if lab is None or lab.kind == "none":
        return None
    if lab.kind == "exact":
        return 1 if lab.b_s <= h + 1e-6 else 0
    if lab.kind == "interval":
        if lab.b_s <= h + 1e-6:
            return 1
        if lab.a_s >= h - 1e-6:
            return 0
        return None
    return 0 if lab.b_s >= h - 1e-6 else None


RULES = {"old": old_event_truth, "new": NEW_RULE}


def use_rule(name: str) -> None:
    te.event_truth = RULES[name]  # evaluate_temporal looks the function up at call time
    tb.event_truth = RULES[name]  # targets of the per-horizon baseline


def rates(exs, horizons, rule, ref=None):
    """{event: {horizon: {n, pos, rate[, ref_rate on the same rows]}}}"""
    out = {}
    for n in EVENTS:
        row = {}
        for h in horizons:
            ys = [(e, rule(e, n, h)) for e in exs]
            ys = [(e, y) for e, y in ys if y is not None]
            r = {"n": len(ys), "pos": sum(y for _, y in ys), "rate": (sum(y for _, y in ys) / len(ys)) if ys else None}
            if ref is not None:
                rr = [NEW_RULE(ref[e.action_id], n, h) for e, _ in ys if e.action_id in ref]
                rr = [y for y in rr if y is not None]
                r["ref_rate_same_rows"] = sum(rr) / len(rr) if rr else None
            row[describe_duration(h)] = r
        out[n] = row
    return out


def kinds(exs):
    k = {}
    for e in exs:
        for n in EVENTS:
            lab = e.events.get(n)
            if lab is not None:
                k[f"{n}:{lab.kind}"] = k.get(f"{n}:{lab.kind}", 0) + 1
    return k


def part_labels():
    res = {}
    for c in range(3):
        b = TemporalTrainingDataBuilder(DelayedOutcomeLedger(RUNS / f"corpus_s{c}" / "net" / "ledger.sqlite"),
                                        lab_temporal_spec())
        for split in ("test", "test_env"):
            exs = b.build(splits=[split], cutoffs=[0])
            res[f"c{c}:{split}"] = {"n_examples": len(exs), "label_kinds": kinds(exs),
                                    **{r: rates(exs, HORIZONS + HELD_HORIZONS, f) for r, f in RULES.items()}}
    save_json("audit_event_truth_labels.json", res)
    # compact table: pooled over corpora and both splits
    print("| event | horizon | old: n / positive rate | new: n / positive rate |")
    print("|---|---|---|---|")
    for n in EVENTS:
        for h in sorted(HORIZONS + HELD_HORIZONS):
            d = describe_duration(h)
            cell = {}
            for r in RULES:
                nn = sum(v[r][n][d]["n"] for v in res.values())
                pp = sum(v[r][n][d]["pos"] for v in res.values())
                cell[r] = f"{nn} / {pp / nn:.3f}" if nn else "0 / —"
            print(f"| {n} | {d}{'' if h in HORIZONS else ' (between)'} | {cell['old']} | {cell['new']} |")


def part_pollonly(seed=10, days=10.0, train_days=7.0):
    from agentrs.sensors.observation_network import NetworkConfig
    from agentrs.socialweblab.engine import LAB_EPOCH, SocialWebLab
    from agentrs.temporal.outcome_labels import polled_events
    from agentrs.temporal.system import build_network

    out = RUNS / f"audit_pollonly_s{seed}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    lab = SocialWebLab(out / "lab.db", seed=seed, n_users=80, lab_id=f"sched{seed}")
    t_train = LAB_EPOCH + train_days * 86400
    t_test = t_train + 1.5 * 86400

    def split(o):
        t = o.effective_action_time()
        return "train" if t < t_train else ("test" if t < t_test else "late")

    modes = {"feed": NetworkConfig(scheduling="fixed", forecasts_enabled=False),
             "poll_only": NetworkConfig(scheduling="fixed", forecasts_enabled=False, use_feed_events=False)}
    nets = {m: build_network(out / m, lab, None, c, split_fn=split) for m, c in modes.items()}
    for _ in range(int(days * 96)):
        lab.advance(900)
        for n in nets.values():
            n.step()
    spec = lab_temporal_spec()
    ex = {m: TemporalTrainingDataBuilder(n.ledger, spec).build(splits=["test"], cutoffs=[0]) for m, n in nets.items()}
    ref = {e.action_id: e for e in ex["feed"]}
    common = [e for e in ex["poll_only"] if e.action_id in ref]
    src = {a["source_id"] for a in nets["poll_only"].ledger.actions()}
    res = {"seed": seed, "days": days, "n_test": {m: len(v) for m, v in ex.items()}, "n_common": len(common),
           "polled_events": {s: sorted(polled_events(nets["poll_only"].ledger, s)) for s in src},
           "label_kinds": {m: kinds(v) for m, v in ex.items()},
           "reference_feed_new_rule": rates([ref[e.action_id] for e in common], HORIZONS + HELD_HORIZONS, NEW_RULE),
           **{f"poll_only_{r}": rates(common, HORIZONS + HELD_HORIZONS, f, ref) for r, f in RULES.items()}}
    save_json(f"audit_event_truth_pollonly_s{seed}.json", res)
    print("polled events:", res["polled_events"], "common test posts:", len(common))
    print("| event | horizon | reference (feed, exact times) | poll-only, old rule: n / rate (reference on those rows) "
          "| poll-only, new rule: n / rate |")
    print("|---|---|---|---|---|")
    for n in EVENTS:
        for h in sorted(HORIZONS + HELD_HORIZONS):
            d = describe_duration(h)
            g, o, w = res["reference_feed_new_rule"][n][d], res["poll_only_old"][n][d], res["poll_only_new"][n][d]

            def f(x):
                return "—" if x is None else f"{x:.3f}"
            print(f"| {n} | {d}{'' if h in HORIZONS else ' (between)'} | {g['n']} / {f(g['rate'])} | "
                  f"{o['n']} / {f(o['rate'])} ({f(o['ref_rate_same_rows'])}) | {w['n']} / {f(w['rate'])} |")


def part_expa(corpus: int, seed: int, steps: int, arms: str):
    import random

    import exp_a_temporal_vs_single_step as A

    from agentrs.evaluation.temporal_baselines import (
        DiscreteHorizonModel,
        cap_example,
        persistence_predict_fn,
    )
    from agentrs.models.backbone import BackboneConfig
    from agentrs.temporal.temporal_evaluation import (
        marginal_predict_fn,
        model_predict_fn,
    )
    from agentrs.training.common import GradientTrainer, OptimConfig
    from agentrs.training.temporal_world_model_trainer import (
        TemporalTrainConfig,
        TemporalWorldModelTrainer,
    )

    threads()
    train_upd, train0, test, test_env = A.load(RUNS / f"corpus_s{corpus}")
    sets = {"test": test, "test_env": test_env}
    cfg = TemporalTrainConfig(steps=steps, batch_size=16, seed=seed,
                              optim=OptimConfig(lr=2e-4, total_steps=steps, schedule="cosine", warmup_steps=30))
    res = {"corpus": corpus, "seed": seed, "steps": steps, "arms": {}}

    def score(fn, s=sets):
        out = {}
        for r in RULES:
            use_rule(r)
            out[r] = A.evaluate(fn, s)
        use_rule("new")
        return out

    for arm in arms.split(","):
        T = Timer()
        if arm == "marginal_km":
            ev = score(marginal_predict_fn(train0, EVENTS, ["responses", "views"]))
        elif arm == "discrete":
            ev = {}
            for r in RULES:  # targets follow the rule too
                use_rule(r)
                m = DiscreteHorizonModel(BackboneConfig(name="EleutherAI/pythia-14m", max_prompt_tokens=320, seed=seed),
                                         EVENTS, A.PROFILE)
                opt = GradientTrainer(m, cfg.optim)
                rng = random.Random(seed)
                for _ in range(steps):
                    loss, _ = m.loss(rng.sample(train0, 16))
                    opt.step(loss)
                m.eval()
                ev[r] = A.evaluate(m.predict_fn(), sets)
            use_rule("new")
        else:
            m = A.make_wm(seed)
            s = sets
            if arm == "temporal":
                data = train_upd
            elif arm == "no_content":
                data = [A.strip_content(e) for e in train_upd]
                s = {k: [A.strip_content(e) for e in v] for k, v in sets.items()}
            elif arm == "single_step":
                data = [cap_example(e, 300.0) for e in train0]
            TemporalWorldModelTrainer(m, cfg, name=arm).train(data)
            fn = model_predict_fn(m)
            if arm == "single_step":
                fn = persistence_predict_fn(fn, 300.0)
            ev = score(fn, s)
        res["arms"][arm] = {"eval": ev, "wall_s": T.s()}
        print(arm, T.s(), {r: {k: (round(v["observed_horizons"]["mean_brier"], 4),
                                   round(v["observed_horizons"]["mean_auroc"] or 0, 3),
                                   round(v["between_horizons"]["mean_brier"], 4)) for k, v in e.items()}
                           for r, e in ev.items()}, flush=True)
        save_json(f"audit_event_truth_expA_c{corpus}_s{seed}_steps{steps}.json", res)


def part_summary():
    """Markdown tables (old rule -> new rule, mean over corpora) from the audit_event_truth_* files."""
    import glob
    import json

    import numpy as np
    from common import RESULTS

    def load(pat):
        return [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(RESULTS / pat)))]

    def f(xs):
        xs = [x for x in xs if x is not None]
        return f"{np.mean(xs):.3f}" if xs else "—"

    for steps in sorted({int(p.rsplit("steps", 1)[1][:-5]) for p in glob.glob(str(RESULTS / "audit_event_truth_expA_*_steps*.json"))}):
        rs = load(f"audit_event_truth_expA_c*_s*_steps{steps}.json")
        print(f"\n### A ({len(rs)} corpora, {steps} steps) — old rule → new rule\n")
        print("| model | test Brier | test AUROC | between-horizon Brier | unseen env Brier | unseen env AUROC |")
        print("|---|---|---|---|---|---|")
        for arm in ("marginal_km", "single_step", "discrete", "temporal", "no_content"):
            vals = [r["arms"][arm]["eval"] for r in rs if arm in r["arms"]]
            if not vals:
                continue
            cells = []
            for split, key, k2 in (("test", "observed_horizons", "mean_brier"), ("test", "observed_horizons", "mean_auroc"),
                                   ("test", "between_horizons", "mean_brier"), ("test_env", "observed_horizons", "mean_brier"),
                                   ("test_env", "observed_horizons", "mean_auroc")):
                cells.append(" → ".join(f([v[r][split][key][k2] for v in vals]) for r in RULES))
            print(f"| {arm} | " + " | ".join(cells) + " |")
    for pat in sorted(glob.glob(str(RESULTS / "audit_event_truth_expB_audit*_s*.json"))):
        for r in load(Path(pat).name):
            print(f"\n### B ({Path(pat).stem.removeprefix('audit_event_truth_')}) — test Brier, old rule → new rule\n")
            print("| scheduling | observation requests | test Brier | gain over lookup table |")
            print("|---|---|---|---|")
            km = r["marginal_km"]
            print(f"| lookup table | — | {km['legacy_rule']['mean_brier']:.3f} → {km['mean_brier']:.3f} | — |")
            for m, v in r["modes"].items():
                e = v["eval"]
                print(f"| {m} | {v['observation_requests']} | {e['legacy_rule']['mean_brier']:.3f} → {e['mean_brier']:.3f} | "
                      f"{km['legacy_rule']['mean_brier'] - e['legacy_rule']['mean_brier']:+.3f} → "
                      f"{km['mean_brier'] - e['mean_brier']:+.3f} |")
    for exp in ("C", "D"):
        rs = load(f"audit_event_truth_exp{exp}_c*_s*.json")
        if not rs:
            continue
        sets = list(next(iter(rs[0]["arms"].values()))["eval"])
        print(f"\n### {exp} ({len(rs)} corpora, reduced size) — Brier / AUROC, old rule → new rule\n")
        print("| arm | " + " | ".join(sets) + " |")
        print("|---|" + "---|" * len(sets))
        for arm in rs[0]["arms"]:
            row = []
            for s_ in sets:
                ev = [r["arms"][arm]["eval"][s_] for r in rs if arm in r["arms"] and s_ in r["arms"][arm]["eval"]]
                b = " → ".join(f([e["legacy_rule"]["mean_brier"] for e in ev] if r_ == "old" else [e["mean_brier"] for e in ev])
                               for r_ in RULES)
                a = " → ".join(f([e["legacy_rule"]["mean_auroc"] for e in ev] if r_ == "old" else [e["mean_auroc"] for e in ev])
                               for r_ in RULES)
                row.append(f"{b} / {a}")
            print(f"| {arm} | " + " | ".join(row) + " |")
    for seed in (20, 21):
        old, new = load(f"audit_event_truth_old_expE_s{seed}.json"), load(f"audit_event_truth_new_expE_s{seed}.json")
        if not (old and new):
            continue
        o, n = old[0], new[0]
        print(f"\n### E seed {seed}: randomized posts scored {o['n_experiment']} → {n['n_experiment']}\n")
        for k in ("detailed", "high", "peak"):
            a, b = o["randomized"]["answered"][k], n["randomized"]["answered"][k]
            print(f"* randomized {k}: {a['effect']:.2f} [{a['ci95'][0]:.2f}, {a['ci95'][1]:.2f}] → "
                  f"{b['effect']:.2f} [{b['ci95'][0]:.2f}, {b['ci95'][1]:.2f}]")
        a, b = o["observational"], n["observational"]
        print(f"* naive observational content: {a['naive_detail_and_polite']['effect']:.2f} → "
              f"{b['naive_detail_and_polite']['effect']:.2f}; follower-stratified {a['adjusted_by_followers']:.2f} → "
              f"{b['adjusted_by_followers']:.2f}; naive audience {a['followers_effect_naive']['effect']:.2f} → "
              f"{b['followers_effect_naive']['effect']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["labels", "pollonly", "expA", "summary"])
    ap.add_argument("--corpus", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--arms", default="marginal_km,temporal,no_content,discrete,single_step")
    a = ap.parse_args()
    if a.part == "labels":
        part_labels()
    elif a.part == "summary":
        part_summary()
    elif a.part == "pollonly":
        part_pollonly(seed=10 if a.seed is None else a.seed)
    else:
        part_expa(a.corpus, a.corpus if a.seed is None else a.seed, a.steps, a.arms)


if __name__ == "__main__":
    main()
