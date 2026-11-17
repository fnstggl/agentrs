"""Experiment report (JSON + Markdown) built only from recorded evaluations and counters."""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path


def _fmt(r):
    if r is None:
        return "-"
    return f"{r['success']}/{r['n']} = {r['rate']:.3f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]"


def _decline_baseline(run_dir: Path, tag: str, split: str) -> dict:
    """Trivial 'always decline' policy on support tasks of an evaluation set (from the stored tasks)."""
    from agentrs.econ.core import REGISTRY, load_all

    load_all()
    p = run_dir / "experience" / f"eval_{tag}_{split}.jsonl.gz"
    if not p.exists():
        return {}
    n = k = 0
    with gzip.open(p, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["domain"] != "support":
                continue
            fam = REGISTRY[f"support/{r['family']}"]
            t = fam.make_task(r["split"], r["seed"])
            n += 1
            k += t.private["expect"]["outcome"] == "declined"
    return {"support_tasks": n, "always_decline_successes": k}


def build_report(run_dir: Path, state: dict) -> dict:
    evals = state["evals"]
    by = {(e["tag"], e["split"]): e for e in evals}
    stages = [("A_pretrained", "A: pretrained Qwen3-4B (no updates)"), ("B_after_bc", "B: after imitation (BC)"),
              ("C_final", "C: after RL (final weights)"), ("C_best", "C*: best-validation checkpoint")]
    lines = ["# AGENTRS overnight multi-domain continual-RL run", ""]
    t0, t1 = state["started_at"], state.get("finished_at", time.time())
    hours = (t1 - t0) / 3600
    dph = state["config"].get("dph_usd") or 0.0
    lines += [f"- started: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(t0))}; wall-clock: {hours:.2f} h; "
              f"stop reason: {state.get('stop_reason', '-')}",
              f"- policy: {state['models']['policy']} @ {state['models']['policy_revision']} + LoRA r16 (shared across all domains); "
              f"world model / critic: {state['models']['small']} + LoRA",
              f"- code: {state['git']}",
              f"- estimated instance cost for the run: ${hours * dph:.2f} (at ${dph:.3f}/h; GPU-hours {hours:.2f})", ""]
    c = state["counters"]
    lines += ["## Training activity", "",
              f"- RL cycles: {c['cycles']}; training task attempts: {c['attempts']}; verified successes: {c['successes']}",
              f"- policy gradient updates (PPO optimizer steps): {c['policy_updates']}; BC steps: {c['bc_updates']}; "
              f"world-model steps: {c['wm_updates']}; critic steps: {c['critic_updates']}",
              f"- rejected stale (off-policy) samples: {c['rejected_stale_samples']}; environment errors: {c['env_errors']}",
              f"- unique task fingerprints: {len(state['fingerprints'])} over {sum(state['fingerprints'].values())} sampled "
              f"tasks (near-duplicate rate {1 - len(state['fingerprints']) / max(1, sum(state['fingerprints'].values())):.3f})",
              f"- BC families: {state.get('bc_families')}; BC stats: {state.get('bc')}",
              f"- cost per verified training success: "
              f"{'$%.3f' % (hours * dph / c['successes']) if c['successes'] and dph else '-'}", ""]
    if c["by_domain"]:
        lines += ["Training rollouts by domain (sampled at T=1, includes exploration):", "",
                  "| domain | attempts | successes | rate |", "|---|---|---|---|"]
        for d, v in sorted(c["by_domain"].items()):
            lines.append(f"| {d} | {v['attempts']} | {v['successes']} | {v['successes'] / max(1, v['attempts']):.3f} |")
        lines.append("")
    lines += ["## Sealed test set (never used for training, task generation or selection)", "",
              "Greedy decoding; Wilson 95% intervals. Training families vs held-out (transfer) families.", "",
              "| stage | all families | training families | held-out families |", "|---|---|---|---|"]
    for tag, name in stages:
        e = by.get((tag, "test"))
        if e:
            lines.append(f"| {name} | {_fmt(e['summary']['overall'])} | {_fmt(e['train_families'])} | {_fmt(e['heldout_families'])} |")
        else:
            lines.append(f"| {name} | not run | | |")
    lines.append("")
    tests = [(tag, by[(tag, 'test')]) for tag, _ in stages if (tag, "test") in by]
    if tests:
        doms = sorted({d for _, e in tests for d in e["summary"]["by_domain"]})
        lines += ["### Sealed test by domain", "", "| domain | " + " | ".join(t for t, _ in tests) + " |",
                  "|---" * (len(tests) + 1) + "|"]
        for d in doms:
            lines.append(f"| {d} | " + " | ".join(_fmt(e["summary"]["by_domain"].get(d)) for _, e in tests) + " |")
        fams = sorted({f for _, e in tests for f in e["summary"]["by_family"]})
        lines += ["", "### Sealed test by family", "", "| family | held-out | " + " | ".join(t for t, _ in tests) + " |",
                  "|---" * (len(tests) + 2) + "|"]
        for f in fams:
            ho = "yes" if f in state["heldout_families"] else ""
            lines.append(f"| {f} | {ho} | " + " | ".join(
                (lambda r: f"{r['success']}/{r['n']}" if r else "-")(e["summary"]["by_family"].get(f)) for _, e in tests) + " |")
        lines.append("")
    lines += ["## Validation curve (used for promotion / best-checkpoint selection)", "",
              "| eval | cycle | version | all | training families | held-out families | WM valid-acc (majority) | WM outcome-acc (majority) | critic EV |",
              "|---|---|---|---|---|---|---|---|---|"]
    for e in evals:
        if e["split"] != "val":
            continue
        wm = e.get("world_model") or {}
        cr = e.get("critic") or {}
        lines.append(f"| {e['tag']} | {e['cycle']} | {e['version']} | {_fmt(e['summary']['overall'])} | {_fmt(e['train_families'])} | "
                     f"{_fmt(e['heldout_families'])} | "
                     f"{wm.get('valid_acc', float('nan')):.3f} ({wm.get('valid_majority', float('nan')):.3f}) | "
                     f"{wm.get('outcome_acc', float('nan')):.3f} ({wm.get('outcome_majority', float('nan')):.3f}) | "
                     f"{cr.get('explained_var') if cr.get('explained_var') is None else round(cr['explained_var'], 3)} |")
    lines += ["", "## Promotion decisions", ""]
    for p in state["promotions"]:
        lines.append(f"- cycle {p['cycle']}: {p['decision']} (val {p['val']:.3f} vs promoted {p['promoted_val']:.3f})"
                     + (" — new best" if p.get("new_best") else "") + (f" — lr -> {p['new_lr']}" if p.get("new_lr") else ""))
    lines += ["", "## RL cycles (training rollouts, T=1)", "",
              "| cycle | success | valid | informative groups | samples | updates | clipfrac | approx KL | re-score mismatch | WM loss | critic loss |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in state["cycles"]:
        p = r["ppo"]
        lines.append(f"| {r['cycle']} | {r['success_rate']:.3f} | {r['valid_action_rate']:.3f} | {r['informative_groups']}/{r['groups']} | "
                     f"{p['samples']} | {p['updates']} | {p['clipfrac']:.3f} | {p['kl_approx']:.4f} | "
                     f"{p['mismatch_first'] if p['mismatch_first'] is None else round(p['mismatch_first'], 5)} | "
                     f"{r['wm_loss'] if r['wm_loss'] is None else round(r['wm_loss'], 3)} | "
                     f"{r['critic_loss'] if r['critic_loss'] is None else round(r['critic_loss'], 3)} |")
    trivial = _decline_baseline(run_dir, "A_pretrained", "test")
    lines += ["", "## Trivial baselines and caveats", "",
              f"- 'always decline' on the sealed-test support tickets would solve {trivial.get('always_decline_successes', '-')}"
              f" of {trivial.get('support_tasks', '-')} support tasks.",
              "- Customer support is a τ²-style CUSTOM environment with a scripted customer, not τ²-Bench.",
              "- Per-family test denominators are small (see tables); differences within the confidence intervals are not evidence.",
              "- D (world-model-assisted planning) and E (internet-derived auxiliary learning for the policy) were NOT run "
              "as policy ablations unless listed in the stage table above.", ""]
    live = run_dir / "live" / "status.json"
    if live.exists():
        try:
            ls = json.loads(live.read_text())
            lines += ["## Live internet observation (separate process)", "",
                      "```", json.dumps({k: ls.get(k) for k in ("uptime_s", "optimizer_steps", "updates", "posts", "observations",
                                                                  "targets")}, indent=1, default=str)[:3000], "```", ""]
        except (OSError, ValueError):
            pass
    gh = run_dir / "github" / "status.json"
    if gh.exists():
        lines += ["## GitHub engineering-work observation", "", "```", gh.read_text()[:2000], "```", ""]
    if state["errors"]:
        lines += ["## Errors", ""] + [f"- {time.strftime('%H:%M', time.gmtime(e['time']))} {e['error'][:300]}" for e in state["errors"][-20:]]
    js = {"state_summary": {k: state[k] for k in ("started_at", "deadline", "phase", "cycle", "version", "counters", "best",
                                                  "promoted", "bc_families", "models", "git", "config")},
          "evals": evals, "cycles": state["cycles"], "promotions": state["promotions"], "trivial_baselines": trivial,
          "stop_reason": state.get("stop_reason")}
    return {"json": js, "md": "\n".join(lines) + "\n"}
