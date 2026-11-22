"""Print Markdown tables from the experiment JSON files in this directory (so that RESULTS.md is transcribed from
measurements, not from memory).   python docs/experiments/render.py [name ...]"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    hits = sorted(glob.glob(os.path.join(HERE, name)))
    return [(os.path.basename(h), json.load(open(h))) for h in hits]


def pct(x):
    return "-" if x is None else f"{100 * x:.0f}%"


def ci(c):
    return "-" if not c else f"[{100 * c[0]:.0f}, {100 * c[1]:.0f}]"


def row(r):
    return f"{pct(r['success'])} {ci(r.get('success_ci'))} | {pct(r.get('valid_rate'))} | {r.get('episodes')}"


def b_multi():
    for name, d in load("exp_b_multi.json"):
        print(f"### Experiment B ({name}): held-out success before and after RL, {d['tasks']['n']} tasks, families {d['tasks']['world_families']}\n")
        arms = list(d["arms"])
        print("| seed | warm start (before RL) | " + " | ".join(arms) + " |\n|---|---|" + "---|" * len(arms))
        for sd, r in d["per_seed"].items():
            print(f"| {sd} | {pct(r['before_rl']['success'])} {ci(r['before_rl']['success_ci'])} | " + " | ".join(
                f"{pct(r['arms'][a]['after_rl']['success'])} {ci(r['arms'][a]['after_rl']['success_ci'])} ({100 * r['arms'][a]['delta_success']:+.0f})" if a in r["arms"] else "-" for a in arms) + " |")
        if all("before_rl_sampled" in r for r in d["per_seed"].values()):
            print("\nSampled actions (same tasks):\n\n| seed | warm start (before RL) | " + " | ".join(arms) + " |\n|---|---|" + "---|" * len(arms))
            for sd, r in d["per_seed"].items():
                print(f"| {sd} | {pct(r['before_rl_sampled']['success'])} {ci(r['before_rl_sampled']['success_ci'])} | " + " | ".join(
                    f"{pct(r['arms'][a]['after_rl_sampled']['success'])} {ci(r['arms'][a]['after_rl_sampled']['success_ci'])} ({100 * r['arms'][a]['delta_success_sampled']:+.0f})" if a in r["arms"] else "-" for a in arms) + " |")
        print()
        for k, v in d["aggregate"].items():
            if isinstance(v, dict) and "diff" in v and "n_cand" in v:
                print(f"* {k}: {100 * v['diff']:+.1f} points, 95% CI [{100 * v['ci'][0]:+.1f}, {100 * v['ci'][1]:+.1f}], P(>0) {v['p_improve']:.2f}, n={v['n_cand']} seeds, established={v.get('established')}")
        for k, v in d["aggregate"].items():
            if isinstance(v, dict) and "tasks_gained" in v:
                print(f"* {k}: {100 * v['diff']:+.1f} points, 95% CI [{100 * v['ci'][0]:+.1f}, {100 * v['ci'][1]:+.1f}] (bootstrap over seeds and tasks), "
                      f"{v['n_pairs']} task pairs: {v['tasks_gained']} gained, {v['tasks_lost']} lost, {v['tasks_unchanged']} unchanged, established={v['established']}")
        print("\n| task arm | before | " + " | ".join(arms) + " |\n|---|---|" + "---|" * len(arms))
        for arm, v in d["aggregate"].get("mean_success_by_task_arm", {}).items():
            print(f"| {arm} | {pct(v.get('before'))} | " + " | ".join(pct(v.get(a)) for a in arms) + " |")
        print("\n| seed | arm | in-training curve (decisions: success) | AULC | decisions | weights moved (L2, relative) | updates that changed weights | replay steps |\n|---|---|---|---|---|---|---|---|")
        for sd, r in d["per_seed"].items():
            for a in arms:
                if a in r["arms"]:
                    x = r["arms"][a]
                    print(f"| {sd} | {a} | {' '.join(f'{c[0]}:{pct(c[1])}' for c in x['in_training_curve'])} | {x['aulc_in_training']:.3f} | {x['env_decisions']} | "
                          f"{x['weights_l2_distance_from_warm_start']:.2f}, {100 * x['relative_change']:.2f}% | {x['updates_that_changed_weights']}/{x['rl_iterations']} | {x['replay_steps']} |")
        print()


def baselines():
    for name, d in load("baselines_*.json"):
        print(f"### Part 5 comparison ({name}): {d['tasks']['n']} held-out tasks, families {d['tasks']['world_families']}\n")
        print("| condition | success [95% CI] | valid actions | episodes | failures by class |\n|---|---|---|---|---|")
        for k in sorted(d["conditions"]):
            r = d["conditions"][k]
            f = ", ".join(f"{a} {b}" for a, b in sorted((r.get("failures") or {}).items(), key=lambda kv: -kv[1])[:4])
            print(f"| {k} | {pct(r['success'])} {ci(r.get('success_ci'))} | {pct(r.get('valid_rate'))} | {r.get('episodes')} | {f} |")
        print(f"\nmemory items {d.get('memory_items')}, agent tools {d.get('library_tools')}, parameter distance RL vs warm start {d.get('param_distance_rl_vs_warm_start')}\n")


def exp_c():
    for name, d in load("exp_c_*.json"):
        print(f"### Experiment C ({name}): transfer\n\n| evaluation set | success [95% CI] | valid actions | n |\n|---|---|---|---|")
        for k, r in d["zero_shot"].items():
            print(f"| {k} | {row(r)} |")
        for k, curve in d.get("adaptation", {}).items():
            print(f"\nadaptation {k}: " + ", ".join(f"{c['decisions']} decisions: {pct(c['success'])}" for c in curve))
        print()


def exp_d():
    for name, d in load("exp_d_retention.json"):
        pa = d["phase_A"]
        print(f"### Experiment D ({name}): retention\n\nA = {d['skill_set_A']}, B = {d['unrelated_set_B']}; iterations {d['iterations']}\n")
        print(f"A at warm start {pct(pa['A_at_warm_start']['success'])}; after learning A {pct(pa['A_after_learning_A']['success'])}; "
              f"B before any B training {pct(pa['B_before_any_B_training']['success'])}\n")
        print("| phase-B strategy | A after training on B | forgetting on A | B after training on B | gain on B |\n|---|---|---|---|---|")
        for k, r in d["strategies"].items():
            print(f"| {k} | {pct(r['A_after_training_on_B']['success'])} {ci(r['A_after_training_on_B']['success_ci'])} | {100 * r['forgetting_on_A']:+.0f} | "
                  f"{pct(r['B_after_training_on_B']['success'])} {ci(r['B_after_training_on_B']['success_ci'])} | {100 * r['gain_on_B']:+.0f} |")
        print()


def exp_d_clone():
    for name, d in load("exp_d_clone.json"):
        print(f"### Experiment D, supervised variant ({name}): A {pct(d['before']['A']['success'])}, B {pct(d['before']['B']['success'])} before; "
              f"demonstrations A {d['demonstrations']['A']}, B {d['demonstrations']['B']}\n")
        print("| phase-B training | A afterwards | forgetting on A | B afterwards | gain on B | weights moved (L2) |\n|---|---|---|---|---|---|")
        for k, r in d["strategies"].items():
            print(f"| {k} | {pct(r['A_after']['success'])} {ci(r['A_after']['success_ci'])} | {100 * r['forgetting_on_A']:+.0f} | "
                  f"{pct(r['B_after']['success'])} {ci(r['B_after']['success_ci'])} | {100 * r['gain_on_B']:+.0f} | {r['weights_moved_l2']:.2f} of {r['weights_norm']:.1f} |")
        print()


def exp_a():
    for name, d in load("exp_a_real_actions.json"):
        print(f"### Experiment A ({name}): building and operating a service from scratch, {len(d['tasks'])} tasks\n")
        print("| who acts | success [95% CI] | valid actions | n | LLM-written values | failures |\n|---|---|---|---|---|---|")
        for k, c in d["conditions"].items():
            r = c["summary"]
            f = ", ".join(f"{a} {b}" for a, b in sorted((r.get("failures") or {}).items(), key=lambda kv: -kv[1])[:4])
            print(f"| {k} | {row(r)} | {r.get('llm_gen')} | {f} |")
        print()


def worldmodel():
    for name, d in load("exp_worldmodel_*.json"):
        h = d["heldout"]
        print(f"### World model ({name}): {d['train_transitions']} training / {d['heldout_transitions']} held-out transitions\n")
        print("| prediction | model (held-out) | trivial predictor |\n|---|---|---|")
        print(f"| action accepted (accuracy) | {h['ok_accuracy']:.3f} | {h['ok_accuracy_base_rate']:.3f} (base rate) |")
        print(f"| action accepted (NLL) | {h['ok_nll']:.3f} | {h['ok_nll_base_rate']:.3f} |")
        print(f"| failure class (accuracy) | {h['fclass_accuracy']:.3f} | {h['fclass_accuracy_majority']:.3f} (majority) |")
        print(f"| reward (MSE) | {h['reward_mse']:.4f} | {h['reward_mse_mean_predictor']:.4f} (mean) |")
        print(f"| return-to-go (MSE) | {h['gain_mse']:.4f} | {h['gain_mse_mean_predictor']:.4f} (mean) |")
        print(f"| next context (cosine) | {h['next_latent_cosine']:.3f} | {h['next_latent_cosine_mean_predictor']:.3f} (mean vector) |")
        p = d.get("planning")
        if p:
            print(f"\nplanning: model-free {row(p['model_free'])}; one-step lookahead {row(p['with_one_step_lookahead'])}; difference {100 * p['difference']:+.1f} points\n")


def selfplay():
    for name, d in load("exp_selfplay*.json"):
        print(f"### Self-play ({name}): world {d.get('world', {'family': 'bazaar', 'seed': 31})}, {d['days']} simulated days\n\n| run | participant | live | listed | requests served | independent revenue (cents) |\n|---|---|---|---|---|---|")
        for label, r in d["runs"].items():
            for a, o in r["final"]["outcomes"].items():
                print(f"| {label} | {a} | {o['live_services']:.0f} | {o['listed_services']:.0f} | {o['requests_served']} | {o['independent_revenue']:.0f} |")
        print("\neffect of presence (with - without): " + json.dumps(d["effects_of_presence"]) + "\n")


def meta():
    for name, d in load("exp_e_meta.json"):
        print(f"### Meta-Researcher ({name}): {d['candidates_screened']} candidates screened ({d['valid_screened']} valid), {d['experiments']} taken further\n")
        print("champion baseline runs: " + "; ".join(f"{b['stage']} seed {b['seed']}: AULC {b['aulc_eval']:.3f}, held-out {b['first_eval']:.2f} -> {b['final_eval']:.2f}" for b in d["champion_baseline_runs"]) + "\n")
        print("| round | candidate | source | function(s) | changed lines | runs | does anything? | AULC - champion (1 seed) | picked | hypothesis / rejection |\n|---|---|---|---|---|---|---|---|---|---|")
        for x in d["screened"]:
            print(f"| {x['round']} | {x['candidate']} | {x['source']} | {', '.join(x['slots'])} | {x['diff_lines']} | {'yes' if x['valid'] else 'no'} | "
                  f"{'no (inert)' if x.get('inert') else 'yes' if x.get('inert') is False else '?'} | "
                  f"{'-' if x['effect_aulc'] is None else format(x['effect_aulc'], '+.3f')} | {'yes' if x['picked_by_surrogate'] else ''} | "
                  f"{(x['hypothesis'] if x['valid'] else x['reject_reason'] + ' / ' + x['hypothesis'])[:150].replace('|', '/')} |")
        print("\n| source | proposed | ran | inert | mean effect | best effect |\n|---|---|---|---|---|---|")
        for k, g in d["by_source"].items():
            print(f"| {k} | {g['n']} | {g['valid']} | {g.get('inert', '?')} | {'-' if g['mean_effect'] is None else format(g['mean_effect'], '+.3f')} | {'-' if g['best_effect'] is None else format(g['best_effect'], '+.3f')} |")
        nz = d.get("noise_from_inert_candidates") or {}
        if nz.get("n"):
            print(f"\ninert candidates (the champion's algorithm run again): {nz['n']}, effects {nz['effects']}, mean |effect| {nz['mean_abs']:.3f}, max {nz['max_abs']:.3f}; "
                  f"{nz['active_effects_beyond_max_noise']} of {nz['active_candidates']} active candidates had |effect| above that maximum")
        print("\n| candidate | source | function(s) | decision | stage | AULC difference [95% CI] | seeds | hypothesis |\n|---|---|---|---|---|---|---|---|")
        for r in d["table"]:
            c = r.get("ci")
            print(f"| {r['candidate']} | {r['source']} | {r['target']} | {r['decision']} | {r['stage']} | "
                  f"{'-' if r.get('diff_aulc') is None else format(r['diff_aulc'], '+.3f')} {'' if not c else '[%+.3f, %+.3f]' % tuple(c)} | {r.get('n')} | {r['hypothesis'][:90]} |")
        print("\n| round | surrogate fitted | value of surrogate picks | pool mean (random) | best of pool | valid fraction |\n|---|---|---|---|---|---|")
        for r in d["rounds"]:
            print(f"| {r['round']} | {r['surrogate_fitted']} | {r['value_surrogate']:+.3f} | {r['value_random_expected']:+.3f} | {r['value_best']:+.3f} | {r['valid_fraction']:.2f} |")
        so, sf = d.get("selection_online") or {}, d.get("selection_offline_full_feedback") or {}
        if so:
            a_ = so["advantage_over_random_all_rounds"]
            f_ = so["advantage_over_random_once_fitted"]
            r3 = lambda v: None if v is None else [round(x, 3) for x in v] if isinstance(v, list) else round(v, 3)      # noqa: E731
            print(f"\nonline selection (learns from its own picks): advantage over random, all rounds {a_['mean']:+.3f} {r3(a_['ci95'])} n={a_['n']}; "
                  f"once fitted {r3(f_['mean'])} {r3(f_['ci95'])} n={f_['n']}; established={so['established']}; leave-one-out {so.get('surrogate_loo')}")
        if sf.get("rounds"):
            a_ = sf["advantage_over_random"]
            print(f"\noffline replay with full feedback: advantage over random {a_['mean']:+.3f} [{a_['ci95'][0]:+.3f}, {a_['ci95'][1]:+.3f}] over {a_['n_rounds']} rounds; "
                  f"mean rank correlation predicted vs measured {sf['mean_spearman']:+.2f}; established={sf['established']}\n")
            for r in sf["rounds"]:
                sp = r['spearman_predicted_vs_measured']
                print(f"* round {r['round']}: trained on {r['trained_on']}, picked {r['picked']:+.3f}, random {r['random']:+.3f}, best {r['best']:+.3f}, rank correlation {'-' if sp is None else format(sp, '+.2f')}")
        print()


def final():
    for name, d in load("exp_final.json"):
        t, g = d["tasks"], d["aggregate"]
        print(f"### Sealed final list ({name}, seal {d['seal'][:12]}): learner `{d['learner']}` ({d['selection_rule'][:120]})\n")
        print(f"{t['attempted']} of {t['sealed']} sealed tasks attempted (arms {t['arms_attempted']}); not attempted: {t['arms_not_attempted']}\n")
        print("| seed | before RL (greedy) | after RL (greedy) | before RL (sampled) | after RL (sampled) |\n|---|---|---|---|---|")
        for sd, r in d["per_seed"].items():
            print(f"| {sd} | {pct(r['before_rl']['success'])} {ci(r['before_rl']['success_ci'])} | {pct(r['after_rl']['success'])} {ci(r['after_rl']['success_ci'])} | "
                  f"{pct(r['before_rl_sampled']['success'])} {ci(r['before_rl_sampled']['success_ci'])} | {pct(r['after_rl_sampled']['success'])} {ci(r['after_rl_sampled']['success_ci'])} |")
        for k in ("after - before (greedy)", "after - before (sampled)"):
            v = g[k]
            print(f"\n* {k}: {100 * v['diff']:+.1f} points, 95% CI [{100 * v['ci'][0]:+.1f}, {100 * v['ci'][1]:+.1f}], {v['n_pairs']} task pairs: "
                  f"{v['tasks_gained']} gained, {v['tasks_lost']} lost; established={v['established']}")
        print("\n| world family | n | before | after |\n|---|---|---|---|")
        for k, v in g["by_world_family"].items():
            print(f"| {k} | {v['n']} | {pct(v['before'])} | {pct(v['after'])} |")
        print()


ALL = {"b": b_multi, "baselines": baselines, "c": exp_c, "d": exp_d, "a": exp_a, "worldmodel": worldmodel, "selfplay": selfplay, "meta": meta, "final": final, "d_clone": exp_d_clone}

def fill(path: str) -> None:
    """Replace every block ``<!-- table:NAME -->...<!-- /table -->`` of a Markdown file with the tables printed by
    NAME's function, so that the tables of RESULTS.md are generated from the JSON files and never typed."""
    import contextlib
    import io
    import re
    text = open(path).read()

    def sub(m):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ALL[m.group(1)]()
        body = "\n".join(l for l in buf.getvalue().strip("\n").split("\n"))
        return f"<!-- table:{m.group(1)} -->\n{body}\n<!-- /table -->"
    new = re.sub(r"<!-- table:(\w+) -->.*?<!-- /table -->", sub, text, flags=re.S)
    open(path, "w").write(new)
    print(f"filled {len(re.findall(r'<!-- table:', new))} table blocks in {path}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--fill"]:
        fill(sys.argv[2])
    else:
        for k in (sys.argv[1:] or list(ALL)):
            ALL[k]()
