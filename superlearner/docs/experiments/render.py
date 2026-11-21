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
        print()
        for k, v in d["aggregate"].items():
            if isinstance(v, dict) and "diff" in v:
                print(f"* {k}: {100 * v['diff']:+.1f} points, 95% CI [{100 * v['ci'][0]:+.1f}, {100 * v['ci'][1]:+.1f}], P(>0) {v['p_improve']:.2f}, n={v['n_cand']} seeds, established={v.get('established')}")
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
    for name, d in load("exp_selfplay.json"):
        print(f"### Self-play ({name}): {d['days']} simulated days\n\n| run | participant | live | listed | requests served | independent revenue (cents) |\n|---|---|---|---|---|---|")
        for label, r in d["runs"].items():
            for a, o in r["final"]["outcomes"].items():
                print(f"| {label} | {a} | {o['live_services']:.0f} | {o['listed_services']:.0f} | {o['requests_served']} | {o['independent_revenue']:.0f} |")
        print("\neffect of presence (with - without): " + json.dumps(d["effects_of_presence"]) + "\n")


def meta():
    for name, d in load("exp_e_meta.json") + load("exp_f_meta.json"):
        print(f"### Meta-Researcher ({name}): {d['experiments']} staged experiments\n\n| candidate | source | function(s) | decision | stage | AULC difference [95% CI] | seeds | hypothesis |\n|---|---|---|---|---|---|---|---|")
        for r in d["table"]:
            c = r.get("ci")
            print(f"| {r['candidate']} | {r['source']} | {r['target']} | {r['decision']} | {r['stage']} | "
                  f"{'-' if r.get('diff_aulc') is None else format(r['diff_aulc'], '+.3f')} {'' if not c else '[%+.3f, %+.3f]' % tuple(c)} | {r.get('n')} | {r['hypothesis'][:90]} |")
        print()


ALL = {"b": b_multi, "baselines": baselines, "c": exp_c, "d": exp_d, "a": exp_a, "worldmodel": worldmodel, "selfplay": selfplay, "meta": meta}

if __name__ == "__main__":
    for k in (sys.argv[1:] or list(ALL)):
        ALL[k]()
