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


# ============================================================================================ continual learning
def _cl():
    hits = load("continual_lifetime.json")
    return hits[0][1] if hits else None


def _p(x, d=0):
    return "-" if x is None else f"{100 * x:.{d}f}%"


def _seeds(v, d=0):
    return "-" if not v else "(" + ", ".join("-" if x is None else f"{100 * x:.{d}f}" for x in v) + ")"


def _cmp(c):
    if not c:
        return "-"
    return f"{100 * c['diff']:+.1f} [{100 * c['ci95'][0]:+.1f}, {100 * c['ci95'][1]:+.1f}]"


def life():
    d = _cl()
    if not d:
        return
    C = d["conditions"]
    print("Held-out success after the whole lifetime (greedy actions; per-seed values in brackets). *Final, six lists*: mean over "
          "the six held-out lists with the final weights. *Taught five*: the five phases that came with demonstrations. "
          "*Right after learning*: each phase's own list at the end of that phase. *Forgetting*: for every earlier phase, its best "
          "success at any time minus its success at the end. *Unseen worlds*: the same arms in two kinds of world no phase trained in.\n")
    print("| condition | seeds | final, six lists | taught five | sampled actions | right after learning | forgetting | unseen worlds | decisions |\n|---|---|---|---|---|---|---|---|---|")
    for c in d["order"]:
        s = C[c]
        print(f"| {d['labels'][c]} | {len(s['seeds'])} | **{_p(s['final_mean']['mean'])}** {_seeds(s['final_mean']['by_seed'])} | {_p(s['final_mean_taught']['mean'])} | "
              f"{_p(s['final_mean_sampled']['mean'])} | {_p(s['right_after_learning']['mean'])} | {_p(s['forgetting']['mean'])} {_seeds(s['forgetting']['by_seed'])} | "
              f"{_p(s['unseen_worlds_final']['mean'])} | {s['env_decisions']:,.0f} |")
    names = C[d["order"][0]]["phases"]
    print("\nFinal success by phase:\n\n| condition | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
    for c in d["order"]:
        print(f"| {d['labels'][c]} | " + " | ".join(_p(x) for x in C[c]["final_by_phase"]) + " |")
    print("\nThe *shift* phase (four earlier arms in two kinds of world they were not learned in, practice only, no demonstrations): "
          "success on its held-out list just before and after the phase.\n\n| condition | before | after | per seed (before, after) |\n|---|---|---|---|")
    for c in d["order"]:
        sh = C[c]["shift"]
        print(f"| {d['labels'][c]} | {_p(sh['before'])} | {_p(sh['after'])} | " + ", ".join(f"({100 * a:.0f}, {100 * b:.0f})" for a, b in sh["by_seed"]) + " |")
    mem = [c for c in d["order"] if "final_mean_memory_off" in C[c]]
    if mem:
        print("\nConditions with episodic memory, final weights evaluated with the memory and with it switched off:\n\n"
              "| condition | with its memory | memory off | memory records | agent-made tools |\n|---|---|---|---|---|")
        for c in mem:
            s = C[c]
            print(f"| {d['labels'][c]} | {_p(s['final_mean']['mean'])} | {_p(s['final_mean_memory_off']['mean'])} | {s['memory_items_final']:.0f} | {s['tools_final']:.0f} |")
    print("\nDifferences between conditions on the same held-out tasks (final weights, all six lists; percentage points; 95% interval "
          "from a bootstrap over seeds and tasks):\n\n| comparison | seeds | tasks | difference | tasks only the first solves / only the second | unseen worlds |\n|---|---|---|---|---|---|")
    for k, v in d["comparisons"].items():
        a, u = v["final, all six lists"], v.get("final, unseen worlds") or {}
        if not a:
            continue
        x, y = k.split(" - ")
        print(f"| {d['labels'][x]} minus {d['labels'][y]} | {a['seeds']} | {a['n_tasks']} | {_cmp(a)} | {a['a_only']} / {a['b_only']} | {_cmp(u)} |")
    print()


def life_matrix():
    d = _cl()
    if not d:
        return
    for c in d["order"]:
        s = d["conditions"][c]
        names = s["phases"]
        print(f"**{d['labels'][c]}** (mean of {len(s['seeds'])} seeds; row: after phase, column: held-out list)\n\n| after | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
        for k, row in enumerate(s["matrix_mean"]):
            print(f"| {names[k]} | " + " | ".join((f"**{_p(x)}**" if j == k else _p(x)) for j, x in enumerate(row)) + " |")
        print()


def life_backbone():
    d = _cl()
    if not d:
        return
    C = d["conditions"]
    rows = [c for c in ("consolidate", "backbone_lora", "backbone_top4", "backbone_full") if c in C]
    print("The same learner and update rule (condition 4b) with four amounts of backbone plasticity.\n\n"
          "| backbone | trainable backbone parameters | moved from pretrained (relative L2) | seeds | final, six lists | right after learning | forgetting | unseen worlds | shift before -> after | update time per lifetime | difference to frozen (paired) |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for c in rows:
        s = C[c]
        b = s.get("backbone") or {}
        cmp_ = (d["comparisons"].get(f"{c} - consolidate") or {}).get("final, all six lists")
        rel = b.get("relative_change_from_pretrained")
        print(f"| {b.get('mode', '?')} | {b.get('trainable', 0):,} | {'-' if not rel else format(rel, '.2e')} | {len(s['seeds'])} | **{_p(s['final_mean']['mean'])}** {_seeds(s['final_mean']['by_seed'])} | "
              f"{_p(s['right_after_learning']['mean'])} | {_p(s['forgetting']['mean'])} | {_p(s['unseen_worlds_final']['mean'])} | {_p(s['shift']['before'])} -> {_p(s['shift']['after'])} | "
              f"{s['wall_s']['update_s'] / 60:.0f} min | {_cmp(cmp_) if c != 'consolidate' else 'reference'} |")
    sw = [(c, C[c]["backbone_swap"]) for c in rows if C[c].get("backbone_swap")]
    if sw:
        print("\nSwap test: the final controller evaluated on the six held-out lists with its trained backbone, and again with the "
              "backbone put back to its pretrained weights (all text embeddings recomputed). What disappears was carried by the "
              "backbone's changed weights.\n\n| backbone | seeds | with its trained backbone | with the pretrained backbone | difference (paired by task) | by phase (trained / pretrained) |\n|---|---|---|---|---|---|")
        for c, w in sw:
            print(f"| {C[c]['backbone']['mode']} | {w['seeds']} | {_p(w['trained'])} | {_p(w['pretrained'])} | {_cmp(w['paired'])} | "
                  + ", ".join(f"{n} {100 * a:.0f}/{100 * b:.0f}" for n, (a, b) in w["by_phase"].items()) + " |")
    print()


def life_plasticity():
    d = _cl()
    if not d:
        return
    C = d["conditions"]
    names = C[d["order"][0]]["phases"]
    f2 = lambda x: "-" if x is None else f"{x:.2f}"      # noqa: E731
    print("Plasticity probe: a copy of the controller is trained for 40 steps to choose arbitrary fixed targets on fixed probe decisions; the "
          f"loss it reaches (lower = easier to train; choosing at random gives {f2(C[d['order'][0]]['plasticity']['probe_chance'])}). Measured after every phase.\n\n"
          "| condition | " + " | ".join(names) + " | dormant units (max) | effective rank of the state, first -> last |\n|---|" + "---|" * (len(names) + 2))
    for c in d["order"]:
        pl = C[c]["plasticity"]
        dm = [x for x in pl["dormant_fraction"] if x is not None]
        er = pl["state_effective_rank"]
        print(f"| {d['labels'][c]} | " + " | ".join(f2(x) for x in pl["probe_loss_after_40_steps"]) + f" | {_p(max(dm) if dm else None, 1)} | {f2(er[0])} -> {f2(er[-1])} |")
    print("\nWorld model, held-out transitions (episodes of exploratory behaviour that are never trained on; all phases seen so far): share of "
          "actions whose acceptance it predicts correctly, against always predicting the majority outcome; and the same for this "
          "iteration's NEW transitions before any update has seen them (predict, then train).\n\n"
          "| condition | held-out accuracy at the end | majority | first phase's transitions, after phase 1 -> after phase 6 | next-state cosine (mean predictor) | new transitions, first iteration | new transitions, all later iterations (majority) |\n|---|---|---|---|---|---|---|")
    for c in d["order"]:
        s = C[c]
        w = s["world_model_heldout"][-1]
        f = s["world_model_first_phase_over_time"]
        pq = s.get("world_model_prequential") or {}
        print(f"| {d['labels'][c]} | {_p(w['ok_accuracy'])} | {_p(w['majority'])} | {_p(f[0])} -> {_p(f[-1])} | {f2(w['next_cosine'])} ({f2(w['next_cosine_mean_predictor'])}) | "
              f"{_p((pq.get('first_iteration') or [None])[0])} | {_p(pq.get('accuracy'))} ({_p(pq.get('majority'))}) |")
    print("\nHow far the controller's weights are from their initial values at the end (relative L2 per parameter group), and whether they "
          "changed after the first phase:\n\n| condition | input | encoder | recurrent | policy | critic | all | changed after phase 1 | optimiser steps |\n|---|---|---|---|---|---|---|---|---|")
    for c in d["order"]:
        s = C[c]
        w = s["weights_since_init_relative"]
        print(f"| {d['labels'][c]} | " + " | ".join(f2(w[g]) for g in ("input", "encoder", "recurrent", "policy", "critic", "all"))
              + f" | {'yes' if all(s['weights_changed_after_first_phase']) else 'NO'} | {s['optimizer_steps']:,.0f} |")
    print("\nLearning efficiency over the lifetime: training success in the FIRST practice iteration of each phase (right after observing "
          "the demonstrations; the last phase has none):\n\n| condition | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
    for c in d["order"]:
        print(f"| {d['labels'][c]} | " + " | ".join(_p(x) for x in C[c]["train_first_iter"]) + " |")
    print()


def pit():
    hits = load("pit.json")
    if not hits:
        return
    d = hits[0][1]
    desc = {"practice": "practice only", "explore_practice": "practice, half exploratory", "observe": "observation only (4 demonstrations)",
            "observe_practice": "observation + practice", "observe_practice_backbone": "observation + practice, backbone fully trainable",
            "frozen_memory": "control: weights frozen, memory written",
            "memory_observe_practice": "observation + practice, memory on"}
    for fam, F in d["families"].items():
        arms = next((f["arms"] for f in d["config"]["families"] if f["name"] == fam), [])
        print(f"### Unfamiliar family `{fam}` ({', '.join(arms)})\n\nHeld-out success with the weights before and after, both evaluated in a fresh process from the "
              "weights file alone (greedy; percentage points; 95% interval over seeds and tasks).\n\n"
              "| route | seeds | weights changed | unfamiliar family: before -> after | difference | tasks gained / lost | in unseen kinds of world | related, never trained | familiar families (retention) | after, sampled | controller weights moved (relative) | decisions used |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
        for route, r in F.items():
            L = r["lists"]
            u = L.get("unfamiliar") or {}
            ba = lambda x: "-" if not x else f"{_p(x['before'])} -> {_p(x['after'])}"       # noqa: E731
            print(f"| {desc.get(route, route)} | {len(r['seeds'])} | {'yes' if all(r['weights_changed']) else 'no'} | **{ba(u)}** | {_cmp(u)} | {u.get('gained', '-')} / {u.get('lost', '-')} | "
                  f"{ba(L.get('unfamiliar_unseen_worlds'))} | {ba(L.get('related_untrained'))} | {ba(L.get('familiar'))} | {_p(r.get('after_sampled_unfamiliar'))} | "
                  f"{r['relative_weight_change']['all']:.3f} | {r['env_decisions']:,.0f} |")
        extra = []
        for route, r in F.items():
            if r.get("after_with_its_memory"):
                m = r["after_with_its_memory"]
                extra.append(f"* {desc.get(route, route)}: with the memory it wrote, {_p(m['before'])} -> {_p(m['after'])} ({_cmp(m)}); "
                             f"weights alone, {_p(r['lists']['unfamiliar']['before'])} -> {_p(r['lists']['unfamiliar']['after'])}.")
            if r.get("after_other_actor_name"):
                m = r["after_other_actor_name"]
                extra.append(f"* {desc.get(route, route)}, the agent registered under another name: {_p(m['after'])} (same weights, same tasks: {_p(r['lists']['unfamiliar']['after'])}).")
        if extra:
            print("\n" + "\n".join(extra))
        st = next(iter(F.values()))["state_at_start_of_fresh_process"]
        print(f"\nState of the fresh process before its first episode: {st}.\n")


def meta_continual():
    path = os.path.join(HERE, "meta_continual", "ledger.jsonl")
    if not os.path.exists(path):
        return
    recs = [json.loads(l) for l in open(path) if l.strip()]
    base = [r for r in recs if r["kind"] == "baseline_run"]
    ff = [r for r in recs if r["kind"] == "full_feedback"]
    exps = [r for r in recs if r["kind"] == "experiment"]
    pool = [dict(e, round=r["round"], picked=e["proposal"] in r["surrogate_picks"]) for r in ff for e in r["pool"]]
    ran = [e for e in pool if e["valid"]]
    f3 = lambda x: "-" if x is None else f"{x:+.3f}"       # noqa: E731
    print(f"{len(pool)} candidate rules proposed, {len(ran)} ran a whole confined lifetime, "
          f"{sum(1 for e in pool if e.get('inert'))} rejected as inert, {len(exps)} taken to more seeds; "
          f"{len(base) + sum(len(e.get('measures') or {}) for e in ran) + sum(sum(len(v) for v in (x.get('stages') or {}).values()) for x in exps)} confined lifetimes in the ledger.\n")
    print("Champion (the shipped rule: rehearsal, online world model, no anchors), per seed:\n\n| seed | score (mean success on phases seen so far) | final | forgetting | right-after-learning gain | retention matrix |\n|---|---|---|---|---|---|")
    seen = set()
    for b in sorted(base, key=lambda r: int(r["seed"])):
        if b["seed"] in seen:
            continue
        seen.add(b["seed"])
        m = b["measures"]
        print(f"| {b['seed']} | {m['aulc_eval']:.3f} | {m['final_eval']:.3f} | {m.get('life_forgetting', 0):.3f} | {m.get('life_acquisition', 0):.3f} | {m.get('life_matrix')} |")
    print("\nScreening (one seed each, same seed as the champion's run). Effect = candidate's score minus the champion's.\n\n"
          "| round | candidate | what it changes | ran | effect | final | forgetting | picked by the learned selector |\n|---|---|---|---|---|---|---|---|")
    for e in pool:
        ms = next(iter((e.get("measures") or {}).values()), {})
        why = e["hypothesis"] if e["valid"] else (e.get("reject_reason", "") + " / " + e["hypothesis"])
        print(f"| {e['round']} | {e['candidate']} | {why[:170].replace('|', '/')} | {'yes' if e['valid'] else 'no'} | {f3(e['effect']) if e['valid'] else '-'} | "
              f"{'-' if not ms else format(ms.get('final_eval', 0), '.3f')} | {'-' if not ms else format(ms.get('life_forgetting', 0), '.3f')} | {'yes' if e['picked'] else ''} |")
    if exps:
        print("\nCandidates taken to more seeds (score difference to the champion on the same seeds, 95% bootstrap interval):\n\n"
              "| candidate | what it changes | decision | stage reached | seeds | score difference | final-success difference |\n|---|---|---|---|---|---|---|")
        for x in exps:
            c = (x.get("comparison") or {})
            a, f = c.get("aulc_eval") or {}, c.get("final_eval") or {}
            ci_ = lambda v: "" if not v.get("ci") else " [%+.3f, %+.3f]" % tuple(v["ci"])        # noqa: E731
            print(f"| {x['candidate']} | {x['hypothesis'][:150].replace('|', '/')} | {x['decision']} | {x['stage_reached']} | {a.get('n_cand', '-')} | "
                  f"{f3(a.get('diff'))}{ci_(a)} | {f3(f.get('diff'))}{ci_(f)} |")
    if ff:
        print("\nLearned selection of what to test (it chooses before the round's outcomes are known):\n\n| round | pool | selector fitted | its picks | pool mean (random choice) | best in pool |\n|---|---|---|---|---|---|")
        for r in ff:
            print(f"| {r['round']} | {len(r['pool'])} | {r['surrogate_fitted']} | {r['value_surrogate']:+.3f} | {r['value_random_expected']:+.3f} | {r['value_best']:+.3f} |")
    print()


ALL = {"b": b_multi, "baselines": baselines, "c": exp_c, "d": exp_d, "a": exp_a, "worldmodel": worldmodel, "selfplay": selfplay, "meta": meta, "final": final, "d_clone": exp_d_clone,
       "life": life, "life_matrix": life_matrix, "life_backbone": life_backbone, "life_plasticity": life_plasticity, "pit": pit,
       "meta_continual": meta_continual}

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
