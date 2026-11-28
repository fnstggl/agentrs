"""G2-learn round 3: per-provider statistics of slow steps in learner logs (accepted and refused), the update projection
and the trust region.

    .venv/bin/python cam_live/docs/verification/repro/g2learn/slow_step_stats.py LOG [LOG ...]

Prints, per provider, the relative paired held-out change of accepted steps (median, largest rise) and how many
accepted steps raised it at all; the largest first-order change after the update projection (must be <= 0); how often
the projection acted; the step-scale trajectory.
"""
import collections
import json
import statistics
import sys

for path in sys.argv[1:]:
    per = collections.defaultdict(list)
    proj_after, n_proj, scales, acc, rej = [], 0, [], 0, 0
    restored = collections.Counter()
    for line in open(path):
        d = json.loads(line)
        d = d.get("data", d)
        if d.get("timescale") != "slow_adapters":
            continue
        n = d.get("notes") or "{}"
        n = json.loads(n) if isinstance(n, str) else n
        for x in n.get("running_state_restored") or []:
            restored[x] += 1
        g = n.get("guard") or {}
        if not g.get("by_provider"):
            continue
        scales.append(n.get("step_scale"))
        up = n.get("update_projection") or {}
        if up:
            n_proj += int(up.get("projections", 0) > 0)
            proj_after += [v for v in (up.get("first_order_change_after") or {}).values()]
        if g.get("accepted"):
            acc += 1
            for k, v in g["by_provider"].items():
                per[k].append(v["paired_change"] / max(abs(v["validation_before"]), 1e-12))
        else:
            rej += 1
    print(path)
    print(f"  judged {acc + rej}: accepted {acc}, refused {rej}; update projection acted in {n_proj}; "
          f"largest first-order change after projection {max(proj_after) if proj_after else None}")
    print(f"  running statistics restored (steps): {dict(restored)}")
    print(f"  step scale over the run: {scales}")
    for k, v in sorted(per.items()):
        rises = [x for x in v if x > 0]
        print(f"  {k:22s} accepted steps {len(v)}: median rel. change {statistics.median(v):+.2e}, "
              f"rose in {len(rises)}, largest rise {max(rises) if rises else 0:+.2e}, sum of changes {sum(v):+.2e}")
