"""SP-12.1: the self-play population mixes current learner instances, historical checkpoints, independent learned
policies, statistical behavioural participants, compact neural participants, semantic generalists and held-out
policy families - every kind acting in one Tier 3 civilisation - and is not homogeneous.

A short campaign (`_selfplay_kit.campaign`, size short): the learning groups (current: alpha, beta; independent:
gamma, delta), a frozen historical checkpoint of alpha from half-time, one semantic generalist, the held-out GRU/PPO
family admitted for the last quarter, and the civilisation's own population (class 1 statistical, class 2 compact
adaptive neural, class 3 strategic neural). A kind counts when it is present in the world at the end and at least
one of its members acted (admitted identities: an accepted state-changing tool action; population classes: an
individual's last activity inside a window). Diversity: distinct weight hashes and the mean pairwise Jensen-Shannon divergence of the
learned groups' tool-host usage."""
import itertools
import json

import pytest

from crucible.acceptance.v5.evidence import record
from crucible.multiagent import analysis as AN

from _selfplay_kit import campaign

pytestmark = [pytest.mark.slow, pytest.mark.timeout(5400)]
CMD = "PYTHONPATH=src python -m crucible.acceptance.v5 run --requirement SP-12.1"
KIND_OF = {"current": "current", "historical": "historical", "independent": "independent", "semantic": "semantic",
           "heldout": "heldout", "statistical": "statistical", "compact_neural": "compact_neural",
           "strategic_neural": "independent"}


def test_SP_12_1_population_mixes_seven_member_kinds(tmp_path):
    run = campaign(str(tmp_path), "short", heldout_join_at=0.75)
    m = run["match"]
    acting: dict[str, int] = {}
    for w in m.windows:
        for k, v in w["kinds"].items():
            kk = KIND_OF.get(k, k)
            acting[kk] = max(acting.get(kk, 0), v["acting"])
    present: dict[str, int] = {}
    for k, v in m.windows[-1]["kinds"].items():              # all of them in the one world at the end
        kk = KIND_OF.get(k, k)
        present[kk] = present.get(kk, 0) + v["present"]
    kinds = sorted(k for k, n in acting.items() if n > 0 and present.get(k, 0) > 0)
    learned = {}
    for p, a in m.agents.items():
        g = getattr(a, "group", None)
        if g is not None:
            learned.setdefault(g.gid, []).extend(a.history)
    dists = {g: AN.usage_distribution(h) for g, h in learned.items() if h}
    js = [AN.js_divergence(dists[a], dists[b]) for a, b in itertools.combinations(sorted(dists), 2)]
    hashes = {g: m.groups[g].hash for g in learned}
    record("SP-12.1", {"population_member_kinds": len(kinds), "kinds_acting": json.dumps(kinds),
                       "kinds_present": json.dumps(present), "acting_counts": json.dumps(acting),
                       "learned_groups": len(learned), "distinct_weight_hashes": len(set(hashes.values())),
                       "mean_pairwise_js_divergence": float(sum(js) / len(js)) if js else 0.0},
           command=CMD, labels={"learned": True, "scripted_demonstrations_for_pretraining": True})
    m.close()
    assert len(set(hashes.values())) == len(hashes)            # no two groups share one latest policy
    assert len(kinds) >= 7, kinds
