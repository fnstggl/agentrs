"""SP-12.2: self-play records model hashes, participant identity, training-data provenance, experience, actions and
outcome history, policy updates and population composition; cross-play between policy versions is reproducible;
league / population-based evaluation runs.

A short campaign with an experience store (`crucible.experience.store`); then, in Tier 3 civilisations, frozen
cross-play between versions (alpha final vs beta final, alpha initial vs beta final, alpha final vs alpha initial),
one match replayed from its recorded specification (world digest and every evaluator outcome compared), and an Elo
league over the cross-play matches."""
import json

import pytest

from crucible.acceptance.v5.evidence import record
from crucible.multiagent import campaign as C
from crucible.multiagent import league as L
from crucible.multiagent.registry import FIELD_KINDS

from _selfplay_kit import PROFILE, campaign

pytestmark = [pytest.mark.slow, pytest.mark.timeout(5400)]
CMD = "PYTHONPATH=src python -m crucible.acceptance.v5 run --requirement SP-12.2"


def test_SP_12_2_versioning_reproducible_crossplay_league(tmp_path):
    run = campaign(str(tmp_path), "short", store=True, heldout=False)
    m, reg = run["match"], run["registry"]
    m.finish()
    m.close()
    kinds = reg.field_kinds()
    a_fin, b_fin, a_ini = reg.vid(run["groups"]["alpha"]), reg.vid(run["groups"]["beta"]), \
        reg.vid(run["initial"]["alpha"])
    pairs = [(a_fin, b_fin), (a_ini, b_fin), (a_fin, a_ini)]
    recs, label = [], {}
    for k, (a, b) in enumerate(pairs):
        r = L.crossplay(reg, a, b, world=f"civ:{PROFILE}", seeds=[101 + k], hours=8, tag=f"xp{k}")[0]
        recs.append(r)
        label[f"{r['match']}:ctrl:A"], label[f"{r['match']}:ctrl:B"] = a, b
    again = L.replay(L.MatchSpec.from_dict(recs[0]["spec"]), reg, recs[0])
    lg = L.league_ratings(recs, label)
    record("SP-12.2", {"recorded_field_kinds": sum(kinds.values()), "field_kinds": json.dumps(kinds),
                       "crossplay_reproduction_mismatches": len(again["mismatches"]),
                       "league_evaluations": 1 if lg["pairs"] >= len(recs) else 0,
                       "league_ratings": json.dumps({k: round(v, 1) for k, v in lg["ratings"].items()}),
                       "crossplay_matches": len(recs), "versions_recorded": len(reg.versions),
                       "experience_records": sum(e["records"] for e in reg.experience),
                       "replayed_digest_equal": again["digest"] == recs[0]["digest"]},
           command=CMD, labels=C.EVIDENCE_LABELS)
    assert set(kinds) == set(FIELD_KINDS)
    assert sum(kinds.values()) >= 6
    assert again["mismatches"] == []
    assert lg["pairs"] >= 3
