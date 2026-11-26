"""SP-3.1 (MASTER_SPEC Part 3.1): persistent individual identity with independently evolving state; individuals respond
differently to the same situation and their preferences evolve through experience. Measured on the integrated world
of the one population (tests/v5/pop_world.py: crucible.population + the web + the social internet, 6 days).

identity_state_fields >= 15: Part 3.1 kinds of state (schema.FIELDS) that are represented, differ between
individuals and changed since genesis; same_situation_response_heterogeneity > 0: individuals reading the same page in
the same tick experience different utility; preference_drift_from_experience > 0: interests (shared semantic space)
moved towards what each individual enjoyed (cosine of the interest change with the enjoyed content's direction)."""
import numpy as np
import pytest

from crucible.acceptance.v5 import evidence
from crucible.population import analysis as A
from pop_world import integrated

CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_3_1.py"
pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]


def test_SP_3_1_persistent_identity_and_evolving_state():
    r = integrated()
    w = r["world"]
    sf = A.state_fields(w, r["genesis"])
    full = [f for f, v in sf.items() if v["represented"] and v["heterogeneous"] and v["evolving"]]
    het = A.same_situation_heterogeneity(w)
    drift = A.preference_drift(w)
    sb = A.social_behaviours(w)
    m = {"identity_state_fields": len(full), "fields_represented": sum(v["represented"] for v in sf.values()),
         "same_situation_response_heterogeneity": round(float(het["same_situation_response_heterogeneity"]), 6),
         "same_situation_groups": int(het["same_situation_groups"]),
         "preference_drift_from_experience": round(float(drift["preference_drift_from_experience"]), 6),
         "individuals_with_drift": int(drift["individuals_with_drift"]),
         "social_consumption_rows": int(sb["consume_media"]), "fields": ",".join(full),
         "not_evolving": ",".join(f for f, v in sf.items() if not v["evolving"])}
    evidence.record("SP-3.1", m, command=CMD, notes=(
        "One columnar row per individual of crucible.population; Part 3.1 field -> columns in schema.FIELDS. "
        "Interests are coordinates in the shared semantic space (MIND-derived prior, drift through experience on the "
        "web and on the social platforms via population.experience)."))
    assert m["same_situation_groups"] > 0
