"""Unit and short integration tests of the community-ecosystem domain (9.4): configurations, communities founded at
genesis through the ordinary API, the population's civic participation with decision provenance (short horizon;
runs the shared semantic space's text path)."""
import crucible

crucible.load_all_modules()
from crucible.domains import _t2kit as K  # noqa: E402
from crucible.domains import base  # noqa: E402
from crucible.domains import community as D  # noqa: E402


def test_configurations_and_genesis_communities():
    rep = base.structure_report(D.DOMAIN, (0, 1))
    assert rep["train_structures"] == 3 and rep["compositional_eval"] == 1 and rep["structural_eval"] == 1
    w = base.world("community", "curated", 0)
    comms = [c for _, c in w.state.scan("communities")]
    assert len(comms) == 4 and all(c["rules"]["governance"] == "moderators" and c["rules"]["join"] == "approval"
                                   for c in comms)
    assert all(K.is_population(c["founder"]) for c in comms)
    w.close()


def test_population_takes_part_and_reference_organises():
    """2 days, LABELLED SCRIPTED organiser: members join its community and post; civic decisions are organic."""
    r = base.run("community", "assemblies", 0, policy="reference", days=2, keep_env=True)
    w = r["env"].world
    assert r["proc_failed"] == 0 and r["invariant_violations"] == 0
    prov = K.decision_provenance(w, ("commlife.",))
    assert prov["batches"] > 0 and prov["forced"] == 0
    det = r["outcome"]["details"]["communities"]
    assert det and max(c["members"] for c in det.values()) >= 3
    for m in ("Persistent memberships", "Shared activities", "Communication", "Social networks"):
        assert r["mechanics"]["counts"][m] > 0, m
    r["env"].close()
