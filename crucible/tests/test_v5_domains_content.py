"""Unit and short integration tests of the content-ecosystem domain (9.3): configurations, difficulty levels toward
10M viewers (built specs), originality rules of the evaluator and the population's consumption (short horizon;
runs the shared semantic space's text path)."""
import crucible

crucible.load_all_modules()
from crucible.domains import base  # noqa: E402
from crucible.domains import content_domain as D  # noqa: E402
from crucible.population import profiles as PP  # noqa: E402


def test_configurations_and_difficulty_levels():
    rep = base.structure_report(D.DOMAIN, (0, 1))
    assert rep["train_structures"] == 3 and rep["compositional_eval"] == 1 and rep["structural_eval"] == 1
    assert D.TARGET[max(D.DOMAIN.difficulties)] == 10_000_000
    for lv in D.DOMAIN.difficulties:
        sp = base.spec("content", "short_social", 0, overrides={"population": D.POPULATION_FOR[lv]})
        n = PP.resolve(next(m["config"] for m in sp.modules if m["name"] == "pop"))["n"]
        assert n >= D.TARGET[lv], (lv, n)


def test_originality_is_exact_against_every_earlier_item():
    """A re-typed copy of the newest of 6,000 earlier items by other creators is a copy (no sample or cap); an
    original text and one sharing well under 90% of its shingles are not; a republished artifact is a copy."""
    import numpy as np
    rng = np.random.default_rng(0)
    vocab = ["".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), size=6)) for _ in range(3000)]
    texts = [" ".join(rng.choice(vocab, size=60)) for _ in range(6000)]
    og = D.Originality()
    for k, t in enumerate(texts):
        og.add_other(f"art{k}", f"p{k % 97}", t)
    me = {"p:creator"}
    assert og.is_copy("mine1", me, texts[-1]) and og.is_copy("mine2", me, texts[17])
    assert og.is_copy("mine3", me, texts[-1] + " " + " ".join(vocab[:3]))      # a few words added: still a copy
    assert not og.is_copy("mine4", me, " ".join(rng.choice(vocab, size=60)))
    half = texts[-2].split()
    assert not og.is_copy("mine5", me, " ".join(half[:30] + list(rng.choice(vocab, size=30))))
    assert og.is_copy("art5999", me, "anything") and not og.is_copy("mine6", me, "")


def test_reference_reaches_viewers_and_copycat_counts_nothing():
    r = base.run("content", "short_social", 0, policy="reference", days=1)
    assert r["proc_failed"] == 0 and r["invariant_violations"] == 0
    assert r["outcome"]["details"]["distinct_viewers"] > 0 and r["outcome"]["details"]["original_items"] > 0
    for m in ("Actual content artifacts", "Autonomous consumption", "Recommendation systems", "Competing creators",
              "Distinct verified viewers"):
        assert r["mechanics"]["counts"][m] > 0, m
    c = base.run("content", "short_social", 0, policy="copycat", days=1)
    assert c["outcome"]["details"]["original_items"] == 0 and c["outcome"]["details"]["distinct_viewers"] == 0
