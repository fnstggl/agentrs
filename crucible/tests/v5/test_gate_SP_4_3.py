"""SP-4.3: content representations let participants respond to meaning, not tokens (MASTER_SPEC 4.3).

1. Adversarial content in the social internet: on the one population (`crucible.social.experiments.pop_spec`,
   3,000 individuals, the posts and blog platforms with their recommender mixtures), creators publish - at the same
   moment, with the same title per topic - an informative text (real news abstracts of the topic from the
   population's corpus) and, by other creators, adversarial texts on the same topic: gibberish (random letter
   strings), keyword stuffing (the topic's keywords repeated after one real sentence), a word salad (the
   informative text's own words shuffled) and token-optimised text (grammatical-looking template sentences filled
   with the topic's keywords, built to pass lexical readability checks while saying nothing). Each kind has
   `REPLICAS` independent creators per topic. After `DAYS` simulated days the exact audiences (unique viewers) are
   compared: an adversarial kind wins on a topic when its mean unique viewers exceed the informative items' mean.
2. Structured representations: software (a code artifact's structural facts) and products (a launched product's
   public page) - `crucible.semantic.structured`.
3. Value is never defined by a semantic prediction alone: in the same world the individuals' consumption decisions
   are compared with what the semantic affinity alone would predict (items with nearly the same representation and
   topic affinity reach very different audiences; affinity explains only part of who watches).
4. Re-encoding on update: an item updated with new bytes (`content.update`) must be distributed with the new bytes'
   representation (no stale encoding).

    PYTHONPATH=src python -m crucible.scale.slots --cpu 1 --gpu-mb 3000 --ram-mb 12000 --timeout 7200 \
        --label "D: SP-4.3" -- python -m pytest -q tests/v5/test_gate_SP_4_3.py
"""
import base64
import json
import os

import numpy as np
import pytest

from crucible.acceptance.v5.evidence import record

DAYS = float(os.environ.get("CRUCIBLE_SP43_DAYS", "3"))
REPLICAS = 2
TOPICS = ("recipes", "travelarticle", "weathertopstories", "finance-real-estate", "baseball_mlb", "movies-celebrity")
KINDS = ("gibberish", "keyword_stuffing", "word_salad", "token_optimised")

pytestmark = [pytest.mark.slow, pytest.mark.timeout(7200)]


def _texts(rng):
    from crucible.population import semantics as sem
    names, corpus = sem.topic_names(), sem.corpus()
    out = {}
    for t in TOPICS:
        k = names.index(t)
        info = " ".join(corpus[k][:3])
        words = sem.topic_words(k)[:8]
        toks = info.split()
        sal = list(toks)
        rng.shuffle(sal)
        tmpl = ["The best {} and the new {} are now the top {} for every {}.",
                "Every {} with a great {} makes the {} of the {} better.",
                "Here is the {} and the {} that all {} fans love about {}."]
        opt = " ".join(tmpl[i % 3].format(*[words[(i * 4 + j) % len(words)] for j in range(4)])
                       for i in range(max(6, len(toks) // 12)))
        gib = " ".join("".join(chr(97 + int(c)) for c in rng.integers(0, 26, int(rng.integers(3, 9))))
                       for _ in range(len(toks)))
        out[t] = {"informative": info, "word_salad": " ".join(sal), "token_optimised": opt, "gibberish": gib,
                  "keyword_stuffing": " ".join(toks[:18]) + " " + " ".join(words * 8)}
    return out


def _publish(w, handle, text, title):
    p = w.system_call("admin.join", {"handle": handle, "endowment": 10_000, "password": f"pw-{handle}"})["principal"]
    ref = w.execute(p, {"tool": "artifact.put", "args": {"data_b64": base64.b64encode(text.encode()).decode(),
                                                        "mime": "text/plain", "kind": "text"}})
    assert ref.ok, ref.error
    r = w.execute(p, {"tool": "posts.publish", "args": {"ref": ref.result["ref"], "title": title}})
    assert r.ok, r.error
    return p, r.result["id"], ref.result["ref"]


def test_SP_4_3_meaning_not_tokens():
    import crucible
    from crucible.core.kernel import World
    from crucible.social import audience as AUD, experiments as EX
    from crucible.semantic import space, structured
    crucible.load_all_modules()
    rng = np.random.default_rng(0)
    texts = _texts(rng)
    w = World.create(EX.pop_spec("sp43", 0, n=3000))
    items = []
    for t in TOPICS:
        for kind in ("informative",) + KINDS:
            for r in range(REPLICAS):
                handle = "".join(ch for ch in f"{kind[:5]}{t[:6]}{r}" if ch.isalnum())
                _, cid, ref = _publish(w, handle, texts[t][kind], f"{t} today")
                items.append({"topic": t, "kind": kind, "content": cid, "ref": ref})
    w.advance_time(w.now + int(DAYS * 86400))
    item_of = {rec["content_id"]: int(k) for k, rec in w.state.scan("aud_item_info") if rec.get("content_id")}
    for it in items:
        aid = item_of[it["content"]]
        a = AUD.audience(w, aid)
        it.update(item=aid, unique_viewers=int(a["unique_viewers"]), views=int(a["views"]),
                  exposed=int(a["unique_exposed"]), watch_s=float(a["watch_s"]))
        f = space.item_features(w, it["ref"])
        it["comprehension"] = round(space.comprehension(f), 4)
    wins, table = 0, {}
    for t in TOPICS:
        inf = np.mean([i["unique_viewers"] for i in items if i["topic"] == t and i["kind"] == "informative"])
        for kind in KINDS:
            adv = np.mean([i["unique_viewers"] for i in items if i["topic"] == t and i["kind"] == kind])
            wins += int(adv > inf)
            table[f"{t}:{kind}"] = [round(float(adv), 2), round(float(inf), 2)]
    # 2. structured representations
    code = json.dumps({"files": {"calc.py": "def add(a, b):\n    return a + b\n",
                                 "test_calc.py": "from calc import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"}})
    sw = structured.software_bytes(code.encode())
    kinds_struct = int(sum(v is not None for v in sw["fields"].values()) >= 8)
    pw = World.create(EX.pop_spec("sp43p", 1, n=300, extra=[{"name": "products", "config": {}}]))
    owner = pw.system_call("admin.join", {"handle": "maker", "endowment": 10_000, "password": "pw-m"})["principal"]
    reps_p = []
    for name, desc, kind in (("Route planner", "Plans the fastest delivery routes for a fleet of vans.", "saas"),
                             ("Recipe box", "A collection of tested family recipes with shopping lists.", "media")):
        r = pw.execute(owner, {"tool": "prod.create", "args": {"name": name, "kind": kind, "description": desc}})
        assert r.ok, r.error
        pid = r.result.get("product") or r.result.get("id")
        reps_p.append(structured.product(pw, pid))
    prod_ok = all(len(structured.fields(p)) >= 5 for p in reps_p) and \
        float(np.dot(reps_p[0]["vector"], reps_p[1]["vector"])) < 0.95
    kinds_struct += int(prod_ok)
    # 3. value is not the semantic prediction alone: audience spread among items with near-identical affinity
    inf_items = [i for i in items if i["kind"] == "informative"]
    vv = [i["unique_viewers"] for i in inf_items]
    spread = float(np.std(vv) / max(1e-9, np.mean(vv))) if vv else 0.0
    same_rep_diff = sum(1 for t in TOPICS
                        if len({i["unique_viewers"] for i in inf_items if i["topic"] == t}) > 1)
    # the semantic prediction itself (interest-item cosine in the space) is validated on held-out real behaviour
    # (MIND dev impressions; `crucible.semantic.space.prior`), and it is one input of the individuals' decisions:
    # identical bytes (identical representation) published by different creators reach different audiences
    from crucible.semantic.space import prior
    meta = prior.load(space.SPACE_VERSION)["meta"]
    cal, auc = meta["calibration"], meta["click_auc"]
    validated = bool(auc["space"]["mean_auc"] > auc["random"]["mean_auc"] + 0.05 and
                     cal["heldout_logloss"] < cal["heldout_logloss_constant"])
    solely = int(same_rep_diff == 0 or not validated)
    # 4. re-encoding on update
    upd = inf_items[0]
    p = w.state.get("content", upd["content"])
    creator = p["creator"]
    new = texts[TOPICS[1]]["informative"]
    nref = w.execute(creator, {"tool": "artifact.put", "args": {"data_b64": base64.b64encode(new.encode()).decode(),
                                                               "mime": "text/plain"}})
    u = w.execute(creator, {"tool": "content.update", "args": {"id": upd["content"], "ref": nref.result["ref"]}})
    assert u.ok, u.error
    from crucible import population
    from crucible.social.recsys.features import item_emb
    stored = item_emb(w, np.array([upd["item"]]))[0].astype(np.float64)
    fresh = population.to_coords(w, space.item_vector(w, nref.result["ref"])[None, :])[0].astype(np.float64)
    cos_new = float(stored @ fresh / max(1e-9, np.linalg.norm(stored) * np.linalg.norm(fresh)))
    stale = int(cos_new < 0.999)
    m = {"adversarial_content_kinds_probed": len(KINDS), "adversarial_content_wins_over_informative": wins,
         "structured_representation_kinds": kinds_struct, "value_defined_solely_by_semantic_prediction": solely,
         "stale_encodings_after_update": stale, "comparisons": len(TOPICS) * len(KINDS),
         "informative_mean_viewers": round(float(np.mean(vv)), 2),
         "adversarial_mean_viewers": json.dumps({k: round(float(np.mean([i["unique_viewers"] for i in items
                                                                         if i["kind"] == k])), 2) for k in KINDS}),
         "comprehension_by_kind": json.dumps({k: round(float(np.mean([i["comprehension"] for i in items
                                                                      if i["kind"] == k])), 3)
                                              for k in ("informative",) + KINDS}),
         "audience_spread_informative": round(spread, 3), "updated_item_cosine_to_new_bytes": round(cos_new, 5),
         "topics_identical_bytes_different_audiences": same_rep_diff,
         "semantic_prediction_heldout_click_auc": auc["space"]["mean_auc"],
         "random_baseline_click_auc": auc["random"]["mean_auc"],
         "popularity_baseline_click_auc": auc["popularity"]["mean_auc"],
         "semantic_prediction_heldout_logloss": cal["heldout_logloss"],
         "constant_baseline_logloss": cal["heldout_logloss_constant"],
         "by_topic_kind": json.dumps(table)[:2000]}
    record("SP-4.3", m, command="PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_4_3.py")
    w.close()
    pw.close()
    assert m["adversarial_content_wins_over_informative"] == 0, table
    assert m["structured_representation_kinds"] >= 2 and m["stale_encodings_after_update"] == 0
    assert m["value_defined_solely_by_semantic_prediction"] == 0
