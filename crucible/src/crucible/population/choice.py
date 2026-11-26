"""Evaluator side: which of the content-choice factors of MASTER_SPEC Part 4.4 the population's decisions actually
respond to (SP-4.4), measured on real individuals of a running world by controlled perturbation of each factor's
input in the decision functions the world uses.

The decisions (the code that takes them in a world):

* `article_open` - which headline of a page's article slate is opened: the fitted runtime model
  `article_open_semantic@1` (crucible.calibration.runtime, scope content_kind="article"; `PopWeb._article_open`);
* `resource_choice` - which known web resource to visit or whether to explore (`web.resource_scores`);
* `continue` - read on (`models` decision `continue`), `evaluate_product`, `purchase`, `abandon` (leave a resource),
  `recommend` (share with a contact), `follow_creator` (subscribe to an author), `join_community` - the plausible
  class-1 decisions (`population.models`).

`FACTORS` maps each of the twelve Part 4.4 factors to the decision inputs that carry it. For every (decision, input)
pair, `factor_effects` evaluates the decision on a cohort's real feature rows (built from the world's state exactly as
the decision code builds them) and again with only that input perturbed (a fixed shift of one standard deviation of
the input across the cohort, or a flip for flags), and reports the mean absolute change of the decision probability
(or, for the resource choice, of the softmax choice distribution). A factor is USED when at least one of its inputs
moves a decision by more than `THRESHOLD`.

Two perturbations follow the structure of the decision rather than one cohort-wide shift:

* alternative-specific inputs of the resource choice (`ALTERNATIVE`: satisfaction with, match of, visits to, recency
  of and contacts' liking of each known resource) are perturbed for ONE alternative (the first known resource), as a
  choice model's attribute effect is measured - shifting every alternative by the same amount leaves a softmax
  almost unchanged and would measure nothing;
* bounded state inputs (`BOUNDED`: the share of today's attention left) are moved by a fixed share of their range
  (a quarter of the day's attention spent), because their spread across a cohort depends on the hour at which the
  world is inspected (at a day boundary nearly everyone has a full budget).
"""
from __future__ import annotations

import numpy as np

from ..substrate import columns
from . import web as W
from .schema import RES_TABLE, TABLE

THRESHOLD = 1e-3

FACTORS = {
    "individual_interests": [("article_open", "text_cos"), ("resource_choice", "match")],
    "relevance": [("evaluate_product", "page_relevance"), ("purchase", "page_relevance")],
    "previous_experience": [("article_open", "own_topic_views"), ("resource_choice", "known_sat")],
    "attention_and_availability": [("continue", "attention_ratio"), ("evaluate_product", "attention_ratio")],
    "content_characteristics": [("continue", "page_richness"), ("article_open", "item_age_hours")],
    "social_relationships": [("resource_choice", "social"), ("join_community", "contacts_in_community"),
                             ("resource_choice", "trust")],
    "creator_familiarity": [("resource_choice", "visits"), ("follow_creator", "page_utility")],
    "platform_distribution": [("article_open", "item_opens_24h"), ("article_open", "topic_population_share")],
    "novelty": [("resource_choice", "recency"), ("resource_choice", "novelty_seeking")],
    "satisfaction": [("abandon", "tolerance_gap"), ("recommend", "resource_satisfaction")],
    "competing_content": [("article_open", "text_cos_z"), ("article_open", "slate_size")],
    "current_needs": [("evaluate_product", "goal_buy"), ("purchase", "goal_buy")],
}
FLAGS = {"goal_buy", "valid"}
ALTERNATIVE = {"known_sat", "match", "visits", "recency", "social"}
BOUNDED = {"attention_ratio": (0.0, 1.0, 0.25)}         # name -> (low, high, shift as a share of the range)


def _contacts_in_community(world, idx: np.ndarray) -> np.ndarray:
    """Per individual: the share of its contacts who are members of the community most of its contacts belong to
    and it has not joined (the `contacts_in_community` input of the join decision, `PopWeb.act_forum`)."""
    allc = columns.col(world, TABLE, "communities").astype(np.int64)
    ct = columns.col(world, TABLE, "contacts")[idx].astype(np.int64)
    own = allc[idx]
    out = np.zeros(idx.size)
    for j in range(idx.size):
        v = ct[j][ct[j] >= 0]
        if v.size == 0:
            continue
        ids = allc[v]
        ids = ids[(ids >= 0)]
        ids = ids[~np.isin(ids, own[j])]
        if ids.size == 0:
            continue
        b = int(np.bincount(ids).argmax())
        out[j] = float((allc[v] == b).any(axis=1).mean())
    return out


def _cohort_features(world, idx: np.ndarray) -> dict:
    """Real feature rows of individuals `idx`: the inputs each decision reads, built from the world's state the way
    the decision code builds them (the first known resource stands for 'the page read' of the plausible decisions)."""
    c = lambda name: columns.col(world, TABLE, name)[idx]  # noqa: E731
    n = idx.size
    prefs = c("prefs").astype(np.float64)
    beliefs = c("beliefs").astype(np.float64)
    known = c("known").astype(np.int64)
    ks = c("known_sat").astype(np.float64)
    kc = c("known_n").astype(np.float64)
    kl = c("known_last").astype(np.int64)
    valid = known >= 0
    R = columns.rows(world, RES_TABLE) if RES_TABLE in columns.tables(world) else 0
    inter = c("interests").astype(np.float32)
    if R:
        rv = columns.col(world, RES_TABLE, "vec")[np.clip(known, 0, R - 1)].astype(np.float32)
        match = np.einsum("nd,nkd->nk", inter, rv)
    else:
        match = np.zeros(known.shape)
    contacts = c("contacts").astype(np.int64)
    cvalid = contacts >= 0
    cc = np.clip(contacts, 0, None)
    ck = columns.col(world, TABLE, "known")[cc].astype(np.int64)
    liked = (columns.col(world, TABLE, "known_sat")[cc] > 0.4) & cvalid[:, :, None]
    ck = np.where(liked, ck, -2)
    social = (ck[:, None, :, :] == known[:, :, None, None]).any(axis=3).sum(axis=2) / \
        np.maximum(cvalid.sum(axis=1), 1)[:, None]
    recency = np.exp(-np.maximum(world.now - kl, 0) / (4 * 3600.0)) * (kl >= 0)
    rs = {"known_sat": ks, "match": match, "visits": kc, "recency": recency, "social": social, "valid": valid,
          "loyalty": prefs[:, 5], "social_conformity": prefs[:, 4], "novelty_seeking": prefs[:, 1],
          "suggest": c("suggest").astype(np.int64), "suggest_src": c("suggest_src").astype(np.int64),
          "trust": c("contact_trust").astype(np.float64).mean(axis=1), "belief_links": beliefs[:, 5],
          "belief_unknown": beliefs[:, 0]}
    bud = c("attention_budget_min").astype(np.float64)
    left = c("attention_left_min").astype(np.float64)
    first_sat = np.where(valid[:, 0], ks[:, 0], 0.0)
    page_rel = np.where(valid[:, 0], match[:, 0], 0.0)
    page_u = 0.1 + 1.5 * page_rel
    goals = c("goals")
    sat = c("satisfaction").astype(np.float64)
    plaus = {
        "continue": {"page_utility": page_u, "attention_ratio": left / np.maximum(bud, 1e-3),
                     "quality_sensitivity": prefs[:, 3], "novelty_seeking": prefs[:, 1],
                     "page_richness": np.full(n, 0.5)},
        "evaluate_product": {"page_relevance": page_rel, "goal_buy": (goals == 2).any(axis=1).astype(float),
                             "impulsiveness": prefs[:, 7], "shopping_skill": c("skills")[:, 3].astype(np.float64),
                             "attention_ratio": left / np.maximum(bud, 1e-3)},
        "purchase": {"price_ratio": np.full(n, 0.2), "page_relevance": page_rel,
                     "goal_buy": (goals == 2).any(axis=1).astype(float), "wallet_cover_log": np.full(n, 1.0),
                     "price_sensitivity": prefs[:, 0], "resource_satisfaction": first_sat,
                     "belief_shops": beliefs[:, 3], "impulsiveness": prefs[:, 7]},
        "abandon": {"tolerance_gap": np.clip(0.6 * sat + 0.15 - 0.25 * prefs[:, 2], 0.05, 0.6) - first_sat,
                    "loyalty": prefs[:, 5], "resource_visits_log": np.log1p(kc[:, 0])},
        "recommend": {"page_utility": page_u, "sociability": prefs[:, 6], "resource_satisfaction": first_sat},
        "follow_creator": {"page_utility": page_u, "sociability": prefs[:, 6],
                           "goal_social": (goals == 3).any(axis=1).astype(float)},
        "join_community": {"board_relevance": page_rel, "sociability": prefs[:, 6],
                           "contacts_in_community": _contacts_in_community(world, idx),
                           "goal_social": (goals == 3).any(axis=1).astype(float)},
    }
    # article slates: the individual's own reading history against the headlines of a page it knows
    arts = _article_inputs(world, idx)
    return {"resource_choice": rs, **plaus, "article_open": arts}


def _article_inputs(world, idx: np.ndarray) -> dict | None:
    """article_open_semantic@1 inputs for each individual against one slate of a page it knows (the home page of its
    first known resource), from the world's recorded pages, page counters and reading history."""
    st = world.state
    hosts = W.PopWeb._hosts(world.modules["popweb"], world) if "popweb" in world.modules else []
    known = columns.col(world, TABLE, "known")[idx].astype(np.int64)
    rows, cos, z, size, topic, opens, age = [], [], [], [], [], [], []
    inter = columns.col(world, TABLE, "interests")[idx].astype(np.float32)
    tv = columns.col(world, W.TOPIC_TABLE, "views").astype(np.float64)
    for j in range(idx.size):
        r = int(known[j, 0])
        if r < 0 or r >= len(hosts):
            continue
        pg = st.get("pop_pages", f"{hosts[r]}|/")
        if not pg or not pg.get("arts"):
            continue
        V = world.modules["popweb"].slate_vectors(world, pg)
        vn = np.maximum(np.linalg.norm(V, axis=1), 1e-6)
        cs = np.clip((V @ inter[j]) / vn, -1, 1)
        sd = cs.std()
        zz = (cs - cs.mean()) / sd if sd > 1e-9 else np.zeros_like(cs)
        for k in range(len(cs)):
            rec = st.get("pop_pagerow", f"{pg['host']}|{pg['arts'][k]}")
            o = float(columns.col(world, W.PG_TABLE, "opens")[rec["row"]]) if rec else 0.0
            t0 = float(columns.col(world, W.PG_TABLE, "t0")[rec["row"]]) if rec else float(world.now)
            rows.append(j)
            cos.append(float(cs[k]))
            z.append(float(zz[k]))
            size.append(float(len(cs)))
            topic.append(int(pg["art_topics"][k]))
            opens.append(o)
            age.append(max(world.now - t0, 0.0) / 3600.0)
    if not rows:
        return None
    rows = np.array(rows, np.int64)
    topic = np.array(topic, np.int64)
    wt = columns.col(world, TABLE, "wtopic")[idx].astype(np.int64)
    wn = columns.col(world, TABLE, "wtopic_n")[idx].astype(np.float64)
    age = np.array(age)
    return {"text_cos": np.array(cos), "text_cos_z": np.array(z),
            "own_topic_views": ((wt[rows] == topic[:, None]) * wn[rows]).sum(axis=1),
            "own_views_total": columns.col(world, TABLE, "web_views")[idx][rows].astype(np.float64),
            "topic_population_share": (tv[topic] + 1.0) / (tv.sum() + tv.size),
            "item_opens_24h": np.array(opens) * np.minimum(1.0, 24.0 / np.maximum(age, 1.0)),
            "active_individuals_24h": np.full(rows.size, float(np.count_nonzero(
                columns.col(world, TABLE, "web_last_at") >= world.now - 86400))),
            "item_age_hours": age, "slate_size": np.array(size)}


def _decide(world, decision: str, F: dict) -> np.ndarray:
    if decision == "article_open":
        from ..calibration import runtime
        return runtime.get(W.ARTICLE_OPEN).scoped(content_kind="article").evaluate(F).values
    if decision == "resource_choice":
        s = W.resource_scores(F)
        s = np.where(np.isfinite(s), s, -1e9)
        e = np.exp((s - s.max(axis=1, keepdims=True)) / 0.45)
        return e / e.sum(axis=1, keepdims=True)
    return world.runtime["pop.models"][decision].prob(F)


def _perturb(F: dict, name: str) -> dict:
    G = dict(F)
    v = np.asarray(F[name])
    if name in BOUNDED:
        lo, hi, share = BOUNDED[name]
        step = share * (hi - lo)
        x = v.astype(np.float64)
        G[name] = np.where(x - step >= lo, x - step, np.minimum(x + step, hi))
        return G
    if name in ALTERNATIVE and v.ndim == 2:
        x = v.astype(np.float64).copy()
        col = x[:, 0]
        sd = float(np.std(col)) if col.size > 1 else 0.0
        x[:, 0] = col + (sd if sd > 1e-9 else max(1.0, abs(float(np.mean(col)))) * 0.5)
        G[name] = x
        return G
    if name in FLAGS or v.dtype == bool:
        G[name] = (1 - v.astype(np.float64)) if v.dtype != bool else ~v
        return G
    sd = float(np.std(v)) if v.size > 1 else 0.0
    shift = sd if sd > 1e-9 else max(1.0, abs(float(np.mean(v)))) * 0.5
    G[name] = v.astype(np.float64) + shift
    if name == "slate_size":
        G[name] = np.maximum(1.0, v + max(1.0, round(shift)))
    if name == "text_cos":
        G[name] = np.clip(G[name], -1, 1)
    return G


def factor_effects(world, idx: np.ndarray) -> dict:
    """{factor: {used, inputs: {decision.input: mean abs change}}} on the cohort `idx` (see the module doc)."""
    feats = _cohort_features(world, np.asarray(idx, np.int64))
    out = {}
    for factor, pairs in FACTORS.items():
        eff = {}
        for decision, name in pairs:
            F = feats.get(decision)
            if F is None or name not in F:
                eff[f"{decision}.{name}"] = None
                continue
            base = _decide(world, decision, F)
            alt = _decide(world, decision, _perturb(F, name))
            eff[f"{decision}.{name}"] = float(np.mean(np.abs(alt - base)))
        out[factor] = {"used": any(v is not None and v > THRESHOLD for v in eff.values()), "inputs": eff}
    return out
