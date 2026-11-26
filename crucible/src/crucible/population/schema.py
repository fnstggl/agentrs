"""Columns of the population table `pop` (contract section 2) and how they represent MASTER_SPEC Part 3.1.

Every individual is row `i` of every column (`crucible.substrate.columns`); a column changes only through a recorded
cohort batch (`columns.batch`). Dimensions come from the profile (`profiles.PROFILES[...]["dims"]`) and can be
overridden in the module config.

`FIELDS` maps each of the 18 kinds of individual state named in Part 3.1 to the columns that hold it, so a test can
check that a field is represented, differs between individuals and evolves (see `analysis.state_fields`).
"""
from __future__ import annotations

import numpy as np

from ..substrate import attention

TABLE = "pop"
RES_TABLE = "res"
NONE_I32 = -1
NEVER = np.iinfo(np.int64).max // 4          # "no activation scheduled" (far future, still safe to add to)

# preference vector components (prefs[P]); a fixed meaning per index
PREF_NAMES = ("price_sensitivity", "novelty_seeking", "error_tolerance", "quality_sensitivity", "social_conformity",
              "loyalty", "sociability", "impulsiveness")
# skills[S]
SKILL_NAMES = ("searching", "writing", "evaluating", "shopping", "technical", "social", "reading", "planning")
# beliefs[B]: subjective expectations formed from experience (0..1)
BELIEF_NAMES = ("unknown_sites_worthwhile", "search_results_useful", "recommendations_useful", "shops_trustworthy",
                "forum_helpful", "links_useful", "ads_honest", "content_improving")
# goal codes (goals[G], u8)
GOALS = {0: "none", 1: "learn", 2: "buy", 3: "socialize", 4: "entertain", 5: "work"}
# participant classes (class u8)
CLASSES = {1: "behavioural", 2: "adaptive", 3: "strategic", 4: "semantic"}
# how an individual reached a resource (visit records, `via`)
VIA = {0: "known", 1: "link", 2: "contact", 3: "search", 4: "forum", 5: "genesis"}


def dims(cfg: dict) -> dict:
    """Column dimensions: interest D (coordinates in the topic subspace of the shared space's topic block,
    population.semantics on crucible.semantic.space), known slots K, habit slots H, memory M, contacts C, goals G, skills S,
    beliefs B, prefs P, recent R, communities Q, abandoned A, purchased U."""
    d = {"D": 64, "K": 12, "H": 16, "M": 32, "C": 8, "G": 4, "S": len(SKILL_NAMES), "B": len(BELIEF_NAMES),
         "P": len(PREF_NAMES), "R": 8, "Q": 4, "A": 4, "U": 4}
    d.update({k: int(v) for k, v in (cfg or {}).items() if k in d})
    if "interest_dim" in (cfg or {}):
        d["D"] = int(cfg["interest_dim"])
    return d


def columns(d: dict) -> dict:
    """{name: (dtype, trailing shape, fill)} of the `pop` table."""
    D, K, H, M, C = d["D"], d["K"], d["H"], d["M"], d["C"]
    cols = {
        # identity and class
        "class": ("uint8", (), 1),
        "policy_group": ("int16", (), -1),
        "segment": ("uint8", (), 0),
        # interests, preferences, knowledge, goals, skills, beliefs
        "interests": ("float16", (D,), 0.0),
        "prefs": ("float16", (d["P"],), 0.0),
        "knowledge": ("float16", (D,), 0.0),
        "goals": ("uint8", (d["G"],), 0),
        "skills": ("float16", (d["S"],), 0.0),
        "beliefs": ("float16", (d["B"],), 0.5),
        "experience": ("float32", (), 0.0),
        # money and work
        "wallet_cents": ("int64", (), 0),
        "income_cents": ("int32", (), 0),
        "wtp_scale": ("float16", (), 1.0),
        "employer": ("int32", (), NONE_I32),
        "commit_start_min": ("int16", (), 0),        # daily commitment (work) window, minutes of day
        "commit_len_min": ("int16", (), 0),
        # routine and scheduling
        "day_start_min": ("int16", (), 7 * 60),
        "day_len_min": ("int16", (), 16 * 60),
        "mean_gap_s": ("float32", (), 6 * 3600.0),
        "session_min": ("float32", (), 20.0),
        "next_active_at": ("int64", (), NEVER),
        "wake_at": ("int64", (), NEVER),
        "wake_act": ("int16", (), -1),
        # habits, satisfaction, behavioural history
        "habit": ("float32", (H,), 0.0),
        "satisfaction": ("float32", (), 0.5),
        "last_utility": ("float32", (), 0.0),
        "last_activity": ("int16", (), -1),
        "last_session_at": ("int64", (), -1),
        "n_sessions": ("int32", (), 0),
        "n_switches": ("int32", (), 0),
        "n_purchases": ("int32", (), 0),
        "spent_cents": ("int64", (), 0),
        "recent": ("int32", (d["R"],), NONE_I32),
        # product / resource usage history (known resources with personal satisfaction)
        "known": ("int32", (K,), NONE_I32),
        "known_sat": ("float16", (K,), 0.0),
        "known_n": ("uint16", (K,), 0),
        "known_last": ("int64", (K,), -1),
        "abandoned": ("int32", (d["A"],), NONE_I32),
        "purchased": ("int32", (d["U"],), NONE_I32),
        "suggest": ("int32", (), NONE_I32),
        "suggest_from": ("int32", (), NONE_I32),
        "suggest_src": ("uint8", (), 0),
        # social connections, trust in them, community memberships
        "contacts": ("int32", (C,), NONE_I32),
        "contact_trust": ("float16", (C,), 0.5),
        "communities": ("int32", (d["Q"],), NONE_I32),
        # class 2 personal memory
        "memory": ("float16", (M,), 0.0),
    }
    cols.update(attention.COLUMNS)
    return cols


def res_columns(d: dict) -> dict:
    """{name: spec} of the resource catalogue table `res` (one row per discovered web resource)."""
    return {
        "vec": ("float16", (d["D"],), 0.0),     # content representation of the entry page (computed from its text)
        "chars": ("int32", (), 0),
        "uniq": ("float16", (), 0.0),
        "links": ("int16", (), 0),
        "items": ("int16", (), 0),
        "kind": ("uint8", (), 0),
        "price_cents": ("int64", (), 0),
        "status": ("int16", (), 0),
        "fetched_at": ("int64", (), -1),
        "ok_n": ("int64", (), 0),
        "err_n": ("int64", (), 0),
        "visits": ("int64", (), 0),
        "first_visits": ("int64", (), 0),
        "return_visits": ("int64", (), 0),
        "abandons": ("int64", (), 0),
        "purchases": ("int64", (), 0),
        "revenue_cents": ("int64", (), 0),
    }


# MASTER_SPEC Part 3.1: the 18 kinds of state and the columns representing each
FIELDS = {
    "interests": ["interests"],
    "preferences": ["prefs", "wtp_scale"],
    "knowledge": ["knowledge"],
    "personal_goals": ["goals"],
    "attention_budget": ["attention_budget_min", "attention_left_min"],
    "financial_resources": ["wallet_cents", "income_cents"],
    "skills": ["skills"],
    "experience": ["experience"],
    "trust_relationships": ["contact_trust"],
    "social_connections": ["contacts"],
    "habits": ["habit"],
    "product_usage_history": ["known", "known_sat", "known_n", "purchased"],
    "community_memberships": ["communities"],
    "employment": ["employer"],
    "satisfaction": ["satisfaction"],
    "beliefs": ["beliefs"],
    "current_commitments": ["commit_start_min", "commit_len_min", "wake_at"],
    "behavioral_history": ["recent", "n_sessions", "last_activity", "n_purchases"],
}
