"""Population scaling profiles (MASTER_SPEC Part 3.3).

| profile     | lightweight | adaptive | strategic | semantic |
|-------------|------------:|---------:|----------:|---------:|
| development |       1,000 |       32 |         8 |        2 |
| standard    |      10,000 |      256 |        32 |        8 |
| large       |     100,000 |    1,000 |       100 |       20 |
| internet    |  10,000,000 | configurable (defaults 1,000 / 100 / 20) |

Every class is a row of the same columnar table, so the classes share identity, attention, money and history;
classes 2-4 add a shared model (never one checkpoint per participant). The internet profile uses compact
dimensions (fewer known-resource slots, smaller interest and memory vectors) so 10M+ rows fit in memory, and
`init: "derived"` so the initial state is not written as one multi-gigabyte genesis blob.
"""
from __future__ import annotations

PROFILES: dict[str, dict] = {
    "development": {"lightweight": 1_000, "adaptive": 32, "strategic": 8, "semantic": 2,
                    "dims": {}, "init": "batch"},
    "standard": {"lightweight": 10_000, "adaptive": 256, "strategic": 32, "semantic": 8,
                 "dims": {}, "init": "batch"},
    "large": {"lightweight": 100_000, "adaptive": 1_000, "strategic": 100, "semantic": 20,
              "dims": {}, "init": "batch"},
    "internet": {"lightweight": 10_000_000, "adaptive": 1_000, "strategic": 100, "semantic": 20, "configurable": True,
                 "dims": {"D": 16, "K": 4, "H": 8, "M": 8, "C": 4, "G": 2, "R": 2, "Q": 2, "A": 2, "U": 2},
                 "init": "derived"},
}
# what Part 3.3 specifies, for checking (the internet row says "10M+" and "configurable")
SPEC_TABLE = {"development": (1_000, 32, 8, 2), "standard": (10_000, 256, 32, 8), "large": (100_000, 1_000, 100, 20),
              "internet": (10_000_000, None, None, None)}
CLASS_KEYS = ("lightweight", "adaptive", "strategic", "semantic")


def resolve(cfg: dict) -> dict:
    """Class counts and dimensions for a module config `{profile, n?, classes?: {adaptive, strategic, semantic}}`.
    Returns {profile, counts: {lightweight, adaptive, strategic, semantic}, n, dims, init}."""
    name = cfg.get("profile", "development")
    if name not in PROFILES:
        raise ValueError(f"unknown population profile {name!r} (one of {sorted(PROFILES)})")
    p = PROFILES[name]
    counts = {k: int(p[k]) for k in CLASS_KEYS}
    if cfg.get("n") is not None:
        counts["lightweight"] = int(cfg["n"])
    for k, v in (cfg.get("classes") or {}).items():
        if k not in CLASS_KEYS[1:]:
            raise ValueError(f"unknown participant class {k!r}")
        counts[k] = int(v)
    if any(v < 0 for v in counts.values()):
        raise ValueError("class counts must be >= 0")
    d = dict(p["dims"])
    d.update(cfg.get("dims") or {})
    if "interest_dim" in cfg:
        d["D"] = int(cfg["interest_dim"])
    return {"profile": name, "counts": counts, "n": sum(counts.values()), "dims": d,
            "init": cfg.get("init", p["init"])}


def matches_spec(name: str) -> bool:
    """Do the profile's class counts match the Part 3.3 table?"""
    p, s = PROFILES[name], SPEC_TABLE[name]
    if name == "internet":
        return p["lightweight"] >= s[0] and bool(p.get("configurable"))
    return (p["lightweight"], p["adaptive"], p["strategic"], p["semantic"]) == s
