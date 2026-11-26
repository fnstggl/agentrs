"""Kernel module `feed` (host `feed`): distribution on every platform (workstream C2; MASTER_SPEC Part 4.5).

Config (module config; evaluator-side - no tool, observation or visible event reveals it):

    {"platforms": {"<platform>": {"k": 10, "pool": 300, "pool_days": 7, "inbox_slots": 2,
                                  "recommenders": [{"name": "interest", "weight": 1.0, "params": {...}}, ...],
                                  "schedule": [{"day": 3, "recommenders": [...]}, ...]}},
     "inbox": 4, "share_delay_s": 900, "train_interval_s": 21600, "behaviour": {...},
     "calibrated": {"<platform>": {"open": "news_click@1"}}}

`calibrated` replaces the built-in decision to open an item on that platform with a fitted model of workstream E
(`crucible.calibration.load`, checkpoint hashes verified at install); see `consumption.CalibratedOpen`.

* The population's browse activity (`crowd`, or B's `pop` when merged) hands this module the activated cohort; each
  individual picks a platform by its learned habit (softmax of `feed_habit`) and runs a session there
  (`consumption.Engine`).
* A platform's recommender mixture is chosen by day from `recommenders` and `schedule` (the change takes effect at
  the start of that simulated day; process `feed.daily` records it as an internal `feed.mixture` event), or by an
  operator override (`set_mixture`, recorded the same way). The materialised mixtures live in
  `world.runtime["recsys_private"]`; overrides in the internal table `feed_private`.
* Process `feed.train` refits every learned recommender used on a platform (collaborative filtering, engagement
  prediction) from the platform's recorded batches; fits are recorded inference (`substrate.inference.recorded`)
  and stored by the internal `feed.model` event.

Tools for in-world actors that are population individuals (`p<i>`): `feed.platforms`, `feed.home` (your feed:
impressions are recorded), `feed.open` (go to an item by its id or link), `feed.search`, `feed.watch` (read or
watch an item you were shown; optionally like, share with your contacts, follow its creator). None of them returns
which recommender placed an item, any score or any parameter.
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np

from ...core import schema as S
from ...core.canonical import canonical_bytes
from ...core.errors import Forbidden, InvalidAction, NotFound, ProtocolError
from ...core.kernel import SYSTEM, Ctx, Module, applier, endpoint, process, register_module
from ...substrate import artifacts, attention, bus, columns, ids, inference
from .. import audience as AUD
from .. import idset, textrep
from ..consumption import BUILTIN, CALIBRATED_OPEN, DEFAULT_PLATFORM, RETIRED_OPEN, Engine
from ..vocab import TOPIC_NAMES
from . import REGISTRY, Mixture, training

DAY = 86400
DEFAULT_MIX = [{"name": "interest", "weight": 1.0}, {"name": "social", "weight": 1.0},
               {"name": "fresh", "weight": 0.5}, {"name": "explore", "weight": 0.3}]
DEFAULTS = {"platforms": {"posts": {"recommenders": DEFAULT_MIX}, "articles": {"recommenders": DEFAULT_MIX}},
            "inbox": 4, "share_delay_s": 900, "train_interval_s": 21600, "behaviour": {}, "calibrated": {}}
_PCFG_KEYS = {"k", "pool", "pool_days", "inbox_slots", "recommenders", "schedule"}


def _mod(world) -> "Feed":
    m = world.modules.get("feed")
    if m is None:
        raise ProtocolError("this world has no 'feed' module")
    return m  # type: ignore[return-value]


def _check_entries(entries) -> list[dict]:
    if not isinstance(entries, list) or not entries:
        raise ProtocolError("a mixture is a non-empty list of {name, weight, params}")
    out = []
    for e in entries:
        if not isinstance(e, dict) or e.get("name") not in REGISTRY or set(e) - {"name", "weight", "params"}:
            raise ProtocolError(f"bad mixture entry {e!r} (recommenders: {sorted(REGISTRY)})")
        out.append({"name": e["name"], "weight": float(e.get("weight", 1.0)), "params": dict(e.get("params") or {})})
    Mixture(out)                                     # validates names, weights and parameters
    return out


def set_mixture(ctx: Ctx, platform: str, entries: list[dict]) -> None:
    """Operator intervention: change a platform's mixture from now on (recorded as an internal event)."""
    m = _mod(ctx.world)
    m.platform_cfg(platform)
    ents = _check_entries(copy.deepcopy(entries))
    ver = int(ctx.state.meta.get(f"feed.override.{platform}", 0)) + 1
    ctx.emit("feed.mixture_set", {"platform": platform, "entries": ents, "from_t": int(ctx.now), "version": ver},
             vis="internal")


@register_module
class Feed(Module):
    """Recommendation and the consumption engine for every platform (see the module docstring)."""
    name = "feed"
    host = "feed"
    version = "1"
    depends = ("aud",)

    def __init__(self, config=None):
        super().__init__(config)
        self.cfg = copy.deepcopy(DEFAULTS)
        for k, v in self.config.items():
            if k not in DEFAULTS:
                raise ProtocolError(f"feed: unknown config key {k!r}")
            self.cfg[k] = copy.deepcopy(v)
        self.platforms = sorted(self.cfg["platforms"])
        if not self.platforms:
            raise ProtocolError("feed: at least one platform")
        self.versions: dict[str, list[tuple[int, list[dict]]]] = {}
        for p, pc in self.cfg["platforms"].items():
            if set(pc) - _PCFG_KEYS:
                raise ProtocolError(f"feed: platform {p}: unknown keys {sorted(set(pc) - _PCFG_KEYS)}")
            vers = [(0, _check_entries(pc.get("recommenders") or DEFAULT_MIX))]
            for ent in sorted(pc.get("schedule") or [], key=lambda e: int(e["day"])):
                d = int(ent["day"])
                if d <= vers[-1][0]:
                    raise ProtocolError(f"feed: platform {p}: schedule days must be increasing and > 0")
                vers.append((d, _check_entries(ent["recommenders"])))
            self.versions[p] = vers
        self.R = int(self.cfg["inbox"])
        if not 1 <= self.R <= 32:
            raise ProtocolError("feed: inbox must be in [1, 32]")
        unknown = set(self.cfg["behaviour"]) - set(BUILTIN)
        if unknown:
            raise ProtocolError(f"feed: unknown behaviour parameters {sorted(unknown)}")
        for p, c in self.cfg["calibrated"].items():
            if p not in self.cfg["platforms"] or set(c) - {"open", "affinity_space", "content_kind"} or \
                    c.get("open") not in CALIBRATED_OPEN:
                raise ProtocolError(f"feed: calibrated must map a platform to {{'open': one of "
                                    f"{sorted(CALIBRATED_OPEN)}, 'affinity_space': ...}}, not {p!r}: {c!r} "
                                    f"(retired: {RETIRED_OPEN})")

    # ------------------------------------------------------------------ configuration
    def platform_cfg(self, platform: str) -> dict:
        pc = self.cfg["platforms"].get(platform)
        if pc is None:
            raise NotFound(f"no platform {platform!r}")
        return {k: v for k, v in {**DEFAULT_PLATFORM, **pc}.items() if k in DEFAULT_PLATFORM}

    def platform_index(self, platform: str) -> int:
        return self.platforms.index(platform)

    def _private(self, world) -> dict:
        priv = world.runtime.get("recsys_private")
        if priv is None:
            priv = world.runtime["recsys_private"] = {"mixtures": {}}
        return priv

    def mixture_entries(self, world, platform: str, now: int) -> tuple[str, list[dict]]:
        """(version id, entries) of the mixture in force on `platform` at time `now` (evaluator side)."""
        ov = world.state.get("feed_private", platform)
        if ov is not None and ov["from_t"] <= now:
            return f"o{ov['version']}", ov["entries"]
        day = int(now // DAY)
        vid, ents = "v0", self.versions[platform][0][1]
        for j, (d, e) in enumerate(self.versions[platform]):
            if d <= day:
                vid, ents = f"v{j}", e
        return vid, ents

    def mixture(self, platform: str, now: int) -> Mixture:
        w = self.world
        vid, ents = self.mixture_entries(w, platform, now)
        cache = self._private(w)["mixtures"]
        key = (platform, vid, hashlib.sha256(canonical_bytes(ents)).hexdigest())
        mx = cache.get(key)
        if mx is None:
            mx = cache[key] = Mixture(copy.deepcopy(ents))
        return mx

    # ------------------------------------------------------------------ install
    def install(self, world):
        self.aud = world.modules["aud"]
        for p in self.platforms:
            self.aud.platform_code(p)                    # every feed platform is an aud platform
        n = columns.rows(world, "pop")
        columns.declare(world, "pop", n, {"feed_inbox": ("int32", (self.R,), -1),
                                          "feed_inbox_gen": ("uint8", (self.R,), 0),
                                          "feed_social": ("int32", (self.R,), -1),
                                          "feed_habit": ("float32", (len(self.platforms),), 0.0)})
        if self.cfg["calibrated"]:                       # per-individual consumption history by topic
            columns.declare(world, "pop", n, {"feed_topic_views": ("uint16", (len(TOPIC_NAMES),), 0)})
        world.state.declare_table("feed_private")
        world.state.declare_table("feed_models")
        self.engine = Engine(self)
        self._private(world)
        if "crowd" in world.modules:
            from ..scale import register_activity
            register_activity(world, "browse", self.browse, owner="feed")
        elif "pop" in world.modules:
            from ... import population
            reg = getattr(population, "register_activity", None)
            if reg is None:
                raise ProtocolError("feed: the population module does not provide register_activity")
            reg(world, "browse", self.browse, owner="feed")
        else:
            raise ProtocolError("feed needs a population module (crowd or pop) listed before it")
        bus.enforce(world, owner="feed")

    def on_load(self, world):
        world.runtime.pop("recsys_private", None)
        self._private(world)

    def genesis(self, ctx):
        ti = int(self.cfg["train_interval_s"])
        if any(REGISTRY[e["name"]].learned for vers in self.versions.values() for _, ents in vers for e in ents):
            ctx.schedule(ti, "feed.train", {}, prio=55, period=ti, key="feed.train")
        if any(len(v) > 1 for v in self.versions.values()):
            ctx.schedule(DAY, "feed.daily", {}, prio=5, period=DAY, key="feed.daily")
        for p in self.platforms:
            ctx.emit("feed.mixture", {"platform": p, "version": "v0", "day": 0, "names":
                                      [e["name"] for e in self.versions[p][0][1]]}, vis="internal")

    # ------------------------------------------------------------------ the population's browse activity
    def browse(self, ctx: Ctx, idx: np.ndarray, now: int) -> None:
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            return
        P = len(self.platforms)
        if P == 1:
            self.engine.run(ctx, self.platforms[0], idx)
            return
        hab = columns.col(ctx.world, "pop", "feed_habit")[idx].astype(np.float64)
        pr = np.exp(hab - hab.max(axis=1, keepdims=True))
        pr /= pr.sum(axis=1, keepdims=True)
        r = ctx.rng("feed.platform_choice").random((idx.size, 1))
        choice = np.minimum((r > np.cumsum(pr, axis=1)).sum(axis=1), P - 1)
        for j, p in enumerate(self.platforms):
            sel = idx[choice == j]
            if sel.size:
                self.engine.run(ctx, p, sel)

    # ------------------------------------------------------------------ processes
    @process("daily")
    def pr_daily(self, ctx):
        day = int(ctx.now // DAY)
        for p in self.platforms:
            for j, (d, ents) in enumerate(self.versions[p]):
                if d == day and j > 0:
                    ctx.emit("feed.mixture", {"platform": p, "version": f"v{j}", "day": day,
                                              "names": [e["name"] for e in ents]}, vis="internal")

    @process("train")
    def pr_train(self, ctx):
        for p in self.platforms:
            seen = set()
            for _, ents in self.versions[p] + [(0, self.mixture_entries(ctx.world, p, ctx.now)[1])]:
                for e in ents:
                    cls = REGISTRY[e["name"]]
                    if not cls.learned or e["name"] in seen:
                        continue
                    seen.add(e["name"])
                    self.fit(ctx, p, e)

    def fit(self, ctx: Ctx, platform: str, entry: dict) -> bool:
        rec = REGISTRY[entry["name"]](**entry.get("params", {}))
        job = rec.fit(ctx, platform)
        if job is None:
            return False
        run, info = job
        mh = hashlib.sha256(canonical_bytes({"name": rec.name, "version": rec.version, "params": rec.p})).hexdigest()
        arrays = inference.recorded(ctx, "recsys.fit", f"{rec.name}@{rec.version}", mh,
                                    {"platform": platform, "t": int(ctx.now), **info}, run)
        ref = artifacts.put_array_blob(ctx, {k: np.asarray(v) for k, v in arrays.items()})
        meta = {k: v for k, v in info.items() if k != "data_sha256"}
        ctx.emit("feed.model", {"platform": platform, "rec": rec.name, "ref": ref, "t": int(ctx.now), "meta": meta},
                 vis="internal")
        return True

    # ------------------------------------------------------------------ appliers
    @applier("feed.mixture")
    def ap_mixture(self, ev):
        st = self.world.state
        st.meta_set(f"feed.changes.{ev.payload['platform']}",
                    int(st.meta.get(f"feed.changes.{ev.payload['platform']}", -1)) + 1)

    @applier("feed.mixture_set")
    def ap_mixture_set(self, ev):
        p = ev.payload
        st = self.world.state
        st.put("feed_private", p["platform"], {"entries": p["entries"], "from_t": p["from_t"],
                                                "version": p["version"]})
        st.meta_set(f"feed.override.{p['platform']}", p["version"])
        st.meta_set(f"feed.changes.{p['platform']}", int(st.meta.get(f"feed.changes.{p['platform']}", -1)) + 1)

    @applier("feed.model")
    def ap_model(self, ev):
        p = ev.payload
        w = self.world
        arrs = artifacts.get_array_blob(w, p["ref"])
        for k, a in arrs.items():
            w.state.set_array(training.model_key(p["platform"], p["rec"], k), np.array(a))
        w.state.put("feed_models", f"{p['platform']}.{p['rec']}", {"ref": p["ref"], "t": p["t"]})

    # ------------------------------------------------------------------ tools (population individuals)
    def _me(self, ctx) -> int:
        i = ids.index(ctx.actor)
        if i is None or i >= columns.rows(ctx.world, "pop"):
            raise InvalidAction("feeds are for population individuals (principals p<i>)")
        return i

    def _platform(self, platform):
        if platform not in self.platforms:
            raise NotFound(f"no platform {platform!r}")
        return platform

    def _spend(self, ctx, i, seconds, activity) -> float:
        left = float(columns.col(ctx.world, "pop", "attention_left_min")[i]) * 60.0
        s = float(min(max(seconds, 0.0), left))
        if s > 0:
            attention.spend(ctx, np.array([i]), np.array([s / 60.0]), activity=activity)
        return s

    @endpoint("set_mixture", S.obj({"platform": S.s_str("platform name", maxLength=32),
                                    "recommenders": S.s_arr(S.s_any_obj("{name, weight, params}"), maxItems=16)},
                                   ["platform", "recommenders"]), cost_s=0)
    def ep_set_mixture(self, ctx, platform, recommenders):
        """Operator only (the simulator, through a system call): change how a platform distributes content from
        now on. Refused for every in-world actor."""
        if ctx.actor != SYSTEM:
            raise Forbidden("only the simulator operator can change a platform's distribution")
        try:
            set_mixture(ctx, self._platform(platform), recommenders)
        except ProtocolError as e:
            raise InvalidAction(str(e)) from None
        return {"platform": platform, "ok": True}

    @endpoint("platforms", S.obj(), read_only=True, cost_s=0)
    def ep_platforms(self, ctx):
        """The platforms of this world and the kinds of content each distributes."""
        out = []
        for p in self.platforms:
            kinds = sorted({ctx.state.tables["aud_item_info"][str(int(i))]["kind"]
                            for i in AUD.items_on(ctx.world, p)[:200]})
            out.append({"platform": p, "kinds": kinds, "items": int(AUD.items_on(ctx.world, p).size)})
        return {"platforms": out}

    @endpoint("home", S.obj({"platform": S.s_str("platform name", maxLength=32),
                             "k": S.s_int("how many items", minimum=1, maximum=50, default=10)}, ["platform"]),
              cost_s=2)
    def ep_home(self, ctx, platform, k=10):
        """Open your feed on a platform: the items the platform shows you now (each one is recorded as shown to
        you, and scrolling past them takes a little of your attention)."""
        i = self._me(ctx)
        self._platform(platform)
        idx = np.array([i], dtype=np.int64)
        feed, src, gen, used = self.engine.build_feed(ctx, platform, idx, ctx.rng("feed.home"))
        feed, src, gen = feed[:, :k], src[:, :k], gen[:, :k]
        scroll = float(self.engine.b["scroll_s"])
        left = float(columns.col(ctx.world, "pop", "attention_left_min")[i]) * 60.0
        n_fit = int(min(int((feed[0] >= 0).sum()), left // scroll)) if scroll > 0 else int((feed[0] >= 0).sum())
        items = [int(x) for x in feed[0] if x >= 0][:n_fit]
        if not items:
            return {"platform": platform, "items": []}
        pos = np.arange(len(items))
        AUD.impressions(ctx, platform, np.full(len(items), i), np.array(items), pos, source=src[0, :len(items)],
                        gen=gen[0, :len(items)])
        self._spend(ctx, i, scroll * len(items), f"feed.{platform}")
        if used.any():
            ring = columns.col(ctx.world, "pop", "feed_inbox")[idx].copy()
            ring[used[:, :ring.shape[1]]] = -1
            columns.batch(ctx, "pop", "feed.inbox_seen", idx, {"feed_inbox": ring}, mode="set")
        return {"platform": platform, "items": [self.aud.public_item(ctx.world, x) for x in items]}

    @endpoint("open", S.obj({"item": S.s_int("item id", minimum=0, maximum=2 ** 31 - 1)}, ["item"]), cost_s=1)
    def ep_open(self, ctx, item):
        """Go to an item directly (by its id, for example from a link someone sent you); it is recorded as shown
        to you. Read or watch it with feed.watch."""
        i = self._me(ctx)
        if item >= AUD.n_items(ctx.world) or bool(AUD.item_col(ctx.world, "removed")[item]):
            raise NotFound(f"no item {item}")
        rec = ctx.state.tables["aud_item_info"][str(int(item))]
        AUD.impressions(ctx, rec["platform"], np.array([i]), np.array([item]), np.array([0]),
                        source=np.array([AUD.SOURCE_CODE["direct"]]))
        self._spend(ctx, i, float(self.engine.b["scroll_s"]), f"feed.{rec['platform']}")
        return self.aud.public_item(ctx.world, item)

    @endpoint("search", S.obj({"platform": S.s_str("platform name", maxLength=32),
                               "q": S.s_str("search words", maxLength=200),
                               "k": S.s_int(minimum=1, maximum=50, default=10)}, ["platform", "q"]), cost_s=2)
    def ep_search(self, ctx, platform, q, k=10):
        """Search a platform's items by their text (results are recorded as shown to you)."""
        from .algorithms import _bm25_matrix
        i = self._me(ctx)
        self._platform(platform)
        terms = sorted({t for t in textrep.tokenize(q) if t not in textrep.STOPWORDS})[:20]
        cand = AUD.items_on(ctx.world, platform)
        if not terms or cand.size == 0:
            return {"platform": platform, "items": []}
        W = _bm25_matrix(ctx.world, cand, terms, 1.2, 0.75)
        s = W.sum(axis=0)
        order = np.argsort(-s, kind="stable")
        hits = [int(cand[j]) for j in order[:k] if s[j] > 0]
        if hits:
            AUD.impressions(ctx, platform, np.full(len(hits), i), np.array(hits), np.arange(len(hits)),
                            source=np.full(len(hits), AUD.SOURCE_CODE["search_tool"]))
            self._spend(ctx, i, float(self.engine.b["scroll_s"]) * len(hits), f"feed.{platform}")
        return {"platform": platform, "items": [self.aud.public_item(ctx.world, x) for x in hits]}

    @endpoint("watch", S.obj({"item": S.s_int("item id", minimum=0, maximum=2 ** 31 - 1),
                              "seconds": S.s_num("how long you read or watch", minimum=0, maximum=86400),
                              "like": S.s_bool(default=False), "share": S.s_bool("share it with your contacts",
                                                                                 default=False),
                              "follow": S.s_bool("follow its creator", default=False)}, ["item", "seconds"]),
              cost_s=5)
    def ep_watch(self, ctx, item, seconds, like=False, share=False, follow=False):
        """Read or watch an item you were shown (feed, search, a share or feed.open), for up to `seconds` (at most
        its duration and the attention you have left today). It counts as a view when you spend at least
        min(30 s, half its duration) on it."""
        i = self._me(ctx)
        if item >= AUD.n_items(ctx.world) or bool(AUD.item_col(ctx.world, "removed")[item]):
            raise NotFound(f"no item {item}")
        if not idset.contains(ctx.state, idset.key("ex", int(item)), np.array([i]))[0]:
            raise InvalidAction("you have not been shown this item (open it first)")
        rec = ctx.state.tables["aud_item_info"][str(int(item))]
        dur = float(AUD.item_col(ctx.world, "duration_s")[item])
        want = float(np.float32(min(float(seconds), dur)))
        got = self._spend(ctx, i, want, f"feed.{rec['platform']}")
        got = float(np.float32(got))
        if got <= 0:
            return {"item": item, "watched_s": 0.0, "counted_as_view": False, "completed": False}
        comp = got >= AUD.COMPLETE_FRAC * dur
        AUD.consumption(ctx, rec["platform"], np.array([i]), np.array([item]), np.array([got]), np.array([comp]),
                        np.array([bool(share)]), np.array([bool(follow)]), np.array([bool(like)]))
        graph = ctx.world.runtime.get("crowd.graph")
        if share and graph is not None:
            con = graph.contacts(np.array([i]))[0]
            con = con[con != i]
            g = int(columns.col(ctx.world, "pop", "aud_i_gen")[i]) if \
                int(columns.col(ctx.world, "pop", "aud_i_item")[i]) == item else 0
            self.engine.deliver(ctx, con, np.full(con.size, item), np.full(con.size, min(g + 1, 255)))
        view = got >= float(AUD.view_threshold(dur)) and rec["creator"] != ctx.actor
        return {"item": item, "watched_s": round(got, 3), "counted_as_view": bool(view), "completed": bool(comp)}


bus.event_type("feed.model", S.obj({"platform": {"type": "string"}, "rec": {"type": "string"},
                                    "ref": {"type": "string", "pattern": artifacts.REF_RE.pattern},
                                    "t": {"type": "integer"}, "meta": {"type": "object", "additionalProperties": True}},
                                   ["platform", "rec", "ref", "t"]), owner="feed",
               doc="a learned recommender was refitted (parameters in the array blob; internal)")
