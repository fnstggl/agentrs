"""Live observational learner: sources (mocked HTTP), sampling, gating, and the full loop on a fake
real-time source with second-scale horizons (no network)."""

from __future__ import annotations

import json
import time

import httpx
import pytest

from agentrs.live.bluesky import BlueskySource, JetstreamSampler, post_uri
from agentrs.live.hackernews import HackerNewsSource
from agentrs.live.reddit import RedditSource, reddit_declaration, reddit_source_from_env
from agentrs.live.spec import payload
from agentrs.sensors.source_adapter import RawEvent, SourceAdapter, SourceDeclaration


def _commit(did, rkey, text="hello world", op="create", reply=False, langs=("en",), t=None):
    rec = {"text": text, "createdAt": "2026-11-09T00:00:00Z", "langs": list(langs)}
    if reply:
        rec["reply"] = {"root": {}, "parent": {}}
    return {"did": did, "time_us": int((t or time.time()) * 1e6), "kind": "commit",
            "commit": {"operation": op, "collection": "app.bsky.feed.post", "rkey": rkey, "cid": "c", "record": rec}}


def test_jetstream_sampler_filters_samples_and_tracks_deletions():
    s = JetstreamSampler(target_per_min=60, langs=["en"], seed=1)
    t0 = 1_000_000.0
    for i in range(201):  # 200 eligible posts in 30 s -> 400/min, keep prob 60/400
        s.handle(_commit("did:plc:a", f"r{i}", t=t0 + i * 0.15), now=t0 + i * 0.15)
    assert s.stats["kept"] <= 1 and s.rate_per_min > 0  # nothing kept while the rate is being estimated
    s.handle(_commit("did:plc:a", "reply", reply=True), now=t0 + 31)
    s.handle(_commit("did:plc:a", "de", langs=("de",)), now=t0 + 31)
    assert s.stats["eligible"] == 201
    for i in range(2000):
        s.handle(_commit("did:plc:b", f"k{i}"), now=t0 + 31 + i * 0.01)
    kept = s.stats["kept"]
    assert 0.07 * 2000 < kept < 0.25 * 2000, kept
    posts, _ = s.drain()
    assert all(0 < p["keep_prob"] <= 1 for p in posts)
    s.tracked[post_uri("did:plc:b", "k1")] = "did:plc:b"
    s.tracked[post_uri("did:plc:c", "z")] = "did:plc:c"
    s.handle(_commit("did:plc:b", "k1", op="delete"))
    s.handle(_commit("did:plc:x", "untracked", op="delete"))
    s.handle({"did": "did:plc:c", "time_us": 1, "kind": "account", "account": {"active": False, "status": "deleted"}})
    _, dels = s.drain()
    assert {d[0] for d in dels} == {post_uri("did:plc:b", "k1"), post_uri("did:plc:c", "z")}


def _bsky_transport(posts_state: dict, profiles: dict):
    def h(req: httpx.Request):
        if req.url.path.endswith("app.bsky.actor.getProfiles"):
            ids = req.url.params.get_list("actors")
            return httpx.Response(200, json={"profiles": [profiles[d] for d in ids if d in profiles]})
        if req.url.path.endswith("app.bsky.feed.getPosts"):
            uris = req.url.params.get_list("uris")
            assert len(uris) <= 25
            return httpx.Response(200, json={"posts": [posts_state[u] for u in uris if u in posts_state]})
        return httpx.Response(404)

    return httpx.MockTransport(h)


def test_bluesky_source_profiles_optout_counts_and_deletion():
    sampler = JetstreamSampler(target_per_min=10)
    sampler.start = lambda: None  # no socket in tests
    profiles = {"did:plc:a": {"did": "did:plc:a", "followersCount": 1234, "postsCount": 50, "followsCount": 9,
                              "createdAt": "2023-05-01T00:00:00Z", "labels": []},
                "did:plc:o": {"did": "did:plc:o", "followersCount": 5, "labels": [{"val": "!no-unauthenticated"}]}}
    ua, uo = post_uri("did:plc:a", "1"), post_uri("did:plc:o", "2")
    state = {ua: {"uri": ua, "likeCount": 12, "replyCount": 3, "repostCount": 1, "quoteCount": 1}}
    src = BlueskySource(sampler=sampler, client=httpx.Client(transport=_bsky_transport(state, profiles)), pseudonym_key="k")
    rec = {"text": "my new paper on ants", "langs": ["en"],
           "embed": {"$type": "app.bsky.embed.external", "external": {"uri": "https://arxiv.org/abs/1", "title": "Ants"}}}
    sampler.posts.extend([{"uri": ua, "did": "did:plc:a", "time": time.time(), "record": rec, "keep_prob": 0.1},
                          {"uri": uo, "did": "did:plc:o", "time": time.time(), "record": rec, "keep_prob": 0.1}])
    raws, cur = src.poll(0)
    assert [r.object_ref for r in raws] == [ua] and src.stats["skipped_opt_out"] == 1
    (obs,) = src.normalize(raws[0])
    assert obs.context_before["author_followers"] == 1234 and obs.context_before["space"] == "link"
    assert obs.context_before["domain"] == "arxiv.org" and "did:plc" not in (obs.actor_ref or "")
    assert "AUTHOR: followers=1234" in obs.context_text()
    p, comp = src.fetch_state(ua)
    assert comp == "complete" and p["counts"] == {"reactions": 12, "responses": 3, "shares": 2}
    assert p["occurred"]["engagement_10"] and not p["occurred"]["engagement_100"]
    state.pop(ua)  # the author deleted it
    src._cache.clear()
    p, _ = src.fetch_state(ua)
    assert p["state"] == "removed"
    raws, _ = src.poll(cur)
    assert raws[0].type == "item_deleted"
    (fu,) = src.normalize(raws[0])
    assert fu.type == "item_deleted" and fu.object_ref == ua


def test_hackernews_source_is_live_only_and_propagates_deletion():
    now = time.time()
    items = {101: {"id": 101, "type": "story", "by": "pg", "time": now - 30, "title": "Show HN: a thing",
                   "url": "https://example.com/x", "score": 1, "descendants": 0},
             102: {"id": 102, "type": "story", "by": "old", "time": now - 7200, "title": "old", "score": 5},
             103: {"id": 103, "type": "comment", "time": now}}
    listing = [100]

    def h(req):
        path = req.url.path
        if path.endswith("newstories.json"):
            return httpx.Response(200, json=listing)
        if "/item/" in path:
            return httpx.Response(200, json=items.get(int(path.split("/")[-1].split(".")[0])))
        if "/user/" in path:
            return httpx.Response(200, json={"karma": 4321})
        return httpx.Response(404)

    src = HackerNewsSource(client=httpx.Client(transport=httpx.MockTransport(h)), pseudonym_key="k")
    raws, cur = src.poll(0)
    assert raws == [] and cur == 100  # first poll only sets the starting point
    listing[:] = [103, 102, 101, 100]
    raws, cur = src.poll(cur)
    assert [r.object_ref for r in raws] == ["101"]  # 102 too old, 103 not a story
    (obs,) = src.normalize(raws[0])
    assert obs.context_before["space"] == "show" and obs.context_before["author_karma"] == 4321
    items[101].update(score=15, descendants=4)
    p, _ = src.fetch_state("101")
    assert p["counts"] == {"reactions": 14, "responses": 4} and p["occurred"]["engagement_10"]
    items[101]["deleted"] = True
    src.fetch_state("101")
    raws, _ = src.poll(cur)
    assert raws[0].type == "item_deleted" and raws[0].object_ref == "101"


def test_reddit_source_refuses_without_explicit_authorisation(monkeypatch):
    for k in ("AGENTRS_REDDIT_AUTHORIZATION_REF", "AGENTRS_REDDIT_TRAINING_AUTHORIZED"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(PermissionError, match="public visibility is not permission"):
        reddit_source_from_env("k")
    with pytest.raises(PermissionError):
        RedditSource(reddit_declaration("", False), "id", "sec", "ua", ["test"], "k")
    monkeypatch.setenv("AGENTRS_REDDIT_AUTHORIZATION_REF", "REF-1")
    with pytest.raises(PermissionError, match="training"):
        reddit_source_from_env("k")  # a reference alone is not enough: training must be stated explicitly


def test_reddit_source_official_api_shape_mocked():
    now = time.time()
    calls = []

    def h(req):
        calls.append(req.url.path)
        if req.url.path == "/api/v1/access_token":
            assert req.headers.get("authorization", "").startswith("Basic ")
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
        assert req.headers["authorization"] == "Bearer tok" and req.headers["user-agent"] == "test:agentrs:0.1"
        if req.url.path.endswith("/new"):
            kids = [{"data": {"name": "t3_b", "created_utc": now - 10, "title": "Q", "subreddit": "test", "is_self": True,
                              "selftext": "body", "author": "u1"}},
                    {"data": {"name": "t3_a", "created_utc": now - 50, "title": "P", "subreddit": "test", "author": "u2"}}]
            return httpx.Response(200, json={"data": {"children": kids}})
        if req.url.path == "/api/info":
            return httpx.Response(200, json={"data": {"children": [{"data": {"name": "t3_b", "score": 8, "num_comments": 2}}]}})
        return httpx.Response(404)

    src = RedditSource(reddit_declaration("REF-1", True), "id", "sec", "test:agentrs:0.1", ["test"], "k",
                       client=httpx.Client(transport=httpx.MockTransport(h)))
    raws, cur = src.poll(0)
    assert raws == [] and cur == pytest.approx(now - 10)
    raws, _ = src.poll(now - 30)
    assert [r.object_ref for r in raws] == ["t3_b"]
    p, _ = src.fetch_state("t3_b")
    assert p["counts"] == {"reactions": 7, "responses": 2}


# ---------------------------------------------------------------------------- end-to-end loop
class FakeLiveSource(SourceAdapter):
    """Real clock; posts containing 'great' collect likes quickly, others rarely."""

    def __init__(self):
        self.declaration = SourceDeclaration(
            source_id="fake", name="fake", access_method="authorized_api", operator="test",
            allowed_data_categories=["public_posts"], training_authorized=True, authorization_ref="test",
            time_base="real", min_poll_interval_s=0, collection_limits={"requests_per_minute": 6000})
        self.posts: dict[str, dict] = {}
        self.n = 0
        self.requests = 0

    def now(self):
        return time.time()

    def env_key(self, ref):
        return f"fake:{ref}"

    def poll(self, cursor):
        out = []
        for _ in range(6):
            self.n += 1
            good = self.n % 2 == 0
            ref = f"p{self.n}"
            self.posts[ref] = {"t": time.time(), "good": good, "deleted": False}
            out.append(RawEvent(provider_event_id=ref, type="item_created", at=time.time(), object_ref=ref,
                                data={"text": ("great news everyone" if good else "meh") + f" #{self.n}", "good": good}))
        return out, int(cursor or 0) + len(out)

    def normalize(self, raw):
        from agentrs.data.observed_actor_action import ObservedAction, ObservedActorAction

        return [ObservedActorAction(
            source_id="fake", provider_object_id=raw.object_ref, actor_type="human", actor_ref="actor-x",
            action_time=raw.at, action_time_known=True, observation_time=time.time(), time_base="real",
            context_before={"env": "fake", "space": "all", "author_followers": 10}, action=ObservedAction(
                operation="create_post", content=raw.data["text"]),
            permission_metadata={"categories": ["public_posts"]}, env_id="fake", group="fake:all")]

    def fetch_state(self, ref):
        self.requests += 1
        p = self.posts[ref]
        age = time.time() - p["t"]
        likes = int(age * 10) if p["good"] else 0
        return payload(likes, 1 if p["good"] else 0, 0, removed=False), "complete"


def test_live_learner_forecasts_first_learns_scores_and_resumes(tmp_path):
    from agentrs.live.service import LiveConfig, LiveLearner

    cfg = LiveConfig(workdir=str(tmp_path / "live"), sources=[], horizons=["1s", "2s"], outcome_window="3s",
                     backbone={"name": "scratch", "scratch": {"hidden_size": 32, "num_layers": 1, "num_heads": 2,
                                                              "vocab_size": 256}, "max_prompt_tokens": 64},
                     batch_size=8, steps_per_update=5, max_steps_per_update=10, min_new_examples=4, ingest_every_s=0,
                     observe_every_s=0, learn_every_s=0, score_every_s=0, checkpoint_every_s=1e9, retention_every_s=1e9,
                     threads=1, n_heads=2)
    src = FakeLiveSource()
    ll = LiveLearner(cfg, sources=[src])
    w0 = {k: v.clone() for k, v in ll.model.named_parameters() if v.requires_grad}
    trained = False
    t_end = time.time() + 9
    while time.time() < t_end:
        did = ll.tick()
        trained |= bool((did.get("learn") or {}).get("trained"))
        time.sleep(0.3)
    assert trained, ll.errors
    moved = sum(float((p - w0[k]).abs().sum()) for k, p in ll.model.named_parameters() if k in w0)
    assert moved > 0
    # every post has a model forecast and a baseline forecast stored before its first observation
    for a in ll.ledger.actions()[:20]:
        fc = ll.ledger.forecasts(a["action_id"])
        obs = ll.ledger.observations(a["action_id"])
        assert {f["kind"] for f in fc} >= {"initial", "baseline"}
        if obs:
            assert max(f["created_time"] for f in fc if f["kind"] in ("initial", "baseline")) <= obs[0]["actual_time"]
    ll.score()
    rep = ll.scorer.report()
    assert rep["n_event_rows"] > 0 and "fake" in rep["overall"]
    st = ll.write_status()
    assert json.loads((tmp_path / "live" / "status.json").read_text())["optimizer_steps"] == st["optimizer_steps"] > 0
    ll.save()
    ck = ll.model.checkpoint_id
    ll2 = LiveLearner(cfg, sources=[FakeLiveSource()])
    assert ll2.state["resumed_from"] == ck and ll2.trainer.opt.global_step == ll.trainer.opt.global_step
    for (k, a), (_, b) in zip(ll.model.named_parameters(), ll2.model.named_parameters()):
        assert bool((a == b).all()), k
    assert len(ll2.trainer.replay) > 0


def test_deleted_posts_are_redacted_and_not_scored(tmp_path):
    from agentrs.live.service import LiveConfig, LiveLearner
    from agentrs.sensors.source_adapter import FollowUp

    class Deleting(FakeLiveSource):
        def poll(self, cursor):
            raws, cur = super().poll(cursor)
            if self.n > 6:
                raws.append(RawEvent(provider_event_id="del:p1", type="item_deleted", at=time.time(), object_ref="p1"))
            return raws, cur

        def normalize(self, raw):
            if raw.type == "item_deleted":
                return [FollowUp(object_ref=raw.object_ref, type="item_deleted", env_event_time=raw.at, actor_ref=None,
                                 by_original_actor=True, provider_event_id=raw.provider_event_id,
                                 categories=["public_posts"])]
            return super().normalize(raw)

    cfg = LiveConfig(workdir=str(tmp_path / "live"), sources=[], horizons=["1s"], outcome_window="2s",
                     backbone={"name": "scratch", "scratch": {"hidden_size": 32, "num_layers": 1, "num_heads": 2,
                                                              "vocab_size": 256}, "max_prompt_tokens": 64},
                     ingest_every_s=0, observe_every_s=0, learn_every_s=1e9, score_every_s=1e9, threads=1, n_heads=2)
    ll = LiveLearner(cfg, sources=[Deleting()])
    ll.tick()
    ll.tick()
    from agentrs.sensors.deduplication import stable_action_id

    a = ll.ledger.action(stable_action_id("fake", "p1"))
    assert a["action"]["content"] is None and not a["training_eligible"]
    assert any(g["op"] == "redact" for g in ll.ledger.governance_log())


def test_network_keeps_the_starting_cursor_of_a_live_source(tmp_path):
    """A live source's first poll returns no events, only a starting point; it must be kept."""
    from agentrs.sensors.observation_network import ObservationSensorNetwork
    from agentrs.sensors.source_registry import SourceRegistry

    now = time.time()
    listing = [100]
    items = {101: {"id": 101, "type": "story", "by": "a", "time": now, "title": "new", "score": 1}}

    def h(req):
        if req.url.path.endswith("newstories.json"):
            return httpx.Response(200, json=listing)
        if "/item/" in req.url.path:
            return httpx.Response(200, json=items.get(int(req.url.path.split("/")[-1].split(".")[0])))
        return httpx.Response(200, json={"karma": 1})

    reg = SourceRegistry(tmp_path / "s.json")
    reg.register(HackerNewsSource(client=httpx.Client(transport=httpx.MockTransport(h)), pseudonym_key="k"))
    net = ObservationSensorNetwork(tmp_path, reg)
    assert net.ingest("hackernews")["new_actions"] == 0 and net.cursors["hackernews"] == 100
    listing[:] = [101, 100]
    assert net.ingest("hackernews")["new_actions"] == 1


def test_scoring_uses_the_actual_follow_up_time(tmp_path):
    """A like first seen by a poll made 38 s after the 10-min horizon must count as a positive."""
    from agentrs.live.prequential import PrequentialScorer, PrequentialStore
    from agentrs.live.spec import live_spec
    from agentrs.temporal.delayed_outcome_ledger import ActionRecord, DelayedOutcomeLedger, ObservationRecord

    led = DelayedOutcomeLedger(tmp_path / "l.sqlite")
    t0 = time.time() - 1000
    spec = live_spec()
    for i, liked in enumerate([True, False]):
        aid = f"a{i}"
        led.record_action(ActionRecord(action_id=aid, source_id="bluesky", source_type="third_party", actor_type="human",
                                       env_id="b", time_base="real", action_time=t0, observed_time=t0,
                                       context_text="c", context={"author_followers": 10}, action_str="post",
                                       action={"operation": "create_post"}, split="train", group="bsky:text",
                                       window_end=t0 + 4 * 86400))
        for kind, p in (("initial", 0.7), ("baseline", 0.3)):
            hp = {"horizon_s": 600.0, "event_prob": {e: p for e in spec.events}, "count_mean": {c: 1.0 for c in spec.counts}}
            led.record_forecast(aid, {"horizons": [hp]}, kind, info_cutoff_time=t0, created_time=t0, horizons=[600.0],
                                kind=kind)
        led.record_observation(ObservationRecord(action_id=aid, actual_time=t0 + 638, payload=payload(int(liked), 0, 0, False),
                                                 horizon_label="10m", intended_time=t0 + 600, completeness="complete",
                                                 source_id="bluesky", idem=f"o{i}", time_base="real"))
    sc = PrequentialScorer(led, spec, PrequentialStore(tmp_path / "p.sqlite"))
    sc.update()
    ys = {r["action_id"]: r["y"] for r in sc.store.q("SELECT * FROM ev WHERE event='positive_feedback'")}
    assert ys == {"a0": 1, "a1": 0}
