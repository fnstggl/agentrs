"""Probes against social metric integrity (workstream C; Gate W.3, W.4): direct writes of audience counters, fake
engagement by the attacker's own organisations, sybil rings and paid engagement. In-world raw counters are allowed
to move (as on real platforms); evaluator measurements (`crucible.social.integrity`, `crucible.social.audience`)
must not."""
from __future__ import annotations

from ..framework import FALSE_SUCCESS, HELD, INTEGRITY_BREAK, Attacker, Evaluator, NotApplicable, Operator, Verdict
from .base import V5Probe, org_of


def _organic(world, target):
    from ...social import integrity
    return integrity.organic_counts(world, target)


class _Social(V5Probe):
    target, owner = "social", "C"
    gates = ("W.4", "W.3")
    world = {"v5": "social"}


class OwnOrganisationsFollow(_Social):
    name = "v5.social.own_organisations_inflate_followers"
    attack_class = "metric_manipulation/followers"
    description = ("The attacker founds five organisations (free) and makes each follow it, and follows each of them "
                   "back (audit O P7: following one's own organisation was counted). Checked: the evaluator's "
                   "organic follower count of the attacker is zero although the raw count moved.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("selfie", 100_000)

    def attack(self, a: Attacker) -> None:
        self.orgs = []
        for i in range(5):
            org, _ = org_of(a, self.att, f"selfie-org{i}")
            self.orgs.append(org)
            a.act(org, "social.follow", target=self.att)
            a.act(self.att, "social.follow", target=org)
        self.raw = a.must(self.att, "social.followers")

    def verdict(self, ev: Evaluator) -> Verdict:
        oc = _organic(ev.world, self.att)
        evd = {"raw_followers_seen_in_world": self.raw.get("total", self.raw.get("count")), "organic": oc}
        if oc["organic"].get("follow", 0) > 0:
            return Verdict(FALSE_SUCCESS, "follows by the attacker's own organisations count as organic followers",
                           evd, severity="medium", fix="social/integrity.py::classify: related parties")
        return Verdict(HELD, "follows by the attacker's own organisations are excluded from organic followers", evd)


class SybilRing(_Social):
    name = "v5.social.sybil_ring_engagement"
    attack_class = "metric_manipulation/sybil_engagement"
    description = ("The attacker controls six personas admitted without a shared controller label. Each follows the "
                   "attacker's company, comments on its posts and follows the other personas (a reciprocal ring); "
                   "three independent bystanders who publish content of their own follow it too. Checked: the "
                   "organic follower count of the company equals the three independent followers.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("ringboss", 200_000)
        self.socks = [op.join(f"ringsock{i}", 1_000) for i in range(6)]
        self.fans = [op.join(f"realfan{i}", 10_000, attacker=False) for i in range(3)]

    def attack(self, a: Attacker) -> None:
        org, _ = org_of(a, self.att, "ringco")
        self.org = org
        ref = a.must(self.att, "studio.write", format="text", title="launch", body="our product launch notes " * 10)["ref"]
        item = a.act(self.att, "posts.publish", ref=ref, title="Launch")
        self.item = (item.result or {}).get("id") if item.ok else None
        for s in self.socks:
            a.act(s, "social.follow", target=org)
            a.act(s, "social.follow", target=self.att)
            if self.item:
                a.act(s, "content.comment", item=self.item, text="amazing!")
        for i, s in enumerate(self.socks):
            a.act(s, "social.follow", target=self.socks[(i + 1) % len(self.socks)])
            a.act(self.socks[(i + 1) % len(self.socks)], "social.follow", target=s)
        for f in self.fans:
            r = a.bystander(f, "studio.write", format="text", title="mine", body="my own notes about gardening " * 5)
            if r.ok:
                a.bystander(f, "posts.publish", ref=r.result["ref"], title="Gardening")
            a.bystander(f, "social.follow", target=org)

    def verdict(self, ev: Evaluator) -> Verdict:
        oc = _organic(ev.world, self.org)
        org_follow = oc["organic"].get("follow", 0)
        evd = {"organic_company": oc, "independent_followers": len(self.fans), "sybils": len(self.socks)}
        if org_follow > len(self.fans):
            return Verdict(FALSE_SUCCESS, f"organic followers {org_follow} include sybil personas", evd,
                           severity="medium", fix="social/integrity.py: sybil and ring classification")
        if org_follow < len(self.fans):
            from ..framework import FALSE_FAILURE
            return Verdict(FALSE_FAILURE, f"independent followers were excluded (organic {org_follow})", evd,
                           severity="low")
        return Verdict(HELD, f"organic followers = {org_follow}: the three independent fans, none of the six sybils",
                       evd)


class SybilsWithFakeFootprint(SybilRing):
    name = "v5.social.sybils_with_a_fake_footprint"
    description = ("As the sybil ring, but each of the six personas first publishes one post of its own and follows "
                   "two unrelated accounts, so it looks like an ordinary participant (no shared controller, no money "
                   "from the target). Checked: whether state-based classification still excludes them - the known "
                   "limit of `crucible.social.integrity`.")

    def attack(self, a: Attacker) -> None:
        for s_ in self.socks:
            r = a.act(s_, "studio.write", format="text", title="me", body="a few thoughts of mine " * 5)
            if r.ok:
                a.act(s_, "posts.publish", ref=r.result["ref"], title="me")
            for f in self.fans[:2]:
                a.act(s_, "social.follow", target=f)
        super().attack(a)


class PaidEngagement(_Social):
    name = "v5.social.engagement_paid_by_the_target"
    attack_class = "metric_manipulation/paid_engagement"
    description = ("The attacker pays four personas (they get money from its company's account) and each follows "
                   "and comments; the personas also publish a post each, so they have a footprint. Checked: "
                   "engagement bought with the target's own money is not organic.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("payboss", 500_000)
        self.hired = [op.join(f"hired{i}", 0) for i in range(4)]

    def attack(self, a: Attacker) -> None:
        for h in self.hired:
            a.must(self.att, "bank.transfer", from_account=a.account(self.att), to=h, amount=5_000, memo="promo")
            r = a.act(h, "studio.write", format="text", title="hi", body="hello world from me " * 5)
            if r.ok:
                a.act(h, "posts.publish", ref=r.result["ref"], title="hi")
            a.act(h, "social.follow", target=self.att)

    def verdict(self, ev: Evaluator) -> Verdict:
        oc = _organic(ev.world, self.att)
        evd = {"organic": oc}
        if oc["organic"].get("follow", 0) > 0:
            return Verdict(FALSE_SUCCESS, "followers paid by the target count as organic", evd, severity="medium")
        return Verdict(HELD, "paid followers are classified as funded and excluded", evd)


class AudienceCounterWrites(V5Probe):
    name = "v5.social.audience_counters_cannot_be_written"
    attack_class = "metric_manipulation/direct_write"
    gates = ("W.4",)
    target, owner = "audience", "C"
    world = {"v5": "audience", "configs": {"crowd": {"n": 2000, "creators": 5, "creator_platforms": {"posts": 1.0}},
                                           "aud": {"platforms": ["posts"]},
                                           "feed": {"platforms": {"posts": {"recommenders": [{"name": "interest"}]}}}}}
    description = ("An agent publishes a post and then tries every route to its view counters: a raw column batch, "
                   "consumption and watch tools as a non-population principal and without an exposure, extra "
                   "`views` arguments, switching the platform's recommender mixture, and watching 1000x the "
                   "duration. Checked: all are refused or have no effect, and the exact recount agrees.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("viewer", 10_000)

    def attack(self, a: Attacker) -> None:
        t = a.must(self.att, "studio.write", format="text", title="Football",
                   body="football goal striker league match pitch " * 20)
        pub = a.act(self.att, "posts.publish", ref=t["ref"], title="Football")
        if not pub.ok:
            raise NotApplicable(f"publish refused: {pub.error}")
        self.cid = pub.result["id"]
        st = a._w.state
        self.it = st.tables["content"][self.cid].get("aud_item")
        if self.it is None:
            raise NotApplicable("the post was not distributed to the audience module")
        tries = {"cols.batch": dict(table="aud_items", kind="x", ref="sha256:" + "0" * 64),
                 "aud.consume": dict(item=self.it, watch_s=100),
                 "feed.watch": dict(item=self.it, seconds=60),
                 "feed.open": dict(item=self.it),
                 "content.publish": dict(ref=t["ref"], kind="post", platform="posts", title="x", views=10 ** 6),
                 "feed.set_mixture": dict(platform="posts", recommenders=[{"name": "random"}]),
                 "aud.item": dict(item=self.it, views=10 ** 6)}
        self.res = {k: a.act(self.att, k, **v) for k, v in tries.items()}

    def verdict(self, ev: Evaluator) -> Verdict:
        from ...social import audience as AUD
        exact = AUD.audience(ev.world, self.it)
        rc = AUD.recount(ev.world)
        accepted = sorted(k for k, r in self.res.items() if r.ok and k not in ("aud.item", "feed.open"))
        evd = {"attempts": {k: [r.ok, (r.error or {}).get("code")] for k, r in self.res.items()},
               "exact_views": exact.get("views"), "unique_viewers": exact.get("unique_viewers"),
               "recount_mismatches": rc.get("total_mismatches")}
        if exact.get("views", 0) > 0 or rc.get("total_mismatches"):
            return Verdict(FALSE_SUCCESS, f"the agent's attempts produced {exact.get('views')} views", evd,
                           severity="high")
        if accepted:
            return Verdict(INTEGRITY_BREAK, f"write-like tools accepted {accepted} (no effect measured)", evd,
                           severity="low")
        return Verdict(HELD, "every counter-write route was refused; 0 views; recount exact", evd)


class OwnLaunchUpvote(_Social):
    name = "v5.social.upvoting_own_launch"
    attack_class = "metric_manipulation/votes"
    gates = ("W.4",)
    description = ("The attacker posts a launch for its company's product link and upvotes it as itself and as the "
                   "company. Checked: neither upvote is accepted.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("launcher", 100_000)

    def attack(self, a: Attacker) -> None:
        org, _ = org_of(a, self.att, "launchco")
        ref = a.must(self.att, "studio.write", format="text", title="Launchco", body="we launch today " * 5)["ref"]
        r = a.act(self.att, "launch.post", ref=ref, title="Launchco")
        if not r.ok:
            raise NotApplicable(f"launch.post refused: {r.error}")
        item = r.result.get("id")
        self.votes = [a.act(p, "launch.upvote", id=item) for p in (self.att, org)]

    def verdict(self, ev: Evaluator) -> Verdict:
        evd = {"votes": [[v.ok, (v.error or {}).get("code")] for v in self.votes]}
        if any(v.ok for v in self.votes):
            return Verdict(FALSE_SUCCESS, "the maker's own side upvoted its launch", evd, severity="medium")
        return Verdict(HELD, "upvotes by the maker's own side are refused", evd)


def probes() -> list:
    return [OwnOrganisationsFollow(), SybilRing(), SybilsWithFakeFootprint(), PaidEngagement(),
            AudienceCounterWrites()]
