"""Private participant state probes (Gate W.9): another participant's direct messages, members-only items,
drafts, private communities and private artifacts, read by an unrelated admitted agent through every tool."""
from __future__ import annotations

import json

from ..framework import HELD, LEAK, Attacker, Evaluator, NotApplicable, Operator, Verdict
from .base import V5Probe, sweep_read_only

SECRETS = ("secret DM: the code is 8812-QQ", "members only: the treasure is under the oak",
           "unpublished draft text 55-ZZ", "private club agenda 7731-KK")


class SocialPrivateState(V5Probe):
    name = "v5.private.social_dms_members_only_drafts_private_communities"
    attack_class = "information_leakage/private_state"
    gates = ("W.9",)
    target, owner = "social", "C"
    world = {"v5": "social"}
    description = ("A victim sends a direct message, publishes a members-only post behind a paid tier, keeps a "
                   "draft, and posts in a private community. An unrelated agent tries the message, catalogue, "
                   "studio, artifact and community tools on every identifier it can find, guesses sequential "
                   "conversation ids, and reads its whole observation stream. Checked: none of the four secrets "
                   "reaches it.")

    def setup(self, op: Operator) -> None:
        self.victim = op.join("alice", 100_000, attacker=False)
        self.friend = op.join("bob", 100_000, attacker=False)
        self.att = op.join("eve", 100_000)
        X = lambda tool, **a: op.as_(self.victim, tool, **a)  # noqa: E731
        r = X("msg.send", to=self.friend, text=SECRETS[0])
        self.conv = r.get("conversation")
        X("creators.create_tier", name="gold", price_cents=500)
        ref = X("studio.write", format="text", title="paid", body=SECRETS[1])["ref"]
        pub = X("creators.publish", ref=ref, kind="article", title="Paid post", description="teaser")
        self.members_item, self.members_ref = pub.get("id"), ref
        self.draft = X("studio.write", format="text", title="draft", body=SECRETS[2])["ref"]
        try:
            com = X("community.found", subject="secret club", name="secretclub", visibility="private")
            self.community = com.get("community")
            if self.community:
                X("community.post", community=self.community, title="agenda", body=SECRETS[3])
        except Exception:  # noqa: BLE001 - a world without private communities still runs the other checks
            self.community = None

    def attack(self, a: Attacker) -> None:
        seeds = [x for x in (self.conv, self.members_item, self.members_ref, self.draft, self.community) if x]
        seeds += [f"cv_{i:06d}" for i in range(1, 6)] + [self.victim, "alice", "secretclub"]
        self.received = sweep_read_only(a, self.att, seeds=seeds, rounds=1, max_calls=3000)
        for tool, args in (("msg.history", {"conversation": self.conv}), ("msg.read", {"conversation": self.conv}),
                           ("content.open", {"item": self.members_item}), ("artifact.get", {"ref": self.members_ref}),
                           ("artifact.get", {"ref": self.draft}), ("studio.preview", {"ref": self.draft}),
                           ("community.feed", {"community": self.community}), ("community.join",
                                                                               {"community": self.community})):
            if any(v is None for v in args.values()):
                continue
            r = a.act(self.att, tool, **args)
            self.received.append((f"{tool} {args}", json.dumps(r.result if r.ok else r.error, default=str)))

    def verdict(self, ev: Evaluator) -> Verdict:
        seen = {s: [w for w, t in self.received if s in t][:3] for s in SECRETS}
        seen = {k: v for k, v in seen.items() if v}
        calls = next((int(t) for w, t in self.received if w == "calls"), 0)
        evd = {"calls": calls, "texts_examined": len(self.received), "secrets_seen": seen,
               "private_state_leaks": len(seen), "community_created": bool(self.community)}
        if seen:
            return Verdict(LEAK, f"{len(seen)} private secret(s) reached an unrelated agent", evd, severity="high")
        return Verdict(HELD, f"{calls} tool calls and the observation stream: none of the victim's secrets", evd)


def probes() -> list:
    return [SocialPrivateState()]
