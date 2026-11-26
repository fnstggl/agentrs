"""Probes against the Crucible 5.0 substrate (workstream A): the artifact store (quotas, permissions, existence side
channels, reserved kinds), column batch blobs (tampering) and generic assets (ownership and atomic sale)."""
from __future__ import annotations

import os

from ..framework import (HELD, INTEGRITY_BREAK, LEAK, MONEY_CREATED, Attacker, AttackRefused, Evaluator, NotApplicable,
                         Operator, Verdict)
from .base import V5Probe, b64, need_modules, org_of, text

MiB = 1 << 20


# ====================================================================================== artifacts: quotas
class ArtifactQuotaAcrossOrganisations(V5Probe):
    name = "v5.artifact.quota_multiplied_by_free_organisations"
    attack_class = "unbounded_resources/storage_quota"
    gates = ("W.7",)
    target, owner = "artifacts", "A"
    world = {"v5": "substrate", "configs": {"artifacts": {"quota_bytes": 1 * MiB, "max_bytes": 256 * 1024}}}
    description = ("The storage quota is per owner. An agent fills its own quota with distinct artifacts, then "
                   "founds organisations (`id.create_org`, free and unlimited) and stores more artifacts in each "
                   "organisation's name (`artifact.put owner=<org>`, allowed for organisations one administers). "
                   "Checked: how many bytes the one participant's group stores in total against the quota a single "
                   "participant is given (quota 1 MiB, 256 KiB per artifact for speed; the default is 512 MiB, so "
                   "the same steps exhaust the host's disk).")
    N_ORGS = 6

    def setup(self, op: Operator) -> None:
        self.att = op.join("hoarder", 10_000)

    def _fill(self, a: Attacker, owner: str, tag: str) -> int:
        stored = 0
        for i in range(8):
            data = (f"{tag}-{i}-".encode() * 20000)[: 200 * 1024]
            kw = {} if owner == self.att else {"owner": owner}
            r = a.act(self.att, "artifact.put", data_b64=b64(data), mime="application/octet-stream", **kw)
            if not r.ok:
                break
            stored += len(data)
        return stored

    def attack(self, a: Attacker) -> None:
        self.own = self._fill(a, self.att, "own")
        self.orgs = {}
        for k in range(self.N_ORGS):
            org, _ = org_of(a, self.att, f"store{k}")
            self.orgs[org] = self._fill(a, org, org)

    def verdict(self, ev: Evaluator) -> Verdict:
        from ...substrate import artifacts as art
        quota = ev.world.modules["artifacts"].cfg["quota_bytes"]
        group = ev.group()
        total = sum(art.usage(ev.world, p)["bytes"] for p in group)
        evd = {"quota_bytes": quota, "own_bytes": self.own, "org_bytes": self.orgs, "group_bytes": total,
               "organisations_founded": len(self.orgs), "multiple_of_quota": round(total / quota, 2)}
        if total > quota:
            return Verdict(INTEGRITY_BREAK, f"one participant stored {total / MiB:.2f} MiB, {total / quota:.1f}x its "
                           f"quota, by founding {len(self.orgs)} free organisations: the storage limit does not bound "
                           "a participant", evd, severity="medium",
                           fix="substrate/artifacts.py::Artifacts._charge: also bound the usage of the founding "
                               "person's whole group (the person plus every organisation it founded) by quota_bytes")
        return Verdict(HELD, f"the participant's group stores {total / MiB:.2f} MiB, within the single quota of "
                             f"{quota / MiB:.2f} MiB, although it founded {len(self.orgs)} organisations", evd)


# ====================================================================================== artifacts: existence oracles
class ArtifactListConfirmsHoldings(V5Probe):
    name = "v5.artifact.listing_confirms_a_guessed_private_file"
    attack_class = "information_leakage/existence_oracle"
    gates = ("W.9",)
    target, owner = "artifacts", "A"
    world = {"v5": "substrate"}
    description = ("A victim stores a short private document whose content is guessable from a small set (an offer "
                   "letter with a salary between 120,000 and 130,000). The attacker stores every candidate itself "
                   "(storing bytes someone else already stores makes the caller a 'holder' of them) and then lists "
                   "the victim's artifacts (`artifact.list owner=<victim>`) and reads each candidate's "
                   "`artifact.info`. Checked: whether anything the attacker received confirms which candidate the "
                   "victim holds (a confirmation-of-file attack: the content is the attacker's guess, the fact that "
                   "the victim holds it is private).")

    def setup(self, op: Operator) -> None:
        self.att = op.join("guesser", 10_000)
        self.victim = op.join("employee", 10_000, attacker=False)
        self.salary = 123_000
        data = f"Offer letter. Annual salary: {self.salary} USD.".encode()
        self.secret_ref = op.as_(self.victim, "artifact.put", data_b64=b64(data), mime="text/plain")["ref"]

    def attack(self, a: Attacker) -> None:
        self.cands = {}
        for s in range(120_000, 130_001, 1_000):
            r = a.must(self.att, "artifact.put", data_b64=b64(f"Offer letter. Annual salary: {s} USD.".encode()),
                       mime="text/plain")
            self.cands[r["ref"]] = s
        r = a.act(self.att, "artifact.list", owner=self.victim)
        self.listing = r.result if r.ok else r.error
        self.infos = {ref: (a.act(self.att, "artifact.info", ref=ref).result) for ref in self.cands}

    def verdict(self, ev: Evaluator) -> Verdict:
        listed = [x["ref"] for x in (self.listing or {}).get("artifacts", [])] if isinstance(self.listing, dict) else []
        confirmed = sorted({self.cands[r] for r in listed if r in self.cands})
        owners_seen = sorted({str((i or {}).get("owner")) for i in self.infos.values()})
        evd = {"victim_listing_as_seen_by_attacker": {"total": (self.listing or {}).get("total"), "refs": listed[:5]},
               "candidates_tried": len(self.cands), "salary_confirmed_by_listing": confirmed,
               "owners_shown_by_info": owners_seen}
        if confirmed:
            return Verdict(LEAK, f"`artifact.list owner=<victim>` lists the victim's private artifact to an attacker "
                           f"that merely stored a guess of its bytes: the attacker learns the victim's salary "
                           f"({confirmed[0]:,}) from {len(self.cands)} guesses", evd, severity="medium",
                           fix="substrate/artifacts.py::Artifacts.ep_list: for another owner's listing, include an "
                               "artifact only when it is published, published to the caller or granted to it - "
                               "never because the caller holds identical bytes")
        if self.secret_ref in self.infos and (self.infos[self.secret_ref] or {}).get("owner") == self.victim:
            return Verdict(LEAK, "artifact.info shows the victim as owner of the attacker's guess", evd, severity="medium")
        return Verdict(HELD, f"none of {len(self.cands)} guesses is confirmed: the victim's listing shows the attacker "
                             "nothing, and the attacker's info on each guess shows only its own record", evd)


class ArtifactUnreadableLooksMissing(V5Probe):
    name = "v5.artifact.unreadable_is_indistinguishable_from_missing"
    attack_class = "information_leakage/existence_oracle"
    gates = ("W.9",)
    target, owner = "artifacts, assets, media, embed", "A, D1"
    world = {"v5": "substrate", "media": True, "extra": ["embed"]}
    description = ("A victim stores a private image. The attacker sends the victim's REF and a REF that exists "
                   "nowhere to every tool that accepts an artifact reference (artifact.get/info/publish/grant/revoke, "
                   "artifact.put with it as a parent, asset.register backed by it, media.probe / media.edit_image / "
                   "media.transcode, embed.compare) and compares the two answers. Checked: whether any answer "
                   "differs (an existence oracle for other participants' private content).")

    def setup(self, op: Operator) -> None:
        need_modules(op.world, "artifacts", "assets")
        self.att = op.join("prober", 10_000)
        self.victim = op.join("artist", 10_000, attacker=False)
        from ...semantic.media import render_text_image
        png = render_text_image("private sketch", width=128, font_size=16)
        self.secret = op.as_(self.victim, "artifact.put", data_b64=b64(png), mime="image/png", kind="image")["ref"]
        self.missing = "sha256:" + "ab" * 32

    def attack(self, a: Attacker) -> None:
        calls = [("artifact.get", lambda r: {"ref": r}), ("artifact.info", lambda r: {"ref": r}),
                 ("artifact.publish", lambda r: {"ref": r}),
                 ("artifact.grant", lambda r: {"ref": r, "grantee": self.att}),
                 ("artifact.revoke", lambda r: {"ref": r, "grantee": self.att}),
                 ("artifact.put", lambda r: {"data_b64": b64(b"derived"), "mime": "text/plain", "parents": [r]}),
                 ("asset.register", lambda r: {"class": "licence", "title": "x", "ref": r})]
        if a.has_host("media"):
            calls += [("media.probe", lambda r: {"ref": r}),
                      ("media.edit_image", lambda r: {"ref": r, "ops": [{"op": "grayscale"}]}),
                      ("media.transcode", lambda r: {"ref": r, "format": "wav"})]
        if a.has_host("embed"):
            calls += [("embed.compare", lambda r: {"a": r, "b": r, "modality": "image"})]
        self.rows = {}
        for tool, mk in calls:
            out = []
            for ref in (self.secret, self.missing):
                res = a.act(self.att, tool, **mk(ref))
                err = res.error or {}
                out.append({"ok": res.ok, "code": err.get("code"),
                            "message": str(err.get("message", "")).replace(ref, "<REF>")[:200]})
            self.rows[tool] = out

    def verdict(self, ev: Evaluator) -> Verdict:
        diff = {t: r for t, r in self.rows.items() if r[0] != r[1]}
        evd = {"tools": len(self.rows), "answers": self.rows, "differing": sorted(diff)}
        if diff:
            return Verdict(LEAK, f"{len(diff)} tool(s) answer differently for an existing private artifact and a "
                           f"missing one: {sorted(diff)}", evd, severity="medium",
                           fix="make the listed tools check `artifacts.readable` first and raise the same NotFound "
                               "for unreadable and missing references")
        return Verdict(HELD, f"all {len(self.rows)} tools give identical answers for the victim's private artifact and "
                             "a reference that exists nowhere", evd)


class ArtifactReservedAndForged(V5Probe):
    name = "v5.artifact.reserved_kinds_owners_and_limits"
    attack_class = "world_integrity/input_validation"
    gates = ("W.6", "W.7")
    target, owner = "artifacts", "A"
    world = {"v5": "substrate", "configs": {"artifacts": {"max_bytes": 64 * 1024, "max_meta_bytes": 512}}}
    description = ("The attacker tries to create artifacts the store reserves or refuses: kind 'batch' (system "
                   "array blobs that feed population columns), owner 'sys', owner = another participant, owner = an "
                   "organisation it does not administer, metadata over the size limit, too many parents, bytes over "
                   "the per-artifact limit, malformed base64 and media types with control characters; it also "
                   "publishes and grants an artifact it does not own. Checked: every attempt is refused and no "
                   "artifact or usage record was created for anyone but the attacker.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("forger", 10_000)
        self.victim = op.join("bystander", 10_000, attacker=False)
        op.as_(self.victim, "id.create_org", handle="vicorg", name="Vic", kind="group")
        self.vref = op.as_(self.victim, "artifact.put", data_b64=b64(b"victim bytes"), mime="text/plain")["ref"]

    def attack(self, a: Attacker) -> None:
        good = b64(b"x" * 100)
        tries = {
            "kind_batch": dict(data_b64=good, mime="application/x-npz", kind="batch"),
            "owner_sys": dict(data_b64=good, mime="text/plain", owner="sys"),
            "owner_victim": dict(data_b64=good, mime="text/plain", owner=self.victim),
            "owner_victim_org": dict(data_b64=good, mime="text/plain", owner="o:vicorg"),
            "meta_too_large": dict(data_b64=good, mime="text/plain", meta={"x": "y" * 2000}),
            "too_many_parents": dict(data_b64=good, mime="text/plain", parents=[self.vref] * 70),
            "over_size_limit": dict(data_b64=b64(b"z" * (64 * 1024 + 1)), mime="text/plain"),
            "bad_base64": dict(data_b64="!!!notbase64", mime="text/plain"),
            "mime_injection": dict(data_b64=good, mime="text/plain\nX-Evil: 1"),
            "meta_non_finite": dict(data_b64=good, mime="text/plain", meta={"x": float("inf")}),
        }
        self.out = {}
        for k, args in tries.items():
            r = a.act(self.att, "artifact.put", **args)
            self.out[k] = "accepted" if r.ok else (r.error or {}).get("code")
        for k, (tool, args) in {"publish_others": ("artifact.publish", {"ref": self.vref}),
                                "grant_others": ("artifact.grant", {"ref": self.vref, "grantee": self.att})}.items():
            r = a.act(self.att, tool, **args)
            self.out[k] = "accepted" if r.ok else (r.error or {}).get("code")

    def verdict(self, ev: Evaluator) -> Verdict:
        st = ev.world.state
        recs = {ref: a["owner"] for ref, a in st.tables["artifacts"].items()}
        others = sorted({o for o in recs.values() if o not in (self.att, self.victim)})
        accepted = sorted(k for k, v in self.out.items() if v == "accepted")
        v = st.tables["artifacts"][self.vref]
        evd = {"attempts": self.out, "accepted": accepted, "artifact_owners_other_than_parties": others,
               "victim_artifact_published": v["published"],
               "attacker_grants_on_victim_artifact": st.has("artifact_grants", f"{self.vref}|{self.att}")}
        if accepted or others or v["published"] or evd["attacker_grants_on_victim_artifact"]:
            return Verdict(INTEGRITY_BREAK, f"accepted: {accepted}; owners created: {others}", evd, severity="high")
        return Verdict(HELD, f"all {len(self.out)} forged or over-limit requests were refused; no record was created "
                             "for the system, the victim or the victim's organisation", evd)


# ====================================================================================== columns: blob tampering
class BatchBlobTampering(V5Probe):
    name = "v5.cols.tampered_batch_blob_is_never_applied"
    attack_class = "world_integrity/tamper_detection"
    gates = ("W.6",)
    target, owner = "cols, artifacts", "A"
    world = {"v5": "substrate", "store": "sqlite", "extra": ["rt_cohort"]}
    description = ("Population-scale decisions are recorded as `cols.batch` events that carry the SHA-256 of an "
                   "npz blob kept in the world's store. A participant records cohort tallies through the real batch "
                   "path; then someone with write access to the store (no participant has it) replaces each stored "
                   "blob with a valid npz holding other values (larger tallies), and the world is reloaded from the "
                   "store and replayed. Checked: the tampered values are never applied - the reload either finds "
                   "the genuine bytes by hash or fails loudly - and an untampered reload is exact.")

    def setup(self, op: Operator) -> None:
        from . import harness  # noqa: F401 - registers rt_cohort
        need_modules(op.world, "rt_cohort", "cols")
        self.att = op.join("tallier", 1_000)

    def attack(self, a: Attacker) -> None:
        self.refs = [a.must(self.att, "cohort.tally", rows=[1, 2, 3, 5, 8], amount=7)["ref"],
                     a.must(self.att, "cohort.tally", rows=[2, 3, 13], amount=11)["ref"]]

    def verdict(self, ev: Evaluator) -> Verdict:
        import numpy as np
        from ...core.kernel import World
        from ...substrate import artifacts as art
        from ...substrate import columns
        w = ev.world
        w.flush()
        truth = columns.col(w, "rtc", "tally").copy()
        clean = World.load(w.store, w.world_id)
        clean_ok = bool(np.array_equal(columns.col(clean, "rtc", "tally"), truth)) and clean.digest() == w.digest()
        clean.close()
        # tamper: every batch blob replaced by a valid npz with inflated values (same rows, same dtypes)
        for ref in self.refs:
            arrs = art.get_array_blob(w, ref)
            fake = {k: (v * 1000 if k == "v.tally" else v) for k, v in arrs.items()}
            w.store.put_blob(art.blob_key(ref[7:]), art.npz_bytes(fake))
        outcome, applied = "loaded", None
        try:
            w2 = World.load(w.store, w.world_id)
            applied = columns.col(w2, "rtc", "tally").copy()
            outcome = "loaded exactly" if w2.digest() == w.digest() else "loaded with a different digest"
            w2.close()
        except Exception as e:  # noqa: BLE001 - a loud failure is the correct response to tampering
            outcome = f"refused: {type(e).__name__}: {str(e)[:160]}"
        tampered_used = applied is not None and not np.array_equal(applied, truth)
        evd = {"batches": len(self.refs), "clean_reload_exact": clean_ok, "tampered_reload": outcome,
               "true_tally_sum": int(truth.sum()), "tally_sum_after_tampered_reload": None if applied is None
               else int(applied.sum())}
        for ref in self.refs:                                   # restore, so the post-check replays the genuine log
            w.store.put_blob(art.blob_key(ref[7:]), art.get(w, ref))
        if tampered_used or not clean_ok:
            return Verdict(INTEGRITY_BREAK, "a tampered batch blob changed the reloaded population columns" if
                           tampered_used else "an untampered reload was not exact", evd, severity="critical")
        return Verdict(HELD, f"tampering with {len(self.refs)} stored batch blobs was {outcome.split(':')[0]}: the "
                             "tampered values were never applied (the applier verifies the SHA-256 carried by the "
                             "hash-chained event); the untampered reload was exact", evd)


# ====================================================================================== assets
class AssetTheftAndAtomicSale(V5Probe):
    name = "v5.asset.theft_double_sale_and_unpaid_transfer"
    attack_class = "payment_loopholes/ownership"
    gates = ("W.6",)
    target, owner = "assets", "A"
    world = {"v5": "substrate"}
    description = ("A victim owns a licence asset and offers it to a buyer. The attacker tries to transfer and "
                   "offer the victim's asset, to accept the offer made to the buyer, to accept its own offer for an "
                   "asset after the asset changed hands, to buy with an empty account (the payment must fail and "
                   "the asset stay), to accept one offer twice, and to register an asset of a class owned by another "
                   "module (equity, money, domain). Checked: ownership changes only by a gift of the owner or a "
                   "fully paid accepted offer, money is conserved, and no reserved class appears.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("fence", 50_000)
        self.poor = op.join("pauper", 0)
        self.victim = op.join("holder", 10_000, attacker=False)
        self.buyer = op.join("collector", 100_000, attacker=False)
        self.asset = op.as_(self.victim, "asset.register", **{"class": "licence", "title": "patent licence"})["asset"]
        self.offer = op.as_(self.victim, "asset.offer", asset=self.asset, to=self.buyer, price_cents=30_000)["offer"]

    def attack(self, a: Attacker) -> None:
        o = {}
        o["transfer_victims"] = a.act(self.att, "asset.transfer", asset=self.asset, to=self.att).ok
        o["offer_victims"] = a.act(self.att, "asset.offer", asset=self.asset, to=self.att, price_cents=1).ok
        o["accept_others_offer"] = a.act(self.att, "asset.accept", offer=self.offer).ok
        mine = a.must(self.att, "asset.register", **{"class": "licence", "title": "my licence"})["asset"]
        off = a.must(self.att, "asset.offer", asset=mine, to=self.poor, price_cents=40_000)["offer"]
        o["unpaid_accept"] = a.act(self.poor, "asset.accept", offer=off).ok
        a.must(self.att, "asset.withdraw", offer=off)
        off2 = a.must(self.att, "asset.offer", asset=mine, to=self.poor, price_cents=0)["offer"]
        o["first_accept"] = a.act(self.poor, "asset.accept", offer=off2).ok
        o["second_accept"] = a.act(self.poor, "asset.accept", offer=off2).ok
        o["stale_offer_after_gift"] = None
        mine2 = a.must(self.att, "asset.register", **{"class": "right", "title": "second"})["asset"]
        off3 = a.must(self.att, "asset.offer", asset=mine2, to=self.poor, price_cents=0)["offer"]
        a.must(self.att, "asset.transfer", asset=mine2, to=self.poor)
        o["stale_offer_after_gift"] = a.act(self.poor, "asset.accept", offer=off3).ok
        for cls in ("equity", "money", "domain", "shares"):
            o[f"reserved_{cls}"] = a.act(self.att, "asset.register", **{"class": cls, "title": "x"}).ok
        self.o, self.mine = o, mine

    def verdict(self, ev: Evaluator) -> Verdict:
        st = ev.world.state
        bad = sorted(k for k, v in self.o.items() if v and k not in ("first_accept",))
        evd = {"attempts": self.o, "victim_asset_owner": st.tables["assets"][self.asset]["owner"],
               "attacker_asset_owner": st.tables["assets"][self.mine]["owner"], "minted_to_group": ev.minted_to(),
               "attacker_group_cash_gain": ev.wealth_gain()}
        if ev.minted_to() > 0 or ev.wealth_gain() > 0:
            return Verdict(MONEY_CREATED, "the attacker group gained money from asset operations", evd, severity="high")
        if bad or evd["victim_asset_owner"] != self.victim:
            return Verdict(INTEGRITY_BREAK, f"refused steps went through: {bad}", evd, severity="high")
        return Verdict(HELD, "theft, unpaid acceptance, double acceptance, stale offers and reserved classes were all "
                             "refused; the victim still owns its asset and the gift reached the recipient once", evd)


def probes() -> list:
    return [ArtifactQuotaAcrossOrganisations(), ArtifactListConfirmsHoldings(), ArtifactUnreadableLooksMissing(),
            ArtifactReservedAndForged(), BatchBlobTampering(), AssetTheftAndAtomicSale()]
