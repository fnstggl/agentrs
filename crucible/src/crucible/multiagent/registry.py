"""Policy registry: versions, model hashes, identities, provenance, experience, actions, outcomes, updates and
population composition of self-play (MASTER_SPEC 12.2).

Everything a self-play claim rests on is recorded here, by the harness and the evaluator (never by a learner):

* ``versions``     every parameter version of every policy group: ``<group>@<n>`` with its weights hash (sha256
                   over the state dict) and its origin (``update``: a reinforcement update from self-play experience;
                   ``behaviour_cloning``: a pre-training step on demonstrations; or how it was made: initialised,
                   frozen copy, loaded), the parent version, algorithm, architecture, parameter count, the world and
                   seed it was trained in and its *training-data provenance* (which identities' decisions, at which
                   world times, went into the update that produced it, and the experience store holding them);
                   checkpoints (content-addressed by hash) for the initial, every `every`-th and the final version and
                   for every version a harness freezes;
* ``identities``   participant identity: principal, handle, controller (the operator that controls it, as admitted
                   with `admin.join`), policy group, role (current / historical / independent / heldout /
                   semantic), objective, world and admission time;
* ``updates``      every update (changed or skipped for lack of signal) with loss, signal sum and hashes;
* ``actions``      per identity: its decisions (time, tool, accepted or rejected, policy version);
* ``outcomes``     evaluator-side outcomes per match and identity (`crucible.multiagent.evaluate`);
* ``composition``  which kinds of participants were present and acting, per window;
* ``matches``      every match (world, profile, seed, roster with versions, event-log digest) - what makes a
                   cross-play between versions reproducible (`crucible.multiagent.league.replay_match`);
* ``experience``   the experience stores (`crucible.experience.store.ExperienceStore`) and their digests.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from .policy import PolicyGroup

FIELD_KINDS = ("versions", "model_hashes", "identities", "provenance", "experience", "actions", "outcomes",
               "updates", "composition")


class PolicyRegistry:
    def __init__(self, root: str | None = None, *, every: int = 10):
        self.root = root
        self.every = max(1, int(every))
        if root:
            os.makedirs(os.path.join(root, "ckpt"), exist_ok=True)
        self.versions: dict[str, dict] = {}
        self.identities: dict[str, dict] = {}
        self.updates: list[dict] = []
        self.actions: dict[str, list] = {}
        self.outcomes: list[dict] = []
        self.composition: list[dict] = []
        self.matches: list[dict] = []
        self.experience: list[dict] = []
        self._ckpt: dict[str, bytes] = {}        # in memory when no root

    # ------------------------------------------------------------------ versions
    @staticmethod
    def vid(group: PolicyGroup, version: int | None = None) -> str:
        return f"{group.gid}@{group.version if version is None else int(version)}"

    def register_group(self, group: PolicyGroup, *, world_id: str | None = None, seed: int | None = None,
                       origin: str = "initialised", checkpoint: bool = True) -> str:
        v = self.vid(group)
        if v not in self.versions:
            self.versions[v] = {"vid": v, "group": group.gid, "lineage": group.lineage, "version": group.version,
                                "hash": group.hash, "parent": None, "algo": group.algo, "arch": group.arch,
                                "n_params": group.n_params(), "world_id": world_id, "seed": seed, "origin": origin,
                                "frozen": group.frozen, "provenance": None, "checkpoint": None}
        if checkpoint:
            self.checkpoint(group)
        group.on_update = self.on_update
        group._registry_world = (world_id, seed)
        return v

    def on_update(self, group: PolicyGroup, info: dict) -> None:
        rec = {k: info.get(k) for k in ("version_before", "version_after", "params_hash_before", "params_hash_after",
                                        "changed", "loss", "signal_sum", "signal_abs_max", "transitions", "algo",
                                        "epochs", "scale", "identities")}
        rec["group"] = group.gid
        self.updates.append(rec)
        if not info.get("changed"):
            return
        world_id, seed = getattr(group, "_registry_world", (None, None))
        per: dict[str, list[int]] = {}
        for ident, t in info.get("decisions") or []:
            per.setdefault(ident, []).append(int(t))
        prov = {"decisions": int(info.get("transitions", 0)), "world_id": world_id, "seed": seed,
                "by_identity": {k: {"n": len(v), "t_first": min(v), "t_last": max(v)} for k, v in sorted(per.items())},
                "decisions_digest": hashlib.sha256(json.dumps(info.get("decisions"), sort_keys=True)
                                                   .encode()).hexdigest()[:24],
                "experience_stores": [e["root"] for e in self.experience if e.get("world_id") == world_id]}
        v = self.vid(group)
        self.versions[v] = {"vid": v, "group": group.gid, "lineage": group.lineage, "version": group.version,
                            "hash": group.hash, "parent": self.vid(group, info.get("version_before")),
                            "parent_hash": info.get("params_hash_before"), "algo": group.algo, "arch": group.arch,
                            "n_params": group.n_params(), "world_id": world_id, "seed": seed,
                            # a reinforcement update from self-play experience ("update") or a supervised
                            # pre-training step on (scripted) demonstrations ("behaviour_cloning")
                            "origin": "behaviour_cloning" if info.get("algo") == "behaviour_cloning" else "update",
                            "frozen": False, "provenance": prov, "checkpoint": None,
                            "signal_sum": info.get("signal_sum"), "loss": info.get("loss")}
        if group.version % self.every == 0:
            self.checkpoint(group)

    def checkpoint(self, group: PolicyGroup) -> str:
        data = group.state_bytes()
        key = group.hash
        if self.root:
            path = os.path.join(self.root, "ckpt", f"{key[:32]}.pt")
            if not os.path.exists(path):
                with open(path, "wb") as fh:
                    fh.write(data)
        else:
            self._ckpt[key] = data
            path = f"memory:{key[:32]}"
        v = self.vid(group)
        if v in self.versions:
            self.versions[v]["checkpoint"] = path
        return path

    def load(self, vid: str, gid: str, *, seed: int | None = None, frozen: bool = True) -> PolicyGroup:
        rec = self.versions[vid]
        path = rec.get("checkpoint")
        if not path:
            raise KeyError(f"{vid} has no checkpoint (only its hash is recorded)")
        if path.startswith("memory:"):
            data = self._ckpt[rec["hash"]]
        else:
            with open(path, "rb") as fh:
                data = fh.read()
        g = PolicyGroup.from_bytes(gid, data, frozen=frozen, seed=seed)
        if g.hash != rec["hash"]:
            raise ValueError(f"checkpoint of {vid} does not match its recorded hash")
        return g

    def freeze(self, group: PolicyGroup, gid: str, *, seed: int | None = None, origin: str = "frozen copy") -> PolicyGroup:
        """A frozen copy of `group`'s current version (a historical checkpoint), registered under `gid@<n>`."""
        self.checkpoint(group)
        g = group.frozen_copy(gid, seed=seed)
        g.version = group.version
        g.lineage = group.lineage
        v = self.vid(g)
        self.versions[v] = dict(self.versions[self.vid(group)], vid=v, group=gid, frozen=True, origin=origin,
                                copy_of=self.vid(group))
        g.on_update = self.on_update
        return g

    # ------------------------------------------------------------------ identities, actions, outcomes
    def add_identity(self, principal: str, **info) -> None:
        self.identities[principal] = dict(info, principal=principal)

    def record_actions(self, principal: str, history: list[dict]) -> None:
        self.actions.setdefault(principal, []).extend(
            {k: h.get(k) for k in ("t", "tool", "ok", "error", "version", "read_only")} for h in history)

    def record_outcomes(self, match_id: str, outcomes: dict) -> None:
        for p, o in outcomes.items():
            self.outcomes.append({"match": match_id, "principal": p, **o})

    def record_composition(self, snap: dict) -> None:
        self.composition.append(snap)

    def record_match(self, rec: dict) -> None:
        self.matches.append(rec)

    def record_experience(self, root: str, world_id: str, digest: str, records: int) -> None:
        self.experience.append({"root": root, "world_id": world_id, "digest": digest, "records": int(records)})

    # ------------------------------------------------------------------ summaries
    def field_kinds(self) -> dict[str, bool]:
        """Which of the 12.2 record kinds hold at least one entry."""
        return {"versions": len(self.versions) > 0,
                "model_hashes": any(v.get("hash") for v in self.versions.values()),
                "identities": len(self.identities) > 0,
                "provenance": any(v.get("provenance") for v in self.versions.values()),
                "experience": any(e["records"] > 0 for e in self.experience),
                "actions": any(self.actions.values()),
                "outcomes": len(self.outcomes) > 0,
                "updates": len(self.updates) > 0,
                "composition": len(self.composition) > 0}

    def lineage(self, vid: str) -> list[str]:
        out = []
        while vid and vid in self.versions:
            out.append(vid)
            vid = self.versions[vid].get("parent")
        return out

    def to_dict(self) -> dict:
        return {"versions": self.versions, "identities": self.identities, "updates": self.updates,
                "actions": {k: v[-2000:] for k, v in self.actions.items()}, "outcomes": self.outcomes,
                "composition": self.composition, "matches": self.matches, "experience": self.experience}

    def save(self, path: str | None = None) -> str:
        path = path or os.path.join(self.root or ".", "registry.json")
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, sort_keys=True, default=str)
        return path
