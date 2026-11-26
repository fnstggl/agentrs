"""Verified outcome labels through a separate, explicitly authorised channel (Clarification 2; MASTER_SPEC 11.3
"authorised verified outcome labels may be supplied separately for training and calibration"; requirement OO.12).

The learner's experience stream (observations, tool results, events, the scalar reward slot) never carries what the
evaluator concluded. An operator who wants a learner's *evaluation* or *meta-learning* machinery to see verified
outcomes opens a `LabelChannel` on the evaluator side:

    ch = LabelChannel(env, authorised_by="operator:alice", purpose="meta_learning")   # or a RemoteEvaluator
    ch.issue()        # -> [label records], one per agent, from the independent verifier (env.verify())
    ch.labels()       # every label issued so far (hash-chained, `crucible-label/1`)

Rules (checked by tests/v5/test_gate_OO.py::test_OO12_*):

* opening a channel needs a named authority and a declared purpose (`PURPOSES`); there is no default channel and
  nothing in `crucible.experience` opens one;
* a label is a separate record type (`SCHEMA`), never an `Observation` field, never an experience record
  (`crucible-exp/1`), and issuing labels changes nothing the learner observes or receives: the environment's reward
  stays 0.0 in observation-only mode and the participant's view is identical with and without the channel;
* labels are issued only when the operator calls `issue()` (no per-step stream): they cannot become handcrafted
  intermediate rewards by construction of the environment. What a learner does with them is the learner's business.
* each record carries the verifier's terminal-outcome keys (`success`, `failed`), the domain/world evaluator's
  verdict when the environment has one, the evaluator's outcome measures, the time and event sequence it refers to,
  and a SHA-256 chain over the channel's records so a consumer can check that none was altered or dropped.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

SCHEMA = "crucible-label/1"
PURPOSES = ("evaluation", "meta_learning", "calibration")


class LabelChannelError(PermissionError):
    """A label channel was opened without authority or for an undeclared purpose."""


def _canon(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), default=str)


class LabelChannel:
    def __init__(self, evaluator: Any, *, authorised_by: str, purpose: str, path: str | None = None):
        if not isinstance(authorised_by, str) or not authorised_by.strip():
            raise LabelChannelError("a label channel needs a named authority (authorised_by)")
        if purpose not in PURPOSES:
            raise LabelChannelError(f"purpose must be one of {PURPOSES}, got {purpose!r}")
        if not callable(getattr(evaluator, "verify", None)):
            raise TypeError("the evaluator side must offer verify() (an ExperienceEnv or a RemoteEvaluator)")
        self._ev = evaluator
        self.authorised_by, self.purpose = authorised_by.strip(), purpose
        self.path = path
        self._records: list[dict] = []
        self._head = "0" * 64

    @staticmethod
    def _outcome(v: dict, agent: str) -> dict:
        st = (v.get("agents") or {}).get(agent) or {}
        out = {"success": bool(st.get("success")), "failed": bool(st.get("failed"))}
        if "progress" in st and st.get("progress") is not None:
            out["progress"] = st["progress"]
        ev = (v.get("evaluation") or {}).get(agent)
        if ev is not None:
            out["evaluation"] = ev
        ms = (v.get("measures") or {}).get(agent)
        if ms:
            out["measures"] = ms
        if "outcome" in v and agent in (v.get("agents") or {agent: None}):
            out["outcome"] = v["outcome"]
        return out

    def issue(self) -> list[dict]:
        """Ask the independent verifier and issue one label per agent (evaluator side)."""
        v = self._ev.verify()
        agents = sorted((v.get("agents") or {}).keys()) or ["*"]
        new = []
        for a in agents:
            body = {"schema": SCHEMA, "agent": a, "t": int(v.get("t", 0) or 0), "seq": int(v.get("seq", 0) or 0),
                    "verifier": v.get("verifier"), "purpose": self.purpose, "authorised_by": self.authorised_by,
                    "outcome": self._outcome(v, a) if a != "*" else {"outcome": v.get("outcome")},
                    "index": len(self._records), "prev": self._head}
            h = hashlib.sha256(_canon(body).encode()).hexdigest()
            rec = dict(body, hash=h)
            self._head = h
            self._records.append(rec)
            new.append(rec)
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
            with open(self.path, "a") as fh:
                for r in new:
                    fh.write(_canon(r) + "\n")
        return [dict(r) for r in new]

    def labels(self) -> list[dict]:
        return [dict(r) for r in self._records]

    @staticmethod
    def verify_chain(records: list[dict]) -> bool:
        head = "0" * 64
        for i, r in enumerate(records):
            body = {k: v for k, v in r.items() if k != "hash"}
            if r.get("schema") != SCHEMA or body.get("index") != i or body.get("prev") != head:
                return False
            if hashlib.sha256(_canon(body).encode()).hexdigest() != r.get("hash"):
                return False
            head = r["hash"]
        return True

    @staticmethod
    def read(path: str) -> list[dict]:
        with open(path) as fh:
            return [json.loads(line) for line in fh if line.strip()]
