"""A society of eight semantic participants in one 4.0 open-economy world (Gate H), with evaluator-side verification.

`build_society(service, family="open_digital", seed=0)` creates the world (with the ``semantic_inference`` recording
module), admits eight participants with their own endowments, and gives each one its own goal. Goals are plain text
and say *what* the participant wants, never which tool achieves it. Some goals need another participant: a question
only another participant can answer, a payment owed to another participant, information that one participant must
find in the world and send to another.

`verify(world, society)` is the independent check, run by the evaluator on world state only (documents, bank
postings, delivered mail, repository contents executed against hidden tests in the sandbox, forum posts, cloud
instances). Nothing a participant says about its own success is used. The verifiers are never reachable from inside
the world.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from ..core.kernel import World
from .participants import BatchedRunner, SemanticGroup, SemanticParticipant

MEETING = "Thursday at 14:00"


@dataclass
class Member:
    handle: str
    goal: str
    check: Callable[[World, dict], dict]
    principal: str = ""
    account: str = ""
    policy: SemanticParticipant | None = None


@dataclass
class Society:
    world: World
    group: SemanticGroup
    members: list[Member] = field(default_factory=list)

    @property
    def principals(self) -> dict:
        return {m.handle: m.principal for m in self.members}

    def agents(self) -> list[tuple[str, SemanticParticipant]]:
        return [(m.principal, m.policy) for m in self.members]


# ---------------------------------------------------------------------------------------------- verifiers (evaluator)
def _mails(world, sender: str, recipient: str) -> list[dict]:
    out = []
    for mid, m in sorted(world.state.tables.get("mail_msgs", {}).items()):
        if m.get("from") == sender and recipient in (m.get("to") or []) and \
                (m.get("status") or {}).get(recipient) == "delivered":
            out.append({"id": mid, "subject": m.get("subject") or "", "body": m.get("body") or "",
                        "in_reply_to": m.get("in_reply_to")})
    return out


def _has(text: str, *words: str) -> bool:
    t = text.lower()
    return all(w.lower() in t for w in words)


def check_document(world, P) -> dict:
    docs = [d for d in world.state.tables.get("documents", {}).values()
            if d.get("owner") == P["ana"] and d.get("visibility") == "public"]
    good = [d for d in docs if "opening" in (d.get("title") or "").lower() and
            re.search(r"\b8\b", d.get("body") or "") and re.search(r"\b18\b", d.get("body") or "")]
    return {"verified": bool(good), "public_docs": len(docs), "matching": len(good)}


def check_payment(world, P) -> dict:
    src = set(world.state.find_ids("accounts", "owner", P["ben"]))
    dst = set(world.state.find_ids("accounts", "owner", P["cara"]))
    paid = 0
    for t in world.state.tables.get("txns", {}).values():
        ps = t.get("postings") or []
        out = -sum(a for acc, a in ps if acc in src and a < 0)
        got = sum(a for acc, a in ps if acc in dst and a > 0)
        if out > 0 and got > 0:
            paid += min(out, got)
    return {"verified": paid == 2500, "paid_cents": paid}


def check_question(world, P) -> dict:
    ms = [m for m in _mails(world, P["cara"], P["dev"]) if _has(m["subject"] + " " + m["body"], "meeting")]
    return {"verified": bool(ms), "mails": len(ms)}


def check_answer(world, P) -> dict:
    """Collaborative: dev answered cara's question with the fact only dev was told (verified from delivered mail)."""
    asked = [m for m in _mails(world, P["cara"], P["dev"]) if _has(m["subject"] + " " + m["body"], "meeting")]
    ans = [m for m in _mails(world, P["dev"], P["cara"])
           if re.search(r"thursday", m["body"] + m["subject"], re.I) and re.search(r"\b14\b|2 ?pm|14:00",
                                                                                   m["body"] + m["subject"], re.I)]
    return {"verified": bool(asked and ans), "asked": len(asked), "answers": len(ans), "collaborative": True}


def check_program(world, P) -> dict:
    """Productive: eli's repository holds word_count, executed in the sandbox against hidden tests."""
    from ..sandbox.runner import run_program
    code = world.modules.get("code")
    owner_prefix = P["eli"].split(":", 1)[1]
    repos = [rid for rid, r in world.state.tables.get("repos", {}).items()
             if r.get("owner") == P["eli"] and rid.split("/", 1)[-1] == "textutils" and r.get("head")]
    if not repos:
        return {"verified": False, "reason": "no textutils repository with a commit", "owner": owner_prefix}
    files = code.files(world.state, repos[0])
    if "textutils.py" not in files:
        return {"verified": False, "reason": "no textutils.py", "files": sorted(files)[:10]}
    cases = [("a b c", 3), ("", 0), ("  hello   world ", 2), ("one", 1), ("x y\nz\tw", 4)]
    harness = ("import textutils\n\ndef run(args, caller):\n    out = []\n    for s in args:\n        try:\n"
               "            out.append(textutils.word_count(s))\n        except Exception as e:\n"
               "            out.append('error: ' + type(e).__name__)\n    return out\n")
    res = run_program(dict(files, **{"_hidden_eval.py": harness}), "_hidden_eval:run", [c for c, _ in cases])
    got = res.get("result") if res.get("ok") else None
    passed = sum(1 for (c, want), g in zip(cases, got or []) if g == want)
    return {"verified": passed == len(cases), "tests_passed": passed, "tests": len(cases), "productive": True,
            "sandbox_ok": bool(res.get("ok"))}


def check_forum(world, P) -> dict:
    posts = [p for p in world.state.tables.get("forum_posts", {}).values()
             if p.get("author") == P["fay"] and p.get("status") == "visible"]
    good = [p for p in posts if re.search(r"cloud|provider|hosting|server", (p.get("title") or "") + " " +
                                          (p.get("body") or ""), re.I)]
    return {"verified": bool(good), "posts": len(posts), "matching": len(good)}


def check_providers_report(world, P) -> dict:
    names = [p.get("name", "") for p in world.state.tables.get("providers", {}).values()]
    ids = list(world.state.tables.get("providers", {}))
    ms = _mails(world, P["gus"], P["hana"])
    good = [m for m in ms if any(n and n.lower() in (m["body"] + m["subject"]).lower() for n in names) or
            any(i in m["body"] for i in ids)]
    return {"verified": bool(good), "mails": len(ms), "naming_real_providers": len(good)}


def check_instance(world, P) -> dict:
    inst = [i for i in world.state.tables.get("instances", {}).values()
            if i.get("owner") == P["hana"] and i.get("status", "running") not in ("released",)]
    return {"verified": bool(inst), "instances": len(inst)}


GOALS = [
    ("ana", "Publish a public document titled 'Opening hours' saying that the bakery is open Monday to Friday from 8 "
            "to 18.", check_document),
    ("ben", "You owe cara (p:cara) 2500 cents for the logo she designed for you. Pay her that amount from your bank "
            "account.", check_payment),
    ("cara", "You do not know when the team meeting is. Ask dev (p:dev) by mail when the team meeting is.",
     check_question),
    ("dev", f"You organise the team meeting; it is on {MEETING}. Check your mail and answer the questions people "
            "send you.", check_answer),
    ("eli", "Create a code repository named textutils and commit a file textutils.py containing a Python function "
            "word_count(text) that returns the number of words in text.", check_program),
    ("fay", "Ask the community, in a forum post, which cloud provider is cheapest for hosting a small website.",
     check_forum),
    ("gus", "Find out which cloud providers exist in this world and send hana (p:hana) a mail that lists their "
            "names.", check_providers_report),
    ("hana", "Rent a cloud instance with 1 core from a cloud provider, paid from your bank account.", check_instance),
]


def build_society(service, family: str = "open_digital", seed: int = 0, endowment: int = 500_000,
                  record: bool = True, **participant_kw) -> Society:
    """World + eight participants sharing `service` (see the module docstring)."""
    import crucible
    from ..worldgen import generator as G
    crucible.load_all_modules()
    spec = G.generate({"family": family, "extra_modules": ["semantic_inference"],
                       "overrides": {"params": {"cost_model": "deterministic"}}}, seed)
    world = World.create(spec)
    group = SemanticGroup(service, world, record=record)
    soc = Society(world, group)
    for handle, goal, check in GOALS:
        r = world.system_call("admin.join", {"handle": handle, "endowment": endowment, "kind": "agent",
                                             "controller": f"semantic:{service.model_id}"})
        m = Member(handle, goal, check, r["principal"], r["account"])
        m.policy = group.participant(goal, **participant_kw)
        soc.members.append(m)
    return soc


def run_society(soc: Society, rounds: int = 10, dt: int = 600, seed: int = 0) -> BatchedRunner:
    runner = BatchedRunner(soc.world, soc.agents(), dt=dt, seed=seed)
    runner.run(rounds)
    return runner


def verify(world: World, soc: Society) -> dict:
    """Evaluator: per goal {verified, details} from world state only."""
    P = soc.principals
    out = {}
    for m in soc.members:
        try:
            out[m.handle] = m.check(world, P)
        except Exception as e:  # noqa: BLE001 - a verifier failure is a failed verification, reported as such
            out[m.handle] = {"verified": False, "error": f"{type(e).__name__}: {e}"}
    return out
