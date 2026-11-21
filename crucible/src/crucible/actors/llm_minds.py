"""Optional Tier-3 social reasoning: simulated people whose replies to offers are drafted by a frozen LLM.

Enabled per world with ``params.llm_minds = {"backend": "hf:Qwen/Qwen3-4B" | "test:accept" | ..., "fraction": 0.2}``
(or per mind with ``mind["llm"] = True``). The LLM reads the person's *persistent* profile (role, a natural
description of its private minimum, temperament, trust in the counterparty, recent memories), the offer's terms
and the counterparty's free-text message, and proposes ``{"action": accept|reject|counter, "amount", "message"}``.

The LLM never decides outcomes:
* a proposed **accept** is executed only if the offer clears the person's private reservation (the Tier-2 utility,
  `negotiation.evaluate`); otherwise it becomes a counter at the person's aspiration;
* a proposed **counter** amount is clamped between the offered amount and 3x what the person would ask;
* contracts, payments and memberships only arise through the deal and contract services.
The LLM's text is recorded as an external outcome of the transaction, so replay and regeneration reuse it and the
world stays reproducible. The message it writes is sent as the reply text (communication, persuasion attempts).
"""
from __future__ import annotations

import json
import re

from ..core.errors import CrucibleError

_BACKENDS: dict = {}


def backend(world):
    cfg = world.spec.params.get("llm_minds") or {}
    name = cfg.get("backend", "")
    if name in _BACKENDS:
        return _BACKENDS[name]
    if name.startswith("hf:"):
        from ..agents.llm import HFBackend
        b = HFBackend(name[3:], max_context=4096)
    elif name.startswith("test:"):
        mode = name[5:]

        class _Test:
            """TEST DOUBLE ONLY: always proposes `mode` (accept/reject/counter) with a fixed message."""
            name = f"test-double:{mode}"

            def generate(self, messages, max_new_tokens=160, temperature=0.0, seed=None):
                from ..agents.llm import GenResult
                txt = json.dumps({"action": mode, "amount": 10 ** 9 if mode == "counter" else None,
                                  "message": f"[{mode}] thank you for the offer"})
                return GenResult(txt, 0, 0)
        b = _Test()
    else:
        return None
    _BACKENDS[name] = b
    return b


def wants_llm(world, m) -> bool:
    if m.get("llm") is not None:
        return bool(m["llm"])
    cfg = world.spec.params.get("llm_minds") or {}
    return bool(cfg) and m["kind"] == "person"


def prompt(m, d, terms, info, you) -> list[dict]:
    t = m["traits"]
    mem = [f"- {e['kind']} with {e['with']}" for e in m["memory"][-5:]]
    pay = [c for c in terms if c.get("type") == "payment"]
    lines = [f"{c['type']}: " + ", ".join(f"{k}={v}" for k, v in c.items() if k != "type") for c in terms]
    sys_ = ("You are a simulated person in a synthetic economy deciding how to answer a job or business offer. "
            "Answer ONLY with JSON: {\"action\": \"accept\"|\"reject\"|\"counter\", \"amount\": integer cents or "
            "null, \"message\": short reply}. You cannot change anything except by this answer.")
    user = (f"Your role: {m.get('role') or 'worker'}. You will not work for less than about "
            f"{int(m.get('reservation', 200000))} cents per month (private - never reveal it exactly). "
            f"Temperament: {'patient' if t['patience'] > 0.93 else 'impatient'}, "
            f"{'bold' if t['risk'] > 0.6 else 'cautious'}. Your trust in {you}: {info['trust']:.2f} (0-1).\n"
            f"Recent dealings:\n" + ("\n".join(mem) or "- none") + "\n"
            f"Offer from {you} (round {d['round']}):\n" + "\n".join(lines) + "\n"
            f"Their message: {d.get('message') or '(none)'}\n"
            f"{'The main payment is ' + str(pay[0]['amount']) + ' cents per period.' if pay else ''}")
    return [{"role": "system", "content": sys_}, {"role": "user", "content": user}]


def parse(text: str) -> dict:
    mt = re.search(r"\{.*\}", text or "", re.S)
    if not mt:
        return {}
    try:
        out = json.loads(mt.group(0))
        return out if isinstance(out, dict) else {}
    except ValueError:
        return {}


def llm_decide(soc, ctx, mid, m, me, d, rng):
    """(action, terms or None, message, meta) - an LLM proposal grounded by the person's own utility."""
    from .negotiation import _subst, decide, evaluate
    b = backend(ctx.world)
    if b is None:
        return None
    terms = _subst(d["terms"], d["from"], d["to"])
    u, info = evaluate(soc, ctx, m, me, d)
    msgs = prompt(m, d, terms, info, d["from"])
    cfg = ctx.world.spec.params.get("llm_minds") or {}

    def run():
        r = b.generate(msgs, max_new_tokens=int(cfg.get("max_new_tokens", 160)), temperature=0.0)
        return {"text": r.text[:2000], "gen_tokens": r.gen_tokens, "backend": getattr(b, "name", "?")}
    rec = ctx.external("llm", run)
    prop = parse(rec["text"])
    base_action, base_terms, base_msg = decide(soc, ctx, mid, m, me, d, rng)
    action = prop.get("action") if prop.get("action") in ("accept", "reject", "counter") else base_action
    message = str(prop.get("message") or base_msg)[:1000]
    grounded = action
    if action == "accept" and base_action != "accept":
        if u < 0:      # below the person's own reservation: the LLM cannot accept against the actor's interest
            grounded = base_action if base_action != "accept" else "reject"
    new_terms = None
    if grounded == "counter":
        new_terms = base_terms
        amt = prop.get("amount")
        if base_terms is None:
            pays = [i for i, c in enumerate(terms) if c.get("type") == "payment" and c["to"] == me]
            if pays:
                new_terms = list(terms)
                new_terms[pays[0]] = dict(terms[pays[0]], amount=int(terms[pays[0]]["amount"] * 1.15) + 1)
        if isinstance(amt, int) and new_terms:
            i = next((k for k, c in enumerate(new_terms) if c.get("type") == "payment" and c["to"] == me), None)
            if i is not None:
                offered = next((c["amount"] for c in terms if c.get("type") == "payment" and c["to"] == me), 1)
                hi = 3 * new_terms[i]["amount"]
                new_terms[i] = dict(new_terms[i], amount=int(min(hi, max(offered, amt))))
        if new_terms is None:
            grounded = "reject"
    meta = {"llm_action": prop.get("action"), "grounded_action": grounded, "tier2_action": base_action,
            "utility": round(u, 1), "backend": rec.get("backend"), "overridden": grounded != (prop.get("action") or grounded)}
    return grounded, new_terms, message, meta


def record(soc, ctx, mid, meta):
    ctx.emit("society.llm_decision", {"mind": mid, **meta}, vis="internal")


__all__ = ["llm_decide", "wants_llm", "backend", "record", "CrucibleError"]
