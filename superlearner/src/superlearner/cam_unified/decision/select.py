"""Selection among CAM's evaluated candidates, its exact log-probability, and the cam_eval authorization stamp.

    k ~ softmax(Q / tau) over the candidates that are not vetoed (Q_k >= veto_q)
    sel_logp = log softmax(Q / tau)_k                         (0 for a greedy / single-candidate deterministic choice)

The behaviour log-probability of an executed DOM action is the candidate-conditional surrogate

    log pi(a | I, C) = log p_gen(a_k | I) + sel_logp_k

over the recorded candidate set C (plan §2, §6: exact and recomputable, conditional on C). `selection_log_prob`
recomputes sel_logp differentiably from re-evaluated Q with the recorded veto mask, so PPO ratios at the behaviour
parameters are exactly 1.

Authorization (DOM-04: "the execution layer must reject actions that lack the required CAM authorization"). Every
action CAM selects carries `action.logp_parts["cam_eval"]` (STAMP_KEY): the candidate set, every Q, the selection
probabilities, gen/sel log-probabilities, the policy / reward-model versions and an HMAC-SHA256 digest over the
action's identity and executable content (tool, arguments, code) plus the selection record, keyed by a per-process
secret. `verify_stamp` recomputes the digest, so an unstamped action, a forged stamp or an action whose arguments were
changed after CAM evaluated them is refused (`require_stamp` raises `UnauthorizedAction`; `authorize(actions)` checks
a batch, None rows = no action). The canonical loop (W1, runtime.run_lifetime) calls `authorize` before
`env.execute`. The key is process-local (`set_stamp_key` installs a shared key for multi-process execution).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
from typing import Optional, Sequence

import torch
from torch import Tensor

STAMP_KEY = "cam_eval"
STAMP_VERSION = 1
_KEY = {"k": os.urandom(32)}


class UnauthorizedAction(RuntimeError):
    """An action without a valid cam_eval stamp reached the execution layer."""


def set_stamp_key(key: bytes) -> None:
    """Install the HMAC key (e.g. one shared by actor processes that execute each other's decisions)."""
    if not isinstance(key, (bytes, bytearray)) or len(key) < 16:
        raise ValueError("stamp key must be >= 16 bytes")
    _KEY["k"] = bytes(key)


# ============================================================================ selection

def _masked(Q: Tensor, vetoed: Optional[Tensor]) -> tuple:
    """(scores with vetoed entries at -inf, the effective veto mask, veto_all [B]). A row whose candidates are all
    vetoed selects among all of them (and is flagged veto_all; the caller records it)."""
    if vetoed is None:
        return Q, torch.zeros_like(Q, dtype=torch.bool), torch.zeros(Q.shape[0], dtype=torch.bool, device=Q.device)
    v = vetoed.to(torch.bool)
    allv = v.all(1)
    v = v & ~allv.unsqueeze(1)
    return Q.masked_fill(v, -math.inf), v, allv


def selection_log_probs(Q: Tensor, tau: float, vetoed: Optional[Tensor] = None) -> Tensor:
    """log softmax(Q / tau) over the non-vetoed candidates [B, K] (vetoed entries -inf). Differentiable."""
    S, _, _ = _masked(Q.float() / max(float(tau), 1e-6), vetoed)
    return torch.log_softmax(S, -1)


def selection_log_prob(Q: Tensor, chosen: Tensor, tau: float, vetoed: Optional[Tensor] = None,
                       greedy: Optional[Tensor] = None) -> tuple:
    """(sel_logp [B], selection entropy [B]) of the recorded choice `chosen` [B] long. Rows with greedy [B] True
    (deterministic choice) get logp 0 and entropy 0. Differentiable in Q."""
    lp = selection_log_probs(Q, tau, vetoed)
    p = lp.exp()
    ent = -(p * lp.masked_fill(~torch.isfinite(lp), 0.0)).sum(-1)
    sel = lp.gather(1, chosen.view(-1, 1).long()).squeeze(1)
    if greedy is not None:
        g = greedy.to(torch.bool).to(sel.device)
        sel = torch.where(g, torch.zeros_like(sel), sel)
        ent = torch.where(g, torch.zeros_like(ent), ent)
    return sel, ent


@torch.no_grad()
def select(Q: Tensor, gen_logp: Tensor, tau: float = 1.0, veto: Optional[float] = None, *, greedy: bool = False,
           generator: Optional[torch.Generator] = None) -> tuple:
    """Choose one candidate per row. Q, gen_logp [B, K] -> (idx [B] long, sel_logp [B], records [B] dict).

    k ~ softmax(Q / tau) over the non-vetoed candidates (Q_k >= veto when veto is not None); greedy: argmax, logp 0.
    Each record holds the selection facts the stamp needs: q, probs, gen_logp, chosen, tau, veto, vetoed, veto_all,
    greedy, sel_logp, sel_entropy."""
    Q = Q.detach().float()
    B, K = Q.shape
    vetoed = None if veto is None else (Q < float(veto))
    lp = selection_log_probs(Q, tau, vetoed)
    _, vmask, allv = _masked(Q, vetoed)
    probs = lp.exp()
    if greedy or K == 1:
        idx = lp.argmax(1)
    elif generator is not None and torch.device(generator.device).type != probs.device.type:
        idx = torch.multinomial(probs.cpu(), 1, generator=generator).squeeze(1).to(probs.device)
    else:
        idx = torch.multinomial(probs, 1, generator=generator).squeeze(1)
    det = bool(greedy or K == 1)
    sel, ent = selection_log_prob(Q, idx, tau, vetoed, torch.full((B,), det, dtype=torch.bool, device=Q.device))
    recs = []
    for b in range(B):
        recs.append({"q": [float(x) for x in Q[b]], "probs": [float(x) for x in probs[b]],
                     "gen_logp": [float(x) for x in gen_logp[b]], "chosen": int(idx[b]), "tau": float(tau),
                     "veto": None if veto is None else float(veto), "vetoed": [bool(x) for x in vmask[b]],
                     "veto_all": bool(allv[b]), "greedy": det, "sel_logp": float(sel[b]),
                     "sel_entropy": float(ent[b]), "n": K})
    return idx, sel, recs


# ============================================================================ the stamp

def _canon(x) -> str:
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _payload(action, stamp: dict) -> bytes:
    body = {"v": stamp.get("v"), "action_id": action.action_id, "tool": action.tool, "args": action.args,
            "code": action.code, "chosen": stamp.get("chosen"), "q": stamp.get("q"), "probs": stamp.get("probs"),
            "gen_logp": stamp.get("gen_logp"), "sel_logp": stamp.get("sel_logp"), "vetoed": stamp.get("vetoed"),
            "greedy": stamp.get("greedy"), "versions": stamp.get("versions"),
            "candidates": [_cand_identity(c) for c in stamp.get("candidates", [])]}
    return _canon(body).encode("utf-8", "surrogatepass")


def _cand_identity(c: dict) -> dict:
    return {"tool": c.get("tool"), "args": c.get("args"), "code": c.get("code"), "action_id": c.get("action_id")}


def stamp_digest(action, stamp: dict) -> str:
    return hmac.new(_KEY["k"], _payload(action, stamp), hashlib.sha256).hexdigest()


def candidate_record(a) -> dict:
    """A candidate's replay record (Action dict without its own stamp; the decode record stays attached)."""
    d = a.to_dict()
    lp = dict(d.get("logp_parts") or {})
    lp.pop(STAMP_KEY, None)
    d["logp_parts"] = lp
    return d


def attach_stamp(action, record: dict, candidates: Sequence, versions: dict, extra: Optional[dict] = None):
    """Write the cam_eval stamp into `action.logp_parts[STAMP_KEY]` (in place) and return the action. `record` is
    the row's selection record from `select`; `candidates` the K candidate Actions (or their dicts)."""
    reserved = {"v", "kind", "candidates", "versions", "digest"}
    clash = reserved & (set(record) | set(extra or {}))
    if clash or (extra and set(extra) & set(record)):
        raise ValueError(f"stamp fields clash with reserved / selection keys: "
                         f"{sorted(clash or (set(extra) & set(record)))}")
    st = {**record, **(extra or {}), "v": STAMP_VERSION, "kind": STAMP_KEY,
          "candidates": [c if isinstance(c, dict) else candidate_record(c) for c in candidates],
          "versions": dict(versions)}
    st["digest"] = stamp_digest(action, st)
    action.logp_parts[STAMP_KEY] = st
    return action


def stamp_of(action) -> Optional[dict]:
    lp = getattr(action, "logp_parts", None)
    if not isinstance(lp, dict):
        return None
    st = lp.get(STAMP_KEY)
    return st if isinstance(st, dict) else None


def verify_stamp(action) -> tuple:
    """(ok, reason): the action carries a cam_eval stamp whose digest matches its current content."""
    st = stamp_of(action)
    if st is None:
        return False, "no cam_eval stamp"
    if st.get("v") != STAMP_VERSION:
        return False, f"stamp version {st.get('v')} != {STAMP_VERSION}"
    dig = st.get("digest")
    if not isinstance(dig, str) or not hmac.compare_digest(dig, stamp_digest(action, st)):
        return False, "stamp digest mismatch (forged stamp, or the action changed after CAM evaluated it)"
    ch = st.get("chosen")
    cands = st.get("candidates") or []
    if not isinstance(ch, int) or not (0 <= ch < len(cands)):
        return False, "stamp has no valid chosen candidate"
    c = cands[ch]
    if c.get("tool") != action.tool or _canon(c.get("args")) != _canon(action.args):
        return False, "executed action differs from the evaluated candidate"
    return True, "ok"


def is_stamped(action) -> bool:
    return verify_stamp(action)[0]


def require_stamp(action) -> None:
    ok, why = verify_stamp(action)
    if not ok:
        raise UnauthorizedAction(f"refusing to execute {getattr(action, 'tool', action)!r}: {why}")


def authorize(actions: Sequence) -> list:
    """The execution gate: every non-None action must carry a valid stamp (raises UnauthorizedAction otherwise).
    Returns the actions unchanged."""
    for a in actions:
        if a is not None:
            require_stamp(a)
    return list(actions)
