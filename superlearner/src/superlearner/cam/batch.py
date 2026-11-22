"""Training batches: whole episodes, padded, with the controller's likelihood recomputed exactly.  [candidate-editable]

`collate` turns a list of episodes (each a list of the decision records written by `CAMAgent.step`) into tensors
shaped [B, T, ...]. `evaluate` runs the controller over them the way it ran when acting - the context encoder per
decision, the recurrent state unrolled from the start of each episode (full-episode backpropagation through time,
no stale hidden states) and teacher-forced argument choices - and returns, per decision, the log-probability of
the action that was taken, the entropy, the value ensemble and the success logit.

For an unchanged controller `evaluate(...)["logp"]` equals the log-probability recorded at acting time; a unit
test checks this, because every policy-gradient quantity depends on it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..grounding.executor import FAILURE_CLASSES
from .features import N_NUM, OBS_DIM, T_TOK
from .model import CAMController, EmbStore, masked_entropy

FC = ("",) + tuple(FAILURE_CLASSES)        # outcome classes of a decision: accepted, or one failure class

AUX_DIM = 16


@dataclass
class Batch:
    t: dict            # name -> tensor
    B: int
    T: int

    def __getitem__(self, k):
        return self.t[k]

    def select(self, idx: torch.Tensor) -> "Batch":
        return Batch({k: v[idx] for k, v in self.t.items()}, int(idx.numel()), self.T)


def collate(episodes: list[list[dict]], store: EmbStore, device="cpu", success: list[bool] | None = None) -> Batch:
    B = len(episodes)
    T = max(len(e) for e in episodes)
    Nc = max(len(d["ctx"]["ek"]) for e in episodes for d in e)
    Nh = max(len(d["head"]["ek"]) for e in episodes for d in e)
    S = max([len(d["slots"]) for e in episodes for d in e] + [1])
    No = max([len(s["opt"]["ek"]) for e in episodes for d in e for s in d["slots"]] + [1])

    def z(shape, dt):
        return np.zeros(shape, dt)
    a = {"ctx_ek": z((B, T, Nc), np.int64), "ctx_tok": z((B, T, Nc, T_TOK), np.int32), "ctx_num": z((B, T, Nc, N_NUM), np.float32),
         "ctx_typ": z((B, T, Nc), np.int32), "ctx_mask": z((B, T, Nc), bool),
         "obsf": z((B, T, OBS_DIM), np.float32), "aux": z((B, T, AUX_DIM), np.float32), "tick": z((B, T), np.float32),
         "head_ek": z((B, T, Nh), np.int64), "head_tok": z((B, T, Nh, T_TOK), np.int32), "head_num": z((B, T, Nh, N_NUM), np.float32),
         "head_typ": z((B, T, Nh), np.int32), "head_mask": z((B, T, Nh), bool), "h": z((B, T), np.int64),
         "par_ek": z((B, T, S), np.int64), "par_tok": z((B, T, S, T_TOK), np.int32), "par_num": z((B, T, S, N_NUM), np.float32),
         "par_typ": z((B, T, S), np.int32),
         "opt_ek": z((B, T, S, No), np.int64), "opt_tok": z((B, T, S, No, T_TOK), np.int32),
         "opt_num": z((B, T, S, No, N_NUM), np.float32), "opt_typ": z((B, T, S, No), np.int32),
         "opt_mask": z((B, T, S, No), bool), "c": np.full((B, T, S), -1, np.int64),
         "valid": z((B, T), bool), "logp_old": z((B, T), np.float32), "value_old": z((B, T), np.float32),
         "reward": z((B, T), np.float32), "done": z((B, T), np.float32), "trunc": z((B, T), np.float32),
         "ok": z((B, T), np.float32), "fclass": z((B, T), np.int64)}
    for b, ep in enumerate(episodes):
        for t, d in enumerate(ep):
            n = len(d["ctx"]["ek"])
            a["ctx_ek"][b, t, :n] = d["ctx"]["ek"]
            a["ctx_tok"][b, t, :n] = d["ctx"]["tok"]
            a["ctx_num"][b, t, :n] = d["ctx"]["num"]
            a["ctx_typ"][b, t, :n] = d["ctx"]["typ"]
            a["ctx_mask"][b, t, :n] = True
            a["obsf"][b, t] = d["obsf"]
            a["aux"][b, t] = d["aux"]
            a["tick"][b, t] = d["tick"]
            n = len(d["head"]["ek"])
            a["head_ek"][b, t, :n] = d["head"]["ek"]
            a["head_tok"][b, t, :n] = d["head"]["tok"]
            a["head_num"][b, t, :n] = d["head"]["num"]
            a["head_typ"][b, t, :n] = d["head"]["typ"]
            a["head_mask"][b, t, :n] = True
            a["h"][b, t] = d["h"]
            for j, s in enumerate(d["slots"]):
                a["par_ek"][b, t, j] = s["par"]["ek"][0]
                a["par_tok"][b, t, j] = s["par"]["tok"][0]
                a["par_num"][b, t, j] = s["par"]["num"][0]
                a["par_typ"][b, t, j] = s["par"]["typ"][0]
                n = len(s["opt"]["ek"])
                a["opt_ek"][b, t, j, :n] = s["opt"]["ek"]
                a["opt_tok"][b, t, j, :n] = s["opt"]["tok"]
                a["opt_num"][b, t, j, :n] = s["opt"]["num"]
                a["opt_typ"][b, t, j, :n] = s["opt"]["typ"]
                a["opt_mask"][b, t, j, :n] = True
                a["c"][b, t, j] = s["c"]
            a["valid"][b, t] = True
            a["logp_old"][b, t] = d.get("logp", 0.0)
            a["value_old"][b, t] = d.get("value", 0.0)
            a["reward"][b, t] = d.get("reward", 0.0)
            a["done"][b, t] = float(d.get("done", False))
            a["trunc"][b, t] = float(d.get("trunc", False))
            a["ok"][b, t] = float(d.get("ok", False))
            fc = d.get("fclass", "") or ""
            a["fclass"][b, t] = FC.index(fc) if fc in FC else len(FC) - 1
    out = {}
    for k, v in a.items():
        if k.endswith("_ek"):
            out[k.replace("_ek", "_row")] = torch.from_numpy(store.rows(v)).to(device)
        else:
            out[k] = torch.from_numpy(v).to(device)
    out["success"] = torch.tensor([float(x) for x in (success or [False] * B)], device=device)
    return Batch(out, B, T)


def evaluate(model: CAMController, batch: Batch, store: EmbStore | None = None) -> dict:
    t = batch.t
    B, T = t["valid"].shape
    N = B * T
    live = getattr(model, "live", None)
    if live is not None:
        # the learner has its own language backbone: the embeddings of the texts this batch uses are recomputed
        # from the backbone's current weights (and, when it is trainable, receive gradients; cam/backbone.py)
        live.attach(model, t)
    elif store is not None:
        model.set_table(store.tensor(t["valid"].device))

    def flat(k):
        v = t[k]
        return v.reshape(N, *v.shape[2:])
    ce = model.embed(flat("ctx_row"), flat("ctx_tok"), flat("ctx_num"), flat("ctx_typ"))
    c = model.encode(ce, flat("ctx_mask"), flat("obsf"), flat("aux")).view(B, T, -1)
    state = model.init_state(B, c.device)
    zs = []
    for i in range(T):
        zi, state = model.recur(c[:, i], state, t["tick"][:, i])
        zs.append(zi)
    z = torch.stack(zs, 1).reshape(N, -1)
    he = model.embed(flat("head_row"), flat("head_tok"), flat("head_num"), flat("head_typ"))
    hmask = flat("head_mask")
    hl = model.head_logits(z, he, hmask)
    hlp = torch.log_softmax(hl, -1)
    h = flat("h")
    logp_h = hlp.gather(1, h.unsqueeze(1)).squeeze(1)
    ent_h = masked_entropy(hl, hmask)
    e_head = he.gather(1, h.view(N, 1, 1).expand(N, 1, he.shape[-1])).squeeze(1)
    pe = model.embed(flat("par_row"), flat("par_tok"), flat("par_num"), flat("par_typ"))
    oe = model.embed(flat("opt_row"), flat("opt_tok"), flat("opt_num"), flat("opt_typ"))
    omask = flat("opt_mask")
    ch = flat("c")
    al = model.arg_logits_all(z, e_head, pe, oe, omask, ch)
    alp = torch.log_softmax(al, -1)
    has = (ch >= 0).to(alp.dtype)
    logp_a = (alp.gather(2, ch.clamp(min=0).unsqueeze(-1)).squeeze(-1) * has).sum(1)
    ent_a = (masked_entropy(al, omask) * has).sum(1) / has.sum(1).clamp(min=1)
    # With `model.detach_critic` the value ensemble and the success predictor read the controller's state without
    # sending gradients back into it: fitting the critic can then never move the policy.
    zc = z.detach() if getattr(model, "detach_critic", False) else z
    d = oe.shape[-1]
    e_args = (oe.gather(2, ch.clamp(min=0).view(N, -1, 1, 1).expand(N, ch.shape[1], 1, d)).squeeze(2) * has.unsqueeze(-1)).sum(1) \
        / has.sum(1, keepdim=True).clamp(min=1)
    return {"c": c, "e_head": e_head.view(B, T, -1), "e_args": e_args.view(B, T, -1),"logp": (logp_h + logp_a).view(B, T), "logp_head": logp_h.view(B, T), "logp_args": logp_a.view(B, T),
            "ent": (ent_h + ent_a).view(B, T), "ent_head": ent_h.view(B, T), "head_logp_all": hlp.view(B, T, -1),
            "values": model.values(zc).view(B, T, -1), "succ_logit": model.p_success(zc).view(B, T), "z": z.view(B, T, -1)}
