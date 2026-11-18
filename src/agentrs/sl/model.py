"""The superlearner core: one continually trained network shared by every world.

    observation (hashed symbols, named numbers, entity sets, egocentric image cells)
      -> perception: set transformer over observation tokens -> observation embedding      [shared]
    goal (same encoder)                                                                    [shared]
      -> belief: GRU over (observation, goal, previous action, previous reward, episode/time flags, last memory read)
      -> fast episodic memory: per-trial key/value store written every step (learned write gate) and read by
         learned attention; persists across the episodes of a trial, cleared between trials; no gradient step
         is needed to use a new observation
      -> skills: K learned skill embeddings; a high-level policy picks a skill when a learned termination fires
         (always at episode start); the low-level policy is conditioned on the active skill
      -> action generation: pointer head scoring candidate action labels (any world's labels, hashed), slot by
         slot for compositional actions
      -> value: time-aware (time features, semi-Markov discount in the learner)
      -> world model: ensemble predicting the next observation embedding, reward, termination and elapsed time
         from (belief, memory read, action); its disagreement is the model-uncertainty signal
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from agentrs.sl.featurize import CELL_F, ENT_F, LABEL_IDS, NB


@dataclass
class SLConfig:
    d: int = 192            # token / embedding width
    dh: int = 384           # belief (GRU) width
    layers: int = 2         # perception transformer layers
    heads: int = 4
    dk: int = 64            # memory key width
    dv: int = 128           # memory value width
    mem_slots: int = 256    # per-trial ring buffer
    skills: int = 8         # learned skill embeddings (1 = no hierarchy)
    ds: int = 32
    da: int = 128           # action-label embedding width
    ens: int = 3            # world-model ensemble
    grid: int = 16          # max image grid side
    memory: bool = True     # external episodic memory available (architecture flag; per-run adapt modes below)


def clear(st: dict, mask: torch.Tensor, keys, fill: float = 0.0) -> None:
    for k in keys:
        t = st[k]
        st[k] = torch.where(mask.view(-1, *([1] * (t.dim() - 1))), torch.full_like(t, fill), t)


def symlog(x):
    return torch.sign(x) * torch.log1p(x.abs())


class Perception(nn.Module):
    def __init__(self, c: SLConfig):
        super().__init__()
        d = c.d
        self.emb = nn.Embedding(NB, d, padding_idx=0)       # symbols, field names, entity types, families
        self.num_val = nn.Sequential(nn.Linear(10, d), nn.GELU(), nn.Linear(d, d))
        self.ent = nn.Linear(ENT_F, d)
        self.cell = nn.Sequential(nn.Linear(CELL_F, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.zeros(c.grid * c.grid, d))
        self.kind = nn.Parameter(torch.zeros(5, d))        # family / token / number / entity / image-cell
        layer = nn.TransformerEncoderLayer(d, c.heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, c.layers, enable_nested_tensor=False)
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, c.heads, batch_first=True)
        self.out = nn.LayerNorm(d)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.kind, std=0.02)
        self.grid = c.grid

    @staticmethod
    def _numfeat(v):
        s = symlog(v)
        fr = torch.tensor([0.5, 1.0, 2.0, 4.0], device=v.device)
        return torch.cat([s.unsqueeze(-1), torch.sign(v).unsqueeze(-1), torch.sin(s.unsqueeze(-1) * fr),
                          torch.cos(s.unsqueeze(-1) * fr)], -1)

    def forward(self, b: dict) -> torch.Tensor:
        parts, masks = [], []
        N = b["fam"].shape[0]
        parts.append((self.emb(b["fam"]) + self.kind[0]).unsqueeze(1))
        masks.append(torch.ones(N, 1, dtype=torch.bool, device=b["fam"].device))
        parts.append(self.emb(b["tok"]).sum(-2) + self.kind[1])
        masks.append(b["tm"])
        parts.append(self.emb(b["numk"]).sum(-2) + self.num_val(self._numfeat(b["numv"])) + self.kind[2])
        masks.append(b["nm"])
        parts.append(self.emb(b["entk"]).sum(-2) + self.ent(symlog(b["entf"])) + self.kind[3])
        masks.append(b["em"])
        if "img" in b:
            g = b["img"].float() / 255.0                      # (N, G, G, CELL_F)
            G = g.shape[1]
            cells = self.cell(g.reshape(N, G * G, CELL_F))
            pos = self.pos.view(self.grid, self.grid, -1)[:G, :G].reshape(G * G, -1)
            parts.append(cells + pos + self.kind[4])
            masks.append(b["imgm"].unsqueeze(1).expand(N, G * G))
        x = torch.cat(parts, 1)
        m = torch.cat(masks, 1)
        x = self.tf(x, src_key_padding_mask=~m)
        y, _ = self.pool(self.q.expand(N, 1, -1), x, x, key_padding_mask=~m)
        return self.out(y.squeeze(1))


class SLCore(nn.Module):
    def __init__(self, c: SLConfig):
        super().__init__()
        self.c = c
        d, dh = c.d, c.dh
        self.perc = Perception(c)
        self.goal_tag = nn.Parameter(torch.zeros(d))
        self.act_emb = nn.Embedding(NB, c.da, padding_idx=0)
        nn.init.normal_(self.act_emb.weight, std=0.02)
        self.act_mlp = nn.Sequential(nn.Linear(c.da, c.da), nn.GELU(), nn.Linear(c.da, c.da))
        self.gru = nn.GRUCell(d + d + c.da + 5 + c.dv, dh)
        # fast episodic memory
        self.mk = nn.Linear(dh, c.dk)
        self.mv = nn.Linear(dh + c.da + 3, c.dv)
        self.mw = nn.Linear(dh, 1)
        self.mq = nn.Linear(dh, c.dk)
        # skills
        self.skill = nn.Parameter(torch.randn(c.skills, c.ds) * 0.5)
        ctx = dh + c.dv + d
        self.hi = nn.Sequential(nn.Linear(ctx, dh), nn.GELU(), nn.Linear(dh, c.skills))
        self.term = nn.Sequential(nn.Linear(ctx + c.ds, dh // 2), nn.GELU(), nn.Linear(dh // 2, 1))
        # low-level policy query + autoregressive slots
        self.query = nn.Sequential(nn.Linear(ctx + c.ds, dh), nn.GELU(), nn.Linear(dh, c.da))
        self.slot = nn.GRUCell(c.da, c.da)
        self.value = nn.Sequential(nn.Linear(ctx + c.ds, dh), nn.GELU(), nn.Linear(dh, 1))
        # world model (ensemble)
        wm_in = dh + c.dv + c.da
        self.wm_next = nn.ModuleList([nn.Sequential(nn.Linear(wm_in, dh), nn.GELU(), nn.Linear(dh, d)) for _ in range(c.ens)])
        self.wm_aux = nn.Sequential(nn.Linear(wm_in, dh), nn.GELU(), nn.Linear(dh, 3))  # reward, done, dt

    # ------------------------------------------------------------------ pieces
    def encode(self, b: dict) -> torch.Tensor:
        return self.perc(b)

    def encode_goal(self, b: dict) -> torch.Tensor:
        return self.perc(b) + self.goal_tag

    def labels(self, ids: torch.Tensor) -> torch.Tensor:
        """(..., LABEL_IDS) hashed label ids -> (..., da) action-label embeddings."""
        return self.act_mlp(self.act_emb(ids).sum(-2))

    def init_state(self, n: int, device) -> dict:
        c = self.c
        return {"h": torch.zeros(n, c.dh, device=device), "read": torch.zeros(n, c.dv, device=device),
                "mk": torch.zeros(n, c.mem_slots, c.dk, device=device), "mv": torch.zeros(n, c.mem_slots, c.dv, device=device),
                "mg": torch.full((n, c.mem_slots), -1e4, device=device),  # log write-gate; -1e4 = empty slot
                "ptr": torch.zeros(n, dtype=torch.long, device=device),
                "z": torch.zeros(n, dtype=torch.long, device=device),
                "prev_a": torch.zeros(n, c.da, device=device), "prev_r": torch.zeros(n, 3, device=device)}

    def reset_rows(self, st: dict, rows: torch.Tensor, keep_memory: bool) -> None:
        """Clear the belief (and, unless keep_memory, the episodic memory) of the given rows. Out-of-place (the
        learner replays through these resets with autograd)."""
        if rows.numel() == 0:
            return
        n = st["h"].shape[0]
        mask = torch.zeros(n, dtype=torch.bool, device=st["h"].device)
        mask[rows] = True
        clear(st, mask, ("h", "read", "prev_a", "prev_r"))
        if not keep_memory:
            clear(st, mask, ("mk", "mv"))
            clear(st, mask, ("mg",), fill=-1e4)
            clear(st, mask, ("ptr",))

    def step_belief(self, st: dict, obs_e, goal_e, flags, use_memory: torch.Tensor) -> dict:
        """One belief update. flags: (N, 2) [new_episode, symlog(elapsed steps)]. use_memory: (N,) bool."""
        c = self.c
        x = torch.cat([obs_e, goal_e, st["prev_a"], st["prev_r"], flags, st["read"]], -1)
        h = self.gru(x, st["h"])
        q = self.mq(h)
        logits = torch.einsum("nk,nmk->nm", q, st["mk"]) / math.sqrt(c.dk) + st["mg"]
        empty = (st["mg"] <= -1e3).all(-1)
        a = torch.softmax(logits, -1)
        read = torch.einsum("nm,nmv->nv", a, st["mv"])
        read = torch.where((empty | ~use_memory).unsqueeze(-1), torch.zeros_like(read), read)
        st = dict(st)
        st["h"], st["read"] = h, read
        return st

    def write_memory(self, st: dict, a_emb, r_feat, use_memory: torch.Tensor) -> dict:
        """Write (belief, action, consequence) of the step just taken; ring buffer per row."""
        c = self.c
        k = self.mk(st["h"])
        v = self.mv(torch.cat([st["h"], a_emb, r_feat], -1))
        g = F.logsigmoid(self.mw(st["h"])).squeeze(-1)
        n = k.shape[0]
        idx = st["ptr"] % c.mem_slots
        rows = torch.arange(n, device=k.device)
        mk, mv, mg = st["mk"].clone(), st["mv"].clone(), st["mg"].clone()
        um = use_memory
        mk[rows[um], idx[um]] = k[um]
        mv[rows[um], idx[um]] = v[um]
        mg[rows[um], idx[um]] = g[um]
        st = dict(st)
        st["mk"], st["mv"], st["mg"] = mk, mv, mg
        st["ptr"] = st["ptr"] + um.long()
        st["prev_a"], st["prev_r"] = a_emb, r_feat
        return st

    def context(self, st: dict, goal_e) -> torch.Tensor:
        return torch.cat([st["h"], st["read"], goal_e], -1)

    def skill_dists(self, ctx, z):
        hi = self.hi(ctx)
        term_logit = self.term(torch.cat([ctx, self.skill[z]], -1)).squeeze(-1)
        return hi, term_logit

    def policy_query(self, ctx, z):
        return self.query(torch.cat([ctx, self.skill[z]], -1))

    def value_of(self, ctx, z):
        return self.value(torch.cat([ctx, self.skill[z]], -1)).squeeze(-1)

    def wm(self, h, read, a_emb):
        x = torch.cat([h, read, a_emb], -1)
        nxt = torch.stack([m(x) for m in self.wm_next], 0)       # (E, N, d)
        aux = self.wm_aux(x)
        return nxt, aux

    def cfg_dict(self) -> dict:
        return asdict(self.c)
