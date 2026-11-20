"""CAM 2.0: the trainable neural agent that works with a frozen LLM.

Per decision step (batched over environment streams):

  frozen-LLM features (no grad)                      CAM (all trainable, fp32)
  ------------------------------                      ---------------------------------------------------------------
  goal: per-token hidden states        ──► goal tokens ┐
  observation elements: pooled states  ──► element tokens (+ role/depth/position features)
  feedback text (errors/outputs)       ──► feedback token     ├─► semantic-integration transformer ─► x_t, element ctx
  previous action text, reward, time   ──► action token        │
  retrieved episodic memories          ──► memory tokens (learned retrieval)
  recurrent state h_{t-1}              ──► state tokens ┘
                                                         x_t ─► recurrent core (GRU / LSTM / none) ─► h_t ─► f_t
  tool docs / enum values: pooled      ──► candidate encoder over (tool, element, enum) ─► c_k
                                                         f_t, c_k ─► policy logits, Q(f,c), V(f), deliberation logits
                                                         f_t, c_a ─► control vectors (P × LLM-hidden) ─► frozen LLM
                                                                     generates the text / JSON argument
                                                         f_t, a  ─► world-model ensemble: next f, next x, reward, done

Nothing here touches the LLM's weights. ``prefix`` outputs live in the LLM's input-embedding space and are
consumed by ``FrozenLLM.sample_with_prefix`` / ``logprob_with_prefix`` (gradient flows back into CAM).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

EL_FEATS = 4  # targetable, depth/8, position/N, is-history/output (non-page element)


@dataclass
class CAMConfig:
    # ---- dimensions supplied by the frozen backbone (not searched)
    llm_pooled_dim: int = 4096
    llm_hidden: int = 2048
    # ---- architecture (searchable; bounds in cam2/search/spec.py)
    d: int = 512                  # integration width
    integ_layers: int = 4
    heads: int = 8
    ff_mult: int = 4
    core: str = "gru"             # gru | lstm | none
    core_dim: int = 1024
    state_tokens: int = 4         # recurrent state fed back to the integration network as tokens
    mem_k: int = 8                # episodic memories retrieved per step (0 = no episodic memory)
    mem_capacity: int = 50_000
    wm_ensemble: int = 5
    wm_hidden: int = 1024
    wm_layers: int = 2
    prefix_len: int = 8           # CAM control vectors given to the frozen LLM (0 = LLM generates unconditioned)
    prefix_rank: int = 256
    head_hidden: int = 512
    q_head: bool = True
    deliberation: bool = False    # learned act-vs-plan decision (planning effort allocation)
    el_goal_cond: bool = True     # frozen LLM encodes each element in the context of the goal
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def mlp(i: int, h: int, o: int, layers: int = 2) -> nn.Sequential:
    mods: list = []
    d = i
    for _ in range(layers - 1):
        mods += [nn.Linear(d, h), nn.GELU()]
        d = h
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


class EpisodicMemory:
    """Persistent store of past decision states with their verified outcomes (non-parametric, CPU/GPU tensor ring
    buffer). Retrieval uses CAM's *learned* query/key projections; the rows retrieved at a decision are stored with
    the transition so the learner can replay the decision exactly. Saved and restored with checkpoints."""

    def __init__(self, raw_dim: int, capacity: int, device: str):
        self.raw_dim, self.capacity, self.device = raw_dim, capacity, device
        self.rows = torch.zeros(capacity, raw_dim, dtype=torch.float16, device=device)
        self.n, self.ptr, self.writes = 0, 0, 0

    def write(self, rows: torch.Tensor) -> None:
        rows = rows.detach().to(self.device, torch.float16)
        for r in rows.split(4096):
            k = r.shape[0]
            idx = (torch.arange(k, device=self.device) + self.ptr) % self.capacity
            self.rows[idx] = r
            self.ptr = int((self.ptr + k) % self.capacity)
            self.n = min(self.capacity, self.n + k)
            self.writes += k

    def state_dict(self) -> dict:
        return {"rows": self.rows[: self.n].cpu() if self.n < self.capacity else self.rows.cpu(), "n": self.n,
                "ptr": self.ptr, "writes": self.writes}

    def load_state_dict(self, s: dict) -> None:
        r = s["rows"].to(self.device, torch.float16)
        self.rows[: r.shape[0]] = r
        self.n, self.ptr, self.writes = int(s["n"]), int(s["ptr"]), int(s["writes"])


class CAM(nn.Module):
    def __init__(self, cfg: CAMConfig, prefix_init: Optional[torch.Tensor] = None):
        super().__init__()
        self.cfg = c = cfg
        d, D, Dp, H = c.d, c.core_dim, c.llm_pooled_dim, c.llm_hidden
        # ---------------- semantic integration network
        self.goal_in = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, d))
        self.el_in = nn.Sequential(nn.LayerNorm(Dp), nn.Linear(Dp, d))
        self.el_feat = nn.Linear(EL_FEATS, d)
        self.el_pos = nn.Embedding(128, d)
        self.text_in = nn.Sequential(nn.LayerNorm(Dp), nn.Linear(Dp, d))
        self.act_in = nn.Sequential(nn.LayerNorm(Dp), nn.Linear(Dp, d))
        self.scalar_in = mlp(4, d, d)                          # prev reward, step frac, first-step flag, invalid flag
        self.type_emb = nn.Parameter(torch.zeros(8, d))        # cls, goal, element, text, action, memory, state, pad
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, c.heads, c.ff_mult * d, dropout=c.dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.integ = nn.TransformerEncoder(layer, c.integ_layers, enable_nested_tensor=False)
        self.integ_norm = nn.LayerNorm(d)
        # ---------------- recurrent core
        if c.core == "gru":
            self.core = nn.GRUCell(2 * d, D)
        elif c.core == "lstm":
            self.core = nn.LSTMCell(2 * d, D)
        elif c.core == "none":
            self.core = mlp(2 * d, D, D)
        else:
            raise ValueError(c.core)
        self.state_out = nn.Linear(D, c.state_tokens * d) if c.state_tokens > 0 else None
        self.f_norm = nn.LayerNorm(D)
        # ---------------- episodic memory (learned retrieval)
        self.mem_raw_dim = D + 4
        if c.mem_k > 0:
            self.mem_q = nn.Linear(D, 128)
            self.mem_key = nn.Linear(self.mem_raw_dim, 128)
            self.mem_v = mlp(self.mem_raw_dim, d, d)
        # ---------------- candidate encoder, policy, values
        self.tool_in = nn.Sequential(nn.LayerNorm(Dp), nn.Linear(Dp, d))
        self.enum_in = nn.Sequential(nn.LayerNorm(Dp), nn.Linear(Dp, d))
        self.null_el = nn.Parameter(torch.zeros(d))
        self.null_enum = nn.Parameter(torch.zeros(d))
        self.cand = nn.Sequential(nn.Linear(3 * d, c.head_hidden), nn.GELU(), nn.Linear(c.head_hidden, d), nn.LayerNorm(d))
        self.pol_q = nn.Linear(D, d)
        self.pol_mlp = mlp(2 * d, c.head_hidden, 1)
        self.value = mlp(D, c.head_hidden, 1, layers=3)
        if c.q_head:
            self.qnet = mlp(D + d, c.head_hidden, 1, layers=3)
        if c.deliberation:
            self.delib = mlp(D, c.head_hidden // 2, 2)
        # ---------------- control vectors for the frozen LLM (CAM-controlled generation)
        if c.prefix_len > 0:
            self.pre_a = nn.Sequential(nn.Linear(D + d, c.prefix_len * c.prefix_rank), nn.GELU())
            self.pre_b = nn.Linear(c.prefix_rank, H)
            base = prefix_init if prefix_init is not None else torch.randn(c.prefix_len, H) * 0.02
            reps = -(-c.prefix_len // base.shape[0])  # tile the initialising embedding rows up to prefix_len
            base = base.repeat(reps, 1)[: c.prefix_len]
            self.pre_base = nn.Parameter(base.clone().float())
            self.pre_scale = nn.Parameter(torch.tensor(float(base.norm(dim=-1).mean()) if prefix_init is not None else 0.5))
            # small (not zero) init: a zero output layer would block every gradient into pre_a at the start;
            # 1e-4 keeps the initial control vectors within a few percent of the initialising embeddings
            nn.init.normal_(self.pre_b.weight, std=1e-4)
            nn.init.zeros_(self.pre_b.bias)
        # ---------------- world model (ensemble; latent next-state, next observation embedding, reward, done)
        self.act_emb = nn.Linear(d + d, d)  # candidate embedding + generated-text embedding
        self.gen_text_in = nn.Sequential(nn.LayerNorm(Dp), nn.Linear(Dp, d))
        self.wm = nn.ModuleList([mlp(D + d, c.wm_hidden, D + d + 2, layers=c.wm_layers + 1) for _ in range(c.wm_ensemble)])
        for p in [self.type_emb, self.cls, self.null_el, self.null_enum]:
            nn.init.normal_(p, std=0.02)

    # ------------------------------------------------------------------ helpers
    def n_params(self) -> dict:
        groups = {"integration": ["goal_in", "el_in", "el_feat", "el_pos", "text_in", "act_in", "scalar_in", "type_emb",
                                  "cls", "integ", "integ_norm"],
                  "core": ["core", "state_out", "f_norm"], "memory": ["mem_q", "mem_key", "mem_v"],
                  "policy_value": ["tool_in", "enum_in", "null_el", "null_enum", "cand", "pol_q", "pol_mlp", "value",
                                   "qnet", "delib"],
                  "llm_control": ["pre_a", "pre_b", "pre_base", "pre_scale"],
                  "world_model": ["act_emb", "gen_text_in", "wm"]}
        out = {}
        for g, names in groups.items():
            out[g] = sum(p.numel() for n, p in self.named_parameters() if n.split(".")[0] in names)
        out["total"] = sum(p.numel() for p in self.parameters())
        return out

    def init_state(self, B: int, device) -> dict:
        D = self.cfg.core_dim
        st = {"h": torch.zeros(B, D, device=device)}
        if self.cfg.core == "lstm":
            st["c"] = torch.zeros(B, D, device=device)
        return st

    def memory_query(self, h_prev: torch.Tensor) -> torch.Tensor:
        return self.mem_q(h_prev)

    @torch.no_grad()
    def retrieve(self, mem: EpisodicMemory, h_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k rows by the learned query/key similarity. Returns (B, K, raw) rows and (B, K) validity mask."""
        B, K = h_prev.shape[0], self.cfg.mem_k
        rows = torch.zeros(B, K, self.mem_raw_dim, device=h_prev.device)
        mask = torch.zeros(B, K, dtype=torch.bool, device=h_prev.device)
        if K == 0 or mem.n == 0:
            return rows, mask
        q = self.mem_q(h_prev)
        allr = mem.rows[: mem.n].float()
        keys = self.mem_key(allr)
        sim = q @ keys.T
        k = min(K, mem.n)
        idx = sim.topk(k, dim=1).indices
        rows[:, :k] = allr[idx]
        mask[:, :k] = True
        return rows, mask

    def memory_rows(self, f: torch.Tensor, ret: torch.Tensor, success: torch.Tensor, frac: torch.Tensor) -> torch.Tensor:
        return torch.cat([f.detach(), ret.unsqueeze(-1), success.unsqueeze(-1), frac.unsqueeze(-1),
                          torch.ones_like(frac).unsqueeze(-1)], -1)

    # ------------------------------------------------------------------ forward
    def integrate(self, b: dict, st: dict) -> tuple[torch.Tensor, torch.Tensor, dict, torch.Tensor]:
        """One decision step. ``b``: batch of step inputs (see ``featurize.collate``). Returns
        (f_t features, element context (B,Ne,d), new state, x_t)."""
        c = self.cfg
        B = b["el"].shape[0]
        te = self.type_emb
        toks = [self.cls.expand(B, 1, -1) + te[0]]
        masks = [torch.ones(B, 1, dtype=torch.bool, device=b["el"].device)]
        toks.append(self.goal_in(b["goal"]) + te[1])
        masks.append(b["goal_m"])
        Ne = b["el"].shape[1]
        pos = self.el_pos(torch.arange(Ne, device=b["el"].device).clamp(max=127))
        toks.append(self.el_in(b["el"]) + self.el_feat(b["el_f"]) + pos + te[2])
        masks.append(b["el_m"])
        toks.append((self.text_in(b["text"]) + te[3]).unsqueeze(1))
        masks.append(b["text_m"].unsqueeze(1))
        toks.append((self.act_in(b["pact"]) + self.scalar_in(b["scal"]) + te[4]).unsqueeze(1))
        masks.append(torch.ones(B, 1, dtype=torch.bool, device=b["el"].device))
        if c.mem_k > 0:
            rows, mm = b["mem"], b["mem_m"]
            q = self.mem_q(st["h"])
            w = torch.softmax((torch.einsum("bd,bkd->bk", q, self.mem_key(rows)) / math.sqrt(128)).masked_fill(~mm, -1e4), -1)
            toks.append(self.mem_v(rows) * (w * mm.float() * rows.shape[1]).unsqueeze(-1) + te[5])
            masks.append(mm)
        if self.state_out is not None:
            toks.append(self.state_out(st["h"]).view(B, c.state_tokens, c.d) + te[6])
            masks.append(torch.ones(B, c.state_tokens, dtype=torch.bool, device=b["el"].device))
        x = torch.cat(toks, 1)
        m = torch.cat(masks, 1)
        y = self.integ_norm(self.integ(x, src_key_padding_mask=~m))
        xt = y[:, 0]
        el_ctx = y[:, 1 + b["goal"].shape[1]: 1 + b["goal"].shape[1] + Ne]
        inp = torch.cat([xt, self.act_in(b["pact"])], -1)
        if c.core == "gru":
            h = self.core(inp, st["h"])
            new = {"h": h}
        elif c.core == "lstm":
            h, cc = self.core(inp, (st["h"], st["c"]))
            new = {"h": h, "c": cc}
        else:
            h = self.core(inp)
            new = {"h": h}
        f = self.f_norm(h)
        return f, el_ctx, new, xt

    def candidates(self, b: dict, el_ctx: torch.Tensor) -> torch.Tensor:
        """(B, Nc, d) candidate embeddings for (tool, element, enum) triples."""
        B, Nc = b["c_tool"].shape
        tool = self.tool_in(b["tool_tab"])[b["c_tool"].clamp(min=0)]                     # (B,Nc,d)
        eli = b["c_el"]
        el = torch.where((eli >= 0).unsqueeze(-1), el_ctx.gather(1, eli.clamp(min=0).unsqueeze(-1).expand(B, Nc, el_ctx.shape[-1])),
                         self.null_el.expand(B, Nc, -1))
        eni = b["c_enum"]
        en = torch.where((eni >= 0).unsqueeze(-1), self.enum_in(b["enum_tab"])[eni.clamp(min=0)],
                         self.null_enum.expand(B, Nc, -1))
        return self.cand(torch.cat([tool, el, en], -1))

    def policy_logits(self, f: torch.Tensor, cand: torch.Tensor, cmask: torch.Tensor) -> torch.Tensor:
        q = self.pol_q(f).unsqueeze(1)
        lg = (cand * q).sum(-1) / math.sqrt(cand.shape[-1]) + self.pol_mlp(torch.cat([cand * q, cand], -1)).squeeze(-1)
        return lg.masked_fill(~cmask, -1e9)

    def q_values(self, f: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        B, Nc, _ = cand.shape
        return self.qnet(torch.cat([f.unsqueeze(1).expand(B, Nc, -1), cand], -1)).squeeze(-1)

    def v(self, f: torch.Tensor) -> torch.Tensor:
        return self.value(f).squeeze(-1)

    def delib_logits(self, f: torch.Tensor) -> torch.Tensor:
        return self.delib(f)

    def prefix(self, f: torch.Tensor, c_sel: torch.Tensor) -> torch.Tensor:
        """(B, P, llm_hidden) control vectors for the frozen LLM, from CAM's state and the chosen candidate."""
        c = self.cfg
        z = self.pre_a(torch.cat([f, c_sel], -1)).view(-1, c.prefix_len, c.prefix_rank)
        delta = self.pre_b(z)
        base = self.pre_base / self.pre_base.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return (base.unsqueeze(0) + delta) * self.pre_scale

    def action_embedding(self, c_sel: torch.Tensor, gen_text: Optional[torch.Tensor], gen_m: Optional[torch.Tensor]):
        g = torch.zeros_like(c_sel) if gen_text is None else self.gen_text_in(gen_text) * gen_m.unsqueeze(-1).float()
        return self.act_emb(torch.cat([c_sel, g], -1))

    def wm_predict(self, f: torch.Tensor, a: torch.Tensor) -> dict:
        """Ensemble predictions; each member: next f (D), next x (d), reward, done-logit."""
        D, d = self.cfg.core_dim, self.cfg.d
        outs = torch.stack([m(torch.cat([f, a], -1)) for m in self.wm], 0)           # (E,B,D+d+2)
        nf = outs[..., :D]
        return {"next_f": nf, "next_x": outs[..., D:D + d], "reward": outs[..., D + d], "done": outs[..., D + d + 1]}


def param_checksum(m: nn.Module) -> float:
    with torch.no_grad():
        return float(sum((p.double() * (i + 1)).sum() for i, p in enumerate(m.parameters())))
