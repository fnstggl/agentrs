"""ActionGenerator: CAM's trainable, latent-conditioned action policy (spec Part 9; architecture §2).

pi(a | R, G, B) = pi(tool | R, TOOL tokens, availability mask)
                x pi(evidence | R, tool, observation tokens)        (<= cfg.max_evidence sequential pointer picks + stop)
                x prod_k pi(decision_k | R, tool, schema slot, previous decisions)      (typed heads, grammar.ArgHeads)
                x prod_t pi(token_t | bridge prefix P(R,G,B), tool schema, evidence, previous tokens)  (Qwen + LoRA)

* tool: pointer attention from a ContextReader query over [R; G; B] onto the TOOL tokens of every listed tool plus the
  built-ins `wait` ({"seconds": 1..cfg.max_wait_s}) and `noop` ({}), which are ordinary tools with their own TOOL
  tokens. Masked to the available tools whose schemas the constrained decoder supports (unsupported schemas are
  counted in `stats["ungenerable"]` with the reason in `ungenerable`). A renamed or unseen tool is just another key.
* evidence: only for tools whose schema can contain strings - CAM (not the language model) picks up to max_evidence
  observations (grouped from the observation tokens through their refs) that are shown verbatim to the decoder.
* arguments: grammar.walk + ArgHeads (skeleton) and grammar.HoleFiller (strings), always schema-valid.

`sample()` returns `Action`s with source CAM_DECODER, `logp` (= sum of the parts) and `logp_parts` {tool, evidence,
args, args_heads, args_tokens, entropy, n_decisions, n_tokens, decode}; `logp_parts["decode"]` is the exact replay
record. `log_prob()` recomputes the behaviour log-probability and the path entropy of given actions with teacher
forcing - differentiable into R (and the G/B tokens), the bridge, the pointers, the typed heads and the LoRA weights;
the pretrained Qwen weights are frozen. Actions without a record (demonstrations) are encoded from their arguments
(numbers snap to the nearest grid point; flagged) and score without an evidence term unless the record names it.

Per-action compute: one tool-pointer read, <= max_evidence pointer steps, one ArgHeads read per non-trivial decision
(batched over rows), and - only when a string hole exists - one Qwen prefill (prompt ~ tool schema + evidence, + n_prefix
prefix vectors) plus one decoder step per forced/sampled token after the first hole; `last_compute` reports the counts.
"""
from __future__ import annotations

import json
import math
from typing import Optional

import torch
import torch.nn as nn

from ..config import CamConfig
from ..interfaces import Action, ActionSource, TokenSet, ToolSchema
from .bridge import LatentBridge, decision_context
from .constraints import tables_for
from .grammar import (ArgHeads, GrammarError, GrammarOptions, HoleFiller, ToolGrammar, drive_walks, encode_value,
                      holes_of, materialize)
from .layers import ContextReader, RMSNorm, masked_entropy, masked_log_softmax
from .tokenizer import parse_obs_ref, render_observation

BUILTIN_NAMES = ("wait", "noop")
DECODE_VERSION = 1


def builtin_tools(cfg: CamConfig) -> list:
    return [
        ToolSchema("wait", "Let simulated time pass without acting, so that slow effects (deployments, sales, "
                           "reports, replies) can be observed before the next decision.",
                   {"type": "object", "properties": {"seconds": {"type": "integer", "minimum": 1,
                                                                 "maximum": int(cfg.max_wait_s),
                                                                 "description": "simulated seconds to wait"}},
                    "required": ["seconds"], "additionalProperties": False},
                   returns="nothing; time advances", meta={"builtin": True}),
        ToolSchema("noop", "Do nothing at this decision.", {"type": "object", "properties": {},
                                                             "additionalProperties": False},
                   meta={"builtin": True}),
    ]


def _validate(schema: dict, value) -> list:
    from crucible.core.schema import validate
    return validate(schema, value)


class ActionGenerator(nn.Module):
    def __init__(self, cfg: CamConfig, encoder, tokenizer, *, opts: Optional[GrammarOptions] = None,
                 include_builtins: bool = True, evidence_chars: int = 1500, max_prompt_tokens: int = 3072,
                 autocast: bool = True, bridge_layers: int = 2, exact_rows: bool = False):
        super().__init__()
        object.__setattr__(self, "enc", encoder)
        object.__setattr__(self, "tok", tokenizer)
        D, H = cfg.d_model, encoder.sem_dim
        self.cfg = cfg
        self.D = D
        self.opts = opts or GrammarOptions(max_string_tokens=cfg.max_arg_tokens,
                                           max_action_tokens=2 * cfg.max_arg_tokens)
        self.include_builtins = include_builtins
        self.evidence_chars = evidence_chars
        self.autocast = autocast
        self.exact_rows = exact_rows
        self.src_emb = nn.Embedding(3, D)
        self.ctx_norm = RMSNorm(D)
        self.tool_query = nn.Parameter(torch.randn(1, D) * 0.02)
        self.tool_reader = ContextReader(D, cfg.n_heads, 1)
        self.tool_q = nn.Linear(D, D)
        self.tool_k = nn.Linear(D, D)
        self.tool_prior = nn.Linear(D, 1)
        self.ev_query = nn.Parameter(torch.randn(1, D) * 0.02)
        self.ev_tool = nn.Linear(D, D)
        self.ev_reader = ContextReader(D, cfg.n_heads, 1)
        self.ev_q = nn.Linear(D, D)
        self.ev_k = nn.Linear(D, D)
        self.ev_stop = nn.Linear(D, 1)
        self.ev_update = nn.Linear(D, D)
        self.heads = ArgHeads(D, cfg.n_heads, H, self.opts)
        self.bridge = LatentBridge(cfg, H, encoder.embedding_rms,
                                   LatentBridge.init_rows_from(encoder, cfg.n_prefix), n_layers=bridge_layers)
        # LoRA tensors of the decoder path: registered so they are optimised and saved with the generator
        self.lora = nn.ParameterList(encoder.lora_parameters())
        self.tables = tables_for(encoder.tok, encoder.vocab_size, encoder.device, encoder.model_name)
        self.filler = HoleFiller(encoder, self.tables, self.opts, max_prompt_tokens)
        self._grammars: dict = {}
        self.ungenerable: dict = {}
        self.builtins = builtin_tools(cfg) if include_builtins else []
        self.stats = {"actions": 0, "ungenerable": 0, "schema_violations": 0, "qwen_steps": 0, "decisions": 0,
                      "sampled_tokens": 0}
        self.last_compute: dict = {}
        self.to(encoder.device)

    # ------------------------------------------------------------------ bookkeeping
    @property
    def device(self):
        return self.tool_query.device

    def _ac(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast and self.device.type == "cuda")

    def count_parameters(self) -> dict:
        lora_ids = {id(p) for p in self.lora}
        own = sum(p.numel() for p in self.parameters() if id(p) not in lora_ids)
        return {"action_generator": own, "bridge": sum(p.numel() for p in self.bridge.parameters()),
                "typed_heads": sum(p.numel() for p in self.heads.parameters()),
                "lora": sum(p.numel() for p in self.lora), "pretrained_semantic": self.enc.n_pretrained}

    def grammar(self, tool: ToolSchema) -> ToolGrammar:
        g = self._grammars.get(tool.key)
        if g is None:
            g = ToolGrammar(tool, self.opts)
            self._grammars[tool.key] = g
            if not g.generable:
                self.ungenerable[tool.name] = g.error
                self.stats["ungenerable"] += 1
        return g

    def tool_lists_with_builtins(self, tool_lists: list) -> list:
        out = []
        for lst in tool_lists:
            names = {t.name for t in lst}
            out.append(list(lst) + [t for t in self.builtins if t.name not in names])
        return out

    def _augment_tools(self, tools: TokenSet, tool_lists: list):
        """Append TOOL tokens of the built-in tools (computed with the same tokenizer) to every row."""
        B = len(tool_lists)
        lists2 = self.tool_lists_with_builtins(tool_lists)
        extra = [l2[len(l):] for l, l2 in zip(tool_lists, lists2)]
        if not any(extra):
            return tools, lists2
        if tools.N != max((len(l) for l in tool_lists), default=0):
            raise ValueError("tools TokenSet does not match tool_lists")
        ext = self.tok.tools(extra)
        # left-align each row's built-ins right after its own tools
        N = max(len(l) for l in lists2)
        dev = self.device
        x = torch.zeros(B, N, self.D, device=dev)
        mask = torch.zeros(B, N, dtype=torch.bool, device=dev)
        refs = [[None] * N for _ in range(B)]
        for b in range(B):
            n0, n1 = len(tool_lists[b]), len(extra[b])
            if n0:
                x[b, :n0] = tools.x[b, :n0].float()
                mask[b, :n0] = tools.mask[b, :n0]
            if n1:
                x[b, n0:n0 + n1] = ext.x[b, :n1]
                mask[b, n0:n0 + n1] = True
            refs[b][: n0 + n1] = [t.key for t in lists2[b]]
        kind = torch.full((B, N), 16, dtype=torch.long, device=dev)
        return TokenSet(x, mask, kind, torch.zeros(B, N, device=dev), torch.full((B, N), -1, dtype=torch.long,
                                                                               device=dev), refs), lists2

    @staticmethod
    def _evidence_groups(obs: Optional[TokenSet], obs_lists: list, B: int):
        """Per row: (observation indices, token positions per observation)."""
        groups = []
        for b in range(B):
            g: dict = {}
            if obs is not None and obs.refs is not None and obs.N:
                m = obs.mask[b].tolist()
                for j, ref in enumerate(obs.refs[b]):
                    i = parse_obs_ref(ref)
                    if i is not None and m[j] and i < len(obs_lists[b]):
                        g.setdefault(i, []).append(j)
            idx = sorted(g)
            groups.append((idx, [g[i] for i in idx]))
        return groups

    def _prompt(self, tool: ToolSchema, ev_obs: list) -> str:
        lines = ["Write the JSON arguments for one call of the tool below. Reply with the JSON object only.",
                 f"Tool: {tool.name}", f"Description: {tool.description}",
                 "Arguments schema: " + json.dumps(tool.parameters, ensure_ascii=False, separators=(",", ":"))]
        if ev_obs:
            lines.append("Evidence selected for this decision:")
            for k, o in enumerate(ev_obs):
                lines.append(f"[{k + 1}] " + render_observation(o, self.evidence_chars))
        return "\n".join(lines)

    # ------------------------------------------------------------------ the shared forward
    def _forward(self, R, goal, belief, tools, obs, tool_lists, obs_lists, *, teacher=None, temperature=1.0,
                 greedy=False, generator=None, tool_mask=None, sample=True, exact_rows=False):
        B, dev, D = R.shape[0], R.device, self.D
        if teacher is None:
            temps = torch.full((B,), float(temperature), device=dev)
            active = list(range(B))
        else:
            temps = torch.tensor([float(t["temperature"]) if t is not None else 1.0 for t in teacher], device=dev)
            active = [b for b in range(B) if teacher[b] is not None]
        zero = R.new_zeros((), dtype=torch.float32)
        parts = {k: [zero for _ in range(B)] for k in ("tool", "evidence", "heads", "tokens")}
        ents = {k: [zero for _ in range(B)] for k in ("tool", "evidence", "heads", "tokens")}
        n_dec, n_tok = [0] * B, [0] * B
        with self._ac():
            ctx_x, ctx_m = decision_context(R, goal, belief, self.src_emb)
            ctx_x = self.ctx_norm(ctx_x)
            tools2, lists2 = self._augment_tools(tools, tool_lists)
            T = tools2.N
            grams = [[self.grammar(t) for t in lists2[b]] for b in range(B)]
            gm = [[g.generable for g in grams[b]] + [False] * (T - len(grams[b])) for b in range(B)]
            gen_mask = torch.tensor(gm, dtype=torch.bool, device=dev).reshape(B, T)
            allowed = tools2.mask & gen_mask
            if tool_mask is not None:
                allowed = allowed & tool_mask.to(dev)[:, :T]
            if teacher is not None:                     # re-apply a restriction recorded at sampling time
                for b in active:
                    rec_allowed = teacher[b].get("allowed")
                    if rec_allowed is not None:
                        rm = torch.zeros(T, dtype=torch.bool, device=dev)
                        rm[torch.tensor(rec_allowed, dtype=torch.long, device=dev)] = True
                        allowed[b] = allowed[b] & rm
            # ---- tool pointer
            qt = self.tool_reader(self.tool_query.unsqueeze(0).expand(B, -1, -1), ctx_x, ctx_m).squeeze(1)
            keys = self.tool_k(tools2.x)
            logits = (self.tool_q(qt).unsqueeze(1) * keys).sum(-1) / math.sqrt(D) + self.tool_prior(tools2.x).squeeze(-1)
            tool_idx = [-1] * B
            for b in active:
                if not bool(allowed[b].any()):
                    raise GrammarError(f"row {b}: no available tool can be generated")
            lp_tool = masked_log_softmax(logits, allowed | ~allowed.any(1, keepdim=True), temps.unsqueeze(1))
            for b in active:
                if teacher is None:
                    if greedy:
                        tool_idx[b] = int(lp_tool[b].masked_fill(~allowed[b], -torch.inf).argmax())
                    else:
                        tool_idx[b] = int(torch.multinomial(lp_tool[b].exp(), 1, generator=generator))
                else:
                    tool_idx[b] = self._teacher_tool(teacher[b], lists2[b])
                if not bool(allowed[b, tool_idx[b]]):
                    raise GrammarError(f"row {b}: tool {lists2[b][tool_idx[b]].name} is not available/generable")
                parts["tool"][b] = lp_tool[b, tool_idx[b]]
                ents["tool"][b] = masked_entropy(lp_tool[b:b + 1], allowed[b:b + 1])[0]
            sel = torch.tensor([max(i, 0) for i in tool_idx], device=dev)
            tool_tok = tools2.x[torch.arange(B, device=dev), sel]                          # [B, D]
            chosen = [lists2[b][tool_idx[b]] if b in active else None for b in range(B)]
            gsel = [grams[b][tool_idx[b]] if b in active else None for b in range(B)]
            # ---- evidence pointer
            evidence = [[] for _ in range(B)]
            groups = self._evidence_groups(obs, obs_lists, B)
            ev_rows = [b for b in active if gsel[b].may_hole and groups[b][0]
                       and not (teacher is not None and teacher[b].get("evidence") is None)]
            if ev_rows:
                self._evidence(ev_rows, groups, obs, tool_tok, ctx_x, ctx_m, temps, teacher, greedy, generator,
                               evidence, parts, ents)
            # ---- skeleton walk (typed heads)
            choices = [[] for _ in range(B)]
            forced = None
            roots = [gsel[b].root if b in active else None for b in range(B)]
            demo_holes: dict = {}
            flags: dict = {}
            if teacher is not None:
                forced = {}
                for b in active:
                    t = teacher[b]
                    if t.get("choices") is not None:
                        forced[b] = list(t["choices"])
                    else:                                                  # demonstration: encode the args
                        cs, hs = [], []
                        encode_value(gsel[b].root, t["args"], self.opts, cs, hs, flags)
                        forced[b] = cs
                        demo_holes[b] = hs
            state = {"h": self.heads.init_state(tool_tok, qt), "ptr": [0] * B}

            def decide(rows, decs):
                rt = torch.tensor(rows, device=dev)
                fc = None
                if forced is not None:
                    fc = []
                    for r in rows:
                        p = state["ptr"][r]
                        if p >= len(forced[r]):
                            raise GrammarError(f"row {r}: recorded choices exhausted")
                        fc.append(forced[r][p])
                        state["ptr"][r] = p + 1
                cs, lp, en, hn = self.heads.decide(self.enc, decs, state["h"][rt], tool_tok[rt], ctx_x[rt], ctx_m[rt],
                                                   temps[rt], greedy, fc, generator)
                state["h"] = state["h"].index_copy(0, rt, hn.to(state["h"].dtype))
                for j, r in enumerate(rows):
                    parts["heads"][r] = parts["heads"][r] + lp[j]
                    ents["heads"][r] = ents["heads"][r] + en[j]
                    choices[r].append(cs[j])
                    n_dec[r] += 1
                return cs

            templates = drive_walks(roots, self.opts, decide)
            if forced is not None:
                for b in active:
                    if state["ptr"][b] != len(forced[b]):
                        raise GrammarError(f"row {b}: {len(forced[b]) - state['ptr'][b]} recorded choices unused")
            # ---- strings (Qwen + LoRA, bridge prefix)
            hole_rows = [b for b in active if holes_of(templates[b])]
            programs, ctx_ids, prompt_sha = {}, {}, {}
            for b in hole_rows:
                programs[b] = self.filler.program(templates[b])
                ev_obs = [obs_lists[b][i] for i in evidence[b]]
                ctx_ids[b] = self.filler.context_ids(self._prompt(chosen[b], ev_obs))
                prompt_sha[b] = self.filler.prompt_hash(ctx_ids[b])
                if teacher is not None and teacher[b].get("prompt") and teacher[b]["prompt"] != prompt_sha[b]:
                    raise GrammarError(f"row {b}: decoder prompt differs from the recorded one (different tool "
                                       f"schema or evidence content)")
            qwen_steps = 0
            if hole_rows:
                rt = torch.tensor(hole_rows, device=dev)
                prefix = self.bridge(R[rt], goal.select(rt) if goal is not None else None,
                                     belief.select(rt) if belief is not None else None)
                progs = [programs[b] for b in hole_rows]
                ctxs = [ctx_ids[b] for b in hole_rows]
                if sample:
                    lp, en, ns, qwen_steps = self.filler.sample(progs, prefix, ctxs, float(temperature), greedy,
                                                                generator)
                    for j, b in enumerate(hole_rows):
                        n_tok[b] = ns[j]
                else:
                    for b in hole_rows:
                        p = programs[b]
                        if b in demo_holes:
                            hs = demo_holes[b]
                            if len(hs) != len(p.holes):
                                raise GrammarError(f"row {b}: demo strings do not match the holes")
                            for k, (h, (kind, val)) in enumerate(zip(p.holes, hs)):
                                h.tokens = self.filler.tokenize_hole(kind, val, h, p.cap[k])
                        else:
                            rec = teacher[b].get("holes") or []
                            if len(rec) != len(p.holes):
                                raise GrammarError(f"row {b}: {len(rec)} recorded holes for {len(p.holes)} slots")
                            for h, tk in zip(p.holes, rec):
                                h.tokens = list(tk)
                    tt = temps[rt].unsqueeze(1)
                    lp, en, nt = self.filler.score(progs, prefix, ctxs, tt, exact_rows=exact_rows)
                    for j, b in enumerate(hole_rows):
                        n_tok[b] = int(nt[j])
                for j, b in enumerate(hole_rows):
                    parts["tokens"][b] = lp[j]
                    ents["tokens"][b] = en[j]
        st = lambda d: torch.stack([v.float() for v in d])
        res = {k: st(v) for k, v in parts.items()}
        ent = {k: st(v) for k, v in ents.items()}
        self.last_compute = {"rows": B, "decisions": sum(n_dec), "evidence_rows": len(ev_rows), "hole_rows": len(hole_rows),
                             "tokens": sum(n_tok), "qwen_steps": qwen_steps,
                             "prompt_tokens": sum(len(ctx_ids[b]) for b in hole_rows)}
        self.stats["decisions"] += sum(n_dec)
        self.stats["qwen_steps"] += qwen_steps
        self.stats["sampled_tokens"] += sum(n_tok)
        restricted = None
        if tool_mask is not None:
            restricted = [torch.nonzero(allowed[b]).flatten().tolist() for b in range(B)]
        return {"logp": res, "entropy": ent, "tool_idx": tool_idx, "tools": chosen, "evidence": evidence,
                "allowed": restricted,
                "choices": choices, "templates": templates, "programs": programs, "prompt_sha": prompt_sha,
                "n_dec": n_dec, "n_tok": n_tok, "active": active, "approx": flags.get("approx", 0)}

    def _teacher_tool(self, t: dict, lst: list) -> int:
        idx = t.get("tool")
        if idx is not None and 0 <= idx < len(lst) and lst[idx].name == t["name"] and \
                (t.get("key") is None or lst[idx].key == t["key"]):
            return int(idx)
        for j, tool in enumerate(lst):
            if tool.name == t["name"] and (t.get("key") is None or tool.key == t["key"]):
                return j
        raise GrammarError(f"tool {t['name']!r} is not in the row's tool list")

    def _evidence(self, rows, groups, obs, tool_tok, ctx_x, ctx_m, temps, teacher, greedy, generator, evidence, parts,
                  ents):
        dev, D = self.device, self.D
        n = len(rows)
        M = max(len(groups[b][0]) for b in rows)
        src_b, src_t, dst, cnt_dst = [], [], [], []
        km = [[False] * M for _ in range(n)]
        for j, b in enumerate(rows):
            for m, toks in enumerate(groups[b][1]):
                src_b += [b] * len(toks)
                src_t += toks
                dst += [j * M + m] * len(toks)
                km[j][m] = True
        vals = obs.x[torch.tensor(src_b, device=dev), torch.tensor(src_t, device=dev)].float()
        di = torch.tensor(dst, device=dev)
        keys = torch.zeros(n * M, D, device=dev).index_add(0, di, vals)
        cnt = torch.zeros(n * M, device=dev).index_add(0, di, torch.ones(len(dst), device=dev))
        keys = (keys / cnt.clamp_min(1.0).unsqueeze(1)).view(n, M, D)       # mean of each observation's tokens
        kmask = torch.tensor(km, dtype=torch.bool, device=dev)
        rt = torch.tensor(rows, device=dev)
        q = self.ev_reader((self.ev_query + self.ev_tool(tool_tok[rt])).unsqueeze(1), ctx_x[rt], ctx_m[rt]).squeeze(1)
        K = self.ev_k(keys)
        avail = kmask.clone()
        stopped = [False] * n
        rec = None
        if teacher is not None:
            rec = []
            for b in rows:
                pos = {i: m for m, i in enumerate(groups[b][0])}
                want = teacher[b]["evidence"]
                miss = [i for i in want if i not in pos]
                if miss:
                    raise GrammarError(f"row {b}: recorded evidence {miss} has no observation tokens")
                rec.append([pos[i] for i in want])
        for step in range(self.cfg.max_evidence):
            live = [j for j in range(n) if not stopped[j] and bool(avail[j].any())]
            if not live:
                break
            lt = torch.tensor(live, device=dev)
            lg = (self.ev_q(q[lt]).unsqueeze(1) * K[lt]).sum(-1) / math.sqrt(D)
            full = torch.cat([lg.float(), self.ev_stop(q[lt]).float()], 1)
            m = torch.cat([avail[lt], torch.ones(len(live), 1, dtype=torch.bool, device=dev)], 1)
            lp = masked_log_softmax(full, m, temps[rt[lt]].unsqueeze(1))
            en = masked_entropy(lp, m)
            upd_rows, upd_vals = [], []
            for a, j in enumerate(live):
                if rec is not None:
                    c = rec[j][step] if step < len(rec[j]) else M
                elif greedy:
                    c = int(lp[a].argmax())
                else:
                    c = int(torch.multinomial(lp[a].exp(), 1, generator=generator))
                if not bool(m[a, c]):
                    raise GrammarError("evidence choice not admissible")
                b = rows[j]
                parts["evidence"][b] = parts["evidence"][b] + lp[a, c]
                ents["evidence"][b] = ents["evidence"][b] + en[a]
                if c == M:
                    stopped[j] = True
                else:
                    evidence[b].append(groups[b][0][c])
                    avail[j, c] = False
                    upd_rows.append(j)
                    upd_vals.append(K[j, c])
            if upd_rows:
                ur = torch.tensor(upd_rows, device=dev)
                q = q.index_add(0, ur, self.ev_update(torch.stack(upd_vals)).to(q.dtype))
        if rec is not None:
            for j, b in enumerate(rows):
                if len(rec[j]) > self.cfg.max_evidence:
                    raise GrammarError(f"row {b}: more recorded evidence than max_evidence")

    # ------------------------------------------------------------------ public API
    @torch.no_grad()
    def sample(self, R, goal: TokenSet, belief: TokenSet, tools: TokenSet, obs: TokenSet, tool_lists: list,
               obs_lists: list, temperature: float = 1.0, greedy: bool = False, *, generator=None, tool_mask=None,
               rescore: bool = False, exact_rows: bool = False,
               source: ActionSource = ActionSource.CAM_DECODER) -> list:
        """[B] Action with logp / logp_parts / decode record. rescore=True replaces the sampling-time logp by the
        learner-path recomputation (`log_prob`), so PPO ratios start at exactly 1 even under bf16 drift."""
        out = self._forward(R, goal, belief, tools, obs, tool_lists, obs_lists, temperature=temperature,
                            greedy=greedy, generator=generator, tool_mask=tool_mask, sample=True)
        actions = []
        for b in range(R.shape[0]):
            tool = out["tools"][b]
            mstats: dict = {}
            args = materialize(out["templates"][b], mstats)
            errs = _validate(tool.parameters or {"type": "object"}, args)
            if errs:
                self.stats["schema_violations"] += 1
                raise GrammarError(f"generated arguments for {tool.name} violate the schema: {errs[:3]}")
            lp = {k: float(v[b]) for k, v in out["logp"].items()}
            ent = sum(float(v[b]) for v in out["entropy"].values())
            rec = {"v": DECODE_VERSION, "tool": out["tool_idx"][b], "name": tool.name, "key": tool.key,
                   "evidence": list(out["evidence"][b]), "choices": list(out["choices"][b]),
                   "holes": [list(h.tokens) for h in (out["programs"][b].holes if b in out["programs"] else [])],
                   "prompt": out["prompt_sha"].get(b), "temperature": float(temperature), "greedy": bool(greedy),
                   "allowed": None if out["allowed"] is None else out["allowed"][b]}
            ev_entities = sorted({obs_lists[b][i].entity for i in out["evidence"][b] if obs_lists[b][i].entity})
            code = args.get("code") if isinstance(args.get("code"), str) else None
            dur = tool.duration_hint
            if tool.name == "wait" and tool.meta.get("builtin"):
                dur = args["seconds"] / 86400.0
            total = lp["tool"] + lp["evidence"] + lp["heads"] + lp["tokens"]
            actions.append(Action(tool=tool.name, args=args, entities=ev_entities, code=code, expected_duration=dur,
                                  expected_cost=tool.cost_hint, source=source, logp=total,
                                  logp_parts={"tool": lp["tool"], "evidence": lp["evidence"],
                                              "args": lp["heads"] + lp["tokens"], "args_heads": lp["heads"],
                                              "args_tokens": lp["tokens"], "entropy": ent,
                                              "n_decisions": out["n_dec"][b], "n_tokens": out["n_tok"][b],
                                              "dropped_keys": mstats.get("dropped_keys", 0), "decode": rec}))
        self.stats["actions"] += len(actions)
        if rescore:
            lpr, _ = self.log_prob_parts(R, goal, belief, tools, obs, tool_lists, obs_lists, actions,
                                         exact_rows=exact_rows)
            for b, a in enumerate(actions):
                for k_src, k_dst in (("tool", "tool"), ("evidence", "evidence"), ("heads", "args_heads"),
                                     ("tokens", "args_tokens")):
                    a.logp_parts[k_dst] = float(lpr[k_src][b])
                a.logp_parts["args"] = a.logp_parts["args_heads"] + a.logp_parts["args_tokens"]
                a.logp = float(sum(lpr[k][b] for k in lpr))
        return actions

    def _teacher(self, actions: list, evidence: Optional[list] = None) -> list:
        out = []
        for b, a in enumerate(actions):
            if a is None:
                out.append(None)
                continue
            rec = (a.logp_parts or {}).get("decode")
            if rec is not None:
                if rec.get("v") != DECODE_VERSION:
                    raise GrammarError(f"decode record version {rec.get('v')} != {DECODE_VERSION}")
                out.append(dict(rec))
            else:                                                   # demonstration / external action
                ev = None if evidence is None else evidence[b]
                out.append({"tool": None, "name": a.tool, "key": None, "evidence": ev, "choices": None, "holes": None,
                            "prompt": None, "temperature": 1.0, "args": a.args})
        return out

    def log_prob_parts(self, R, goal, belief, tools, obs, tool_lists, obs_lists, actions, evidence=None,
                       exact_rows: Optional[bool] = None):
        """({tool, evidence, heads, tokens}: [B] each, {same}: entropies [B]); differentiable. exact_rows (default
        self.exact_rows) scores the Qwen part row by row (batch-invariant under bf16)."""
        out = self._forward(R, goal, belief, tools, obs, tool_lists, obs_lists, teacher=self._teacher(actions, evidence),
                            sample=False, exact_rows=self.exact_rows if exact_rows is None else exact_rows)
        return out["logp"], out["entropy"]

    def log_prob(self, R, goal: TokenSet, belief: TokenSet, tools: TokenSet, obs: TokenSet, tool_lists: list,
                 obs_lists: list, actions: list, *, exact_rows: Optional[bool] = None) -> tuple:
        """ActionGeneratorP.log_prob -> (logp [B], entropy [B]) of the given actions (None rows give 0)."""
        lp, en = self.log_prob_parts(R, goal, belief, tools, obs, tool_lists, obs_lists, actions,
                                     exact_rows=exact_rows)
        return sum(lp.values()), sum(en.values())

    def evidence_log_prob(self, R, goal, belief, tools, obs, tool_lists, obs_lists, tool_names: list,
                          evidence: list) -> tuple:
        """log pi(evidence = labels | R, tool, obs) [B] and its entropy [B] (evidence-retrieval supervision).
        tool_names[b] names the tool the evidence is for; evidence[b] lists observation indices (<= max_evidence)."""
        B, dev = R.shape[0], R.device
        zero = R.new_zeros((), dtype=torch.float32)
        parts = {"evidence": [zero for _ in range(B)]}
        ents = {"evidence": [zero for _ in range(B)]}
        with self._ac():
            ctx_x, ctx_m = decision_context(R, goal, belief, self.src_emb)
            ctx_x = self.ctx_norm(ctx_x)
            tools2, lists2 = self._augment_tools(tools, tool_lists)
            idx = [self._teacher_tool({"name": n, "tool": None}, lists2[b]) for b, n in enumerate(tool_names)]
            tool_tok = tools2.x[torch.arange(B, device=dev), torch.tensor(idx, device=dev)]
            groups = self._evidence_groups(obs, obs_lists, B)
            rows = [b for b in range(B) if groups[b][0]]
            teacher = [{"evidence": list(e)} for e in evidence]
            out_ev = [[] for _ in range(B)]
            if rows:
                self._evidence(rows, groups, obs, tool_tok, ctx_x, ctx_m, torch.ones(B, device=dev), teacher, False,
                               None, out_ev, parts, ents)
        return torch.stack(parts["evidence"]).float(), torch.stack(ents["evidence"]).float()

    # ------------------------------------------------------------------ candidates / provenance (spec 9.5)
    @torch.no_grad()
    def sample_candidates(self, R, goal, belief, tools, obs, tool_lists, obs_lists, n: int,
                          temperature: float = 1.0, generator=None) -> list:
        """n independent CAM-decoder samples per row -> [B][n] Action (each with its own logp and record)."""
        B = R.shape[0]
        idx = torch.arange(B, device=R.device).repeat_interleave(n)
        rep = lambda ts: None if ts is None else ts.select(idx)
        acts = self.sample(R[idx], rep(goal), rep(belief), rep(tools), rep(obs),
                           [tool_lists[i] for i in idx.tolist()], [obs_lists[i] for i in idx.tolist()],
                           temperature=temperature, generator=generator)
        return [acts[b * n:(b + 1) * n] for b in range(B)]

    @staticmethod
    def select_candidate(candidates: list, probs: list, chosen: int, source: ActionSource = ActionSource.CAM_PLANNER,
                         scores: Optional[list] = None) -> Action:
        """Mark the candidate chosen by a selector (planner / baseline) with its provenance and the proposal
        distribution {n, probs, chosen, proposal_logp, scores}; the decoder record and logp stay attached."""
        if not (0 <= chosen < len(candidates)) or len(probs) != len(candidates):
            raise ValueError("bad candidate selection")
        a = Action.from_dict(candidates[chosen].to_dict())
        a.source = source
        a.proposal = {"n": len(candidates), "probs": [float(p) for p in probs], "chosen": int(chosen),
                      "proposal_logp": [c.logp for c in candidates],
                      "scores": None if scores is None else [float(s) for s in scores]}
        return a
