"""ActionGenerator: CAM's trainable, latent-conditioned action policy (spec Part 9; architecture §2).

pi(a | R, G, B) = pi(tool | R, TOOL tokens, availability mask)
                x pi(evidence | R, tool, observation tokens)        (<= cfg.max_evidence sequential pointer picks + stop)
                x prod_k pi(decision_k | R, tool, schema slot, previous decisions)      (typed heads, grammar.ArgHeads)
                x prod_t pi(token_t | bridge prefix P(R,G,B), tool schema, evidence, previous tokens)  (Qwen + LoRA)

* tool: pointer attention from a ContextReader query over [R; G; B] onto the TOOL tokens of every listed tool plus the
  built-ins `wait` ({"seconds": 1..cfg.max_wait_s}) and `noop` ({}), which are ordinary tools with their own TOOL
  tokens. Masked to the available tools whose schemas the constrained decoder supports (unsupported schemas are
  counted in `stats["ungenerable"]` with the reason in `ungenerable`; capability reductions of supported schemas are
  listed per tool in `reduced` and counted in `reduction_counts`). A renamed or unseen tool is just another key.
* evidence: for tools whose schema contains strings or numbers - CAM (not the language model) picks up to
  max_evidence observations (grouped from the observation tokens through their refs); they are shown verbatim to the
  string decoder and their numbers (values and numeric literals) are offered to the number-copy pointer.
* arguments: grammar.walk + ArgHeads (skeleton, exact numbers) and grammar.HoleFiller (strings), always schema-valid.

Behaviour log-probability. `sample()` decodes incrementally (KV cache, bf16 on GPU). In bf16 that incremental value
differs from a full teacher-forced forward by up to ~1 nat per action on free strings (verifier: median 0.16, p90
0.57, max 1.07 nats on Qwen3-0.6B, 3-80 token strings; 0.42 nats for a 1024-token Qwen3-1.7B string; this package's
test: median 0.13, max 0.53 over 16 actions), and a padded *batched* teacher-forced forward differs from the row-wise
one by a median ~0.19 nats. So by default (`rescore=True`) every sampled action's `logp` and `logp_parts` are
recomputed through the learner path `log_prob_parts(exact_rows=self.exact_rows)` before the action is returned; the
incremental value is kept only as the diagnostic `logp_parts["sample_logp"]` (`max_rescore_drift` tracks the largest
gap seen). With `exact_rows=True` (default) every component - pointers, typed heads, bridge and Qwen - is evaluated
row by row on that row's own (un-padded) tokens and every sum is a deterministic reduction, so a row's
log-probability does not depend on the minibatch it is scored in nor on grad mode: PPO ratios at the behaviour
parameters are exactly 1 when the learner also uses exact_rows (tested bit-for-bit, with gradients enabled, in a
permuted half batch). The cost is one forward per row; exact_rows=False batches the learner path at the price of the
drift above. The guarantee covers this module given the same R/G/B/TOOL/observation tensors: R recomputed by the
backbone in another batch is outside it. Greedy actions have behaviour log-probability 0 (a deterministic choice):
`Action.logp = 0.0`, `logp_parts["greedy"] = True` and the policy's tau-softmax log-probability of the greedy path in
`logp_parts["policy_logp"]`; `importance_weight_usable(a)` is False for them.

Replay safety. `logp_parts["decode"]` is the exact replay record: tool index/name/key, a fingerprint of the ordered
tool list (keys incl. built-ins), whether the evidence pointer was evaluated and a fingerprint of the evidence
candidates (observation index, ref and rendered content of every observation with tokens), the evidence picks, every
head choice, every hole's token ids, the decoder-prompt hash and its truncation count, temperature, greedy flag and
any tool restriction. `log_prob()` re-walks it with teacher forcing and raises GrammarError when the tool list, the
evidence candidates or the decoder prompt differ - it never silently scores a different decision. `log_prob(...,
temperature=t)` evaluates the same path under another temperature (e.g. the target policy at tau = 1). Actions without
a record (demonstrations) are encoded from their arguments (exact numbers; a number equal to an evidence number is
encoded as a copy) and score without an evidence term unless evidence labels are given.

Per-action compute: one tool-pointer read, <= max_evidence pointer steps, one ArgHeads read per non-trivial decision
(batched over rows; an exact decimal number costs 2-3 + #digits decisions), and - only when a string hole exists -
one Qwen prefill (prompt ~ tool schema + evidence, + n_prefix prefix vectors) plus one decoder step per forced/sampled
token after the first hole; with rescore=True one more teacher-forced Qwen forward per action with holes.
`last_compute` reports the counts.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Optional

import torch
import torch.nn as nn

from ..config import CamConfig
from ..interfaces import Action, ActionSource, TokenSet, ToolSchema
from .bridge import LatentBridge, decision_context
from .constraints import tables_for
from .grammar import (ArgHeads, GrammarError, GrammarOptions, HoleFiller, LengthHead, ToolGrammar, drive_walks,
                      encode_value, holes_of, materialize, number_candidates)
from .layers import ContextReader, RMSNorm, masked_entropy, masked_log_softmax
from .tokenizer import numeric_descriptor, obs_ref, parse_obs_ref, render_observation

BUILTIN_NAMES = ("wait", "noop")
DECODE_VERSION = 3          # 3: free strings carry the LengthHead stop-hazard mixture (grammar.LengthHead)
PART_KEYS = ("tool", "evidence", "heads", "tokens")


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


def _sha(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", "surrogatepass"))
        h.update(b"\x00")
    return h.hexdigest()[:24]


def row_tokens(ts: Optional[TokenSet], b: int) -> Optional[TokenSet]:
    """Row b of a TokenSet cut to its own length (up to its last real token), so a row's computation does not depend
    on the padding of the batch it came from."""
    if ts is None:
        return None
    m = ts.mask[b]
    n = int(torch.nonzero(m).max()) + 1 if bool(m.any()) else 0
    return TokenSet(ts.x[b:b + 1, :n], ts.mask[b:b + 1, :n], ts.kind[b:b + 1, :n], ts.time[b:b + 1, :n],
                    ts.entity[b:b + 1, :n], None if ts.refs is None else [list(ts.refs[b][:n])],
                    None if ts.entity2 is None else ts.entity2[b:b + 1, :n])


class ActionGenerator(nn.Module):
    def __init__(self, cfg: CamConfig, encoder, tokenizer, *, opts: Optional[GrammarOptions] = None,
                 include_builtins: bool = True, evidence_chars: int = 1500, max_prompt_tokens: int = 3072,
                 autocast: bool = True, bridge_layers: int = 2, exact_rows: bool = True):
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
        # learned stop hazard of free strings (length prior of the untrained decoder; grammar.LengthHead)
        self.length_head = LengthHead(D, H) if self.opts.length_prior else None
        self.filler.length_head = self.length_head
        self._grammars: dict = {}
        self.ungenerable: dict = {}
        self.reduced: dict = {}                      # tool name -> [{"path", "kind", "detail"}]
        self.reduction_counts: dict = {}             # reduction kind -> number of slots (over compiled tools)
        self.builtins = builtin_tools(cfg) if include_builtins else []
        self.stats = {"actions": 0, "ungenerable": 0, "reduced_tools": 0, "schema_violations": 0, "qwen_steps": 0,
                      "decisions": 0, "sampled_tokens": 0, "prompt_truncated_tokens": 0, "copied_numbers": 0,
                      "rescored": 0, "demo_numbers": 0, "demo_numbers_approx": 0, "demo_numbers_copied": 0}
        self.max_rescore_drift = 0.0                 # max |incremental sampling logp - learner-path logp| seen
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
            elif g.reductions:
                self.reduced[tool.name] = list(g.reductions)
                self.stats["reduced_tools"] += 1
                for r in g.reductions:
                    self.reduction_counts[r["kind"]] = self.reduction_counts.get(r["kind"], 0) + 1
        return g

    def tool_lists_with_builtins(self, tool_lists: list) -> list:
        out = []
        for lst in tool_lists:
            names = {t.name for t in lst}
            out.append(list(lst) + [t for t in self.builtins if t.name not in names])
        return out

    @staticmethod
    def tools_fingerprint(lst: list) -> str:
        return _sha(*[t.key for t in lst])

    def evidence_fingerprint(self, idx: list, obs_list: list) -> str:
        """Identity + content of the evidence candidates (observations that have tokens)."""
        return _sha(*[f"{obs_ref(i, obs_list[i])}|{_sha(render_observation(obs_list[i], self.evidence_chars))}"
                      for i in idx])

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

    def _prompt(self, tool: ToolSchema, ev_obs: list) -> list:
        """Decoder prompt as prioritised pieces (HoleFiller.fit_pieces shortens evidence first, then the description,
        the arguments schema last)."""
        pieces = [("Write the JSON arguments for one call of the tool below. Reply with the JSON object only.\n"
                   f"Tool: {tool.name}\n", 0),
                  (f"Description: {tool.description}\n", 2),
                  ("Arguments schema: " + json.dumps(tool.parameters, ensure_ascii=False, separators=(",", ":"))
                   + ("\n" if ev_obs else ""), 3)]
        if ev_obs:
            pieces.append(("Evidence selected for this decision:\n", 0))
            for k, o in enumerate(ev_obs):
                pieces.append((f"[{k + 1}] " + render_observation(o, self.evidence_chars)
                               + ("\n" if k + 1 < len(ev_obs) else ""), 1))
        return pieces

    def number_sources(self, ev_obs: list) -> list:
        """Copy candidates of the selected evidence (same rendering as the decoder prompt)."""
        src = []
        for o in ev_obs:
            vals = []
            if o.value is not None:
                vals.append((o.value, f"{numeric_descriptor(o)} = {o.value!r} {o.unit or ''}".strip()))
            src.append((vals, render_observation(o, self.evidence_chars)))
        return number_candidates(src, self.opts.max_copy_candidates)

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
        parts = {k: [zero for _ in range(B)] for k in PART_KEYS}
        ents = {k: [zero for _ in range(B)] for k in PART_KEYS}
        n_dec, n_tok, n_copy = [0] * B, [0] * B, [0] * B
        with self._ac():
            ctx_x, ctx_m = decision_context(R, goal, belief, self.src_emb)
            ctx_x = self.ctx_norm(ctx_x)
            tools2, lists2 = self._augment_tools(tools, tool_lists)
            T = tools2.N
            tools_sha = [self.tools_fingerprint(lists2[b]) for b in range(B)]
            if teacher is not None:
                for b in active:
                    rec_sha = teacher[b].get("tools")
                    if rec_sha is not None and rec_sha != tools_sha[b]:
                        raise GrammarError(f"row {b}: the tool list differs from the one the action was sampled with")
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
            want = {b: gsel[b].uses_evidence and bool(groups[b][0]) for b in active}
            ev_sha = {b: self.evidence_fingerprint(groups[b][0], obs_lists[b]) for b in active if want[b]}
            ev_rows = []
            for b in active:
                t = teacher[b] if teacher is not None else None
                if t is not None and "ev" in t:                               # sampled action: replay exactly
                    if bool(t["ev"]) != want[b]:
                        raise GrammarError(f"row {b}: the evidence pointer was {'' if t['ev'] else 'not '}evaluated "
                                           f"when the action was sampled, but the replay has "
                                           f"{'no' if not want[b] else 'some'} evidence candidates")
                    if want[b] and t.get("cands") != ev_sha[b]:
                        raise GrammarError(f"row {b}: the evidence candidates (observations or their content) differ "
                                           f"from the ones the action was sampled with")
                elif t is not None:                                           # demonstration
                    if t.get("evidence") is None:
                        continue
                    if t["evidence"] and not want[b]:
                        raise GrammarError(f"row {b}: evidence labels given but the row has no evidence candidates")
                if want[b]:
                    ev_rows.append(b)
            if ev_rows:
                self._evidence(ev_rows, groups, obs, tool_tok, ctx_x, ctx_m, temps, teacher, greedy, generator,
                               evidence, parts, ents)
            ev_obs = {b: [obs_lists[b][i] for i in evidence[b]] for b in active}
            cands = [self.number_sources(ev_obs[b]) if b in active and evidence[b] else [] for b in range(B)]
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
                        encode_value(gsel[b].root, t["args"], self.opts, cs, hs, flags, cands[b])
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
                    if decs[j].kind == "numsrc" and cs[j] > 0:
                        n_copy[r] += 1
                return cs

            templates = drive_walks(roots, self.opts, decide, cands)
            if forced is not None:
                for b in active:
                    if state["ptr"][b] != len(forced[b]):
                        raise GrammarError(f"row {b}: {len(forced[b]) - state['ptr'][b]} recorded choices unused")
            # ---- strings (Qwen + LoRA, bridge prefix)
            hole_rows = [b for b in active if holes_of(templates[b])]
            programs, ctx_ids, prompt_sha, prompt_trunc = {}, {}, {}, {}
            for b in hole_rows:
                programs[b] = self.filler.program(templates[b])
                ctx_ids[b], prompt_trunc[b] = self.filler.context_ids(self._prompt(chosen[b], ev_obs[b]))
                prompt_sha[b] = self.filler.prompt_hash(ctx_ids[b])
                if teacher is not None and teacher[b].get("prompt") and teacher[b]["prompt"] != prompt_sha[b]:
                    raise GrammarError(f"row {b}: decoder prompt differs from the recorded one (different tool "
                                       f"schema or evidence content)")
                if teacher is not None and "len_prior" in teacher[b] and \
                        bool(teacher[b]["len_prior"]) != (self.length_head is not None):
                    raise GrammarError(f"row {b}: the action was sampled with{'' if teacher[b]['len_prior'] else 'out'}"
                                       f" the string length prior; this generator has{'' if self.length_head else ' no'}"
                                       f" length head")
            if hole_rows and self.length_head is not None:
                # per-hole learned stop-logit offsets from the slot's schema embedding, the tool token and the
                # decision summary qt (a function of R, G, B): CAM's state conditions how long strings get
                items = [(b, k) for b in hole_rows for k in range(len(programs[b].holes))]
                emb = self.enc.embed([programs[b].holes[k].node.slot_text[: self.opts.slot_chars] for b, k in items])
                ri = torch.tensor([b for b, _ in items], device=dev)
                off = self.length_head.offsets(emb.to(dev), tool_tok[ri], qt[ri])
                i0 = 0
                for b in hole_rows:
                    n_h = len(programs[b].holes)
                    programs[b].hz_off = off[i0: i0 + n_h]
                    i0 += n_h
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
        trunc = sum(prompt_trunc.values())
        self.last_compute = {"rows": B, "decisions": sum(n_dec), "evidence_rows": len(ev_rows),
                             "hole_rows": len(hole_rows), "tokens": sum(n_tok), "qwen_steps": qwen_steps,
                             "prompt_tokens": sum(len(ctx_ids[b]) for b in hole_rows),
                             "prompt_truncated_tokens": trunc}
        self.stats["decisions"] += sum(n_dec)
        self.stats["qwen_steps"] += qwen_steps
        self.stats["sampled_tokens"] += sum(n_tok)
        if teacher is not None and flags:
            self.stats["demo_numbers"] += flags.get("numbers", 0)
            self.stats["demo_numbers_approx"] += flags.get("approx", 0)
            self.stats["demo_numbers_copied"] += flags.get("copied", 0)
        restricted = None
        if tool_mask is not None:
            restricted = [torch.nonzero(allowed[b]).flatten().tolist() for b in range(B)]
        return {"logp": res, "entropy": ent, "tool_idx": tool_idx, "tools": chosen, "evidence": evidence,
                "allowed": restricted, "tools_sha": tools_sha, "ev": [b in ev_rows for b in range(B)],
                "ev_sha": ev_sha, "choices": choices, "templates": templates, "programs": programs,
                "prompt_sha": prompt_sha, "prompt_trunc": prompt_trunc, "n_dec": n_dec, "n_tok": n_tok,
                "n_copy": n_copy, "active": active, "approx": flags.get("approx", 0)}

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
        G = max(len(toks) for b in rows for toks in groups[b][1])
        gb = torch.zeros(n, M, G, dtype=torch.long)
        gt = torch.zeros(n, M, G, dtype=torch.long)
        gm = torch.zeros(n, M, G, dtype=torch.bool)
        km = [[False] * M for _ in range(n)]
        for j, b in enumerate(rows):
            for m, toks in enumerate(groups[b][1]):
                gb[j, m, : len(toks)] = b
                gt[j, m, : len(toks)] = torch.tensor(toks)
                gm[j, m, : len(toks)] = True
                km[j][m] = True
        gb, gt, gm = gb.to(dev), gt.to(dev), gm.to(dev)
        vals = obs.x[gb, gt].float() * gm.unsqueeze(-1)                          # [n, M, G, D]
        # mean of each observation's tokens (dense masked reduction: deterministic, no atomic adds)
        keys = vals.sum(2) / gm.sum(2).clamp_min(1).unsqueeze(-1).float()
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
                if len(want) > self.cfg.max_evidence:
                    raise GrammarError(f"row {b}: more recorded evidence than max_evidence")
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
                if len(evidence[b]) != len(rec[j]):
                    raise GrammarError(f"row {b}: recorded evidence could not be replayed")

    # ------------------------------------------------------------------ public API
    @torch.no_grad()
    def sample(self, R, goal: TokenSet, belief: TokenSet, tools: TokenSet, obs: TokenSet, tool_lists: list,
               obs_lists: list, temperature: float = 1.0, greedy: bool = False, *, generator=None, tool_mask=None,
               rescore: bool = True, exact_rows: Optional[bool] = None,
               source: ActionSource = ActionSource.CAM_DECODER) -> list:
        """[B] Action with logp / logp_parts / decode record.

        rescore=True (default): `logp` and its parts are the learner-path values (`log_prob_parts` with exact_rows,
        default self.exact_rows), so importance ratios start at exactly 1; the incremental-decoding value is kept as
        logp_parts["sample_logp"]. rescore=False keeps the incremental value as `logp` (diagnostics only: under bf16
        it differs from log_prob by up to ~1 nat per action). Greedy actions: logp = 0.0 (deterministic behaviour),
        logp_parts["greedy"] = True, logp_parts["policy_logp"] = tau-softmax log-probability of the greedy path."""
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
            g = self.grammar(tool)
            rec = {"v": DECODE_VERSION, "tool": out["tool_idx"][b], "name": tool.name, "key": tool.key,
                   "tools": out["tools_sha"][b], "ev": bool(out["ev"][b]), "cands": out["ev_sha"].get(b),
                   "evidence": list(out["evidence"][b]), "choices": list(out["choices"][b]),
                   "holes": [list(h.tokens) for h in (out["programs"][b].holes if b in out["programs"] else [])],
                   "prompt": out["prompt_sha"].get(b), "prompt_truncated": int(out["prompt_trunc"].get(b, 0)),
                   "temperature": float(temperature), "greedy": bool(greedy),
                   "allowed": None if out["allowed"] is None else out["allowed"][b],
                   "reduced": len(g.reductions), "len_prior": self.length_head is not None}
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
                                              "n_copied": out["n_copy"][b], "sample_logp": total,
                                              "rescored": False, "greedy": bool(greedy),
                                              "dropped_keys": mstats.get("dropped_keys", 0), "decode": rec}))
            self.stats["prompt_truncated_tokens"] += rec["prompt_truncated"]
            self.stats["copied_numbers"] += out["n_copy"][b]
        self.stats["actions"] += len(actions)
        compute = dict(self.last_compute)
        if rescore:
            lpr, enr = self.log_prob_parts(R, goal, belief, tools, obs, tool_lists, obs_lists, actions,
                                           exact_rows=exact_rows)
            for b, a in enumerate(actions):
                for k_src, k_dst in (("tool", "tool"), ("evidence", "evidence"), ("heads", "args_heads"),
                                     ("tokens", "args_tokens")):
                    a.logp_parts[k_dst] = float(lpr[k_src][b])
                a.logp_parts["args"] = a.logp_parts["args_heads"] + a.logp_parts["args_tokens"]
                a.logp_parts["entropy"] = float(sum(enr[k][b] for k in enr))
                a.logp = float(sum(lpr[k][b] for k in lpr))
                a.logp_parts["rescored"] = True
                self.max_rescore_drift = max(self.max_rescore_drift, abs(a.logp - a.logp_parts["sample_logp"]))
            self.stats["rescored"] += len(actions)
            compute["rescore"] = dict(self.last_compute)
        if greedy:
            for a in actions:
                a.logp_parts["policy_logp"] = a.logp
                a.logp = 0.0
        self.last_compute = compute
        return actions

    @property
    def demo_number_stats(self) -> dict:
        """Numbers seen in encoded demonstrations: how many, how many had to be rounded (more than float_digits
        fractional digits), how many were encoded as evidence copies, and the rounded fraction."""
        n = self.stats["demo_numbers"]
        return {"numbers": n, "approx": self.stats["demo_numbers_approx"], "copied": self.stats["demo_numbers_copied"],
                "approx_fraction": (self.stats["demo_numbers_approx"] / n) if n else 0.0}

    @staticmethod
    def importance_weight_usable(action: Action) -> bool:
        """True when Action.logp is a stochastic behaviour log-probability recomputed on the learner path."""
        p = action.logp_parts or {}
        return action.logp is not None and bool(p.get("rescored")) and not p.get("greedy")

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
                       exact_rows: Optional[bool] = None, temperature: Optional[float] = None):
        """({tool, evidence, heads, tokens}: [B] each, {same}: entropies [B]); differentiable.

        exact_rows (default self.exact_rows): every row is evaluated alone on its own un-padded tokens (all components),
        so the result is independent of the minibatch composition. temperature: evaluate the recorded paths under
        this temperature instead of the recorded one (e.g. the target policy at 1.0)."""
        teacher = self._teacher(actions, evidence)
        if temperature is not None:
            for t in teacher:
                if t is not None:
                    t["temperature"] = float(temperature)
        exact = self.exact_rows if exact_rows is None else exact_rows
        B = R.shape[0]
        if not exact or B == 1:
            out = self._forward(R, goal, belief, tools, obs, tool_lists, obs_lists, teacher=teacher, sample=False,
                                exact_rows=exact)
            return out["logp"], out["entropy"]
        zero = R.new_zeros((), dtype=torch.float32)
        lps = {k: [zero] * B for k in PART_KEYS}
        ens = {k: [zero] * B for k in PART_KEYS}
        tot: dict = {}
        for b in range(B):
            if teacher[b] is None:
                continue
            o = self._forward(R[b:b + 1], row_tokens(goal, b), row_tokens(belief, b), row_tokens(tools, b),
                              row_tokens(obs, b), [tool_lists[b]], [obs_lists[b]], teacher=[teacher[b]], sample=False,
                              exact_rows=True)
            for k in PART_KEYS:
                lps[k][b] = o["logp"][k][0]
                ens[k][b] = o["entropy"][k][0]
            for k, v in self.last_compute.items():
                tot[k] = tot.get(k, 0) + v
        self.last_compute = tot
        return {k: torch.stack(v) for k, v in lps.items()}, {k: torch.stack(v) for k, v in ens.items()}

    def log_prob(self, R, goal: TokenSet, belief: TokenSet, tools: TokenSet, obs: TokenSet, tool_lists: list,
                 obs_lists: list, actions: list, *, exact_rows: Optional[bool] = None,
                 temperature: Optional[float] = None) -> tuple:
        """ActionGeneratorP.log_prob -> (logp [B], entropy [B]) of the given actions (None rows give 0)."""
        lp, en = self.log_prob_parts(R, goal, belief, tools, obs, tool_lists, obs_lists, actions,
                                     exact_rows=exact_rows, temperature=temperature)
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
