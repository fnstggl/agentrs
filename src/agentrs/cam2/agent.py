"""Acting: CAM chooses the structured action, the frozen LLM writes text arguments under CAM's control vectors.

The behaviour distribution of one decision is the product of
  * (optional) deliberation choice  d ~ softmax(delib_logits(f))           [CAM]
  * structured candidate            k ~ softmax(policy_logits(f, c) + plan_bias)   [CAM; plan_bias from the
                                      world-model planner when d = plan, stored so the learner replays it exactly]
  * text/JSON argument tokens       y ~ prod_t softmax(LLM(ctx, prefix(f, c_k), y_<t) / T)   [frozen LLM, CAM prefix]
and its log-probability is the sum of the three parts. The learner recomputes all three with gradients.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from agentrs.cam2.featurize import StepInput, ToolRegistry, collate
from agentrs.cam2.model import CAM
from agentrs.cam2.planner import PlanConfig, plan_bias
from agentrs.cam2.tools import ToolCall

GEN_MAX = {"text": 24, "json": 96}


def gen_prompt(step: StepInput, spec, el_text: Optional[str], max_feedback: int = 2500) -> str:
    a = spec.gen_arg
    lines = [f"Task: {step.goal}"]
    if step.text:
        lines.append(f"Recent results:\n{step.text[-max_feedback:]}")
    lines.append(f"Chosen action: {spec.name} — {spec.description}")
    if el_text:
        lines.append(f"Target element: {el_text}")
    if a.type == "json":
        lines.append(f"Arguments: {a.description}")
        lines.append("Write only the JSON object of arguments for this call.")
    else:
        lines.append(f"Write only the {a.description or a.name} for this action, nothing else.")
    return "\n".join(lines)


@dataclass
class ActOut:
    cand: torch.Tensor             # (B,) chosen candidate index
    logp_cand: torch.Tensor        # (B,)
    delib: torch.Tensor            # (B,) 0 = act directly, 1 = plan (always 0 without deliberation head)
    logp_delib: torch.Tensor
    bias: Optional[torch.Tensor]   # (B, Nc) planner logit bias (zeros where not planning)
    value: torch.Tensor
    f: torch.Tensor
    state: dict
    gen: list = field(default_factory=list)   # per row: None or dict(ctx, stub, tokens, text, logp)
    logp_text: Optional[torch.Tensor] = None
    calls: list = field(default_factory=list)
    plan_info: dict = field(default_factory=dict)
    delib_used: bool = False


class Agent:
    def __init__(self, llm, cam: CAM, reg: ToolRegistry, device, temperature: float = 1.0,
                 plan: Optional[PlanConfig] = None, use_prefix: bool = True):
        self.llm, self.cam, self.reg, self.device = llm, cam, reg, device
        self.temperature = temperature
        self.plan = plan
        self.use_prefix = use_prefix and cam.cfg.prefix_len > 0

    def gen_ids(self, step: StepInput, cand: tuple) -> tuple[list, list, str]:
        t, e, _ = cand
        spec = self.reg.sets[step.tools_key]["specs"][t]
        el_text = step.el_texts[e] if e >= 0 else None
        ctx, stub = self.llm.chat_ids(gen_prompt(step, spec, el_text))
        return ctx, stub, spec.gen_arg.type

    @torch.no_grad()
    def act(self, steps: list, state: dict, mem_rows: Optional[torch.Tensor], mem_m: Optional[torch.Tensor],
            greedy: bool = False, gen: Optional[torch.Generator] = None) -> ActOut:
        cam = self.cam
        b = collate(self.llm, self.reg, steps, mem_rows, mem_m, self.device, goal_cond=cam.cfg.el_goal_cond)
        f, el_ctx, new_state, _ = cam.integrate(b, state)
        cand = cam.candidates(b, el_ctx)
        logits = cam.policy_logits(f, cand, b["cmask"])
        B = f.shape[0]
        delib = torch.zeros(B, dtype=torch.long, device=self.device)
        logp_d = torch.zeros(B, device=self.device)
        bias = None
        plan_info: dict = {}
        if self.plan is not None and self.plan.enabled:
            if cam.cfg.deliberation:
                dl = torch.log_softmax(cam.delib_logits(f), -1)
                delib = dl.argmax(-1) if greedy else torch.multinomial(dl.exp(), 1, generator=gen).squeeze(-1)
                logp_d = dl.gather(1, delib.unsqueeze(1)).squeeze(1)
                do_plan = delib == 1
            else:
                do_plan = torch.ones(B, dtype=torch.bool, device=self.device)
            bias, plan_info = plan_bias(cam, f, cand, b["cmask"], logits, self.plan)
            bias = bias * do_plan.float().unsqueeze(1)
        lg = logits if bias is None else logits + bias
        lp = torch.log_softmax(lg, -1)
        k = lp.argmax(-1) if greedy else torch.multinomial(lp.exp(), 1, generator=gen).squeeze(-1)
        logp_c = lp.gather(1, k.unsqueeze(1)).squeeze(1)
        value = cam.v(f)
        out = ActOut(cand=k, logp_cand=logp_c, delib=delib, logp_delib=logp_d, bias=bias, value=value, f=f,
                     state=new_state, gen=[None] * B, logp_text=torch.zeros(B, device=self.device), plan_info=plan_info,
                     delib_used=bool(self.plan is not None and self.plan.enabled and cam.cfg.deliberation))
        out.logits = logits
        # ---- text / JSON arguments from the frozen LLM under CAM's control vectors
        rows = [i for i in range(B) if self.reg.sets[steps[i].tools_key]["gen"][steps[i].cands[int(k[i])][0]] is not None]
        for kind in ("text", "json"):
            rr = [i for i in rows if self.reg.sets[steps[i].tools_key]["gen"][steps[i].cands[int(k[i])][0]] == kind]
            if not rr:
                continue
            ctx, stub = [], []
            for i in rr:
                c_, s_, _ = self.gen_ids(steps[i], steps[i].cands[int(k[i])])
                ctx.append(c_)
                stub.append(s_)
            pre = None
            if self.use_prefix:
                idx = torch.tensor(rr, device=self.device)
                pre = cam.prefix(f[idx], cand[idx, k[idx]])
            g = self.llm.sample_with_prefix(pre, ctx, stub, max_new=GEN_MAX[kind], temperature=self.temperature,
                                            greedy=greedy, gen=gen)
            # behaviour log-prob through the same full-sequence computation the learner uses (incremental KV-cache
            # decoding and a full forward differ by bf16 rounding; the stored value must match the learner's path)
            lp = self.llm.logprob_with_prefix(pre, ctx, stub, g.tokens, temperature=self.temperature) if not greedy else g.logp
            pre_lo = None if pre is None else pre.detach().to(self.llm.dtype).cpu()  # what the LLM actually received
            for j, i in enumerate(rr):
                out.gen[i] = {"ctx": ctx[j], "stub": stub[j], "tokens": g.tokens[j], "text": g.texts[j], "kind": kind,
                              "prefix": None if pre_lo is None else pre_lo[j]}
                out.logp_text[i] = lp[j]
        # ---- tool calls
        for i, s in enumerate(steps):
            t, e, en = s.cands[int(k[i])]
            spec = self.reg.sets[s.tools_key]["specs"][t]
            out.calls.append(ToolCall(tool=spec.name, element=s.el_ids[e] if e >= 0 else None, enum=en,
                                      text=out.gen[i]["text"] if out.gen[i] else None))
        return out


def action_text(call: ToolCall, step: StepInput) -> str:
    """Agent-visible description of the action just taken (the next step's ``pact``)."""
    el = ""
    if call.element is not None and call.element in step.el_ids:
        el = " on " + step.el_texts[step.el_ids.index(call.element)]
    extra = f" {call.enum}" if call.enum else ""
    txt = f" with {call.text[:200]!r}" if call.text is not None else ""
    return f"{call.tool}{extra}{el}{txt}"
