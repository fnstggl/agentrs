"""Throughput / memory benchmark of the workstream-B stack at the dev, main and large scales (spec 1.1, A01).

One "step" = EntityEncoder over the observation tokens -> UnifiedCore over [obs; entities; relations; belief; memory;
goal; action] context -> belief tokens -> Reasoner (fixed J, or the default adaptive halting with `--J adaptive`:
bounded per-row unroll, iteration checkpointing in training, compute penalty in the loss) -> scalar loss -> backward
(+ AdamW step). The "core" mode times the backbone alone. Run under the shared GPU lock:

    flock /workspace/agentrs/runs/cam_unified/gpu_heavy.lock python -m superlearner.cam_unified.core.bench \
        --scales dev main large --B 8 --N 384 --J 4 adaptive

Full-model mode (W1, plan 2026-11-26 §2 memory budget / §3 W1): the real `build.build_agent(preset)` system (1B
core + Qwen3-4B for "dom") acting on real Crucible rapid worlds through the canonical loop, then one full optimiser
step over the collected decisions:

    flock ... python -m superlearner.cam_unified.core.bench --full --preset dom --B 16 --K 4 --rounds 8 \
        --microbatch 4 [--share-ema-target] [--adafactor-satellites] [--planned-adapters] --out FILE

Measured: build memory and parameter counts per group; decide time at B (agent.act, real observations and numeric
tools) and at B*K rows (one batched decide of B*K rows: the cost model of K candidates per environment until W5's
candidate decoding exists); one full optimiser step = policy_evaluate fwd+bwd over every collected decision in
microbatches + world-model and adaptation losses on replayed sequences + the optimiser step (AdamW; Adafactor for
the satellites with --adafactor-satellites), its wall time and peak memory, the core hash before and after; optionally
the agent's own (unmicrobatched) update on a reduced buffer. --planned-adapters adds the parameters (weights, grads,
optimiser states) of the DOM adapters not yet wired by their owners (Mechanism C cross-attention in the core every
cfg.sem_xattn_every blocks, Mechanism A decoder cross-attention every cfg.dec_xattn_every decoder layers) so the
memory of the final system is counted; their activations are not. --share-ema-target makes the world model's EMA
target the online core (the plan removes the EMA target; W6 implements it).
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Optional

import torch

from ..config import make_config
from ..interfaces import TokenKind, TokenSet
from .backbone import UnifiedCore
from .bayes import BayesLastLayer
from .beliefs import BeliefModule
from .blocks import count_parameters
from .entities import EntityEncoder
from .reasoning import Reasoner


def build_stack(scale: str, device="cuda", grad_checkpoint: bool = False, bf16: bool = True, **over) -> dict:
    cfg = make_config(scale, grad_checkpoint=grad_checkpoint, **over)
    core = UnifiedCore(cfg, autocast=bf16).to(device)
    stack = {
        "cfg": cfg, "core": core, "reasoner": Reasoner(cfg, core).to(device),
        "entities": EntityEncoder(cfg, autocast=bf16).to(device), "beliefs": BeliefModule(cfg, autocast=bf16).to(device),
        "bayes": BayesLastLayer(cfg).to(device),
    }
    return stack


def parameter_report(stack: dict) -> dict:
    core = stack["core"]
    rep = {"core": core.count_parameters(), "reasoner_own": count_parameters(stack["reasoner"]),
           "entity_encoder": count_parameters(stack["entities"]), "belief_module": count_parameters(stack["beliefs"]),
           "bayes_last_layer": count_parameters(stack["bayes"])}
    rep["workstream_B_total"] = (rep["core"]["total"] + rep["reasoner_own"] + rep["entity_encoder"]
                                 + rep["belief_module"] + rep["bayes_last_layer"])
    return rep


def synthetic_context(cfg, B: int, N: int, device, seed: int = 0) -> tuple:
    """A realistic per-step working context of N tokens: observation tokens with entity bindings (60%), memory,
    goal and action tokens; the entity encoder adds ENTITY/RELATION tokens and the belief K+1 tokens are appended by
    the step, so the observation share is sized to make the total ~N."""
    g = torch.Generator().manual_seed(seed)
    D = cfg.d_model
    n_belief = cfg.n_particles + 1
    n_goal, n_act, n_mem = 4, 4, 32
    n_ent = min(48, cfg.max_entities)
    n_obs = max(8, N - n_belief - n_goal - n_act - n_mem - n_ent - n_ent)

    def ts(n, kind, ent=False):
        x = torch.randn(B, n, D, generator=g)
        e = torch.randint(0, n_ent, (B, n), generator=g) if ent else torch.full((B, n), -1)
        e2 = torch.where(torch.rand(B, n, generator=g) < 0.15, torch.randint(0, n_ent, (B, n), generator=g),
                         torch.full((B, n), -1)) if ent else None
        return TokenSet(x, torch.ones(B, n, dtype=torch.bool), torch.full((B, n), int(kind)),
                        -torch.rand(B, n, generator=g) * 30, e, None, e2).to(device)
    obs = ts(n_obs, TokenKind.OBS_SEM, ent=True)
    rest = TokenSet.cat(ts(n_mem, TokenKind.MEMORY), ts(n_goal, TokenKind.GOAL), ts(n_act, TokenKind.ACTION))
    rest.entity2 = None
    return obs, rest


def _step(stack, obs, rest, Z0, fast, J):
    """One step; J = None runs the reasoner's default (adaptive halting when cfg.adaptive_reasoning)."""
    ents = stack["entities"](obs)
    btok = stack["beliefs"].tokens(fast)
    obs_flat = TokenSet(obs.x, obs.mask, obs.kind, obs.time, obs.entity, None, obs.entity2)
    ctx = TokenSet.cat(obs_flat, ents, btok, rest)
    Z, ctx_out = stack["core"](Z0, ctx)
    R, info = stack["reasoner"](Z, TokenSet.cat(btok, rest), n_iters=J)
    loss = R.float().pow(2).mean() + ctx_out.float().pow(2).mean() * 0.1 + info["ponder"].mean()
    return loss, ctx.N, info


def benchmark(scale: str, B: int = 8, N: int = 384, steps: int = 10, warmup: int = 3, bf16: bool = True,
              grad_checkpoint: bool = False, J: Optional[int] = 4, mode: str = "step", optimizer: bool = True,
              device: str = "cuda", halt_bias: Optional[float] = None) -> dict:
    """Forward and training-step time / throughput / peak memory. J = None: the default adaptive reasoning.
    For adaptive reasoning pass optimizer=False to time the step at fixed parameters (optimizer updates on the
    synthetic loss would retrain the halting depth within the warmup steps); halt_bias (e.g. -20) sets the halting
    head bias to measure the worst case (every row runs max_reason_iters)."""
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    stack = build_stack(scale, device, grad_checkpoint, bf16)
    if halt_bias is not None:
        with torch.no_grad():
            stack["reasoner"].halt_head.bias.fill_(halt_bias)
    cfg, core = stack["cfg"], stack["core"]
    params = parameter_report(stack)
    mods = [stack[k] for k in ("core", "reasoner", "entities", "beliefs")]
    ps = [p for m in mods for p in m.parameters()]
    seen, plist = set(), []
    for p in ps:
        if id(p) not in seen:
            seen.add(id(p))
            plist.append(p)
    opt = torch.optim.AdamW(plist, lr=1e-4, fused=True) if optimizer else None
    obs, rest = synthetic_context(cfg, B, N, device)

    iters_seen = []

    def one(train: bool):
        Z0 = core.init_state(B, device)                      # fresh graph per step (learned initial state)
        fast = stack["beliefs"].init_fast(B, device)
        if mode == "core":
            ctx = TokenSet.cat(TokenSet(obs.x, obs.mask, obs.kind, obs.time, obs.entity, None, obs.entity2), rest)
            Z, co = core(Z0, ctx)
            loss, n_ctx = Z.pow(2).mean() + 0.1 * co.pow(2).mean(), ctx.N
        else:
            loss, n_ctx, info = _step(stack, obs, rest, Z0, fast, J)
            # tensors (no host sync inside the timed loop); read after synchronising
            iters_seen.append((info["n_computed"].detach().float().mean(), info["expected_iters"].detach().mean()))
        if train:
            loss.backward()
            if opt is not None:
                opt.step()
                opt.zero_grad(set_to_none=True)
        return n_ctx

    res = {"scale": scale, "B": B, "N_target": N, "mode": mode, "bf16": bf16, "grad_checkpoint": grad_checkpoint,
           "reason_iters": "adaptive" if J is None else J, "optimizer_step": opt is not None,
           "halt_bias": halt_bias, "params": params}
    for train in (False, True):
        for m in mods:
            m.train(train)
        ctxm = torch.enable_grad() if train else torch.no_grad()
        with ctxm:
            for _ in range(warmup):
                n_ctx = one(train)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for _ in range(steps):
                n_ctx = one(train)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / steps
        key = "train" if train else "forward"
        res[key] = {"ms_per_step": 1000 * dt, "tokens_per_s": B * (cfg.n_state + n_ctx) / dt,
                    "peak_mem_GB": (torch.cuda.max_memory_allocated() - base_mem) / 2 ** 30}
        if iters_seen:
            res[key]["reason_iters_computed"] = float(sum(a for a, _ in iters_seen[-steps:])) / steps
            res[key]["reason_iters_expected"] = float(sum(b for _, b in iters_seen[-steps:])) / steps
            iters_seen.clear()
        res["ctx_tokens"] = n_ctx
    del stack, opt
    torch.cuda.empty_cache()
    return res


# ============================================================================ full-model mode (real system)

GB = 2 ** 30


def _mem() -> dict:
    return {"allocated_GB": torch.cuda.memory_allocated() / GB, "reserved_GB": torch.cuda.memory_reserved() / GB,
            "peak_GB": torch.cuda.max_memory_allocated() / GB}


def rapid_lever_specs(n: int, seed0: int = 0, split: str = "train") -> list:
    """`n` WorldSpecs of rapid families of `split` whose only action kind is numeric levers (numeric tools), in
    registry order, cycled with fresh seeds when there are fewer families than n."""
    import crucible
    crucible.load_all_modules()
    from crucible.rapid import families as F
    from ..envs.experience import BROKEN_MULTIAGENT_RAPID, WorldSpec
    fams = [k for k, v in F.REGISTRY.items() if v.get("split") == split and list(v.get("action_kinds") or []) ==
            ["levers"] and not v.get("participants") and k not in BROKEN_MULTIAGENT_RAPID]
    if not fams:
        raise RuntimeError(f"no rapid lever families in split {split!r}")
    return [WorldSpec(1, f"rapid:{fams[i % len(fams)]}", seed=seed0 + i) for i in range(n)]


class _PlannedAdapters(torch.nn.Module):
    """Parameters of DOM adapters that their owners have not wired yet, at the preset's sizes: Mechanism C (gated
    semantic cross-attention after every cfg.sem_xattn_every core blocks, W3) and Mechanism A (latent cross-attention
    in every cfg.dec_xattn_every-th decoder layer, W4). Only for memory accounting: they get zero gradients so the
    optimiser allocates their states; they compute nothing."""

    def __init__(self, cfg, sem, core, wired_a: bool = False):
        super().__init__()
        from ..semantic.decoder_xattn import LatentDecoderConditioning
        from .semantic_xattn import GatedSemanticXAttn, SemanticProjection
        self.report = {}
        have_c = bool(getattr(core, "xattn", None)) and len(core.xattn) > 0
        every = int(getattr(cfg, "sem_xattn_every", 0) or 0)
        sdim = int(getattr(cfg, "sem_xattn_dim", 0) or 0)
        if every and sdim and not have_c:
            layers = [i for i in range(cfg.n_layers) if (i + 1) % every == 0]
            self.mech_c = torch.nn.ModuleDict({str(i): GatedSemanticXAttn(cfg.d_model, cfg.n_heads)
                                               for i in layers})
            self.mech_c_proj = SemanticProjection(sdim, cfg.d_model)
            self.report["mech_c"] = {"layers": layers, "params": count_parameters(self.mech_c)
                                     + count_parameters(self.mech_c_proj)}
        de = int(getattr(cfg, "dec_xattn_every", 0) or 0)
        n_lm = int(getattr(sem, "n_layers", 0) or 0)
        d_lm = int(getattr(sem, "sem_dim", 0) or sdim)
        if de and n_lm and not wired_a:
            layers = [i for i in range(n_lm) if (i + 1) % de == 0]
            self.mech_a = LatentDecoderConditioning(d_lm, cfg.d_model, layers)
            self.report["mech_a"] = {"layers": layers, "params": count_parameters(self.mech_a)}
        self.report["params"] = count_parameters(self)


def _optimizers(system, extra: Optional[torch.nn.Module], adafactor_satellites: bool, lr: float = 1e-5) -> list:
    """The trainer's AdamW over every trained group (+ the planned adapters), or AdamW for the core and Adafactor for
    every other trainable parameter."""
    tr = system.trainer
    extra_ps = [p for p in extra.parameters()] if extra is not None else []
    if not adafactor_satellites:
        opts = [tr.opt]
        if extra_ps:
            opts.append(torch.optim.AdamW(extra_ps, lr=lr))
        return opts
    core = list(tr.groups["core"])
    sat = list(tr.groups["heads"]) + list(tr.groups["semantic"]) + extra_ps
    if not hasattr(torch.optim, "Adafactor"):
        raise RuntimeError("torch.optim.Adafactor needs torch >= 2.5")
    return [torch.optim.AdamW(core, lr=lr, fused=True), torch.optim.Adafactor(sat, lr=lr)]


def _opt_state_GB(opts) -> float:
    n = 0
    for o in opts:
        for st in o.state.values():
            for v in st.values():
                if torch.is_tensor(v):
                    n += v.numel() * v.element_size()
    return n / GB


def full_optimiser_step(system, *, microbatch: int = 4, world: bool = True, adapt: bool = True,
                        extra: Optional[torch.nn.Module] = None, opts: Optional[list] = None) -> dict:
    """One full optimiser step over the agent's fresh buffer (every collected decision): policy_evaluate fwd + bwd in
    microbatches of `microbatch` decisions (surrogate -logp - c*entropy + value^2, the graph of the policy objective),
    the world-model loss and the support/query adaptation loss on replayed sequences, zero gradients for trainable
    parameters the step did not reach (so the optimiser state of the whole model is allocated), clip, step."""
    from ..model import StepInputs
    from ..stage1.harness_agent import core_hash
    agent, model = system.agent, system.model
    opts = opts or [system.trainer.opt]
    rows = [(st, j) for st in agent.fresh for j in range(len(st.rows))]
    N = len(rows)
    out = {"decisions": N, "microbatch": microbatch}
    h0 = core_hash(model)
    for o in opts:
        o.zero_grad(set_to_none=True)
    model.train()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    tp = 0.0
    for i in range(0, N, microbatch):
        chunk = rows[i:i + microbatch]
        subs = [st.subset([j]) for st, j in chunk]
        inp = StepInputs.cat([s.inputs for s in subs])
        acts = [a for s in subs for a in s.actions]
        r = model.policy_evaluate(inp, acts)
        loss = (-r["logp"].float().sum() - 1e-3 * r["entropy"].float().sum()
                + 0.5 * r["value"].float().pow(2).sum()) / max(N, 1)
        loss.backward()
    torch.cuda.synchronize()
    tp = time.perf_counter() - t0
    out["policy_s"] = tp
    out["policy_peak_GB"] = torch.cuda.max_memory_allocated() / GB
    tw = time.perf_counter()
    if world or adapt:
        if hasattr(agent.replay, "sync"):
            agent.replay.sync()
    if world:
        seqs = agent._sample(agent.n_seqs, agent.seq_len)
        if seqs:
            seq, aux = agent.tokenise_sequences(seqs)
            wl = model.world_loss(seq, model.filter_sequence(seq, aux))["world"]
            wl.backward()
            out["world_loss"] = float(wl.detach())
            out["world_T"] = len(seq.steps)
    if adapt and model.cfg.use_fast_adaptation:
        aseqs = agent._sample(agent.adapt_n_seqs, agent.adapt_seq_len)
        if aseqs:
            aseq, aaux = agent.tokenise_sequences(aseqs)
            k = min(4, len(aseq.steps) - 1)
            al, _ = model.adapt_loss(aseq, model.filter_sequence(aseq, aaux), k)
            if al.requires_grad:
                al.backward()
                out["adapt_loss"] = float(al.detach())
    torch.cuda.synchronize()
    out["world_adapt_s"] = time.perf_counter() - tw
    ps = [p for o in opts for g in o.param_groups for p in g["params"]]
    n_zero = 0
    for p in ps:
        if p.grad is None and p.requires_grad:
            p.grad = torch.zeros_like(p)
            n_zero += p.numel()
    out["params_without_grad"] = n_zero
    gn = torch.nn.utils.clip_grad_norm_(ps, 1.0)
    to = time.perf_counter()
    for o in opts:
        o.step()
    torch.cuda.synchronize()
    out["optimizer_s"] = time.perf_counter() - to
    out["step_s"] = time.perf_counter() - t0
    out["grad_norm"] = float(gn)
    out["peak_GB"] = torch.cuda.max_memory_allocated() / GB
    out["optimizer_state_GB"] = _opt_state_GB(opts)
    for o in opts:
        o.zero_grad(set_to_none=True)
    model.eval()
    h1 = core_hash(model)
    out["core_hash_before"], out["core_hash_after"], out["core_hash_changed"] = h0, h1, h0 != h1
    return out


def core_hash_of(model) -> str:
    from ..stage1.harness_agent import core_hash
    return core_hash(model)


PARTIAL_OUT: Optional[str] = None             # benchmark_full writes its results so far here after every phase


class PhaseTimer:
    """Wall time (CUDA-synchronised) of the sub-steps of CamUnifiedModel.decide, by monkeypatching the component
    methods while active: encode_step (tokenizer + core + reasoner), tokenizer.tools, action_generator.sample (Qwen
    decoding), planner.plan, world_model.prior, predict_outcomes, value_head."""

    TARGETS = (("model", "encode_step"), ("tokenizer", "tools"), ("action_generator", "sample"), ("planner", "plan"),
               ("world_model", "prior"), ("model", "predict_outcomes"), ("value_head", "forward"))

    def __init__(self, model):
        self.model = model
        self.t: dict = {}
        self._saved: list = []

    def __enter__(self):
        for owner, name in self.TARGETS:
            obj = self.model if owner == "model" else getattr(self.model, owner, None)
            if obj is None or not hasattr(obj, name):
                continue
            fn = getattr(obj, name)
            key = f"{owner}.{name}"

            def wrap(*a, _fn=fn, _key=key, **k):
                torch.cuda.synchronize()
                t = time.perf_counter()
                out = _fn(*a, **k)
                torch.cuda.synchronize()
                self.t[_key] = self.t.get(_key, 0.0) + time.perf_counter() - t
                return out
            self._saved.append((obj, name, obj.__dict__.get(name)))
            setattr(obj, name, wrap)
        return self

    def __exit__(self, *exc):
        for obj, name, prev in reversed(self._saved):
            if prev is None:
                delattr(obj, name)
            else:
                setattr(obj, name, prev)
        self._saved = []


def wait_free(need_GB: float, max_wait_s: float, own_GB: float = 0.0) -> dict:
    """Wait (polling every 5 s, at most max_wait_s) until the GPU has `need_GB` free beyond what this process holds
    already counted in `own_GB`; -> {"free_GB", "waited_s", "ok"} (the GPU is shared; other jobs' memory is not ours)."""
    t0 = time.perf_counter()
    while True:
        free = torch.cuda.mem_get_info()[0] / GB
        if free >= need_GB - own_GB or time.perf_counter() - t0 >= max_wait_s:
            return {"free_GB": free, "waited_s": time.perf_counter() - t0, "ok": free >= need_GB - own_GB}
        time.sleep(5)


def wait_free_gpu(need_GB: float, timeout_s: float = 0.0) -> dict:
    """wait_free for callers outside the bench (tests): {"free_GB", "waited_s", "ok"} after at most timeout_s."""
    return wait_free(need_GB, timeout_s)


def _partial(res: dict) -> None:
    if PARTIAL_OUT:
        with open(PARTIAL_OUT + ".partial", "w") as fh:
            json.dump(res, fh, indent=1, default=str)


def benchmark_full(preset: str = "dom", *, B: int = 16, K: int = 4, rounds: int = 8, microbatch: int = 4,
                   decide_reps: int = 3, overrides: Optional[dict] = None, share_ema_target: bool = False,
                   adafactor_satellites: bool = False, planned_adapters: bool = False, grad_checkpoint: bool = True,
                   agent_update_rows: int = 0, run_dir: Optional[str] = None, sem_model: Optional[str] = None,
                   seed: int = 0, ow_envs: int = 0, variants: Optional[list] = None,
                   min_free_GB: float = 0.0, max_wait_s: float = 0.0, bk_cost_model: bool = False,
                   learner_variants: Optional[list] = None) -> dict:
    """The real system at `preset` (see the module docstring). `ow_envs` > 0 adds that many ow_tiny OpenWorld worlds
    to the B rapid worlds (one ConcatEnv batch)."""
    import tempfile
    from ..build import ConcatLabels, OperatorLabels, SystemConfig, build_agent, describe
    from ..envs.experience import ExperienceOperator
    from ..runtime import ConcatEnv, GuardedEnv, run_lifetime
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    over = {"grad_checkpoint": bool(grad_checkpoint), **dict(overrides or {})}
    run_dir = run_dir or tempfile.mkdtemp(prefix=f"bench_full_{preset}_")
    res: dict = {"mode": "full", "preset": preset, "B": B, "K": K, "rounds": rounds, "microbatch": microbatch,
                 "overrides": over, "share_ema_target": share_ema_target, "planned_adapters": planned_adapters,
                 "adafactor_satellites": adafactor_satellites, "ow_envs": ow_envs, "run_dir": run_dir,
                 "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    res["gpu_free_before_build"] = wait_free(min_free_GB, max_wait_s) if min_free_GB else \
        {"free_GB": torch.cuda.mem_get_info()[0] / GB}
    t0 = time.perf_counter()
    sysm = build_agent(preset, env_kind="mixed" if ow_envs else "rapid", device="cuda", run_dir=run_dir,
                       overrides=over, sem_model=sem_model, seed=seed,
                       system=SystemConfig(deterministic=False, grad_probe_every=0, archive_autoflush=False,
                                           fresh_max_steps=rounds + 2, checkpoint_semantic_cache=False))
    res["build_s"] = time.perf_counter() - t0
    model, agent = sysm.model, sysm.agent
    if share_ema_target and getattr(model.world_model, "target_core", None) is not None:
        del model.world_model.target_core
        model.world_model.target_core = model.world_model.core
        model.world_model.update_target = lambda *a, **k: None
        torch.cuda.empty_cache()
    wired_a = getattr(model.action_generator, "dec_cond", None) is not None      # W4 x W5 merge fix d564665
    extra = _PlannedAdapters(sysm.cfg, sysm.semantic, model.core, wired_a).cuda() if planned_adapters else None
    res["build_mem"] = {k: v - (base / GB if k == "allocated_GB" else 0.0) for k, v in _mem().items()}
    _partial(res)
    d = describe(sysm)
    res["parameters"] = d["parameters"]
    res["trainer_groups"] = d["trainer"]["groups"]
    res["planned_adapters_report"] = extra.report if extra is not None else None
    tcore = getattr(model.world_model, "target_core", None)
    res["ema_target_params"] = sum(p.numel() for p in tcore.parameters()) \
        if tcore is not None and tcore is not model.world_model.core else 0
    res["cfg"] = {k: getattr(sysm.cfg, k, None) for k in (
        "n_layers", "d_model", "n_heads", "n_state", "d_satellite", "n_intent", "n_candidates", "wm_ema_target",
        "microbatch", "ppo_epochs", "sem_xattn_every", "dec_xattn_every", "reward_mode", "update_every")}
    res["semantic_model"] = d["semantic_model"]
    # ---------------- acting through the canonical loop (real rapid worlds, numeric tools)
    op = ExperienceOperator(rapid_lever_specs(B, seed0=1000 + seed))
    parts, srcs = [op.adapter()], [OperatorLabels(op)]
    ow_ad = None
    if ow_envs:
        from crucible.research import llm_eval as L
        from ..envs.openworld import open_world
        goal = L.open_economy_goal(world_family="ow_tiny", seed=1, overrides={"n_consumers": 800}, days=8,
                                   window_days=3, revenue_cents=10_000, requests=100, sustain_checks=2,
                                   endowment=3_000_000)
        ow_ad, ow_op = open_world([goal] * ow_envs, seed=seed)
        parts.append(ow_ad)
    env = GuardedEnv(ConcatEnv(parts) if len(parts) > 1 else parts[0])
    act_s: list = []
    act0 = agent.act

    def timed_act(obs, tools):
        torch.cuda.synchronize()
        t = time.perf_counter()
        a = act0(obs, tools)
        torch.cuda.synchronize()
        act_s.append(time.perf_counter() - t)
        agent._bench_last_actions = list(a) if isinstance(a, (list, tuple)) else list(getattr(a, "actions", []) or [])
        return a
    dec_timings: list = []
    _dec0 = sysm.model.decide

    def _timed_decide(*a_, **k_):
        d_ = _dec0(*a_, **k_)
        dec_timings.append({k: float(v) for k, v in (getattr(d_, "timings", None) or {}).items()
                            if isinstance(v, (int, float))})
        return d_
    sysm.model.decide = _timed_decide
    agent.act = timed_act
    torch.cuda.reset_peak_memory_stats()
    try:
        life = run_lifetime(agent, env, labels=ConcatLabels(srcs), label_store=sysm.label_store, learner=sysm.learner,
                            max_decisions=rounds, update_every=0, require_cam_eval=None)   # config-driven gate (verifier DOM-4)
    finally:
        agent.act = act0
        sysm.model.__dict__.pop("decide", None)
    res["acting"] = {"decide_s": act_s, "decide_s_warm_median": sorted(act_s[1:])[len(act_s[1:]) // 2]
                     if len(act_s) > 1 else None, "peak_GB": torch.cuda.max_memory_allocated() / GB,
                     "decisions": life.compute["decisions"], "rounds": life.steps, "reward_mode": life.reward_mode,
                     "labels_appended": life.labels_appended, "authorisation": life.authorisation,
                     "tools_per_env": [len(t) for t in env.available_tools()][:4],
                     "versions": life.versions, "phase_wall_s": life.accounting["phase_wall_s"],
                     "env_exceptions": list(env.errors), "decide_phase_s": dec_timings}
    _partial(res)
    last = [a for a in (getattr(agent, "_bench_last_actions", None) or []) if a is not None]
    res["acting"]["n_candidates_cfg"] = int(getattr(sysm.cfg, "n_candidates", 1) or 1)
    from ..decision.select import stamp_of
    res["acting"]["n_candidates_actions"] = sorted({(stamp_of(a) or {}).get("n") for a in last}, key=str)
    res["acting"]["actions_last_round"] = len(last)
    # ---------------- B*K rows in one decide (the pre-W5 cost model of K candidates per environment; with
    # cfg.n_candidates = K the acting decide above already decodes K candidates per row sharing one encode)
    if K and K > 1 and (bk_cost_model or int(getattr(sysm.cfg, "n_candidates", 1) or 1) <= 1):
        obs = env.observe()
        tools = env.available_tools()
        goals = env.goals()
        nb = len(obs.env_ids)
        rowsK = nb * K
        st = model.init_state(rowsK, device=agent.device, seed=seed)
        times = []
        torch.cuda.reset_peak_memory_stats()
        phases = {}
        for r in range(decide_reps + 1):
            st = model.init_state(rowsK, device=agent.device, seed=seed + r)
            torch.cuda.synchronize()
            t = time.perf_counter()
            with PhaseTimer(model) as pt:
                model.decide(st, [list(o) for o in obs.observations] * K, [float(x) for x in obs.t] * K, tools * K,
                             list(goals) * K, generator=torch.Generator().manual_seed(r))
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
            phases = pt.t
        st = model.init_state(nb, device=agent.device, seed=seed)
        with PhaseTimer(model) as pt:
            torch.cuda.synchronize()
            t = time.perf_counter()
            model.decide(st, [list(o) for o in obs.observations], [float(x) for x in obs.t], tools, list(goals),
                         generator=torch.Generator().manual_seed(0))
            torch.cuda.synchronize()
        res["decide_B_phases"] = {"rows": nb, "decide_s": time.perf_counter() - t, "phases_s": pt.t}
        res["decide_BK"] = {"rows": rowsK, "decide_s": times, "phases_s_last": phases, "decide_s_warm_median": sorted(times[1:])[len(times[1:]) // 2],
                            "peak_GB": torch.cuda.max_memory_allocated() / GB,
                            "note": "one batched decide of B*K rows; K-candidate decoding (W5) not merged"}
        _partial(res)
    # ---------------- one full optimiser step per variant (optimiser states freed between variants)
    res["steps"] = {}
    sur = [] if variants == ["none"] else (variants or ["base"])
    for var in sur:
        parts_v = set(var.split("+"))
        if "share_ema" in parts_v and getattr(model.world_model, "target_core", None) is not None \
                and model.world_model.target_core is not model.world_model.core:
            del model.world_model.target_core
            model.world_model.target_core = model.world_model.core
            model.world_model.update_target = lambda *a, **k: None
        mb = microbatch
        for q in parts_v:
            if q.startswith("mb"):
                mb = int(q[2:])
        sysm.trainer.opt.state.clear()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        own = torch.cuda.memory_reserved() / GB
        gate = wait_free(min_free_GB, max_wait_s, own) if min_free_GB else {"free_GB": torch.cuda.mem_get_info()[0] / GB}
        opts = _optimizers(sysm, extra, adafactor_satellites or "adafactor" in parts_v)
        try:
            r = full_optimiser_step(sysm, microbatch=mb, extra=extra, opts=opts)
        except torch.OutOfMemoryError as e:
            r = {"error": "OutOfMemoryError", "detail": str(e)[:400], "peak_GB": torch.cuda.max_memory_allocated() / GB}
            for o in opts:
                o.zero_grad(set_to_none=True)
            model.eval()
        r["static_after_GB"] = torch.cuda.memory_allocated() / GB
        r["gpu_free_at_start"] = gate
        res["steps"][var] = r
        _partial(res)
        for o in opts:
            o.state.clear()
        del opts
        torch.cuda.empty_cache()
    res["step"] = res["steps"][sur[-1]] if sur else None
    res["after_step_mem"] = _mem()
    # ---------------- the system's real update (IntegratedLearner.step: replay world/adapt losses, policy recompute
    # in cfg.microbatch microbatches, PPO epochs, the named-group probe, role optimisers) on the last
    # `agent_update_rows` decisions; one measurement per learner variant ('+'-joined: base, mb<k>, adafactor),
    # each from the same fresh-buffer snapshot with fresh optimiser moments
    if agent_update_rows:
        keep, n = [], 0
        for stp in reversed(agent.fresh):
            if n >= agent_update_rows:
                break
            keep.append(stp)
            n += len(stp.rows)
        snap = list(reversed(keep))
        res["agent_updates"] = {}
        lr = sysm.learner
        cfg0, tcfg0 = lr.cfg, lr.trainer.cfg
        wm = model.world_model
        ols0 = getattr(wm, "open_loop_starts", None)
        from dataclasses import replace as _replace
        import gc
        for var in (learner_variants or ["base"]):
            parts_v = set(var.split("+"))
            mb = next((int(q[2:]) for q in parts_v if q.startswith("mb")), None)
            ols = next((int(q[3:]) for q in parts_v if q.startswith("ols")), None)
            nseq = next((int(q[3:]) for q in parts_v if q.startswith("seq")), None)
            nseq0 = agent.n_seqs
            if nseq is not None:
                agent.n_seqs = nseq
            sat_mode = "adafactor_all" if "adafactorall" in parts_v else ("adafactor" if "adafactor" in parts_v
                                                                           else None)
            lr.cfg = cfg0 if mb is None else _replace(cfg0, microbatch=mb)
            lr.trainer.cfg = _replace(tcfg0, sat_optimizer=sat_mode) if sat_mode else tcfg0
            if ols is not None:
                wm.open_loop_starts = ols
            lr.trainer.reset_optimizer()
            runs = []
            for rep in range(2 if "once" not in parts_v else 1):   # 2nd update: optimiser moments resident
                agent.fresh = list(snap)
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                alloc0 = torch.cuda.memory_allocated() / GB
                h0 = core_hash_of(model)
                t = time.perf_counter()
                try:
                    u = lr.step()
                    torch.cuda.synchronize()
                    mbs_ = u.get("microbatches")
                    r = {"rows": n, "microbatch": int(getattr(lr.cfg, "microbatch", 0) or 0),
                         "updated": bool(u.get("updated")), "s": time.perf_counter() - t,
                         "peak_GB": torch.cuda.max_memory_allocated() / GB,
                         "optimizer_state_GB": _opt_state_GB([o for o in lr.trainer.opts.values() if o is not None]),
                         "optimizer": [type(o).__name__ for o in lr.trainer.opts.values() if o is not None],
                         "losses": u.get("losses"), "reason": u.get("reason"),
                         "decisions_recomputed": sum(mbs_) if mbs_ else None, "microbatches": mbs_,
                         "epochs": u.get("epochs"), "stop": u.get("stop"), "probe": u.get("probe"),
                         "core_share": u.get("core_share"), "core_share_measured": u.get("core_share_measured"),
                         "share_guard": u.get("share_guard"), "dom08": u.get("dom08"),
                         "grad_absent": (u.get("grad_table") or {}).get("absent"),
                         "hook_multi": (u.get("grad_table") or {}).get("hook_multi")}
                except torch.OutOfMemoryError as e:
                    r = {"rows": n, "microbatch": int(getattr(lr.cfg, "microbatch", 0) or 0),
                         "error": "OutOfMemoryError", "detail": str(e)[:300], "s": time.perf_counter() - t,
                         "peak_GB": torch.cuda.max_memory_allocated() / GB}
                    for p_ in model.parameters():
                        p_.grad = None
                    model.eval()
                r["open_loop_starts"] = getattr(wm, "open_loop_starts", None)
                r["n_seqs"] = agent.n_seqs
                r["alloc_start_GB"] = alloc0
                r["core_hash_changed"] = h0 != core_hash_of(model)
                runs.append(r)
                if "error" in r:
                    break
            res["agent_updates"][var] = {"first": runs[0], "second": runs[1] if len(runs) > 1 else None,
                                         "peak_GB": max(x["peak_GB"] for x in runs),
                                         "s_max": max(x["s"] for x in runs),
                                         "error": next((x.get("error") for x in runs if x.get("error")), None)}
            res["agent_update"] = res["agent_updates"][var]
            _partial(res)
            lr.cfg, lr.trainer.cfg = cfg0, tcfg0
            wm.open_loop_starts = ols0
            agent.n_seqs = nseq0
            lr.trainer.reset_optimizer()
            gc.collect()
            torch.cuda.empty_cache()
    res["overall_peak_GB"] = max(v.get("peak_GB", 0.0) if isinstance(v, dict) else 0.0
                                 for v in (res.get("acting"), res.get("decide_BK"), *res["steps"].values(),
                                           *(res.get("agent_updates") or {}).values()) if v is not None)
    if ow_ad is not None:
        ow_ad.close()
    return res


def main(argv: Optional[list] = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", nargs="+", default=["dev", "main", "large"])
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--N", type=int, default=384)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--J", nargs="+", default=["4"],
                    help="reasoning iterations per run; 'adaptive' = default halting at the initial parameters (no "
                         "optimizer step), 'adaptive_max' = adaptive with every row running max_reason_iters")
    ap.add_argument("--mode", default="step", choices=["step", "core"])
    ap.add_argument("--checkpoint", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--out", default=None)
    # full-model mode
    ap.add_argument("--full", action="store_true", help="the real build_agent system (see module docstring)")
    ap.add_argument("--preset", default="dom")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--microbatch", type=int, default=4)
    ap.add_argument("--ow-envs", type=int, default=0)
    ap.add_argument("--share-ema-target", action="store_true")
    ap.add_argument("--adafactor-satellites", action="store_true")
    ap.add_argument("--planned-adapters", action="store_true")
    ap.add_argument("--agent-update-rows", type=int, default=0)
    ap.add_argument("--override", action="append", default=[], help="CamConfig field=value (JSON value)")
    ap.add_argument("--sem-model", default=None)
    ap.add_argument("--min-free-gb", type=float, default=0.0, help="wait until this much GPU memory is free")
    ap.add_argument("--max-wait-s", type=float, default=0.0)
    ap.add_argument("--decide-reps", type=int, default=3)
    ap.add_argument("--learner-variants", nargs="*", default=None,
                    help="real IntegratedLearner.step variants on --agent-update-rows decisions, '+'-joined: base, "
                         "mb<k> (cfg.microbatch), adafactor (AdamW core + Adafactor satellites)")
    ap.add_argument("--bk-cost-model", action="store_true",
                    help="also time one batched decide of B*K rows (the pre-W5 cost model)")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="optimiser-step variants in order, '+'-joined flags: base, share_ema, adafactor, mb<k>")
    a = ap.parse_args(argv)
    if a.full:
        global PARTIAL_OUT
        PARTIAL_OUT = a.out
        over = {}
        for kv in a.override:
            k, v = kv.split("=", 1)
            try:
                over[k] = json.loads(v)
            except ValueError:
                over[k] = v
        r = benchmark_full(a.preset, B=a.B, K=a.K, rounds=a.rounds, microbatch=a.microbatch, overrides=over,
                           share_ema_target=a.share_ema_target, adafactor_satellites=a.adafactor_satellites,
                           planned_adapters=a.planned_adapters, agent_update_rows=a.agent_update_rows,
                           sem_model=a.sem_model, ow_envs=a.ow_envs, grad_checkpoint=a.checkpoint != "off",
                           variants=a.variants, min_free_GB=a.min_free_gb, max_wait_s=a.max_wait_s,
                           decide_reps=a.decide_reps, bk_cost_model=a.bk_cost_model,
                           learner_variants=a.learner_variants)
        txt = json.dumps(r, indent=1, default=str)
        print(txt, flush=True)
        if a.out:
            with open(a.out, "w") as fh:
                fh.write(txt)
        return r
    out = []
    for s in a.scales:
        ck = {"on": True, "off": False}.get(a.checkpoint, s == "large")
        for j in a.J:
            J = None if j.startswith("adaptive") else int(j)
            r = benchmark(s, a.B, a.N, a.steps, bf16=not a.fp32, grad_checkpoint=ck, J=J, mode=a.mode,
                          optimizer=J is not None, halt_bias=-20.0 if j == "adaptive_max" else None)
            print(json.dumps({k: v for k, v in r.items() if k != "params"}), flush=True)
            out.append(r)
        print(json.dumps({"scale": s, "params": out[-1]["params"]}), flush=True)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    main()
