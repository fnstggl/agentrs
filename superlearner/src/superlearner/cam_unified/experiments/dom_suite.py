"""DOM-11 strategic-dominance conditions A-I (CAM_UNIFIED_SPEC_DOM.md §6; plan 2026-11-26 §3 W9 and §5 E0/E1/E2),
run ONLY through the canonical loop ``runtime.run_lifetime`` in reward_mode observation_only, one provenance row per
(condition, seed) and the agent's TraceRecorder on.

    python -m superlearner.cam_unified.experiments.dom_suite run --condition G --smoke --run-dir DIR
    python -m superlearner.cam_unified.experiments.dom_suite run --condition C --ckpt W.safetensors --seed 1 ...
    python -m superlearner.cam_unified.experiments.dom_suite train --run-dir DIR --rounds N      # source lifetime
    python -m superlearner.cam_unified.experiments.dom_suite ckpt-e2e --run-dir DIR --updates 200  # W9 e2e check

Conditions (DOM spec §6; how each is realised here, all on the same builder, families, budgets and seeds):

  A  original pretrained semantic model: envs.baselines.FrozenSemanticAgent (the preset's semantic model, same
     tools) as the acting agent, no CAM; token-matched budgets are the caller's responsibility (tokens recorded).
  B  untrained CAM: the freshly built model, no learning during the evaluation lifetime.
  C  continually trained CAM: trained weights (weights-only load), learning ON during the lifetime.
  D  reset CAM strategic weights: trained weights, then the strategic core (STRATEGIC_PREFIXES: core, reasoner,
     intent / decision / value / option / exploration heads) re-loaded from the INITIAL checkpoint of the same
     source run; the semantic side (tokenizer, action generator, bridge, adapters) keeps its trained weights.
  E  remove historical memory: trained weights only, the source lifetime's archive / replay / memory never attached
     (use_memory is off in every DOM run, declared; E therefore equals weights-only evaluation with a fresh archive).
  F  decoder conditioning disabled: the latent inputs of the decoder conditioning (the bridge's R; decoder latent
     cross-attention when the model exposes ``set_decoder_conditioning``) are replaced by zeros.
  G  latent-state intervention: observations held fixed, each row's latent state Z replaced by another row's
     (a derangement of the batch) on probe decisions; the executed action is the unswapped one, the swapped
     decision is computed on a snapshot and discarded. Reports action-change and donor-agreement rates.
  H  frozen CAM core: learning ON, the core's learning rate 0 (no permanent core-weight update; heads, adapters and
     fast adaptation learn); the core digest is checked unchanged afterwards.
  I  semantic-only continual learning: learning ON with every learning rate 0 except the semantic group.

Provenance row (``<run>/eval/provenance.jsonl``; rendered by observe.dashboard "Controlled experiments"):
experiment, condition, seed, set, smoke, run_id, t_wall, code_commit / code_dirty, loop (which canonical loop ran),
builder, preset, cfg_hash, arch_hash, ckpt / ckpt_sha256 / init_ckpt_sha256, reward_mode, semantic_model, params,
families, n_envs, decisions, env_steps, updates, gpu_busy_s, gpu_span_s, semantic_tokens, wall_s, peak_gpu_mb,
integrity (run-directory scan), metrics, condition_detail, deviations. Evaluation rows (``eval/<condition>.jsonl``):
{"metric", "value", "version", "family", "seed", "set", "condition", "t"} with verified utilities computed on the
operator side only (never shown to the agent).

Canonical loop and factory: the plan's ``runtime.run_lifetime`` and ``build.build_agent`` (owner W1) are used when
present. Until W1's merge they do not exist at this commit; the driver then runs the existing loop in runtime.py
(``runtime.run_steps``, with NO evaluator, so no environment reward reaches the agent) and builds the stack from
``stage1.system`` helpers with memory off. Both fallbacks are recorded in the row (``loop`` / ``builder``) and listed
in ``deviations``; no act-execute-advance loop is written here.

Confirm-set runs are refused unless the experiment's pre-registration (docs/experiments/U-DOM-1.md) is committed and
unmodified in the working tree, and training families come from the train split only (val / test for confirm).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Sequence

EXPERIMENT_ID = "U-DOM-1"
PREREG = "superlearner/docs/experiments/U-DOM-1.md"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))

CONDITIONS = {
    "A": "original pretrained semantic model (no CAM)",
    "B": "untrained CAM",
    "C": "continually trained CAM",
    "D": "reset CAM strategic weights",
    "E": "remove historical memory",
    "F": "disable decoder conditioning",
    "G": "latent-state intervention",
    "H": "frozen CAM core",
    "I": "semantic-only continual learning",
}
LEARNING = {"C": True, "H": True, "I": True}               # conditions that learn during the evaluated lifetime
NEEDS_TRAINED = {"C", "D", "E", "F", "G", "H", "I"}
STRATEGIC_PREFIXES = ("core.", "reasoner.", "intent_head.", "decision.", "evaluator.", "value_head.",
                      "option_policy.", "exploration_head.", "entity_encoder.")
SMOKE_FAMILIES = ("mk_elastic", "hc_confounded")
DEV_SEED0, CONFIRM_SEED0 = 100_000, 500_000                 # plan §5: confirm seeds >= 500000


# ============================================================================================ provenance helpers
def code_state() -> dict:
    from ..memory.checkpoint import code_version
    return code_version(REPO)


def cfg_hash(cfg) -> str:
    d = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=list).encode()).hexdigest()


def sha256_file(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row, default=_jsonable) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _jsonable(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    if hasattr(x, "__dict__"):
        return {k: v for k, v in vars(x).items() if not k.startswith("_")}
    return str(x)


def prereg_committed(path: str = PREREG) -> tuple:
    """(ok, why): the pre-registration exists in HEAD and has no uncommitted change."""
    full = os.path.join(REPO, path)
    if not os.path.isfile(full):
        return False, f"{path} does not exist"
    try:
        r = subprocess.run(["git", "cat-file", "-e", f"HEAD:{path}"], cwd=REPO, capture_output=True, timeout=30)
        if r.returncode != 0:
            return False, f"{path} is not committed"
        st = subprocess.run(["git", "status", "--porcelain", "--", path], cwd=REPO, capture_output=True, text=True,
                            timeout=30)
        if st.stdout.strip():
            return False, f"{path} has uncommitted changes"
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"git unavailable: {e}"
    return True, "committed"


# ============================================================================================ canonical loop
def resolve_loop() -> tuple:
    """(run_lifetime callable, name). The plan's runtime.run_lifetime when it exists, else a delegation to the
    existing loop of runtime.py (runtime.run_steps) with the run_lifetime keyword signature."""
    from .. import runtime
    fn = getattr(runtime, "run_lifetime", None)
    if callable(fn):
        return fn, "runtime.run_lifetime"
    return _run_lifetime_via_run_steps, "runtime.run_steps (run_lifetime absent at this commit)"


def _run_lifetime_via_run_steps(agent, env, *, labels=None, label_store=None, reward_mode="observation_only",
                                learner=None, days=None, max_decisions=None, update_every=8, think_s=None,
                                checkpoint=None, on_step=None, seeds=None, profile_gpu=False):
    """run_lifetime's keyword signature over runtime.run_steps (the only act -> execute -> advance loop at this
    commit). No evaluator is passed, so the agent receives no environment reward (run_steps hands it zeros); labels
    are drained into the trainer-side label store after every round. `learner` is not supported by run_steps (the
    agent's own update() runs every `update_every` rounds); `days` is not enforced (the decision budget is)."""
    from .. import runtime
    if reward_mode != "observation_only":
        raise ValueError("the run_steps delegation implements observation_only only")
    if learner is not None:
        raise NotImplementedError("a LearnerHook needs runtime.run_lifetime (W1); run_steps drives agent.update()")
    B = getattr(env, "B", None) or len(env.observe().env_ids)
    n = max(1, int(math.ceil((max_decisions or B) / B)))
    joined = []

    def step(i, row):
        if labels is not None:
            for lab in labels.take_labels():
                if lab is not None and label_store is not None and label_store.append(lab):
                    joined.append(lab)
        if on_step is not None:
            on_step(i, row)
    rep = runtime.run_steps(agent, env, n, evaluator=None, update_every=update_every or None, seeds=seeds,
                            think_s=think_s, profile_gpu=profile_gpu, on_step=step)
    return {"report": rep, "labels_joined": len(joined), "rounds": n, "compute": rep.compute.to_dict(),
            "updates": rep.updates, "steps": rep.steps}


def _lifetime_stats(res) -> dict:
    """decisions / env_steps / updates / compute of a LifetimeResult (W1) or of the run_steps delegation."""
    if isinstance(res, dict):
        comp = res.get("compute") or {}
        ups = res.get("updates") or []
        return {"decisions": comp.get("decisions"), "env_steps": comp.get("env_steps"),
                "updates": sum(1 for u in ups if isinstance(u, dict) and u.get("updated")),
                "gpu_span_s": comp.get("gpu_span_s"), "wall_s": comp.get("wall_s"),
                "peak_gpu_mb": (comp.get("peak_gpu_mem_bytes") or 0) / 2 ** 20, "labels_joined": res.get("labels_joined")}
    out = {}
    for k in ("decisions", "env_steps", "updates", "labels_joined"):
        v = getattr(res, k, None)
        out[k] = len(v) if isinstance(v, list) else v
    comp = getattr(res, "compute", None)
    if comp is not None:
        c = comp.to_dict() if hasattr(comp, "to_dict") else dict(comp)
        out.update(decisions=out.get("decisions") or c.get("decisions"), env_steps=c.get("env_steps"),
                   gpu_span_s=c.get("gpu_span_s"), wall_s=c.get("wall_s"),
                   peak_gpu_mb=(c.get("peak_gpu_mem_bytes") or 0) / 2 ** 20)
    return out


# ============================================================================================ builder
@dataclass
class Stack:
    cfg: object
    agent: object
    model: object
    semantic: object
    trainer: object
    archive: object
    recorder: object
    ops: object
    run_dir: str
    builder: str
    learner: object = None


def build_stack(preset: str, run_dir: str, *, device: str = "cuda", seed: int = 0, overrides: Optional[dict] = None,
                semantic=None, agent_kw: Optional[dict] = None) -> Stack:
    """The canonical factory build.build_agent when present; else the stage-1 assembly with memory off."""
    overrides = dict(overrides or {})
    try:
        from .. import build as B
    except ImportError:
        B = None
    if B is not None and callable(getattr(B, "build_agent", None)):
        S = B.build_agent(preset, env_kind="experience", sem_model=None, device=device, run_dir=run_dir,
                          overrides=overrides)
        return Stack(S.cfg, S.agent, S.model, getattr(S, "semantic", None), getattr(S, "trainer", None),
                     getattr(S, "archive", None), getattr(S, "recorder", None), S.ops, run_dir, "build.build_agent",
                     getattr(S, "learner", None))
    return _build_stage1(preset, run_dir, device=device, seed=seed, overrides=overrides, semantic=semantic,
                         agent_kw=agent_kw)


def _build_stage1(preset, run_dir, *, device, seed, overrides, semantic=None, agent_kw=None) -> Stack:
    import torch
    from ..agent import CamUnifiedAgent
    from ..config import make_preset
    from ..memory.archive import EventArchive
    from ..memory.artifacts import ArtifactStore
    from ..memory.checkpoint import CheckpointOps
    from ..memory.replay import ReplaySampler
    from ..model import build_model
    from ..observe.trace import TraceRecorder
    from ..stage1 import system as SY
    os.makedirs(run_dir, exist_ok=True)
    cfg = make_preset(preset, **overrides)
    sc = SY.SystemConfig(scale=cfg.scale, overrides={}, sem_model=cfg.sem_model, device=device, seed=seed,
                         memory_sync_every=None, **{k: v for k, v in (agent_kw or {}).items()
                                                    if k in SY.SystemConfig.__dataclass_fields__})
    SY.set_deterministic(True)
    SY.seed_everything(seed)
    dev = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
    # a loaded semantic model is shared between stacks only when no decoder conditioning is configured: decoder
    # latent cross-attention installs hooks inside the semantic model, which must belong to exactly one CAM model
    share = semantic is not None and not int(getattr(cfg, "dec_xattn_every", 0) or 0)
    sem = semantic if share else SY.make_semantic(cfg, cfg.sem_model, dev)
    index = SY.make_index(cfg, os.path.join(run_dir, "memory_index"), dev) if cfg.use_memory else None
    model = build_model(cfg, components={"semantic": sem},
                        component_kwargs={"retriever": {"index": index}} if index is not None else None)
    model.to(dev)
    model.eval()
    trainer = SY.make_trainer(cfg, model, sc, os.path.join(run_dir, "learner", "trainer.jsonl"))
    arts = ArtifactStore(os.path.join(run_dir, "artifacts"))
    archive = EventArchive(os.path.join(run_dir, "archive"), cfg, artifacts=arts, autoflush=True)
    view = SY.BranchView(archive, set())
    replay = ReplaySampler(view, seed=seed)
    recorder = TraceRecorder(os.path.join(run_dir, "traces"))
    kw = dict(seq_len=sc.seq_len, n_seqs=sc.n_seqs, adapt_seq_len=sc.adapt_seq_len, adapt_n_seqs=sc.adapt_n_seqs,
              fresh_max_steps=sc.fresh_max_steps, loss_cap=sc.loss_cap, greedy=sc.greedy)
    kw.update({k: v for k, v in (agent_kw or {}).items() if k in ("seq_len", "n_seqs", "adapt_seq_len",
                                                                  "adapt_n_seqs", "fresh_max_steps", "greedy")})
    agent = CamUnifiedAgent(model, archive=archive, recorder=recorder, replay=replay, trainer=trainer, index=index,
                            consolidator=None, actor_id="cam0", seed=seed, memory_sync_every=None, **kw)
    agent.exclude_events = view.exclude
    return Stack(cfg, agent, model, sem, trainer, archive, recorder, CheckpointOps(agent), run_dir,
                 "stage1.system helpers (build.build_agent absent at this commit; memory off)")


def attach_fresh_experience(st: Stack, run_dir: str) -> None:
    """A new lifetime's own (empty) archive + replay (never the source lifetime's)."""
    from ..memory.archive import EventArchive
    from ..memory.replay import ReplaySampler
    from ..stage1 import system as SY
    arc = EventArchive(os.path.join(run_dir, "archive_target"), st.cfg, autoflush=True)
    view = SY.BranchView(arc, set())
    st.agent.archive, st.agent.replay = arc, ReplaySampler(view, seed=0)
    st.agent.exclude_events = view.exclude
    st.archive = arc


# ============================================================================================ environments
class OperatorLabels:
    """LabelSource over an ExperienceOperator: resolved episodes' labels, each exactly once (operator side)."""

    def __init__(self, operator):
        self.operator = operator
        self.seen: set = set()

    def take_labels(self) -> list:
        from ..envs.labels import from_outcome_label
        out = []
        for lab in self.operator.release_labels():
            key = (lab.env_id, lab.episode)
            if lab.status == "pending" or key in self.seen:
                continue
            self.seen.add(key)
            rec = from_outcome_label(lab, evaluator="experience_operator")
            if rec is not None:
                out.append(rec)
        return out


def families_for(split: str, names: Optional[Sequence[str]] = None) -> list:
    """Rapid families of a split (train for training / dev, val for confirm); names must belong to the split."""
    from crucible.rapid.families import families
    from ..envs.experience import BROKEN_MULTIAGENT_RAPID
    allowed = [f for f in families(split) if f not in BROKEN_MULTIAGENT_RAPID]
    if names:
        bad = [n for n in names if n not in allowed]
        if bad:
            raise ValueError(f"families {bad} are not in the {split!r} split (split guard)")
        return list(names)
    return allowed


def _spec_of(name: str, seed: int):
    """'mk_elastic' / 'rapid:mk_elastic' (tier 1), 'mech:<f>' (tier 1), 'integrity:<variant>' (CAM-side, tier -1)."""
    from ..envs.experience import WorldSpec
    if name.startswith("integrity:"):
        return WorldSpec(-1, name, seed)
    if name.startswith(("rapid:", "mech:")):
        return WorldSpec(1, name, seed)
    return WorldSpec(1, f"rapid:{name}", seed)


def make_world(families: Sequence[str], seed0: int, n_envs: int, split: str = "train"):
    """(adapter, operator, label source, source name, family per env) over Crucible experience worlds: W2's
    envs.factory.make_env when present, else the ExperienceOperator directly. reward_mode is observation_only."""
    try:
        from ..envs import factory as F
    except ImportError:
        F = None
    fams = [families[i % len(families)] for i in range(n_envs)]
    specs = [_spec_of(f, seed0 + i) for i, f in enumerate(fams)]
    if F is not None and callable(getattr(F, "make_env", None)):
        adapter, operator, labels = F.make_env("experience", specs, n_envs, "observation_only", split)
        return adapter, operator, labels, "envs.factory.make_env", fams
    from ..envs.experience import ExperienceOperator
    op = ExperienceOperator(specs)
    return op.adapter(), op, OperatorLabels(op), "envs.experience.ExperienceOperator", fams


def operator_utilities(operator) -> list:
    """Evaluator-side verified utility of every world now (PRIVATE: scoring only, never to the agent)."""
    out = []
    for i, w in enumerate(getattr(operator, "worlds", [])):
        try:
            lab = operator._label(w)
            out.append({"env_id": w.env_id, "family": w.spec.name, "utility": lab.utility, "status": lab.status,
                        "value": lab.value, "resolved": bool(operator.resolved(i))})
        except Exception as e:
            out.append({"env_id": getattr(w, "env_id", str(i)), "error": repr(e)[:200]})
    return out


# ============================================================================================ condition mechanics
def load_subset(ops, path: str, prefixes: Sequence[str]) -> list:
    """Overwrite the model tensors under `prefixes` with those of a weights file (same layout required)."""
    import torch
    from ..memory.checkpoint import read_weights
    sd, _, _ = read_weights(path)
    msd = ops.agent.model.state_dict()
    keys = [k for k in sd if any(k.startswith(p) for p in prefixes) and k in msd]
    with torch.no_grad():
        for k in keys:
            if tuple(msd[k].shape) != tuple(sd[k].shape):
                raise ValueError(f"{k}: shape {tuple(sd[k].shape)} != {tuple(msd[k].shape)}")
            msd[k].copy_(sd[k].to(msd[k].device, msd[k].dtype))
    return keys


def _set_group_lr(trainer, keep: Callable[[str], bool]) -> list:
    """Learning rate 0 for every optimiser group whose name fails `keep` (current optimiser AND the spec its
    resets rebuild from). Returns the names of the frozen groups."""
    frozen = []
    for g in list(getattr(trainer, "_pg_spec", [])) + [g for o in (getattr(trainer, "opt", None),) if o is not None
                                                       for g in o.param_groups]:
        name = str(g.get("name", ""))
        if not keep(name):
            g["lr"] = 0.0
            if "base_lr" in g:
                g["base_lr"] = 0.0
            if "initial_lr" in g:
                g["initial_lr"] = 0.0
            frozen.append(name)
    return sorted(set(frozen))


def _digest(params) -> str:
    h = hashlib.sha256()
    for p in params:
        h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def disable_decoder_conditioning(model) -> dict:
    """Condition F: remove CAM's latent influence on the decoder. Prefers a component switch
    (``set_decoder_conditioning(False)`` on the model or its action generator); otherwise the bridge's latent input
    R is replaced by zeros with a forward pre-hook. Returns what was done (and anything left on)."""
    import torch
    for obj in (model, getattr(model, "action_generator", None)):
        if obj is not None and callable(getattr(obj, "set_decoder_conditioning", None)):
            obj.set_decoder_conditioning(False)
            return {"method": f"{type(obj).__name__}.set_decoder_conditioning(False)", "hooks": 0}
    ag = getattr(model, "action_generator", None)
    hooks, left = [], []
    br = getattr(ag, "bridge", None)
    if br is not None:
        def zero_latent(mod, args, kwargs):
            if args:
                args = (torch.zeros_like(args[0]),) + tuple(args[1:])
            return args, kwargs
        hooks.append(br.register_forward_pre_hook(zero_latent, with_kwargs=True))
    for name, m in model.named_modules():
        if "dec_xattn" in name or "decoder_xattn" in name or type(m).__name__.lower().startswith("decoderlatent"):
            left.append(name)
    return {"method": "bridge latent input R -> 0 (forward pre-hook)" if hooks else "none found",
            "hooks": len(hooks), "_handles": hooks, "not_disabled": left[:20]}


class LatentSwapAgent:
    """Condition G wrapper: on every probe decision, computes the batch's actions with each row's latent state Z
    replaced by row perm[b]'s (observations unchanged), on a snapshot that is then discarded; the executed decision
    is the agent's own. Everything else delegates to the wrapped agent, so the canonical loop drives it unchanged."""

    def __init__(self, agent, probe_every: int = 1, seed: int = 0):
        object.__setattr__(self, "_agent", agent)
        object.__setattr__(self, "_probe_every", int(probe_every))
        object.__setattr__(self, "_rng", random.Random(seed))
        object.__setattr__(self, "probes", [])
        object.__setattr__(self, "_n", 0)

    def __getattr__(self, k):
        return getattr(self._agent, k)

    def __setattr__(self, k, v):
        setattr(self._agent, k, v)

    @staticmethod
    def _key(a) -> Optional[str]:
        if a is None:
            return None
        return json.dumps({"tool": a.tool, "args": a.args}, sort_keys=True, default=str)

    def _snapshot(self) -> dict:
        import torch
        a = self._agent
        snap = {"state": copy.deepcopy(a.state), "_pending": copy.deepcopy(a._pending),
                "_post": copy.deepcopy(a._post), "_t0": list(a._t0), "_funds": copy.deepcopy(a._funds),
                "_funds0": list(a._funds0), "trng": a.torch_rng.get_state(), "prng": a.py_rng.getstate(),
                "grng": torch.get_rng_state(), "timing": dict(a.timing)}
        if torch.cuda.is_available():
            snap["crng"] = torch.cuda.get_rng_state_all()
        return snap

    def _restore(self, s: dict) -> None:
        import torch
        a = self._agent
        a.state, a._pending, a._post = copy.deepcopy(s["state"]), copy.deepcopy(s["_pending"]), copy.deepcopy(s["_post"])
        a._t0, a._funds, a._funds0 = list(s["_t0"]), copy.deepcopy(s["_funds"]), list(s["_funds0"])
        a.torch_rng.set_state(s["trng"])
        a.py_rng.setstate(s["prng"])
        torch.set_rng_state(s["grng"])
        if "crng" in s:
            torch.cuda.set_rng_state_all(s["crng"])
        a.timing = dict(s["timing"])

    def act(self, obs, tools):
        a = self._agent
        n = self._n
        object.__setattr__(self, "_n", n + 1)
        if a.state is None:
            a.reset(obs.env_ids)
        B = len(obs.env_ids)
        live = [b for b in range(B) if not obs.done[b]]
        if self._probe_every and n % self._probe_every == 0 and len(live) >= 2:
            snap = self._snapshot()
            arc = a.archive
            a.archive = None                            # probes never write experience
            try:
                own = a.act(obs, tools)
                z_own = a.state.Z.detach().float().clone()
                self._restore(snap)
                perm = live[1:] + live[:1]              # derangement of the live rows
                Z = a.state.Z.clone()
                for b, d in zip(live, perm):
                    a.state.Z[b] = Z[d]
                swap = a.act(obs, tools)
                z_swap = a.state.Z.detach().float().clone()
                self._restore(snap)
            finally:
                a.archive = arc
            Zf = Z.detach().float()
            for b, d in zip(live, perm):
                rel_in = float((Zf[d] - Zf[b]).norm() / Zf[b].norm().clamp(min=1e-12))
                rel_out = float((z_swap[b] - z_own[b]).norm() / z_own[b].norm().clamp(min=1e-12))
                self.probes.append({"probe": n, "row": b, "donor": d, "z_in_rel_diff": rel_in,
                                    "z_out_rel_diff": rel_out, "own": self._key(own[b]),
                                    "swap": self._key(swap[b]), "donor_own": self._key(own[d]),
                                    "own_tool": getattr(own[b], "tool", None), "swap_tool": getattr(swap[b], "tool", None),
                                    "donor_tool": getattr(own[d], "tool", None)})
        return a.act(obs, tools)

    def metrics(self) -> dict:
        P = [p for p in self.probes if p["own"] is not None and p["swap"] is not None]
        if not P:
            return {"g_probes": 0}
        ch = sum(p["own"] != p["swap"] for p in P) / len(P)
        tch = sum(p["own_tool"] != p["swap_tool"] for p in P) / len(P)
        dag = sum(p["swap_tool"] == p["donor_tool"] for p in P) / len(P)
        base = sum(p["own_tool"] == p["donor_tool"] for p in P) / len(P)
        zin = [p["z_in_rel_diff"] for p in P if "z_in_rel_diff" in p]
        zout = [p["z_out_rel_diff"] for p in P if "z_out_rel_diff" in p]
        return {"g_probes": len(P), "g_action_change_rate": ch, "g_tool_change_rate": tch,
                "g_donor_tool_agreement": dag, "g_own_donor_tool_agreement": base,
                "g_donor_follow_excess": dag - base,
                # the intervention reached the latent: relative difference of the swapped-in Z_{t-1} and of the
                # resulting Z_t (a nonzero latent change with no action change measures a decision bypass)
                "g_z_prev_rel_diff_mean": sum(zin) / len(zin) if zin else None,
                "g_z_next_rel_diff_mean": sum(zout) / len(zout) if zout else None}


# ============================================================================================ integrity scan
def integrity_scan(archive, cfg, label_store=None) -> dict:
    """Run-directory scan of a CAM condition: reward_mode observation_only, every archived actor record carries
    reward 0 (no environment reward reached the agent) and no label key in its actor-visible fields; verified label
    values from the trainer-side store never appear as archived rewards."""
    out = {"reward_mode": getattr(cfg, "reward_mode", None), "records": 0, "nonzero_rewards": 0,
           "label_keys_in_extra": 0}
    if archive is None:
        out["ok"] = out["reward_mode"] == "observation_only"
        out["note"] = "no archive attached"
        return out
    archive.flush()
    for batch in archive.scan(batch_size=2048):
        for r in batch:
            out["records"] += 1
            if float(r.reward or 0.0) != 0.0:
                out["nonzero_rewards"] += 1
            if any(k in (r.extra or {}) for k in ("success", "utility", "verified_outcome", "label")):
                out["label_keys_in_extra"] += 1
    out["labels"] = len(label_store) if label_store is not None else None
    out["ok"] = (out["reward_mode"] == "observation_only" and out["nonzero_rewards"] == 0 and
                 out["label_keys_in_extra"] == 0)
    return out


# ============================================================================================ one condition
@dataclass
class RunSpec:
    condition: str
    run_dir: str
    preset: str = "dom_dev"
    seed: int = 0
    set: str = "dev"                       # dev (train families) | confirm (val families, prereg required)
    smoke: bool = False
    families: tuple = ()
    n_envs: int = 2
    decisions: int = 16
    update_every: int = 4
    ckpt: Optional[str] = None             # trained weights (C-I); "auto" = a short source run inside this call
    init_ckpt: Optional[str] = None        # the source run's initial weights (D)
    source_rounds: int = 4                 # "auto" source run length (rounds)
    device: str = "cuda"
    profile_gpu: bool = True
    overrides: dict = field(default_factory=dict)
    agent_kw: dict = field(default_factory=dict)
    probe_every: int = 1                   # G
    experiment: str = EXPERIMENT_ID


def _seed0(spec: RunSpec) -> int:
    return (CONFIRM_SEED0 if spec.set == "confirm" else DEV_SEED0) + 1000 * spec.seed


def source_run(spec: RunSpec, run_dir: str, semantic=None) -> dict:
    """A short source lifetime (train families) through the canonical loop: initial and trained weights."""
    st = build_stack(spec.preset, run_dir, device=spec.device, seed=spec.seed, overrides=spec.overrides,
                     semantic=semantic, agent_kw=spec.agent_kw)
    init = os.path.join(run_dir, "init.safetensors")
    st.ops.save_weights(init)
    fams = families_for("train", spec.families or SMOKE_FAMILIES)
    adapter, operator, labels, env_src, fam_list = make_world(fams, DEV_SEED0 + 77_000 + 1000 * spec.seed,
                                                              spec.n_envs)
    from ..envs.labels import LabelStore
    store = LabelStore(os.path.join(run_dir, "labels"))
    run_lifetime, loop = resolve_loop()
    res = run_lifetime(st.agent, adapter, labels=labels, label_store=store, reward_mode="observation_only",
                       learner=st.learner, days=None, max_decisions=spec.source_rounds * spec.n_envs,
                       update_every=max(1, spec.update_every // 2), think_s=None, checkpoint=None, on_step=None)
    trained = os.path.join(run_dir, "trained.safetensors")
    info = st.ops.save_weights(trained)
    return {"init": init, "trained": trained, "trained_info": info, "loop": loop, "stats": _lifetime_stats(res),
            "semantic": st.semantic, "integrity": integrity_scan(st.archive, st.cfg, store)}


def run_condition(spec: RunSpec, semantic=None) -> dict:
    """One (condition, seed) lifetime through the canonical loop; appends the provenance row and eval rows to
    <run_dir>/eval/ and returns the row."""
    import torch
    from ..envs.labels import LabelStore
    from ..memory.checkpoint import model_arch_hash
    from ..observe.accounting import GpuBusy, TokenMeter
    c = spec.condition.upper()
    if c not in CONDITIONS:
        raise ValueError(f"unknown condition {c!r}; known: {sorted(CONDITIONS)}")
    deviations = ["smoke: engineering check with a tiny budget, not an evaluation"] if spec.smoke else []
    if spec.set == "confirm":
        ok, why = prereg_committed()
        if not ok:
            raise RuntimeError(f"confirm-set evaluation refused: pre-registration {PREREG}: {why}")
        if spec.smoke:
            raise ValueError("a smoke run is never a confirm evaluation")
    split = "val" if spec.set == "confirm" else "train"
    fams = families_for(split, spec.families or (SMOKE_FAMILIES if split == "train" else None))
    run_dir = os.path.abspath(os.path.join(spec.run_dir, c, f"seed{spec.seed}"))
    os.makedirs(run_dir, exist_ok=True)
    row_id = f"{EXPERIMENT_ID}-{c}-s{spec.seed}-{uuid.uuid4().hex[:8]}"
    t_start = time.time()
    detail: dict = {"description": CONDITIONS[c]}
    src = None
    ckpt, init_ckpt = spec.ckpt, spec.init_ckpt
    if c in NEEDS_TRAINED and (ckpt in (None, "auto")):
        if not spec.smoke and ckpt is None:
            raise ValueError(f"condition {c} needs --ckpt (trained weights) or --ckpt auto")
        src = source_run(spec, os.path.join(run_dir, "source"), semantic=semantic)
        semantic = src["semantic"]
        ckpt, init_ckpt = src["trained"], src["init"]
        detail["source"] = {k: src[k] for k in ("loop", "stats", "integrity")}
        deviations.append(f"trained weights from a {spec.source_rounds}-round source run inside this call "
                          "(smoke / auto), not a converged training run")
    if c == "A":
        return _run_condition_a(spec, run_dir, fams, row_id, t_start)
    st = build_stack(spec.preset, os.path.join(run_dir, "target"), device=spec.device, seed=spec.seed,
                     overrides=spec.overrides, semantic=semantic, agent_kw=spec.agent_kw)
    if st.cfg.reward_mode != "observation_only":
        raise RuntimeError(f"CAM conditions run in reward_mode observation_only; the preset has {st.cfg.reward_mode}")
    handles = []
    if c in NEEDS_TRAINED:
        info = st.ops.load_weights_only(ckpt, seed=spec.seed)
        attach_fresh_experience(st, run_dir)
        detail["weights"] = {"sha256": info.get("sha256"), "arch_hash": info.get("arch_hash"),
                             "migrated": info.get("migrated"), "reset": info["reset"]["reset"]}
    if c == "D":
        if not init_ckpt:
            raise ValueError("condition D needs --init-ckpt (the source run's initial weights)")
        keys = load_subset(st.ops, init_ckpt, STRATEGIC_PREFIXES)
        detail["reset_to_init"] = {"prefixes": list(STRATEGIC_PREFIXES), "tensors": len(keys)}
    if c == "F":
        r = disable_decoder_conditioning(st.model)
        handles = r.pop("_handles", [])
        detail["decoder_conditioning"] = r
        if r.get("not_disabled"):
            deviations.append(f"decoder latent cross-attention modules present but not switchable: "
                              f"{r['not_disabled'][:3]}")
    core_digest = None
    if c in ("H", "I"):
        tr = st.trainer
        keep = (lambda n: not n.startswith("core/")) if c == "H" else (lambda n: n.startswith("semantic/"))
        detail["frozen_groups"] = _set_group_lr(tr, keep)
        core_digest = _digest(st.model.core.parameters())
    agent = st.agent
    if c == "G":
        agent.greedy = True
        agent = LatentSwapAgent(st.agent, probe_every=spec.probe_every, seed=spec.seed)
        detail["probe_every"] = spec.probe_every
        detail["acting"] = "greedy (deterministic decisions: the swap is the only difference)"
    adapter, operator, labels, env_src, fam_list = make_world(fams, _seed0(spec), spec.n_envs, split)
    store = LabelStore(os.path.join(run_dir, "labels"))
    run_lifetime, loop = resolve_loop()
    if loop != "runtime.run_lifetime":
        deviations.append(f"canonical loop: {loop}")
    if not st.builder.startswith("build.build_agent"):
        deviations.append(f"factory: {st.builder}")
    tm = TokenMeter()
    lm = getattr(st.semantic, "lm", None)
    if lm is not None:
        tm.attach(lm)
    learn = LEARNING.get(c, False)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with GpuBusy(enabled=spec.profile_gpu) as gb:
        res = run_lifetime(agent, adapter, labels=labels, label_store=store, reward_mode="observation_only",
                           learner=st.learner if learn else None, days=None, max_decisions=spec.decisions,
                           update_every=spec.update_every if learn else 0, think_s=None, checkpoint=None,
                           on_step=None)
    tm.detach()
    for h in handles:
        h.remove()
    stats = _lifetime_stats(res)
    metrics = {}
    utils = operator_utilities(operator)
    us = [u["utility"] for u in utils if isinstance(u.get("utility"), (int, float))]
    metrics["verified_utility_end_mean"] = sum(us) / len(us) if us else None
    metrics["worlds_resolved"] = sum(1 for u in utils if u.get("resolved"))
    metrics["labels_joined"] = len(store)
    if c == "G":
        metrics.update(agent.metrics())
        _write_jsonl_rows(os.path.join(run_dir, "g_probes.jsonl"), agent.probes)
    if core_digest is not None:
        detail["core_unchanged"] = _digest(st.model.core.parameters()) == core_digest
    if c in LEARNING:
        metrics["updates"] = stats.get("updates")
    integ = integrity_scan(st.archive, st.cfg, store)
    counts = st.model.count_parameters() if hasattr(st.model, "count_parameters") else {}
    cv = code_state()
    row = {"experiment": spec.experiment, "condition": c, "seed": spec.seed, "set": spec.set, "smoke": spec.smoke,
           "run_id": row_id, "t_wall": time.time(), "t_start": t_start, "code_commit": cv["commit"],
           "code_dirty": cv["dirty"], "loop": loop, "builder": st.builder, "env_source": env_src,
           "preset": spec.preset, "cfg_hash": cfg_hash(st.cfg), "arch_hash": model_arch_hash(
               st.model, getattr(st.agent, "weights_exclude", ())), "ckpt": ckpt,
           "ckpt_sha256": sha256_file(ckpt), "init_ckpt_sha256": sha256_file(init_ckpt),
           "reward_mode": st.cfg.reward_mode, "semantic_model": st.cfg.sem_model,
           "params": {"cam_total": counts.get("cam_total"), "cam_trainable": counts.get("cam_trainable")},
           "families": fam_list, "world_seed0": _seed0(spec), "n_envs": spec.n_envs, "learning": learn,
           "decisions": stats.get("decisions"), "env_steps": stats.get("env_steps"), "updates": stats.get("updates"),
           "gpu_busy_s": gb.busy_s, "gpu_span_s": stats.get("gpu_span_s"), "semantic_tokens": tm.tokens,
           "semantic_calls": tm.calls, "wall_s": time.time() - t_start, "peak_gpu_mb": stats.get("peak_gpu_mb"),
           "integrity": integ, "metrics": metrics, "condition_detail": detail, "deviations": deviations,
           "utilities": utils}
    _emit(spec, row, run_dir, fam_list, getattr(st.agent, "version", None))
    return row


def _write_jsonl_rows(path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=_jsonable) + "\n")


def _emit(spec: RunSpec, row: dict, run_dir: str, fam_list, version) -> None:
    """Provenance row + per-metric eval rows, both under <spec.run_dir>/eval/ (one dashboard per suite run) and the
    condition's own run directory."""
    for root in {os.path.abspath(spec.run_dir), run_dir}:
        append_jsonl(os.path.join(root, "eval", "provenance.jsonl"), row)
        for k, v in sorted(row["metrics"].items()):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                append_jsonl(os.path.join(root, "eval", f"{row['condition']}.jsonl"),
                             {"metric": f"{row['condition']}/{k}", "value": v, "version": version,
                              "family": ",".join(sorted(set(fam_list))), "seed": row["seed"], "set": row["set"],
                              "condition": row["condition"], "t": row["t_wall"], "experiment": row["experiment"],
                              "run_id": row["run_id"], "learner": f"cam_{row['condition']}"})


def _run_condition_a(spec: RunSpec, run_dir: str, fams, row_id: str, t_start: float) -> dict:
    """Condition A: the frozen pretrained semantic model with the same tools, through the canonical loop."""
    from ..config import make_preset
    from ..envs.baselines import FrozenSemanticAgent
    from ..envs.labels import LabelStore
    from ..observe.accounting import GpuBusy, TokenMeter
    cfg = make_preset(spec.preset, **spec.overrides)
    agent = FrozenSemanticAgent("tools", model=cfg.sem_model, device=spec.device, seed=spec.seed)
    adapter, operator, labels, env_src, fam_list = make_world(fams, _seed0(spec), spec.n_envs)
    store = LabelStore(os.path.join(run_dir, "labels"))
    run_lifetime, loop = resolve_loop()
    tm = TokenMeter()
    m = getattr(getattr(agent, "backend", None), "model", None)
    if m is not None:
        tm.attach(m)
    with GpuBusy(enabled=spec.profile_gpu) as gb:
        res = run_lifetime(agent, adapter, labels=labels, label_store=store, reward_mode="observation_only",
                           learner=None, days=None, max_decisions=spec.decisions, update_every=0, think_s=None,
                           checkpoint=None, on_step=None)
    tm.detach()
    stats = _lifetime_stats(res)
    utils = operator_utilities(operator)
    us = [u["utility"] for u in utils if isinstance(u.get("utility"), (int, float))]
    cv = code_state()
    row = {"experiment": spec.experiment, "condition": "A", "seed": spec.seed, "set": spec.set, "smoke": spec.smoke,
           "run_id": row_id, "t_wall": time.time(), "t_start": t_start, "code_commit": cv["commit"],
           "code_dirty": cv["dirty"], "loop": loop, "builder": "envs.baselines.FrozenSemanticAgent",
           "env_source": env_src, "preset": spec.preset, "cfg_hash": cfg_hash(cfg), "ckpt": None,
           "ckpt_sha256": None, "reward_mode": "observation_only", "semantic_model": cfg.sem_model,
           "families": fam_list, "world_seed0": _seed0(spec), "n_envs": spec.n_envs, "learning": False,
           "decisions": stats.get("decisions"), "env_steps": stats.get("env_steps"), "updates": 0,
           "gpu_busy_s": gb.busy_s, "gpu_span_s": stats.get("gpu_span_s"), "semantic_tokens": tm.tokens,
           "wall_s": time.time() - t_start, "integrity": {"ok": True, "note": "baseline agent (no CAM)"},
           "metrics": {"verified_utility_end_mean": sum(us) / len(us) if us else None, "labels_joined": len(store)},
           "condition_detail": {"description": CONDITIONS["A"]}, "deviations": [], "utilities": utils}
    _emit(spec, row, run_dir, fam_list, None)
    return row


# ============================================================================================ checkpoint e2e (W9)
def ckpt_e2e(run_dir: str, *, updates: int = 200, n_envs: int = 2, seed: int = 0, preset: str = "dom_dev",
             act_rounds: int = 4, device: str = "cuda", chunk: Optional[int] = None, families=SMOKE_FAMILIES,
             agent_kw: Optional[dict] = None, budget_s: float = 600.0) -> dict:
    """Plan §3 W9 real end-to-end check, part 1 (training): build the real `preset` stack, train until the agent
    has made `updates` learner updates (one update per round), resumable in chunks of `chunk` updates via
    full-lifetime checkpoints (each call does at most one chunk), then save weights-only. Returns progress."""
    os.makedirs(run_dir, exist_ok=True)
    state_p = os.path.join(run_dir, "e2e_state.json")
    stt = json.load(open(state_p)) if os.path.exists(state_p) else {"updates": 0, "chunks": []}
    if stt["updates"] >= updates and os.path.exists(os.path.join(run_dir, "weights.safetensors")):
        return stt
    agent_kw = dict(agent_kw or {"seq_len": 4, "n_seqs": 1, "adapt_seq_len": 5, "adapt_n_seqs": 1,
                                 "fresh_max_steps": 32})
    st = build_stack(preset, os.path.join(run_dir, "source"), device=device, seed=seed, agent_kw=agent_kw)
    life = os.path.join(run_dir, "life")
    if stt["updates"] > 0:
        st.ops.load_full_lifetime(life)
        # the lifetime archive is reattached by pointer; replay keeps sampling it
    fams = families_for("train", families)
    adapter, operator, labels, env_src, _ = make_world(fams, DEV_SEED0 + 90_000 + 1000 * seed, n_envs)
    from ..envs.labels import LabelStore
    store = LabelStore(os.path.join(run_dir, "labels"))
    run_lifetime, loop = resolve_loop()
    todo = min(updates - stt["updates"], chunk or updates)
    t0 = time.time()
    done = 0
    while done < todo and time.time() - t0 < budget_s:     # sub-chunks: the GPU-lock hold stays bounded
        k = min(20, todo - done)
        res = run_lifetime(st.agent, adapter, labels=labels, label_store=store, reward_mode="observation_only",
                           learner=st.learner, days=None, max_decisions=k * n_envs, update_every=1, think_s=None,
                           checkpoint=None, on_step=None)
        s = _lifetime_stats(res)
        done += int(s.get("updates") or 0)
        stt["chunks"].append({"t": time.time(), "wall_s": time.time() - t0, "loop": loop, **s})
        if not s.get("updates"):
            break
    stt["updates"] += done
    st.ops.save_full_lifetime(life)
    if stt["updates"] >= updates:
        info = st.ops.save_weights(os.path.join(run_dir, "weights.safetensors"))
        stt["weights"] = {k: info[k] for k in ("sha256", "arch_hash", "code_commit", "tensors", "bytes")}
        stt["agent_version"] = st.agent.version
        stt["trainer_n_updates"] = getattr(st.trainer, "n_updates", None)
        stt["source_event_ids_sample"] = None
        stt["source_archive_count"] = st.archive.count() if st.archive is not None else None
    json.dump(stt, open(state_p, "w"), indent=1, default=str)
    return stt


def ckpt_e2e_load(run_dir: str, out: str, *, seed: int = 7, preset: str = "dom_dev", n_envs: int = 2,
                  act_rounds: int = 4, device: str = "cuda", families=SMOKE_FAMILIES, world_seed: int = 123_000,
                  agent_kw: Optional[dict] = None) -> dict:
    """Part 2 (run in a FRESH process, twice): build, load_weights_only(seed), act greedily for `act_rounds` rounds
    on fixed world seeds through the canonical loop with one update at the end; record the actions, the update's
    consumed / replayed event ids and the reset state, as JSON at `out`."""
    import torch
    agent_kw = dict(agent_kw or {"seq_len": 4, "n_seqs": 1, "adapt_seq_len": 5, "adapt_n_seqs": 1,
                                 "fresh_max_steps": 32, "greedy": True})
    tgt = os.path.join(os.path.dirname(out), f"target_{os.path.basename(out)[:-5]}")
    st = build_stack(preset, tgt, device=device, seed=seed, agent_kw=agent_kw)
    info = st.ops.load_weights_only(os.path.join(run_dir, "weights.safetensors"), seed=seed)
    attach_fresh_experience(st, tgt)
    a = st.agent
    after_load = {"fresh": len(a.fresh), "loss_mag": len(a.loss_mag), "update_log": len(a.update_log),
                  "trainer_n_updates": getattr(st.trainer, "n_updates", None), "version": a.version,
                  "reset": info["reset"]["reset"], "weights_sha256": info["sha256"], "migrated": info["migrated"]}
    fams = families_for("train", families)
    adapter, operator, labels, env_src, _ = make_world(fams, world_seed, n_envs)
    run_lifetime, loop = resolve_loop()
    res = run_lifetime(a, adapter, labels=labels, label_store=None, reward_mode="observation_only",
                       learner=st.learner, days=None, max_decisions=act_rounds * n_envs, update_every=act_rounds,
                       think_s=None, checkpoint=None, on_step=None)
    steps = res["steps"] if isinstance(res, dict) else getattr(res, "steps", [])
    ups = res["updates"] if isinstance(res, dict) else getattr(res, "updates", [])
    ups = [u for u in ups if isinstance(u, dict) and u.get("updated")]
    new_ids = set()
    for b in st.archive.scan(batch_size=4096):
        new_ids |= {r.event_id for r in b}
    rec = {"loop": loop, "after_load": after_load, "tools": [s.get("tools") for s in steps],
           "actions": [], "updates": [{"consumed_events": u.get("consumed_events"),
                                       "replay_events": u.get("replay_events"),
                                       "consumed_decisions": u.get("consumed_decisions")} for u in ups],
           "new_archive_ids": sorted(new_ids), "seed": seed, "world_seed": world_seed,
           "cuda": torch.cuda.is_available()}
    for b in st.archive.scan(batch_size=4096):
        for r in b:
            if r.action is not None:
                rec["actions"].append({"step": r.step, "env": r.env_id, "tool": r.action.tool,
                                       "args": json.loads(json.dumps(r.action.args, default=str))})
    rec["actions"].sort(key=lambda x: (x["env"], x["step"]))
    json.dump(rec, open(out, "w"), indent=1, default=str)
    return rec


# ============================================================================================ CLI
def _spec_from(a) -> RunSpec:
    return RunSpec(condition=a.condition, run_dir=a.run_dir, preset=a.preset, seed=a.seed, set=a.set,
                   smoke=a.smoke, families=tuple(a.families.split(",")) if a.families else (), n_envs=a.n_envs,
                   decisions=a.decisions, update_every=a.update_every, ckpt=a.ckpt, init_ckpt=a.init_ckpt,
                   source_rounds=a.source_rounds, device=a.device, profile_gpu=not a.no_profile,
                   probe_every=a.probe_every)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run conditions (comma-separated) through the canonical loop")
    r.add_argument("--condition", required=True)
    r.add_argument("--run-dir", required=True)
    r.add_argument("--preset", default="dom_dev")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--set", default="dev", choices=("dev", "confirm"))
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--families", default="")
    r.add_argument("--n-envs", type=int, default=2)
    r.add_argument("--decisions", type=int, default=None)
    r.add_argument("--update-every", type=int, default=4)
    r.add_argument("--ckpt", default=None)
    r.add_argument("--init-ckpt", default=None)
    r.add_argument("--source-rounds", type=int, default=4)
    r.add_argument("--device", default="cuda")
    r.add_argument("--no-profile", action="store_true")
    r.add_argument("--probe-every", type=int, default=1)
    r.add_argument("--dashboard", action="store_true", help="render <run-dir>/dashboard afterwards")
    t = sub.add_parser("ckpt-e2e", help="W9 real end-to-end checkpoint check (training part, resumable)")
    t.add_argument("--run-dir", required=True)
    t.add_argument("--updates", type=int, default=200)
    t.add_argument("--chunk", type=int, default=None)
    t.add_argument("--n-envs", type=int, default=2)
    t.add_argument("--device", default="cuda")
    t2 = sub.add_parser("ckpt-e2e-load", help="W9 e2e: fresh-process weights-only load + act + one update")
    t2.add_argument("--run-dir", required=True)
    t2.add_argument("--out", required=True)
    t2.add_argument("--seed", type=int, default=7)
    t2.add_argument("--act-rounds", type=int, default=4)
    t2.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        if a.decisions is None:
            a.decisions = 6 * a.n_envs if a.smoke else 64 * a.n_envs
        sem = None
        conds = [c.strip().upper() for c in a.condition.split(",") if c.strip()]
        if a.smoke and a.ckpt in (None, "auto") and any(c in NEEDS_TRAINED for c in conds):
            a.condition = conds[0]
            src = source_run(_spec_from(a), os.path.join(os.path.abspath(a.run_dir), "source"))
            a.ckpt, a.init_ckpt, sem = src["trained"], src["init"], src["semantic"]
            append_jsonl(os.path.join(os.path.abspath(a.run_dir), "eval", "source.jsonl"),
                         {k: src[k] for k in ("init", "trained", "loop", "stats", "integrity")})
        for c in conds:
            a.condition = c
            row = run_condition(_spec_from(a), semantic=sem)
            print(json.dumps({k: row.get(k) for k in ("experiment", "condition", "seed", "loop", "decisions",
                                                      "updates", "gpu_busy_s", "semantic_tokens", "metrics",
                                                      "integrity")}, default=str))
        if a.dashboard:
            from ..observe.dashboard import build_dashboard
            print(build_dashboard(a.run_dir))
        return 0
    if a.cmd == "ckpt-e2e":
        print(json.dumps(ckpt_e2e(a.run_dir, updates=a.updates, chunk=a.chunk, n_envs=a.n_envs, device=a.device),
                         default=str)[:4000])
        return 0
    if a.cmd == "ckpt-e2e-load":
        rec = ckpt_e2e_load(a.run_dir, a.out, seed=a.seed, act_rounds=a.act_rounds, device=a.device)
        print(json.dumps({"after_load": rec["after_load"], "n_actions": len(rec["actions"]),
                          "updates": len(rec["updates"])}, default=str))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
