"""The integrated CAM Unified model as the central learner's model (`learner.LearnerModelP`; spec 12, 14.1, 14.4;
manifest A16/F03).

`CamLearnerModel` holds the learner's copy of the same `CamUnifiedModel` the actors run (`cam_policy.build_cam_model`:
real components of workstreams A-E), workstream D's `CombinedTrainer` (one AdamW over the core / heads / semantic
LoRA + bridge groups, per-objective gradient contributions into the shared backbone) and D's
`ActorCriticObjective` (PPO on fresh rows, V-trace on stale rows), and implements one optimiser step per
`LearnerBatch`:

1. **Inputs.** For every decision of every trajectory piece (burn-in context + offered decisions + bootstrap
   decision) the exact forward inputs the actor used are read from the run's artifact store (`cam_policy`), with the
   record's observations and the decision's tool list.
2. **Latent state under the current weights** (spec 14.3: no blind reuse of stale latent state). A piece whose
   decisions were all produced by the learner's current version from a latent state built under that version
   (``model_version == state_version == learner version`` for every decision) uses the stored Z_{t-1} — it *is* what
   the current weights compute. Any other piece is re-filtered: Z is recomputed along the piece under the current
   weights from the stored state of its first event (the burn-in prefix washes out the stale start, R2D2-style;
   ``recompute_state="always"`` forces this, "never" disables it).
3. **Policy terms.** `CamUnifiedModel.policy_evaluate` recomputes, with gradients, the full behaviour
   log-probability of every recorded action (tool pointer + evidence pointer + typed heads + Qwen/LoRA tokens + the
   option decision), the entropy, the value (two-hot logits) and the reasoner's ponder cost; the bootstrap decision
   contributes its value only. Each piece becomes one row of D's `Trajectory` (verified reward + ``lp_coef`` x the
   hindsight learning progress recorded on the next decision, semi-Markov dt, done, behaviour logp, model version,
   behaviour-time values); D routes rows with any lag > 0 (or a fresh-KL above ``max_fresh_kl``) to V-trace with
   truncated importance weights from the *recorded* behaviour log-probability, and only exactly fresh lag-0 rows to
   PPO (spec 14.4: stale data never gets a PPO ratio).
4. **World / adaptation terms** on the same real experience: the trained decisions are cut into windows of
   ``world_seq_len``, tokenised from the archived records (`CamUnifiedAgent.tokenise_sequences`), filtered with the
   acting context (`filter_sequence`) and scored by the world model's loss and the support/query adaptation loss.
5. `CombinedTrainer.step`; the model version becomes ``learner_version + 1``.

Metrics returned per update include the losses, D's routing counts, V-trace ρ statistics (mean, clipped fraction,
effective sample size), the deviation between the recomputed and the recorded behaviour log-probability on lag-0
decisions (exactness check), per-actor counts of decisions trained by the policy objective, and wall-time phases.
Decisions the model cannot evaluate (missing inputs, a replay mismatch the action generator refuses) are returned in
``skipped_event_ids`` and logged by the learner as skipped, never as trained.
"""
from __future__ import annotations

import collections
import math
import os
import time
from typing import Optional

import torch

from ..interfaces import ToolSchema
from .cam_policy import (InputsStore, alias_map, artifacts_dir, build_cam_model, cam_config, load_unique_state_dict,
                         unique_state_dict, unpack_step_inputs)


def rho_stats(log_rho: torch.Tensor, rho_bar: float = 1.0) -> dict:
    """Truncated importance weights rho = min(rho_bar, exp(log_rho)) of trained decisions: mean, fraction clipped,
    fraction below 0.5, normalised effective sample size (sum rho)^2 / (n sum rho^2)."""
    if log_rho.numel() == 0:
        return {"n": 0}
    r = log_rho.float().clamp(max=30.0).exp()
    rc = r.clamp(max=rho_bar)
    n = rc.numel()
    ess = float(rc.sum() ** 2 / (n * (rc ** 2).sum()).clamp_min(1e-30))
    return {"n": int(n), "rho_mean": float(rc.mean()), "rho_raw_mean": float(r.clamp(max=1e6).mean()),
            "clipped_frac": float((r > rho_bar).float().mean()), "below_half_frac": float((r < 0.5).float().mean()),
            "ess_frac": ess, "log_rho_mean": float(log_rho.float().mean()),
            "log_rho_absmax": float(log_rho.float().abs().max())}


class CamLearnerModel:
    """See the module docstring. Constructor arguments: run_dir (the run directory: artifacts), cfg (`cam_config`
    overrides incl. ``scale``), seed, device, lr / lr_sem (override the config), stage (D's training stage; default
    "continual" = every objective), world_seq_len (world/adapt window), world_objectives (False: policy terms only),
    max_fresh_kl (D's freshness check), recompute_state ("auto" | "always" | "never"), grad_probe_every (D's backbone
    gradient probe), policy_rows (decisions per policy-evaluation call; default 1 = each decision evaluated alone, the
    shape it was acted in, which keeps the lag-0 recomputation exact; larger values batch them at a numerical cost of
    up to ~0.3 nats on long string actions in bf16; the graph of all calls is kept until the single backward)."""

    def __init__(self, run_dir: str, cfg: Optional[dict] = None, seed: int = 0, device: Optional[str] = None,
                 lr: Optional[float] = None, lr_sem: Optional[float] = None, stage: str = "continual",
                 world_seq_len: int = 16, world_objectives: bool = True, max_fresh_kl: float = 0.05,
                 recompute_state: str = "auto", grad_probe_every: int = 10, policy_rows: int = 1,
                 model_kwargs: Optional[dict] = None, fsync: bool = True):
        from ..agent import CamUnifiedAgent
        from ..learn.ppo import ActorCriticObjective
        from ..learn.trainer import CombinedTrainer
        if recompute_state not in ("auto", "always", "never"):
            raise ValueError(recompute_state)
        over = dict(cfg or {})
        scale = over.pop("scale", "dev")
        if lr is not None:
            over["lr"] = float(lr)
        if lr_sem is not None:
            over["lr_sem"] = float(lr_sem)
        self.cfg = cam_config(scale, **over)
        self.cfg_overrides = {"scale": scale, **over}
        t0 = time.time()
        self.model = build_cam_model(self.cfg, device=device, seed=seed, **(model_kwargs or {}))
        self.build_s = time.time() - t0
        m = self.model
        self.aliases = alias_map(m)
        core = [p for p in m.core.parameters() if p.requires_grad]
        ag = m.action_generator
        sem = [p for p in list(ag.lora) + list(ag.bridge.parameters()) if p.requires_grad]
        taken = {id(p) for p in core + sem}
        heads, seen = [], set()
        for p in m.parameters():
            if p.requires_grad and id(p) not in taken and id(p) not in seen:
                heads.append(p)
                seen.add(id(p))
        self.trainer = CombinedTrainer(self.cfg, core=core, heads=heads, semantic=sem, stage=stage, amp=False,
                                       grad_probe_every=grad_probe_every)
        self.objective = ActorCriticObjective.from_config(self.cfg, max_policy_lag=0, max_fresh_kl=max_fresh_kl)
        self.tok_agent = CamUnifiedAgent(m, seed=seed, actor_id="learner")      # tokenise_sequences only
        self.store = InputsStore(artifacts_dir(run_dir), fsync=fsync)
        self.world_seq_len = int(world_seq_len)
        self.world_objectives = bool(world_objectives)
        self.recompute_state = recompute_state
        self.policy_rows = max(1, int(policy_rows))
        self.rng = torch.Generator().manual_seed(int(seed))
        self.counts = collections.Counter()
        self.errors: list = []                     # first refusals of the policy evaluation (diagnostics)
        self.trained_by_actor = collections.Counter()
        self.history: list = []

    # ---------------------------------------------------------------- LearnerModelP
    def state_dict(self) -> dict:
        return {k: v.detach() for k, v in unique_state_dict(self.model).items()}

    def load_state_dict(self, sd: dict) -> None:
        with torch.no_grad():
            load_unique_state_dict(self.model, sd, self.aliases)

    def config(self) -> dict:
        return {"cam_config": self.cfg.to_dict(), "overrides": self.cfg_overrides,
                "params": self.model.count_parameters().get("cam_total"), "learner": "dist.cam_learner.CamLearnerModel",
                "recompute_state": self.recompute_state, "world_seq_len": self.world_seq_len}

    # ---------------------------------------------------------------- helpers
    def _load(self, tr) -> Optional[list]:
        m = self.model
        out = []
        for e in tr.events:
            ex = e.extra or {}
            ref, tref = ex.get("inputs_ref"), ex.get("tools_ref")
            if not ref or not tref:
                self.counts["no_inputs"] += 1
                return None
            try:
                d = self.store.inputs(ref)
                tools = self.store.tools(tref)
            except (OSError, ValueError, KeyError, RuntimeError):
                self.counts["inputs_unreadable"] += 1
                return None
            m.tokenizer.register_tools(tools)
            if d.get("last_tool"):
                m.tokenizer.register_tools([ToolSchema.from_dict(d["last_tool"])])
            out.append((d, tools))
        return out

    @staticmethod
    def _consistent(tr, lv: int) -> bool:
        """Every decision of the piece was produced by version lv from a latent state built under lv."""
        for e in tr.events:
            sv = (e.extra or {}).get("state_version")
            if int(e.model_version) != lv or sv is None or int(sv) != lv:
                return False
        return True

    def _inputs(self, d, e, tools, Z):
        return unpack_step_inputs(d, e.observations, tools, self.model.device, Z_prev=Z)

    def _value_out(self, enc) -> tuple:
        """(value [1], two-hot logits [1, bins] | None) of a one-row StepEncoding."""
        vo = self.model.value_head(enc.R, enc.goal)
        lg = getattr(vo, "logits", None)
        return self.model._value(vo), lg

    # ---------------------------------------------------------------- the update
    def update(self, batch) -> dict:
        from ..learn.trajectory import Trajectory as DTraj
        from ..model import StepInputs, acting_mode
        t_all = time.perf_counter()
        m = self.model
        dev = m.device
        lv = int(batch.learner_version)
        if m.version != lv:
            m.version = lv
        phases = collections.Counter()
        skipped: list = []
        pieces = []                                   # (trajectory, [(inputs dict, tools)])
        t = time.perf_counter()
        for tr in batch.trajectories:
            items = self._load(tr)
            if items is None:
                skipped += tr.trained_ids()
                continue
            pieces.append((tr, items))
        phases["load_s"] += time.perf_counter() - t
        if not pieces:
            return {"optimizer_step": False, "skipped_event_ids": skipped, "reason": "no evaluable trajectory"}
        n = len(pieces)
        # ---- spans (offered decisions after routing) and the bootstrap event after each span
        spans = []
        for tr, _ in pieces:
            pos = [i for i, mk in enumerate(tr.train_mask) if mk]
            if not pos:
                spans.append(None)
                continue
            lo, hi = pos[0], pos[-1]
            boot = hi + 1 if hi + 1 < len(tr.events) else None
            spans.append((lo, hi, boot))
        # ---- 2. latent state under the current weights
        t = time.perf_counter()
        zin = [[None] * len(tr.events) for tr, _ in pieces]
        recompute = []
        for k, (tr, items) in enumerate(pieces):
            rc = self.recompute_state == "always" or (self.recompute_state == "auto" and not self._consistent(tr, lv))
            recompute.append(rc and spans[k] is not None)
            for i, (d, _) in enumerate(items):
                zin[k][i] = d["Z_prev"].to(dev)
        zdev = []
        boot_out = {}                                  # k -> (value [1], logits [1, bins])
        with torch.no_grad(), acting_mode(m), m._autocast():
            # one environment row per forward pass (B = 1, as the actors act): in bf16 a padded batch of several
            # pieces changes Z by ~1e-4, which moves long string actions' log-probabilities by tenths of a nat
            for k in range(n):
                if spans[k] is None:
                    continue
                tr, items = pieces[k]
                last = spans[k][2] if spans[k][2] is not None else spans[k][1]
                if recompute[k]:
                    zc = zin[k][0]
                    pure = True     # every decision so far by version lv from a state built under lv: exact check
                    for i in range(last + 1):
                        zin[k][i] = zc
                        d, tools = items[i]
                        enc = m.encode_step(self._inputs(d, tr.events[i], tools, zc))
                        if i == spans[k][2]:
                            boot_out[k] = self._value_out(enc)
                        ev = tr.events[i]
                        sv = (ev.extra or {}).get("state_version")
                        pure = pure and int(ev.model_version) == lv and sv is not None and int(sv) == lv
                        zc = enc.Z.float()
                        if pure and i + 1 < len(tr.events):
                            zdev.append(float((zc - items[i + 1][0]["Z_prev"].to(dev)).abs().max()))
                elif spans[k][2] is not None:          # stored Z_{t-1} of the bootstrap decision is exact
                    b = spans[k][2]
                    d, tools = items[b]
                    boot_out[k] = self._value_out(m.encode_step(self._inputs(d, tr.events[b], tools, zin[k][b])))
        phases["state_s"] += time.perf_counter() - t
        # ---- 3. policy terms (with gradients) for every decision of every span
        t = time.perf_counter()
        m.train()
        rows = [(k, i) for k in range(n) if spans[k] is not None for i in range(spans[k][0], spans[k][1] + 1)]
        res = {}
        bad = set()

        def evaluate(chunk):
            inps = [self._inputs(pieces[k][1][i][0], pieces[k][0].events[i], pieces[k][1][i][1], zin[k][i].detach())
                    for k, i in chunk]
            acts = [pieces[k][0].events[i].action for k, i in chunk]
            return m.policy_evaluate(StepInputs.cat(inps), acts)

        for c0 in range(0, len(rows), self.policy_rows):
            chunk = rows[c0:c0 + self.policy_rows]
            try:
                out = evaluate(chunk)
                for j, ki in enumerate(chunk):
                    res[ki] = {kk: (None if vv is None or not torch.is_tensor(vv) else vv[j])
                               for kk, vv in out.items()}
            except Exception:                      # noqa: BLE001 — isolate the rows the model refuses
                for ki in chunk:
                    try:
                        out = evaluate([ki])
                        res[ki] = {kk: (None if vv is None or not torch.is_tensor(vv) else vv[0])
                                   for kk, vv in out.items()}
                    except Exception as e:          # noqa: BLE001
                        bad.add(ki)
                        self.counts[f"policy_eval_error:{type(e).__name__}"] += 1
                        if len(self.errors) < 50:
                            import traceback
                            tb = traceback.extract_tb(e.__traceback__)[-2:]
                            where = "; ".join(f"{os.path.basename(f.filename)}:{f.lineno}" for f in tb)
                            self.errors.append(f"{pieces[ki[0]][0].events[ki[1]].event_id}: {type(e).__name__}: "
                                               f"{str(e)[:300]} [{where}]")
        phases["policy_eval_s"] += time.perf_counter() - t
        # ---- D's trajectory rows
        t = time.perf_counter()
        T = max((s[1] - s[0] + 1 for s in spans if s is not None), default=0)
        cfg = self.cfg
        lp_coef = cfg.lp_coef if cfg.use_learned_exploration else 0.0
        z = lambda: torch.zeros(n, T, device=dev)               # noqa: E731
        reward, dt, beh, lp_new, ent = z(), z(), z(), z(), z()
        done = torch.zeros(n, T, dtype=torch.bool, device=dev)
        mask = torch.zeros(n, T, dtype=torch.bool, device=dev)
        mv = torch.full((n, T), lv, dtype=torch.long, device=dev)
        old_v = torch.zeros(n, T + 1, device=dev)
        nb = None
        for r in res.values():
            if r.get("value_logits") is not None:
                nb = int(r["value_logits"].shape[-1])
                break
        vals = [[torch.zeros((), device=dev) for _ in range(T + 1)] for _ in range(n)]
        logits = [[None] * (T + 1) for _ in range(n)]
        ponder = []
        exact_dev, lag0_dev = [], []
        planner_skipped = []
        by_actor = collections.Counter()
        trained_ids = []
        for k, (tr, items) in enumerate(pieces):
            if spans[k] is None:
                continue
            lo, hi, boot = spans[k]
            for c, i in enumerate(range(lo, hi + 1)):
                e = tr.events[i]
                ex = e.extra or {}
                nxt = tr.events[i + 1] if i + 1 < len(tr.events) else None
                lp_t = float((nxt.extra or {}).get("lp_prev", 0.0) or 0.0) if nxt is not None else 0.0
                reward[k, c] = float(e.reward) + lp_coef * lp_t
                dt[k, c] = max(0.0, float(ex.get("dt", 0.0) or 0.0))
                done[k, c] = bool(e.done)
                beh[k, c] = float(e.behavior_logp if e.behavior_logp is not None else 0.0)
                mv[k, c] = int(e.model_version)
                old_v[k, c] = float(ex.get("value", 0.0) or 0.0)
                planner = str((ex.get("behavior") or {}).get("policy", "")).startswith("planner")
                trainable = bool(tr.train_mask[i]) and (k, i) not in bad and not planner
                if planner and tr.train_mask[i]:
                    planner_skipped.append(e.event_id)
                if (k, i) in bad and tr.train_mask[i]:
                    skipped.append(e.event_id)
                r = res.get((k, i))
                if r is not None:
                    lp_new[k, c] = r["logp"]
                    ent[k, c] = r["entropy"]
                    vals[k][c] = r["value"]
                    logits[k][c] = r.get("value_logits")
                    if trainable and r.get("ponder") is not None:
                        ponder.append(r["ponder"])
                    if trainable and int(e.model_version) == lv and e.behavior_logp is not None:
                        dv = abs(float(r["logp"].detach()) - float(e.behavior_logp))
                        lag0_dev.append(dv)
                        if tr.exact and tr.exact[i]:
                            exact_dev.append(dv)
                mask[k, c] = trainable
                if trainable:
                    by_actor[tr.actor_id] += 1
                    trained_ids.append(e.event_id)
            if boot is not None and k in boot_out:
                vals[k][hi - lo + 1] = boot_out[k][0][0]
                logits[k][hi - lo + 1] = None if boot_out[k][1] is None else boot_out[k][1][0]
                old_v[k, hi - lo + 1] = float((tr.events[boot].extra or {}).get("value", 0.0) or 0.0)
        skipped += planner_skipped
        if not trained_ids:
            m.eval()
            return {"optimizer_step": False, "skipped_event_ids": sorted(set(skipped)),
                    "reason": "no decision the policy objective can train", **self._counts()}
        new_value = torch.stack([torch.stack([v.float().reshape(()) for v in row]) for row in vals])
        vlog = None
        if nb is not None:
            zero_l = torch.zeros(nb, device=dev)
            vlog = torch.stack([torch.stack([(l if l is not None else zero_l).float() for l in row]) for row in logits])
        dtraj = DTraj(reward=reward, dt=dt, done=done, behavior_logp=beh, model_version=mv, mask=mask, value=old_v)
        prep = self.objective.prepare(dtraj, lv)
        if vlog is not None:
            ac = self.objective.loss(dtraj, prep, lp_new, ent, value_logits=vlog, value_head=m.value_head)
        else:
            ac = self.objective.loss(dtraj, prep, lp_new, ent, new_value=new_value)
        actor = ac["actor"]
        if ponder and cfg.use_reasoning and cfg.adaptive_reasoning:
            reason_cost = cfg.reason_compute_cost * torch.stack([p.float().reshape(()) for p in ponder]).mean()
            actor = actor + reason_cost
        else:
            reason_cost = None
        losses = {"actor": actor, "value": ac["value"]}
        # V-trace diagnostics on the trained decisions (recorded behaviour logp vs the current policy)
        with torch.no_grad():
            lr_all = (lp_new.detach() - beh)[mask]
            stale = mask & ~prep.fresh.unsqueeze(1)
            vt = rho_stats((lp_new.detach() - beh)[stale], cfg.vtrace_rho)
        phases["actor_critic_s"] += time.perf_counter() - t
        # ---- 4. world / adaptation terms on the same experience
        t = time.perf_counter()
        wm = {}
        if self.world_objectives:
            windows = []
            for k, (tr, _) in enumerate(pieces):
                if spans[k] is None:
                    continue
                lo, hi, _ = spans[k]
                evs = [e for e in tr.events[lo:hi + 1] if e.action is not None]
                for w0 in range(0, len(evs), self.world_seq_len):
                    w = evs[w0:w0 + self.world_seq_len]
                    if len(w) >= 2:
                        windows.append(w)
            if windows:
                seq, aux = self.tok_agent.tokenise_sequences(windows)
                filt = m.filter_sequence(seq, aux)
                wout = m.world_loss(seq, filt)
                if torch.is_tensor(wout.get("world")) and wout["world"].requires_grad:
                    losses["world"] = wout["world"]
                wm.update({f"world/{kk}": float(vv.detach()) for kk, vv in wout.items()
                           if torch.is_tensor(vv) and vv.numel() == 1})
                if cfg.use_fast_adaptation:
                    Tw = len(seq.steps)
                    ks = [kk for kk in (0, 1, 2, 4, 8, 16, 32, 64) if kk < Tw]
                    kk = ks[int(torch.randint(len(ks), (1,), generator=self.rng))]
                    al, am = m.adapt_loss(seq, filt, kk)
                    if al.requires_grad:
                        losses["adapt"] = al
                    wm.update({f"adapt/{a}": b for a, b in am.items()})
                wm["world/windows"] = len(windows)
        phases["world_s"] += time.perf_counter() - t
        # ---- 5. one optimiser step
        t = time.perf_counter()
        bad_loss = [kk for kk, vv in losses.items() if not torch.isfinite(vv).all()]
        if bad_loss:
            m.eval()
            self.counts["nonfinite_losses"] += 1
            return {"optimizer_step": False, "skipped_event_ids": sorted(set(skipped)),
                    "reason": f"non-finite losses {bad_loss}", **self._counts()}
        info = self.trainer.step(losses)
        phases["step_s"] += time.perf_counter() - t
        m.eval()
        if info.skipped:
            self.counts["trainer_skipped"] += 1
            return {"optimizer_step": False, "skipped_event_ids": sorted(set(skipped)),
                    "reason": "trainer skipped a non-finite step", **self._counts()}
        m.version = lv + 1
        enc = m.semantic
        if enc is not None and hasattr(enc, "mark_updated"):
            enc.mark_updated()
        self.trained_by_actor.update(by_actor)
        self.counts["updates"] += 1
        self.counts["policy_trained"] += len(trained_ids)
        phases["total_s"] = time.perf_counter() - t_all
        out = {"optimizer_step": True, "skipped_event_ids": sorted(set(skipped)), "update_id_trainer": info.update_id,
               "loss": info.total, "losses": dict(info.losses), "weighted": dict(info.weighted),
               "grad_norm": info.grad_norm, "group_grad_norm": dict(info.group_grad_norm), "clipped": info.clipped,
               "backbone": info.backbone, "actor": float(ac["actor"].detach()), "value": float(ac["value"].detach()),
               "entropy": ac.get("entropy"), "kl_fresh": ac.get("kl"), "n_fresh_steps": ac.get("n_fresh_steps"),
               "n_stale_steps": ac.get("n_stale_steps"), "rerouted_by_kl": ac.get("rerouted_by_kl"),
               "fresh_kl_max": ac.get("fresh_kl_max"), "vtrace_rho_mean": ac.get("vtrace_rho_mean"),
               "vtrace": vt, "log_rho_all": rho_stats(lr_all, cfg.vtrace_rho),
               "reason_compute": None if reason_cost is None else float(reason_cost.detach()),
               "mean_reward": float(reward[mask].mean()) if bool(mask.any()) else None,
               "policy_trained_by_actor": dict(by_actor), "n_policy_trained": len(trained_ids),
               "n_pieces": n, "n_recomputed_pieces": int(sum(recompute)),
               "exact_logp_dev_max": max(exact_dev) if exact_dev else None, "n_exact_checked": len(exact_dev),
               "lag0_logp_dev_max": max(lag0_dev) if lag0_dev else None,
               "lag0_logp_dev_mean": (sum(lag0_dev) / len(lag0_dev)) if lag0_dev else None,
               "lag0_logp_within_1e-3": (sum(1 for x in lag0_dev if x < 1e-3) / len(lag0_dev)) if lag0_dev else None,
               "z_recompute_dev_max": max(zdev) if zdev else None, "planner_skipped": len(planner_skipped),
               "phases": dict(phases), **self._counts(), "gpu_peak_mb": (torch.cuda.max_memory_allocated() / 2 ** 20
                                                       if dev.type == "cuda" else 0.0), **wm}
        self.history.append({k: v for k, v in out.items() if k not in ("skipped_event_ids", "backbone")})
        return out

    def _counts(self) -> dict:
        return {"model_counts": dict(self.counts), "model_errors": list(self.errors[-5:])}


def make_cam_learner(**kw) -> CamLearnerModel:
    return CamLearnerModel(**kw)


def finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False
