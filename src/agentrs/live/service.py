"""Live observational learner: forecast new public posts, watch what happens, update the weights.

One loop, run indefinitely (``python -m agentrs.live run``):

1. **sense** – every ``ingest_every_s``: new posts from each source (Hacker News new stories,
   a random sample of new Bluesky posts). Nothing older than the start of the service is read.
2. **forecast before outcome** – each new post gets a forecast from the current weights
   (P(first reply / any like / 10 / 100 engagements / removal by 10 min, 1 h, 6 h, 1 d, 3 d) and
   expected counts) and one from the stratified baseline, stored immutably with the checkpoint id.
3. **observe** – durable follow-up jobs read each post's public state at those real horizons
   (Bluesky batched 25 posts per request); deletions are propagated (stored text redacted).
4. **learn** – every ``learn_every_s``: posts whose labels changed become censoring-aware examples
   (initial forecasts and forecasts conditioned on what was already observed), mixed with replay,
   and a few gradient steps update the world model in place. Forecasts made after an update use it.
5. **score** – once an outcome at horizon h is final, the stored forecasts are scored against it
   (Brier, log loss, AUROC; counts: log-MAE) — model vs baseline, overall and over time.
6. **persist** – weights + optimizer state every ``checkpoint_every_s`` (atomic) and on shutdown;
   on restart everything resumes from the checkpoint, the ledger and the job queue.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import secrets
import shutil
import signal
import time
from pathlib import Path
from typing import Optional

import torch
from pydantic import BaseModel, ConfigDict, Field

from agentrs.live.prequential import PrequentialScorer, PrequentialStore, StratifiedBaseline
from agentrs.live.spec import live_horizon_config, live_spec
from agentrs.models.backbone import BackboneConfig
from agentrs.sensors.observation_network import ForecastService, NetworkConfig, ObservationSensorNetwork
from agentrs.sensors.source_registry import SourceRegistry
from agentrs.temporal.delayed_outcome_ledger import DelayedOutcomeLedger
from agentrs.temporal.temporal_world_model import TemporalWorldModel, TemporalWorldModelConfig
from agentrs.training.common import OptimConfig, load_weights, read_meta
from agentrs.training.temporal_world_model_trainer import (
    TemporalTrainConfig,
    TemporalTrainingDataBuilder,
    TemporalWorldModelTrainer,
)

log = logging.getLogger("agentrs.live")


class Target(BaseModel):
    """A goal for the learner, e.g. Brier skill >= 0.10 on Bluesky over >= 2000 scored outcomes."""

    model_config = ConfigDict(extra="forbid")
    source: str
    metric: str = "skill"  # skill (1 - Brier/Brier_baseline) | auroc_model
    min_value: float
    min_n: int = 1000
    window_s: float = 86400.0  # only forecasts made in the last window count


class LiveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workdir: str = "runs/live"
    sources: list[str] = Field(default_factory=lambda: ["hackernews", "bluesky"])
    bluesky_per_min: float = 20.0
    bluesky_langs: list[str] = Field(default_factory=lambda: ["en"])
    horizons: list[str] = Field(default_factory=lambda: ["10m", "1h", "6h", "1d", "3d"])
    outcome_window: str = "4d"
    model: str = "EleutherAI/pythia-14m"
    max_prompt_tokens: int = 320
    backbone: Optional[dict] = None  # full BackboneConfig override (tests use a tiny scratch trunk)
    n_heads: int = 5
    lr: float = 1e-4
    weight_decay: float = 0.1
    batch_size: int = 16
    steps_per_update: int = 60
    max_steps_per_update: int = 200
    min_new_examples: int = 16
    replay_cap: int = 20000
    ingest_every_s: float = 30.0
    observe_every_s: float = 30.0
    learn_every_s: float = 120.0
    score_every_s: float = 300.0
    checkpoint_every_s: float = 600.0
    snapshot_every_s: float = 3600.0
    keep_snapshots: int = 48
    retention_every_s: float = 3600.0
    threads: int = 4
    seed: int = 0
    updated_forecasts: bool = False
    targets: list[Target] = Field(default_factory=list)
    stop_when_targets_met: bool = False


class BaselineForecastService(ForecastService):
    """Batched model forecasts, plus a baseline forecast recorded at the same moment for each post."""

    def __init__(self, model, ledger, baseline: StratifiedBaseline, **kw):
        super().__init__(model, ledger, **kw)
        self.baseline = baseline

    def flush(self):
        with self.lock:
            pending = [(c[0], c[2], c[3], c[4]) for c in self.q if c[5] == "initial"]
        out = super().flush()
        for aid, hs, created, cutoff in pending:
            a = self.ledger.action(aid)
            if a is not None:
                self.ledger.record_forecast(aid, self.baseline.predict(a, hs), "baseline-stratified",
                                            info_cutoff_time=cutoff, created_time=created, horizons=hs, kind="baseline")
        return out


def pseudonym_key(workdir: Path) -> str:
    """Per-deployment secret for actor pseudonyms (never committed; file mode 600)."""
    env = os.environ.get("AGENTRS_PSEUDONYM_KEY")
    if env:
        return env
    p = workdir / "pseudonym.key"
    if not p.exists():
        p.write_text(secrets.token_hex(32))
        p.chmod(0o600)
    return p.read_text().strip()


def build_sources(cfg: LiveConfig, key: str) -> list:
    out = []
    for s in cfg.sources:
        if s == "hackernews":
            from agentrs.live.hackernews import HackerNewsSource

            out.append(HackerNewsSource(pseudonym_key=key))
        elif s == "bluesky":
            from agentrs.live.bluesky import BlueskySource, JetstreamSampler

            out.append(BlueskySource(sampler=JetstreamSampler(cfg.bluesky_per_min, cfg.bluesky_langs, seed=cfg.seed),
                                     pseudonym_key=key))
        elif s == "reddit":
            from agentrs.live.reddit import reddit_source_from_env

            out.append(reddit_source_from_env(key))  # refuses without explicit Reddit authorisation
        else:
            raise ValueError(f"unknown source {s}")
    return out


class LiveLearner:
    def __init__(self, cfg: LiveConfig, sources: Optional[list] = None):
        self.cfg = cfg
        self.dir = Path(cfg.workdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(cfg.threads)
        torch.manual_seed(cfg.seed)
        self.spec = live_spec()
        self.ckpt = self.dir / "checkpoints" / "latest"
        self.model = TemporalWorldModel(TemporalWorldModelConfig(
            backbone=BackboneConfig(**cfg.backbone) if cfg.backbone else
            BackboneConfig(name=cfg.model, max_prompt_tokens=cfg.max_prompt_tokens, seed=cfg.seed),
            n_heads=cfg.n_heads, spec=self.spec))
        self.trainer = TemporalWorldModelTrainer(self.model, TemporalTrainConfig(
            batch_size=cfg.batch_size, progressive_steps=cfg.steps_per_update, progressive_min_new=cfg.min_new_examples,
            optim=OptimConfig(lr=cfg.lr, weight_decay=cfg.weight_decay, schedule="constant"), seed=cfg.seed),
            name="live-wm")
        self.state = {"started_wall": time.time(), "updates": 0, "learn_seconds": 0.0, "last_learn": None,
                      "resumed_from": None, "learned_through": 0.0}
        self._restore()
        self.model.eval()
        self.ledger = DelayedOutcomeLedger(self.dir / "ledger.sqlite")
        self.store = PrequentialStore(self.dir / "prequential.sqlite")
        self.baseline = StratifiedBaseline(self.store, self.spec)
        self.baseline.fit()
        self.scorer = PrequentialScorer(self.ledger, self.spec, self.store)
        self.forecaster = BaselineForecastService(self.model, self.ledger, self.baseline, batch=32)
        reg = SourceRegistry(self.dir / "sources.json")
        for src in sources if sources is not None else build_sources(cfg, pseudonym_key(self.dir)):
            reg.register(src, coverage_tags=[src.declaration.source_id, "social", "create_post"])
        self.net = ObservationSensorNetwork(
            self.dir, reg, self.forecaster, NetworkConfig(scheduling="fixed", updated_forecasts=cfg.updated_forecasts,
                                                          max_updated_per_action=2),
            horizon_cfg=live_horizon_config(cfg.horizons, cfg.outcome_window), ledger=self.ledger)
        self.builder = TemporalTrainingDataBuilder(self.ledger, self.spec)
        self._retrack()
        self._rebuild_replay()
        self.timers = {k: 0.0 for k in ("ingest", "observe", "learn", "score", "ckpt", "snap", "retention")}
        self.stop = False
        self.errors: list[str] = []

    # ------------------------------------------------------------------ persistence
    def _restore(self) -> None:
        if not self.ckpt.exists():
            return
        meta = read_meta(self.ckpt)
        load_weights(self.ckpt, self.model)
        self.trainer.load_optimizer(self.ckpt)
        ex = meta.get("extra", {})
        self.trainer.version = int(ex.get("version", 0))
        self.trainer.examples_seen = int(ex.get("examples_seen", 0))
        self.model.checkpoint_id = ex.get("checkpoint_id", self.model.checkpoint_id)
        self.state.update({k: ex[k] for k in ("updates", "learn_seconds", "learned_through") if k in ex})
        self.state["resumed_from"] = self.model.checkpoint_id
        log.info("resumed weights %s (step %d)", self.model.checkpoint_id, self.trainer.opt.global_step)

    def save(self, snapshot: bool = False) -> Path:
        p = self.trainer.save(self.ckpt, extra={"version": self.trainer.version, "updates": self.state["updates"],
                                                "learn_seconds": self.state["learn_seconds"],
                                                "learned_through": self.state["learned_through"],
                                                "saved_wall": time.time()})
        if snapshot:
            snaps = self.dir / "checkpoints" / "snapshots"
            snaps.mkdir(parents=True, exist_ok=True)
            dst = snaps / time.strftime("%Y%m%d-%H%M%S", time.gmtime())
            shutil.copytree(p, dst, ignore=shutil.ignore_patterns("trainer.pt"))
            for old in sorted(snaps.iterdir())[:-self.cfg.keep_snapshots]:
                shutil.rmtree(old, ignore_errors=True)
        return p

    def _retrack(self) -> None:
        """After a restart, keep propagating deletions for posts still being followed."""
        for sid, ad in self.net.registry.adapters.items():
            sampler = getattr(ad, "sampler", None)
            if sampler is None:
                continue
            for r in self.ledger._q("SELECT action_id, context_json FROM actions WHERE source_id=? AND status='open'", (sid,)):
                ref = json.loads(r["context_json"] or "{}").get("object_ref")
                if ref and ref.startswith("at://"):
                    sampler.tracked[ref] = ref.split("/")[2]

    def _rebuild_replay(self) -> None:
        rows = self.ledger._q("SELECT action_id FROM observations GROUP BY action_id ORDER BY MAX(ingest_wall) DESC LIMIT ?",
                              (self.cfg.replay_cap // 3,))
        if rows:
            self.trainer.replay = self.builder.build(as_of={"real": time.time()}, action_ids=[r["action_id"] for r in rows],
                                                     cutoffs="observations", max_cutoffs=2)

    # ------------------------------------------------------------------ loop steps
    def ingest(self) -> dict:
        out = {}
        for sid in self.net.registry.active():
            try:
                out[sid] = self.net.ingest(sid)["new_actions"]
            except Exception as e:  # noqa: BLE001 - a failing source must not stop the others
                self._err(f"ingest {sid}: {type(e).__name__}: {e}")
        return out

    def observe(self) -> int:
        now = time.time()
        for sid, ad in self.net.registry.adapters.items():
            if hasattr(ad, "prefetch"):
                rows = self.net.scheduler._q(
                    "SELECT j.action_id FROM jobs j WHERE j.source_id=? AND j.status='pending' AND j.next_attempt_time<=?",
                    (sid, now))
                refs = []
                for r in rows:
                    a = self.ledger.action(r["action_id"])
                    ref = ((a or {}).get("context") or {}).get("object_ref")
                    if ref:
                        refs.append(ref)
                try:
                    ad.prefetch(refs)
                except Exception as e:  # noqa: BLE001
                    self._err(f"prefetch {sid}: {type(e).__name__}: {e}")
        n = self.net.observe()
        self.net.close_windows()
        return n

    def learn(self) -> dict:
        t0 = time.perf_counter()
        since = self.state["learned_through"]
        now = time.time()
        changed = {r["action_id"] for r in self.ledger._q(
            "SELECT DISTINCT action_id FROM observations WHERE ingest_wall>? AND ingest_wall<=? "
            "UNION SELECT DISTINCT action_id FROM events WHERE ingest_wall>? AND ingest_wall<=?", (since, now, since, now))}
        if not changed:
            return {"trained": False, "new_examples": 0}
        n_new_est = len(changed)
        steps = int(min(self.cfg.max_steps_per_update,
                        max(self.cfg.steps_per_update, math.ceil(2 * n_new_est / max(1, self.cfg.batch_size // 2)))))
        res = self.trainer.progressive_update(self.builder, as_of={"real": now}, changed=sorted(changed), steps=steps,
                                              cutoffs="observations", max_cutoffs=2)
        if res.get("trained"):
            self.state["learned_through"] = now
            self.state["updates"] += 1
            if len(self.trainer.replay) > self.cfg.replay_cap:
                rng = random.Random(self.trainer.opt.global_step)
                self.trainer.replay = rng.sample(self.trainer.replay, self.cfg.replay_cap)
        dt = time.perf_counter() - t0
        self.state["learn_seconds"] += dt
        res["seconds"] = round(dt, 2)
        self.state["last_learn"] = {k: v for k, v in res.items() if k != "checkpoint"} | {
            "checkpoint": self.model.checkpoint_id, "wall": now}
        return res

    def score(self) -> dict:
        res = self.scorer.update()
        self.baseline.fit()
        return res

    def _err(self, msg: str) -> None:
        log.warning(msg)
        self.errors = (self.errors + [f"{time.strftime('%H:%M:%S')} {msg}"])[-20:]

    # ------------------------------------------------------------------ status / goals
    def targets_met(self, report_window: dict[float, dict]) -> list[dict]:
        out = []
        for t in self.cfg.targets:
            r = report_window[t.window_s]["overall"].get(t.source, {})
            v = r.get(t.metric)
            out.append({**t.model_dump(), "value": v, "n": r.get("n", 0),
                        "met": v is not None and r.get("n", 0) >= t.min_n and v >= t.min_value})
        return out

    def status(self, full: bool = True) -> dict:
        now = time.time()
        windows = sorted({t.window_s for t in self.cfg.targets} | {86400.0})
        rep_win = {w: self.scorer.report(since_forecast_time=now - w, ci=False) for w in windows}
        led = self.ledger._q("SELECT source_id, COUNT(*) n, SUM(training_eligible) e FROM actions GROUP BY source_id")
        st = {
            "now": now, "uptime_s": round(now - self.state["started_wall"]), "checkpoint": self.model.checkpoint_id,
            "optimizer_steps": self.trainer.opt.global_step, "updates": self.state["updates"],
            "learn_seconds": round(self.state["learn_seconds"], 1), "last_learn": self.state["last_learn"],
            "resumed_from": self.state["resumed_from"], "replay": len(self.trainer.replay),
            "posts": {r["source_id"]: {"tracked": r["n"], "eligible": r["e"]} for r in led},
            "observations": self.ledger._q("SELECT COUNT(*) n FROM observations")[0]["n"],
            "jobs": self.net.scheduler.stats(), "network": self.net.stats,
            "sources": {sid: {"requests": getattr(ad, "requests", None),
                              **({"sampler": ad.sampler.stats, "keep_prob": round(ad.sampler.keep_prob(), 4),
                                  "eligible_per_min": round(ad.sampler.rate_per_min, 1), "stream_alive": ad.sampler.alive()}
                                 if hasattr(ad, "sampler") else {}), **getattr(ad, "stats", {})}
                        for sid, ad in self.net.registry.adapters.items()},
            "governance": len(self.ledger.governance_log()),
            "targets": self.targets_met(rep_win), "errors": self.errors[-5:]}
        if full:
            st["scores_all_time"] = self.scorer.report(ci=True)
            st["scores_last_24h"] = rep_win[86400.0]
            st["trend_hourly"] = self.scorer.trend(3600.0)
        return st

    def write_status(self) -> dict:
        st = self.status()
        tmp = self.dir / "status.json.tmp"
        tmp.write_text(json.dumps(st, indent=1, default=str))
        os.replace(tmp, self.dir / "status.json")
        return st

    # ------------------------------------------------------------------ main loop
    def tick(self, now: Optional[float] = None) -> dict:
        now = now or time.time()
        did = {}
        c = self.cfg

        def due(k, every):
            if now - self.timers[k] >= every:
                self.timers[k] = now
                return True
            return False

        if due("ingest", c.ingest_every_s):
            did["ingest"] = self.ingest()
        if due("observe", c.observe_every_s):
            try:
                did["observe"] = self.observe()
            except Exception as e:  # noqa: BLE001
                self._err(f"observe: {type(e).__name__}: {e}")
        if due("learn", c.learn_every_s):
            try:
                did["learn"] = self.learn()
            except Exception as e:  # noqa: BLE001
                self._err(f"learn: {type(e).__name__}: {e}")
        if due("score", c.score_every_s):
            did["score"] = self.score()
            st = self.write_status()
            did["targets"] = st["targets"]
        if due("ckpt", c.checkpoint_every_s) and self.state["updates"]:
            self.save(snapshot=due("snap", c.snapshot_every_s))
        if due("retention", c.retention_every_s):
            did["purged"] = self.net.enforce_retention()
        return did

    def run(self, max_seconds: Optional[float] = None) -> None:
        t_end = time.time() + max_seconds if max_seconds else None
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, lambda *_: setattr(self, "stop", True))
            except ValueError:  # not in the main thread
                pass
        self.timers["ckpt"] = self.timers["snap"] = time.time()
        log.info("live learner started: sources=%s checkpoint=%s", self.net.registry.active(), self.model.checkpoint_id)
        try:
            while not self.stop and (t_end is None or time.time() < t_end):
                did = self.tick()
                if did:
                    log.info("%s", json.dumps({k: v for k, v in did.items() if k != "targets"}, default=str))
                if self.cfg.stop_when_targets_met and did.get("targets") and all(t["met"] for t in did["targets"]):
                    log.info("all targets met: %s", did["targets"])
                    break
                time.sleep(1.0)
        finally:
            for ad in self.net.registry.adapters.values():
                if hasattr(ad, "sampler"):
                    ad.sampler.stop()
            self.score()
            self.write_status()
            if self.state["updates"]:
                self.save()
                log.info("stopped; weights saved at %s", self.ckpt)
            else:
                log.info("stopped before the first weight update (nothing to save)")
