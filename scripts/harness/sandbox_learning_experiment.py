"""Full autonomous sandbox practice loop through the internet-agent harness (integration test 18,
build phases 10 and 13).

Everything runs as separate services talking over real HTTP on localhost:
  * SocialWebLab (self-hosted multi-user app, virtual clock) — the sandbox environment;
  * the shared inference gateway (uvicorn) serving the Pythia-14M content policy to all agents;
  * the control plane on PostgreSQL + Redis (when URLs are given) or SQLite;
  * N agent runtimes (asyncio, concurrent) acting only through the tool gateway, sharing one
    programmatically provisioned sandbox account;
  * the outcome worker (follow-ups on horizons), the verification service (helpful answer within
    24 h, from the lab's privileged truth) and the learner (delayed-reward RL on a training copy,
    candidate checkpoints, held-out evaluation in identical lab snapshots, gated promotion).

Measured: held-out success of the initial (imitation) checkpoint vs the final production
checkpoint on the same tasks and lab state, plus throughput, reliability and compute.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from agentrs.agent.belief import BeliefState  # noqa: E402
from agentrs.checkpoints.registry import CheckpointRegistry, PromotionGate  # noqa: E402
from agentrs.content.policy import ContentGenConfig, ContentGeneratingPolicy, generation_prompt, op_text  # noqa: E402
from agentrs.content.training import ContentDemo, ContentRLConfig, content_bc_loss  # noqa: E402
from agentrs.evaluation.metrics import bootstrap_ci  # noqa: E402
from agentrs.harness.db import Experience  # noqa: E402
from agentrs.harness.inference import BatchingInferenceServer, InferenceClient, LocalPolicyBackend, create_inference_app  # noqa: E402
from agentrs.harness.learning import HarnessLearner  # noqa: E402
from agentrs.harness.runtime import AgentRuntime, SocialWebLabHelpAdapter  # noqa: E402
from agentrs.harness.sandbox import build_sandbox_harness, sandbox_fetchers  # noqa: E402
from agentrs.harness.tasks import PracticeTaskGenerator  # noqa: E402
from agentrs.models.backbone import BackboneConfig  # noqa: E402
from agentrs.models.policy import LMPolicy, PolicyConfig  # noqa: E402
from agentrs.socialweblab.agent_tasks import HelpTeacher, SocialWebLabContentEnv, make_help_task  # noqa: E402
from agentrs.socialweblab.engine import SocialWebLab  # noqa: E402
from agentrs.socialweblab.server import SocialWebLabServer  # noqa: E402
from agentrs.training.common import GradientTrainer, OptimConfig  # noqa: E402
from agentrs.training.delayed_reward_trainer import DelayedRewardConfig  # noqa: E402

DAY = 86400.0


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_inference(server: BatchingInferenceServer, port: int):
    import uvicorn

    cfg = uvicorn.Config(create_inference_app(server, admin_token="exp-admin"), host="127.0.0.1", port=port,
                         log_level="warning")
    srv = uvicorn.Server(cfg)
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    while not srv.started:
        time.sleep(0.05)
    return srv


def imitation(cp, lab, seed, n_demos, steps, log):
    """Behaviour cloning on a scripted author's posts (authorised demonstrations, provenance: teacher)."""
    env = SocialWebLabContentEnv(lab)
    teacher = HelpTeacher(seed=seed)
    demos = []
    for i in range(n_demos):
        t = make_help_task("train", 20000 + i, account="demo-author")
        obs = env.reset(t)
        b = BeliefState(task=t.public())
        b.reset(obs)
        cands = [c for c in env.available_actions() if c.type.value != "finish"]
        op, content = teacher.write(t)
        k = [c.name for c in cands].index(op)
        demos.append(ContentDemo(b.prompt(), [op_text(c) for c in cands], k, generation_prompt(b.prompt(), op),
                                 cp.demo_token_ids(content), 1.0))
    opt = GradientTrainer(cp.policy, OptimConfig(lr=5e-4, total_steps=steps, warmup_steps=20, schedule="cosine"))
    rng = random.Random(seed)
    cp.policy.train()
    for _ in range(steps):
        loss, st = content_bc_loss(cp, rng.sample(demos, 16))
        opt.step(loss)
    cp.policy.eval()
    log["imitation"] = {"demos": n_demos, "steps": steps, "last_loss": float(loss.detach()), **st,
                        "provenance": "scripted teacher (authorised demonstrations)"}


async def run_tasks(rts, h, gen, split, n_groups, group, start, checkpoint=None, attempt_base=0, lab=None,
                    gap_s=600.0):
    """Agents work in rounds (one task each, concurrently); simulated time passes between rounds, so the
    agents act over time like real users (and the lab's per-session rate limits refill)."""
    for i in range(n_groups):
        for k in range(group):
            gen.generate(split, start + i, attempt=attempt_base + k)
    out = []
    while True:
        res = await asyncio.gather(*(rt.loop(1, environments=["socialweblab"], split=split, checkpoint=checkpoint)
                                     for rt in rts))
        got = [x for r in res for x in r]
        if not got:
            return out
        out += got
        if lab is not None:
            lab.advance(gap_s)


async def advance(h, lab, seconds, step=1800.0):
    end = lab.now() + seconds
    stats = {"done": 0, "retry": 0, "failed": 0}
    while lab.now() < end - 1e-6:
        lab.advance(min(step, end - lab.now()))
        st = await h.outcomes.run_due(sandbox_fetchers(h), {lab.time_base: lab.now()})
        for k in stats:
            stats[k] += st[k]
    return stats


def success(h, split, checkpoint, since):
    with h.db.session() as db:
        ex = [e for e in db.query(Experience).all() if e.split == split and e.model_checkpoint == checkpoint
              and e.created_at >= since and e.reward_status == "matured"]
    r = [float(e.reward) for e in ex]
    return {"n": len(r), "success_rate": float(np.mean(r)) if r else 0.0, "ci95": list(bootstrap_ci(r)) if r else None}


async def main_async(a):
    T0 = time.time()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    torch.manual_seed(a.seed)
    log: dict = {"args": vars(a), "cycles": []}
    lab = SocialWebLab(out / "lab.db", seed=100 + a.seed, n_users=80, lab_id=f"harness{a.seed}")
    lab.advance(DAY)
    srv = SocialWebLabServer(lab)
    srv.start()
    agents = tuple(f"sandbox_agent_{i:03d}" for i in range(1, a.agents + 1))
    db_url = a.db_url or f"sqlite:///{out}/harness.db"
    if a.db_url and a.db_url.startswith("postgresql"):
        from sqlalchemy import create_engine, text

        name = f"harness_exp_{int(time.time())}"
        with create_engine(a.db_url.rsplit("/", 1)[0] + "/postgres", isolation_level="AUTOCOMMIT").connect() as c:
            c.execute(text(f"CREATE DATABASE {name}"))
        db_url = a.db_url.rsplit("/", 1)[0] + "/" + name
    h = build_sandbox_harness(out / "harness", lab, srv, db_url, agents, redis_url=a.redis_url)
    log["backends"] = {"database": "postgresql" if h.db.is_postgres else "sqlite",
                       "rate_limiter": "redis" if h.policy.limiter.r is not None else "memory"}
    # ---------------------------------------------------------------- policy: imitation, then served
    t = time.time()
    if a.model == "scratch":
        from agentrs.models.backbone import ScratchSettings

        bb = BackboneConfig(name="scratch", scratch=ScratchSettings(hidden_size=48, num_layers=2, num_heads=4,
                                                                     vocab_size=512), max_prompt_tokens=384, seed=a.seed)
    else:
        bb = BackboneConfig(name=a.model, max_prompt_tokens=320, seed=a.seed)
    cp = ContentGeneratingPolicy(LMPolicy(PolicyConfig(backbone=bb)), ContentGenConfig(max_content_tokens=64),
                                 checkpoint_id="bc-v0")
    imitation(cp, lab, a.seed, a.bc_demos, a.bc_steps, log)
    log["imitation"]["wall_s"] = time.time() - t
    reg = CheckpointRegistry(out / "registry.sqlite")
    prod_rid = reg.register("policy", "bc-v0 (in memory)", metrics={"source": "imitation"}, notes="bc-v0")
    reg.promote(prod_rid, "initial imitation checkpoint")
    backend = LocalPolicyBackend({"bc-v0": cp}, "bc-v0", seed=a.seed)
    inf_server = BatchingInferenceServer(backend, max_batch=16, max_wait_ms=25)
    port = free_port()
    uv = start_inference(inf_server, port)
    client = InferenceClient(base_url=f"http://127.0.0.1:{port}")  # agents use the HTTP service
    adapter = SocialWebLabHelpAdapter(lab.now, lab.time_base)
    rts = [AgentRuntime(ag, h.gateway, client, h.tasks, h.provisioning, h.verification, {"socialweblab": adapter})
           for ag in agents]
    # one shared, programmatically provisioned sandbox account for all agents
    req = h.provisioning.request("socialweblab", ["create_post", "read_content"], "sandbox practice", agents[0])
    req = await h.provisioning.verify(req.id)
    acc = h.accounts.get(req.account_id)
    h.accounts.update(acc.id, allowed_agents=list(agents))
    log["account"] = {"provisioning_state": req.state, "history": [x["to"] for x in req.history]}
    learner = HarnessLearner(out / "learner", h.outcomes.ledger, cp, reg, DelayedRewardConfig(
        objective="task:socialweblab_help_within", max_staleness=50,
        rl=ContentRLConfig(optim=OptimConfig(lr=3e-5, total_steps=10_000), minibatch_size=16, kl_coef=0.02, epochs=2)))
    gen = PracticeTaskGenerator(h.tasks)
    eval_round = {"n": 0}

    async def evaluate(ck, snap, split="val", n_tasks=None):
        """Held-out tasks in a throwaway copy of the current lab state (same state for every arm)."""
        from agentrs.harness.db import ActionLog, Task

        srv.control("restore", path=str(snap))
        since = time.time()
        eval_round["n"] += 1
        await run_tasks(rts, h, gen, split, n_tasks or a.eval_tasks, 1, 0, checkpoint=ck,
                        attempt_base=10_000 * eval_round["n"], lab=lab)
        lab.advance(DAY + 3600)
        h.verification.run()
        with h.db.session() as db:  # this copy of the lab is discarded: stop following its posts
            ids = [x.id for x in db.query(ActionLog).join(Task, Task.id == ActionLog.task_id)
                   .filter(Task.split == split, ActionLog.proposed_at >= since).all()]
        for i in ids:
            h.outcomes.scheduler.cancel(i, "evaluation copy discarded")
        return success(h, split, ck, since)

    async def compare(cks, split, n_tasks):
        snap = out / f"snap_{eval_round['n']}.db"
        srv.control("snapshot", path=str(snap))
        res = {ck: await evaluate(ck, snap, split, n_tasks) for ck in cks}
        srv.control("restore", path=str(snap))  # back to the practice timeline
        return res

    # ---------------------------------------------------------------- practice cycles
    t_rl = time.time()
    task_ix = 0
    for cyc in range(a.cycles):
        tc = time.time()
        recs = await run_tasks(rts, h, gen, "train", a.groups_per_cycle, a.group, task_ix, lab=lab)
        task_ix += a.groups_per_cycle
        fol = await advance(h, lab, DAY / 2)
        ver = h.verification.run()
        cand = learner.train_candidate()
        c = {"cycle": cyc, "tasks": len(recs), "executed": sum(r["status"] == "executed" for r in recs),
             "max_batch": max(r["batch_size"] for r in recs), "followups": fol, "verification": ver,
             "update": {k: v for k, v in cand.items() if k != "candidate"}, "production": backend.production,
             "wall_s": time.time() - tc}
        if cand.get("candidate") is not None:
            backend.add(cand["checkpoint_id"], cand["candidate"])
            if (cyc + 1) % a.promote_every == 0:
                # held-out evaluation in identical lab copies, then the gate (never promoted on training loss)
                m = await compare([cand["checkpoint_id"], backend.production], "val", a.eval_tasks)
                m_c, m_p = m[cand["checkpoint_id"]], m[backend.production]
                ok, why = PromotionGate(metric="success_rate", min_improvement=a.min_improvement).decide(m_c, m_p)
                reg.update_metrics(cand["registry_id"], {"heldout": m_c, "production_heldout": m_p})
                if ok:
                    reg.promote(cand["registry_id"], why)
                    backend.production = cand["checkpoint_id"]
                    learner.production = cand["candidate"]
                else:
                    reg.reject(cand["registry_id"], why)
                c["promotion"] = {"candidate": m_c, "production": m_p, "promoted": ok, "reason": why}
        log["cycles"].append(c)
        print(json.dumps({k: c[k] for k in ("cycle", "executed", "max_batch", "verification", "production")} |
                         {"mean_reward": c["update"].get("mean_reward"), "promotion": c.get("promotion")}), flush=True)
    await advance(h, lab, DAY + 3600)
    h.verification.run()
    log["rl_wall_s"] = time.time() - t_rl
    # ---------------------------------------------------------------- final held-out test: initial vs final
    t = time.time()
    final_ck = backend.production
    last = max((k for k in backend.models if k.startswith("harness-policy-v")), default=None,
               key=lambda k: int(k.rsplit("v", 1)[1]))
    arms = ["bc-v0"] + [k for k in dict.fromkeys([final_ck, last]) if k and k != "bc-v0"]
    res = await compare(arms, "test", a.test_tasks)
    log["final"] = {"initial_checkpoint": "bc-v0", "final_production": final_ck, "last_candidate": last,
                    "results": res, "wall_s": time.time() - t}
    # ---------------------------------------------------------------- throughput / reliability / compute
    from agentrs.harness.db import ActionLog

    with h.db.session() as db:
        acts = db.query(ActionLog).all()
        st = {}
        for x in acts:
            st[x.status] = st.get(x.status, 0) + 1
    log["throughput"] = {"wall_s": time.time() - T0, "gateway_actions": len(acts), "action_status": st,
                         "inference": inf_server.stats, "followup_scheduler": h.outcomes.scheduler.stats(),
                         "ledger": h.outcomes.ledger.stats(), "egress_requests": len(h.gateway.egress_log),
                         "egress_denied": sum(e.get("outcome") == "denied" for e in h.gateway.egress_log),
                         "cpu_cores": torch.get_num_threads(), "gpu_hours": 0.0,
                         "registry": [{"id": r.id, "status": r.status, "notes": r.notes, "metrics": r.metrics}
                                      for r in reg.list("policy")]}
    (ROOT / "results" / "harness").mkdir(parents=True, exist_ok=True)
    name = a.results_name or f"sandbox_learning_s{a.seed}.json"
    dest = Path(name) if name.startswith("/") else ROOT / "results" / "harness" / name
    dest.write_text(json.dumps(log, indent=1, default=str))
    print(json.dumps(log["final"], default=str))
    uv.should_exit = True
    srv.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--cycles", type=int, default=12)
    ap.add_argument("--groups-per-cycle", type=int, default=8)
    ap.add_argument("--group", type=int, default=3)
    ap.add_argument("--promote-every", type=int, default=4)
    ap.add_argument("--min-improvement", type=float, default=0.0)
    ap.add_argument("--eval-tasks", type=int, default=32)
    ap.add_argument("--test-tasks", type=int, default=72)
    ap.add_argument("--bc-demos", type=int, default=400)
    ap.add_argument("--bc-steps", type=int, default=400)
    ap.add_argument("--db-url", default=None)
    ap.add_argument("--redis-url", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--model", default="EleutherAI/pythia-14m")
    ap.add_argument("--results-name", default=None)
    ap.add_argument("--out", default=str(ROOT / "runs" / "harness_experiment"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
