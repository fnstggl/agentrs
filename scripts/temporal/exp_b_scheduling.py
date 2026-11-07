"""Experiment B: fixed vs event-driven vs adaptive observation scheduling.

Four sensor networks observe the SAME lab run concurrently (observer reads never change the lab,
so all four see identical histories):
  fixed      event feed + polls at every profile horizon (8 per action)
  poll_only  polls at every profile horizon, no event feed (events only interval-censored)
  event      event feed (exact event times) + polls only at 1 h, 1 d and the window end
  adaptive   event feed + polls chosen from each action's stored forecast (skip low-information
             checks, add checks near uncertain predicted event times)
Each network's train labels train the same model (same steps, same seed); all are scored on one
common test set (the fixed network's labels for later actions). Reported: observation requests,
label quality, detection latency, held-out Brier, and Brier gain over the marginal per 1000 requests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shutil  # noqa: E402

import numpy as np  # noqa: E402
from common import EVENTS, HORIZONS, RUNS, TUNED, Timer, make_wm, save_json, threads  # noqa: E402

from agentrs.sensors.observation_network import NetworkConfig  # noqa: E402
from agentrs.sensors.socialweblab_source import lab_temporal_spec  # noqa: E402
from agentrs.socialweblab.engine import LAB_EPOCH, SocialWebLab  # noqa: E402
from agentrs.temporal.delayed_outcome_ledger import DelayedOutcomeLedger  # noqa: E402
from agentrs.temporal.system import build_network  # noqa: E402
from agentrs.temporal.temporal_evaluation import evaluate_temporal, marginal_predict_fn, model_predict_fn, summarize  # noqa: E402
from agentrs.training.common import OptimConfig  # noqa: E402
from agentrs.training.temporal_world_model_trainer import (  # noqa: E402
    TemporalTrainConfig, TemporalTrainingDataBuilder, TemporalWorldModelTrainer)

MODES = {"fixed": NetworkConfig(scheduling="fixed", forecasts_enabled=False),
         "poll_only": NetworkConfig(scheduling="fixed", forecasts_enabled=False, use_feed_events=False),
         "event": NetworkConfig(scheduling="event", forecasts_enabled=False),
         "adaptive": NetworkConfig(scheduling="adaptive", forecasts_enabled=True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--days", type=float, default=10)
    ap.add_argument("--train-days", type=float, default=7)
    ap.add_argument("--steps", type=int, default=TUNED["steps"])
    ap.add_argument("--forecaster-steps", type=int, default=300)
    a = ap.parse_args()
    threads()
    out = RUNS / f"expB_s{a.seed}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    T = Timer()
    cfg = TemporalTrainConfig(steps=a.steps, batch_size=16, seed=0,
                              optim=OptimConfig(lr=TUNED["lr"], weight_decay=TUNED["wd"], total_steps=a.steps, schedule="cosine", warmup_steps=30))
    # forecaster for the adaptive network: trained on an independent corpus (corpus_s0)
    fc = make_wm(0)
    b0 = TemporalTrainingDataBuilder(DelayedOutcomeLedger(RUNS / "corpus_s0" / "net" / "ledger.sqlite"), lab_temporal_spec())
    TemporalWorldModelTrainer(fc, cfg.model_copy(update={"steps": a.forecaster_steps}), name="forecaster").train(
        b0.build(splits=["train"], cutoffs=[0]))
    lab = SocialWebLab(out / "lab.db", seed=a.seed, n_users=80, lab_id=f"sched{a.seed}")
    t_train = LAB_EPOCH + a.train_days * 86400
    t_test = t_train + 1.5 * 86400

    def split(o):
        t = o.effective_action_time()
        return "train" if t < t_train else ("test" if t < t_test else "late")

    nets = {m: build_network(out / m, lab, fc if c.forecasts_enabled else None, c, split_fn=split)
            for m, c in MODES.items()}
    wall = dict.fromkeys(nets, 0.0)
    for _ in range(int(a.days * 96)):
        lab.advance(900)
        for m, n in nets.items():
            t0 = Timer()
            n.step()
            wall[m] += t0.s()
    res = {"seed": a.seed, "days": a.days, "lab": lab.stats(), "modes": {}}
    ref = TemporalTrainingDataBuilder(nets["fixed"].ledger, lab_temporal_spec())
    test = ref.build(splits=["test"], cutoffs=[0])
    train_ref = ref.build(splits=["train"], cutoffs=[0])
    km = summarize(evaluate_temporal(marginal_predict_fn(train_ref, EVENTS, []), test, HORIZONS, EVENTS), EVENTS)
    res["marginal_km"] = km
    for m, n in nets.items():
        st = n.scheduler.stats()
        fetches = sum(v for k, v in st.items() if k.endswith(":done"))
        b = TemporalTrainingDataBuilder(n.ledger, lab_temporal_spec())
        tr = b.build(splits=["train"], cutoffs=[0])
        kinds = {"exact": 0, "interval": 0, "right_censored": 0}
        widths = []
        for e in tr:
            for name in ("answered", "first_response", "positive_feedback"):
                lab_ = e.events.get(name)
                if lab_ is not None and lab_.kind in kinds:
                    kinds[lab_.kind] += 1
                    if lab_.kind == "interval":
                        widths.append(lab_.b_s - lab_.a_s)
        # detection latency: when did this network first know that an answer had happened?
        lat = []
        for act in n.ledger.actions("split='train'"):
            evs = [x for x in n.ledger.events(act["action_id"]) if x["type"] == "answer_posted"]
            if evs:
                lat.append(evs[0]["observed_time"] - evs[0]["env_event_time"])
                continue
            obs = [o for o in n.ledger.observations(act["action_id"]) if (o["payload"].get("occurred") or {}).get("answered")]
            at = (obs[0]["payload"].get("occurred_at") or {}).get("answered") if obs else None
            if obs and at:
                lat.append(obs[0]["actual_time"] - at)
        m_ = make_wm(0)
        TT = TemporalWorldModelTrainer(m_, cfg, name=m)
        info = TT.train(tr)
        ev = summarize(evaluate_temporal(model_predict_fn(m_), test, HORIZONS, EVENTS, ["responses", "views"]), EVENTS)
        gain = km["mean_brier"] - ev["mean_brier"]
        res["modes"][m] = {"observation_requests": fetches, "feed_polls": n.stats["polled_events"],
                           "scheduler": st, "train_actions": len(tr), "label_kinds": kinds,
                           "mean_interval_width_h": float(np.mean(widths) / 3600) if widths else None,
                           "detection_latency_h_median": float(np.median(lat) / 3600) if lat else None,
                           "eval": ev, "brier_gain_vs_km": gain,
                           "brier_gain_per_1000_requests": 1000 * gain / max(1, fetches),
                           "collection_wall_s": wall[m], "train": {k: info[k] for k in ("last_loss", "seconds")}}
        print(m, fetches, kinds, round(ev["mean_brier"], 4), round(gain, 4), flush=True)
        save_json(f"expB_s{a.seed}.json", res)
    res["wall_s"] = T.s()
    save_json(f"expB_s{a.seed}.json", res)


if __name__ == "__main__":
    main()
