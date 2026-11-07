"""Throughput of the observational-learning pipeline on this machine (CPU, Pythia-14M):
observed actions / s, forecasts / s (batched, all profile horizons in one pass), matured labels / s,
training examples / s and gradient updates / hour."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS, make_wm, save_json, threads  # noqa: E402

from agentrs.sensors.observation_network import ForecastService, NetworkConfig  # noqa: E402
from agentrs.sensors.socialweblab_source import lab_temporal_spec  # noqa: E402
from agentrs.socialweblab.engine import SocialWebLab  # noqa: E402
from agentrs.temporal.delayed_outcome_ledger import DelayedOutcomeLedger  # noqa: E402
from agentrs.temporal.system import build_network, run_lab  # noqa: E402
from agentrs.temporal.temporal_world_model import TemporalQuery  # noqa: E402
from agentrs.training.temporal_world_model_trainer import (  # noqa: E402
    TemporalTrainConfig, TemporalTrainingDataBuilder, TemporalWorldModelTrainer)


def main():
    threads(4)
    out = RUNS / "throughput"
    import shutil

    shutil.rmtree(out, ignore_errors=True)
    lab = SocialWebLab(out / "lab.db", seed=7, n_users=80, lab_id="tp7")
    net = build_network(out / "net", lab, None, NetworkConfig(scheduling="fixed", forecasts_enabled=False))
    t = time.perf_counter()
    st = run_lab(lab, net, 5 * 86400, 900)
    sim = time.perf_counter() - t
    res = {"simulated_days": 5, "lab_and_sensor_wall_s": sim, "observed_actions": st["new_actions"],
           "observed_actions_per_s": st["new_actions"] / st["wall_ingest"],
           "follow_up_events_per_s": st["follow_ups"] / st["wall_ingest"],
           "matured_labels_per_s": st["observations"] / st["wall_observe"]}
    m = make_wm(0)
    L = DelayedOutcomeLedger(RUNS / "corpus_s0" / "net" / "ledger.sqlite")
    ex = TemporalTrainingDataBuilder(L, lab_temporal_spec()).build(splits=["train"], cutoffs=[0])
    qs = [TemporalQuery.from_example(e) for e in ex[:256]]
    hs = [60, 300, 900, 3600, 21600, 86400, 259200, 604800]
    fs = ForecastService(m, L, batch=64)
    t = time.perf_counter()
    m.predict_batch(qs, hs, chunk=64)
    dt = time.perf_counter() - t
    res.update({"forecasts_per_s": len(qs) / dt, "horizon_predictions_per_s": len(qs) * len(hs) / dt})
    T = TemporalWorldModelTrainer(m, TemporalTrainConfig(steps=40, batch_size=16))
    info = T.train(ex, steps=40)
    res.update({"training_examples_per_s": info["examples_per_s"],
                "gradient_updates_per_hour": 40 / info["seconds"] * 3600, "batch_size": 16, "threads": 4,
                "device": "cpu (4 cores)", "model": "EleutherAI/pythia-14m"})
    del fs
    save_json("throughput.json", res)
    print(res)


if __name__ == "__main__":
    main()
