"""CAM 2.0 experiment runner: (optional) evaluation of the untrained CAM, training, checkpoint-reload verification,
and evaluation without parameter updates.

  PYTHONPATH=src python scripts/cam2/train_eval.py --config configs/cam2/main.yaml [--set key=value ...]
      [--eval-initial] [--verify-reload] [--eval-train] [--eval-limit N]

Writes ``<run_dir>/results.json``:
  initial_val    - B2 (frozen LLM + fixed, untrained CAM) on outer-validation tasks
  final_val      - the trained CAM on the same tasks (greedy; parameter checksum asserted unchanged)
  final_train    - fixed TRAIN-family instances (train seed range, not sampled during training by construction of
                   the sampler's RNG draw - reported as fit / retention, not generalisation)
  reload         - a fresh CAM built from the saved checkpoint reproduces the trained parameters and its action
                   distributions exactly on recorded decision states
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agentrs.cam2.featurize import collate  # noqa: E402
from agentrs.cam2.model import CAM, CAMConfig, param_checksum  # noqa: E402
from agentrs.cam2.tasks import train_eval_tasks, val_tasks  # noqa: E402
from agentrs.cam2.trainer import RunConfig, Trainer  # noqa: E402


def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--eval-initial", action="store_true")
    ap.add_argument("--verify-reload", action="store_true")
    ap.add_argument("--eval-train", action="store_true")
    ap.add_argument("--eval-per-family", type=int, default=4)
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--no-train", action="store_true")
    a = ap.parse_args(argv)
    cfg = {}
    if a.config:
        import yaml

        cfg = yaml.safe_load(open(a.config))
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except Exception:  # noqa: BLE001
            pass
        d = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return a, RunConfig(**cfg)


@torch.no_grad()
def verify_reload(tr: Trainer, ck_path: str, n_states: int = 64) -> dict:
    """Build a fresh CAM from the checkpoint file and compare it with the in-memory trained CAM: parameter
    checksum, every tensor, and the policy distribution + value on recorded decision states (replay buffer)."""
    ck = torch.load(ck_path, map_location=tr.device, weights_only=False)
    cam2 = CAM(CAMConfig(**ck["cam_config"])).to(tr.device)
    cam2.load_state_dict(ck["cam"])
    cam2.eval()
    tr.cam.eval()
    same_tensors = all(torch.equal(a, b) for a, b in zip(tr.cam.state_dict().values(), cam2.state_dict().values()))
    trs = [x for e in tr.replay[-32:] for x in e.tr][:n_states]
    max_dlogit, max_dv = 0.0, 0.0
    for i in range(0, len(trs), 16):
        chunk = trs[i:i + 16]
        st = {k: torch.stack([x.h0[k] for x in chunk]).to(tr.device).float() for k in chunk[0].h0}
        mem = torch.stack([x.mem for x in chunk]) if chunk[0].mem is not None else None
        mm = torch.stack([x.mem_m for x in chunk]) if chunk[0].mem_m is not None else None
        b = collate(tr.llm, tr.reg, [x.step for x in chunk], mem, mm, tr.device, goal_cond=tr.cam.cfg.el_goal_cond)
        outs = []
        for m in (tr.cam, cam2):
            f, el, _, _ = m.integrate(b, st)
            lg = torch.log_softmax(m.policy_logits(f, m.candidates(b, el), b["cmask"]), -1).masked_fill(~b["cmask"], 0)
            outs.append((lg, m.v(f)))
        max_dlogit = max(max_dlogit, float((outs[0][0] - outs[1][0]).abs().max()))
        max_dv = max(max_dv, float((outs[0][1] - outs[1][1]).abs().max()))
    tr.cam.train()
    return {"checkpoint": ck_path, "checksum_saved": ck["checksum"], "checksum_reloaded": param_checksum(cam2),
            "checksum_trained": param_checksum(tr.cam), "all_tensors_equal": same_tensors, "states_compared": len(trs),
            "max_abs_logprob_diff": max_dlogit, "max_abs_value_diff": max_dv,
            "llm_fingerprint_matches": ck["llm"]["fingerprint"] == tr.llm.fingerprint()}


def main(argv=None):
    import os

    if os.environ.get("CAM2_STACKDUMP"):   # periodic stack dumps (py-spy needs ptrace, unavailable here)
        import faulthandler

        faulthandler.dump_traceback_later(float(os.environ["CAM2_STACKDUMP"]), repeat=True)
    a, rc = parse(argv)
    tr = Trainer(rc)
    out_p = Path(rc.run_dir) / "results.json"
    res = json.loads(out_p.read_text()) if out_p.exists() else {}
    res.update({"config": rc.to_dict(), "cam_params": tr.cam.n_params(), "llm": tr.llm.describe(),
                "llm_load_s": tr.llm_load_s})
    vt = val_tasks(rc.env, a.eval_per_family, rc.eval_families, limit=a.eval_limit)

    def save():
        out_p.write_text(json.dumps(res, indent=1, default=float))

    try:
        if a.eval_initial and "initial_val" not in res and tr.counters["updates"] == 0:
            ev = tr.evaluate(vt, label="initial_val")
            res["initial_val"] = ev
            print(f"[smoke] initial (untrained CAM) val success={ev['success']:.3f} ci={ev['ci95']}", flush=True)
            save()
        if not a.no_train:
            t0 = time.time()
            res["train_summary"] = tr.train()
            res["train_wall_s"] = time.time() - t0
            save()
        if a.verify_reload:
            res["reload"] = verify_reload(tr, str(Path(rc.run_dir) / "ckpt" / "final.pt"))
            print(f"[smoke] reload: {json.dumps(res['reload'])}", flush=True)
            save()
        ev = tr.evaluate(vt, label="final_val")
        res["final_val"] = ev
        print(f"[smoke] final val success={ev['success']:.3f} ci={ev['ci95']} per_family={ev['per_family']}", flush=True)
        save()
        if a.eval_train:
            ev = tr.evaluate(train_eval_tasks(rc.env, 3, rc.train_families), label="final_train")
            res["final_train"] = ev
            print(f"[smoke] final train-family success={ev['success']:.3f}", flush=True)
            save()
        res["llm_fingerprint_end"] = tr.llm.fingerprint()
        res["llm_unchanged"] = res["llm_fingerprint_end"] == tr.llm_fp_start
        res["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        save()
    finally:
        tr.close()


if __name__ == "__main__":
    main()
