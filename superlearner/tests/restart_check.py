"""Helper of tests/test_continual.py, run as its own process with one math thread (set before any library starts).

A smoke lifetime runs its first phase and is checkpointed. Then two learners exist side by side: the ORIGINAL
object, which simply goes on, and a NEW object built from the checkpoint file alone. The script prints, as JSON:

  * a fingerprint of everything that learns or steers learning, for both, right after the reload (controller
    weights, world model, both optimisers' moments, replay reservoir and experience buffer, consolidation anchors,
    counters, every random-number state) - they must be equal;
  * the weights of both after ONE MORE UPDATE ON THE SAME EPISODES - they must be bit-identical, which is what
    "the restart continues the same learning process" means;
  * that the reloaded lifetime then runs to its end.

Why not compare two whole lifetimes bit by bit: the simulator itself is not perfectly repeatable. Some observations
contain a value that depends on wall-clock timing, so two IDENTICAL uninterrupted runs occasionally differ from the
first practice iteration on (measured: about one pair in five; docs/experiments/NOTES.md). That has nothing to do
with restarting, so the comparison is made on fixed episodes."""
import copy
import hashlib
import json
import os
import sys

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import torch  # noqa: E402

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from superlearner.cli import load_config  # noqa: E402
from superlearner.continual import plasticity as PL  # noqa: E402
from superlearner.continual.lifetime import Lifetime  # noqa: E402


def h(tensors) -> str:
    m = hashlib.sha256()
    for t in tensors:
        m.update(t.detach().cpu().double().numpy().tobytes())
    return m.hexdigest()[:16]


def fingerprint(life) -> dict:
    tr, a = life.tr, life.tr.algo
    opt = a.opt.state_dict()["state"]
    wopt = a.wm_opt.state_dict()["state"]
    return {"weights": PL.weights_digest(tr.model), "world_model": h(a.wm.parameters()),
            "adam_m": h([v["exp_avg"] for _, v in sorted(opt.items())]), "adam_v": h([v["exp_avg_sq"] for _, v in sorted(opt.items())]),
            "adam_steps": int(max(float(v["step"]) for v in opt.values())), "wm_adam_m": h([v["exp_avg"] for _, v in sorted(wopt.items())]),
            "anchors": h(a.anchors) if a.anchors is not None else None, "fisher": h(a.fisher) if a.fisher is not None else None,
            "buffer": [e["summary"]["task"] for e in a.buffer], "experience": [e["summary"]["task"] for e in a.experience],
            "counters": [tr.iter, tr.env_decisions, tr.episodes, a.n_updates, a.n_calls, a.n_policy_steps, a.rollbacks, a.lr_scale, a.phase],
            "rng_algo": str(a.rng.bit_generator.state["state"]), "rng_trainer": str(tr.rng.bit_generator.state["state"]),
            "rng_torch": h([torch.get_rng_state().double()]), "lifetime": [life.x["phase"], life.x["stage"], life.x["it"], life.x["counter"], len(life.x["rows"])],
            "store_rows": tr.store.n, "store": hashlib.sha256(tr.store.mat[: tr.store.n].tobytes()).hexdigest()[:16]}


def main(out_dir: str) -> None:
    cfg = load_config(os.path.join(HERE, "..", "configs", "life_smoke.yaml"))
    cfg["lifetime"]["phases"] = cfg["lifetime"]["phases"][:2]
    cfg["lifetime"]["eval"]["sampled"] = False
    a = Lifetime(copy.deepcopy(cfg), os.path.join(out_dir, "life"), verbose=False)
    init = a.x["digest_init"]
    for _ in range(3):                                   # begin + observe, one practice iteration, close of the first phase
        a.step()
    b = Lifetime(copy.deepcopy(cfg), os.path.join(out_dir, "life"), verbose=False)       # everything comes from the file
    fa, fb = fingerprint(a), fingerprint(b)
    eps = copy.deepcopy(list(a.tr.algo.experience))
    rs = torch.get_rng_state()
    a.tr.algo.update(copy.deepcopy(eps), a.tr.store)
    torch.set_rng_state(rs)                              # both learners share this process's torch generator
    b.tr.algo.update(copy.deepcopy(eps), b.tr.store)
    after = [fingerprint(a), fingerprint(b)]
    a.tr.close()
    while b.step():
        pass
    res = json.load(open(os.path.join(out_dir, "life", "results.json")))
    b.tr.close()
    print("RESTART_CHECK " + json.dumps({
        "original": fa, "reloaded": fb, "restarts": b.x["restarts"], "weights_at_start": init,
        "after_same_update": [{k: f[k] for k in ("weights", "world_model", "adam_m", "wm_adam_m", "counters")} for f in after],
        "finished": res["finished"], "matrix_rows": len(res["derived"]["matrix"]), "rows": len(res["rows"]),
        "done_marker": os.path.exists(os.path.join(out_dir, "life", "DONE")),
        "last_row": {"moved": res["rows"][-1]["weights"]["since_init"]["all"]["l2"], "wm": bool(res["rows"][-1]["world_model"]),
                     "probe": "probe_fit" in res["rows"][-1]["plasticity"]}}))


if __name__ == "__main__":
    main(sys.argv[1])
