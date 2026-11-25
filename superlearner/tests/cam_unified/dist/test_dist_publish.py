"""Atomic weights-only checkpoint publication, validation gating, concurrent readers, safe-boundary refresh
(manifest F02)."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time

import pytest
import torch

from superlearner.cam_unified.dist.publish import (CheckpointPublisher, PolicyClient, PublishRejected, load_weights,
                                                    read_latest)

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))


def _sd(v: float, n: int = 1000):
    return {"layer.weight": torch.full((n, 8), float(v)), "layer.bias": torch.full((8,), float(v)),
            "step": torch.tensor(int(v))}


def test_publish_versions_latest_and_weights_only_file(tmp_path):
    pub = CheckpointPublisher(str(tmp_path / "ckpt"), keep_last=10)
    assert pub.latest() is None
    m0 = pub.publish(_sd(1.0), 0, meta={"note": "init"}, config={"d_model": 64})
    assert m0["validated"] and m0["rel_change"] is None
    m3 = pub.publish(_sd(2.0), 3)
    assert m3["validated"] and m3["rel_change"] == pytest.approx(1.0, rel=1e-6)   # ‖2−1‖/‖1‖ on float tensors
    lt = read_latest(str(tmp_path / "ckpt"))
    assert lt["version"] == 3 and lt["dir"] == "v00000003"
    with pytest.raises(PublishRejected):
        pub.publish(_sd(3.0), 3)                  # versions are strictly monotone
    with pytest.raises(PublishRejected):
        pub.publish(_sd(3.0), 1)
    # the file is weights-only: loadable with weights_only=True, holds tensors + plain config, no optimiser state
    path = tmp_path / "ckpt" / "v00000003" / "weights.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert set(payload) == {"format", "version", "config", "state_dict"}
    assert all(isinstance(v, torch.Tensor) for v in payload["state_dict"].values())
    assert not any("optim" in k or "exp_avg" in k for k in payload["state_dict"])
    p = load_weights(str(path), lt["sha256"])
    assert torch.equal(p["state_dict"]["layer.weight"], _sd(2.0)["layer.weight"])
    with pytest.raises(RuntimeError, match="sha256"):
        load_weights(str(path), "0" * 64)
    with pytest.raises(TypeError):
        pub.publish({"x": [1, 2, 3]}, 9)          # non-tensor values are refused


def test_validation_failures_never_reach_latest(tmp_path):
    calls = []

    def probe(sd, meta):                          # e.g. a held-out probe; rejects weights whose bias is negative
        calls.append(meta["version"])
        ok = bool((sd["layer.bias"] >= 0).all())
        return ok, {"bias_min": float(sd["layer.bias"].min())}
    pub = CheckpointPublisher(str(tmp_path / "ckpt"), validate_fn=probe)
    assert pub.publish(_sd(1.0), 1)["validated"]
    bad = _sd(1.0)
    bad["layer.weight"][0, 0] = float("nan")
    m = pub.publish(bad, 2)
    assert not m["validated"] and "non-finite" in m["validation"]["reason"]
    m = pub.publish(_sd(-1.0), 3)
    assert not m["validated"] and m["validation"]["reason"] == "validate_fn rejected"
    assert read_latest(str(tmp_path / "ckpt"))["version"] == 1
    assert json.load(open(tmp_path / "ckpt" / "v00000003" / "meta.json"))["validated"] is False
    assert calls == [1, 3]                        # the probe runs only on structurally valid versions
    assert pub.publish(_sd(4.0), 4)["validated"]
    assert read_latest(str(tmp_path / "ckpt"))["version"] == 4


def test_crashed_publication_leftovers_are_ignored_and_pruning_keeps_latest(tmp_path):
    root = tmp_path / "ckpt"
    pub = CheckpointPublisher(str(root), keep_last=2)
    for v in range(5):
        pub.publish(_sd(v), v)
    assert pub.versions() == [3, 4] and read_latest(str(root))["version"] == 4
    os.makedirs(root / ".tmp-v00000005-999")       # a publisher died mid-write
    (root / ".tmp-v00000005-999" / "weights.pt").write_bytes(b"garbage")
    pub2 = CheckpointPublisher(str(root), keep_last=2)
    assert not any(f.startswith(".tmp-") for f in os.listdir(root))
    assert pub2.last_version == 4
    m = pub2.publish(_sd(6.0), 6)
    assert m["rel_change"] is not None             # the restarted publisher reloaded the previous weights


_PUBLISHER = r"""
import sys, time
sys.path.insert(0, {src!r})
import torch
from superlearner.cam_unified.dist.publish import CheckpointPublisher
pub = CheckpointPublisher({root!r}, keep_last=2)
for v in range(1, {n} + 1):
    pub.publish({{"w": torch.full((512, 512), float(v)), "b": torch.full((512,), float(v))}}, v)
print("published", flush=True)
"""

_READER = r"""
import json, os, sys, time
sys.path.insert(0, {src!r})
import torch
from superlearner.cam_unified.dist.publish import read_latest, load_weights
root = {root!r}
seen, loads, torn, regress, vanished = [], 0, 0, 0, 0
t0 = time.time()
while time.time() - t0 < {secs}:
    lt = read_latest(root)
    if lt is None:
        continue
    try:
        sd = load_weights(os.path.join(root, lt["dir"], "weights.pt"), lt["sha256"])["state_dict"]
    except FileNotFoundError:
        vanished += 1          # pruned between reading LATEST and opening the file: retry
        continue
    v = float(lt["version"])
    if not (bool((sd["w"] == v).all()) and bool((sd["b"] == v).all())):
        torn += 1
    if seen and lt["version"] < seen[-1]:
        regress += 1
    seen.append(lt["version"])
    loads += 1
print(json.dumps({{"loads": loads, "torn": torn, "regress": regress, "vanished": vanished,
                  "distinct": len(set(seen)), "last": seen[-1] if seen else None}}), flush=True)
"""


def test_concurrent_readers_see_only_complete_monotone_versions(tmp_path):
    root = str(tmp_path / "ckpt")
    CheckpointPublisher(root).publish({"w": torch.zeros(512, 512), "b": torch.zeros(512)}, 0)
    readers = [subprocess.Popen([sys.executable, "-c", _READER.format(src=SRC, root=root, secs=6)],
                                stdout=subprocess.PIPE, text=True) for _ in range(3)]
    time.sleep(1.5)
    pub = subprocess.Popen([sys.executable, "-c", _PUBLISHER.format(src=SRC, root=root, n=40)],
                           stdout=subprocess.PIPE, text=True)
    assert pub.wait(120) == 0
    for r in readers:
        out, _ = r.communicate(timeout=120)
        res = json.loads(out.strip().splitlines()[-1])
        assert res["torn"] == 0, res               # never a partially written or mismatched version
        assert res["regress"] == 0, res            # LATEST never goes backwards
        assert res["loads"] > 20 and res["distinct"] >= 3, res
        assert res["last"] == 40, res
    assert read_latest(root)["version"] == 40


class _Policy:
    def __init__(self):
        self.loaded = []

    def load_weights(self, sd, version):
        self.loaded.append((version, float(sd["layer.bias"][0])))


def test_policy_client_refreshes_only_at_safe_boundaries(tmp_path):
    root = str(tmp_path / "ckpt")
    pub = CheckpointPublisher(root)
    pol = _Policy()
    traj = PolicyClient(root, refresh_at="trajectory")
    with pytest.raises(TimeoutError):
        traj.wait_for_version(pol, 0, timeout_s=0.2)
    pub.publish(_sd(1.0), 0)
    assert traj.wait_for_version(pol, 0, timeout_s=5) == 0
    pub.publish(_sd(2.0), 1)
    assert traj.maybe_refresh(pol, "decision") is False          # mid-trajectory: not a safe point for this actor
    assert traj.version == 0
    assert traj.maybe_refresh(pol, "trajectory") is True
    assert traj.version == 1 and pol.loaded[-1] == (1, 2.0)
    assert traj.drift == pytest.approx(1.0, rel=1e-6)            # ‖θ1−θ0‖/‖θ0‖ = ‖2−1‖/‖1‖
    assert traj.maybe_refresh(pol, "trajectory") is False        # nothing newer
    dec = PolicyClient(root, refresh_at="decision")
    dec.wait_for_version(pol, 0)
    assert dec.version == 1 and dec.drift == 0.0
    pub.publish(_sd(3.0), 2)
    assert dec.maybe_refresh(pol, "decision") is True and dec.version == 2
    assert dec.drift == pytest.approx(0.5, rel=1e-6)             # ‖3−2‖/‖2‖
    dec.state_rebuilt()
    assert dec.drift == 0.0
    assert [h[2] for h in dec.history] == ["trajectory", "decision"]


def _ones(v: float, n: int = 1000):
    return {"layer.weight": torch.full((n, 8), float(v)), "layer.bias": torch.full((8,), float(v))}


@pytest.mark.parametrize("mode", ["exact", "meta_sum"])
def test_drift_accounts_for_skipped_versions(tmp_path, mode):
    """Verifier probe P2-drift: an actor jumping v0 -> v3 sees the full change ‖θ3 − θ0‖/‖θ0‖ = 3, not the last
    published step's 1/3; and drift keeps accumulating over several refreshes without a latent rebuild."""
    root = str(tmp_path / "ckpt")
    pub = CheckpointPublisher(root, keep_last=10)
    pol = _Policy()
    pub.publish(_ones(1.0), 0)
    c = PolicyClient(root, refresh_at="decision", drift_mode=mode)
    c.wait_for_version(pol, 0)
    for v in (1, 2, 3):
        pub.publish(_ones(v + 1.0), v)
    assert c.maybe_refresh(pol, "decision") and c.version == 3
    assert c.drift == pytest.approx(3.0, rel=1e-6)
    assert c.state_version == 0                                  # latent still built under v0
    pub.publish(_ones(5.0), 4)
    assert c.maybe_refresh(pol, "decision")
    assert c.drift == pytest.approx(4.0, rel=1e-6)               # ‖5−1‖/‖1‖ since the state was built
    c.state_rebuilt()
    assert c.drift == 0.0 and c.state_version == 4
    pub.publish(_ones(6.0), 5)
    assert c.maybe_refresh(pol, "decision") and c.drift == pytest.approx(0.2, rel=1e-6)   # ‖6−5‖/‖5‖


def test_meta_sum_drift_is_an_upper_bound_and_infinite_when_the_chain_is_pruned(tmp_path):
    root = str(tmp_path / "ckpt")
    pub = CheckpointPublisher(root, keep_last=2)
    pol = _Policy()
    g = torch.Generator().manual_seed(0)
    sds = [{"layer.weight": torch.randn(50, 8, generator=g), "layer.bias": torch.randn(8, generator=g)}
           for _ in range(4)]
    pub.publish(sds[0], 0)
    exact, bound = PolicyClient(root, drift_mode="exact"), PolicyClient(root, drift_mode="meta_sum")
    for c in (exact, bound):
        c.wait_for_version(pol, 0)
    pub.publish(sds[1], 1)
    for c in (exact, bound):
        assert c.maybe_refresh(pol, "trajectory")
    assert bound.drift >= exact.drift - 1e-9 and exact.drift > 0
    pub.publish(sds[2], 2)
    pub.publish(sds[3], 3)                    # keep_last=2: v2 and v3 on disk, so the chain 3 -> 2 -> 1 holds
    for c in (exact, bound):
        assert c.maybe_refresh(pol, "trajectory")
    assert bound.drift >= exact.drift - 1e-9
    pub.publish(sds[0], 4)
    pub.publish(sds[1], 5)
    pub.publish(sds[2], 6)                    # v4, v5 pruned: the chain 6 -> 5 -> 4 -> 3 is broken
    assert bound.maybe_refresh(pol, "trajectory") and bound.drift == float("inf")


def test_behavioural_probe_keeps_a_degraded_version_from_actors(tmp_path):
    """A version is validated only if it does not do worse on held-out worlds than the best validated one: a
    sabotaged update is published but rejected, LATEST keeps the last good version, and the actor keeps running it."""
    from superlearner.cam_unified.dist.actor import ActorProcess, ActorSpec, run_paths
    from superlearner.cam_unified.dist.learner import CentralLearner
    from superlearner.cam_unified.dist.sinks import JsonlArchive
    from superlearner.cam_unified.dist.toyworld import PayoutProbe, best_tool_names, make_tiny_learner
    run = str(tmp_path / "run")
    probe = PayoutProbe(mechanism_seed=0, margin=0.1)
    model = make_tiny_learner(seed=0, lr=1e-2)
    sabotage_at = 16
    orig = model.update

    def update(batch):
        out = orig(batch)
        if batch.learner_version + 1 == sabotage_at:        # this update produces version `sabotage_at`
            with torch.no_grad():
                model.net.pi.weight.zero_()
                model.net.pi.bias.copy_(torch.tensor([0.0, 0.0, 0.0, 30.0]))   # always lever_3
        return out
    model.update = update
    L = CentralLearner(run, model, sink=JsonlArchive(run_paths(run)["archive"]), min_decisions=256,
                       max_batch_decisions=256, validate_fn=probe, record_traces=False)
    a = ActorProcess(ActorSpec("a0", run, env_kwargs={"B": 16, "mechanism_seed": 0}, seed=7, batch_records=10 ** 6,
                               record_traces=False))
    a.client.wait_for_version(a.policy, 0)
    while L.version < sabotage_at + 1:
        a._episode()
        while L.step() is not None:
            pass
    metas = {m: json.load(open(os.path.join(run, "ckpt", f"v{m:08d}", "meta.json")))
             for m in (sabotage_at - 1, sabotage_at)}
    good, bad = metas[sabotage_at - 1], metas[sabotage_at]
    assert good["validated"] and good["validation"]["validate_fn"]["score"] > 0.5
    assert not bad["validated"] and bad["validation"]["reason"] == "validate_fn rejected"
    frac_lever3 = best_tool_names(mechanism_seed=0).count("lever_3") / 4
    assert bad["validation"]["validate_fn"]["score"] == pytest.approx(frac_lever3, abs=0.12)
    lt = read_latest(os.path.join(run, "ckpt"))
    assert lt["version"] != sabotage_at
    a._episode()                                             # trajectory boundary: refresh only to validated weights
    assert a.client.version != sabotage_at
    assert [v for v, _, ok in probe.history if not ok][:1] == [sabotage_at]
    L.close()
