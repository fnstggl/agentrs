"""Crucible 5.0 substrate: recorded inference (contract 1.7) with a real neural network."""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")   # torch is part of the environment (rl extra); a missing torch is reported

import crucible
from crucible.core import schema as S
from crucible.core.kernel import Module, World, WorldSpec, applier, endpoint, register_module
from crucible.substrate import inference as I

crucible.load_all_modules()
torch.set_num_threads(1)

CALLS = {"policy": 0, "embed": 0}


def _model(seed: int):
    g = torch.Generator().manual_seed(seed)
    m = torch.nn.Sequential(torch.nn.Linear(6, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4))
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.5)
    return m


@register_module
class InfTest(Module):
    """Participants choose actions with a shared small network; sampling is genuinely nondeterministic (fresh OS
    entropy), so only the record makes the world reproducible."""
    name = "inftest"
    host = "inft"

    def __init__(self, config=None):
        super().__init__(config)
        self.model = _model(int(self.config.get("weights_seed", 0)))
        self.whash = I.weights_hash(self.model)

    def install(self, world):
        world.state.declare_table("inf_decisions")

    @endpoint("decide", S.obj({"n": S.s_int(minimum=1, maximum=64)}, ["n"]), doc="a cohort decision by the network")
    def ep_decide(self, ctx, n):
        obs = ctx.rng("obs").normal(size=(n, 6)).astype(np.float32)

        def run():
            CALLS["policy"] += 1
            with torch.no_grad():
                logits = self.model(torch.from_numpy(obs))
            gen = torch.Generator().manual_seed(int.from_bytes(os.urandom(8), "little"))
            return torch.multinomial(torch.softmax(logits, -1), 1, generator=gen).squeeze(1).tolist()
        acts = I.recorded(ctx, "policy", "inftest-mlp", self.whash, {"obs": obs}, run,
                          sampling={"temperature": 1.0, "method": "multinomial"})
        ctx.emit("inft.decision", {"actions": acts}, vis="internal")
        return {"actions": acts}

    @endpoint("embed", S.obj({"n": S.s_int(minimum=1, maximum=64)}, ["n"]), doc="embeddings, stored as a blob")
    def ep_embed(self, ctx, n):
        obs = ctx.rng("emb").normal(size=(n, 6)).astype(np.float32)

        def run():
            CALLS["embed"] += 1
            with torch.no_grad():
                e = self.model[0](torch.from_numpy(obs))
            return (e + torch.from_numpy(np.frombuffer(os.urandom(4 * e.numel()), dtype=np.uint32)
                                         .reshape(e.shape).astype(np.float32) / 2**40)).numpy()
        emb = I.recorded(ctx, "embed", "inftest-mlp.layer0", self.whash, obs, run, store_inputs=True)
        ctx.emit("inft.embedding", {"norm": float(np.linalg.norm(emb.astype(np.float64)))}, vis="internal")
        return {"shape": list(emb.shape), "first": float(emb[0, 0])}

    @applier("inft.decision")
    def ap_decision(self, ev):
        self.world.state.put("inf_decisions", str(ev.seq), {"actions": ev.payload["actions"]})

    @applier("inft.embedding")
    def ap_emb(self, ev):
        self.world.state.put("inf_decisions", f"e{ev.seq}", {"norm": ev.payload["norm"]})


def make():
    sp = WorldSpec("inf-test", 1, [{"name": "identity", "config": {}}, {"name": "bank", "config": {}},
                                   {"name": "admin", "config": {}}, {"name": "artifacts", "config": {}},
                                   {"name": "inftest", "config": {}}])
    w = World.create(sp, strict=True)
    w.system_call("admin.join", {"handle": "eve", "endowment": 0})
    return w


def test_recorded_inference_replays_and_regenerates_without_rerunning():
    w = make()
    for i in range(6):
        assert w.execute("p:eve", {"tool": "inft.decide", "args": {"n": 16}}).ok
        assert w.execute("p:eve", {"tool": "inft.embed", "args": {"n": 8}}).ok
    live = dict(CALLS)
    assert live["policy"] >= 6 and live["embed"] >= 6
    evs = [e.to_dict() for e in w.log]
    recs = I.records(evs)
    assert len([r for r in recs if r["kind"] == "policy"]) == 6 and len([r for r in recs if r["kind"] == "embed"]) == 6
    m = w.modules["inftest"]
    for r in recs:
        assert r["model_hash"] == m.whash and len(r["inputs_digest"]) == 64
    assert all("output" in r for r in recs if r["kind"] == "policy")
    assert all(r["output_ref"].startswith("sha256:") and r["inputs_ref"].startswith("sha256:")
               for r in recs if r["kind"] == "embed")
    # the sampling really is nondeterministic: rerunning the same inputs gives other decisions (with high probability)
    g = World.regenerate(w.spec, evs)
    assert g.last_hash == w.last_hash and g.digest() == w.digest()
    r = World.replay(w.spec, evs)
    assert r.digest() == w.digest()
    assert CALLS == live                                     # neither regeneration nor replay ran the network
    obs = np.random.default_rng(0).normal(size=(16, 6)).astype(np.float32)
    outs = set()
    for _ in range(5):
        with torch.no_grad():
            logits = m.model(torch.from_numpy(obs))
        gen = torch.Generator().manual_seed(int.from_bytes(os.urandom(8), "little"))
        outs.add(tuple(torch.multinomial(torch.softmax(logits, -1), 1, generator=gen).squeeze(1).tolist()))
    assert len(outs) > 1


def test_diverging_inputs_are_detected_during_regeneration():
    w = make()
    w.execute("p:eve", {"tool": "inft.decide", "args": {"n": 4}})
    evs = [e.to_dict() for e in w.log]
    for e in evs:
        if e["type"] == "ext":
            for o in e["payload"]["outcomes"]:
                o["out"]["inputs_digest"] = "0" * 64
    g = World.regenerate(w.spec, evs)
    assert g.last_hash != w.last_hash
    rej = [e for e in g.log if e.type == "cmd.rejected"]
    assert rej and "diverged" in rej[-1].payload["error"]["message"]


def test_weights_hash():
    a, b = _model(3), _model(3)
    assert I.weights_hash(a) == I.weights_hash(b) == I.weights_hash(a.state_dict())
    assert len(I.weights_hash(a)) == 64
    with torch.no_grad():
        p = next(b.parameters())
        p[0, 0] = torch.nextafter(p[0, 0], torch.tensor(1e9))
    assert I.weights_hash(a) != I.weights_hash(b)
    sd = {k: v.numpy() for k, v in a.state_dict().items()}
    rev = dict(reversed(list(sd.items())))
    assert I.weights_hash(sd) == I.weights_hash(rev) == I.weights_hash(a)
    assert I.weights_hash({"w": np.zeros(3, np.float32)}) != I.weights_hash({"w": np.zeros(3, np.float64)})
