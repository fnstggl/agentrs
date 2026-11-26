"""Numerical inference backends for the population's neural participants, benchmarked and chosen by measurement
(MASTER_SPEC 13.3, SP-13.3, Gate V item 4).

Workloads are the real networks with the real dimensions of a development-profile world: the shared class-2 policy
(`population.nets.AdaptiveNet`: observation encoder, GRU memory, candidate scorer) and the two class-3 architectures
(`StrategicMLP`, `StrategicGRU`), initialised exactly as the population initialises them. Observations are the class-2
members' observations computed from the generated world (`population.adaptive.obs_features`); candidate features are
drawn in their documented ranges (match, satisfaction, log-visits, recency, social signal, validity, the explore slot)
because a freshly generated world has no browsing history yet.

Backends (each timed on one CPU thread, or on the GPU):

| backend | what |
|---|---|
| `torch_per_participant` | one eager forward pass per participant (the unbatched reference) |
| `torch_cpu` | one eager forward pass for the whole cohort |
| `numpy` | the same network written in NumPy (float32 matrix products) |
| `torchscript` | JIT-compiled graph (`torch.jit.trace` + freeze): the available Numba-equivalent JIT, since Numba is not installed and may not be installed into the shared environment |
| `inductor_cpu` | compiled numerical kernels: `torch.compile` with the Inductor backend generating C++/OpenMP code (gcc) |
| `int8_dynamic` | quantized inference: dynamic int8 quantization of the linear and recurrent layers |
| `gpu_fp32`, `gpu_fp16` | batched GPU inference (when the job was granted GPU memory) |

Behavioural equivalence is measured, not assumed: the population chooses `argmax(score + Gumbel * 0.15)` over valid
candidates, so for every backend we draw the same Gumbel noise and report the share of identical choices (paired
agreement), the total-variation distance between choice distributions over many noise draws, and the mean
fp32-utility loss of the backend's choices. A backend is equivalent when paired agreement >= 0.99 and TV <= 0.01.
`choose()` picks, per batch size, the fastest equivalent backend.
"""
from __future__ import annotations

import os
import time

import numpy as np

EQUIV_AGREEMENT = 0.99
EQUIV_TV = 0.01
TEMPERATURE = 0.15


def _torch():
    import torch
    return torch


def adaptive_workload(B: int, seed: int = 0, profile: str = "development") -> dict:
    """The class-2 network of a generated world and a cohort of B decision inputs."""
    import crucible
    crucible.load_all_modules()
    from ..core.kernel import World
    from ..population import adaptive as AD
    from ..population import nets
    from ..population.worlds import population_world
    from ..substrate import columns
    w = World.create(population_world(profile, seed))
    mod = w.modules["pop_adaptive"]
    members = np.flatnonzero(columns.col(w, "pop", "class") == 2)
    idx = members[np.arange(B) % members.size]
    obs = AD.obs_features(w, idx)
    K = mod.K
    rng = np.random.default_rng(seed + 1)
    feats = {"known": rng.integers(0, max(1, columns.rows(w, "popweb_res") if "popweb_res" in columns.tables(w)
                                             else 24), (B, K)),
             "valid": rng.random((B, K)) < 0.7, "match": rng.random((B, K)).astype(np.float32),
             "known_sat": rng.uniform(-1, 1, (B, K)).astype(np.float32), "known_n": rng.poisson(3, (B, K)),
             "recency": rng.random((B, K)).astype(np.float32), "social": rng.random((B, K)).astype(np.float32),
             "suggest": np.where(rng.random(B) < 0.5, rng.integers(0, 24, B), -1),
             "trust": rng.random(B).astype(np.float32)}
    try:
        cand, mask = AD._cand_features(w, feats, K)
    except Exception:                   # noqa: BLE001 - a world without the resource table: same layout, zeros
        cand = np.zeros((B, K + 1, mod.cand_dim), np.float32)
        mask = np.ones((B, K + 1), bool)
    net = nets.AdaptiveNet(mod.obs_dim, mod.cand_dim, mod.M, int(mod.cfg["hidden"]))
    nets.init_params(net, np.random.default_rng(1000))
    net.eval()
    mem = np.tanh(rng.normal(0, 0.5, (B, mod.M))).astype(np.float32)
    w.close()
    return {"net": net, "obs": obs.astype(np.float32), "mem": mem, "cand": cand.astype(np.float32), "mask": mask,
            "dims": {"obs": int(mod.obs_dim), "cand": int(mod.cand_dim), "mem": int(mod.M), "K": int(K) + 1,
                     "hidden": int(mod.cfg["hidden"])}, "params": int(nets.n_params(net))}


# ====================================================================== backends of the adaptive network
def numpy_forward(params: dict, obs, mem, cand):
    """AdaptiveNet in NumPy (float32): enc -> relu -> GRUCell -> scorer."""
    relu = lambda x: np.maximum(x, 0.0)          # noqa: E731
    h = relu(obs @ params["enc.weight"].T + params["enc.bias"])
    gi = h @ params["gru.weight_ih"].T + params["gru.bias_ih"]
    gh = mem @ params["gru.weight_hh"].T + params["gru.bias_hh"]
    H = mem.shape[1]
    r = 1 / (1 + np.exp(-(gi[:, :H] + gh[:, :H])))
    z = 1 / (1 + np.exp(-(gi[:, H:2 * H] + gh[:, H:2 * H])))
    n = np.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
    m2 = (1 - z) * n + z * mem
    B, K, _ = cand.shape
    x = np.concatenate([np.broadcast_to(h[:, None, :], (B, K, h.shape[1])),
                        np.broadcast_to(m2[:, None, :], (B, K, H)), cand], -1)
    y = relu(x @ params["s1.weight"].T + params["s1.bias"])
    y = relu(y @ params["s2.weight"].T + params["s2.bias"])
    return (y @ params["s3.weight"].T + params["s3.bias"])[..., 0], m2


def make_backends(net, sample: tuple, gpu: bool) -> dict:
    """name -> callable(obs, mem, cand) -> (scores ndarray [B, K], memory ndarray)."""
    torch = _torch()
    torch.set_num_threads(1)
    out = {}

    def eager(o, m, c):
        with torch.no_grad():
            s, m2 = net(torch.from_numpy(o), torch.from_numpy(m), torch.from_numpy(c))
        return s.numpy(), m2.numpy()

    def per_participant(o, m, c):
        S, M2 = [], []
        with torch.no_grad():
            for i in range(o.shape[0]):
                s, m2 = net(torch.from_numpy(o[i:i + 1]), torch.from_numpy(m[i:i + 1]), torch.from_numpy(c[i:i + 1]))
                S.append(s.numpy())
                M2.append(m2.numpy())
        return np.concatenate(S), np.concatenate(M2)
    out["torch_per_participant"] = per_participant
    out["torch_cpu"] = eager
    params = {k: v.detach().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    out["numpy"] = lambda o, m, c: numpy_forward(params, o, m, c)
    o, m, c = (torch.from_numpy(x) for x in sample)
    with torch.no_grad():
        ts = torch.jit.freeze(torch.jit.trace(net.eval(), (o, m, c), check_trace=False))

    def scripted(o, m, c):
        with torch.no_grad():
            s, m2 = ts(torch.from_numpy(o), torch.from_numpy(m), torch.from_numpy(c))
        return s.numpy(), m2.numpy()
    out["torchscript"] = scripted
    try:
        comp = torch.compile(net, backend="inductor", dynamic=True)

        def inductor(o, m, c):
            with torch.no_grad():
                s, m2 = comp(torch.from_numpy(o), torch.from_numpy(m), torch.from_numpy(c))
            return s.numpy(), m2.numpy()
        inductor(*sample)                      # compile once (timed separately)
        out["inductor_cpu"] = inductor
    except Exception as exc:                   # noqa: BLE001 - reported as not available, never silently
        out["inductor_cpu"] = exc
    q = torch.ao.quantization.quantize_dynamic(net, {torch.nn.Linear, torch.nn.GRUCell}, dtype=torch.qint8)

    def int8(o, m, c):
        with torch.no_grad():
            s, m2 = q(torch.from_numpy(o), torch.from_numpy(m), torch.from_numpy(c))
        return s.numpy(), m2.numpy()
    out["int8_dynamic"] = int8
    if gpu:
        import copy
        g32 = copy.deepcopy(net).cuda().eval()
        g16 = copy.deepcopy(net).cuda().half().eval()

        def gpu_fn(model, dt):
            def f(o, m, c):
                with torch.no_grad():
                    s, m2 = model(torch.from_numpy(o).cuda().to(dt), torch.from_numpy(m).cuda().to(dt),
                                  torch.from_numpy(c).cuda().to(dt))
                    torch.cuda.synchronize()
                return s.float().cpu().numpy(), m2.float().cpu().numpy()
            return f
        out["gpu_fp32"] = gpu_fn(g32, torch.float32)
        out["gpu_fp16"] = gpu_fn(g16, torch.float16)
    return out


def _time(fn, args, reps: int, gpu: bool) -> dict:
    fn(*args)                                   # warm up
    walls, cpus = [], []
    for _ in range(reps):
        t, c = time.perf_counter(), time.process_time()
        fn(*args)
        walls.append(time.perf_counter() - t)
        cpus.append(time.process_time() - c)
    w = float(np.median(walls))
    return {"wall_s": w, "cpu_s": float(np.median(cpus)), "per_decision_us": w / args[0].shape[0] * 1e6}


def equivalence(ref_scores: np.ndarray, scores: np.ndarray, mask: np.ndarray, draws: int = 50, seed: int = 7) -> dict:
    """Behavioural equivalence of a backend's scores to the fp32 reference under the population's choice rule."""
    rng = np.random.default_rng(seed)
    B, K = ref_scores.shape
    agree, ref_counts, counts, loss = 0, np.zeros((B, K)), np.zeros((B, K)), 0.0
    for _ in range(draws):
        g = rng.gumbel(size=(B, K)) * TEMPERATURE
        a = np.argmax(np.where(mask, ref_scores + g, -np.inf), 1)
        b = np.argmax(np.where(mask, scores + g, -np.inf), 1)
        agree += int((a == b).sum())
        ref_counts[np.arange(B), a] += 1
        counts[np.arange(B), b] += 1
        loss += float((ref_scores[np.arange(B), a] - ref_scores[np.arange(B), b]).mean())
    tv = float(0.5 * np.abs(ref_counts - counts).sum(1).mean() / draws)
    ag = agree / (B * draws)
    return {"paired_agreement": round(ag, 5), "tv_distance": round(tv, 5), "mean_utility_loss": round(loss / draws, 6),
            "max_abs_score_diff": float(np.max(np.abs(np.where(mask, ref_scores - scores, 0.0)))),
            "equivalent": bool(ag >= EQUIV_AGREEMENT and tv <= EQUIV_TV)}


def bench_adaptive(batches=(1, 8, 32, 64, 256, 1024), reps: int = 5, seed: int = 0, gpu: bool | None = None) -> dict:
    torch = _torch()
    if gpu is None:
        gpu = bool(torch.cuda.is_available())
    big = adaptive_workload(max(batches), seed)
    net = big["net"]
    t = time.process_time()
    bk = make_backends(net, (big["obs"][:8], big["mem"][:8], big["cand"][:8]), gpu)
    setup_cpu = time.process_time() - t
    rows = []
    for B in batches:
        args = (big["obs"][:B].copy(), big["mem"][:B].copy(), big["cand"][:B].copy())
        mask = big["mask"][:B]
        ref, _ = bk["torch_cpu"](*args)
        for name, fn in bk.items():
            if isinstance(fn, Exception):
                rows.append({"backend": name, "B": B, "available": False, "error": f"{type(fn).__name__}: {fn}"[:300]})
                continue
            if name == "torch_per_participant" and B > 256:
                continue
            tm = _time(fn, args, reps if B >= 32 else reps * 4, name.startswith("gpu"))
            s, _ = fn(*args)
            eq = equivalence(ref, s.astype(np.float64), mask)
            rows.append({"backend": name, "B": B, "available": True, **tm,
                         "decisions_per_s": round(B / max(tm["wall_s"], 1e-12), 1),
                         "decisions_per_cpu_s": round(B / max(tm["cpu_s"], 1e-12), 1) if not name.startswith("gpu")
                         else None, **eq})
    return {"workload": "class-2 AdaptiveNet", "dims": big["dims"], "params": big["params"], "rows": rows,
            "setup_cpu_s": round(setup_cpu, 2), "gpu": gpu, "threads": 1}


def choose(bench: dict) -> dict:
    """Per batch size: the fastest backend whose choices are behaviourally equivalent to fp32 (wall time; GPU
    backends compete only when the GPU was part of the benchmark)."""
    out = {}
    for B in sorted({r["B"] for r in bench["rows"]}):
        cands = [r for r in bench["rows"] if r["B"] == B and r.get("available") and r.get("equivalent")]
        if not cands:
            continue
        best = min(cands, key=lambda r: r["wall_s"])
        cpu_only = [r for r in cands if not r["backend"].startswith("gpu")]
        best_cpu = min(cpu_only, key=lambda r: r["wall_s"]) if cpu_only else None
        rejected = [r["backend"] for r in bench["rows"] if r["B"] == B and r.get("available")
                    and not r.get("equivalent")]
        out[str(B)] = {"best": best["backend"], "best_wall_s": best["wall_s"],
                       "best_cpu": best_cpu["backend"] if best_cpu else None,
                       "rejected_not_equivalent": rejected}
    return out


def batching_speedup(bench: dict, batch_sizes: list[int] | None = None) -> dict:
    """Batched (torch_cpu) vs one pass per participant at equal decisions; with `batch_sizes` (the in-world batch
    sizes of a run), the cost of those exact batches each way (per-participant cost from the measured per-decision
    time, batched cost interpolated over the measured batch sizes)."""
    rows = {(r["backend"], r["B"]): r for r in bench["rows"] if r.get("available")}
    Bs = sorted(B for (b, B) in rows if b == "torch_per_participant")
    per = {B: rows[("torch_per_participant", B)]["wall_s"] / B for B in Bs}
    bat = {B: rows[("torch_cpu", B)]["wall_s"] for (b, B) in rows if b == "torch_cpu"}
    micro = {str(B): round(per[B] * B / bat[B], 3) for B in Bs if B in bat}
    out = {"micro": micro}
    if batch_sizes:
        xs = sorted(bat)
        per_dec = float(np.median(list(per.values())))
        b_cost = sum(float(np.interp(b, xs, [bat[x] for x in xs])) for b in batch_sizes)
        p_cost = per_dec * sum(batch_sizes)
        out["in_world"] = {"batches": len(batch_sizes), "decisions": int(sum(batch_sizes)),
                           "mean_batch": round(float(np.mean(batch_sizes)), 2), "max_batch": int(max(batch_sizes)),
                           "batched_s": round(b_cost, 6), "per_participant_s": round(p_cost, 6),
                           "speedup": round(p_cost / max(b_cost, 1e-12), 3)}
    return out


def bench_strategic(B: int = 100, reps: int = 5, gpu: bool | None = None) -> dict:
    """The class-3 architectures at their default sizes: batched vs per-participant, CPU and GPU."""
    torch = _torch()
    from ..population import nets
    if gpu is None:
        gpu = bool(torch.cuda.is_available())
    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    obs_dim, heads = 64, {"price": 9, "product": 6, "hire": 4, "invest": 5}
    out = {}
    for name, cls in (("mlp", nets.StrategicMLP), ("gru", nets.StrategicGRU)):
        net = cls(obs_dim, heads)
        nets.init_params(net, np.random.default_rng(1))
        net.eval()
        obs = torch.from_numpy(rng.normal(0, 1, (B, obs_dim)).astype(np.float32))
        hist = torch.from_numpy(rng.normal(0, 1, (B, 8, obs_dim)).astype(np.float32)) if name == "gru" else None

        def batched():
            with torch.no_grad():
                net(obs, hist)

        def single():
            with torch.no_grad():
                for i in range(B):
                    net(obs[i:i + 1], hist[i:i + 1] if hist is not None else None)
        r = {"params": int(nets.n_params(net)), "B": B,
             "batched_s": _time(lambda *_: batched(), (np.zeros((B, 1)),), reps, False)["wall_s"],
             "per_participant_s": _time(lambda *_: single(), (np.zeros((B, 1)),), max(2, reps // 2), False)["wall_s"]}
        r["speedup"] = round(r["per_participant_s"] / max(r["batched_s"], 1e-12), 3)
        if gpu:
            g = net.cuda()
            go, gh = obs.cuda(), (hist.cuda() if hist is not None else None)

            def gb():
                with torch.no_grad():
                    g(go, gh)
                torch.cuda.synchronize()
            r["gpu_batched_s"] = _time(lambda *_: gb(), (np.zeros((B, 1)),), reps, True)["wall_s"]
            net.cpu()
        out[name] = r
    return out


def bench_encoder(n: int = 16, seed: int = 0) -> dict:
    """The shared space's text pipeline (encoder + language-model features) on n page-sized texts: one batched
    call vs one call per text (fresh texts each time, so no cache hit hides the cost)."""
    from ..semantic import space
    rng = np.random.default_rng(seed)
    words = ("market price people city water energy school model data research company local team new growth "
             "network policy report health science music travel food sports history design open market").split()

    def texts(tag):
        return [(" ".join(rng.choice(words, 160)) + f" {tag}-{i}").encode() for i in range(n)]
    t = time.perf_counter()
    space.vectors_of_bytes(texts("warm"))
    warm = time.perf_counter() - t
    a = texts("batched")
    t = time.perf_counter()
    space.vectors_of_bytes(a)
    batched = time.perf_counter() - t
    b = texts("single")
    t = time.perf_counter()
    for x in b:
        space.vectors_of_bytes([x])
    single = time.perf_counter() - t
    return {"texts": n, "words_per_text": 161, "batched_s": round(batched, 4), "per_item_s": round(single, 4),
            "speedup": round(single / max(batched, 1e-12), 3), "items_per_s_batched": round(n / batched, 3),
            "warmup_s": round(warm, 3), "threads": int(_torch().get_num_threads()),
            "device": os.environ.get("CUDA_VISIBLE_DEVICES", "unset")}
