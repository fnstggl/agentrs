"""Length control + acting throughput on Qwen3-1.7B (cfg.sem_model) over the real ow_tiny catalogue - slow; run under
the shared GPU lock, one condition group per locked job:

    CAM_LEN_BENCH=baseline,length   flock /workspace/agentrs/runs/cam_unified/gpu_heavy.lock \\
        python -m pytest -q -s -m slow superlearner/tests/cam_unified/semantic/test_benchmark_length.py
    CAM_LEN_BENCH=merged,trained    flock ... (same command)

Conditions (untrained decoder: random R/G/B, zero-initialised LoRA unless stated; dev config, bf16, eager decode):
* baseline - wave-1 decoding: every free string capped at cfg.max_arg_tokens, no length prior, LoRA path when
  acting (a subset of the tools: it is slow by construction, which is the point);
* length   - schema-derived caps + LengthHead prior, LoRA path when acting;
* merged   - length control + merged-LoRA acting (contract v3 #12; zero delta at init = adapter disabled);
* trained  - a non-zero LoRA delta (as after training): LoRA path vs merged acting, same tools and seed.
Per condition and batch size B in {1, 8}: decoded string tokens per action (mean / max), Qwen decode steps, schema
validity, wall time per action of `sample()` (which includes the learner-path rescoring that reports logp), and a
decode-step microbenchmark (LoRA path vs acting path). Results are merged into
`benchmarks/length_control_qwen3_1p7b.json` next to this file (committed)."""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import time

import pytest
import torch

from superlearner.cam_unified.config import make_config
from superlearner.cam_unified.interfaces import EntityRegistry, ToolSchema
from superlearner.cam_unified.testing import sample_observations

from sem_helpers import make_context
from test_benchmark_qwen17b import _gpu_sharing

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "benchmarks", "length_control_qwen3_1p7b.json")


def _validate(schema, value):
    from crucible.core.schema import validate
    return validate(schema, value)


def _ow_tiny_tools():
    import crucible
    crucible.load_all_modules()
    from crucible.api import WorldServer
    from crucible.worldgen import generate
    srv = WorldServer("memory://", lease_ttl_s=None)
    wid = srv.create_world(generate({"family": "ow_tiny"}, 1))
    me = srv.join_world(wid, "nova", endowment=3_000_000)
    return [ToolSchema(t["name"], t.get("description", ""), t["inputSchema"],
                       duration_hint=(t.get("annotations") or {}).get("costSeconds", 0) / 86400.0)
            for t in srv.available_tools(wid, me)]


def _run(gen, tok, cfg, tools, names, B, seed=0):
    """Sample one action per named tool (forced by the tool mask) in batches of B; returns the measurements."""
    obs = sample_observations(5, seed=2)
    all_names = [t.name for t in tools]
    g = torch.Generator(device=gen.device).manual_seed(seed)
    toks, steps, times, invalid, prompt = [], [], [], 0, []
    for s in range(0, len(names), B):
        rows = names[s: s + B]
        n = len(rows)
        tl = [tools] * n
        R, goal, belief = make_context(cfg, n, seed=seed + s)
        T = tok.tools(tl)
        O = tok.observations([obs] * n, [EntityRegistry(cfg.max_entities, seed=b) for b in range(n)], [0.0] * n)
        m = torch.zeros(n, T.N + 2, dtype=torch.bool)
        for b, nm in enumerate(rows):
            m[b, all_names.index(nm)] = True
        torch.cuda.synchronize()
        t0 = time.time()
        acts = gen.sample(R, goal, belief, T, O, tl, [obs] * n, tool_mask=m, generator=g)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        steps.append(gen.last_compute.get("qwen_steps", 0))
        prompt.append(gen.last_compute.get("prompt_tokens", 0))
        for nm, a in zip(rows, acts):
            assert a.tool == nm
            invalid += bool(_validate(next(t for t in tools if t.name == nm).parameters, a.args))
            toks.append(a.logp_parts["n_tokens"])
    n = len(toks)
    return {"B": B, "n_actions": n, "schema_valid": n - invalid, "schema_valid_frac": (n - invalid) / n,
            "tokens_mean": round(sum(toks) / n, 2), "tokens_max": max(toks),
            "tokens_p90": sorted(toks)[int(0.9 * (n - 1))], "qwen_steps_total": sum(steps),
            "prompt_tokens_mean_per_batch": round(sum(prompt) / len(prompt), 1),
            "wall_s_total": round(sum(times), 2), "wall_s_per_action": round(sum(times) / n, 4)}


def _step_bench(enc, B, L=768, reps=30):
    """ms per decoder step (B rows, L-token cache): LoRA path vs acting path."""
    out = {}
    ids = torch.randint(100, 20000, (B, L), device=enc.device)
    att = torch.ones_like(ids)
    pos = torch.arange(L, device=enc.device).unsqueeze(0).expand(B, -1)
    for name, ctx in (("lora_path", lambda: torch.no_grad()), ("acting_path", lambda: enc.acting())):
        with torch.no_grad(), ctx():
            _, cache = enc.step(enc.embed_ids(ids), att, pos, enc.new_cache())
            a2, cur = att, pos[:, -1:]
            vals = []
            for r in range(reps):
                a2 = torch.cat([a2, torch.ones(B, 1, dtype=att.dtype, device=enc.device)], 1)
                cur = cur + 1
                torch.cuda.synchronize()
                t0 = time.time()
                _, cache = enc.step(enc.embed_ids(ids[:, :1]), a2, cur, cache)
                torch.cuda.synchronize()
                vals.append(1000 * (time.time() - t0))
        out[name] = {"median_ms": round(statistics.median(vals[5:]), 2), "min_ms": round(min(vals[5:]), 2)}
    return out


@pytest.mark.slow
def test_length_control_and_acting_throughput_qwen3_1p7b(cuda):
    from superlearner.cam_unified.semantic.actions import ActionGenerator
    from superlearner.cam_unified.semantic.encoder import QwenSemantic
    from superlearner.cam_unified.semantic.grammar import GrammarOptions
    from superlearner.cam_unified.semantic.tokenizer import ObservationTokenizer

    conds = [c.strip() for c in os.environ.get("CAM_LEN_BENCH", "length,merged").split(",") if c.strip()]
    cfg = make_config("dev")
    assert cfg.sem_model == "Qwen/Qwen3-1.7B"
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault("model", cfg.sem_model)
    res.setdefault("conditions", {})
    meta = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "load_avg_start": os.getloadavg(),
            "gpu_start": _gpu_sharing()}
    try:
        meta["commit"] = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True,
                                        text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    enc = QwenSemantic(cfg)
    tok = ObservationTokenizer(cfg, enc)
    tools = _ow_tiny_tools()
    names = [t.name for t in tools]
    subset = names[::9]
    for c in conds:
        torch.manual_seed(0)
        rec = {"meta": dict(meta), "runs": []}
        if c == "baseline":
            opts = GrammarOptions(max_string_tokens=cfg.max_arg_tokens, max_action_tokens=2 * cfg.max_arg_tokens,
                                  short_string_tokens=cfg.max_arg_tokens, text_string_tokens=cfg.max_arg_tokens,
                                  length_prior=False)
            gen = ActionGenerator(cfg, enc, tok, opts=opts, merged_acting=False)
            rec["tools"] = "every 9th ow_tiny tool (%d)" % len(subset)
            rec["runs"] = [_run(gen, tok, cfg, tools, subset, 8), _run(gen, tok, cfg, tools, subset, 1)]
        elif c in ("length", "merged"):
            gen = ActionGenerator(cfg, enc, tok, merged_acting=(c == "merged"))
            rec["tools"] = "all %d ow_tiny tools" % len(names)
            rec["runs"] = [_run(gen, tok, cfg, tools, names, 8), _run(gen, tok, cfg, tools, names, 1)]
            if c == "length":
                rec["subset_runs"] = [_run(gen, tok, cfg, tools, subset, 8), _run(gen, tok, cfg, tools, subset, 1)]
            rec["expected_length_prior"] = {k: round(gen.length_head.expected_length(0.0, k), 1)
                                            for k in ("short", "text", "code")}
            rec["step_ms"] = {"B1": _step_bench(enc, 1), "B8": _step_bench(enc, 8)}
        elif c == "trained":
            layers = enc._lora_layers()
            saved = [Bw.detach().clone() for _, _, Bw, _ in layers]
            gcpu = torch.Generator().manual_seed(0)
            with torch.no_grad():
                for _, _, Bw, _ in layers:
                    Bw.copy_((torch.randn(Bw.shape, generator=gcpu) * 0.02).to(Bw.device, Bw.dtype))
            enc.mark_updated()
            try:
                for merged in (False, True):
                    gen = ActionGenerator(cfg, enc, tok, merged_acting=merged)
                    r8 = _run(gen, tok, cfg, tools, names[::3], 8)
                    r1 = _run(gen, tok, cfg, tools, names[::3], 1)
                    rec["runs"] += [{**r8, "merged_acting": merged}, {**r1, "merged_acting": merged}]
                rec["tools"] = "every 3rd ow_tiny tool (%d), LoRA B ~ N(0, 0.02^2)" % len(names[::3])
                rec["step_ms"] = {"B1": _step_bench(enc, 1), "B8": _step_bench(enc, 8)}
            finally:
                with torch.no_grad():
                    for (_, _, Bw, _), s in zip(layers, saved):
                        Bw.copy_(s)
                enc.mark_updated()
        rec["meta"]["load_avg_end"] = os.getloadavg()
        rec["meta"]["gpu_end"] = _gpu_sharing()
        rec["meta"]["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
        res["conditions"][c] = rec
        for r in rec["runs"]:
            assert r["schema_valid"] == r["n_actions"]
        with open(OUT, "w") as f:
            json.dump(res, f, indent=1)
        print(c, json.dumps(rec, indent=1))
