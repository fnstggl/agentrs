"""Qwen3-1.7B (cfg.sem_model) throughput benchmark - slow; run under the shared GPU lock:

    flock /workspace/agentrs/runs/cam_unified/gpu_heavy.lock \\
        python -m pytest -q -s -m slow superlearner/tests/cam_unified/semantic/test_benchmark_qwen17b.py

Measures (bf16, LoRA attached, eager decode), each repeated CAM_BENCH_REPS times (default 3; 2 for the 1024-token
cases) and reported as {min, median, max} because the GPU and CPUs are shared with other jobs: encoder prefill
tokens/s and cache-hit cost, observation tokenisation of a realistic step, ActionGenerator decode latency for exactly
64- and 1024-token strings (pattern holes: one token per character, so the token count is exact) at B=1 and B=8 and a
free-text JSON-string hole (vocabulary-wide masks), the learner-path rescoring that sample() runs by default, the
training-path log_prob forward+backward of a 1024-token action, and one decoder step with and without LoRA. The record
includes the host load average and the other GPU processes (shared-machine metadata) and is written to
$CAM_BENCH_OUT, default `benchmarks/qwen3_1p7b.json` next to this file (committed)."""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import time

import pytest
import torch

from superlearner.cam_unified.config import make_config
from superlearner.cam_unified.interfaces import EntityRegistry, Observation, ObsKind, TokenKind, ToolSchema
from superlearner.cam_unified.testing import rand_tokens, sample_observations

HERE = os.path.dirname(os.path.abspath(__file__))


def _sync():
    torch.cuda.synchronize()
    return time.time()


def _rng(vals):
    return {"min": round(min(vals), 4), "median": round(statistics.median(vals), 4), "max": round(max(vals), 4),
            "n": len(vals)}


def _gpu_sharing() -> dict:
    out = {}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        name, used, total, util = [x.strip() for x in q.strip().splitlines()[0].split(",")]
        out = {"gpu": name, "memory_used_mb": int(used), "memory_total_mb": int(total), "utilization_pct": int(util)}
        p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout
        apps = [ln.split(",") for ln in p.strip().splitlines() if ln.strip()]
        out["compute_processes"] = len(apps)
        out["other_processes_mb"] = sum(int(m) for pid, m in apps if int(pid) != os.getpid())
    except Exception as e:  # noqa: BLE001 - metadata only
        out["error"] = f"{type(e).__name__}: {e}"
    return out


@pytest.mark.slow
def test_qwen3_1p7b_throughput(cuda):
    from superlearner.cam_unified.semantic.actions import ActionGenerator
    from superlearner.cam_unified.semantic.encoder import QwenSemantic
    from superlearner.cam_unified.semantic.grammar import GrammarOptions
    from superlearner.cam_unified.semantic.tokenizer import ObservationTokenizer

    reps = int(os.environ.get("CAM_BENCH_REPS", "3"))
    cfg = make_config("dev")
    assert cfg.sem_model == "Qwen/Qwen3-1.7B"
    out: dict = {"model": cfg.sem_model, "scale": "dev", "d_model": cfg.d_model, "reps": reps,
                 "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "load_avg_start": os.getloadavg(),
                 "gpu_start": _gpu_sharing(), "cpu_count": os.cpu_count(),
                 "omp_threads": os.environ.get("OMP_NUM_THREADS")}
    try:
        out["commit"] = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True,
                                       text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    enc = QwenSemantic(cfg)
    out["load_s"] = round(time.time() - t0, 2)
    out["params"] = enc.parameter_counts()
    assert out["params"]["pretrained"] > 1.5e9 and out["params"]["lora"] > 0
    # ---- encoder prefill (fresh texts per repetition: every repetition is a cache miss)
    enc_tps, hit_s = [], []
    for r in range(reps):
        texts = [(" ".join(f"w{i}x{j}r{r}" for j in range(400)))[:2400] for i in range(64)]
        n_tok = sum(len(enc._token_ids(t)) for t in texts)
        m0 = enc.stats["misses"]
        t0 = _sync()
        enc.token_states(texts)
        t1 = _sync()
        enc.token_states(texts)
        t2 = _sync()
        assert enc.stats["misses"] - m0 == len(texts)
        enc_tps.append(n_tok / (t1 - t0))
        hit_s.append(t2 - t1)
    out["encode"] = {"texts": 64, "tokens": n_tok, "tokens_per_s": _rng(enc_tps), "cache_hit_s": _rng(hit_s)}
    # ---- tokenizer: one realistic decision step (8 actors x 24 observations)
    tok = ObservationTokenizer(cfg, enc)
    cold, warm = [], []
    for r in range(reps):
        obs_lists = [[Observation(ObsKind.TEXT, f"actor {b} event {i} rep {r}: " + "details " * 30, t=float(i),
                                  source="feed") for i in range(20)] + sample_observations(4, seed=b) for b in range(8)]
        t0 = _sync()
        ts = tok.observations(obs_lists, [EntityRegistry(cfg.max_entities, seed=b) for b in range(8)], [30.0] * 8)
        t1 = _sync()
        ts = tok.observations(obs_lists, [EntityRegistry(cfg.max_entities, seed=b) for b in range(8)], [30.0] * 8)
        t2 = _sync()
        cold.append(t1 - t0)
        warm.append(t2 - t1)
    out["tokenize_step"] = {"rows": 8, "obs_per_row": 24, "tokens": int(ts.mask.sum()), "cold_s": _rng(cold),
                            "warm_s": _rng(warm)}
    # ---- action generation with exact string lengths
    gen = ActionGenerator(cfg, enc, tok, opts=GrammarOptions(max_string_tokens=1100, max_action_tokens=2200))

    def tool(n_chars):
        return ToolSchema(f"write_{n_chars}", "Write a text of the requested length.",
                          {"type": "object", "properties": {"text": {"type": "string",
                                                                     "pattern": f"[a-z ]{{{n_chars}}}"}},
                           "required": ["text"], "additionalProperties": False})

    free = ToolSchema("write_free", "Write a paragraph.",
                      {"type": "object", "properties": {"text": {"type": "string", "minLength": 300,
                                                                 "maxLength": 2000}},
                       "required": ["text"], "additionalProperties": False})
    gens = {}
    for name, t, B in [("str64_b1", tool(63), 1), ("str64_b8", tool(63), 8), ("str1024_b1", tool(1023), 1),
                       ("str1024_b8", tool(1023), 8), ("free_b8", free, 8)]:
        n_rep = reps if not name.startswith("str1024") else max(1, min(reps, 2))
        tool_lists = [[t]] * B
        obs_l = [sample_observations(4, seed=1)] * B
        dec_s, step_ms, tps, resc_s, drift = [], [], [], [], []
        for r in range(n_rep):
            R = torch.randn(B, cfg.n_state, cfg.d_model, device=cuda)
            goal = rand_tokens(B, 3, cfg.d_model, TokenKind.GOAL, device=cuda, ragged=False)
            belief = rand_tokens(B, cfg.n_particles + 1, cfg.d_model, TokenKind.BELIEF, device=cuda, ragged=False)
            tools = tok.tools(tool_lists)
            obs = tok.observations(obs_l, None, [0.0] * B)
            mask = torch.zeros(B, 3, dtype=torch.bool, device=cuda)
            mask[:, 0] = True
            t0 = _sync()
            acts = gen.sample(R, goal, belief, tools, obs, tool_lists, obs_l, tool_mask=mask, rescore=False)
            t1 = _sync()
            steps = max(gen.last_compute["qwen_steps"], 1)
            prompt_tokens = gen.last_compute["prompt_tokens"]
            with torch.no_grad():                        # the learner-path rescoring sample() does by default
                lp, _ = gen.log_prob(R, goal, belief, tools, obs, tool_lists, obs_l, acts)
            t2 = _sync()
            n = [a.logp_parts["n_tokens"] for a in acts]
            for a, k in zip(acts, n):
                assert gen.tool_lists_with_builtins([[t]])[0][0].name == a.tool
                if name.startswith("str"):
                    assert len(a.args["text"]) + 1 == k  # exact token count (one per character + closing quote)
                else:
                    assert 300 <= len(a.args["text"]) <= 2000
            dec_s.append(t1 - t0)
            step_ms.append(1000 * (t1 - t0) / steps)
            tps.append(sum(n) / (t1 - t0))
            resc_s.append(t2 - t1)
            drift.append(max(abs(float(lp[b]) - a.logp_parts["sample_logp"]) for b, a in enumerate(acts)))
        gens[name] = {"B": B, "tokens_per_action": n[0] if len(set(n)) == 1 else n, "decode_s": _rng(dec_s),
                      "ms_per_decode_step": _rng(step_ms), "tokens_per_s": _rng(tps), "rescore_s": _rng(resc_s),
                      "prompt_tokens": prompt_tokens, "max_incremental_vs_learner_nats": _rng(drift)}
        if name == "str1024_b1":                         # default path: behaviour logp == learner logp, + backward
            torch.cuda.reset_peak_memory_stats()
            a_def = gen.sample(R, goal, belief, tools, obs, tool_lists, obs_l, tool_mask=mask)
            Rg = R.clone().requires_grad_(True)
            fw, bw = [], []
            for r in range(n_rep):
                t0 = _sync()
                lp, ent = gen.log_prob(Rg, goal, belief, tools, obs, tool_lists, obs_l, a_def)
                t1 = _sync()
                lp.sum().backward()
                t2 = _sync()
                fw.append(t1 - t0)
                bw.append(t2 - t1)
                assert float(lp[0]) == a_def[0].logp          # PPO ratio exactly 1 at the behaviour parameters
                gen.zero_grad(set_to_none=True)
                Rg.grad = None
            out["log_prob_1024"] = {"forward_s": _rng(fw), "backward_s": _rng(bw),
                                    "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                    "default_path_abs_diff": abs(float(lp[0]) - a_def[0].logp),
                                    "incremental_vs_learner_nats": round(abs(a_def[0].logp
                                                                             - a_def[0].logp_parts["sample_logp"]), 4)}
    out["generate"] = gens
    with torch.no_grad():                                 # reference: one decoder step with and without LoRA
        ids = torch.full((8, 1), 1000, device=cuda)
        e = enc.embed_ids(enc.tok("x " * 300, return_tensors="pt")["input_ids"].to(cuda).repeat(8, 1))
        att = torch.ones(e.shape[:2], dtype=torch.long, device=cuda)
        for label, ctx in (("lora_on", None), ("lora_off", enc.peft.disable_adapter)):
            vals = []
            for r in range(reps):
                cm = ctx() if ctx else torch.no_grad()
                with cm:
                    cache = enc.new_cache()
                    _, cache = enc.step(e, att, (att.cumsum(1) - 1), cache)
                    a2, p2 = att, (att.sum(1, keepdim=True) - 1)
                    t0 = _sync()
                    for _ in range(20):
                        a2 = torch.cat([a2, torch.ones(8, 1, dtype=torch.long, device=cuda)], 1)
                        p2 = p2 + 1
                        _, cache = enc.step(enc.embed_ids(ids), a2, p2, cache)
                    t1 = _sync()
                vals.append((t1 - t0) / 20 * 1000)
            out[f"decode_step_ms_{label}"] = _rng(vals)
    out["peak_gb_total"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    out["load_avg_end"] = os.getloadavg()
    out["gpu_end"] = _gpu_sharing()
    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    assert out["encode"]["tokens_per_s"]["min"] > 1000 and out["log_prob_1024"]["default_path_abs_diff"] == 0.0
    print(json.dumps(out, indent=1))
    path = os.environ.get("CAM_BENCH_OUT") or os.path.join(HERE, "benchmarks", "qwen3_1p7b.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
