"""Qwen3-1.7B (cfg.sem_model) throughput benchmark - slow; run under the shared GPU lock:

    flock /workspace/agentrs/runs/cam_unified/gpu_heavy.lock \\
        python -m pytest -q -s -m slow superlearner/tests/cam_unified/semantic/test_benchmark_qwen17b.py

Measures (bf16, LoRA attached, eager decode): encoder prefill tokens/s and cache-hit cost, observation tokenisation of a
realistic step, end-to-end ActionGenerator latency for exactly 64- and 1024-token strings (pattern holes: one token per
character, so the token count is exact) at B=1 and B=8, a free-text JSON-string hole (vocabulary-wide masks), and the
training-path log_prob forward+backward of a 1024-token action. Results are printed as JSON and written to
$CAM_BENCH_OUT (if set)."""
from __future__ import annotations

import json
import os
import time

import pytest
import torch

from superlearner.cam_unified.config import make_config
from superlearner.cam_unified.interfaces import EntityRegistry, Observation, ObsKind, TokenKind, ToolSchema
from superlearner.cam_unified.testing import rand_tokens, sample_observations


def _sync():
    torch.cuda.synchronize()
    return time.time()


@pytest.mark.slow
def test_qwen3_1p7b_throughput(cuda):
    from superlearner.cam_unified.semantic.actions import ActionGenerator
    from superlearner.cam_unified.semantic.encoder import QwenSemantic
    from superlearner.cam_unified.semantic.grammar import GrammarOptions
    from superlearner.cam_unified.semantic.tokenizer import ObservationTokenizer

    cfg = make_config("dev")
    assert cfg.sem_model == "Qwen/Qwen3-1.7B"
    out: dict = {"model": cfg.sem_model, "scale": "dev", "d_model": cfg.d_model}
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    enc = QwenSemantic(cfg)
    out["load_s"] = round(time.time() - t0, 2)
    out["params"] = enc.parameter_counts()
    # ---- encoder prefill
    texts = [(" ".join(f"w{i}x{j}" for j in range(400)))[:2400] for i in range(64)]
    n_tok = sum(len(enc._token_ids(t)) for t in texts)
    t0 = _sync()
    enc.token_states(texts)
    t1 = _sync()
    enc.token_states(texts)
    t2 = _sync()
    out["encode"] = {"texts": len(texts), "tokens": n_tok, "s": round(t1 - t0, 3),
                     "tokens_per_s": round(n_tok / (t1 - t0)), "cache_hit_s": round(t2 - t1, 4)}
    # ---- tokenizer: one realistic decision step (8 actors x 24 observations)
    tok = ObservationTokenizer(cfg, enc)
    obs_lists = [[Observation(ObsKind.TEXT, f"actor {b} event {i}: " + "details " * 30, t=float(i), source="feed")
                  for i in range(20)] + sample_observations(4, seed=b) for b in range(8)]
    regs = [EntityRegistry(cfg.max_entities, seed=b) for b in range(8)]
    t0 = _sync()
    ts = tok.observations(obs_lists, regs, [30.0] * 8)
    t1 = _sync()
    ts = tok.observations(obs_lists, [EntityRegistry(cfg.max_entities, seed=b) for b in range(8)], [30.0] * 8)
    t2 = _sync()
    out["tokenize_step"] = {"rows": 8, "obs_per_row": 24, "tokens": int(ts.mask.sum()), "cold_s": round(t1 - t0, 3),
                            "warm_s": round(t2 - t1, 3)}
    # ---- end-to-end action generation with exact string lengths
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
        tool_lists = [[t]] * B
        R = torch.randn(B, cfg.n_state, cfg.d_model, device=cuda)
        goal = rand_tokens(B, 3, cfg.d_model, TokenKind.GOAL, device=cuda, ragged=False)
        belief = rand_tokens(B, cfg.n_particles + 1, cfg.d_model, TokenKind.BELIEF, device=cuda, ragged=False)
        tools = tok.tools(tool_lists)
        obs = tok.observations([sample_observations(4, seed=1)] * B, None, [0.0] * B)
        mask = torch.zeros(B, 3, dtype=torch.bool, device=cuda)
        mask[:, 0] = True
        t0 = _sync()
        acts = gen.sample(R, goal, belief, tools, obs, tool_lists, [sample_observations(4, seed=1)] * B,
                          tool_mask=mask)
        t1 = _sync()
        n = [a.logp_parts["n_tokens"] for a in acts]
        if name.startswith("str"):
            assert all(len(a.args["text"]) + 1 == k for a, k in zip(acts, n))
        gens[name] = {"B": B, "tokens_per_action": n[0] if len(set(n)) == 1 else n, "s": round(t1 - t0, 3),
                      "ms_per_decode_step": round(1000 * (t1 - t0) / max(gen.last_compute["qwen_steps"], 1), 1),
                      "tokens_per_s": round(sum(n) / (t1 - t0), 1), "prompt_tokens": gen.last_compute["prompt_tokens"]}
        if name == "str1024_b1":
            torch.cuda.reset_peak_memory_stats()
            Rg = R.clone().requires_grad_(True)
            t0 = _sync()
            lp, ent = gen.log_prob(Rg, goal, belief, tools, obs, tool_lists, [sample_observations(4, seed=1)], acts)
            t1 = _sync()
            lp.sum().backward()
            t2 = _sync()
            out["log_prob_1024"] = {"forward_s": round(t1 - t0, 3), "backward_s": round(t2 - t1, 3),
                                    "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                    "abs_logp_diff_vs_sampling": round(abs(float(lp[0]) - acts[0].logp), 4)}
            gen.zero_grad(set_to_none=True)
    out["generate"] = gens
    with torch.no_grad():                                 # reference: one decoder step with and without LoRA
        ids = torch.full((8, 1), 1000, device=cuda)
        e = enc.embed_ids(enc.tok("x " * 300, return_tensors="pt")["input_ids"].to(cuda).repeat(8, 1))
        att = torch.ones(e.shape[:2], dtype=torch.long, device=cuda)
        for label, ctx in (("lora_on", None), ("lora_off", enc.peft.disable_adapter)):
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
            out[f"decode_step_ms_{label}"] = round((t1 - t0) / 20 * 1000, 1)
    out["peak_gb_total"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    out["load_avg"] = os.getloadavg()
    print(json.dumps(out, indent=1))
    if os.environ.get("CAM_BENCH_OUT"):
        with open(os.environ["CAM_BENCH_OUT"], "w") as f:
            json.dump(out, f, indent=1)
