"""Per-capability model routing and the selective-GPU policy of the shared semantic service.

**Routing** (MASTER_SPEC 3.2 class 4: "use the smallest model or combination of models that meets each capability
requirement"; DECISION_LOG D-10). `route(capability)` names the model for a decision of that capability:

* default: ``Qwen/Qwen3-0.6B`` (smallest effective model for content understanding, situation interpretation, tool
  choice, communication and incentive adaptation in SP-3.2b);
* escalation to ``Qwen/Qwen3-1.7B`` for ``programming``, ``negotiation`` and ``strategic_decisions``, where the
  SP-3.2b held-out results show the 0.6B model is not effective.

When an SP-3.2b report is present (`$CRUCIBLE_DATA/semantic/capbench/sp32b.json`, written by `capbench.run`), its
per-capability recommendation on held-out templates replaces this table entry by entry (`table()` says which
source each entry came from). `service(capability)` returns the process-wide `SemanticService` of the routed model,
so every participant routed to a model shares one weight copy.

**Selective GPU** (13.4). `device_for(model_id)` puts a model on the GPU only when the slot scheduler granted GPU
memory to this process (`CUDA_VISIBLE_DEVICES` non-empty) and the grant (`CRUCIBLE_SLOT_GPU_MB`, set by
`crucible.scale.slots`) holds the weights with room for activations; otherwise CPU (float32, or int8 dynamic
quantization for models of 1B parameters and more, which are too slow on CPU in float32). `limit_gpu_memory()`
caps this process's CUDA allocator at the granted MB, so a run cannot exceed its slot.
"""
from __future__ import annotations

import json
import os

DEFAULT = "Qwen/Qwen3-0.6B"
ESCALATION = "Qwen/Qwen3-1.7B"
TABLE = {"content_understanding": DEFAULT, "situation_interpretation": DEFAULT, "tool_choice": DEFAULT,
         "communication": DEFAULT, "incentive_adaptation": DEFAULT, "negotiation": ESCALATION,
         "programming": ESCALATION, "strategic_decisions": ESCALATION}
PARAMS = {"Qwen/Qwen3-0.6B": 596_049_920, "Qwen/Qwen3-1.7B": 1_720_574_976,
          "HuggingFaceTB/SmolLM2-135M-Instruct": 134_515_008, "HuggingFaceTB/SmolLM2-360M-Instruct": 361_821_120}
HEADROOM = 1.6          # weights x this must fit the GPU grant (activations, KV cache, allocator)


def report_path() -> str:
    return os.path.join(os.environ.get("CRUCIBLE_DATA", "/workspace/data/crucible"), "semantic", "capbench",
                        "sp32b.json")


DECISION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "routing.json")
EFFECTIVE_ACCURACY = 0.6     # SP-3.2b effectiveness: accuracy (or generation pass rate) >= 0.6, p <= 0.01


def decision() -> dict | None:
    """The committed routing decision (`data/routing.json`): per capability the held-out measurements of every
    benchmarked model, the routed model and why (derived by `derive` from an SP-3.2b report)."""
    try:
        with open(DECISION_FILE) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def derive(report: dict) -> dict:
    """Routes from an SP-3.2b report: for each capability the SMALLEST model effective on held-out templates
    (`recommendation_held_out`); where no candidate is effective, the best-measured language model (largest value,
    then smallest size). Returns {capability: {model, reason, measured: {model: value}}}."""
    from .capbench.run import primary
    out = {}
    lms = [r for r in report.get("results", []) if (r.get("model") or {}).get("name") in PARAMS]
    rec = report.get("recommendation_held_out") or {}
    for cap in TABLE:
        meas = {}
        for r in lms:
            held = ((r.get("capabilities") or {}).get(cap) or {})
            p = primary(cap, held)
            ho = (held.get("mc") or {}).get("held_out_templates") or {}
            meas[r["model"]["name"]] = ho.get("accuracy", p.get("value"))
        r = rec.get(cap) or {}
        model = r.get("recommended") if isinstance(r, dict) else None
        if model in PARAMS:
            out[cap] = {"model": model, "reason": "smallest model effective on held-out templates", "measured": meas}
        else:
            vals = [(v, -PARAMS[m], m) for m, v in meas.items() if v is not None]
            best = max(vals)[2] if vals else DEFAULT
            out[cap] = {"model": best, "reason": "no candidate effective on held-out templates: best measured model",
                        "measured": meas}
    return out


def table() -> dict[str, dict]:
    """capability -> {model, source}: the committed routing decision (derived from the SP-3.2b report), else an
    SP-3.2b report under $CRUCIBLE_DATA, else D-10."""
    out = {c: {"model": m, "source": "DECISION_LOG D-10 (SP-3.2b held-out results)"} for c, m in TABLE.items()}
    dec = decision()
    if dec:
        for cap, r in (dec.get("routes") or {}).items():
            if cap in out and r.get("model") in PARAMS:
                out[cap] = {"model": r["model"], "source": f"{DECISION_FILE} ({r.get('reason', '')})"}
        return out
    try:
        with open(report_path()) as fh:
            rep = json.load(fh)
        rec = rep.get("recommendation_held_out") or {}
        for cap, r in rec.items():
            # capbench.run.recommend_held_out writes {"recommended": <model name>, ...}
            model = (r.get("recommended") or r.get("model")) if isinstance(r, dict) else r
            if cap in out and isinstance(model, str) and "/" in model:
                out[cap] = {"model": model, "source": report_path()}
    except (OSError, ValueError):
        pass
    return out


def route(capability: str) -> str:
    return table().get(capability, {"model": DEFAULT})["model"]


def granted_gpu_mb() -> int:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis in ("", "-1"):
        return 0
    try:
        return int(os.environ.get("CRUCIBLE_SLOT_GPU_MB", "0") or 0)
    except ValueError:
        return 0


def device_for(model_id: str, dtype_bytes: int = 2) -> tuple[str, str | None]:
    """(device, quantization) for a model under this process's slot."""
    params = PARAMS.get(model_id, 600_000_000)
    need_mb = params * dtype_bytes * HEADROOM / 2 ** 20
    try:
        import torch
        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = False
    gmb = granted_gpu_mb()
    if cuda and gmb and need_mb <= gmb:
        return "cuda", None
    return "cpu", ("int8-dynamic" if params >= 1_000_000_000 else None)


def limit_gpu_memory() -> float | None:
    """Cap the CUDA allocator of this process at the slot's grant; returns the fraction set (None without a grant)."""
    gmb = granted_gpu_mb()
    if not gmb:
        return None
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
        frac = max(0.01, min(1.0, gmb / total))
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        return frac
    except Exception:  # noqa: BLE001
        return None


_SERVICES: dict[str, object] = {}


def service_for_model(model: str, **kw):
    """The process-wide shared service of `model` on the device the slot allows (one per model and process)."""
    from .service import SemanticService
    s = _SERVICES.get(model)
    if s is None:
        dev, quant = device_for(model)
        if dev == "cuda":
            limit_gpu_memory()
        s = _SERVICES[model] = SemanticService(model, device=dev, quantization=quant, **kw)
    return s


def service(capability: str, **kw):
    """The shared service of the model routed for `capability` (one per model and process)."""
    return service_for_model(route(capability), **kw)


# ====================================================================== licences and tool use of the checkpoints
def model_card(model_id: str) -> dict:
    """Licence fields of the local model card (README front matter) of `model_id`; no network."""
    from .service import hub_cache_dir, local_revision
    rev = local_revision(model_id)
    path = os.path.join(hub_cache_dir(), "models--" + model_id.replace("/", "--"), "snapshots", rev, "README.md")
    out = {"model": model_id, "revision": rev, "card": path, "license": None, "license_link": None}
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return out
    if text.startswith("---"):
        head = text.split("---", 2)[1]
        for line in head.splitlines():
            k, _, v = line.partition(":")
            if k.strip() in ("license", "license_link") and v.strip():
                out[k.strip()] = v.strip()
    return out


TOOL_PROBES = (
    ("Send a mail to p7 with the subject 'Lunch' saying that lunch is at noon.", "mail_send"),
    ("Pay 2500 cents from my account acct_000012 to p9.", "bank_transfer"),
    ("Create a public document titled 'Opening hours' that says we open at 9.", "docs_create"),
    ("Look up how much money is in my account acct_000012.", "bank_balance"),
)
PROBE_TOOLS = [
    {"type": "function", "function": {"name": "mail_send", "description": "Send a private message.",
                                      "parameters": {"type": "object", "properties": {
                                          "to": {"type": "array", "items": {"type": "string"}},
                                          "subject": {"type": "string"}, "body": {"type": "string"}},
                                          "required": ["to", "subject", "body"]}}},
    {"type": "function", "function": {"name": "bank_transfer", "description": "Send money from an account.",
                                      "parameters": {"type": "object", "properties": {
                                          "from_account": {"type": "string"}, "to": {"type": "string"},
                                          "amount": {"type": "integer"}}, "required": ["from_account", "to", "amount"]}}},
    {"type": "function", "function": {"name": "docs_create", "description": "Create a document.",
                                      "parameters": {"type": "object", "properties": {
                                          "title": {"type": "string"}, "body": {"type": "string"},
                                          "visibility": {"type": "string", "enum": ["private", "public"]}},
                                          "required": ["title", "body"]}}},
    {"type": "function", "function": {"name": "bank_balance", "description": "Balance of an account.",
                                      "parameters": {"type": "object", "properties": {"account": {"type": "string"}},
                                                     "required": ["account"]}}},
]


def tool_use(model_id: str, device: str | None = None) -> dict:
    """Tool-use capability of a checkpoint, verified by running it: (a) whether its chat template renders tool
    definitions (native function calling); (b) native calls: for each probe request, the model's reply parsed as a
    tool call (`<tool_call>{...}</tool_call>` or a bare JSON object) naming the right tool with valid JSON
    arguments; (c) choice-based selection (how semantic participants pick tools): the right tool's name has the
    highest mean log probability among the four."""
    import re
    from .service import SemanticService
    dev = device or device_for(model_id)[0]
    svc = SemanticService(model_id, device=dev, max_batch=4, max_new_tokens=96)
    tok = svc.tokenizer
    try:
        rendered = tok.apply_chat_template([{"role": "user", "content": "hi"}], tools=PROBE_TOOLS, tokenize=False,
                                           add_generation_prompt=True)
        native_template = "mail_send" in rendered
    except Exception:  # noqa: BLE001 - templates without tool support raise various errors: no native template
        native_template = False
    native_ok = 0
    if native_template:
        import torch
        for q, want in TOOL_PROBES:
            kw = {"enable_thinking": False} if "qwen3" in model_id.lower() else {}
            ids = tok.apply_chat_template([{"role": "user", "content": q}], tools=PROBE_TOOLS, tokenize=True,
                                          add_generation_prompt=True, return_tensors="pt", **kw)
            ids = ids["input_ids"] if hasattr(ids, "keys") else ids
            with torch.inference_mode():
                out = svc.model.generate(ids.to(svc.device), max_new_tokens=96, do_sample=False,
                                         pad_token_id=svc.weights.pad_id)
            text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
            m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.S) or re.search(r"(\{.*\})", text,
                                                                                                    flags=re.S)
            try:
                call = json.loads(m.group(1)) if m else {}
            except ValueError:
                call = {}
            native_ok += int(call.get("name") == want and isinstance(call.get("arguments"), dict))
    sysm = {"role": "system", "content": "Tools: " + "; ".join(f"{t['function']['name']}: {t['function']['description']}"
                                                                for t in PROBE_TOOLS)}
    reqs = [{"messages": [sysm, {"role": "user", "content": q + " Which tool do you call? Answer with its name."}],
             "choices": [t["function"]["name"] for t in PROBE_TOOLS], "normalize": "mean"} for q, _ in TOOL_PROBES]
    picks = [o["choice"] for o in svc.choose(None, reqs)]
    choose_ok = sum(int(PROBE_TOOLS[p]["function"]["name"] == w) for p, (_, w) in zip(picks, TOOL_PROBES))
    return {"model": model_id, "native_tool_template": native_template, "native_calls_correct": native_ok,
            "choice_correct": choose_ok, "probes": len(TOOL_PROBES), "device": svc.device,
            "verified": (native_ok >= 3) or (choose_ok >= 3)}
