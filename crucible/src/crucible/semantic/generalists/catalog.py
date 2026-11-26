"""Tool retrieval over the universal tool interface (MASTER_SPEC 13.4 "retrieval of relevant state").

A participant sees every tool `World.available_tools(principal)` lists for it (the interface learners use through
`crucible.experience`), several hundred in a civilisation. Its prompt cannot hold them all, so each decision step
retrieves a shortlist: the cosine similarity between a query (the participant's goal and its latest situation) and
every tool's text (name, description, argument names) in the pretrained all-MiniLM-L6-v2 sentence space (the text
encoder of `crucible.semantic.encoders`). The model then chooses among the shortlist; nothing maps goals to tools.

Tool vectors are computed once per process for each distinct tool text and cached; queries are embedded per call.
The ranked shortlists of a step are recorded in the world (`crucible.substrate.inference.recorded`, kind
``semp.retrieve``), so replay and regeneration on another device return exactly the recorded shortlists.
"""
from __future__ import annotations

import hashlib
import threading

import numpy as np

# hosts whose tools are world meta-tools (descriptions of the API itself) or operator-only controls of other
# participant classes: never useful as a participant's action, kept out of shortlists
EXCLUDED_HOSTS = ("admin", "world", "pop_adaptive", "pop_strategic", "space")
_LOCK = threading.RLock()
_VEC: dict[str, np.ndarray] = {}
_IDENT: dict[str, str] = {}


def tool_text(t: dict) -> str:
    props = ((t.get("inputSchema") or {}).get("properties") or {})
    name = t.get("name", "")
    words = name.replace(".", " ").replace("_", " ")
    desc = " ".join(str(t.get("description") or "").split())
    return f"{words}: {desc} Arguments: {', '.join(sorted(props))}."[:600]


def usable(tools: list[dict]) -> list[dict]:
    return [t for t in tools if t.get("name", "").split(".", 1)[0] not in EXCLUDED_HOSTS]


def identity() -> tuple[str, str]:
    """(model id, model hash) of the sentence encoder used for retrieval."""
    with _LOCK:
        if not _IDENT:
            from ..encoders.models import weights_file_sha256
            from ..encoders.text import TEXT
            _IDENT["id"] = f"{TEXT.id}@{TEXT.version()}"
            _IDENT["hash"] = weights_file_sha256("text")
        return _IDENT["id"], _IDENT["hash"]


def embed(texts: list[str]) -> np.ndarray:
    """Unit MiniLM vectors of `texts` (cached per text in this process)."""
    from ..encoders.text import TEXT
    keys = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
    with _LOCK:
        todo = sorted({k: t for k, t in zip(keys, texts) if k not in _VEC}.items())
    if todo:
        vecs = TEXT.embed_texts([t for _, t in todo])
        with _LOCK:
            for (k, _), v in zip(todo, vecs):
                _VEC[k] = v.astype(np.float32)
            if len(_VEC) > 200_000:
                _VEC.clear()
    with _LOCK:
        return np.stack([_VEC[k] for k in keys]) if keys else np.zeros((0, 384), np.float32)


def rank(queries: list[str], tools: list[list[dict]], k: int) -> list[list[str]]:
    """For each query, the names of the `k` most similar of its tools (ties broken by name)."""
    out = []
    qv = embed(queries) if queries else np.zeros((0, 384), np.float32)
    for q, ts in zip(qv, tools):
        if not ts:
            out.append([])
            continue
        names = [t["name"] for t in ts]
        tv = embed([tool_text(t) for t in ts])
        sc = np.round(tv @ q, 5)
        order = sorted(range(len(names)), key=lambda j: (-float(sc[j]), names[j]))
        out.append([names[j] for j in order[:k]])
    return out


def shortlists(ctx, queries: list[str], tools: list[list[dict]], k: int) -> list[list[str]]:
    """`rank`, recorded in the world when `ctx` is given (replay returns the recorded shortlists)."""
    if ctx is None:
        return rank(queries, tools, k)
    from ...substrate import inference
    mid, mh = identity()
    inputs = {"k": int(k), "queries": [hashlib.sha256(q.encode()).hexdigest() for q in queries],
              "tools": [hashlib.sha256("\n".join(tool_text(t) for t in ts).encode()).hexdigest() for ts in tools]}
    return inference.recorded(ctx, "semp.retrieve", mid, mh, inputs, lambda: rank(queries, tools, k))
