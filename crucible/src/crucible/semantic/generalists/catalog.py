"""Tool retrieval over the universal tool interface (MASTER_SPEC 13.4 "retrieval of relevant state").

A participant sees every tool `World.available_tools(principal)` lists for it (the interface learners use through
`crucible.experience`), several hundred tools of some sixty services in a civilisation. Its prompt cannot hold them
all, so each decision step narrows them in two levels, the way a person browses a service directory:

1. `rank_hosts`: the services (tool hosts) most relevant to the step the participant says it intends to take next -
   a service scores by its best tool's sentence similarity to the intention (pretrained all-MiniLM-L6-v2, the text
   encoder of `crucible.semantic.encoders`, over each tool's name in words and the first sentence of its
   description), then by similarity to the participant's goal and by lexical overlap (Okapi BM25); the model chooses
   one service from its one-line description (`world.hosts`) and tool names;
2. `rank_tools`: the chosen service's tools in order of similarity to the intention; the model chooses the tool.

Nothing maps goals or intentions to tools; retrieval only orders what is shown.

Tool vectors are computed once per process for each distinct tool text and cached; queries are embedded per call.
The ranked shortlists of a step are recorded in the world (`crucible.substrate.inference.recorded`, kind
``semp.retrieve``), so replay and regeneration on another device return exactly the recorded shortlists.
"""
from __future__ import annotations

import hashlib
import math
import re
import threading

import numpy as np

# hosts whose tools are world meta-tools (descriptions of the API itself) or operator-only controls of other
# participant classes: never useful as a participant's action, kept out of shortlists
EXCLUDED_HOSTS = ("admin", "world", "pop_adaptive", "pop_strategic", "space")
_LOCK = threading.RLock()
_VEC: dict[str, np.ndarray] = {}
_IDENT: dict[str, str] = {}


def tool_text(t: dict) -> str:
    """What a tool does, for retrieval: its name in words and the first sentence of its description."""
    name = t.get("name", "")
    words = name.replace(".", " ").replace("_", " ")
    desc = " ".join(str(t.get("description") or "").split())
    first = re.split(r"(?<=[.!?])\s", desc, maxsplit=1)[0]
    return f"{words}: {first}"[:300]


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def lexical(query: str, ts: list[dict]) -> np.ndarray:
    """Okapi BM25 of `query` against the tools' name words and whole descriptions."""
    docs = [_tokens(f"{t['name'].replace('.', ' ').replace('_', ' ')} {t.get('description', '')}") for t in ts]
    n = len(docs)
    avg = sum(len(d) for d in docs) / max(1, n)
    df: dict[str, int] = {}
    for d in docs:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    q = set(_tokens(query))
    out = np.zeros(n, np.float64)
    for j, d in enumerate(docs):
        norm = 1.2 * (0.25 + 0.75 * len(d) / max(1e-9, avg))
        for w in q:
            f = d.count(w)
            if f:
                out[j] += math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) * f * 2.2 / (f + norm)
    return out


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


def _order(names: list[str], sc: np.ndarray) -> list[str]:
    sc = np.round(np.asarray(sc, np.float64), 5)
    return [names[i] for i in sorted(range(len(names)), key=lambda i: (-float(sc[i]), names[i]))]


_STOP = frozenset("next i will my me the a an to of for and in on with about at it this that is be from by or as "
                  "now then so".split())


def strip(query: str) -> str:
    """The content of an intention without its 'Next I will' frame."""
    q = re.sub(r"^\s*next,?\s+i\s+(will|'ll|am going to)\s+", "", query.strip(), flags=re.I)
    return q or query


def rank(queries: list[str], tools: list[list[dict]], k: int, goals: list[str] | None = None) -> list[list[str]]:
    """Flat ranking (used by the held-out benchmark's flat baseline): `k` tools by sentence similarity to the
    query, two by similarity to the goal, two by lexical overlap."""
    goals = list(goals) if goals is not None else [""] * len(queries)
    qs = [strip(q) for q in queries]
    qv = embed(qs) if qs else np.zeros((0, 384), np.float32)
    gv = embed([g or q for g, q in zip(goals, qs)]) if qs else qv
    out = []
    for j, ts in enumerate(tools):
        if not ts:
            out.append([])
            continue
        names = [t["name"] for t in ts]
        tv = embed([tool_text(t) for t in ts])
        by_q = _order(names, tv @ qv[j])
        by_g = _order(names, tv @ gv[j]) if goals[j] else []
        lex = lexical(qs[j], ts)
        by_lex = [n for n in _order(names, lex) if lex[names.index(n)] > 0][:2]
        out.append(list(dict.fromkeys(by_q[: max(1, k - 4)] + by_g[:2] + by_lex + by_q))[:k])
    return out


def host_of(name: str) -> str:
    return name.split(".", 1)[0]


def rank_hosts(queries: list[str], tools: list[list[dict]], k: int, goals: list[str] | None = None) -> list[list[str]]:
    """For each query, `k` services: by their best tool's similarity to the query (k - 3), to the goal (2), and by
    lexical overlap with the query (1), then by query similarity again (no duplicates)."""
    goals = list(goals) if goals is not None else [""] * len(queries)
    qs = [strip(q) for q in queries]
    qv = embed(qs) if qs else np.zeros((0, 384), np.float32)
    gv = embed([g or q for g, q in zip(goals, qs)]) if qs else qv
    out = []
    for j, ts in enumerate(tools):
        if not ts:
            out.append([])
            continue
        hosts = sorted({host_of(t["name"]) for t in ts})
        hi = {h: i for i, h in enumerate(hosts)}
        tv = embed([tool_text(t) for t in ts])
        owner = np.array([hi[host_of(t["name"])] for t in ts])

        def best(sc):
            b = np.full(len(hosts), -9.0)
            np.maximum.at(b, owner, sc)
            return b
        by_q = _order(hosts, best(tv @ qv[j]))
        by_g = _order(hosts, best(tv @ gv[j])) if goals[j] else []
        lex = best(lexical(qs[j], ts))
        by_lex = [h for h in _order(hosts, lex) if lex[hi[h]] > 0][:1]
        out.append(list(dict.fromkeys(by_q[: max(1, k - 3)] + by_g[:2] + by_lex + by_q))[:k])
    return out


def rank_tools(queries: list[str], tools: list[list[dict]], k: int) -> list[list[str]]:
    """For each query, its tools (of one service) in order of similarity to the query, at most `k`."""
    qs = [strip(q) for q in queries]
    qv = embed(qs) if qs else np.zeros((0, 384), np.float32)
    out = []
    for j, ts in enumerate(tools):
        names = [t["name"] for t in ts]
        if not ts:
            out.append([])
            continue
        tv = embed([tool_text(t) for t in ts])
        sc = tv @ qv[j] + 0.02 * lexical(qs[j], ts)
        out.append(_order(names, sc)[:k])
    return out


def host_line(host: str, summary: str, tools: list[dict], n_names: int = 30) -> str:
    names = [t["name"].split(".", 1)[1] for t in tools if host_of(t["name"]) == host]
    s = re.sub(r"\(kernel module `[^`]*`[^)]*\)", "", summary or "").strip()
    s = re.sub(r"\s+", " ", s).replace(" :", ":").strip(" .;:")
    return f"- {host}: {clip_words(s, 90)}. Tools: {', '.join(names[:n_names])}{', ...' if len(names) > n_names else ''}"


def clip_words(s: str, n: int) -> str:
    return s if len(s) <= n else s[: max(0, n - 3)].rsplit(" ", 1)[0] + "..."


def recorded(ctx, kind: str, inputs: dict, fn):
    """`fn()` recorded in the world under the retrieval encoder's identity (replay returns the record)."""
    if ctx is None:
        return fn()
    from ...substrate import inference
    mid, mh = identity()
    return inference.recorded(ctx, kind, mid, mh, inputs, fn)


def digest(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def shortlists(ctx, queries: list[str], tools: list[list[dict]], k: int,
               goals: list[str] | None = None) -> list[list[str]]:
    """`rank`, recorded in the world when `ctx` is given (replay returns the recorded shortlists)."""
    inputs = {"k": int(k), "queries": [digest(q) for q in queries], "goals": [digest(g) for g in (goals or [])],
              "tools": [digest("\n".join(tool_text(t) for t in ts)) for ts in tools]}
    return recorded(ctx, "semp.retrieve", inputs, lambda: rank(queries, tools, k, goals))


def host_shortlists(ctx, queries, tools, k, goals=None) -> list[list[str]]:
    inputs = {"k": int(k), "queries": [digest(q) for q in queries], "goals": [digest(g) for g in (goals or [])],
              "tools": [digest("\n".join(tool_text(t) for t in ts)) for ts in tools]}
    return recorded(ctx, "semp.retrieve_hosts", inputs, lambda: rank_hosts(queries, tools, k, goals))


def tool_shortlists(ctx, queries, tools, k) -> list[list[str]]:
    inputs = {"k": int(k), "queries": [digest(q) for q in queries],
              "tools": [digest("\n".join(tool_text(t) for t in ts)) for ts in tools]}
    return recorded(ctx, "semp.retrieve_tools", inputs, lambda: rank_tools(queries, tools, k))
