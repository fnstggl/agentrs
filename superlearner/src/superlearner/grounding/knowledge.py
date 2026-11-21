"""What the actor knows about the world it is in: identifiers, numbers, documents it has read, and the tools it
may call. Everything here is built from the actor's own observations and action results - never from privileged
simulator state.

* `Knowledge.ents`  - `crucible.integrations.EntityMemory`: identifiers and integers by the JSON key they appeared
                      under (``{"account": "acct_000017"}`` -> key ``account``).
* `Knowledge.docs`  - text-bearing things the actor has read (search results, documents, listings, API
                      descriptions, problem reports, error messages), most recent last.
* aliases           - stable short names for identifiers (``ACCT1``, ``ORG2``), used when a situation is shown to the
                      frozen language model, so that the same situation in two worlds produces the same prompt, and a
                      made-up identifier is detected as an unresolved reference instead of reaching the world.
* `CapabilityIndex` - the tools currently available to the actor, with semantic retrieval over their descriptions.
"""
from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import ensure_crucible
from ..llm.base import LLM, text_key

ensure_crucible()
from crucible.api import render as R  # noqa: E402
from crucible.integrations import EntityMemory, id_kind  # noqa: E402

_ALIAS_NAME = {"o": "ORG", "p": "PERSON", "acct": "ACCT", "svc": "SVC", "board": "BOARD", "lst": "LISTING",
               "inst": "INSTANCE", "doc": "DOC", "repo": "REPO"}
_ALIAS_RE = re.compile(r"\b([A-Z]{2,12})(\d{1,3})\b")
_TEXT_FIELDS = ("description", "doc", "body", "snippet", "text", "title", "summary", "message", "note", "name")
_ID_LIKE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:_./@-]{0,80}$")
_PROSE_KEYS = {"text", "body", "description", "memo", "note", "message", "title", "bio", "doc", "q", "comment",
               "reason", "name", "content", "stdout", "stderr", "error", "balance_fmt", "news"}


def alias_kind(v: str) -> str | None:
    if "/" in v and ":" in v:
        return "repo"
    k = id_kind(v)
    return None if k in ("word", "dotted", "path") else k


class Ents(EntityMemory):
    """`EntityMemory` that also keeps identifiers appearing under prose-like keys (``{"doc": "doc_000002"}``):
    a string that has an identifier's lexical shape is an identifier whatever its key is called."""

    def observe_obj(self, x: Any, key: str = "", depth: int = 0) -> None:
        if depth > 6:
            return
        if isinstance(x, dict):
            for k, v in sorted(x.items(), key=lambda kv: str(kv[0])):
                ks = str(k)
                is_id = bool(_ID_LIKE.match(ks)) and id_kind(ks) != "word"
                if is_id and key:
                    self._put_id(key, ks)
                self.observe_obj(v, key if is_id else ks, depth + 1)
        elif isinstance(x, list):
            for v in x[:40]:
                self.observe_obj(v, key, depth + 1)
        elif isinstance(x, bool) or x is None:
            return
        elif isinstance(x, int):
            if key and abs(x) < 10**9:
                self._put_num(key, int(x))
        elif isinstance(x, str):
            if not _ID_LIKE.match(x) or x.isdigit():
                return
            if key in _PROSE_KEYS and id_kind(x) == "word":
                return
            self._put_id(key or "_", x)


@dataclass
class Doc:
    key: int            # text key (also the embedding key)
    text: str           # normalised, truncated text shown to the language model
    source: str         # tool that returned it
    title: str
    t: int              # simulated time it was read
    raw: Any = None     # the JSON item it came from (identifiers intact)


class Knowledge:
    def __init__(self, actor: str, max_docs: int = 48):
        self.actor = actor
        self.ents = Ents(per_key=8, recent=48)
        self.docs: "OrderedDict[int, Doc]" = OrderedDict()
        self.max_docs = max_docs
        self._alias: dict[str, str] = {}
        self._unalias: dict[str, str] = {}
        self._count: dict[str, int] = {}
        self.alias_of(actor)

    # ------------------------------------------------------------------ aliases
    def alias_of(self, ident: str) -> str:
        a = self._alias.get(ident)
        if a is not None:
            return a
        k = alias_kind(ident)
        if k is None:
            return ident
        name = _ALIAS_NAME.get(k, k.upper()[:10])
        n = self._count.get(name, 0) + 1
        self._count[name] = n
        a = f"{name}{n}"
        self._alias[ident] = a
        self._unalias[a] = ident
        return a

    def aliased(self, x: Any) -> Any:
        """Copy of a JSON value with every known identifier replaced by its alias."""
        if isinstance(x, str):
            if x in self._alias:
                return self._alias[x]
            if _ID_LIKE.match(x) and alias_kind(x) is not None:
                return self.alias_of(x)
            return x
        if isinstance(x, dict):
            return {self.aliased(k) if isinstance(k, str) else k: self.aliased(v) for k, v in x.items()}
        if isinstance(x, list):
            return [self.aliased(v) for v in x]
        return x

    def alias_text(self, s: str) -> str:
        for ident in sorted(self._alias, key=len, reverse=True):
            if ident in s:
                s = s.replace(ident, self._alias[ident])
        return s

    def resolve(self, x: Any, unresolved: list | None = None) -> Any:
        """Inverse of `aliased`. Strings that look like an alias but are unknown are reported in `unresolved`."""
        if isinstance(x, str):
            if x in self._unalias:
                return self._unalias[x]
            m = _ALIAS_RE.fullmatch(x)
            if m and m.group(1) in set(_ALIAS_NAME.values()) | {k.upper()[:10] for k in self._count}:
                if unresolved is not None:
                    unresolved.append(x)
                return x
            if any(a in x for a in self._unalias) and len(x) < 4000:
                for a in sorted(self._unalias, key=len, reverse=True):
                    x = re.sub(rf"\b{re.escape(a)}\b", self._unalias[a], x)
            return x
        if isinstance(x, dict):
            return {self.resolve(k) if isinstance(k, str) else k: self.resolve(v, unresolved) for k, v in x.items()}
        if isinstance(x, list):
            return [self.resolve(v, unresolved) for v in x]
        return x

    # ------------------------------------------------------------------ observation
    def observe(self, obs: dict | None = None, action: dict | None = None, result: dict | None = None,
                now: int = 0) -> None:
        self.ents.observe(obs, action, result)
        if result is not None and result.get("ok"):
            self._collect_docs(result.get("result"), (action or {}).get("tool", result.get("tool", "")), now)
        if obs:
            for e in (obs.get("events") or [])[-20:]:
                pl = e.get("payload")
                if isinstance(pl, dict):
                    self._collect_docs(pl, "event:" + str(e.get("type", "")), now, depth=1)

    def _collect_docs(self, x: Any, source: str, now: int, depth: int = 0) -> None:
        if depth > 4 or x is None:
            return
        if isinstance(x, list):
            for v in x[:24]:
                self._collect_docs(v, source, now, depth + 1)
            return
        if not isinstance(x, dict):
            return
        texts = [(k, v) for k, v in x.items() if k in _TEXT_FIELDS and isinstance(v, str) and len(v) >= 24]
        if texts:
            self.add_doc(x, source, now)
        for k, v in x.items():
            if isinstance(v, (dict, list)):
                self._collect_docs(v, source, now, depth + 1)

    def add_doc(self, item: Any, source: str, now: int = 0, title: str | None = None) -> Doc:
        if isinstance(item, str):
            text, raw = item, item
            ttl = title or item[:60]
        else:
            raw = item
            ttl = title or str(item.get("title") or item.get("name") or item.get("need") or item.get("host") or source)[:80]
            fields = []
            for k in sorted(item, key=lambda k: (k not in ("title", "name", "need", "kind"), k)):
                v = item[k]
                if isinstance(v, (str, int, float, bool)) and not isinstance(v, bool) or isinstance(v, bool):
                    fields.append(f"{k}: {v}")
            text = "; ".join(fields)
        text = f"[{source}] " + self.alias_text(text)[:900]
        key = text_key(text)
        self.docs.pop(key, None)
        d = self.docs[key] = Doc(key, text, source, ttl, now, raw)
        while len(self.docs) > self.max_docs:
            self.docs.popitem(last=False)
        return d

    def known_ids(self, param: str, limit: int = 6) -> list[str]:
        return [v for v, _ in self.ents.ids_for(param, limit)]

    def entity_summary(self, per_key: int = 3, max_keys: int = 14) -> dict:
        """Recently seen identifiers by key (aliased), for prompts."""
        out = {}
        keys = sorted(self.ents.by_key, key=lambda k: -max(self.ents.by_key[k].values(), default=0))
        for k in keys[:max_keys]:
            vals = [v for v in reversed(self.ents.by_key[k]) if alias_kind(v) is not None or k in ("host", "need", "repo")]
            if vals:
                out[k] = [self.alias_of(v) for v in vals[:per_key]]
        return out

    def state_dict(self) -> dict:
        return {"ents": self.ents.state_dict(), "alias": dict(self._alias), "count": dict(self._count),
                "docs": [(d.key, d.text, d.source, d.title, d.t) for d in self.docs.values()]}


# ================================================================================================ capabilities
def _signature(tool: dict) -> str:
    schema = tool.get("inputSchema") or {}
    props = schema.get("properties") or {}
    req = set(schema.get("required") or [])
    params = [f"{n}{'' if n in req else '?'}: {R._type_str(props[n])}" for n in sorted(props, key=lambda n: (n not in req, n))]
    line = f"{tool.get('name')}({', '.join(params)})"
    cost = (tool.get("annotations") or {}).get("costSeconds")
    return line + (f" [{cost}s]" if cost else "")


def tool_doc(tool: dict, chars: int = 400) -> str:
    """Signature + the WHOLE description on one line (the text that is embedded and shown to the language
    model). Crucible's `render_tool` keeps only the first line of a description, which drops, for example, the
    manifest format that ``cloud.deploy`` documents on its following lines."""
    d = " ".join((tool.get("description") or "").split())
    if len(d) > chars:
        d = d[: chars - 3] + "..."
    return _signature(tool) + (f" - {d}" if d else "")


def tool_full_doc(tool: dict) -> str:
    """Signature, whole description and per-argument schema details (enum values, bounds, patterns, descriptions)."""
    lines = [tool_doc(tool, 900)]
    sc = tool.get("inputSchema") or {}
    req = set(sc.get("required") or [])
    for k, p in (sc.get("properties") or {}).items():
        bits = []
        if p.get("description"):
            bits.append(str(p["description"]))
        for kw in ("enum", "pattern", "minimum", "maximum", "maxLength", "minItems", "maxItems"):
            if kw in p:
                bits.append(f"{kw}={json.dumps(p[kw])}")
        if p.get("type") == "object" and p.get("additionalProperties") not in (None, False):
            bits.append(f"values={json.dumps(p['additionalProperties'])[:120]}")
        if p.get("type") == "array" and p.get("items"):
            bits.append(f"items={json.dumps(p['items'])[:200]}")
        lines.append(f"  - {k} ({'required' if k in req else 'optional'}): " + "; ".join(bits))
    return "\n".join(lines)


class CapabilityIndex:
    """The tools the actor can call right now, discovered from the world, with semantic retrieval.

    `update(tools)` is called with `World.available_tools(actor)` whenever it may have changed (a service the actor
    learned about adds tools); it returns the names that are new. `search(query)` ranks tools by cosine similarity
    between the frozen model's embedding of the query and of each tool's description, plus a lexical bonus for
    words shared with the tool's name."""

    def __init__(self, llm: LLM):
        self.llm = llm
        self.tools: dict[str, dict] = {}
        self.order: list[str] = []
        self.keys: dict[str, int] = {}
        self._mat: np.ndarray | None = None
        self._sig: tuple | None = None

    def update(self, tools: list[dict]) -> list[str]:
        sig = (len(tools), tools[0]["name"] if tools else "", tools[-1]["name"] if tools else "")
        if sig == self._sig and len(tools) == len(self.order):
            return []
        new = [t["name"] for t in tools if t["name"] not in self.tools]
        gone = set(self.tools) - {t["name"] for t in tools}
        if not new and not gone:
            self._sig = sig
            return []
        self.tools = {t["name"]: t for t in tools}
        self.order = [t["name"] for t in tools]
        docs = [tool_doc(self.tools[n]) for n in self.order]
        self.keys = {n: text_key(d) for n, d in zip(self.order, docs)}
        self._mat = self.llm.embed(docs)
        self._sig = sig
        return new

    def has(self, name: str) -> bool:
        return name in self.tools

    def vec(self, name: str) -> np.ndarray:
        return self._mat[self.order.index(name)]

    def consumers(self, name: str) -> list[str]:
        """Tools that take an argument this tool also takes (they operate on the same kind of thing: a tool that
        writes to a ``repo`` is related to the tools that deploy, run or sell a ``repo``)."""
        mine = {k for k, p in ((self.tools[name].get("inputSchema") or {}).get("properties") or {}).items()
                if p.get("type") == "string" and p.get("maxLength", 4000) <= 200 and "enum" not in p}
        out = []
        for n in self.order:
            if n != name and mine & set(((self.tools[n].get("inputSchema") or {}).get("properties") or {})):
                out.append(n)
        return out

    def search(self, query: str, k: int = 8, exclude: tuple = (), restrict: list | None = None) -> list[tuple[str, float]]:
        if not self.order:
            return []
        q = self.llm.embed([query])[0]
        mu = self._mat.mean(0)                     # centre on this tool set: scores compare tools with each other
        d = self._mat - mu
        d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
        qc = q - mu
        s = d @ (qc / max(float(np.linalg.norm(qc)), 1e-6))
        if restrict is not None:
            keep = set(restrict)
            s = s + np.array([0.0 if n in keep else -10.0 for n in self.order], np.float32)
        words = set(re.findall(r"[a-z]{3,}", query.lower()))
        for i, n in enumerate(self.order):
            parts = set(re.findall(r"[a-z]{3,}", n.lower()))
            if parts & words:
                s[i] += 0.08 * len(parts & words)
        idx = np.argsort(-s)
        out = [(self.order[i], float(s[i])) for i in idx if self.order[i] not in exclude]
        return out[:k]

    def nearest_names(self, name: str, k: int = 3) -> list[str]:
        """Tools whose names resemble an unknown name (for unknown-tool feedback)."""
        import difflib
        cand = difflib.get_close_matches(name, self.order, n=k, cutoff=0.5)
        host, _, ep = name.partition(".")
        for n in self.order:
            if len(cand) >= k:
                break
            if n not in cand and (n.startswith(host + ".") or (ep and n.endswith("." + ep))):
                cand.append(n)
        return cand[:k]
