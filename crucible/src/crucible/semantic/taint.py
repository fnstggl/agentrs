"""Gate E.5 by data flow: does any value an artifact's or offer's creator declares about their own item reach a
participant's decision, other than through the bytes (the shared space) or as a binding term?

This replaces the name-regex audit. It is a static, field-sensitive taint analysis over the source tree
(`src/crucible`), with world tables and event payloads as the channels between modules:

**Sources.** Every parameter of every kernel endpoint (`@endpoint` handler) is actor-declared. A parameter whose
schema is a number, integer, boolean, an enum or an array of such is a *declared value*; other strings are *declared
text*. Taint labels name the tool and parameter.

**Self-declaration.** A declared value becomes a *self-declared field* of a world table when an applier writes it
into a record that the same event ties to the declaring actor: the record's key, or another field written in the
same record, carries the taint of `ctx.actor`/`ctx.origin` of the handler that emitted the event (e.g. a listing
stored with ``seller = ctx.actor``). Values the actor states about someone else's item (a review's stars, a bid on
another's listing) are not self-declared: they are third-party signals.

**Independent signals.** A handler that refuses when the actor owns the record it acts on (an `if` comparing the
actor with an owner read from a world record, raising) declares third-party values (ratings, reviews, bids on
others' items); its parameters are not self-declarations. A rating endpoint *without* such a check lets creators
rate their own items, and its values count as self-declared.

**Verified declarations.** A declared value that the code compares with a measurement (an expression without
declared taint that calls something, e.g. the measured duration of the media) and refuses on mismatch is verified.

**Binding terms.** A self-declared value that also flows into a money movement (`bank.move`/`transfer`/`pay`/...
amounts) is a term the world enforces (a price is charged exactly), not an unverified claim; it is exempt.

**Sanitisers.** Values computed by the shared space or the encoders from bytes (`crucible.semantic.space.*`,
`encoders.encode*`, `artifacts.put`) carry no creator taint: they are measurements of the content.

**Sinks.** Participant decision inputs: the arguments of `record_decisions(...)` and `experience(...)`, and the
return value of every handler registered with `register_activity(...)`.

**Propagation.** Within a function, flow-insensitive assignment propagation over names, attributes, subscripts and
calls (a call's result carries its arguments' taint, unless it is a sanitiser); records read from a table
(`state.get/scan`, `columns.col`) carry the taint of the fields read from them; calls to functions defined in the
tree use their summaries (fields they read, parameters that reach their return value), iterated to a fixpoint.

A violation is a self-declared, non-binding declared value reaching a sink. Declared *text* reaching a sink outside
the space is reported separately (`text_flows`) but not counted: titles and descriptions are content that readers
perceive. A line can be marked ``# taint-ok: <reason>``; marked lines are reported with the reason, never dropped.

Precision limits (documented, measured by the planted-violation tests): aliasing through containers is
approximated by key-insensitive taint, dynamic dispatch through `getattr` strings is not followed, and records
passed whole to other functions keep their table only when the callee reads fields by constant names.
"""
from __future__ import annotations

import ast
import os
from collections import defaultdict
from dataclasses import dataclass, field

PRAGMA = "# taint-ok:"
MONEY_CALLS = {"move", "transfer", "pay", "pay_invoice", "charge", "refund", "escrow", "settle", "debit", "credit"}
MONEY_TOOLS = ("bank.", "payments.", "escrow.")
SANITISERS = {"item_vector", "item_vectors", "item_chunks", "item_features", "text_vector", "text_vectors", "reps",
              "reps_of_bytes", "vectors_of_bytes", "features_of_bytes", "comprehension", "encode", "encode_many",
              "encode_bytes", "encode_bytes_many", "put_artifact", "ref_of", "interest_prior"}
SANITISER_PUT = {("artifacts", "put"), ("art", "put")}
SINK_CALLS = {"record_decisions", "experience"}
ACTOR = "@actor"
MAX_DEFS = 6               # a call name defined more often than this is not resolved to a summary
SKIP = ("__pycache__",)


@dataclass
class Fn:
    module: str
    qual: str
    node: ast.AST
    params: list[str]
    endpoint: dict | None = None
    applier: str | None = None
    ret: set = field(default_factory=set)            # labels reaching the return value
    param_to_ret: set = field(default_factory=set)   # indices of params whose taint reaches the return
    activity: bool = False
    ret_tables: set = field(default_factory=set)     # tables of the records this function returns
    param_tables: dict = field(default_factory=dict) # param index -> tables of records passed by callers
    nodes: list | None = None


def _const_str(node, consts: dict) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in consts:
        return consts[node.id]
    if isinstance(node, ast.Attribute) and node.attr in consts:
        return consts[node.attr]
    return None


def _call_name(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


def _call_base(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
        return f.value.id
    return ""


def _schema_kind(schema: dict) -> str:
    if not isinstance(schema, dict):
        return "text"
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    if schema.get("enum") is not None or t in ("number", "integer", "boolean"):
        return "value"
    if t == "array":
        return _schema_kind(schema.get("items") or {})
    return "text"


class Analysis:
    def __init__(self, root: str):
        self.root = root
        self.fns: dict[str, Fn] = {}
        self.by_name: dict[str, list[str]] = defaultdict(list)
        self.consts: dict[str, dict] = {}
        self.sources: dict[str, str] = {}              # label -> value|text
        self.event_fields: dict[tuple, set] = defaultdict(set)     # (event type, payload key or '*') -> labels
        self.table_fields: dict[tuple, set] = defaultdict(set)     # (table, field) -> labels
        self.self_declared: dict[tuple, set] = defaultdict(set)    # (table, field) -> labels
        self.money: set = set()
        self.sink_hits: list[dict] = []
        self.lines: dict[str, list[str]] = {}
        self.activity_handlers: set[str] = set()

    # ------------------------------------------------------------------------------------------ parsing
    def load(self) -> None:
        for d, dirs, files in os.walk(self.root):
            dirs[:] = sorted(x for x in dirs if x not in SKIP)
            for fn in sorted(files):
                if not fn.endswith(".py"):
                    continue
                p = os.path.join(d, fn)
                rel = os.path.relpath(p, self.root).replace(os.sep, "/")
                try:
                    text = open(p, encoding="utf-8").read()
                    tree = ast.parse(text)
                except (SyntaxError, UnicodeDecodeError, ValueError):
                    continue
                self.lines[rel] = text.splitlines()
                consts = {}
                for n in tree.body:
                    if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
                        for t in n.targets:
                            if isinstance(t, ast.Name):
                                consts[t.id] = n.value.value
                self.consts[rel] = consts
                self._collect(rel, tree, consts)

    def _collect(self, rel: str, tree, consts) -> None:
        def visit(body, prefix, cls=None):
            for n in body:
                if isinstance(n, ast.ClassDef):
                    for b in n.body:      # class-level string constants (e.g. TABLE = "...")
                        if isinstance(b, ast.Assign) and isinstance(b.value, ast.Constant) and \
                                isinstance(b.value.value, str):
                            for t in b.targets:
                                if isinstance(t, ast.Name):
                                    consts.setdefault(t.id, b.value.value)
                    visit(n.body, prefix + n.name + ".", n)
                elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    q = f"{rel}:{prefix}{n.name}"
                    params = [a.arg for a in n.args.args + n.args.kwonlyargs]
                    f = Fn(rel, q, n, params)
                    for dec in n.decorator_list:
                        if isinstance(dec, ast.Call) and _call_name(dec) == "endpoint" and dec.args:
                            name = _const_str(dec.args[0], consts) or "?"
                            schema = self._literal_schema(dec.args[1]) if len(dec.args) > 1 else {}
                            f.endpoint = {"name": name, "schema": schema, "class": cls.name if cls else ""}
                        if isinstance(dec, ast.Call) and _call_name(dec) == "applier" and dec.args:
                            f.applier = _const_str(dec.args[0], consts)
                    self.fns[q] = f
                    self.by_name[n.name].append(q)
                    visit(n.body, prefix + n.name + ".", cls)
        visit(tree.body, "")

    @staticmethod
    def _literal_schema(node) -> dict:
        """Best-effort reading of an `S.obj({...}, [...])` schema: property name -> {type/enum} from `S.s_*`
        helper names (s_int/s_num/s_bool/s_enum are values; s_str/s_text are text)."""
        out = {"properties": {}}
        if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Dict):
            for k, v in zip(node.args[0].keys, node.args[0].values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    kind = {}
                    if isinstance(v, ast.Call):
                        nm = _call_name(v)
                        if nm in ("s_int", "s_integer", "s_num", "s_number", "s_bool", "s_money", "s_cents"):
                            kind = {"type": "number"}
                        elif nm == "s_enum":
                            kind = {"enum": ["?"]}
                        elif nm == "s_arr" and v.args and isinstance(v.args[0], ast.Call) and \
                                _call_name(v.args[0]) in ("s_int", "s_num", "s_bool", "s_enum"):
                            kind = {"type": "array", "items": {"type": "number"}}
                        else:
                            kind = {"type": "string"}
                    elif isinstance(v, ast.Dict):
                        for kk, vv in zip(v.keys, v.values):
                            if isinstance(kk, ast.Constant) and kk.value == "type" and isinstance(vv, ast.Constant):
                                kind = {"type": vv.value}
                            if isinstance(kk, ast.Constant) and kk.value == "enum":
                                kind = {"enum": ["?"]}
                    out["properties"][k.value] = kind
        return out

    # ------------------------------------------------------------------------------------------ intra-procedural
    def _analyse(self, f: Fn) -> dict:
        """One pass over `f`: local taints, event emissions, table writes, money calls and sinks."""
        consts = self.consts.get(f.module, {})
        taint: dict[str, set] = defaultdict(set)
        rec_table: dict[str, set] = defaultdict(set)     # local name -> tables of the record(s) it holds
        alias: set[str] = set()                 # names bound to ev.payload in an applier
        for i, ts in f.param_tables.items():
            if i < len(f.params):
                rec_table[f.params[i]] |= ts

        def field_read(tables, key) -> set:
            out = set()
            for t in tables:
                out |= self.table_fields.get((t, key), set())
                out.add(("read", t, key))
            return out
        if f.endpoint:
            props = f.endpoint["schema"].get("properties", {})
            for p in f.params:
                if p in ("self", "ctx"):
                    continue
                lab = f"{f.endpoint['class']}.{f.endpoint['name']}:{p}"
                self.sources.setdefault(lab, _schema_kind(props.get(p, {"type": "string"})))
                taint[p].add(lab)
        for i, p in enumerate(f.params):
            taint[p].add(f"#param{i}")
        if f.applier:
            pass
        ret: set = set()

        def expr(e) -> set:
            if e is None:
                return set()
            if isinstance(e, ast.Name):
                return set(taint.get(e.id, ()))
            if isinstance(e, ast.Attribute):
                if isinstance(e.value, ast.Name) and e.value.id == "ctx" and e.attr in ("actor", "origin"):
                    return {ACTOR}
                if isinstance(e.value, ast.Attribute) and e.attr == "payload":
                    return set()
                base = expr(e.value)
                if f.applier and isinstance(e.value, ast.Name) and e.attr == "payload":
                    return {("ev", f.applier, "*")}
                return base
            if isinstance(e, ast.Subscript):
                key = _const_str(e.slice, consts) if not isinstance(e.slice, ast.Slice) else None
                # ev.payload["k"] (or an alias of ev.payload) in an applier
                if f.applier and ((isinstance(e.value, ast.Attribute) and e.value.attr == "payload") or
                                  (isinstance(e.value, ast.Name) and e.value.id in alias)):
                    return set(self.event_fields.get((f.applier, key), set())) | \
                        set(self.event_fields.get((f.applier, "*"), set()))
                if isinstance(e.value, ast.Name) and rec_table.get(e.value.id) and key is not None:
                    return field_read(rec_table[e.value.id], key)
                return expr(e.value) | expr(e.slice)
            if isinstance(e, ast.Call):
                nm = _call_name(e)
                args = list(e.args) + [k.value for k in e.keywords]
                if nm in SANITISERS or (_call_base(e), nm) in SANITISER_PUT:
                    return set()
                if nm == "get" and isinstance(e.func, ast.Attribute):
                    base = e.func.value
                    if f.applier and ((isinstance(base, ast.Attribute) and base.attr == "payload") or
                                      (isinstance(base, ast.Name) and base.id in alias)) and e.args:
                        key = _const_str(e.args[0], consts)
                        return set(self.event_fields.get((f.applier, key), set())) | \
                            set(self.event_fields.get((f.applier, "*"), set()))
                    if isinstance(base, ast.Name) and rec_table.get(base.id) and e.args:
                        key = _const_str(e.args[0], consts)
                        if key is not None:
                            return field_read(rec_table[base.id], key)
                if nm == "col" and len(e.args) >= 3:
                    t, c = _const_str(e.args[1], consts), _const_str(e.args[2], consts)
                    if t and c:
                        return set(self.table_fields.get((t, c), set())) | {("read", t, c)}
                out = set()
                for a in args:
                    out |= expr(a)
                if isinstance(e.func, ast.Attribute):
                    out |= expr(e.func.value)
                cands = self.by_name.get(nm, ())
                for q in (cands if len(cands) <= MAX_DEFS else ()):     # ambiguous names: arguments only
                    g = self.fns[q]
                    out |= {x for x in g.ret if not (isinstance(x, str) and x.startswith("#param"))}
                    for i in g.param_to_ret:
                        if i < len(e.args):
                            out |= expr(e.args[i])
                return out
            if isinstance(e, (ast.BinOp,)):
                return expr(e.left) | expr(e.right)
            if isinstance(e, ast.UnaryOp):
                return expr(e.operand)
            if isinstance(e, ast.BoolOp):
                return set().union(*[expr(v) for v in e.values])
            if isinstance(e, ast.Compare):
                return expr(e.left).union(*[expr(c) for c in e.comparators])
            if isinstance(e, ast.IfExp):
                return expr(e.body) | expr(e.orelse) | expr(e.test)
            if isinstance(e, (ast.List, ast.Tuple, ast.Set)):
                return set().union(*[expr(x) for x in e.elts]) if e.elts else set()
            if isinstance(e, ast.Dict):
                return set().union(*[expr(v) for v in e.values]) if e.values else set()
            if isinstance(e, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
                out = expr(e.elt)
                for g in e.generators:
                    out |= expr(g.iter)
                return out
            if isinstance(e, ast.DictComp):
                out = expr(e.key) | expr(e.value)
                for g in e.generators:
                    out |= expr(g.iter)
                return out
            if isinstance(e, ast.JoinedStr):
                return set().union(*[expr(v) for v in e.values]) if e.values else set()
            if isinstance(e, ast.FormattedValue):
                return expr(e.value)
            if isinstance(e, ast.Starred):
                return expr(e.value)
            return set()

        def table_of_read(e) -> set:
            """Tables of the record(s) an expression evaluates to: state.get/scan/..., a record-holding name, a call
            of a function that returns records, an element of a record list."""
            if isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute) and \
                    _call_name(e) in ("get", "scan", "find", "query", "rows", "values", "items", "get_many"):
                b = e.func.value
                is_state = (isinstance(b, ast.Attribute) and b.attr == "state") or \
                    (isinstance(b, ast.Name) and b.id in ("st", "state"))
                if is_state and e.args:
                    t = _const_str(e.args[0], consts)
                    return {t} if t else set()
            if isinstance(e, ast.Name):
                return set(rec_table.get(e.id, ()))
            if isinstance(e, ast.Subscript) and isinstance(e.value, ast.Name) and \
                    not isinstance(e.slice, ast.Constant):
                return set(rec_table.get(e.value.id, ()))
            if isinstance(e, ast.Call):
                cands = self.by_name.get(_call_name(e), ())
                if 0 < len(cands) <= MAX_DEFS:
                    out = set()
                    for q in cands:
                        out |= self.fns[q].ret_tables
                    return out
            return set()

        def assign(target, val: set, src=None):
            if isinstance(target, ast.Name):
                if f.applier and isinstance(src, ast.Attribute) and src.attr == "payload":
                    alias.add(target.id)
                    return
                taint[target.id] |= val
                if src is not None:
                    rec_table[target.id] |= table_of_read(src)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for el in target.elts:
                    assign(el, val, src)
            elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                taint[target.value.id] |= val
            elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                taint[target.value.id] |= val

        if f.nodes is None:
            f.nodes = list(ast.walk(f.node))
        nodes = f.nodes
        for _ in range(3):                    # flow-insensitive: iterate the assignments to a local fixpoint
            for n in nodes:
                if isinstance(n, ast.Assign):
                    v = expr(n.value)
                    for t in n.targets:
                        assign(t, v, n.value)
                elif isinstance(n, (ast.AugAssign, ast.AnnAssign)) and getattr(n, "value", None) is not None:
                    assign(n.target, expr(n.value))
                elif isinstance(n, (ast.For, ast.comprehension)):
                    it = n.iter
                    t = table_of_read(it)
                    assign(n.target, expr(it))
                    if t and isinstance(n.target, ast.Tuple) and len(n.target.elts) == 2 and \
                            isinstance(n.target.elts[1], ast.Name):
                        rec_table[n.target.elts[1].id] |= t
                    elif t and isinstance(n.target, ast.Name):
                        rec_table[n.target.id] |= t
                elif isinstance(n, ast.withitem) and n.optional_vars is not None:
                    assign(n.optional_vars, expr(n.context_expr))
        if f.endpoint and self._independence_check(f.node, expr):
            for p in f.params:
                lab = f"{f.endpoint['class']}.{f.endpoint['name']}:{p}"
                if self.sources.get(lab) == "value":
                    self.sources[lab] = "third_party"
        emits, writes, money, sinks = [], [], [], []
        ret_tables: set = set()
        param_tables: list = []
        for n in nodes:
            if isinstance(n, ast.Return) and n.value is not None:
                ret |= expr(n.value)
                ret_tables |= table_of_read(n.value)
            if isinstance(n, ast.Call):
                cands = self.by_name.get(_call_name(n), ())
                if 0 < len(cands) <= MAX_DEFS:
                    for i, a in enumerate(n.args):
                        ts = table_of_read(a)
                        if ts:
                            param_tables.append((cands, i, isinstance(n.func, ast.Attribute), ts))
            if not isinstance(n, ast.Call):
                continue
            nm = _call_name(n)
            if nm == "emit":
                a = list(n.args)
                if a and isinstance(a[0], ast.Name) and a[0].id == "ctx":     # bus.emit(ctx, type, payload)
                    a = a[1:]
                et = _const_str(a[0], consts) if a else None
                if et and len(a) > 1:
                    pl = a[1]
                    if isinstance(pl, ast.Dict):
                        emits.append((et, {(_const_str(k, consts) if k is not None else "*"): expr(v)
                                           for k, v in zip(pl.keys, pl.values)}))
                    else:
                        emits.append((et, {"*": expr(pl)}))
            elif nm in ("put", "insert", "update", "upsert", "set") and isinstance(n.func, ast.Attribute) and \
                    len(n.args) >= 3:
                t = _const_str(n.args[0], consts)
                if t:
                    keyt = expr(n.args[1])
                    rec = n.args[2]
                    if isinstance(rec, ast.Dict):
                        flds = {(_const_str(k, consts) if k is not None else "*"): expr(v)
                                for k, v in zip(rec.keys, rec.values)}
                    elif isinstance(rec, ast.BinOp) and isinstance(rec.right, ast.Dict):
                        flds = {"*": expr(rec.left)}
                        flds.update({(_const_str(k, consts) or "*"): expr(v)
                                     for k, v in zip(rec.right.keys, rec.right.values)})
                    else:
                        flds = {"*": expr(rec)}
                    writes.append((t, keyt, flds))
            elif nm == "batch" and len(n.args) >= 5:
                t = _const_str(n.args[1], consts)
                vals = n.args[4]
                if t and isinstance(vals, ast.Dict):
                    writes.append((t, set(), {(_const_str(k, consts) or "*"): expr(v)
                                              for k, v in zip(vals.keys, vals.values)}))
            if nm in MONEY_CALLS or (nm == "call" and n.args and (_const_str(n.args[0], consts) or "").startswith(
                    MONEY_TOOLS)):
                for a in list(n.args) + [k.value for k in n.keywords]:
                    money.append(expr(a))
            if nm in SINK_CALLS:
                sinks.append((f.module, n.lineno, nm, set().union(*[expr(a) for a in list(n.args[1:]) +
                                                                    [k.value for k in n.keywords]] or [set()])))
            if nm == "register_activity" and len(n.args) >= 3:
                h = n.args[2]
                hn = h.attr if isinstance(h, ast.Attribute) else h.id if isinstance(h, ast.Name) else None
                if hn:
                    self.activity_handlers.add(hn)
        verified = self._verified(f.node, expr, set(f.params))
        for n in nodes:                       # effects of called functions, their parameters bound to our arguments
            if not isinstance(n, ast.Call):
                continue
            cands = self.by_name.get(_call_name(n), ())
            if not cands or len(cands) > MAX_DEFS:
                continue
            for q in cands:
                g, eff = self.fns[q], self.prev.get(q)
                if eff is None or g is f:
                    continue
                off = 1 if g.params and g.params[0] in ("self", "cls") and isinstance(n.func, ast.Attribute) else 0
                argt = {i + off: expr(a) for i, a in enumerate(n.args)}
                for kw in n.keywords:
                    if kw.arg in g.params:
                        argt[g.params.index(kw.arg)] = expr(kw.value)

                def sub(labs, argt=argt):
                    out = set()
                    for x in labs:
                        if isinstance(x, str) and x.startswith("#param"):
                            out |= argt.get(int(x[6:]), set())
                        else:
                            out.add(x)
                    return out
                emits += [(et, {k: sub(v)}) for (et, k), v in eff["emits"].items()]
                writes += [(t, sub(w["key"]), {k: sub(v) for k, v in w["fields"].items()})
                           for t, w in eff["writes"].items()]
                money.append(sub(eff["money"]))
                sinks += [(m, ln, nm2, sub(x)) for (m, ln, nm2), x in eff["sinks"].items()]
                verified |= sub(eff["verified"])
        # merged (bounded) effect summaries
        em: dict = defaultdict(set)
        for et, flds in emits:
            for k, v in flds.items():
                em[(et, k)] |= v
        wr: dict = {}
        for t, kt, flds in writes:
            w = wr.setdefault(t, {"key": set(), "fields": defaultdict(set), "owner": False})
            w["key"] |= kt
            if ACTOR in kt or any(ACTOR in v for v in flds.values()):
                w["owner"] = True
            for k, v in flds.items():
                w["fields"][k] |= v
        sk: dict = defaultdict(set)
        for m, ln, nm2, x in sinks:
            sk[(m, ln, nm2)] |= x
        return {"ret": ret, "emits": dict(em), "writes": wr, "money": set().union(*money) if money else set(),
                "sinks": dict(sk), "verified": verified, "ret_tables": ret_tables, "param_tables": param_tables}

    def _verified(self, node, expr, params=()) -> set:
        """Labels of declared values the function checks against a measurement and refuses on mismatch: an `if`
        comparing (for equality, or by an absolute difference) a declared-tainted expression with a non-constant
        expression that carries no declared value (a measurement of the artifact or the world), whose body raises."""
        out = set()
        for n in ast.walk(node):
            if not isinstance(n, ast.If):
                continue
            raises = any(isinstance(b, ast.Raise) for b in ast.walk(n))
            if not raises:
                continue
            for c in ast.walk(n.test):
                if not isinstance(c, ast.Compare):
                    continue
                sides = [c.left] + list(c.comparators)
                if len(sides) != 2:
                    continue
                inner = []
                for sd in sides:          # abs(a - b) > tol: look inside the call
                    if isinstance(sd, ast.Call) and _call_name(sd) == "abs" and sd.args and \
                            isinstance(sd.args[0], ast.BinOp) and isinstance(sd.args[0].op, ast.Sub):
                        inner += [sd.args[0].left, sd.args[0].right]
                if not inner and not isinstance(c.ops[0], (ast.NotEq, ast.Eq)):
                    continue              # bounds (x > limit) constrain a value, they do not verify a claim
                cand = inner or sides
                if len(cand) != 2:
                    continue
                ta, tb = expr(cand[0]), expr(cand[1])

                def declared(t):
                    return {x for x in t if isinstance(x, str) and (x.startswith("#param") or ":" in x)}

                def declared_values(t):    # a measurement may derive from content (refs, text), not from values
                    return {x for x in t if isinstance(x, str) and self.sources.get(x) in ("value", "verified")}
                for decl_side, other, onode, dnode in ((ta, tb, cand[1], cand[0]), (tb, ta, cand[0], cand[1])):
                    if not (isinstance(dnode, ast.Name) and dnode.id in params):
                        continue          # only a parameter itself is verified, not an expression involving it
                    if declared(decl_side) and not declared_values(other) and \
                            not all(isinstance(z, (ast.Constant, ast.UnaryOp, ast.operator, ast.unaryop))
                                    for z in ast.walk(onode)):
                        out |= declared(decl_side) - declared(other)
        return out

    @staticmethod
    def _independence_check(node, expr) -> bool:
        """True if the handler refuses when the actor is the owner of what it acts on: an `if` comparing an
        actor-tainted value with a value read from a world record, whose body raises (or calls ctx.fail)."""
        for n in ast.walk(node):
            if not isinstance(n, ast.If) or not isinstance(n.test, ast.Compare):
                continue
            sides = [n.test.left] + list(n.test.comparators)
            t = [expr(x) for x in sides]
            has_actor = any(ACTOR in x for x in t)
            has_read = any(any(isinstance(y, tuple) and y[0] == "read" for y in x) for x in t)
            raises = any(isinstance(b, ast.Raise) or (isinstance(b, ast.Expr) and isinstance(b.value, ast.Call) and
                                                       _call_name(b.value) == "fail") for b in ast.walk(n))
            if has_actor and has_read and raises:
                return True
        return False

    # ------------------------------------------------------------------------------------------ fixpoint
    def run(self, rounds: int = 6) -> None:
        self.load()
        results: dict = {}
        self.prev = {}
        for _ in range(rounds):
            changed = False
            self.prev = dict(results)
            for q, f in self.fns.items():
                r = self._analyse(f)
                if q not in results or _size(r) != _size(results[q]):
                    changed = True
                results[q] = r
                for x in r["verified"]:
                    if isinstance(x, str) and self.sources.get(x) == "value":
                        self.sources[x] = "verified"
                        changed = True
                if r["ret_tables"] - f.ret_tables:
                    f.ret_tables |= r["ret_tables"]
                    changed = True
                for cands, i, method, ts in r["param_tables"]:
                    for qq in cands:
                        g = self.fns[qq]
                        j = i + (1 if method and g.params and g.params[0] in ("self", "cls") else 0)
                        if ts - g.param_tables.get(j, set()):
                            g.param_tables.setdefault(j, set()).update(ts)
                            changed = True
                newret = {x for x in r["ret"] if not (isinstance(x, str) and x.startswith("#param"))}
                p2r = {int(x[6:]) for x in r["ret"] if isinstance(x, str) and x.startswith("#param")}
                if newret - f.ret or p2r - f.param_to_ret:
                    f.ret |= newret
                    f.param_to_ret |= p2r
                    changed = True
                for (et, k), labs in r["emits"].items():
                    labs = {x for x in labs if not (isinstance(x, str) and x.startswith("#param"))}
                    if labs - self.event_fields[(et, k)]:
                        self.event_fields[(et, k)] |= labs
                        changed = True
                for t, w in r["writes"].items():
                    owner = w["owner"]
                    for k, labs in w["fields"].items():
                        labs = {x for x in labs if not (isinstance(x, str) and x.startswith("#param"))}
                        if labs - self.table_fields[(t, k)]:
                            self.table_fields[(t, k)] |= labs
                            changed = True
                        if owner:
                            decl = {x for x in labs if isinstance(x, str) and self.sources.get(x) == "value"}
                            if decl - self.self_declared[(t, k)]:
                                self.self_declared[(t, k)] |= decl
                                changed = True
                for x in r["money"]:
                    if isinstance(x, str) and x in self.sources and x not in self.money:
                        self.money.add(x)
                        changed = True
                    if isinstance(x, tuple) and x[0] == "read" and x not in self.money:
                        self.money.add(x)
                        changed = True
            if not changed:
                break
        self.results = results

    # ------------------------------------------------------------------------------------------ report
    def _declared(self, labs: set) -> tuple[set, set]:
        """(self-declared non-binding value labels, declared text labels) in a label set."""
        vals, texts = set(), set()
        for x in labs:
            if isinstance(x, tuple) and x[0] == "read":
                t, c = x[1], x[2]
                if x in self.money:
                    continue
                for lab in self.self_declared.get((t, c), ()):
                    if lab not in self.money:
                        vals.add(f"{t}.{c} <- {lab}")
            elif isinstance(x, str) and x in self.sources and self.sources[x] == "text":
                texts.add(x)
        return vals, texts

    def report(self) -> dict:
        violations, text_flows, marked = [], [], []
        for q, r in self.results.items():
            f = self.fns[q]
            checks = [(m, ln, nm, labs) for (m, ln, nm), labs in r["sinks"].items()]
            if f.node.name in self.activity_handlers:
                checks.append((f.module, f.node.lineno, "activity handler return", r["ret"]))
            for mod, ln, nm, labs in checks:
                vals, texts = self._declared(labs)
                src = self.lines.get(mod, [])
                line = src[ln - 1] if 0 < ln <= len(src) else ""
                if PRAGMA in line and (vals or texts):
                    marked.append({"file": mod, "line": ln, "reason": line.split(PRAGMA, 1)[1].strip()[:200]})
                    continue
                for v in sorted(vals):
                    violations.append({"file": mod, "function": q.split(":", 1)[1], "line": ln, "sink": nm,
                                       "flow": v})
                for t in sorted(texts):
                    text_flows.append({"file": mod, "function": q.split(":", 1)[1], "line": ln, "sink": nm,
                                       "source": t})
        sd = {f"{t}.{c}": sorted(v) for (t, c), v in sorted(self.self_declared.items(), key=lambda kv: str(kv[0])) if v}
        binding = sorted(x if isinstance(x, str) else f"{x[1]}.{x[2]}" for x in self.money)
        return {"functions": len(self.fns), "endpoint_parameters": len(self.sources),
                "declared_value_parameters": sum(1 for k in self.sources.values() if k == "value"),
                "self_declared_fields": sd, "binding_terms": binding[:400],
                "activity_handlers": sorted(self.activity_handlers),
                "sinks_checked": len({k for r in self.results.values() for k in r["sinks"]}) + len(self.activity_handlers),
                "verified_parameters": sorted(k for k, v in self.sources.items() if v == "verified"),
                "third_party_parameters": sorted(k for k, v in self.sources.items() if v == "third_party"),
                "violations": violations, "text_flows": text_flows, "marked": marked,
                "creator_assigned_quality_fields_read": len(violations)}


def _size(r: dict) -> tuple:
    return (len(r["emits"]), sum(len(v) for v in r["emits"].values()), len(r["writes"]),
            sum(len(v) for w in r["writes"].values() for v in w["fields"].values()), len(r["money"]),
            sum(len(v) for v in r["sinks"].values()), len(r["verified"]), len(r["ret_tables"]))


def audit(root: str | None = None) -> dict:
    """Run the analysis over `src/crucible` (or `root`)."""
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    a = Analysis(root)
    a.run()
    return a.report()
