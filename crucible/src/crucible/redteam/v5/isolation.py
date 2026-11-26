"""Evaluator isolation probes (Gate W.2): static reachability of evaluator-side code from world mechanics, what a
sandboxed program can import, and what an in-process learner holding an environment object can reach.

`reach(roots)` is the 5.0 import-closure analysis. Compared with the 4.0 check
(`crucible.redteam.independence.StaticReachability`) it follows what Python executes, not only import statements:

* every parent package of a reached module (importing ``a.b.c`` runs ``a/__init__`` and ``a/b/__init__``);
* literal dynamic imports (``importlib.import_module("x")``, ``importlib.import_module(".x", __name__)``,
  ``__import__("x")``);
* imports inside a package's module-level ``__getattr__`` are *lazy exports*: followed only for an importer that asks
  the package for a name that is neither defined at its top level nor a submodule.
"""
from __future__ import annotations

import ast
import os

from ..framework import HELD, INTEGRITY_BREAK, LEAK, Attacker, Evaluator, NotApplicable, Operator, Verdict
from .base import V5Probe, repo, run

PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # .../src/crucible

FORBIDDEN = ("crucible.realworld", "crucible.redteam", "crucible.evaluation.harness", "crucible.research.sealed",
             "crucible.acceptance", "crucible.calibration.datasets", "crucible.calibration.fit_",
             "crucible.calibration.gharchive", "crucible.calibration.criteo", "crucible.rapid.oracles",
             "crucible.rapid.eval_", "crucible.rapid.evaluator_h2", "crucible.experience.leakage",
             "crucible.semantic.capbench", "crucible.innovation.measure", "crucible.social.integrity")
TIER1_FORBIDDEN = ("crucible.rapid.oracles", "crucible.rapid.eval_", "crucible.rapid.evaluator_h2",
                   "crucible.rapid.references", "crucible.rapid.refs_h3", "crucible.acceptance", "crucible.redteam",
                   "crucible.realworld")


def _path(mod: str) -> str | None:
    rel = mod.split(".")[1:]
    for cand in (os.path.join(PKG, *rel) + ".py", os.path.join(PKG, *rel, "__init__.py")):
        if os.path.isfile(cand):
            return cand
    return None


def _literal_import(node: ast.Call, pkg: str) -> str | None:
    f = node.func
    name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
    if name not in ("import_module", "__import__") or not node.args or not isinstance(node.args[0], ast.Constant):
        return None
    target = node.args[0].value
    if not isinstance(target, str):
        return None
    if target.startswith("."):
        return pkg + target if target != "." else pkg
    return target


_CACHE: dict[str, tuple] = {}


def _parse(path: str, modname: str):
    """(eager imports, lazy imports of a package __getattr__, names defined at top level)."""
    if path in _CACHE:
        return _CACHE[path]
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except (OSError, SyntaxError):
        _CACHE[path] = (set(), set(), set(), {})
        return _CACHE[path]
    is_pkg = path.endswith("__init__.py")
    pkg = modname if is_pkg else modname.rsplit(".", 1)[0]
    eager: set[str] = set()
    lazy: set[str] = set()
    top: set[str] = set()
    names_from: dict[str, set[str]] = {}

    def visit(node, in_lazy: bool):
        for ch in ast.iter_child_nodes(node):
            sink = lazy if in_lazy else eager
            if isinstance(ch, ast.Import):
                sink.update(n.name for n in ch.names if n.name.startswith("crucible"))
            elif isinstance(ch, ast.ImportFrom):
                base = ch.module or ""
                if ch.level:
                    parts = pkg.split(".")
                    parts = parts[: len(parts) - (ch.level - 1)]
                    base = ".".join(parts + ([ch.module] if ch.module else []))
                if base.startswith("crucible"):
                    sink.add(base)
                    for n in ch.names:
                        sink.add(f"{base}.{n.name}")
                        names_from.setdefault(base, set()).add(n.name)
            elif isinstance(ch, ast.Call):
                t = _literal_import(ch, pkg)
                if t and t.startswith("crucible"):
                    sink.add(t)
            lazy_here = in_lazy or (is_pkg and isinstance(ch, ast.FunctionDef) and ch.name == "__getattr__"
                                    and node is tree)
            visit(ch, lazy_here)
    visit(tree, False)
    for ch in tree.body:
        if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            top.add(ch.name)
        elif isinstance(ch, (ast.Assign, ast.AnnAssign)):
            for t in (ch.targets if isinstance(ch, ast.Assign) else [ch.target]):
                if isinstance(t, ast.Name):
                    top.add(t.id)
        elif isinstance(ch, (ast.Import, ast.ImportFrom)):
            for n in ch.names:
                top.add((n.asname or n.name).split(".")[0])
    _CACHE[path] = (eager, lazy, top, names_from)
    return _CACHE[path]


def reach(roots) -> dict[str, str | None]:
    """Module -> the module that first imported it (None for roots), following Python's import semantics."""
    seen: dict[str, str | None] = {}
    asked: dict[str, set[str]] = {}         # package -> names importers asked it for
    stack = [(r, None) for r in sorted(roots)]
    lazy_of: dict[str, set[str]] = {}
    while stack:
        mod, parent = stack.pop()
        path = _path(mod)
        if path is None:
            # `from pkg import name` where name is not a module: a name asked of the package
            pkg, _, name = mod.rpartition(".")
            if pkg and _path(pkg):
                asked.setdefault(pkg, set()).add(name)
                p = _path(pkg)
                _, lazy, top, _ = _parse(p, pkg)
                if name not in top and lazy and pkg in seen:
                    for t in lazy:
                        if t not in seen:
                            stack.append((t, pkg))
            continue
        if mod in seen:
            continue
        seen[mod] = parent
        parts = mod.split(".")
        for i in range(2, len(parts)):          # parent packages run first
            pp = ".".join(parts[:i])
            if pp not in seen:
                stack.append((pp, mod))
        eager, lazy, top, _ = _parse(path, mod)
        lazy_of[mod] = lazy
        for imp in sorted(eager):
            if imp not in seen:
                stack.append((imp, mod))
        for name in sorted(asked.get(mod, ())):
            if name not in top and lazy:
                for t in lazy:
                    if t not in seen:
                        stack.append((t, mod))
    return seen


def chain(seen: dict, mod: str) -> list[str]:
    out, cur = [mod], seen.get(mod)
    while cur is not None and len(out) < 30:
        out.append(cur)
        cur = seen.get(cur)
    return list(reversed(out))


def world_roots() -> list[str]:
    import crucible
    crucible.load_all_modules()
    from ...core.kernel import MODULES
    roots = {cls.__module__ for cls in MODULES.values() if cls.__module__.startswith("crucible.")
             and not cls.__module__.startswith("crucible.redteam")}
    roots |= {"crucible.core.kernel", "crucible.core.state", "crucible.sandbox.bridge", "crucible.sandbox.runner",
              "crucible.sandbox.metering", "crucible.sandbox.isolation", "crucible.goals.metrics",
              "crucible.goals.selfgoals"}
    return sorted(roots)


def tier1_roots() -> list[str]:
    d = os.path.join(PKG, "rapid", "engines")
    if not os.path.isdir(d):
        return []
    return sorted("crucible.rapid.engines." + f[:-3] for f in os.listdir(d) if f.endswith(".py") and f != "__init__.py")


class StaticReachabilityV5(V5Probe):
    name = "v5.isolation.world_code_cannot_import_evaluator_side_code"
    attack_class = "evaluator_independence/code_reachability"
    gates = ("W.2",)
    target, owner = "all world modules", "O"
    world = {"v5": "substrate"}
    description = ("Static import closure, with Python's import semantics, from every registered kernel module, the "
                   "kernel, the sandbox runtime and the goal metrics. Checked: none of the evaluator-side packages "
                   "is reachable - the real-world bridge, the red team, sealed sets, the acceptance gates, the "
                   "calibration dataset downloaders and fitting code, Tier 1 oracles and evaluators, the leakage "
                   "battery, capability-benchmark splits, innovation gate measures and engagement integrity.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("reacher", 100)
        self.roots = world_roots()

    def attack(self, a: Attacker) -> None:
        a.note("no in-world action: static import-graph analysis")

    def verdict(self, ev: Evaluator) -> Verdict:
        seen = reach(self.roots)
        hits = {m: chain(seen, m) for m in sorted(seen) if m.startswith(FORBIDDEN)}
        evd = {"roots": len(self.roots), "modules_reachable": len(seen), "forbidden_prefixes": list(FORBIDDEN),
               "forbidden_reached": hits}
        if hits:
            return Verdict(INTEGRITY_BREAK, f"world mechanics can import {sorted(hits)}", evd, severity="high",
                           fix="remove the import chain shown in evidence.forbidden_reached")
        return Verdict(HELD, f"{len(seen)} modules reachable from {len(self.roots)} world roots; no evaluator-side "
                             "package among them", evd)


class Tier1EngineReachability(V5Probe):
    name = "v5.isolation.tier1_engines_cannot_import_oracles"
    attack_class = "evaluator_independence/code_reachability"
    gates = ("W.2",)
    target, owner = "rapid", "H"
    world = {"v5": "substrate"}
    description = ("The same closure from the Tier 1 engines (the transition code of every rapid world). Checked: "
                   "the Tier 1 oracles, evaluators and hidden references are not reachable.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("reacher-t1", 100)
        self.roots = tier1_roots()
        if not self.roots:
            raise NotApplicable("no Tier 1 engines in this build")

    def attack(self, a: Attacker) -> None:
        a.note("no in-world action: static import-graph analysis")

    def verdict(self, ev: Evaluator) -> Verdict:
        seen = reach(self.roots)
        hits = {m: chain(seen, m) for m in sorted(seen) if m.startswith(TIER1_FORBIDDEN)}
        evd = {"roots": self.roots, "modules_reachable": len(seen), "forbidden_reached": hits}
        if hits:
            via_init = all("crucible.rapid" in c[:3] for c in hits.values())
            return Verdict(INTEGRITY_BREAK, f"Tier 1 transition code can import {sorted(hits)}"
                           + (" (through the eager imports of crucible/rapid/__init__.py)" if via_init else ""),
                           evd, severity="low",
                           fix="rapid/__init__.py: import admission/references lazily; engines import only "
                               "crucible.rapid.engines.base and primitives. Exploitability: none from a participant - "
                               "Tier 1 actions are data and programs run in the sandbox, which cannot import crucible")
        return Verdict(HELD, "no Tier 1 oracle or evaluator is reachable from the engines", evd)


IMPORTS = r'''
import os
def tryall(args, caller):
    out = {}
    for m in ("crucible", "crucible.rapid.oracles", "crucible.acceptance.v5", "crucible.redteam", "crucible.realworld",
              "crucible.calibration.datasets"):
        try:
            __import__(m)
            out[m] = "imported"
        except BaseException as e:
            out[m] = type(e).__name__
    for p in args.get("paths", []):
        try:
            open(p).read(10)
            out[p] = "read"
        except BaseException as e:
            out[p] = type(e).__name__
    return out
'''


class SandboxCannotReachEvaluator(V5Probe):
    name = "v5.isolation.sandboxed_code_cannot_import_or_read_evaluator"
    attack_class = "evaluator_independence/sandbox_import"
    gates = ("W.2", "W.1")
    target, owner = "sandbox", "O"
    needs_sandbox = True
    world = {"v5": "commerce"}
    description = ("Agent code run in the sandbox tries to import the simulator, the Tier 1 oracles, the acceptance "
                   "gates, the red team, the real-world bridge and the dataset downloader, and to read the acceptance "
                   "manifest, the oracle source and the sealed holdout directory by absolute path. Checked: all fail.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("importer", 2_000_000)
        root = os.path.dirname(os.path.dirname(PKG))
        self.paths = [os.path.join(root, "docs", "v5", "ACCEPTANCE_MANIFEST.json"),
                      os.path.join(PKG, "rapid", "oracles.py"), os.path.join(PKG, "research", "sealed.py")]

    def attack(self, a: Attacker) -> None:
        rid = repo(a, self.att, "imp", {"i.py": IMPORTS})
        self.out = run(a, self.att, rid, "i.py:tryall", {"paths": self.paths})

    def verdict(self, ev: Evaluator) -> Verdict:
        if not self.out.get("ok"):
            raise NotApplicable(f"probe program failed: {self.out.get('error')}")
        res = self.out["result"]
        got = sorted(k for k, v in res.items() if v in ("imported", "read"))
        evd = {"results": res}
        if got:
            return Verdict(LEAK, f"sandboxed code reached {got}", evd, severity="critical")
        return Verdict(HELD, f"all {len(res)} imports and reads of evaluator-side code failed", evd)


class InProcessLearnerReachesGrader(V5Probe):
    name = "v5.isolation.in_process_learner_reads_hidden_grader"
    attack_class = "evaluator_independence/in_process_learner"
    gates = ("W.2",)
    target, owner = "experience", "L"
    world = {"v5": "substrate"}
    description = ("A learner that is handed the reference Tier 1 environment object (`crucible.experience.make`, "
                   "observation_only) walks the Python object graph reachable from it looking for the scenario's "
                   "hidden parameters and holdout suite (audit O skeptic S.2: rule parameters and the 50-record "
                   "holdout were reachable at env.env._engine.sc). Checked: nothing the evaluator grades against is "
                   "reachable from the learner-facing object.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("learner", 100)
        try:
            from ...experience import make
            self.env = make(1, "pe_rule_induction", 3, reward_mode="observation_only")
        except Exception as e:  # noqa: BLE001
            raise NotApplicable(f"Tier 1 rule-induction environment unavailable: {type(e).__name__}: {e}")

    def attack(self, a: Attacker) -> None:
        import collections
        seen, found = set(), []
        q = collections.deque([(self.env, "env", 0)])
        while q and len(seen) < 20000:
            obj, path, d = q.popleft()
            if id(obj) in seen or d > 7:
                continue
            seen.add(id(obj))
            h = getattr(obj, "h", None) if not isinstance(obj, (dict, list, tuple)) else None
            if isinstance(h, dict) and ({"holdout", "theta", "hidden"} & set(h)):
                found.append({"at": path + ".h", "keys": sorted(h)[:12]})
            if isinstance(obj, dict) and ({"holdout", "hidden"} & set(obj)):
                found.append({"at": path, "keys": sorted(map(str, obj))[:12]})
            try:
                items = list(vars(obj).items()) if hasattr(obj, "__dict__") and not isinstance(obj, type) else []
            except TypeError:
                items = []
            if isinstance(obj, dict):
                items += [(f"[{k!r}]", v) for k, v in list(obj.items())[:60]]
            elif isinstance(obj, (list, tuple)):
                items += [(f"[{i}]", v) for i, v in enumerate(obj[:30])]
            for k, v in items:
                if v is None or isinstance(v, (int, float, str, bool, bytes, type)) or callable(v) and not hasattr(v, "__dict__"):
                    continue
                q.append((v, f"{path}.{k}" if not k.startswith("[") else path + k, d + 1))
        self.found = found[:10]
        self.visited = len(seen)

    def verdict(self, ev: Evaluator) -> Verdict:
        evd = {"objects_visited": self.visited, "hidden_found": self.found}
        try:
            self.env.close()
        except Exception:  # noqa: BLE001
            pass
        if self.found:
            return Verdict(LEAK, f"the grader's hidden answers are reachable from the learner's environment object "
                           f"at {self.found[0]['at']}", evd, severity="high",
                           fix="experience: serve evaluated learners out of process (the HTTP API or a subprocess "
                               "bridge) so only observations cross; keep graders in the evaluator process")
        return Verdict(HELD, f"no hidden grader state among {self.visited} objects reachable from the environment",
                       evd)


def probes() -> list:
    return [StaticReachabilityV5(), Tier1EngineReachability(), SandboxCannotReachEvaluator(),
            InProcessLearnerReachesGrader()]
