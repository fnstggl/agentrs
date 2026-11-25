"""SP-11.6: a documented path for future learners (CAM included) to connect without environment changes, proven by
an external adapter that uses only the public interface.

The adapter is `examples/experience_external_adapter.py` (outside the `crucible` package). Measured:
* its imports: only the standard library and public names of `crucible.experience` (`__all__`);
* every source file of the environment (`src/crucible/**`) hashed before and after running it;
* the identity of every public attribute of `crucible.experience` and of the adapter classes' methods, before and
  after (no monkeypatching);
* that it really ran: actions in a Tier 1 world and a kernel world, events collected, experience stored, the external
  learner's parameters updated; and that docs/v5/LEARNER_GUIDE.md documents the path."""
import ast
import hashlib
import importlib.util
import os
import sys

import pytest

import crucible.experience as X
from crucible.acceptance.v5.evidence import record
from crucible.experience import ExperienceStore

pytestmark = pytest.mark.timeout(900)
CMD = "PYTHONPATH=src python -m pytest tests/v5/test_gate_SP_11_6.py"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ADAPTER = os.path.join(ROOT, "examples", "experience_external_adapter.py")
GUIDE = os.path.join(ROOT, "docs", "v5", "LEARNER_GUIDE.md")
STDLIB = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}


def _tree_hash() -> dict[str, str]:
    out = {}
    base = os.path.join(ROOT, "src", "crucible")
    for d, _, files in os.walk(base):
        for f in files:
            if f.endswith((".py", ".json", ".yaml", ".yml", ".html")):
                p = os.path.join(d, f)
                with open(p, "rb") as fh:
                    out[os.path.relpath(p, base)] = hashlib.sha256(fh.read()).hexdigest()
    return out


def _import_violations(path: str) -> list[str]:
    tree = ast.parse(open(path).read())
    bad = []
    public = set(X.__all__)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                if top == "crucible" or top not in STDLIB:
                    bad.append(f"import {a.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            top = mod.split(".")[0]
            if top == "crucible":
                if mod != "crucible.experience":
                    bad.append(f"from {mod} import ...")
                else:
                    bad += [f"crucible.experience.{a.name} (not public)" for a in node.names if a.name not in public]
            elif top not in STDLIB:
                bad.append(f"from {mod} import ...")
    return bad


def _public_ids() -> dict[str, int]:
    out = {}
    for name in X.__all__:
        obj = getattr(X, name)
        out[name] = id(obj)
        for attr in ("observe", "execute", "observe_since", "tools", "advance", "reward", "verify", "append",
                     "iterate", "act", "update"):
            fn = getattr(obj, attr, None)
            if fn is not None:
                out[f"{name}.{attr}"] = id(getattr(fn, "__code__", fn))
    return out


def test_SP_11_6_external_adapter_without_environment_changes(tmp_path):
    violations = _import_violations(ADAPTER)
    before_tree, before_ids = _tree_hash(), _public_ids()
    spec = importlib.util.spec_from_file_location("experience_external_adapter", ADAPTER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                     # an ordinary import of an external module
    spec.loader.exec_module(mod)
    specs = [{"tier": 1, "name": "mech:scm_additive", "kw": {"validate": False}},
             {"tier": 1, "name": "mech:abm_duopoly", "kw": {"validate": False}},
             {"tier": 0, "name": "tiny", "kw": {"agents": ["external"]}}]
    stats = mod.run(specs, [0, 1, 2], 60, store_dir=str(tmp_path / "ext"))
    after_tree, after_ids = _tree_hash(), _public_ids()
    changed_files = sorted(k for k in set(before_tree) | set(after_tree) if before_tree.get(k) != after_tree.get(k))
    patched = sorted(k for k in before_ids if after_ids.get(k) != before_ids[k])
    stored = sum(1 for _ in ExperienceStore(str(tmp_path / "ext")).iterate())
    guide = open(GUIDE).read() if os.path.exists(GUIDE) else ""
    sections = ["What the learner observes", "Acting", "Reward modes", "Worlds", "Experience, checkpoints",
                "Existing learner contracts", "How a future CAM version connects", "experience_external_adapter.py"]
    guide_ok = bool(guide) and all(s in guide for s in sections)
    metrics = {"external_adapter_env_changes": len(violations) + len(changed_files) + len(patched),
               "connection_guide_present": bool(guide_ok), "adapter_import_violations": len(violations),
               "environment_files_changed": len(changed_files), "public_attributes_patched": len(patched),
               "adapter_actions": stats["actions"], "adapter_events": stats["events"], "adapter_records": stored,
               "external_learner_updates": stats["updates"], "worlds": len(specs)}
    record("SP-11.6", metrics, command=CMD, artifacts=[os.path.relpath(GUIDE, ROOT), os.path.relpath(ADAPTER, ROOT)],
           notes=f"adapter: examples/experience_external_adapter.py (CAM Unified EnvAdapter shape on crucible.experience); "
                 f"violations {violations}; changed files {changed_files[:5]}; patched {patched[:5]}")
    assert metrics["external_adapter_env_changes"] == 0, (violations, changed_files, patched)
    assert guide_ok and stats["actions"] >= 100 and stored == stats["records"] and stats["updates"] >= 1
