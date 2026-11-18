"""Task specifications, admission control and curriculum sampling (see package docstring)."""
from __future__ import annotations

import hashlib
import inspect
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from agentrs.econ.core import REGISTRY, SPLIT_SEED_BASE, Task, load_all, make_task

GENERATOR_SCHEMA = "agentrs-task-factory-v1"
COUNTERPARTIES = {
    "support": "deterministic scripted customer (reveals account details on request, confirms only when asked)",
    "crm": "deterministic scripted prospect / counterparty",
}


def _src_hash(obj) -> str:
    try:
        return hashlib.sha1(inspect.getsource(inspect.getmodule(obj)).encode()).hexdigest()[:10]
    except (OSError, TypeError):
        return "unknown"


def _stable(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


@dataclass
class TaskSpec:
    """Everything that defines one executable task instance. ``private_keys`` names the hidden state
    (expected final state / solution) without exposing it; the worker never sees ``Task.private``."""
    key: str
    domain: str
    family: str
    split: str
    index: int
    seed: int
    generator_version: str
    environment: str
    environment_version: str
    goal: str
    deliverable: str
    initial_observation: str
    tools: list = field(default_factory=list)
    rules: list = field(default_factory=list)
    observable_schema: str = ""
    private_keys: list = field(default_factory=list)
    counterparties: Optional[str] = None
    step_budget: int = 12
    difficulty: Optional[float] = None
    expert_steps: Optional[int] = None
    verifier: str = ""
    heldout_family: bool = False
    fingerprint: str = ""
    instance_hash: str = ""
    provenance: dict = field(default_factory=dict)

    def public_dict(self) -> dict:
        return asdict(self)


def build_spec(task: Task, index: int, env=None) -> TaskSpec:
    fam = REGISTRY[f"{task.domain}/{task.family}"]
    env = env if env is not None else fam.make_env(task)
    tools = []
    if hasattr(env, "tools") and isinstance(env.tools, dict):
        tools = [{"name": t.name, "writes": t.writes, "terminal": t.terminal} for t in env.tools.values()]
        term = [t for t in env.tools.values() if t.terminal]
        deliverable = "; ".join(t.doc for t in term) or "final application state"
    else:  # MiniWeb browser apps: free-form browser actions, server-side verifier
        deliverable = "final state of the web application (server-side check)"
    gen_v = f"{GENERATOR_SCHEMA}:{fam.key}:{_src_hash(fam.make_task)}"
    return TaskSpec(
        key=fam.key, domain=task.domain, family=task.family, split=task.split, index=index, seed=task.seed,
        generator_version=gen_v, environment=type(env).__name__, environment_version=_src_hash(type(env)),
        goal=task.goal, deliverable=deliverable, initial_observation=str(env.initial_observation())[:2000],
        tools=tools, rules=list(getattr(env, "rules", []) or []),
        observable_schema=env.schema_text() if hasattr(env, "schema_text") else "",
        private_keys=sorted(task.private.keys()), counterparties=COUNTERPARTIES.get(task.domain),
        step_budget=task.max_steps, verifier=f"{type(env).__name__}.verify (hidden; inspects final state)",
        heldout_family=fam.heldout, fingerprint=task.fingerprint(),
        instance_hash=hashlib.sha1(f"{fam.key}|{task.goal}|{_stable(task.private)}".encode()).hexdigest()[:16],
        provenance={"generator": f"agentrs.econ.domains.{task.domain}", "data": "synthetic (local generators)",
                    "third_party_content": False, "training_permitted": task.split == "train" and not fam.heldout,
                    "split_seed_base": SPLIT_SEED_BASE[task.split]})


@dataclass
class Admission:
    admitted: bool
    reason: str
    spec: Optional[TaskSpec] = None
    expert_success: Optional[bool] = None


class TaskFactory:
    """Generates, validates and samples tasks. All mutable state lives in ``state`` (a dict inside
    the trainer's run state) so that it is checkpointed and restored with the run."""

    def __init__(self, state: dict, families: list[str], out_dir: Optional[Path] = None, validate_expert: bool = True,
                 seed: int = 0):
        load_all()
        self.families = sorted(families)
        self.domains = sorted({k.split("/")[0] for k in self.families})
        self.out_dir = out_dir
        self.validate_expert = validate_expert
        self.seed = seed
        st = state.setdefault("factory", {})
        st.setdefault("generated", 0)
        st.setdefault("admitted", 0)
        st.setdefault("rejected", {})
        st.setdefault("by_family", {})
        st.setdefault("instance_hashes", [])
        st.setdefault("fingerprints", {})
        self.st = st
        self._seen = set(st["instance_hashes"])
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ admission
    def admit(self, task: Task, index: int) -> Admission:
        from agentrs.econ.rollout import make_env, run_expert

        self.st["generated"] += 1
        fam = REGISTRY[f"{task.domain}/{task.family}"]
        try:
            env = make_env(task)
            spec = build_spec(task, index, env)
        except Exception as e:  # noqa: BLE001 - a generator bug must not stop training
            return self._reject(task, f"environment setup failed: {type(e).__name__}")
        if spec.instance_hash in self._seen:
            return self._reject(task, "duplicate instance")
        try:
            already, _ = env.verify()
        except Exception as e:  # noqa: BLE001
            return self._reject(task, f"verifier crashed on initial state: {type(e).__name__}")
        if already:
            return self._reject(task, "initial state already satisfies the verifier (trivial)")
        ok = None
        if self.validate_expert and (fam.oracle is not None or hasattr(env, "expert_action")):
            try:
                ep = run_expert(task)
                ok = bool(ep.success)
                spec.expert_steps = len(ep.steps)
            except Exception as e:  # noqa: BLE001
                return self._reject(task, f"expert crashed: {type(e).__name__}")
            if not ok:
                return self._reject(task, "expert could not solve it (possibly impossible / unverifiable)")
        self._seen.add(spec.instance_hash)
        self.st["instance_hashes"].append(spec.instance_hash)
        self.st["instance_hashes"] = self.st["instance_hashes"][-20000:]
        self.st["admitted"] += 1
        bf = self.st["by_family"].setdefault(spec.key, {"admitted": 0, "rejected": 0})
        bf["admitted"] += 1
        self.st["fingerprints"][spec.fingerprint] = self.st["fingerprints"].get(spec.fingerprint, 0) + 1
        if self.out_dir:
            with open(self.out_dir / "admitted_specs.jsonl", "a") as f:
                f.write(json.dumps(spec.public_dict(), default=str) + "\n")
        return Admission(True, "ok", spec, ok)

    def _reject(self, task: Task, reason: str) -> Admission:
        key = f"{task.domain}/{task.family}"
        self.st["rejected"][reason] = self.st["rejected"].get(reason, 0) + 1
        bf = self.st["by_family"].setdefault(key, {"admitted": 0, "rejected": 0})
        bf["rejected"] += 1
        return Admission(False, reason)

    # ------------------------------------------------------------------ curriculum
    @staticmethod
    def family_weight(cur: dict) -> float:
        """Learning-progress weighting: most weight where success is neither ~0 nor ~1; unsolved
        families keep a bonus (hard tasks stay in the mix), mastered ones keep a small floor so
        forgetting is noticed, but they stop consuming the budget."""
        ema, n = cur.get("ema"), cur.get("n", 0)
        if ema is None:
            return 1.0
        if ema >= 0.92 and n >= 24:
            return 0.03
        return max(0.08, 4 * ema * (1 - ema)) + (0.15 if ema < 0.05 else 0.0)

    def sample(self, n: int, cycle: int, train_index: dict, curriculum: dict, max_tries: int = 6) -> list[Task]:
        """Domain-balanced curriculum batch of admitted training tasks (fresh instances every time)."""
        rng = random.Random(self.seed * 7919 + cycle)
        doms: list[str] = []
        while len(doms) < n:
            d = list(self.domains)
            rng.shuffle(d)
            doms += d
        picks = []
        for d in doms[:n]:
            fams = [k for k in self.families if k.startswith(d + "/")]
            w = [self.family_weight(curriculum.get(k, {})) for k in fams]
            k = rng.choices(fams, w)[0]
            for _ in range(max_tries):
                idx = train_index[k]
                train_index[k] = idx + 1
                try:
                    t = make_task(k, "train", idx)
                except Exception as e:  # noqa: BLE001 - generator bug: reject this index, keep training
                    self.st["generated"] += 1
                    r = f"generator crashed: {type(e).__name__}"
                    self.st["rejected"][r] = self.st["rejected"].get(r, 0) + 1
                    continue
                if self.admit(t, idx).admitted:
                    picks.append(t)
                    break
        return picks

    def coverage(self) -> dict:
        fp = self.st["fingerprints"]
        return {"generated": self.st["generated"], "admitted": self.st["admitted"], "rejected": self.st["rejected"],
                "unique_instances": len(self._seen), "unique_goal_templates": len(fp),
                "families_admitted": len([k for k, v in self.st["by_family"].items() if v["admitted"]]),
                "domains": self.domains}
