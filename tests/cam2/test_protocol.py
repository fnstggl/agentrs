"""The three-way evaluation protocol cannot be reached or modified by the outer learner."""
import ast
from pathlib import Path

import pytest

from agentrs.cam2 import tasks
from agentrs.cam2.envs.miniwob import SEED_RANGES, TEST_FAMILIES, TRAIN_FAMILIES, VAL_FAMILIES

SEARCH = Path(tasks.__file__).parent / "search"
FORBIDDEN = {"locked_test_tasks", "ACK", "ACCESS_LOG", "TEST_FAMILIES"}


def _names(path: Path) -> set:
    tree = ast.parse(path.read_text())
    docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
                  and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    out = set()
    for n in ast.walk(tree):
        if id(n) in docstrings:
            continue
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, ast.alias):
            out.add(n.name.split(".")[-1])
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and "locked" in n.value.lower():
            out.add("locked-string")
    return out


@pytest.mark.parametrize("path", sorted(SEARCH.glob("*.py")), ids=lambda p: p.name)
def test_outer_learner_never_references_the_locked_test(path):
    bad = (_names(path) & FORBIDDEN) | ({"locked-string"} & _names(path))
    assert not bad, f"{path.name} references {bad}"


def test_splits_are_disjoint():
    assert not (set(TRAIN_FAMILIES) & set(VAL_FAMILIES)) and not (set(TRAIN_FAMILIES) & set(TEST_FAMILIES))
    assert not (set(VAL_FAMILIES) & set(TEST_FAMILIES))
    r = sorted(SEED_RANGES.values())
    assert all(a[1] <= b[0] for a, b in zip(r, r[1:]))


def test_training_sampler_refuses_held_out_families():
    with pytest.raises(ValueError):
        tasks.TrainSampler("miniwob", ["click-dialog-2"])
    s = tasks.TrainSampler("miniwob", seed=0)
    for _ in range(200):
        t = s()
        assert t.family in TRAIN_FAMILIES and SEED_RANGES["train"][0] <= t.seed < SEED_RANGES["train"][1]


def test_validation_tasks_are_outside_training_and_test():
    v = tasks.val_tasks("miniwob", 2)
    assert all(t.seed >= SEED_RANGES["val"][0] and t.seed < SEED_RANGES["val"][1] for t in v)
    assert not ({t.family for t in v} & set(TEST_FAMILIES))


def test_locked_test_needs_acknowledgement(tmp_path, monkeypatch):
    monkeypatch.setattr(tasks, "ACCESS_LOG", tmp_path / "log")
    with pytest.raises(PermissionError):
        tasks.locked_test_tasks("miniwob", "please")
    t = tasks.locked_test_tasks("miniwob", tasks.ACK, per_family=1, who="test")
    assert {x.family for x in t} == set(TEST_FAMILIES) and (tmp_path / "log").read_text().count("\n") == 1


def test_protocol_hash_is_stable():
    assert tasks.protocol_hash() == tasks.protocol_hash()
