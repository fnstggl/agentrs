"""Software engineering on small generated Python repositories with executable tests.

Families: bug fixing, API repair after a rename, config repair against a validator, dependency pin
repair against a compatibility matrix, and implementing a stubbed function from its docstring.
The agent edits files with an exact-string ``replace`` tool and can run the visible tests; on
``submit`` a HIDDEN test suite (more cases) runs. Reward = all hidden tests pass.

Isolation: tests run in a subprocess as the unprivileged ``nobody`` user (when started as root),
with CPU / memory / file-size / process limits, a wall-clock timeout, ``python -I`` (no site
packages), an empty environment and a read-only working directory. Code is generated from local
templates; agent edits to .py files may not introduce imports or dunder / I/O / reflection calls.
No internet repository code is executed.
"""
from __future__ import annotations

import json
import os
import random
import resource
import shutil
import subprocess
import sys
import tempfile

from agentrs.econ.core import Family, Task, ToolEnv, ToolError, register

BANNED = ("import ", "__", "open(", "exec(", "eval(", "os.", "sys.", "subprocess", "socket", "shutil", "pathlib",
          "compile(", "globals(", "locals(", "getattr(", "setattr(", "delattr(", "input(", "breakpoint(", "while True")

TEMPLATES = [
    dict(fn="sum_evens", params="nums", doc="Return the sum of the even numbers in nums.",
         body="total = 0\nfor x in nums:\n    if x % 2 == 0:\n        total += x\nreturn total",
         bugs=[("x % 2 == 0", "x % 2 == 1"), ("total += x", "total = x")],
         gen="([r.randint(-9, 30) for _ in range(r.randint(1, 7))],)"),
    dict(fn="clamp", params="x, lo, hi", doc="Return x limited to the closed interval [lo, hi].",
         body="return max(lo, min(x, hi))", bugs=[("max(lo, min(x, hi))", "min(lo, max(x, hi))"),
                                                  ("max(lo, min(x, hi))", "max(lo, min(x, lo))")],
         gen="(r.randint(-50, 50), r.randint(-20, 0), r.randint(1, 20))"),
    dict(fn="moving_average", params="xs, k", doc="Return the list of averages of every window of k consecutive values.",
         body="out = []\nfor i in range(len(xs) - k + 1):\n    out.append(sum(xs[i:i + k]) / k)\nreturn out",
         bugs=[("range(len(xs) - k + 1)", "range(len(xs) - k)"), ("/ k)", "/ len(xs))")],
         gen="([r.randint(0, 20) for _ in range(r.randint(3, 8))], r.randint(1, 3))"),
    dict(fn="dedupe", params="items", doc="Return items without duplicates, keeping the first occurrence order.",
         body="seen = set()\nout = []\nfor it in items:\n    if it not in seen:\n        seen.add(it)\n        out.append(it)\nreturn out",
         bugs=[("        seen.add(it)\n", ""), ("out.append(it)\nreturn out", "out.append(it)\nreturn sorted(out)")],
         gen="([r.randint(0, 6) for _ in range(r.randint(2, 9))],)"),
    dict(fn="paginate", params="items, page, size", doc="Return page number `page` (1-based) of `items` with `size` items per page.",
         body="start = (page - 1) * size\nreturn items[start:start + size]",
         bugs=[("(page - 1) * size", "page * size"), ("start + size]", "start + size - 1]")],
         gen="(list(range(r.randint(3, 15))), r.randint(1, 3), r.randint(1, 5))"),
    dict(fn="apply_discount", params="price, pct", doc="Return price reduced by pct percent, rounded to 2 decimals.",
         body="return round(price * (1 - pct / 100), 2)", bugs=[("(1 - pct / 100)", "(1 - pct)"), ("round(price * (1 - pct / 100), 2)", "int(round(price * (1 - pct / 100), 2))")],
         gen="(round(r.uniform(1, 500), 2), r.choice([5, 10, 15, 20, 25, 50]))"),
    dict(fn="word_freq", params="text", doc="Return a dict mapping each lower-cased word of text to its count.",
         body="counts = {}\nfor w in text.lower().split():\n    counts[w] = counts.get(w, 0) + 1\nreturn counts",
         bugs=[("text.lower().split()", "text.split()"), ("counts.get(w, 0) + 1", "counts.get(w, 1)")],
         gen="(' '.join(r.choice(['Apple', 'apple', 'Pear', 'kiwi', 'KIWI', 'fig']) for _ in range(r.randint(2, 7))),)"),
    dict(fn="is_leap", params="year", doc="Return True if year is a Gregorian leap year.",
         body="return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)",
         bugs=[("(year % 100 != 0 or year % 400 == 0)", "year % 100 != 0"), ("year % 4 == 0 and ", "")],
         gen="(r.choice([1900, 2000, 2024, 2023, 2100, 1996, 2400, 2019, 1800, 2016]),)"),
    dict(fn="parse_kv", params="s", doc="Parse 'a=1; b=2' into {'a': '1', 'b': '2'} (keys/values stripped; values may contain '=').",
         body="out = {}\nfor part in s.split(';'):\n    if '=' in part:\n        k, v = part.split('=', 1)\n        out[k.strip()] = v.strip()\nreturn out",
         bugs=[("part.split('=', 1)", "part.split('=')"), ("out[k.strip()] = v.strip()", "out[k] = v")],
         gen="('; '.join(f'{k} = {v}' for k, v in zip(['a', 'b', 'mode', 'x'], [r.randint(0, 9), 'on', 'k=v', r.randint(10, 99)][:r.randint(1, 4)])),)"),
    dict(fn="median", params="xs", doc="Return the median of a non-empty list of numbers.",
         body="s = sorted(xs)\nn = len(s)\nmid = n // 2\nif n % 2 == 1:\n    return s[mid]\nreturn (s[mid - 1] + s[mid]) / 2",
         bugs=[("s = sorted(xs)", "s = list(xs)"), ("(s[mid - 1] + s[mid]) / 2", "s[mid]")],
         gen="([r.randint(-10, 40) for _ in range(r.randint(1, 8))],)"),
    dict(fn="invoice_total", params="lines, tax_rate", doc="lines are (qty, unit_price) pairs; return subtotal plus tax, rounded to 2 decimals.",
         body="subtotal = sum(q * p for q, p in lines)\nreturn round(subtotal * (1 + tax_rate), 2)",
         bugs=[("(1 + tax_rate)", "tax_rate"), ("q * p", "q + p")],
         gen="([(r.randint(1, 9), round(r.uniform(1, 80), 2)) for _ in range(r.randint(1, 4))], r.choice([0.0, 0.05, 0.08, 0.2]))"),
    dict(fn="max_index", params="xs", doc="Return the index of the first maximum element of a non-empty list.",
         body="best = 0\nfor i in range(1, len(xs)):\n    if xs[i] > xs[best]:\n        best = i\nreturn best",
         bugs=[("if xs[i] > xs[best]", "if xs[i] >= xs[best]"), ("best = 0\n", "best = 1\n")],
         gen="([r.randint(0, 9) for _ in range(r.randint(2, 8))],)"),
]
MODULES = ["utils", "helpers", "core", "calc", "billing", "stats", "textops", "pricing", "records", "tools"]
R_HELPER = ("def _r(v):\n    if isinstance(v, float):\n        return round(v, 6)\n    if isinstance(v, (list, tuple)):\n"
            "        return type(v)(_r(x) for x in v)\n    if isinstance(v, dict):\n        return {k: _r(x) for k, x in v.items()}\n"
            "    return v\n")


def _indent(body: str, n: int = 4) -> str:
    return "\n".join((" " * n + ln) if ln else ln for ln in body.split("\n"))


def _func_src(t, body=None, name=None):
    return f"def {name or t['fn']}({t['params']}):\n    \"\"\"{t['doc']}\"\"\"\n{_indent(body or t['body'])}\n"


def _call(src: str, fn: str, args):
    ns: dict = {}
    exec(src, ns)  # our own template code, not agent code
    try:
        return ("ok", ns[fn](*args))
    except Exception as e:  # noqa: BLE001 - the buggy variant may raise
        return ("err", type(e).__name__)


def _cases(t, rng, buggy_src=None, n=8, need_diff=True):
    good = _func_src(t)
    cases, diff = [], None
    for _ in range(400):
        args = eval(t["gen"], {"r": rng})  # generator expressions are local template constants
        st, out = _call(good, t["fn"], args)
        if st != "ok":
            continue
        if buggy_src is not None and diff is None:
            if _call(buggy_src, t["fn"], args) != ("ok", out):
                diff = (args, out)
                continue
        if len(cases) < n:
            cases.append((args, out))
        if len(cases) >= n and (diff is not None or not need_diff):
            break
    return cases, diff


def _test_file(module, fn, cases, cls="TestVisible"):
    lines = ["import unittest", f"from {module} import {fn}", "", R_HELPER, "", f"class {cls}(unittest.TestCase):"]
    for i, (args, out) in enumerate(cases):
        a = ", ".join(repr(x) for x in args)
        lines += [f"    def test_{i}(self):", f"        self.assertEqual(_r({fn}({a})), _r({out!r}))", ""]
    lines += ["", "if __name__ == '__main__':", "    unittest.main()", ""]
    return "\n".join(lines)


def run_sandboxed(files: dict, timeout: float = 15.0) -> tuple[bool, str]:
    """Write ``files`` into a fresh read-only temp dir and run unittest discovery there."""
    d = tempfile.mkdtemp(prefix="agentrs_swe_")
    try:
        for name, content in files.items():
            with open(os.path.join(d, name), "w") as f:
                f.write(content)
            os.chmod(os.path.join(d, name), 0o644)
        os.chmod(d, 0o755)
        drop = os.geteuid() == 0

        def pre():
            resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
            resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
            resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
            os.setsid()
            if drop:
                os.setgid(65534)
                os.setuid(65534)

        r = subprocess.run([sys.executable, "-I", "-B", "-m", "unittest", "discover", "-q", "-p", "test_*.py"], cwd=d,
                           env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True,
                           timeout=timeout, preexec_fn=pre)
        out = (r.stdout + r.stderr).strip()
        return r.returncode == 0, out
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT: tests took too long"
    finally:
        shutil.rmtree(d, ignore_errors=True)


class SoftwareEnv(ToolEnv):
    domain = "software"
    role = "a software engineer working in a small Python repository"
    rules = ["Test files (test_*.py) must not be modified.",
             "Edits to .py files may not add imports or use dunder names, I/O, eval/exec or reflection.",
             "Call submit() when you believe the hidden test-suite will pass."]

    def setup(self):
        p = self.task.private
        self.files = dict(p["files"])
        self.final_ok = None

        @self.tool("list_files", "list_files() — list the repository files")
        def list_files():
            return "\n".join(sorted(self.files))

        @self.tool("read_file", "read_file(path: str) — show a file")
        def read_file(path):
            if path not in self.files:
                raise ToolError("no such file")
            return self.files[path]

        @self.tool("replace", "replace(path: str, old: str, new: str) — replace an exact snippet (must occur once)",
                   writes=True)
        def replace(path, old, new):
            if path not in self.files:
                raise ToolError("no such file")
            if path.startswith("test_"):
                raise ToolError("test files are read-only")
            old, new = str(old), str(new)
            c = self.files[path].count(old) if old else 0
            if c != 1:
                raise ToolError(f"'old' must occur exactly once in {path} (found {c} times)")
            if path.endswith(".py") and any(b in new for b in BANNED):
                raise ToolError("edit rejected by sandbox policy (imports / dunder / I/O / reflection not allowed)")
            if len(new) > 4000:
                raise ToolError("edit too large")
            self.files[path] = self.files[path].replace(old, new, 1)
            return f"{path} updated"

        @self.tool("run_tests", "run_tests() — run the visible test-suite")
        def run_tests():
            ok, out = run_sandboxed(self.files)
            return ("PASSED\n" if ok else "FAILED\n") + out[-1500:]

        @self.tool("submit", "submit() — finish; the hidden test-suite is run (terminal)", terminal=True)
        def submit():
            return "submitted"

    def initial_observation(self):
        return "Repository files:\n" + "\n".join(sorted(self.files))

    def verify(self):
        files = {k: v for k, v in self.files.items() if not k.startswith("test_")}
        files.update(self.task.private["hidden_tests"])
        ok, out = run_sandboxed(files)
        return ok, {"hidden_tests_output": out[-400:]}


def _rng(f, split, seed):
    return random.Random(f"software:{f}:{split}:{seed}")


def _pick_template(rng, split):
    # held-out instances for the sealed test split: disjoint template subset ensures new code bases
    ts = TEMPLATES[8:] if split == "test" else TEMPLATES[:8]
    return rng.choice(ts)


def make_bugfix(split, seed):
    rng = _rng("bug", split, seed)
    for _ in range(20):
        t = _pick_template(rng, split)
        old, new = rng.choice(t["bugs"])
        body_b = t["body"].replace(old, new, 1)
        buggy = _func_src(t, body_b)
        cases, diff = _cases(t, rng, buggy)
        if diff is not None:
            break
    mod = rng.choice(MODULES)
    visible = [diff] + cases[:2]
    hidden = [diff] + cases
    files = {f"{mod}.py": buggy, f"test_{mod}.py": _test_file(mod, t["fn"], visible)}
    goal = f"The tests for `{t['fn']}` in {mod}.py are failing. Find and fix the bug, then submit()."
    plan = [json.dumps({"tool": "run_tests", "args": {}}), json.dumps({"tool": "read_file", "args": {"path": f"{mod}.py"}})]
    if new.strip() and buggy.count(new) == 1:  # minimal edit: undo the injected change
        plan.append(json.dumps({"tool": "replace", "args": {"path": f"{mod}.py", "old": new, "new": old}}))
    else:  # deleted line: rewrite the function
        plan.append(json.dumps({"tool": "replace", "args": {"path": f"{mod}.py", "old": buggy, "new": _func_src(t)}}))
    plan += [json.dumps({"tool": "run_tests", "args": {}}), json.dumps({"tool": "submit", "args": {}})]
    return Task("software", "bugfix", split, seed, goal,
                {"files": files, "hidden_tests": {f"test_{mod}_hidden.py": _test_file(mod, t["fn"], hidden, "TestHidden")},
                 "plan": plan}, max_steps=8)


def make_implement(split, seed):
    rng = _rng("impl", split, seed)
    ts = [t for t in (TEMPLATES[8:] if split == "test" else TEMPLATES[:8])]
    t = rng.choice(ts)
    cases, _ = _cases(t, rng, None, n=8, need_diff=False)
    mod = rng.choice(MODULES)
    stub = _func_src(t, "raise NotImplementedError")
    files = {f"{mod}.py": stub, f"test_{mod}.py": _test_file(mod, t["fn"], cases[:3])}
    goal = f"Implement `{t['fn']}` in {mod}.py according to its docstring (replace the stub), then submit()."
    plan = [json.dumps({"tool": "read_file", "args": {"path": f"{mod}.py"}}),
            json.dumps({"tool": "replace", "args": {"path": f"{mod}.py", "old": "    raise NotImplementedError",
                                                    "new": _indent(t["body"])}}),
            json.dumps({"tool": "run_tests", "args": {}}), json.dumps({"tool": "submit", "args": {}})]
    return Task("software", "implement_function", split, seed, goal,
                {"files": files, "hidden_tests": {f"test_{mod}_hidden.py": _test_file(mod, t["fn"], cases, "TestHidden")},
                 "plan": plan}, max_steps=8)


def make_api_repair(split, seed):
    rng = _rng("api", split, seed)
    t = _pick_template(rng, split)
    cases, _ = _cases(t, rng, None, n=8, need_diff=False)
    new_name = t["fn"] + rng.choice(["_v2", "_safe", "_impl", "_strict"])
    lib = "# CHANGELOG: `%s` was renamed to `%s` in release 2.0.\n\n" % (t["fn"], new_name) + _func_src(t, name=new_name)
    nargs = len([p for p in t["params"].split(",") if p.strip()])
    params = ", ".join(f"a{i}" for i in range(nargs))
    app = f"import lib\n\n\ndef {t['fn']}({params}):\n    return lib.{t['fn']}({params})\n"
    files = {"lib.py": lib, "app.py": app, "test_app.py": _test_file("app", t["fn"], cases[:3])}
    goal = "The application in app.py broke after upgrading lib.py to release 2.0. Repair app.py, then submit()."
    plan = [json.dumps({"tool": "run_tests", "args": {}}), json.dumps({"tool": "read_file", "args": {"path": "lib.py"}}),
            json.dumps({"tool": "replace", "args": {"path": "app.py", "old": f"lib.{t['fn']}(", "new": f"lib.{new_name}("}}),
            json.dumps({"tool": "run_tests", "args": {}}), json.dumps({"tool": "submit", "args": {}})]
    return Task("software", "api_repair", split, seed, goal,
                {"files": files, "hidden_tests": {"test_app_hidden.py": _test_file("app", t["fn"], cases, "TestHidden")},
                 "plan": plan}, max_steps=8)


SETTINGS = '''import json

ALLOWED_LEVELS = ["debug", "info", "warning", "error"]


def load(path="config.json"):
    with open(path) as f:
        cfg = json.load(f)
    for key in ("service", "timeout_s", "retries", "workers", "log_level"):
        if key not in cfg:
            raise ValueError(f"missing key: {key}")
    if not isinstance(cfg["timeout_s"], (int, float)) or cfg["timeout_s"] <= 0:
        raise ValueError("timeout_s must be a positive number")
    if not isinstance(cfg["retries"], int) or not 0 <= cfg["retries"] <= 10:
        raise ValueError("retries must be an integer between 0 and 10")
    if not isinstance(cfg["workers"], int) or not 1 <= cfg["workers"] <= 32:
        raise ValueError("workers must be an integer between 1 and 32")
    if cfg["log_level"] not in ALLOWED_LEVELS:
        raise ValueError(f"log_level must be one of {ALLOWED_LEVELS}")
    return cfg
'''


def make_config_fix(split, seed):
    rng = _rng("cfg", split, seed)
    good = {"service": rng.choice(["billing-api", "search", "ingest", "auth"]), "timeout_s": rng.choice([5, 10, 30]),
            "retries": rng.randint(1, 5), "workers": rng.choice([2, 4, 8]), "log_level": rng.choice(["info", "warning"])}
    bad = dict(good)
    kind = rng.choice(["type", "rename", "range", "enum"])
    if kind == "type":
        bad["retries"] = str(good["retries"])
    elif kind == "rename":
        bad["timeout"] = bad.pop("timeout_s")
    elif kind == "range":
        bad["workers"] = rng.choice([0, 64, 100])
    else:
        bad["log_level"] = rng.choice(["verbose", "INFO", "trace"])
    bad_txt = json.dumps(bad, indent=2)
    test = ("import unittest\nimport settings\n\n\nclass TestConfig(unittest.TestCase):\n    def test_loads(self):\n"
            "        cfg = settings.load()\n        self.assertEqual(cfg['service'], %r)\n\n\nif __name__ == '__main__':\n"
            "    unittest.main()\n" % good["service"])
    hidden = ("import json\nimport unittest\nimport settings\n\n\nclass TestHidden(unittest.TestCase):\n"
              "    def test_loads(self):\n        cfg = settings.load()\n        self.assertEqual(cfg['service'], %r)\n"
              "        self.assertEqual(cfg['timeout_s'], %r)\n\n\nif __name__ == '__main__':\n    unittest.main()\n"
              % (good["service"], good["timeout_s"]))
    files = {"settings.py": SETTINGS, "config.json": bad_txt, "test_settings.py": test}
    goal = ("The service fails to start because config.json does not pass validation in settings.py. Fix config.json "
            "(do not change settings.py or the service name/timeout values), then submit().")
    fix_old, fix_new = {"type": (f'"retries": "{good["retries"]}"', f'"retries": {good["retries"]}'),
                        "rename": ('"timeout":', '"timeout_s":'),
                        "range": (f'"workers": {bad["workers"]}', f'"workers": {good["workers"]}'),
                        "enum": (f'"log_level": "{bad["log_level"]}"', f'"log_level": "{good["log_level"]}"')}[kind]
    if kind == "enum" and bad["log_level"] == "INFO":
        fix_new = '"log_level": "info"'
    plan = [json.dumps({"tool": "run_tests", "args": {}}), json.dumps({"tool": "read_file", "args": {"path": "config.json"}}),
            json.dumps({"tool": "replace", "args": {"path": "config.json", "old": fix_old, "new": fix_new}}),
            json.dumps({"tool": "run_tests", "args": {}}), json.dumps({"tool": "submit", "args": {}})]
    return Task("software", "config_fix", split, seed, goal,
                {"files": files, "hidden_tests": {"test_settings_hidden.py": hidden}, "plan": plan}, max_steps=8)


CHECK_DEPS = '''import json


def parse_req(text):
    pins = {}
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, ver = line.split("==")
            pins[name.strip()] = tuple(int(x) for x in ver.strip().split("."))
    return pins


def _ok(ver, spec):
    op, v = spec[:2] if spec[:2] in (">=", "<=") else spec[:1], spec[2:] if spec[:2] in (">=", "<=") else spec[1:]
    v = tuple(int(x) for x in v.split("."))
    return {">=": ver >= v, "<=": ver <= v, "<": ver < v, ">": ver > v}[op]


def problems(req_path="requirements.txt", compat_path="compat.json", avail_path="available.json"):
    pins = parse_req(open(req_path).read())
    compat = json.load(open(compat_path))
    avail = json.load(open(avail_path))
    out = []
    for name, ver in pins.items():
        if ".".join(map(str, ver)) not in avail.get(name, []):
            out.append(f"{name}=={'.'.join(map(str, ver))} is not a published version")
    for key, needs in compat.items():
        name, ver = key.split("==")
        if name in pins and pins[name] == tuple(int(x) for x in ver.split(".")):
            for dep, specs in needs.items():
                for spec in specs:
                    if dep in pins and not _ok(pins[dep], spec):
                        out.append(f"{key} requires {dep}{spec}")
    return out
'''


def make_dependency_fix(split, seed):
    rng = _rng("dep", split, seed)
    pk = rng.sample(["alpha", "requestor", "datakit", "fastjson", "plotter", "netcore", "cachelib"], 3)
    avail = {p: [f"{m}.{n}.0" for m in range(1, 4) for n in range(0, 4)] for p in pk}
    a, b, c = pk
    a_ver = rng.choice(["2.1.0", "3.0.0", "2.3.0"])
    lo, hi = rng.choice([("1.2.0", "2.0.0"), ("2.0.0", "3.0.0"), ("1.1.0", "1.3.0")])
    good_b = rng.choice([v for v in avail[b] if tuple(map(int, lo.split("."))) <= tuple(map(int, v.split("."))) <
                         tuple(map(int, hi.split(".")))])
    bad_b = rng.choice([v for v in avail[b] if not (tuple(map(int, lo.split("."))) <= tuple(map(int, v.split("."))) <
                                                    tuple(map(int, hi.split("."))))])
    c_ver = rng.choice(avail[c])
    req_bad = f"{a}=={a_ver}\n{b}=={bad_b}\n{c}=={c_ver}\n"
    compat = {f"{a}=={a_ver}": {b: [f">={lo}", f"<{hi}"]}}
    files = {"requirements.txt": req_bad, "compat.json": json.dumps(compat, indent=2),
             "available.json": json.dumps(avail), "check_deps.py": CHECK_DEPS,
             "test_deps.py": ("import unittest\nimport check_deps\n\n\nclass TestDeps(unittest.TestCase):\n"
                              "    def test_consistent(self):\n        self.assertEqual(check_deps.problems(), [])\n\n\n"
                              "if __name__ == '__main__':\n    unittest.main()\n")}
    hidden = ("import unittest\nimport check_deps\n\n\nclass TestHidden(unittest.TestCase):\n    def test_consistent(self):\n"
              "        self.assertEqual(check_deps.problems(), [])\n        pins = check_deps.parse_req(open('requirements.txt').read())\n"
              f"        self.assertEqual(pins[{a!r}], {tuple(map(int, a_ver.split('.')))!r})\n"
              f"        self.assertEqual(pins[{c!r}], {tuple(map(int, c_ver.split('.')))!r})\n\n\n"
              "if __name__ == '__main__':\n    unittest.main()\n")
    goal = (f"The dependency check fails. Change ONLY the pin of the conflicting package in requirements.txt to a "
            f"published version that satisfies compat.json (keep {a} and {c} as they are), then submit().")
    plan = [json.dumps({"tool": "run_tests", "args": {}}), json.dumps({"tool": "read_file", "args": {"path": "compat.json"}}),
            json.dumps({"tool": "replace", "args": {"path": "requirements.txt", "old": f"{b}=={bad_b}", "new": f"{b}=={good_b}"}}),
            json.dumps({"tool": "run_tests", "args": {}}), json.dumps({"tool": "submit", "args": {}})]
    return Task("software", "dependency_fix", split, seed, goal,
                {"files": files, "hidden_tests": {"test_deps_hidden.py": hidden}, "plan": plan}, max_steps=8)


def oracle(task: Task) -> list[str]:
    return list(task.private["plan"])


for _n, _f in [("bugfix", make_bugfix), ("implement_function", make_implement), ("api_repair", make_api_repair),
               ("config_fix", make_config_fix), ("dependency_fix", make_dependency_fix)]:
    register(Family("software", _n, _f, SoftwareEnv, oracle))
