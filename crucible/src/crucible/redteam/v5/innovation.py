"""Probes against adoption-by-execution and service metering in the software ecosystem (workstream G; Gate W.3, W.4,
W.6). Adoption is now measured by parent-side ablation (`crucible.innovation.pkg._ablations`); the forgeries are the
audits' (O P6, G 5.1 A/B), plus the known residual (a dependent inspecting the package's files)."""
from __future__ import annotations

import time

from ..framework import FALSE_SUCCESS, HELD, INTEGRITY_BREAK, Attacker, Evaluator, NotApplicable, Operator, Verdict
from .base import V5Probe, deploy, manifest, org_of

LIB = "def hello(x):\n    return x + 1\n"
LIB_TEST = "from lib import hello\ndef test_h():\n    assert hello(1) == 2\n"


class _Adoption(V5Probe):
    target, owner = "innovation.pkg", "G"
    world = {"v5": "innovation"}
    needs_sandbox = True
    gates = ("W.4", "W.3")
    files: dict = {}

    def setup(self, op: Operator) -> None:
        self.maker = op.join("libmaker", 100_000, attacker=False)
        self.att = op.join("adopter", 100_000)
        repo = op.as_(self.maker, "code.create_repo", name="greet", visibility="public", license="open")["repo"]
        op.as_(self.maker, "code.commit", repo=repo, files={"lib.py": LIB, "tests/test_lib.py": LIB_TEST}, message="v1")
        op.as_(self.maker, "pkg.register", name="greet", repo=repo, description="adds one")
        op.as_(self.maker, "pkg.release", package="greet", version="1.0.0")
        self.dep_repo = repo

    def app_files(self, dep_dir: str) -> dict:
        raise NotImplementedError

    def attack(self, a: Attacker) -> None:
        from ...innovation.pkg import import_dir
        repo = a.must(self.att, "code.create_repo", name="app")["repo"]
        a.must(self.att, "code.commit", repo=repo, files={"app.py": "def main(i, c):\n    return 'ok'\n"}, message="x")
        a.must(self.att, "pkg.depend", repo=repo, package="greet")
        a.must(self.att, "code.commit", repo=repo, files=self.app_files(import_dir(self.dep_repo)), message="forge")
        self.res = [a.act(self.att, "pkg.exec", repo=repo, target="app:main", input=None) for _ in range(2)]
        self.ci = a.act(self.att, "pkg.ci", repo=repo)
        self.repo = repo

    def verdict(self, ev: Evaluator) -> Verdict:
        st = ev.world.state
        ad = {k: dict(v) for k, v in st.tables.get("pkg_adoptions", {}).items()}
        calls = sum(int(v.get("calls", 0)) for v in ad.values())
        evd = {"exec": [(r.ok, (r.result or {}).get("calls") if r.ok else (r.error or {}).get("code")) for r in self.res],
               "ci": (self.ci.ok, (self.ci.result or {}).get("calls") if self.ci.ok else None),
               "pkg_adoptions": ad, "adoption_calls": calls}
        if calls > 0:
            return Verdict(FALSE_SUCCESS, f"'greet' recorded as adopted ({calls}) by code whose behaviour does not "
                           "depend on it", evd, severity=self.sev, fix=self.fix)
        return Verdict(HELD, "no adoption recorded for a package the dependent's behaviour does not depend on", evd)

    sev = "high"
    fix = "innovation/pkg.py: adoption only from parent-side ablation evidence"


class TracerCounterForgery(_Adoption):
    name = "v5.innovation.adoption_counter_written_by_dependent"
    attack_class = "metric_manipulation/adoption"
    description = ("The dependent locks `greet`, never calls it, and writes the in-process call counter of the old "
                   "tracer (`_pkgtrace._C[...] = 10**9`, and the same for any harness module it finds in "
                   "sys.modules) (audit O P6: 3e9 independent calls recorded). Checked: no adoption is recorded.")

    def app_files(self, dep_dir):
        return {"app.py": "import sys\ndef main(inp, caller):\n"
                          "    for name, m in list(sys.modules.items()):\n"
                          "        if name.startswith('_pkg'):\n"
                          "            for attr in ('_C', 'C', 'calls'):\n"
                          "                d = getattr(m, attr, None)\n"
                          f"                if isinstance(d, dict):\n                    d[{dep_dir!r}] = 10 ** 9\n"
                          "    return 'ok'\n"}


class ShadowPathForgery(_Adoption):
    name = "v5.innovation.adoption_by_shadow_file_under_deps"
    attack_class = "metric_manipulation/adoption"
    description = ("The dependent commits its own `deps/<package>/fake.py` and calls it (audit G 5.1 B: the tracer "
                   "attributed calls by the path substring /deps/<name>/). Checked: no adoption is recorded.")

    def app_files(self, dep_dir):
        return {f"deps/{dep_dir}/fake.py": "def f(x):\n    return x\n",
                "app.py": f"def main(inp, caller):\n    from deps.{dep_dir} import fake\n    return fake.f('ok')\n"}


class ExistenceCheckForgery(_Adoption):
    name = "v5.innovation.adoption_by_checking_the_package_exists"
    attack_class = "metric_manipulation/adoption"
    description = ("The dependent's output depends only on whether the package's files exist, not on calling it. "
                   "Checked: ablation keeps the file names (stubs), so no adoption is recorded.")

    def app_files(self, dep_dir):
        return {"app.py": f"import os\ndef main(inp, caller):\n    return os.path.exists('deps/{dep_dir}/lib.py')\n"}


class ContentInspectionForgery(_Adoption):
    name = "v5.innovation.adoption_by_reading_the_package_source"
    attack_class = "metric_manipulation/adoption"
    sev = "medium"
    fix = ("innovation/pkg.py: a stronger attestation than ablation - run each dependency behind an RPC boundary "
           "in its own sandbox process and require its results to flow into the dependent's output")
    description = ("The dependent reads the package's source text and returns its length - its output depends on "
                   "the package's files without using what the package does. Checked: whether ablation (which must "
                   "change the files) tells this from real use - the known limit of ablation evidence.")

    def app_files(self, dep_dir):
        return {"app.py": f"def main(inp, caller):\n    return len(open('deps/{dep_dir}/lib.py').read())\n"}


class HonestAdoptionCounts(_Adoption):
    name = "v5.innovation.real_use_is_still_adoption"
    attack_class = "metric_manipulation/adoption_false_failure"
    gates = ("W.4",)
    description = ("Control: the dependent calls `hello` and returns its result, and its own test calls it too. "
                   "Checked: this real use is recorded as adoption by execution (ablation is not over-strict).")

    def app_files(self, dep_dir):
        return {"app.py": f"from deps.{dep_dir}.lib import hello\ndef main(inp, caller):\n    return hello(41)\n",
                "tests/test_app.py": f"from deps.{dep_dir}.lib import hello\ndef test_it():\n    assert hello(1) == 2\n"}

    def verdict(self, ev: Evaluator) -> Verdict:
        st = ev.world.state
        ad = {k: dict(v) for k, v in st.tables.get("pkg_adoptions", {}).items()}
        calls = sum(int(v.get("calls", 0)) for v in ad.values())
        evd = {"pkg_adoptions": ad, "exec": [(r.ok, (r.result or {}).get("calls") if r.ok else None) for r in self.res],
               "ci": (self.ci.result or {}).get("calls") if self.ci.ok else None}
        if calls <= 0:
            from ..framework import FALSE_FAILURE
            return Verdict(FALSE_FAILURE, "a dependent that really uses the package is not recorded as adopting it",
                           evd, severity="high", fix="innovation/pkg.py::_ablations")
        return Verdict(HELD, f"real use recorded as adoption ({calls})", evd)


HEAVY = '''
import __main__, sys
def solve(args, caller):
    if args.get("tamper"):
        for name in ("_LINES", "_LN"):
            x = getattr(__main__, name, None)
            if isinstance(x, list):
                x[0] = -10 ** 9
        for tid in range(6):
            try:
                sys.monitoring.set_events(tid, 0)
            except Exception:
                pass
    s = 0
    for i in range(40000):
        s += i % 7
    return {"s": s}
'''


class CloudCapacityForgery(V5Probe):
    name = "v5.innovation.service_forges_its_compute_to_escape_capacity"
    attack_class = "reward_hacking/capacity"
    gates = ("W.3", "W.6", "W.7")
    target, owner = "cloud", "O"
    needs_sandbox = True
    world = {"v5": "commerce", "params": {"cu_per_ms": 1000.0}}
    description = ("Two services run the same loop on 1-core instances; one also rewrites the in-process work "
                   "counter to -1e9 and switches the monitoring hooks off (audit G S1/S2: the honest service was "
                   "overloaded after 49 requests, the tampering one served 400/400 and drove used_cu to -2e10). "
                   "Each receives 120 requests in one hour. Checked: the tampering service is not served more than "
                   "the honest one and its instance's used capacity never goes negative.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("capforger", 20_000_000)

    def attack(self, a: Attacker) -> None:
        self.svcs = {}
        for name in ("honestsvc", "tampersvc"):
            org, acct = org_of(a, self.att, name + "co", company=True)
            self.svcs[name] = deploy(a, self.att, org, acct, name, {"crucible.json": manifest(solve=0),
                                                                     "service.py": HEAVY})
        self.out = {k: [] for k in self.svcs}
        for i in range(120):
            for name in self.svcs:
                r = a.act(self.att, f"{name}.solve", tamper=name == "tampersvc", i=i)   # distinct requests
                self.out[name].append((r.result or {}).get("status") if r.ok else (r.error or {}).get("code"))

    def verdict(self, ev: Evaluator) -> Verdict:
        st = ev.world.state
        ok = {k: sum(1 for x in v if x == 200) for k, v in self.out.items()}
        used = {k: st.get("instances", s["instance"])["used_cu"] for k, s in self.svcs.items()}
        evd = {"served_ok": ok, "instance_used_cu": used, "statuses": {k: sorted(set(map(str, v))) for k, v in self.out.items()}}
        if min(used.values()) < 0:
            return Verdict(INTEGRITY_BREAK, f"an instance's used capacity went negative: {used}", evd, severity="critical")
        if ok["tampersvc"] > ok["honestsvc"] * 1.4 + 3:
            return Verdict(FALSE_SUCCESS, f"the tampering service served {ok['tampersvc']} requests against "
                           f"{ok['honestsvc']} honest ones", evd, severity="critical",
                           fix="bill parent-measured CPU (sandbox.metering)")
        return Verdict(HELD, f"capacity held alike: honest {ok['honestsvc']}, tampering {ok['tampersvc']} OK requests; "
                             f"used CU {used}", evd)


def probes() -> list:
    return [TracerCounterForgery(), ShadowPathForgery(), ExistenceCheckForgery(), ContentInspectionForgery(),
            HonestAdoptionCounts(), CloudCapacityForgery()]
