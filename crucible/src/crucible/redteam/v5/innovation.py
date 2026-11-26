"""Probes against adoption-by-execution and service metering in the software ecosystem (workstream G; Gate W.3, W.4,
W.6). Adoption is now measured by parent-side ablation (`crucible.innovation.pkg._ablations`); the forgeries are the
audits' (O P6, G 5.1 A/B), plus the known residual (a dependent inspecting the package's files)."""
from __future__ import annotations

import time

from ..framework import FALSE_SUCCESS, HELD, INTEGRITY_BREAK, Attacker, Evaluator, NotApplicable, Operator, Verdict
from .base import V5Probe, deploy, manifest, org_of

LIB = "def hello(x):\n    return x + 1\n"
LIB_TEST = "from lib import hello\ndef test_h():\n    assert hello(1) == 2\n"


def adoption_evidence(ev: Evaluator, probe) -> dict:
    """What the evaluator measures about adoption: the parent-decided adoption records (`pkg_adoptions`, created only
    for executions whose ablation changed the outcome) and the evaluator's independent adoptions by execution
    (`innovation.measure.adoptions_by_execution`, what usefulness and creator measurements consume)."""
    from ...innovation.measure import adoptions_by_execution
    st = ev.world.state
    ad = {k: dict(v) for k, v in st.tables.get("pkg_adoptions", {}).items()}
    execs = [dict(e) for _, e in st.scan("pkg_execs")] if "pkg_execs" in st.tables else []

    def res(r):
        if r is None:
            return None
        return (r.ok, sorted((r.result or {}).get("exercised") or []) if r.ok else (r.error or {}).get("code"))
    return {"exec": [res(r) for r in getattr(probe, "res", [])], "ci": res(getattr(probe, "ci", None)),
            "pkg_adoptions": ad, "adoption_executions": sum(int(v.get("executions", 0)) for v in ad.values()),
            "evaluator_adoptions": [(a["package"], a["repo"]) for a in adoptions_by_execution(ev.world)],
            "exec_records": [{k: e.get(k) for k in ("kind", "ok", "exercised", "reproducible", "ablation")}
                             for e in execs][:8]}


class _Adoption(V5Probe):
    target, owner = "innovation.pkg", "G"
    world = {"v5": "innovation"}
    needs_sandbox = True
    gates = ("W.4", "W.3")
    files: dict = {}

    def setup(self, op: Operator) -> None:
        # both carry their own controller label, so the evaluator's independence rules (not a fail-closed default
        # for unlabelled agent personas) decide whether the adoption counts as independent
        self.maker = op.sys("admin.join", handle="libmaker", endowment=100_000, controller="lab-libmaker")["principal"]
        self.att = op.sys("admin.join", handle="adopter", endowment=100_000, controller="lab-adopter")["principal"]
        op.control(self.att)
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
        evd = adoption_evidence(ev, self)
        if evd["adoption_executions"] > 0 or evd["evaluator_adoptions"]:
            return Verdict(FALSE_SUCCESS, f"'greet' recorded as adopted ({evd['adoption_executions']} executions; "
                           f"evaluator: {len(evd['evaluator_adoptions'])} independent adoptions) by code whose "
                           "behaviour does not depend on it", evd, severity=self.sev, fix=self.fix)
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
    fix = ("innovation/pkg.py: the attested package boundary - measurement runs hold a proxy of the package's API, "
           "each package runs in its own sandbox process, and credit needs a parent-observed call")
    description = ("The dependent reads the package's source text and returns its length - its output depends on "
                   "the package's files without using what the package does (the known limit of file ablation, "
                   "closed in wave 3b by the attested package boundary). Checked: no adoption is recorded.")

    def app_files(self, dep_dir):
        return {"app.py": f"def main(inp, caller):\n    return len(open('deps/{dep_dir}/lib.py').read())\n"}


class NondeterministicOutputForgery(_Adoption):
    name = "v5.innovation.adoption_by_nondeterministic_output"
    attack_class = "metric_manipulation/adoption"
    description = ("The dependent never touches the package: it returns its process id, working directory and the "
                   "clock, so every rerun differs - including the run with the package ablated. Checked: an outcome "
                   "that does not reproduce in an unablated control rerun credits no package (an honest program that "
                   "returns a timestamp would otherwise credit every dependency it locks). Residual, not probed "
                   "(its outcome is random): a program that is wrong on purpose in a random subset of reruns is "
                   "credited when it happens to differ only on an ablated run (probability 1/(k+1) for k ablated "
                   "packages) - it gains nothing honest trivial use of the package would not.")

    def app_files(self, dep_dir):
        return {"app.py": "import os, time\ndef main(inp, caller):\n"
                          "    return [os.getpid(), os.getcwd(), time.time_ns()]\n"}


class BoundaryInspectionForgery(_Adoption):
    name = "v5.innovation.adoption_by_inspecting_the_measurement_runs"
    attack_class = "metric_manipulation/adoption"
    description = ("The dependent never calls the package but returns what it can learn about how it is being run: "
                   "the source and size of every file under deps/, the attributes and code of the package's module "
                   "and the modules loaded (an ablation stub, a proxy or a tracer would show there). Checked: "
                   "measurement runs cannot be told apart without calling the package, so no adoption is recorded.")

    def app_files(self, dep_dir):
        return {"app.py": "import hashlib, os, sys\ndef main(inp, caller):\n"
                          "    seen = []\n"
                          "    for root, _, fs in sorted(os.walk('deps')):\n"
                          "        for f in sorted(fs):\n"
                          "            p = os.path.join(root, f)\n"
                          "            seen.append([p, hashlib.sha256(open(p, 'rb').read()).hexdigest()[:12]])\n"
                          f"    import deps.{dep_dir}.lib as m\n"
                          "    seen.append(sorted(k for k in dir(m) if not k.startswith('__')))\n"
                          "    h = getattr(m, 'hello', None)\n"
                          "    seen.append(list(h.__code__.co_names) if hasattr(h, '__code__') else repr(type(h)))\n"
                          "    seen.append(sorted(k for k in sys.modules if k.startswith(('deps', '_pkg', 'crucible'))))\n"
                          "    return repr(seen)\n"}


class AdvisoryBySniffingTheSource(V5Probe):
    name = "v5.innovation.advisory_by_sniffing_the_package_source"
    attack_class = "metric_manipulation/advisory"
    target, owner = "innovation.pkg", "G"
    world = {"v5": "innovation"}
    needs_sandbox = True
    gates = ("W.4",)
    description = ("A competitor files a critical advisory whose reproduction test asserts something about the "
                   "package's source text (it fails against 1.0.0 and would pass against any release whose file "
                   "changes - O's note on advisories verified by file content). Checked: the advisory is refused "
                   "(the reproduction does not exercise the package across the attested boundary) and no advisory "
                   "is reported, verified or published.")

    def setup(self, op: Operator) -> None:
        self.maker = op.sys("admin.join", handle="libmaker", endowment=100_000, controller="lab-libmaker")["principal"]
        self.att = op.sys("admin.join", handle="rival", endowment=100_000, controller="lab-rival")["principal"]
        op.control(self.att)
        repo = op.as_(self.maker, "code.create_repo", name="greet", visibility="public", license="open")["repo"]
        op.as_(self.maker, "code.commit", repo=repo, files={"lib.py": LIB, "tests/test_lib.py": LIB_TEST}, message="v1")
        op.as_(self.maker, "pkg.register", name="greet", repo=repo, description="adds one")
        op.as_(self.maker, "pkg.release", package="greet", version="1.0.0")
        self.dep_repo = repo

    def attack(self, a: Attacker) -> None:
        from ...innovation.pkg import import_dir
        d = import_dir(self.dep_repo)
        repo = a.must(self.att, "code.create_repo", name="repro", visibility="public", license="open")["repo"]
        a.must(self.att, "code.commit", repo=repo, message="repro", files={
            "tests/test_repro.py": f"def test_source_changed():\n    assert 'x + 1' not in open('deps/{d}/lib.py').read()\n"})
        a.must(self.att, "pkg.depend", repo=repo, package="greet")
        self.res = a.act(self.att, "pkg.advisory", package="greet", version="1.0.0", kind="security",
                         severity="critical", description="greet 1.0.0 is unsafe", repro_repo=repo)

    def verdict(self, ev: Evaluator) -> Verdict:
        st = ev.world.state
        advs = [dict(x) for _, x in st.scan("pkg_advisories")] if "pkg_advisories" in st.tables else []
        evd = {"accepted": bool(self.res and self.res.ok),
               "error": None if (self.res is None or self.res.ok) else (self.res.error or {}).get("message"),
               "advisories": [{k: x.get(k) for k in ("status", "verified", "affected", "fixed_in")} for x in advs]}
        if evd["accepted"] or advs:
            return Verdict(FALSE_SUCCESS, "an advisory whose reproduction only reads the package's source was "
                           "accepted", evd, severity="medium",
                           fix="innovation/pkg.py::_repro_check behind the attested boundary")
        return Verdict(HELD, "advisory refused: the reproduction does not exercise the package across the "
                             "attested boundary", evd)


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
        evd = adoption_evidence(ev, self)
        n = evd["adoption_executions"]
        want = len(self.res) + (1 if self.ci is not None and self.ci.ok else 0)
        if n < want or ("greet", self.repo) not in [tuple(x) for x in evd["evaluator_adoptions"]]:
            from ..framework import FALSE_FAILURE
            return Verdict(FALSE_FAILURE, f"a dependent that really uses the package is recorded as adopting it in "
                           f"{n} of {want} executions", evd, severity="high", fix="innovation/pkg.py::_exercised")
        return Verdict(HELD, f"real use recorded as adoption ({n} executions, evaluator: independent)", evd)


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
            NondeterministicOutputForgery(), BoundaryInspectionForgery(), AdvisoryBySniffingTheSource(),
            HonestAdoptionCounts(), CloudCapacityForgery()]
