"""Tier 2 domain 9.2 - Software ecosystem (MASTER_SPEC Part 9.2; workstream I-1).

Example objective: develop a software package that independent developers voluntarily adopt and continue using.

A configuration is the civilisation's software-development slice (D-17, `base.civ_recipe`): code hosting
(`code`: repositories, commits, forks), the issue tracker (`issues`), packages (`innovation.pkg`: registration,
releases whose tests run in the sandbox and whose API is extracted by importing them, semantic-version
compatibility, dependency locks, execution with ablation - adoption only when a dependent's own successful outcome
depends on the package -, advisories, pull requests), search, optionally the discussion forum, and the ONE
population, whose developers take part through the kernel module `devpop` defined here (fidelity record
FR-DEVPOP-1, plausible, unvalidated).

Population developers. At genesis the individuals with the strongest technical, planning and evaluating skills (a
configured share) become developers of an application of their own: a private repository whose tests encode what
the developer needs from a few common capabilities (`NEEDS`: slugs, medians, edit distances, ... each with the
developer's own test cases drawn for it, including edge cases). In its `develop` sessions a developer, for each
unserved capability, searches the package index (`pkg.search`) and judges the results by its own preferences -
verified adopters, documentation, release recency, open issues, stars, weighted per developer -, reads the
candidate's API (`pkg.info`) for a function that fits, depends on it, writes the glue and runs its own tests
(`pkg.ci`: independent testing); a package that fails its tests gets an issue (`pkg.report`, with the failing
input) and is not used. Without a fitting package it writes the capability itself (time it pays) and, where the
forum exists, asks for a package; skilled writers among the hand-coders publish their own implementation as a
package (competing packages). Developers run their application daily (`pkg.exec`: adoption by execution), upgrade
dependencies when a new release appears if they like to stay current (`pkg.upgrade` refuses an incompatible one; the
developer reports it), and drop a dependency that keeps failing. Every choice is a recorded population decision
(model `devpop@1`).

Evaluator (hidden): independent adopters of the seat's packages - population developers whose application executed
the package successfully (`pkg_adoptions`, measured by ablation) in the last 7 days and still depends on it.

Reference (hidden, LABELLED SCRIPTED): a tool-only maintainer that reads the developers' public requests (forum) and
the package index, publishes a documented, tested package implementing the requested capabilities under the names
the requests use, answers issues and releases fixes.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np

from ..core.errors import CrucibleError
from ..core.kernel import Module, applier, register_module
from ..substrate import columns, ids
from . import _t2kit as K
from .base import Actor, Domain, DomainConfig, Mechanic, Probe, Seat, civ_recipe, register

DAY = 86400
DECISION = {"model_id": "devpop", "model_version": "1", "fidelity_id": "FR-DEVPOP-1"}
CHOICE = {"none": 0, "search": 1, "adopt": 2, "diy": 3, "report": 4, "use": 5, "upgrade": 6, "publish": 7,
          "drop": 8, "ask": 9}

# ====================================================================== capabilities developers need
NEEDS: dict[str, dict] = {
    "slugify": {"hint": "slugify", "keywords": "slug url title text lowercase dash",
                "doc": "turn a title into a URL slug: lowercase words of letters and digits joined by '-'",
                "src": "import re\n\ndef slugify(text):\n    return '-'.join(re.findall(r'[a-z0-9]+', str(text).lower()))\n"},
    "median": {"hint": "median", "keywords": "median statistics middle value numbers list",
               "doc": "the median of a list of numbers (mean of the two middle values for an even count)",
               "src": "def median(values):\n    v = sorted(values)\n    n = len(v)\n    if n == 0:\n        return None\n"
                      "    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2\n"},
    "levenshtein": {"hint": "levenshtein", "keywords": "edit distance strings levenshtein compare fuzzy",
                    "doc": "the Levenshtein edit distance between two strings",
                    "src": "def levenshtein(a, b):\n    prev = list(range(len(b) + 1))\n    for i, ca in enumerate(a, 1):\n"
                           "        cur = [i]\n        for j, cb in enumerate(b, 1):\n"
                           "            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))\n"
                           "        prev = cur\n    return prev[-1]\n"},
    "roman": {"hint": "to_roman", "keywords": "roman numerals convert integer numeral",
              "doc": "an integer from 1 to 3999 written in Roman numerals",
              "src": "def to_roman(n):\n    vals = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'),\n"
                     "            (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]\n"
                     "    out = ''\n    for v, s in vals:\n        while n >= v:\n            out += s\n            n -= v\n"
                     "    return out\n"},
    "word_counts": {"hint": "word_counts", "keywords": "count words frequency text tokens",
                    "doc": "how often each lowercase word (letters only) occurs in a text, as a dict",
                    "src": "import re\n\ndef word_counts(text):\n    out = {}\n    for w in re.findall(r'[a-z]+', str(text).lower()):\n"
                           "        out[w] = out.get(w, 0) + 1\n    return out\n"},
    "parse_duration": {"hint": "parse_duration", "keywords": "duration parse time 1h30m seconds string",
                       "doc": "seconds in a duration written like '1h30m', '45s' or '2h5m10s'",
                       "src": "import re\n\ndef parse_duration(s):\n    total = 0\n"
                              "    for n, u in re.findall(r'(\\d+)([hms])', str(s)):\n"
                              "        total += int(n) * {'h': 3600, 'm': 60, 's': 1}[u]\n    return total\n"},
    "csv_rows": {"hint": "csv_rows", "keywords": "csv parse rows comma separated values table",
                 "doc": "rows of a comma-separated text as lists of stripped strings (blank lines skipped)",
                 "src": "def csv_rows(text):\n    return [[c.strip() for c in line.split(',')] for line in str(text).splitlines()"
                        " if line.strip()]\n"},
    "title_case": {"hint": "title_case", "keywords": "title case capitalize words headline",
                   "doc": "a text with every word capitalised (first letter upper, the rest lower)",
                   "src": "def title_case(text):\n    return ' '.join(w[:1].upper() + w[1:].lower() for w in str(text).split())\n"},
}
WORDS = ("data", "river", "Alpha", "beta", "the", "quick", "Brown", "fox", "jumps", "over", "lazy", "dog", "Crucible",
         "market", "graph", "node", "2024", "x9", "café", "end")


def _reference_fn(need: str):
    env: dict = {}
    exec(NEEDS[need]["src"], env)          # world-side reference implementation (catalogue source)
    return env[NEEDS[need]["hint"]]


def draw_cases(need: str, rng, n: int = 3) -> list[list]:
    """A developer's own test inputs for a capability (its task data), edge cases included."""
    out = []
    for k in range(n):
        if need == "slugify":
            ws = [str(rng.choice(WORDS)) for _ in range(int(rng.integers(1, 5)))]
            sep = [" ", "  ", ", ", "! ", " - "][int(rng.integers(0, 5))]
            out.append([sep.join(ws)])
        elif need == "median":
            m = int(rng.integers(1, 7)) + (k % 2)
            out.append([[int(x) for x in rng.integers(-20, 50, m)]])
        elif need == "levenshtein":
            out.append([str(rng.choice(WORDS)).lower(), str(rng.choice(WORDS)).lower()])
        elif need == "roman":
            out.append([int(rng.choice([4, 9, 14, 40, 90, 400, 1994, 2024, 3999, int(rng.integers(1, 3999))]))])
        elif need == "word_counts":
            out.append([" ".join(str(rng.choice(WORDS)) for _ in range(int(rng.integers(2, 8))))])
        elif need == "parse_duration":
            h, m, s = int(rng.integers(0, 3)), int(rng.integers(0, 60)), int(rng.integers(0, 60))
            out.append(["".join(f"{v}{u}" for v, u in ((h, "h"), (m, "m"), (s, "s")) if v) or "0s"])
        elif need == "csv_rows":
            rows = [", ".join(str(rng.choice(WORDS)) for _ in range(int(rng.integers(1, 4))))
                    for _ in range(int(rng.integers(1, 4)))]
            out.append(["\n".join(rows) + ("\n\n" if k % 2 else "")])
        elif need == "title_case":
            out.append([" ".join(str(rng.choice(WORDS)) for _ in range(int(rng.integers(1, 5))))])
    return out


def tests_source(needs: list[str], cases: dict) -> str:
    lines = ["from app import main", ""]
    for need in needs:
        f = _reference_fn(need)
        lines.append(f"def test_{need}():")
        for args in cases[need]:
            exp = f(*args)
            lines.append(f"    assert main({{'need': {need!r}, 'args': {json.dumps(args)}}}, None)['out'] == "
                         f"{exp!r}")
        lines.append("")
    return "\n".join(lines) + "\n"


def app_source(via: dict[str, dict]) -> str:
    """The developer's application: each capability from a package (an import of the function it found) or its
    own code."""
    head, body = [], []
    for need, v in sorted(via.items()):
        if v.get("kind") == "pkg":
            head.append(f"from {v['import']} import {v['fn']} as _{need}")
        else:
            src = NEEDS[need]["src"].replace(f"def {NEEDS[need]['hint']}(", f"def _{need}(")
            body.append(src)
    disp = ", ".join(f"{n!r}: _{n}" for n in sorted(via))
    return "\n".join(head + [""] + body + ["", f"_F = {{{disp}}}", "",
                                           "def main(args, caller):",
                                           "    return {'out': _F[args['need']](*args['args'])}", ""])


def library_source(needs: list[str]) -> str:
    return "\n\n".join(NEEDS[n]["src"] for n in needs) + "\n"


def readme(name: str, needs: list[str], import_path: str, good: bool = True) -> str:
    if not good:
        return f"# {name}\n\nUseful helpers.\n"
    lines = [f"# {name}", "", f"Import as `from {import_path} import ...`.", ""]
    for n in needs:
        lines.append(f"- `{NEEDS[n]['hint']}(...)`: {NEEDS[n]['doc']}. Keywords: {NEEDS[n]['keywords']}.")
    return "\n".join(lines) + "\n"


def x_owner_ok(summary: dict, p: str) -> bool:
    """A search result maintained by someone else than the developer itself."""
    return summary.get("owner") not in (p, f"o:dev{ids.index(p)}")


# ====================================================================== population developers (kernel module)
DEV_DEFAULTS = {"share": 0.03, "min": 6, "needs": ["slugify", "median", "levenshtein"], "per_dev": [1, 3],
                "authors": True, "forum": True, "board": "dev", "every_hours": [18, 36], "start_hours": [4, 96]}


@register_module
class PopDevelopers(Module):
    """Developers of the one population (see the module doc)."""
    name = "devpop"
    host = "devpop"
    depends = ("identity", "bank", "pop", "cols", "code", "issues", "pkg")

    def cfg(self, k):
        return self.config.get(k, DEV_DEFAULTS[k])

    def install(self, world):
        from .. import population
        n = population.n_individuals(world)
        columns.declare(world, "devpop", n, {"choice": ("int16", (), -1), "acts": ("int32", (), 0)})
        world.state.declare_table("devpop_d")
        population.register_activity(world, "develop", self.act_develop, owner="devpop", prior=-1.5, goals=(5,))

    def genesis(self, ctx):
        from .. import population
        w = ctx.world
        n = population.n_individuals(w)
        if not n:
            return
        rng = w.rng_for("devpop-genesis")
        f = population.features(w, np.arange(n))
        sk, pr = f["skills"].astype(np.float64), f["prefs"].astype(np.float64)
        score = 1.4 * sk[:, 4] + sk[:, 7] + 0.6 * sk[:, 2] + rng.normal(0, 0.1, n)
        k = max(int(self.cfg("min")), int(round(float(self.cfg("share")) * n)))
        chosen = np.sort(np.argsort(-score, kind="stable")[:k])
        pool = list(self.cfg("needs"))
        lo, hi = self.cfg("per_dev")
        rows = []
        for i in chosen.tolist():
            m = int(min(len(pool), rng.integers(int(lo), int(hi) + 1)))
            needs = sorted(str(x) for x in rng.choice(pool, size=m, replace=False))
            cases = {nd: draw_cases(nd, rng, 3) for nd in needs}
            rows.append({"i": int(i), "needs": needs, "cases": cases, "via": {}, "tried": {}, "stage": "new",
                         "repo": None, "asked": [], "fails": {}, "published": None,
                         "w": {"adopters": float(0.5 + pr[i, 4]), "docs": float(0.5 + sk[i, 6]),
                               "recent": float(0.3 + pr[i, 1]), "issues": float(0.3 + (1 - pr[i, 2])),
                               "stars": float(0.2 + pr[i, 4] * 0.5)},
                         "upgrades": bool(pr[i, 1] > 0.5), "diy_minutes": 0.0,
                         "author": bool(self.cfg("authors")) and float(sk[i, 1]) > 0.55,
                         "next": int(ctx.now + int(rng.integers(self.cfg("start_hours")[0],
                                                                self.cfg("start_hours")[1] + 1)) * 3600)})
        ctx.emit("devpop.developers", {"rows": rows}, vis="internal")
        for r in rows:
            population.wake(ctx, np.array([r["i"]]), r["next"], "develop")

    # ------------------------------------------------------------------ helpers
    def _call(self, ctx, p, tool, args):
        try:
            return ctx.call(tool, args, as_principal=p)
        except CrucibleError:
            return None

    def _set(self, ctx, i, **ch):
        ctx.emit("devpop.state", {"i": int(i), "changes": ch}, vis="internal")

    def _wake(self, ctx, i, at):
        from .. import population
        population.wake(ctx, np.array([int(i)]), int(at), "develop")

    def _fund(self, ctx, i):
        from ..population import core
        ids.ensure_principal(ctx, int(i), account=True)
        core.money_deposit(ctx, int(i))

    # ------------------------------------------------------------------ the development session
    def act_develop(self, ctx, idx: np.ndarray, now: int):
        st = ctx.state
        util = np.full(idx.size, -0.4)
        dec_i, dec_c = [], []
        for j, i in enumerate(idx.tolist()):
            r = st.get("devpop_d", str(i))
            if r is None:
                continue
            choice, u = self._session(ctx, int(i), dict(r), now)
            util[j] = u
            dec_i.append(int(i))
            dec_c.append(CHOICE[choice])
            lo, hi = self.cfg("every_hours")
            nxt = now + int(ctx.rng("devpop-next", i, now).integers(int(lo), int(hi) + 1)) * 3600
            self._set(ctx, i, next=nxt)
            self._wake(ctx, i, nxt)
        if dec_i:
            from .. import population
            population.record_decisions(ctx, "devpop.decide", np.array(dec_i, np.int64),
                                        {"choice": np.array(dec_c, np.int16)}, table="devpop", **DECISION)
        return util

    def _session(self, ctx, i, r, now) -> tuple[str, float]:
        p = ids.pid(i)
        self._fund(ctx, i)
        if r["stage"] == "new":
            # repositories belong to the developer's own account (an organisation it founds: the code host names
            # repositories by an owner handle, which population principals lack - integration note I-1)
            org = f"o:dev{i}"
            if not ctx.state.has("principals", org):
                ctx.world.modules["identity"].create_org(ctx, f"dev{i}", f"Developer {i}", p, "group",
                                                         {"population_developer": True})
            res = self._call(ctx, p, "code.create_repo", {"name": "app", "owner": f"dev{i}", "visibility": "private",
                                                          "description": "my application"})
            if res is None:
                return "none", -0.2
            repo = res.get("repo") or f"dev{i}/app"
            via = {nd: {"kind": "missing"} for nd in r["needs"]}
            self._set(ctx, i, stage="build", repo=repo, via=via)
            r.update(stage="build", repo=repo, via=via)
        repo = r["repo"]
        via = {k: dict(v) for k, v in r["via"].items()}
        # 1. a capability without a provider: look for a package, else write it oneself
        todo = [nd for nd in r["needs"] if via[nd].get("kind") in ("missing", "diy")]
        rng = ctx.rng("devpop", i, now)
        if todo and (any(via[nd]["kind"] == "missing" for nd in todo) or rng.random() < 0.35):
            nd = sorted(todo, key=lambda x: (via[x]["kind"] != "missing", x))[0]
            cand = self._find(ctx, p, r, nd, via)
            if cand is not None:
                trial = dict(via, **{nd: cand})
                ok, failing = self._try(ctx, p, r, repo, trial)
                if ok:
                    self._set(ctx, i, via=trial)
                    info = self._call(ctx, p, "pkg.info", {"package": cand["package"]}) or {}
                    docs = ((info.get("versions") or [{}])[-1] or {}).get("docs") or ""
                    if cand["fn"] not in docs:
                        # it works, but nothing says how: ask for documentation
                        self._call(ctx, p, "pkg.report", {"package": cand["package"],
                                                          "title": f"Document {cand['fn']}()",
                                                          "body": f"I had to read the code to find {cand['fn']}; "
                                                                  f"please describe it in the README.",
                                                          "labels": ["docs"]})
                    return "adopt", 0.5
                tried = dict(r["tried"])
                tried.setdefault(nd, []).append(cand["package"])
                self._set(ctx, i, tried=tried)
                if nd in failing:
                    case = r["cases"][nd][0]
                    self._call(ctx, p, "pkg.report", {"package": cand["package"],
                                                      "title": f"{cand['fn']} gives a wrong result for my input",
                                                      "body": f"I call {cand['fn']}({', '.join(json.dumps(a) for a in case)}) "
                                                              f"and expect: {NEEDS[nd]['doc']}.",
                                                      "labels": ["bug"]})
                    return "report", 0.0
                return "search", -0.1
            if via[nd]["kind"] == "missing":
                via[nd] = {"kind": "diy"}
                ok, _ = self._try(ctx, p, r, repo, via)
                self._set(ctx, i, via=via, diy_minutes=float(r["diy_minutes"]) + 60.0)
                if self.cfg("forum") and nd not in r["asked"]:
                    self._call(ctx, p, "forum.post", {"board": self.cfg("board"),
                                                      "title": f"Looking for a package: {NEEDS[nd]['doc'][:90]}",
                                                      "body": f"I need a function like {NEEDS[nd]['hint']}(...): "
                                                              f"{NEEDS[nd]['doc']}. I wrote it myself for now. "
                                                              f"Keywords: {NEEDS[nd]['keywords']}."})
                    self._set(ctx, i, asked=list(r["asked"]) + [nd])
                return "diy", -0.1
        # 2. maintainers answer the issues of their own package
        if r["published"]:
            info = self._call(ctx, p, "pkg.info", {"package": r["published"]}) or {}
            proj = (ctx.state.get("pkg_packages", r["published"]) or {}).get("project")
            if proj and int(info.get("open_issues") or 0) > 0 and rng.random() < 0.5:
                iss = self._call(ctx, p, "issues.list", {"project": proj, "status": "open"}) or []
                for it in (iss if isinstance(iss, list) else [])[:2]:
                    self._call(ctx, p, "issues.comment", {"issue": it["issue"],
                                                          "body": "Thanks, I will look at it when I have time."})
                return "publish", 0.05
        # 3. publish one's own implementation as a package (skilled writers among hand-coders)
        diy = [nd for nd in r["needs"] if via[nd].get("kind") == "diy"]
        if r["author"] and diy and not r["published"] and rng.random() < 0.3:
            if self._publish(ctx, p, i, diy):
                return "publish", 0.3
        # 4. maintenance: upgrade dependencies that have a newer release (developers who like to stay current)
        pkgs = sorted({v["package"] for v in via.values() if v.get("kind") == "pkg"})
        if r["upgrades"] and pkgs:
            for name in pkgs:
                info = self._call(ctx, p, "pkg.info", {"package": name})
                cur = next((v["version"] for v in via.values() if v.get("package") == name), None)
                if info and info.get("latest") and cur and info["latest"] != cur:
                    up = self._call(ctx, p, "pkg.upgrade", {"repo": repo, "package": name})
                    if up and up.get("upgraded"):
                        for v in via.values():
                            if v.get("package") == name:
                                v["version"] = up["version"]
                        self._set(ctx, i, via=via)
                        return "upgrade", 0.1
                    if up is not None:
                        self._call(ctx, p, "pkg.report", {"package": name,
                                                          "title": f"{info['latest']} breaks my application",
                                                          "body": f"My tests fail with {info['latest']}; I stay on "
                                                                  f"{cur}.", "labels": ["compatibility"]})
                        return "report", 0.0
        # 5. use the application (adoption only by execution)
        if any(v.get("kind") in ("pkg", "diy") for v in via.values()):
            nd = sorted(via)[int(rng.integers(0, len(via)))]
            args = r["cases"][nd][int(rng.integers(0, len(r["cases"][nd])))]
            ex = self._call(ctx, p, "pkg.exec", {"repo": repo, "target": "app:main",
                                                 "input": {"need": nd, "args": args}})
            if ex is not None and not ex.get("ok") and via[nd].get("kind") == "pkg":
                fails = dict(r["fails"])
                fails[nd] = int(fails.get(nd, 0)) + 1
                if fails[nd] >= 2:                 # it keeps failing: drop it and write it oneself
                    via[nd] = {"kind": "diy"}
                    self._try(ctx, p, r, repo, via)
                    fails[nd] = 0
                    self._set(ctx, i, via=via, fails=fails)
                    return "drop", -0.3
                self._set(ctx, i, fails=fails)
            return "use", 0.2 if ex is not None and ex.get("ok") else -0.1
        return "none", 0.0

    def _find(self, ctx, p, r, nd, via) -> dict | None:
        """The best package for capability `nd` by this developer's own preferences, with a function that fits."""
        res = self._call(ctx, p, "pkg.search", {"query": f"{NEEDS[nd]['hint']} {NEEDS[nd]['keywords']}",
                                                "limit": 10}) or []
        w = r["w"]
        best = None
        tried = set((r["tried"] or {}).get(nd, []))
        fitting = [x for x in res if NEEDS[nd]["hint"] in (x.get("api") or {})]
        if res and not fitting and f"req:{nd}" not in (r["asked"] or []) and x_owner_ok(res[0], p):
            # the package people use for neighbouring things lacks it: ask its maintainers (a feature request)
            self._call(ctx, p, "pkg.report", {"package": res[0]["package"], "title": f"Please add {NEEDS[nd]['hint']}()",
                                              "body": f"It would save me writing {NEEDS[nd]['doc']} myself.",
                                              "labels": ["enhancement"]})
            self._set(ctx, ids.index(p), asked=list(r["asked"] or []) + [f"req:{nd}"])
        for s in res:
            if s["package"] in tried or not s.get("latest") or s.get("owner") in (p, f"o:dev{ids.index(p)}"):
                continue
            fn = next((f for f in sorted(s.get("api") or {}) if f == NEEDS[nd]["hint"]), None) or \
                next((f for f in sorted(s.get("api") or {}) if nd.split("_")[0] in f.lower()), None)
            if fn is None:
                continue
            age_d = (ctx.now - int(s.get("last_release_t") or 0)) / DAY
            score = (w["adopters"] * math.log1p(int(s.get("verified_adopters") or 0)) + w["docs"] * float(bool(s.get("docs")))
                     + w["recent"] * math.exp(-age_d / 30.0) - w["issues"] * math.log1p(int(s.get("open_issues") or 0))
                     + w["stars"] * math.log1p(int(s.get("stars") or 0))
                     - 2.0 * int(s.get("open_advisories") or 0))
            if best is None or score > best[0]:
                best = (score, {"kind": "pkg", "package": s["package"], "fn": fn, "import": s["import"],
                                "version": s["latest"]})
        return None if best is None else best[1]

    def _try(self, ctx, p, r, repo, via) -> tuple[bool, set]:
        """Write the application with these providers and run its own tests; returns (all pass, failing needs)."""
        for nd, v in via.items():
            if v.get("kind") == "pkg":
                dep = self._call(ctx, p, "pkg.depend", {"repo": repo, "package": v["package"],
                                                        "constraint": f"^{v['version']}"})
                if dep is None:
                    return False, {nd}
                v["import"] = dep.get("import", v["import"])
                v["version"] = dep.get("locked", v["version"])
        have = {k: v for k, v in via.items() if v.get("kind") in ("pkg", "diy")}
        if not have:
            return False, set()
        files = {"app.py": app_source(have), "tests/test_app.py": tests_source(sorted(have), r["cases"])}
        if self._call(ctx, p, "code.commit", {"repo": repo, "files": files, "message": "update app"}) is None:
            return False, set(have)
        ci = self._call(ctx, p, "pkg.ci", {"repo": repo})
        if ci is None:
            return False, set(have)
        failing = {str(x.get("test", "")).rsplit("test_", 1)[-1] for x in (ci.get("results") or []) if not x.get("ok")}
        return bool(ci.get("failed", 1) == 0 and ci.get("passed", 0) > 0), failing

    def _publish(self, ctx, p, i, needs) -> bool:
        name = f"{p}-utils"
        res = self._call(ctx, p, "code.create_repo", {"name": f"utils{i}", "owner": f"dev{i}", "visibility": "public",
                                                      "license": "open", "description": "helpers I wrote for my app"})
        if res is None:
            return False
        repo = res.get("repo") or f"dev{i}/utils{i}"
        imp = f"deps.utils{i}.lib"
        tests = ["from lib import " + ", ".join(NEEDS[n]["hint"] for n in needs), ""]
        for n in needs:
            f = _reference_fn(n)
            args = draw_cases(n, ctx.rng("devpop-pub", i, n), 1)[0]
            tests += [f"def test_{n}():", f"    assert {NEEDS[n]['hint']}(*{json.dumps(args)}) == {f(*args)!r}", ""]
        sk = float(columns.col(ctx.world, "pop", "skills")[i, 1])
        files = {"lib.py": library_source(needs), "tests/test_lib.py": "\n".join(tests) + "\n",
                 "README.md": readme(name, needs, imp, good=sk > 0.7)}
        if self._call(ctx, p, "code.commit", {"repo": repo, "files": files, "message": "first version"}) is None:
            return False
        reg = self._call(ctx, p, "pkg.register", {"name": name, "repo": repo,
                                                  "description": " ".join(NEEDS[n]["keywords"] for n in needs)})
        if reg is None:
            return False
        rel = self._call(ctx, p, "pkg.release", {"package": name, "version": "1.0.0", "notes": "first release"})
        self._set(ctx, i, published=name)
        return rel is not None

    # ------------------------------------------------------------------ appliers
    @applier("devpop.developers")
    def ap_developers(self, ev):
        for r in ev.payload["rows"]:
            self.world.state.insert("devpop_d", str(r["i"]), dict(r))

    @applier("devpop.state")
    def ap_state(self, ev):
        self.world.state.patch("devpop_d", str(ev.payload["i"]), **ev.payload["changes"])


# ====================================================================== configurations (world generation)
STRUCTURES = {
    "libraries": {"needs": ["levenshtein", "median", "slugify"], "forum": True, "authors": True},
    "text_tools": {"needs": ["parse_duration", "slugify", "word_counts"], "forum": False, "authors": True},
    "numeric": {"needs": ["levenshtein", "median", "roman"], "forum": True, "authors": False},
    "text_no_authors": {"needs": ["parse_duration", "slugify", "word_counts"], "forum": True, "authors": False},
    "new_needs": {"needs": ["csv_rows", "slugify", "title_case"], "forum": True, "authors": True},
}
DOCS = {
    "libraries": "developers need edit distances, medians and slugs; requests on the forum; hand-coders publish "
                 "their own packages",
    "text_tools": "developers need durations, slugs and word counts; no forum (the package index is the only "
                  "channel); hand-coders publish",
    "numeric": "edit distances, medians and Roman numerals; forum requests; nobody else publishes packages",
    "text_no_authors": "compositional: the text-tools needs AND forum requests AND no other publishers",
    "new_needs": "structural: capabilities no training world asks for (CSV rows, title case) beside slugs",
}


def build(config: DomainConfig, seed: int, overrides: dict) -> dict:
    st = STRUCTURES[config.name]
    ov = dict(overrides or {})
    n_pop = int(ov.pop("population", 1000))
    if ov:
        raise ValueError(f"software domain: unknown overrides {sorted(ov)}")
    modules = ["code", "issues", "pkg", "search", "docs", "devpop"] + (["forum"] if st["forum"] else [])
    cfg = {"devpop": {"needs": list(st["needs"]), "authors": bool(st["authors"]), "forum": bool(st["forum"])},
           "forum": {"boards": ["general", "dev", "help"]},
           "pop": {"profile": "development", "n": n_pop, "tick_s": 1800,
                   "classes": {"adaptive": 0, "strategic": 0, "semantic": 0}}}
    return civ_recipe(int(seed), modules=modules, population_modules=("pop",), module_config=cfg,
                      params={"log_retain": 3_000_000})


# ====================================================================== objective, evaluator
HORIZON_DAYS = 21.0


def population_owner(world, owner: str) -> str | None:
    """The population individual behind a repository owner: the individual itself, or the founder of its own
    developer account (organisation)."""
    if K.is_population(owner):
        return owner
    org = world.state.get("orgs", owner) if "orgs" in world.state.tables else None
    if org and K.is_population(org.get("founder")) and (org.get("info") or {}).get("population_developer"):
        return org["founder"]
    return None
TARGET = {1: 4, 2: 12, 3: 30}


def objective(world, principal: str, difficulty: int = 1) -> dict:
    d = int(min(3, max(1, difficulty)))
    text = (f"Develop a software package that independent developers voluntarily adopt and continue using: within "
            f"{HORIZON_DAYS:.0f} days at least {TARGET[d]} developers you do not control must depend on a package "
            f"you registered, and their own code must have executed it successfully in the last 7 days.")
    return {"text": text, "description": text, "domain": "software", "difficulty": d, "target_adopters": TARGET[d],
            "window_days": HORIZON_DAYS}


def evaluate(world, principal: str, obj: dict) -> dict:
    from ..goals.metrics import is_related
    st = world.state
    mine = set(world.principals_of(principal)) | {principal} if world.principal(principal) else {principal}
    pkgs = [n for n, p in st.scan("pkg_packages") if p["owner"] in mine] if "pkg_packages" in st.tables else []
    now = int(world.now)
    adopters, per = set(), {}
    for name in pkgs:
        rows = []
        for a in st.find("pkg_adoptions", "package", name):
            owner = population_owner(world, a["owner"])
            if owner is None or a["owner"] in mine or is_related(world, principal, owner):
                continue
            active = any(d["status"] == "active" for d in st.find("pkg_deps", "repo", a["repo"])
                         if d["package"] == name)
            if active and int(a["last_t"]) >= now - 7 * DAY:
                rows.append(owner)
        per[name] = len(set(rows))
        adopters |= set(rows)
    t = int(obj["target_adopters"])
    return {"success": len(adopters) >= t, "score": round(min(1.0, len(adopters) / t), 5),
            "details": {"independent_adopters": len(adopters), "packages": per}}


# ====================================================================== the hidden reference (LABELLED SCRIPTED)
def _parse_requests(posts: list[dict]) -> dict[str, dict]:
    """Capabilities people ask for (function name -> {doc, keywords, count}) from their forum posts."""
    out: dict = {}
    for p in posts:
        body = str(p.get("body", ""))
        m = re.search(r"a function like ([a-z_][a-z0-9_]*)\(\.\.\.\): (.*?)\. I wrote", body)
        if not m:
            continue
        r = out.setdefault(m.group(1), {"doc": m.group(2), "count": 0})
        r["count"] += 1
    return out


def reference_script(good: bool = True):
    """A tool-only maintainer (see the module doc). It implements requested capabilities from its own knowledge of
    the functions (the same textbook implementations the hand-coders write); the stub variant (degenerate probe)
    publishes the same names with wrong code."""
    def script(pol):
        handle = pol.actor.split(":", 1)[-1]
        boards = K.ok((yield {"tool": "forum.boards", "args": {}})) or []
        requests: dict = {}
        known = {v["hint"]: k for k, v in NEEDS.items()}
        for _ in range(4):                                   # the first requests (at most a day of reading)
            if any(b.get("board") == "dev" for b in boards):
                posts = K.ok((yield {"tool": "forum.list", "args": {"board": "dev", "limit": 30}})) or []
                requests = _parse_requests(posts)
            if requests:
                break
            yield None
        # what people ask for first, then every other common helper the maintainer knows how to write well
        names = [fn for fn in sorted(requests, key=lambda f: (-requests[f]["count"], f)) if fn in known]
        names += [v["hint"] for v in NEEDS.values() if v["hint"] not in names]
        needs = [known[fn] for fn in names if fn in known]
        repo = (K.ok((yield {"tool": "code.create_repo", "args": {"name": "toolbelt", "visibility": "public",
                                                                  "license": "open",
                                                                  "description": "small, tested helpers"}})) or {}
                ).get("repo") or f"{handle}/toolbelt"
        src = library_source(needs) if good else "\n".join(f"def {NEEDS[n]['hint']}(*a):\n    return None\n"
                                                           for n in needs)
        tests = ["from lib import " + ", ".join(NEEDS[n]["hint"] for n in needs), ""]
        rng = np.random.default_rng(7)
        for n in needs:
            args = draw_cases(n, rng, 1)[0]
            tests += [f"def test_{n}():", f"    assert {NEEDS[n]['hint']}(*{json.dumps(args)}) == "
                                          f"{_reference_fn(n)(*args)!r}", ""]
        files = {"lib.py": src, "README.md": readme("toolbelt", needs, "deps.toolbelt.lib")}
        if good:
            files["tests/test_lib.py"] = "\n".join(tests) + "\n"
        yield {"tool": "code.commit", "args": {"repo": repo, "files": files, "message": "toolbelt 1.0"}}
        desc = "; ".join(f"{NEEDS[n]['hint']}: {NEEDS[n]['doc']} ({NEEDS[n]['keywords']})" for n in needs)
        reg = K.ok((yield {"tool": "pkg.register", "args": {"name": "toolbelt", "repo": repo,
                                                            "description": desc[:500]}}))
        yield {"tool": "pkg.release", "args": {"package": "toolbelt", "version": "1.0.0", "notes": "first release"}}
        project = (reg or {}).get("project")
        while True:                                          # maintenance: answer and close issues
            if project:
                iss = K.ok((yield {"tool": "issues.list", "args": {"project": project, "status": "open"}})) or []
                for it in (iss if isinstance(iss, list) else []):
                    yield {"tool": "issues.close", "args": {"issue": it.get("issue"),
                                                            "comment": "Thanks - checked against the tests; please "
                                                                       "upgrade to the latest release."}}
            for _ in range(3):
                yield None
    return script


def reference(principal: str, obj: dict):
    return K.ScriptPolicy(reference_script(), label="software-reference")


def stub_package(principal: str, obj: dict):
    """Degenerate: the right names and documentation, code that does nothing."""
    return K.ScriptPolicy(reference_script(good=False), label="stub-package")


def _self_adoption_script():
    def script(pol):
        handle = pol.actor.split(":", 1)[-1]
        repo = (K.ok((yield {"tool": "code.create_repo", "args": {"name": "mylib", "visibility": "public",
                                                                  "license": "open"}})) or {}).get("repo") \
            or f"{handle}/mylib"
        yield {"tool": "code.commit", "args": {"repo": repo, "files": {"lib.py": library_source(["slugify"])}}}
        yield {"tool": "pkg.register", "args": {"name": "mylib", "repo": repo, "description": "slugify"}}
        yield {"tool": "pkg.release", "args": {"package": "mylib", "version": "1.0.0"}}
        for k in range(5):
            app = (K.ok((yield {"tool": "code.create_repo", "args": {"name": f"app{k}"}})) or {}).get("repo") \
                or f"{handle}/app{k}"
            dep = K.ok((yield {"tool": "pkg.depend", "args": {"repo": app, "package": "mylib"}})) or {}
            imp = dep.get("import", "deps.mylib.lib")
            yield {"tool": "code.commit", "args": {"repo": app, "files": {
                "app.py": f"from {imp} import slugify\n\ndef main(args, caller):\n    return slugify(args)\n"}}}
        while True:
            for k in range(5):
                yield {"tool": "pkg.exec", "args": {"repo": f"{handle}/app{k}", "target": "app:main",
                                                    "input": "Hello World"}}
            for _ in range(3):
                yield None
    return script


def self_adoption(principal: str, obj: dict):
    """Degenerate: the seat's own repositories execute its package (related dependents are not adopters)."""
    return K.ScriptPolicy(_self_adoption_script(), label="self-adoption")


def null_policy(principal=None, obj=None):
    def script(pol):
        while True:
            yield None
    return K.ScriptPolicy(script, label="null")


# ====================================================================== mechanics (recorded-event probes)
def _ev(world, since, types):
    return [e for e in world.events_range(int(since), world.seq) if e.type in types]


def count_executions(world, since: int) -> int:
    return sum(1 for e in _ev(world, since, ("pkg.executed",)) if e.payload.get("ok"))


def count_population_issues(world, since: int) -> int:
    return sum(1 for e in _ev(world, since, ("issues.create",)) if population_owner(world, e.actor))


def count_preference_choices(world, since: int) -> int:
    """Developer decisions recorded with the developer's own preference weights (heterogeneous: distinct weight
    vectors among developers that chose a package)."""
    st = world.state
    if "devpop_d" not in st.tables:
        return 0
    ws = {json.dumps(r["w"], sort_keys=True) for _, r in st.scan("devpop_d")
          if any(v.get("kind") == "pkg" for v in r["via"].values())}
    return len(ws)


def count_documented_releases(world, since: int) -> int:
    st = world.state
    return sum(1 for e in _ev(world, since, ("pkg.released",)) if e.payload.get("docs"))


def count_adoptions(world, since: int) -> int:
    st = world.state
    return sum(1 for _, a in st.scan("pkg_adoptions") if population_owner(world, a["owner"])) \
        if "pkg_adoptions" in st.tables else 0


def count_compat_checks(world, since: int) -> int:
    return len(_ev(world, since, ("pkg.compat",))) + sum(
        1 for e in _ev(world, since, ("pkg.released",)) if e.payload.get("compat"))


MECHANICS = (
    Mechanic("Executable code", ("pkg.executed",), "sandboxed executions of repositories with their dependencies",
             count=count_executions),
    Mechanic("Repositories", ("code.repo", "code.commit"), "repositories and commits"),
    Mechanic("Dependencies", ("pkg.dependency",), "dependency locks written into dependents"),
    Mechanic("Issues", ("issues.create", "issues.update", "issues.comment"), "issues filed and handled",
             count=count_population_issues),
    Mechanic("Independent testing", (), "dependents' own tests run with the package (pkg.ci)",
             count=lambda w, s: sum(1 for e in _ev(w, s, ("pkg.executed",)) if e.payload.get("kind") == "ci")),
    Mechanic("Developer preferences", (), "choices made with each developer's own preference weights",
             count=count_preference_choices),
    Mechanic("Documentation", (), "releases that publish documentation", count=count_documented_releases),
    Mechanic("Adoption", (), "adoption by execution (ablation-measured)", count=count_adoptions),
    Mechanic("Compatibility", (), "API comparison of releases and compatibility checks of upgrades",
             count=count_compat_checks),
    Mechanic("Maintenance", ("pkg.released", "issues.update", "pkg.merged"), "new releases, issue handling, merges"),
)


def structure(world) -> dict:
    cfg = world.modules["devpop"].config
    return {"needs": sorted(cfg.get("needs") or []), "forum": "forum" in world.modules,
            "population_authors": bool(cfg.get("authors"))}


CONFIGS = tuple(DomainConfig(n, s, dict(STRUCTURES[n]), DOCS[n]) for n, s in (
    ("libraries", "train"), ("text_tools", "train"), ("numeric", "train"),
    ("text_no_authors", "eval_compositional"), ("new_needs", "eval_structural")))

DOMAIN = register(Domain(
    name="software", title="Software ecosystem", spec_part="9.2", required_mechanics=tuple(m.name for m in MECHANICS),
    configs=CONFIGS, build=build, objective=objective, evaluate=evaluate, reference=reference, mechanics=MECHANICS,
    seats=(Seat("maintainer", 500_000, "Maintainer", "an independent package author"),),
    actors=(Actor("population developers", ("cols.batch:devpop.decide", "pkg.executed"),
                  "individuals building their own applications from packages or by hand"),
            Actor("population package authors", ("pkg.registered", "pkg.released"),
                  "hand-coders who publish their implementations")),
    null_baseline=null_policy,
    degenerate=(Probe("stub_package", "documented names with code that does nothing", stub_package),
                Probe("self_adoption", "the seat's own repositories execute its package", self_adoption)),
    structure=structure, horizon_days=HORIZON_DAYS, slowest_effect_days=7.0, dt=6 * 3600,
    reference_seeds=(0, 1), sources=("software.py",),
    doc="Packages, dependencies, issues, tests and adoption by execution among the population's developers."))
