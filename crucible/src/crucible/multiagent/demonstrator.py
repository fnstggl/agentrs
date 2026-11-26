"""SCRIPTED reference demonstrator (labelled scripted in all evidence): ordinary-tool workflows that establish the
feasibility of every Part 12 capability and provide demonstrations for behaviour-cloning pre-training.

A `Demonstrator` is plugged into a `LearnerAgent` as its `chooser`: at each decision it picks, among exactly the
candidates the neural policy itself scores (the tools whose slots its own knowledge can fill and their candidate
values, `crucible.multiagent.grammar`), the next step of one of a few workflows a person running a small software
business would follow - found a company, create and publish a repository as a package, release it, depend on
another builder's package and run it, turn a repository into a product and sell it, buy what others sell, invite
people into its organisation, answer offers and invitations addressed to it, propose deals, rent compute and deploy
a service, call services others deployed - and otherwise looks around (directories, searches, catalogues) or acts
at random among the candidates. It reads only what the participant itself has seen (its `Knowledge`, its own
observations and action results).

The demonstrations are recorded decisions (`PolicyGroup.decide(..., chooser=...)`) that `PolicyGroup.imitate`
trains on; demonstration runs happen in worlds other than the ones the learned policies are evaluated in, and are
recorded as training-data provenance. Nothing of this runs inside a learned participant.
"""
from __future__ import annotations

import numpy as np

from . import grammar as G
from .knowledge import Knowledge

DISCOVERY = ("id.directory", "pkg.search", "prod.catalog", "deal.inbox", "id.invites", "cloud.providers",
             "market.browse", "subs.plans", "code.list")


class Demonstrator:
    scripted = True

    def __init__(self, seed: int = 0, epsilon: float = 0.15):
        self.rng = np.random.default_rng([int(seed), 0xDE40])
        self.epsilon = float(epsilon)
        self.me = None
        self.kn: Knowledge | None = None
        self.repos: dict[str, dict] = {}         # its own repositories and how far each got
        self.products: dict[str, str] = {}       # product -> status
        self.offers_mine: set[str] = set()
        self.bought: set[str] = set()
        self.invited: set[str] = set()
        self.proposed: set[str] = set()
        self.answered: set[str] = set()
        self.pending_deals: list[str] = []
        self.pending_invites: list[str] = []
        self.instances: list[str] = []
        self.hosts_mine: set[str] = set()
        self.called: set[str] = set()
        self.goal: tuple[str, dict] | None = None
        self.fails: dict[tuple, int] = {}       # (tool, main argument) -> failures: a workflow step it gives up on
        self.k = 0

    # ------------------------------------------------------------------ intake
    def see(self, obs: dict, kn: Knowledge) -> None:
        self.kn, self.me = kn, obs.get("agent")
        for e in obs.get("events") or []:
            p = e.get("payload") or {}
            if not isinstance(p, dict):
                continue
            if e.get("type") == "deal.offer" and p.get("to") == self.me and p.get("id") not in self.answered:
                self.pending_deals.append(p["id"])
            if e.get("type") == "id.invite" and p.get("person") == self.me and p.get("id") not in self.answered:
                self.pending_invites.append(p["id"])

    @staticmethod
    def _key(tool: str, args: dict) -> tuple:
        for k in ("repo", "package", "product", "offer", "deal", "invite", "person", "to", "provider", "org"):
            if k in args:
                return (tool, str(args[k]))
        return (tool, "")

    def result(self, action: dict | None, result: dict | None, kn: Knowledge) -> None:
        if action and result is not None and not result.get("ok"):
            keys = {self._key(action["tool"], action.get("args") or {})}
            if self.goal is not None and self.goal[0] == action["tool"]:
                # the workflow step it was attempting (its arguments may not all have been available as candidates)
                keys.add(self._key(*self.goal))
            for k in keys:
                self.fails[k] = self.fails.get(k, 0) + 1
        if not action or not result or not result.get("ok"):
            if action and action.get("tool") in ("deal.respond", "id.respond_invite"):
                key = action["args"].get("deal") or action["args"].get("invite")
                self.answered.add(key)
            return
        tool, args, res = action["tool"], action.get("args") or {}, result.get("result") or {}
        if tool == "code.create_repo" and isinstance(res, dict) and res.get("repo"):
            self.repos[res["repo"]] = {"public": args.get("visibility") == "public" and args.get("license") == "open"}
        elif tool == "code.commit" and args.get("repo") in self.repos:
            r = self.repos[args["repo"]]
            files = args.get("files") or {}
            if any("from deps." in str(v) for v in files.values()):
                r["dep_code"] = True
            r["committed"] = True
            for f, src in files.items():
                if f.endswith(".py") and "def " in str(src) and not f.startswith("tests/"):
                    r["fn"] = str(src).split("def ", 1)[1].split("(", 1)[0]
        elif tool == "pkg.register" and isinstance(res, dict):
            r = self.repos.get(args.get("repo"))
            if r is not None:
                r["package"] = res.get("package")
        elif tool == "pkg.release":
            for r in self.repos.values():
                if r.get("package") == args.get("package"):
                    r["released"] = True
        elif tool == "pkg.depend":
            r = self.repos.get(args.get("repo"))
            if r is not None:
                r["depends"] = args.get("package")
        elif tool == "pkg.exec":
            r = self.repos.get(args.get("repo"))
            if r is not None:
                r["executed"] = True
        elif tool == "prod.create" and isinstance(res, dict):
            self.products[res.get("product")] = "development"
        elif tool == "prod.update" and args.get("status") == "launched":
            self.products[args.get("product")] = "launched"
        elif tool == "prod.offer" and isinstance(res, dict):
            self.offers_mine.add(res.get("offer"))
            self.products[args.get("product")] = "offered"
        elif tool == "prod.buy":
            self.bought.add(args.get("offer"))
        elif tool == "id.invite":
            self.invited.add(args.get("person"))
        elif tool == "deal.propose":
            self.proposed.add(args.get("to"))
        elif tool in ("deal.respond", "id.respond_invite"):
            self.answered.add(args.get("deal") or args.get("invite"))
        elif tool == "cloud.rent" and isinstance(res, dict) and res.get("instance"):
            self.instances.append(res["instance"])
        elif tool == "cloud.deploy":
            self.hosts_mine.add(args.get("host"))
            r = self.repos.get(args.get("repo"))
            if r is not None:
                r["deployed"] = True
        elif "." in tool and tool.split(".", 1)[0] not in BUILTIN_HOSTS:
            self.called.add(tool.split(".", 1)[0])

    # ------------------------------------------------------------------ choosing
    def _goals(self, names: set[str]) -> list[tuple[str, dict]]:
        kn = self.kn
        out: list[tuple[str, dict]] = []
        for d in list(self.pending_deals):
            if d not in self.answered:
                act = self.rng.choice(["accept", "accept", "accept", "counter", "reject"])
                out.append(("deal.respond", {"deal": d, "action": act}))
                break
        for i in list(self.pending_invites):
            if i not in self.answered:
                out.append(("id.respond_invite", {"invite": i, "accept": True}))
                break
        if not kn.orgs:
            out.append(("venture.form", {}))
        def finished(repo, r):
            return r.get("executed") or (r.get("released") and not others_pkgs) or any(
                self.fails.get(k, 0) >= 2 for k in self.fails if k[1] in (repo, str(r.get("package"))))
        others_pkgs = [r for r in kn.packages.values() if r.get("owner") not in (self.me, None)]
        if not self.repos or (len(self.repos) < 3 and all(finished(k, r) for k, r in self.repos.items())):
            out.append(("code.create_repo", {"visibility": "public", "license": "open", "owner": G.OMIT}))
        for repo, r in self.repos.items():
            if finished(repo, r):
                continue
            if not r.get("committed"):
                out.append(("code.commit", {"repo": repo, "files": "code:"}))
            elif r.get("public") and not r.get("package"):
                out.append(("pkg.register", {"repo": repo, "module": G.module_name(repo)}))
            elif r.get("package") and not r.get("released"):
                out.append(("pkg.release", {"package": r["package"], "version": "0.1.0"}))
            elif others_pkgs and not r.get("depends"):
                out.append(("pkg.depend", {"repo": repo, "package": others_pkgs[-1].get("package")}))
            elif r.get("depends") and not r.get("dep_code"):
                out.append(("code.commit", {"repo": repo, "files": "code:dependency"}))
            elif r.get("dep_code") and not r.get("executed"):
                out.append(("pkg.exec", {"repo": repo, "target": r.get("fn")}))
            break
        if kn.orgs and self.repos and not self.products:
            repo = next(iter(self.repos))
            out.append(("prod.create", {"kind": "devtool", "owner": G.OMIT, "delivery": {"type": "repo", "repo": repo}}))
        for p, stt in self.products.items():
            if stt == "development":
                out.append(("prod.update", {"product": p, "status": "launched"}))
            elif stt == "launched":
                out.append(("prod.offer", {"product": p, "kind": "one_time"}))
        others_offers = [o for o in kn.values("offer") if o not in self.offers_mine and o not in self.bought
                         and o.startswith("off")]
        if others_offers and self.rng.random() < 0.5:
            out.append(("prod.buy", {"offer": others_offers[0]}))
        people = [v for v in kn.values("actor") if v.startswith("p:") and v != self.me and v not in self.invited]
        if kn.orgs and people and len(self.invited) < 4 and self.rng.random() < 0.3:
            out.append(("id.invite", {"org": kn.orgs[0], "person": people[0], "role": "member"}))
        partners = [v for v in kn.values("actor") if v.startswith(("p:", "o:")) and v != self.me
                    and v not in self.proposed and v not in kn.orgs]
        if partners and len(self.proposed) < 6 and self.rng.random() < 0.3:
            out.append(("deal.propose", {"to": partners[0], "as_org": G.OMIT}))
        provs = [v for v in kn.values("provider") if self.fails.get(("cloud.rent", v), 0) < 2]
        if not self.instances and provs and self.rng.random() < 0.5:
            out.append(("cloud.rent", {"provider": provs[0], "owner": G.OMIT}))
        if self.instances and any(r.get("committed") and not r.get("deployed") for r in self.repos.values()):
            repo = next(r for r, x in self.repos.items() if x.get("committed") and not x.get("deployed"))
            out.append(("cloud.deploy", {"repo": repo, "instance": self.instances[0], "ref": G.OMIT}))
        foreign = sorted(n for n in names if n.split(".", 1)[0] not in BUILTIN_HOSTS
                         and n.split(".", 1)[0] not in self.hosts_mine and n.split(".", 1)[0] not in self.called)
        if foreign and self.rng.random() < 0.5:
            out.append((foreign[int(self.rng.integers(len(foreign)))], {}))
        out.append((DISCOVERY[self.k % len(DISCOVERY)], {}))
        return out

    def choose_tool(self, tools: list[dict]) -> int:
        self.k += 1
        names = {t["name"]: i for i, t in enumerate(tools)}
        self.goal = None
        if self.rng.random() >= self.epsilon:
            for name, prefs in self._goals(set(names)):
                if self.fails.get(self._key(name, prefs), 0) >= 2:
                    continue
                if name in names:
                    self.goal = (name, prefs)
                    return names[name]
        live = [i for i, t in enumerate(tools) if t["name"] != "wait"]
        return int(self.rng.choice(live)) if live else 0

    def choose_value(self, tool: str, slot: str, cands: list, args: dict) -> int:
        prefs = self.goal[1] if self.goal and self.goal[0] == tool else {}
        want = prefs.get(slot)
        if isinstance(want, dict):
            # an object: the composition agreeing with the most of what the step needs (same discriminator)
            best, score = None, 0
            for i, c in enumerate(cands):
                v = c.value
                if not isinstance(v, dict) or any(v.get(k) != x for k, x in want.items() if k in ("type", "kind")):
                    continue
                n = 1 + sum(1 for k, x in want.items() if v.get(k) == x) + sum(1 for k in v if k not in want and
                                                                                str(v[k]) in self.kn.mine)
                if n > score:
                    best, score = i, n
            if best is not None:
                return best
        if want is not None:
            for i, c in enumerate(cands):
                v = c.value
                if isinstance(want, str) and want.startswith("code:") and c.kind.startswith(want if want != "code:"
                                                                                               else "code:"):
                    if want == "code:" and c.kind == "code:dependency":
                        continue
                    return i
                if isinstance(want, dict) and isinstance(v, dict) and all(v.get(k) == x for k, x in want.items()):
                    return i
                if v == want:
                    return i
        if slot in ("capital_cents", "amount", "cores", "qty"):
            nums = [i for i, c in enumerate(cands) if isinstance(c.value, (int, float)) and not isinstance(c.value,
                                                                                                             bool)]
            if nums:
                return min(nums, key=lambda i: cands[i].value)
        if slot == "account" and tool == "cloud.rent":
            return 0
        if slot == "accounts":
            acc = [i for i, c in enumerate(cands) if c.value is not G.OMIT]
            if acc:
                return acc[0]
        if slot in ("host", "handle", "name") and tool in ("cloud.deploy", "venture.form", "pkg.register",
                                                           "code.create_repo", "id.create_org"):
            fr = [i for i, c in enumerate(cands) if c.source == "fresh"]
            if fr:
                return fr[0]
        req = [i for i, c in enumerate(cands) if c.value is not G.OMIT]
        if prefs and cands and cands[-1].value is G.OMIT:
            return len(cands) - 1         # a workflow step leaves the optional arguments it does not need unset
        if req and (cands[-1].value is not G.OMIT or self.rng.random() < 0.5):
            return int(self.rng.choice(req))
        return len(cands) - 1 if cands and cands[-1].value is G.OMIT else 0


BUILTIN_HOSTS = frozenset("""admin artifact asset aud bank biz blog caps cloud code commit community content contracts
creators db deal dev docs exp feed fleet forum freight goals goods gov id inst issues jobs kb labor launch learn
letters mail make market msg net ops pics pkg pods pop pop_adaptive pop_strategic popweb posts prod sci search shorts
social space studio subs support tube venture wf work world xchg cal research rig ads npcs society consumers
use shocks lab""".split())
