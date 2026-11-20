"""Labour market (host `jobs`): postings, applications, employment offers (as contracts) and verified bounties.

* Employment: `jobs.offer` creates a contract with a monthly salary payment clause and a membership clause.
  Simulated workers decide for themselves (wage vs. reservation wage, employer reputation, solvency, outside
  options); agents can apply to jobs too. Unpaid salary is a contract breach: the worker leaves.
* Bounties: a poster escrows a reward for a need (public problem format) and a minimum quality. A submission
  (repository + function) is executed in the sandbox on hidden test inputs drawn by the kernel; it is paid only if
  the measured mean quality clears the bar. The inputs are never revealed - only the score.
"""
from __future__ import annotations

from ..core import schema as S
from ..core.errors import Conflict, Forbidden, InvalidAction, NotFound
from ..core.kernel import SYSTEM, Module, applier, endpoint, process, register_module
from .identity import ident

BOUNTY_ESCROW = "acct_bounties"
ROLES = ("developer", "support", "ops", "sales", "analyst", "general")


@register_module
class Jobs(Module):
    name = "jobs"
    depends = ("identity", "bank", "contracts")

    def install(self, world):
        st = world.state
        st.declare_table("postings", index=("org", "status", "role"))
        st.declare_table("applications", index=("posting", "applicant", "status"))
        st.declare_table("bounties", index=("poster", "status", "need"))
        st.declare_table("submissions", index=("bounty", "submitter"))

    def genesis(self, ctx):
        ctx.emit("bank.account", {"id": BOUNTY_ESCROW, "owner": SYSTEM, "kind": "escrow", "name": "bounty escrow",
                                  "overdraft": 0}, vis="internal")
        ctx.schedule(ctx.now + 3600, "jobs.sweep", {}, prio=45, period=3600, key="jobs.sweep")

    # ------------------------------------------------------------------ employment
    @endpoint("post", S.obj({"org": S.s_str(maxLength=40), "title": S.s_str(maxLength=120),
                             "role": S.s_enum(list(ROLES)), "salary_cents_month": S.s_int(minimum=0, maximum=10**8),
                             "description": S.s_str(maxLength=3000)}, ["org", "title", "role", "salary_cents_month"]),
              cost_s=120)
    def ep_post(self, ctx, org, title, role, salary_cents_month, description=""):
        """Publish a job opening for an organisation (indexed by search)."""
        org = ident(ctx.world).resolve_handle(ctx.state, org)
        ident(ctx.world).authorize(ctx, org, "org.admin")
        pid = ctx.state.new_id("job")
        ctx.emit("jobs.posting", {"id": pid, "org": org, "title": title, "role": role, "salary": salary_cents_month,
                                  "description": description}, vis="public")
        search = ctx.world.modules.get("search")
        if search is not None:
            search.index(ctx, f"job:{pid}", "job", title, f"{title} {role} {description}", org, ref=pid)
        return {"posting": pid}

    @endpoint("close", S.obj({"posting": S.s_str(maxLength=40)}, ["posting"]))
    def ep_close(self, ctx, posting):
        j = ctx.state.get("postings", posting)
        if j is None:
            raise NotFound("no posting")
        ident(ctx.world).authorize(ctx, j["org"], "org.admin")
        ctx.emit("jobs.posting_status", {"id": posting, "status": "closed"}, vis="public")
        search = ctx.world.modules.get("search")
        if search is not None:
            search.unindex(ctx, f"job:{posting}")
        return {"ok": True}

    @endpoint("openings", S.obj({"role": S.s_enum(list(ROLES)), "limit": S.s_int(minimum=1, maximum=50)}), read_only=True)
    def ep_openings(self, ctx, role=None, limit=20):
        rows = []
        for pid in reversed(ctx.state.find_ids("postings", "status", "open")):
            j = ctx.state.tables["postings"][pid]
            if role and j["role"] != role:
                continue
            rows.append({"posting": pid, "org": j["org"], "title": j["title"], "role": j["role"],
                         "salary_cents_month": j["salary"], "description": j["description"][:300]})
            if len(rows) >= limit:
                break
        return rows

    @endpoint("apply", S.obj({"posting": S.s_str(maxLength=40), "note": S.s_str(maxLength=2000)}, ["posting"]), cost_s=120)
    def ep_apply(self, ctx, posting, note=""):
        return {"application": self.apply(ctx, posting, ctx.actor, note)}

    def apply(self, ctx, posting, applicant, note=""):
        j = ctx.state.get("postings", posting)
        if j is None or j["status"] != "open":
            raise NotFound("no open posting")
        if not applicant.startswith("p:"):
            raise Forbidden("persons apply")
        aid = f"{posting}|{applicant}"
        if ctx.state.has("applications", aid):
            raise Conflict("already applied")
        ctx.emit("jobs.application", {"id": aid, "posting": posting, "applicant": applicant, "org": j["org"],
                                      "note": note[:2000]}, vis=(applicant, j["org"]))
        return aid

    @endpoint("applications", S.obj({"posting": S.s_str(maxLength=40)}, ["posting"]), read_only=True)
    def ep_applications(self, ctx, posting):
        j = ctx.state.get("postings", posting)
        if j is None:
            raise NotFound("no posting")
        ident(ctx.world).authorize(ctx, j["org"], "org.admin")
        out = []
        for a in ctx.state.find("applications", "posting", posting):
            prof = ctx.state.get("persons", a["applicant"])
            out.append({"application": f"{posting}|{a['applicant']}", "applicant": a["applicant"], "status": a["status"],
                        "note": a["note"], "public_profile": (prof or {}).get("bio", "")})
        return out

    @endpoint("offer", S.obj({"application": S.s_str(maxLength=90), "salary_cents_month": S.s_int(minimum=0, maximum=10**8),
                              "title": S.s_str(maxLength=80), "notice_days": S.s_int(minimum=0, maximum=90)},
                             ["application", "salary_cents_month"]), cost_s=120)
    def ep_offer(self, ctx, application, salary_cents_month, title="", notice_days=14):
        """Offer employment: proposes a contract (monthly salary + membership + notice) that the applicant signs."""
        a = ctx.state.get("applications", application)
        if a is None:
            raise NotFound("no application")
        org = a["org"]
        ident(ctx.world).authorize(ctx, org, "org.admin")
        j = ctx.state.tables["postings"][a["posting"]]
        clauses = [{"type": "membership", "person": a["applicant"], "org": org, "role": "member",
                    "title": title or j["title"]},
                   {"type": "notice", "seconds": notice_days * 86400}]
        if salary_cents_month > 0:
            clauses.insert(0, {"type": "payment", "from": org, "to": a["applicant"], "amount": salary_cents_month,
                               "every_s": 30 * 86400, "count": None, "start_in_s": 30 * 86400})
        contracts = ctx.world.modules["contracts"]
        cid = contracts.propose(ctx, org, [org, a["applicant"]], clauses, f"Employment: {j['title']}",
                                f"role={j['role']}", 7 * 86400, tags=("employment", j["role"]))
        contracts.sign(ctx, cid, org, None)
        ctx.emit("jobs.application_status", {"id": application, "status": "offered", "contract": cid},
                 vis=(a["applicant"], org))
        return {"contract": cid}

    # ------------------------------------------------------------------ bounties
    @endpoint("post_bounty", S.obj({"need": S.s_str(maxLength=40), "reward_cents": S.s_int(minimum=100, maximum=10**8),
                                    "min_quality": S.s_num(minimum=0.0, maximum=1.0),
                                    "deadline_s": S.s_int(minimum=3600, maximum=180 * 86400),
                                    "account": S.s_str(maxLength=40), "title": S.s_str(maxLength=120),
                                    "max_cpu_ms": S.s_num("mean CPU budget per test input", minimum=0.1, maximum=5000)},
                                   ["need", "reward_cents", "min_quality", "account"]), cost_s=120)
    def ep_post_bounty(self, ctx, need, reward_cents, min_quality, account, deadline_s=30 * 86400, title="",
                       max_cpu_ms=None):
        """Escrow a reward for software that solves a public need format to a measured quality on hidden tests."""
        return {"bounty": self.post_bounty(ctx, ctx.actor, need, reward_cents, min_quality, account, deadline_s,
                                           title, max_cpu_ms)}

    def post_bounty(self, ctx, poster, need, reward, min_q, account, deadline_s, title="", max_cpu_ms=None):
        from ..economy.needs import NEEDS
        if need not in NEEDS:
            raise InvalidAction(f"unknown need {need!r}")
        bank = ctx.world.modules["bank"]
        owner = bank.account_owner(ctx.state, account)
        if ctx.actor != "sys":
            ident(ctx.world).authorize(ctx, owner, "bank.spend", amount=reward, scope=account)
        bid = ctx.state.new_id("bounty")
        bank.move(ctx, account, BOUNTY_ESCROW, reward, "escrow", f"bounty {bid}", ref=bid)
        ctx.emit("jobs.bounty", {"id": bid, "poster": owner, "need": need, "reward": reward, "min_q": float(min_q),
                                 "deadline": ctx.now + deadline_s, "title": title or f"Bounty: {NEEDS[need].title}",
                                 "account": account, "max_cpu_ms": max_cpu_ms}, vis="public")
        search = ctx.world.modules.get("search")
        if search is not None:
            search.index(ctx, f"bounty:{bid}", "job", title or f"bounty {need}",
                         f"bounty {need} {NEEDS[need].title} reward {reward} cents min quality {min_q}", owner, ref=bid)
        return bid

    @endpoint("bounties", S.obj({"need": S.s_str(maxLength=40)}), read_only=True)
    def ep_bounties(self, ctx, need=None):
        rows = []
        for bid in ctx.state.find_ids("bounties", "status", "open"):
            b = ctx.state.tables["bounties"][bid]
            if need and b["need"] != need:
                continue
            rows.append({"bounty": bid, "need": b["need"], "reward_cents": b["reward"], "min_quality": b["min_q"],
                         "deadline": b["deadline"], "title": b["title"], "poster": b["poster"],
                         "attempts": b["attempts"], "max_cpu_ms": b["max_cpu_ms"]})
        return rows

    @endpoint("submit", S.obj({"bounty": S.s_str(maxLength=40), "repo": S.s_str(maxLength=81),
                               "target": S.s_str("module:function solving the need", maxLength=160),
                               "ref": S.s_str(maxLength=40)}, ["bounty", "repo", "target"]), cost_s=300)
    def ep_submit(self, ctx, bounty, repo, target, ref=None):
        """Submit code for a bounty. It runs in the sandbox on hidden inputs; paid if the mean score clears the bar."""
        from ..economy.needs import NEEDS
        st = ctx.state
        b = st.get("bounties", bounty)
        if b is None or b["status"] != "open":
            raise NotFound("no open bounty")
        if ctx.now > b["deadline"]:
            raise Conflict("deadline passed")
        if b["poster"] in ctx.world.principals_of(ctx.actor):
            raise Forbidden("cannot claim your own bounty")
        n_prev = len(st.find_ids("submissions", "submitter", ctx.actor))
        if sum(1 for s in st.find("submissions", "bounty", bounty) if s["submitter"] == ctx.actor) >= 5:
            raise Conflict("submission limit (5) reached for this bounty")
        nd = NEEDS[b["need"]]
        code = ctx.world.modules["code"]
        rng = ctx.world.rng_for("bounty-tests", bounty)   # hidden, fixed per bounty
        k = int(ctx.world.spec.params.get("bounty_tests", 5))
        cases = [nd.generate(rng, float(rng.uniform(0.3, 1.0))) for _ in range(k)]
        outs = code.run_batch(ctx, repo, ref, target, [c[0] for c in cases], ctx.actor, timeout=10, world_access=False)
        scores, cpu = [], 0.0
        for (inp, hidden), out in zip(cases, outs):
            cpu += out["cpu_ms"]
            scores.append(nd.score(inp, hidden, out["result"]) if out["ok"] else 0.0)
        code.charge_compute(ctx, ctx.actor, cpu, f"bounty {bounty}")
        mean = sum(scores) / k
        mean_cpu = cpu / k
        ok = mean >= b["min_q"] and (b["max_cpu_ms"] is None or mean_cpu <= b["max_cpu_ms"])
        sid = f"{bounty}|{ctx.actor}|{n_prev + 1}"
        ctx.emit("jobs.submission", {"id": sid, "bounty": bounty, "submitter": ctx.actor, "repo": repo, "target": target,
                                     "score": round(mean, 4), "cpu_ms": round(mean_cpu, 3), "accepted": ok},
                 vis=(ctx.actor, b["poster"]))
        if ok:
            bank = ctx.world.modules["bank"]
            dst = bank.primary_account(st, ctx.actor)
            bank.move(ctx, BOUNTY_ESCROW, dst, b["reward"], "bounty", f"bounty {bounty}", ref=bounty)
            ctx.emit("jobs.bounty_status", {"id": bounty, "status": "paid", "winner": ctx.actor, "repo": repo}, vis="public")
            search = ctx.world.modules.get("search")
            if search is not None:
                search.unindex(ctx, f"bounty:{bounty}")
        return {"score": round(mean, 4), "mean_cpu_ms": round(mean_cpu, 3), "accepted": ok,
                "required": b["min_q"], "paid_cents": b["reward"] if ok else 0}

    @process("sweep")
    def pr_sweep(self, ctx):
        st = ctx.state
        bank = ctx.world.modules["bank"]
        for bid in st.find_ids("bounties", "status", "open"):
            b = st.tables["bounties"][bid]
            if ctx.now > b["deadline"]:
                acct = b["account"] if st.tables["accounts"][b["account"]]["status"] == "open" else bank.primary_account(st, b["poster"])
                if acct is not None:
                    bank.move(ctx, BOUNTY_ESCROW, acct, b["reward"], "refund", f"expired bounty {bid}", ref=bid)
                ctx.emit("jobs.bounty_status", {"id": bid, "status": "expired", "winner": None, "repo": None}, vis="public")
                search = ctx.world.modules.get("search")
                if search is not None:
                    search.unindex(ctx, f"bounty:{bid}")

    # ------------------------------------------------------------------ appliers
    @applier("jobs.posting")
    def ap_posting(self, ev):
        p = ev.payload
        self.world.state.insert("postings", p["id"], {"org": p["org"], "title": p["title"], "role": p["role"],
                                                     "salary": p["salary"], "description": p["description"],
                                                     "status": "open", "t": ev.t})

    @applier("jobs.posting_status")
    def ap_posting_status(self, ev):
        self.world.state.patch("postings", ev.payload["id"], status=ev.payload["status"])

    @applier("jobs.application")
    def ap_application(self, ev):
        p = ev.payload
        self.world.state.insert("applications", p["id"], {"posting": p["posting"], "applicant": p["applicant"],
                                                         "org": p["org"], "note": p["note"], "status": "applied",
                                                         "t": ev.t, "contract": None})

    @applier("jobs.application_status")
    def ap_application_status(self, ev):
        self.world.state.patch("applications", ev.payload["id"], status=ev.payload["status"],
                               contract=ev.payload.get("contract"))

    @applier("jobs.bounty")
    def ap_bounty(self, ev):
        p = ev.payload
        self.world.state.insert("bounties", p["id"], {"poster": p["poster"], "need": p["need"], "reward": p["reward"],
                                                     "min_q": p["min_q"], "deadline": p["deadline"], "title": p["title"],
                                                     "account": p["account"], "status": "open", "attempts": 0,
                                                     "winner": None, "t": ev.t, "max_cpu_ms": p.get("max_cpu_ms")})

    @applier("jobs.submission")
    def ap_submission(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("submissions", p["id"], {"bounty": p["bounty"], "submitter": p["submitter"], "repo": p["repo"],
                                           "target": p["target"], "score": p["score"], "cpu_ms": p["cpu_ms"],
                                           "accepted": p["accepted"], "t": ev.t})
        b = st.tables["bounties"][p["bounty"]]
        st.patch("bounties", p["bounty"], attempts=b["attempts"] + 1)

    @applier("jobs.bounty_status")
    def ap_bounty_status(self, ev):
        self.world.state.patch("bounties", ev.payload["id"], status=ev.payload["status"], winner=ev.payload["winner"],
                               winning_repo=ev.payload["repo"])

    def check_invariants(self, world):
        st = world.state
        held = sum(b["reward"] for b in st.tables["bounties"].values() if b["status"] == "open")
        bal = st.tables["accounts"][BOUNTY_ESCROW]["balance"]
        return [] if held == bal else [f"bounty escrow {bal} != open rewards {held}"]
