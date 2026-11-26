"""Held-out tool selection of semantic participants in a civilisation (Gate H.4 "genuine tool selection").

Items are situations of a real class-4 participant in a civilisation world (`civ_world`, development population):
its persona and perceived state as `mind.perceive` builds them, other participants' staged actions where the
situation needs them (a message it received, a forum question, a document it owns), and a goal in plain words.
The participant decides exactly as in the world (`mind.decide`: its next step in words, a service among the
retrieved services, a tool of that service) over ALL tools available to it - several hundred - not over a short
list that is guaranteed to contain the answer. An item is correct when the chosen tool is one of the tools that
make progress on the goal's first step (`accept`).

Templates are split: ``dev`` templates were used while designing prompts and retrieval; ``test`` templates (held out,
other actions, other services) are only measured. Per item, `chance` is the probability that a participant choosing
uniformly at random among what it is shown (each shortlisted service, then each shortlisted tool) picks an
acceptable tool; the p-value is the Poisson-binomial tail of the correct count under these chances. A retrieval-only
baseline (the first-ranked tool of the first-ranked service, no model) is reported beside it.
"""
from __future__ import annotations

import numpy as np

NAMES_T = ["route planner", "invoice tool", "photo resizer", "recipe finder", "chat widget", "backup service",
           "language course", "bike repair guide", "garden planner", "budget sheet"]
TOPICS_T = ["cheap hosting", "growing tomatoes", "learning Spanish", "training a puppy", "saving for retirement",
            "fixing a bicycle", "baking bread", "running a marathon", "home insulation", "starting a podcast"]
THINGS_T = ["logo", "website text", "translation", "market survey", "user guide", "photo shoot", "data cleanup"]


def _r(rng, seq):
    return seq[int(rng.integers(0, len(seq)))]


# template -> (split, builder(rng, me, other) -> (goal, staging [(actor_is_other, tool, args)], acceptable tools))
def _templates():
    T = {}

    def t(name, split):
        def deco(fn):
            T[name] = (split, fn)
            return fn
        return deco

    @t("pay_debt", "dev")
    def _(rng, me, other):
        return (f"I will pay {other} the {int(rng.integers(5, 90)) * 100} cents I owe them for the {_r(rng, THINGS_T)}.",
                [], {"bank.transfer"})

    @t("check_money", "dev")
    def _(rng, me, other):
        return ("I will find out exactly how much money I have before I buy anything.", [],
                {"bank.accounts", "bank.balance", "bank.statement"})

    @t("ask_privately", "dev")
    def _(rng, me, other):
        return (f"I will ask {other} privately whether they can review my {_r(rng, THINGS_T)} this week.", [],
                {"mail.send", "msg.send"})

    @t("read_mail", "dev")
    def _(rng, me, other):
        th = _r(rng, THINGS_T)
        return (f"I will find out what {other} wrote to me about the {th}.",
                [(True, "mail.send", {"to": [me], "subject": f"About the {th}", "body": f"I have news about the {th}."})],
                {"mail.read", "mail.inbox", "mail.search"})

    @t("write_article", "dev")
    def _(rng, me, other):
        return (f"I will write an article about {_r(rng, TOPICS_T)} and put it on my blog.", [],
                {"studio.write", "docs.create"})

    @t("ask_forum", "dev")
    def _(rng, me, other):
        return (f"I will ask people on the public forum for advice about {_r(rng, TOPICS_T)}.", [], {"forum.post"})

    @t("search_web", "dev")
    def _(rng, me, other):
        return (f"I will find out whether anybody in this world already offers a {_r(rng, NAMES_T)}.", [],
                {"search.query", "market.browse", "prod.catalog", "xchg.browse", "launch.latest"})

    @t("new_repo", "dev")
    def _(rng, me, other):
        return (f"I will start a new code repository called {_r(rng, NAMES_T).replace(' ', '-')} for my program.", [],
                {"code.create_repo"})

    @t("find_job", "dev")
    def _(rng, me, other):
        return ("I will look at which job positions are open right now.", [], {"jobs.openings", "labor.postings"})

    @t("follow", "dev")
    def _(rng, me, other):
        return (f"I will follow {other} so that I see everything they publish.", [], {"social.follow"})

    @t("notes_doc", "dev")
    def _(rng, me, other):
        return ("I will write my meeting notes down in a document.", [], {"docs.create", "studio.write"})

    @t("read_feed", "dev")
    def _(rng, me, other):
        return ("I will read the newest text posts that are recommended to me.", [],
                {"posts.feed", "feed.home", "posts.latest", "social.timeline"})

    # ---------------------------------------------------------------- held out
    @t("rent_server", "test")
    def _(rng, me, other):
        return (f"I will rent a server with {int(rng.integers(1, 4))} cores to run my web service.", [],
                {"cloud.rent", "cloud.providers"})

    @t("answer_forum", "test")
    def _(rng, me, other):
        top = _r(rng, TOPICS_T)
        return (f"I will answer the question {other} asked on the forum about {top}.",
                [(True, "forum.post", {"board": "help", "title": f"Question about {top}",
                                       "body": f"Does anyone know a good way to get started with {top}?"})],
                {"forum.reply", "forum.read", "forum.list", "forum.feed"})

    @t("join_community", "test")
    def _(rng, me, other):
        return (f"I will join a community of people who care about {_r(rng, TOPICS_T)}.", [],
                {"community.join", "community.directory"})

    @t("share_doc", "test")
    def _(rng, me, other):
        return (f"I will give {other} access to my document with the project plan.",
                [(False, "docs.create", {"title": "Project plan", "body": "Step one: plan the work."})],
                {"docs.share", "docs.list"})

    @t("my_stats", "test")
    def _(rng, me, other):
        return ("I will check how many people have seen the things I published.", [],
                {"aud.mine", "aud.analytics", "content.mine"})

    @t("send_invoice", "test")
    def _(rng, me, other):
        return (f"I will send {other} a bill of {int(rng.integers(10, 90)) * 100} cents for the work I did.", [],
                {"bank.invoice"})

    @t("post_job", "test")
    def _(rng, me, other):
        return (f"I will advertise an open position for a {_r(rng, ['developer', 'support agent', 'analyst'])} "
                "publicly.", [], {"jobs.post", "labor.post"})

    @t("found_company", "test")
    def _(rng, me, other):
        return (f"I will found a company called {_r(rng, ['Brightbyte', 'Copperleaf', 'Redkite', 'Lumen'])} with "
                "myself as the owner.", [], {"id.create_org", "biz.register", "venture.form"})

    @t("comment_item", "test")
    def _(rng, me, other):
        return ("I will tell the author of the newest article on my feed what I think of it, publicly.", [],
                {"content.comment", "blog.comment", "blog.feed", "posts.comment", "feed.home"})

    @t("report_issue", "test")
    def _(rng, me, other):
        return ("I will record a bug report in an issue tracker: the login page of my app crashes.", [],
                {"issues.create", "issues.create_project"})

    @t("newsletter", "test")
    def _(rng, me, other):
        return (f"I will start a newsletter about {_r(rng, TOPICS_T)} that people can subscribe to.", [],
                {"letters.create", "creators.create_tier"})

    @t("schedule_talk", "test")
    def _(rng, me, other):
        return (f"I will let {other} know in a private chat message that I am free on Friday.", [],
                {"msg.send", "mail.send"})
    return T


TEMPLATES = _templates()


def build(seed: int = 0, per_template: int = 5, splits=("dev", "test")) -> list[dict]:
    """Benchmark items: situations staged in a civilisation for its class-4 participants."""
    import crucible
    from . import module as GM, world as GW
    crucible.load_all_modules()
    w = GW.civ_world("balanced", seed, population="development", config={"brain": "ablated"})
    rows = GM.rows(w)
    me_i, other_i = rows[0], rows[1]
    rng = np.random.default_rng(seed)
    items = []
    for name, (split, fn) in sorted(TEMPLATES.items()):
        if split not in splits:
            continue
        for k in range(per_template):
            goal, staging, accept = fn(rng, me_i["principal"], other_i["principal"])
            for by_other, tool, args in staging:
                actor = other_i["principal"] if by_other else me_i["principal"]
                r = w.execute(actor, {"tool": tool, "args": args})
                if not r.ok:
                    raise RuntimeError(f"staging {tool} failed: {r.error}")
            w.advance_time(w.now + 600)
            items.append({"id": f"{name}/{k}", "template": name, "split": split, "goal": goal,
                          "accept": sorted(accept), "t": int(w.now), "state": _snapshot(w, me_i["i"])})
    w.close()
    return items


def _snapshot(world, i: int) -> dict:
    """The participant's perceived situation and tools at this moment (perception runs as in the world)."""
    from . import mind as M
    from .module import TABLE
    row = world.state.get(TABLE, str(int(i)))
    a = M.Agent.from_row(row)
    world._run_system("bench.perceive", lambda ctx: M.perceive(ctx, a))
    return {"row": {k: row[k] for k in ("i", "principal", "account", "persona")}, "situation": list(a.situation),
            "memory": list(a.memory), "known": dict(a.known), "tools": a.tools,
            "summaries": M.host_summaries(world)}


def agent_of(item: dict):
    from . import mind as M
    st = item["state"]
    a = M.Agent.from_row({**st["row"], "goals": [], "memory": st["memory"], "known": st["known"]})
    a.situation, a.tools = list(st["situation"]), list(st["tools"])
    a.new_goal(item["t"], item["goal"])
    return a


def evaluate(items: list[dict], brain, cfg=None, batch: int = 16) -> dict:
    """Run the participants' decision pipeline on every item; accuracy, chance, p-value, retrieval baseline."""
    from . import catalog, mind as M
    from ..service import poisson_binom_sf, wilson
    cfg = cfg or M.Config()
    out = []
    for s in range(0, len(items), batch):
        chunk = items[s:s + batch]
        agents = [agent_of(it) for it in chunk]
        cfg.summaries = chunk[0]["state"]["summaries"]
        acting = M.decide(None, brain, agents, chunk[0]["t"], cfg)
        picked = {id(a): t["name"] for a, t, _ in acting}
        for it, a in zip(chunk, agents):
            log = a.step_log[-1] if a.step_log else {}
            pick = picked.get(id(a)) or log.get("service")
            hosts = log.get("services") or []
            acc = set(it["accept"])
            ch = 0.0
            for h in hosts:
                ts = [t for t in a.tools if catalog.host_of(t["name"]) == h]
                tl = catalog.rank_tools([a.intent or it["goal"]], [ts], cfg.shortlist)[0]
                if tl:
                    ch += (1.0 / (len(hosts) + len(M.endings(a)))) * sum(n in acc for n in tl) / len(tl)
            base = None
            if hosts:
                ts = [t for t in a.tools if catalog.host_of(t["name"]) == hosts[0]]
                tl = catalog.rank_tools([a.intent or it["goal"]], [ts], 1)[0]
                base = tl[0] if tl else None
            out.append({"id": it["id"], "split": it["split"], "template": it["template"], "intent": a.intent,
                        "services": hosts, "service": log.get("service"), "pick": pick,
                        "correct": bool(pick in acc), "chance": round(ch, 5),
                        "host_recall": bool({catalog.host_of(x) for x in acc} & set(hosts)),
                        "retrieval_only_correct": bool(base in acc)})
    res = {}
    for split in sorted({o["split"] for o in out}):
        rs = [o for o in out if o["split"] == split]
        k, n = sum(o["correct"] for o in rs), len(rs)
        lo, hi = wilson(k, n)
        res[split] = {"n": n, "correct": k, "accuracy": round(k / max(1, n), 4), "ci95": [round(lo, 4), round(hi, 4)],
                      "chance_mean": round(float(np.mean([o["chance"] for o in rs])), 4),
                      "p_vs_random": float(f"{poisson_binom_sf(k, [o['chance'] for o in rs]):.3g}"),
                      "host_recall": round(float(np.mean([o["host_recall"] for o in rs])), 4),
                      "retrieval_only_accuracy": round(float(np.mean([o["retrieval_only_correct"] for o in rs])), 4),
                      "by_template": {t: round(float(np.mean([o["correct"] for o in rs if o["template"] == t])), 3)
                                      for t in sorted({o["template"] for o in rs})}}
    return {"splits": res, "items": out}
