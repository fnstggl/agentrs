"""Information-equivalent natural-language observations and goals for the CAM environments.

Every text is produced *only* from the agent's own observation / goal tokens (``env.obs_tokens()``,
``env.goal_tokens()``), never from simulator state, so it cannot leak hidden parameters, reward
functions or verifier state, and it carries exactly the information of the structured tokens (the
structured tokens are the information-equivalent control). Numbers are rendered from the token bins'
representative values, so the bin is recoverable.

Two wording styles:
  * ``train``   — the wording seen during training (several phrasings per fact, shuffled sentence order);
  * ``heldout`` — different vocabulary (vendor, order volume, cash on hand, retained customers ...),
                  number formats (fractions, "k", "x/10"), sentence structures; used only to test
                  semantic generalisation to unfamiliar language about the same situation.
``parse_obs`` / ``parse_goal`` invert the ``train`` wording exactly (the rule-based text-parser
baseline, and the information-equivalence check); they are not expected to read ``heldout`` text.
"""
from __future__ import annotations

import math
import random
import re
import zlib

ISSUES = ["price", "qty", "delivery", "warranty"]
ISSUE_TXT = {"train": {"price": "price", "qty": "quantity", "delivery": "delivery", "warranty": "warranty"},
             "heldout": {"price": "pricing", "qty": "order volume", "delivery": "delivery timing", "warranty": "warranty coverage"}}
IMP_TXT = {"train": {"HIGH": "high", "MED": "medium", "LOW": "low"},
           "heldout": {"HIGH": "a top priority", "MED": "moderately important", "LOW": "a minor concern"}}
MOOD_TXT = {"train": {"OK": "calm", "TENSE": "tense", "ANGRY": "angry"},
            "heldout": {"OK": "relaxed", "TENSE": "strained", "ANGRY": "hostile"}}
RESP_TXT = {"train": {"NONE": ["The supplier has not answered a proposal yet.", "No proposal has been answered by the supplier yet."],
                      "COUNTER": ["The supplier rejected your last proposal and made a counter-offer.",
                                  "The supplier turned down your proposal and countered."],
                      "ACCEPTED": ["The supplier accepted the deal.", "The supplier said yes to the deal."],
                      "REJECTED": ["The supplier rejected your request.", "The supplier refused your request."],
                      "LEFT": ["The supplier has left the negotiation.", "The supplier walked out of the negotiation."]},
            "heldout": {"NONE": ["No reply to a proposal has come back from the vendor."],
                        "COUNTER": ["Your counterpart declined the previous terms and came back with terms of their own."],
                        "ACCEPTED": ["Your counterpart agreed to the terms."],
                        "REJECTED": ["Your counterpart did not go along with that."],
                        "LEFT": ["Your counterpart broke off the talks."]}}
WANT_TXT = {"train": {"share": "revenue share", "exclusive": "exclusivity", "speed": "speed", "proof": "social proof"},
            "heldout": {"share": "a bigger cut of revenue", "exclusive": "an exclusive arrangement", "speed": "fast execution",
                        "proof": "evidence that others have already signed"}}
STATUS_TXT = {"train": {"UNKNOWN": "not contacted", "PENDING": "waiting for a reply", "INTERESTED": "interested",
                        "OBJECTION": "raised an objection", "DECLINED": "declined", "COMMITTED": "committed"},
              "heldout": {"UNKNOWN": "no outreach so far", "PENDING": "reply outstanding", "INTERESTED": "keen",
                          "OBJECTION": "pushed back", "DECLINED": "said no", "COMMITTED": "signed up"}}
SENS_TXT = {"train": {"HIGH": "high", "MED": "medium", "LOW": "low"},
            "heldout": {"HIGH": "very price-sensitive", "MED": "somewhat price-sensitive", "LOW": "not very price-sensitive"}}
CHANNELS = ["ads", "content", "partners", "referral"]
CH_TXT = {"train": {c: c for c in CHANNELS},
          "heldout": {"ads": "paid advertising", "content": "content marketing", "partners": "partner programs", "referral": "word of mouth"}}
PRICES = [0, 5, 10, 20, 40, 80]
HI = {"BUDGET": 300_000, "ACTIVE": 200_000, "SIGNUPS": 20_000, "CHURN": 20_000, "REVENUE": 100_000, "CH": 20_000, "TARGET": 200_000}


def bi(tok: str) -> int:
    return int(tok[1:])


def logval(k: int, hi: float) -> float:
    return math.expm1(k / 20 * math.log1p(hi))


def logbin(x: float, hi: float) -> int:
    return int(round(min(1.0, max(0.0, math.log1p(max(0.0, x)) / math.log1p(hi))) * 20))


def rng_for(*key) -> random.Random:
    return random.Random(zlib.crc32(repr(key).encode()))


# ---------------------------------------------------------------------------- number formats
def pct(k: int, style: str, r: random.Random) -> str:
    return f"{5 * k}%" if style == "train" else r.choice([f"{k / 20:.2f}", f"{5 * k} per cent"])


def amount(k: int, hi: float, style: str, r: random.Random, money: bool = False) -> str:
    x = logval(k, hi)
    if style == "train":
        return f"about {'$' if money else ''}{int(round(x)):,}"
    if x >= 1000:
        return r.choice([f"roughly {x / 1000:.2f}k{' dollars' if money else ''}", f"~{x / 1000:.2f} thousand{' dollars' if money else ''}"])
    return f"roughly {x:.1f}{' dollars' if money else ''}"


def level(L: int, style: str, r: random.Random) -> str:
    return f"level {L}" if style == "train" else r.choice([f"{L}/10", f"{L} out of 10"])


# ---------------------------------------------------------------------------- negotiation
def _neg_fields(t: list[str]) -> dict:
    f = {"round": bi(t[1]), "resp": t[3], "last": None, "best": None, "mood": None, "answer": None, "know": []}
    i = 4
    while i < len(t):
        if t[i] == "LAST":
            f["last"] = [bi(x) for x in t[i + 1:i + 5]]
            f["best"] = bi(t[i + 6])
            i += 7
        elif t[i] == "MOOD":
            f["mood"] = t[i + 1]
            i += 2
        elif t[i] == "ANSWER":
            f["answer"] = (t[i + 2], t[i + 3])
            i += 4
        elif t[i] == "KNOW":
            f["know"].append((t[i + 1], t[i + 2]))
            i += 3
        else:
            raise ValueError(f"unexpected negotiation token {t[i]!r} in {t}")
    return f


def _neg_text(t, style, r):
    f = _neg_fields(t)
    it, im = ISSUE_TXT[style], IMP_TXT[style]
    s = []
    if style == "train":
        s.append(r.choice(["Negotiation progress: {p} of the allowed rounds have been used.",
                           "So far {p} of the allowed rounds have been used."]).format(p=pct(f["round"], style, r)))
    else:
        s.append(f"At this point {pct(f['round'], style, r)} of the negotiation window has elapsed.")
    s.append(r.choice(RESP_TXT[style][f["resp"]]))
    if f["last"] is not None:
        a = [level(x, style, r) for x in f["last"]]
        if style == "train":
            s.append(r.choice(["Their current offer: price {0}, quantity {1}, delivery {2}, warranty {3}.",
                               "Offer on the table from them: price {0}, quantity {1}, delivery {2}, warranty {3}."]).format(*a))
            s.append(f"The best offer they have made so far is worth {pct(f['best'], style, r)} of your maximum possible value.")
        else:
            s.append("Their side currently proposes {0} on pricing, {1} on order volume, {2} on delivery timing and {3} on "
                     "warranty coverage.".format(*a))
            s.append(f"Measured against the ideal outcome for you, their strongest proposal so far reaches {pct(f['best'], style, r)}.")
    s.append(f"Their mood appears {MOOD_TXT[style][f['mood']]}." if style == "train" else f"The vendor's tone is {MOOD_TXT[style][f['mood']]}.")
    extra = []
    if f["answer"]:
        i, v = f["answer"]
        extra.append(f"In answer to your question about {it[i]}, they said it is of {im[v]} importance to them." if style == "train"
                     else f"Asked about {it[i]}, the vendor described it as {im[v]}.")
    for i, v in f["know"]:
        extra.append(f"You know that {it[i]} is of {im[v]} importance to them." if style == "train"
                     else f"Earlier inquiries showed that {it[i]} is {im[v]} for them.")
    head, rest = s[:1], s[1:] + extra
    r.shuffle(rest)
    return " ".join(head + rest)


def _neg_parse(x: str) -> list[str]:
    t = ["ROUND", f"N{int(re.search(r'(\d+)% of the allowed rounds', x).group(1)) // 5}", "CP"]
    for tok, vs in RESP_TXT["train"].items():
        if any(v in x for v in vs):
            t.append(tok)
            break
    m = re.search(r"price level (\d+), quantity level (\d+), delivery level (\d+), warranty level (\d+)", x)
    if m:
        t += ["LAST"] + [f"N{int(g)}" for g in m.groups()]
        t += ["BEST", f"N{int(re.search(r'so far is worth (\d+)%', x).group(1)) // 5}"]
    inv_m = {v: k for k, v in MOOD_TXT["train"].items()}
    t += ["MOOD", inv_m[re.search(r"mood appears (\w+)", x).group(1)]]
    inv_i = {v: k for k, v in ISSUE_TXT["train"].items()}
    inv_v = {v: k for k, v in IMP_TXT["train"].items()}
    m = re.search(r"question about (\w+), they said it is of (\w+) importance", x)
    if m:
        t += ["ANSWER", "ASK", inv_i[m.group(1)], inv_v[m.group(2)]]
    kn = re.findall(r"You know that (\w+) is of (\w+) importance", x)
    for i, v in sorted(kn, key=lambda p: ISSUES.index(inv_i[p[0]])):
        t += ["KNOW", inv_i[i], inv_v[v]]
    return t


def _neg_goal_text(t, style, r):
    parts, i = [], 3
    w = {}
    while t[i] == "W":
        w[t[i + 1]] = (bi(t[i + 2]), t[i + 4])
        i += 5
    tgt = bi(t[i + 1])
    it = ISSUE_TXT[style]
    if style == "train":
        for iss in ISSUES:
            k, d = w[iss]
            parts.append(f"{it[iss]} weight {pct(k, style, r)} ({'higher' if d == 'UP' else 'lower'} is better)")
        return ("Goal: negotiate a purchase as the buyer. How much each issue matters to you: " + ", ".join(parts)
                + f". Reach a deal worth at least {pct(tgt, style, r)} of the best possible value.")
    for iss in ISSUES:
        k, d = w[iss]
        parts.append(f"{it[iss]} counts for {pct(k, style, r)} and you want it {'as high' if d == 'UP' else 'as low'} as possible")
    return ("You are buying in a procurement deal. Your priorities: " + "; ".join(parts)
            + f". An agreement only counts if it delivers no less than {pct(tgt, style, r)} of the ideal value.")


def _neg_goal_parse(x: str) -> list[str]:
    t = ["GOAL", "NEGOTIATE", "BUY"]
    for iss in ISSUES:
        m = re.search(rf"{ISSUE_TXT['train'][iss]} weight (\d+)% \((higher|lower) is better\)", x)
        t += ["W", iss, f"N{int(m.group(1)) // 5}", "DIR", "UP" if m.group(2) == "higher" else "DOWN"]
    return t + ["TARGET", f"N{int(re.search(r'at least (\d+)% of the best', x).group(1)) // 5}"]


# ---------------------------------------------------------------------------- product
def _prod_fields(t):
    f = {"built": [], "building": [], "ch": {}, "needs": []}
    i = 0
    while i < len(t):
        k = t[i]
        if k in ("WEEK", "BUDGET", "ACTIVE", "SIGNUPS", "CHURN", "REVENUE", "ONB"):
            f[k] = bi(t[i + 1])
            i += 2
        elif k == "PRICE":
            f["PRICE"] = int(t[i + 1][1])
            i += 2
        elif k in ("BUILT", "BUILDING"):
            j = i + 1
            while j < len(t) and t[j].startswith("F") and t[j][1:].isdigit():
                f[k.lower()].append(t[j])
                j += 1
            i = j
        elif k == "CH":
            f["ch"][t[i + 1]] = bi(t[i + 2])
            i += 3
        elif k == "NEEDS":
            f["needs"].append((t[i + 1], t[i + 2], t[i + 3], t[i + 5]))
            i += 6
        else:
            raise ValueError(f"unexpected product token {k!r} in {t}")
    return f


def _prod_text(t, style, r):
    f = _prod_fields(t)
    s = []
    if style == "train":
        s.append(f"{pct(f['WEEK'], style, r)} of the 26-week period has passed.")
        s += [f"Cash: {amount(f['BUDGET'], HI['BUDGET'], style, r, True)}.",
              f"Active users: {amount(f['ACTIVE'], HI['ACTIVE'], style, r)}.",
              f"New signups last week: {amount(f['SIGNUPS'], HI['SIGNUPS'], style, r)}.",
              f"Users lost last week: {amount(f['CHURN'], HI['CHURN'], style, r)}.",
              f"Revenue last week: {amount(f['REVENUE'], HI['REVENUE'], style, r, True)}.",
              "Price: free." if f["PRICE"] == 0 else f"Price: ${PRICES[f['PRICE']]} per month.",
              f"Onboarding level: {round(f['ONB'] / 20 * 3)} of 3.",
              ("Built features: " + ", ".join(f["built"]) + ".") if f["built"] else "No features built yet.",
              ("In development: " + ", ".join(f["building"]) + ".") if f["building"] else "Nothing in development.",
              "Signups by channel last week: " + ", ".join(f"{c} {amount(f['ch'][c], HI['CH'], style, r)}" for c in CHANNELS) + "."]
        for seg, a, b, sens in f["needs"]:
            s.append(f"Research on segment {seg}: they need {a} and {b} most; price sensitivity {SENS_TXT[style][sens]}.")
    else:
        s.append(f"Of the 26-week runway window, {pct(f['WEEK'], style, r)} is behind us.")
        s += [f"Cash on hand is {amount(f['BUDGET'], HI['BUDGET'], style, r, True)}.",
              f"We currently retain {amount(f['ACTIVE'], HI['ACTIVE'], style, r)} customers.",
              f"Last week brought {amount(f['SIGNUPS'], HI['SIGNUPS'], style, r)} sign-ups and we lost "
              f"{amount(f['CHURN'], HI['CHURN'], style, r)} customers.",
              f"Weekly takings came to {amount(f['REVENUE'], HI['REVENUE'], style, r, True)}.",
              "The product is currently offered at no charge." if f["PRICE"] == 0 else
              f"Subscriptions cost {PRICES[f['PRICE']]} dollars a month.",
              f"Onboarding has been improved {round(f['ONB'] / 20 * 3)} time(s) out of a possible 3.",
              ("Shipped capabilities: " + " and ".join(f["built"]) + ".") if f["built"] else "Nothing has shipped yet.",
              ("Engineering is working on " + " and ".join(f["building"]) + ".") if f["building"] else "Engineering is idle.",
              "Acquisition by source last week: " + "; ".join(f"{CH_TXT[style][c]} {amount(f['ch'][c], HI['CH'], style, r)}"
                                                             for c in CHANNELS) + "."]
        for seg, a, b, sens in f["needs"]:
            s.append(f"Interviews with customer segment {seg} point to {a} and {b} as what they want most; they are "
                     f"{SENS_TXT[style][sens]}.")
    head, rest = s[:1], s[1:]
    r.shuffle(rest)
    return " ".join(head + rest)


def _num(x: str) -> float:
    return float(x.replace(",", "").replace("$", ""))


def _prod_parse(x: str) -> list[str]:
    g = lambda pat: re.search(pat, x).group(1)  # noqa: E731
    t = ["WEEK", f"N{int(g(r'(\d+)% of the 26-week'))// 5}",
         "BUDGET", f"N{logbin(_num(g(r'Cash: about (\$?[\d,]+)')), HI['BUDGET'])}",
         "ACTIVE", f"N{logbin(_num(g(r'Active users: about ([\d,]+)')), HI['ACTIVE'])}",
         "SIGNUPS", f"N{logbin(_num(g(r'New signups last week: about ([\d,]+)')), HI['SIGNUPS'])}",
         "CHURN", f"N{logbin(_num(g(r'Users lost last week: about ([\d,]+)')), HI['CHURN'])}",
         "REVENUE", f"N{logbin(_num(g(r'Revenue last week: about (\$?[\d,]+)')), HI['REVENUE'])}"]
    t += ["PRICE", "P0" if "Price: free." in x else f"P{PRICES.index(int(g(r'Price: \$(\d+) per month')))}"]
    t += ["ONB", f"N{int(round(int(g(r'Onboarding level: (\d) of 3')) / 3 * 20))}"]
    m = re.search(r"Built features: ([F\d, ]+)\.", x)
    t += ["BUILT"] + (m.group(1).split(", ") if m else [])
    m = re.search(r"In development: ([F\d, ]+)\.", x)
    t += ["BUILDING"] + (m.group(1).split(", ") if m else [])
    for c in CHANNELS:
        t += ["CH", c, f"N{logbin(_num(re.search(rf'{c} about ([\d,]+)', x).group(1)), HI['CH'])}"]
    inv_s = {v: k for k, v in SENS_TXT["train"].items()}
    for seg, a, b, sens in sorted(re.findall(r"Research on segment (S\d): they need (F\d) and (F\d) most; price sensitivity (\w+)", x)):
        t += ["NEEDS", seg, a, b, "SENS", inv_s[sens]]
    return t


def _prod_goal_text(t, style, r):
    k = bi(t[4])
    if style == "train":
        return f"Goal: grow to at least {amount(k, HI['TARGET'], style, r)} active users by the end of week 26 without running out of cash."
    return f"Objective: reach {amount(k, HI['TARGET'], style, r)} retained customers by the 26-week mark while staying solvent."


def _prod_goal_parse(x: str) -> list[str]:
    k = logbin(_num(re.search(r"at least about ([\d,]+) active users", x).group(1)), HI["TARGET"])
    return ["GOAL", "GROW", "USERS", "TARGET", f"N{k}", "BY", "WEEK", "N20"]


# ---------------------------------------------------------------------------- business development
def _biz_fields(t):
    f = {"step": bi(t[1]), "pool": bi(t[3]), "p": []}
    i = 4
    while i < len(t):
        assert t[i] == "PARTNER", t
        d = {"id": t[i + 1], "status": t[i + 3], "obj": None, "want": None}
        i += 4
        while i < len(t) and t[i] in ("OBJECTION", "WANTS"):
            d["obj" if t[i] == "OBJECTION" else "want"] = t[i + 1]
            i += 2
        f["p"].append(d)
    return f


def _biz_text(t, style, r):
    f = _biz_fields(t)
    wt, st = WANT_TXT[style], STATUS_TXT[style]
    if style == "train":
        s = [f"Step {f['step']} of 20.", f"Revenue share still available to promise: {pct(f['pool'], style, r)}."]
        for d in f["p"]:
            x = f"Partner {d['id']}: {st[d['status']]}"
            if d["obj"]:
                x += f", objection about {wt[d['obj']]}"
            if d["want"]:
                x += f", known to want {wt[d['want']]}"
            s.append(x + ".")
    else:
        s = [f"We are on turn {f['step']} of 20.", f"Uncommitted revenue-share budget: {pct(f['pool'], style, r)}."]
        for d in f["p"]:
            x = f"{d['id']} — {st[d['status']]}"
            if d["obj"]:
                x += f"; their concern was {wt[d['obj']]}"
            if d["want"]:
                x += f"; research suggests they value {wt[d['want']]}"
            s.append(x + ".")
    head, rest = s[:1], s[1:]
    r.shuffle(rest)
    return " ".join(head + rest)


def _biz_parse(x: str) -> list[str]:
    t = ["STEP", f"N{int(re.search(r'Step (\d+) of 20', x).group(1))}", "POOL",
         f"N{int(re.search(r'available to promise: (\d+)%', x).group(1)) // 5}"]
    inv_s = {v: k for k, v in STATUS_TXT["train"].items()}
    inv_w = {v: k for k, v in WANT_TXT["train"].items()}
    ws = "|".join(map(re.escape, inv_w))
    for pid, stx, obj, want in sorted(re.findall(rf"Partner (P\d): ([a-z ]+?)(?:, objection about ({ws}))?(?:, known to want ({ws}))?\.", x)):
        t += ["PARTNER", pid, "STATUS", inv_s[stx]]
        if obj:
            t += ["OBJECTION", inv_w[obj]]
        if want:
            t += ["WANTS", inv_w[want]]
    return t


def _biz_goal_text(t, style, r):
    k, cap = round(bi(t[2]) / 20 * 5), bi(t[5])
    if style == "train":
        return f"Goal: secure commitments from {k} partners while promising at most {pct(cap, style, r)} revenue share in total."
    return f"Objective: line up {k} committed partners, keeping total pledged revenue share within {pct(cap, style, r)}."


def _biz_goal_parse(x: str) -> list[str]:
    m = re.search(r"commitments from (\d) partners while promising at most (\d+)%", x)
    return ["GOAL", "SECURE", f"N{int(round(int(m.group(1)) / 5 * 20))}", "COMMITMENTS", "CAP", f"N{int(m.group(2)) // 5}"]


# ---------------------------------------------------------------------------- public API
_OBS = {"negotiation": (_neg_text, _neg_parse), "product": (_prod_text, _prod_parse), "bizdev": (_biz_text, _biz_parse)}
_GOAL = {"negotiation": (_neg_goal_text, _neg_goal_parse), "product": (_prod_goal_text, _prod_goal_parse),
         "bizdev": (_biz_goal_text, _biz_goal_parse)}
STYLES = ("train", "heldout")


def obs_text(family: str, tokens: list[str], style: str = "train", key=None) -> str:
    """Natural-language observation from the agent's observation tokens (``key`` fixes the wording draw)."""
    return _OBS[family][0](tokens, style, rng_for("obs", family, style, key, tuple(tokens)))


def goal_text(family: str, tokens: list[str], style: str = "train", key=None) -> str:
    return _GOAL[family][0](tokens, style, rng_for("goal", family, style, key, tuple(tokens)))


def parse_obs(family: str, text: str) -> list[str]:
    return _OBS[family][1](text)


def parse_goal(family: str, text: str) -> list[str]:
    return _GOAL[family][1](text)
