"""Simulated users of SocialWebLab: population, writing styles and session behaviour.

Every simulated user acts only through the lab's service layer (the same functions the HTTP
handlers call, with the same authorisation checks) and only reacts to information an ordinary
user of the application could see. All generated activity is marked simulated in the database
(``users.kind``) and stays inside the lab.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from agentrs.socialweblab.apps import SPACES

PROBLEMS = ["fails", "crashes", "hangs", "is very slow", "returns wrong results", "won't start", "keeps timing out"]
TRIED = ["I tried reinstalling.", "I already restarted twice.", "Searching did not help.", "I checked the docs.",
         "Rolling back did not fix it."]
POLITE = ["Thanks in advance!", "Any help appreciated.", "Thank you!", "Please advise, thanks."]
SHOUT = ["HELP!!! NOTHING WORKS!!!", "THIS IS BROKEN AGAIN!!!", "WHY DOES THIS ALWAYS HAPPEN!!!"]
CLICK = ["URGENT:", "You won't believe this:", "ASAP:", "Shocking:"]
SPAM = ["Buy now at www.cheap-deals.example and get free money!", "Crypto giveaway, click here: http://win.example",
        "Limited offer!!! www.best-price.example"]
DISCUSS = ["What is your favourite way to {k}?", "Sharing my notes on {k} and {k2}.", "Thoughts on {k} this year?"]


@dataclass
class Style:
    p_detail: float
    p_polite: float
    p_question: float = 0.85
    p_code: float = 0.3
    p_shout: float = 0.03
    p_clickbait: float = 0.03
    p_spam: float = 0.0
    p_offtopic: float = 0.03
    p_tests: float = 0.6


STYLES = {
    "veteran": Style(p_detail=0.75, p_polite=0.6, p_code=0.45, p_shout=0.01, p_clickbait=0.02, p_tests=0.8),
    "regular": Style(p_detail=0.45, p_polite=0.4),
    "newbie": Style(p_detail=0.2, p_polite=0.3, p_shout=0.15, p_clickbait=0.12, p_offtopic=0.08, p_tests=0.3),
    "spammer": Style(p_detail=0.0, p_polite=0.1, p_question=0.2, p_spam=0.9, p_shout=0.4, p_clickbait=0.5),
    "expert": Style(p_detail=0.8, p_polite=0.5, p_code=0.6, p_tests=0.9),
}


@dataclass
class Persona:
    handle: str
    role: str  # veteran | regular | newbie | spammer | expert | moderator | staff | maintainer | bot
    spaces: list[str]
    expertise: list[str] = field(default_factory=list)
    sessions_per_day: float = 4.0
    p_post: float = 0.12
    peak_hour: int = 19


def make_population(n_users: int, seed: int, spaces: list[str]) -> list[Persona]:
    rng = random.Random(seed * 1009 + 7)
    forum = [s for s in spaces if SPACES[s][0] == "forum"]
    help_ = [s for s in spaces if SPACES[s][0] == "helpdesk"]
    repo = [s for s in spaces if SPACES[s][0] == "codereview"]
    out: list[Persona] = []
    counts = {"veteran": max(2, n_users // 8), "expert": max(3, n_users // 6), "spammer": max(1, n_users // 25)}
    roles = ["veteran"] * counts["veteran"] + ["expert"] * counts["expert"] + ["spammer"] * counts["spammer"]
    roles += ["newbie" if rng.random() < 0.45 else "regular" for _ in range(max(0, n_users - len(roles)))]
    n_exp = 0
    for i, role in enumerate(roles):
        k = rng.randint(2, 4)
        subs = rng.sample(spaces, min(k, len(spaces)))
        exp = []
        if role == "expert":
            # every space gets experts (round robin), plus random extra expertise
            exp = sorted({spaces[n_exp % len(spaces)], spaces[(n_exp + 3) % len(spaces)]} |
                         set(rng.sample(spaces, min(rng.randint(0, 1), len(spaces)))))
            n_exp += 1
            subs = sorted(set(subs) | set(exp))
        out.append(Persona(handle=f"{role[:3]}{i:03d}", role=role, spaces=subs, expertise=exp,
                           sessions_per_day={"veteran": 5, "expert": 6, "spammer": 3}.get(role, 3.5) * rng.uniform(0.6, 1.4),
                           p_post={"veteran": 0.3, "expert": 0.05, "spammer": 0.5, "newbie": 0.3}.get(role, 0.25),
                           peak_hour=(19 + rng.randint(-3, 3)) % 24))
    for j in range(2):
        out.append(Persona(handle=f"mod{j}", role="moderator", spaces=forum, sessions_per_day=10, p_post=0.0,
                           peak_hour=14))
    if help_:
        for j in range(3):
            out.append(Persona(handle=f"staff{j}", role="staff", spaces=help_, expertise=help_, sessions_per_day=12,
                               p_post=0.0, peak_hour=13))
    if repo:
        for j in range(2):
            out.append(Persona(handle=f"maint{j}", role="maintainer", spaces=repo, expertise=repo, sessions_per_day=6,
                               p_post=0.0, peak_hour=15))
        out.append(Persona(handle="ci-bot", role="bot", spaces=repo, sessions_per_day=0, p_post=0.0))
    return out


def activity(hour: float, peak: int) -> float:
    """Diurnal multiplier in [0.15, 1]."""
    d = min(abs(hour - peak), 24 - abs(hour - peak))
    return 0.15 + 0.85 * math.exp(-(d / 4.0) ** 2)


def write_item(rng: random.Random, space: str, style: Style, overrides: dict | None = None) -> tuple[str, str, dict]:
    """Returns (title, body, the style flags actually used). ``overrides`` forces flags (experiments)."""
    app, kws = SPACES[space]
    o = overrides or {}
    f = {
        "detail": o.get("detail", rng.random() < style.p_detail),
        "polite": o.get("polite", rng.random() < style.p_polite),
        "question": o.get("question", rng.random() < style.p_question),
        "code": o.get("code", rng.random() < style.p_code),
        "shout": o.get("shout", rng.random() < style.p_shout),
        "clickbait": o.get("clickbait", rng.random() < style.p_clickbait),
        "spam": o.get("spam", rng.random() < style.p_spam),
        "offtopic": o.get("offtopic", rng.random() < style.p_offtopic),
        "tests": o.get("tests", rng.random() < style.p_tests),
    }
    if f["offtopic"]:
        others = [s for s in SPACES if SPACES[s][0] == app and s != space] or [s for s in SPACES if s != space]
        kws = SPACES[rng.choice(others)][1]
    k, k2 = rng.sample(kws, 2)
    if f["spam"]:
        body = rng.choice(SPAM) + f" Best {k} deals."
        return body[:70], body, f
    if app == "codereview":
        title = f"Fix {k} handling in {k2}"
        lines = [f"{'Fix' if rng.random() < 0.5 else 'Improve'} {k} when {k2} {rng.choice(PROBLEMS)}."]
        if f["detail"]:
            lines.append(f"Fixes issue #{rng.randint(100, 999)} (error E{rng.randint(1000, 9999)} on line {rng.randint(10, 400)}).")
        lines.append(f"```\ndef {k}_{k2}(x, retries=3):\n    return handle(x, retries)\n```")
        if f["tests"]:
            lines.append(f"Adds def test_{k}_{k2}(): assert {k}_{k2}(1) == 1")
        if rng.random() < 0.25:
            lines.append("TODO: clean up later")
        if f["polite"]:
            lines.append(rng.choice(POLITE))
        return title, "\n".join(lines), f
    if app == "helpdesk":
        issue = {"billing": [f"I was charged twice for my {k}", f"My {k} {k2} did not go through", f"I need a {k} for my {k2}"],
                 "accounts": [f"I cannot {k} to my {k2}", f"My {k} {k2} is not arriving", f"My {k2} is {k}"]}
        base = rng.choice(issue.get(space, [f"Problem with {k} and {k2}"]))
        lines = [base + ("?" if f["question"] else ".")]
        if f["detail"]:
            lines.append(rng.choice([f"Order #{rng.randint(10000, 99999)}.", f"Account id {rng.randint(100000, 999999)}.",
                                     f"Error code E{rng.randint(100, 999)} shown."]))
        if f["shout"]:
            lines.append(rng.choice(SHOUT))
        if f["polite"]:
            lines.append(rng.choice(POLITE))
        title = ("URGENT: " if f["clickbait"] else "") + base[:60]
        return title, " ".join(lines), f
    # forum
    if not f["question"] and rng.random() < 0.5:
        body = rng.choice(DISCUSS).format(k=k, k2=k2)
        if f["polite"]:
            body += " " + rng.choice(POLITE)
        return body[:70], body, f
    prob = rng.choice(PROBLEMS)
    head = f"How do I fix {k} when {k2} {prob}?" if f["question"] else f"My {k} {prob} with {k2}."
    lines = [head]
    if f["detail"]:
        lines.append(rng.choice([f"Getting error: E{rng.randint(1000, 9999)} on line {rng.randint(3, 300)}.",
                                 f"Using version {rng.randint(1, 4)}.{rng.randint(0, 12)}.{rng.randint(0, 9)}.",
                                 f"Traceback says {k}Error: cannot open {k2}."]))
    if f["code"]:
        lines.append(f"`{k} --{k2} = true`")
    lines.append(rng.choice(TRIED))
    if f["shout"]:
        lines.append(rng.choice(SHOUT))
    if f["polite"]:
        lines.append(rng.choice(POLITE))
    title = (rng.choice(CLICK) + " " if f["clickbait"] else "") + head[:64]
    return title, " ".join(lines), f


ANSWERS = ["Solution: upgrade {k} to the latest release and clear the {k2} cache, then retry.",
           "Solution: the {k} setting conflicts with {k2}; set it explicitly and restart.",
           "Solution: this is a known {k} bug, apply the patch for {k2} and it works."]
WEAK_ANSWERS = ["Have you tried turning {k} off and on again?", "Maybe reinstall {k}? Not sure.",
                "I think {k2} is the problem but I don't know how to fix it."]
COMMENTS = ["Same problem here.", "+1, following this.", "Interesting, never seen that with {k}.", "Nice post!"]
REVIEWS = ["Please add a test for the {k} path.", "Looks good, but rename {k2}.", "LGTM once CI passes."]
