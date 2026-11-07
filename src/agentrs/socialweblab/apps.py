"""Application kinds hosted by SocialWebLab and the observable content features their simulated
users react to.

Three applications share one data model (items, responses, reactions, views, notifications):

* ``forum``      — questions / discussion posts in topic spaces; answers and comments; upvotes;
                   accepted answers; moderation of spam;
* ``helpdesk``   — support tickets; staff answers during business hours; the customer confirms
                   resolution;
* ``codereview`` — patches; an asynchronous CI job posts a result; reviewers approve or comment;
                   a maintainer merges approved patches that pass CI.

Simulated users react only to what they can see: the text (via ``content_features``), the space,
the author's public follower count, and when the item reaches them. The *effects* of the features
are causal by construction (they are inputs of the users' decision rules); the audience and the
posting hour change exposure, so they confound naive content/outcome associations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SPACES = {
    # forum
    "python-help": ("forum", ["python", "pip", "venv", "import", "pandas", "numpy", "traceback"]),
    "linux": ("forum", ["kernel", "systemd", "bash", "apt", "ssh", "permissions", "cron"]),
    "webdev": ("forum", ["css", "javascript", "react", "html", "fetch", "cors", "webpack"]),
    "databases": ("forum", ["sql", "postgres", "index", "query", "join", "migration", "sqlite"]),
    "gardening": ("forum", ["tomato", "soil", "compost", "seeds", "watering", "pests", "mulch"]),
    "cooking": ("forum", ["bread", "yeast", "oven", "recipe", "sauce", "dough", "knife"]),
    # helpdesk
    "billing": ("helpdesk", ["invoice", "refund", "charge", "payment", "card", "subscription"]),
    "accounts": ("helpdesk", ["login", "password", "account", "email", "verification", "locked"]),
    # code review
    "repo-core": ("codereview", ["parser", "cache", "scheduler", "config", "logging", "retry"]),
    "repo-ui": ("codereview", ["button", "layout", "theme", "modal", "render", "tooltip"]),
}

APP_OPERATION = {"forum": "create_post", "helpdesk": "open_ticket", "codereview": "submit_patch"}

_SPAM = re.compile(r"(https?://|www\.|buy now|free money|click here|limited offer|crypto giveaway)", re.IGNORECASE)
_DETAIL = re.compile(r"(\b\d+\.\d+(\.\d+)?\b|error[: ]|exception|traceback|errno|code [A-Z]?\d{2,}|\bE\d{3,}\b|"
                     r"order #?\d{4,}|account id|ticket #?\d+|line \d+|version)", re.IGNORECASE)
_POLITE = re.compile(r"\b(please|thanks|thank you|appreciate|grateful)\b", re.IGNORECASE)
_CLICKBAIT = re.compile(r"\b(urgent|asap|you won't believe|insane|shocking|must see|!!!)", re.IGNORECASE)
_TESTS = re.compile(r"def test_|assert ", re.IGNORECASE)


@dataclass
class Features:
    is_question: bool
    detail: bool
    polite: bool
    code: bool
    shouting: bool
    clickbait: bool
    spam: bool
    topic_match: float  # fraction of the space's keywords family present (0..1, capped)
    off_topic: bool
    length_words: int
    has_tests: bool
    todo: bool

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def content_features(text: str, space: str) -> Features:
    words = re.findall(r"[A-Za-z']+", text)
    lw = [w.lower() for w in words]
    kw = SPACES.get(space, ("forum", []))[1]
    hits = sum(1 for k in kw if k in lw)
    other = 0
    for s, (_, kws) in SPACES.items():
        if s != space:
            other = max(other, sum(1 for k in kws if k in lw))
    caps = sum(1 for w in words if len(w) >= 3 and w.isupper())
    return Features(
        is_question="?" in text or bool(re.search(r"\b(how|why|what|can|does|is there)\b", text, re.IGNORECASE)),
        detail=bool(_DETAIL.search(text)),
        polite=bool(_POLITE.search(text)),
        code="`" in text or "()" in text or "def " in text or "=" in text,
        shouting=caps >= 3 or text.count("!") >= 3,
        clickbait=bool(_CLICKBAIT.search(text)),
        spam=bool(_SPAM.search(text)),
        topic_match=min(1.0, hits / 2.0),
        off_topic=hits == 0 and other >= 2,
        length_words=len(words),
        has_tests=bool(_TESTS.search(text)),
        todo="TODO" in text or "FIXME" in text,
    )


@dataclass
class AppRules:
    """Coefficients of the simulated users' decision rules (logits)."""

    open_base: float = -0.6
    open_clickbait: float = 1.2
    open_shouting: float = 0.6
    open_interest: float = 0.8
    answer_base: float = -1.6
    answer_detail: float = 1.6
    answer_question: float = 0.8
    answer_polite: float = 0.5
    answer_code: float = 0.4
    answer_topic: float = 0.8
    answer_shouting: float = -1.4
    answer_clickbait: float = -1.2
    answer_spam: float = -3.0
    answer_off_topic: float = -2.2
    answer_per_existing: float = -0.5
    answer_short: float = -0.8  # fewer than 8 words
    helpful_base: float = 0.3
    helpful_detail: float = 1.6
    helpful_code: float = 0.4
    comment_prob: float = 0.04
    up_base: float = -2.0
    up_detail: float = 0.9
    up_polite: float = 0.4
    up_spam: float = -2.5
    down_shouting: float = 0.25
    down_spam: float = 0.5
    report_spam: float = 0.35
    accept_prob: float = 0.85
    edit_after_s: float = 6 * 3600
    edit_prob: float = 0.35
    delete_prob: float = 0.02
    business_hours_only: bool = False
    experts_only_answer: bool = True


RULES = {
    "forum": AppRules(),
    "helpdesk": AppRules(answer_base=-0.4, answer_detail=2.0, answer_polite=0.2, answer_question=0.2, answer_topic=0.4,
                         open_clickbait=0.3, business_hours_only=True, comment_prob=0.01, up_base=-3.0),
    "codereview": AppRules(answer_base=-0.8, answer_detail=0.6, answer_question=-0.2, answer_code=1.4, answer_topic=0.6,
                           open_clickbait=0.0, comment_prob=0.08, up_base=-1.2, up_detail=0.4),
}
