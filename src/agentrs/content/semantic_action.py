"""Semantic actions: an operation together with the complete content it carries.

    Action = (operation, generated_content, parameters)

A ``SemanticAction`` is the record of one content-bearing action. It wraps the executable
typed action (``Action.submit(operation, content)``, whose canonical string the world model
reads, so two actions that differ only in content are different inputs) and keeps everything
learning needs: the complete content (never only a summary), its modality and format,
structured parameters, the target environment, the goal/context it was generated for, the
generating checkpoint, the exact sampled token ids with their per-token log-probabilities,
the log-probability of the chosen operation, and execution status / permission decisions.

Arbitrary-length content: the full text is always kept in ``content``; ``content_ref`` is its
SHA-256 (the ExperienceStore blob key when content is stored out of line), and
``ContentActionEncoder`` splits long content into token chunks so that models can read all of
it within a bounded context.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional, Sequence

from pydantic import BaseModel, Field

from agentrs.actions.schema import Action, ActionType

MODALITIES = ("text", "code", "structured", "email", "document")


class SemanticAction(BaseModel):
    operation: str  # e.g. send_email, post_reply, submit_code, apply_config, write_document
    content: str  # complete generated content (original, unsummarised)
    content_ref: str = ""  # sha256 of content (blob key for out-of-line storage)
    modality: str = "text"
    format: str = "plain"  # plain | expression | key_value | markdown | ...
    parameters: dict[str, Any] = Field(default_factory=dict)
    target_env: str = ""
    goal: str = ""
    context: str = ""  # the belief prompt / state description the content was generated for
    generation_prompt: Optional[str] = None  # exact text the content tokens were conditioned on
    policy_checkpoint: Optional[str] = None
    op_candidates: list[str] = Field(default_factory=list)
    op_index: Optional[int] = None
    op_logprob: Optional[float] = None  # log pi(operation | state)
    token_ids: Optional[list[int]] = None  # exact sampled content tokens (incl. terminator)
    token_logprobs: Optional[list[float]] = None  # behaviour log-prob of each sampled token
    sampling_temperature: Optional[float] = None
    source: str = "policy"  # policy | planner | teacher | human | dataset | imagined
    execution_status: str = "proposed"  # proposed | executed | declined | denied
    permission: str = ""  # decision of the permission layer, if any
    truncated_generation: bool = False  # generation hit the token budget before the terminator

    def model_post_init(self, __context) -> None:
        if not self.content_ref:
            self.content_ref = content_hash(self.content)

    @property
    def content_logprob(self) -> Optional[float]:
        return float(sum(self.token_logprobs)) if self.token_logprobs is not None else None

    @property
    def total_logprob(self) -> Optional[float]:
        if self.op_logprob is None or self.content_logprob is None:
            return None
        return self.op_logprob + self.content_logprob

    @property
    def has_behavior_logprobs(self) -> bool:
        return bool(self.token_ids) and self.token_logprobs is not None and len(self.token_ids) == len(
            self.token_logprobs) and self.op_logprob is not None and self.generation_prompt is not None

    def to_action(self) -> Action:
        return Action.submit(self.operation, self.content)

    @classmethod
    def from_action(cls, a: Action, **kw) -> "SemanticAction":
        if a.type != ActionType.SUBMIT:
            raise ValueError("not a content action")
        return cls(operation=a.name or "", content=a.text or "", **kw)


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def unescape_generated(raw: str) -> str:
    """Generated content is produced inside a quoted string; undo the escapes the canonical
    action grammar uses (\\n, \\", \\\\)."""
    out, i = [], 0
    while i < len(raw):
        c = raw[i]
        if c == "\\" and i + 1 < len(raw):
            out.append({"n": "\n", '"': '"', "\\": "\\"}.get(raw[i + 1], raw[i + 1]))
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def content_from_generation(text: str, terminator: str = '")') -> tuple[str, bool]:
    """Split generated text at the closing terminator. Returns (content, terminated)."""
    k = text.find(terminator)
    if k < 0:
        return unescape_generated(text), False
    return unescape_generated(text[:k]), True


class ContentActionEncoder:
    """Renders complete actions for the models and chunks long content.

    ``render`` gives the canonical action text (what the world model's prompt contains).
    ``chunks`` splits content into pieces of at most ``chunk_tokens`` tokens so that a model
    with a bounded context still reads ALL of the content (chunk representations are pooled by
    the content-conditioned world model)."""

    def __init__(self, tokenizer=None, chunk_tokens: int = 160):
        self.tokenizer = tokenizer
        self.chunk_tokens = chunk_tokens

    def render(self, sa: SemanticAction) -> str:
        return sa.to_action().to_str()

    def n_tokens(self, text: str) -> int:
        if self.tokenizer is None:
            return len(text.split())
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def chunks(self, content: str) -> list[str]:
        if self.tokenizer is None:
            words = content.split(" ")
            return [" ".join(words[i: i + self.chunk_tokens]) for i in range(0, max(1, len(words)),
                                                                             self.chunk_tokens)]
        ids = self.tokenizer(content, add_special_tokens=False)["input_ids"]
        if not ids:
            return [""]
        return [self.tokenizer.decode(ids[i: i + self.chunk_tokens]) for i in range(0, len(ids), self.chunk_tokens)]

    def chunked_action_texts(self, sa_or_action: SemanticAction | Action) -> list[str]:
        a = sa_or_action.to_action() if isinstance(sa_or_action, SemanticAction) else sa_or_action
        if a.type != ActionType.SUBMIT or self.n_tokens(a.text or "") <= self.chunk_tokens:
            return [a.to_str()]
        parts = self.chunks(a.text or "")
        return [f'submit({a.name}, [part {i + 1}/{len(parts)}] "{p}")' for i, p in enumerate(parts)]


def summarize_for_log(contents: Sequence[str], width: int = 60) -> list[str]:
    return [c.replace("\n", " ")[:width] for c in contents]
