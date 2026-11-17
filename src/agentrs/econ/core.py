"""Shared machinery for the economic task environments.

Every environment instance holds real application state (SQLite tables, files on disk, or a
MiniWeb app), exposes a small set of tools, and has a hidden verifier that inspects the final state.
Actions are single-line JSON tool calls: {"tool": "<name>", "args": {...}}. A task is fully
determined by (domain, family, split, seed), so episodes are reproducible and groups of rollouts
on the same task are possible (GRPO-style baselines).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

SPLITS = ("train", "val", "test")
SPLIT_SEED_BASE = {"train": 0, "val": 1_000_000, "test": 2_000_000}


@dataclass
class Task:
    domain: str
    family: str
    split: str
    seed: int
    goal: str
    private: dict = field(default_factory=dict)  # hidden: verifier expectations, oracle plan
    max_steps: int = 12

    @property
    def id(self) -> str:
        return f"{self.domain}/{self.family}/{self.split}/{self.seed}"

    def fingerprint(self) -> str:
        """Near-duplicate key: goal text with numbers / quoted values normalised."""
        g = re.sub(r"\d+(\.\d+)?", "#", self.goal.lower())
        g = re.sub(r'"[^"]*"', '"_"', g)
        return hashlib.sha1(f"{self.family}|{g}".encode()).hexdigest()[:12]


@dataclass
class StepOut:
    observation: str
    valid: bool = True
    error: Optional[str] = None
    done: bool = False
    info: dict = field(default_factory=dict)


class ToolError(Exception):
    """A well-formed tool call that the application rejected (shown to the agent)."""


@dataclass
class Tool:
    name: str
    fn: Callable[..., Any]
    doc: str  # signature + one-line description shown to the agent
    writes: bool = False
    terminal: bool = False


def parse_tool_call(text: str) -> tuple[Optional[str], Optional[dict], Optional[str]]:
    """Parse the first JSON object in ``text`` as a tool call. Returns (tool, args, error)."""
    s = text.strip()
    i = s.find("{")
    if i < 0:
        return None, None, "no JSON object found"
    try:
        obj, _ = json.JSONDecoder().raw_decode(s[i:])
    except json.JSONDecodeError as e:
        return None, None, f"invalid JSON: {e.msg}"
    if not isinstance(obj, dict) or not isinstance(obj.get("tool"), str):
        return None, None, 'expected {"tool": "<name>", "args": {...}}'
    args = obj.get("args", {})
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None, None, '"args" must be an object'
    return obj["tool"], args, None


def fmt_rows(cur: sqlite3.Cursor, rows: list, limit: int = 40) -> str:
    cols = [d[0] for d in cur.description] if cur.description else []
    if not rows:
        return "(no rows)"
    out = [" | ".join(cols)]
    for r in rows[:limit]:
        out.append(" | ".join("NULL" if v is None else str(v) for v in r))
    if len(rows) > limit:
        out.append(f"... ({len(rows) - limit} more rows)")
    return "\n".join(out)


class ToolEnv:
    """Base class: subclasses build state in ``setup`` and register tools; ``verify`` is hidden."""

    domain = "base"
    role = "an operations agent"
    rules: list[str] = []

    def __init__(self, task: Task):
        self.task = task
        self.rng = random.Random(f"{task.id}:env")
        self.tools: dict[str, Tool] = {}
        self.log: list[dict] = []  # executed tool calls (name, args, ok)
        self.done = False
        self.submitted: Optional[dict] = None
        self.db: Optional[sqlite3.Connection] = None
        self.setup()

    # ----------------------------------------------------------------- subclass API
    def setup(self) -> None:
        raise NotImplementedError

    def verify(self) -> tuple[bool, dict]:
        raise NotImplementedError

    def initial_observation(self) -> str:
        return "Environment ready."

    # ----------------------------------------------------------------- helpers
    def tool(self, name: str, doc: str, writes: bool = False, terminal: bool = False):
        def deco(fn):
            self.tools[name] = Tool(name, fn, doc, writes, terminal)
            return fn
        return deco

    def new_db(self, schema: str) -> sqlite3.Connection:
        self.db = sqlite3.connect(":memory:")
        self.db.executescript(schema)
        return self.db

    def add_sql_tool(self, max_rows: int = 40) -> None:
        def sql(query: str) -> str:
            q = str(query).strip().rstrip(";")
            if not re.match(r"(?is)^\s*(select|with)\b", q) or ";" in q:
                raise ToolError("only a single read-only SELECT statement is allowed")
            self.db.set_authorizer(_read_only_authorizer)
            try:
                cur = self.db.execute(q)
                rows = cur.fetchall()
            except sqlite3.Error as e:
                raise ToolError(f"SQL error: {e}")
            finally:
                self.db.set_authorizer(None)
            return fmt_rows(cur, rows, max_rows)
        self.tools["sql"] = Tool("sql", sql, 'sql(query: str) — run one read-only SELECT over the database')

    def schema_text(self) -> str:
        if self.db is None:
            return ""
        rows = self.db.execute("SELECT name, sql FROM sqlite_master WHERE type='table'").fetchall()
        return "\n".join(r[1].replace("\n", " ") for r in rows)

    # ----------------------------------------------------------------- stepping
    def step(self, action_text: str) -> StepOut:
        name, args, err = parse_tool_call(action_text)
        if err:
            return StepOut(f"Invalid action: {err}", valid=False, error=err)
        tool = self.tools.get(name)
        if tool is None:
            return StepOut(f"Invalid action: unknown tool '{name}'. Available: {', '.join(self.tools)}",
                           valid=False, error="unknown tool")
        try:
            res = tool.fn(**args)
            ok = True
        except TypeError as e:
            self.log.append({"tool": name, "args": args, "ok": False})
            return StepOut(f"Invalid arguments for {name}: {e}", valid=False, error="bad arguments")
        except ToolError as e:
            res, ok = f"Error: {e}", False
        self.log.append({"tool": name, "args": args, "ok": ok})
        if tool.terminal and ok:
            self.done = True
        return StepOut(str(res), valid=True, error=None if ok else str(res)[:200], done=self.done)

    def tool_docs(self) -> str:
        return "\n".join(f"- {t.doc}" for t in self.tools.values())

    def system_prompt(self) -> str:
        parts = [f"You are {self.role}. You complete the task by calling tools, one call per turn.",
                 "TOOLS:", self.tool_docs()]
        if self.rules:
            parts += ["RULES:"] + [f"- {r}" for r in self.rules]
        sch = self.schema_text()
        if sch:
            parts += ["DATABASE SCHEMA:", sch]
        parts += ["Respond with exactly one line of JSON: {\"tool\": \"<name>\", \"args\": {...}}. "
                  "Use only information you have observed. Finish with the terminal tool when done."]
        return "\n".join(parts)


def _read_only_authorizer(action, *args):
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


# ---------------------------------------------------------------------------- pools
FIRST = ["Ava", "Liam", "Noah", "Emma", "Mia", "Lucas", "Zoe", "Ethan", "Ivy", "Omar", "Nina", "Raj", "Sara",
         "Tom", "Yuki", "Leo", "Aria", "Ben", "Chloe", "Diego", "Elena", "Felix", "Grace", "Hugo", "Isla", "Jon",
         "Kira", "Luca", "Maya", "Nils", "Olga", "Pia", "Quinn", "Rosa", "Sven", "Tara", "Umar", "Vera", "Wen",
         "Xena", "Yara", "Zane", "Ada", "Bo", "Cleo", "Dev", "Eli", "Fay"]
LAST = ["Smith", "Garcia", "Chen", "Patel", "Kim", "Nguyen", "Lopez", "Brown", "Ito", "Silva", "Novak", "Haas",
        "Rossi", "Khan", "Moreau", "Berg", "Costa", "Dubois", "Evans", "Fischer", "Gomez", "Hughes", "Ivanov",
        "Jensen", "Klein", "Larsen", "Meyer", "Nowak", "Olsen", "Park", "Quinn", "Reyes", "Sato", "Torres",
        "Ueda", "Vargas", "Weber", "Young", "Zhang", "Abe", "Baker", "Cruz", "Diaz", "Ek", "Ford", "Gil"]
CO_A = ["Blue", "North", "Bright", "Iron", "Silver", "Rapid", "Green", "Prime", "Atlas", "Nova", "Summit", "Coral",
        "Delta", "Echo", "Granite", "Harbor", "Lumen", "Maple", "Orbit", "Pine", "Quartz", "River", "Solar", "Tidal",
        "Vertex", "Willow", "Zenith", "Amber", "Beacon", "Cobalt", "Dune", "Ember", "Falcon", "Glacier"]
CO_B = ["Labs", "Logistics", "Systems", "Foods", "Health", "Robotics", "Analytics", "Energy", "Retail", "Media",
        "Works", "Dynamics", "Networks", "Capital", "Studios", "Materials", "Bio", "Freight", "Software", "Textiles"]
PRODUCTS = ["Desk Lamp", "Office Chair", "Water Bottle", "Backpack", "Running Shoes", "Headphones", "Keyboard",
            "Coffee Grinder", "Yoga Mat", "Rain Jacket", "Monitor Stand", "Wool Sweater", "Travel Mug", "Tent",
            "Bike Helmet", "Blender", "Notebook Set", "Desk Fan", "Wireless Mouse", "Hiking Boots", "Sun Hat",
            "Cutting Board", "Phone Case", "Duffel Bag", "Table Lamp", "Cast Iron Pan", "Throw Blanket", "Sneakers"]
CITIES = ["Austin", "Denver", "Boston", "Seattle", "Chicago", "Portland", "Miami", "Atlanta", "Phoenix", "Dallas",
          "Detroit", "Raleigh", "Tampa", "Omaha", "Tucson", "Fresno", "Boise", "Reno", "Madison", "Albany"]


def pool(items: list, split: str) -> list:
    """Disjoint vocabulary for the sealed test split (held-out instances AND held-out names)."""
    k = (len(items) * 2) // 3
    return items[k:] if split == "test" else items[:k]


def person(rng: random.Random, split: str) -> str:
    return f"{rng.choice(pool(FIRST, split))} {rng.choice(pool(LAST, split))}"


def company(rng: random.Random, split: str) -> str:
    return f"{rng.choice(pool(CO_A, split))} {rng.choice(pool(CO_B, split))}"


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


# ---------------------------------------------------------------------------- registry
@dataclass
class Family:
    domain: str
    name: str
    make_task: Callable[[str, int], Task]          # (split, seed) -> Task
    make_env: Callable[[Task], Any]                # Task -> env
    oracle: Optional[Callable[[Task], list[str]]]  # Task -> action strings (expert demonstration)
    heldout: bool = False                          # never used for policy training (transfer test)

    @property
    def key(self) -> str:
        return f"{self.domain}/{self.name}"


REGISTRY: dict[str, Family] = {}


def register(fam: Family) -> Family:
    REGISTRY[fam.key] = fam
    return fam


def load_all() -> dict[str, Family]:
    from agentrs.econ.domains import browser, crm, finance, research, software, support  # noqa: F401

    return REGISTRY


def make_task(key: str, split: str, index: int) -> Task:
    fam = REGISTRY[key]
    seed = SPLIT_SEED_BASE[split] + index
    return fam.make_task(split, seed)
