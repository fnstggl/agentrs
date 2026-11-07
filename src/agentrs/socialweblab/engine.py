"""SocialWebLab backend: persistent multi-user application state, service layer, background jobs
and the clock that drives them.

* State lives in one SQLite database (users, follows, spaces, items + versions, responses,
  reactions, views, notifications, sessions, background jobs, analytics events). Restarting the
  process on the same file resumes the application where it was.
* Every state change goes through the service methods below (which check authorisation: only the
  author edits/deletes/accepts, only moderators moderate, one reaction/view per user and item).
  The HTTP handlers (``server.py``) and the simulated users (``actors.py``) call the same methods.
* Background jobs (simulated users' browsing sessions, auto-moderation, moderator reviews, CI runs,
  notification digests) sit in a persistent queue with a ``run_at`` time. ``advance(seconds)``
  executes them in time order under a **virtual clock** (one simulated week in seconds of compute,
  deterministic given the seed and the sequence of external requests); with a **real clock** a
  worker thread executes them as wall time passes. The clock is part of the database, so snapshots
  capture it.
* Public vs privileged: ``public_item`` / ``public_events`` / ``public_user`` return what any user
  of the application can see; content features, answer helpfulness, CI ground truth and the job
  queue are privileged (verifiers and tests only).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import secrets
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from agentrs.socialweblab.actors import (
    ANSWERS,
    COMMENTS,
    REVIEWS,
    STYLES,
    WEAK_ANSWERS,
    Persona,
    activity,
    make_population,
    write_item,
)
from agentrs.socialweblab.apps import APP_OPERATION, RULES, SPACES, content_features

LAB_EPOCH = 1767571200.0  # 2026-01-05 00:00:00 UTC (a Monday)
DEFAULT_SPACES = list(SPACES)

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, handle TEXT UNIQUE NOT NULL, kind TEXT NOT NULL,
  role TEXT, pw_hash TEXT, created_at REAL, persona_json TEXT);
CREATE TABLE IF NOT EXISTS follows (follower INTEGER, followee INTEGER, at REAL, PRIMARY KEY (follower, followee));
CREATE INDEX IF NOT EXISTS follows_followee ON follows(followee);
CREATE TABLE IF NOT EXISTS spaces (name TEXT PRIMARY KEY, app TEXT, keywords TEXT);
CREATE TABLE IF NOT EXISTS subs (user_id INTEGER, space TEXT, PRIMARY KEY (user_id, space));
CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY, space TEXT, author INTEGER, title TEXT, body TEXT,
  state TEXT, status TEXT, created_at REAL, edited_at REAL, version INTEGER, accepted_response INTEGER,
  n_responses INTEGER DEFAULT 0, n_answers INTEGER DEFAULT 0, ups INTEGER DEFAULT 0, downs INTEGER DEFAULT 0,
  views INTEGER DEFAULT 0, reports INTEGER DEFAULT 0, ci TEXT, client_ref TEXT);
CREATE INDEX IF NOT EXISTS items_space ON items(space, created_at);
CREATE INDEX IF NOT EXISTS items_author ON items(author);
CREATE TABLE IF NOT EXISTS item_versions (item_id INTEGER, version INTEGER, title TEXT, body TEXT, at REAL);
CREATE TABLE IF NOT EXISTS item_truth (item_id INTEGER PRIMARY KEY, features TEXT, ci_pass INTEGER);
CREATE TABLE IF NOT EXISTS responses (id INTEGER PRIMARY KEY, item_id INTEGER, author INTEGER, kind TEXT, body TEXT,
  created_at REAL, state TEXT DEFAULT 'visible');
CREATE INDEX IF NOT EXISTS responses_item ON responses(item_id);
CREATE TABLE IF NOT EXISTS response_truth (response_id INTEGER PRIMARY KEY, helpful INTEGER);
CREATE TABLE IF NOT EXISTS reactions (item_id INTEGER, user_id INTEGER, kind TEXT, at REAL, PRIMARY KEY (item_id, user_id));
CREATE TABLE IF NOT EXISTS views (item_id INTEGER, user_id INTEGER, at REAL, PRIMARY KEY (item_id, user_id));
CREATE TABLE IF NOT EXISTS notifications (id INTEGER PRIMARY KEY, user_id INTEGER, kind TEXT, item_id INTEGER,
  response_id INTEGER, created_at REAL, read INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS notif_user ON notifications(user_id, read);
CREATE TABLE IF NOT EXISTS jobs (id INTEGER PRIMARY KEY, run_at REAL, kind TEXT, payload TEXT,
  status TEXT DEFAULT 'pending', done_at REAL);
CREATE INDEX IF NOT EXISTS jobs_due ON jobs(status, run_at, id);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, at REAL, type TEXT, actor INTEGER, item_id INTEGER,
  response_id INTEGER, public INTEGER, payload TEXT);
CREATE INDEX IF NOT EXISTS events_item ON events(item_id);
CREATE TABLE IF NOT EXISTS sessions (token TEXT PRIMARY KEY, user_id INTEGER, created_at REAL);
"""


class LabError(RuntimeError):
    """A request the application refuses (not found, not allowed, invalid)."""

    def __init__(self, msg: str, status: int = 400):
        super().__init__(msg)
        self.status = status


def _pw(pw: str, salt: str) -> str:
    return salt + "$" + hashlib.sha256((salt + pw).encode()).hexdigest()


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class SocialWebLab:
    def __init__(self, path: str | Path, seed: int = 0, clock: str = "virtual", start_time: float = LAB_EPOCH,
                 spaces: list[str] | None = None, n_users: int = 60, populate: bool = True, lab_id: str | None = None,
                 rules_override: dict | None = None, feed_k: int = 8):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(SCHEMA)
        self.feed_k = feed_k
        self.rules = {k: v for k, v in RULES.items()}
        for app, kv in (rules_override or {}).items():
            r = self.rules[app]
            self.rules[app] = type(r)(**{**r.__dict__, **kv})
        fresh = self._meta("seed") is None
        if fresh:
            self._set_meta("seed", str(seed))
            self._set_meta("clock_mode", clock)
            self._set_meta("now", repr(float(start_time if clock == "virtual" else time.time())))
            self._set_meta("lab_id", lab_id or secrets.token_hex(4))
            self._set_meta("schema", "1")
        self.seed = int(self._meta("seed"))
        self.clock_mode = self._meta("clock_mode")
        self.lab_id = self._meta("lab_id")
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        if fresh:
            for s in spaces or DEFAULT_SPACES:
                self._x("INSERT INTO spaces VALUES (?,?,?)", (s, SPACES[s][0], json.dumps(SPACES[s][1])))
            if populate:
                self._populate(n_users)

    # ------------------------------------------------------------------ plumbing
    @property
    def time_base(self) -> str:
        return f"virtual:socialweblab-{self.lab_id}" if self.clock_mode == "virtual" else "real"

    def _q(self, sql, args=()):
        with self._lock:
            return self._db.execute(sql, tuple(args)).fetchall()

    def _x(self, sql, args=()):
        with self._lock:
            cur = self._db.execute(sql, tuple(args))
            if sql.lstrip().upper().startswith("INSERT"):
                return cur.lastrowid if cur.rowcount > 0 else 0
            return cur.rowcount

    def _meta(self, k):
        r = self._q("SELECT v FROM meta WHERE k=?", (k,))
        return r[0]["v"] if r else None

    def _set_meta(self, k, v):
        self._x("INSERT INTO meta VALUES (?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, str(v)))

    def now(self) -> float:
        if self.clock_mode == "real":
            return time.time()
        return float(self._meta("now"))

    def spaces(self) -> list[dict]:
        return [{"name": r["name"], "app": r["app"]} for r in self._q("SELECT name, app FROM spaces ORDER BY name")]

    def close(self) -> None:
        self.stop_worker()
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------------ population
    def _populate(self, n_users: int) -> None:
        spaces = [s["name"] for s in self.spaces()]
        rng = random.Random(self.seed * 31 + 1)
        personas = make_population(n_users, self.seed, spaces)
        now = self.now()
        with self._lock:
            self._db.execute("BEGIN")
            for p in personas:
                uid = self._db.execute("INSERT INTO users (handle, kind, role, pw_hash, created_at, persona_json) VALUES "
                                       "(?,?,?,?,?,?)", (p.handle, "simulated", p.role, None, now,
                                                         json.dumps(p.__dict__))).lastrowid
                for s in p.spaces:
                    self._db.execute("INSERT OR IGNORE INTO subs VALUES (?,?)", (uid, s))
            ids = [r["id"] for r in self._db.execute("SELECT id, role FROM users").fetchall()]
            roles = {r["id"]: r["role"] for r in self._db.execute("SELECT id, role FROM users").fetchall()}
            # heavy-tailed follow graph: veterans and experts attract followers
            weight = {u: {"veteran": 8.0, "expert": 4.0, "spammer": 0.3, "regular": 1.0, "newbie": 0.5}.get(roles[u], 0.2)
                      for u in ids}
            for u in ids:
                if roles[u] in ("bot",):
                    continue
                k = rng.randint(1, 6)
                cands = [v for v in ids if v != u and roles[v] not in ("bot", "moderator", "staff", "maintainer")]
                ws = [weight[v] for v in cands]
                for v in set(rng.choices(cands, weights=ws, k=k)):
                    self._db.execute("INSERT OR IGNORE INTO follows VALUES (?,?,?)", (u, v, now))
            self._db.execute("COMMIT")
        for p in personas:
            if p.sessions_per_day > 0:
                uid = self._uid(p.handle)
                self._schedule_session(uid, p, now, rng)

    def _uid(self, handle: str) -> int:
        r = self._q("SELECT id FROM users WHERE handle=?", (handle,))
        if not r:
            raise LabError(f"no user {handle}", 404)
        return r[0]["id"]

    def _persona(self, uid: int) -> Persona | None:
        r = self._q("SELECT persona_json FROM users WHERE id=?", (uid,))
        if not r or not r[0]["persona_json"]:
            return None
        return Persona(**json.loads(r[0]["persona_json"]))

    def followers(self, uid: int) -> int:
        return self._q("SELECT COUNT(*) AS n FROM follows WHERE followee=?", (uid,))[0]["n"]

    # ------------------------------------------------------------------ accounts (test accounts for agents)
    def create_account(self, handle: str, password: str, kind: str = "agent", spaces: list[str] | None = None,
                       followers: int = 0, follower_seed: int = 0) -> int:
        """Test account for the agent (or a human tester). ``followers`` simulated users are made to
        follow it (an experimental audience-size condition)."""
        salt = secrets.token_hex(4)
        uid = self._x("INSERT INTO users (handle, kind, role, pw_hash, created_at, persona_json) VALUES (?,?,?,?,?,NULL)",
                      (handle, kind, kind, _pw(password, salt), self.now()))
        for s in spaces or []:
            self._x("INSERT OR IGNORE INTO subs VALUES (?,?)", (uid, s))
        if followers:
            self.set_followers(uid, followers, follower_seed)
        return uid

    def set_followers(self, uid: int, n: int, seed: int = 0) -> None:
        sims = [r["id"] for r in self._q("SELECT id FROM users WHERE kind='simulated' AND role NOT IN "
                                         "('bot','moderator','staff','maintainer') ORDER BY id")]
        rng = random.Random(seed * 7 + uid)
        self._x("DELETE FROM follows WHERE followee=?", (uid,))
        for v in rng.sample(sims, min(n, len(sims))):
            self._x("INSERT OR IGNORE INTO follows VALUES (?,?,?)", (v, uid, self.now()))

    def login(self, handle: str, password: str) -> str:
        r = self._q("SELECT id, pw_hash FROM users WHERE handle=?", (handle,))
        if not r or not r[0]["pw_hash"]:
            raise LabError("invalid credentials", 401)
        salt = r[0]["pw_hash"].split("$", 1)[0]
        if not secrets.compare_digest(_pw(password, salt), r[0]["pw_hash"]):
            raise LabError("invalid credentials", 401)
        tok = secrets.token_urlsafe(18)
        self._x("INSERT INTO sessions VALUES (?,?,?)", (tok, r[0]["id"], self.now()))
        return tok

    def user_for_token(self, token: str | None) -> int | None:
        if not token:
            return None
        r = self._q("SELECT user_id FROM sessions WHERE token=?", (token,))
        return r[0]["user_id"] if r else None

    def handle(self, uid: int) -> str:
        r = self._q("SELECT handle FROM users WHERE id=?", (uid,))
        return r[0]["handle"] if r else "?"

    # ------------------------------------------------------------------ events
    def _event(self, type_: str, actor: int | None, item: int | None, response: int | None, public: bool,
               payload: dict | None = None, at: float | None = None) -> int:
        return self._x("INSERT INTO events (at, type, actor, item_id, response_id, public, payload) VALUES (?,?,?,?,?,?,?)",
                       (at if at is not None else self.now(), type_, actor, item, response, int(public),
                        json.dumps(payload or {})))

    def _notify(self, uid: int, kind: str, item: int, response: int | None = None) -> None:
        self._x("INSERT INTO notifications (user_id, kind, item_id, response_id, created_at) VALUES (?,?,?,?,?)",
                (uid, kind, item, response, self.now()))

    # ------------------------------------------------------------------ service layer (authorised operations)
    def _item(self, item_id: int) -> sqlite3.Row:
        r = self._q("SELECT * FROM items WHERE id=?", (item_id,))
        if not r:
            raise LabError("no such item", 404)
        return r[0]

    def create_item(self, uid: int, space: str, title: str, body: str, client_ref: str | None = None) -> int:
        if not title.strip() or not body.strip():
            raise LabError("title and body are required")
        if len(title) > 200 or len(body) > 20000:
            raise LabError("too long")
        sp = self._q("SELECT app FROM spaces WHERE name=?", (space,))
        if not sp:
            raise LabError("no such space", 404)
        if client_ref:
            dup = self._q("SELECT id FROM items WHERE author=? AND client_ref=?", (uid, client_ref))
            if dup:  # idempotent submission (double click / retried request)
                return dup[0]["id"]
        app = sp[0]["app"]
        now = self.now()
        with self._lock:
            iid = self._x("INSERT INTO items (space, author, title, body, state, status, created_at, edited_at, version, "
                          "client_ref) VALUES (?,?,?,?,?,?,?,NULL,1,?)",
                          (space, uid, title.strip(), body.strip(), "visible", "open", now, client_ref))
            self._x("INSERT INTO item_versions VALUES (?,?,?,?,?)", (iid, 1, title, body, now))
            f = content_features(title + " " + body, space)
            ci_pass = None
            if app == "codereview":
                rng = random.Random(f"{self.seed}:ci:{iid}")
                ok = f.code and f.has_tests and not f.todo
                ci_pass = int(ok if rng.random() > 0.1 else not ok)
            self._x("INSERT INTO item_truth VALUES (?,?,?)", (iid, json.dumps(f.as_dict()), ci_pass))
            self._event("item_created", uid, iid, None, True,
                        {"space": space, "app": app, "operation": APP_OPERATION[app], "title": title, "body": body,
                         "author": self.handle(uid), "author_followers": self.followers(uid)})
            self._job(now + 600 + random.Random(f"{self.seed}:am:{iid}").uniform(0, 120), "automod", {"item": iid})
            if app == "codereview":
                dur = 300 + 60 * min(20, f.length_words / 15)
                self._job(now + dur, "ci", {"item": iid})
            for (fo,) in self._q("SELECT follower FROM follows WHERE followee=?", (uid,)):
                if random.Random(f"{self.seed}:nf:{iid}:{fo}").random() < 0.5:
                    self._notify(fo, "followed_posted", iid)
        return iid

    def respond(self, uid: int, item_id: int, body: str, kind: str = "comment", helpful: bool | None = None) -> int:
        it = self._item(item_id)
        if it["state"] not in ("visible",):
            raise LabError("item is not open for responses", 409)
        if kind not in ("comment", "answer", "ci", "review"):
            raise LabError("bad response kind")
        if not body.strip():
            raise LabError("empty response")
        now = self.now()
        rid = self._x("INSERT INTO responses (item_id, author, kind, body, created_at) VALUES (?,?,?,?,?)",
                      (item_id, uid, kind, body.strip(), now))
        if helpful is not None:
            self._x("INSERT INTO response_truth VALUES (?,?)", (rid, int(helpful)))
        self._x("UPDATE items SET n_responses=n_responses+1, n_answers=n_answers+? WHERE id=?",
                (1 if kind == "answer" else 0, item_id))
        by_author = uid == it["author"]
        self._event("response_created", uid, item_id, rid, True,
                    {"kind": kind, "body": body[:500], "author": self.handle(uid), "by_author": by_author})
        if kind == "answer" and not by_author:
            self._event("answer_posted", uid, item_id, rid, True, {"author": self.handle(uid)})
        if not by_author:
            self._notify(it["author"], "response", item_id, rid)
        return rid

    def react(self, uid: int, item_id: int, kind: str) -> bool:
        if kind not in ("up", "down"):
            raise LabError("bad reaction")
        it = self._item(item_id)
        if it["state"] != "visible":
            raise LabError("item not visible", 409)
        if uid == it["author"]:
            raise LabError("cannot react to own item", 403)
        n = self._x("INSERT OR IGNORE INTO reactions VALUES (?,?,?,?)", (item_id, uid, kind, self.now()))
        if not n:
            return False
        self._x(f"UPDATE items SET {'ups' if kind == 'up' else 'downs'}={'ups' if kind == 'up' else 'downs'}+1 WHERE id=?",
                (item_id,))
        self._event(f"reaction_{kind}", uid, item_id, None, True, {})
        return True

    def view(self, uid: int | None, item_id: int) -> None:
        if uid is None:
            return
        n = self._x("INSERT OR IGNORE INTO views VALUES (?,?,?)", (item_id, uid, self.now()))
        if n:
            self._x("UPDATE items SET views=views+1 WHERE id=?", (item_id,))

    def edit_item(self, uid: int, item_id: int, body: str, title: str | None = None) -> int:
        it = self._item(item_id)
        if it["author"] != uid:
            raise LabError("only the author can edit", 403)
        if it["state"] in ("removed", "deleted"):
            raise LabError("item was removed", 409)
        v = it["version"] + 1
        title = title or it["title"]
        self._x("UPDATE items SET body=?, title=?, version=?, edited_at=? WHERE id=?", (body, title, v, self.now(), item_id))
        self._x("INSERT INTO item_versions VALUES (?,?,?,?,?)", (item_id, v, title, body, self.now()))
        self._x("UPDATE item_truth SET features=? WHERE item_id=?",
                (json.dumps(content_features(title + " " + body, it["space"]).as_dict()), item_id))
        self._event("item_edited", uid, item_id, None, True, {"title": title, "body": body, "version": v})
        return v

    def delete_item(self, uid: int, item_id: int) -> None:
        it = self._item(item_id)
        if it["author"] != uid:
            raise LabError("only the author can delete", 403)
        if it["state"] == "deleted":
            return
        self._x("UPDATE items SET state='deleted' WHERE id=?", (item_id,))
        self._event("item_deleted", uid, item_id, None, True, {"state": "deleted"})

    def accept(self, uid: int, item_id: int, response_id: int) -> None:
        it = self._item(item_id)
        if it["author"] != uid:
            raise LabError("only the author can accept an answer", 403)
        r = self._q("SELECT * FROM responses WHERE id=? AND item_id=?", (response_id, item_id))
        if not r:
            raise LabError("no such response", 404)
        if it["status"] == "resolved":
            return
        self._x("UPDATE items SET status='resolved', accepted_response=? WHERE id=?", (response_id, item_id))
        self._event("item_resolved", uid, item_id, response_id, True, {})

    def report(self, uid: int, item_id: int) -> None:
        self._x("UPDATE items SET reports=reports+1 WHERE id=?", (item_id,))
        it = self._item(item_id)
        self._event("item_reported", uid, item_id, None, False, {})
        if it["reports"] >= 2 and it["state"] == "visible":
            self._set_state(item_id, "hidden", None, "community reports")

    def _set_state(self, item_id: int, state: str, actor: int | None, why: str) -> None:
        self._x("UPDATE items SET state=? WHERE id=?", (state, item_id))
        typ = {"hidden": "item_hidden", "removed": "item_removed", "visible": "item_restored"}[state]
        self._event(typ, actor, item_id, None, True, {"state": state, "reason": why})

    def moderate(self, uid: int, item_id: int, decision: str) -> None:
        r = self._q("SELECT role FROM users WHERE id=?", (uid,))
        if not r or r[0]["role"] != "moderator":
            raise LabError("moderators only", 403)
        if decision not in ("remove", "restore"):
            raise LabError("bad decision")
        self._set_state(item_id, "removed" if decision == "remove" else "visible", uid, "moderator review")

    def merge(self, uid: int, item_id: int) -> None:
        r = self._q("SELECT role FROM users WHERE id=?", (uid,))
        if not r or r[0]["role"] != "maintainer":
            raise LabError("maintainers only", 403)
        it = self._item(item_id)
        if it["ci"] != "passed":
            raise LabError("CI has not passed", 409)
        self._x("UPDATE items SET status='resolved' WHERE id=?", (item_id,))
        self._event("item_resolved", uid, item_id, None, True, {"merged": True})

    def mark_read(self, uid: int) -> int:
        return self._x("UPDATE notifications SET read=1 WHERE user_id=? AND read=0", (uid,))

    # ------------------------------------------------------------------ queries (public views)
    def public_item(self, item_id: int, viewer: int | None = None) -> dict:
        it = self._item(item_id)
        if viewer is not None:
            self.view(viewer, item_id)
            it = self._item(item_id)
        gone = it["state"] in ("removed", "deleted", "hidden")
        resp = [{"id": r["id"], "kind": r["kind"], "author": self.handle(r["author"]), "body": r["body"],
                 "created_at": r["created_at"], "by_author": r["author"] == it["author"],
                 "accepted": r["id"] == it["accepted_response"]}
                for r in self._q("SELECT * FROM responses WHERE item_id=? ORDER BY id", (item_id,))]
        return {"id": it["id"], "space": it["space"], "author": self.handle(it["author"]),
                "author_followers": self.followers(it["author"]),
                "title": "[removed]" if gone else it["title"], "body": "[removed]" if gone else it["body"],
                "state": it["state"], "status": it["status"], "created_at": it["created_at"], "edited_at": it["edited_at"],
                "version": it["version"], "ci": it["ci"], "counts": {"responses": it["n_responses"],
                "answers": it["n_answers"], "reactions": it["ups"], "downs": it["downs"], "views": it["views"]},
                "responses": [] if gone else resp, "accepted_response": it["accepted_response"], "now": self.now()}

    def public_events(self, since_id: int = 0, limit: int = 200) -> list[dict]:
        rows = self._q("SELECT * FROM events WHERE public=1 AND id>? ORDER BY id LIMIT ?", (since_id, limit))
        out = []
        for r in rows:
            p = json.loads(r["payload"])
            out.append({"id": r["id"], "at": r["at"], "type": r["type"], "actor": self.handle(r["actor"]) if r["actor"]
                        else "system", "item_id": r["item_id"], "response_id": r["response_id"], "payload": p})
        return out

    def public_user(self, handle: str) -> dict:
        uid = self._uid(handle)
        n_items = self._q("SELECT COUNT(*) AS n FROM items WHERE author=?", (uid,))[0]["n"]
        return {"handle": handle, "followers": self.followers(uid), "items": n_items}

    def feed(self, uid: int | None, space: str | None = None, limit: int = 20, sort: str = "hot") -> list[dict]:
        now = self.now()
        if space:
            spaces = [space]
        elif uid is not None:
            spaces = [r["space"] for r in self._q("SELECT space FROM subs WHERE user_id=?", (uid,))]
        else:
            spaces = [s["name"] for s in self.spaces()]
        if not spaces:
            return []
        ph = ",".join("?" * len(spaces))
        rows = self._q(f"SELECT * FROM items WHERE state='visible' AND space IN ({ph}) AND created_at>? AND created_at<=? "
                       f"ORDER BY created_at DESC LIMIT 300", (*spaces, now - 7 * 86400, now))
        followees = set()
        if uid is not None:
            followees = {r["followee"] for r in self._q("SELECT followee FROM follows WHERE follower=?", (uid,))}
            if not space:
                fr = self._q("SELECT * FROM items WHERE state='visible' AND author IN (SELECT followee FROM follows WHERE "
                             "follower=?) AND created_at>? AND created_at<=? ORDER BY created_at DESC LIMIT 50",
                             (uid, now - 7 * 86400, now))
                seen = {r["id"] for r in rows}
                rows = list(rows) + [r for r in fr if r["id"] not in seen]
        fol = {}

        def score(r):
            if r["author"] not in fol:
                fol[r["author"]] = self.followers(r["author"])
            age_h = max(0.0, (now - r["created_at"]) / 3600)
            s = (1 + r["ups"] + 2 * r["n_responses"]) / (age_h + 2) ** 1.3 * (1 + 0.15 * math.log1p(fol[r["author"]]))
            return s * (3.0 if r["author"] in followees else 1.0)

        rows = sorted(rows, key=(score if sort == "hot" else (lambda r: r["created_at"])), reverse=True)[:limit]
        return [{"id": r["id"], "space": r["space"], "title": r["title"], "author": self.handle(r["author"]),
                 "created_at": r["created_at"], "responses": r["n_responses"], "reactions": r["ups"],
                 "views": r["views"], "status": r["status"]} for r in rows]

    def search(self, query: str, space: str | None = None, limit: int = 20) -> list[dict]:
        q = f"%{query.strip()}%"
        sql = "SELECT * FROM items WHERE state='visible' AND (title LIKE ? OR body LIKE ?) AND created_at<=?"
        args: list[Any] = [q, q, self.now()]
        if space:
            sql += " AND space=?"
            args.append(space)
        rows = self._q(sql + " ORDER BY created_at DESC LIMIT ?", (*args, limit))
        return [{"id": r["id"], "space": r["space"], "title": r["title"], "author": self.handle(r["author"]),
                 "created_at": r["created_at"], "responses": r["n_responses"]} for r in rows]

    def notifications(self, uid: int, unread_only: bool = False) -> list[dict]:
        sql = "SELECT * FROM notifications WHERE user_id=?" + (" AND read=0" if unread_only else "")
        return [dict(r) for r in self._q(sql + " ORDER BY id DESC LIMIT 50", (uid,))]

    # ------------------------------------------------------------------ background jobs and clock
    def _job(self, run_at: float, kind: str, payload: dict) -> int:
        return self._x("INSERT INTO jobs (run_at, kind, payload) VALUES (?,?,?)", (run_at, kind, json.dumps(payload)))

    def _schedule_session(self, uid: int, p: Persona, after: float, rng: random.Random) -> None:
        rate = max(1e-6, p.sessions_per_day / 86400.0 / 0.42)  # / mean diurnal activity: thinning keeps the daily rate
        t = after
        for _ in range(200):  # thinning for the diurnal profile
            t += rng.expovariate(rate)
            hour = ((t - LAB_EPOCH) % 86400) / 3600
            if p.role == "staff":
                wd = int((t - LAB_EPOCH) // 86400) % 7
                if wd >= 5 or not (9 <= hour < 17):
                    continue
                break
            if rng.random() < activity(hour, p.peak_hour):
                break
        self._job(t, "session", {"user": uid})

    def process_due(self, until: float, max_jobs: int = 1_000_000) -> int:
        n = 0
        while n < max_jobs:
            with self._lock:
                r = self._q("SELECT * FROM jobs WHERE status='pending' AND run_at<=? ORDER BY run_at, id LIMIT 1", (until,))
                if not r:
                    break
                job = r[0]
                if self.clock_mode == "virtual":
                    self._set_meta("now", repr(max(self.now(), float(job["run_at"]))))
                self._x("UPDATE jobs SET status='running' WHERE id=?", (job["id"],))
                try:
                    self._run_job(job)
                    self._x("UPDATE jobs SET status='done', done_at=? WHERE id=?", (self.now(), job["id"]))
                except LabError:
                    self._x("UPDATE jobs SET status='done', done_at=? WHERE id=?", (self.now(), job["id"]))
            n += 1
        return n

    def advance(self, seconds: float) -> int:
        """Virtual clock only: run every job due in the next ``seconds`` in time order."""
        if self.clock_mode != "virtual":
            raise LabError("advance() is only available with the virtual clock")
        target = self.now() + seconds
        n = self.process_due(target)
        self._set_meta("now", repr(max(self.now(), target)))
        return n

    def advance_to(self, t: float) -> int:
        return self.advance(max(0.0, t - self.now()))

    def start_worker(self, poll_s: float = 0.25) -> None:
        if self.clock_mode != "real" or self._worker is not None:
            return
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                try:
                    self.process_due(time.time(), max_jobs=200)
                except Exception:  # pragma: no cover - keep the worker alive
                    pass
                self._stop.wait(poll_s)

        self._worker = threading.Thread(target=loop, name="socialweblab-worker", daemon=True)
        self._worker.start()

    def stop_worker(self) -> None:
        if self._worker is not None:
            self._stop.set()
            self._worker.join(timeout=5)
            self._worker = None

    # ------------------------------------------------------------------ job handlers
    def _run_job(self, job: sqlite3.Row) -> None:
        p = json.loads(job["payload"])
        rng = random.Random(f"{self.seed}:job:{job['id']}")
        kind = job["kind"]
        if kind == "session":
            uid = p["user"]
            persona = self._persona(uid)
            if persona is None:
                return
            self._session(uid, persona, rng)
            self._schedule_session(uid, persona, self.now(), rng)
        elif kind == "automod":
            it = self._item(p["item"])
            f = json.loads(self._q("SELECT features FROM item_truth WHERE item_id=?", (it["id"],))[0]["features"])
            if it["state"] == "visible" and f["spam"]:
                self._set_state(it["id"], "hidden", None, "automatic spam filter")
        elif kind == "ci":
            it = self._item(p["item"])
            if it["state"] != "visible":
                return
            ok = bool(self._q("SELECT ci_pass FROM item_truth WHERE item_id=?", (it["id"],))[0]["ci_pass"])
            self._x("UPDATE items SET ci=? WHERE id=?", ("passed" if ok else "failed", it["id"]))
            bot = self._uid("ci-bot")
            self.respond(bot, it["id"], "CI passed: all checks green." if ok else
                         "CI failed: " + rng.choice(["test_retry_timeout", "lint", "test_render", "build"]) + " failing.",
                         kind="ci")

    def _features(self, item_id: int) -> dict:
        return json.loads(self._q("SELECT features FROM item_truth WHERE item_id=?", (item_id,))[0]["features"])

    def _session(self, uid: int, persona: Persona, rng: random.Random) -> None:
        now = self.now()
        role = persona.role
        if role == "moderator":
            for it in self._q("SELECT id FROM items WHERE state='hidden' ORDER BY id LIMIT 30"):
                f = self._features(it["id"])
                self.moderate(uid, it["id"], "remove" if f["spam"] or f["shouting"] and rng.random() < 0.3 else "restore")
            return
        # 1. own items: accept helpful answers, edit unanswered posts, merge (maintainers)
        if role == "maintainer":
            for it in self._q("SELECT * FROM items WHERE state='visible' AND status='open' AND ci='passed' AND ups>=1"):
                if rng.random() < 0.7:
                    self.merge(uid, it["id"])
        for it in self._q("SELECT * FROM items WHERE author=? AND state='visible' AND status='open' AND created_at>?",
                          (uid, now - 14 * 86400)):
            app = SPACES[it["space"]][0]
            rules = self.rules[app]
            if app in ("forum", "helpdesk"):
                helpful = self._q("SELECT r.id FROM responses r JOIN response_truth t ON t.response_id=r.id WHERE "
                                  "r.item_id=? AND t.helpful=1 ORDER BY r.id LIMIT 1", (it["id"],))
                if helpful and rng.random() < rules.accept_prob:
                    self.accept(uid, it["id"], helpful[0]["id"])
                    continue
            if it["n_responses"] == 0 and it["version"] == 1 and now - it["created_at"] > rules.edit_after_s:
                f = self._features(it["id"])
                if rng.random() < rules.edit_prob * (1.5 if not f["detail"] else 0.5):
                    extra = rng.choice([f" Update: using version {rng.randint(1, 4)}.{rng.randint(0, 9)}.{rng.randint(0, 9)}.",
                                        f" Update: error E{rng.randint(1000, 9999)} on line {rng.randint(3, 99)}."])
                    self.edit_item(uid, it["id"], it["body"] + extra)
                elif rng.random() < rules.delete_prob:
                    self.delete_item(uid, it["id"])
        self.mark_read(uid)
        # 2. browse: hot feed + newest items in subscribed spaces
        hot = self.feed(uid, limit=self.feed_k)
        new = self.feed(uid, limit=3, sort="new")
        seen = set()
        for entry in hot + new:
            if entry["id"] in seen:
                continue
            seen.add(entry["id"])
            it = self._item(entry["id"])
            if it["author"] == uid or it["state"] != "visible":
                continue
            if self._q("SELECT 1 FROM views WHERE item_id=? AND user_id=?", (it["id"], uid)):
                continue
            app = SPACES[it["space"]][0]
            rules = self.rules[app]
            f = self._features(it["id"])
            interest = 1.0 if it["space"] in persona.spaces else 0.0
            p_open = _sig(rules.open_base + rules.open_clickbait * f["clickbait"] + rules.open_shouting * f["shouting"]
                          + rules.open_interest * interest)
            if rng.random() >= p_open:
                continue
            self.view(uid, it["id"])
            expert = it["space"] in persona.expertise
            answered = False
            if (expert or not rules.experts_only_answer) and it["status"] == "open" and role != "bot":
                logit = (rules.answer_base + rules.answer_detail * f["detail"] + rules.answer_question * f["is_question"]
                         + rules.answer_polite * f["polite"] + rules.answer_code * f["code"]
                         + rules.answer_topic * f["topic_match"] + rules.answer_shouting * f["shouting"]
                         + rules.answer_clickbait * f["clickbait"] + rules.answer_spam * f["spam"]
                         + rules.answer_off_topic * f["off_topic"] + rules.answer_per_existing * it["n_answers"]
                         + rules.answer_short * (f["length_words"] < 8))
                if rng.random() < _sig(logit):
                    kws = SPACES[it["space"]][1]
                    k, k2 = rng.sample(kws, 2)
                    helpful = rng.random() < _sig(rules.helpful_base + rules.helpful_detail * f["detail"]
                                                  + rules.helpful_code * f["code"] - 1.5 * f["off_topic"])
                    if app == "codereview":
                        body = rng.choice(REVIEWS).format(k=k, k2=k2)
                    else:
                        body = (rng.choice(ANSWERS) if helpful else rng.choice(WEAK_ANSWERS)).format(k=k, k2=k2)
                    self.respond(uid, it["id"], body, kind="answer", helpful=helpful)
                    answered = True
            if not answered and rng.random() < rules.comment_prob * (1 + f["shouting"]):
                k = rng.choice(SPACES[it["space"]][1])
                self.respond(uid, it["id"], rng.choice(COMMENTS).format(k=k), kind="comment", helpful=False)
            up_logit = rules.up_base + rules.up_detail * f["detail"] + rules.up_polite * f["polite"] + rules.up_spam * f["spam"]
            if app == "codereview" and expert:
                up_logit += 2.0 if it["ci"] == "passed" else -2.0
            u = rng.random()
            if u < _sig(up_logit):
                self.react(uid, it["id"], "up")
            elif u > 1 - (rules.down_shouting * f["shouting"] + rules.down_spam * f["spam"]):
                self.react(uid, it["id"], "down")
            if f["spam"] and rng.random() < rules.report_spam:
                self.report(uid, it["id"])
        # 3. post something new (a third-party action)
        if persona.p_post > 0 and rng.random() < persona.p_post and persona.spaces:
            space = rng.choice(persona.spaces)
            style = STYLES.get(persona.role, STYLES["regular"])
            title, body, _ = write_item(rng, space, style)
            self.create_item(uid, space, title, body)

    # ------------------------------------------------------------------ privileged: verification, snapshots, analytics
    def helpful_answer_within(self, item_id: int, within_s: float) -> dict:
        it = self._item(item_id)
        rows = self._q("SELECT r.id, r.created_at, r.author, t.helpful FROM responses r JOIN response_truth t ON "
                       "t.response_id=r.id WHERE r.item_id=? AND r.kind='answer' AND r.author!=? ORDER BY r.created_at",
                       (item_id, it["author"]))
        hit = [r for r in rows if r["helpful"] and r["created_at"] - it["created_at"] <= within_s]
        return {"success": bool(hit), "first_helpful_at": hit[0]["created_at"] - it["created_at"] if hit else None,
                "n_answers": len(rows), "state": it["state"]}

    def item_truth(self, item_id: int) -> dict:
        r = self._q("SELECT * FROM item_truth WHERE item_id=?", (item_id,))
        return {"features": json.loads(r[0]["features"]), "ci_pass": r[0]["ci_pass"]} if r else {}

    def all_events(self, item_id: int | None = None) -> list[dict]:
        sql, args = "SELECT * FROM events", ()
        if item_id is not None:
            sql, args = sql + " WHERE item_id=?", (item_id,)
        return [dict(r) for r in self._q(sql + " ORDER BY id", args)]

    def items(self, where: str = "1=1", args=()) -> list[dict]:
        return [dict(r) for r in self._q(f"SELECT * FROM items WHERE {where} ORDER BY id", args)]

    def snapshot(self, path: str | Path) -> str:
        with self._lock:
            dst = sqlite3.connect(str(path))
            self._db.backup(dst)
            dst.close()
        return str(path)

    def restore(self, path: str | Path) -> None:
        with self._lock:
            src = sqlite3.connect(str(path))
            src.backup(self._db)
            src.close()
        self.seed = int(self._meta("seed"))

    def stats(self) -> dict:
        r = {}
        for t in ("users", "items", "responses", "reactions", "views", "events", "jobs"):
            r[t] = self._q(f"SELECT COUNT(*) AS n FROM {t}")[0]["n"]
        r["now"] = self.now()
        return r

    @staticmethod
    def copy(src: str | Path, dst: str | Path) -> None:
        shutil.copyfile(src, dst)
