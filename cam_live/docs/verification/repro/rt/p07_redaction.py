"""G-rt probe 07 — logs and redaction: plant secrets, tokens, e-mail addresses and member identifiers in what the
environment returns, then search EVERY file of the run directory (mid-run and at the end). Also the log envelope.

    .venv/bin/python cam_live/docs/verification/repro/rt/p07_redaction.py

All planted values are synthetic (made up for this probe); none is a real credential or a real account. The ones
shaped like real credentials are assembled at run time so that no credential-shaped literal is committed.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from standin import LOGS, Report, StandInAgent, check_seq, make_slow_run, read_log  # noqa: E402

from cam_live.contracts import (ActionSpec, AuthorizationDecision, AuthVerdict, Cost, Quantity, RunFiles, SimClock,  # noqa: E402
                                new_id)
from cam_live.diagnostics.world import AllowAllPolicy  # noqa: E402
from cam_live.runtime.journal import write_journal  # noqa: E402

R = Report("p07_redaction")

# label -> (needle searched in files, how it is planted)
P = {
    # --- secrets
    "S01 bot-token-shaped string in page text": ".".join(["MTA1NzYxMjM0NTY3" + "ODkwMTIzNA", "Gh" + "IjKl", "abcdefghijklmnopqrstuvwxyz" + "0123456789AB"]),
    "S02 'Authorization: Bearer <value>' in page text": "abcdef0123456789abcdef0123456789",
    "S03 sk-... API key in page text": "sk-" + "proj-" + "abcdefghijklmnopqrstuvwx",
    "S04 env-style DISCORD_BOT_TOKEN=<value> in page text": "Zx9Qp2Lm7Rt4Vb6Nc8Yd",
    "S05 JSON text {\"access_token\": \"<value>\"} in page text": "Kd83hsLq02mXz71pQw",
    "S06 JSON text {\"password\": \"<value>\"} in page text": "hunter2-Longer-Passphrase",
    "S07 structured data key 'token'": "tok-PLAIN-12345678",
    "S08 structured data key 'client_secret'": "cs-PLAIN-87654321",
    "S09 password inside a URL (https://user:<value>@host)": "Pa55w0rdInUrlXyz",
    "S10 webhook URL": "https://discord.com/api/" + "webhooks/" + "123456789012345678" + "/" + "AbCdEfGhIjKlMnOpQrStUvWxYz",
    "S11 secret carried as bytes in structured data": "sk-" + "live-bytes-" + "0123456789abcdef",
    "S12 'Bearer <value>' in an executor exception message": "excSECRET9f8e7d6c5b4a3210",
    "S13 'token=<value>' in an action argument awaiting approval": "argSECRET0a1b2c3d4e5f6789",
    "S14 structured data key 'DISCORD_TOKEN'": "envkey-PLAIN-55554444",
    "S15 '?access_token=<value>' in a URL in page text": "qs-PLAIN-33332222",
    # --- e-mail
    "E01 e-mail address in page text": "alice.smith@example.org",
    "E02 e-mail address as a mapping key": "bob.jones@example.org",
    "E03 e-mail address in failure_detail": "carol.fail@example.org",
    # --- member identifiers
    "M01 structured key member_id": "111111111111111111",
    "M02 bare snowflake id in text": "222222222222222222",
    "M03 <@mention> in text": "333333333333333333",
    "M04 list of ids under key 'members'": "444444444444444444",
    "M05 user object {id, username}": "555555555555555555",
    "M06 username#discriminator in text": "carolQ#1234",
    "M07 id in Observation.entities": "666666666666666666",
    "M08 id inside a metric name": "888888888888888888",
    "M09 list of {id} objects under key 'joined'": "999999999999999999",
    "M10 id under key 'user_ids'": "121212121212121212",
    # --- content that needs a data permission
    "C01 message_content without the store_message_content permission": "this is a private message body",
}
V = lambda k: [v for kk, v in P.items() if kk.startswith(k)][0]  # noqa: E731

TEXT = (f"Settings page. token on page: {V('S01')} | Authorization: Bearer {V('S02')} | key {V('S03')} | "
        f"DISCORD_BOT_TOKEN={V('S04')} | config: {{\"access_token\": \"{V('S05')}\", \"password\": \"{V('S06')}\"}} | "
        f"repo https://deploy:{V('S09')}@git.example.org/x.git | hook {V('S10')} | "
        f"callback https://api.example.org/cb?access_token={V('S15')} | "
        f"contact {V('E01')} | user {V('M02')} joined | hello <@{V('M03')}> | {V('M06')} said hi")
DATA = {"token": V("S07"), "client_secret": V("S08"), "blob": V("S11").encode(), "DISCORD_TOKEN": V("S14"),
        V("E02"): "owner", "member_id": V("M01"), "members": [V("M04"), "444444444444444445"],
        "user": {"id": V("M05"), "username": "carolQ"}, "joined": [{"id": V("M09")}], "user_ids": [V("M10")],
        "message_content": V("C01")}


def scan(run_dir):
    """needle label -> sorted list of files (relative) containing it."""
    hits = {}
    for root, _dirs, files in os.walk(run_dir):
        for fn in files:
            p = os.path.join(root, fn)
            try:
                blob = open(p, "rb").read().decode("utf-8", "replace")
            except OSError:
                continue
            for label, needle in P.items():
                if needle in blob:
                    hits.setdefault(label, set()).add(os.path.relpath(p, run_dir))
    return {k: sorted(v) for k, v in hits.items()}


class NeedsApproval(AllowAllPolicy):
    def __init__(self):
        super().__init__()
        self.first = True

    def authorize(self, action, task, context=None):
        if action.arguments.get("note"):
            return AuthorizationDecision(AuthVerdict.NEEDS_APPROVAL, action.content_hash(), ["operator approval required"],
                                         approval_id="apr_g1")
        return super().authorize(action, task, context)

    def approvals(self):
        return [{"approval_id": "apr_g1", "status": "pending"}]


class Agent(StandInAgent):
    """First proposes an action whose argument repeats something read from the page; afterwards plain actions."""

    def propose(self, tools, k):
        self._t("propose", tools, k)
        t = {x.name: x for x in tools}["slow.act"]
        args = {"note": f"open with token={V('S13')}"} if self.n_select == 0 else {"i": self.n_select}
        return [ActionSpec(action_id=new_id("act"), operation="slow.act", modality=t.modality, level=t.level,
                           arguments=args, est_cost=Cost(money=0.0), provenance="g-standin")]


clock = SimClock()
L = make_slow_run("p07", clock, agent=Agent(clock, durable=False), policy=NeedsApproval(),
                  env_kw={"text": TEXT, "outcome_horizon": 10 ** 9}, limits={"max_decisions": 6},
                  rt_kw={"checkpoint_every_s": 10 ** 9})        # one commit at start: the journal (WAL) stays populated
n = {"exec": 0}


def edit(res):
    n["exec"] += 1
    o = res.observations[0]
    o.text, o.data, o.entities = TEXT, dict(DATA), [V("M07")]
    o.quantities = [Quantity(f"member.{V('M08')}.messages", 3.0)]
    res.artifacts = {"note": f"see {V('E01')}"}
    if n["exec"] == 2:
        res.status, res.failure_class, res.failure_detail = "failed", "unknown", f"mail to {V('E03')} bounced; token {V('S01')}"
    if n["exec"] == 3:
        raise RuntimeError(f"HTTP 401 for request sent with Authorization: Bearer {V('S12')}")


L.env.result_edit = edit
L.rt.start()
for _ in range(5):
    L.rt.step()
L.rt.log.sync()
mid = scan(L.run_dir)
L.rt.run(max_iterations=50)
write_journal(L.run_dir)
end = scan(L.run_dir)
R.note("files in the run directory", sorted(os.path.relpath(os.path.join(r, f), L.run_dir) for r, _, fs in os.walk(L.run_dir) for f in fs))

logset = set(LOGS) | {RunFiles.JOURNAL}
for label in P:
    in_logs = sorted(set(mid.get(label, []) + end.get(label, [])) & logset)
    R.check(f"{label}: absent from the redacted logs and the journal", not in_logs, in_logs)
for label in P:
    other_mid = sorted(set(mid.get(label, [])) - logset)
    other_end = sorted(set(end.get(label, [])) - logset)
    R.check(f"{label}: absent from every OTHER run file (state.json, wal.jsonl, checkpoints, task.json ...)",
            not other_mid and not other_end, {"mid-run": other_mid, "after the run ended": other_end})

# ---- log envelope and global seq ----
recs = [r for nme in LOGS for r in read_log(L.run_dir, nme)]
need = {"rid", "seq", "ts", "type", "run_id", "provenance", "source", "data"}
R.check("X1 every log line has the envelope {rid, seq, ts, type, run_id, provenance, source, data}",
        bool(recs) and all(need <= set(r) for r in recs), len(recs))
R.check("X2 provenance of every record equals the task mode (diagnostic)", {r["provenance"] for r in recs} == {"diagnostic"})
sq = check_seq(L.run_dir)
R.check("X3 seq is one counter across all files: unique, increasing, no gaps", not sq["duplicate_seq"] and not sq["non_monotonic"]
        and sq["n_gaps"] == 0, sq)
obs = [r for r in read_log(L.run_dir, RunFiles.EVENTS) if r["type"] == "observation"]
R.check("X4 environment observations are marked trust=untrusted", bool(obs) and all(r.get("trust") == "untrusted" for r in obs
                                                                                     if r["data"]["observation"]["provenance"]["source"] == "slowenv"))
sys.exit(R.finish())
