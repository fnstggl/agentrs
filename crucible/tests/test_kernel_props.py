"""Property tests: random command sequences against identity/bank/admin must keep every kernel guarantee.

Each example runs up to ~60 random steps (joins, transfers between random accounts with random - also invalid -
amounts, grants with spending limits and expiries, organisations with accounts, invites and responses, keyed
retries, clock advances, snapshots and reloads from the store) and checks after every step:

* kernel and module invariants, ledger consistency (balances == postings), money conservation, overdrafts;
* idempotent retries return the recorded outcome and change nothing;
* a successful transfer was authorised (owner, org owner/admin, or a matching live grant within its limit);

and at the end that replay (appliers), regeneration (handlers) and snapshot/fork reloads reproduce the world.
"""
from __future__ import annotations

from hypothesis import HealthCheck, given, settings, strategies as st

from crucible.core.errors import CrucibleError
from crucible.core.kernel import World, WorldSpec
from crucible.services import admin as _admin, bank as _bank, identity as _identity  # noqa: F401  (registers modules)
from crucible.services.identity import WINDOW, _cap_match
from crucible.storage.memory import MemoryStore

SPEC = WorldSpec("props", 3, [{"name": "identity"}, {"name": "bank"}, {"name": "admin"}])
HANDLES = ["ann", "ben", "cat", "dan", "eve"]
ORGS = ["acme", "bolt", "core"]
SYSTEM_ACCTS = ["acct_genesis", "acct_households", "acct_external", "acct_fees"]
WORLD_ID = "props"

# ---------------------------------------------------------------------- strategies
person = st.sampled_from([0, 0, 0, 1, 1, 1, 2, 2, 3, 4])      # mostly actors that exist (ann, ben, cat)
org = st.sampled_from([0, 0, 0, 1, 2])
account = st.one_of(st.integers(1, 8).map(lambda i: f"acct_{i:06d}"), st.integers(1, 8).map(lambda i: f"acct_{i:06d}"),
                    st.sampled_from(SYSTEM_ACCTS + ["acct_nope"]))
source = st.one_of(st.just("own"), st.just("own"), st.just("org"), account)   # resolved against the state
target = st.one_of(st.sampled_from(["ann", "ben", "cat"]), account,
                   st.sampled_from(HANDLES + ["o:acme", "o:bolt", "p:ann", "nobody", "sys"]))
valid_amount = st.one_of(st.integers(1, 3000), st.integers(1, 10**7))
amount = st.one_of(*[valid_amount] * 6, st.sampled_from([-3, 0, 10**13, 1.5, True]))  # some schema-invalid
key = st.sampled_from([None, None, None, "k0", "k1", "k2", ""])

op = st.one_of(
    st.tuples(st.just("join"), person, st.sampled_from([0, 1, 500, 5000, 10**6])),
    st.tuples(st.just("transfer"), person, source, target, amount, key),
    st.tuples(st.just("transfer"), person, source, target, amount, key),
    st.tuples(st.just("transfer"), person, source, target, amount, key),
    st.tuples(st.just("transfer"), person, source, target, amount, key),
    st.tuples(st.just("grant"), person, st.sampled_from(["ann", "ben", "ben", "cat", "cat", "dan", "o:bolt", "nobody"]),
              st.sampled_from(["bank.spend", "bank.spend", "bank.*", "bank.read", "*", "Bad Cap", "org.admin"]),
              st.sampled_from([None, None, "acme", "acme", "acme", "bolt", "ann"]),
              st.one_of(st.just("*"), account), st.sampled_from([None, 0, 100, 2500]),
              st.sampled_from([None, 60, 3600, 86400]), key),
    st.tuples(st.just("org"), person, org),
    st.tuples(st.just("org_account"), person, org),
    st.tuples(st.just("org_account"), person, org),
    st.tuples(st.just("invite"), person, org, person, st.sampled_from(["member", "admin", "owner"])),
    st.tuples(st.just("respond"), person, st.one_of(st.just("own"), st.integers(1, 5)), st.booleans()),
    st.tuples(st.just("respond"), person, st.one_of(st.just("own"), st.integers(1, 5)), st.booleans()),
    st.tuples(st.just("revoke"), person, st.one_of(st.just("own"), st.integers(1, 6))),
    st.tuples(st.just("retry"),),
    st.tuples(st.just("retry"),),
    st.tuples(st.just("advance"), st.sampled_from([0, 1, 60, 3600, 86400, 31 * 86400])),
    st.tuples(st.just("snapshot"),),
    st.tuples(st.just("reload"),),
)


# ---------------------------------------------------------------------- model
class Run:
    def __init__(self) -> None:
        self.store = MemoryStore()
        self.w = World.create(SPEC, WORLD_ID, store=self.store)
        self.endowed = 0
        self.last_keyed: tuple | None = None   # (actor, action) of the last keyed command
        for h, e in (("ann", 5000), ("ben", 3000), ("cat", 0)):     # acct_000001..3
            self.join(h, e)
        # ann founds acme and opens its account (acct_000004), so delegated spending is reachable early
        assert self.w.execute("p:ann", {"tool": "id.create_org", "args": {"handle": "acme", "name": "Acme"}}).ok
        assert self.w.execute("p:ann", {"tool": "bank.open_account", "args": {"owner": "acme", "name": "ops"}}).ok

    # -------------------------------------------------------------- helpers
    def join(self, handle: str, endowment: int) -> None:
        try:
            self.w.system_call("admin.join", {"handle": handle, "endowment": endowment})
            self.endowed += endowment
        except CrucibleError:
            pass

    def execute(self, actor: str, action: dict):
        st_ = self.w.state
        k = action.get("idempotency_key")
        idem_id = f"{actor}|{k}"
        recorded = st_.get("idem", idem_id) if k is not None else None
        before = (self.w.seq, self.w.digest()) if recorded is not None else None
        allowed_src = self._may_spend(actor, action) if action["tool"] == "bank.transfer" else None
        r = self.w.execute(actor, action)
        if recorded is not None:
            assert r.replayed and r.ok == recorded["ok"] and r.result == recorded["result"]
            assert r.error == recorded.get("error")
            assert (self.w.seq, self.w.digest()) == before, "a replayed command changed the world"
        else:
            assert not r.replayed
        if k is not None:
            self.last_keyed = (actor, action)
        if r.ok and action["tool"] == "bank.transfer" and not r.replayed:
            assert allowed_src, f"unauthorised transfer succeeded: {actor} {action}"
        return r

    def _may_spend(self, actor: str, action: dict) -> bool:
        """Independent oracle of identity.authorize for bank.spend on the source account (state before)."""
        st_ = self.w.state
        args = action["args"]
        a = st_.get("accounts", args.get("from_account")) if isinstance(args.get("from_account"), str) else None
        if a is None:
            return False
        owner = a["owner"]
        if owner == actor:
            return True
        m = st_.get("members", f"{owner}|{actor}")
        o = st_.get("orgs", owner)
        if m is not None and o is not None and o["status"] == "active" and m["role"] in ("owner", "admin"):
            return True
        amt = args.get("amount") if isinstance(args.get("amount"), int) else 0
        for gid in st_.find_ids("grants", "grantee", actor):
            g = st_.tables["grants"][gid]
            if g["owner"] != owner or g["status"] != "active" or not _cap_match(g["cap"], "bank.spend"):
                continue
            if g["scope"] not in ("*", None) and g["scope"] != args["from_account"]:
                continue
            if g["expires"] is not None and g["expires"] <= self.w.now:
                continue
            if g["limit"] is not None:
                used = g["used"] if self.w.now - g["window_start"] < WINDOW else 0
                if used + amt > g["limit"]:
                    continue
            return True
        return False

    # -------------------------------------------------------------- steps
    def step(self, o: tuple) -> None:
        kind = o[0]
        if kind == "join":
            self.join(HANDLES[o[1]], o[2])
        elif kind == "transfer":
            _, p, src, dst, amt, k = o
            if src == "own":     # the actor's own checking account
                src = self.w.modules["bank"].primary_account(self.w.state, f"p:{HANDLES[p]}") or "acct_nope"
            elif src == "org":   # acme's account: spendable by its owners/admins and by grantees
                src = "acct_000004"
            a = {"tool": "bank.transfer", "args": {"from_account": src, "to": dst, "amount": amt}}
            if k is not None:
                a["idempotency_key"] = k
            self.execute(f"p:{HANDLES[p]}", a)
        elif kind == "grant":
            _, p, grantee, cap, owner, scope, limit, expires, k = o
            args = {"grantee": grantee, "cap": cap, "scope": scope}
            if owner is not None:
                args["owner"] = owner
            if limit is not None:
                args["limit"] = limit
            if expires is not None:
                args["expires_in"] = expires
            a = {"tool": "id.grant", "args": args}
            if k is not None:
                a["idempotency_key"] = k
            self.execute(f"p:{HANDLES[p]}", a)
        elif kind == "org":
            self.execute(f"p:{HANDLES[o[1]]}", {"tool": "id.create_org",
                                               "args": {"handle": ORGS[o[2]], "name": ORGS[o[2]].title()}})
        elif kind == "org_account":
            self.execute(f"p:{HANDLES[o[1]]}", {"tool": "bank.open_account",
                                               "args": {"owner": ORGS[o[2]], "name": "ops"}})
        elif kind == "invite":
            _, p, g, q, role = o
            self.execute(f"p:{HANDLES[p]}", {"tool": "id.invite",
                                            "args": {"org": ORGS[g], "person": HANDLES[q], "role": role}})
        elif kind == "respond":
            actor, inv = f"p:{HANDLES[o[1]]}", o[2]
            if inv == "own":     # the actor's first pending invite, if any
                mine = [i for i in self.w.state.find_ids("invites", "person", actor)
                        if self.w.state.tables["invites"][i]["status"] == "pending"]
                inv = mine[0] if mine else "inv_999999"
            else:
                inv = f"inv_{inv:06d}"
            self.execute(actor, {"tool": "id.respond_invite", "args": {"invite": inv, "accept": o[3]}})
        elif kind == "revoke":
            actor, g = f"p:{HANDLES[o[1]]}", o[2]
            if g == "own":       # a live grant this actor issued, if any
                mine = [gid for gid, x in self.w.state.scan("grants") if x["by"] == actor and x["status"] == "active"]
                g = mine[0] if mine else "grant_999999"
            else:
                g = f"grant_{g:06d}"
            self.execute(actor, {"tool": "id.revoke", "args": {"grant": g}})
        elif kind == "retry":
            if self.last_keyed is not None:
                actor, action = self.last_keyed
                self.execute(actor, action)
        elif kind == "advance":
            self.w.advance_time(self.w.now + o[1])
        elif kind == "snapshot":
            self.w.snapshot()
        elif kind == "reload":
            self.w.flush()
            old = self.w
            self.w = World.load(self.store, WORLD_ID)
            assert (self.w.digest(), self.w.last_hash, self.w.seq) == (old.digest(), old.last_hash, old.seq)
        else:  # pragma: no cover
            raise AssertionError(kind)

    # -------------------------------------------------------------- checks
    def check_step(self) -> None:
        w = self.w
        assert not w.state.in_tx
        assert w.check_invariants() == []
        assert w.modules["bank"].check_ledger_consistency(w) == []
        accts = w.state.tables["accounts"]
        assert sum(a["balance"] for a in accts.values()) == 0
        for aid, a in accts.items():
            if a["kind"] != "system":
                assert a["balance"] >= -a["overdraft"], aid
        assert accts["acct_genesis"]["balance"] == -self.endowed
        assert sum(a["balance"] for a in accts.values() if a["kind"] != "system") == self.endowed
        for x in SYSTEM_ACCTS[1:]:
            assert accts[x]["balance"] == 0, x
        for gid, g in w.state.tables["grants"].items():
            if g["limit"] is not None:
                assert 0 <= g["used"] <= g["limit"], gid

    def check_final(self) -> None:
        w = self.w
        w.flush()
        evs = self.store.read_events(WORLD_ID)
        assert [e["seq"] for e in evs] == list(range(1, w.seq + 1))
        r = World.replay(SPEC, evs)
        assert (r.digest(), r.last_hash, r.seq) == (w.digest(), w.last_hash, w.seq)
        g = World.regenerate(SPEC, evs)
        assert (g.last_hash, g.seq) == (w.last_hash, w.seq)
        assert g.digest() == w.digest()
        # snapshot + reload, and a fork persisted in the same store
        sid = w.snapshot()
        assert World.from_blob(self.store.get_snapshot(sid)[1]).digest() == w.digest()
        loaded = World.load(self.store, WORLD_ID)
        assert (loaded.digest(), loaded.last_hash) == (w.digest(), w.last_hash)
        f = w.fork("props-fork", store=self.store)
        fl = World.load(self.store, "props-fork")
        assert (fl.digest(), fl.last_hash, fl.seq) == (w.digest(), w.last_hash, w.seq)
        # the same commands on parent and fork keep them identical (forks are exact copies)
        for x in (w, f):
            x.execute("p:ann", {"tool": "bank.transfer", "args": {"from_account": "acct_000001", "to": "ben",
                                                                  "amount": 1}})
            x.advance_time(x.now + 7200)
        assert (f.digest(), f.last_hash) == (w.digest(), w.last_hash)


@settings(max_examples=40, deadline=None, database=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
@given(st.lists(op, min_size=20, max_size=60))
def test_random_command_sequences_preserve_kernel_guarantees(ops):
    run = Run()
    run.check_step()
    for o in ops:
        run.step(o)
        run.check_step()
    run.check_final()


def test_targeted_grant_limit_sequence():
    """A deterministic sequence through the delegated-spending paths the random test may rarely combine."""
    run = Run()
    steps = [("org", 0, 0),                                             # acme exists (founded in Run): conflict
             ("invite", 0, 0, 1, "member"), ("respond", 1, 1, True),     # ben joins acme as a member
             ("transfer", 0, "acct_000001", "o:acme", 2000, None),       # ann funds acme
             ("transfer", 1, "acct_000004", "cat", 10, None),            # ben: no grant yet -> forbidden
             ("grant", 0, "ben", "bank.spend", "acme", "*", 100, None, "g"),
             ("retry",),
             ("transfer", 1, "acct_000004", "cat", 60, "t1"),            # within the limit
             ("transfer", 1, "acct_000004", "cat", 60, "t2"),            # would exceed 100 -> forbidden
             ("retry",), ("retry",),
             ("advance", 31 * 86400),                                    # new window
             ("transfer", 1, "acct_000004", "cat", 60, "t3"),
             ("revoke", 0, 1),
             ("transfer", 1, "acct_000004", "cat", 1, None),             # revoked -> forbidden
             ("snapshot",), ("reload",), ("transfer", 0, "acct_000004", "ben", 5, None)]
    for o in steps:
        run.step(o)
        run.check_step()
    accts = run.w.state.tables["accounts"]
    assert accts["acct_000003"]["balance"] == 120          # cat received two delegated payments of 60
    assert accts["acct_000004"]["balance"] == 2000 - 120 - 5
    run.check_final()
