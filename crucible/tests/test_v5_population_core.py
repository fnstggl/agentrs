"""Workstream B (population): kernel rules, scheduling, attention, money, learning plumbing, admission.

Small worlds only (a few hundred individuals, a few sites); the gate tests in tests/v5/ measure the requirements at
scale."""
import numpy as np
import pytest

import crucible
from crucible.core.kernel import World
from crucible.population import analysis as A
from crucible.population import core, profiles, schema, text
from crucible.population.worlds import population_world
from crucible.substrate import attention, columns

crucible.load_all_modules()
DAY = 86400

SMALL_STRAT = {"decide_every_s": 8 * 3600, "min_examples": 1, "checkpoint_every": 2,
               "groups": [{"group": "A", "arch": "mlp", "hidden": 64, "algo": "reinforce", "lr": 1e-3},
                          {"group": "B", "arch": "gru", "hidden": 48, "algo": "a2c", "lr": 1e-3}]}
SMALL_AD = {"learn_every_s": 3 * 3600, "min_examples": 8, "checkpoint_every": 2}


def small_spec(seed=3, **kw):
    return population_world("development", seed, n=kw.pop("n", 150), sites=kw.pop("sites", 5),
                            tick_s=kw.pop("tick_s", 1800),
                            classes=kw.pop("classes", {"adaptive": 24, "strategic": 2, "semantic": 1}),
                            adaptive=kw.pop("adaptive", SMALL_AD), strategic=kw.pop("strategic", SMALL_STRAT), **kw)


@pytest.fixture(scope="module")
def run2():
    spec = small_spec()
    w = World.create(spec)
    w.advance_time(w.now + DAY)
    mid = w.snapshot_blob()
    w.advance_time(w.now + DAY)
    return spec, w, mid


def test_profiles_match_part_3_3():
    for name in ("development", "standard", "large", "internet"):
        assert profiles.matches_spec(name), name
    r = profiles.resolve({"profile": "standard"})
    assert r["counts"] == {"lightweight": 10_000, "adaptive": 256, "strategic": 32, "semantic": 8}
    assert r["n"] == 10_296
    assert profiles.resolve({"profile": "internet"})["counts"]["lightweight"] >= 10_000_000
    assert profiles.resolve({"profile": "internet", "classes": {"adaptive": 5000}})["counts"]["adaptive"] == 5000
    with pytest.raises(ValueError):
        profiles.resolve({"profile": "nope"})


def test_schema_represents_part_3_1_fields():
    cols = {**schema.columns(schema.dims({})), **schema.web_columns(schema.dims({}))}   # web columns: popweb
    assert len(schema.FIELDS) == 18
    for f, cs in schema.FIELDS.items():
        assert all(c in cols or c == "income_day" for c in cs), f
    # the contract's columns of table pop
    for c in ("interests", "knowledge", "skills", "goals", "attention_budget_min", "attention_left_min", "wallet_cents",
              "satisfaction", "habit", "employer", "beliefs", "next_active_at", "recent", "class", "policy_group",
              "memory"):
        assert c in cols, c


def test_text_representation_is_content_based():
    a, sa = text.text_vector("fresh bread soup recipe for a winter dinner in the kitchen", 32)
    b, _ = text.text_vector("bread recipe and soup for dinner: kitchen tips", 32)
    c, _ = text.text_vector("road bike repair: wheel, tyre and helmet for the commute", 32)
    assert float(a @ b) > float(a @ c) + 0.3
    stuffed, ss = text.text_vector("bread " * 80, 32)
    assert ss["uniq"] < 0.05 < sa["uniq"]
    p = text.parse_page("<html><title>T</title><a href='/a/1'>x</a><a href='http://bikebay.crucible/'>b</a>"
                        "<button>Buy for $12.50</button></html>")
    assert p["paths"] == ["/a/1"] and p["hosts"] == ["bikebay"] and p["prices"] == [1250]


def test_same_seed_same_hashes_replay_regeneration_and_resume(run2):
    spec, w, mid = run2
    w2 = World.create(spec)
    w2.advance_time(w2.now + 2 * DAY)
    assert w2.last_hash == w.last_hash
    assert w.check_invariants() == []
    evs = [e.to_dict() for e in w.log]
    assert World.replay(spec, evs).digest() == w.digest()
    assert World.regenerate(spec, evs).last_hash == w.last_hash
    # a world resumed from a mid-run snapshot rebuilds the learned weights (checkpoint + logged updates, hashes
    # checked) and continues exactly as the uninterrupted world did
    f = World.from_blob(mid, world_id="resumed")
    f.advance_time(f.now + DAY)
    assert f.last_hash == w.last_hash and f.digest() == w.digest()


def test_population_acts_autonomously_and_within_attention(run2):
    _, w, _ = run2
    li = A.learner_initiated(w)
    assert li["population_decision_events"] > 100 and li["learner_initiated"] == 0
    assert attention.check(w) == []
    spent = columns.col(w, "pop", "attention_spent_min")
    bud = columns.col(w, "pop", "attention_budget_min")
    assert np.all(spent <= bud * (1 + 1e-5) + 1e-3)
    log = A.visit_log(w)
    assert len(log["ind"]) > 100
    # every visit was paid for with attention: an individual's minutes in one tick never exceed its budget
    assert float(log["minutes"].max()) <= float(bud.max()) + 1e-3


def test_purchases_are_bank_moves_from_individual_accounts(run2):
    _, w, _ = run2
    paid = [e for e in w.log if e.type == "bank.invoice_paid"]
    log = A.visit_log(w)
    assert int(log["bought"].sum()) == len(paid) > 0
    bank = w.modules["bank"]
    for e in paid[:20]:
        inv = w.state.get("invoices", e.payload["id"])
        assert inv["status"] == "paid"
        payer = inv["payer"]
        assert payer.startswith("p") and payer[1:].isdigit()          # an individual of the population
        txn = w.state.get("txns", e.payload["txn"])
        debit = [a for a, d in txn["postings"] if d < 0][0]
        assert bank.account_owner(w.state, debit) == payer           # its own account, not a households pool
    assert not any(e.type == "bank.txn" and any(a == "acct_households" for a, _ in e.payload["postings"])
                   for e in w.log)


def test_wake_triggers_an_activation():
    spec = small_spec(seed=5, classes={"adaptive": 0, "strategic": 0, "semantic": 0}, n=60)
    w = World.create(spec)
    who = np.arange(0, 20)
    target = w.now + 3600
    w.modules["pop"].config  # noqa: B018

    def do(ctx):
        core.wake(ctx, who, target, "web")
    w._run_system("test.wake", do)
    assert np.all(columns.col(w, "pop", "wake_at")[who] == target)
    # one batch with a time per row; the earliest pending wake-up of an individual is kept
    who2 = np.arange(30, 36)
    at2 = w.now + 7200 + 600 * np.arange(6)

    def do2(ctx):
        core.wake(ctx, np.concatenate([who2, who2[:2]]), np.concatenate([at2, at2[:2] + 99_999]), "web")
    n0 = sum(1 for e in w.log if e.type == "cols.batch" and e.payload["kind"] == "pop.wake")
    w._run_system("test.wake2", do2)
    assert np.all(columns.col(w, "pop", "wake_at")[who2] == at2)
    assert sum(1 for e in w.log if e.type == "cols.batch" and e.payload["kind"] == "pop.wake") == n0 + 1
    w.advance_time(target + 1800)
    sess = [e for e in w.log if e.type == "cols.batch" and e.payload["kind"] == "pop.session"]
    woken = sum(e.payload["extra"]["woken"] for e in sess)
    assert woken >= 20
    assert np.all(columns.col(w, "pop", "wake_at")[who] > target + 1800)


def test_adaptive_policy_learns_from_examples():
    from crucible.population.adaptive import AdaptivePolicy
    rng = np.random.default_rng(0)
    pol = AdaptivePolicy(0, obs_dim=20, cand_dim=8, mem_dim=6, hidden=32, seed=1)
    assert 0 < pol.params < 2_000_000
    obs = rng.normal(size=(256, 20)).astype(np.float32)
    cand = rng.normal(size=(256, 8)).astype(np.float32)
    batch = {"obs": obs, "mem": np.zeros((256, 6), np.float32), "cand": cand,
             "reward": (cand[:, 0] - 0.5 * cand[:, 1]).astype(np.float32)}
    h0 = pol.weights_hash()
    out = pol.learn(batch, steps=200, batch_size=64, lr=3e-3, seed=7)
    assert out["weights_hash"] != h0 and out["loss"] < out["loss_before"]
    # determinism: the same update from the same weights gives the same weights
    pol2 = AdaptivePolicy(0, obs_dim=20, cand_dim=8, mem_dim=6, hidden=32, seed=1)
    assert pol2.learn(batch, steps=200, batch_size=64, lr=3e-3, seed=7)["weights_hash"] == out["weights_hash"]
    act, mem = pol.act_batch({"obs": obs[:5], "cand": np.stack([cand[:5]] * 3, 1), "mask": np.ones((5, 3), bool)},
                             np.zeros((5, 6), np.float32), np.random.default_rng(1))
    assert act.shape == (5,) and mem.shape == (5, 6)
    # the activity decision steps the personal memory (once per session)
    from crucible.population.adaptive import ACT_DIM
    s, mem2 = pol.act_scores(obs[:5], np.zeros((5, 6), np.float32), rng.normal(size=(5, 4, ACT_DIM)).astype(np.float32))
    assert s.shape == (5, 4) and np.abs(mem2).sum() > 0


def test_adaptive_memory_learns_through_time():
    """Activity examples are replayed as sequences through the GRU (BPTT): a reward that depends on what happened
    several sessions earlier (not in the current observation) is learnable with memory and not without."""
    from crucible.population.adaptive import ACT_DIM, AdaptivePolicy
    rng = np.random.default_rng(3)
    n_ind, L = 60, 8
    obs = rng.normal(size=(n_ind * L, 10)).astype(np.float32)
    key = rng.integers(0, 2, n_ind).astype(np.float32) * 2 - 1          # a hidden personal trait
    obs.reshape(n_ind, L, 10)[:, 0, 0] = key                              # seen only at the first session
    obs.reshape(n_ind, L, 10)[:, 1:, 0] = 0.0
    ac = np.zeros((n_ind * L, ACT_DIM), np.float32)
    ac[:, 0] = rng.integers(0, 2, n_ind * L) * 2 - 1
    reward = (np.repeat(key, L) * ac[:, 0]).astype(np.float32)
    batch = {"act": {"obs": obs, "mem": np.zeros((n_ind * L, 8), np.float32), "acand": ac, "reward": reward,
                     "ind": np.repeat(np.arange(n_ind), L).astype(np.int32),
                     "t": np.tile(np.arange(L), n_ind).astype(np.int64)}, "web": None}
    res = {}
    for mem in (True, False):
        pol = AdaptivePolicy(0, obs_dim=10, cand_dim=4, mem_dim=8, hidden=32, seed=1, memory=mem, bptt=L)
        out = pol.learn(batch, steps=300, lr=3e-3, seed=5)
        res[mem] = out["loss"]
    assert res[True] < 0.5 * res[False], res


def test_policy_updates_are_recorded(run2):
    _, w, _ = run2
    ups = [e for e in w.log if e.type == "pop.policy_update"]
    groups = {e.payload["group"] for e in ups}
    assert {"adaptive:0", "strategic:A", "strategic:B"} <= groups
    for g in groups:
        steps = [e.payload for e in ups if e.payload["group"] == g]
        assert steps[0]["checkpoint"]                       # an initial checkpoint
        if len(steps) > 1:
            assert steps[-1]["weights_hash"] != steps[0]["weights_hash"]
            assert steps[-1]["n_examples"] > 0 and steps[-1]["examples"]


def test_strategic_participants_act_through_tools(run2):
    from crucible.population import strategic as STR
    _, w, _ = run2
    exps = list(w.state.tables["pop_strat_exp"].values())
    assert exps and {r["group"] for r in exps} <= {"A", "B"}
    ents = [e for e in w.log if e.type == "cloud.deployment" and STR.host_member(e.payload["host"]) is not None]
    if any(r.get("outcome") == "ok" and STR.ACTIONS[r["act"]["act"]] == "launch" for r in exps):
        assert ents, "ventures deployed sites with code/cloud tools"
    for e in ents:
        assert e.payload["owner"].startswith("o:ent") and e.payload["trusted"] is False
    # decisions are recorded inside the population's own sessions (class runtime 3)
    sess = [e for e in w.log if e.type == "cols.batch" and e.payload["kind"] == "pop.session" and
            (e.payload["extra"] or {}).get("class") == 3]
    assert sess


def test_modules_pass_admission():
    from crucible.worldgen.admission import admit
    base = ["identity", "bank", "admin", "artifacts", "cols", "space", "code", "search", "forum", "cloud", "net", "appdb"]
    cfg = {"popsites": {"n": 3, "mix": {"rich": 1.0}}, "pop": {"profile": "development", "n": 40, "tick_s": 3600,
                                                               "classes": {"adaptive": 6, "strategic": 1,
                                                                           "semantic": 0}},
           "pop_adaptive": {"learn_every_s": 6 * 3600, "min_examples": 4}, "pop_strategic": SMALL_STRAT}
    rep = admit(["popsites", "pop", "popweb", "pop_adaptive", "pop_strategic"], base=base, steps=120, actors=3,
                seed=1, configs=cfg)
    assert rep.passed, (rep.checks, rep.errors[:5])


def test_reader_model_does_not_reward_keyword_stuffing():
    """DEF-B-6 regression: the readable-content factor of a page (population.web: length x legibility of the text,
    the shared space's reader model) ranks a real article above the same topic's words as a word salad or a keyword
    list, so a page cannot gain utility by repeating topic words."""
    from crucible.population import semantics, writing
    rng = np.random.default_rng(4)
    t = semantics.topic_names().index("recipes")
    _, real = writing.compose(rng, t, 12, 1.0)
    words = [w for w in real.split() if w.isalpha()]
    salad = " ".join(rng.permutation(words))
    keywords = " ".join(semantics.topic_words(t)[:8] * 30)
    leg = {k: semantics.legibility(semantics.text_features(v)) for k, v in
           (("real", real), ("salad", salad), ("keywords", keywords))}
    rich = {k: min(1.0, len(v) / 2000.0) ** 0.5 * leg[k] for k, v in
            (("real", real), ("salad", salad), ("keywords", keywords))}
    assert rich["real"] > rich["salad"] and rich["real"] > 2 * rich["keywords"], (leg, rich)


def test_fitted_runtime_models_take_population_decisions(run2):
    """DEF-B-2 / DEF-E-1 regression: daily activation and sessions per active day are decisions of workstream E's
    fitted runtime models, recorded with their provenance inside the population's own sessions."""
    from crucible.population import check_provenance
    _, w, _ = run2
    day = [e.payload for e in w.log if e.type == "cols.batch" and e.payload["kind"] == "pop.day_active"]
    ses = [e.payload for e in w.log if e.type == "cols.batch" and e.payload["kind"] == "pop.day_sessions"]
    assert day and ses
    assert {p["extra"]["decision"]["model_id"] for p in day} == {"return_daily"}
    assert {p["extra"]["decision"]["model_id"] for p in ses} == {"session_rate"}
    cp = check_provenance(w, prefixes=("pop.day_",))
    assert cp["organic_population_rows"] > 0 and cp["forced_rows"] == 0
    # every population-table batch outside the non-decision kinds carries provenance
    from crucible.calibration import inworld
    nd = {e.payload["kind"] for e in w.log if e.type == "cols.batch" and e.payload["table"] == "pop"
          and inworld.non_decision_reason(e.payload["kind"]) is None
          and not isinstance((e.payload.get("extra") or {}).get("decision"), dict)}
    assert not nd, nd
