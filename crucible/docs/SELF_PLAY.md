# Self-play and leagues

Code: `crucible/src/crucible/training/league.py` (`League`, `policy_from_spec`, rating math),
`crucible/src/crucible/cli.py` (`crucible selfplay run`). Tests: `crucible/tests/test_tooling.py`
(`test_league_*`, `test_selfplay_short_real_run`).

## 1. What a match is

A **match** puts several policies into one freshly generated world (league family + seed). Each policy controls its
own actor (`p:seat1`, `p:seat2`, ...; joined through the recorded `admin.join` with the same endowment) and acts
once per simulated hour (`Runner`, seeded order). The world's own mechanics (simulated consumers, incumbents,
compute billing, search, ads, ratings) decide what happens. All of them compete for **the same simulated
customers**: e.g. two `Builder` variants serving `route.shortest` with different prices and advertising, or a
`Composer` that resells incumbents' answers.

Outcomes are measured per agent from authoritative state after the match (`measure_agent`), never from what the
agent says:

| outcome | source |
|---|---|
| `revenue` | `independent_revenue` of every organisation the agent owns, over the match (related-party and circular flows excluded) |
| `customers`, `share` | simulated consumers currently served by the agent's hosts; share within the needs it serves |
| `requests_ok` | successful requests its services served (`svc_days`) |
| `cash`, `cash_delta` | person + owned organisations, relative to the endowment |
| `goal_success` | optional `GoalContract` predicate evaluated by `GoalVerifier` |

## 2. Modes: competition, cooperation, mixed motive

`League(mode=...)` turns outcomes into the score used for ranking:

* **competition** - own independent revenue (customers break ties). Zero-sum-ish over the shared customer pool.
* **cooperation** - every participant gets the team total (sum of independent revenues). Ratings then measure
  who makes teams succeed; use `sample_collaborators(method="best")` to pick partners with high past team scores.
* **mixed** - own revenue + `coop_weight` x others' revenue: agents gain from the economy growing but still compete
  for share. This is the natural setting for studying supply chains (a composer paying upstream providers raises
  their revenue too).

Revenue that agents route between their own principals never counts (the metric excludes related parties), so
collusion inside one agent's organisations cannot inflate a score; collusion *between* seats is visible in the
matchup records and through `evaluation.audit.integrity_report`.

## 3. Paired worlds

Every pairing is played on the same list of world seeds (same `WorldSpec`, same hash: `MatchRecord.spec_hash`) and,
by default, in **both seatings** (seat handles swapped), so a difference between A and B is not a difference between
worlds or seats (a Builder's host name derives from its handle). `round_robin(seeds)` plays all pairings;
`play_pairing(a, b, seeds)` one. By default leagues force `params.cost_model = "deterministic"` so a match is
reproducible from (spec, seed, policies); `--measured-cost` uses measured CPU (faster for trusted incumbents,
not bit-reproducible).

## 4. Ratings

* **Elo** (K = 24 / (n-1) per pairwise result within a match).
* **TrueSkill-style** Gaussian skill (mu, sigma) with the two-player update, a small draw margin and dynamics
  tau; leaderboards sort by the conservative `mu - 3 sigma`.
* A match with n players contributes n(n-1)/2 pairwise results (1 / 0 / 0.5 within `draw_tol` of the larger score).
* A child checkpoint (`add_policy(..., parent=...)`) starts from its parent's rating with widened sigma and
  `generation = parent + 1`; `generation_compare(g_a, g_b)` reports head-to-head win rate and score difference of
  generation b against generation a plus mean ratings.

## 5. Opponent and collaborator sampling

`sample_opponents(learner, k, method)` draws from the **training pool** only:

* `uniform`;
* `pfsp` - prioritised fictitious self-play: weight f(p) with p = P[learner beats opponent] from recorded training
  matches (Beta(1,1) prior); `weighting="hard"` uses (1-p)^2 (focus on opponents the learner loses to),
  `"variance"` uses p(1-p) (focus on even matches);
* `recent` - newer checkpoints more likely (decay^rank).

`train_schedule(learner, seeds, n, method)` plays n matches against sampled opponents on the paired seeds.

## 6. Held-out evaluation population and leakage prevention

* Entries added with `heldout=True` form the evaluation population. They are **never sampled** for training,
  `play()` refuses them in training matches, and they have no influence on training ratings.
* `evaluate_heldout(pid, seeds)` plays a policy against every held-out entry; those records carry
  `heldout_eval=True` and are excluded from `win_rate` (hence from PFSP weights), from `_update_ratings` and from
  the training matchup table. Results come back as a separate report.
* Keep held-out *worlds* separate too: train leagues on `train` world families, select on `val`, and use the locked
  `test` generators only through `crucible.evaluation.harness.locked_test_goals` (logged access). The CLI
  `evaluate` command accepts `train` and `val` only.
* Agents observe only their own actor's view; match outcomes are computed by the evaluator after the match and are
  never written into the world, so a policy cannot read its rating or its opponents' identity from the world
  (seat handles are neutral and the policy id is not recorded in the world: person records are public).

## 7. Running a league

```bash
crucible selfplay run --world tiny --seeds 1,2 --days 3 \
  --policies "builder:need=route.shortest:undercut=0.9,builder:need=route.shortest:undercut=0.7:advertise=0,composer:need=route.shortest" \
  --heldout "builder:need=route.shortest:undercut=0.8" --mode competition --out league.json
crucible report league.json            # matchup table + ratings (markdown)
crucible dashboard --experiments league.json
```

Policy specs are `name[:key=value...]` with `name` in builder | employer | composer | bounty | random | idle and the
keyword arguments of the scripted policy's constructor (`need`, `capital`, `undercut`, `advertise`, `salary`,
`margin`). For learned policies pass factories: `League.add_policy("ckpt-17", factory=lambda: MyPolicy(path),
parent="ckpt-16", tags=("ppo",))`.

Measured example (tiny world, seed 1, 2 simulated days, both seatings): `builder undercut=0.9` with advertising
beat `builder undercut=0.6` without advertising 2-0 ($102.90 vs $36.15 mean independent revenue; 177 vs 84
customers; 50% vs 24% share) - discovery (search + sponsored results) mattered more than a lower price.

## 8. Limitations

* Scripted policies keep their plan in Python generators, so a match always starts them from scratch; long-lived
  learning agents should hold their own state (or checkpoint it) between matches.
* Matches are sequential in one process (each world may start sandbox workers; the container PID limit bounds
  parallelism). Distribute matches over processes for large leagues.
* Ratings assume a stationary game; world families with shocks make outcomes seed-dependent, so always compare on
  paired seeds and report confidence intervals over seeds.
