# Limitations and known technical debt

Everything below was observed or follows directly from how the code works.

## World fidelity

* **The economy is a model.** Demand comes from a vectorised consumer population with hand-designed utility,
  shopping and word-of-mouth rules. Strategies that work here may not work in real markets; the rules are explicit
  and documented (ECONOMY.md) so researchers can see what is and is not modelled.
* **The set of valuable work is finite per world** (eight need formats implemented). Agents can create *new*
  services, protocols and tools, but simulated consumers only pay for need formats they have; demand for genuinely
  new kinds of work comes from other agents, NPC firms (bounties, licences) or worlds that add needs
  (`novel_needs`, `new_need` shocks). A mechanism by which simulated consumers adopt entirely new categories is
  not implemented.
* **Sampled execution.** Each offering executes a sample of real requests per hour (default 4) and extrapolates
  the rest; quality and CPU noise are therefore sampling noise around real measurements, not per-request truth.
* **NPC policies are rule-based** with bounded rationality. No LLM-backed NPCs run by default (an LLM policy class
  exists for agents); NPC firms do not form alliances or negotiate contracts with agents beyond the implemented
  employment, lending and licence-buying decisions.
* **Search** is BM25 + rating signal + ads; no link graph / PageRank. **Mail** has latency and loss but no spam
  filtering. **Calendar** and **docs** exist but few mechanisms depend on them.
* **Physical goods** are only inventory records with conservation; there is no production/supply chain beyond
  hardware purchases for compute providers.
* **Engineering lab**: static per-hour flow model, link capacity shared by both directions, no fixed point after
  drops; production compute of the lab is not billed.

## Reproducibility

* Event replay is exact. Regeneration from commands is exact when external outcomes are recorded (they are). In
  the default *measured* cost model, re-running agent programs is not bit-reproducible (CPU timing), so two forks
  advanced independently diverge slightly. The *deterministic* cost model fixes this but undercounts work done in
  native code (numpy, builtins), which an agent could exploit to get cheap compute in that mode.
* Browser-mode sessions and web caches are in memory; restarting the web server signs users out.

## Security

* The sandbox is Landlock + seccomp + an unprivileged uid inside an unprivileged container — not a VM or gVisor.
  A host-kernel exploit through permitted syscalls would escape it. No mount namespaces are available here.
* There is no login rate limiting in the web frontend.

## Scale

* One world is single-threaded; scaling is by replicas (measured up to 16 concurrent worlds, 6.9 simulated
  days/s aggregate). Snapshots are full copies (≈200 ms at 100k consumers); no structural sharing between forks.
* Measured maximum here: 100,000 consumers, 2,000 NPC workers, 16 world replicas, a few agents per world. Hundreds
  of concurrently active LLM agents were not run (GPU-bound; see RL_INTEGRATION.md).

## Evaluation

* Goal families implemented: revenue, recruit, service, solvency (+ engineering-lab improvement). The
  "outperform incumbents on a held-out battery" goal is evaluated via counterfactual forks and need verifiers but
  is not packaged as a goal family.
* The autocurriculum is implemented and unit-tested; it has not yet been used to train a learner to convergence.
* Self-generated goals are implemented; the experiment "do self-generated goals improve held-out performance" has
  not been run (it needs a learner that improves at all — see EXPERIMENTS.md).
* Statistical power: the reported demonstrations are single runs or a few seeds.

## Learners

* No learner has yet been shown to acquire a broad strategy (start a business, recruit) from experience. The
  scripted controls succeed; the frozen LLM and CAM results are reported in RL_INTEGRATION.md / EXPERIMENTS.md.

## Open-world layer (feat/crucible-open-world)

The full list with code locations is in `docs/OPEN_ENDEDNESS_AUDIT.md` §5. In short:

* **Simulated actors are Tier 2 (rule-based, utility-driven).** No LLM-driven NPC minds were built; free text in
  offers and messages does not influence NPC decisions (only structured terms and verifiable references do).
* **Verification of usefulness is limited by noise and design.** Switchback trials use the installer's daily
  operating result; lumpy business purchases make it noisy, lagged effects carry over between on/off days, and
  10-16 day trials miss modest real gains. There is no universal usefulness evaluator, by design.
* **Calibration of the industrial economy.** Downstream assemblers run deficits in several seeds and the firm
  sector's total operating result in the 90-day mixed run is negative; prices are cost-plus with simple
  adjustments; the economy runs but is not in a sustainable steady state.
* **Fixed mechanism vocabulary**: 5 production operations, 4 forms, 4 property and 4 attribute dimensions, contract
  clause types, governance rule types, loss mechanism labels. New *kinds* of mechanism need developer code; there
  is no runtime developer-module extension interface or schema migration on this branch.
* **NPC entrepreneurs imitate (fork open source); they do not invent.** NPC developers only implement legacy needs.
* **Secondary needs**: mechanisms for them exist (dependency failures, compute/grid/freight load); in the recorded
  runs the observed secondary effects were competitive entry and upstream investment, not new problem categories.
* **Knowledge diffusion depends on contacts**, which form only through dealings; in the long mixed run few firms
  shared evidence.
* **Isolation**: only Landlock/seccomp/rlimits on this host; gVisor, bubblewrap and Firecracker backends are wired
  but unavailable here (no runsc, no bwrap, no user namespaces, no /dev/kvm) and are refused, not emulated.

## Crucible 3.0 (feat/crucible-internet)

* **The synthetic internet has no browsing population.** Simulated people and firms read the forum, mail and the
  capability registry. They do not visit web sites. Web traffic comes from agents, the crawler and site seeding.
  Audience-type goals therefore depend on agents.
* **Sites are server-rendered.** There are no cross-site cookies, TLS, CDN or DNS hierarchy. All generated sites
  share one template; agent-written sites are arbitrary code.
* **Second-order emergence has one mechanism family.** It was shown for compute capacity (autoscaling ->
  provider exhaustion -> placement), with scripted competent controls on both sides, in 3 seeds. It has not
  been shown with learned or LLM inventors, or in other domains.
* **Most open-world "discoveries" are made by scripted controls.** NPC innovators invent and iterate, but none of
  their versions was adopted in the recorded runs. The frozen LLM agents (Qwen3-4B, Qwen3-14B) built nothing in the final challenge (8 runs).
* **Trials are imprecise.** Neither trial design recovers maintenance effects reliably against matched
  counterfactuals (sign agreement 0.44-0.50). In-world firms face the same difficulty; the numbers are
  documented rather than tuned away.
* **The industrial sector is only partly calibrated.** One of three seeds is clearly viable; the other two are
  marginal.
* **Stylised social mechanics.**
  * Effort under profit sharing is a formula, and it rewarded a labour-based rule that paid members nothing
    (demo C). This is an inconsistency, not fixed.
  * Simulated voters decide from trust in the proposer, not from their payoffs.
* **Information channels.**
  * Ablating communication changed beliefs and trials, but no measure in the Part 21 suite differed
    consistently.
  * The evidence-aggregator site died when its capital ran out.
* **Task generation is narrow.**
  * There are three mined task families. All 7 `relieve_loss` tasks were rejected in the three validated worlds:
    6 had no feasibility witness and 1 was not attributable. Only 4 of 14 mined tasks were accepted.
  * Validation cannot tell "impossible" from "beyond the competent controls".
  * Difficulty is estimated from one competent control per family.
* **LLM evaluation is small.** One world seed, a 15-day horizon, a decision every 3 simulated hours, Qwen3-4B
  and Qwen3-14B only. No API frontier models were run.
  * The first attempt was confounded by the interface. With a 9,000-character tool budget, alphabetical order
    gave full signatures only to the first services (`ads`, `bank`, `caps`...). All three runs then fixated on
    advertising tools.
  * The rerun used interleaved ordering and a 24,000-character budget. A second confound followed: the
    backend's 8k context truncated the tool list in the middle. It was fixed with a 32k context.
  * On the corrected interface no model founded an organisation or built anything. All runs perseverated on one
    action. Scaffolded or frontier agents were not tried.
* **Liveness checks were too weak.** "Demand served > 0" missed a collapse to about 0.1% of normal. The weekly
  process-failure count caught it. Thresholds should be relative to recent levels.
* **Isolation is not VM-grade here.** Workloads that require VM-grade isolation are refused, not run.

## Crucible 4.0 (feat/crucible-4)

Measured results and negative findings are in [OPENWORLD_REPORT.md](OPENWORLD_REPORT.md).

* **No learner has yet learned to succeed in unfamiliar worlds.** On the sealed test no trained learner beat a scripted
  experimenter. The instrument's sensitivity to real meta-learning is therefore shown only on purpose-built learners.
* **Nothing is calibrated.** 0 of 55 registered mechanisms are compared with real data: 37 are plausible hypotheses and 18
  are known to be unrealistic. No result here predicts real-world performance.
* **Competence in the world is mostly scripted.** Inventors, builders and experimenters that succeed are scripted controls;
  simulated firms and founders are rule-based minds.
* **Self-play runs on hidden-mechanism facilities**, not inside the full open economy, with small controllers.
* **Discovery by one-lever experiments does not work on the technical-systems engine**, where levers interact and lag.
* **Second-order effects are established for one invention class** (capacity); pricing and latent-demand classes are weak.
* **Remaining exploits.**
  * Two formally unrelated colluders can each be credited if the two legs of a payment are further apart than the netting
    window. A one-way payment from a colluder is indistinguishable from a purchase.
  * A first deliberate loan default still nets the owner the principal (limited liability), at the cost of a public breach,
    a claim, bankruptcy and no further credit.
  * Free requests from a colluding unrelated agent count as service usage; a colluding real business counts as an adopter; a
    paid one-cent order buys a rating; a doorway page padded past 300 characters can still outrank thin content.
  * N distinct agents can cost an advertiser N impressions a day.
  * Sustained goals are watched only at harness steps.
  * Anyone with write access to the event store can rewrite a log suffix; `storage.base.make_anchor` detects it, but nothing
    verifies anchors automatically.
* **Isolation is process-level.** Workers share one uid and the host kernel. VM-grade workloads are refused.
* **The founder pool is finite** (14 per world, activated over a year), so horizons well beyond a year can run out of entrants.
* **Simulated prices can ratchet down:** firm minds have no cost floor and rarely raise prices once entrants arrive.
* **The real-world bridge** is tested against a loopback server only; its HTTPS path is untested.
* **Performance numbers** come from the quick suite on a shared machine with an 11.5-core quota. The 30-day episode costs
  and the full family table were not measured.
* **Known flaky test under heavy load:** `test_internet.py::test_real_chromium_*` failed once and passed on rerun.
* **The sealed set `ow-final` is spent**, and its salt is outside the repository: a new comparison needs a new set.

