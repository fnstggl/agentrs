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
