# World fidelity: what is mechanistic, what is stylised, what is absent

Crucible is not a faithful model of the real economy or the real internet. This table states, per subsystem,
how each outcome is produced, so that results can be read at the right strength.

Levels:
* **mechanistic**: outcomes follow from executed rules over recorded state that an agent can change by acting;
* **stylised**: an explicit formula with chosen parameters stands in for a process;
* **absent**: not modelled.

| Subsystem | Level | How outcomes arise | Known simplifications |
|---|---|---|---|
| Money and ledgers | mechanistic | double-entry postings; every transfer is a recorded command; invariants checked | no credit creation beyond lenders; no taxes; no FX |
| Compute | mechanistic | providers have finite cores; rent/resize refuse when full (`capacity_unavailable`); prices move with utilisation; hardware arrives after a lead time | one region; prices follow a fixed rule (+8% above 85% utilisation, -3% below 35%); no spot market |
| Service demand | stylised | consumers sample services each hour by price and quality with a daily cycle (`diurnal`, `peak_hour`); overload turns requests away | demand curve and daily cycle are parameters; consumers do not browse web sites |
| Agent software | mechanistic | agent code really runs in the sandbox (CPU time, errors, timeouts); quality is measured on hidden test batteries | sandbox is a restricted process, not a VM (SECURITY.md) |
| Firm decisions | stylised | firm minds keep their own books, publish problems, run randomised trials of capabilities and adopt on their own measured effect; pricing and capacity rules | decision rules are fixed heuristics with traits; no learning within firms beyond beliefs |
| Trials and causal inference | mechanistic | switchback and sequential-with-washout designs executed on real days; matched-counterfactual evaluator (`evaluation/causal.py`) | daily operating results are noisy; both designs recover maintenance effects poorly (SECOND_ORDER_EMERGENCE.md) |
| Industry | mechanistic | machines, energy grids with curtailment, yields that depend on settings, wear and breakdown, freight on a link network, B2B trade, retail demand from households | yields and wear are parametric; household demand is a budgeted choice model |
| Labour | stylised | reservation wages, offers, counter-offers and contracts; effort under profit-sharing rules follows `gov.effort` (e = 0.55 + 2.5 s, clipped) | the effort formula is an assumption, not an emergent behaviour |
| Governance | mechanistic + stylised | constitutions are enforced at authorisation; proposals execute when votes pass | simulated members vote from trust in the proposer, not from modelled payoffs |
| Negotiation and reputation | mechanistic + stylised | offers against private reservations; breaches create public records; trust is Beta-updated | concession rules are heuristics; optional LLM actors decide only inside grounded bounds |
| Information | mechanistic | forum posts, mail, public evidence records; `#evidence` references are checkable; beliefs update by source | people read a bounded number of posts; no rumours beyond posted text |
| Synthetic web | mechanistic | real HTTP semantics through `net.fetch`, sessions, consent login, per-app storage, rate limits, outages, a crawler and a link-ranked index | server-rendered only; no cross-site cookies, TLS or DNS hierarchy |
| Browser | mechanistic | real Chromium against the front end; hermetic resolver | Chromium sees the front end's rendering of world pages |
| Adversarial content | mechanistic | injection pages, fraudulent capability, spam and false docs created at genesis by simulated actors | static adversaries; no adaptive attackers |
| NPC invention | mechanistic | innovators generate controller programs from a grammar, deploy them, read trial evidence and iterate | one grammar family (capacity/placement); field trials are the only test |
| Science and technology | mechanistic | designed experiments on real production runs; process notes sold with world-verified evidence | the response surfaces are parametric |
| Population scale | stylised | thousands of statistical consumers and households; tens of minds | statistical consumers are not reasoning agents and are reported separately (SCALING.md) |

Reading guide:
* A result that depends on a **stylised** row shows the consequences of that row's assumption. Examples: an
  effort gain after a sharing change, or a firm's adoption rule. It is not independent evidence about the real
  world.
* Results on **mechanistic** rows can still be artefacts of parameters, but the causal path is executable and
  inspectable in the event log.
