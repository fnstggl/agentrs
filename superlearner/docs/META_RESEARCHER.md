# The Meta-Researcher

The outer learner. Its object of study is the inner learner's *learning algorithm*; its measure of success is
how much capability a freshly initialised controller acquires from a fixed amount of experience. Results of the
runs done so far are in [RESULTS.md](RESULTS.md); this file describes the mechanism.

## The loop (`meta/researcher.py`, one `round`)

| Step | What happens | Code |
|---|---|---|
| A diagnose | From the champion's runs: the trusted learning curve (first/last held-out success, AULC, late training success) and the learner's own diagnostics (KL, entropy, explained variance of the critic, clip fraction, gradient norm, valid-action rate, early stops). Rules turn these into bottleneck tags (`credit`, `trust_region`, `exploration`, `stability`, `validity`, `plateau`) and a paragraph of text | `diagnose` |
| B hypothesize | A pool of executable proposals from three generators (below); the trainable surrogate picks which get an experiment (or a uniform random choice, for the control arm) | `propose_pool`, `select` |
| C implement | The candidate's own source tree is created with the proposed files; the diff and proposal are stored with it | `make_candidate` |
| D verify | Protected files byte-identical to the evaluator's tree; source parses; then a self-test **inside the confinement** on real episodes from the hub: an episode runs, recomputed likelihood equals acting-time likelihood, advantages and losses are finite, an update changes the weights and keeps them finite, gradients reach policy and critic | `verify`, `candidate_run.selftest` |
| D' probe | One update and the rule's loss and gradient on a fixed, deliberately varied batch (seeded controller; perturbed old log-probabilities; finished and truncated episodes; successes and failures), **inside the confinement**; a candidate whose fingerprint equals the champion's changes nothing on this data and is rejected unrun (`reject_inert`, default on) | `probe`, `is_inert`, `candidate_run.probe` |
| E train | A fresh controller is trained by the candidate's algorithm, confined, against its own env hub, under an experience budget the hub enforces | `run_one` |
| F compare | Learning curves from the hub's results file against the champion's runs on the same seeds, demonstrations, budgets, evaluation tasks and hardware | `compare` |
| G analyze | Differences in AULC (primary), final held-out success, training-time success and wall-clock, each with a bootstrap 95% interval over seeds and the probability of improvement; Pareto status across measures | `meta/compare.py` |
| H decide | Screening → development → confirmation. Promotion only if the predeclared rule holds at confirmation | `experiment`, `verdict` |
| I learn | The outcome goes to the ledger; the surrogate is refitted by gradient descent; the learned-rule search takes an evolution-strategy step when a generation is complete | `_finish` |
| J repeat | State (champion, measurements, surrogate data, search state, random states) is saved after every experiment; `--resume` continues | `save_state` |

## Where proposals come from (`meta/proposals.py`)

1. **`LLMProposer` — open-ended.** The frozen LLM is shown the diagnosis, the ledger's history and the source of
   one function of the learning rule, and writes a new version of that function: any computation with the same
   signature. `splice_function` accepts it only if it parses, defines that function with the same arguments, and
   adds nothing but functions and constants at top level. Unusable replies are recorded in the ledger
   (`proposal_rejected`).
2. **`RuleSpace` — a compositional program space.** The learning rule is four slots (credit assignment, policy
   objective, critic objective, combination), each filled by a code fragment with free constants, plus the rule's
   constants. Fragments: 5 × 6 × 4 × 4 with continuous parameters; proposals are mutations and crossovers of
   the champion's genome, rendered to a complete `algo/rule.py`. Every fragment is unit-tested to render to a
   working rule, and the baseline genome reproduces the hand-written rule numerically.
3. **`LearnedRule` — learning-rule discovery.** A 5→8→1 meta-network inside the rule computes a per-decision
   weight on the policy gradient from (advantage, likelihood ratio − 1, their product, |advantage|, 1); two more
   meta-parameters set the credit-assignment horizon λ and the trust region. The 59 parameters start at the baseline
   rule and are searched by an evolution strategy (antithetic Gaussian perturbations, rank-shaped update) whose
   fitness is the trusted AULC difference of an inner training run. No gradient is taken through training.

These are sources of *code*, not a menu of named algorithms. They are also not unbounded: the first is limited by
what a 14B model can write correctly, the second by its fragments, the third by its parameterisation
([LIMITATIONS.md](LIMITATIONS.md)).

## The trainable research policy (`meta/surrogate.py`)

An ensemble of small networks trained by gradient descent on the ledger. Input: a random projection of the frozen
model's embedding of the proposal text and diff, and structural features (source, which functions change, size).
Outputs: expected AULC difference against the champion, and the probability of passing verification. Selection is
by upper confidence bound. It is refitted after every experiment; with fewer than three outcomes it chooses
uniformly. `loo_report` gives its leave-one-out rank correlation, the honest measure of whether it has learned
anything yet.

## What "better" means

| Measure | Definition |
|---|---|
| sample efficiency (primary) | **AULC**: area under the held-out success curve over the experience budget, normalised to [0, 1] |
| final capability | held-out success at the last evaluation round |
| training-time success | mean verified success of training episodes over ten equal slices of the budget |
| generalisation | the evaluation list has unseen seeds and one world family that is never trained on |
| compute | wall-clock of the confined run; update and rollout seconds; LLM tokens (when the inner task uses the LLM) |
| robustness | every claim needs ≥ 3 seeds; confirmation uses 5 |

Decision rule (predeclared in `verdict`): **improved** if the AULC difference's 95% interval is above the margin on
≥ 3 seeds and final success did not drop by more than 0.05; **harmful** if the interval is entirely below zero;
otherwise **inconclusive**. A candidate that is faster but ends lower is not promoted. Trade-offs are reported per
measure and as Pareto status, never folded into one number.

## Multi-fidelity stages

| Stage | Seeds | Purpose | Outcome |
|---|---|---|---|
| verify + self-test | — | reject broken code in about a minute | `rejected_invalid` |
| screen | 1 | drop clearly worse candidates | `rejected_screen` if AULC is below the champion's by more than the margin |
| develop | 3 | estimate the effect | continue only if P(improvement) ≥ 0.8 |
| confirm | 5 | promotion test | `promoted` / `inconclusive` / `harmful` |

In the reported search the three stages use identical run settings and differ only in their seeds, so a seed that
was run at an earlier stage is reused, for the candidate and for the champion alike. Baseline (champion) runs are
made once per seed. Several candidates are evaluated side by side within a round; confined runs share a fixed
number of slots.

What makes two runs comparable is the experience budget (training decisions, counted and enforced by the env hub),
not wall-clock time: `timeout_s` is deliberately generous and only stops runaway candidates, because wall time
depends on what else the host is doing (it is recorded as a measure). The confined self-test checks the loss a
candidate uses *after* any critic-only warm-up phase.

## Changes that do nothing

A proposal can parse, keep its signature, pass the self-test and still be the champion's algorithm: an added
variable that is never used, an edit to a function the configuration never calls (the replay loss when no replay
steps are configured), a reformulation that is mathematically the same. Run as an experiment, such a candidate
produces a "difference" that is only the run-to-run noise of training. The probe (`candidate_run.probe`) computes,
confined and on one CPU thread, a fingerprint of what the candidate's learning code does on fixed data; equal
fingerprints mean no functional change on that data. The reported search was run before the probe existed, so its
inert candidates were run — which turned them into free A/A tests: their measured "effects" calibrate the noise of
single-seed screening (RESULTS, section 11). `experiment e` applies the probe to a finished search.

## Interruptions

The ledger is appended after every round; `state.json` holds the champion, the surrogate, the learned-rule search
and the random states. The trusted measures of every finished confined run are stored
(`trusted/<candidate>/<stage>_s<seed>.measures.json`, tied to the hash of the candidate's source tree and the stage
settings). `meta-search --resume` on the same directory continues with the remaining rounds; runs that had finished
are read back instead of spending their experience again, half-finished runs are set aside (`*.old<time>`) and
started clean.

## Reproducibility

Every candidate directory holds its full source tree, the proposal, the diff, hashes of editable and protected
code, each run's job file (configuration, seeds), log, checkpoints, and the trusted results file. The ledger is
append-only. Nothing a candidate does can overwrite the champion's tree or measurements.
