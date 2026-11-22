# Known limitations

Read together with [RESULTS.md](RESULTS.md) (what was measured) and [STATUS.md](STATUS.md) (implemented versus
demonstrated).

## The inner learner

* **The frozen LLM's content is not trained and only indirectly steered.** The controller decides when the LLM
  writes, about which document, which sampled variant, and whether its draft steps are followed, edited or
  dropped. It cannot change the tokens the LLM produces: no gradient reaches the frozen model, and a learned bias
  on its output logits (which would make the generated text itself a trained distribution) was designed but not
  built.
* **LLM-written services are unreliable.** With the documented formats in the prompt, reasoning enabled for
  structured files and error-specific repair, Qwen3-14B still produces a deployable, good-enough service only some
  of the time (numbers in RESULTS). Every arm that needs agent-written code inherits this ceiling.
* **Warm start carries most of the competence.** Reinforcement learning here improves and repairs a policy that
  imitation started (+8 to +12 points of held-out success, mostly where the warm start was weakest); from a random
  controller, multi-step tasks in a space of ~140 tools with structured arguments are not discovered by exploration
  in the budgets used. No experiment shows learning of an undemonstrated multi-step skill from reward alone:
  arms at 0% after the warm start stayed at 0% (`fund` with greedy actions), and six iterations of RL on three
  undemonstrated arms left them at 0%.
* **The update rule throttles itself.** A policy step that exceeds the KL limit is rolled back and the step size
  halved; the size recovers slowly. In the reported runs it was between 1/13 and 1/80 of its initial value after 12
  iterations, and the weights moved by 0.1–0.3% of their norm. Stability was bought with learning speed.
* **Episodic memory and agent-made tools are off in the reported learning runs.** A controller cloned without
  memory was thrown off by remembered episodes appearing in its context (42% with the memory it had accumulated,
  70% with memory reset). Training the controller *with* memory present from the start was not done; whether
  memory can help this agent is open. No agent-made tool was ever verified and reused.
* **Greedy and sampled behaviour differ.** A greedy controller repeats one failing call until the retry guard
  stops it (about 40% of its calls are accepted, against 100% for scripts); sampling gets past some of these and
  fails elsewhere. Both are reported; neither is the better policy everywhere.
* **Other learners in the world break it.** In the self-play arena the same checkpoint that lists and operates a
  service in 100% of single-actor tasks never got a listing accepted (permission and missing-resource errors).
* **Forgetting.** Learning a new skill by cloning erased two of three earlier skills unless the earlier
  demonstrations were mixed in. Nothing in the RL loop protects old skills except continued sampling of old tasks.
* **Demonstrations come from scripts that encode ordinary developer knowledge** (found a company before trading,
  read the format before writing code). They are recorded with provenance and limited to training-split worlds,
  but they are a strong prior on *what to try*.
* **Tasks are short-horizon.** Trained and evaluated arms take 3–40 decisions and up to 10 simulated days. Level-1
  goals (5,000 requests; revenue targets) and the mined open-world tasks of Crucible 3.0 were not trained.
  "Long-horizon" is supported by the architecture (recurrent state, waits, drafts, agent tools, whole-episode
  credit) and only lightly exercised.
* **Heads that are available but hardly learned.** `propose`, `draft_run`, `define` and `use` exist, execute and
  are tested, and the baseline without a controller uses `propose`/`draft_run` exclusively; whether the trained
  controller chooses them is reported in RESULTS. Tool-making earns nothing by itself and nothing in the trained
  tasks requires it.
* **Curriculum.** The adaptive curriculum responds to competence, uncertainty, prerequisites and failure classes,
  but with a warm start prerequisite gating was harmful (arms the warm start had already solved were locked,
  not rehearsed, and forgotten) and was switched off for the reported runs.
* **Value ensemble spread is used as an uncertainty signal for the curriculum only**; exploration is entropy-
  regularised sampling.
* **One simulated actor per world in training.** Other actors are Crucible's own simulated firms and people.
* **Operating an LLM-written service to a traffic goal was achieved by nobody** — not the controller, not the
  scripted teacher (0 of 8 and 0 of 11). There are therefore no demonstrations for it, and the reason the
  teacher's correctly listed services fall short was not diagnosed.
* **RL with the LLM in the loop is slow.** A training episode that needs a service written takes minutes of
  generation; four iterations took over an hour and showed no effect. Experiment B's positive result is for the
  eight arms that need no LLM-written content.

## The Meta-Researcher

* **Scope of what candidates changed.** All evaluated candidates changed `algo/rule.py` (the learning rule,
  including its replay component). Candidates may also replace the controller architecture, features, memory and
  curriculum, and the pipeline would run them, but no proposal generator writes those files yet, and
  demonstrations recorded with the champion's featurisation would not fit a candidate that changes features.
* **Three proposal sources, each bounded**: what a 14B model writes correctly; a hand-written fragment set
  (5 × 6 × 4 × 4 × 4 structural choices with continuous constants); a 59-parameter learned rule. This is a large
  space of programs, not an unbounded one.
* **The inner task of the search is small.** A 2.5M-parameter controller with hash text embeddings on CPU (the GPU
  is not reachable inside the confinement), six task arms that need no LLM content, a few thousand decisions per
  run. An algorithm that wins there has not been shown to win for the full controller. No candidate won there,
  so no transfer check was made.
* **Statistical power.** Screening uses one seed; development three; confirmation five. One screening run of the
  champion's own algorithm differs from another by about ±0.03–0.04 of AULC (measured on candidates that turned
  out to change nothing), which is as large as most screened effects; the best screened candidate lost its
  advantage when two more seeds were added. Effects of the size found here cannot be established at this scale.
* **The surrogate has little data and did not help.** It is a real trained model with a leave-one-out report; in
  four rounds its picks were not better than the pool average (RESULTS, section 11). With screening outcomes that
  are mostly noise there was little for it to learn.
* **A 14B model as algorithm author.** Of eight functions it wrote, one did not parse, one failed the self-test,
  three ran but changed nothing (an unused variable; a code path the configuration never reaches; a clipping
  variant that is mathematically the same), and three were real changes of one to three lines. The search that produced the
  reported results did not yet reject inert candidates before running them; the probe that does was added
  afterwards and is on by default for new searches. The probe compares computations on one varied batch, so a rule
  that differs only on data unlike that batch would still be called inert.
* **Diagnosis is rule-based.** Bottleneck tags come from fixed thresholds over training diagnostics; the LLM turns
  them into hypotheses but does not derive them.
* **No recursive self-improvement is claimed.** A promoted rule becomes the champion that later candidates are
  compared with and mutated from; whether successive promotions compound was not tested.

## Security

* Confinement is uid + rlimits + Landlock + seccomp inside one unprivileged container; no VM or container-grade
  isolation is available on this host (see [SECURITY.md](SECURITY.md)).
* The champion's own training (`superlearner train`) runs unconfined; it is repository code.

## Engineering

* One GPU for everything. The frozen-LLM server and the training runs share one 46 GB card; before the job queue
  existed, runs died of out-of-memory or stalled. Generation throughput (about 30–40 tokens per second in total)
  bounds every experiment that needs LLM-written content.
* The host is shared with other tenants and the instance stopped once for six hours; everything is restartable
  from files, but timings in RESULTS are from a loaded machine.

* Rollouts are CPU processes with per-worker copies of the controller; there is no batched GPU inference server
  for the policy.
* A few world seeds fail inside Crucible's own world generation (an incumbent cannot afford its genesis bounty).
  Fixed task lists skip them deterministically; sampled training tasks that hit one are dropped and counted in
  `errors.log`.
* GPU nondeterminism: two cloning runs with the same seed and demonstrations can give different warm starts
  (see RESULTS), which is why compared runs branch from one saved warm-start checkpoint.
* The frozen-LLM cache makes repeated prompts free but means that a run's LLM cost depends on what earlier runs
  asked; `gen_tokens` (generated in this run) and `served_tokens` (delivered, cached or not) are both recorded.
* **Evaluation is not perfectly repeatable outside `tiny` worlds.** Crucible charges services by *measured* CPU
  time unless a world asks for its deterministic line-counting model, which is about four times slower (74 s
  instead of 17 s for three simulated days of a bazaar world under load). Training and evaluation use the measured
  model, so the same checkpoint on the same tasks can differ by a few tasks out of 80 between runs made under
  different host load (observed: 0.613 and 0.662). Compared conditions are run side by side and over several
  seeds for this reason.
