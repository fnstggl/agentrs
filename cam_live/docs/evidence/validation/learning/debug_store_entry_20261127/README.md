# Repeated staging task, debug scale (2026-11-27, commit 3597af2 + d74f810 runner)

`run_arm.py` x 4 arms x 4 seeds (0-3), 12 objectives each ("increase the entries of the entry book to n+1",
staging `variants` site layout A, 30 decisions per objective, the browser put on the site's start page before
each objective), tiny random backbone + debug core on the CPU, **backbone proposals off** (the tiny random LM
writes no valid action) and planner off. 16 processes ran concurrently on a loaded machine (10–15 s per decision).
Summary: `summary.json` (`aggregate.py`: Wilson intervals per episode, bootstrap over seeds for half-to-half
changes and arm differences).

Result — an honest negative:

| arm | verified successes | 1st half -> 2nd half success (mean over seeds, 95 % bootstrap CI of the change) |
|---|---|---|
| learn_skills | 12 / 48 | 0.38 -> 0.12, change -0.25 [-0.46, +0.04] |
| learn_primitive | 16 / 48 | 0.42 -> 0.25, change -0.17 [-0.29, -0.04] |
| frozen_skills | 7 / 48 | 0.17 -> 0.12, change -0.04 [-0.25, +0.13] |
| frozen_primitive | 10 / 48 | 0.21 -> 0.21, change 0.00 [-0.17, +0.33] |

* The agent completes the unfamiliar two-step task (fill the designation field with an observed string, then press
  the store button — the browser actions are grounded in the observed page) in 9–33 % of objectives, also without
  any learning; nothing site-specific is in the code.
* It does **not** get better with repetition. With learning on, the second half is worse than the first (learn_primitive:
  CI excludes 0); learning vs frozen in the second half: +0.04 [-0.21, +0.29] (primitive), 0.00 [-0.17, +0.21]
  (skills). Primitive operations per objective do not fall.
* Skills: 9 (learning) / 4 (frozen) `skill.induce` attempts over verified-successful traces, none registered a
  skill (validation of the induced procedure failed), so `skill.run` was never available: primitive-only vs
  skills-enabled shows no difference that can be attributed to skills.
* Debug scale only (random tiny LM, no backbone proposals): per the operator's rule this is an engineering
  diagnostic, not a conclusion about learning of the real configuration. The diagnosis of why learning does not
  help is in `docs/INTEGRATION.md` §9.
