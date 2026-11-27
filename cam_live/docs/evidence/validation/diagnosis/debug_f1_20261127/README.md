# Why the integrated agent chose near random in LabWorld F1 — measured (2026-11-27, debug scale)

Engineering diagnostic (tiny random backbone + debug core on the CPU; per the operator's rule conclusions about
learning come from the real configuration — see `../../meta_real/`). One agent per arm, 8 lifetimes of the S
evaluation family F1 (same world seeds 10000–10007 in every arm, 16 decisions, k = 16, close-out to the deadline),
`diagnose_f1.py`; `summary.json` from `summarise.py` (bootstrap 95 % intervals over lifetimes / decisions).
`base` ran 4 lifetimes (stopped to free the CPU); `fix2` 7.

| arm | what changed (cumulative) | verified progress | choice quality (chosen / best in C_t) | uniform over C_t | interventions offered of 16 | zero-value decisions | waits |
|---|---|---|---|---|---|---|---|
| base | round-2 code before the diagnosis | 0.31 [0.07, 0.58] | 0.09 | 0.13 | 4.0 | 78 % | 27 % |
| fix1 | walker enumerates small spaces completely; verified progress = the decision's attributed signed outcomes (level change elsewhere) | 0.65 [0.45, 0.85] | 0.20 | 0.20 | 9.1 | 37 % | 18 % |
| fix2 | + 0 for unattributed decisions in attributing objectives | 0.56 [0.33, 0.76] | 0.19 | 0.22 | 9.9 | 42 % | 19 % |
| fix2lr | fix2 + live inner policy step `optim.inner_lr` 0.3 (default 0.01) | 0.79 [0.58, 0.95] | 0.25 | 0.22 | 9.6 | 36 % | 9 % |
| fix3 | fix2 + selection = policy logit + standardised predicted goal value; no level query where outcomes are attributed | 0.72 [0.43, 0.96] | 0.29 | 0.22 | 9.9 | 10 % | 7 % |

Mechanisms found (each measured in the files here):
1. **Candidate generation.** The schema walker took 4 evenly spaced settings per tool, i.e. 4 of the 12 interventions
   (4 channels x 3 efforts); the best one was usually not offered (`n_act_offered` = 4.0, base). With complete
   enumeration of small spaces it is offered in almost every decision. This is the largest single effect.
2. **Reward.** At record time the bootstrap reward is ~0 for every decision (|r| <= 0.002, base): the verified term
   never had an input in a runtime run (INT2-1). Crediting the *level change* of the objective metric to the
   decision in flight paid waits (0.57) and probes (0.33) more than interventions (0.43) (fix1,
   `mean_inner_bootstrap_by_op`); crediting attributed outcomes only, the inner update's reward pays interventions
   alone (fix2lr: lab.act 0.43, everything else 0.001); Spearman(inner reward, truth of the chosen action) 0.05 ->
   0.25. The verified term saturates at the reward bound (INT2-6).
3. **Policy step.** One live policy update moves the heads by ~0.005 (L2) and the selection probabilities stay within
   0.059–0.073 for every candidate of 16 (uniform 0.0625) in every arm with the default step; `sep_debug.json`: one
   SGD step of 0.1 on log pi(candidate 0) moves its probability from 0.0826 to 0.0852 — the policy head's gain is
   tiny relative to the step (INT2-7). With step 0.3 the heads move by 0.19 per update and waits fall to 9 %.
4. **Selection ignored the world model** (S-5). Adding the standardised predicted goal value to the logits cuts
   zero-value decisions from 42 % to 10 % and raises choice quality above uniform-in-offer (0.29 vs 0.22).
5. **Meta-updates**: rarely applied at this scale (insufficient independent verified groups per window; zero
   advantage); the reward-parameter hash changed in most lifetimes after C-10.

Not established here: whether these hold on the real configuration (separate runs), and CIs of the fix variants
overlap each other (8 lifetimes each); only base vs fix2lr / fix3 is clearly apart.
