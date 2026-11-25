# E1 evaluation plan (prespecified before any fit)

The binding, machine-readable version is [E1_evaluation_plan.json](E1_evaluation_plan.json). Every model card records
the SHA-256 of that file and the commit that holds it; `tests/v5/test_gate_SP_5_5.py` checks that this commit precedes
the commit that added the checkpoint, and that the card's metrics are the ones named here.

## What is fitted

| model | domain | data | estimand | identification | primary metric |
|---|---|---|---|---|---|
| content_consumption | content selection | KuaiRand-Pure random-exposure log | P(consume \| shown in the featured feed), exposure not selected on preferences | randomised exposure | log loss |
| content_consumption_feed | content selection | KuaiRand-Pure standard log | P(consume \| shown by the production recommender) | observational | log loss |
| watch_duration | watch duration | KuaiRand-Pure, both logs | distribution of log1p(watch seconds) given consumed, duration, affinity | observational | mean pinball loss over 21 quantiles |
| return_daily | browsing frequency | KuaiRand-Pure, both logs | P(active on day d \| activity before d) | observational | log loss |
| session_rate | browsing frequency | KuaiRand-Pure, both logs | heterogeneity of sessions per active day | observational | Poisson deviance |
| news_click | content selection | MIND-small | P(click \| impression, history, text, category, popularity, age) | observational | log loss |

## Rule

1. Fit every candidate and every simple baseline on the train split only.
2. Select the candidate with the best primary metric on the validation split (ties go to the simpler one).
3. On the test split (held-out users and a later period), keep the selected model only if no simple baseline is
   strictly better on the primary metric; otherwise keep the baseline and explain the failure in the report.
4. Report a user-cluster bootstrap interval (200 resamples) of the kept model's test metric and of its difference from
   the best baseline.
5. Secondary metrics: Brier score, expected calibration error (15 equal-width bins), AUC and, for MIND, impression
   AUC, MRR and nDCG@5/10; distributional checks (PIT uniformity, interval coverage, Wasserstein distance, completion
   and view rates by duration decile; simulated active-day distributions and retention curves).

## Splits

Users are assigned to train / validation / test by a hash of their anonymised ID (70/15/15 for KuaiRand). KuaiRand
tests use test users on 2022-05-03..05-08 (choice and watch time) or 05-05..05-08 (return); MIND tests use the dev
file (2019-11-15, mostly users absent from the train file). The random-exposure log validates the observational models:
every observational choice model is also scored on random-exposure rows of test users.
