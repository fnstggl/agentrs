# E3 and E4 plans (workstream E, 2026-11-26)

Both plans are committed before any of the fits or reviews they govern; SP-5.5 checks the commit order against the
cards and reviews.

* `E3_review_plan.json` - review of the nine kept models: each is scored on its own test rows against the
  simulator's existing rule-based behaviour (not only constants and popularity), a long-horizon measure (forward
  simulation for models whose outputs feed their inputs, multi-day stability otherwise) and compute cost, with a
  prespecified rule for which contender is kept. Results: `docs/v5/calibration/reviews/<model>.json`.
* `E4_new_domains_plan.json` - six new behavioural domains on legitimate public data (KuaiRand session length and
  creator return; Online Retail II customer return and product price response; GH Archive participation return and
  star cascades) and `news_click_semantic`, a news-choice model whose text affinity is a cosine in the shared semantic
  encoder space so that the simulator can compute it without batch-relative mapping.

Identification: all E4 models are observational (stated per model); the randomised models remain
`content_consumption@1` (KuaiRand random exposure) and `commercial_response@1` (Criteo randomised trials).
