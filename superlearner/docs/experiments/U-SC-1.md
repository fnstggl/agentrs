# U-SC-1 — compute-normalised scaling comparison (pre-registration)

Spec: `CAM_UNIFIED_SPEC_DOM.md` §7; plan `CAM_UNIFIED_REDESIGN_PLAN_2026-11-26.md` §5 E5 (run only if time allows).
Manifest: DOM-12. Committed **before** any scaling run. With 3 / 2 / 1 seeds this comparison is **descriptive**:
it carries no inferential claim and cannot support EMPIRICALLY_DEMONSTRATED (≥ 3 seeds per compared learner are
required; only dom_dev has them).

## Scales (same DOM switches, same rapid suite, observation-only)

| Scale | Preset | Core | Semantic model | Seeds |
|---|---|---|---|---|
| ~14.5 M | dom_dev (8 × 384) | measured by `count_parameters` | Qwen3-0.6B | 3 (0–2) |
| ~150 M | dom_150m (16 × 768) | measured | Qwen3-4B | 2 (0–1) |
| ~1 B | dom (32 × 1536) | measured | Qwen3-4B | 1 (the U-DOM-1 E1 run) |

The 500 M point is dropped (plan §5). Training: the rapid train split, the U-DOM-1 world-seed scheme, B = 4, the same
decision budget per scale (the E1 1B budget truncated to the smaller scales' budget where they stop earlier).
Evaluation: the U-DOM-1 confirm protocol (rapid val split, seeds ≥ 500,000), weights-only, at matched decisions.

## Reported (fixed in advance)

* decisions to competence: the first decision count at which the rolling 512-decision mean training-world U ≥ 0.5
  (∞ if never), per scale and seed;
* confirm U at equal environment decisions, and at equal GPU-busy seconds (the smaller scales' checkpoint closest to
  the larger scale's GPU-busy seconds);
* unfamiliar-task adaptation speed (confirm U after 256 and 2,000 decisions of continual learning);
* compute per scale (GPU-busy seconds, semantic tokens, wall time, peak memory).

Intervals (stats.two_level method "t") are shown where a scale has ≥ 2 seeds; the 1B row is a single point.

## Amendments

None.
