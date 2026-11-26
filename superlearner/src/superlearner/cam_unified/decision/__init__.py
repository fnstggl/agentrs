"""Mechanism B: CAM's mandatory consequential decision bottleneck (redesign plan 2026-11-26 §2, §3 W5; DOM-04, DOM-09).

    Observation -> Z_t = F(Z_{t-1}, ctx) -> R_t = Reason(Z_t) -> I_t = Intent(R_t, ctx_out)
                -> K candidates ~ ActionGenerator(I_t, ctx_out[TOOL], ctx_out[OBS])      (one batched decode)
                -> Q_k = evaluator.score(R_t, B_t, G_t, A_k, dt_k)                          (the CONCRETE actions)
                -> k ~ softmax(Q / tau) (veto Q_k < veto_q)  -> stamped Action -> execution

* `intent.IntentHead`            I_t [B, n_intent, D]: learned queries reading [R_t; ctx_out; OPTION; BUDGET]
* `evaluator.CandidateEvaluator` Q of every concrete candidate (world-model prior of the candidate's action tokens,
                                 a learned residual head, the reward model's potential and V_int)
* `select`                       selection over the candidates, its exact log-probability and the cam_eval stamp the
                                 execution layer checks (`require_stamp` / `authorize`)
* `fast.fast_params`             the fast parameter set phi of the meta-gradient (W8)

The goal reaches the action only through Z_t / R_t / ctx_out / I_t: the generator in intent mode has no access to
raw goal, belief or tokenizer tool tokens and no R-independent tool prior (closed by construction; tests in
tests/cam_unified/dom/test_dom_bottleneck.py).
"""
from .intent import IntentHead, intent_tokens                                   # noqa: F401
from .select import (STAMP_KEY, UnauthorizedAction, authorize, is_stamped, require_stamp, select,  # noqa: F401
                     selection_log_prob, verify_stamp)
