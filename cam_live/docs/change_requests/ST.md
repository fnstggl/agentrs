# Change requests of ST (autonomous strategic invention; integrator of agent.py)

## ST-1 (configs/cam_live.yaml, owner M): proposer prompt window 6144 -> 8192 tokens

**What.**

```diff
 nn:
-  max_prompt_tokens: 6144
+  max_prompt_tokens: 8192
```

**Why (measured).** The live planning task's proposer prompt (59 permitted tools before the hotfix, every one listed
by S-6b) was 5,252-5,900 tokens in the live run's first 35 decisions against the 6,144-token window (decision
diagnostics `generation.prompt_tokens`, `prompt_fit`). ST's OWN RECORDS section (the agent's executed actions of the
objective with arguments, expectation and result, and the text of its own notes / requests; `agent.proposer_memory_chars`
2,400 characters, ~650 tokens) does not fit beside the 3,000-character observation digest: the fitting shrinks both
(the backbone keeps a too-long prompt's END, so nothing is ever sent over the window, but the agent's history is then
cut to a few lines). Qwen3-4B's native context is 32 k (40,960 positions). Cost: KV cache ~147 KB per token per row
(36 layers x 8 KV heads x 128 x 2 x bf16); 4 samples x 8,192 tokens = ~4.8 GB at most (vs ~3.6 GB at 6,144). All ST
staging runs used 8,192 through their configuration overlay (equal windows across conditions).

**Workaround.** Without it the agent still runs: the fitting keeps every tool listed and shortens the observation
digest and the OWN RECORDS section together (`CamLiveAgent._fitted_digest`).

## ST-2 (informational, IMP): goal-value Thompson sampling concentrates on a candidate whose predicted value is fitted noise

Live run decisions 15-35 (`decisions.jsonl`, rule `thompson_policy`): walker `http.request` (refused every time,
P-TOOL-NOT-AUTHORIZED) had predicted objective-metric values 0.3-0.8 vs ~0.13 for every other candidate, although the
objective metric was 0 at every reading; standardised within the decision, its Thompson samples won with probability
up to 0.64. The hotfix (a0f3411) removes unauthorized tools from the candidate set, but the mechanism — noise-fitted
goal-value differences standardised by the spread of the posterior means — can single out any candidate while the
objective metric has no variation (planning phase). Not changed by ST (predict/ is IMP's).

## ST-3 (reward/assemble.py, reward/inner.py, reward/bootstrap.py, owner RW): realised information value as a bootstrap input — APPLIED by RW in a055332

**What.** The agent writes, for every executed decision, `reward_signals.information_value` (ST,
`CamLiveAgent._information_value`): the world model's mean NLL of the next 2 decisions' matured outcomes with the
decision's context items (its action item and its result's observations) removed, minus with them (same weights, same
targets; nats; > 0 = what the action brought into the context improved CAM's later predictions). This request makes it
a bootstrap input exactly like `learning_progress` (the assembler's bootstrap-only extras path, so R_psi's signal
vector and checkpoint shapes are unchanged): component `information_value = w * tanh(IV)`, `w` 1.0.

**Why (master's question, 2026-11-28 12:10).** In the planning phase the objective metric is 0 at every reading; every
existing bootstrap term (learning progress of the decision's own outcomes, delta_v, the drop of the goal-value
epistemic std after the objective reading, verified progress) is the same for an action whose only effect is
information (reading a document, writing a note) as for any other action, and VOI covers only outcome queries. An
information-only action therefore had no pathway to be reinforced. With this input the reward the live policy update
uses values what an action's result did to CAM's own later predictions — generic, from the model's own predictions,
never a bonus for a tool.

**Exact diff** (against HEAD 0e2c541's reward files; also applies to a0f3411's; tested on a patched copy: `tests/reward`
+ `tests/integration/test_st_iv.py::test_information_value_reaches_the_bootstrap_reward_when_rw_applies_st3`, which is
skipped until applied; the ST long staging run used it on a scratch copy):

```diff
--- a/cam_live/src/cam_live/reward/assemble.py	2026-11-28 12:10:17.528557512 +0000
+++ b/cam_live/src/cam_live/reward/assemble.py	2026-11-28 12:10:17.564556797 +0000
@@ -198,7 +198,7 @@
                 continue
             if k in SCALAR_NAMES:
                 out[k] = f
-            elif k == "learning_progress":
+            elif k in ("learning_progress", "information_value"):     # bootstrap-only inputs (ST-3)
                 extras[k] = f
     return out, extras, t_r
 
@@ -211,6 +211,9 @@
     learning_progress: Tensor         # [B] float64, NaN where no value was recorded (bootstrap input only)
     reward_time: Tensor               # [B] float64
     keys: List[str]                   # noisy-TV context key of each row
+    #: ST-3: realised information value (agent annotation ``information_value``: later prediction loss without minus
+    #: with the decision's context items), NaN where not measured; bootstrap input only
+    information_value: Optional[Tensor] = None
 
 
 class SignalAssembler:
@@ -234,11 +237,12 @@
                  b: Optional[Tensor] = None, b_next: Optional[Tensor] = None, e_g: Optional[Tensor] = None,
                  has_next: Optional[Tensor] = None, device: Optional[torch.device] = None,
                  dtype: torch.dtype = torch.float32) -> AssembledSignals:
-        rows, lps, times = [], [], []
+        rows, lps, ivs, times = [], [], [], []
         for rec in records:
             sig, extras, t_r = record_signals(rec, store)
             rows.append(sig)
             lps.append(extras.get("learning_progress", float("nan")))
+            ivs.append(extras.get("information_value", float("nan")))
             times.append(t_r)
         if self.evaluator is not None and b is not None and e_g is not None and records:
             hz = torch.tensor([self._horizons(r) for r in records], dtype=torch.float64, device=b.device)
@@ -261,7 +265,8 @@
         return AssembledSignals(signals=sig, tensor=sig.to_tensor(dtype, device), mask=sig.mask(dtype, device),
                                 learning_progress=torch.tensor(lps, dtype=torch.float64),
                                 reward_time=torch.tensor(times, dtype=torch.float64),
-                                keys=[context_key(r) for r in records])
+                                keys=[context_key(r) for r in records],
+                                information_value=torch.tensor(ivs, dtype=torch.float64))
 
     def assemble_batch(self, batch: DecisionBatch, store: ExperienceStore, *, b_next: Optional[Tensor] = None,
                        has_next: Optional[Tensor] = None) -> AssembledSignals:
--- a/cam_live/src/cam_live/reward/inner.py	2026-11-28 12:10:17.532557433 +0000
+++ b/cam_live/src/cam_live/reward/inner.py	2026-11-28 12:10:17.564556797 +0000
@@ -312,6 +312,7 @@
             ok = torch.isfinite(v)
             return torch.where(ok, v, torch.zeros_like(v)), ok.to(torch.float64)
         lp, lp_ok = col(asm.learning_progress)
+        iv, iv_ok = col(getattr(asm, "information_value", None))      # ST-3
         dv, _ = col(s.delta_v)
         vd, vd_ok = col(s.verified_progress_delta)
         eb, eb_ok = col(s.epistemic_before)
@@ -321,7 +322,8 @@
         ea = torch.where(both, ea, torch.ones_like(ea))           # no measured drop: no exploration reward
         keys = [k if bool(ok) else None for k, ok in zip(asm.keys, both)]
         return self.bootstrap(learning_progress=lp * lp_ok, delta_v=dv, epistemic_before=eb, epistemic_after=ea,
-                              verified_delta=vd, verified_mask=vd_ok, context_keys=keys, gate_update=gate_update)
+                              verified_delta=vd, verified_mask=vd_ok, context_keys=keys, gate_update=gate_update,
+                              information_value=iv * iv_ok)
 
     def build_batch(self, store: ExperienceStore, records: Sequence[ExperienceRecord], *,
                     batch: Optional[DecisionBatch] = None, policy_only: bool = False,
--- a/cam_live/src/cam_live/reward/bootstrap.py	2026-11-28 12:10:17.536557353 +0000
+++ b/cam_live/src/cam_live/reward/bootstrap.py	2026-11-28 12:10:17.564556797 +0000
@@ -79,6 +79,7 @@
     delta_v: float = 1.0
     exploration: float = 0.5
     verified: float = 2.0
+    information_value: float = 1.0    # ST-3: tanh of the realised information value (nats), like learning progress
 
 
 @dataclass
@@ -99,7 +100,8 @@
     def __call__(self, *, learning_progress: Optional[Tensor] = None, delta_v: Optional[Tensor] = None,
                  epistemic_before: Optional[Tensor] = None, epistemic_after: Optional[Tensor] = None,
                  verified_delta: Optional[Tensor] = None, verified_mask: Optional[Tensor] = None,
-                 context_keys: Optional[Sequence[str]] = None, gate_update: bool = True) -> BootstrapOutput:
+                 context_keys: Optional[Sequence[str]] = None, gate_update: bool = True,
+                 information_value: Optional[Tensor] = None) -> BootstrapOutput:
         """All inputs [B] (None = not available, contributes 0). ``context_keys`` names the context of each row
         for the noisy-TV gate (rows without a key are not gated). ``gate_update=False`` reads the gate without
         advancing it (replay of a step the gate has already seen; the gate moves once per real step)."""
@@ -109,6 +111,9 @@
         comp["learning_progress"] = self.w.learning_progress * (
             zeros if learning_progress is None else torch.tanh(learning_progress.to(ref.dtype)))
         comp["delta_v"] = self.w.delta_v * (zeros if delta_v is None else delta_v.to(ref.dtype))
+        # ST-3: what an action's result changed in CAM's later predictions (realised information value, nats)
+        comp["information_value"] = self.w.information_value * (
+            zeros if information_value is None else torch.tanh(information_value.to(ref.dtype)))
         if epistemic_before is None or epistemic_after is None:
             comp["exploration"] = zeros
         else:
```
