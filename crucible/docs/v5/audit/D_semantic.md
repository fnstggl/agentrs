# Audit D_semantic: semantic and multimodal intelligence (workstream D)

Auditor: independent audit of branch `feat/crucible-5-complete` @ `a881d18`, worktree
`/workspace/agentrs/.worktrees/c5-complete`. Scope: MASTER_SPEC Parts 3.2 class 4, 4.3, 13.4, 13.5, Gates E and H,
SP-3.2b, `crucible/src/crucible/semantic`. Clarification 1 (no CAM dependency) and the two new user amendments
(observation-only default; recovery preamble: nothing counts without reproducible evidence) were applied.

Probe scripts, logs and the raw evidence JSON from my runs are in
`/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/D_semantic/`.

## 1. Bottom line

* The **encoders are real** (MiniLM-L6, CLIP ViT-B/32, AST + Whisper-tiny, video = frames + audio + temporal
  features, code = declaration summary + structure). I reproduced E.1, E.2, E.4 and E.n1 on a fresh cache.
* **They do not drive participant behaviour.** No E.3 probe is registered, and the default social engine uses a
  lexical random-indexing encoder (`social/audience.py:112`). When I switched a social world to the pretrained
  encoders (`aud.encoder="semantic"`), the audience stopped responding to what the content is about. The cause is a
  random projection of MiniLM vectors into an interest space they have no alignment with (`social/audience.py:470-479`).
  E.3, "controlled artifact changes affect participant behaviour through learned representations", is not met. The
  present design cannot meet it.
* **E.5 now fails** on the merged branch: the static audit finds 5 violations. They look like false positives:
  task quality in `domains/firms.py` and a `relevance()` method in `social/reference.py`. The audit itself only
  matches field names, so it cannot prove the absence of a creator-assigned score.
* The **shared LM service** is well engineered: batching, bounded generation, a prefix KV cache, recording and
  replay, one weight copy. Two parts do not deliver what they claim:
  * The prefix cache gave no speedup (0.986x).
  * In bfloat16 the prefix cache changed 3 of 8 greedy outputs.
* **Gate H reproduces exactly** (same counts as the interrupted session reported), but the substance is weak:
  * 74 of 129 calls were rejected.
  * 2 of 8 goals were met: a debt payment and a server rental, each one tool call.
  * No collaborative result and no productive artifact (document or program) was verified.
  * Only 2 of the 15 "consequential actions" serve a goal. My per-participant dump shows the other 13 are junk:
    `bank.open_account` x4 after the goal was done, `cal.create` x6 titled "cal", and `mail.send` x3 whose body is
    the literal string "mail.send".
  * Everything ran in the 4.0 `open_digital` family, not a v5 standard world.
* **Semantic participants are not integrated anywhere.** Nothing outside `crucible/semantic` and the tests imports
  `SemanticParticipant`, `SemanticGroup`, `SemanticService` or `build_society`. The population's class-4 rows are
  inert: `population/core.py:323` gives them `policy_group` 200, and `:333` sets `next_active_at = NEVER` for
  classes above 2. `civ` and `multiagent` are still stubs.
* **SP-3.2b is honest but not acted on.** Its own result is that Qwen3-0.6B is not effective on held-out templates
  for programming, negotiation and strategy (Qwen3-1.7B is needed). Even so, DECISION_LOG D-10 makes 0.6B the model
  for everything, and there is no per-capability routing. Also, the "~1B" candidate is a 1.7B model.

## 2. What I ran

| # | Command / probe | Result |
|---|---|---|
| 1 | `slots --cpu 1 --gpu-mb 3000 -- pytest -q tests/v5/test_gate_E.py` (fresh `$CRUCIBLE_CACHE`, 96 s) | 4 passed (E.1, E.2, E.4, E.n1); **2 failed: E.3 (no probe registered), E.5 (5 violations)** |
| 2 | `slots --cpu 1 --gpu-mb 7000 -- pytest -q -s tests/v5/test_gate_H.py` (846 s) | 9 passed. The tests only record evidence and never assert; metrics below |
| 3 | `pytest -q tests/test_v5_semantic_{service,participants,media,media_world}.py` (CPU, 1 thread) | 36 passed |
| 4 | `slots -- pytest -q tests/test_v5_semantic_encoders.py` | 19 passed |
| 5 | `probe_encoders.py` (CPU): my own artifacts; cache edge cases; world record | see section 3 |
| 6 | `probe_aud_semantic_space.py`: interest-space alignment of the `aud` semantic path | margin -0.011 +/- 0.046 (lexical 0.425) |
| 7 | `probe_e3_integrated.py`: two crowd/aud/feed worlds (1,500 people, 3 days) differing only in `aud.encoder` | semantic path gives no topic match (section 4) |
| 8 | `probe_service.py` (SmolLM2-135M, CPU) | bound enforced, 16/16 prefix hits, deterministic, prompt not in log |
| 9 | `probe_society.py` (Gate H society re-run with per-participant dump and a regenerate-without-model check) | section 5 |

Consolidated regression XMLs (`docs/v5/evidence/integration/regression_consolidated_g*.xml`) show no failure in the
semantic test files. The Gate E and H tests are not part of that regression.

## 3. Encoders and cache (Parts 4.3, 13.5; E.1, E.2, E.4, E.n1)

Code: `semantic/encoders/{models,base,text,image,audio,video,code}.py`, `semantic/cache.py`,
`semantic/encoders/__init__.py`, `semantic/encoders/module.py` (kernel module `embed`).

**Reproduced (fresh cache, GPU slot):**
* E.1:
  * `real_encoder_modalities=4` (5 with code), `stub_vectors_detected=0`, `min_encoder_params=22,713,216`.
  * Cosine against separately loaded transformers models: text 1.0, image 0.999999, AST 1.0.
  * A video of one still image against the image vector: 0.994.
  * The Whisper transcript is correct.
* E.2: minimum separation ratio 15.1 (text). Image 22.4, audio 18.3, video 65.8, code 154.6.
* E.4: 0 failures, 10 hits, 5 recomputes.
* E.n1: 9 forward passes. Changing one weight matrix changes the vector of every modality (cosine 0.64-0.95), and
  restoring it restores the vector.

**My own artifacts (CPU, MiniLM/CLIP/AST/Whisper), cosine similarity:**

| change | cosine | reading |
|---|---|---|
| text paraphrase (nuisance) | 0.831 | |
| text "approved three ... April" vs "thirty ... December" | 0.831 | numbers and dates are not distinguished from a paraphrase |
| text "safe for children" vs "**not** safe for children" | **0.965** | negation is *closer* than a paraphrase |
| review: excellent/love vs terrible/hate | 0.824 | a sentiment flip looks like a paraphrase |
| text: different topic | -0.013 | topic is captured well |
| 3,200-word article, last word changed | 0.9997 | local edits vanish in long documents (windowed mean) |
| image: red circle vs JPEG q60 copy | 0.956 | |
| image: red vs green circle / circle vs square | 0.881 / 0.851 | |
| image: "SALE 50% OFF" vs "SHOP CLOSED FOREVER" poster | 0.784 | |
| audio: same words, other voice / same voice, other words | 0.866 / 0.608 | good |
| audio speech posterior, flite "stock market" clip | 0.563 | just above the hard 0.5 gate (`audio.py:60`); below it the transcript is dropped |
| video: same 4 frames in reverse order / different pictures | 0.869 / 0.773 | |

So the encoders separate topic, event and scene well, which is what the E.2 suite measures. They separate polarity,
negation, quantities and quality poorly. Participants reading these vectors could not tell "safe" from "not safe" or a
rave from a pan. The E.2 suite only contains topic- and event-level changes (`encoders/validation.py`), so its 1.5
ratio is a weak criterion.

**Cache:**
* What works:
  * The key is `sha256:<ref>|<encoder id>@<revisions>.c<CODE_VERSION>|<preprocessing hash>` (`encoders/__init__.py:92-95`).
  * Content changes recompute; disk hits are verified; tampering is detected.
  * The world record (`embed.recorded`, internal visibility) makes replay exact. I observed 1 record for two calls.
* Defects:
  * `preprocessing` is a hand-written class dictionary (`text.py:49-51`). It is not derived from the parameters
    that are used: with `TextEncoder.window` changed from 254 to 32, the key stayed the same and the cache served a
    stale vector (cosine 0.83 to the freshly computed one).
  * `CODE_VERSION` (`base.py:13`) is a manual bump.
  * Library versions and device are not in the key.
  * The E.4 test builds the "new version" and "new preprocessing" keys itself (`test_gate_E.py:270-279`), so it
    never exercises a real change to the encoder.

**Video:** it encodes sampled frames, the audio track, the Whisper transcript and motion features. It does not use
subtitle or caption streams, which are "available transcripts". `aud` encodes once per item at publication; viewers
read the stored vector (consumption engine), so the encoder is not run per viewer.

**Products:** there is no structured product representation (4.3 asks for "software and products"). The code
encoder covers software only.

## 4. Do learned representations drive behaviour? (E.3, E.5; Part 4.3/4.4)

* `encoders/effects.py` has the paired permutation machinery, but `PROBES` is empty: `register_probe` is called
  nowhere. E.3 fails with p = 1.0, as recorded.
* The social consumption engine (`social/consumption.py`; `recsys/features.py:66-81`) scores affinity as
  cos(population interests, `aud_items.emb`).
  * Default `aud.encoder = "c2-random-indexing"`: a 32-d lexical random projection (`social/textrep.py`), not a
    learned representation.
  * The optional `"semantic"` path (`social/audience.py:470-479`) multiplies the MiniLM vector by a seeded Gaussian
    matrix into the same 32 dimensions. The interests, however, are built from lexical topic centroids
    (`social/scale.py:188-199`; the population module uses yet another token-vector space, `population/text.py`).
    The two spaces have no relation.
* Probe 6:
  * For interest vectors of 8 topics, the on-topic minus off-topic affinity is 0.425 with the lexical encoder.
  * Through the semantic path it is **-0.011 +/- 0.046** over 30 projections.
  * In raw MiniLM space the margin on synonym-only texts would be 0.396.
* Probe 7: two worlds identical except for `aud.encoder` (crowd 1,500, 20 creators, 3 days, Gate D mixtures):

| | lexical (default) | semantic (pretrained) |
|---|---|---|
| impressions / consumptions | 223,569 / 95,486 | 92,560 / 28,016 |
| topic match of opened items | 0.444 | 0.171 |
| topic match of not-opened items | 0.250 | **0.221** (opened items are *less* on topic) |
| Spearman(topic match, watch time) | 0.203 | **0.026** |
| cos(stored item vector, lexical vector) | 1.0 | -0.05 |

  With the pretrained encoders switched on, viewers' choices and watch time no longer follow what the content is
  about. Behaviour responds to content only through the lexical bag of words, and only when the viewer's interests
  and the item share lexicon words.
* E.5 (`encoders/audit.py`):
  * It is a regex over identifier names (`audit.py:32-35`). It currently reports 5 reads in participant scope:
    `domains/firms.py:290,295,711` (`inst_tasks.quality`, set by the labour market or a lead's acceptance in
    `commerce/institutions.py:1258`) and `social/reference.py:151` (`self.relevance(...)`, a lexical method).
    These are probably false positives, but the gate fails.
  * By construction the audit cannot see a creator-controlled signal under any other name (`score`, `boost`,
    `rank`, `weight`, tags or duration used as quality proxies).
  * The right test is dynamic: vary creator-controlled metadata with the bytes held fixed, and show that behaviour
    does not change.

## 5. Shared inference service, recording and semantic participants (13.4, 3.2 class 4, Gate H)

**Service** (`semantic/service.py`):
* 12 unit tests pass. They cover:
  * batched left-padded generation equal to unbatched HF `generate`;
  * bounded generation;
  * seeded sampling;
  * `choose` against direct log probabilities;
  * one weight copy;
  * a weights hash over the parameter bytes;
  * int8 dynamic quantization on CPU;
  * replay and regeneration without the model;
  * an operator-only recording tool.
* My probe (SmolLM2-135M, CPU, 1 thread):
  * A request for 500 tokens was capped at the service's 20.
  * 16/16 prefix hits (86 tokens reused per row).
  * A rerun was identical.
  * Two service objects share one weights object.
  * The world log holds the `cmd` plus a `semantic.inference` event with digests only; the prompt text is absent.
    The recorded outputs equal the unrecorded ones.
* Gate H (Qwen3-0.6B, bf16, A6000):
  * Peak 5,813 MiB, 237.5 tokens/s on GPU, 1.9 tokens/s on CPU with 1 thread.
  * Prefix-cache speedup 0.986 (none).
  * The prefix-cached outputs were identical in only 5 of 8 cases. In bf16 the prefix cache is not exact; outputs
    are recorded, so replay is unaffected, but a run is not reproducible with the cache toggled.
* Missing:
  * continuous batching across independent callers (a batch is only what one `act_batch` call collects);
  * any policy for "selective GPU execution";
  * per-capability model routing;
  * a GPU int8/4-bit path, and int8 output quality was never benchmarked.

**Recording** (`semantic/recording.py`, module `semantic_inference`, host `admin`, operator-only): sound. Requests
are staged in `world.runtime`, and outputs are recorded through `substrate.inference.recorded`. The prompts
themselves are not recoverable from the log (only their digests), which limits post-hoc auditing of decisions.

**Participants** (`semantic/participants.py`):
* Design:
  * The tool is chosen by Qwen3 thinking (384 tokens), then by scoring the candidate names.
  * Candidates are the BM25 top tools for the goal plus the tools the participant found itself.
  * Arguments are filled one slot at a time, among values that appear in the participant's own goal and
    observations.
  * The prompts carry only the participant's own goal, state and memory; the unit test checks this.
  * Observations contain no reward fields (I inspected `world.observe`: identity/bank/mail only). The
    `BatchedRunner` reward is 0 unless the caller supplies objectives, which is consistent with the
    observation-only default.
* Gaps:
  * A text `goal` is mandatory (`participants.py:204`). There is no operation without an assigned objective and no
    self-generated goals (the world's `goals.propose` tool is never prompted for).
  * Memory and history are Python lists on the policy object (`participants.py:212-216`). They are not persisted
    with world snapshots or forks, so there is no persistent identity (Part 3.1) and no long-horizon continuity.
  * No learning or adaptation of any kind.
  * Only 4.0 worldgen families can host them: no v5 family is registered in `worldgen.generator.FAMILIES`.

**Gate H society** (`semantic/scenarios.py:147-183`): eight one-line tasks assigned by the evaluator, in the 4.0
family `open_digital`. My reproduction gave exactly the evidence of the interrupted session:

| metric | value |
|---|---|
| H.1 participants acting | 8 (129 actions) |
| H.2 weight copies / sharing | 1 / 8 |
| H.3 distinct goals / histories | 8 / 8 |
| H.4 held-out tool choice | 0.6083 (73/120, CI 0.519-0.691; BM25 0.167, random 0.15) |
| H.5 comprehension | 0.900 (held-out genres 0.817) |
| H.6 consequential actions | 15 by 4 participants: bank.open_account 4, cal.create 6, mail.send 3, bank.transfer 1, cloud.rent 1 (only the last 2 serve a goal) |
| H.7 verified goals | 2/8: ben (2,500-cent payment) and hana (instance). ana/cara/dev/eli/fay/gus all false |
| rejected actions | 74 of 129 (57%) |
| H.8 | peak 5,812.6 MiB, 237.5 tok/s GPU, prefix speedup 0.986 |
| H.n1 | 596,049,920 params; SmolLM2-135M/360M tool choice 0.14/0.12, comprehension 0.38/0.23 |

The H.4 benchmark guarantees that the correct tool is among 10 clean options. In the live world the same model
failed most calls and reached no multi-step or collaborative goal. Most "consequential" actions were not goal-directed.

**Society re-run with a per-participant dump (`probe_society.py`):**

* The run gave the same verification as the gate run: ben and hana verified, the other six false.
* Regenerating the world from its log, with `SemanticService.run` patched to raise, reproduced the digest
  (`regenerated_digest_equal: true`). Replay never runs the model.

What each participant did:

* **ana** ("publish a public document 'Opening hours'"): never called `docs.create`. Called `forum.post` 3 times
  with board values such as `"- forum.post: {\"title\": \"Openin"` and body `"<host>.crucible"` (rejected by the
  schema), then `market.list` for an "api" listing.
* **ben** ("pay cara 2500 cents"): his first action was correct (`bank.transfer` 2500 to `p:cara`). He then did not
  wait. Over 23 more turns he sent invoices to cara addressed to `to_account: "p:cara"`, opened 4 new bank accounts
  and tried to pay invoices that did not exist.
* **cara** ("ask dev by mail when the meeting is"): never sent mail. 11 times `net.fetch` of `"-502"` and once
  `http://p:dev.crucible/path?query=team_meeting`. All are "ok" calls, so her 12/12 success rate is meaningless.
* **dev** (answer questions): created 6 calendar events with junk arguments (title "cal", duration 500000). Two
  `cal.create` calls failed ("start is in the past"), as did `wf.trigger {"tool": "acct_000032"}`.
* **eli** (create repo `textutils` + commit): never created the repository. `code.commit` failed 8 times on the
  argument shape (`files.files` list instead of a path map); `code.ls`/`code.run` failed on "code: textutils".
* **fay** (forum question): `forum.post` with board `"- \"cloud provider"`, rejected.
* **gus** (mail hana the provider names): mailed the provider orgs `o:cloud1`/`o:cloud2` with body `"mail.send"`
  and never mailed hana.
* **hana** (rent a 1-core instance): 9 failed `cloud.register_provider` attempts (and 2 failed `cloud.set_price`)
  before one successful `cloud.rent`.

Diagnosis:

* Tool choice is often plausible, but the argument values are frequently prompt fragments echoed back: list
  markers, tool names, placeholders.
* Participants have no notion of "done", so actions continue after the goal is met.
* The world tools accept nonsense with `ok` (net.fetch, mail.send with junk bodies, cal.create).

So the H.6 count rewards activity, not competence. These participants would teach a learner that the other agents
are noise generators.

## 6. SP-3.2b benchmark (`semantic/capbench`)

* Coverage:
  * Candidates: hashed n-gram head (2.1M, from scratch), frozen MiniLM with heads (22.7M), SmolLM2-135M and -360M,
    Qwen3-0.6B, Qwen3-1.7B.
  * Eight capabilities, with train/dev/test splits and held-out templates.
  * The scoring method is chosen on dev; results carry Wilson intervals, p-values against chance, VRAM and latency.
* What I reproduced: I did not rerun the full 40-minute, 10 GB benchmark. Its evaluation sets were regenerated
  bit-identically in my Gate H run (`sets_5dd8f4ce29d7eb22.json`, the same hash as the cached copy under
  `/workspace/data/crucible/semantic/capbench`). The tool-choice, comprehension and smaller-model numbers it shares
  with Gate H reproduced exactly.
* It is honest:
  * The limitations are reported (agent report D2).
  * Recommendations are given over all test items and over held-out templates.
* Weaknesses:
  * Every item is template-generated.
  * "Negotiating simple agreements" is a single accept/counter decision against a known limit
    (`tasks.py:618-690`), not a negotiation with another agent.
  * "Communicating coherently" is checked by copying facts plus a repetition and alphabetic-share heuristic
    (`tasks.py:968-980`), with no independent judge.
  * The "~1B" slot is filled by Qwen3-1.7B. No checkpoint licences are recorded, although 3.2 asks to verify
    "licenses and tool-use capabilities".
* Not followed through:
  * The benchmark's held-out result says 0.6B is not effective for programming, negotiation or strategy.
  * DECISION_LOG D-10 still uses 0.6B everywhere and only "allows" an escalation that is not implemented.
  * That contradicts "use the smallest model or combination of models that meets each capability requirement".

## 7. Transfer to the real internet

Topic-level semantics transfer, because the encoders are real pretrained models. What would not transfer:

1. An audience whose consumption follows bag-of-words overlap with a 16-topic lexicon. Creators learn keyword
   matching, not quality. With the pretrained encoders the audience follows nothing semantic at all, because of the
   projection mismatch.
2. Encoders that are blind to negation and sentiment. If they became the audience's representation, misleading and
   accurate content would look the same.
3. `embed.compare` exposes the world's exact encoder to agents. Once audiences read that space, creators could
   optimise content against the platform's representation directly, which real platforms do not allow.
4. Semantic participants given one-line tasks by an evaluator, with no persistence, no needs and no self-generated
   goals. A learner meets them as scripted task-doers, not as an autonomous population.

## 8. Subsystem verdicts

| subsystem | verdict | effort |
|---|---|---|
| Pretrained content encoders (text/image/audio/video/code) | EXTEND | M |
| Embedding cache and world `embed` record | REPAIR | S |
| Media creator tools (`semantic/media.py`, module `media`) | REPAIR | S |
| Encoder to participant-behaviour integration (`aud` semantic path, E.3 probes) | REDESIGN | L |
| Creator-assigned-quality audit (E.5) | REPAIR | M |
| Shared small-LM inference service | REPAIR | M |
| Recorded semantic inference module | RETAIN | S |
| Semantic participants (policy, retrieval, argument filling) | REDESIGN | L |
| Gate H society scenario and verifiers | REBUILD | M |
| Population class-4 binding (semantic rows of the standard profile) | IMPLEMENT | M |
| SP-3.2b capability benchmark | EXTEND | M |

Notes on the verdicts:

* **Media tools:** they work (13 + 3 tests). They are not in the social internet module list: the social `studio`
  module duplicates them through `social/render.py`.
* **Inference service:** sound. Prefix cache: no speedup and not exact in bf16. No routing or selective GPU policy.
  It is used only by the participants, which are not integrated anywhere.
* **Participants:** these are ephemeral Policy objects outside the world, they need an assigned goal, and they have
  no persistence, needs or self-goals.
* **Gate H society:** the 4.0 family and one-line assigned tasks do not make a "standard world".

The JSON file holds per-requirement verdicts, criteria weaknesses and missing requirements.

## Skeptic review

Reviewer: skeptic for area D_semantic. I tried to refute every RETAIN, every REPAIR/EXTEND that might deserve a harsher
verdict, and every harsh verdict. The worktree HEAD is now `dd745b9`. That commit changes only docs and the acceptance
manifest; `git diff a881d18 HEAD` is empty for `src/crucible/{semantic,social,population,experience}` and `tests/`.
My probe scripts, logs and JSON outputs are in
`/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/D_semantic_skeptic/`.

Several probes used no model at all. `fake_service.py` is a `SemanticService` subclass that returns random choices
and replays recorded outputs. Probes that touch the `aud` module ran in the social worlds that the gates build
(`social/experiments.py:40`). Every model run went through the slot scheduler, with `CRUCIBLE_CACHE`, `CRUCIBLE_DATA`
and `TMPDIR` pointed at the scratch directory, so nothing was written to the shared caches.

### S1. What I ran

| # | Probe | Result |
|---|---|---|
| 1 | `probe_e1_integration.py`: default social world (300 people, 8 creators, half a day), CPU | **0 pretrained forward passes** over 306 consumptions; item vectors are 32-d (`c2-random-indexing`). The default audience refuses a real PNG, WAV and MP4: `InvalidAction: no encoder for image/png in this world` (`social/audience.py:482-484`). The gate social worlds only have the `articles` and `posts` platforms. |
| 2 | same probe, `aud.encoder="semantic"`, 6 texts in 3 topic pairs | The stored 32-d vectors **keep** MiniLM's geometry: Pearson 0.907 between the pairwise cosines; same topic 0.65 -> 0.63, other topic 0.01 -> -0.05 |
| 3 | `probe_experience_semantic.py`: `KernelExperienceEnv` on a social world (fresh cache) | 8 MiniLM 384-d vectors in the learner's observation, 8 forward passes, `recorded: 0`. On a warm cache: 0 passes (reuse across worlds works). Env default `reward_mode = "legacy_shaped_reward"` (`experience/kernel.py:59`) |
| 4 | `probe_stubcheck.py` | `stub_check` (`encoders/audit.py:186-207`) flags **0** SHA-256-seeded random vectors in all 5 suites. The lexical random-indexing vectors also pass, with an **E.2 text separation ratio of 4.95** (criterion 1.5) and no model involved |
| 5 | `probe_semantic_regen.py`: semantic-`aud` world on CPU (16 items, 694 events, 0 `embed.recorded`/`ext` records), then `World.regenerate` from the log with an empty cache | regenerated on CPU: digest **equal**; regenerated on GPU: digest **differs** |
| 6 | `probe_h5_shortcuts.py`: surface rules on the cached Gate H test sets (no model) | "Pick the option mentioned last in the context" scores **0.583** (70/120) on comprehension (chance 0.25). Tool-choice rules score at most 0.175 |
| 7 | `probe_h5_peritem.py`: Qwen3-0.6B on the Gate H comprehension set, think mode (GPU, 101 s) | 0.900 reproduced. **0.986** on the 70 items the shortcut gets right; **0.78 [0.648, 0.872]** on the 50 items it gets wrong |
| 8 | `probe_gateh_nomodel.py` (FakeService, `open_digital`, seed 0) | See S2-S4. Untouched world verifies 0/8 goals. Turn-0 candidates miss the goal tool for ana, cara and eli. One transfer gives `verified_productive_results = 1`. hana's 64-core rental is verified. Fork and snapshot lose the participants' state. Recorded outputs re-derive 22/22 actions. An unrecorded run also regenerates with an equal digest. `admin.semantic_infer` is hidden and forbidden |
| 9 | `probe_no_objective.py`: the Gate H society (Qwen3-0.6B) with `goal = ""`, 6 rounds (GPU, 94 s) | 48 turns: 12 waits and 22 actions, every action read-only (`world.find_tools`, `world.validate`, `bank.balance`, ...). **0 consequential actions**, 0 `goals.*` calls |
| 10 | `probe_h2_h8_cpu.py`: SmolLM2-135M on CPU | `loaded_copies` is 1 for two services with the same key, 2 after an int8 service and 3 after a bf16 service. On CPU, `usage()["peak_vram_mb"]` is None, there is no host-memory field, and `peak_vram_mb_measured` is False |
| 11 | `python -m crucible.semantic.capbench --models Qwen/Qwen3-0.6B --caps programming,strategic_decisions --no-encoders` (sets rebuilt from the current sources in 20 s; GPU, 124 s) | programming generation 14/28 = 0.50 [0.326, 0.674], MC 1.0; strategy 0.4917 [0.404, 0.58], held-out 0.35. **Identical to D2.md** |
| 12 | `encoders.audit.quality_fields()` on HEAD | 5 violations: `domains/firms.py:290,295,711(x2)` and `social/reference.py:151` (reproduces the E.5 failure) |

### S2. Overturned verdicts

| target | from | to | why (probe) |
|---|---|---|---|
| E.1 | RETAIN | REPAIR | Gate E feeds the encoders hand-made suites. In the world, the default audience runs 0 forward passes and refuses image, audio and video outright (probe 1). The only in-world caller is the learner-observation path, which encodes text it does not record (probe 3). Encoders that no participant reads do not "process the world's artifacts". |
| E.n1 | RETAIN | REPAIR | The vectors participants actually consume involve no model: 32-d lexical hashes (probe 1). The stub detector passes SHA-256 random vectors and the lexical vectors (probe 4), so `stub_vectors_detected == 0` cannot detect a stub. The weight-perturbation check that does discriminate exists only in the test, not in the manifest criteria. |
| H.3 | RETAIN | REPAIR | Goals, memory and history live only on the Python policy (`participants.py:212-216`). A world fork after round 2 holds 1-2 commands per participant, while the policies continue with 3-4 (probe 8, part 5). Goal and memory text are absent from the 137 kB snapshot. Separate histories cannot be persisted, restored or forked consistently, and the `history_digest` metric (`participants.py:249`) is distinct by construction. |
| H.8 | RETAIN | EXTEND | The measurement is `torch.cuda.max_memory_allocated` for the whole process, on GPU only. On CPU, a supported mode under 13.4, `peak_vram_mb_measured` is False and no host memory is reported (probe 10). Participant state memory, per-decision latency and the standard-world or per-profile scaling are not measured. The auditor's own `work_needed` was not empty. |
| H.4 | EXTEND | REDESIGN | The benchmark always offers the correct tool. In the world, the BM25 shortlist over the goal text (`participants.py:306-325`) leaves out the needed tool on turn 0 for 3/8 goals: ana `docs.create`, cara `mail.send`, eli `code.create_repo` (probe 8, part 2). In the 24-round run those three never called those tools, even after `world.find_tools`. The failure comes from the architecture, which picks from an 8-item lexical shortlist. More model scale would not fix it. |
| H.6 | REPAIR | REDESIGN | The count can be satisfied by junk: 8 `bank.open_account` calls by one participant add 4 "consequential actions" (probe 8, part 4). With no assigned objective, which is the default under amendment (a), the society produces **0** consequential actions (probe 9). Meaningful consequential action needs participants with their own needs, which is the participant redesign. |
| H.7 | REPAIR | REDESIGN | One `bank.transfer` meets the criterion: `verified_productive_results = 1` with collaborative 0 and program 0 (probe 8, part 3). The metric counts every verified goal (`test_gate_H.py:176`). `check_instance` (`scenarios.py:141`) verifies a 64-core rental for a 1-core goal. Without an assigned objective nothing is produced (probe 9). A collaborative or productive result is out of reach without the participant redesign and a rebuilt scenario. |

### S3. Verdicts that survived

* **H.2 RETAIN**: the counter discriminates. It counts one copy per (model, revision, device, dtype, quantization), going 1 -> 2 -> 3 (probe 10). Other services with the same key get the same `_Weights` object.
* **H.5 RETAIN**, with weaker criteria than reported:
  * A no-understanding rule reaches 0.583, close to the 0.6 bar, with p far below 0.01 (probe 6).
  * Even so, the model scores 0.78 on the items the rule gets wrong (probe 7), so its comprehension is real.
  * The criterion should be scored on a shortcut-controlled subset.
* **Recorded semantic inference module RETAIN**, kept on new evidence:
  * The auditor's evidence ("regenerate without the model gives the same digest") does not discriminate: a run with `record=False` also regenerates with an equal digest (probe 8, part 6b).
  * The discriminating test passes. Feeding only the recorded outputs, taken from the `ext` events of the log, to fresh participants re-derives all 22 actions exactly (6a).
  * Participants cannot list or call `admin.semantic_infer` (`forbidden`) and see 0 inference or `ext` events (part 7).
  * Its only caller is still the unintegrated participant code.
* **SP-3.2b EXTEND**: the Qwen3-0.6B programming (0.50) and strategy (0.4917 / held-out 0.35) numbers reproduce bit-for-bit from the merged tree (probe 11). The 1.7B numbers that support routing were not re-run. No report file is committed. The only copy I found is in another session's scratchpad, written by an older `run.py` that has no `recommendation_held_out`.
* **E.2 EXTEND**: kept, but the criterion is weaker than the auditor said. A lexical bag of words with no model passes the text suite at 4.95 (probe 4).
* **E.3 REDESIGN**: kept, with one correction to the mechanism. The seeded projection preserves MiniLM's geometry (Pearson 0.907, probe 2). The misalignment is on the **interest** side: interests are mixtures of lexical topic words (`social/textrep.py`, `population/text.py`). The fix is still cross-module: one shared learned space for interests, needs, search and the calibrated open model.
* **Harsh verdicts are justified:**
  * **H.1 and participants REDESIGN.** With no assigned goal they only do read-only actions (probe 9), and retrieval hides the needed tools (probe 8).
  * **Gate H society REBUILD.**
  * **Class-4 IMPLEMENT.** Confirmed at `population/core.py:323` and `:333`; nothing else in `population/` or `scale/` binds class 4.
* **E.4 / cache REPAIR**: kept, with a new defect. The `aud` semantic path calls `semantic.encode(world, ref, modality)` with no ctx (`social/audience.py:474`), so the world records no embedding. The cache key carries no device. As a result, regenerating a world log on another device changes the world digest (probe 5). I would raise the effort from S to M.
* **E.5 / audit REPAIR**: kept. None of the 5 hits is creator-assigned. The `inst_tasks.quality` value comes from the labour market's skill-match formula (`commerce/labor.py:450-453`), and a lead's acceptance passes `quality=None` (`commerce/institutions.py:1580`).

### S4. Additional findings

1. **Non-deterministic regeneration** for `aud.encoder="semantic"`: a CPU-built world regenerates with a different digest on GPU (probe 5). The embeddings should be recorded (`ctx` into `encode_many` -> `embed.recorded`), and the device or numeric class belongs in the cache key.
2. **Learners and participants see different representations**:
   * The experience env gives learners 384-d MiniLM vectors (`experience/kernel.py:362-377`), but the audience responds to 32-d lexical vectors.
   * A learner's content features therefore do not describe the mechanism that produces audience behaviour.
   * The same env defaults to `legacy_shaped_reward` (`experience/kernel.py:59`), which contradicts amendment (a). That is workstream L's area, but it is the path by which D's encoders reach learners.
3. **Retrieval scaffold, not the model, explains 3 of 6 failed Gate H goals** (probe 8, part 2, against the auditor's dump). The pretrained MiniLM is available but tool retrieval does not use it.
4. **The stub detector does not discriminate** (probe 4). Any criterion that relies on `stub_vectors_detected` (E.1, E.n1) should add a learned-structure control: a paraphrase/synonym margin above a lexical baseline, plus weight dependence.
5. **Code reading, not probed (workstream I):** `domains/firms.py:290` computes `q = float(t["quality"] or 0.0)`. A task accepted by an agent lead has `quality=None` (`commerce/institutions.py:1580`), so it would earn 0 external sale revenue.

### S5. Net requirement verdicts after skeptic review

E.1=REPAIR, E.2=EXTEND, E.3=REDESIGN, E.4=REPAIR, E.5=REPAIR, E.n1=REPAIR, H.1=REDESIGN, H.2=RETAIN, H.3=REPAIR,
H.4=REDESIGN, H.5=RETAIN, H.6=REDESIGN, H.7=REDESIGN, H.8=EXTEND, H.n1=EXTEND, SP-3.2b=EXTEND. Of the 16 requirements,
2 are RETAIN.
