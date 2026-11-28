# CAM's live planning phase, 2026-11-28 — CAM's own records (exported with `cam-live plan-report`, unedited)

Objective: `tasks/discord_100_planning.yaml` (the operator's objective verbatim), mode live, the operator's Discord
server "CAM's lab"; authorization signed on the operator's explicit approval (public web read; writes to the
operator's own server pre-authorized; everything else queued; hard constraints unchanged). Full-size agent
(Qwen3-4B + 50m core, open-ended generation on). Nothing in this directory was written or edited by Claude Code except
this README; the reports reproduce CAM's records verbatim (operator answers are marked as such).

| | attempt 1 | attempt 2 |
|---|---|---|
| run dir | runs/cam_live/live/discord-100-planning-20261128T083454Z | runs/cam_live/live/discord-100-planning-20261128T104208Z |
| code | worktree at d1d06dc | snapshot of a0f3411 (ST hotfix: only authorized tools offered, no wildcard values, no truncated proposals) |
| decisions | 36 (stopped; see ledger) | 120 (limit max_decisions exhausted) |
| notes written by CAM | 0 | 0 |
| requests to the operator | 2 | 14 |
| executed / refused actions | — / 18 of 35 refused (attempt 1 defects) | 89 / 31 |
| most frequent action | discord.modify_guild | discord.modify_guild (53 of 120; 22+ refused by the run's write-rate limit) |
| web pages visited | none | none (browser.open_tab / read / scroll on blank tabs only; no navigation to an address) |
| learning updates | — | fast 120, frequent 68, slow adapter 4, meta-updates applied 0 (bootstrap mix 1.0) |

What this shows (measured, not interpreted for CAM): on the pre-amendment architecture CAM did not research the
web, wrote no notes, hypotheses or plan, repeated one reversible server setting change, and asked the operator for
information it could read itself, for permissions it already had, and once for the niche (declined per the
objective: the operator provides resources, not strategic direction). The architectural causes are being
investigated under the 2026-11-28 core-intelligence amendment (docs/ST.md: information-only actions had no pathway
to be valued; docs/RW.md: the learned reward never took effect in one objective).

## Attempt 3 (amended code 6cba496, standing profile routine-reversible-1)

Run dir runs/cam_live/live/discord-100-planning-20261128T144554Z, code snapshot runs/cam_live/live/code_6cba496 (all
seven amendment agents integrated; defaults: imagination on with learned weight, evidence-based R_psi promotion,
segment credit, persistent state off, information value as bootstrap input, prompt window 8192; ST's OWN RECORDS /
expected / proposal learning off by default). Signed with the standing profile (routine writes to the operator's
server and package installs from pypi without approval; irreversible / new permissions / messages to people / money
keep approval).

| | attempt 3 |
|---|---|
| decisions | 120 (limit) |
| executed / refused | 101 / 19 |
| notes written by CAM | 0 |
| requests to the operator | 7 |
| learning | fast 119, frequent 68, slow adapter 4, meta-updates applied 0 (mix 1.0) |
| consequential actions | none (12 x discord.modify_guild ok, 1 discord.create_channel; no money, no messages to people) |
| web navigation | none to an address (open_tab / read / scroll / inspect / screenshot on blank tabs) |

Same qualitative outcome as attempt 2. The independent verifiers' explanations: G3-act measured that the full-size
predictor barely uses its context (removing a decision's items moves later beliefs by a median 3.3e-4 relative), so
information carries no measurable value (docs/verification/G3_act.md); ST: the frozen generator never proposes notes.
