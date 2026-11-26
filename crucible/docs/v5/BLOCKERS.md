# Crucible 5.0 blockers and risks

A blocker is something the project cannot resolve itself. A requirement marked BLOCKED in the manifest must cite an
entry here by id: an open blocker is a `### BLK-<n> <title>` heading under "## Open blockers" (the checker reads
only that section; a resolved blocker moves to "## Resolved blockers"). The project status is INCOMPLETE_BLOCKED only
when every unverified requirement cites an open blocker. Risks are listed so they are watched; they are not blockers.

## Open blockers

No open blockers.

## Risks being managed

* **R-1 Disk.** 26 GB free on a 300 GB overlay that is 92% full (another worktree holds 165 GB of checkpoints). Datasets,
  encoder weights, worktrees and long-run logs must fit. Budgets in INTERFACE_CONTRACTS section 0.
* **R-2 Persistence.** `/workspace` is not a persistent volume. Pushing to GitHub needs a credential the user provides
  for each push; until then the only durable copy of new work is local. Local bundles are written to `/workspace/backups`.
* **R-3 Shared compute.** One 11.5-CPU quota and one GPU shared with other tenants; schedule through slots.
* **R-4 Dataset access.** KuaiRand (Zenodo), MIND (Azure blob) and Criteo (go.criteo.net) host the calibration data;
  first probes returned 200 for Zenodo and Hugging Face, and 400/404 for the bare MIND and Criteo hostnames (file URLs
  still to be tried). If a dataset is unreachable, an alternative legitimate dataset is used and recorded.
* **R-5 Scale of the assignment.** The specification is far larger than one integration cycle; the manifest and the
  checker keep the status honest (IN_PROGRESS) until every requirement is verified.

## Resolved blockers

### B-1 MIND-small licence acceptance (RESOLVED 2026-11-26 by the project owner; recorded by workstream E)

`mind_small` (used by `news_click@1` and, from the E4 plan, `news_click@2`) was downloaded on 2026-11-25 from the
ungated Hugging Face mirror `huyva/MIND-small` (pinned revision `d771a250`). That route skips the licence acceptance the
official distribution requires: msnews.github.io links to the gated repository `yjw1029/MIND`, whose files answer
HTTP 401 without an account that has clicked through the Microsoft Research License Terms (checked again
2026-11-26: `gated: "auto"`, MINDsmall_train.zip -> 401), and the original Azure host
`mind201910small.blob.core.windows.net` answers HTTP 409 "public access is not permitted". The terms are
research-only and non-commercial. Matching row counts do not prove the mirror's files are byte-identical to the
official ones.

Workstream E has **not** accepted the licence on the user's behalf and did not use any credential. The fitted
results are kept (they are research use, nothing from the data is redistributed; checkpoints hold only model
parameters), and every artifact derived from MIND is marked non-commercial research (`docs/v5/calibration/models/
LICENSE_NOTICE.md`).

Decision needed from the user, one of:

1. accept the Microsoft Research License Terms with their own Hugging Face account and provide a token for one
   download of `yjw1029/MIND` (MINDsmall_train.zip, MINDsmall_dev.zip); E then verifies the mirror's files against
   the official ones (or re-fits from the official files) and records the acceptance (who, when, licence version)
   in the dataset entry; or
2. confirm in writing that the mirror copy may be used for this research project under the MSR terms; or
3. withdraw MIND: delete `mind_small`, drop `news_click@1`/`@2` from the registry and rely on KuaiRand for content
   selection (Gate F still has content consumption through `content_consumption@1`).

Until then F.7 counts MIND's checksums and provenance, but its acquisition route is recorded as unauthorised in
`datasets()["mind_small"]["license"]["acceptance"]`.

Resolution (2026-11-26, project owner, recorded by the orchestrator as DECISION_LOG D-28): option 2. The owner
instructed: "disregard the MIND license. use MIND dataset now for purely experimental use and I will currently
get the license for it." MIND-small (the pinned mirror copy) is used for purely experimental, non-commercial
research; the owner is obtaining the official licence. Derived artifacts stay marked non-commercial research only
and nothing from the data is redistributed. When the official files are available, verify the mirror against them
(or re-fit) and record the licence acceptance (who, when, version) in `datasets()["mind_small"]["license"]`.
