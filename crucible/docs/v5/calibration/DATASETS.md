# Calibration datasets: recipes, checksums, licences, provenance

The machine-readable source of truth is `src/crucible/calibration/datasets.py` (`datasets()`); this page explains
it. No dataset file is committed. `acquire(name)` downloads into `$CRUCIBLE_DATA/<name>` (default
`/workspace/data/crucible/<name>`), refuses any file whose SHA-256 differs from the value below, extracts archives
with Python's safe `data` tar filter, checks the SHA-256 of every extracted file the models read, and validates the
schema (column names and row counts).

```bash
cd crucible
PYTHONPATH=src python -c "from crucible.calibration import acquire; print(acquire('kuairand_pure'), acquire('mind_small'))"
```

Disk: KuaiRand-Pure 240 MB (47 MB archive + 195 MB extracted), MIND-small 210 MB, Criteo 1.0 GB, GH Archive window
290 MB (extracted tables), Online Retail II 50 MB. The calibration budget is 8 GB in total.

## kuairand_pure: KuaiRand-Pure

| | |
|---|---|
| file | `download/KuaiRand-Pure.tar.gz`, 47,432,272 bytes |
| URL | https://zenodo.org/api/records/10439422/files/KuaiRand-Pure.tar.gz/content (Zenodo record 10439422, DOI 10.5281/zenodo.10439422; landing page https://kuairand.com/) |
| SHA-256 | `c814bf6f3624c0cfae83c57de3df26b2ed206e5c57bab4c4dcbfabbabe20cbf0` (MD5 `0820331067a3784d9691136f772b35a7` as published by Zenodo, checked) |
| extracted files | the three logs, user and video features, `LICENSE`; each SHA-256 is in `datasets.py` |
| licence | CC BY-SA 4.0; full text in the archive at `raw/KuaiRand-Pure/LICENSE` (the Zenodo metadata says cc-by-4.0) |
| citation | Gao et al., KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos, CIKM 2022, doi:10.1145/3511808.3557624 |
| population | 27,285 anonymised Kuaishou users chosen by the authors; only interactions with the 7,583 videos of the random-exposure pool |
| platform | Kuaishou short-video app, mostly the single-column featured feed (tab 1) |
| period | standard logs 2022-04-08..05-08; random-exposure logs 2022-04-22..05-08 |
| identification | standard logs: observational (the production recommender chose the exposures); random logs: videos inserted into feeds uniformly at random from the pool (randomised exposure) |

## mind_small: MIND-small

| | |
|---|---|
| files | `download/{train,dev}/{behaviors,news}.tsv` |
| URLs | https://huggingface.co/datasets/huyva/MIND-small/resolve/d771a25043d6352e49e76ca1fe3075c9ac6862bb/{train,dev}/{behaviors,news}.tsv |
| SHA-256 | train/behaviors `a424547c...4658`, train/news `cd49d7da...09d6`, dev/behaviors `b6c460e3...43ef`, dev/news `e5d14466...b822` (full values in `datasets.py`; they equal the Hugging Face LFS object ids) |
| licence | Microsoft Research License Terms, non-commercial research use; text at https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf |
| citation | Wu et al., MIND: A Large-scale Dataset for News Recommendation, ACL 2020, doi:10.18653/v1/2020.acl-main.331 |
| population | 50,000 Microsoft News users sampled by the authors (at least 5 clicks), anonymised IDs |
| platform | Microsoft News (msn.com), English, US market |
| period | impressions 2019-11-09..14 (train) and 2019-11-15 (dev); click histories reach back to 2019-10-12 |
| identification | observational (the production recommender chose the impressions) |

Why a mirror: the official page (msnews.github.io) now links to the gated Hugging Face repository `yjw1029/MIND`,
which needs an account that has accepted the licence, and the original host `mind201910small.blob.core.windows.net`
answers HTTP 409 "Public access is not permitted on this storage account" (checked 2026-11-25). No credential is
configured on this machine, so the files come from the ungated mirror `huyva/MIND-small` at a pinned revision. The
mirror's files are the original MINDsmall_train / MINDsmall_dev TSVs: row counts equal the official statistics
(156,965 and 73,152 impressions; 51,282 and 42,416 news; 50,000 users per split). Use of the data follows the
Microsoft Research licence (non-commercial research); nothing from the dataset is redistributed.

**Licence status (2026-11-26): unresolved, user decision pending (docs/v5/BLOCKERS.md B-1).** Re-checked: the official
gated repository answers 401 without an account that accepted the licence, the Azure host answers 409. Workstream E
did not accept the licence on the user's behalf. The fitted results are kept; every derived artifact is marked
non-commercial research in `models/LICENSE_NOTICE.md`.

## criteo_uplift_v2.1 and gharchive_2024-10-07_96h

Both are described by their committed manifests in `datasets/` (licence, provenance, per-file SHA-256).

GH Archive privacy (MASTER_SPEC 5.1). Since extractor version 2 (2026-11-26) the extracted hour tables hold **no GitHub
id**: actor and repository ids are replaced at extraction by keyed 63-bit pseudonyms (blake2b under a 32-byte secret
salt in `$CRUCIBLE_DATA/gharchive_2024-10-07_96h/pseudonym_key`, mode 0600, never committed or logged). The 96
extractor-1 tables were converted in place after each was checked against its recorded digest, the derived table
`tables_v1.npz` (dense ranks of raw ids) was deleted and rebuilt as `tables_v2.npz` from the pseudonymised events, and
the software reproducers return the same held-out metrics (abs diff 0.0 and 1e-16). Verification uses digests in
which identity columns are replaced by their first-appearance order, so it does not depend on the salt. Residual
risk: exact timestamps and event types could be matched against the public GH Archive by someone who downloads it,
so the extracted tables stay local and only aggregate model parameters are committed.

## online_retail_ii: UCI Online Retail II

| | |
|---|---|
| source | official UCI archive https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip (45,622,418 bytes, SHA-256 `572e3627...`) |
| extracted | `extracted/online_retail_ii_v1.npz` (1,067,371 invoice lines, 4.5 MB) and `products.json` (5,304 product codes with descriptions, 43 countries); parsed from the xlsx with the standard library |
| licence | CC BY 4.0 (UCI dataset page); cite Chen, D. (2012), doi:10.24432/C5CG6D |
| population | 5,942 customers of one UK online gift retailer (many wholesalers), 43 countries; 243,007 lines without a customer id |
| platform | the retailer's own non-store online shop |
| period | invoices 2009-12-01 to 2011-12-09 |
| identification | observational: prices set by the retailer, quantity discounts make unit price depend on quantity |
| privacy | customer ids replaced by keyed pseudonyms (salt outside the repository) |

## Not used

* KuaiRand's `video_features_statistic_pure.csv` is not used by any model: its aggregates have no timestamps and
  include the test period (time leakage).
* KuaiRand-1K / 27K (1.1 GB / 9.9 GB) were not downloaded; Pure contains the random-exposure logs the validation
  needs.
