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

Disk: KuaiRand-Pure 240 MB (47 MB archive + 195 MB extracted), MIND-small 210 MB. E1's budget is 4 GB.

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

## Not used

* KuaiRand's `video_features_statistic_pure.csv` is not used by any model: its aggregates have no timestamps and
  include the test period (time leakage).
* KuaiRand-1K / 27K (1.1 GB / 9.9 GB) were not downloaded; Pure contains the random-exposure logs the validation
  needs.
