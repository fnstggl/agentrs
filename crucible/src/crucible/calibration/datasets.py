"""Calibration datasets: download recipes, checksums, licences, schemas and provenance (INTERFACE_CONTRACTS section 5).

Datasets are never committed. `acquire(name)` downloads the files named here into `$CRUCIBLE_DATA/<name>`
(default `/workspace/data/crucible/<name>`), verifies each file's SHA-256 before it is kept, extracts archives with the
safe `data` tar filter, verifies the SHA-256 of every extracted file it relies on and validates the schema (columns and
row counts). World code never imports this module; it loads fitted checkpoints through `crucible.calibration.load`.

Entries are plain dicts so other calibration workstreams can register their own datasets with `register(name, entry)`
from their own modules (E2 does this for its datasets) without editing this file.

Entry fields:

* ``title``, ``description``
* ``files``: list of ``{path, url, sha256, size, md5?, extract?}``; ``path`` is relative to the dataset directory
* ``extracted``: ``{relative path: sha256}`` for files produced by extraction that models depend on
* ``schema``: ``{relative path: {format, columns, rows, header}}``
* ``license``: ``{name, url, text_location, terms}``
* ``provenance``: ``{publisher, source, landing_page, doi?, citation, population, platform, period, collection,
  mirror?, retrieved}``
* ``synthetic``: False for every empirical dataset (gate F counts only non-synthetic datasets)
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile

DEFAULT_ROOT = "/workspace/data/crucible"


def data_root() -> str:
    return os.environ.get("CRUCIBLE_DATA") or DEFAULT_ROOT


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_KR = "KuaiRand-Pure/data/"
_KR_LOG_COLS = ["user_id", "video_id", "date", "hourmin", "time_ms", "is_click", "is_like", "is_follow", "is_comment",
                "is_forward", "is_hate", "long_view", "play_time_ms", "duration_ms", "profile_stay_time",
                "comment_stay_time", "is_profile_enter", "is_rand", "tab"]

_DATASETS: dict[str, dict] = {
    "kuairand_pure": {
        "title": "KuaiRand-Pure",
        "description": ("Interaction logs of 27,285 users of the Kuaishou short-video app with 7,583 videos, "
                        "2022-04-08 to 2022-05-08: standard logs (videos exposed by the production recommender) and "
                        "logs of videos inserted into the featured feed uniformly at random (is_rand = 1)."),
        "files": [{
            "path": "download/KuaiRand-Pure.tar.gz",
            "url": "https://zenodo.org/api/records/10439422/files/KuaiRand-Pure.tar.gz/content",
            "sha256": "c814bf6f3624c0cfae83c57de3df26b2ed206e5c57bab4c4dcbfabbabe20cbf0",
            "md5": "0820331067a3784d9691136f772b35a7",          # as published in the Zenodo record
            "size": 47432272,
            "extract": {"format": "tar.gz", "into": "raw"},
        }],
        "extracted": {
            "raw/KuaiRand-Pure/LICENSE": "187442db4df3afd21f2f0525739fd4beac28a62daaba3ee8d3533f60e7c33ec7",
            "raw/" + _KR + "log_standard_4_08_to_4_21_pure.csv":
                "5bb6eb0b3d9f47e5436cb5dc82ee1899b845ebf9750a5560b801e929e18bd41c",
            "raw/" + _KR + "log_standard_4_22_to_5_08_pure.csv":
                "429e3b948828942e572f2c3a5be5a25799ffe75591d22d18cf417b9b534d31fd",
            "raw/" + _KR + "log_random_4_22_to_5_08_pure.csv":
                "60b80994da969cd53da4d50c37ba3dafd6fb185df804c92c8410df34845a9d2c",
            "raw/" + _KR + "user_features_pure.csv":
                "dc729a656301b4c6d07f713fe41d05ec9bfaab670b90e531c70037caf033c011",
            "raw/" + _KR + "video_features_basic_pure.csv":
                "a6f7ee02684c5777422306cdc416e170302288aa89aca9dfea995edbd625bcc2",
            "raw/" + _KR + "video_features_statistic_pure.csv":
                "d5c9e237ef2c6c1fc0e7f27e952f215d6626ecd934b01a6c53ecfcc72540f6b6",
        },
        "schema": {
            "raw/" + _KR + "log_standard_4_08_to_4_21_pure.csv": {"format": "csv", "header": True,
                                                                  "columns": _KR_LOG_COLS, "rows": 1141112},
            "raw/" + _KR + "log_standard_4_22_to_5_08_pure.csv": {"format": "csv", "header": True,
                                                                  "columns": _KR_LOG_COLS, "rows": 295497},
            "raw/" + _KR + "log_random_4_22_to_5_08_pure.csv": {"format": "csv", "header": True,
                                                                "columns": _KR_LOG_COLS, "rows": 1186059},
            "raw/" + _KR + "video_features_basic_pure.csv": {
                "format": "csv", "header": True, "rows": 7583,
                "columns": ["video_id", "author_id", "video_type", "upload_dt", "upload_type", "visible_status",
                            "video_duration", "server_width", "server_height", "music_id", "music_type", "tag"]},
            "raw/" + _KR + "user_features_pure.csv": {"format": "csv", "header": True, "rows": 27285,
                                                     "columns_prefix": ["user_id", "user_active_degree",
                                                                        "is_lowactive_period", "is_live_streamer",
                                                                        "is_video_author", "follow_user_num"]},
        },
        "license": {
            "name": "CC BY-SA 4.0",
            "url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "text_location": "raw/KuaiRand-Pure/LICENSE (full licence text shipped inside the archive)",
            "terms": ("Attribution-ShareAlike 4.0 International per the LICENSE file in the archive; the Zenodo "
                      "record metadata lists cc-by-4.0. Crucible redistributes none of the data; fitted parameters "
                      "are derived statistics and the dataset is cited in every model card."),
        },
        "provenance": {
            "publisher": "Kuaishou Technology and University of Science and Technology of China (Gao et al.)",
            "source": "Zenodo record 10439422",
            "landing_page": "https://kuairand.com/",
            "doi": "10.5281/zenodo.10439422",
            "citation": ("Chongming Gao et al. KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly "
                         "Exposed Videos. CIKM 2022. doi:10.1145/3511808.3557624"),
            "population": ("27,285 anonymised users of the Kuaishou app (China) selected by the dataset authors; the "
                           "'Pure' release keeps only interactions with the 7,583 videos of the random-exposure pool"),
            "platform": "Kuaishou short-video app, mainly the single-column featured feed (tab 1)",
            "period": "2022-04-08 to 2022-05-08 (standard logs 04-08..05-08, random-exposure logs 04-22..05-08)",
            "collection": ("Production logs. In the random-exposure logs a video drawn uniformly from the pool was "
                           "inserted into the user's feed, so exposure is randomised with respect to the user's "
                           "preferences; the standard logs are exposures chosen by the production recommender."),
            "identification": "observational (standard logs) and randomized exposure (random logs)",
            "retrieved": "2026-11-25",
        },
        "synthetic": False,
    },
    "mind_small": {
        "title": "MIND-small (Microsoft News Dataset, small version)",
        "description": ("Impression logs of 50,000 sampled anonymised Microsoft News users, 2019-11-09 to "
                        "2019-11-15: for each impression the user's click history and which displayed news were "
                        "clicked; news titles, abstracts and categories."),
        "files": [
            {"path": "download/train/behaviors.tsv", "size": 92019716,
             "url": ("https://huggingface.co/datasets/huyva/MIND-small/resolve/"
                     "d771a25043d6352e49e76ca1fe3075c9ac6862bb/train/behaviors.tsv"),
             "sha256": "a424547c8fa17c9ea4879c2110221c2b2f2709f4f7615ede0b0d8e4765158658"},
            {"path": "download/train/news.tsv", "size": 41202121,
             "url": ("https://huggingface.co/datasets/huyva/MIND-small/resolve/"
                     "d771a25043d6352e49e76ca1fe3075c9ac6862bb/train/news.tsv"),
             "sha256": "cd49d7da275e9c405f627d75a0de9d4ec6ad43ff8a8db5d23a447591676e09d6"},
            {"path": "download/dev/behaviors.tsv", "size": 42838544,
             "url": ("https://huggingface.co/datasets/huyva/MIND-small/resolve/"
                     "d771a25043d6352e49e76ca1fe3075c9ac6862bb/dev/behaviors.tsv"),
             "sha256": "b6c460e33b1a8693252ded6e626da7d3ccf78920eea2ec11889020bb7d8443ef"},
            {"path": "download/dev/news.tsv", "size": 33519092,
             "url": ("https://huggingface.co/datasets/huyva/MIND-small/resolve/"
                     "d771a25043d6352e49e76ca1fe3075c9ac6862bb/dev/news.tsv"),
             "sha256": "e5d144667558c449d16084fe6bfa01940c2f5d5785ea9f9110292bc3b94eb822"},
        ],
        "extracted": {},
        "schema": {
            "download/train/behaviors.tsv": {"format": "tsv", "header": False, "rows": 156965,
                                             "columns": ["impression_id", "user_id", "time", "history", "impressions"]},
            "download/dev/behaviors.tsv": {"format": "tsv", "header": False, "rows": 73152,
                                           "columns": ["impression_id", "user_id", "time", "history", "impressions"]},
            "download/train/news.tsv": {"format": "tsv", "header": False, "rows": 51282,
                                        "columns": ["news_id", "category", "subcategory", "title", "abstract", "url",
                                                    "title_entities", "abstract_entities"]},
            "download/dev/news.tsv": {"format": "tsv", "header": False, "rows": 42416,
                                      "columns": ["news_id", "category", "subcategory", "title", "abstract", "url",
                                                  "title_entities", "abstract_entities"]},
        },
        "license": {
            "name": "Microsoft Research License Terms (MIND), non-commercial research use",
            "url": "https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf",
            "text_location": "https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf (PDF, not mirrored)",
            "terms": ("Free for non-commercial research use under the Microsoft Research License Terms. Crucible "
                      "redistributes none of the data; only derived model parameters and aggregate metrics are "
                      "committed."),
        },
        "provenance": {
            "publisher": "Microsoft (Wu et al.)",
            "source": ("Official distribution: msnews.github.io, now linking to the gated Hugging Face repository "
                       "yjw1029/MIND (requires an account that accepted the licence; the old "
                       "mind201910small.blob.core.windows.net host answers 409 'public access is not permitted'). "
                       "Downloaded from the ungated Hugging Face mirror huyva/MIND-small at revision "
                       "d771a25043d6352e49e76ca1fe3075c9ac6862bb, which holds the MINDsmall_train and MINDsmall_dev "
                       "files unchanged in the original TSV format."),
            "landing_page": "https://msnews.github.io/",
            "mirror": "https://huggingface.co/datasets/huyva/MIND-small (revision d771a250)",
            "mirror_check": ("row counts equal the official MIND-small statistics: train 156,965 impressions and "
                             "51,282 news, dev 73,152 impressions and 42,416 news; 50,000 train users"),
            "citation": ("Fangzhao Wu et al. MIND: A Large-scale Dataset for News Recommendation. ACL 2020. "
                         "doi:10.18653/v1/2020.acl-main.331"),
            "population": "50,000 users sampled from Microsoft News users with at least 5 clicks, anonymised IDs",
            "platform": "Microsoft News (msn.com) news feed, English-language, US market",
            "period": "2019-10-12 to 2019-11-15 (click histories); impressions 2019-11-09 to 2019-11-14 (train) "
                      "and 2019-11-15 (dev)",
            "collection": "Production impression logs; the displayed news were chosen by the production recommender",
            "identification": "observational",
            "retrieved": "2026-11-25",
        },
        "synthetic": False,
    },
}


def register(name: str, entry: dict) -> None:
    """Add a dataset entry (used by other calibration modules for their own datasets)."""
    need = ("title", "files", "license", "provenance")
    missing = [k for k in need if k not in entry]
    if missing:
        raise ValueError(f"dataset {name}: missing fields {missing}")
    for f in entry["files"]:
        if not f.get("sha256") or not f.get("url") or not f.get("path"):
            raise ValueError(f"dataset {name}: every file needs path, url and sha256")
    _DATASETS[name] = dict(entry, synthetic=bool(entry.get("synthetic", False)))


def _load_plugins() -> None:
    # Other calibration modules register their datasets on import; they are found by naming convention so this
    # file never needs editing: any module crucible.calibration.datasets_<x> is imported once.
    import importlib
    import pkgutil
    from . import __path__ as pkg_path
    for m in pkgutil.iter_modules(pkg_path):
        if m.name.startswith("datasets_"):
            importlib.import_module(f"{__package__}.{m.name}")


_PLUGINS_LOADED = False


def datasets() -> dict[str, dict]:
    """name -> {url, license, sha256 per file, size, schema, provenance, ...} (copies)."""
    global _PLUGINS_LOADED
    if not _PLUGINS_LOADED:
        _PLUGINS_LOADED = True
        _load_plugins()
    out = {}
    for name, e in _DATASETS.items():
        d = dict(e)
        d["url"] = e["files"][0]["url"]
        d["sha256"] = {f["path"]: f["sha256"] for f in e["files"]}
        d["size"] = sum(int(f.get("size") or 0) for f in e["files"])
        out[name] = d
    return out


def dataset_dir(name: str, dest: str | None = None) -> str:
    return dest or os.path.join(data_root(), name)


def _download(url: str, path: str, sha256: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".dl-", dir=os.path.dirname(path))
    os.close(fd)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "crucible-calibration/5"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as fh:
            shutil.copyfileobj(r, fh, 1 << 20)
        got = sha256_file(tmp)
        if got != sha256:
            raise IOError(f"checksum mismatch for {url}: expected {sha256}, got {got}")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _extract(archive: str, spec: dict, root: str) -> None:
    into = os.path.join(root, spec.get("into", "."))
    os.makedirs(into, exist_ok=True)
    if spec["format"] in ("tar.gz", "tar"):
        with tarfile.open(archive) as tf:
            tf.extractall(into, filter="data")          # rejects absolute paths, links outside, devices
    elif spec["format"] == "zip":
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                target = os.path.realpath(os.path.join(into, member))
                if not target.startswith(os.path.realpath(into) + os.sep):
                    raise IOError(f"unsafe path in archive: {member}")
            zf.extractall(into)
    else:
        raise ValueError(f"unknown archive format {spec['format']}")


def verify(name: str, dest: str | None = None) -> dict:
    """Check every downloaded and extracted file against its SHA-256. Returns {path: ok|missing|mismatch}."""
    e = _DATASETS[name] if name in _DATASETS else datasets()[name]
    root = dataset_dir(name, dest)
    out = {}
    wanted = {f["path"]: f["sha256"] for f in e["files"]}
    wanted.update(e.get("extracted", {}))
    for rel, sha in wanted.items():
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            out[rel] = "missing"
        else:
            out[rel] = "ok" if sha256_file(p) == sha else "mismatch"
    return out


def validate_schema(name: str, dest: str | None = None) -> list[str]:
    """Schema problems (columns, header, row counts); [] when the files match the recorded schema."""
    e = datasets()[name]
    root = dataset_dir(name, dest)
    errs = []
    for rel, sch in e.get("schema", {}).items():
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            errs.append(f"{rel}: missing")
            continue
        sep = "\t" if sch["format"] == "tsv" else ","
        with open(p, encoding="utf-8", newline="") as fh:
            first = fh.readline().rstrip("\r\n")
            rows = 1 + sum(1 for _ in fh)
        if sch.get("header"):
            rows -= 1
            cols = first.split(sep)
            if "columns" in sch and cols != sch["columns"]:
                errs.append(f"{rel}: columns {cols[:6]}... differ from the recorded schema")
            if "columns_prefix" in sch and cols[:len(sch["columns_prefix"])] != sch["columns_prefix"]:
                errs.append(f"{rel}: leading columns differ from the recorded schema")
        elif "columns" in sch and len(first.split(sep)) != len(sch["columns"]):
            errs.append(f"{rel}: {len(first.split(sep))} fields, expected {len(sch['columns'])}")
        if "rows" in sch and rows != sch["rows"]:
            errs.append(f"{rel}: {rows} rows, expected {sch['rows']}")
    return errs


def acquire(name: str, dest: str | None = None) -> str:
    """Download (if needed), verify checksums, extract and validate the schema; returns the dataset directory."""
    e = _DATASETS[name] if name in _DATASETS else datasets()[name]
    root = dataset_dir(name, dest)
    os.makedirs(root, exist_ok=True)
    for f in e["files"]:
        p = os.path.join(root, f["path"])
        if not (os.path.exists(p) and sha256_file(p) == f["sha256"]):
            _download(f["url"], p, f["sha256"])
        if f.get("extract"):
            wanted = [r for r in e.get("extracted", {}) if r.startswith(f["extract"].get("into", ""))]
            if not wanted or any(not os.path.exists(os.path.join(root, r)) for r in wanted):
                _extract(p, f["extract"], root)
    bad = {k: v for k, v in verify(name, root).items() if v != "ok"}
    if bad:
        raise IOError(f"dataset {name}: files failed verification: {bad}")
    errs = validate_schema(name, root)
    if errs:
        raise IOError(f"dataset {name}: schema validation failed: {errs}")
    return root
