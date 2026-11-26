"""Criteo Uplift Prediction dataset v2.1: acquisition, schema validation, conversion and splits (workstream E2).

Offline and evaluator side only; world code never imports this module. The file is downloaded from the
Criteo-owned Hugging Face mirror at a pinned revision (the original go.criteo.net link returned 404 on 2026-11-25),
its SHA-256 is checked, the schema is validated, and the CSV is converted once into NumPy `.npy` arrays
(features float32 [N, 12]; treatment, conversion, visit, exposure uint8 [N]).
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request

import numpy as np

DATASET = "criteo_uplift_v2.1"
HF_REPO = "criteo/criteo-uplift"
HF_REVISION = "2424920019e49d52d72c13ac1143ec5d53af276b"
FILE = "criteo-research-uplift-v2.1.csv.gz"
URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{FILE}"
SHA256 = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
BYTES = 311422618
ROWS = 13979592
FEATURES = [f"f{i}" for i in range(12)]
LABELS = ["treatment", "conversion", "visit", "exposure"]
HEADER = FEATURES + LABELS
SPLIT_SEED = 20261125
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)
SPLIT_NAMES = ("train", "validation", "test")


def data_root() -> str:
    return os.environ.get("CRUCIBLE_DATA", "/workspace/data/crucible")


def dataset_dir(dest: str | None = None) -> str:
    return dest or os.path.join(data_root(), DATASET)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest: str | None = None) -> str:
    d = dataset_dir(dest)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, FILE)
    if os.path.exists(path) and os.path.getsize(path) == BYTES and sha256_file(path) == SHA256:
        return path
    tmp = path + ".part"
    req = urllib.request.Request(URL, headers={"User-Agent": "crucible-calibration/1"})
    with urllib.request.urlopen(req, timeout=300) as resp, open(tmp, "wb") as out:
        while True:
            chunk = resp.read(1 << 22)
            if not chunk:
                break
            out.write(chunk)
    got = sha256_file(tmp)
    if got != SHA256:
        os.remove(tmp)
        raise RuntimeError(f"checksum mismatch for {URL}: {got} != {SHA256}")
    os.replace(tmp, path)
    return path


def convert(dest: str | None = None, chunk_rows: int = 1_000_000) -> dict:
    """Validate the schema and write features.npy / labels .npy. Returns the conversion record."""
    import gzip
    import pandas as pd
    d = dataset_dir(dest)
    src = os.path.join(d, FILE)
    rec_path = os.path.join(d, "converted.json")
    if os.path.exists(rec_path):
        with open(rec_path) as fh:
            rec = json.load(fh)
        if rec.get("source_sha256") == SHA256 and all(os.path.exists(os.path.join(d, f)) for f in rec["files"]):
            return rec
    with gzip.open(src, "rt") as fh:
        header = fh.readline().strip().split(",")
    if header != HEADER:
        raise ValueError(f"schema mismatch: header {header} != {HEADER}")
    feats = np.lib.format.open_memmap(os.path.join(d, "features.npy"), mode="w+", dtype=np.float32, shape=(ROWS, 12))
    labs = {k: np.lib.format.open_memmap(os.path.join(d, f"{k}.npy"), mode="w+", dtype=np.uint8, shape=(ROWS,))
            for k in LABELS}
    dtypes = {c: np.float64 for c in FEATURES} | {c: np.int64 for c in LABELS}
    pos = 0; nan_features = 0; bad_binary = 0
    for df in pd.read_csv(src, chunksize=chunk_rows, dtype=dtypes):
        n = len(df)
        if pos + n > ROWS:
            raise ValueError("more rows than the published row count")
        X = df[FEATURES].to_numpy(dtype=np.float64)
        nan_features += int(np.isnan(X).sum())
        feats[pos:pos + n] = X.astype(np.float32)
        for k in LABELS:
            v = df[k].to_numpy()
            bad_binary += int(((v != 0) & (v != 1)).sum())
            labs[k][pos:pos + n] = v.astype(np.uint8)
        pos += n
    if pos != ROWS:
        raise ValueError(f"row count {pos} != published {ROWS}")
    feats.flush()
    for a in labs.values():
        a.flush()
    t = np.asarray(labs["treatment"]); e = np.asarray(labs["exposure"])
    v = np.asarray(labs["visit"]); c = np.asarray(labs["conversion"])
    checks = {"rows": pos, "nan_features": nan_features, "non_binary_label_values": bad_binary,
              "exposed_in_control": int((e[t == 0] == 1).sum()), "conversion_without_visit": int(((c == 1) & (v == 0)).sum()),
              "treatment_ratio": float(t.mean()), "visit_rate": float(v.mean()), "conversion_rate": float(c.mean()),
              "exposure_rate_treated": float(e[t == 1].mean())}
    if nan_features or bad_binary:
        raise ValueError(f"schema validation failed: {checks}")
    files = ["features.npy"] + [f"{k}.npy" for k in LABELS]
    rec = {"dataset": DATASET, "source_file": FILE, "source_sha256": SHA256, "checks": checks, "files": files,
           "file_sha256": {f: sha256_file(os.path.join(d, f)) for f in files},
           "feature_dtype": "float32 (converted from the CSV's decimal text)"}
    with open(rec_path, "w") as fh:
        json.dump(rec, fh, indent=1)
    return rec


LICENSE = {
    "spdx": "CC-BY-NC-SA-4.0",
    "summary": "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (dataset card on the Criteo-owned Hugging Face "
               "repository). Attribution: Diemert, Betlei, Renaudin and Amini, 'A Large Scale Benchmark for Uplift "
               "Modeling', AdKDD 2018. Fitted parameters derived from it are shared under the same license, "
               "non-commercially.",
    "text_locations": ["https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode",
                       f"https://huggingface.co/datasets/{HF_REPO}/blob/{HF_REVISION}/README.md"],
}


TITLE = "Criteo Uplift Prediction Dataset v2.1"
DESCRIPTION = ("Pooled incrementality tests of Criteo display advertising: 13,979,592 users with 12 anonymised "
               "features, a randomised treatment (eligibility for targeting), exposure, visit and conversion labels.")
# Provenance facts that the calibration registry requires of every dataset (datasets.py, audit.provenance_complete).
PROVENANCE = {
    "publisher": "Criteo AI Lab (Diemert, Betlei, Renaudin, Amini)",
    "citation": "Diemert, Betlei, Renaudin, Amini. A Large Scale Benchmark for Uplift Modeling. AdKDD 2018.",
    "population": "users reached by the display-advertising campaigns of Criteo's advertisers in several pooled "
                  "incrementality tests (advertisers, countries and dates not disclosed), sub-sampled non-uniformly "
                  "by the publisher to 13,979,592 users",
    "platform": "Criteo display advertising; outcomes are a visit to and a conversion on the advertiser's website",
    "period": "collection period not disclosed by the publisher; released with Diemert et al. (AdKDD 2018)",
    "identification": "randomized controlled trial of eligibility for targeting (exposure is not randomised)",
    "retrieved": "2026-11-25",
}


def manifest(dest: str | None = None) -> dict:
    """The committed dataset manifest (built from a completed conversion record)."""
    with open(os.path.join(dataset_dir(dest), "converted.json")) as fh:
        conv = json.load(fh)
    return {
        "schema": "crucible.v5.calibration.dataset/1", "name": DATASET, "workstream": "E2", "synthetic": False,
        "title": TITLE, "description": DESCRIPTION, "acquirer": __name__,
        "url": URL, "homepage": "https://ailab.criteo.com/criteo-uplift-prediction-dataset/",
        "mirror": {"kind": "huggingface", "repo": HF_REPO, "revision": HF_REVISION,
                   "note": "go.criteo.net/criteo-research-uplift-v2.1.csv.gz returned HTTP 404 on 2026-11-25; the "
                           "Criteo-owned Hugging Face repository holds the same file (LFS SHA-256 below)"},
        "license": LICENSE,
        "files": [{"file": FILE, "url": URL, "sha256": SHA256, "bytes": BYTES}],
        "derived_files": [{"file": f, "sha256": s} for f, s in conv["file_sha256"].items()],
        "size": {"rows": ROWS, "compressed_bytes": BYTES, "converted": "features float32 [N,12] + 4 uint8 label columns"},
        "schema_columns": {**{c: "float (anonymised random projection)" for c in FEATURES},
                           "treatment": "0/1 randomised assignment (1 = may be targeted)", "conversion": "0/1",
                           "visit": "0/1", "exposure": "0/1, user was effectively shown an ad (treated arm only)"},
        "validation": conv["checks"],
        "provenance": {
            "source": "Criteo AI Lab incrementality tests, pooled; released with Diemert et al. (AdKDD 2018)",
            "design": "randomised controlled trials: a random part of the population is prevented from being targeted",
            "privacy": "sub-sampled non-uniformly so that the original incrementality level cannot be deduced; features "
                       "anonymised and randomly projected",
            "period_and_population": "not disclosed by the publisher (users of Criteo's advertisers' campaigns)",
            "acquired_by": "crucible.calibration.criteo.acquire",
            **PROVENANCE,
        },
    }


def acquire(dest: str | None = None) -> str:
    download(dest)
    convert(dest)
    return dataset_dir(dest)


# ---------------------------------------------------------------- registry hooks (crucible.calibration.datasets)
def acquire_dataset(entry: dict | None = None, dest: str | None = None) -> str:
    """`datasets.acquire` hook: download (checksum-verified), validate the schema and convert once."""
    return acquire(dest)


def verify_dataset(entry: dict, dest: str | None = None) -> dict[str, str]:
    """`datasets.verify` hook, {file: ok|missing|mismatch}: the downloaded file and the converted arrays against the
    manifest's SHA-256."""
    d = dataset_dir(dest)
    want = {f["file"]: f["sha256"] for f in entry.get("files", [])}
    want.update({f["file"]: f["sha256"] for f in entry.get("derived_files", [])})
    out = {}
    for fn, sha in want.items():
        p = os.path.join(d, fn)
        out[fn] = "missing" if not os.path.exists(p) else ("ok" if sha256_file(p) == sha else "mismatch")
    return out


def validate_dataset(entry: dict, dest: str | None = None) -> list[str]:
    """`datasets.validate_schema` hook: the conversion record's schema checks (header, published row count, no NaN
    feature, binary labels, no exposure in control, no conversion without a visit) agree with the manifest, and the
    converted arrays have the published shapes."""
    d = dataset_dir(dest)
    p = os.path.join(d, "converted.json")
    if not os.path.exists(p):
        return [f"{p} missing: acquire the dataset"]
    with open(p) as fh:
        rec = json.load(fh)
    errs = []
    if rec.get("source_sha256") != SHA256:
        errs.append("conversion record is not of the published file")
    checks = rec.get("checks") or {}
    if checks.get("rows") != ROWS:
        errs.append(f"{checks.get('rows')} rows, published {ROWS}")
    for k in ("nan_features", "non_binary_label_values", "exposed_in_control", "conversion_without_visit"):
        if checks.get(k) != 0:
            errs.append(f"{k} = {checks.get(k)}")
    if entry.get("validation") and entry["validation"] != checks:
        errs.append("conversion checks differ from the manifest's validation record")
    for f, shape in [("features.npy", (ROWS, len(FEATURES)))] + [(f"{k}.npy", (ROWS,)) for k in LABELS]:
        fp = os.path.join(d, f)
        if not os.path.exists(fp):
            errs.append(f"{f} missing")
        elif np.load(fp, mmap_mode="r").shape != shape:
            errs.append(f"{f} has shape {np.load(fp, mmap_mode='r').shape}, expected {shape}")
    return errs


def load(dest: str | None = None, mmap: bool = True) -> dict[str, np.ndarray]:
    d = dataset_dir(dest)
    for f in ["features.npy"] + [f"{k}.npy" for k in LABELS]:
        if not os.path.exists(os.path.join(d, f)):
            raise FileNotFoundError(f"{d}/{f} missing: run crucible.calibration.acquire('{DATASET}')")
    mode = "r" if mmap else None
    out = {"X": np.load(os.path.join(d, "features.npy"), mmap_mode=mode)}
    for k in LABELS:
        out[k] = np.load(os.path.join(d, f"{k}.npy"), mmap_mode=mode)
    return out


def splits(n: int = ROWS) -> np.ndarray:
    """Split label per row (0 train, 1 validation, 2 test): seeded PCG64 permutation, as prespecified in the plan."""
    perm = np.random.Generator(np.random.PCG64(SPLIT_SEED)).permutation(n)
    lab = np.empty(n, dtype=np.uint8)
    a = int(round(SPLIT_FRACTIONS[0] * n)); b = a + int(round(SPLIT_FRACTIONS[1] * n))
    lab[perm[:a]] = 0; lab[perm[a:b]] = 1; lab[perm[b:]] = 2
    return lab


def split_digest(lab: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(lab, dtype=np.uint8).tobytes()).hexdigest()


def audit_indices(lab: np.ndarray, n: int = 100_000) -> np.ndarray:
    """Fixed random audit subsample of the test split (seeded, sorted) on which the unit tests recompute metrics."""
    test = np.flatnonzero(lab == 2)
    pick = np.random.Generator(np.random.PCG64(SPLIT_SEED + 1)).choice(test.size, size=min(n, test.size), replace=False)
    return np.sort(test[pick])
