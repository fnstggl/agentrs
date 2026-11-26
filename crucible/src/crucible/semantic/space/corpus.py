"""The real topic corpus behind the shared space: MIND-small (Microsoft News Dataset, Wu et al. 2020).

Read from `$CRUCIBLE_DATA/mind_small/download/{train,dev}` (acquired and checksummed by
`crucible.calibration.datasets`; MSR research licence, non-commercial: neither the data nor per-user profiles are
committed to the repository). Users are anonymised MIND ids; nothing here reconstructs a real identity.

* `news()` - {news id: (category, subcategory, title, abstract)} over train and dev.
* `behaviors(split)` - (user id, time, history ids, [(news id, clicked)]) per impression.
"""
from __future__ import annotations

import csv
import os
import sys
from functools import lru_cache


def data_root() -> str:
    return os.environ.get("CRUCIBLE_DATA") or "/workspace/data/crucible"


def mind_dir(split: str) -> str:
    d = os.path.join(data_root(), "mind_small", "download", split)
    if not os.path.exists(os.path.join(d, "news.tsv")):
        raise FileNotFoundError(f"MIND-small {split} is not at {d}; acquire it with "
                                "`python -m crucible.calibration.datasets acquire mind_small`")
    return d


def news_text(rec: tuple[str, str, str, str]) -> str:
    """What a reader sees of a news item: its title and abstract."""
    _, _, title, abstract = rec
    return (title.strip() + (". " + abstract.strip() if abstract.strip() else "")).strip()


@lru_cache(maxsize=1)
def news() -> dict[str, tuple[str, str, str, str]]:
    csv.field_size_limit(sys.maxsize)
    out: dict[str, tuple[str, str, str, str]] = {}
    for split in ("train", "dev"):
        with open(os.path.join(mind_dir(split), "news.tsv"), newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
                if len(row) >= 5 and row[0] not in out:
                    out[row[0]] = (row[1], row[2], row[3], row[4])
    return out


def behaviors(split: str):
    with open(os.path.join(mind_dir(split), "behaviors.tsv"), encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            hist = parts[3].split() if parts[3] else []
            imps = []
            for x in parts[4].split():
                nid, _, lab = x.rpartition("-")
                imps.append((nid, lab == "1"))
            yield parts[1], parts[2], hist, imps


def corpus_files() -> dict[str, str]:
    """Paths of the MIND files the space reads (for provenance hashing)."""
    return {f"{s}/{n}": os.path.join(mind_dir(s), n) for s in ("train", "dev") for n in ("news.tsv", "behaviors.tsv")}
