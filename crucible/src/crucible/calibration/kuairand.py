"""KuaiRand-Pure feature pipeline for the content-consumption, watch-duration and browsing-frequency models.

Everything here is offline fitting code: it reads the dataset that `datasets.acquire("kuairand_pure")` verified and
builds model inputs whose definitions are written in `docs/v5/calibration/plans/E1_evaluation_plan.json`. History
features use only events of days strictly before the row's day, and only the standard (recommender-exposed) logs, so
the random-exposure rows are scored with the same information a production system had.

Simulator mapping of every feature is in `FEATURE_MAP` (copied into each model card).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import os

import numpy as np
import pandas as pd

from .datasets import acquire

D0 = _dt.date(2022, 4, 8)                 # day 0
PRIOR_N = 20.0                             # pseudo-count for user / item consume rates
TAG_PRIOR_N = 5.0                          # pseudo-count for topic shares
SESSION_GAP_MS = 30 * 60 * 1000

CHOICE_FEATURES = ["user_log_plays_7d", "user_click_rate", "item_log_pop_7d", "item_click_rate", "item_log_age_days",
                   "item_log_duration_s", "user_tag_share", "user_tag_lift", "creator_familiarity", "item_seen_before",
                   "item_played_before", "hour_sin", "hour_cos"]
POPULARITY_FEATURES = ["item_click_rate", "item_log_pop_7d"]
WATCH_FEATURES = ["consumed", "item_log_duration_s", "user_tag_share", "user_tag_lift", "creator_familiarity",
                  "user_log_plays_7d", "user_click_rate", "item_click_rate", "item_log_pop_7d", "item_log_age_days",
                  "item_played_before"]
RETURN_FEATURES = ["active_days_7d", "days_since_last_active", "log_rows_7d", "log_sessions_7d", "active_yesterday",
                   "weekend"]

FEATURE_MAP = {
    "user_log_plays_7d": "log1p(items the individual consumed (views) in the previous 7 simulated days). Scale: the "
                         "dataset counts only plays of the 7,583 pool videos, so map by quantile (card "
                         "feature_quantiles) rather than by raw value",
    "user_click_rate": "(consumed + 20 m) / (feed exposures + 20) over the individual's whole history before today, m "
                       "= the card's prior_rate; scale-free",
    "item_log_pop_7d": "log1p(1000 * views of the item in the previous 7 days / individuals active in those 7 days); "
                       "per-capita, so independent of population size",
    "item_click_rate": "(views + 20 m) / (impressions + 20) of the item on the platform before today (audience "
                       "counters views / impressions); scale-free",
    "item_log_age_days": "log1p((now - published_at) / 86400); the data cover ages 0-30 days only",
    "item_log_duration_s": "log1p(duration_s) of the content item",
    "user_tag_share": "(individual's past views in the item's topic + 5 s) / (past views + 5), s = population share "
                      "of the topic; topic = the item's primary tag or the nearest topic centroid of its embedding",
    "user_tag_lift": "log(user_tag_share / s): relative interest in the item's topic (0 = average)",
    "creator_familiarity": "log1p(the individual's past views of this creator's items)",
    "item_seen_before": "log1p(times the individual was exposed to this item before today)",
    "item_played_before": "1 if the individual consumed this item before today, else 0 (repeat consumption)",
    "hour_sin": "sin(2 pi hour / 24) of the local simulated time",
    "hour_cos": "cos(2 pi hour / 24) of the local simulated time",
    "consumed": "1 if the choice model's draw was 'consume', 0 for a skip",
    "active_days_7d": "number of the previous 7 days on which the individual was active on the platform",
    "days_since_last_active": "days since the individual's last active day, capped at 14",
    "log_rows_7d": "log1p(feed exposures in the previous 7 days); map by quantile (dataset logs only a sample of "
                   "exposures)",
    "log_sessions_7d": "log1p(sessions in the previous 7 days; a session ends after 30 idle minutes); map by quantile",
    "active_yesterday": "1 if the individual was active the previous day",
    "weekend": "1 on Saturdays and Sundays",
}


def user_bucket(uid) -> int:
    return int(hashlib.sha256(f"e1:{int(uid)}".encode()).hexdigest()[:8], 16) % 100


def split_of_bucket(b: np.ndarray) -> np.ndarray:
    return np.where(b < 70, "train", np.where(b < 85, "val", "test"))


def day_of(date_int: pd.Series) -> np.ndarray:
    d = pd.to_datetime(date_int.astype(str), format="%Y%m%d")
    return (d - pd.Timestamp(D0)).dt.days.to_numpy()


def day_num(y: int, m: int, d: int) -> int:
    return (_dt.date(y, m, d) - D0).days


def load_raw(root: str | None = None) -> dict:
    root = root or acquire("kuairand_pure")
    d = os.path.join(root, "raw", "KuaiRand-Pure", "data")
    s = pd.concat([pd.read_csv(os.path.join(d, "log_standard_4_08_to_4_21_pure.csv")),
                   pd.read_csv(os.path.join(d, "log_standard_4_22_to_5_08_pure.csv"))], ignore_index=True)
    r = pd.read_csv(os.path.join(d, "log_random_4_22_to_5_08_pure.csv"))
    v = pd.read_csv(os.path.join(d, "video_features_basic_pure.csv"))
    logs = pd.concat([s, r], ignore_index=True)
    logs["day"] = day_of(logs["date"])
    first = v["tag"].astype("string").str.split(",").str[0]
    v["primary_tag"] = pd.to_numeric(first, errors="coerce").fillna(-1).astype(int)   # 96 videos have no tag
    v["author"] = pd.factorize(v["author_id"])[0]
    v["upload_day"] = (pd.to_datetime(v["upload_dt"]) - pd.Timestamp(D0)).dt.days
    logs = logs.merge(v[["video_id", "primary_tag", "author", "upload_day", "video_duration"]], on="video_id",
                      how="left")
    dur = logs["duration_ms"].where(logs["duration_ms"] > 0, logs["video_duration"])
    logs["duration_s"] = dur / 1000.0
    logs["play_s"] = logs["play_time_ms"] / 1000.0
    logs["bucket"] = logs["user_id"].map({u: user_bucket(u) for u in logs["user_id"].unique()})
    logs = logs.sort_values(["time_ms", "user_id", "video_id"], kind="mergesort").reset_index(drop=True)
    return {"logs": logs, "videos": v, "root": root}


# ------------------------------------------------------------------ prior-day cumulative lookups
class PriorCounts:
    """Sum of `w` over events with the same key and day < query day (and, with `window`, day >= query day - window)."""

    def __init__(self, keys: np.ndarray, days: np.ndarray, w: np.ndarray):
        comb = keys.astype(np.int64) * 64 + days.astype(np.int64)
        order = np.argsort(comb, kind="mergesort")
        comb, w = comb[order], w[order].astype(float)
        uniq, start = np.unique(comb, return_index=True)
        sums = np.add.reduceat(w, start) if len(w) else np.zeros(0)
        self.comb = uniq
        self.key = uniq // 64
        self.cum = np.cumsum(sums)

    def before(self, keys: np.ndarray, days: np.ndarray) -> np.ndarray:
        q = keys.astype(np.int64) * 64 + days.astype(np.int64)
        pos = np.searchsorted(self.comb, q, side="left") - 1
        ok = pos >= 0
        same = np.zeros(len(q), bool)
        same[ok] = self.key[pos[ok]] == keys[ok]
        total = np.zeros(len(q))
        total[same] = self.cum[pos[same]]
        # subtract everything of smaller keys: cum is global, so take the difference to the key's first entry
        first = np.searchsorted(self.comb, keys.astype(np.int64) * 64, side="left") - 1
        base = np.where(first >= 0, self.cum[np.maximum(first, 0)], 0.0)
        return np.where(same, total - base, 0.0)

    def window(self, keys, days, width: int) -> np.ndarray:
        return self.before(keys, days) - self.before(keys, days - width)


def choice_frame(raw: dict, prior_rate: float | None = None) -> pd.DataFrame:
    """Tab-1 rows of both logs with history features computed from earlier days of the standard logs."""
    L = raw["logs"]
    std = L[L["is_rand"] == 0]
    n_users = int(L["user_id"].max()) + 1
    n_vid = int(L["video_id"].max()) + 1
    s_click = std["is_click"].to_numpy(float)
    s_tab1 = (std["tab"] == 1).to_numpy(float)
    su, sv, sd = std["user_id"].to_numpy(), std["video_id"].to_numpy(), std["day"].to_numpy()
    rows = L[(L["tab"] == 1)].copy()
    dropped = int((~(rows["duration_s"] > 0)).sum())
    rows = rows[rows["duration_s"] > 0].copy()
    ru, rv, rd = rows["user_id"].to_numpy(), rows["video_id"].to_numpy(), rows["day"].to_numpy()
    if prior_rate is None:
        prior_rate = float(s_click[s_tab1 > 0].mean())

    user_plays = PriorCounts(su, sd, s_click)
    user_exp1 = PriorCounts(su, sd, s_tab1)
    user_clk1 = PriorCounts(su, sd, s_click * s_tab1)
    item_plays = PriorCounts(sv, sd, s_click)
    item_exp1 = PriorCounts(sv, sd, s_tab1)
    item_clk1 = PriorCounts(sv, sd, s_click * s_tab1)
    # distinct active users in the 7 days before each day (per-capita popularity)
    act = {d: set(std.loc[std["day"] == d, "user_id"].unique()) for d in range(int(L["day"].max()) + 2)}
    active7 = np.array([len(set().union(*[act.get(x, set()) for x in range(d - 7, d)])) for d in range(64)], float)

    feats = pd.DataFrame(index=rows.index)
    feats["user_log_plays_7d"] = np.log1p(user_plays.window(ru, rd, 7))
    feats["user_click_rate"] = (user_clk1.before(ru, rd) + PRIOR_N * prior_rate) / (user_exp1.before(ru, rd) + PRIOR_N)
    feats["item_log_pop_7d"] = np.log1p(1000.0 * item_plays.window(rv, rd, 7) / np.maximum(active7[rd], 1.0))
    feats["item_click_rate"] = (item_clk1.before(rv, rd) + PRIOR_N * prior_rate) / (item_exp1.before(rv, rd) + PRIOR_N)
    feats["item_log_age_days"] = np.log1p(np.maximum(rd - rows["upload_day"].to_numpy(), 0))
    feats["item_log_duration_s"] = np.log1p(rows["duration_s"].to_numpy())
    # topic (primary tag) affinity
    stag = std["primary_tag"].to_numpy() + 1
    rtag = rows["primary_tag"].to_numpy() + 1
    n_tag = int(max(stag.max(), rtag.max())) + 1
    tag_all = PriorCounts(stag, sd, s_click)
    tot_all = PriorCounts(np.zeros(len(sd), np.int64), sd, s_click)
    share_tag = (tag_all.before(rtag, rd) + 1.0) / (tot_all.before(np.zeros(len(rd), np.int64), rd) + n_tag)
    ut = PriorCounts(su.astype(np.int64) * n_tag + stag, sd, s_click)
    u_tag = ut.before(ru.astype(np.int64) * n_tag + rtag, rd)
    u_tot = user_plays.before(ru, rd)
    feats["user_tag_share"] = (u_tag + TAG_PRIOR_N * share_tag) / (u_tot + TAG_PRIOR_N)
    feats["user_tag_lift"] = np.log(feats["user_tag_share"].to_numpy() / share_tag)
    sa = std["author"].to_numpy()
    n_auth = int(raw["videos"]["author"].max()) + 2
    ua = PriorCounts(su.astype(np.int64) * n_auth + sa, sd, s_click)
    feats["creator_familiarity"] = np.log1p(ua.before(ru.astype(np.int64) * n_auth + rows["author"].to_numpy(), rd))
    uv_exp = PriorCounts(su.astype(np.int64) * n_vid + sv, sd, np.ones(len(sd)))
    uv_play = PriorCounts(su.astype(np.int64) * n_vid + sv, sd, s_click)
    uvq = ru.astype(np.int64) * n_vid + rv
    feats["item_seen_before"] = np.log1p(uv_exp.before(uvq, rd))
    feats["item_played_before"] = (uv_play.before(uvq, rd) > 0).astype(float)
    hour = (rows["hourmin"].to_numpy() // 100) % 24
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out = pd.concat([rows[["user_id", "video_id", "day", "is_rand", "is_click", "play_s", "duration_s", "bucket"]],
                     feats], axis=1)
    out["consumed"] = out["is_click"].astype(float)
    out["split_user"] = split_of_bucket(out["bucket"].to_numpy())
    out.attrs.update({"prior_rate": prior_rate, "dropped_no_duration": dropped, "n_users": n_users})
    return out.reset_index(drop=True)


# ------------------------------------------------------------------ activity (return / sessions)
def sessions_by_day(raw: dict) -> pd.DataFrame:
    """Per user and day: logged rows and sessions (30-minute gap rule, from both logs) with at least one row that day
    (a session crossing midnight counts on both days, so every active day has at least one session)."""
    L = raw["logs"][["user_id", "time_ms", "day"]].sort_values(["user_id", "time_ms"], kind="mergesort")
    u = L["user_id"].to_numpy()
    t = L["time_ms"].to_numpy()
    new = np.ones(len(L), bool)
    new[1:] = (u[1:] != u[:-1]) | (t[1:] - t[:-1] > SESSION_GAP_MS)
    L = L.assign(session_id=np.cumsum(new))
    g = L.groupby(["user_id", "day"]).agg(rows=("time_ms", "size"), sessions=("session_id", "nunique")).reset_index()
    return g


def activity_frame(raw: dict, first_day: int, last_day: int) -> pd.DataFrame:
    """One row per (user, day) for days first_day..last_day with the return features and the label `active`."""
    g = sessions_by_day(raw)
    users = np.sort(raw["logs"]["user_id"].unique())
    D = int(raw["logs"]["day"].max()) + 1
    U = int(users.max()) + 1
    rows = np.zeros((U, D))
    sess = np.zeros((U, D))
    rows[g["user_id"], g["day"]] = g["rows"]
    sess[g["user_id"], g["day"]] = g["sessions"]
    active = rows > 0
    recs = []
    for d in range(first_day, last_day + 1):
        lo = max(d - 7, 0)
        a7 = active[users, lo:d].sum(1)
        last = np.full(len(users), 14.0)
        hist = active[users, :d]
        any_ = hist.any(1)
        idx_last = d - 1 - np.argmax(hist[:, ::-1], axis=1)
        last[any_] = np.minimum(d - idx_last[any_], 14)
        wd = (D0 + _dt.timedelta(days=d)).weekday()
        recs.append(pd.DataFrame({
            "user_id": users, "day": d, "active": active[users, d].astype(float),
            "active_days_7d": a7.astype(float), "days_since_last_active": last,
            "log_rows_7d": np.log1p(rows[users, lo:d].sum(1)), "log_sessions_7d": np.log1p(sess[users, lo:d].sum(1)),
            "active_yesterday": active[users, d - 1].astype(float), "weekend": float(wd >= 5)}))
    out = pd.concat(recs, ignore_index=True)
    out["bucket"] = out["user_id"].map({u: user_bucket(u) for u in users}).to_numpy()
    out["split_user"] = split_of_bucket(out["bucket"].to_numpy())
    return out


def activity_matrices(raw: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(users, rows[U, D], sessions[U, D]) for forward simulation."""
    g = sessions_by_day(raw)
    users = np.sort(raw["logs"]["user_id"].unique())
    D = int(raw["logs"]["day"].max()) + 1
    U = int(users.max()) + 1
    rows = np.zeros((U, D))
    sess = np.zeros((U, D))
    rows[g["user_id"], g["day"]] = g["rows"]
    sess[g["user_id"], g["day"]] = g["sessions"]
    return users, rows, sess


def return_features_from(active: np.ndarray, rows: np.ndarray, sess: np.ndarray, d: int) -> np.ndarray:
    """RETURN_FEATURES for day d from (possibly simulated) per-user day matrices [n, D]."""
    lo = max(d - 7, 0)
    hist = active[:, :d]
    last = np.full(len(active), 14.0)
    any_ = hist.any(1)
    idx_last = d - 1 - np.argmax(hist[:, ::-1], axis=1)
    last[any_] = np.minimum(d - idx_last[any_], 14)
    wd = (D0 + _dt.timedelta(days=d)).weekday()
    return np.column_stack([active[:, lo:d].sum(1), last, np.log1p(rows[:, lo:d].sum(1)),
                            np.log1p(sess[:, lo:d].sum(1)), active[:, d - 1].astype(float),
                            np.full(len(active), float(wd >= 5))])
