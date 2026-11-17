"""GitHub engineering-work observer (``python -m agentrs.econ.github_observer``).

Source: GH Archive (https://www.gharchive.org) hourly dumps of public GitHub events — read-only,
no API rate limits, no credentials. Archives appear ~1-2 h after the hour, so this is near-real-time
replay, processed strictly in chronological order.

For every sampled newly opened issue / pull request it records a forecast BEFORE reading any later
event: P(closed within h), P(merged within h) for PRs, P(any comment within h) for issues, for
h in {10m, 1h, 6h}. Forecasts are persisted with the event time, wall time and model version. When
the archive has advanced past created_at + h, the outcome is resolved (from events actually seen —
never invented) and the online model takes gradient steps on the matured examples. Forecasts are
scored prequentially against a non-neural base-rate baseline. Longer horizons are not tracked.

IMPORTANT data caveats (measured on the 2026 archive): the archive holds only a few thousand events
per hour (far fewer than GitHub's real volume) and pull-request payloads are trimmed to ids and
branch refs (no title/body). Outcomes are therefore defined as "the archive RECORDS event X within
h" — absence from the archive is not proof that the event did not happen on GitHub.

Issue->fix relationships: a PR body saying "fixes/closes/resolves #N" links it to issue N, and an
issue counts as solved by code only if that PR was observed MERGED (never merely because the issue
closed). With trimmed PR payloads such links are usually unavailable; the count is reported as is.
Links are metadata-only candidate task templates (licences unverified; no code is downloaded).
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
import os
import random
import re
import time
import urllib.request
from pathlib import Path

import torch

HORIZONS = {"10m": 600, "1h": 3600, "6h": 21600}
TARGETS = ["closed", "merged", "commented"]
DIM = 1 << 18
FIX_RE = re.compile(r"\b(?:fix(?:es|ed)?|close[sd]?|resolve[sd]?)\s+#(\d+)", re.I)


def h(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little") % DIM


def features(ev: dict, kind: str, repo_seen: int) -> list[int]:
    p = ev.get("payload") or {}
    obj = p.get("issue") or p.get("pull_request") or {}
    title = (obj.get("title") or "")[:200].lower()
    head_ref = ((obj.get("head") or {}).get("ref") or "").lower()
    base_ref = ((obj.get("base") or {}).get("ref") or "").lower()
    body = obj.get("body") or ""
    f = [f"kind={kind}", f"assoc={obj.get('author_association')}", f"hour={ev['created_at'][11:13]}",
         f"repo_seen={min(6, int(math.log2(1 + repo_seen)))}", f"body_len={min(8, int(math.log2(1 + len(body))))}",
         f"org={'org' in ev}", f"draft={obj.get('draft')}", f"repo={ev['repo']['name'].lower()}",
         f"owner={ev['repo']['name'].split('/')[0].lower()}", f"nlabels={min(3, len(obj.get('labels') or []))}",
         f"bot={'[bot]' in (ev.get('actor') or {}).get('login', '')}", f"fixref={bool(FIX_RE.search(body))}"]
    f += [f"w={w}" for w in re.findall(r"[a-z]{3,}", title)[:20]]
    f += [f"head={w}" for w in re.findall(r"[a-z]{3,}", head_ref)[:6]] + [f"base={base_ref[:30]}"]
    f += [f"actor={(ev.get('actor') or {}).get('login', '').lower()}"]
    f += [f"label={(lb.get('name') or '').lower()}" for lb in (obj.get("labels") or [])[:5]]
    return [h(x) for x in f] + [h("bias")]


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.EmbeddingBag(DIM, len(TARGETS) * len(HORIZONS), mode="sum")
        torch.nn.init.zeros_(self.emb.weight)
        self.bias = torch.nn.Parameter(torch.full((len(TARGETS) * len(HORIZONS),), -1.0))

    def forward(self, feats: list[list[int]]) -> torch.Tensor:
        idx = torch.tensor([i for f in feats for i in f])
        off = torch.tensor([0] + [len(f) for f in feats[:-1]]).cumsum(0)
        return self.emb(idx, off) + self.bias


def head(target: str, hz: str) -> int:
    return TARGETS.index(target) * len(HORIZONS) + list(HORIZONS).index(hz)


def applicable(kind: str, target: str) -> bool:
    return not ((kind == "issue" and target == "merged") or (kind == "pr" and target == "commented"))


def auroc(ps, ys):
    pos = [p for p, y in zip(ps, ys) if y]
    neg = [p for p, y in zip(ps, ys) if not y]
    if not pos or not neg:
        return None
    allv = sorted([(p, 1) for p in pos] + [(p, 0) for p in neg])
    r, rank_sum, i = 0, 0.0, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        rank_sum += avg * sum(1 for k in range(i, j) if allv[k][1])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


class Observer:
    def __init__(self, workdir: Path, sample: float, seed: int = 0):
        self.dir = workdir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(seed)
        self.sample = sample
        self.model = Model()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=0.02)
        self.version = 0
        self.tracked: dict[str, dict] = {}       # key -> {created, kind, feats, pred, base, outcome times}
        self.repo_seen: dict[str, int] = {}
        self.base = {}                            # (kind, target, hz) -> [pos, n] (running base rate)
        self.scores = {}                          # (kind,target,hz) -> {"ll_model","ll_base","n","p","b","y"}
        self.links: dict[str, dict] = {}          # "repo#issue" -> {pr, merged, issue_closed}
        self.counts = {"hours": 0, "events_seen": 0, "tracked": 0, "matured": 0, "gradient_steps": 0, "links": 0,
                       "links_merged": 0}
        self.next_hour = None
        st = self.dir / "state.pt"
        if st.exists():
            s = torch.load(st, weights_only=False)
            self.model.load_state_dict(s["model"])
            self.opt.load_state_dict(s["opt"])
            for k in ("version", "tracked", "repo_seen", "base", "scores", "links", "counts", "next_hour"):
                setattr(self, k, s[k])

    def save(self):
        tmp = self.dir / "state.pt.tmp"
        torch.save({k: getattr(self, k) for k in ("version", "tracked", "repo_seen", "base", "scores", "links", "counts",
                                                  "next_hour")} | {"model": self.model.state_dict(),
                                                                   "opt": self.opt.state_dict()}, tmp)
        os.replace(tmp, self.dir / "state.pt")
        self.write_status()

    def write_status(self):
        sc = {}
        for (kind, t, hz), v in self.scores.items():
            n = v["n"]
            if n == 0:
                continue
            sc[f"{kind}/{t}/{hz}"] = {"n": n, "positive_rate": round(sum(v["y"][-5000:]) / max(1, len(v["y"][-5000:])), 4),
                                      "logloss_model": round(v["ll_model"] / n, 4), "logloss_baseline": round(v["ll_base"] / n, 4),
                                      "skill_vs_baseline": round(1 - v["ll_model"] / v["ll_base"], 4) if v["ll_base"] else None,
                                      "auroc_model": auroc(v["p"][-5000:], v["y"][-5000:]),
                                      "auroc_baseline": auroc(v["b"][-5000:], v["y"][-5000:])}
        st = {"time": time.time(), "archive_next_hour": self.next_hour, "model_version": self.version,
              "outcome_definition": "event RECORDED in GH Archive within the horizon (archive is a reduced sample)",
              "counts": self.counts, "pending": len(self.tracked), "scores": sc,
              "source": "GH Archive hourly public events (read-only replay, ~1-2h behind real time)"}
        tmp = self.dir / "status.json.tmp"
        tmp.write_text(json.dumps(st, indent=1, default=str))
        os.replace(tmp, self.dir / "status.json")

    # ------------------------------------------------------------------ archive processing
    def process_hour(self, hour: dt.datetime) -> bool:
        url = f"https://data.gharchive.org/{hour:%Y-%m-%d}-{hour.hour}.json.gz"
        try:
            resp = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "agentrs-research-observer"}),
                                          timeout=120)
        except Exception as e:  # noqa: BLE001 - not yet published / transient
            print(f"[github] {url} not available: {e}", flush=True)
            return False
        preds_f = open(self.dir / "predictions.jsonl", "a")
        n = 0
        clock = None
        with gzip.GzipFile(fileobj=resp) as gz:
            for line in gz:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                n += 1
                clock = ev.get("created_at") or clock
                self.handle(ev, preds_f)
                if n % 200000 == 0 and clock:
                    self.mature(clock)
        preds_f.close()
        if clock:
            self.mature(clock)
        self.counts["hours"] += 1
        self.counts["events_seen"] += n
        print(f"[github] {url}: {n} events, tracked {len(self.tracked)}, matured {self.counts['matured']}", flush=True)
        return True

    def handle(self, ev: dict, preds_f) -> None:
        typ = ev.get("type")
        if typ not in ("IssuesEvent", "PullRequestEvent", "IssueCommentEvent", "PullRequestReviewCommentEvent"):
            return
        repo = ev["repo"]["name"]
        p = ev.get("payload") or {}
        act = p.get("action")
        if typ == "IssuesEvent" or typ == "PullRequestEvent":
            kind = "issue" if typ == "IssuesEvent" else "pr"
            obj = p.get("issue") or p.get("pull_request") or {}
            num = obj.get("number") or p.get("number")
            key = f"{kind}:{repo}#{num}"
            if act == "opened":
                self.repo_seen[repo] = self.repo_seen.get(repo, 0) + 1
                if self.rng.random() > self.sample:
                    return
                feats = features(ev, kind, self.repo_seen[repo])
                with torch.no_grad():
                    prob = torch.sigmoid(self.model([feats])[0]).tolist()
                rec = {"created": ev["created_at"], "kind": kind, "feats": feats, "pred": prob, "version": self.version,
                       "base": {}, "events": {}}
                for t in TARGETS:
                    for hz in HORIZONS:
                        if applicable(kind, t):
                            pos, cnt = self.base.get((kind, t, hz), [1, 2])
                            rec["base"][f"{t}/{hz}"] = pos / cnt
                self.tracked[key] = rec
                self.counts["tracked"] += 1
                preds_f.write(json.dumps({"key": key, "event_time": ev["created_at"], "wall_time": time.time(),
                                          "model_version": self.version,
                                          "pred": {f"{t}/{hz}": round(prob[head(t, hz)], 4) for t in TARGETS for hz in HORIZONS
                                                   if applicable(kind, t)}}) + "\n")
                if kind == "pr":
                    for m in FIX_RE.findall(obj.get("body") or "")[:3]:
                        lk = f"{repo}#{m}"
                        if lk not in self.links:
                            self.links[lk] = {"pr": num, "pr_title": (obj.get("title") or "")[:160], "merged": False,
                                              "issue_closed": False, "time": ev["created_at"]}
                            self.counts["links"] += 1
            elif act in ("closed", "merged"):
                merged = kind == "pr" and (act == "merged" or bool(obj.get("merged")))
                if key in self.tracked:
                    self.tracked[key]["events"].setdefault("closed", ev["created_at"])
                    if merged:
                        self.tracked[key]["events"].setdefault("merged", ev["created_at"])
                if merged:
                    for lk, v in self.links.items():
                        if lk.startswith(repo + "#") and v["pr"] == num and not v["merged"]:
                            v["merged"] = True
                            self.counts["links_merged"] += 1
                if kind == "issue" and f"{repo}#{num}" in self.links:
                    self.links[f"{repo}#{num}"]["issue_closed"] = True
        else:
            obj = p.get("issue") or p.get("pull_request") or {}
            num = obj.get("number")
            key = f"issue:{repo}#{num}"
            if key in self.tracked and act in ("created", None):
                self.tracked[key]["events"].setdefault("commented", ev["created_at"])

    def mature(self, clock: str) -> None:
        now = dt.datetime.fromisoformat(clock.replace("Z", "+00:00"))
        batch_feats, batch_y, batch_mask = [], [], []
        done = []
        for key, r in self.tracked.items():
            c = dt.datetime.fromisoformat(r["created"].replace("Z", "+00:00"))
            age = (now - c).total_seconds()
            y = [0.0] * (len(TARGETS) * len(HORIZONS))
            mk = [0.0] * len(y)
            newly = False
            for t in TARGETS:
                if not applicable(r["kind"], t):
                    continue
                for hz, sec in HORIZONS.items():
                    tag = f"{t}/{hz}"
                    if age < sec or tag in r.get("resolved", {}):
                        continue
                    et = r["events"].get(t)
                    occurred = et is not None and (dt.datetime.fromisoformat(et.replace("Z", "+00:00")) - c).total_seconds() <= sec
                    r.setdefault("resolved", {})[tag] = int(occurred)
                    newly = True
                    i = head(t, hz)
                    y[i], mk[i] = float(occurred), 1.0
                    k3 = (r["kind"], t, hz)
                    pm = min(max(r["pred"][i], 1e-4), 1 - 1e-4)
                    pb = min(max(r["base"][tag], 1e-4), 1 - 1e-4)
                    s = self.scores.setdefault(k3, {"ll_model": 0.0, "ll_base": 0.0, "n": 0, "p": [], "b": [], "y": []})
                    s["ll_model"] -= math.log(pm if occurred else 1 - pm)
                    s["ll_base"] -= math.log(pb if occurred else 1 - pb)
                    s["n"] += 1
                    s["p"].append(pm)
                    s["b"].append(pb)
                    s["y"].append(int(occurred))
                    for lst in ("p", "b", "y"):
                        s[lst] = s[lst][-20000:]
                    bp = self.base.setdefault(k3, [1, 2])
                    bp[0] += int(occurred)
                    bp[1] += 1
            if newly:
                batch_feats.append(r["feats"])
                batch_y.append(y)
                batch_mask.append(mk)
                self.counts["matured"] += 1
            if age > max(HORIZONS.values()):
                done.append(key)
        for k in done:
            r = self.tracked.pop(k)
            with open(self.dir / "outcomes.jsonl", "a") as f:
                f.write(json.dumps({"key": k, "created": r["created"], "resolved": r.get("resolved"),
                                    "pred_version": r["version"]}) + "\n")
        # gradient steps on the newly matured labels (only the resolved heads carry loss)
        for s in range(0, len(batch_feats), 256):
            fb = batch_feats[s:s + 256]
            yb = torch.tensor(batch_y[s:s + 256])
            mb = torch.tensor(batch_mask[s:s + 256])
            logits = self.model(fb)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none") * mb).sum() / mb.sum().clamp(min=1)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.counts["gradient_steps"] += 1
            self.version += 1

    def write_templates(self) -> None:
        with open(self.dir / "task_templates.jsonl", "w") as f:
            for lk, v in self.links.items():
                if v["merged"]:
                    f.write(json.dumps({"issue": lk, **v, "status": "fix PR merged (licence unverified; metadata only)"}) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="runs/overnight/github")
    ap.add_argument("--max-hours", type=float, default=10.0)
    ap.add_argument("--sample", type=float, default=1.0)
    ap.add_argument("--backfill-hours", type=int, default=3)
    a = ap.parse_args(argv)
    obs = Observer(Path(a.workdir), a.sample)
    t_end = time.time() + a.max_hours * 3600
    if obs.next_hour is None:
        start = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0) - dt.timedelta(hours=a.backfill_hours)
        obs.next_hour = start.isoformat()
    obs.write_status()
    while time.time() < t_end:
        hour = dt.datetime.fromisoformat(obs.next_hour)
        if hour + dt.timedelta(hours=1, minutes=10) > dt.datetime.now(dt.timezone.utc):
            time.sleep(300)
            continue
        if obs.process_hour(hour):
            obs.next_hour = (hour + dt.timedelta(hours=1)).isoformat()
            obs.write_templates()
            obs.save()
        else:
            time.sleep(300)
    obs.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
