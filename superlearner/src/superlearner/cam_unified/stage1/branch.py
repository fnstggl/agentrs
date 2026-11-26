"""Abandoned branches of a resumed lifetime.

A lifetime resumes from its last checkpoint: the operator restores the world snapshot, the agent its weights,
optimiser, fast / latent state, RNG and counters. Events the interrupted process archived *after* that checkpoint
are real experience of a world state that no longer exists (the resumed world continues from the checkpoint and
re-lives those simulated hours differently). The archive is append-only, so they stay on disk; `BranchView` is the
archive as the learner's readers see it - the replay sampler (E's `ReplaySampler`) and the consolidator (E's
`Consolidator`) read through it - with those events left out:

* ``scan_meta`` / ``scan`` rows whose ``gseq`` (or ``event_id``) belongs to an abandoned branch are skipped (replay
  membership and memory segments only ever contain events of the lifetime as it continued);
* ``stream_window`` (replay slices) drops them and re-indexes the anchor;
* everything else (``get``, ``count``, ``pointer``, ``flush``, ``mark_used``, ``tag_changes`` ...) is the archive's own.

Writers (the agent's ``archive.append``, checkpoints' ``archive.pointer``) use the archive itself, so the on-disk
record of what happened is complete; `abandoned_after(archive, count)` lists the events beyond a checkpoint's archive
pointer.
"""
from __future__ import annotations

from typing import Iterable, Optional


class BranchView:
    def __init__(self, archive, exclude: Optional[set] = None, exclude_gseq: Optional[set] = None):
        object.__setattr__(self, "_a", archive)
        object.__setattr__(self, "exclude", exclude if exclude is not None else set())      # event ids
        object.__setattr__(self, "exclude_gseq", exclude_gseq if exclude_gseq is not None else set())

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_a"), name)

    @property
    def archive(self):
        return self._a

    def add(self, items: Iterable[tuple]) -> None:
        """Exclude [(event_id, gseq)]."""
        for e, g in items:
            self.exclude.add(e)
            self.exclude_gseq.add(int(g))

    def _keep(self, row: dict) -> bool:
        g = row.get("gseq")
        if g is not None and int(g) in self.exclude_gseq:
            return False
        e = row.get("event_id")
        return not (e is not None and e in self.exclude)

    def scan_meta(self, *a, **kw):
        for rows in self._a.scan_meta(*a, **kw):
            if self.exclude or self.exclude_gseq:
                rows = [r for r in rows if self._keep(r)]
            if rows:
                yield rows

    def scan(self, *a, **kw):
        for recs in self._a.scan(*a, **kw):
            if self.exclude:
                recs = [r for r in recs if r.event_id not in self.exclude]
            if recs:
                yield recs

    def stream_window(self, env_id, actor_id, episode_id, step, gseq, before, after):
        """The archive's window with abandoned events removed; the window is widened until it again holds `before`
        / `after` kept events on each side (or the stream ends there)."""
        if not (self.exclude or self.exclude_gseq):
            return self._a.stream_window(env_id, actor_id, episode_id, step, gseq, before, after)
        keep = lambda r: r[1] not in self.exclude and int(r[2]) not in self.exclude_gseq   # noqa: E731
        extra = 0
        while True:
            rows, i = self._a.stream_window(env_id, actor_id, episode_id, step, gseq, before + extra, after + extra)
            pre = [r for r in rows[:i] if keep(r)]
            post = [r for r in rows[i + 1:] if keep(r)]
            full_pre = len(pre) >= before or i < before + extra
            full_post = len(post) >= after or len(rows) - i - 1 < after + extra
            if (full_pre and full_post) or extra > len(self.exclude) + len(self.exclude_gseq):
                pre, post = pre[max(0, len(pre) - before):], post[:after]
                return pre + [rows[i]] + post, len(pre)
            extra = 2 * extra + max(before, after, 1)

    def __len__(self) -> int:
        return len(self._a)


def abandoned_after(archive, count: int) -> list:
    """[(event_id, gseq)] of archived events with gseq >= `count` (a checkpoint's archive pointer count)."""
    out = []
    for rows in archive.scan_meta(columns=["gseq", "event_id"], gseq_min=int(count)):
        out.extend((r["event_id"], int(r["gseq"])) for r in rows)
    return out
