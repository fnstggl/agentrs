"""G3-act: independent read of the realised information value in an ST staging run's experience store (a COPY of
the store; nothing in the run directory is written). For every record: chosen operation, the
reward_signals.information_value annotation, the matured targets of the record (metric names), and whether the record's
result added context items. Prints the distribution per operation and the target metrics.

    PYTHONPATH=<tree>/cam_live/src .venv/bin/python iv_store_report.py <copy of experience.sqlite>
(the copy needs its ``.blobs`` directory beside it; a read-only symlink to the original is enough)."""
import collections
import json
import sys

from cam_live.credit.replay import SqliteExperienceStore
from cam_live.learning_api import matured_outcomes


def main(path):
    st = SqliteExperienceStore(path, readonly=True)
    recs = sorted(st.all(), key=lambda r: r.seq)
    by_op = collections.defaultdict(list)
    targets = collections.Counter()
    n_new = 0
    rows = []
    for r in recs:
        d = r.decision.diagnostics if isinstance(r.decision.diagnostics, dict) else {}
        iv = d.get("reward_signals.information_value")
        new = set(r.post_item_refs or []) - set(r.context_item_refs or [])
        n_new += bool(new)
        for q, o in matured_outcomes(r):
            targets[q.metric] += 1
        if iv is not None:
            by_op[r.decision.chosen.operation].append(float(iv))
        rows.append({"seq": r.seq, "op": r.decision.chosen.operation, "iv": iv, "new_items": len(new),
                     "ctx_items": len(r.context_item_refs or []), "targets": [q.metric for q, _ in matured_outcomes(r)]})
    ivs = [v for vs in by_op.values() for v in vs]
    print(json.dumps({"records": len(recs), "with_new_items": n_new, "iv_measured": len(ivs),
                      "iv_min": min(ivs) if ivs else None, "iv_max": max(ivs) if ivs else None,
                      "iv_mean": sum(ivs) / len(ivs) if ivs else None,
                      "iv_abs_max": max(abs(v) for v in ivs) if ivs else None,
                      "by_operation": {k: {"n": len(v), "mean": sum(v) / len(v), "max_abs": max(abs(x) for x in v)}
                                       for k, v in sorted(by_op.items())},
                      "target_metrics": dict(targets.most_common()),
                      "context_items_median": sorted(x["ctx_items"] for x in rows)[len(rows) // 2] if rows else None},
                     indent=1))
    st.close()


if __name__ == "__main__":
    main(sys.argv[1])
