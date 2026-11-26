"""Measurements of the shared space (used by Gate E tests and the workstream report).

* `finegrained(split)` - separation of negation / polarity / quantity changes from paraphrases (`finegrained.py`
  pairs): mean cosine distance of changed pairs over mean distance of paraphrase pairs, per kind, and the share of
  bases whose changed sentence is farther than their paraphrase. Also for the topic block alone (what plain
  mean-pooled MiniLM gives).
* `long_document_locality()` - a ~3,000-word article and copies with one local edit: item-vector distance, the
  edited chunk's distance, and whether a query about the edited passage changes its best chunk match.
* `shortcut_resistance()` - informative articles vs keyword stuffing, word salads and meaningless tokens on the
  same topics: intrinsic features (LM coherence of novel tokens, language validity, compression) and topic affinity
  to an on-topic interest.
* `blocks(vec)` / `reweight(vec, ...)` - recover the blocks of a space vector and re-assemble with other weights
  (block-weight selection on DEV).
"""
from __future__ import annotations

import math

import numpy as np

from .. import finegrained as FG
from . import compute as CP


def _cosd(a, b) -> float:
    return float(1.0 - np.dot(a, b) / max(1e-12, np.linalg.norm(a) * np.linalg.norm(b)))


def blocks(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = np.asarray(vec, np.float32)
    t = v[CP.TOPIC] / math.sqrt(CP.W_TOPIC)
    return t / max(1e-12, float(np.linalg.norm(t))), v[CP.AFFECT] / math.sqrt(CP.W_AFFECT), \
        v[CP.QUANT] / math.sqrt(CP.W_QUANT)


def reweight(vec: np.ndarray, wt: float, wa: float, wq: float) -> np.ndarray:
    t, a, q = blocks(vec)
    qn = float(np.linalg.norm(q))
    q = q / qn if qn > 1e-9 else q
    v = np.concatenate([math.sqrt(wt) * t, math.sqrt(wa) * a, math.sqrt(wq) * q])
    return v / max(1e-12, float(np.linalg.norm(v)))


def finegrained_from(vecs: dict[str, np.ndarray], pairs, transform=None) -> dict:
    tf = transform or (lambda v: v)
    out = {"by_kind": {}}
    para_all, chg_all, wins = [], [], 0
    for kind in FG.KINDS:
        P, Cg, w = [], [], 0
        for b, p, c, k in pairs:
            if k != kind:
                continue
            dp, dc = _cosd(tf(vecs[b]), tf(vecs[p])), _cosd(tf(vecs[b]), tf(vecs[c]))
            P.append(dp)
            Cg.append(dc)
            w += dc > dp
        para_all += P
        chg_all += Cg
        wins += w
        out["by_kind"][kind] = {"mean_paraphrase_distance": round(float(np.mean(P)), 4),
                                "mean_changed_distance": round(float(np.mean(Cg)), 4),
                                "separation_ratio": round(float(np.mean(Cg) / max(1e-9, np.mean(P))), 4),
                                "changed_farther_share": round(w / len(P), 4), "pairs": len(P)}
    out["separation_ratio"] = round(float(np.mean(chg_all) / max(1e-9, np.mean(para_all))), 4)
    out["min_kind_separation_ratio"] = min(v["separation_ratio"] for v in out["by_kind"].values())
    out["changed_farther_share"] = round(wins / len(para_all), 4)
    return out


def finegrained(split: str = "test") -> dict:
    """Fine-grained separation in the full space and in its topic block (plain MiniLM-chunk topic)."""
    from . import text_vectors
    pairs = FG.TEST if split == "test" else FG.DEV
    texts = sorted({t for b, p, c, _ in pairs for t in (b, p, c)})
    V = text_vectors(None, texts)
    vecs = dict(zip(texts, V))
    full = finegrained_from(vecs, pairs)
    topic = finegrained_from(vecs, pairs, transform=lambda v: blocks(v)[0])
    return {"split": split, "space": full, "topic_block_only": topic}


LONG_PARAGRAPHS = [
    "The regional transport authority published its annual report on Monday. Ridership on buses grew steadily over "
    "the year, while the tram network recovered more slowly after the long closure of the central line.",
    "Officials said the new ticketing system reduced queues at the main stations. Passengers can now pay with a "
    "phone or a bank card, and paper tickets will be withdrawn gradually over the next two years.",
    "The report also describes the condition of the fleet. Many of the older diesel buses will be replaced by "
    "electric vehicles, and three depots are being rebuilt with charging equipment and new workshops.",
    "Cycling lanes were extended along the river and near the university. Surveys of residents show broad support, "
    "although shop owners on the market street complained about the loss of parking spaces.",
    "Funding remains the main concern. The authority depends on grants from the national government, and the "
    "budget for maintenance has not kept pace with the cost of repairs to bridges and tunnels.",
    "Staff shortages affected services in the winter. The authority recruited new drivers and engineers, offering "
    "training programmes for young people from the surrounding towns.",
]


def long_document(n_words: int = 3000) -> str:
    paras, k = [], 0
    while sum(len(p.split()) for p in paras) < n_words:
        paras.append(LONG_PARAGRAPHS[k % len(LONG_PARAGRAPHS)] + f" (Section {k + 1}.)")
        k += 1
    return "\n\n".join(paras)


def long_document_locality() -> dict:
    """One-word, one-sentence and one-paragraph edits near the end of a ~3,000-word article."""
    from . import reps_of_bytes, text_vector
    base = long_document()
    words = base.split(" ")
    one_word = " ".join(words[:-3] + ["cancelled"] + words[-2:])
    sentence = base + "\n\nThe authority announced that the night bus service will be cancelled from next month."
    para = base.replace(LONG_PARAGRAPHS[3] + " (Section", "A severe flood closed the river tunnel and the city "
                        "declared an emergency, evacuating hundreds of homes near the old harbour." + " (Section", 1)
    datas = [d.encode() for d in (base, one_word, sentence, para)]
    rs = reps_of_bytes(datas, ["text"] * 4)
    q = text_vector(None, "flood emergency evacuation of homes near the harbour")
    out = {"words": len(words), "chunks": int(rs[0][1].shape[0])}
    for name, r in zip(("one_word", "added_sentence", "replaced_paragraph"), rs[1:]):
        b, e = rs[0], r
        k = min(len(b[1]), len(e[1]))
        cd = [_cosd(b[1][i], e[1][i]) for i in range(k)]
        out[name] = {"item_cosine": round(1 - _cosd(b[0], e[0]), 6), "max_chunk_distance": round(max(cd), 4),
                     "chunks_changed": int(sum(1 for d in cd if d > 1e-4)) + abs(len(b[1]) - len(e[1])),
                     "query_best_chunk_cos_before": round(float(np.max(b[1] @ q)), 4),
                     "query_best_chunk_cos_after": round(float(np.max(e[1] @ q)), 4)}
    return out


SHORTCUT_TOPICS = {
    "football": ("The home side pressed high from the first whistle, forcing mistakes in midfield. Their striker "
                 "scored twice before half time, and the coach praised the defence for keeping the visitors away "
                 "from goal. Supporters left the stadium singing after a win that lifts the club into third place "
                 "in the league table."),
    "cooking": ("Roast the tomatoes with garlic and olive oil until they soften, then blend them with a little stock. "
                "Season the soup with salt, pepper and a pinch of sugar, and finish it with fresh basil. Served with "
                "warm bread, it makes a simple dinner that keeps well in the fridge for three days."),
    "finance": ("The central bank kept interest rates unchanged but signalled that cuts could come later in the year "
                "if inflation keeps falling. Mortgage lenders expect demand to recover slowly, and analysts warned "
                "that savers should compare accounts because many banks have not passed on higher rates."),
}
STUFFING = {"football": "football soccer goal striker league football match goal football club stadium coach",
            "cooking": "recipe cooking soup dinner kitchen recipe cooking tomato basil recipe food cooking",
            "finance": "finance bank interest rates money savings mortgage finance bank inflation money"}


def shortcut_resistance() -> dict:
    """Features and on-topic affinity of informative text vs keyword stuffing, word salad and meaningless tokens."""
    from . import features_of_bytes, text_vector, text_vectors
    rng = np.random.default_rng(0)
    rows = {}
    for topic, article in SHORTCUT_TOPICS.items():
        words = article.split()
        n = len(words)
        stuffing = " ".join((STUFFING[topic].split() * 20)[:n])
        salad = " ".join(rng.permutation(words))
        letters = "bcdfghjklmnpqrstvwxz"
        noise = " ".join("".join(rng.choice(list(letters + "aeiou"), size=int(rng.integers(3, 9)))) for _ in range(n))
        variants = {"informative": article, "keyword_stuffing": stuffing, "word_salad": salad, "meaningless": noise}
        names = list(variants)
        V = text_vectors(None, [variants[k] for k in names])
        F = features_of_bytes([variants[k].encode() for k in names], ["text"] * len(names))
        interest = text_vector(None, f"news and stories about {topic}")
        rows[topic] = {k: {"affinity": round(float(V[i] @ interest), 4),
                           "lm_novel_logprob_per_token": F[i]["lm_novel_logprob_per_token"],
                           "lm_logprob_per_token": F[i]["lm_logprob_per_token"],
                           "language_validity": F[i]["language_validity"], "compression_ratio": F[i]["compression_ratio"],
                           "repeated_4gram_share": F[i]["repeated_4gram_share"]} for i, k in enumerate(names)}
    worse = {k: all(rows[t]["informative"]["lm_novel_logprob_per_token"] > rows[t][k]["lm_novel_logprob_per_token"]
                    for t in rows) for k in ("keyword_stuffing", "word_salad", "meaningless")}
    return {"topics": rows, "informative_more_coherent_than": worse}
