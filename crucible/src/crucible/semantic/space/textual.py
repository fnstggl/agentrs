"""Text path of the shared space: chunked MiniLM topic vectors, lexical cue features and the quantity block.

* **Chunks.** The text (HTML reduced to its visible text, NFC, collapsed whitespace: `encoders.text.to_text`) is cut
  into windows of `CHUNK_WORDPIECES` MiniLM word pieces (character spans kept), at most `MAX_CHUNKS` windows spread
  evenly over the document. Each chunk is mean-pooled by all-MiniLM-L6-v2 and L2-normalised. The document topic block
  is the token-weighted mean of its chunks, L2-normalised. Short windows mean a local edit moves one chunk vector a
  lot (and the document vector measurably), instead of vanishing in one 256-piece window mean.
* **Cues** (per chunk, word level): share of negation cues (not, no, never, n't, without, cannot, ...), of hedges
  (may, might, possibly, reportedly, ...), and of sentences that are questions.
* **Quantity block** ([`QUANT_DIM`]): every number in the text is parsed (digits with separators and decimals,
  number words such as "thirty" or "three thousand", scale words k/m/million/billion) and typed by its context:
  money (currency sign or word), percent, year (1800-2100 without unit), or count; each number adds Gaussian radial
  basis functions of its log10 magnitude (years: of the year) to its type's sub-block. Months, weekdays and clock
  times add points on a circle. The block is L2-normalised (zero when the text has no quantities). So "three ... in
  April" and "thirty ... in December" differ, while a paraphrase with the same numbers does not.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

CHUNK_WORDPIECES = 128
MAX_CHUNKS = 32
RBF_CENTERS = tuple(float(c) for c in range(0, 7))          # log10 magnitude 1 .. 1e6
RBF_SIGMA = 0.45
YEAR_CENTERS = (1900.0, 1925.0, 1950.0, 1975.0, 2000.0, 2015.0, 2030.0)
YEAR_SIGMA = 9.0
QTYPES = ("money", "percent", "year", "count")
QUANT_DIM = len(QTYPES) * len(RBF_CENTERS) + 6              # + month, weekday, hour circles
MAX_NUMBERS = 96

NEGATION = re.compile(r"\b(?:not|no|never|none|nobody|nothing|neither|nor|nowhere|without|cannot|lack|lacks|"
                      r"lacking|fails?|failed|refuses?|refused|denies|denied)\b|n't\b", re.I)
HEDGE = re.compile(r"\b(?:may|might|could|possibly|perhaps|maybe|reportedly|allegedly|apparently|likely|unlikely|"
                   r"suggests?|uncertain|unclear|rumou?red?)\b", re.I)
WORD = re.compile(r"[A-Za-zÀ-ɏ]+(?:'[A-Za-z]+)?")
SENT = re.compile(r"[^.!?]+[.!?]*")

UNITS_MONEY = r"(?:dollars?|usd|euros?|eur|pounds?|gbp|yen|yuan|cents?|bucks)"
SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "mn": 1e6, "bn": 1e9, "billion": 1e9, "b": 1e9,
         "trillion": 1e12, "hundred": 1e2, "dozen": 12.0}
SMALL = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
MONTHS = {m: i for i, m in enumerate("january february march april may june july august september october "
                                     "november december".split())}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items()) if m != "may"})
DAYS = {d: i for i, d in enumerate("monday tuesday wednesday thursday friday saturday sunday".split())}
DAYPARTS = {"morning": 8.0, "noon": 12.0, "midday": 12.0, "afternoon": 15.0, "evening": 19.0, "night": 22.0,
            "midnight": 0.0}

NUM_TOKEN = re.compile(
    r"(?P<cur>[$€£¥])?\s?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s?(?P<scale>k|m|mn|bn|b|thousand|million|"
    r"billion|trillion)?\b(?:\s?(?P<pct>%|percent|per cent|pct)|\s?(?P<money>" + UNITS_MONEY + r")\b|"
    r"\s?(?P<ampm>a\.?m\.?|p\.?m\.?|o'clock)|(?P<clock>:\d{2}))?", re.I)
WORD_NUM = re.compile(r"\b(?:(?:" + "|".join(list(SMALL) + list(TENS)) + r")(?:[\s-](?:" + "|".join(list(SMALL)) +
                      r"))?)(?:\s(?:hundred|thousand|million|billion|dozen))*\b", re.I)


@dataclass
class Chunk:
    text: str
    start: int
    end: int
    n_pieces: int


@dataclass
class Quantities:
    values: list[tuple[str, float]] = field(default_factory=list)   # (type, value)
    months: list[int] = field(default_factory=list)
    weekdays: list[int] = field(default_factory=list)
    hours: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------- chunking
def chunks(text: str, tokenizer, size: int = CHUNK_WORDPIECES, max_chunks: int = MAX_CHUNKS) -> list[Chunk]:
    """Word-piece windows of `size` with their character spans (at most `max_chunks`, spread over the text)."""
    if not text:
        return [Chunk("", 0, 0, 0)]
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    offs = enc["offset_mapping"]
    n = len(offs)
    if n == 0:
        return [Chunk(text, 0, len(text), 0)]
    starts = list(range(0, n, size))
    if len(starts) > max_chunks:                     # evenly spread windows across the whole document
        pick = np.linspace(0, len(starts) - 1, max_chunks).round().astype(int)
        starts = [starts[int(i)] for i in pick]
    out = []
    for s in starts:
        e = min(n, s + size)
        a, b = offs[s][0], offs[e - 1][1]
        out.append(Chunk(text[a:b], a, b, e - s))
    return out


# ---------------------------------------------------------------------------------------------- cues
def cues(text: str) -> np.ndarray:
    """[negation, hedge, question] in [0, 1]: cue shares (per word, scaled) and the share of question sentences."""
    words = WORD.findall(text)
    nw = max(1, len(words))
    neg = len(NEGATION.findall(text))
    hed = len(HEDGE.findall(text))
    sents = [s for s in SENT.findall(text) if s.strip()]
    q = sum(1 for s in sents if s.strip().endswith("?")) / max(1, len(sents))
    return np.array([min(1.0, 8.0 * neg / nw), min(1.0, 8.0 * hed / nw), q], dtype=np.float32)


# ---------------------------------------------------------------------------------------------- quantities
def _word_number(s: str) -> float | None:
    total, cur = 0.0, 0.0
    seen = False
    for w in re.split(r"[\s-]+", s.lower()):
        if w in SMALL:
            cur += SMALL[w]
            seen = True
        elif w in TENS:
            cur += TENS[w]
            seen = True
        elif w == "hundred":
            cur = max(cur, 1.0) * 100.0
        elif w in ("thousand", "million", "billion"):
            total += max(cur, 1.0) * SCALE[w]
            cur = 0.0
        elif w == "dozen":
            cur = max(cur, 1.0) * 12.0
    return total + cur if seen else None


def quantities(text: str) -> Quantities:
    q = Quantities()
    low = text.lower()
    spans = []
    for m in NUM_TOKEN.finditer(text):
        if len(q.values) >= MAX_NUMBERS:
            break
        raw = m.group("num")
        try:
            v = float(raw.replace(",", ""))
        except ValueError:
            continue
        sc = (m.group("scale") or "").lower()
        if sc:
            v *= SCALE.get(sc, 1.0)
        spans.append((m.start(), m.end()))
        if m.group("ampm") or m.group("clock"):
            ap = (m.group("ampm") or "").lower()
            q.hours.append(v % 12.0 + 12.0 if ap.startswith("p") else v % 12.0 if ap.startswith("a") else v % 24.0)
            continue
        tail = low[m.end():m.end() + 20]
        if not (m.group("cur") or m.group("money") or m.group("pct")) and v <= 12 and \
                re.match(r"\s?(in the )?(morning|afternoon|evening)|\s?at night", tail):
            q.hours.append(v % 12.0 + (0.0 if "morning" in tail[:16] else 12.0))
            continue
        if m.group("cur") or m.group("money"):
            unit = (m.group("money") or "").lower()
            q.values.append(("money", v / 100.0 if unit.startswith("cent") else v))
        elif m.group("pct"):
            q.values.append(("percent", v))
        elif 1800 <= v <= 2100 and float(v).is_integer() and "," not in raw and not sc:
            q.values.append(("year", v))
        else:
            q.values.append(("count", v))
    for m in WORD_NUM.finditer(text):
        if any(a <= m.start() < b for a, b in spans) or len(q.values) >= MAX_NUMBERS:
            continue
        v = _word_number(m.group(0))
        if v is None:
            continue
        tail = low[m.end():m.end() + 16]
        if re.match(r"\s?(%|percent|per cent)", tail):
            q.values.append(("percent", v))
        elif re.match(r"\s?" + UNITS_MONEY, tail):
            q.values.append(("money", v))
        elif re.match(r"\s?o'clock", tail) or re.match(r"\s?(in the )?(morning|afternoon|evening|at night)", tail):
            q.hours.append(v % 12.0 + (12.0 if re.match(r"\s?(in the )?(afternoon|evening|at night)", tail) else 0.0))
        else:
            q.values.append(("count", v))
    for w in WORD.findall(low):
        if w in MONTHS and w != "may":
            q.months.append(MONTHS[w])
        elif w in DAYS:
            q.weekdays.append(DAYS[w])
        elif w in DAYPARTS and not q.hours:
            q.hours.append(DAYPARTS[w])
    return q


def quantity_block(text: str) -> np.ndarray:
    """[QUANT_DIM] float32: L2-normalised quantity features (zeros when the text states no quantity)."""
    q = quantities(text)
    out = np.zeros(QUANT_DIM, dtype=np.float64)
    k = len(RBF_CENTERS)
    for t, v in q.values:
        base = QTYPES.index(t) * k
        if t == "year":
            out[base:base + k] += np.exp(-0.5 * ((v - np.asarray(YEAR_CENTERS)) / YEAR_SIGMA) ** 2)
        else:
            lv = math.log10(max(abs(v), 1e-3) + 1.0)
            out[base:base + k] += np.exp(-0.5 * ((lv - np.asarray(RBF_CENTERS)) / RBF_SIGMA) ** 2)
    c = len(QTYPES) * k
    for i, period, items in ((c, 12.0, q.months), (c + 2, 7.0, q.weekdays), (c + 4, 24.0, q.hours)):
        for x in items:
            a = 2.0 * math.pi * float(x) / period
            out[i] += math.cos(a)
            out[i + 1] += math.sin(a)
    n = float(np.linalg.norm(out))
    return (out / n).astype(np.float32) if n > 1e-9 else out.astype(np.float32)


def count_quantities(text: str) -> int:
    q = quantities(text)
    return len(q.values) + len(q.months) + len(q.weekdays) + len(q.hours)
