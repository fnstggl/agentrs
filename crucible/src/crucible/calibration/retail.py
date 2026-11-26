"""UCI Online Retail II: acquisition, compact extraction and the purchase / retention / price-response tables
(workstream E, E4 plan). Offline fitting code only; world code never imports it.

The dataset (Daqing Chen, UCI Machine Learning Repository id 502, doi:10.24432/C5CG6D, licence CC BY 4.0) holds every
invoice line of a UK-based non-store online gift retailer, 2009-12-01 to 2011-12-09; many customers are wholesalers.
`acquire_dataset` downloads the official UCI archive, checks its SHA-256, parses the two worksheets of the xlsx with
the standard library (no spreadsheet package is installed) and keeps one compact NumPy table:

    t         int64    seconds since 2009-12-01T00:00 (local time of the retailer, from the Excel serial date)
    invoice   int64    invoice number without the cancellation prefix
    cancel    uint8    1 if the invoice is a cancellation (number starting with "C")
    product   int32    index into the product table (StockCode)
    quantity  int32    units (negative on cancellations and adjustments)
    price     float64  unit price, GBP
    customer  int64    keyed pseudonym of the customer id (-1 when the line has no customer id)
    country   int16    index into the country table

Privacy: the published customer ids are already anonymous five-digit codes; they are still replaced by keyed
pseudonyms (blake2b-64 under a secret salt kept outside the repository, `gharchive.pseudonym_key`), so the derived
files hold no identifier of the source. Product codes and descriptions are the retailer's catalogue, not personal
data, and are kept in a side table (`products.json`) for product representations.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import os
import re
import zipfile

import numpy as np

from . import gharchive as G

DATASET = "online_retail_ii"
URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
ZIP_SHA256 = "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"
ZIP_BYTES = 45622418
EXTRACTOR_VERSION = "1"
T0 = _dt.datetime(2009, 12, 1)
EXCEL_EPOCH = _dt.datetime(1899, 12, 30)
COLUMNS = (("t", np.int64), ("invoice", np.int64), ("cancel", np.uint8), ("product", np.int32),
           ("quantity", np.int32), ("price", np.float64), ("customer", np.int64), ("country", np.int16))
HEADER = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
SHEET_ROWS = {"xl/worksheets/sheet1.xml": 525461, "xl/worksheets/sheet2.xml": 541910}
_CELL = re.compile(rb'<c r="([A-Z]+)\d+"([^>]*?)(?:/>|>(.*?)</c>)', re.S)
_TYPE = re.compile(rb' t="(\w+)"')
_V = re.compile(rb"<v>([^<]*)</v>")
_ROW = re.compile(rb"<row [^>]*>(.*?)</row>", re.S)
_SI = re.compile(rb"<si>(.*?)</si>", re.S)
_T = re.compile(rb"<t[^>]*>([^<]*)</t>")


def dataset_dir(dest: str | None = None) -> str:
    return dest or os.path.join(G.data_root(), DATASET)


def _xml_text(b: bytes) -> str:
    s = b.decode("utf-8")
    return (s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&apos;", "'")
            .replace("&amp;", "&"))


def _shared_strings(x: zipfile.ZipFile) -> list[str]:
    data = x.read("xl/sharedStrings.xml")
    return ["".join(_xml_text(t) for t in _T.findall(si)) for si in _SI.findall(data)]


def parse_xlsx(xlsx_bytes: bytes) -> list[list]:
    """Every data row of both worksheets as [Invoice, StockCode, Description, Quantity, InvoiceDate (Excel serial),
    Price, Customer ID, Country] (strings or floats; None for an empty cell)."""
    x = zipfile.ZipFile(io.BytesIO(xlsx_bytes))
    ss = _shared_strings(x)
    cols = {c: i for i, c in enumerate("ABCDEFGH")}
    rows = []
    for sheet, n_expected in SHEET_ROWS.items():
        data = x.read(sheet)
        got = 0
        for k, m in enumerate(_ROW.finditer(data)):
            rec = [None] * 8
            for c, attrs, body in _CELL.findall(m.group(1)):
                j = cols.get(c.decode())
                if j is None or not body:
                    continue
                tm = _TYPE.search(attrs)
                typ = tm.group(1) if tm else b"n"
                if typ == b"inlineStr":
                    rec[j] = "".join(_xml_text(t) for t in _T.findall(body))
                    continue
                vm = _V.search(body)
                if vm is None:
                    continue
                v = vm.group(1)
                if typ == b"s":
                    rec[j] = ss[int(v)]
                elif typ == b"str":
                    rec[j] = _xml_text(v)
                else:
                    rec[j] = float(v) if v else None
            if k == 0:
                if rec != HEADER:
                    raise ValueError(f"{sheet}: header {rec} != {HEADER}")
                continue
            rows.append(rec)
            got += 1
        if got != n_expected:
            raise ValueError(f"{sheet}: {got} data rows, expected {n_expected}")
    return rows


def _code(v) -> str:
    if isinstance(v, float):
        return str(int(v)) if v == int(v) else repr(v)
    return str(v).strip()


def extract(rows: list[list], key: bytes) -> tuple[dict[str, np.ndarray], dict]:
    """The compact table and the side tables (products with their most frequent description, countries)."""
    n = len(rows)
    inv = np.zeros(n, np.int64); cancel = np.zeros(n, np.uint8); prod = np.zeros(n, np.int32)
    qty = np.zeros(n, np.int32); t = np.zeros(n, np.int64); price = np.zeros(n, np.float64)
    cust_raw = np.full(n, -1, np.int64); ctry = np.zeros(n, np.int16)
    pidx: dict[str, int] = {}; cidx: dict[str, int] = {}; desc: dict[int, dict[str, int]] = {}
    t0 = (T0 - EXCEL_EPOCH).total_seconds()
    for i, (iv, sc, de, q, d, p, cu, co) in enumerate(rows):
        s = _code(iv)
        cancel[i] = s[:1].upper() == "C"
        digits = re.sub(r"\D", "", s)
        inv[i] = int(digits) if digits else -1
        code = _code(sc)
        j = pidx.setdefault(code, len(pidx))
        prod[i] = j
        if de:
            dd = desc.setdefault(j, {})
            dd[str(de)] = dd.get(str(de), 0) + 1
        qty[i] = int(q)
        t[i] = int(round(float(d) * 86400.0 - t0))
        price[i] = float(p)
        if cu is not None:
            cust_raw[i] = int(cu)
        ctry[i] = cidx.setdefault(str(co), len(cidx))
    has = cust_raw >= 0
    cust = np.full(n, -1, np.int64)
    cust[has] = G.pseudonyms(cust_raw[has], key)
    del cust_raw
    arrays = {"t": t, "invoice": inv, "cancel": cancel, "product": prod, "quantity": qty, "price": price,
              "customer": cust, "country": ctry}
    products = [{"code": c, "description": max(desc.get(j, {"": 0}).items(), key=lambda kv: (kv[1], kv[0]))[0]}
                for c, j in sorted(pidx.items(), key=lambda kv: kv[1])]
    return {k: np.asarray(arrays[k], dtype=dt) for k, dt in COLUMNS}, {"products": products,
                                                                         "countries": list(cidx)}


def content_digest(arrays: dict[str, np.ndarray]) -> str:
    """SHA-256 over the columns (customer in canonical first-appearance form, so salt-independent)."""
    h = hashlib.sha256()
    for name, dtype in COLUMNS:
        a = np.ascontiguousarray(arrays[name], dtype=dtype)
        if name == "customer":
            c = np.full(a.shape, -1, np.int64)
            m = a >= 0
            c[m] = G.canonical_ids(a[m])
            a = c
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def table_path(dest: str | None = None) -> str:
    return os.path.join(dataset_dir(dest), "extracted", f"online_retail_ii_v{EXTRACTOR_VERSION}.npz")


# ---------------------------------------------------------------- registry hooks (crucible.calibration.datasets)
def acquire_dataset(entry: dict, dest: str | None = None) -> str:
    from .datasets import _download, sha256_file
    d = dataset_dir(dest)
    z = os.path.join(d, "download", "online_retail_ii.zip")
    f = entry["files"][0]
    if not (os.path.exists(z) and sha256_file(z) == f["sha256"]):
        _download(f["url"], z, f["sha256"])
    out = table_path(d)
    if not os.path.exists(out):
        with zipfile.ZipFile(z) as zz:
            xlsx = zz.read("online_retail_II.xlsx")
        arrays, side = extract(parse_xlsx(xlsx), G.pseudonym_key(d))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out + ".tmp.npz", **arrays)
        os.replace(out + ".tmp.npz", out)
        with open(os.path.join(d, "extracted", "products.json"), "w", encoding="utf-8") as fh:
            json.dump(side, fh)
        with open(os.path.join(d, "extracted", "extract.json"), "w") as fh:
            json.dump({"extractor_version": EXTRACTOR_VERSION, "rows": int(arrays["t"].size),
                       "extract_digest": content_digest(arrays), "zip_sha256": f["sha256"]}, fh, indent=1)
    return d


def verify_dataset(entry: dict, dest: str | None = None) -> dict[str, str]:
    from .datasets import sha256_file
    d = dataset_dir(dest)
    f = entry["files"][0]
    z = os.path.join(d, f["path"])
    out = {f["path"]: "missing" if not os.path.exists(z) else ("ok" if sha256_file(z) == f["sha256"] else "mismatch")}
    p = table_path(d)
    if not os.path.exists(p):
        out["extracted"] = "missing"
    else:
        with np.load(p) as zz:
            dig = content_digest({k: zz[k] for k in zz.files})
        out["extracted"] = "ok" if dig == entry.get("extract_digest") else "mismatch"
    return out


def validate_dataset(entry: dict, dest: str | None = None) -> list[str]:
    p = table_path(dataset_dir(dest))
    if not os.path.exists(p):
        return ["extracted table missing"]
    errs = []
    with np.load(p) as z:
        got = {k: str(z[k].dtype) for k in z.files}
        n = {int(z[k].shape[0]) for k in z.files}
    want = {k: str(np.dtype(dt)) for k, dt in COLUMNS}
    if got != want:
        errs.append(f"columns {got} differ from {want}")
    if n != {int(entry.get("rows", -1))}:
        errs.append(f"{sorted(n)} rows, manifest records {entry.get('rows')}")
    return errs


def load(dest: str | None = None) -> dict:
    """The verified compact table plus the side tables."""
    from .datasets import acquire
    d = acquire(DATASET, dest)
    with np.load(table_path(d)) as z:
        out = {k: z[k] for k in z.files}
    with open(os.path.join(d, "extracted", "products.json"), encoding="utf-8") as fh:
        out.update(json.load(fh))
    return out
