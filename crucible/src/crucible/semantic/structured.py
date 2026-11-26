"""Structured representations of software and products (MASTER_SPEC 4.3: "structured representations for software
and products"; SP-4.3).

Both are computed from what a buyer or user can observe - the artifact's bytes, a product's public page - never from
a quality, relevance or popularity score its creator declares:

* `software(world_or_ctx, ref)` - a code artifact (a repository tree, a zip/tar archive or a JSON file map) in the
  shared space: its vector (the declaration summary through the text path), and structural facts read from the
  source - files, lines, functions, classes, tests present, manifest present, comment ratio, HTTP endpoints,
  imports, share of Python files that parse, branches per function, LM naturalness of the source
  (`crucible.semantic.space.compute.code_reps`).
* `product(world_or_ctx, product_id)` - a launched product from its public page (`prod.view`, what anyone sees):
  the space vector of its name and description, its kind and delivery type, its price structure (cheapest one-time
  price, cheapest licence, number of offers and plans), the number of releases and the star ratings of paying
  customers (count, mean) - ratings are other participants' recorded judgements, not the creator's.

`fields(rep)` lists the structured field names of a representation; nothing here is an in-world tool.
"""
from __future__ import annotations

import math

import numpy as np

SOFTWARE_FIELDS = ("files", "lines", "functions", "classes", "tests_present", "manifest_present", "comment_ratio",
                   "endpoints", "imports", "python_parse_share", "branches_per_function", "lm_logprob_per_token")
PRODUCT_KINDS = ("data", "devtool", "education", "industrial", "infrastructure", "media", "other", "saas")
DELIVERY = ("service", "repo", "content", "goods")


def _world(x):
    return x.world if hasattr(x, "emit") and hasattr(x, "world") else x


def software(world_or_ctx, ref: str) -> dict:
    """{vector, fields, modality} of a code artifact (see the module docstring)."""
    from . import space
    vec, _, feats = space.reps(world_or_ctx, [ref], modalities=["code"])[0]
    return {"modality": "code", "vector": np.asarray(vec, np.float32),
            "fields": {k: feats.get(k) for k in SOFTWARE_FIELDS}}


def software_bytes(data: bytes) -> dict:
    """`software` for code bytes outside a world (tools, tests, analysis)."""
    from . import space
    vec, _, feats = space.reps(None, [space.ref_of(data)], modalities=["code"], data=[bytes(data)])[0]
    return {"modality": "code", "vector": np.asarray(vec, np.float32),
            "fields": {k: feats.get(k) for k in SOFTWARE_FIELDS}}


def product(world_or_ctx, product_id: str) -> dict:
    """{vector, fields, kind} of a product from its public page (see the module docstring)."""
    from . import space
    world = _world(world_or_ctx)
    mod = world.modules.get("products")
    if mod is None:
        raise ValueError("the world has no products module")
    st = world.state
    p = mod.product(st, product_id)
    pub = mod._public(product_id, p)
    text = f"{p.get('name', '')}\n\n{p.get('description', '')}".strip()
    vec = space.text_vector(world_or_ctx, text) if text else np.zeros(space.DIM, np.float32)
    offers = list(pub.get("offers") or [])
    plans = list(pub.get("plans") or [])
    one = [int(o["price_cents"]) for o in offers if o.get("kind") == "one_time" and o.get("price_cents")]
    lic = [int(o["price_cents"]) for o in offers if o.get("kind") == "license" and o.get("price_cents")]
    ratings = [int(r["stars"]) for r in st.find("prod_ratings", "product", product_id)]
    kind = p.get("kind", "other")
    delivery = ((p.get("delivery") or {}).get("type")) or "none"
    fields = {"kind": kind, "delivery": delivery, "offers": len(offers), "plans": len(plans),
              "log_min_one_time_price": math.log1p(min(one) / 100.0) if one else None,
              "log_min_licence_price": math.log1p(min(lic) / 100.0) if lic else None,
              "releases": len(st.find("prod_releases", "product", product_id)),
              "ratings": len(ratings), "mean_stars": float(np.mean(ratings)) if ratings else None,
              "description_words": len(text.split())}
    onehot = np.array([1.0 if kind == k else 0.0 for k in PRODUCT_KINDS] +
                      [1.0 if delivery == d else 0.0 for d in DELIVERY], np.float32)
    return {"modality": "product", "vector": np.asarray(vec, np.float32), "fields": fields, "categorical": onehot}


def fields(rep: dict) -> list[str]:
    return sorted(k for k, v in (rep.get("fields") or {}).items() if v is not None)
