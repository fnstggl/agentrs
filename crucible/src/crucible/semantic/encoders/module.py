"""Kernel module `embed` (host `embed`): the world's record of content embeddings.

`encode(world, ref, modality, ctx=ctx)` stores every vector it records in this world in table `embeddings` (one
`embed.recorded` event per batch, internal visibility), keyed by `cache_key(ref, modality)`. Later calls in the same
world - and replays, regenerations, snapshots and forks of it - read the recorded vector instead of running a model,
so a world's representations never drift with hardware or library versions.

Tools (for every actor):

* `embed.encoders {}` - the encoders this world uses: id, version, dimension and preprocessing hash per modality.
* `embed.compare {a, b, modality}` - cosine similarity of two artifacts the caller may read (needs the substrate
  artifact store). It returns only a similarity of content the caller can already see; no quality, popularity or
  evaluator data exists in or is derived from this module.
"""
from __future__ import annotations

import base64
import hashlib

import numpy as np

from ...core import schema as S
from ...core.errors import InvalidAction, NotFound
from ...core.kernel import Module, applier, endpoint, register_module
from . import MODALITIES, TABLE, EncodingError, cache_key, encode, encoder, similarity

REF_PATTERN = "^sha256:[0-9a-f]{64}$"


@register_module
class Embeddings(Module):
    name = "embed"
    host = "embed"
    version = "1"

    def install(self, world):
        world.state.declare_table(TABLE, index=("ref",))

    @applier("embed.recorded")
    def ap_recorded(self, ev):
        st = self.world.state
        for it in ev.payload["items"]:
            if st.has(TABLE, it["key"]):
                continue
            st.put(TABLE, it["key"], {"ref": it["ref"], "modality": it["modality"], "encoder": it["encoder"],
                                     "version": it["version"], "prep": it["prep"], "dim": int(it["dim"]),
                                     "sha256": it["sha256"], "vec": it["vec"], "t": ev.t})

    @endpoint("encoders", S.obj(), doc="The content encoders of this world: id, version, vector dimension and "
              "preprocessing hash for each modality (text, image, audio, video, code).", cost_s=0, read_only=True)
    def ep_encoders(self, ctx):
        out = {}
        for m in MODALITIES:
            e = encoder(m)
            out[m] = {"id": e.id, "version": e.version(), "dim": int(e.dim), "preprocessing_hash": e.preprocessing_hash()}
        return {"encoders": out}      # nothing about what others encoded (the table is world-internal)

    @endpoint("compare", S.obj({"a": S.s_str("artifact reference sha256:<hex>", pattern=REF_PATTERN),
                                "b": S.s_str("artifact reference sha256:<hex>", pattern=REF_PATTERN),
                                "modality": S.s_enum(list(MODALITIES), "how to read both artifacts")},
                               ["a", "b", "modality"]),
              doc="Cosine similarity (-1..1) of the content of two artifacts you can read, as seen by the world's "
                  "encoder for the modality.", cost_s=2)
    def ep_compare(self, ctx, a, b, modality):
        if "artifacts" not in ctx.world.modules:
            raise NotFound("this world has no artifact store")
        from ...substrate import artifacts as art
        for r in (a, b):
            if not art.readable(ctx.world, ctx.actor, r):     # unreadable looks like missing (no existence oracle)
                raise NotFound(f"no artifact {r}")
        try:
            va = encode(ctx.world, a, modality, ctx=ctx)
            vb = encode(ctx.world, b, modality, ctx=ctx)
        except EncodingError as e:
            raise InvalidAction(str(e)[:300]) from None
        return {"cosine": round(similarity(va, vb), 6), "modality": modality}

    def check_invariants(self, world):
        errs = []
        for key, rec in world.state.scan(TABLE):
            raw = base64.b64decode(rec["vec"])
            if hashlib.sha256(raw).hexdigest() != rec["sha256"]:
                errs.append(f"embedding {key[:60]} does not match its digest")
            if len(raw) != 4 * int(rec["dim"]):
                errs.append(f"embedding {key[:60]} has the wrong size")
            elif not np.all(np.isfinite(np.frombuffer(raw, dtype="<f4"))):
                errs.append(f"embedding {key[:60]} is not finite")
            try:
                if cache_key(rec["ref"], rec["modality"]).split("|")[0] != key.split("|")[0]:
                    errs.append(f"embedding {key[:60]} is filed under the wrong artifact")
            except EncodingError:
                errs.append(f"embedding {key[:60]} has a bad reference")
        return errs
