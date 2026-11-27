"""Content-addressed, immutable artifact store (kernel module `artifacts`, host `artifact`).

An artifact is a byte string identified by its SHA-256: `REF = "sha256:<64 lowercase hex>"`. Storing the same bytes
twice gives the same REF. The world's *state* keeps only metadata (table `artifacts`: mime, size, owner, parents,
meta, kind, publication, time); the *bytes* are kept out of events and snapshots:

* in the world's store blob table under key `art/<hex>` when the world has a store,
* in this process's content-addressed memory pool when it has none,
* and in a bounded per-world cache (`world.runtime["artifacts"]`).

Every read verifies the SHA-256 of bytes that come from outside the world's own cache, so a damaged or substituted
blob is detected, never used. Replay, regeneration, snapshots and forks keep working because bytes are looked up by
hash: replay (with the world's store, or the process pool / any store this process wrote artifacts to) finds them,
regeneration re-creates them, and a fork reads its parent's. `export(world, dest_store)` copies every referenced blob
when a world moves to another store.

Permissions: an artifact is readable by its owner (and the owners/admins of an owning organisation), by principals it
was granted to, by holders (principals that stored identical bytes themselves), and by everyone once published
(`audience="public"`) or by an audience principal it was published to. An unreadable artifact looks exactly like a
missing one to in-world callers (`not_found`), so the tools are not an existence oracle.

Limits: `max_bytes` per artifact (32 MiB), `quota_bytes` stored per owner (512 MiB), both from the module config.
System batch blobs (`owner="sys"`, `kind="batch"`, written by `crucible.substrate.columns`) have their own bound
(`batch_max_bytes`, 1 GiB) and no quota.

System array blobs are not world state (D-78). A cohort decision's blob, a neural checkpoint or a recorded inference
output is named by the hash-chained event that uses it (its SHA-256 is in the REF the event carries); it gets no
`artifacts` row and no `artifact.stored` event. Before D-78 every one of them kept a metadata row in the state for
ever, so a snapshot of a 365-day world held 222,702 rows of blob metadata and retained snapshots grew
superlinearly. A world reaches a system blob only by a REF from its own record, and only from bytes of its own
lineage (`lineage(world)`: the spec hash, shared by the world, its forks, replays and regenerations): its cache, its
store, its read-only sources, process-pool bytes or registered stores that a world of the same lineage wrote. A
participant never reaches one (the tools serve artifacts with a row only). An optional storage fee (`fee_cents_per_mib`, default 0) is charged through
`bank` in the same transaction, so a payer that cannot pay leaves no trace of the put.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import re
import weakref
import zipfile
from collections import OrderedDict
from typing import Any, Sequence

import numpy as np

from ..core import schema as S
from ..core.canonical import canonical_bytes, to_jsonable
from ..core.errors import CapacityExceeded, Forbidden, InvalidAction, NotFound, ProtocolError
from ..core.kernel import SYSTEM, Ctx, Module, applier, endpoint, register_module
from . import bus

REF_PREFIX = "sha256:"
REF_RE = re.compile(r"sha256:[0-9a-f]{64}")
KIND_RE = r"[a-z][a-z0-9_.-]{0,31}"
MIME_RE = r"[a-zA-Z0-9][a-zA-Z0-9!#$&^_.+-]{0,63}/[a-zA-Z0-9][a-zA-Z0-9!#$&^_.+-]{0,63}"
MiB = 1 << 20
DEFAULTS = {"max_bytes": 32 * MiB, "quota_bytes": 512 * MiB, "batch_max_bytes": 1024 * MiB, "fee_cents_per_mib": 0,
            "max_meta_bytes": 4096, "max_parents": 64, "cache_bytes": 256 * MiB,
            # bound on a founding person plus every organisation it founded (0: quota_bytes). Organisations are free
            # to create, so a per-owner quota alone would not bound a participant (red-team finding, ws-O).
            "group_quota_bytes": 0}
NPZ_MIME = "application/x-npz"

# ---------------------------------------------------------------------------------------------------- byte storage
# Content-addressed bytes of worlds that have no store (the "memory store" of this process). Bytes are immutable and
# verified on insertion, so sharing them between worlds cannot change any world's state: a world can only reach
# bytes whose REF its own state records.
#
# 5.0 (D-21, audit A2 / W.5): the pool is reference-counted. Every store-less world holds a *lease* on the blobs it
# wrote or read (world.runtime["artifact_pool"]); closing the world (World.close) or garbage-collecting it releases
# the lease, and a blob no live world holds leaves the pool. A fork or replay of a store-less world takes its own
# lease on every blob its state references, so it keeps working after its parent is closed.
_POOL: dict[str, bytes] = {}
_POOL_REFS: dict[str, int] = {}
_POOL_LIN: dict[str, set] = {}        # pool blob -> lineages of the worlds that wrote it (system blobs, D-78)
# Stores that received artifact bytes in this process: a fork or replay without a store finds its parent's blobs
_SOURCES: list[weakref.ref] = []
_SOURCE_LIN: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()   # store -> lineages that wrote through it


def lineage(world) -> str:
    """The world's blob lineage (D-78): the hash of the spec its genesis ran from (`World.genesis_spec`, the value
    its genesis note records), shared by the world, its forks, snapshots, replays and regenerations. System array
    blobs are reachable only from bytes a world of the same lineage wrote (or the world's own store and read-only
    sources). It is never recomputed from the spec object: the first version cached `spec.hash()` at the first
    system-blob write, and a civilisation whose module annotated its nested config at genesis then wrote its blobs
    under one hash while every replay and fork computed another (none could read them)."""
    lin = getattr(world, "genesis_spec", None)
    return lin if lin is not None else world.spec.hash()


class _PoolLease:
    """The set of pool blobs one store-less world holds. Released exactly once (close or garbage collection)."""

    __slots__ = ("held", "__weakref__")

    def __init__(self):
        self.held: set[str] = set()

    def hold(self, h: str) -> None:
        if h not in self.held:
            self.held.add(h)
            _POOL_REFS[h] = _POOL_REFS.get(h, 0) + 1


def _release(held: set) -> None:
    for h in held:
        n = _POOL_REFS.get(h, 0) - 1
        if n <= 0:
            _POOL_REFS.pop(h, None)
            _POOL.pop(h, None)
            _POOL_LIN.pop(h, None)
        else:
            _POOL_REFS[h] = n
    held.clear()


def _lease(world) -> _PoolLease:
    lease = world.runtime.get("artifact_pool")
    if lease is None:
        lease = world.runtime["artifact_pool"] = _PoolLease()
        weakref.finalize(world, _release, lease.held)
    return lease


def release_pool(world) -> int:
    """Release this world's lease on the process pool (called by World.close through the artifacts module).
    Returns the number of blobs it held."""
    lease = world.runtime.pop("artifact_pool", None)
    if lease is None:
        return 0
    n = len(lease.held)
    _release(lease.held)
    return n


def pool_stats() -> dict:
    """{blobs, bytes} currently in this process's pool (worlds without a store)."""
    return {"blobs": len(_POOL), "bytes": sum(len(b) for b in _POOL.values())}


# Packed blobs (crucible.substrate.retention.pack): many system array blobs compressed into one chunk `pack/<hex>`;
# the blob's own key `art/<hex>` then holds a fixed-size pointer. Reads resolve the pointer and still verify the
# blob's own SHA-256, so packing can never change what a world reads.
PACK_MAGIC = b"CRUCIBLE-PACKREF/1\n"
PTR_LEN = len(PACK_MAGIC) + 64 + 16 + 16 + 1
_CHUNKS: "OrderedDict[str, bytes]" = OrderedDict()
_CHUNKS_CAP = 192 * (1 << 20)


def pack_pointer(pack_hex: str, off: int, length: int, codec: str) -> bytes:
    return PACK_MAGIC + pack_hex.encode() + b"%016x%016x" % (off, length) + codec.encode()


def _parse_pointer(data: bytes) -> tuple[str, int, int, str] | None:
    if len(data) != PTR_LEN or not data.startswith(PACK_MAGIC):
        return None
    body = data[len(PACK_MAGIC):]
    try:
        return body[:64].decode(), int(body[64:80], 16), int(body[80:96], 16), body[96:97].decode()
    except ValueError:
        return None


def _decompress(codec: str, data: bytes) -> bytes:
    if codec == "x":
        import lzma
        return lzma.decompress(data)
    if codec == "z":
        import zlib
        return zlib.decompress(data)
    raise ValueError(f"unknown pack codec {codec!r}")


def _chunk(store, pack_hex: str, codec: str) -> bytes | None:
    key = pack_hex          # content-addressed: the same key is the same bytes in any store
    raw = _CHUNKS.get(key)
    if raw is not None:
        _CHUNKS.move_to_end(key)
        return raw
    comp = store.get_blob(f"pack/{pack_hex}")
    if comp is None or sha256_hex(comp) != pack_hex:
        return None            # missing or damaged chunk: its members are unavailable, never wrong
    raw = _decompress(codec, comp)
    _CHUNKS[key] = raw
    size = sum(len(v) for v in _CHUNKS.values())
    while size > _CHUNKS_CAP and len(_CHUNKS) > 1:
        _, old = _CHUNKS.popitem(last=False)
        size -= len(old)
    return raw


def read_store_blob(store, h: str) -> bytes | None:
    """The verified bytes of blob `h` in `store` (resolving a packed pointer), or None."""
    data = store.get_blob(blob_key(h))
    if data is None:
        return None
    ptr = _parse_pointer(data) if len(data) == PTR_LEN else None
    if ptr is not None:
        pack_hex, off, length, codec = ptr
        raw = _chunk(store, pack_hex, codec)
        if raw is None:
            return None
        data = raw[off:off + length]
    return _verified(h, data)


def _register_source(store, lin: str | None = None) -> None:
    if lin is not None:
        try:
            _SOURCE_LIN.setdefault(store, set()).add(lin)
        except TypeError:
            pass
    for r in _SOURCES:
        if r() is store:
            return
    try:
        _SOURCES.append(weakref.ref(store))
    except TypeError:
        pass
    _SOURCES[:] = [r for r in _SOURCES if r() is not None]


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ref_of(data: bytes) -> str:
    return REF_PREFIX + sha256_hex(data)


def parse_ref(ref: str) -> str:
    """The 64-hex digest of a REF; raises InvalidAction for anything else."""
    if not isinstance(ref, str) or not REF_RE.fullmatch(ref):
        raise InvalidAction(f"not an artifact reference: {str(ref)[:80]!r} (expected sha256:<64 hex>)")
    return ref[len(REF_PREFIX):]


def blob_key(hexd: str) -> str:
    return f"art/{hexd}"


class _Cache:
    """Per-world byte cache (verified bytes only). Unbounded for store-less worlds (it shares the pool's objects);
    LRU-bounded by bytes when a store holds the authoritative copy."""

    def __init__(self, cap: int):
        self.cap = int(cap)
        self.d: OrderedDict[str, bytes] = OrderedDict()
        self.size = 0
        self.arrays: OrderedDict[str, dict] = OrderedDict()    # decoded array blobs (read-only arrays)
        self.arrays_size = 0
        self.persisted: set[str] = set()                         # blobs known to be in world.store

    def get(self, h: str) -> bytes | None:
        b = self.d.get(h)
        if b is not None:
            self.d.move_to_end(h)
        return b

    def put(self, h: str, data: bytes, bounded: bool) -> None:
        if h in self.d:
            self.d.move_to_end(h)
            return
        self.d[h] = data
        self.size += len(data)
        if bounded:
            while self.size > self.cap and len(self.d) > 1:
                _, old = self.d.popitem(last=False)
                self.size -= len(old)

    def get_arrays(self, h: str) -> dict | None:
        a = self.arrays.get(h)
        if a is not None:
            self.arrays.move_to_end(h)
        return a

    def put_arrays(self, h: str, arrays: dict) -> None:
        if h in self.arrays:
            return
        n = sum(a.nbytes for a in arrays.values())
        self.arrays[h] = arrays
        self.arrays_size += n
        while self.arrays_size > self.cap and len(self.arrays) > 1:
            _, old = self.arrays.popitem(last=False)
            self.arrays_size -= sum(a.nbytes for a in old.values())


def _cache(world) -> _Cache:
    c = world.runtime.get("artifacts")
    if c is None:
        mod = world.modules.get("artifacts")
        cap = (mod.cfg["cache_bytes"] if mod is not None else DEFAULTS["cache_bytes"])
        c = world.runtime["artifacts"] = _Cache(cap)
    return c


def _write_bytes(world, h: str, data: bytes, *, system: bool = False) -> None:
    """Make `data` (whose SHA-256 is `h`) retrievable for this world: cache plus store or process pool. `system`:
    a system array blob, which worlds of the same lineage may read without a row (D-78)."""
    c = _cache(world)
    store = world.store
    if store is not None:
        if h not in c.persisted:
            if store.get_blob(blob_key(h)) is None:
                store.put_blob(blob_key(h), data)
            c.persisted.add(h)
            _register_source(store, lineage(world) if system else None)
        elif system:
            _register_source(store, lineage(world))
        c.put(h, data, bounded=True)
    else:
        data = _POOL.setdefault(h, data)
        _lease(world).hold(h)
        if system:
            _POOL_LIN.setdefault(h, set()).add(lineage(world))
        c.put(h, data, bounded=False)


def _verified(h: str, data: bytes | None) -> bytes | None:
    if data is None:
        return None
    if sha256_hex(data) != h:
        return None     # damaged or substituted blob: never used
    return bytes(data)


def _read_bytes(world, h: str, *, claimed: bool = True) -> bytes | None:
    """Verified bytes of blob `h` for `world`, or None. `claimed`: the world's state records an artifact with these
    bytes, so any copy in this process may serve (all are hash-verified). Unclaimed (a system array blob, D-78): only
    the world's cache, store and read-only sources, and system blobs that its lineage wrote to the process pool or
    to a registered store."""
    c = _cache(world)
    b = c.get(h)
    if b is not None:
        return b
    found = None
    lin = None if claimed else lineage(world)
    if world.store is not None:
        found = read_store_blob(world.store, h)
        if found is not None:
            c.persisted.add(h)
    if found is None:
        found = _POOL.get(h)
        if found is not None and not claimed and lin not in _POOL_LIN.get(h, ()):
            found = None
        if found is not None and world.store is None:
            _lease(world).hold(h)
    if found is None:
        for s in world.runtime.get("blob_sources", ()):     # read-only sources (World.regenerate(store=...))
            found = read_store_blob(s, h)
            if found is not None:
                break
    if found is None:
        for r in list(_SOURCES):
            s = r()
            if s is None or s is world.store:
                continue
            if not claimed and lin not in _SOURCE_LIN.get(s, ()):
                continue
            try:
                found = read_store_blob(s, h)
            except Exception:   # a closed store cannot serve: try the next one
                found = None
            if found is not None:
                break
    if found is not None:
        c.put(h, found, bounded=world.store is not None)
    return found


# ---------------------------------------------------------------------------------------------------- npz (no pickle)
def npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """Deterministic, uncompressed .npz bytes (fixed zip timestamps, sorted members, no pickled objects): the same
    arrays always give the same bytes and therefore the same REF."""
    if not arrays:
        raise InvalidAction("an array blob needs at least one array")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for name in sorted(arrays):
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", name):
                raise InvalidAction(f"bad array name {name!r}")
            a = np.ascontiguousarray(arrays[name])
            if a.dtype.hasobject:
                raise InvalidAction(f"array {name!r} has object dtype (pickle is not allowed)")
            zi = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_STORED
            zi.external_attr = 0o644 << 16
            with zf.open(zi, "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, a, allow_pickle=False)
    return buf.getvalue()


def npz_arrays(data: bytes) -> dict[str, np.ndarray]:
    """Decode npz bytes into read-only arrays; refuses pickled objects and compressed members."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            out = {}
            for zi in zf.infolist():
                if zi.compress_type != zipfile.ZIP_STORED or not zi.filename.endswith(".npy"):
                    raise InvalidAction(f"array blob member {zi.filename!r} is not a stored .npy")
                with zf.open(zi) as fh:
                    a = np.lib.format.read_array(fh, allow_pickle=False)
                a.flags.writeable = False
                out[zi.filename[:-4]] = a
            return out
    except (zipfile.BadZipFile, ValueError, OSError) as e:
        raise InvalidAction(f"not an array blob: {e}") from None


# ---------------------------------------------------------------------------------------------------- Python API
def _mod(world) -> "Artifacts":
    m = world.modules.get("artifacts")
    if m is None:
        raise ProtocolError("this world has no 'artifacts' module (add {'name': 'artifacts'} to its spec)")
    return m  # type: ignore[return-value]


def put(ctx: Ctx, data: bytes, mime: str, *, owner: str, parents: Sequence[str] = (), meta: dict | None = None,
        kind: str = "file") -> str:
    """Store `data` for `owner`; emits `artifact.stored` the first time `owner` stores these bytes (the artifact, or
    a holding of bytes another principal already stores) and returns the REF. Idempotent."""
    return _mod(ctx.world).store_bytes(ctx, data, mime, owner=owner, parents=parents, meta=meta, kind=kind)


def get(world, ref: str) -> bytes:
    """The bytes of an artifact of this world, SHA-256 verified: an artifact its state records, or a system array blob
    of its lineage (D-78). Raises NotFound if this world has no such artifact or its bytes are unavailable.
    World-side API: no permission check (tools check `readable`)."""
    h = parse_ref(ref)
    claimed = world.state.has("artifacts", ref)
    b = _read_bytes(world, h, claimed=claimed)
    if b is None:
        if not claimed:
            raise NotFound(f"no artifact {ref}: no state record, and the bytes are not available as a system blob "
                           f"of this world's lineage")
        raise NotFound(f"bytes of {ref} are not available in this world's store (export them with "
                       f"crucible.substrate.artifacts.export)")
    return b


def info(world, ref: str) -> dict:
    """{ref, mime, size, owner, parents, meta, kind, published, t} of an artifact of this world."""
    parse_ref(ref)
    a = world.state.get("artifacts", ref)
    if a is None:
        raise NotFound(f"no artifact {ref}")
    return {"ref": ref, "mime": a["mime"], "size": a["size"], "owner": a["owner"], "parents": list(a["parents"]),
            "meta": to_jsonable(a["meta"]), "kind": a["kind"], "published": bool(a["published"]), "t": a["t"]}


def exists(world, ref: str) -> bool:
    """Whether the world's state records an artifact with this REF (system array blobs are not artifacts, D-78)."""
    return isinstance(ref, str) and REF_RE.fullmatch(ref) is not None and world.state.has("artifacts", ref)


def publish(ctx: Ctx, ref: str, *, audience: str = "public") -> None:
    """Make an artifact readable by everyone (`audience="public"`) or by one principal (and the owners/admins of an
    organisation given as audience). Only the owner side (or the system) may publish. Emits `artifact.published`."""
    _mod(ctx.world).publish(ctx, ref, audience)


def readable(world, principal: str, ref: str) -> bool:
    """Owner (incl. owners/admins of an owning organisation), grantee, holder of identical bytes, or published (by
    the first owner or by any holder: each holder publishes and grants its own record)."""
    if principal == SYSTEM:
        return exists(world, ref)
    if not exists(world, ref):
        return False
    return _record_for(world.state, ref, world.principals_of(principal)) is not None


def _records(st, ref: str) -> list[tuple[str, dict]]:
    """(owner, record) of every record of these bytes: the first owner's artifact record, then each holder's."""
    out = [(st.tables["artifacts"][ref]["owner"], st.tables["artifacts"][ref])]
    for hid in sorted(st.find_ids("artifact_holders", "ref", ref)):
        h = st.tables["artifact_holders"][hid]
        out.append((h["holder"], h))
    return out


def grant_key(ref: str, grantee: str, record: str | None = None) -> str:
    """Key of a read grant: `ref|grantee` for the first owner's record, `ref|grantee|holder` for a holder's."""
    return f"{ref}|{grantee}" if not record else f"{ref}|{grantee}|{record}"


def _record_visible(st, ref: str, rec_owner: str, rec: dict, ps: set) -> bool:
    """Whether principals `ps` may see this one record without owning it: published, published to them, or
    granted to them by the record's owner side."""
    if rec.get("published") or any(x in ps for x in rec.get("audiences", ())):
        return True
    first = st.tables["artifacts"][ref]["owner"] == rec_owner
    return any(st.has("artifact_grants", grant_key(ref, p, None if first else rec_owner)) for p in ps)


def _record_for(st, ref: str, ps: set) -> tuple[str, dict] | None:
    """The record principals `ps` see of an artifact: their own (as owner or holder) first, then one that was
    published or granted to them; None when they may see none. Holders see only their own record, so storing bytes
    someone else already stores never reveals that person's record."""
    recs = _records(st, ref)
    for owner, rec in recs:
        if owner in ps:
            return owner, rec
    for owner, rec in recs:
        if _record_visible(st, ref, owner, rec, ps):
            return owner, rec
    return None


def put_array_blob(ctx: Ctx, arrays: dict[str, np.ndarray]) -> str:
    """Store named arrays as an npz blob (no pickle) owned by 'sys', kind 'batch'. Returns its REF."""
    data = npz_bytes(arrays)
    ref = put(ctx, data, NPZ_MIME, owner=SYSTEM, kind="batch")
    ro = {}
    for k, v in arrays.items():
        a = np.array(v, copy=True, order="C")
        a.flags.writeable = False
        ro[k] = a
    _cache(ctx.world).put_arrays(ref[len(REF_PREFIX):], ro)
    return ref


def get_array_blob(world, ref: str) -> dict[str, np.ndarray]:
    """The arrays of an npz blob of this world (read-only, SHA-256 verified): a system array blob named by the
    world's own record (cols.batch, checkpoints, recorded inference) or an npz artifact its state records."""
    h = parse_ref(ref)
    c = _cache(world)
    arrs = c.get_arrays(h)          # the per-world cache holds only arrays this world wrote or read verified
    if arrs is not None:
        return arrs
    arrs = npz_arrays(get(world, ref))
    c.put_arrays(h, arrs)
    return arrs


def export(world, dest_store) -> dict:
    """Copy the bytes of every artifact this world references into `dest_store` (when a world moves to another
    store): the artifacts its state records and its system array blobs (D-78: every `art/` blob of its store, or
    every pool blob it holds when it has none). Packed blobs are written in full. Returns {copied, present, bytes,
    missing: [refs whose bytes are unavailable]}."""
    copied = present = nbytes = 0
    missing = []
    refs = list(world.state.ids("artifacts"))
    have = set(refs)
    if world.store is not None and hasattr(world.store, "blob_sizes"):
        sys_hex = [k[4:] for k in world.store.blob_sizes("art/")]
    else:
        lease = world.runtime.get("artifact_pool")
        sys_hex = sorted(lease.held) if lease is not None else []
    refs += [REF_PREFIX + h for h in sys_hex if REF_PREFIX + h not in have]
    for ref in refs:
        h = ref[len(REF_PREFIX):]
        if dest_store.get_blob(blob_key(h)) is not None:
            present += 1
            continue
        b = _read_bytes(world, h, claimed=ref in have)
        if b is None:
            missing.append(ref)
            continue
        dest_store.put_blob(blob_key(h), b)
        copied += 1
        nbytes += len(b)
    _register_source(dest_store, lineage(world) if sys_hex else None)
    return {"copied": copied, "present": present, "bytes": nbytes, "missing": missing}


def usage(world, owner: str) -> dict:
    u = world.state.get("artifact_usage", owner)
    return {"owner": owner, "bytes": u["bytes"] if u else 0, "count": u["count"] if u else 0}


# ---------------------------------------------------------------------------------------------------- the module
_PUT_PARAMS = S.obj({
    "data_b64": S.s_str("the bytes, base64-encoded (standard alphabet, padded)", maxLength=4 * ((32 * MiB + 2) // 3)),
    "mime": S.s_str("media type, e.g. text/plain, image/png, video/mp4", pattern=MIME_RE, maxLength=129),
    "kind": S.s_str("what it is: file, text, image, audio, video, code, dataset, model, document, ...",
                    pattern=KIND_RE, maxLength=32),
    "parents": S.s_arr(S.s_str("REF of an artifact this one derives from", maxLength=71),
                       "provenance: artifacts this one was derived from", maxItems=64),
    "meta": S.s_any_obj("small JSON metadata (title, language, duration_s, ...)"),
    "owner": S.s_str("organisation to store it for (you must own or administer it); default: yourself",
                     maxLength=60),
}, ["data_b64", "mime"])
_REF = S.s_str("artifact reference sha256:<64 hex>", maxLength=71)

_EV_REF = {"type": "string", "pattern": REF_RE.pattern}
_EV_PRINCIPAL = {"type": "string", "minLength": 1, "maxLength": 120}
bus.event_type("artifact.stored", S.obj({
    "ref": _EV_REF, "mime": {"type": "string", "pattern": MIME_RE}, "size": {"type": "integer", "minimum": 0},
    "owner": _EV_PRINCIPAL, "parents": {"type": "array", "items": _EV_REF, "maxItems": 64},
    "meta": {"type": "object", "additionalProperties": True}, "kind": {"type": "string", "pattern": KIND_RE}},
    ["ref", "mime", "size", "owner", "parents", "meta", "kind"]), owner="artifacts",
    doc="a principal stored these bytes (metadata only; bytes live in the blob store). The first store creates the "
        "artifact; a later one by another principal makes it a holder of identical bytes - the event is the same, "
        "so a caller cannot learn whether somebody else already had them")
bus.event_type("artifact.published", S.obj({"ref": _EV_REF, "audience": _EV_PRINCIPAL, "holder": _EV_PRINCIPAL},
                                           ["ref", "audience"]),
               owner="artifacts", doc="an artifact became readable by an audience ('public' or a principal); with "
                                      "`holder`, that holder's own record of the bytes was published")
bus.event_type("artifact.granted", S.obj({"ref": _EV_REF, "grantee": _EV_PRINCIPAL, "by": _EV_PRINCIPAL,
                                          "record": _EV_PRINCIPAL},
                                         ["ref", "grantee", "by"]), owner="artifacts",
               doc="read access granted (with `record`: the owner of the holder record the grant shows)")
bus.event_type("artifact.revoked", S.obj({"ref": _EV_REF, "grantee": _EV_PRINCIPAL, "by": _EV_PRINCIPAL,
                                          "record": _EV_PRINCIPAL},
                                         ["ref", "grantee", "by"]), owner="artifacts", doc="read access revoked")


@register_module
class Artifacts(Module):
    name = "artifacts"
    host = "artifact"
    version = "1"

    def __init__(self, config=None):
        super().__init__(config)
        self.cfg = dict(DEFAULTS)
        for k, v in self.config.items():
            if k not in DEFAULTS:
                raise ProtocolError(f"artifacts: unknown config key {k!r}")
            self.cfg[k] = int(v)

    def install(self, world):
        st = world.state
        st.declare_table("artifacts", index=("owner", "kind"))
        st.declare_table("artifact_grants", index=("ref", "grantee"))
        st.declare_table("artifact_holders", index=("ref", "holder"))
        st.declare_table("artifact_usage")
        if self.cfg["fee_cents_per_mib"] and not any(m["name"] == "bank" for m in world.spec.modules):
            raise ProtocolError("artifacts: fee_cents_per_mib needs the bank module")
        bus.enforce(world, owner="artifacts")

    def on_load(self, world):
        world.runtime.pop("artifacts", None)
        if world.store is None:
            # a fork or replay of a store-less world: hold every pooled blob its state references, so closing the
            # world it came from cannot take them away
            lease = _lease(world)
            for ref in world.state.tables["artifacts"]:
                h = ref[len(REF_PREFIX):]
                if h in _POOL:
                    lease.hold(h)

    def on_fork(self, parent, child):
        """A store-less fork holds every pool blob its parent holds (its system array blobs have no rows to lease
        by, D-78), so closing the parent cannot take them away."""
        if child.store is None:
            held = parent.runtime.get("artifact_pool")
            if held is not None:
                lease = _lease(child)
                lin = lineage(parent)
                for h in sorted(held.held):
                    if lin in _POOL_LIN.get(h, ()):       # the parent's system blobs (artifacts: on_load)
                        lease.hold(h)

    def close(self):
        if self.world is not None:
            release_pool(self.world)

    # ------------------------------------------------------------------ internal API
    def _owner_side(self, ctx: Ctx, owner: str) -> bool:
        return ctx.actor == SYSTEM or owner in ctx.world.principals_of(ctx.actor)

    def store_bytes(self, ctx: Ctx, data: bytes, mime: str, *, owner: str, parents: Sequence[str] = (),
                    meta: dict | None = None, kind: str = "file") -> str:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise InvalidAction("artifact data must be bytes")
        data = bytes(data)
        if not isinstance(mime, str) or not re.fullmatch(MIME_RE, mime):
            raise InvalidAction(f"bad media type {str(mime)[:80]!r}")
        if not isinstance(kind, str) or not re.fullmatch(KIND_RE, kind):
            raise InvalidAction(f"bad artifact kind {str(kind)[:40]!r}")
        if not isinstance(owner, str) or not owner:
            raise InvalidAction("artifact owner required")
        if owner != SYSTEM and not ctx.state.has("principals", owner):
            raise NotFound(f"no principal {owner}")
        if kind == "batch" and owner != SYSTEM:
            raise InvalidAction("kind 'batch' is reserved for system array blobs")
        size = len(data)
        limit = self.cfg["batch_max_bytes"] if (owner == SYSTEM and kind == "batch") else self.cfg["max_bytes"]
        if size > limit:
            raise CapacityExceeded(f"artifact of {size} bytes exceeds the limit of {limit} bytes", size=size,
                                   limit=limit)
        meta = to_jsonable(meta or {})
        if not isinstance(meta, dict):
            raise InvalidAction("meta must be an object")
        try:
            mb = len(canonical_bytes(meta))
        except (TypeError, ValueError):
            raise InvalidAction("meta must be JSON (finite numbers)") from None
        if mb > self.cfg["max_meta_bytes"]:
            raise InvalidAction(f"meta of {mb} bytes exceeds {self.cfg['max_meta_bytes']}")
        parents = [str(p) for p in parents]
        if len(parents) > self.cfg["max_parents"]:
            raise InvalidAction(f"at most {self.cfg['max_parents']} parents")
        for p in parents:
            parse_ref(p)
            if not readable(ctx.world, owner, p):
                raise NotFound(f"no artifact {p}")        # unreadable parents look missing: no existence oracle
        parents = sorted(set(parents))
        h = sha256_hex(data)
        ref = REF_PREFIX + h
        if owner == SYSTEM and kind == "batch":
            # a system array blob (D-78): named by the event that uses it, never a row of the state
            if parents or meta:
                raise InvalidAction("system array blobs carry no parents or metadata")
            _write_bytes(ctx.world, h, data, system=True)
            return ref
        st = ctx.state
        rec = st.get("artifacts", ref)
        if rec is not None and (rec["owner"] == owner or owner == SYSTEM or
                                st.has("artifact_holders", f"{ref}|{owner}")):
            _write_bytes(ctx.world, h, data)       # this principal already stores these bytes: nothing to record
            return ref
        # new bytes, or bytes another principal already stores: the same event either way (the applier records the
        # artifact or a holder), the same quota and fee, so the caller learns nothing about other principals
        self._charge(ctx, owner, size)
        _write_bytes(ctx.world, h, data)
        ctx.emit("artifact.stored", {"ref": ref, "mime": mime, "size": size, "owner": owner, "parents": parents,
                                     "meta": meta, "kind": kind},
                 vis="internal" if owner == SYSTEM else (owner,))
        return ref

    def quota_group(self, st, owner: str) -> tuple[str, list[str]]:
        """(accountable person, [that person and every organisation it founded]) for storage made in `owner`'s
        name. An organisation's founder never changes, so leaving or handing over an organisation does not take
        its storage out of the founder's bound."""
        person = owner
        if "orgs" in st.tables and st.has("orgs", owner):
            person = st.tables["orgs"][owner].get("founder") or owner
        group = [person]
        if "orgs" in st.tables:
            group += sorted(o for o, rec in st.tables["orgs"].items() if rec.get("founder") == person and o != person)
        return person, group

    def _charge(self, ctx: Ctx, owner: str, size: int) -> None:
        if owner == SYSTEM:
            return
        used = usage(ctx.world, owner)["bytes"]
        if used + size > self.cfg["quota_bytes"]:
            raise CapacityExceeded(f"storage quota of {owner} exceeded ({used} + {size} > {self.cfg['quota_bytes']})",
                                   used=used, size=size, quota=self.cfg["quota_bytes"])
        gq = self.cfg["group_quota_bytes"] or self.cfg["quota_bytes"]
        person, group = self.quota_group(ctx.state, owner)
        if len(group) > 1:
            gused = sum(usage(ctx.world, p)["bytes"] for p in group)
            if gused + size > gq:
                if ctx.actor == person:
                    raise CapacityExceeded(f"storage quota of {person} and the organisations it founded exceeded "
                                           f"({gused} + {size} > {gq})", used=gused, size=size, quota=gq)
                # another member storing in the organisation's name learns nothing about the founder's other use
                raise CapacityExceeded(f"storage quota of {owner}'s founder group exceeded", size=size, quota=gq)
        fee = self.cfg["fee_cents_per_mib"]
        if fee and size:
            bank = ctx.world.modules["bank"]
            acct = bank.primary_account(ctx.state, owner)
            if acct is None:
                raise NotFound(f"{owner} has no open account to pay the storage fee")
            cents = -(-size * fee // MiB)
            bank.move(ctx, acct, "acct_fees", int(cents), "fee", "artifact storage")

    def own_record(self, ctx: Ctx, ref: str, what: str) -> tuple[str, dict, bool]:
        """(record owner, record, is the first owner's record) of the record `ctx.actor` owns: the artifact record
        if it is on the owner side, else its holder record. A holder of identical bytes publishes, grants and
        revokes its own record exactly as an owner does, so these tools never tell whether somebody else stored
        the bytes first (red-team finding, ws-O). NotFound if unreadable; Forbidden if readable but not owned."""
        a = ctx.state.get("artifacts", ref) if isinstance(ref, str) else None
        if a is None or not readable(ctx.world, ctx.actor, ref):
            raise NotFound(f"no artifact {ref}")
        if self._owner_side(ctx, a["owner"]):
            return a["owner"], a, True
        for p in sorted(ctx.world.principals_of(ctx.actor)):
            h = ctx.state.get("artifact_holders", f"{ref}|{p}")
            if h is not None:
                return p, h, False
        raise Forbidden(f"only the owner can {what} an artifact")

    def publish(self, ctx: Ctx, ref: str, audience: str) -> None:
        owner, rec, first = self.own_record(ctx, ref, "publish")
        if owner == SYSTEM and ctx.actor != SYSTEM:
            raise Forbidden("system artifacts are not published")
        if audience != "public" and not ctx.state.has("principals", audience):
            raise NotFound(f"no principal {audience}")
        if (audience == "public" and rec.get("published")) or audience in rec.get("audiences", ()):
            return
        payload = {"ref": ref, "audience": audience} if first else {"ref": ref, "audience": audience, "holder": owner}
        ctx.emit("artifact.published", payload,
                 vis="public" if audience == "public" else tuple(sorted({owner, audience} - {SYSTEM})))

    # ------------------------------------------------------------------ appliers
    @applier("artifact.stored")
    def ap_stored(self, ev):
        p = ev.payload
        st = self.world.state
        if st.has("artifacts", p["ref"]):
            st.insert("artifact_holders", f"{p['ref']}|{p['owner']}",
                      {"ref": p["ref"], "holder": p["owner"], "mime": p["mime"], "kind": p["kind"],
                       "parents": p["parents"], "meta": p["meta"], "t": ev.t})
        else:
            st.insert("artifacts", p["ref"], {"mime": p["mime"], "size": p["size"], "owner": p["owner"],
                                              "parents": p["parents"], "meta": p["meta"], "kind": p["kind"],
                                              "published": False, "audiences": [], "t": ev.t})
        self._use(p["owner"], p["size"])

    def _use(self, owner, size):
        if owner == SYSTEM:
            return
        st = self.world.state
        u = st.get("artifact_usage", owner)
        st.put("artifact_usage", owner, {"bytes": (u["bytes"] if u else 0) + size, "count": (u["count"] if u else 0) + 1})

    @applier("artifact.published")
    def ap_published(self, ev):
        p = ev.payload
        table, key = ("artifact_holders", f"{p['ref']}|{p['holder']}") if p.get("holder") else ("artifacts", p["ref"])
        a = self.world.state.tables[table][key]
        if p["audience"] == "public":
            self.world.state.patch(table, key, published=True)
        else:
            self.world.state.patch(table, key, audiences=sorted(set(a.get("audiences", ())) | {p["audience"]}))

    @applier("artifact.granted")
    def ap_granted(self, ev):
        p = ev.payload
        rec = {"ref": p["ref"], "grantee": p["grantee"], "by": p["by"], "t": ev.t}
        if p.get("record"):
            rec["record"] = p["record"]
        self.world.state.put("artifact_grants", grant_key(p["ref"], p["grantee"], p.get("record")), rec)

    @applier("artifact.revoked")
    def ap_revoked(self, ev):
        p = ev.payload
        self.world.state.delete("artifact_grants", grant_key(p["ref"], p["grantee"], p.get("record")))

    # ------------------------------------------------------------------ tools
    @staticmethod
    def _describe(st, ref: str, owner: str, rec: dict) -> dict:
        """One record as a tool shows it (the first owner's artifact record or a holder's own record)."""
        a = st.tables["artifacts"][ref]
        return {"ref": ref, "mime": rec["mime"], "size": a["size"], "owner": owner, "parents": list(rec["parents"]),
                "meta": to_jsonable(rec["meta"]), "kind": rec["kind"], "published": bool(rec.get("published")),
                "t": rec["t"]}

    def _view(self, ctx: Ctx, ref: str) -> dict:
        """What `ctx.actor` may see of an artifact (NotFound when it may see nothing): its own record (as owner or
        as a holder of identical bytes), else a record published or granted to it - never the record of somebody
        it merely shares bytes with."""
        if not exists(ctx.world, ref):
            raise NotFound(f"no artifact {ref}")
        got = _record_for(ctx.state, ref, ctx.world.principals_of(ctx.actor))
        if got is None:
            raise NotFound(f"no artifact {ref}")
        return self._describe(ctx.state, ref, *got)

    @endpoint("put", _PUT_PARAMS, cost_s=2)
    def ep_put(self, ctx, data_b64, mime, kind="file", parents=None, meta=None, owner=None):
        """Store bytes as an immutable, content-addressed artifact (owned by you, or by an organisation you
        administer). Returns its reference sha256:<hex>; storing identical bytes again returns the same reference."""
        owner = owner or ctx.actor
        if owner != ctx.actor and owner not in ctx.world.principals_of(ctx.actor):
            raise Forbidden(f"you cannot store artifacts for {owner}")
        try:
            data = base64.b64decode(data_b64.encode("ascii"), validate=True)
        except (binascii.Error, ValueError, UnicodeEncodeError):
            raise InvalidAction("data_b64 is not valid base64") from None
        if kind == "batch":
            raise InvalidAction("kind 'batch' is reserved for system array blobs")
        existed = readable(ctx.world, owner, ref_of(data))      # relative to the caller: no existence oracle
        ref = self.store_bytes(ctx, data, mime, owner=owner, parents=parents or (), meta=meta, kind=kind)
        return {"ref": ref, "size": len(data), "new": not existed}

    @endpoint("get", S.obj({"ref": _REF}, ["ref"]), read_only=True, cost_s=1)
    def ep_get(self, ctx, ref):
        """Fetch the bytes (base64) of an artifact you can read: your own, granted to you, or published."""
        parse_ref(ref)
        d = self._view(ctx, ref)
        data = get(ctx.world, ref)
        return {"ref": ref, "mime": d["mime"], "kind": d["kind"], "size": len(data),
                "data_b64": base64.b64encode(data).decode("ascii")}

    @endpoint("info", S.obj({"ref": _REF}, ["ref"]), read_only=True, cost_s=0)
    def ep_info(self, ctx, ref):
        """Metadata of an artifact you can read: media type, size, owner, provenance (parents), kind, publication."""
        parse_ref(ref)
        return self._view(ctx, ref)

    @endpoint("list", S.obj({"owner": S.s_str("whose artifacts (default: yours)", maxLength=60),
                             "kind": S.s_str("only this kind", pattern=KIND_RE, maxLength=32),
                             "limit": S.s_int(minimum=1, maximum=200, default=50),
                             "offset": S.s_int(minimum=0, default=0)}), read_only=True, cost_s=0)
    def ep_list(self, ctx, owner=None, kind=None, limit=50, offset=0):
        """List artifacts of an owner that you can read (all of your own; others' only when published or granted
        to you), oldest first."""
        owner = owner or ctx.actor
        st = ctx.state
        ps = ctx.world.principals_of(ctx.actor)
        mine = owner in ps or owner == ctx.actor
        recs = [(ref, st.tables["artifacts"][ref]) for ref in st.find_ids("artifacts", "owner", owner)]
        recs += [(h["ref"], h) for h in (st.tables["artifact_holders"][hid]
                                         for hid in st.find_ids("artifact_holders", "holder", owner))]
        rows = []
        for ref, rec in recs:
            if kind is not None and rec["kind"] != kind:
                continue
            # another owner's listing shows only the records that owner published or granted to the caller: being
            # able to read the bytes because the caller holds them too must not confirm that the owner holds them
            if mine or _record_visible(st, ref, owner, rec, ps):
                rows.append((rec["t"], ref, rec))
        rows.sort(key=lambda x: (x[0], x[1]))
        page = rows[offset:offset + limit]
        return {"owner": owner, "total": len(rows), "artifacts": [self._describe(st, r, owner, rec) for _, r, rec in page]}

    @endpoint("publish", S.obj({"ref": _REF, "audience": S.s_str("'public' (default) or one principal",
                                                                maxLength=60)}, ["ref"]), cost_s=1)
    def ep_publish(self, ctx, ref, audience="public"):
        """Publish an artifact you own: readable by everyone, or by one principal (an organisation's owners and
        admins when the audience is an organisation). Publication cannot be withdrawn."""
        parse_ref(ref)
        self.publish(ctx, ref, audience)
        return {"ref": ref, "audience": audience}

    @endpoint("grant", S.obj({"ref": _REF, "grantee": S.s_str("principal to grant read access", maxLength=60)},
                             ["ref", "grantee"]), cost_s=1)
    def ep_grant(self, ctx, ref, grantee):
        """Let one principal read an artifact you own."""
        parse_ref(ref)
        owner, rec, first = self.own_record(ctx, ref, "grant")
        if owner == SYSTEM:
            raise Forbidden("only the owner can grant access")
        if not ctx.state.has("principals", grantee):
            raise NotFound(f"no principal {grantee}")
        if not ctx.state.has("artifact_grants", grant_key(ref, grantee, None if first else owner)):
            payload = {"ref": ref, "grantee": grantee, "by": ctx.actor}
            if not first:
                payload["record"] = owner
            ctx.emit("artifact.granted", payload, vis=tuple(sorted({owner, grantee, ctx.actor})))
        return {"ref": ref, "grantee": grantee}

    @endpoint("revoke", S.obj({"ref": _REF, "grantee": S.s_str(maxLength=60)}, ["ref", "grantee"]), cost_s=1)
    def ep_revoke(self, ctx, ref, grantee):
        """Withdraw a read grant you gave (copies the grantee already fetched are theirs)."""
        parse_ref(ref)
        owner, rec, first = self.own_record(ctx, ref, "revoke")
        if not ctx.state.has("artifact_grants", grant_key(ref, grantee, None if first else owner)):
            raise NotFound(f"{grantee} has no grant on {ref}")
        payload = {"ref": ref, "grantee": grantee, "by": ctx.actor}
        if not first:
            payload["record"] = owner
        ctx.emit("artifact.revoked", payload, vis=tuple(sorted({owner, grantee, ctx.actor})))
        return {"ref": ref, "grantee": grantee, "revoked": True}

    @endpoint("usage", S.obj(), read_only=True, cost_s=0)
    def ep_usage(self, ctx):
        """Bytes and artifacts you store, and your quota."""
        u = usage(ctx.world, ctx.actor)
        return dict(u, quota_bytes=self.cfg["quota_bytes"], max_bytes=self.cfg["max_bytes"],
                    fee_cents_per_mib=self.cfg["fee_cents_per_mib"])

    def check_invariants(self, world):
        errs = []
        st = world.state
        use: dict[str, list[int]] = {}
        for ref, a in st.tables["artifacts"].items():
            if not REF_RE.fullmatch(ref):
                errs.append(f"bad artifact id {ref}")
            for p in a["parents"]:
                if not st.has("artifacts", p):
                    errs.append(f"{ref} has unknown parent {p}")
            if a["owner"] != SYSTEM:
                u = use.setdefault(a["owner"], [0, 0])
                u[0] += a["size"]
                u[1] += 1
        for hid, h in st.tables["artifact_holders"].items():
            a = st.tables["artifacts"].get(h["ref"])
            if a is None:
                errs.append(f"holder record {hid} of unknown artifact")
                continue
            u = use.setdefault(h["holder"], [0, 0])
            u[0] += a["size"]
            u[1] += 1
        for owner, u in st.tables["artifact_usage"].items():
            if use.get(owner, [0, 0]) != [u["bytes"], u["count"]]:
                errs.append(f"usage of {owner} recorded {u['bytes']}/{u['count']} but artifacts sum to "
                            f"{use.get(owner, [0, 0])}")
            if u["bytes"] > self.cfg["quota_bytes"]:
                errs.append(f"{owner} exceeds its quota")
        for owner in use:
            if not st.has("artifact_usage", owner):
                errs.append(f"usage of {owner} not recorded")
        return errs
