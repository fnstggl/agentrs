"""Kernel module `civgen`: the cross-module genesis of a civilisation (ordinary APIs only).

The modules of a civilisation create most of their initial content in their own genesis from the configs the
generator writes (industrial firms and their minds in `industrygen`, labs and phenomena in `sci`, web sites in
`popsites`, boards in `forum`). `civgen` does what needs several modules at once:

* the compute market of the profile: providers (organisations founded by members of the population, so their
  owners are principals of the same identity registry), their capacity and prices, and where their datacenters
  stand (registered with `civlinks`, whose `compute_energy` coupling makes them draw from that region's grid);
* the institutions of the profile: constitutions (`gov.adopt`, by each organisation's own owner) whose profit-sharing
  rule changes members' effort (`gov.effort`, read by industry), and public ownership of utilities;
* the R&D routine of producer firms when the profile has the `science_process` coupling (an entry in each producer
  mind's routines; the routine itself is `crucible.civ.couplings.rnd_routine`).

Nothing here decides what happens later.
"""
from __future__ import annotations

import numpy as np

from ..core.errors import CrucibleError
from ..core.kernel import Module, register_module
from ..substrate import ids
from . import popapi

DAY = 86400


@register_module
class CivGenesis(Module):
    name = "civgen"
    host = "civgen"
    version = "1"
    depends = ("identity", "bank", "cloud", "pop")

    def genesis(self, ctx):
        w = ctx.world
        cfg = self.config
        prof = cfg.get("profile") or {}
        rng = w.rng_for("civgen")
        idm, bank, cloud = w.modules["identity"], w.modules["bank"], w.modules["cloud"]
        n = popapi.n_individuals(w)
        from ..substrate import columns
        cls = columns.col(w, "pop", "class") if n else np.zeros(0)
        pool = np.flatnonzero(cls == 1)
        founders = iter(rng.permutation(pool).tolist()) if pool.size else iter(())

        def founder() -> str | None:
            try:
                return ids.ensure_principal(ctx, int(next(founders)), account=True)
            except StopIteration:
                return None
        regions = sorted(ctx.state.tables["regions"]) if "regions" in ctx.state.tables else []
        comp = cfg.get("compute") or {}
        k_prov = int(comp.get("providers", 1))
        cores = int(comp.get("cores", 256))
        made = []
        for k in range(k_prov):
            owner = founder()
            handle = f"compute{k}"
            public = (prof.get("institutions") or {}).get("utilities") == "public" and k == 0
            oid = idm.create_org(ctx, handle, f"{'Public ' if public else ''}Compute {k}", owner,
                                 "public" if public else "company", {"sector": "compute"})
            acct = bank.open_account(ctx, oid, "operating")
            bank.move(ctx, "acct_genesis", acct, 5_000_000, "endowment", f"initial capital of {oid}")
            price = int(rng.integers(2, 5)) + k
            cloud.add_provider(ctx, oid, f"Compute {k}", price, cores, round(float(rng.uniform(0.995, 0.999)), 4))
            if regions:
                reg = regions[0] if comp.get("datacenters") == "colocated" else regions[k % len(regions)]
                if "civlinks" in w.modules:
                    w.modules["civlinks"].register_datacenter(ctx, oid, reg)
            made.append(oid)
        # where the apparatus operators of the science system stand (their measurements load that region's grid
        # under the coupling `lab_energy`); placed round the regions, after the first one
        if "civlinks" in w.modules and "sci_app" in ctx.state.tables and regions:
            ops = sorted({a["operator"] for _, a in ctx.state.scan("sci_app")})
            for k, op in enumerate(ops):
                w.modules["civlinks"].register_lab(ctx, op, regions[(k + 1) % len(regions)])
        self._institutions(ctx, prof, made)
        self._rnd(ctx)
        ctx.emit("note", {"what": "civgen", "providers": made, "profile": prof.get("name")}, vis="internal")

    def _institutions(self, ctx, prof, providers):
        w = ctx.world
        if "gov" not in w.modules:
            return
        inst = prof.get("institutions") or {}
        gov_kind = inst.get("governance", "equity")
        st = ctx.state
        plans = []
        if "minds" in st.tables:
            for mid, m in sorted(st.scan("minds")):
                if m["kind"] != "firm" or not m.get("org"):
                    continue
                role = m.get("role", "")
                dist = None
                if gov_kind == "cooperative":
                    dist = {"products": {"basis": "labor", "share": 0.6},
                            "components": {"basis": "equal", "share": 0.5}}.get(role)
                elif gov_kind == "mixed" and role in ("utility", "freight"):
                    dist = {"basis": "equal", "share": 0.4}
                if dist:
                    plans.append((m["org"], m["owner"], dist))
        if inst.get("utilities") == "public":
            for org in providers[:1]:
                owner = next((x["person"] for x in st.find("members", "org", org) if x.get("role") == "owner"), None)
                if owner:
                    plans.append((org, owner, {"basis": "equal", "share": 0.4}))
        for org, owner, dist in plans:
            try:
                ctx.call("gov.adopt", {"org": org, "rules": [], "distribution": dist}, as_principal=owner)
            except CrucibleError as e:
                ctx.emit("note", {"what": "civgen.gov_failed", "org": org, "error": e.message[:200]}, vis="internal")

    def _rnd(self, ctx):
        w = ctx.world
        links = w.modules.get("civlinks")
        soc = w.modules.get("society")
        if links is None or soc is None or "science_process" not in (links.config.get("couplings") or []):
            return
        st = ctx.state
        for mid, m in sorted(st.scan("minds")):
            if m["kind"] == "firm" and m.get("role") in ("mining", "components", "products") and \
                    "production" in (m.get("routines") or {}):
                soc.set(ctx, mid, routines=dict(m["routines"], rnd={}))
