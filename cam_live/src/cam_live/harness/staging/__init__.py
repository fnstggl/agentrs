"""cam_live.harness.staging — local controlled web services for acceptance tests (workstream EB).

Five sites, each on its own localhost port, all labelled ``staging`` in capabilities:

* ``structured`` — forms, tables, a three-step workflow, upload/download, a sign-in cookie, an untrusted-text page
* ``canvas``     — the same task painted on a canvas (no DOM semantics) for visual computer use
* ``dynamic``    — content that changes after load and after actions
* ``variants``   — two differently labelled/structured layouts of one task and a mid-run layout switch
* ``labsite``    — ``diagnostics.world.LabWorld`` behind a web UI and JSON pages (delayed consequences)

Content is abstract and neutral; these are not models of any real platform and results on them are staging results.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ...contracts import SimClock
from ...diagnostics.world import LabConfig
from .canvas import LAYOUT as CANVAS_LAYOUT, CanvasSite, center as canvas_center
from .labsite import LabSite, LabSiteSource
from .server import Site, SiteServer
from .sites import DynamicSite, StructuredSite, VariantSite

__all__ = ["StagingSites", "LabSiteSource", "StructuredSite", "CanvasSite", "DynamicSite", "VariantSite", "LabSite",
           "CANVAS_LAYOUT", "canvas_center"]


class StagingSites:
    """Starts the staging sites on localhost. ``urls[name]`` is each site's origin; ``sites[name]`` its object."""

    def __init__(self, *, lab_config: Optional[LabConfig] = None, clock: Optional[SimClock] = None,
                 host: str = "127.0.0.1", ports: Optional[Dict[str, int]] = None):
        self.host, self._ports = host, dict(ports or {})
        self.sites: Dict[str, Site] = {"structured": StructuredSite(), "canvas": CanvasSite(),
                                       "dynamic": DynamicSite(), "variants": VariantSite(),
                                       "labsite": LabSite(lab_config, clock)}
        self.servers: Dict[str, SiteServer] = {}
        self.urls: Dict[str, str] = {}

    structured = property(lambda self: self.sites["structured"])
    canvas = property(lambda self: self.sites["canvas"])
    dynamic = property(lambda self: self.sites["dynamic"])
    variants = property(lambda self: self.sites["variants"])
    labsite = property(lambda self: self.sites["labsite"])

    def start(self) -> "StagingSites":
        for name, site in self.sites.items():
            srv = SiteServer(site, self.host, self._ports.get(name, 0)).start()
            self.servers[name], self.urls[name] = srv, srv.url
        return self

    def stop(self) -> None:
        for srv in self.servers.values():
            srv.stop()
        self.servers.clear()

    def __enter__(self) -> "StagingSites":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def outcome_source(self) -> LabSiteSource:
        return LabSiteSource(self.urls["labsite"])

    def capabilities(self) -> Dict[str, Any]:
        return {"provenance": "staging", "live": False,
                "sites": {n: {"url": self.urls.get(n, ""), "label": "staging", "description": s.description}
                          for n, s in self.sites.items()}}
