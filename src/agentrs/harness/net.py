"""Controlled internet egress for every connector.

``SafeHTTPClient`` is the only HTTP client connectors get. Every request (and every redirect hop)
is checked before it leaves the process:

* scheme: ``https`` (``http`` only for explicitly allowed local hosts such as provider simulators);
* host on the egress allowlist (exact or ``.suffix``), and on the provider's own domain list;
* DNS resolution must not yield private, loopback, link-local, multicast or reserved addresses, nor
  the cloud metadata endpoints (``169.254.169.254``, ``fd00:ec2::254``, ``metadata.google.internal``)
  — unless the exact host is in ``allow_private`` (local development only);
* timeouts, connection limits, response-size cap; no automatic redirects (each hop re-checked);
* an optional proxy for legitimate corporate routing; there is no IP rotation of any kind;
* every request is logged (time, provider, method, host, path, status, bytes, outcome).

Residual risk (documented): a DNS answer can change between the check and the connection (DNS
rebinding). Production deployments should additionally block private ranges at the network layer
(see ``deploy/k8s/networkpolicy.yaml``).
"""

from __future__ import annotations

import ipaddress
import json
import logging
import socket
import time
from typing import Any, Optional
from urllib.parse import urljoin, urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger("agentrs.egress")

METADATA_HOSTS = {"169.254.169.254", "fd00:ec2::254", "metadata.google.internal", "metadata", "100.100.100.200"}


class EgressDenied(PermissionError):
    pass


class EgressPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed_domains: list[str] = Field(default_factory=list)
    allow_private: list[str] = Field(default_factory=list)  # exact hosts, e.g. ["127.0.0.1"] for simulators
    allow_http_for: list[str] = Field(default_factory=list)  # hosts allowed without TLS (local only)
    connect_timeout_s: float = 5.0
    read_timeout_s: float = 20.0
    max_connections: int = 20
    max_connections_per_host: int = 5
    max_response_bytes: int = 10 * 1024 * 1024
    max_redirects: int = 3
    proxy_url: Optional[str] = None  # corporate egress proxy (stable egress IP), never rotation
    egress_pool: str = "default"  # label of the approved egress (NAT gateway / static IP) for this worker pool


def _host_allowed(host: str, domains: list[str]) -> bool:
    host = host.lower().rstrip(".")
    for d in domains:
        d = d.lower().rstrip(".")
        if d.startswith("."):
            if host.endswith(d) or host == d[1:]:
                return True
        elif host == d:
            return True
    return False


def _ip_forbidden(ip: str) -> bool:
    a = ipaddress.ip_address(ip)
    return (a.is_private or a.is_loopback or a.is_link_local or a.is_multicast or a.is_reserved
            or a.is_unspecified or str(a) in METADATA_HOSTS)


def check_url(url: str, policy: EgressPolicy, provider_domains: Optional[list[str]] = None) -> str:
    """Raise ``EgressDenied`` unless ``url`` may be fetched. Returns the normalised host."""
    p = urlsplit(url)
    host = (p.hostname or "").lower()
    if not host:
        raise EgressDenied(f"no host in {url!r}")
    if host in METADATA_HOSTS:
        raise EgressDenied(f"cloud metadata endpoint {host} is never reachable")
    if p.scheme not in ("https", "http"):
        raise EgressDenied(f"scheme {p.scheme!r} not allowed")
    if p.scheme == "http" and host not in policy.allow_http_for:
        raise EgressDenied(f"plain http to {host} not allowed")
    if not _host_allowed(host, policy.allowed_domains):
        raise EgressDenied(f"{host} is not on the egress allowlist")
    if provider_domains is not None and not _host_allowed(host, provider_domains):
        raise EgressDenied(f"{host} is not one of this provider's domains")
    if host in policy.allow_private:
        return host
    try:
        infos = socket.getaddrinfo(host, p.port or (443 if p.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise EgressDenied(f"DNS resolution failed for {host}: {e}") from e
    for info in infos:
        ip = info[4][0]
        if _ip_forbidden(ip):
            raise EgressDenied(f"{host} resolves to a non-public address {ip} (SSRF protection)")
    return host


class SafeHTTPClient:
    """Async HTTP client bound to an egress policy and (optionally) one provider's domains."""

    def __init__(self, policy: EgressPolicy, provider: str = "", provider_domains: Optional[list[str]] = None,
                 egress_log: Optional[list] = None, transport: Optional[httpx.AsyncBaseTransport] = None):
        self.policy = policy
        self.provider = provider
        self.provider_domains = provider_domains
        self.log = egress_log if egress_log is not None else []
        self._transport = transport
        self._client: Optional[httpx.AsyncClient] = None
        self._loop = None

    def _get_client(self) -> httpx.AsyncClient:
        """One pooled client per event loop (an async client cannot outlive the loop it was made in)."""
        import asyncio

        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop:
            p = self.policy
            limits = httpx.Limits(max_connections=p.max_connections, max_keepalive_connections=p.max_connections_per_host)
            timeout = httpx.Timeout(p.read_timeout_s, connect=p.connect_timeout_s)
            self._client = httpx.AsyncClient(limits=limits, timeout=timeout, follow_redirects=False, proxy=p.proxy_url,
                                             transport=self._transport, trust_env=False)
            self._loop = loop
        return self._client

    async def request(self, method: str, url: str, **kw) -> httpx.Response:
        hops = 0
        while True:
            t0 = time.time()
            entry: dict[str, Any] = {"t": t0, "provider": self.provider, "method": method, "url_host": urlsplit(url).hostname,
                                     "path": urlsplit(url).path, "pool": self.policy.egress_pool}
            try:
                check_url(url, self.policy, self.provider_domains)
            except EgressDenied as e:
                entry.update(outcome="denied", reason=str(e))
                self._record(entry)
                raise
            try:
                r = await self._get_client().request(method, url, **kw)
            except httpx.HTTPError as e:
                entry.update(outcome="network_error", reason=type(e).__name__)
                self._record(entry)
                raise
            size = len(r.content)
            entry.update(outcome="ok", status=r.status_code, bytes=size, ms=int(1000 * (time.time() - t0)))
            self._record(entry)
            if size > self.policy.max_response_bytes:
                raise EgressDenied(f"response larger than {self.policy.max_response_bytes} bytes")
            if r.is_redirect and hops < self.policy.max_redirects:
                url = urljoin(url, r.headers["location"])
                hops += 1
                if r.status_code in (301, 302, 303):
                    method, kw = "GET", {k: v for k, v in kw.items() if k in ("headers",)}
                continue
            return r

    def _record(self, entry: dict) -> None:
        self.log.append(entry)
        log.info(json.dumps({"event": "egress", **entry}))

    async def get(self, url: str, **kw) -> httpx.Response:
        return await self.request("GET", url, **kw)

    async def post(self, url: str, **kw) -> httpx.Response:
        return await self.request("POST", url, **kw)

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except RuntimeError:  # its loop is already gone
                pass
            self._client = None
