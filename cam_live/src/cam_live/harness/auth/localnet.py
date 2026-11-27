"""This machine's own addresses and listening sockets, for the local-address rule of the authorization layer.

No mode may reach the instance's control plane (the instance portal, Jupyter, SSH, tunnel managers, IDE servers,
the cloud metadata service ...). A URL destination whose host is an address of this machine or of its link —
loopback, the unspecified address, link-local and metadata addresses, private networks, this machine's own
interface addresses and names that ``/etc/hosts`` maps to any of those — is therefore denied in every mode unless
its ``(host, port)`` is a *registered local service*:

* listed by the operator in the policy file (``destinations.local_services``);
* registered at run start through ``PolicyService.register_local_service`` (operator only), e.g. the staging sites
  that the computer-environment service started;
* (modes with ``own_process_services: true`` only) a loopback port whose every listening socket belongs to the
  process the authorization service runs in — staging services the harness started in-process. A port that any
  other process listens on is never such a service.

Everything here reads Linux ``/proc`` and ``/etc/hosts``; nothing performs DNS. A public host name whose DNS record
points at this machine (DNS rebinding) is not detected at decision time: executors report the connected peer
address (``peer_ip`` in the primitive) so that the rule can be applied to it too (docs/EA/SECURITY.md).
"""
from __future__ import annotations

import ipaddress
import os
import re
import threading
import time
from typing import Dict, List, Optional, Set, Tuple

#: every loopback / unspecified spelling is canonicalised to this host (destinations._canon_host)
LOOPBACK = "127.0.0.1"
#: well-known cloud metadata addresses outside 169.254.0.0/16 (that whole block is link-local and covered anyway)
METADATA_IPS = frozenset({"100.100.100.200", "fd00:ec2::254", "fd00:ec2::23"})
METADATA_NAMES = frozenset({"metadata", "metadata.google.internal", "metadata.goog", "instance-data",
                            "instance-data.ec2.internal", "metadata.azure.internal", "metadata.packet.net",
                            "metadata.tencentyun.com", "metadata.platformequinix.com"})
#: host-name suffixes that only resolve inside a private network
PRIVATE_SUFFIXES = (".local", ".internal", ".localdomain", ".home.arpa", ".lan", ".intranet", ".corp")
_PRIVATE_NETS = [ipaddress.ip_network(n) for n in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10", "198.18.0.0/15", "192.0.0.0/24",
    "fc00::/7", "fec0::/10")]
_LINK_LOCAL_NETS = [ipaddress.ip_network(n) for n in ("169.254.0.0/16", "fe80::/10")]
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_V4COMPAT = ipaddress.ip_network("::/96")
_ZERO8 = ipaddress.ip_network("0.0.0.0/8")

_lock = threading.Lock()
_cache: Dict[str, Tuple[float, object]] = {}
_SELF_TTL_S = 30.0


def canonical_ip(ip: "ipaddress._BaseAddress") -> str:
    """IPv4-mapped / NAT64 / IPv4-compatible IPv6 -> the embedded IPv4; every loopback and unspecified address ->
    ``LOOPBACK``. Anything else: the address in its standard notation."""
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        elif ip in _NAT64:
            ip = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        elif ip in _V4COMPAT and int(ip) > 1:
            ip = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    if ip.is_loopback or ip.is_unspecified or (isinstance(ip, ipaddress.IPv4Address) and ip in _ZERO8):
        return LOOPBACK
    return str(ip)


def _cached(name: str, ttl: float, fn):
    now = time.monotonic()
    with _lock:
        hit = _cache.get(name)
        if hit is not None and now - hit[0] < ttl:
            return hit[1]
    val = fn()
    with _lock:
        _cache[name] = (now, val)
    return val


def _read(path: str) -> str:
    try:
        with open(path, "r", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _self_addresses() -> Set[str]:
    out: Set[str] = set()
    pending = None
    for line in _read("/proc/net/fib_trie").splitlines():
        m = re.match(r"^\s*\|--\s+(\d+\.\d+\.\d+\.\d+)\s*$", line)
        if m:
            pending = m.group(1)
            continue
        if pending and "/32 host LOCAL" in line:
            out.add(pending)
        pending = None if not line.strip().startswith("/") else pending
    for line in _read("/proc/net/if_inet6").splitlines():
        parts = line.split()
        if parts and len(parts[0]) == 32:
            try:
                out.add(str(ipaddress.IPv6Address(int(parts[0], 16))))
            except ValueError:
                pass
    for var in ("PUBLIC_IPADDR",):
        v = os.environ.get(var, "").strip()
        try:
            out.add(canonical_ip(ipaddress.ip_address(v)))
        except ValueError:
            pass
    return {canonical_ip(ipaddress.ip_address(a)) for a in out if a} - {LOOPBACK}


def self_addresses() -> Set[str]:
    """Addresses of this machine's own interfaces (and ``$PUBLIC_IPADDR``), canonical, loopback excluded."""
    return _cached("self", _SELF_TTL_S, _self_addresses)  # type: ignore[return-value]


def _hosts_names() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in _read("/etc/hosts").splitlines():
        line = line.split("#", 1)[0].split()
        if len(line) < 2:
            continue
        try:
            ip = ipaddress.ip_address(line[0].split("%", 1)[0])
        except ValueError:
            continue
        cls = ip_class(ip, use_hosts=False)
        if not cls:
            continue
        for name in line[1:]:
            out[name.lower().rstrip(".")] = cls
    return out


def ip_class(ip: "ipaddress._BaseAddress", use_hosts: bool = True) -> str:
    """'' for an ordinary public address, else the reason it belongs to this machine or its link."""
    c = canonical_ip(ip)
    if c == LOOPBACK:
        return "loopback"
    ip = ipaddress.ip_address(c)
    if any(ip in n for n in _LINK_LOCAL_NETS):
        return "link_local"
    if c in METADATA_IPS:
        return "metadata"
    if c in self_addresses():
        return "self"
    if any(ip in n for n in _PRIVATE_NETS) or ip.is_multicast or ip.is_reserved:
        return "private"
    return ""


def local_kind(host: str) -> str:
    """Classify a canonical host (``destinations._canon_host`` output). '' = not local."""
    h = (host or "").lower().rstrip(".")
    if not h:
        return ""
    try:
        return ip_class(ipaddress.ip_address(h))
    except ValueError:
        pass
    if h in METADATA_NAMES:
        return "metadata"
    if h == "localhost" or h.endswith(".localhost"):
        return "loopback"
    if any(h.endswith(s) for s in PRIVATE_SUFFIXES) or "." not in h:
        return "private_name"          # single-label names resolve through the local search domain only
    names = _cached("hosts", _SELF_TTL_S, _hosts_names)
    return names.get(h, "")            # type: ignore[union-attr]


# ---------------------------------------------------------------------------------------------------------------
# listening sockets
# ---------------------------------------------------------------------------------------------------------------
def _decode(addr_hex: str, v6: bool) -> Optional["ipaddress._BaseAddress"]:
    try:
        if not v6:
            return ipaddress.IPv4Address(bytes.fromhex(addr_hex)[::-1])
        raw = b"".join(bytes.fromhex(addr_hex[i:i + 8])[::-1] for i in range(0, 32, 8))
        return ipaddress.IPv6Address(raw)
    except ValueError:
        return None


def listeners() -> List[Tuple[str, int, int]]:
    """(canonical address, port, socket inode) of every listening TCP socket in this network namespace."""
    out: List[Tuple[str, int, int]] = []
    for fn, v6 in (("/proc/net/tcp", False), ("/proc/net/tcp6", True)):
        for line in _read(fn).splitlines()[1:]:
            parts = line.split()
            if len(parts) < 10 or parts[3] != "0A":
                continue
            try:
                addr_hex, port_hex = parts[1].split(":")
                port, inode = int(port_hex, 16), int(parts[9])
            except ValueError:
                continue
            ip = _decode(addr_hex, v6)
            if ip is not None:
                out.append((canonical_ip(ip), port, inode))
    return out


def own_socket_inodes() -> Set[int]:
    out: Set[int] = set()
    try:
        fds = os.listdir("/proc/self/fd")
    except OSError:
        return out
    for fd in fds:
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if target.startswith("socket:["):
            try:
                out.add(int(target[8:-1]))
            except ValueError:
                pass
    return out


_LISTEN_TTL_S = 1.0


def own_listener(port: int) -> bool:
    """True when at least one socket listens on ``port`` on a loopback-reachable address and every such socket
    belongs to this process. A port with no listener, or with any listener of another process, is not.

    The kernel's socket table is expensive to read on a busy machine; a reading up to one second old may confirm a
    port (the listener was this process's a moment ago), but a refusal is always re-checked on a fresh reading."""
    if not isinstance(port, int) or not 0 < port < 65536:
        return False
    own = own_socket_inodes()
    for fresh in (False, True):
        table = listeners() if fresh else _cached("listen", _LISTEN_TTL_S, listeners)
        mine = [ino for addr, p, ino in table if p == port and addr == LOOPBACK]      # type: ignore[union-attr]
        if mine and all(ino in own for ino in mine):
            return True
    return False
