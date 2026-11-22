"""Socket-level guard: prove that a block of code opens no network connection.

`NetworkGuard` patches the socket module for the duration of a `with` block. Every attempt to resolve a name or
to connect / send to an address is recorded in `guard.attempts`; attempts to anything but the explicitly allowed
loopback endpoints raise `ExternalConnectionAttempt` *before* the operating system is asked. Unix-domain sockets
and socket pairs (used for pipes between local processes) are not network connections and are left alone.

It is used by `tests/test_realworld_gate.py` to show that creating and running a Crucible world, with adversarial
tool calls, never touches the network, and that the bridge itself only talks to the address it resolved and
checked.
"""
from __future__ import annotations

import ipaddress
import socket


class ExternalConnectionAttempt(AssertionError):
    pass


class NetworkGuard:
    def __init__(self, allow: tuple = (), allow_loopback: bool = False):
        self.allow = {(str(h), int(p)) for h, p in allow}
        self.allow_loopback = allow_loopback
        self.attempts: list[tuple] = []
        self._saved: dict = {}

    def _permitted(self, addr) -> bool:
        if not isinstance(addr, tuple) or len(addr) < 2:
            return True                                   # AF_UNIX path: local inter-process communication
        host, port = str(addr[0]), int(addr[1])
        if (host, port) in self.allow:
            return True
        try:
            return self.allow_loopback and ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    def _check(self, what: str, addr) -> None:
        if isinstance(addr, tuple) and len(addr) >= 2:
            self.attempts.append((what, str(addr[0]), int(addr[1])))
            if not self._permitted(addr):
                raise ExternalConnectionAttempt(f"{what} to {addr[0]}:{addr[1]} attempted")

    def __enter__(self) -> "NetworkGuard":
        g = self
        S = socket.socket
        self._saved = {"connect": S.connect, "connect_ex": S.connect_ex, "sendto": S.sendto,
                       "getaddrinfo": socket.getaddrinfo, "gethostbyname": socket.gethostbyname,
                       "create_connection": socket.create_connection}

        def connect(self, addr):
            g._check("connect", addr)
            return g._saved["connect"](self, addr)

        def connect_ex(self, addr):
            g._check("connect", addr)
            return g._saved["connect_ex"](self, addr)

        def sendto(self, data, *rest):
            g._check("sendto", rest[-1] if rest else None)
            return g._saved["sendto"](self, data, *rest)

        def getaddrinfo(host, port, *a, **k):
            g.attempts.append(("resolve", str(host), int(port or 0)))
            try:
                literal = ipaddress.ip_address(str(host))
            except ValueError:
                literal = None
            if literal is None or not g._permitted((str(host), int(port or 0))):
                raise ExternalConnectionAttempt(f"name resolution for {host!r} attempted")
            return g._saved["getaddrinfo"](host, port, *a, **k)

        def gethostbyname(host):
            g.attempts.append(("resolve", str(host), 0))
            raise ExternalConnectionAttempt(f"name resolution for {host!r} attempted")

        def create_connection(address, *a, **k):
            g._check("connect", address)
            return g._saved["create_connection"](address, *a, **k)

        S.connect, S.connect_ex, S.sendto = connect, connect_ex, sendto
        socket.getaddrinfo, socket.gethostbyname, socket.create_connection = getaddrinfo, gethostbyname, create_connection
        return self

    def __exit__(self, *exc) -> None:
        S = socket.socket
        S.connect, S.connect_ex, S.sendto = self._saved["connect"], self._saved["connect_ex"], self._saved["sendto"]
        socket.getaddrinfo = self._saved["getaddrinfo"]
        socket.gethostbyname = self._saved["gethostbyname"]
        socket.create_connection = self._saved["create_connection"]
