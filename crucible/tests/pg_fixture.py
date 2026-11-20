"""Throwaway PostgreSQL 16 cluster for the storage tests.

`pg_dsn` (session-scoped) runs `initdb` into a fresh temp directory (as the `postgres` OS user when the tests
run as root, because initdb refuses to run as root), starts the server listening only on a unix socket in that
directory (`listen_addresses=''`, so nothing is exposed on TCP), creates a database and yields a libpq DSN
(`host=<socket dir> ...`). On teardown the server is stopped and the directory removed, also if the session is
interrupted (atexit fallback). Tests that use it are skipped when the binaries or psycopg are missing.

`pg_fresh_dsn` (function-scoped) creates a new, empty database in that cluster for each test.

Usage in a test module (import into exactly one module, so that only one cluster is started):

    from pg_fixture import pg_dsn, pg_fresh_dsn  # noqa: F401  (pytest fixtures)
"""
from __future__ import annotations

import atexit
import itertools
import os
import re
import shutil
import socket
import subprocess
import tempfile

import pytest

PG_BIN = os.environ.get("CRUCIBLE_PG_BIN", "/usr/lib/postgresql/16/bin")
DBNAME = "crucible_test"
_db_counter = itertools.count()


def _free_port() -> int:
    # The server listens on a unix socket only; the port just names the socket file (.s.PGSQL.<port>).
    # Pick one that is also free on TCP so that a misconfiguration can never collide with a real server.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _as_pg(cmd: list[str]) -> list[str]:
    """initdb/pg_ctl refuse to run as root: drop to the postgres OS user when we are root."""
    if os.geteuid() == 0:
        return ["runuser", "-u", "postgres", "--", *cmd]
    return cmd


class PgCluster:
    def __init__(self) -> None:
        self.root = tempfile.mkdtemp(prefix="crucible-pg-")
        self.data = os.path.join(self.root, "data")
        self.sock = self.root
        self.log = os.path.join(self.root, "server.log")
        self.port = _free_port()
        self.running = False

    def _run(self, cmd: list[str], timeout: float = 120) -> subprocess.CompletedProcess:
        return subprocess.run(_as_pg(cmd), capture_output=True, text=True, timeout=timeout)

    def start(self) -> str:
        if os.geteuid() == 0:
            shutil.chown(self.root, "postgres", "postgres")
        os.chmod(self.root, 0o700)
        r = self._run([f"{PG_BIN}/initdb", "-D", self.data, "-U", "postgres", "--auth=trust", "-E", "UTF8",
                       "--locale=C", "--no-sync"])
        if r.returncode != 0:
            raise RuntimeError(f"initdb failed:\n{r.stdout}\n{r.stderr}")
        opts = (f"-k {self.sock} -c listen_addresses='' -p {self.port} -c fsync=off "
                f"-c synchronous_commit=off -c full_page_writes=off -c max_connections=60")
        r = self._run([f"{PG_BIN}/pg_ctl", "-D", self.data, "-l", self.log, "-w", "-t", "60", "-o", opts, "start"])
        if r.returncode != 0:
            log = open(self.log).read() if os.path.exists(self.log) else ""
            raise RuntimeError(f"pg_ctl start failed:\n{r.stdout}\n{r.stderr}\n{log}")
        self.running = True
        import psycopg
        with psycopg.connect(self.dsn("postgres"), autocommit=True) as c:
            c.execute(f"CREATE DATABASE {DBNAME}")
        return self.dsn(DBNAME)

    def dsn(self, db: str = DBNAME) -> str:
        return f"host={self.sock} port={self.port} user=postgres dbname={db}"

    def stop(self) -> None:
        if self.running:
            r = self._run([f"{PG_BIN}/pg_ctl", "-D", self.data, "-m", "fast", "-w", "-t", "60", "stop"])
            if r.returncode != 0:  # never leave a server behind
                self._run([f"{PG_BIN}/pg_ctl", "-D", self.data, "-m", "immediate", "-w", "-t", "30", "stop"])
            self.running = False
        shutil.rmtree(self.root, ignore_errors=True)


def _skip_reason() -> str | None:
    for b in ("initdb", "pg_ctl", "postgres"):
        if not os.access(os.path.join(PG_BIN, b), os.X_OK):
            return f"PostgreSQL binary {PG_BIN}/{b} not found"
    if os.geteuid() == 0:
        if shutil.which("runuser") is None:
            return "running as root and runuser is not available"
        try:
            import pwd
            pwd.getpwnam("postgres")
        except KeyError:
            return "running as root and there is no 'postgres' OS user"
    try:
        import psycopg  # noqa: F401
    except ImportError:
        return "psycopg (crucible[postgres]) is not installed"
    return None


def fresh_database(dsn: str) -> str:
    """Create an empty database next to the one `dsn` points at and return its DSN."""
    import psycopg
    name = f"crucible_t{os.getpid()}_{next(_db_counter)}"
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute(f"CREATE DATABASE {name}")
    return re.sub(r"dbname=\S+", f"dbname={name}", dsn)


@pytest.fixture(scope="session")
def pg_dsn():
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    cluster = PgCluster()
    atexit.register(cluster.stop)  # safety net if the session dies before fixture teardown
    try:
        dsn = cluster.start()
    except BaseException:
        cluster.stop()
        raise
    try:
        yield dsn
    finally:
        cluster.stop()


@pytest.fixture
def pg_fresh_dsn(pg_dsn):
    return fresh_database(pg_dsn)
