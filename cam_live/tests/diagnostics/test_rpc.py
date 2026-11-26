import os, tempfile
import pytest
from cam_live import contracts as C
from cam_live.harness.rpc import RpcClient, RpcServer, RemoteError, caller_uid


class Svc:
    def echo(self, a): return a
    def who(self): return caller_uid()
    def boom(self): raise ValueError("nope")
    def secret(self): return "hidden"


def test_rpc_roundtrip_records_allowlist_and_peer_uid():
    d = tempfile.mkdtemp()
    srv = RpcServer(os.path.join(d, "s.sock"), Svc(), {"echo", "who", "boom"}, allowed_uids={os.getuid()}).start()
    try:
        c = RpcClient(srv.path)
        a = C.ActionSpec("a1", "browser.click", steps=[C.ActionSpec("a2", "browser.fill", arguments={"v": 1})])
        r = C.ActionResult("a1", C.ResultStatus.OK, observations=[C.Observation("o", 1.0, "page", quantities=[C.Quantity("n", 3.0)])])
        out = c.echo({"a": a, "r": [r], "b": b"\x00\x01"})
        assert out["a"] == a and out["r"][0] == r and out["b"] == b"\x00\x01"
        assert c.who() == os.getuid()
        with pytest.raises(RemoteError) as e:
            c.boom()
        assert e.value.kind == "ValueError"
        with pytest.raises(RemoteError) as e:
            c.secret()
        assert e.value.kind == "PermissionError"
        assert "echo" in c.call("__ping__")["methods"]
    finally:
        srv.stop()
