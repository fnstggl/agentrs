"""G3-act round 2, end to end (composed environment, real PolicyService, real HTTP / Discord executors, staging
Discord, local hosting API, record built by authorize_live.py --standing --routine 'PUT <hosting>/sites/cam/**').

* redirect: a declared-routine PUT answered with 307 to an undeclared, destructive route of the same host -- the
  redirect hop must not be sent without approval (each hop is checked as a primitive);
* a typed adapter operation the policy lists routine (discord.set_channel_permissions) handing @everyone
  MANAGE_CHANNELS + MANAGE_MESSAGES in a channel of CAM's server: does it run without approval? (A test FAILS when it
  does: the property 'consequence keeps approval' is violated for that operation.)

Run: PYTHONPATH=<tree>/cam_live/src .venv/bin/python -m pytest <this file> -q -s --rootdir=<tree>/cam_live -p no:cacheprovider
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
for sub in ("tests/harness/autonomy", "tests/harness/auth", "tests/harness/env"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from cam_live import contracts as C  # noqa: E402
from test_au_e2e_standing import run, world  # noqa: E402,F401
from au_support import ALLOW  # noqa: E402


class _Any:
    def __eq__(self, other):
        return True


def test_a_redirect_from_a_declared_route_to_a_destructive_route_is_not_followed_without_approval(world):
    w = world
    orig = w["hosting"].routes
    purged = []

    class R(dict):
        def get(self, path, default=None):
            if path == "/sites/cam/redir.html":
                return lambda m, p, q, h, b: (307, {"Location": w["hosting"].url + "/admin/purge-all"}, "")
            if path == "/admin/purge-all":
                def purge(m, p, q, h, b):
                    purged.append(m)
                    return 200, {"Content-Type": "application/json"}, '{"purged": true}'
                return purge
            return orig.get(path, default)
    w["hosting"].routes = R()
    a, d, res = run(w, "http.request", expect=_Any(), method="PUT", url=f"{w['hosting'].url}/sites/cam/redir.html",
                    body="<p>x</p>", auth={"secret": "secret:hosting_token"})
    print("REDIRECT verdict", d.verdict, d.policy_ids, "result", res and res.status, (res and res.failure_detail or "")[:160],
          "purge requests", purged)
    assert d.verdict == ALLOW                       # the declared-routine PUT itself is routine
    assert not purged, "VIOLATED: the redirect hop to an undeclared destructive route was sent without approval"


def test_routine_listed_adapter_operation_hands_channel_moderation_to_everyone(world):
    w, st = world, world["st"]
    _, _, res = run(w, "discord.create_channel", guild_id=st.guild_id, name="open-room", type="text")
    assert res.status == C.ResultStatus.OK, res.failure_detail
    ch = res.artifacts.get("channel_id") or res.observations[-1].data.get("channel_id") or \
        (res.observations[-1].data.get("channel") or {}).get("id")
    a, d, res = run(w, "discord.set_channel_permissions", expect=_Any(), guild_id=st.guild_id, channel_id=ch,
                    role_id="everyone", allow=["MANAGE_CHANNELS", "MANAGE_MESSAGES"])
    ows = (st.channels.get(ch) or {}).get("permission_overwrites") if hasattr(st, "channels") else None
    print("OVERWRITE verdict", d.verdict, d.policy_ids, "result", res and res.status,
          (res and res.failure_detail or "")[:160], "overwrites now", ows)
    executed = res is not None and res.status == C.ResultStatus.OK
    assert not (d.verdict == ALLOW and executed), (
        "VIOLATED: under the routine profile CAM gave @everyone MANAGE_CHANNELS + MANAGE_MESSAGES in a channel of its "
        "server without any approval (any member can then delete the channel and others' messages)")
