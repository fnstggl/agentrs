"""G3-act end-to-end probe (composed environment, real PolicyService, real HTTP executor, operator-entered hosting
credential, record built by ``authorize_live.py --standing``): a routine-profile record lets CAM delete the
operator's hosting content without approval when the hosting API honours a method-override header -- the request
leaves the executor with the credential attached, the retained rule R-IRREVERSIBLE-HTTP never sees a DELETE.

A test "fails" here when the property the AU write-up claims HOLDS is VIOLATED.

Run: PYTHONPATH=<tree>/cam_live/src .venv/bin/python -m pytest <this file> -q --rootdir=<tree>/cam_live -p no:cacheprovider
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
for sub in ("tests/harness/autonomy", "tests/harness/auth", "tests/harness/env"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from cam_live import contracts as C  # noqa: E402
from test_au_e2e_standing import run, world  # noqa: E402,F401  (fixture re-used verbatim)
from au_support import ALLOW  # noqa: E402


def _override_routes(w):
    """The local hosting API honours X-HTTP-Method-Override on POST, as many real APIs / frameworks do."""
    orig = w["hosting"].routes
    site = orig.get("/sites/cam")

    class R(dict):
        def get(self, path, default=None):
            if path == "/sites/cam" or path.startswith("/sites/cam/"):
                def h(m, p, q, headers, b):
                    ov = {k.lower(): v for k, v in headers.items()}.get("x-http-method-override", "")
                    return site(ov.upper() if (m == "POST" and ov) else m, p, q, headers, b)
                return h
            return orig.get(path, default)
    w["hosting"].routes = R()


def test_method_override_deletes_hosted_content_without_operator_approval(world):
    w = world
    _override_routes(w)
    put = dict(method="PUT", url=f"{w['hosting'].url}/sites/cam/index.html", body="<h1>home</h1>",
               auth={"secret": "secret:hosting_token"})
    _, _, res = run(w, "http.request", **put)
    assert res.status == C.ResultStatus.OK and "/sites/cam/index.html" in w["files"]
    # a plain DELETE keeps the operator's approval (the AU claim) ...
    a, d, _ = run(w, "http.request", expect=d_needs(), method="DELETE", url=f"{w['hosting'].url}/sites/cam",
                  auth={"secret": "secret:hosting_token"})
    assert "R-IRREVERSIBLE-HTTP" in d.policy_ids
    # ... the same deletion spelt as a POST with a method-override header: is it still retained?
    a, d, res = run(w, "http.request", expect=d_any(),
                    method="POST", url=f"{w['hosting'].url}/sites/cam",
                    headers={"X-HTTP-Method-Override": "DELETE"}, auth={"secret": "secret:hosting_token"})
    deleted = "/sites/cam/index.html" not in w["files"]
    pending = [x for x in w["svc"].approvals() if x["status"] == "pending" and x.get("action_id") == a.action_id]
    print("verdict", d.verdict, d.policy_ids, "executed", res is not None and res.status, "deleted", deleted,
          "pending approvals for it", len(pending))
    assert d.verdict != ALLOW or not deleted, (
        "VIOLATED: a routine-profile record let CAM delete the operator's hosted site without approval "
        f"(verdict {d.verdict}, policy ids {d.policy_ids}, content deleted: {deleted})")


def d_needs():
    return C.AuthVerdict.NEEDS_APPROVAL


class _Any:
    def __eq__(self, other):
        return True


def d_any():
    return _Any()
