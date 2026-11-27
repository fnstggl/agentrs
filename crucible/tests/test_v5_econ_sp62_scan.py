"""Gate SP-6.2's systems-of-record scan (`class_billing_sites`) on synthetic positive and negative controls.

Regression of the verifier's finding on c5c/w3b-econ1 (DEF-F-12): the scan read only the body of the registered
module class, so `caps` (innovation/caps.py CapabilityRegistry, billing through its base class
services/capabilities.Capabilities._bill) was not counted. The scan must see recurring charges a module posts
directly, through a base class in another file, through a top-level helper of its own file, and through helpers of
other files it calls by an imported name or module attribute; it must not flag one-time payments."""
import importlib
import importlib.util
import os
import sys
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def _gate():
    spec = importlib.util.spec_from_file_location("gate_sp62_scan", os.path.join(HERE, "v5", "test_gate_SP_6_2.py"))
    sys.path.insert(0, os.path.join(HERE, "v5"))
    try:
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
    finally:
        sys.path.remove(os.path.join(HERE, "v5"))
    return m


FILES = {
    "__init__.py": "",
    "helpers.py": '''
        def charge(bank, ctx, a, b, amount):
            bank.move(ctx, a, b, amount, "subscription", "monthly fee")

        def charge_kw(bank, ctx, a, b, amount):
            return bank.debit(ctx, a, amount, kind="subscription")

        def deeper(bank, ctx, a, b):
            return charge(bank, ctx, a, b, 1)

        def one_time(bank, ctx, a, b, amount):
            bank.move(ctx, a, b, amount, "purchase", "an order")
        ''',
    "base.py": '''
        class BillingBase:
            def _bill(self, bank, ctx, a, b):
                bank.move(ctx, a, b, 5, "subscription", "capability fee")
        ''',
    "mods.py": '''
        from . import helpers
        from .base import BillingBase
        from .helpers import charge, deeper, one_time

        def _local(bank, ctx, a, b):
            bank.move(ctx, a, b, 3, "subscription", "x")

        class Direct:
            def run(self, bank, ctx, a, b):
                bank.move(ctx, a, b, 1, "subscription", "fee")

        class Inherits(BillingBase):
            pass

        class ByName:
            def run(self, bank, ctx, a, b):
                charge(bank, ctx, a, b, 2)

        class ByModuleAttribute:
            def run(self, bank, ctx, a, b):
                helpers.charge_kw(bank, ctx, a, b, 2)

        class TwoDeep:
            def run(self, bank, ctx, a, b):
                deeper(bank, ctx, a, b)

        class PopulationPay:
            def run(self, pop, ctx, i, pid):
                pop.pay(ctx, i, 100, memo=f"subscription {pid}")

        class OneTimeOnly:
            def run(self, bank, ctx, a, b, pop, i):
                one_time(bank, ctx, a, b, 9)
                bank.move(ctx, a, b, 9, "purchase", "order 1")
                pop.pay(ctx, i, 9, memo="order 2")
        ''',
}


@pytest.fixture(scope="module")
def mods(tmp_path_factory):
    root = tmp_path_factory.mktemp("sp62scan")
    pkg = root / "sp62pkg"
    pkg.mkdir()
    for name, src in FILES.items():
        (pkg / name).write_text(textwrap.dedent(src))
    sys.path.insert(0, str(root))
    try:
        yield importlib.import_module("sp62pkg.mods")
    finally:
        sys.path.remove(str(root))
        for k in [k for k in sys.modules if k.startswith("sp62pkg")]:
            del sys.modules[k]


def test_scan_sees_direct_inherited_and_helper_billing_and_not_one_time_payments(mods):
    g = _gate()

    def sites(cls):
        return {s.split("/")[-1].split(":")[0] for s in g.class_billing_sites(cls, package="sp62pkg")}
    # every class of mods.py also has the file's own top-level helper `_local` (a recurring charge in mods.py)
    assert sites(mods.Direct) == {"mods.py"}
    assert sites(mods.Inherits) == {"mods.py", "base.py"}            # billing through a base class in another file
    assert sites(mods.ByName) == {"mods.py", "helpers.py"}           # a helper imported by name
    assert sites(mods.ByModuleAttribute) == {"mods.py", "helpers.py"}  # a helper reached as module.attribute
    assert sites(mods.TwoDeep) == {"mods.py", "helpers.py"}          # a helper calling a helper
    pay = g.class_billing_sites(mods.PopulationPay, package="sp62pkg")
    assert any(s.endswith("mods.py:" + str(i)) for s in pay for i in range(1, 60))
    # one-time payments only: the only hit is the file's top-level recurring helper, never the class's own calls
    one = g.class_billing_sites(mods.OneTimeOnly, package="sp62pkg")
    lines = {int(s.rsplit(":", 1)[1]) for s in one}
    src = textwrap.dedent(FILES["mods.py"]).splitlines()
    assert all("subscription" in src[ln - 1] for ln in lines), [src[ln - 1] for ln in lines]
    assert all("helpers.py" not in s for s in one)                     # one_time() is not a recurring charge


def test_shipped_registry_scan_finds_the_subscription_module():
    """Positive control on the real registry: `subs` is found where it charges subscriptions."""
    g = _gate()
    out = g.recurring_billing_modules()
    assert "subs" in out and all(s.startswith("crucible/commerce/subscriptions.py:") for s in out["subs"]), out
