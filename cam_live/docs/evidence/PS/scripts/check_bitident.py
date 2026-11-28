"""PS: the pre-PS experiential core (``git show 87eab34:cam_live/src/cam_live/nn/core.py``) and the PS core side by side
on the same weights (debug model, CPU): fresh states bit-identical; a carried state with rho = 0 bit-identical to the
fresh computation; with rho = 1 the difference the newest-event elapsed time makes.

    python check_bitident.py   (from anywhere inside the repository)
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = subprocess.check_output(["git", "-C", HERE, "rev-parse", "--show-toplevel"], text=True).strip()
sys.path.insert(0, os.path.join(REPO, "cam_live", "src"))
sys.path.insert(0, os.path.join(REPO, "cam_live", "tests", "nn"))
from cam_live.config import CamLiveConfig  # noqa: E402
from cam_live.nn.api import CoreState  # noqa: E402
from cam_live.nn.model import build_model  # noqa: E402
from conftest import make_items  # noqa: E402

src = subprocess.check_output(["git", "-C", REPO, "show", "87eab34:cam_live/src/cam_live/nn/core.py"], text=True)
path = os.path.join(tempfile.mkdtemp(), "core_orig.py")
open(path, "w").write(src)
spec = importlib.util.spec_from_file_location("cam_live.nn.core_orig", path)
orig = importlib.util.module_from_spec(spec)
orig.__package__ = "cam_live.nn"
spec.loader.exec_module(orig)

torch.manual_seed(0)
m = build_model(CamLiveConfig.debug())
core = m.exp.core
oc = orig.ExperientialCore(core.d_model, core.n_layers, core.n_heads, core.blocks[0].ff.w1.out_features, core.n_slots,
                           state_every=2, gate_bias=-2.0)
oc.load_state_dict({k: v for k, v in core.state_dict().items() if k != "persist_rho"}, strict=True)
tb = m.tokenize([make_items(7, tag="a"), make_items(3, tag="b")])
with torch.no_grad():
    a, b = core(tb), oc(tb)
    out = {"fresh_b_bit_identical": torch.equal(a.b, b.b), "fresh_h_bit_identical": torch.equal(a.h, b.h),
           "fresh_slots_bit_identical": torch.equal(a.state.slots, b.state.slots)}
    st = CoreState(a.state.slots * 1.3 + 0.1, torch.tensor([1000.0, 900.0], dtype=torch.float64), 1)
    core.persist_rho.zero_()
    c = core(tb, st)
    out["rho0_carried_equals_fresh_b"] = torch.equal(c.b, a.b)
    out["rho0_carried_equals_fresh_h"] = torch.equal(c.h, a.h)
    core.persist_rho.fill_(1.0)
    d, e = core(tb, st), oc(tb, st)
    out["rho1_carried_vs_old_core_max_abs_b"] = float((d.b - e.b).abs().max())
print(json.dumps(out, indent=1))
