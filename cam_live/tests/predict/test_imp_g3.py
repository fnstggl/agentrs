"""IMP regression tests for the G3-core findings (docs/verification/G3_core.md §2.2).

D-IMP1: the paired ablation was not paired (same-seed arms differed before imagination could act) — the driver now
removes the run environment's nondeterminism, so the same arm twice is identical and the two arms are identical up to
the first decision at which imagination can affect the choice.
D-IMP2: the content target moved with the LoRA adapters — it now comes from the frozen base backbone.
D-IMP3: overlapping verified windows made the ImaginationCalibration overconfident — each window now counts 1/(h+1).
"""
import math
import uuid

import torch

from cam_live.predict.policy import ImaginationCalibration, SelectionCalibration


# ---------------------------------------------------------------------------------------------------------- D-IMP3
def _simulate(h, rho, T, reps, seed=0):
    """True beta_im = 0; windows of h + 1 decisions fed overlapping, exactly as refresh_calibration does."""
    g = torch.Generator().manual_seed(seed)
    pos = excl = 0
    means, sds = [], []
    for _ in range(reps):
        z = torch.randn(T + h + 1, generator=g, dtype=torch.float64)
        e = torch.randn(T + h + 1, generator=g, dtype=torch.float64)
        inn = torch.randn(T + h + 1, generator=g, dtype=torch.float64)
        m = torch.zeros(T + h + 1, dtype=torch.float64)
        for t in range(T + h + 1):
            m[t] = (rho * m[t - 1] if t else 0.0) + math.sqrt(1 - rho * rho) * inn[t]
        y = 0.3 * z + e
        cal = ImaginationCalibration()
        for t in range(T):
            cal.observe(f"d{t}", h, float(z[t]), float(m[t]), float(y[t:t + h + 1].sum()))
        s = cal.summary(h)
        means.append(s["beta_im"])
        sds.append(s["beta_im_sd"])
        pos += int(s["beta_im"] > 1.96 * s["beta_im_sd"])
        excl += int(abs(s["beta_im"]) > 1.96 * s["beta_im_sd"])
    mu = sum(means) / reps
    emp = math.sqrt(sum((x - mu) ** 2 for x in means) / (reps - 1))
    return pos / reps, excl / reps, emp / (sum(sds) / reps)


def test_overlapping_windows_do_not_make_the_imagination_calibration_overconfident(measure):
    pos, excl, ratio = _simulate(h=4, rho=0.9, T=100, reps=80)
    measure("imp_g3_overlap_h4_rho09", {"confidently_positive": pos, "interval_excludes_0": excl,
                                        "empirical_over_posterior_sd": ratio})
    assert pos <= 0.05 and excl <= 0.10 and ratio <= 1.1          # before the fix: 9.5 % / 18 % / 1.5
    cal = ImaginationCalibration()
    assert cal.get(4).obs_weight == 1.0 / 5 and cal.get(1).obs_weight == 0.5
    # the one-step selection calibration (independent observations) is unchanged
    assert SelectionCalibration().obs_weight == 1.0
    # a state saved before the correction loads with it
    st = cal.state_dict()
    for v in st["by_h"].values():
        v.pop("obs_weight")
    back = ImaginationCalibration()
    back.load_state_dict(st)
    assert back.get(4).obs_weight == 0.2 and back.get(1).obs_weight == 0.5


# ---------------------------------------------------------------------------------------------------------- D-IMP2
def test_the_content_target_does_not_move_with_the_lora_adapters():
    from cam_live.config import CamLiveConfig
    from cam_live.contracts import ExperienceItem, ItemKind
    from cam_live.nn.model import build_model
    from cam_live.predict.content import ContentTarget
    torch.manual_seed(0)
    model = build_model(CamLiveConfig.debug(seed=0))
    bb = model.backbone
    texts = ["the server answered with status 200 and three new members", "error: permission denied"]
    groups = [[ExperienceItem(kind=ItemKind.OBSERVATION, text=t)] for t in texts]
    y0, _ = ContentTarget(bb.d_sem, seed=0)(model, groups)
    e0 = model.embed_texts(texts).clone()
    scales = [m.scale for m in bb.lora_modules]
    with torch.no_grad():
        g = torch.Generator().manual_seed(1)
        for p in bb.lora_parameters():
            p.add_(torch.randn(p.shape, generator=g).to(p.device, p.dtype) * 1e-2)
    bb.mark_adapters_updated()
    t1 = ContentTarget(bb.d_sem, seed=0)                     # a fresh instance: nothing cached
    y1, _ = t1(model, groups)
    e1 = model.embed_texts(texts)
    assert t1.fixed_target
    assert float((y1 - y0).norm() / y0.norm()) < 1e-6         # the target: fixed (before the fix: 22 %)
    assert float((e1 - e0).norm() / e0.norm()) > 1e-3         # the adapted embedding the tokenizer uses: moved
    assert [m.scale for m in bb.lora_modules] == scales       # the adapters are switched back on
    assert not y1.requires_grad


# ---------------------------------------------------------------------------------------------------------- D-IMP1
def test_paired_runs_are_identical_until_imagination_can_act(tmp_path):
    from cam_live.predict.imp_eval import _row_inputs_differ, pairing, run
    torch.set_num_threads(1)
    real_uuid4 = uuid.uuid4
    kw = dict(family="T1", lifetimes=1, decisions=5, real=False, backbone=False, threads=1, world_base=None,
              extra_predict={}, budget_s=0.0)
    a = run("on", 1, out=str(tmp_path / "a.json"), workdir=str(tmp_path / "wa"), **kw)
    b = run("on", 1, out=str(tmp_path / "b.json"), workdir=str(tmp_path / "wb"), **kw)
    c = run("off", 1, out=str(tmp_path / "c.json"), workdir=str(tmp_path / "wc"), **kw)
    assert uuid.uuid4 is real_uuid4                           # the driver gives the process its uuid4 back
    ra, rb = a["lifetimes"][0]["decision_rows"], b["lifetimes"][0]["decision_rows"]
    assert len(ra) == len(rb) >= 4
    assert all(not _row_inputs_differ(x, y) and (x["operation"], x["arguments"]) == (y["operation"], y["arguments"])
               for x, y in zip(ra, rb))                       # the same arm twice: identical
    p = pairing(a, c)
    assert p["paired_until_influence"] and not p["action_differs_before_influence"]
    assert a["determinism"]["paired"] and a["determinism"]["clock"] == "SimClock"
