"""Exp C exact part: key-decision gradient SNR versus delay (16 ... 1024 decisions)."""
from superlearner.cam_unified.credit.longdelay import LongDelay


def test_snr_of_contribution_methods_is_flat_in_delay_while_reinforce_decays():
    snr = {T: LongDelay(T=T).stats(n=20000, seed=T) for T in (16, 64, 256, 1024)}
    for k in ("cocoa", "rhca"):
        vals = [snr[T][k]["snr"] for T in snr]
        assert max(vals) / min(vals) < 1.2, (k, vals)
    rf = [snr[T]["reinforce"]["snr"] for T in snr]
    assert rf[0] > rf[1] > rf[2] > rf[3] and rf[0] / rf[3] > 20
    for T in snr:
        for k, v in snr[T].items():
            assert v["bias_norm"] <= 5 * v["se"] + 1e-12, (T, k, v)


def test_rhca_beats_cocoa_when_the_key_step_has_noisy_immediate_reward():
    st = LongDelay(T=64, noise0=1.5).stats(n=20000)
    assert st["rhca"]["snr"] > 2 * st["cocoa"]["snr"]
