"""D-40 institution migration, readers (wave-3b repair, verifier probe p_civ_fp): where `inst` is installed (every
civilisation) gov keeps no constitution or proposal of its own, so readers of the 4.0 tables read them through
`services.governance.constitutions_view` / `proposals_view`. civ.integration.fingerprint read fp['institutions']
from the always-empty gov table: the profile defined by differing institutional structures showed none and its
structural distances halved."""
import crucible

crucible.load_all_modules()
from crucible.civ import generator, integration as I  # noqa: E402


def test_civilisation_fingerprints_show_institutions_kept_as_inst_charters():
    fps = {}
    for name in ("balanced", "cooperative_institutions", "mixed_economy"):
        w = generator.generate(name, 5)
        w.advance_time(3600)
        assert not w.state.tables.get("constitutions"), "gov keeps no record of its own where inst is installed"
        fps[name] = I.fingerprint(w)
        w.close()
    inst = {k: [tuple(x) for x in fp["institutions"]] for k, fp in fps.items()}
    assert inst == {"balanced": [], "cooperative_institutions": [("equal", 2), ("labor", 2)],
                    "mixed_economy": [("equal", 2)]}, inst
    d_coop = I.structural_distance(fps["balanced"], fps["cooperative_institutions"])
    d_mix = I.structural_distance(fps["balanced"], fps["mixed_economy"])
    d_cm = I.structural_distance(fps["cooperative_institutions"], fps["mixed_economy"])
    assert d_coop >= 0.16 and d_mix >= 0.16 and d_cm >= 0.24, (d_coop, d_mix, d_cm)   # 0.083/0.083/0.167 when broken
