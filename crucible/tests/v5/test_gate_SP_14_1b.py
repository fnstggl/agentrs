"""SP-14.1b - The sampler is compatible with external learner infrastructure, mixtures are balanced and
configurable, and the distribution is versioned (MASTER_SPEC Part 14.1; workstream N).

External round trip: episode specs drawn here are written as JSON files and handed to a separate, isolated Python
process (``python -I``, empty HOME, no PYTHONPATH) that rebuilds each world with nothing but the spec
(`python -m crucible.research.v5.sampler make <spec.json>`); the world identity it reports (environment digest,
first-observation digest, tool list, reward mode) must equal the in-process build. The same specs also run through
the Gymnasium wrapper of `crucible.experience` (any RL stack).

Mixtures: a balanced mixture gives every family the same share within tolerance; an explicit weighting gives the
configured shares; the realised shares are measured on 6,000 draws. Versioning: the registry has a version and a
content digest, adding one generated world changes it and records the parent, and the version is carried by every
spec and sampler state.
"""
from __future__ import annotations

import collections
import json
import os
import subprocess
import sys

import pytest

from crucible.acceptance.v5.evidence import record
from crucible.research.v5 import distribution as D
from crucible.research.v5 import sampler as S
from crucible.research.v5.evaluator import isolated_env, src_root

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]


def test_SP14_1b_external_round_trip_mixtures_and_versions(tmp_path):
    reg = D.build_registry()
    smp = S.Sampler(reg, S.Mixture(tiers={1: 1.0}, name="t1"), seed=5)
    specs = []
    while len(specs) < 6:
        sp = smp.next()
        if sp["entry"].startswith("t1:rapid:") or (sp["entry"].startswith("t1:mech:") and
                                                   not any(s["entry"].startswith("t1:mech:") for s in specs)):
            specs.append(sp)
    ok = 0
    details = []
    for i, sp in enumerate(specs):
        p = tmp_path / f"spec{i}.json"
        p.write_text(json.dumps(sp))
        env = S.make_env(json.loads(p.read_text()), reg)
        a = env.agents()[0]
        local = {"describe": env.describe(), "env_digest": env.digest(), "observation_digest": D.sha(env.observe(a)),
                 "tools": sorted(t["name"] for t in env.tools(a)), "reward_mode": env.reward_mode}
        boot = ("import sys, runpy; sys.path.insert(0, sys.argv[1]); sys.argv = ['sampler', 'make', sys.argv[2]]; "
                "runpy.run_module('crucible.research.v5.sampler', run_name='__main__')")
        out = subprocess.run([sys.executable, "-I", "-c", boot, src_root(), str(p)], capture_output=True,
                             env=isolated_env(), timeout=600)
        remote = json.loads(out.stdout.decode().strip().splitlines()[-1]) if out.returncode == 0 else {}
        same = remote == json.loads(D.canonical(local))
        ok += int(same)
        details.append({"entry": sp["entry"], "same": same, "reward_mode": remote.get("reward_mode")})
    # Gymnasium compatibility of the same specs
    from crucible.experience.gym import ExperienceGymEnv
    gym_ok = 0
    for sp in specs[:2]:
        g = ExperienceGymEnv(env=S.make_env(sp, reg))
        obs, info = g.reset(seed=0)
        o2, r, term, trunc, info = g.step("")
        gym_ok += int(r == 0.0 and g.observation_space.contains(o2) and info["reward_mode"] == "observation_only")
    # balanced and configured mixtures
    fams = [w["id"] for w in reg.worlds(tier=1, family_partition="train", source="crucible.rapid")][:6]
    bal = S.Sampler(reg, S.Mixture(tiers={1: 1.0}, entries=tuple(fams), name="bal"), seed=1)
    cnt = collections.Counter(bal.next()["entry"] for _ in range(6000))
    shares = {f: cnt[f] / 6000 for f in fams if f not in bal._refused}
    balanced_err = max(abs(v - 1 / len(shares)) for v in shares.values())
    w = {f: float(i + 1) for i, f in enumerate(fams)}
    cfgd = S.Sampler(reg, S.Mixture(tiers={1: 1.0}, entries=tuple(fams), within=w, name="cfg"), seed=2)
    cnt2 = collections.Counter(cfgd.next()["entry"] for _ in range(6000))
    tot = sum(w[f] for f in fams if f not in cfgd._refused)
    cfg_err = max(abs(cnt2[f] / 6000 - w[f] / tot) for f in fams if f not in cfgd._refused)
    # versioning
    child = reg.extend([D.generated_entry({"topology": "fork", "motifs": ["threshold", "decay"]})])
    versioned = bool(reg.version.startswith(D.SCHEMA)) and child.digest != reg.digest and child.parent == reg.digest \
        and all(s["registry_digest"] == reg.digest for s in specs) and smp.state()["registry_digest"] == reg.digest
    round_trip = ok == len(specs) and gym_ok == 2
    record("SP-14.1b", {"external_sampler_round_trip": bool(round_trip), "distribution_versioned": bool(versioned),
                        "specs_round_tripped": ok, "specs_tried": len(specs), "gym_round_trips": gym_ok,
                        "balanced_max_share_error": float(balanced_err), "configured_max_share_error": float(cfg_err),
                        "registry_version": reg.version},
           notes=f"External process rebuilds: {details}. Balanced shares {shares}.")
    assert round_trip, details
    assert versioned
    assert balanced_err < 0.03 and cfg_err < 0.03
