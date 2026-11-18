"""Natural-language layer: information equivalence (exact inversion of the training wording),
no information loss in the held-out wording (injective), and wording actually differs."""
from __future__ import annotations

import random

from agentrs.cam.envs import ENVS
from agentrs.cam.text.render import goal_text, obs_text, parse_goal, parse_obs


def _states(fam, n_eps=40, seed=0):
    r = random.Random(seed)
    out = []
    for s in range(n_eps):
        e = ENVS[fam]()
        e.reset(s, r.choice(["train", "val", "test"]))
        g = e.grammar()
        out.append((e.goal_tokens(), e.obs_tokens()))
        while not e.done:
            pre = []
            while not pre or pre[-1] != "END":
                pre.append(r.choice(g.allowed(pre)))
            e.step(pre[:-1])
            out.append((e.goal_tokens(), e.obs_tokens()))
    return out


def test_training_wording_is_information_equivalent_to_the_tokens():
    for fam in ENVS:
        for k, (gt, ot) in enumerate(_states(fam)):
            assert parse_obs(fam, obs_text(fam, ot, "train", key=k)) == ot, (fam, ot, obs_text(fam, ot, "train", key=k))
            assert parse_goal(fam, goal_text(fam, gt, "train", key=k)) == gt


def test_heldout_wording_loses_no_information_and_is_different():
    for fam in ENVS:
        seen = {}
        for k, (gt, ot) in enumerate(_states(fam, seed=1)):
            x = obs_text(fam, ot, "heldout", key=0)
            assert seen.setdefault(x, tuple(ot)) == tuple(ot), "two different observations rendered to the same text"
            try:
                bad = parse_obs(fam, x) != ot
            except (AttributeError, KeyError, ValueError, StopIteration):
                bad = True
            assert bad, "the training-wording parser should not read held-out wording"
