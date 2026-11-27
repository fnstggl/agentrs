"""G-pr D6 on the real backbone: are reworded self-generated goals near-duplicates in the backbone's semantic text
embedding (``CamLiveModel.embed_texts``), the embedding ``SelfGoalDiscount(embed=...)`` uses? Measured, written to
docs/evidence/C/semantic_goal_repeats.json."""
import itertools
import json
import os

import pytest
import torch

EVIDENCE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "evidence", "C"))

PARAPHRASES = [["Reach 10 members", "Get ten members", "Grow the community to ten people", "Recruit 10 members"],
               ["Learn which posting time gets the most replies", "Find out when posts receive the most answers",
                "Discover the best hour to post for replies"]]
DIFFERENT = ["Translate the welcome page into Spanish", "Fix the broken link in the rules channel",
             "Measure how long the bot takes to respond", "Reduce the budget spent on advertising"]


@pytest.mark.gpu
def test_reworded_goals_are_closer_than_different_goals_in_the_real_embedding(gpu_heavy):
    from cam_live.config import CamLiveConfig
    from cam_live.nn.model import build_model
    from cam_live.reward.integrity import SelfGoalDiscount
    model = build_model(CamLiveConfig(device="cuda"))
    texts = [t for g in PARAPHRASES for t in g] + DIFFERENT
    with torch.no_grad():
        e = model.embed_texts(texts).to(torch.float64).cpu()
    u = e / e.norm(dim=1, keepdim=True)
    cos = u @ u.T
    idx = {t: i for i, t in enumerate(texts)}
    same = [float(cos[idx[a], idx[b]]) for g in PARAPHRASES for a, b in itertools.combinations(g, 2)]
    groups = PARAPHRASES + [[d] for d in DIFFERENT]
    diff = [float(cos[idx[a], idx[b]]) for g, h in itertools.combinations(groups, 2) for a in g for b in h]
    thr = (min(same) + max(diff)) / 2 if min(same) > max(diff) else None
    res = {"backbone": model.backbone.name, "revision": model.backbone.revision, "n_same": len(same),
           "n_different": len(diff), "same_goal_cosine": {"min": min(same), "mean": sum(same) / len(same)},
           "different_goal_cosine": {"max": max(diff), "mean": sum(diff) / len(diff)},
           "separating_threshold": thr, "texts": texts}
    if thr is not None:
        d = SelfGoalDiscount(embed=lambda t: model.embed_texts([t])[0], similar_above=thr)
        res["weights_with_threshold"] = {t: d.weight(t, 0.3) for t in texts}
    os.makedirs(EVIDENCE, exist_ok=True)
    with open(os.path.join(EVIDENCE, "semantic_goal_repeats.json"), "w") as f:
        json.dump(res, f, indent=1)
    del model
    torch.cuda.empty_cache()
    assert sum(same) / len(same) > sum(diff) / len(diff)          # on average a rewording is closer
