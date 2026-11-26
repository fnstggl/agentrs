"""Gate E.3 probe through the social internet's feeds (`space_feed`, registered with
`crucible.semantic.encoders.effects`).

The world is the social internet on the one population (`crucible.social.experiments.pop_spec`: the posts and blog
platforms with their recommender mixtures, the consumption engine, the population's own creators). At time 0 six
creators publish real news abstracts of the population's corpus (the same in both arms) and one more creator
publishes the probed artifact as a post, same title in both arms. Individuals then browse their feeds for `DAYS`
simulated days; whether a post is recommended to them, whether they open it and how long they read is decided by the
feed's recommenders and the population's consumption decisions from the item's representation in the shared space,
its intrinsic features, social signals and their own state. The outcome is each individual's reading time on the
probed post (seconds, summed over the recorded consumption batches). The two arms differ only in the probed
artifact's bytes; random draws are not common between the arms once their decisions differ, so differences mix the
representation's effect with that divergence, which the paired sign-flip test treats as noise.
"""
from __future__ import annotations

import base64

import numpy as np

# 6,000 individuals over 4 days: at 3,000 x 3 days only about 50-150 individuals' reading of the probed post differed
# between the arms (Gate E dev run, 2026-11-26), too few for the smaller effects of the controlled pairs
N, DAYS = 6000, 4.0
CATALOGUE_TOPICS = ("finance-real-estate", "recipes", "weathertopstories", "movies-celebrity", "travelarticle",
                    "baseball_mlb")
TITLE = "What happened this week"


def _catalogue() -> list[tuple[str, bytes]]:
    from ..population import semantics
    names, corpus = semantics.topic_names(), semantics.corpus()
    return [(t, corpus[names.index(t)][0].encode("utf-8")) for t in CATALOGUE_TOPICS]


def _publish(w, handle: str, data: bytes, mime: str, title: str) -> str:
    p = w.system_call("admin.join", {"handle": handle, "endowment": 10_000, "password": f"pw-{handle}"})["principal"]
    r = w.execute(p, {"tool": "artifact.put", "args": {"data_b64": base64.b64encode(data).decode("ascii"),
                                                      "mime": mime, "kind": "text"}})
    if not r.ok:
        raise RuntimeError(f"probe artifact refused: {r.error}")
    pub = w.execute(p, {"tool": "posts.publish", "args": {"ref": r.result["ref"], "title": title}})
    if not pub.ok:
        raise RuntimeError(f"probe post refused: {pub.error}")
    return pub.result["id"]


def make_world(data: bytes, mime: str, seed: int):
    from ..core.kernel import World
    from ..social import experiments as EX
    w = World.create(EX.pop_spec("e3_space_feed", int(seed), n=N, platforms={"posts": EX.GATE_D_POP["posts"]}))
    for k, (t, b) in enumerate(_catalogue()):
        _publish(w, f"news{k}", b, "text/plain", f"{t.replace('-', ' ')} news")
    cid = _publish(w, "probe", bytes(data), mime, TITLE)
    w.runtime["space_feed.target"] = cid
    return w


def run(world) -> np.ndarray:
    from ..population.core import n_individuals
    from ..social import experiments as EX
    world.advance_time(world.now + int(DAYS * 86400))
    cid = world.runtime["space_feed.target"]
    item = next(int(k) for k, rec in world.state.scan("aud_item_info") if rec.get("content_id") == cid)
    rows = EX.decision_rows(world, "consume")
    out = np.zeros(n_individuals(world), np.float64)
    sel = rows["item"] == item
    if sel.any():
        np.add.at(out, rows["viewer"][sel], rows["watch_s"][sel])
    return out


def register() -> None:
    from .encoders import effects
    effects.register_probe("space_feed", make_world, run, modality="text",
                           doc="the social internet's feeds on the one population; outcome: seconds each individual "
                               "spent reading the probed post (semantic.feedprobe)")


register()
