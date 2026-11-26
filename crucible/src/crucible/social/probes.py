"""Gate E.3 probes on the social internet's feeds (`crucible.semantic.encoders.effects`; workstream C, wave 3b).

`social_feed` (text) and `social_feed_image` (image): the integrated social internet of the ONE population
(`crucible.social.worlds.social_world`: pop, the catalogue and sites, `aud` with the shared-space encoder, the feeds
with the platform's recommender mixture and the consumption engine, finite attention) where 6,000 individuals browse
one platform's feed for three simulated days. The catalogue is fixed and identical in both arms - six real news
abstracts of the population's corpus (as text posts, or as poster cards of the same text on the image platform),
each published by an experimenter account at time zero - plus the probed artifact, published by its own account at
the same moment with the same title. The population's own creation is switched off (a world with nothing else to
read keeps the probed item in the competition for every individual's attention). Both arms differ only in the
probed artifact's bytes: whether individuals are shown it (interest retrieval and engagement ranking read its
representation), open it and how long they stay (the engine's decisions read the affinity of their interests with
it) all come from the recorded representation of those bytes. Outcome: each individual's seconds of consumption of
the probed item (0 if never opened).

Fidelity: the consumption engine's built-in decisions (FIDELITY_REGISTER `social-consumption-builtin`); the shared
space (`module:space`).
"""
from __future__ import annotations

import numpy as np

PROBE_N, PROBE_DAYS = 6000, 3.0
TOPICS = ("finance-real-estate", "recipes", "weathertopstories", "movies-celebrity", "travelarticle", "baseball_mlb")
PROBE_HANDLE = "probe"
MIX = [{"name": "interest", "weight": 1.0}, {"name": "engagement", "weight": 0.6}, {"name": "fresh", "weight": 0.4},
       {"name": "explore", "weight": 0.2}]


def _corpus() -> list[str]:
    from ..population import semantics
    names, corpus = semantics.topic_names(), semantics.corpus()
    return [corpus[names.index(t)][0] for t in TOPICS]


def make_world(data: bytes, mime: str, seed: int, *, platform: str = "posts"):
    from ..core.kernel import World
    from .worlds import social_world
    w = World.create(social_world("development", int(seed), n=PROBE_N, platforms=[platform],
                                  feed={"platforms": {platform: {"recommenders": MIX, "k": 8}}},
                                  creators={"prior": -30.0, "inbox_prior": -30.0, "community_prior": -30.0},
                                  family="social_feed_probe"), strict=True)
    texts = _corpus()
    for k, text in enumerate(texts):
        p = w.system_call("admin.join", {"handle": f"src{k}", "endowment": 100, "password": f"pw-src{k}"})["principal"]
        title = " ".join(text.split()[:8])
        _publish(w, p, text, title, platform)
    p = w.system_call("admin.join", {"handle": PROBE_HANDLE, "endowment": 100, "password": "pw-probe"})["principal"]
    a = w.execute(p, {"tool": "artifact.put", "args": {"data_b64": __import__("base64").b64encode(bytes(data))
                                                       .decode("ascii"), "mime": mime,
                                                       "kind": "image" if mime.startswith("image/") else "text"}})
    if not a.ok:
        raise RuntimeError(f"probe artifact refused: {a.error}")
    kind = "image" if platform == "pics" else "post"
    r = w.execute(p, {"tool": f"{platform}.publish", "args": {"ref": a.result["ref"], "title": "Today", "kind": kind}})
    if not r.ok:
        raise RuntimeError(f"probe artifact not published: {r.error}")
    w.state.meta_set("probe.item", int(w.state.tables["content"][r.result["id"]]["aud_item"]))
    return w


def _publish(w, p: str, text: str, title: str, platform: str) -> None:
    import base64
    if platform == "pics":
        from . import popmedia as PM
        data, mime, kind, akind = PM.image_card(title, "", PM.style_of(len(text))), "image/png", "image", "image"
    else:
        data, mime, kind, akind = f"{title}\n\n{text}\n".encode("utf-8"), "text/plain", "post", "text"
    a = w.execute(p, {"tool": "artifact.put", "args": {"data_b64": base64.b64encode(data).decode("ascii"),
                                                       "mime": mime, "kind": akind}})
    assert a.ok, a.error
    r = w.execute(p, {"tool": f"{platform}.publish", "args": {"ref": a.result["ref"], "title": title, "kind": kind}})
    assert r.ok, r.error


def run(world) -> np.ndarray:
    from . import audience as AUD
    from . import experiments as X
    item = int(world.state.meta["probe.item"])
    s0 = 0
    world.advance_time(world.now + int(PROBE_DAYS * 86400))
    con = X.decision_rows(world, "consume", s0)
    out = np.zeros(AUD.n_people(world))
    m = con["item"] == item
    np.add.at(out, con["viewer"][m], con["watch_s"][m])
    return out


def _register() -> None:
    from ..semantic.encoders import effects
    effects.register_probe("social_feed", lambda d, m, s: make_world(d, m, s, platform="posts"), run,
                           modality="text", doc="the one population browsing the text-post feed of the social "
                                                "internet (recommenders + consumption engine); outcome: seconds "
                                                "of consumption of the probed post per individual")
    effects.register_probe("social_feed_image", lambda d, m, s: make_world(d, m, s, platform="pics"), run,
                           modality="image", doc="the same on the image platform: the probed image among poster "
                                                 "cards of real news; outcome: seconds looking at it per individual")


_register()
