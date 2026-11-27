import sys, json, re
sys.path.insert(0, "/workspace/agentrs/crucible/src")
import crucible; crucible.load_all_modules()
from crucible.rapid import families as F
from superlearner.cam_unified.envs.experience import ExperienceOperator, WorldSpec, BROKEN_MULTIAGENT_RAPID, PRIVATE_PAYLOAD_KEYS
from superlearner.cam_unified.envs import integrity as I
from superlearner.cam_unified.interfaces import Action
fams = [k for k, v in F.REGISTRY.items() if v.get("split") == "train" and k not in BROKEN_MULTIAGENT_RAPID]
out = {"n_train_families": len(fams), "hits": {}, "errors": {}, "privkey_hits": {}}
for fam in fams:
    try:
        op = ExperienceOperator([WorldSpec(1, f"rapid:{fam}", 7)])
        ad = op.adapter(deliver_results=True)
        stream = [ad.reset()]
        tools = ad.available_tools(); stream.append(tools)
        script = []
        for t in tools[0]:
            props = (t.parameters or {}).get("properties", {})
            for v in props.get("target", {}).get("enum", []):
                script.append(Action(t.name, {"kind": "query", "target": v}))
            for v in props.get("kind", {}).get("enum", []):
                if v != "query": script.append(Action(t.name, {"kind": v}))
        script += [Action(tools[0][0].name, {})] * 3
        for a in script[:40]:
            r = ad.execute([a]); stream.append(r); stream.append(ad.advance_time([float(r[0].duration) + 1/24]))
        stream.append(ad.goals())
        dump = I.actor_dump(*stream).lower()
        refs = op.worlds[0].refs or {}
        toks = [fam.lower(), f"rapid:{fam}".lower(), "#7"]
        for k in ("null", "oracle"):
            v = refs.get(k)
            if isinstance(v, float) and abs(v) > 1:
                toks.append(repr(v)[:8])
        h = [t for t in toks if t in dump]
        if h: out["hits"][fam] = h
        pk = I.privileged_keys(stream)
        if pk: out["privkey_hits"][fam] = pk[:5]
    except Exception as e:
        out["errors"][fam] = repr(e)[:200]
print(json.dumps(out, indent=1))
