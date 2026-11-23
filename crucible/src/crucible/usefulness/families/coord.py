"""Organisational coordination: assign people to slots under generated availability and conflict constraints
(specified; staffed positions / positions).

Each world draws which constraints bind: how sparse availability is, how many pairs of people cannot share a slot,
whether slots require a skill, whether nobody may work two consecutive slots, and how much load people accept.
Every instance has a planted assignment that staffs every position, so a perfect score is attainable.
"""
from __future__ import annotations

from ..category import Population, category_family
from ._common import Stream, pick

SKILLS = ("lead", "medic", "tech")

_SOLVER = r'''
def solve(args, caller):
    slots, people = args['slots'], args['people']
    S, P = len(slots), len(people)
    sidx = {s['id']: k for k, s in enumerate(slots)}
    pidx = {p['id']: k for k, p in enumerate(people)}
    avail = [set(sidx[x] for x in p['avail'].split() if x in sidx) for p in people]
    skills = [set(p['skills'].split()) for p in people]
    cap = [p['max'] for p in people]
    conf = [set() for _ in range(P)]
    for a, b in args['conflicts']:
        conf[pidx[a]].add(pidx[b])
        conf[pidx[b]].add(pidx[a])
    adj = args['no_adjacent']
    load = [0] * P
    assign = [[] for _ in range(S)]

    def ok(p, s, skip=-1):
        if s not in avail[p] or p in assign[s] or (load[p] >= cap[p] and skip < 0):
            return False
        if adj and ((s > 0 and s - 1 != skip and p in assign[s - 1]) or
                    (s + 1 < S and s + 1 != skip and p in assign[s + 1])):
            return False
        for q in assign[s]:
            if q in conf[p]:
                return False
        return True

    def put(p, s):
        assign[s].append(p)
        load[p] += 1

    if MODE < 2:
        order = list(range(S))
    else:
        order = sorted(range(S), key=lambda s: (sum(1 for p in range(P) if s in avail[p]) - slots[s]['need'], s))
    for s in order:
        need, sk = (1 if MODE == 0 else slots[s]['need']), slots[s]['skill']
        cands = list(range(P))
        if MODE == 2:
            cands.sort(key=lambda p: (len(avail[p]) - load[p], p))
            if sk:
                cands.sort(key=lambda p: sk not in skills[p])
        for p in cands:
            if len(assign[s]) >= need:
                break
            if ok(p, s):
                put(p, s)
    if MODE == 2:
        for s in order:
            while len(assign[s]) < slots[s]['need']:
                moved = False
                for s2 in range(S):
                    if s2 == s or moved:
                        continue
                    for q in list(assign[s2]):
                        if not ok(q, s, skip=s2):
                            continue
                        sk2 = slots[s2]['skill']
                        must = bool(sk2) and sk2 in skills[q] and \
                            not any(sk2 in skills[x] for x in assign[s2] if x != q)
                        assign[s2].remove(q)
                        load[q] -= 1
                        r = next((r for r in range(P) if r != q and ok(r, s2) and (not must or sk2 in skills[r])), -1)
                        if r < 0 or not ok(q, s):
                            put(q, s2)
                            continue
                        put(r, s2)
                        put(q, s)
                        moved = True
                        break
                if not moved:
                    break
    return {'assign': {slots[s]['id']: [people[p]['id'] for p in assign[s]] for s in range(S)}}
'''


def _score(inp, hidden, out) -> float:
    asg = out.get("assign")
    slots, people = inp["slots"], inp["people"]
    if not isinstance(asg, dict):
        return 0.0
    sidx = {s["id"]: k for k, s in enumerate(slots)}
    pinfo = {p["id"]: p for p in people}
    conf = {tuple(sorted(c)) for c in inp["conflicts"]}
    load: dict[str, int] = {}
    by_slot: list[list] = [[] for _ in slots]
    for sid, plist in asg.items():
        if sid not in sidx or not isinstance(plist, list) or len(plist) > slots[sidx[sid]]["need"]:
            return 0.0
        for p in plist:
            if not isinstance(p, str) or p not in pinfo or p in by_slot[sidx[sid]] or \
                    sid not in pinfo[p]["avail"].split():
                return 0.0
            by_slot[sidx[sid]].append(p)
            load[p] = load.get(p, 0) + 1
    if any(n > pinfo[p]["max"] for p, n in load.items()):
        return 0.0
    for k, plist in enumerate(by_slot):
        for i in range(len(plist)):
            for j in range(i + 1, len(plist)):
                if tuple(sorted((plist[i], plist[j]))) in conf:
                    return 0.0
        if inp["no_adjacent"] and k + 1 < len(by_slot) and set(plist) & set(by_slot[k + 1]):
            return 0.0
    staffed = 0
    for s, plist in zip(slots, by_slot):
        if not s["skill"] or any(s["skill"] in pinfo[p]["skills"].split() for p in plist):
            staffed += len(plist)
    return staffed / hidden


@category_family("coord", kind="organisational coordination", relation="specified", split="test",
                 doc="Assign people to slots under generated availability and conflict constraints.",
                 population=Population(0.8, (3.0, 16.0), (0.8, 3.5), "team leads staffing shifts and meetings"),
                 demand_links=("schedule", "triage"), resource_intensity=1.2)
def build(rng, params):
    p_conf = float(pick(rng, (0.0, 0.1, 0.22)))
    use_skills = bool(rng.random() < 0.6)
    no_adjacent = bool(rng.random() < 0.5)
    extra = int(pick(rng, (2, 3, 4)))
    tight = float(pick(rng, (0.6, 0.8, 0.95)))
    loads = (1, 2) if rng.random() < 0.5 else (2, 3)
    if p_conf == 0.0 and not use_skills and not no_adjacent:
        no_adjacent = True                                   # every world binds at least one constraint
    rule = {"p_conflict": p_conf, "skills": use_skills, "no_adjacent": no_adjacent, "extra_availability": extra,
            "max_load": list(loads), "tight": tight}

    def generate(rng, size):
        st = Stream(rng)
        S = int(5 + 5 * size)
        need = [st.between(1, 3) for _ in range(S)]
        cap, planted = [], []                                # per person: accepted load, planted slots
        for s in range(S):
            elig = [p for p in range(len(cap)) if len(planted[p]) < cap[p]
                    and not (no_adjacent and s - 1 in planted[p])]
            while len(elig) < need[s]:
                cap.append(st.between(loads[0], loads[1]))
                planted.append([])
                elig.append(len(cap) - 1)
            for p in st.sample(elig, need[s]):
                planted[p].append(s)
        P = len(cap)
        skill_of = [""] * S
        pskills: list[set] = [set() for _ in range(P)]
        if use_skills:
            for s in range(S):
                if st.chance(0.5):
                    skill_of[s] = st.pick(SKILLS)
                    pskills[st.pick([p for p in range(P) if s in planted[p]])].add(skill_of[s])
            for p in range(P):
                if st.chance(0.15):
                    pskills[p].add(st.pick(SKILLS))
        order = st.shuffled(range(P))                        # people are listed in an order unrelated to the plant
        name = {p: f"p{k:02d}" for k, p in enumerate(order)}
        conflicts = sorted(sorted((name[a], name[b])) for a in range(P) for b in range(a + 1, P)
                           if not set(planted[a]) & set(planted[b]) and st.chance(p_conf))
        people = []
        for p in order:
            av = set(planted[p]) | {st.below(S) for _ in range(extra)}
            # most people accept exactly the load the plant gives them: capacity is tight, choices matter
            people.append({"id": name[p], "avail": " ".join(f"s{s}" for s in sorted(av)),
                           "skills": " ".join(sorted(pskills[p])),
                           "max": len(planted[p]) if st.chance(tight) else cap[p]})
        slots = [{"id": f"s{s}", "need": need[s], "skill": skill_of[s]} for s in range(S)]
        return {"slots": slots, "people": people, "conflicts": conflicts, "no_adjacent": no_adjacent}, float(sum(need))

    binds = [b for b, on in (("some pairs of people cannot share a slot", p_conf > 0),
                             ("some slots require a skill", use_skills),
                             ("nobody may work two consecutive slots", no_adjacent)) if on]
    doc = ("Staff every slot with available people.\n"
           "Input: {\"slots\": [{\"id\": str, \"need\": int, \"skill\": str}, ...] (in time order), \"people\": "
           "[{\"id\": str, \"avail\": \"space separated slot ids\", \"skills\": \"space separated skills\", "
           "\"max\": int}, ...], \"conflicts\": [[person, person], ...], \"no_adjacent\": bool}.\n"
           "Output: {\"assign\": {slot id: [person ids]}}.\n"
           "Rules (any violation scores 0): a person serves only slots they are available for, at most \"max\" "
           "slots in total, never with someone they conflict with, never twice in a slot, and - when no_adjacent "
           "is true - never in two consecutive slots; a slot takes at most \"need\" people.\n"
           "A slot whose \"skill\" is not empty counts only if at least one of its people has that skill.\n"
           f"In this world: {'; '.join(binds)}; people accept {loads[0]} to {loads[1]} slots.\n"
           "Score: staffed positions / positions needed (every position can be staffed).")
    return {"title": "Staff slots with available people", "doc": doc,
            "keywords": ("staffing", "rota", "assignment", "coordination", "availability", "shifts", "team"),
            "generate_raw": generate, "score_raw": _score, "base_price_cents": 4.0,
            "tiers": [("one-per-slot", "MODE = 0\n" + _SOLVER), ("first-fit", "MODE = 1\n" + _SOLVER),
                      ("constrained-first-repair", "MODE = 2\n" + _SOLVER)],
            "rule": rule, "table_key": "people", "output_key": "assign"}
