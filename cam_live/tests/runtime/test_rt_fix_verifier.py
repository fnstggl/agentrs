"""Regression tests for the verifier defects D9 and D22 of docs/verification/G_rt.md (fixed by F-fix).

D9  the membership verifier counted accounts it should not (linked to excluded accounts, an id listed as human and
    as bot, flags in another letter case, ids differing only in whitespace / case, records without a join time,
    unknown account age with ``min_account_age_s`` set) and reported ``achieved``.
D22 a restarted verifier compared only the last entry with its head; raw member ids reached ``verifier.jsonl``.
"""
import json
import os

import pytest

from cam_live.contracts import GoalStatus, Limits, SimClock, TaskSpec
from cam_live.runtime.verifier import (DAY, HashChainLog, MemberRecord, MembershipVerifier, classify_members,
                                       DEFAULT_RULES, norm_id)

T0 = 1_800_000_000.0


class Source:
    name, live = "test-members", False

    def __init__(self, members):
        self.members = members

    def list_members(self):
        return [MemberRecord.from_dict(m) if isinstance(m, dict) else m for m in self.members]


def task(target=100, **excl):
    return TaskSpec(objective_id="obj_members", objective="members", family="community/membership", mode="staging",
                    success={"verifier": "membership", "metric": "membership.confirmed", "target": target,
                             "confidence": 0.95, "exclusions": excl},
                    limits=Limits(duration_s=30 * DAY, max_decisions=10))


def humans(n, prefix="u", **kw):
    return [{"id": f"{prefix}{i}", "joined_at": T0 - 1000 + i, **kw} for i in range(n)]


def verify(tmp_path, members, name="v", **excl):
    clock = SimClock(T0)
    t = task(**excl)
    v = MembershipVerifier(Source(members), clock, str(tmp_path / name), log_path=str(tmp_path / f"{name}.jsonl"),
                           pinned_task=t)
    return v, v.verify(t)


# ---------------------------------------------------------------------------------------------------- D9
@pytest.mark.parametrize("excluded_record", [
    {"id": "T", "joined_at": 1.0, "link_key": "person"},                                   # a declared test account
    {"id": "B", "joined_at": 1.0, "link_key": "person", "is_bot": True},                   # a bot
    {"id": "P", "joined_at": 1.0, "link_key": "person", "flags": ["purchased"]},           # a purchased account
])
def test_an_account_linked_to_an_excluded_account_is_not_counted(tmp_path, excluded_record):
    alt = {"id": "alt", "joined_at": 2.0, "link_key": "person"}
    _, o = verify(tmp_path, humans(99) + [excluded_record, alt], test_account_ids=["t"])     # ids match in any case
    assert o.value == 99 and o.status == GoalStatus.PENDING
    assert o.excluded.get("linked_to_excluded") == 1


def test_links_are_followed_transitively_and_order_does_not_matter():
    rules = dict(DEFAULT_RULES)
    chain = [MemberRecord("c", joined_at=3.0, link_key="k2"), MemberRecord("b1", joined_at=2.0, link_key="k1"),
             MemberRecord("b2", joined_at=2.5, link_key="k2"), MemberRecord("bot", joined_at=1.0, is_bot=True,
                                                                            link_key="k1")]
    # b1 and b2 are one account (same id after normalisation) holding both keys: bot -k1- b -k2- c
    chain[2].id = "B1 "
    for members in (chain, list(reversed(chain))):
        c = classify_members(members, rules, T0)
        assert not c["counted"]
        assert {r: len(v) for r, v in c["excluded"].items()} == {"bots": 1, "duplicates": 1, "linked_to_excluded": 2}


def test_an_id_listed_as_human_and_as_bot_is_a_bot(tmp_path):
    _, o = verify(tmp_path, humans(99) + [{"id": "x", "joined_at": 1.0}, {"id": "x", "joined_at": 2.0, "is_bot": True}])
    assert o.value == 99 and o.excluded == {"bots": 1, "duplicates": 1} and o.status == GoalStatus.PENDING


def test_exclusion_and_uncertain_flags_match_in_any_letter_case(tmp_path):
    _, o = verify(tmp_path, humans(60) + humans(60, "p", flags=["Purchased"]) + humans(5, "s", flags=[" AUTOMATION_suspected"]))
    assert o.value == 60 and o.excluded == {"purchased": 60} and "automation_suspected 5" in o.notes


def test_ids_differing_only_in_whitespace_or_case_are_one_account(tmp_path):
    assert norm_id(" U5 ") == norm_id("u5") == "u5"
    _, o = verify(tmp_path, humans(99) + [{"id": " u5"}, {"id": "u5 "}, {"id": "U5"}, {"id": "   "}])
    assert o.value == 99 and o.excluded == {"duplicates": 3, "invalid_id": 1} and o.status == GoalStatus.PENDING


def test_records_without_a_join_time_are_uncertain_not_confirmed(tmp_path):
    _, o = verify(tmp_path, [{"id": f"n{i}"} for i in range(100)])
    assert o.status == GoalStatus.PENDING and o.value == 0 and o.uncertainty > 0
    assert "join_time_unknown 100" in o.notes and "success is not claimed" in o.notes
    _, o = verify(tmp_path, humans(100), name="control")              # the same accounts with a join time
    assert o.status == GoalStatus.ACHIEVED and o.value == 100


def test_wrongly_typed_fields_are_not_evidence_for_a_member(tmp_path):
    m = MemberRecord.from_dict({"id": 7, "joined_at": "yesterday", "is_bot": 1, "left_at": "x", "flags": "Purchased"})
    assert m.id == "7" and m.joined_at == 0.0 and m.is_bot is True and m.left_at is None and m.flags == ["Purchased"]
    _, o = verify(tmp_path, humans(100, joined_at="unknown"))
    assert o.status == GoalStatus.PENDING and o.value == 0


def test_unknown_account_age_is_uncertain_when_a_minimum_age_is_set(tmp_path):
    _, o = verify(tmp_path, humans(100, account_created_at=0.0), min_account_age_s=30 * DAY)
    assert o.status == GoalStatus.PENDING and o.value == 0 and "account_age_unknown 100" in o.notes
    old = humans(100, account_created_at=T0 - 400 * DAY)
    _, o = verify(tmp_path, old, name="known", min_account_age_s=30 * DAY)
    assert o.status == GoalStatus.ACHIEVED and o.value == 100


# ---------------------------------------------------------------------------------------------------- D22
def test_a_restarted_verifier_refuses_a_log_edited_before_its_last_entry(tmp_path):
    v, _ = verify(tmp_path, humans(100))
    for _ in range(3):
        v.verify(v.pinned_task)
    path, state = v.log.path, str(tmp_path / "v")
    lines = open(path).read().splitlines()
    e = json.loads(lines[1])
    e["payload"]["outcome"]["value"] = 5.0                            # an edit in the middle; the last entry is intact
    open(path, "w").write("\n".join(lines[:1] + [json.dumps(e, sort_keys=True)] + lines[2:]) + "\n")
    with pytest.raises(RuntimeError, match="not an intact hash chain"):
        MembershipVerifier(Source(humans(100)), SimClock(T0), state, log_path=path, pinned_task=task())
    with pytest.raises(RuntimeError):
        HashChainLog(path, os.path.join(state, "head.json"))


def test_a_crash_between_entry_and_head_or_mid_append_is_recovered_without_breaking_the_chain(tmp_path):
    v, _ = verify(tmp_path, humans(10))
    path, state = v.log.path, str(tmp_path / "v")
    head_path = os.path.join(state, "head.json")
    before = json.load(open(head_path))
    v.verify(v.pinned_task)                                          # one more entry ...
    json.dump(before, open(head_path, "w"))                           # ... whose head update was lost in a crash
    with open(path, "ab") as f:                                       # and a half-written entry after it
        f.write(b'{"seq": 99, "kind": "verify", "payl')
    v2 = MembershipVerifier(Source(humans(10)), SimClock(T0), state, log_path=path, pinned_task=task())
    assert v2.log.repaired_bytes > 0 and v2.log.seq == before["seq"] + 1
    v2.verify(v2.pinned_task)
    ok, n, why = HashChainLog.verify_chain(path, json.load(open(head_path)))
    assert ok and n == before["seq"] + 3, why


def test_the_verifier_log_holds_no_raw_member_or_invite_identifier(tmp_path):
    ids = [f"9{i:017d}" for i in range(100)]
    members = [{"id": i, "joined_at": 1.0} for i in ids] + [{"id": "inv", "joined_at": 1.0, "invite_code": "BULKCODE7"}]
    v, o = verify(tmp_path, members, test_account_ids=ids[:2], suspect_member_ids=ids[2:4],
                  suspect_invite_codes=["BULKCODE7"])
    blob = open(v.log.path).read()
    assert not [i for i in ids if i in blob] and "BULKCODE7" not in blob and "inv" not in json.dumps(o.to_dict())
    rules = [json.loads(ln)["payload"]["detail"]["rules"] for ln in blob.splitlines() if '"verify"' in ln][-1]
    assert len(rules["test_account_ids"]) == 2 and all(r.startswith("member:") for r in rules["test_account_ids"])
    assert rules["suspect_invite_codes"][0].startswith("invite:")
    assert o.excluded.get("test_accounts") == 2 and "purchase_indicator 3" in o.notes
