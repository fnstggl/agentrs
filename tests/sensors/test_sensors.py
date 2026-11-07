"""Source permissions, rate limits, revocation, deletion, retention, deduplication (Test 9)."""

import json

import pytest

from agentrs.sensors.observation_network import NetworkConfig, ObservationSensorNetwork
from agentrs.sensors.permissions import check_record, redact_pii
from agentrs.sensors.source_adapter import (
    JsonlDatasetAdapter,
    SourceDeclaration,
    UnauthorizedLiveSource,
)
from agentrs.sensors.source_registry import SourceEntry, SourceRegistry
from agentrs.socialweblab.engine import SocialWebLab
from agentrs.temporal.delayed_outcome_ledger import ActionRecord
from agentrs.temporal.system import build_network, lab_source


@pytest.fixture()
def lab(tmp_path):
    lab = SocialWebLab(tmp_path / "lab.db", seed=5, n_users=25)
    lab.advance(20 * 3600)
    yield lab
    lab.close()


def _dataset(tmp_path, authorized: bool, ref=None, categories=None):
    rows = [{"id": i, "time": 1000.0 + i, "operation": "create_post", "content": f"post {i} mail me at a.b@example.com",
             "categories": categories or ["public_posts"], "follow_ups": [{"type": "response_created", "t": 1100.0 + i}]}
            for i in range(5)]
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    decl = SourceDeclaration(source_id="ds", name="dataset", access_method="licensed_dataset", operator="vendor",
                             allowed_data_categories=["public_posts", "public_replies"], training_authorized=authorized,
                             authorization_ref=ref, time_base="real")
    return JsonlDatasetAdapter(decl, p)


def test_unauthorised_sources_are_rejected(tmp_path):
    reg = SourceRegistry()
    with pytest.raises(PermissionError, match="training use is not authorised"):
        reg.register(_dataset(tmp_path, authorized=False, ref="lic-1"))
    with pytest.raises(PermissionError, match="authorisation reference"):
        reg.register(_dataset(tmp_path, authorized=True, ref=None))
    decl = SourceDeclaration(source_id="reddit", name="reddit", access_method="live_web", operator="reddit",
                             allowed_data_categories=["public_posts"], training_authorized=False)
    with pytest.raises(PermissionError, match="explicit"):
        UnauthorizedLiveSource(decl)  # public visibility is not permission to train
    ranked = SourceRegistry.discover([SourceEntry(declaration=decl, coverage_tags=["forum"])], {}, 3)
    assert ranked[0][1] == 0.0 and ranked[0][2].startswith("refused")


def test_authorised_dataset_ingests_with_pii_redaction_and_category_checks(tmp_path):
    reg = SourceRegistry()
    reg.register(_dataset(tmp_path, authorized=True, ref="license-2026-001"))
    net = ObservationSensorNetwork(tmp_path / "net", reg)
    net.ingest("ds")
    acts = net.ledger.actions()
    assert len(acts) == 5 and all("[email]" in a["action"]["content"] for a in acts)
    assert all(len(net.ledger.events(a["action_id"])) == 1 for a in acts)
    assert check_record(reg.declaration("ds"), ["private_messages"])[0] is False
    reg2 = SourceRegistry()
    reg2.register(_dataset(tmp_path / ".." / tmp_path.name, authorized=True, ref="lic", categories=["private_messages"]))
    net2 = ObservationSensorNetwork(tmp_path / "net2", reg2)
    net2.ingest("ds")
    assert net2.ledger.stats()["actions"] == 0 and net2.stats["refused"] >= 5
    assert redact_pii("call +1 (555) 123-4567 or x@y.org") == "call [phone] or [email]"


def test_rate_limits_revocation_deletion_retention_and_dedup(lab, tmp_path):
    net = build_network(tmp_path / "net", lab, None, NetworkConfig(scheduling="fixed"))
    net.scheduler.set_budget("swl-feed", capacity=5, refill_per_s=5 / 3600, now=lab.now())  # 5 checks per hour
    net.ingest("swl-feed")
    n_actions = net.ledger.stats()["actions"]
    assert n_actions > 5
    # re-polling from the start creates no duplicates
    net.cursors["swl-feed"] = 0
    net.ingest("swl-feed")
    assert net.ledger.stats()["actions"] == n_actions and net.dedup.stats["same_source_duplicate"] >= n_actions
    lab.advance(3600)
    net.observe()
    lab.advance(600)
    net.observe()
    assert net.stats["observations"] <= 5 + 1  # budget in the source's (virtual) clock
    # deletion at the source: content redacted, no longer trainable, follow-ups cancelled
    it = lab.items("state='visible'")[0]
    lab.delete_item(it["author"], it["id"])
    net.ingest("swl-feed")
    from agentrs.sensors.deduplication import stable_action_id

    a = net.ledger.action(stable_action_id("swl-feed", str(it["id"])))
    assert a["action"]["content"] is None and not a["training_eligible"]
    assert all(j["status"] in ("cancelled", "done", "failed") for j in net.scheduler.jobs(a["action_id"]))
    # our own action is not re-ingested as a third-party action
    uid = lab.create_account("agent0", "pw", spaces=["linux"])
    aid = net.register_self_action(ActionRecord(source_id="self", source_type="self", actor_type="agent",
                                                time_base=lab.time_base, action_time=lab.now(), observed_time=lab.now(),
                                                action_str='submit(create_post, "How do I fix ssh?")',
                                                action={"operation": "create_post", "content": "How do I fix ssh?"}))
    iid = lab.create_item(uid, "linux", "How do I fix ssh?", "How do I fix ssh? error E1234 with bash 5.1")
    net.link_self_action(aid, "swl-feed", str(iid))
    before = net.ledger.stats()["actions"]
    net.ingest("swl-feed")
    assert net.ledger.stats()["actions"] == before and net.dedup.stats["own_action"] == 1
    assert net.ledger.action(aid)["source_type"] == "self"
    # retention: everything older than the retention period is purged
    assert net.enforce_retention(now_wall=__import__("time").time() + 400 * 86400) > 0
    # revocation: data purged, jobs cancelled, the source cannot silently come back
    net.registry.entries["swl-feed"].declaration.retention_days = None
    counts = net.revoke_source("swl-feed", "authorisation withdrawn")
    assert net.ledger.stats()["actions"] <= 1  # only our own action remains (source 'self')
    assert all(j["status"] != "pending" for j in net.scheduler.jobs())
    with pytest.raises(PermissionError, match="revoked"):
        net.registry.register(lab_source(lab))
    with pytest.raises(PermissionError):
        net.ingest("swl-feed")
    assert counts["actions"] >= 0
