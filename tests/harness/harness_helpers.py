import os
from types import SimpleNamespace

PG_URL = os.environ.get("AGENTRS_TEST_PG_URL")
REDIS_URL = os.environ.get("AGENTRS_TEST_REDIS_URL")


def make_sandbox(tmp_path, scratch_bb, agents=("sandbox_agent_001",), db_url=None, n_users=40, seed=3, redis_url=None,
                 transports=None, extra_domains=(), max_tier=1, secret_key=None, lab=None, srv=None):
    import torch

    from agentrs.content.policy import ContentGenConfig, ContentGeneratingPolicy
    from agentrs.harness.inference import BatchingInferenceServer, InferenceClient, LocalPolicyBackend
    from agentrs.harness.runtime import AgentRuntime, SocialWebLabHelpAdapter
    from agentrs.harness.sandbox import build_sandbox_harness
    from agentrs.models.policy import LMPolicy, PolicyConfig
    from agentrs.socialweblab.engine import SocialWebLab
    from agentrs.socialweblab.server import SocialWebLabServer

    torch.manual_seed(0)
    if lab is None:
        lab = SocialWebLab(tmp_path / "lab.db", seed=seed, n_users=n_users, lab_id=f"t{seed}")
        lab.advance(14 * 3600)
        srv = SocialWebLabServer(lab)
        srv.start()
    h = build_sandbox_harness(tmp_path / "harness", lab, srv, db_url or f"sqlite:///{tmp_path}/harness.db", agents,
                              redis_url=redis_url, transports=transports, extra_domains=extra_domains, max_tier=max_tier,
                              secret_key=secret_key)
    cp = ContentGeneratingPolicy(LMPolicy(PolicyConfig(backbone=scratch_bb)), ContentGenConfig(max_content_tokens=16),
                                 checkpoint_id="policy-v0")
    backend = LocalPolicyBackend({"policy-v0": cp}, "policy-v0")
    inf = BatchingInferenceServer(backend, max_batch=8, max_wait_ms=20)
    client = InferenceClient(server=inf)
    adapter = SocialWebLabHelpAdapter(lab.now, lab.time_base)
    rts = {a: AgentRuntime(a, h.gateway, client, h.tasks, h.provisioning, h.verification, {"socialweblab": adapter})
           for a in agents}
    return SimpleNamespace(h=h, lab=lab, srv=srv, cp=cp, backend=backend, inf=inf, client=client, rts=rts, adapter=adapter)
